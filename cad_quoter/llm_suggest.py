from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from cad_quoter.coerce import to_float, to_int
from cad_quoter.domain_models import DEFAULT_MATERIAL_DISPLAY
from cad_quoter.llm_overrides import coerce_bounds
from cad_quoter.utils import _first_non_none, compact_dict, jdump

try:  # pragma: no cover - tolerate optional dependency removal
    from cad_quoter.llm import LLMClient, parse_llm_json
except Exception:  # pragma: no cover - fallback keeps quoting functional
    LLMClient = cast("type", None)  # type: ignore[assignment]

    def parse_llm_json(_text: str) -> dict:  # type: ignore[override]
        return {}

from appkit.scrap_helpers import normalize_scrap_pct

T = TypeVar("T")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion used in multiple pricing paths."""

    try:
        coerced = float(value or 0.0)
    except Exception:
        return default
    if coerced != coerced or coerced in {float("inf"), float("-inf")}:
        return default
    return coerced


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"y", "yes", "true", "1", "on"}:
            return True
        if text in {"n", "no", "false", "0", "off"}:
            return False
    return None


@dataclass(slots=True)
class SuggestionSet:
    """Normalized representation of an LLM suggestion bundle."""

    process_hour_multipliers: dict[str, float] = field(
        default_factory=lambda: {"drilling": 1.0, "milling": 1.0}
    )
    process_hour_adders: dict[str, float] = field(
        default_factory=lambda: {"inspection": 0.0}
    )
    scrap_pct: float = 0.0
    setups: int = 1
    fixture: str = "standard"
    notes: list[str] = field(default_factory=list)
    no_change_reason: str = ""
    fixture_build_hr: float | None = None
    soft_jaw_hr: float | None = None
    soft_jaw_material_cost: float | None = None
    operation_sequence: list[str] = field(default_factory=list)
    handling_adder_hr: float | None = None
    drilling_strategy: dict[str, Any] | None = None
    cmm_minutes: float | None = None
    in_process_inspection_hr: float | None = None
    fai_required: bool | None = None
    fai_prep_hr: float | None = None
    packaging_hours: float | None = None
    packaging_flat_cost: float | None = None
    shipping_cost: float | None = None
    shipping_hint: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "process_hour_multipliers": self.process_hour_multipliers
            or {"drilling": 1.0, "milling": 1.0},
            "process_hour_adders": self.process_hour_adders or {"inspection": 0.0},
            "scrap_pct": float(self.scrap_pct),
            "setups": int(self.setups),
            "fixture": self.fixture,
            "notes": list(self.notes),
            "no_change_reason": self.no_change_reason,
        }

        optional = {
            "fixture_build_hr": self.fixture_build_hr,
            "soft_jaw_hr": self.soft_jaw_hr,
            "soft_jaw_material_cost": self.soft_jaw_material_cost,
            "operation_sequence": self.operation_sequence,
            "handling_adder_hr": self.handling_adder_hr,
            "drilling_strategy": self.drilling_strategy,
            "cmm_minutes": self.cmm_minutes,
            "in_process_inspection_hr": self.in_process_inspection_hr,
            "fai_required": self.fai_required,
            "fai_prep_hr": self.fai_prep_hr,
            "packaging_hours": self.packaging_hours,
            "packaging_flat_cost": self.packaging_flat_cost,
            "shipping_cost": self.shipping_cost,
            "shipping_hint": self.shipping_hint,
        }
        for key, value in optional.items():
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            if isinstance(value, dict) and not value:
                continue
            data[key] = value

        if self.meta:
            data["_meta"] = self.meta

        return data


def coerce(cls: type[T], raw: Mapping[str, Any] | None, **context: Any) -> T:
    """Instantiate ``cls`` from an arbitrary mapping using helper coercions."""

    if cls is SuggestionSet:
        return cast(T, _coerce_suggestion_set(raw, context.get("bounds")))
    raise TypeError(f"Unsupported coercion target: {cls!r}")


def sanitize_suggestions(raw: Mapping[str, Any] | None, bounds: Mapping[str, Any] | None) -> dict:
    """Public wrapper mirroring the legacy sanitize helper."""

    suggestion = coerce(SuggestionSet, raw or {}, bounds=bounds)
    return suggestion.to_dict()


def _coerce_suggestion_set(
    raw: Mapping[str, Any] | None, bounds: Mapping[str, Any] | None
) -> SuggestionSet:
    suggestion_map: dict[str, Any] = dict(raw or {}) if isinstance(raw, Mapping) else {}
    coerced_bounds = coerce_bounds(bounds or {})

    mult_min = coerced_bounds["mult_min"]
    mult_max = coerced_bounds["mult_max"]
    adder_min = coerced_bounds["adder_min_hr"]
    base_adder_max = coerced_bounds["adder_max_hr"]
    scrap_min = coerced_bounds["scrap_min"]
    scrap_max = coerced_bounds["scrap_max"]
    bucket_caps = coerced_bounds.get("adder_bucket_max", {})

    meta_info: dict[str, Any] = {}

    def _normalize_conf(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            mapping = {
                "low": 0.3,
                "medium": 0.6,
                "med": 0.6,
                "mid": 0.6,
                "high": 0.85,
                "very high": 0.95,
                "certain": 0.98,
            }
            key = value.strip().lower()
            if key in mapping:
                return mapping[key]
        conf = to_float(value)
        if conf is None:
            return None
        return max(0.0, min(1.0, float(conf)))

    def _extract_detail(raw_val: Any) -> tuple[Any, dict[str, Any]]:
        detail: dict[str, Any] = {}
        value = raw_val
        if isinstance(raw_val, Mapping):
            if "value" in raw_val:
                value = raw_val.get("value")
            reason = str(raw_val.get("reason") or "").strip()
            if reason:
                detail["reason"] = reason[:160]
            conf = _normalize_conf(raw_val.get("confidence"))
            if conf is not None:
                detail["confidence"] = conf
            source_raw = raw_val.get("source") or raw_val.get("sources")
            sources: Sequence[Any] | None
            if isinstance(source_raw, str):
                sources = [source_raw]
            elif isinstance(source_raw, Sequence):
                sources = source_raw
            else:
                sources = None
            cleaned_sources: list[str] = []
            if sources:
                for src in sources:
                    if not src:
                        continue
                    cleaned_sources.append(str(src).strip().upper()[:24])
            if cleaned_sources:
                detail["source"] = cleaned_sources
        return value, detail

    def _store_meta(path: tuple[str, ...], detail: Mapping[str, Any], value: Any) -> None:
        cleaned = compact_dict(detail)
        if not cleaned:
            return
        cleaned["value"] = value
        node = meta_info
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = cleaned

    def _extract_float_field(
        raw_val: Any, lo: float | None, hi: float | None, path: tuple[str, ...]
    ) -> float | None:
        if raw_val is None:
            return None
        value, detail = _extract_detail(raw_val)
        num = to_float(value)
        if num is None:
            return None
        if lo is not None:
            num = max(lo, num)
        if hi is not None:
            num = min(hi, num)
        _store_meta(path, detail, num)
        return num

    def _extract_bool_field(raw_val: Any, path: tuple[str, ...]) -> bool | None:
        if raw_val is None:
            return None
        value, detail = _extract_detail(raw_val)
        flag = _coerce_bool(value)
        if flag is None:
            return None
        _store_meta(path, detail, flag)
        return flag

    setup_block: dict[str, Any] | None = None
    setup_reco = suggestion_map.get("setup_recommendation")
    if isinstance(setup_reco, Mapping):
        setup_block = dict(setup_reco)
    else:
        setup_plan = suggestion_map.get("setup_plan")
        if isinstance(setup_plan, Mapping):
            setup_block = dict(setup_plan)
    if setup_block:
        for key in ("setups", "fixture", "notes"):
            if key not in suggestion_map and setup_block.get(key) is not None:
                suggestion_map[key] = setup_block.get(key)
        if "fixture_build_hr" in setup_block and "fixture_build_hr" not in suggestion_map:
            suggestion_map["fixture_build_hr"] = setup_block.get("fixture_build_hr")

    mults: dict[str, float] = {}
    for proc, raw_val in (suggestion_map.get("process_hour_multipliers") or {}).items():
        value, detail = _extract_detail(raw_val)
        num = to_float(value)
        if num is None:
            continue
        num = max(mult_min, min(mult_max, num))
        proc_key = str(proc)
        mults[proc_key] = num
        _store_meta(("process_hour_multipliers", proc_key), detail, num)

    adders: dict[str, float] = {}
    for proc, raw_val in (suggestion_map.get("process_hour_adders") or {}).items():
        value, detail = _extract_detail(raw_val)
        num = to_float(value)
        if num is None:
            continue
        proc_key = str(proc)
        bucket_cap = bucket_caps.get(proc_key.lower())
        limit = bucket_cap if bucket_cap is not None else base_adder_max
        limit = max(adder_min, float(limit))
        num = max(adder_min, min(limit, num))
        adders[proc_key] = num
        _store_meta(("process_hour_adders", proc_key), detail, num)

    raw_scrap = suggestion_map.get("scrap_pct", 0.0)
    scrap_val, scrap_detail = _extract_detail(raw_scrap)
    scrap_float = to_float(scrap_val)
    if scrap_float is None:
        scrap_float = 0.0
    scrap_float = max(scrap_min, min(scrap_max, scrap_float))
    _store_meta(("scrap_pct",), scrap_detail, scrap_float)

    raw_setups = suggestion_map.get("setups", 1)
    setups_val, setups_detail = _extract_detail(raw_setups)
    try:
        setups_int = int(round(float(setups_val)))
    except Exception:
        setups_int = 1
    setups_int = max(1, min(4, setups_int))
    _store_meta(("setups",), setups_detail, setups_int)

    raw_fixture = suggestion_map.get("fixture", "standard")
    fixture_val, fixture_detail = _extract_detail(raw_fixture)
    fixture_str = str(fixture_val).strip() if fixture_val is not None else "standard"
    if not fixture_str:
        fixture_str = "standard"
    fixture_str = fixture_str[:120]
    _store_meta(("fixture",), fixture_detail, fixture_str)

    notes_raw = suggestion_map.get("notes") or []
    notes: list[str] = []
    if isinstance(notes_raw, Sequence):
        for note in notes_raw:
            if isinstance(note, Mapping):
                value, detail = _extract_detail(note)
                text = str(value).strip()
                if text:
                    trimmed = text[:160]
                    notes.append(trimmed)
                    _store_meta(("notes", str(len(notes) - 1)), detail, trimmed)
                continue
            if not isinstance(note, str):
                continue
            cleaned = note.strip()
            if cleaned:
                notes.append(cleaned[:160])

    raw_no_change = suggestion_map.get("no_change_reason")
    no_change_val, no_change_detail = _extract_detail(raw_no_change)
    no_change_str = str(no_change_val or "").strip()
    if no_change_str:
        _store_meta(("no_change_reason",), no_change_detail, no_change_str)

    extra: dict[str, Any] = {}

    fixture_build_raw = suggestion_map.get("fixture_build_hr")
    if fixture_build_raw is None and setup_block:
        fixture_build_raw = setup_block.get("fixture_build_hr")
    fixture_build_hr = _extract_float_field(
        fixture_build_raw, 0.0, 2.0, ("fixture_build_hr",)
    )
    if fixture_build_hr is not None:
        extra["fixture_build_hr"] = fixture_build_hr

    soft_block = suggestion_map.get("soft_jaw_plan")
    if not isinstance(soft_block, Mapping):
        soft_block = None
    soft_hr_raw = suggestion_map.get("soft_jaw_hr")
    soft_cost_raw = suggestion_map.get("soft_jaw_material_cost")
    if soft_block:
        if soft_hr_raw is None:
            soft_hr_raw = soft_block.get("hours") or soft_block.get("hr")
        if soft_cost_raw is None:
            soft_cost_raw = soft_block.get("stock_cost") or soft_block.get("material_cost")
    soft_hr = _extract_float_field(soft_hr_raw, 0.0, 1.0, ("soft_jaw_hr",))
    if soft_hr is not None:
        extra["soft_jaw_hr"] = soft_hr
    soft_cost = _extract_float_field(
        soft_cost_raw, 0.0, 60.0, ("soft_jaw_material_cost",)
    )
    if soft_cost is not None:
        extra["soft_jaw_material_cost"] = soft_cost

    op_block = suggestion_map.get("operation_sequence")
    op_steps_raw: Any = None
    op_handling_raw: Any = None
    if isinstance(op_block, Mapping):
        op_steps_raw = op_block.get("ops") or op_block.get("sequence")
        op_handling_raw = (
            op_block.get("handling_adder_hr") or op_block.get("handling_hr")
        )
    elif isinstance(op_block, Sequence):
        op_steps_raw = op_block
    if op_handling_raw is None:
        op_handling_raw = suggestion_map.get("handling_adder_hr")
    op_steps_clean: list[str] = []
    if isinstance(op_steps_raw, Sequence):
        for step in op_steps_raw:
            step_str = str(step).strip()
            if step_str:
                op_steps_clean.append(step_str[:80])
    if op_steps_clean:
        extra["operation_sequence"] = op_steps_clean[:12]
    handling_hr = _extract_float_field(op_handling_raw, 0.0, 0.2, ("handling_adder_hr",))
    if handling_hr is not None:
        extra["handling_adder_hr"] = handling_hr

    drilling_block = suggestion_map.get("drilling_strategy") or suggestion_map.get(
        "drilling_plan"
    )
    drilling_clean: dict[str, Any] = {}
    if isinstance(drilling_block, Mapping):
        mult_val, mult_detail = _extract_detail(drilling_block.get("multiplier"))
        mult = to_float(mult_val)
        if mult is not None:
            mult = max(0.8, min(1.5, mult))
            drilling_clean["multiplier"] = mult
            _store_meta(("drilling_strategy", "multiplier"), mult_detail, mult)
        floor_val, floor_detail = _extract_detail(
            drilling_block.get("per_hole_floor_sec")
            or drilling_block.get("floor_sec_per_hole")
        )
        floor = to_float(floor_val)
        if floor is not None:
            floor = max(0.0, floor)
            drilling_clean["per_hole_floor_sec"] = floor
            _store_meta(("drilling_strategy", "per_hole_floor_sec"), floor_detail, floor)
        if drilling_block.get("note") or drilling_block.get("reason"):
            note_text = str(drilling_block.get("note") or drilling_block.get("reason")).strip()
            if note_text:
                drilling_clean["note"] = note_text[:160]
    if drilling_clean:
        extra["drilling_strategy"] = drilling_clean

    cmm_minutes = _extract_float_field(
        suggestion_map.get("cmm_minutes") or suggestion_map.get("cmm_min"),
        0.0,
        60.0,
        ("cmm_minutes",),
    )
    if cmm_minutes is not None:
        extra["cmm_minutes"] = cmm_minutes

    inproc_hr = _extract_float_field(
        suggestion_map.get("in_process_inspection_hr"),
        0.0,
        0.5,
        ("in_process_inspection_hr",),
    )
    if inproc_hr is not None:
        extra["in_process_inspection_hr"] = inproc_hr

    fai_flag = _extract_bool_field(suggestion_map.get("fai_required"), ("fai_required",))
    if fai_flag is not None:
        extra["fai_required"] = fai_flag

    fai_prep = _extract_float_field(
        suggestion_map.get("fai_prep_hr"), 0.0, 1.0, ("fai_prep_hr",)
    )
    if fai_prep is not None:
        extra["fai_prep_hr"] = fai_prep

    packaging_hr = _extract_float_field(
        suggestion_map.get("packaging_hours"), 0.0, 0.5, ("packaging_hours",)
    )
    if packaging_hr is not None:
        extra["packaging_hours"] = packaging_hr

    packaging_cost = _extract_float_field(
        suggestion_map.get("packaging_flat_cost"), 0.0, 25.0, ("packaging_flat_cost",)
    )
    if packaging_cost is not None:
        extra["packaging_flat_cost"] = packaging_cost

    shipping_override_val = _extract_float_field(
        suggestion_map.get("shipping_cost"), 0.0, None, ("shipping_cost",)
    )
    if shipping_override_val is not None:
        extra["shipping_cost"] = shipping_override_val

    shipping_hint = suggestion_map.get("shipping_hint") or suggestion_map.get(
        "shipping_class"
    )
    shipping_hint_value: str | None = None
    if isinstance(shipping_hint, Mapping):
        value, detail = _extract_detail(shipping_hint)
        hint = str(value).strip()
        if hint:
            shipping_hint_value = hint[:80]
            _store_meta(("shipping_hint",), detail, shipping_hint_value)
    elif isinstance(shipping_hint, str) and shipping_hint.strip():
        shipping_hint_value = shipping_hint.strip()[:80]

    sanitized = SuggestionSet(
        process_hour_multipliers=mults or {"drilling": 1.0, "milling": 1.0},
        process_hour_adders=adders or {"inspection": 0.0},
        scrap_pct=scrap_float,
        setups=setups_int,
        fixture=fixture_str,
        notes=notes,
        no_change_reason=no_change_str,
        fixture_build_hr=extra.get("fixture_build_hr"),
        soft_jaw_hr=extra.get("soft_jaw_hr"),
        soft_jaw_material_cost=extra.get("soft_jaw_material_cost"),
        operation_sequence=list(extra.get("operation_sequence", [])),
        handling_adder_hr=extra.get("handling_adder_hr"),
        drilling_strategy=extra.get("drilling_strategy"),
        cmm_minutes=extra.get("cmm_minutes"),
        in_process_inspection_hr=extra.get("in_process_inspection_hr"),
        fai_required=extra.get("fai_required"),
        fai_prep_hr=extra.get("fai_prep_hr"),
        packaging_hours=extra.get("packaging_hours"),
        packaging_flat_cost=extra.get("packaging_flat_cost"),
        shipping_cost=extra.get("shipping_cost"),
        shipping_hint=shipping_hint_value,
        meta=meta_info,
    )

    return sanitized


def build_suggest_payload(
    geo: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    rates: Mapping[str, Any] | None,
    bounds: Mapping[str, Any] | None,
) -> dict:
    """Assemble the JSON payload passed to the suggestion LLM."""

    geo = dict(geo or {})
    baseline = dict(baseline or {})
    rates = dict(rates or {})
    bounds = dict(bounds or {})

    derived = geo.get("derived") or {}

    def _clean_nested(value: Any, depth: int = 0, max_depth: int = 3, limit: int = 24):
        if depth >= max_depth:
            return None
        if isinstance(value, Mapping):
            cleaned: dict[str, Any] = {}
            for idx, (key, val) in enumerate(value.items()):
                if idx >= limit:
                    break
                cleaned_val = _clean_nested(val, depth + 1, max_depth, limit)
                if cleaned_val is not None:
                    cleaned[str(key)] = cleaned_val
            return cleaned
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            cleaned_list: list[Any] = []
            for idx, item in enumerate(value):
                if idx >= limit:
                    break
                cleaned_val = _clean_nested(item, depth + 1, max_depth, limit)
                if cleaned_val is not None:
                    cleaned_list.append(cleaned_val)
            return cleaned_list
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:  # pragma: no cover - defensive
                return None
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            return value.strip()
        coerced = to_float(value)
        if coerced is not None:
            return coerced
        try:
            return str(value)
        except Exception:  # pragma: no cover - defensive
            return None

    hole_bins = derived.get("hole_bins") or {}
    hole_bins_top: dict[str, int] = {}
    if isinstance(hole_bins, Mapping):
        sorted_bins = sorted(
            ((str(k), to_int(v) or 0) for k, v in hole_bins.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        hole_bins_top = {k: int(v) for k, v in sorted_bins[:8] if v}

    raw_thickness = geo.get("thickness_mm")
    thickness_candidates: list[float | None] = []
    if isinstance(raw_thickness, Mapping):
        thickness_candidates.extend(
            to_float(raw_thickness.get(key)) for key in ("value", "mm", "thickness_mm")
        )
    else:
        thickness_candidates.append(to_float(raw_thickness))
    thickness_mm = _first_non_none(
        *thickness_candidates,
        to_float(geo.get("thickness")),
    )

    material_val = geo.get("material")
    if isinstance(material_val, Mapping):
        material_name = (
            material_val.get("name")
            or material_val.get("display")
            or material_val.get("material")
        )
        if material_name is not None:
            material_name = str(material_name).strip() or None
    elif isinstance(material_val, str):
        material_name = material_val.strip() or None
    else:
        material_name = None

    hole_count_val = _first_non_none(
        to_float(geo.get("hole_count")),
        to_float(derived.get("hole_count")),
    )
    hole_count = int(hole_count_val or 0)

    tap_qty = to_int(derived.get("tap_qty")) or 0
    cbore_qty = to_int(derived.get("cbore_qty")) or 0
    csk_qty = to_int(derived.get("csk_qty")) or 0

    finish_flags: list[str] = []
    raw_finish = derived.get("finish_flags") or geo.get("finish_flags")
    if isinstance(raw_finish, Sequence):
        finish_flags = [
            str(flag).strip().upper() for flag in raw_finish if str(flag).strip()
        ]

    needs_back_face = bool(
        derived.get("needs_back_face")
        or geo.get("needs_back_face")
        or geo.get("from_back")
    )

    derived_summary: dict[str, Any] = {}
    for key in (
        "tap_minutes_hint",
        "cbore_minutes_hint",
        "csk_minutes_hint",
        "tap_class_counts",
        "tap_details",
        "npt_qty",
        "max_hole_depth_in",
        "plate_area_in2",
        "finish_flags",
        "stock_guess",
        "has_ldr_notes",
        "has_tight_tol",
        "dfm_geo",
        "tolerance_inputs",
        "default_tolerance_note",
        "stock_catalog",
        "machine_limits",
        "fixture_plan",
        "fai_required",
        "pocket_area_total_in2",
        "slot_count",
        "edge_len_in",
        "deburr_edge_length_in",
        "dfm_summary",
    ):
        if key in derived and derived.get(key) not in (None, "", [], {}):
            derived_summary[key] = _clean_nested(derived.get(key))

    hole_groups = []
    hole_group_list = derived.get("hole_groups")
    if isinstance(hole_group_list, Sequence):
        for group in hole_group_list[:12]:
            if not isinstance(group, Mapping):
                continue
            cleaned_entry = {
                "count": to_int(group.get("count")) or 0,
                "tap": bool(group.get("tap")),
                "cbore": bool(group.get("cbore")),
                "csk": bool(group.get("csk")),
                "dia_mm": to_float(group.get("dia_mm")),
                "depth_mm": to_float(group.get("depth_mm")),
            }
            hole_groups.append(compact_dict(cleaned_entry, drop_values=(None, "")))

    payload = {
        "geo": {
            "material": material_name,
            "thickness_mm": thickness_mm,
            "hole_count": hole_count,
            "finish_flags": finish_flags,
            "needs_back_face": needs_back_face,
            "hole_bins_top": hole_bins_top,
            "hole_groups": hole_groups,
        },
        "derived": derived_summary,
        "baseline": _clean_nested(baseline),
        "rates": _clean_nested(rates),
        "bounds": coerce_bounds(bounds),
    }

    return payload


def get_llm_quote_explanation(
    result: Mapping[str, Any],
    model_path: str,
    *,
    debug_enabled: bool = False,
    debug_dir: Path | None = None,
) -> str:
    """Return a single-paragraph explanation of the main cost drivers."""

    breakdown_raw = result.get("breakdown")
    breakdown: Mapping[str, Any] = (
        breakdown_raw if isinstance(breakdown_raw, Mapping) else {}
    )
    totals_raw = breakdown.get("totals")
    totals: Mapping[str, Any] = (
        totals_raw if isinstance(totals_raw, Mapping) else {}
    )
    process_costs_raw = breakdown.get("process_costs")
    process_costs: Mapping[str, Any] = (
        process_costs_raw if isinstance(process_costs_raw, Mapping) else {}
    )
    material_detail_raw = breakdown.get("material")
    material_detail: Mapping[str, Any] = (
        material_detail_raw if isinstance(material_detail_raw, Mapping) else {}
    )
    pass_meta_raw = breakdown.get("pass_meta")
    pass_meta: Mapping[str, Any] = (
        pass_meta_raw if isinstance(pass_meta_raw, Mapping) else {}
    )

    geo_raw = result.get("geo")
    geo: Mapping[str, Any] = geo_raw if isinstance(geo_raw, Mapping) else {}
    ui_vars_raw = result.get("ui_vars")
    ui_vars: Mapping[str, Any] = (
        ui_vars_raw if isinstance(ui_vars_raw, Mapping) else {}
    )

    final_price = _safe_float(result.get("price"), _safe_float(totals.get("price")))
    declared_labor_cost = _safe_float(totals.get("labor_cost"))
    labor_cost_rendered_val = breakdown.get(
        "labor_cost_rendered", declared_labor_cost
    )
    labor_cost = _safe_float(labor_cost_rendered_val, declared_labor_cost)
    direct_costs = _safe_float(
        breakdown.get("total_direct_costs"), _safe_float(totals.get("direct_costs"))
    )
    subtotal = _safe_float(
        breakdown.get("ladder_subtotal"), labor_cost + direct_costs
    )

    def _to_float(value):
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                return float(cleaned)
        except Exception:
            return None
        return None

    proc_pairs = []
    for key, value in (process_costs or {}).items():
        val = _to_float(value)
        if val is None:
            continue
        proc_pairs.append((str(key), val))
    proc_pairs.sort(key=lambda kv: kv[1], reverse=True)
    proc_top = proc_pairs[:3]

    def _to_int(value):
        try:
            if isinstance(value, bool):
                return int(value)
            return int(float(value))
        except Exception:
            return None

    hole_count = _to_int((geo or {}).get("hole_count"))

    thickness_in_val = _to_float(geo.get("thickness_in"))
    thickness_mm = _to_float(geo.get("thickness_mm"))
    if thickness_in_val is None and thickness_mm is not None:
        try:
            thickness_in_val = float(thickness_mm) / 25.4
        except Exception:
            thickness_in_val = None
    ui_thickness_in = _to_float(ui_vars.get("Thickness (in)"))
    if thickness_in_val is None and ui_thickness_in is not None:
        thickness_in_val = float(ui_thickness_in)
    thickness_in = round(float(thickness_in_val), 2) if thickness_in_val is not None else None

    material_name = geo.get("material") or ui_vars.get("Material")
    if isinstance(material_name, str):
        material_name = material_name.strip() or None

    material_display = material_name or DEFAULT_MATERIAL_DISPLAY

    hole_count_val = hole_count if isinstance(hole_count, int) else None
    geo_notes_default: list[str] = []
    try:
        thickness_note_val = float(thickness_in_val) if thickness_in_val is not None else None
    except Exception:
        thickness_note_val = None
    if hole_count_val and thickness_note_val and material_display:
        geo_notes_default = [f"{hole_count_val} holes in {thickness_note_val:.2f} in {material_display}"]

    material_source = (
        result.get("material_source")
        or material_detail.get("source")
        or (pass_meta.get("Material", {}) or {}).get("basis")
        or "shop defaults"
    )

    effective_scrap = material_detail.get("scrap_pct")
    if effective_scrap is None:
        effective_scrap = breakdown.get("scrap_pct")
    effective_scrap = normalize_scrap_pct(effective_scrap)
    scrap_pct_percent = round(100.0 * effective_scrap, 1)

    ctx: dict[str, Any] = {
        "purpose": "quote_explanation",
        "rollup": {
            "final_price": final_price,
            "labor_cost": labor_cost,
            "direct_costs": direct_costs,
            "subtotal": subtotal,
            "labor_pct": round(100.0 * labor_cost / subtotal, 1) if subtotal else 0.0,
            "directs_pct": round(100.0 * direct_costs / subtotal, 1) if subtotal else 0.0,
            "top_processes": [
                {"name": name.replace("_", " "), "usd": val}
                for name, val in proc_top
            ],
        },
        "geo_summary": {
            "hole_count": hole_count,
            "thickness_in": thickness_in,
            "material": material_name,
        },
        "geo_notes": geo_notes_default,
        "material_source": material_source,
        "scrap_pct": scrap_pct_percent,
    }

    def _render_explanation(parsed: dict | None = None, raw_text: str = "") -> str:
        data = parsed if isinstance(parsed, dict) else {}
        if not data and raw_text:
            data = parse_llm_json(raw_text) or {}

        fallback_drivers = [
            {
                "label": "Labor",
                "usd": labor_cost,
                "pct_of_subtotal": round(100.0 * labor_cost / subtotal, 1) if subtotal else 0.0,
            },
            {
                "label": "Directs",
                "usd": direct_costs,
                "pct_of_subtotal": round(100.0 * direct_costs / subtotal, 1) if subtotal else 0.0,
            },
        ]

        drivers_source = data.get("drivers")
        drivers_raw = drivers_source if isinstance(drivers_source, list) else []

        class _DriverEntry(dict):
            label: str
            usd: float
            pct_of_subtotal: float

        drivers: list[_DriverEntry] = []
        for idx, default in enumerate(fallback_drivers):
            raw_entry = drivers_raw[idx] if idx < len(drivers_raw) else None
            entry = raw_entry if isinstance(raw_entry, Mapping) else {}
            label = str(entry.get("label") or default["label"]).strip() or default["label"]
            usd_val = _to_float(entry.get("usd"))
            pct_val = _to_float(entry.get("pct_of_subtotal"))
            drivers.append(
                {
                    "label": label,
                    "usd": float(usd_val) if usd_val is not None else float(default["usd"]),
                    "pct_of_subtotal": float(pct_val)
                    if pct_val is not None
                    else float(default["pct_of_subtotal"]),
                }
            )
        default_driver: _DriverEntry = {"label": "Labor", "usd": 0.0, "pct_of_subtotal": 0.0}
        while len(drivers) < 2:
            drivers.append(
                {
                    "label": f"Bucket {len(drivers) + 1}",
                    "usd": 0.0,
                    "pct_of_subtotal": 0.0,
                }
            )

        driver_primary = drivers[0] if drivers else default_driver
        driver_secondary = drivers[1] if len(drivers) > 1 else driver_primary

        geo_notes_source = data.get("geo_notes")
        geo_notes_raw = geo_notes_source if isinstance(geo_notes_source, list) else []
        geo_notes = [str(note).strip() for note in geo_notes_raw if str(note).strip()]
        if not geo_notes:
            ctx_geo_notes = ctx.get("geo_notes")
            default_notes = ctx_geo_notes if isinstance(ctx_geo_notes, list) else []
            if default_notes:
                geo_notes = [str(note).strip() for note in default_notes if str(note).strip()]
        if not geo_notes:
            geo_summary_ctx = ctx.get("geo_summary")
            summary = geo_summary_ctx if isinstance(geo_summary_ctx, Mapping) else {}
            hc = summary.get("hole_count")
            thk = summary.get("thickness_in")
            mat = summary.get("material")
            if hc and thk and mat:
                try:
                    geo_notes = [f"{int(hc)} holes in {float(thk):.2f} in {mat}"]
                except Exception:
                    geo_notes = []

        top_processes_source = data.get("top_processes")
        top_processes_raw = top_processes_source if isinstance(top_processes_source, list) else []

        top_processes: list[dict[str, Any]] = []
        for entry in top_processes_raw:
            if not isinstance(entry, Mapping):
                continue
            name_val = str(entry.get("name") or "").strip()
            usd_val = _to_float(entry.get("usd"))
            if name_val and usd_val is not None:
                top_processes.append({"name": name_val, "usd": float(usd_val)})
        if not top_processes:
            rollup_ctx = ctx.get("rollup")
            rollup = rollup_ctx if isinstance(rollup_ctx, Mapping) else {}
            rollup_processes = rollup.get("top_processes")
            if isinstance(rollup_processes, list):
                for proc in rollup_processes:
                    if not isinstance(proc, Mapping):
                        continue
                    name_val = str(proc.get("name") or "").strip()
                    usd_val = _to_float(proc.get("usd"))
                    if name_val and usd_val is not None:
                        top_processes.append({"name": name_val, "usd": float(usd_val)})

        material_section_obj = data.get("material")
        material_section = material_section_obj if isinstance(material_section_obj, Mapping) else {}
        scrap_val = _to_float(material_section.get("scrap_pct"))
        if scrap_val is None:
            scrap_fallback = ctx.get("scrap_pct")
            scrap_val = _to_float(scrap_fallback)
            if scrap_val is None and isinstance(scrap_fallback, (int, float)):
                scrap_val = float(scrap_fallback)
            if scrap_val is None:
                scrap_val = 0.0
        material_struct = {
            "source": str(material_section.get("source") or material_source).strip()
            or material_source,
            "scrap_pct": round(float(scrap_val), 1),
        }

        explanation_raw = data.get("explanation")
        explanation = explanation_raw.strip() if isinstance(explanation_raw, str) else ""

        if not explanation:
            top_text = ""
            if top_processes:
                top_bits = [
                    f"{proc['name']} ${proc['usd']:.0f}"
                    for proc in top_processes
                    if proc.get("name") and _to_float(proc.get("usd")) is not None
                ]
                if top_bits:
                    top_text = "Top processes: " + ", ".join(top_bits) + ". "

            geo_text = ""
            if geo_notes:
                geo_text = "Geometry: " + ", ".join(geo_notes) + ". "

            scrap_display = cast(float | None, material_struct.get("scrap_pct"))
            if scrap_display is None:
                scrap_display = _to_float(ctx.get("scrap_pct"))
            if scrap_display is None:
                scrap_display = 0.0
            try:
                scrap_str = f"{float(scrap_display):.1f}"
            except Exception:
                scrap_str = str(scrap_display)

            material_text = ""
            source_val = material_struct.get("source")
            if source_val:
                material_text = f"Material via {source_val}; scrap {scrap_str}% applied."
            else:
                material_text = f"Scrap {scrap_str}% applied."

            explanation = (
                f"Labor ${labor_cost:.2f} ({driver_primary['pct_of_subtotal']:.1f}%) and directs "
                f"${direct_costs:.2f} ({driver_secondary['pct_of_subtotal']:.1f}%) drive cost. "
                + top_text
                + geo_text
                + material_text
            ).strip()

        return explanation

    if not model_path:
        return _render_explanation()

    if LLMClient is None or not isinstance(LLMClient, type):
        return _render_explanation()

    try:
        client = LLMClient(
            model_path,
            debug_enabled=debug_enabled,
            debug_dir=debug_dir,
        )
    except Exception:
        return _render_explanation()

    try:
        system_prompt = (
            "You are a manufacturing estimator. Using ONLY the provided fields, produce a concise JSON explanation.\n"
            "Do not invent numbers. Mention the biggest cost buckets and key geometry drivers if present.\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "explanation": "…1–3 sentences…",\n'
            '  "drivers": [\n'
            '    {"label":"Labor","usd": <number>,"pct_of_subtotal": <number>},\n'
            '    {"label":"Directs","usd": <number>,"pct_of_subtotal": <number>}\n'
            "  ],\n"
            '  "top_processes": [{"name":"drilling","usd": <number>}],\n'
            '  "geo_notes": ["<e.g., 163 holes in 0.25 in steel>"],\n'
            '  "material": {"source":"<string>","scrap_pct": <number>}\n'
            "}"
        )

        user_prompt = "```json\n" + jdump(ctx, sort_keys=True) + "\n```"

        parsed, raw_text, _usage = client.ask_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=256,
            context=ctx,
        )
        return _render_explanation(parsed, raw_text)
    except Exception:
        return _render_explanation()
    finally:
        try:
            client.close()
        except Exception:
            pass
