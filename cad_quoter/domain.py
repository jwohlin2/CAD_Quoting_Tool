"""Domain utilities and thin proxies for UI-bound helpers."""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Mapping as _MappingABC
from types import MappingProxyType
from typing import Any, Callable, Mapping, TYPE_CHECKING, cast

from cad_quoter.coerce import to_float, to_int
from cad_quoter.config import logger
from cad_quoter.domain_models import DEFAULT_MATERIAL_DISPLAY, QuoteState
from cad_quoter.utils import _first_non_none, compact_dict

if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from appV5 import (  # pylint: disable=unused-import
        compute_effective_state as _compute_effective_state,
        effective_to_overrides as _effective_to_overrides,
        merge_effective as _merge_effective,
        reprice_with_effective as _reprice_with_effective,
    )

__all__ = [
    "QuoteState",
    "merge_effective",
    "compute_effective_state",
    "effective_to_overrides",
    "overrides_to_suggestions",
    "suggestions_to_overrides",
    "reprice_with_effective",
    "HARDWARE_PASS_LABEL",
    "LEGACY_HARDWARE_PASS_LABEL",
    "_canonical_pass_label",
    "_as_float_or_none",
    "canonicalize_pass_through_map",
    "coerce_bounds",
    "get_llm_bound_defaults",
    "LLM_BOUND_DEFAULTS",
    "build_suggest_payload",
]


def _app_module():
    """Return the lazily-imported :mod:`appV5` module."""

    module = sys.modules.get("appV5")
    if module is None:
        module = importlib.import_module("appV5")
    return module


def merge_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.merge_effective` for test visibility."""

    app = _app_module()
    return app.merge_effective(*args, **kwargs)


def compute_effective_state(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.compute_effective_state` for test visibility."""

    app = _app_module()
    return app.compute_effective_state(*args, **kwargs)


def reprice_with_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.reprice_with_effective` for test visibility."""

    app = _app_module()
    return app.reprice_with_effective(*args, **kwargs)


def effective_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.effective_to_overrides` for test visibility."""

    app = _app_module()
    return app.effective_to_overrides(*args, **kwargs)


HARDWARE_PASS_LABEL = "Hardware"
LEGACY_HARDWARE_PASS_LABEL = "Hardware / BOM"
_HARDWARE_LABEL_ALIASES = {
    HARDWARE_PASS_LABEL.lower(),
    LEGACY_HARDWARE_PASS_LABEL.lower(),
    "hardware/bom",
    "hardware bom",
}


def _canonical_pass_label(label: str | None) -> str:
    name = str(label or "").strip()
    if name.lower() in _HARDWARE_LABEL_ALIASES:
        return HARDWARE_PASS_LABEL
    return name


def _canonicalize_pass_through_map(data: Any) -> dict[str, float]:
    """Normalize a pass-through dictionary into ``{label: float}``."""

    result: dict[str, float] = {}

    def _add(label: Any, amount: Any) -> None:
        key = _canonical_pass_label(label)
        try:
            val = to_float(amount)
        except Exception:  # pragma: no cover - defensive
            val = None
        if key and val is not None and math.isfinite(float(val)):
            result[key] = result.get(key, 0.0) + float(val)

    if isinstance(data, _MappingABC):
        for key, value in data.items():
            if isinstance(value, _MappingABC):
                inner = value
                amount = inner.get("amount", inner.get("value", inner.get("cost", inner.get("price"))))
                _add(key, amount)
            else:
                _add(key, value)
        return result

    if isinstance(data, (list, tuple)):
        for entry in data:
            if isinstance(entry, _MappingABC):
                label = entry.get("label") or entry.get("name") or entry.get("key") or entry.get("type")
                amount = entry.get("amount", entry.get("value", entry.get("cost", entry.get("price"))))
                if label is None and len(entry) == 1:
                    key = next(iter(entry.keys()))
                    _add(key, entry.get(key))
                else:
                    _add(label, amount)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                _add(entry[0], entry[1])
        return result

    return result


def canonicalize_pass_through_map(data: Any) -> dict[str, float]:
    """Return a canonicalized pass-through map with defensive fallback."""

    canonicalizer_obj = globals().get("_canonicalize_pass_through_map")
    if callable(canonicalizer_obj):
        canonicalizer = cast(Callable[[Any], dict[str, float]], canonicalizer_obj)
        try:
            return canonicalizer(data)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to canonicalize pass-through map; using fallback")

    result: dict[str, float] = {}

    def _add(label: Any, amount: Any) -> None:
        key = _canonical_pass_label(label)
        try:
            val = float(amount)
        except Exception:  # pragma: no cover - defensive
            return
        if key and math.isfinite(val):
            result[key] = result.get(key, 0.0) + float(val)

    if isinstance(data, _MappingABC):
        for key, value in data.items():
            if isinstance(value, _MappingABC):
                inner = value
                amount = inner.get("amount") or inner.get("value") or inner.get("cost") or inner.get("price")
                _add(key, amount)
            else:
                _add(key, value)
    elif isinstance(data, (list, tuple)):
        for entry in data:
            if isinstance(entry, _MappingABC):
                label = entry.get("label") or entry.get("name") or entry.get("key") or entry.get("type")
                amount = entry.get("amount") or entry.get("value") or entry.get("cost") or entry.get("price")
                if label is None and len(entry) == 1:
                    key = next(iter(entry.keys()))
                    _add(key, entry.get(key))
                else:
                    _add(label, amount)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                _add(entry[0], entry[1])

    return result


LLM_MULTIPLIER_MIN = 0.25
LLM_MULTIPLIER_MAX = 4.0
LLM_ADDER_MAX = 8.0


def _as_float_or_none(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            return float(cleaned)
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def coerce_bounds(bounds: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize LLM bounds into a canonical structure."""

    if bounds is None:
        bounds_map: Mapping[str, Any] = {}
    else:
        bounds_map = bounds

    mult_min = _as_float_or_none(bounds_map.get("mult_min"))
    if mult_min is None:
        mult_min = LLM_MULTIPLIER_MIN
    else:
        mult_min = max(LLM_MULTIPLIER_MIN, float(mult_min))

    mult_max = _as_float_or_none(bounds_map.get("mult_max"))
    if mult_max is None:
        mult_max = LLM_MULTIPLIER_MAX
    else:
        mult_max = min(LLM_MULTIPLIER_MAX, float(mult_max))
    mult_max = max(mult_max, mult_min)

    adder_min = _as_float_or_none(bounds_map.get("adder_min_hr"))
    if adder_min is None:
        adder_min = _as_float_or_none(bounds_map.get("add_hr_min"))
    adder_min = max(0.0, float(adder_min)) if adder_min is not None else 0.0

    adder_max = _as_float_or_none(bounds_map.get("adder_max_hr"))
    add_hr_cap = _as_float_or_none(bounds_map.get("add_hr_max"))
    if adder_max is None and add_hr_cap is not None:
        adder_max = float(add_hr_cap)
    elif adder_max is not None and add_hr_cap is not None:
        adder_max = min(float(adder_max), float(add_hr_cap))
    if adder_max is None:
        adder_max = LLM_ADDER_MAX
    adder_max = max(adder_min, min(LLM_ADDER_MAX, float(adder_max)))

    scrap_min = _as_float_or_none(bounds_map.get("scrap_min"))
    scrap_min = max(0.0, float(scrap_min)) if scrap_min is not None else 0.0

    scrap_max = _as_float_or_none(bounds_map.get("scrap_max"))
    scrap_max = float(scrap_max) if scrap_max is not None else 0.25
    scrap_max = max(scrap_max, scrap_min)

    bucket_caps_raw = bounds_map.get("adder_bucket_max") or bounds_map.get("add_hr_bucket_max")
    bucket_caps: dict[str, float] = {}
    if isinstance(bucket_caps_raw, _MappingABC):
        for key, raw in bucket_caps_raw.items():
            cap_val = _as_float_or_none(raw)
            if cap_val is None:
                continue
            bucket_caps[str(key).lower()] = max(adder_min, min(adder_max, float(cap_val)))

    return {
        "mult_min": mult_min,
        "mult_max": mult_max,
        "adder_min_hr": adder_min,
        "adder_max_hr": adder_max,
        "scrap_min": scrap_min,
        "scrap_max": scrap_max,
        "adder_bucket_max": bucket_caps,
    }


def _default_llm_bounds_dict() -> dict[str, Any]:
    """Return the sanitized default LLM guardrail bounds."""

    return coerce_bounds({})


def get_llm_bound_defaults() -> dict[str, Any]:
    """Return a mutable copy of the default LLM guardrail bounds."""

    try:
        base = LLM_BOUND_DEFAULTS  # type: ignore[name-defined]
    except NameError:  # pragma: no cover - defensive
        base = MappingProxyType(_default_llm_bounds_dict())
    if not isinstance(base, _MappingABC):
        base = MappingProxyType(coerce_bounds(base))
    return dict(base)


LLM_BOUND_DEFAULTS: Mapping[str, Any] = MappingProxyType(_default_llm_bounds_dict())


def _ensure_llm_bound_defaults_initialized() -> Mapping[str, Any]:
    """Return a mapping of LLM guardrail defaults, rebuilding when missing."""

    base = globals().get("LLM_BOUND_DEFAULTS")
    if isinstance(base, _MappingABC):
        return base
    if base is None:
        rebuilt = _default_llm_bounds_dict()
    else:
        try:
            rebuilt = coerce_bounds(base if isinstance(base, _MappingABC) else {})
        except Exception:  # pragma: no cover - defensive
            rebuilt = _default_llm_bounds_dict()
    mapping = MappingProxyType(rebuilt)
    globals()["LLM_BOUND_DEFAULTS"] = mapping
    return mapping


_ensure_llm_bound_defaults_initialized()


def build_suggest_payload(
    geo: dict | None,
    baseline: dict | None,
    rates: dict | None,
    bounds: dict | None,
) -> dict:
    """Assemble the JSON payload passed to the suggestion LLM."""

    geo = geo or {}
    baseline = baseline or {}
    rates = rates or {}
    bounds = bounds or {}

    derived = geo.get("derived") or {}

    def _clean_nested(value: Any, depth: int = 0, max_depth: int = 3, limit: int = 24):
        if depth >= max_depth:
            return None
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for idx, (key, val) in enumerate(value.items()):
                if idx >= limit:
                    break
                cleaned_val = _clean_nested(val, depth + 1, max_depth, limit)
                if cleaned_val is not None:
                    cleaned[str(key)] = cleaned_val
            return cleaned
        if isinstance(value, (list, tuple, set)):
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
    if isinstance(hole_bins, dict):
        sorted_bins = sorted(
            ((str(k), to_int(v) or 0) for k, v in hole_bins.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        hole_bins_top = {k: int(v) for k, v in sorted_bins[:8] if v}

    raw_thickness = geo.get("thickness_mm")
    thickness_candidates: list[float | None] = []
    if isinstance(raw_thickness, dict):
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
    if isinstance(material_val, dict):
        material_name = (
            material_val.get("name")
            or material_val.get("display")
            or material_val.get("material")
        )
    else:
        material_name = material_val
    material_name = (
        str(material_name).strip() if material_name else DEFAULT_MATERIAL_DISPLAY
    )

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
    if isinstance(raw_finish, (list, tuple, set)):
        finish_flags = [str(flag).strip().upper() for flag in raw_finish if str(flag).strip()]

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
        "hole_table_source",
    ):
        value = derived.get(key)
        if value in (None, ""):
            continue
        if key in {"tap_minutes_hint", "cbore_minutes_hint", "csk_minutes_hint"}:
            cleaned = to_float(value)
        elif key in {"npt_qty", "slot_count"}:
            cleaned = to_int(value)
        elif key in {"has_ldr_notes"}:
            cleaned = bool(value)
        else:
            cleaned = _clean_nested(value, limit=12)
        if cleaned not in (None, "", [], {}):
            derived_summary[key] = cleaned

    if finish_flags and "finish_flags" not in derived_summary:
        derived_summary["finish_flags"] = finish_flags

    bbox_mm = _clean_nested(geo.get("bbox_mm"), limit=6)

    hole_groups: list[dict[str, Any]] = []
    raw_groups = geo.get("hole_groups")
    if isinstance(raw_groups, (list, tuple)):
        for idx, entry in enumerate(raw_groups):
            if idx >= 12:
                break
            if not isinstance(entry, dict):
                continue
            cleaned_entry = {
                "dia_mm": to_float(entry.get("dia_mm")),
                "depth_mm": to_float(entry.get("depth_mm")),
                "through": bool(entry.get("through")) if entry.get("through") is not None else None,
                "count": to_int(entry.get("count")),
            }
            hole_groups.append(compact_dict(cleaned_entry, drop_values=(None, "")))

    geo_notes: list[str] = []
    raw_notes = geo.get("notes")
    if isinstance(raw_notes, (list, tuple, set)):
        geo_notes = [str(note).strip() for note in raw_notes if str(note).strip()][:8]

    gdt_counts: dict[str, int] = {}
    raw_gdt = geo.get("gdt")
    if isinstance(raw_gdt, dict):
        for key, value in raw_gdt.items():
            val = to_int(value)
            if val:
                gdt_counts[str(key)] = val

    baseline_hours_raw = baseline.get("process_hours") if isinstance(baseline.get("process_hours"), dict) else {}
    baseline_hours: dict[str, float] = {}
    for proc, hours in (baseline_hours_raw or {}).items():
        val = to_float(hours)
        if val is not None:
            baseline_hours[str(proc)] = float(val)

    baseline_pass_raw = baseline.get("pass_through") if isinstance(baseline.get("pass_through"), dict) else {}
    baseline_pass = canonicalize_pass_through_map(baseline_pass_raw)

    top_process_hours = sorted(baseline_hours.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    top_pass_through = sorted(baseline_pass.items(), key=lambda kv: (-kv[1], kv[0]))[:6]

    baseline_summary = {
        "scrap_pct": to_float(baseline.get("scrap_pct")) or 0.0,
        "setups": to_int(baseline.get("setups")) or 1,
        "fixture": baseline.get("fixture"),
        "process_hours": baseline_hours,
        "pass_through": baseline_pass,
        "top_process_hours": top_process_hours,
        "top_pass_through": top_pass_through,
    }

    rates_of_interest = {
        key: to_float(rates.get(key))
        for key in (
            "MillingRate",
            "TurningRate",
            "WireEDMRate",
            "SinkerEDMRate",
            "SurfaceGrindRate",
            "InspectionRate",
            "FixtureBuildRate",
            "AssemblyRate",
            "PackagingRate",
            "DeburrRate",
            "DrillingRate",
        )
        if rates.get(key) is not None and to_float(rates.get(key)) is not None
    }

    signals: dict[str, Any] = {
        "hole_bins_top": hole_bins_top,
        "tap_qty": tap_qty,
        "cbore_qty": cbore_qty,
        "csk_qty": csk_qty,
        "needs_back_face": needs_back_face,
    }

    for key in (
        "tap_minutes_hint",
        "cbore_minutes_hint",
        "csk_minutes_hint",
        "tap_class_counts",
        "tap_details",
        "npt_qty",
        "stock_guess",
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
    ):
        if key in derived_summary:
            signals[key] = derived_summary[key]

    if gdt_counts:
        signals["gdt_counts"] = gdt_counts

    geo_summary = {
        "material": material_name,
        "thickness_mm": thickness_mm,
        "hole_count": hole_count,
        "finish_flags": finish_flags,
        "needs_back_face": needs_back_face,
        "bbox_mm": bbox_mm,
        "hole_groups": hole_groups,
        "notes": geo_notes,
        "derived": derived_summary,
        "gdt": gdt_counts,
    }

    if geo.get("meta"):
        geo_summary["meta"] = _clean_nested(geo.get("meta"), limit=12)
    if geo.get("provenance"):
        geo_summary["provenance"] = _clean_nested(geo.get("provenance"), limit=12)

    coerced_bounds = coerce_bounds(bounds)
    bounds_summary = {
        "mult_min": coerced_bounds["mult_min"],
        "mult_max": coerced_bounds["mult_max"],
        "adder_max_hr": coerced_bounds["adder_max_hr"],
        "scrap_min": coerced_bounds["scrap_min"],
        "scrap_max": coerced_bounds["scrap_max"],
    }

    seed_extra: dict[str, Any] = {}
    dfm_geo_summary = derived_summary.get("dfm_geo")
    if isinstance(dfm_geo_summary, dict) and dfm_geo_summary:
        dfm_bits: list[str] = []
        min_wall = to_float(dfm_geo_summary.get("min_wall_mm"))
        if min_wall is not None:
            dfm_bits.append(f"min_wall≈{min_wall:.1f}mm")
        if dfm_geo_summary.get("thin_wall"):
            dfm_bits.append("thin_walls")
        unique_normals = to_int(dfm_geo_summary.get("unique_normals"))
        if unique_normals:
            dfm_bits.append(f"{unique_normals} normals")
        deburr_edge = to_float(dfm_geo_summary.get("deburr_edge_len_mm"))
        if deburr_edge and deburr_edge > 0:
            dfm_bits.append(f"deburr_edge≈{deburr_edge:.0f}mm")
        face_count = to_int(dfm_geo_summary.get("face_count"))
        if face_count and face_count > 0:
            dfm_bits.append(f"{face_count} faces")
        if dfm_bits:
            seed_extra["dfm_summary"] = dfm_bits[:5]

    tol_inputs_summary = derived_summary.get("tolerance_inputs")
    if isinstance(tol_inputs_summary, dict) and tol_inputs_summary:
        tol_labels = [str(k).strip() for k in tol_inputs_summary.keys() if str(k).strip()]
        if tol_labels:
            seed_extra["tolerance_focus"] = sorted(tol_labels)[:6]

    has_tight_tol_seed = derived_summary.get("has_tight_tol")
    if has_tight_tol_seed is not None:
        seed_extra["has_tight_tol"] = bool(has_tight_tol_seed)

    default_tol_note = derived_summary.get("default_tolerance_note")
    if isinstance(default_tol_note, str) and default_tol_note.strip():
        seed_extra["default_tolerance_note"] = default_tol_note.strip()[:160]

    if derived_summary.get("fai_required") is not None:
        seed_extra["fai_required"] = bool(derived_summary.get("fai_required"))

    stock_catalog_summary = derived_summary.get("stock_catalog")
    stock_focus: list[str] = []
    if isinstance(stock_catalog_summary, dict):
        for entry in stock_catalog_summary.values():
            label = None
            if isinstance(entry, dict):
                label = entry.get("item") or entry.get("name") or entry.get("stock")
            elif isinstance(entry, str):
                label = entry
            if label:
                label = str(label).strip()
            if label:
                stock_focus.append(label)
        if not stock_focus:
            stock_focus = [str(k).strip() for k in stock_catalog_summary.keys() if str(k).strip()]
    elif isinstance(stock_catalog_summary, (list, tuple, set)):
        for entry in stock_catalog_summary:
            label = str(entry).strip()
            if label:
                stock_focus.append(label)
    if stock_focus:
        seed_extra["stock_focus"] = stock_focus[:6]

    seed = {
        "top_process_hours": top_process_hours,
        "top_pass_through": top_pass_through,
        "hole_count": hole_count,
        "setups": baseline_summary["setups"],
        "finish_flags": finish_flags,
    }
    if seed_extra:
        seed.update(seed_extra)

    payload = {
        "geo": geo_summary,
        "baseline": baseline_summary,
        "signals": signals,
        "rates": rates_of_interest,
        "bounds": bounds_summary,
        "seed": seed,
    }

    return payload


def overrides_to_suggestions(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.overrides_to_suggestions` for test visibility."""

    return _overrides_to_suggestions(*args, **kwargs)


def suggestions_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.suggestions_to_overrides` for test visibility."""

    return _suggestions_to_overrides(*args, **kwargs)

