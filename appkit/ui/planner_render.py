from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict
from collections.abc import Iterable, Mapping as _MappingABC, MutableMapping as _MutableMappingABC, Sequence

from cad_quoter.config import logger
from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.utils import sdict
from cad_quoter.utils.render_utils import fmt_hours, fmt_money

from appkit.utils.text_rules import canonicalize_amortized_label as _canonical_amortized_label

from .services import QuoteConfiguration


PROGRAMMING_PER_PART_LABEL = "Programming (per part)"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


_HIDE_IN_BUCKET_VIEW: frozenset[str] = frozenset({*PLANNER_META, "misc"})
_PREFERRED_BUCKET_VIEW_ORDER: tuple[str, ...] = (
    "programming",
    "programming_amortized",
    "fixture_build",
    "fixture_build_amortized",
    "milling",
    "drilling",
    "counterbore",
    "countersink",
    "tapping",
    "grinding",
    "finishing_deburr",
    "saw_waterjet",
    "wire_edm",
    "sinker_edm",
    "inspection",
    "assembly",
    "toolmaker_support",
    "packaging",
    "ehs_compliance",
    "turning",
    "lapping_honing",
)

CANON_MAP: dict[str, str] = {
    "deburr": "finishing_deburr",
    "deburring": "finishing_deburr",
    "finish_deburr": "finishing_deburr",
    "finishing_deburr": "finishing_deburr",
    "finishing": "finishing_deburr",
    "finishing/deburr": "finishing_deburr",
    "inspection": "inspection",
    "milling": "milling",
    "drilling": "drilling",
    "deep_drill": "drilling",
    "counterbore": "counterbore",
    "tapping": "tapping",
    "grinding": "grinding",
    "wire_edm": "wire_edm",
    "wire_edm_windows": "wire_edm",
    "wire_edm_outline": "wire_edm",
    "wire_edm_open_id": "wire_edm",
    "wire_edm_cam_slot_or_profile": "wire_edm",
    "wire_edm_id_leave": "wire_edm",
    "wedm": "wire_edm",
    "wireedm": "wire_edm",
    "wire-edm": "wire_edm",
    "wire edm": "wire_edm",
    "sinker_edm": "sinker_edm",
    "sinker_edm_finish_burn": "sinker_edm",
    "sinker": "sinker_edm",
    "ram_edm": "sinker_edm",
    "ramedm": "sinker_edm",
    "ram-edm": "sinker_edm",
    "ram edm": "sinker_edm",
    "saw_waterjet": "saw_waterjet",
    "fixture_build_amortized": "fixture_build_amortized",
    "programming_amortized": "programming_amortized",
    "misc": "misc",
}

PLANNER_META: frozenset[str] = frozenset({"planner_labor", "planner_machine", "planner_total"})

# ---------- Bucket & Operation Roles ----------
BUCKET_ROLE: dict[str, str] = {
    "programming": "labor_only",
    "inspection": "labor_only",
    "deburr": "labor_only",
    "deburring": "labor_only",
    "finishing_deburr": "labor_only",
    "finishing": "labor_only",
    "finishing/deburr": "labor_only",

    "drilling": "split",
    "milling": "split",
    "grinding": "split",
    "sinker_edm": "split",

    "wedm": "machine_only",
    "wire_edm": "machine_only",
    "waterjet": "machine_only",
    "saw": "machine_only",
    "saw_waterjet": "machine_only",

    "_default": "machine_only",
}

OP_ROLE: dict[str, str] = {
    "assemble_pair_on_fixture": "labor_only",
    "prep_carrier_or_tab": "labor_only",
    "indicate_hardened_blank": "labor_only",
    "indicate_on_shank": "labor_only",
    "stability_check_after_ops": "labor_only",
    "mark_id": "labor_only",
    "saw_blank": "machine_only",
    "saw_or_mill_rough_blocks": "machine_only",
    "waterjet_or_saw_blanks": "machine_only",
    "face_mill_pre": "split",
    "cnc_rough_mill": "split",
    "cnc_mill_rough": "split",
    "finish_mill_windows": "split",
    "finish_mill_cam_slot_or_profile": "split",
    "spot_drill_all": "split",
    "drill_patterns": "split",
    "interpolate_critical_bores": "split",
    "drill_ream_bore": "split",
    "drill_ream_dowel_press": "split",
    "ream_slip_in_assembly": "split",
    "rigid_tap": "split",
    "thread_mill": "split",
    "drill_or_trepan_id": "split",
    "wire_edm_windows": "machine_only",
    "wire_edm_outline": "machine_only",
    "wire_edm_open_id": "machine_only",
    "wire_edm_cam_slot_or_profile": "machine_only",
    "wire_edm_id_leave": "machine_only",
    "machine_electrode": "labor_only",
    "sinker_edm_finish_burn": "split",
    "blanchard_grind_pre": "split",
    "surface_grind_faces": "split",
    "surface_grind_datums": "split",
    "surface_or_profile_grind_bearing": "split",
    "surface_or_profile_grind_od_cleanup": "split",
    "profile_or_surface_grind_wear_faces": "split",
    "profile_grind_pilot_od_to_tir": "split",
    "profile_grind_flanks_and_reliefs_to_spec": "split",
    "jig_bore_or_jig_grind_coaxial_bores": "split",
    "jig_grind_id_to_size_and_roundness": "split",
    "jig_grind_id_to_tenths_and_straightness": "split",
    "light_grind_cleanup": "split",
    "match_grind_set_for_gap_and_parallelism": "split",
    "turn_or_mill_od": "split",
    "purchase_od_ground_blank": "outsourced",
    "lap_bearing_land": "labor_only",
    "lap_id": "labor_only",
    "lap_edges": "labor_only",
    "hone_edge": "labor_only",
    "edge_break": "labor_only",
    "edge_prep": "labor_only",
    "heat_treat": "outsourced",
    "heat_treat_to_spec": "outsourced",
    "heat_treat_if_wear_part": "outsourced",
    "apply_coating": "outsourced",
    "clean_degas_for_coating": "labor_only",
    "start_ground_carbide_blank": "outsourced",
    "start_ground_carbide_ring": "outsourced",
    "verify_connected_passage_and_masking": "labor_only",
    "abrasive_flow_polish": "outsourced",
    "clean_and_flush_media": "labor_only",
}


def _bucket_role_for_key(key: str) -> str:
    canon = _canonical_bucket_key(key) or _normalize_bucket_key(key) or ""
    return BUCKET_ROLE.get(canon, BUCKET_ROLE["_default"])


def _op_role_for_name(name: str) -> str:
    return OP_ROLE.get((name or "").strip(), "machine_only")
_HIDE_IN_BUCKET_VIEW: frozenset[str] = frozenset({*PLANNER_META, "misc"})
_PREFERRED_BUCKET_VIEW_ORDER: tuple[str, ...] = (
    "programming",
    "programming_amortized",
    "fixture_build",
    "fixture_build_amortized",
    "milling",
    "drilling",
    "counterbore",
    "countersink",
    "tapping",
    "grinding",
    "finishing_deburr",
    "saw_waterjet",
    "wire_edm",
    "sinker_edm",
    "inspection",
    "assembly",
    "toolmaker_support",
    "packaging",
    "ehs_compliance",
    "turning",
    "lapping_honing",
)

def _normalize_bucket_key(name: str | None) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    aliases = {
        "finishing": "finishing_deburr",
        "deburring": "finishing_deburr",
        "finish_deburr": "finishing_deburr",
        "finishing_deburr": "finishing_deburr",
        "deburr": "finishing_deburr",
        "saw": "saw_waterjet",
        "saw_waterjet": "saw_waterjet",
        "waterjet": "saw_waterjet",
        "counter_bore": "counterbore",
        "counter_bores": "counterbore",
        "counterbore": "counterbore",
        "counter_sink": "countersink",
        "counter_sinks": "countersink",
        "countersink": "countersink",
        "thread_mill": "tapping",
    }
    return aliases.get(text, text)

def _rate_key_for_bucket(bucket: str | None) -> str | None:
    canon = _normalize_bucket_key(bucket)
    mapping = {
        "milling": "MillingRate",
        # Policy: price drilling as labor-only, using the shop's milling labor rate
        # so that Process and Bucket sections reconcile consistently.
        "drilling": "MillingRate",
        "counterbore": "DrillingRate",
        "countersink": "DrillingRate",
        "tapping": "TappingRate",
        "grinding": "SurfaceGrindRate",
        "wire_edm": "WireEDMRate",
        "sinker_edm": "SinkerEDMRate",
        "inspection": "InspectionRate",
        "finishing_deburr": "DeburrRate",
        "assembly": "AssemblyRate",
        "packaging": "PackagingRate",
        "saw_waterjet": "SawWaterjetRate",
        "misc": "MillingRate",
    }
    return mapping.get(canon)


_LABORISH_BUCKET_KEYS: frozenset[str] = frozenset(
    {
        "finishing_deburr",
        "inspection",
        "assembly",
        "toolmaker_support",
        "ehs_compliance",
        "fixture_build_amortized",
        "programming_amortized",
    }
)

_BUCKET_RATE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "milling": ("MillingRate",),
    "drilling": ("DrillingRate", "MillingRate"),
    "counterbore": ("CounterboreRate", "DrillingRate"),
    "countersink": ("CountersinkRate", "DrillingRate"),
    "tapping": ("TappingRate", "DrillingRate"),
    "grinding": (
        "GrindingRate",
        "SurfaceGrindRate",
        "ODIDGrindRate",
        "JigGrindRate",
    ),
    "finishing_deburr": ("FinishingRate", "DeburrRate"),
    "saw_waterjet": ("SawWaterjetRate", "SawRate", "WaterjetRate"),
    "inspection": ("InspectionRate",),
    "wire_edm": ("WireEDMRate", "EDMRate"),
    "sinker_edm": ("SinkerEDMRate", "EDMRate"),
    "lapping_honing": ("LappingRate", "HoningRate"),
}


def _bucket_cost_mode(key: str | None) -> str:
    norm = _normalize_bucket_key(key)
    if norm in _LABORISH_BUCKET_KEYS:
        return "labor"
    return "machine"


def _lookup_bucket_rate(
    bucket_key: str | None, rates: Mapping[str, Any] | None
) -> float:
    if not isinstance(rates, _MappingABC):
        rates_map: Mapping[str, Any] = {}
    else:
        rates_map = rates

    norm = _normalize_bucket_key(bucket_key)
    candidates = _BUCKET_RATE_CANDIDATES.get(norm, ())
    for candidate in candidates:
        rate_val = rates_map.get(candidate)
        if rate_val is None:
            rate_val = rates_map.get(_normalize_bucket_key(candidate))
        coerced = _safe_float(rate_val, default=0.0)
        if coerced > 0:
            return coerced
    canon_key = _canonical_bucket_key(bucket_key)
    canon_norm = _normalize_bucket_key(canon_key)
    if canon_key and canon_norm and canon_norm != norm:
        for candidate in _BUCKET_RATE_CANDIDATES.get(canon_norm, ()): 
            rate_val = rates_map.get(candidate)
            if rate_val is None:
                rate_val = rates_map.get(_normalize_bucket_key(candidate))
            coerced = _safe_float(rate_val, default=0.0)
            if coerced > 0:
                return coerced

    fallback_key = "LaborRate" if _bucket_cost_mode(norm) == "labor" else "MachineRate"
    fallback_val = rates_map.get(fallback_key)
    if fallback_val is None:
        fallback_val = rates_map.get(_normalize_bucket_key(fallback_key))
    coerced_fallback = _safe_float(fallback_val, default=0.0)
    return coerced_fallback if coerced_fallback > 0 else 0.0


@dataclass
class PlannerBucketRenderState:
    canonical_order: list[str] = field(default_factory=list)
    canonical_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    table_rows: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    label_to_canon: dict[str, str] = field(default_factory=dict)
    canon_to_display_label: dict[str, str] = field(default_factory=dict)
    detail_lookup: dict[str, str] = field(default_factory=dict)
    labor_costs_display: dict[str, float] = field(default_factory=dict)
    hour_entries: dict[str, tuple[float, bool]] = field(default_factory=dict)
    display_labor_total: float = 0.0
    display_machine_total: float = 0.0
    bucket_minutes_detail: dict[str, float] = field(default_factory=dict)
    process_costs_for_render: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)


class _BucketOpEntry(TypedDict):
    name: str
    minutes: float


def _split_hours_for_bucket(
    label: str,
    hours: float,
    render_state: "PlannerBucketRenderState | None",
    cfg: QuoteConfiguration | None,
) -> tuple[float, float]:
    total_h = max(0.0, float(hours or 0.0))
    if not cfg or not getattr(cfg, "separate_machine_labor", False):
        return (0.0, total_h)

    canon_label = _canonical_bucket_key(label)
    key = canon_label or _normalize_bucket_key(label)
    if not key:
        key = str(label or "")

    extra: Mapping[str, Any] | None = None
    if render_state is not None:
        extra_candidate = getattr(render_state, "extra", None)
        if isinstance(extra_candidate, _MappingABC):
            extra = extra_candidate
    if extra is None:
        extra = {}

    if canon_label == "drilling" or key == "drilling":
        m_min = _coerce_float_or_none(extra.get("drill_machine_minutes"))
        l_min = _coerce_float_or_none(extra.get("drill_labor_minutes"))
        if (
            m_min is not None
            and l_min is not None
            and (float(m_min) + float(l_min)) > 0.0
        ):
            return (float(m_min) / 60.0, float(l_min) / 60.0)
        return (total_h, 0.0)

    bucket_ops: Mapping[str, Any] | None = None
    if isinstance(extra, _MappingABC):
        bucket_ops_candidate = extra.get("bucket_ops")
        if isinstance(bucket_ops_candidate, _MappingABC):
            bucket_ops = bucket_ops_candidate

    if bucket_ops is not None:
        ops_list = bucket_ops.get(key)
        if isinstance(ops_list, Sequence):
            machine_minutes = 0.0
            labor_minutes = 0.0
            for entry in ops_list:
                if not isinstance(entry, _MappingABC):
                    continue
                name_val = entry.get("name")
                if not isinstance(name_val, str):
                    continue
                role = _op_role_for_name(name_val)
                minutes_val = _coerce_float_or_none(entry.get("minutes"))
                if minutes_val is None or minutes_val <= 0:
                    continue
                if role == "labor_only":
                    labor_minutes += minutes_val
                elif role == "machine_only":
                    machine_minutes += minutes_val
                elif role == "split":
                    machine_minutes += minutes_val
                else:
                    # outsourced / unknown → skip from internal hours
                    continue
            if (machine_minutes + labor_minutes) > 0.0:
                return (machine_minutes / 60.0, labor_minutes / 60.0)

    role = _bucket_role_for_key(key)
    if role == "labor_only":
        return (0.0, total_h)
    if role == "machine_only":
        return (total_h, 0.0)
    if total_h > 0.0:
        return (total_h, 0.0)
    return (0.0, 0.0)


def _build_planner_bucket_render_state(
    bucket_view: Mapping[str, Any] | None,
    *,
    label_overrides: Mapping[str, str] | None = None,
    labor_cost_details: Mapping[str, Any] | None = None,
    labor_cost_details_input: Mapping[str, Any] | None = None,
    process_costs_canon: Mapping[str, float] | None = None,
    rates: Mapping[str, Any] | None = None,
    removal_drilling_hours: float | None = None,
    prefer_removal_drilling_hours: bool = True,
    cfg: QuoteConfiguration | None = None,
    bucket_ops: Mapping[str, typing.Sequence[Mapping[str, Any]]] | None = None,
    drill_machine_minutes: float | None = None,
    drill_labor_minutes: float | None = None,
    drill_total_minutes: float | None = None,
) -> PlannerBucketRenderState:
    state = PlannerBucketRenderState()

    def _flatten_rate_map(value: Any) -> dict[str, float]:
        flat: dict[str, float] = {}
        if not isinstance(value, _MappingABC):
            return flat

        def _walk(container: Mapping[str, Any]) -> None:
            for key, raw in container.items():
                if isinstance(raw, _MappingABC):
                    _walk(raw)
                    continue
                try:
                    numeric = float(raw)
                except Exception:
                    continue
                if numeric > 0:
                    flat[str(key)] = numeric

        _walk(value)
        return flat

    state.rates = _flatten_rate_map(rates)

    # The canonical bucket view is the single source of truth for the Process & Labor table.
    # Start with an empty structure and allow the canonical buckets to populate it below,
    # preventing any stale entries from ``process_costs`` from sneaking into the render.
    state.process_costs_for_render = {}

    if drill_machine_minutes is not None:
        try:
            state.extra["drill_machine_minutes"] = max(0.0, float(drill_machine_minutes))
        except Exception:
            state.extra["drill_machine_minutes"] = drill_machine_minutes
    if drill_labor_minutes is not None:
        try:
            state.extra["drill_labor_minutes"] = max(0.0, float(drill_labor_minutes))
        except Exception:
            state.extra["drill_labor_minutes"] = drill_labor_minutes
    if drill_total_minutes is not None and drill_total_minutes > 0.0:
        try:
            state.extra["drill_total_minutes"] = float(drill_total_minutes)
        except Exception:
            state.extra["drill_total_minutes"] = drill_total_minutes

    bucket_ops_map: dict[str, list[_BucketOpEntry]] = {}

    def _ingest_bucket_ops(source: Any) -> None:
        if isinstance(source, _MappingABC):
            items = source.items()
        else:
            return
        for raw_key, raw_list in items:
            canon_key = _canonical_bucket_key(raw_key) or _normalize_bucket_key(raw_key)
            if not canon_key:
                continue
            entries: list[_BucketOpEntry] = bucket_ops_map.setdefault(canon_key, [])
            if isinstance(raw_list, Sequence):
                for item in raw_list:
                    if not isinstance(item, _MappingABC):
                        continue
                    op_name = (item.get("name") or item.get("op") or "").strip()
                    if not op_name:
                        continue
                    minutes_val = _coerce_float_or_none(item.get("minutes"))
                    if minutes_val is None or minutes_val <= 0:
                        minutes_val = _coerce_float_or_none(item.get("mins"))
                    if minutes_val is None or minutes_val <= 0:
                        continue
                    entries.append(
                        {
                            "name": op_name,
                            "minutes": float(minutes_val),
                        }
                    )

    if isinstance(bucket_view, _MappingABC):
        _ingest_bucket_ops(bucket_view.get("bucket_ops"))
    if isinstance(bucket_ops, _MappingABC):
        _ingest_bucket_ops(bucket_ops)

    if bucket_ops_map:
        for ops in bucket_ops_map.values():
            ops.sort(key=lambda entry: (-float(entry.get("minutes", 0.0) or 0.0), entry.get("name", "")))
        state.extra["bucket_ops"] = bucket_ops_map

    if not isinstance(bucket_view, _MappingABC):
        return state

    try:
        removal_hr = (
            float(removal_drilling_hours) if removal_drilling_hours is not None else None
        )
    except Exception:
        removal_hr = None
    if removal_hr is not None and removal_hr < 0:
        removal_hr = None

    if removal_hr is not None:
        try:
            state.extra["removal_drilling_hours"] = float(removal_hr)
        except Exception:
            state.extra["removal_drilling_hours"] = removal_hr

    buckets = bucket_view.get("buckets") if isinstance(bucket_view, _MappingABC) else None
    if not isinstance(buckets, _MappingABC):
        buckets = {}

    order = bucket_view.get("order") if isinstance(bucket_view, _MappingABC) else None
    if not isinstance(order, Sequence):
        order = _preferred_order_then_alpha(buckets.keys())

    details_map = (
        dict(labor_cost_details)
        if isinstance(labor_cost_details, _MappingABC)
        else {}
    )
    detail_inputs_map = (
        dict(labor_cost_details_input)
        if isinstance(labor_cost_details_input, _MappingABC)
        else {}
    )

    machine_hours_total = 0.0
    labor_hours_total = 0.0

    for canon_key in order:
        info = buckets.get(canon_key)
        if not isinstance(info, _MappingABC):
            continue

        minutes_val = _safe_float(info.get("minutes"), default=0.0)
        labor_raw = _safe_float(info.get("labor$"), default=0.0)
        machine_raw = _safe_float(info.get("machine$"), default=0.0)

        hours_raw = minutes_val / 60.0 if minutes_val else 0.0
        original_hours = hours_raw

        if _canonical_bucket_key(canon_key) == "drilling" and removal_hr is not None:
            if prefer_removal_drilling_hours:
                override_hours = max(0.0, float(removal_hr))
                override_minutes = override_hours * 60.0
                if not math.isclose(original_hours, override_hours, rel_tol=1e-9, abs_tol=1e-6):
                    state.notes["drilling_source"] = "removal_card"
                    state.extra["drilling_hours_override"] = (
                        float(original_hours),
                        float(override_hours),
                    )
                hours_raw = override_hours
                minutes_val = override_minutes
            elif not math.isclose(original_hours, float(removal_hr), rel_tol=1e-9, abs_tol=1e-6):
                state.extra["drilling_hours_override"] = (
                    float(original_hours),
                    float(removal_hr),
                )

        split_machine_hours = 0.0
        split_labor_hours = 0.0
        used_split = False
        if cfg and getattr(cfg, "separate_machine_labor", False):
            split_machine_hours, split_labor_hours = _split_hours_for_bucket(
                canon_key, hours_raw, state, cfg
            )
            total_split_hours = (split_machine_hours or 0.0) + (split_labor_hours or 0.0)
            if total_split_hours > 0.0:
                hours_raw = total_split_hours
                machine_raw = float(split_machine_hours) * float(cfg.machine_rate_per_hr)
                labor_raw = float(split_labor_hours) * float(cfg.labor_rate_per_hr)
                used_split = True

        total_raw = labor_raw + machine_raw

        if total_raw <= 0.01 and minutes_val > 0:
            inferred_rate = _lookup_bucket_rate(canon_key, rates)
            if inferred_rate > 0:
                injected_total = (minutes_val / 60.0) * inferred_rate
                if _bucket_cost_mode(canon_key) == "labor":
                    labor_raw = injected_total
                    machine_raw = 0.0
                else:
                    machine_raw = injected_total
                    labor_raw = 0.0
                total_raw = labor_raw + machine_raw

        if total_raw <= 0.01 and hours_raw <= 0.01:
            continue

        state.canonical_order.append(canon_key)
        state.canonical_summary[canon_key] = {
            "minutes": minutes_val,
            "hours": hours_raw,
            "labor": labor_raw,
            "machine": machine_raw,
            "total": total_raw,
        }

        label = _display_bucket_label(canon_key, label_overrides)
        hours_val = round(hours_raw, 2)
        labor_val = round(labor_raw, 2)
        machine_val = round(machine_raw, 2)
        total_val = round(total_raw, 2)

        state.table_rows.append((label, hours_val, labor_val, machine_val, total_val))
        state.label_to_canon[label] = canon_key
        state.canon_to_display_label.setdefault(canon_key, label)
        state.labor_costs_display[label] = total_val
        state.display_labor_total += labor_raw
        state.display_machine_total += machine_raw
        state.hour_entries[label] = (hours_val, True)

        split_detail_line: str | None = None
        if cfg and getattr(cfg, "separate_machine_labor", False):
            state.extra.setdefault("bucket_hour_split", {})[canon_key] = {
                "machine_hours": round(split_machine_hours, 4),
                "labor_hours": round(split_labor_hours, 4),
            }
            machine_hours_total += split_machine_hours
            labor_hours_total += split_labor_hours
            if used_split or (split_machine_hours > 0.0 or split_labor_hours > 0.0):
                if split_machine_hours > 0.0 and split_labor_hours > 0.0:
                    split_detail_line = (
                        f"machine {split_machine_hours:.2f} hr @ ${cfg.machine_rate_per_hr:.0f}/hr, "
                        f"labor {split_labor_hours:.2f} hr @ ${cfg.labor_rate_per_hr:.0f}/hr"
                    )

        detail_text: str | None = None
        for candidate in (canon_key, label):
            candidate_key = str(candidate)
            if candidate_key in details_map and details_map[candidate_key]:
                detail_text = details_map[candidate_key]
                break
            if (
                candidate_key in detail_inputs_map
                and detail_inputs_map[candidate_key]
            ):
                detail_text = detail_inputs_map[candidate_key]
                break
        if split_detail_line:
            if detail_text not in (None, ""):
                detail_text = f"{detail_text}; {split_detail_line}"
            else:
                detail_text = split_detail_line
        if detail_text not in (None, ""):
            state.detail_lookup[label] = str(detail_text)

    for canon_key, metrics in state.canonical_summary.items():
        state.bucket_minutes_detail[canon_key] = _safe_float(
            metrics.get("minutes"), default=0.0
        )
        state.process_costs_for_render[canon_key] = _safe_float(
            metrics.get("total"), default=0.0
        )
        label = _display_bucket_label(canon_key, label_overrides)
        state.label_to_canon.setdefault(label, canon_key)
        state.canon_to_display_label.setdefault(canon_key, label)

    if cfg and getattr(cfg, "separate_machine_labor", False):
        state.extra["_machine_total_hours"] = round(machine_hours_total, 2)
        state.extra["_labor_total_hours"] = round(labor_hours_total, 2)

    return state


def _display_rate_for_row(
    label: str,
    *,
    cfg: QuoteConfiguration | None,
    render_state: PlannerBucketRenderState | None,
    hours: float | None,
) -> str:
    total_hours = max(0.0, float(hours or 0.0))
    cfg_obj = cfg
    if cfg_obj and getattr(cfg_obj, "separate_machine_labor", True):
        machine_hours, labor_hours = _split_hours_for_bucket(
            label, total_hours, render_state, cfg_obj
        )
        pieces: list[str] = []
        if machine_hours > 0:
            pieces.append(f"mach ${float(cfg_obj.machine_rate_per_hr):.2f}/hr")
        if labor_hours > 0:
            pieces.append(f"labor ${float(cfg_obj.labor_rate_per_hr):.2f}/hr")
        if pieces:
            return " / ".join(pieces)
        fallback_rate = float(getattr(cfg_obj, "labor_rate_per_hr", 0.0) or 0.0)
        if fallback_rate <= 0:
            fallback_rate = float(getattr(cfg_obj, "machine_rate_per_hr", 0.0) or 0.0)
        return f"${fallback_rate:.2f}/hr"

    summary_map: Mapping[str, Any] | None = None
    if isinstance(render_state, PlannerBucketRenderState):
        summary_map = render_state.canonical_summary
    if isinstance(summary_map, _MappingABC):
        canon_key = _canonical_bucket_key(label)
        candidates = [
            canon_key,
            _normalize_bucket_key(label),
            str(label or ""),
        ] if canon_key else [
            _normalize_bucket_key(label),
            str(label or ""),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            metrics = summary_map.get(candidate)
            if not isinstance(metrics, _MappingABC):
                continue
            total_cost = _safe_float(metrics.get("total"), default=0.0)
            hours_val = _safe_float(metrics.get("hours"), default=0.0)
            if hours_val <= 0.0:
                hours_val = total_hours
            if hours_val > 0.0 and total_cost > 0.0:
                return f"${(total_cost / hours_val):.2f}/hr"

    rate_lookup = 0.0
    if isinstance(render_state, PlannerBucketRenderState) and render_state.rates:
        rate_lookup = _lookup_rate(str(label), render_state.rates, fallback=0.0)
        if rate_lookup <= 0.0:
            canon = _canonical_bucket_key(label)
            if canon:
                rate_lookup = _lookup_rate(canon, render_state.rates, fallback=0.0)
    if rate_lookup <= 0.0 and cfg_obj is not None:
        rate_lookup = float(getattr(cfg_obj, "labor_rate_per_hr", 0.0) or 0.0)
        if rate_lookup <= 0.0:
            rate_lookup = float(getattr(cfg_obj, "machine_rate_per_hr", 0.0) or 0.0)
    return f"${rate_lookup:.2f}/hr"


def _sync_drilling_bucket_view(
    bucket_view: Mapping[str, Any] | None,
    *,
    billed_minutes: float,
    billed_cost: float | None = None,
) -> bool:
    """Ensure the drilling bucket mirrors the authoritative planner minutes."""

    if not isinstance(bucket_view, _MutableMappingABC):
        return False

    if billed_minutes <= 0.0:
        return False

    buckets_obj = bucket_view.get("buckets")
    if not isinstance(buckets_obj, _MutableMappingABC):
        if isinstance(bucket_view, dict):
            buckets_obj = bucket_view.setdefault("buckets", {})
        else:
            return False

    entry = buckets_obj.get("drilling")
    if not isinstance(entry, _MutableMappingABC):
        if isinstance(buckets_obj, dict):
            entry = buckets_obj.setdefault("drilling", {})
        else:
            return False

    old_minutes = _safe_float(entry.get("minutes"))
    old_machine = _safe_float(entry.get("machine$"))
    old_labor = _safe_float(entry.get("labor$"))
    old_total = _safe_float(entry.get("total$"))

    new_minutes = round(float(billed_minutes), 2)
    entry["minutes"] = new_minutes
    entry["synced_minutes"] = new_minutes
    entry["synced_hours"] = round(new_minutes / 60.0, 4)

    new_machine = old_machine
    new_labor = old_labor

    if billed_cost is not None and billed_cost > 0.0:
        billed_cost_val = round(float(billed_cost), 2)
        if _bucket_cost_mode("drilling") == "labor":
            new_labor = billed_cost_val
            if new_machine <= 0.0:
                new_machine = 0.0
        else:
            new_machine = billed_cost_val
            if new_labor <= 0.0:
                new_labor = 0.0

    entry["machine$"] = round(new_machine, 2)
    entry["labor$"] = round(new_labor, 2)
    entry["total$"] = round(entry["machine$"] + entry["labor$"], 2)

    totals_map = bucket_view.get("totals")
    if isinstance(totals_map, _MutableMappingABC):
        totals_map["minutes"] = round(
            _safe_float(totals_map.get("minutes")) - old_minutes + new_minutes,
            2,
        )
        totals_map["machine$"] = round(
            _safe_float(totals_map.get("machine$"))
            - old_machine
            + entry["machine$"],
            2,
        )
        totals_map["labor$"] = round(
            _safe_float(totals_map.get("labor$")) - old_labor + entry["labor$"],
            2,
        )
        totals_map["total$"] = round(
            _safe_float(totals_map.get("total$")) - old_total + entry["total$"],
            2,
        )

    return True


def _seed_bucket_minutes(
    breakdown: MutableMapping[str, Any],
    *,
    tapping_min: float = 0.0,
    cbore_min: float = 0.0,
    spot_min: float = 0.0,
    jig_min: float = 0.0,
) -> None:
    bucket_view_obj = breakdown.setdefault("bucket_view", {})
    buckets_obj = bucket_view_obj.setdefault("buckets", {})

    def _ins(name: str, minutes: float) -> None:
        if minutes <= 0:
            return
        entry = buckets_obj.setdefault(
            name,
            {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0},
        )
        entry["minutes"] = float(entry.get("minutes") or 0.0) + float(minutes)

    _ins("tapping", tapping_min)
    _ins("counterbore", cbore_min)
    # spot and jig_grind can roll into "drilling" or "grinding"; keep explicit names if you expose them
    _ins("drilling", spot_min)
    _ins("grinding", jig_min)


def _hole_table_minutes_from_geo(
    geo: Mapping[str, Any] | None,
) -> tuple[float, float, float, float]:
    """Return (tap, cbore, spot, jig) minutes inferred from ``geo``."""

    if not isinstance(geo, _MappingABC):
        return (0.0, 0.0, 0.0, 0.0)

    ops_summary = geo.get("ops_summary") if isinstance(geo, _MappingABC) else None
    if isinstance(ops_summary, _MappingABC):
        totals_map = ops_summary.get("totals")
    else:
        totals_map = None
    if not isinstance(totals_map, _MappingABC):
        totals: Mapping[str, Any] = {}
    else:
        totals = totals_map

    def _ops_total(*keys: str) -> float:
        total = 0.0
        for key in keys:
            total += _safe_float(totals.get(key), 0.0)
        return total

    tap_minutes = _safe_float(geo.get("tap_minutes_hint"), 0.0)
    if tap_minutes <= 0.0:
        details = geo.get("tap_details")
        if isinstance(details, (list, tuple)):
            tap_minutes = 0.0
            for entry in details:
                if isinstance(entry, _MappingABC):
                    tap_minutes += _safe_float(entry.get("total_minutes"), 0.0)
        if tap_minutes <= 0.0:
            tap_count = _ops_total("tap_front", "tap_back")
            if tap_count > 0.0:
                tap_minutes = tap_count * TAP_MINUTES_BY_CLASS.get("medium", 0.3)

    cbore_minutes = _safe_float(geo.get("cbore_minutes_hint"), 0.0)
    if cbore_minutes <= 0.0:
        cbore_qty = _ops_total("cbore_front", "cbore_back")
        if cbore_qty > 0.0:
            cbore_minutes = cbore_qty * CBORE_MIN_PER_SIDE_MIN

    spot_minutes = _ops_total("spot_front", "spot_back") * SPOT_DRILL_MIN_PER_SIDE_MIN

    jig_minutes = _ops_total("jig_grind",) * JIG_GRIND_MIN_PER_FEATURE

    return (
        float(max(tap_minutes, 0.0)),
        float(max(cbore_minutes, 0.0)),
        float(max(spot_minutes, 0.0)),
        float(max(jig_minutes, 0.0)),
    )


def _charged_hours_by_bucket(
    process_costs,
    process_meta,
    rates,
    *,
    render_state: PlannerBucketRenderState | None = None,
    removal_drilling_hours: float | None = None,
    prefer_removal_drilling_hours: bool = True,
    cfg: QuoteConfiguration | None = None,
):
    """Return the hours that correspond to what we actually charged."""
    out: dict[str, float] = {}
    for key, amount in (process_costs or {}).items():
        norm = _normalize_bucket_key(key)
        if norm.startswith("planner_"):
            continue
        # Prefer explicit final hours if meta provided them
        meta_source = process_meta or {}
        meta = (
            meta_source.get(key)
            or meta_source.get(_normalize_bucket_key(key))
            or meta_source.get(_final_bucket_key(key))
            or {}
        )
        hr = meta.get("final_hr") or meta.get("planner_hr") or meta.get("hr")
        if hr is None:
            # Derive from amount ÷ rate if needed
            rate_key = _rate_key_for_bucket(norm)
            rate_source = rates if isinstance(rates, _MappingABC) else {}
            rate = float(rate_source.get(rate_key, 0.0)) if rate_key else 0.0
            hr = (float(amount) / rate) if rate > 0 else None
        if hr is not None:
            label = _process_label(key)
            out[label] = out.get(label, 0.0) + float(hr)
    removal_hr = None
    render_extra: Mapping[str, Any] | None = None
    if render_state is not None:
        try:
            extra = getattr(render_state, "extra", {})
        except Exception:
            extra = {}
        if isinstance(extra, _MappingABC):
            render_extra = extra
            removal_candidate = extra.get("removal_drilling_hours")
            removal_hr = _coerce_float_or_none(removal_candidate)
    if removal_hr is None:
        removal_hr = _coerce_float_or_none(removal_drilling_hours)
    if removal_hr is not None and removal_hr < 0:
        removal_hr = None
    prefer_drill_hours = prefer_removal_drilling_hours
    if cfg is not None:
        prefer_from_cfg = getattr(cfg, "prefer_removal_drilling_hours", None)
        if prefer_from_cfg is not None:
            prefer_drill_hours = bool(prefer_from_cfg)
    if removal_hr is not None and prefer_drill_hours:
        desired = max(0.0, float(removal_hr))
        drill_labels = [
            label
            for label in out
            if _canonical_bucket_key(label) in {"drilling", "drill"}
        ]
        if drill_labels:
            for label in drill_labels:
                current = float(out.get(label, 0.0) or 0.0)
                if not math.isclose(current, desired, rel_tol=1e-9, abs_tol=1e-6):
                    logger.info(
                        "[hours-sync] Overriding Drilling bucket hours from %.2f -> %.2f (source=removal_card)",
                        current,
                        desired,
                    )
                out[label] = desired
        else:
            label = _process_label("drilling")
            logger.info(
                "[hours-sync] Injecting Drilling bucket hours %.2f (source=removal_card)",
                desired,
            )
            out[label] = desired

    if prefer_drill_hours and isinstance(render_extra, _MappingABC):
        machine_minutes = render_extra.get("drill_machine_minutes")
        labor_minutes = render_extra.get("drill_labor_minutes")
        if isinstance(machine_minutes, (int, float)) and isinstance(labor_minutes, (int, float)):
            drill_total_hr = (float(machine_minutes) + float(labor_minutes)) / 60.0
            for key in list(out.keys()):
                if _canonical_bucket_key(key) in {"drilling", "drill"}:
                    out[key] = drill_total_hr

    return out

def _planner_bucket_key_for_name(name: Any) -> str:
    text = str(name or "").lower()
    if not text:
        return "milling"
    if any(token in text for token in ("c'bore", "counterbore")):
        return "counterbore"
    if any(token in text for token in ("csk", "countersink")):
        return "countersink"
    if any(
        token in text
        for token in (
            "tap",
            "thread mill",
            "thread_mill",
            "rigid tap",
            "rigid_tap",
        )
    ):
        return "tapping"
    if any(token in text for token in ("drill", "ream", "bore")):
        return "drilling"
    if any(
        token in text
        for token in (
            "grind",
            "od grind",
            "id grind",
            "surface grind",
            "jig grind",
        )
    ):
        return "grinding"
    if "wire" in text or "wedm" in text:
        return "wire_edm"
    if "edm" in text:
        return "sinker_edm"
    if any(token in text for token in ("saw", "waterjet")):
        return "saw_waterjet"
    if any(token in text for token in ("deburr", "finish")):
        return "finishing_deburr"
    if "inspect" in text or "cmm" in text or "fai" in text:
        return "inspection"
    return "milling"


def _canonical_bucket_key(name: str | None) -> str:
    normalized = _normalize_bucket_key(name)
    if not normalized:
        return ""
    canon = CANON_MAP.get(normalized)
    if canon:
        return canon

    tokens = [tok for tok in normalized.split("_") if tok]
    if any(tok.startswith("deburr") or tok.startswith("debur") for tok in tokens):
        return "finishing_deburr"
    if any(tok.startswith("finish") for tok in tokens):
        return "finishing_deburr"

    return normalized

def _bucket_cost(info: Mapping[str, Any] | None, *keys: str) -> float:
    """Safely extract a numeric cost value from a mapping."""

    if not isinstance(info, _MappingABC):
        return 0.0
    for key in keys:
        if key in info:
            try:
                return float(info.get(key) or 0.0)
            except Exception:
                continue
    return 0.0

def _preferred_order_then_alpha(keys: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    remaining = {key for key in keys if key}
    ordered: list[str] = []

    for preferred in _PREFERRED_BUCKET_VIEW_ORDER:
        if preferred in remaining:
            ordered.append(preferred)
            seen.add(preferred)
    remaining -= seen

    if remaining:
        ordered.extend(sorted(remaining))

    return ordered

def _coerce_bucket_metric(data: Mapping[str, Any] | None, *candidates: str) -> float:
    if not isinstance(data, _MappingABC):
        return 0.0
    for key in candidates:
        if key in data:
            try:
                return float(data.get(key) or 0.0)
            except Exception:
                continue
    return 0.0

_FINAL_BUCKET_HIDE_KEYS = {"planner_total", "planner_labor", "planner_machine", "misc"}

SHOW_BUCKET_DIAGNOSTICS_OVERRIDE = False

def _final_bucket_key(raw_key: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(raw_key or "").lower()).strip("_")
    if not text:
        return ""
    return {
        "deburr": "finishing_deburr",
        "finishing": "finishing_deburr",
        "finishing_deburr": "finishing_deburr",
    }.get(text, text)

class BucketOpEntry(TypedDict):
    """Canonical representation for a bucket operation entry."""

    name: str
    minutes: float


def _prepare_bucket_view(raw_view: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the canonical bucket view used for display and rollups."""

    prepared: dict[str, Any] = {}
    if isinstance(raw_view, _MappingABC):
        for key, value in raw_view.items():
            if key == "buckets":
                continue
            prepared[key] = copy.deepcopy(value)

    bucket_ops: dict[str, list[BucketOpEntry]] = {}
    if isinstance(raw_view, _MappingABC):
        operations = raw_view.get("operations")
        if isinstance(operations, Sequence):
            for entry in operations:
                if not isinstance(entry, _MappingABC):
                    continue
                bucket_key = entry.get("bucket") or entry.get("name") or ""
                canon_bucket = _canonical_bucket_key(str(bucket_key))
                if not canon_bucket:
                    canon_bucket = _normalize_bucket_key(bucket_key)
                if not canon_bucket:
                    continue
                minutes_val = _coerce_float_or_none(entry.get("minutes"))
                if minutes_val is None or minutes_val <= 0:
                    minutes_val = _coerce_float_or_none(entry.get("mins"))
                if minutes_val is None or minutes_val <= 0:
                    continue
                op_name = (entry.get("name") or "").strip()
                if not op_name:
                    continue
                if canon_bucket not in bucket_ops:
                    bucket_ops[canon_bucket] = []
                bucket_ops[canon_bucket].append(
                    {"name": op_name, "minutes": float(minutes_val)}
                )
    if bucket_ops:
        prepared.setdefault("bucket_ops", bucket_ops)

    source = raw_view.get("buckets") if isinstance(raw_view, _MappingABC) else None
    if not isinstance(source, _MappingABC):
        source = raw_view if isinstance(raw_view, _MappingABC) else {}

    folded: dict[str, dict[str, float]] = {}

    for raw_key, raw_info in source.items():
        canon = _final_bucket_key(raw_key)
        if not canon or canon in _FINAL_BUCKET_HIDE_KEYS:
            continue
        info_map = raw_info if isinstance(raw_info, _MappingABC) else {}
        bucket = folded.setdefault(
            canon,
            {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0},
        )

        minutes = _coerce_bucket_metric(info_map, "minutes")
        labor = _coerce_bucket_metric(info_map, "labor$", "labor_cost", "labor")
        machine = _coerce_bucket_metric(info_map, "machine$", "machine_cost", "machine")

        bucket["minutes"] += minutes
        bucket["labor$"] += labor
        bucket["machine$"] += machine

    cleaned: dict[str, dict[str, float]] = {}
    totals = {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0}

    for canon, metrics in folded.items():
        minutes = round(float(metrics.get("minutes", 0.0)), 2)
        labor = round(float(metrics.get("labor$", 0.0)), 2)
        machine = round(float(metrics.get("machine$", 0.0)), 2)
        total = round(labor + machine, 2)

        if (
            math.isclose(minutes, 0.0, abs_tol=0.01)
            and math.isclose(labor, 0.0, abs_tol=0.01)
            and math.isclose(machine, 0.0, abs_tol=0.01)
            and math.isclose(total, 0.0, abs_tol=0.01)
        ):
            continue

        cleaned[canon] = {
            "minutes": minutes,
            "labor$": labor,
            "machine$": machine,
            "total$": total,
        }

        totals["minutes"] += minutes
        totals["labor$"] += labor
        totals["machine$"] += machine
        totals["total$"] += total

    prepared["buckets"] = cleaned
    prepared["order"] = _preferred_order_then_alpha(cleaned.keys())
    prepared["totals"] = {key: round(value, 2) for key, value in totals.items()}

    return prepared

def canonicalize_costs(process_costs: Mapping[str, Any] | None) -> dict[str, float]:
    items: Iterable[tuple[Any, Any]]
    if isinstance(process_costs, _MappingABC):
        items = process_costs.items()
    else:
        try:
            items = dict(process_costs or {}).items()  # type: ignore[arg-type]
        except Exception:
            items = []

    debug_misc = os.environ.get("DEBUG_MISC") == "1"

    out: dict[str, float] = {}
    for raw_key, raw_value in items:
        key = str(raw_key).strip().lower()
        if not key:
            continue
        key = key.replace(" ", "_").replace("-", "_").replace("/", "_")
        if not key:
            continue
        if key.startswith("planner_") or key in PLANNER_META:
            continue
        canon_key = CANON_MAP.get(key, key)
        try:
            amount = float(raw_value or 0.0)
        except Exception:
            amount = 0.0
        out[canon_key] = out.get(canon_key, 0.0) + amount

    misc_amount = out.get("misc")
    if misc_amount is not None and not debug_misc:
        try:
            misc_val = float(misc_amount)
        except Exception:
            misc_val = 0.0
        if abs(misc_val) < 50.0:
            out.pop("misc", None)

    return out

def _process_label(key: str | None) -> str:
    raw = str(key or "").strip().lower().replace(" ", "_")
    canon = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    alias = {
        "finishing_deburr": "finishing/deburr",
        "deburr": "finishing/deburr",
        "deburring": "finishing/deburr",
        "finish_deburr": "finishing/deburr",
        "saw_waterjet": "saw / waterjet",
        "counter_bore": "counterbore",
        "counter_sink": "countersink",
        "prog_amortized": PROGRAMMING_PER_PART_LABEL.lower(),
        "programming_amortized": PROGRAMMING_PER_PART_LABEL.lower(),
        "fixture_build_amortized": "fixture build (amortized)",
    }.get(canon, canon)
    if alias == "saw / waterjet":
        return "Saw / Waterjet"
    text = alias.replace("_", " ")
    if "(" in text:
        prefix, suffix = text.split("(", 1)
        return prefix.title().rstrip() + " (" + suffix
    return text.title()

def _canonical_hour_label(label: str | None) -> str:
    text = re.sub(r"\s+", " ", str(label or "").strip())
    if not text:
        return ""
    canonical_label, _ = _canonical_amortized_label(text)
    if canonical_label:
        text = canonical_label
    lookup = {
        "programming": "Programming",
        "programming (lot)": "Programming",
        PROGRAMMING_PER_PART_LABEL.lower(): PROGRAMMING_PER_PART_LABEL,
        "fixture build": "Fixture Build",
        "fixture build (lot)": "Fixture Build",
        "fixture build (amortized)": "Fixture Build (amortized)",
        "fixture build (amortized per part)": "Fixture Build (amortized)",
    }
    return lookup.get(text.lower(), text)

def _display_bucket_label(
    canon_key: str,
    label_overrides: Mapping[str, str] | None = None,
) -> str:
    overrides = sdict(label_overrides)
    if canon_key in overrides:
        return overrides[canon_key]
    return _process_label(canon_key)

def _format_planner_bucket_line(
    canon_key: str,
    amount: float,
    meta: Mapping[str, Any] | None,
    *,
    planner_bucket_display_map: Mapping[str, Mapping[str, Any]] | None = None,
    label_overrides: Mapping[str, str] | None = None,
    currency_formatter: Callable[[float], str] | None = None,
) -> tuple[str | None, float, float, float]:
    if not planner_bucket_display_map:
        return (None, amount, 0.0, 0.0)
    info = planner_bucket_display_map.get(canon_key)
    if not isinstance(info, _MappingABC):
        return (None, amount, 0.0, 0.0)

    try:
        minutes_val = float(info.get("minutes", 0.0) or 0.0)
    except Exception:
        minutes_val = 0.0

    hr_val = 0.0
    if _canonical_bucket_key(canon_key) == "drilling":
        synced_hr_val: float | None = None
        synced_source = info.get("synced_hours")
        if synced_source is not None:
            try:
                synced_hr_val = float(synced_source)
            except Exception:
                synced_hr_val = None
        if synced_hr_val is None:
            synced_minutes = info.get("synced_minutes")
            if synced_minutes is not None:
                try:
                    synced_hr_val = float(synced_minutes) / 60.0
                except Exception:
                    synced_hr_val = None
        if synced_hr_val is not None and synced_hr_val > 0:
            hr_val = float(synced_hr_val)
            minutes_val = hr_val * 60.0
    if isinstance(meta, _MappingABC):
        try:
            hr_val = float(meta.get("hr", 0.0) or 0.0)
        except Exception:
            hr_val = 0.0
    if hr_val <= 0 and minutes_val > 0:
        hr_val = minutes_val / 60.0

    total_cost = amount
    for key_option in ("total_cost", "total$", "total"):
        if key_option in info:
            try:
                candidate = float(info.get(key_option) or 0.0)
            except Exception:
                continue
            if candidate:
                total_cost = candidate
                break

    rate_val = 0.0
    if isinstance(meta, _MappingABC):
        try:
            rate_val = float(meta.get("rate", 0.0) or 0.0)
        except Exception:
            rate_val = 0.0
    if rate_val <= 0 and hr_val > 0 and total_cost > 0:
        rate_val = total_cost / hr_val

    formatter: Callable[[float], str]
    if currency_formatter is None:
        formatter = lambda x: fmt_money(x, "$")  # pragma: no cover
    else:
        formatter = currency_formatter

    hours_text = fmt_hours(hr_val) # pragma: no cover
    if rate_val > 0:
        rate_text = f"{formatter(rate_val)}/hr"
    else:
        rate_text = "—"

    display_override = (
        f"{_display_bucket_label(canon_key, label_overrides)}: {hours_text} × {rate_text} →"
    )
    return (display_override, float(total_cost), hr_val, rate_val)

def _extract_bucket_map(source: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    bucket_map: dict[str, dict[str, Any]] = {}
    if not isinstance(source, _MappingABC):
        return bucket_map
    struct: Mapping[str, Any] = source
    buckets_obj = source.get("buckets") if isinstance(source, _MappingABC) else None
    if isinstance(buckets_obj, _MappingABC):
        struct = buckets_obj
    for raw_key, raw_value in struct.items():
        canon = _canonical_bucket_key(raw_key)
        if not canon:
            continue
        if isinstance(raw_value, _MappingABC):
            bucket_map[canon] = {str(k): v for k, v in raw_value.items()}
        else:
            bucket_map[canon] = {}
    return bucket_map


__all__ = [
    "PROGRAMMING_PER_PART_LABEL",
    "PlannerBucketRenderState",
    "_BucketOpEntry",
    "_bucket_role_for_key",
    "_op_role_for_name",
    "_normalize_bucket_key",
    "_rate_key_for_bucket",
    "_lookup_bucket_rate",
    "_split_hours_for_bucket",
    "_build_planner_bucket_render_state",
    "_display_rate_for_row",
    "_sync_drilling_bucket_view",
    "_seed_bucket_minutes",
    "_hole_table_minutes_from_geo",
    "_charged_hours_by_bucket",
    "_planner_bucket_key_for_name",
    "_canonical_bucket_key",
    "_bucket_cost",
    "_preferred_order_then_alpha",
    "_coerce_bucket_metric",
    "_final_bucket_key",
    "_prepare_bucket_view",
    "canonicalize_costs",
    "_process_label",
    "_canonical_hour_label",
    "_display_bucket_label",
    "_format_planner_bucket_line",
    "_extract_bucket_map",
]
