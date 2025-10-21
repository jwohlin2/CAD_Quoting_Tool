"""Legacy drilling estimator helpers lifted from the Tkinter app."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping as _MappingABC, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, TypeAlias

try:  # pragma: no cover - optional dependency
    from pandas import DataFrame as PandasDataFrame  # type: ignore
except Exception:  # pragma: no cover - fallback when pandas is unavailable
    PandasDataFrame = Any  # type: ignore[misc, assignment]

from appkit.data import load_json
from appkit.utils import _jsonify_debug_value
from cad_quoter.domain_models import MATERIAL_DISPLAY_BY_KEY, normalize_material_key
from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.domain_models.values import to_float
from cad_quoter.estimators.base import SpeedsFeedsUnavailableError
from cad_quoter.pricing import time_estimator as _time_estimator
from cad_quoter.pricing.speeds_feeds_selector import (
    pick_speeds_row as _pick_speeds_row,
    unit_hp_cap as _unit_hp_cap,
)
from cad_quoter.speeds_feeds import (
    coerce_table_to_records as _coerce_table_to_records,
    material_label_from_records as _material_label_from_records,
    select_group_rows as _select_group_rows,
    select_operation_rows as _select_operation_rows,
)
from cad_quoter.pricing.time_estimator import (
    MachineParams as _TimeMachineParams,
    OperationGeometry as _TimeOperationGeometry,
    OverheadParams as _TimeOverheadParams,
    ToolParams as _TimeToolParams,
    estimate_time_min as _estimate_time_min,
)

__all__ = [
    "_clean_hole_groups",
    "_coerce_overhead_dataclass",
    "_default_drill_index_seconds",
    "_drill_minutes_per_hole_bounds",
    "_drill_overhead_from_params",
    "_legacy_estimate_drilling_hours",
    "_machine_params_from_params",
    "legacy_estimate_drilling_hours",
]

_NormalizedKey = str

_normalize_lookup_key = normalize_material_key

try:  # pragma: no cover - resource access
    _DRILLING_COEFFS = load_json("drilling.json")
except FileNotFoundError:  # pragma: no cover - defensive default
    _DRILLING_COEFFS = {}

MIN_DRILL_MIN_PER_HOLE = float(_DRILLING_COEFFS.get("min_minutes_per_hole", 0.10))
DEFAULT_MAX_DRILL_MIN_PER_HOLE = float(_DRILLING_COEFFS.get("max_minutes_per_hole", 3.00))

DEEP_DRILL_SFM_FACTOR = float(
    _DRILLING_COEFFS.get("deep_drill", {}).get("sfm_factor", 0.65)
)
DEEP_DRILL_IPR_FACTOR = float(
    _DRILLING_COEFFS.get("deep_drill", {}).get("ipr_factor", 0.70)
)
DEEP_DRILL_PECK_PENALTY_MIN_PER_IN = float(
    _DRILLING_COEFFS.get("deep_drill", {}).get("peck_penalty_min_per_in", 0.07)
)

DEFAULT_DRILL_INDEX_SEC_PER_HOLE = float(
    _DRILLING_COEFFS.get("standard_drill", {}).get("index_sec_per_hole", 5.3746248)
)
DEFAULT_DEEP_DRILL_INDEX_SEC_PER_HOLE = float(
    _DRILLING_COEFFS.get("deep_drill", {}).get("index_sec_per_hole", 4.3038756)
)

def _material_label_from_table(
    table: Any | None,
    material_key: str | None,
    normalized_lookup: _NormalizedKey,
) -> str | None:
    records = _coerce_table_to_records(table)
    if not records:
        return None
    return _material_label_from_records(
        records,
        normalized_lookup=normalized_lookup,
        material_group=material_key,
    )


_DRILL_OPERATION_ALIASES: Mapping[str, Sequence[str]] = {
    "drill": ("drill", "drilling"),
    "deep_drill": (
        "deep_drill",
        "deep drilling",
        "deepdrill",
        "deep drill",
    ),
}


def _select_speeds_feeds_row(
    table: PandasDataFrame | None,
    operation: str,
    material_key: str | None = None,
    *,
    material_group: str | None = None,
) -> Mapping[str, Any] | None:
    if table is None:
        return None
    records = _coerce_table_to_records(table)
    if not records:
        return None

    op_rows = _select_operation_rows(records, operation, aliases=_DRILL_OPERATION_ALIASES)
    if not op_rows:
        return None

    if material_group:
        group_rows = _select_group_rows(
            records,
            operation,
            material_group,
            aliases=_DRILL_OPERATION_ALIASES,
        )
        if group_rows:
            return group_rows[0]

    normalized_target = _normalize_lookup_key(material_key)
    if normalized_target:
        for row in op_rows:
            row_key = str(row.get("_norm_material_key") or "")
            if row_key and row_key == normalized_target:
                return row

    return op_rows[0]


def _machine_params_from_params(params: Mapping[str, Any] | None) -> _TimeMachineParams:
    rapid = _coerce_float_or_none(params.get("MachineRapidIPM")) if isinstance(params, _MappingABC) else None
    hp = _coerce_float_or_none(params.get("MachineHorsepower")) if isinstance(params, _MappingABC) else None
    mrr_factor = (
        _coerce_float_or_none(params.get("MachineHpToMrrFactor"))
        if isinstance(params, _MappingABC)
        else None
    )
    return _TimeMachineParams(
        rapid_ipm=float(rapid) if rapid and rapid > 0 else 300.0,
        hp_available=float(hp) if hp and hp > 0 else None,
        hp_to_mrr_factor=float(mrr_factor) if mrr_factor and mrr_factor > 0 else None,
    )


OverheadLike: TypeAlias = _TimeOverheadParams | SimpleNamespace | Mapping[str, Any]


def _drill_overhead_from_params(params: Mapping[str, Any] | None) -> OverheadLike:
    defaults = _DRILLING_COEFFS.get(
        "overhead_defaults",
        {
            "toolchange_min": 0.5,
            "approach_retract_in": 0.25,
            "peck_penalty_min_per_in_depth": 0.03,
            "dwell_min": None,
            "index_sec_per_hole": 8.0,
        },
    )
    toolchange = (
        _coerce_float_or_none(params.get("DrillToolchangeMinutes"))
        if isinstance(params, _MappingABC)
        else None
    )
    approach = (
        _coerce_float_or_none(params.get("DrillApproachRetractIn"))
        if isinstance(params, _MappingABC)
        else None
    )
    peck = (
        _coerce_float_or_none(params.get("DrillPeckPenaltyMinPerIn"))
        if isinstance(params, _MappingABC)
        else None
    )
    dwell = (
        _coerce_float_or_none(params.get("DrillDwellMinutes"))
        if isinstance(params, _MappingABC)
        else None
    )
    default_index_sec = _coerce_float_or_none(defaults.get("index_sec_per_hole")) or 8.0
    index_source: object | None = default_index_sec
    if isinstance(params, _MappingABC):
        if "DrillIndexSecPerHole" in params:
            index_source = params.get("DrillIndexSecPerHole", default_index_sec)
        elif "DrillIndexSecondsPerHole" in params:
            index_source = params.get("DrillIndexSecondsPerHole")
    index_sec = _coerce_float_or_none(index_source)
    if index_sec is None:
        index_sec = default_index_sec

    payload = {
        "toolchange_min": float(toolchange)
        if toolchange is not None and toolchange >= 0
        else float(defaults.get("toolchange_min", 0.5)),
        "approach_retract_in": float(approach)
        if approach is not None and approach >= 0
        else float(defaults.get("approach_retract_in", 0.25)),
        "peck_penalty_min_per_in_depth": float(peck)
        if peck is not None and peck >= 0
        else float(defaults.get("peck_penalty_min_per_in_depth", 0.03)),
        "dwell_min": float(dwell)
        if dwell is not None and dwell >= 0
        else defaults.get("dwell_min"),
        "index_sec_per_hole": float(index_sec) if index_sec is not None and index_sec >= 0 else None,
    }

    try:
        return _TimeOverheadParams(**payload)
    except TypeError:
        overhead = _TimeOverheadParams(
            toolchange_min=payload.get("toolchange_min"),
            approach_retract_in=payload.get("approach_retract_in"),
            peck_penalty_min_per_in_depth=payload.get("peck_penalty_min_per_in_depth"),
            dwell_min=payload.get("dwell_min"),
            peck_min=defaults.get("peck_min"),
        )
        index_val = payload.get("index_sec_per_hole")
        if index_val is not None:
            try:
                object.__setattr__(overhead, "index_sec_per_hole", index_val)
            except Exception:
                setattr(overhead, "index_sec_per_hole", index_val)
        return overhead

def _make_time_overhead_params(
    params: Mapping[str, Any] | None,
) -> tuple[_TimeOverheadParams, bool]:
    """Instantiate ``OverheadParams`` handling optional index compatibility."""

    kwargs: dict[str, Any] = {}
    if isinstance(params, _MappingABC):
        kwargs = {str(k): v for k, v in params.items()}

    valid_keys = {
        "toolchange_min",
        "approach_retract_in",
        "peck_penalty_min_per_in_depth",
        "dwell_min",
        "peck_min",
        "index_sec_per_hole",
    }
    filtered: dict[str, Any] = {k: kwargs[k] for k in valid_keys if k in kwargs}

    index_kwarg: float | None = None
    if "index_sec_per_hole" in filtered:
        index_val = filtered.pop("index_sec_per_hole")
        coerced = _coerce_float_or_none(index_val)
        if coerced is not None and coerced >= 0:
            index_kwarg = float(coerced)

    try:
        overhead = _TimeOverheadParams(**filtered)
    except TypeError:
        overhead = _TimeOverheadParams(
            toolchange_min=filtered.get("toolchange_min"),
            approach_retract_in=filtered.get("approach_retract_in"),
            peck_penalty_min_per_in_depth=filtered.get("peck_penalty_min_per_in_depth"),
            dwell_min=filtered.get("dwell_min"),
            peck_min=filtered.get("peck_min"),
        )

    if index_kwarg is not None:
        try:
            overhead = replace(overhead, index_sec_per_hole=index_kwarg)
        except Exception:
            try:
                object.__setattr__(overhead, "index_sec_per_hole", index_kwarg)
            except Exception:
                setattr(overhead, "index_sec_per_hole", index_kwarg)

    return overhead, False


def _coerce_overhead_dataclass(overhead: OverheadLike) -> _TimeOverheadParams:
    """Return a dataclass instance compatible with :func:`replace`."""

    if isinstance(overhead, _TimeOverheadParams):
        return overhead

    payload: dict[str, Any] = {}
    source_mapping: Mapping[str, Any] | None = None
    if isinstance(overhead, _MappingABC):
        source_mapping = overhead

    for name in (
        "toolchange_min",
        "approach_retract_in",
        "peck_penalty_min_per_in_depth",
        "dwell_min",
        "peck_min",
    ):
        if source_mapping and name in source_mapping:
            payload[name] = source_mapping.get(name)
        elif hasattr(overhead, name):
            payload[name] = getattr(overhead, name)

    index_value = None
    if source_mapping and "index_sec_per_hole" in source_mapping:
        index_value = source_mapping.get("index_sec_per_hole")
    elif hasattr(overhead, "index_sec_per_hole"):
        index_value = getattr(overhead, "index_sec_per_hole")

    try:
        coerced = _TimeOverheadParams(**payload)
    except TypeError:
        coerced = _TimeOverheadParams(
            toolchange_min=payload.get("toolchange_min"),
            approach_retract_in=payload.get("approach_retract_in"),
            peck_penalty_min_per_in_depth=payload.get("peck_penalty_min_per_in_depth"),
            dwell_min=payload.get("dwell_min"),
            peck_min=payload.get("peck_min"),
        )

    coerced_index = _coerce_float_or_none(index_value)
    if coerced_index is not None and coerced_index >= 0:
        try:
            coerced = replace(coerced, index_sec_per_hole=float(coerced_index))
        except Exception:
            try:
                object.__setattr__(coerced, "index_sec_per_hole", float(coerced_index))
            except Exception:
                setattr(coerced, "index_sec_per_hole", float(coerced_index))

    return coerced


def _clean_hole_groups(raw: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        dia = _coerce_float_or_none(entry.get("dia_mm"))
        depth = _coerce_float_or_none(entry.get("depth_mm"))
        count = _coerce_float_or_none(entry.get("count"))
        if dia is None or dia <= 0:
            continue
        qty = int(round(count)) if count is not None else 0
        if qty <= 0:
            qty = 1
        cleaned.append(
            {
                "dia_mm": float(dia),
                "depth_mm": float(depth) if depth is not None else None,
                "count": qty,
                "through": bool(entry.get("through")),
            }
        )
    return cleaned if cleaned else None

def _default_drill_index_seconds(operation: str | None) -> float:
    """Return the default indexing time (seconds) for a drill operation."""

    op_name = (operation or "").strip().lower()
    if op_name == "deep_drill":
        return DEFAULT_DEEP_DRILL_INDEX_SEC_PER_HOLE
    return DEFAULT_DRILL_INDEX_SEC_PER_HOLE


def _drill_minutes_per_hole_bounds(
    material_group: str | None = None,
    *,
    depth_in: float | None = None,
) -> tuple[float, float]:
    """Return the (min, max) minutes-per-hole bounds for drilling."""

    min_minutes = MIN_DRILL_MIN_PER_HOLE
    max_minutes = DEFAULT_MAX_DRILL_MIN_PER_HOLE
    depth_value = None
    if depth_in is not None:
        try:
            depth_value = float(depth_in)
        except (TypeError, ValueError):
            depth_value = None
    if depth_value is not None and depth_value <= 0:
        depth_value = None

    caps = {
        str(k): float(v)
        for k, v in _DRILLING_COEFFS.get(
            "material_group_caps", {"N": 2.0, "P": 5.0, "M": 5.0, "S": 6.0, "H": 6.0}
        ).items()
    }
    group_key: str | None = None
    if material_group:
        raw_key = str(material_group).strip()
        key_upper = raw_key.upper()
        normalized_key = "".join(ch for ch in key_upper if ch.isalnum())
        if normalized_key in caps:
            group_key = normalized_key
        elif (
            normalized_key
            and normalized_key[0] in caps
            and normalized_key[1:].isdigit()
        ):
            group_key = normalized_key[0]
        if group_key is None:
            key_lower = raw_key.lower()
            if (
                "inconel" in key_lower
                or "titanium" in key_lower
                or key_upper.startswith("TI")
            ):
                group_key = "S"
            elif "stainless" in key_lower:
                group_key = "M"
            elif "steel" in key_lower:
                group_key = "P"
            elif (
                "alum" in key_lower
                or "copper" in key_lower
                or "brass" in key_lower
                or "bronze" in key_lower
                or key_upper.startswith("C")
            ):
                group_key = "N"
            elif key_upper.startswith("H"):
                group_key = "H"
    if group_key:
        max_minutes = caps.get(group_key, DEFAULT_MAX_DRILL_MIN_PER_HOLE)

    depth_penalty = float(_DRILLING_COEFFS.get("depth_penalty_minutes_per_in", 0.2))
    if depth_value is not None:
        max_minutes += depth_penalty * max(0.0, depth_value - 1.0)

    max_minutes = max(max_minutes, min_minutes)
    return min_minutes, max_minutes


def _apply_drill_minutes_clamp(
    hours: float,
    hole_count: int,
    *,
    material_group: str | None = None,
    depth_in: float | None = None,
) -> float:
    if hours <= 0.0 or hole_count <= 0:
        return hours
    min_min_per_hole, max_min_per_hole = _drill_minutes_per_hole_bounds(
        material_group,
        depth_in=depth_in,
    )
    min_hr = (hole_count * min_min_per_hole) / 60.0
    max_hr = (hole_count * max_min_per_hole) / 60.0
    return max(min(hours, max_hr), min_hr)

def _legacy_estimate_drilling_hours(
    hole_diams_mm: list[float],
    thickness_in: float,
    mat_key: str,
    *,
    material_group: str | None = None,
    hole_groups: Sequence[Mapping[str, Any]] | None = None,
    speeds_feeds_table: PandasDataFrame | None = None,
    machine_params: _TimeMachineParams | None = None,
    overhead_params: _TimeOverheadParams | None = None,
    warnings: list[str] | None = None,
    debug_lines: list[str] | None = None,
    debug_summary: dict[str, dict[str, Any]] | None = None,
) -> float:
    """
    Conservative plate-drilling model with floors so 100+ holes don't collapse to minutes.

    ``hole_diams_mm`` is measured in millimetres; ``thickness_in`` is the plate thickness in inches.
    """
    material_lookup = _normalize_lookup_key(mat_key) if mat_key else ""
    material_label = MATERIAL_DISPLAY_BY_KEY.get(material_lookup, mat_key)
    if speeds_feeds_table is None and warnings is None:
        raise SpeedsFeedsUnavailableError(
            "Speeds/feeds table required when estimating drilling hours without a warnings sink."
        )
    # Use any material group passed in; avoid external scope references.
    material_group_override = str(material_group or "").strip().upper()

    thickness_mm_val = 0.0
    try:
        thickness_mm_val = float(thickness_in) * 25.4
    except (TypeError, ValueError):
        pass
    if not math.isfinite(thickness_mm_val) or thickness_mm_val <= 0:
        thickness_mm_val = 0.0
    thickness_in_val = thickness_mm_val / 25.4 if thickness_mm_val else 0.0
    if (
        speeds_feeds_table is not None
        and (not material_label or material_label == mat_key)
    ):
        alt_label = _material_label_from_table(
            speeds_feeds_table,
            mat_key,
            material_lookup,
        )
        if alt_label:
            material_label = alt_label
    mat = str(material_label or mat_key or "").lower()
    material_factor = _unit_hp_cap(material_label)
    # ``debug_state`` collects aggregate drilling metrics for callers that
    # requested debugging information.  Older revisions attempted to update a
    # ``debug`` mapping without first defining it, which triggered a
    # ``NameError`` during quoting.  Initialise the container up-front and create a
    # ``debug`` alias for any legacy references inside this function.
    debug_state: dict[str, Any] | None = None
    if (debug_lines is not None) or (debug_summary is not None):
        debug_state = {}
    debug: dict[str, Any] | None = None
    if debug_state is not None:
        debug = debug_state

    debug_list = debug_lines if debug_lines is not None else None
    if debug_summary is not None:
        debug_summary.clear()
    avg_dia_in = 0.0
    seen_debug: set[str] = set()
    chosen_material_label: str = ""
    operation_debug_data: dict[str, dict[str, Any]] = {}

    def _update_debug_aggregate(
        *,
        hole_count: int,
        avg_diameter: Any,
        minutes_per_hole: float | None,
    ) -> None:
        if debug is None:
            return

        try:
            avg_val = float(avg_diameter)
        except Exception:
            avg_val = 0.0
        if not math.isfinite(avg_val):
            avg_val = 0.0

        min_per_hole_val: float | None
        if minutes_per_hole is None:
            min_per_hole_val = None
        else:
            try:
                min_candidate = float(minutes_per_hole)
            except Exception:
                min_per_hole_val = None
            else:
                min_per_hole_val = min_candidate if math.isfinite(min_candidate) else None

        debug.update(
            {
                "thickness_in": float(thickness_in or 0.0),
                "avg_dia_in": avg_val,
                "sfm": None,
                "ipr": None,
                "rpm": None,
                "ipm": None,
                "min_per_hole": min_per_hole_val,
                "hole_count": int(hole_count),
            }
        )

    def _log_debug(entry: str) -> None:
        if debug_list is None:
            return
        text = str(entry or "").strip()
        if not text or text in seen_debug:
            return
        debug_list.append(text)
        seen_debug.add(text)

    def _update_range(target: dict[str, Any], min_key: str, max_key: str, value: Any) -> None:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val):
            return
        current_min = target.get(min_key)
        if current_min is None or val < current_min:
            target[min_key] = val
        current_max = target.get(max_key)
        if current_max is None or val > current_max:
            target[max_key] = val

    if debug_list is not None and speeds_feeds_table is None:
        _log_debug("MISS table: using heuristic fallback")

    group_specs: list[tuple[float, int, float]] = []
    # Optional aggregate debug dict for callers that want structured details
    fallback_counts: Counter[float] | None = None

    if hole_groups:
        fallback_counts = Counter()
        for entry in hole_groups:
            if not isinstance(entry, _MappingABC):
                continue
            dia_mm = _coerce_float_or_none(entry.get("dia_mm"))
            count = _coerce_float_or_none(entry.get("count"))
            depth_mm = _coerce_float_or_none(entry.get("depth_mm"))
            if dia_mm is None or dia_mm <= 0:
                continue
            qty = int(round(count)) if count is not None else 0
            if qty <= 0:
                qty = int(max(1, round(count or 1)))
            diameter_in = float(dia_mm) / 25.4
            depth_in = 0.0
            if depth_mm and depth_mm > 0:
                depth_in = float(depth_mm) / 25.4
            elif thickness_in_val and thickness_in_val > 0:
                depth_in = float(thickness_in_val)
            breakthrough_in = max(
                float(_DRILLING_COEFFS.get("breakthrough_min_in", 0.04)),
                float(_DRILLING_COEFFS.get("breakthrough_factor", 0.2)) * diameter_in,
            )
            if depth_in > 0:
                depth_in += breakthrough_in
            else:
                depth_in = breakthrough_in
            group_specs.append((diameter_in, qty, depth_in))
            fallback_counts[round(float(dia_mm), 3)] += qty
    if not group_specs:
        if not hole_diams_mm or thickness_in <= 0:
            return 0.0
        thickness_in_for_depth = float(thickness_in_val)
        counts = Counter(round(float(d), 3) for d in hole_diams_mm if d and math.isfinite(d))
        for dia_mm, qty in counts.items():
            if qty <= 0:
                continue
            diameter_in = float(dia_mm) / 25.4
            breakthrough_in = max(
                float(_DRILLING_COEFFS.get("breakthrough_min_in", 0.04)),
                float(_DRILLING_COEFFS.get("breakthrough_factor", 0.2)) * diameter_in,
            )
            total_depth_in = (
                thickness_in_for_depth + breakthrough_in
                if thickness_in_for_depth > 0
                else breakthrough_in
            )
            group_specs.append((diameter_in, int(qty), total_depth_in))
        fallback_counts = counts
    elif fallback_counts is None:
        fallback_counts = Counter()
        for dia_in, qty, _ in group_specs:
            fallback_counts[round(dia_in * 25.4, 3)] += qty

    if group_specs:
        base_machine = machine_params or _machine_params_from_params(None)
        overhead = overhead_params or _drill_overhead_from_params(None)
        overhead_for_calc = _coerce_overhead_dataclass(overhead)
        per_hole_overhead = replace(overhead_for_calc, toolchange_min=0.0)
        total_min = 0.0
        total_toolchange_min = 0.0
        total_holes = 0
        material_cap_val = to_float(material_factor)
        if material_cap_val is not None and material_cap_val <= 0:
            material_cap_val = None

        total_qty_for_avg = 0
        weighted_dia_sum = 0.0
        depth_candidates: list[float] = []
        for diameter_in, qty, depth_val in group_specs:
            try:
                qty_int = int(qty)
            except Exception:
                qty_int = 0
            if qty_int <= 0:
                continue
            total_qty_for_avg += qty_int
            weighted_dia_sum += float(diameter_in) * qty_int
            if depth_val and depth_val > 0:
                depth_candidates.append(float(depth_val))
        if total_qty_for_avg > 0:
            avg_dia_in = weighted_dia_sum / total_qty_for_avg
        depth_for_bounds = max(depth_candidates) if depth_candidates else None

        hp_cap_val = to_float(getattr(base_machine, "hp_to_mrr_factor", None))
        combined_cap = None
        if material_cap_val is not None and hp_cap_val is not None:
            combined_cap = min(hp_cap_val, material_cap_val)
        elif material_cap_val is not None:
            combined_cap = material_cap_val
        elif hp_cap_val is not None:
            combined_cap = hp_cap_val
        if combined_cap is not None:
            machine_for_cut = _TimeMachineParams(
                rapid_ipm=base_machine.rapid_ipm,
                hp_available=base_machine.hp_available,
                hp_to_mrr_factor=float(combined_cap),
            )
        else:
            machine_for_cut = base_machine

        row_cache: dict[tuple[str, float], tuple[Mapping[str, Any], _TimeToolParams] | None] = {}
        missing_row_messages: set[tuple[str, str, float]] = set()
        debug_summary_entries: dict[str, dict[str, Any]] = {}

        def _build_tool_params(row: Mapping[str, Any]) -> _TimeToolParams:
            key_map = {
                str(k).strip().lower().replace("-", "_").replace(" ", "_"): k
                for k in row.keys()
            }

            def _row_float(*names: str) -> float | None:
                for name in names:
                    actual = key_map.get(name)
                    if actual is None:
                        continue
                    val = to_float(row.get(actual))
                    if val is not None:
                        return float(val)
                return None

            teeth_val = _row_float("teeth_z", "flutes", "flute_count", "teeth")
            teeth_int: int | None = None
            if teeth_val is not None and teeth_val > 0:
                try:
                    teeth_int = int(round(teeth_val))
                except Exception:
                    teeth_int = None
            if teeth_int is None or teeth_int <= 0:
                teeth_int = 1
            return _TimeToolParams(teeth_z=teeth_int)

        for diameter_in, qty, depth_in in group_specs:
            qty_i = int(qty)
            if qty_i <= 0 or diameter_in <= 0 or depth_in <= 0:
                continue
            tool_dia_in = float(diameter_in)
            l_over_d = 0.0
            if tool_dia_in > 0 and depth_in and depth_in > 0:
                l_over_d = float(depth_in) / max(float(tool_dia_in), 1e-6)
            op_name = "deep_drill" if l_over_d >= 3.0 else "drill"
            cache_key = (op_name, round(float(diameter_in), 4))
            row: Mapping[str, Any] | None = None
            expected_group_value = material_group_override
            cache_entry = row_cache.get(cache_key)
            is_new_row_entry = False
            if cache_entry is None:
                material_for_lookup: str | None = None
                lookup_candidates = (
                    material_group_override,
                    material_label,
                    mat_key,
                    material_lookup,
                )
                for idx, candidate in enumerate(lookup_candidates):
                    text = str(candidate or "").strip()
                    if not text:
                        continue
                    material_for_lookup = text if idx else text.upper()
                    break

                canonical_lookup = str(
                    material_label or mat_key or material_lookup or ""
                ).strip()

                def _pick_with_key(
                    operation_name: str,
                    lookup_key: str | None,
                ) -> Mapping[str, Any] | None:
                    if not lookup_key or speeds_feeds_table is None:
                        return None
                    return _pick_speeds_row(
                        material_label=material_label,
                        operation=operation_name,
                        tool_diameter_in=float(diameter_in),
                        table=speeds_feeds_table,
                        material_group=material_group_override or None,
                        material_key=lookup_key,
                    )

                row = None
                # Prefer selection by material group first (normalized like N1/N2 -> N),
                # then fall back to canonical material name. Keep existing fallbacks after.
                if speeds_feeds_table is not None and material_group_override:
                    row = _pick_speeds_row(
                        material_label=material_label,
                        operation=op_name,
                        tool_diameter_in=float(diameter_in),
                        table=speeds_feeds_table,
                        material_group=material_group_override,
                        material_key=None,
                    )
                if not row and canonical_lookup:
                    row = _pick_with_key(op_name, canonical_lookup)
                if not row and speeds_feeds_table is not None:
                    row = _select_speeds_feeds_row(
                        speeds_feeds_table,
                        operation=op_name,
                        material_key=material_for_lookup,
                        material_group=material_group_override,
                    )
                    if not row and op_name.lower() == "deep_drill":
                        row = _select_speeds_feeds_row(
                            speeds_feeds_table,
                            operation="Drill",
                            material_key=material_for_lookup,
                            material_group=material_group_override,
                        )
                if not row:
                    row = _pick_with_key(op_name, material_group_override)
                if not row and canonical_lookup:
                    row = _pick_with_key(op_name, canonical_lookup)
                if not row:
                    row = _pick_speeds_row(
                        material_label=material_label,
                        operation=op_name,
                        tool_diameter_in=float(diameter_in),
                        table=speeds_feeds_table,
                        material_group=material_group_override or None,
                    )
                if not row and op_name.lower() == "deep_drill":
                    row = _pick_with_key("drill", material_group_override)
                if not row and op_name.lower() == "deep_drill" and canonical_lookup:
                    row = _pick_with_key("drill", canonical_lookup)
                if not row and op_name.lower() == "deep_drill":
                    row = _pick_speeds_row(
                        material_label=material_label,
                        operation="drill",
                        tool_diameter_in=float(diameter_in),
                        table=speeds_feeds_table,
                        material_group=material_group_override or None,
                    )
                if row and isinstance(row, _MappingABC):
                    cache_entry = (row, _build_tool_params(row))
                    # Always use one material label for both Debug and Calc.
                    chosen_material_label = str(
                        row.get("material")
                        or row.get("material_family")
                        or material_label
                        or mat_key
                        or material_lookup
                        or ""
                    ).strip()
                    if material_label:
                        chosen_material_label = str(material_label).strip()
                else:
                    cache_entry = None
                    material_display = str(material_label or mat_key or material_lookup or "material").strip()
                    if not material_display:
                        material_display = "material"
                    op_display = "deep drilling" if op_name.lower() == "deep_drill" else "drilling"
                    missing_row_messages.add(
                        (
                            op_display,
                            material_display,
                            round(float(diameter_in), 4),
                        )
                    )
                    _log_debug(
                        f"MISS {op_display} {material_display.lower()} {round(float(diameter_in), 4):.3f}\""
                    )
                    is_new_row_entry = True
                    row_cache[cache_key] = cache_entry
            else:
                try:
                    row = cache_entry[0]
                except Exception:
                    row = None
            row_group_value = ""
            if row and isinstance(row, _MappingABC):
                row_group_value = str(
                    row.get("material_group")
                    or row.get("material_family")
                    or row.get("iso_group")
                    or ""
                ).strip().upper()
            geom = _TimeOperationGeometry(
                diameter_in=float(diameter_in),
                hole_depth_in=float(depth_in),
                point_angle_deg=118.0,
                ld_ratio=l_over_d,
            )
            diameter_float = to_float(diameter_in)
            if diameter_float is None:
                try:
                    diameter_float = float(diameter_in)
                except Exception:
                    diameter_float = None
            precomputed_speeds: dict[str, float] = {}
            row_view = _time_estimator._RowView(row)
            sfm_candidate = getattr(row_view, "sfm_start", None)
            if sfm_candidate is None:
                sfm_candidate = getattr(row_view, "sfm", None)
            sfm_val = _time_estimator.to_num(sfm_candidate)
            if sfm_val is not None and math.isfinite(sfm_val):
                precomputed_speeds["sfm"] = float(sfm_val)
            ipr_val = _time_estimator.pick_feed_value(row_view, float(diameter_float or 0.0))
            if ipr_val is not None and math.isfinite(ipr_val):
                precomputed_speeds["ipr"] = float(ipr_val)
            rpm_val: float | None = None
            if diameter_float and diameter_float > 0 and "sfm" in precomputed_speeds:
                rpm_val = (precomputed_speeds["sfm"] * 12.0) / (math.pi * float(diameter_float))
                if math.isfinite(rpm_val):
                    precomputed_speeds["rpm"] = float(rpm_val)
                else:
                    rpm_val = None
            ipm_val: float | None = None
            if rpm_val is not None and "ipr" in precomputed_speeds:
                ipm_val = float(rpm_val) * precomputed_speeds["ipr"]
                if math.isfinite(ipm_val):
                    precomputed_speeds["ipm"] = float(ipm_val)
                else:
                    ipm_val = None
            is_deep_drill = op_name.lower() == "deep_drill"
            if is_deep_drill:
                sfm_pre = precomputed_speeds.get("sfm")
                if sfm_pre is not None and math.isfinite(sfm_pre):
                    new_sfm = float(sfm_pre) * DEEP_DRILL_SFM_FACTOR
                    precomputed_speeds["sfm"] = new_sfm
                    if diameter_float and diameter_float > 0:
                        rpm_val = (new_sfm * 12.0) / (math.pi * float(diameter_float))
                        if math.isfinite(rpm_val):
                            precomputed_speeds["rpm"] = float(rpm_val)
                elif "rpm" in precomputed_speeds:
                    rpm_only = precomputed_speeds.get("rpm")
                    if rpm_only is not None and math.isfinite(rpm_only):
                        precomputed_speeds["rpm"] = float(rpm_only) * DEEP_DRILL_SFM_FACTOR
                ipr_pre = precomputed_speeds.get("ipr")
                if ipr_pre is not None and math.isfinite(ipr_pre):
                    precomputed_speeds["ipr"] = float(ipr_pre) * DEEP_DRILL_IPR_FACTOR
                if "rpm" in precomputed_speeds and "ipr" in precomputed_speeds:
                    rpm_calc = precomputed_speeds["rpm"]
                    ipr_calc = precomputed_speeds["ipr"]
                    if math.isfinite(rpm_calc) and math.isfinite(ipr_calc):
                        precomputed_speeds["ipm"] = float(rpm_calc) * float(ipr_calc)
            bin_speed_snapshot: dict[str, float | None] = {}
            if precomputed_speeds:
                sfm_for_bin = to_float(precomputed_speeds.get("sfm"))
                ipr_for_bin = to_float(precomputed_speeds.get("ipr"))
                rpm_for_bin = to_float(precomputed_speeds.get("rpm"))
                ipm_for_bin = to_float(precomputed_speeds.get("ipm"))
                if (
                    (rpm_for_bin is None or not math.isfinite(rpm_for_bin) or rpm_for_bin <= 0.0)
                    and sfm_for_bin is not None
                    and math.isfinite(sfm_for_bin)
                    and diameter_float is not None
                    and diameter_float > 0
                ):
                    try:
                        rpm_candidate = (float(sfm_for_bin) * 12.0) / (
                            math.pi * float(diameter_float)
                        )
                    except (TypeError, ValueError, ZeroDivisionError):
                        rpm_candidate = None
                    if rpm_candidate is not None and math.isfinite(rpm_candidate):
                        rpm_for_bin = float(rpm_candidate)
                        precomputed_speeds["rpm"] = rpm_for_bin
                if (
                    (ipm_for_bin is None or not math.isfinite(ipm_for_bin) or ipm_for_bin <= 0.0)
                    and rpm_for_bin is not None
                    and math.isfinite(rpm_for_bin)
                    and ipr_for_bin is not None
                    and math.isfinite(ipr_for_bin)
                ):
                    ipm_candidate = float(rpm_for_bin) * float(ipr_for_bin)
                    if math.isfinite(ipm_candidate):
                        ipm_for_bin = float(ipm_candidate)
                        precomputed_speeds["ipm"] = ipm_for_bin
                bin_speed_snapshot = {
                    "sfm": float(sfm_for_bin)
                    if sfm_for_bin is not None and math.isfinite(sfm_for_bin)
                    else None,
                    "ipr": float(ipr_for_bin)
                    if ipr_for_bin is not None and math.isfinite(ipr_for_bin)
                    else None,
                    "rpm": float(rpm_for_bin)
                    if rpm_for_bin is not None and math.isfinite(rpm_for_bin)
                    else None,
                    "ipm": float(ipm_for_bin)
                    if ipm_for_bin is not None and math.isfinite(ipm_for_bin)
                    else None,
                }
            debug_payload: dict[str, Any] | None = None
            tool_params: _TimeToolParams
            minutes: float
            overhead_for_calc = per_hole_overhead
            if is_deep_drill:
                peck_rate_val = to_float(
                    per_hole_overhead.peck_penalty_min_per_in_depth
                )
                adjusted_peck = max(
                    DEEP_DRILL_PECK_PENALTY_MIN_PER_IN,
                    float(peck_rate_val) if peck_rate_val and peck_rate_val > 0 else 0.0,
                )
                overhead_for_calc = replace(
                    per_hole_overhead,
                    peck_penalty_min_per_in_depth=adjusted_peck,
                )
            if cache_entry:
                row, tool_params = cache_entry
                if debug_lines is not None:
                    debug_payload = {}
                operation_for_time = "drill" if is_deep_drill else op_name
                minutes = _estimate_time_min(
                    row,
                    geom,
                    tool_params,
                    machine_for_cut,
                    overhead_for_calc,
                    material_factor=material_cap_val,
                    debug=debug_payload,
                    precomputed=precomputed_speeds,
                    operation=operation_for_time,
                )
                overhead_for_calc = per_hole_overhead
                if debug_payload is not None and is_deep_drill:
                    debug_payload["operation"] = op_name
                if debug_payload is not None:
                    for key in ("sfm", "ipr", "rpm", "ipm"):
                        snapshot_val = bin_speed_snapshot.get(key)
                        if snapshot_val is not None and math.isfinite(snapshot_val):
                            debug_payload[key] = float(snapshot_val)
                        else:
                            coerced = to_float(debug_payload.get(key))
                            if coerced is not None and math.isfinite(coerced):
                                bin_speed_snapshot[key] = float(coerced)
            else:
                overhead_local = per_hole_overhead
                try:
                    overhead_local = overhead_for_calc
                except (UnboundLocalError, NameError):  # pragma: no cover - safety net
                    pass
                peck_rate = to_float(
                    overhead_local.peck_penalty_min_per_in_depth
                )
                peck_min = None
                if peck_rate and depth_in and depth_in > 0:
                    peck_min = float(peck_rate) * float(depth_in)
                dwell_val = to_float(overhead_local.dwell_min)
                legacy_kwargs = {
                    "toolchange_min": 0.0,
                    "approach_retract_in": overhead_local.approach_retract_in,
                    "peck_penalty_min_per_in_depth": None,
                    "dwell_min": dwell_val,
                    "peck_min": peck_min,
                }
                legacy_index_kwarg = to_float(
                    getattr(overhead_for_calc, "index_sec_per_hole", None)
                )
                if legacy_index_kwarg is not None and legacy_index_kwarg >= 0:
                    legacy_kwargs["index_sec_per_hole"] = legacy_index_kwarg
                legacy_overhead, _ = _make_time_overhead_params(legacy_kwargs)
                legacy_overhead = _coerce_overhead_dataclass(legacy_overhead)
                overhead_for_calc = legacy_overhead
                tool_params = _TimeToolParams(teeth_z=1)
                if debug_lines is not None:
                    debug_payload = {}
                effective_index_sec = to_float(
                    getattr(overhead_for_calc, "index_sec_per_hole", None)
                )
                if effective_index_sec is None or not math.isfinite(effective_index_sec):
                    effective_index_sec = _default_drill_index_seconds(op_name)
                legacy_overhead = replace(
                    legacy_overhead,
                    index_sec_per_hole=effective_index_sec,
                )
                overhead_for_calc = legacy_overhead
                minutes = _estimate_time_min(
                    geom=geom,
                    tool=tool_params,
                    machine=machine_for_cut,
                    overhead=overhead_for_calc,
                    material_factor=material_cap_val,
                    operation=op_name,
                    debug=debug_payload,
                    precomputed=precomputed_speeds,
                )
                overhead_for_calc = legacy_overhead
                if minutes <= 0:
                    continue
                overhead_for_calc = legacy_overhead
            if minutes <= 0:
                continue
            try:
                qty_int = int(qty)
            except Exception:
                continue
            if qty_int <= 0:
                continue
            op_key = str(op_name or "").strip().lower() or "drill"
            op_entry = operation_debug_data.setdefault(
                op_key,
                {
                    "qty": 0,
                    "row": None,
                    "precomputed": None,
                    "material": None,
                    "diameter_weight_sum": 0.0,
                    "diameter_qty_sum": 0,
                },
            )
            op_entry["qty"] += qty_int
            if row and isinstance(row, _MappingABC):
                op_entry["row"] = row
            if expected_group_value:
                op_entry.setdefault("expected_group", expected_group_value)
            if row_group_value:
                op_entry["row_group"] = row_group_value
            if precomputed_speeds:
                op_entry["precomputed"] = dict(precomputed_speeds)
            if chosen_material_label:
                op_entry["material"] = chosen_material_label
            else:
                fallback_material = str(
                    material_label or mat_key or material_lookup or ""
                ).strip()
                if fallback_material:
                    op_entry.setdefault("material", fallback_material)
            if (
                diameter_float is not None
                and math.isfinite(float(diameter_float))
            ):
                op_entry["diameter_weight_sum"] += float(diameter_float) * qty_int
                op_entry["diameter_qty_sum"] += qty_int
            total_holes += qty_int
            total_min += minutes * qty_int
            toolchange_added = 0.0
            if overhead.toolchange_min and qty_int > 0:
                toolchange_added = float(overhead.toolchange_min)
                total_toolchange_min += toolchange_added
            if debug_payload is not None:
                if row_group_value:
                    debug_payload.setdefault("row_group", row_group_value)
                if expected_group_value:
                    debug_payload.setdefault("expected_group", expected_group_value)
                try:
                    operation_name = str(debug_payload.get("operation") or op_name).lower()
                except Exception:
                    operation_name = op_name.lower()
                if precomputed_speeds:
                    for key, value in precomputed_speeds.items():
                        debug_payload.setdefault(key, value)
                sfm_val = precomputed_speeds.get("sfm") if precomputed_speeds else None
                if sfm_val is None:
                    sfm_val = debug_payload.get("sfm")
                ipr_val = precomputed_speeds.get("ipr") if precomputed_speeds else None
                if ipr_val is None:
                    ipr_val = debug_payload.get("ipr")
                rpm_val = precomputed_speeds.get("rpm") if precomputed_speeds else None
                if rpm_val is None:
                    rpm_val = debug_payload.get("rpm")
                ipm_val = precomputed_speeds.get("ipm") if precomputed_speeds else None
                if ipm_val is None:
                    ipm_val = debug_payload.get("ipm")
                depth_val = debug_payload.get("axial_depth_in")
                minutes_per = debug_payload.get("minutes_per_hole")
                qty_for_debug = int(qty) if qty else 0
                mat_display = chosen_material_label or str(
                    material_label or mat_key or material_lookup or ""
                ).strip()
                if not mat_display:
                    mat_display = "material"
                if debug_lines is not None:
                    summary = debug_summary_entries.setdefault(
                        operation_name,
                        {
                            "operation": operation_name,
                            "material": mat_display,
                            "qty": 0,
                            "total_minutes": 0.0,
                            "toolchange_total": 0.0,
                            "sfm_sum": 0.0,
                            "sfm_count": 0,
                            "sfm_min": None,
                            "sfm_max": None,
                            "ipr_sum": 0.0,
                            "ipr_count": 0,
                            "rpm_sum": 0.0,
                            "rpm_count": 0,
                            "ipm_sum": 0.0,
                            "ipm_count": 0,
                            "rpm_min": None,
                            "rpm_max": None,
                            "ipm_min": None,
                            "ipm_max": None,
                            "ipr_min": None,
                            "ipr_max": None,
                            "ipr_effective_min": None,
                            "ipr_effective_max": None,
                            "sfm_min": None,
                            "sfm_max": None,
                            "bins": {},
                            "diameter_weight_sum": 0.0,
                            "diameter_qty_sum": 0,
                            "diam_min": None,
                            "diam_max": None,
                            "depth_weight_sum": 0.0,
                            "depth_qty_sum": 0,
                            "depth_min": None,
                            "depth_max": None,
                            "peck_sum": 0.0,
                            "peck_count": 0,
                            "dwell_sum": 0.0,
                            "dwell_count": 0,
                            "index_sum": 0.0,
                            "index_count": 0,
                        },
                    )
                    if chosen_material_label:
                        summary["material"] = chosen_material_label
                    elif mat_display and (
                        not summary.get("material")
                        or summary.get("material") == "material"
                    ):
                        summary["material"] = mat_display
                    if expected_group_value:
                        summary.setdefault(
                            "expected_material_group",
                            str(expected_group_value).strip().upper(),
                        )
                    if row_group_value:
                        summary["material_group"] = row_group_value
                    minutes_val = to_float(minutes_per)
                    minutes_per_hole = minutes_val if minutes_val is not None else float(minutes)
                    summary["qty"] += qty_for_debug
                    summary["total_minutes"] += minutes_per_hole * qty_for_debug
                    summary["toolchange_total"] += toolchange_added
                    # Accumulate per-bin minutes for compact drilling table
                    try:
                        bins_map = summary.setdefault("bins", {})
                        bin_key = f"{float(tool_dia_in):.4f}"
                        b = bins_map.get(bin_key)
                        if isinstance(b, dict):
                            prior = _coerce_float_or_none(b.get("minutes")) or 0.0
                            b["minutes"] = prior + (minutes_per_hole * qty_for_debug)
                    except Exception:
                        pass
                    sfm_float = to_float(sfm_val)
                    if sfm_float is not None and math.isfinite(sfm_float):
                        summary["sfm_sum"] += sfm_float * qty_for_debug
                        summary["sfm_count"] += qty_for_debug
                        _update_range(summary, "sfm_min", "sfm_max", sfm_float)
                    rpm_float = to_float(rpm_val)
                    ipr_float = to_float(ipr_val)
                    ipm_float = to_float(ipm_val)
                    if (
                        (ipm_float is None or not math.isfinite(ipm_float))
                        and rpm_float is not None
                        and math.isfinite(rpm_float)
                        and ipr_float is not None
                        and math.isfinite(ipr_float)
                    ):
                        ipm_float = float(rpm_float) * float(ipr_float)
                    ipr_effective_float: float | None = None
                    if (
                        rpm_float is not None
                        and math.isfinite(rpm_float)
                        and ipm_float is not None
                        and math.isfinite(ipm_float)
                        and abs(float(rpm_float)) > 1e-9
                    ):
                        ipr_effective_float = float(ipm_float) / float(rpm_float)
                    elif ipr_float is not None and math.isfinite(ipr_float):
                        ipr_effective_float = float(ipr_float)
                    if debug_payload is not None:
                        if rpm_float is not None and math.isfinite(rpm_float):
                            debug_payload["rpm"] = float(rpm_float)
                        if ipm_float is not None and math.isfinite(ipm_float):
                            debug_payload["ipm"] = float(ipm_float)
                        if ipr_effective_float is not None and math.isfinite(ipr_effective_float):
                            debug_payload["ipr_effective"] = float(ipr_effective_float)
                            debug_payload["ipr"] = float(ipr_effective_float)
                        elif ipr_float is not None and math.isfinite(ipr_float):
                            debug_payload["ipr"] = float(ipr_float)
                    if rpm_float is not None and math.isfinite(rpm_float):
                        summary["rpm_sum"] += rpm_float * qty_for_debug
                        summary["rpm_count"] += qty_for_debug
                        _update_range(summary, "rpm_min", "rpm_max", rpm_float)
                    if ipm_float is not None and math.isfinite(ipm_float):
                        summary["ipm_sum"] += ipm_float * qty_for_debug
                        summary["ipm_count"] += qty_for_debug
                        _update_range(summary, "ipm_min", "ipm_max", ipm_float)
                    if ipr_effective_float is not None and math.isfinite(ipr_effective_float):
                        summary["ipr_sum"] += ipr_effective_float * qty_for_debug
                        summary["ipr_count"] += qty_for_debug
                        _update_range(summary, "ipr_min", "ipr_max", ipr_effective_float)
                        _update_range(
                            summary,
                            "ipr_effective_min",
                            "ipr_effective_max",
                            ipr_effective_float,
                        )
                    bins = summary.setdefault("bins", {})
                    bin_key = f"{float(tool_dia_in):.4f}"
                    bin_summary = bins.setdefault(
                        bin_key,
                        {
                            "diameter_in": float(tool_dia_in),
                            "qty": 0,
                            "sfm_min": None,
                            "sfm_max": None,
                            "rpm_min": None,
                            "rpm_max": None,
                            "ipm_min": None,
                            "ipm_max": None,
                            "ipr_min": None,
                            "ipr_max": None,
                            "ipr_effective_min": None,
                            "ipr_effective_max": None,
                            "depth_min": None,
                            "depth_max": None,
                            "minutes": 0.0,
                        },
                    )
                    bin_summary["qty"] += qty_for_debug
                    speeds_for_bin = bin_summary.setdefault("speeds", {})
                    for speed_key, fallback_value in (
                        ("sfm", sfm_float),
                        ("ipr", ipr_effective_float if ipr_effective_float is not None else ipr_float),
                        ("rpm", rpm_float),
                        ("ipm", ipm_float),
                    ):
                        if bin_speed_snapshot and bin_speed_snapshot.get(speed_key) is not None:
                            fallback_value = bin_speed_snapshot.get(speed_key)
                        try:
                            numeric = float(fallback_value) if fallback_value is not None else None
                        except (TypeError, ValueError):
                            numeric = None
                        if numeric is not None and math.isfinite(numeric):
                            speeds_for_bin[speed_key] = numeric
                    if sfm_float is not None and math.isfinite(sfm_float):
                        _update_range(bin_summary, "sfm_min", "sfm_max", sfm_float)
                    if rpm_float is not None and math.isfinite(rpm_float):
                        _update_range(bin_summary, "rpm_min", "rpm_max", rpm_float)
                    if ipm_float is not None and math.isfinite(ipm_float):
                        _update_range(bin_summary, "ipm_min", "ipm_max", ipm_float)
                    if ipr_effective_float is not None and math.isfinite(ipr_effective_float):
                        _update_range(bin_summary, "ipr_min", "ipr_max", ipr_effective_float)
                        _update_range(
                            bin_summary,
                            "ipr_effective_min",
                            "ipr_effective_max",
                            ipr_effective_float,
                        )
                    summary["diameter_weight_sum"] += float(tool_dia_in) * qty_for_debug
                    summary["diameter_qty_sum"] += qty_for_debug
                    diam_min = summary.get("diam_min")
                    diam_max = summary.get("diam_max")
                    if diam_min is None or float(tool_dia_in) < diam_min:
                        summary["diam_min"] = float(tool_dia_in)
                    if diam_max is None or float(tool_dia_in) > diam_max:
                        summary["diam_max"] = float(tool_dia_in)
                    depth_float = to_float(depth_val)
                    if depth_float is None:
                        try:
                            depth_float = float(depth_in)
                        except Exception:
                            depth_float = None
                    if depth_float is not None:
                        summary["depth_weight_sum"] += float(depth_float) * qty_for_debug
                        summary["depth_qty_sum"] += qty_for_debug
                        depth_min = summary.get("depth_min")
                        depth_max = summary.get("depth_max")
                        if depth_min is None or float(depth_float) < depth_min:
                            summary["depth_min"] = float(depth_float)
                        if depth_max is None or float(depth_float) > depth_max:
                            summary["depth_max"] = float(depth_float)
                        _update_range(bin_summary, "depth_min", "depth_max", depth_float)
                    overhead_local = per_hole_overhead
                    try:
                        overhead_local = overhead_for_calc
                    except (UnboundLocalError, NameError):  # pragma: no cover - safety net
                        pass
                    peck_rate = to_float(
                        overhead_local.peck_penalty_min_per_in_depth
                    )
                    if depth_float is not None and peck_rate is not None and peck_rate > 0:
                        peck_total = float(peck_rate) * float(depth_float)
                        if math.isfinite(peck_total) and peck_total > 0:
                            summary["peck_sum"] += peck_total * qty_for_debug
                            summary["peck_count"] += qty_for_debug
                    dwell_val_float = to_float(overhead_local.dwell_min)
                    if dwell_val_float is not None and dwell_val_float > 0:
                        summary["dwell_sum"] += float(dwell_val_float) * qty_for_debug
                        summary["dwell_count"] += qty_for_debug
                    index_min_val = None
                    if debug_payload is not None:
                        index_min_val = to_float(
                            debug_payload.get("index_min")
                        )
                    if index_min_val is None:
                        index_sec_val = to_float(
                            getattr(
                                overhead_local,
                                "index_sec_per_hole",
                                None,
                            )
                        )
                        if index_sec_val is not None and index_sec_val > 0:
                            index_min_val = float(index_sec_val) / 60.0
                    if index_min_val is not None and index_min_val > 0:
                        summary["index_sum"] += float(index_min_val) * qty_for_debug
                        summary["index_count"] += qty_for_debug
                    if not summary.get("material"):
                        summary["material"] = "material"
                qty_int = qty_for_debug
        hole_count_for_clamp = total_holes
        if hole_count_for_clamp <= 0 and fallback_counts:
            hole_count_for_clamp = sum(
                max(0, int(qty)) for qty in fallback_counts.values() if qty
            )

        clamp_ratio = 1.0
        if total_min > 0 and hole_count_for_clamp > 0:
            uncapped_minutes = total_min
            clamped_hours = _apply_drill_minutes_clamp(
                total_min / 60.0,
                hole_count_for_clamp,
                material_group=material_label,
                depth_in=depth_for_bounds,
            )
            total_min = clamped_hours * 60.0
            if uncapped_minutes > 1e-9:
                clamp_ratio = total_min / uncapped_minutes

        if clamp_ratio != 1.0 and debug_summary_entries:
            for summary in debug_summary_entries.values():
                minutes_total = summary.get("total_minutes", 0.0) or 0.0
                summary["total_minutes"] = minutes_total * clamp_ratio

        if debug_lines is not None and debug_summary_entries:
            for op_key, summary in sorted(debug_summary_entries.items()):
                bins_map = summary.get("bins")
                if isinstance(bins_map, _MappingABC):
                    for bin_summary in bins_map.values():
                        if not isinstance(bin_summary, _MappingABC):
                            continue

                        def _merge_range(base_key: str) -> None:
                            min_key = f"{base_key}_min"
                            max_key = f"{base_key}_max"
                            min_val = _coerce_float_or_none(bin_summary.get(min_key))
                            max_val = _coerce_float_or_none(bin_summary.get(max_key))
                            if min_val is not None:
                                _update_range(summary, min_key, max_key, min_val)
                            if max_val is not None:
                                _update_range(summary, min_key, max_key, max_val)

                        _merge_range("sfm")
                        _merge_range("rpm")
                        _merge_range("ipm")
                        _merge_range("ipr")
                        _merge_range("ipr_effective")
                        _merge_range("depth")

                        dia_candidate = _coerce_float_or_none(
                            bin_summary.get("diameter_in")
                        )
                        if dia_candidate is not None:
                            current_min = summary.get("diam_min")
                            current_max = summary.get("diam_max")
                            if current_min is None or dia_candidate < current_min:
                                summary["diam_min"] = dia_candidate
                            if current_max is None or dia_candidate > current_max:
                                summary["diam_max"] = dia_candidate
                qty_total = summary.get("qty", 0)
                if qty_total <= 0:
                    continue
                minutes_total = summary.get("total_minutes", 0.0) or 0.0
                minutes_avg = minutes_total / qty_total if qty_total else 0.0
                toolchange_total = summary.get("toolchange_total", 0.0) or 0.0
                total_hours = (minutes_total + toolchange_total) / 60.0

                def _avg_value(sum_key: str, count_key: str) -> float | None:
                    total = summary.get(sum_key, 0.0) or 0.0
                    count = summary.get(count_key, 0) or 0
                    if count <= 0:
                        return None
                    return float(total) / float(count)

                def _format_avg(value: float | None, fmt: str) -> str:
                    coerced = _coerce_float_or_none(value)
                    if coerced is None or not math.isfinite(float(coerced)):
                        return "-"
                    return fmt.format(float(coerced))

                def _format_range(
                    min_val: float | None,
                    max_val: float | None,
                    fmt: str,
                    *,
                    tolerance: float = 0.0,
                ) -> str:
                    min_f = _coerce_float_or_none(min_val)
                    max_f = _coerce_float_or_none(max_val)
                    if min_f is None and max_f is None:
                        return "-"
                    if min_f is None:
                        min_f = max_f
                    if max_f is None:
                        max_f = min_f
                    if min_f is None or max_f is None:
                        return "-"
                    source_min = min_val if min_val is not None else max_val
                    source_max = max_val if max_val is not None else min_val
                    if source_min is None or source_max is None:
                        return "-"
                    try:
                        min_float = float(source_min)
                        max_float = float(source_max)
                    except (TypeError, ValueError):
                        return "-"
                    if not math.isfinite(min_float) or not math.isfinite(max_float):
                        return "-"
                    if tolerance and abs(max_float - min_float) <= tolerance:
                        return fmt.format(max_float)
                    if abs(max_float - min_float) <= 1e-12:
                        return fmt.format(max_float)
                    return f"{fmt.format(min_float)}–{fmt.format(max_float)}"

                sfm_avg = _avg_value("sfm_sum", "sfm_count")
                rpm_avg = _avg_value("rpm_sum", "rpm_count")
                ipm_avg = _avg_value("ipm_sum", "ipm_count")
                summary["rpm"] = rpm_avg
                summary["ipm"] = ipm_avg
                summary["minutes_per_hole"] = minutes_avg
                sfm_text = _format_range(
                    summary.get("sfm_min"), summary.get("sfm_max"), "{:.0f}", tolerance=0.5
                )
                if sfm_text == "-":
                    sfm_text = _format_avg(sfm_avg, "{:.0f}")
                ipr_min_val = summary.get("ipr_effective_min")
                if ipr_min_val is None:
                    ipr_min_val = summary.get("ipr_min")
                ipr_max_val = summary.get("ipr_effective_max")
                if ipr_max_val is None:
                    ipr_max_val = summary.get("ipr_max")
                ipr_text = _format_range(ipr_min_val, ipr_max_val, "{:.4f}", tolerance=5e-5)
                rpm_text = _format_range(summary.get("rpm_min"), summary.get("rpm_max"), "{:.0f}", tolerance=0.5)
                ipm_text = _format_range(summary.get("ipm_min"), summary.get("ipm_max"), "{:.1f}", tolerance=0.05)

                diam_qty = summary.get("diameter_qty_sum", 0) or 0
                dia_segment = "Ø -"
                if diam_qty > 0:
                    diam_sum = summary.get("diameter_weight_sum", 0.0) or 0.0
                    avg_dia = diam_sum / diam_qty if diam_qty else 0.0
                    diam_min = summary.get("diam_min")
                    diam_max = summary.get("diam_max")
                    dia_range_text = _format_range(diam_min, diam_max, "{:.3f}\"", tolerance=5e-4)
                    if dia_range_text == "-":
                        dia_range_text = f"{float(avg_dia):.3f}\""
                    dia_segment = f"Ø {dia_range_text}"

                depth_qty = summary.get("depth_qty_sum", 0) or 0
                depth_text = "-"
                if depth_qty > 0:
                    depth_sum = summary.get("depth_weight_sum", 0.0) or 0.0
                    avg_depth = depth_sum / depth_qty if depth_qty else 0.0
                    depth_min = summary.get("depth_min")
                    depth_max = summary.get("depth_max")
                    depth_range_text = _format_range(
                        depth_min, depth_max, "{:.2f}", tolerance=5e-3
                    )
                    if depth_range_text == "-":
                        depth_range_text = f"{float(avg_depth):.2f}"
                    depth_text = depth_range_text

                peck_avg = _avg_value("peck_sum", "peck_count")
                dwell_avg = _avg_value("dwell_sum", "dwell_count")
                index_avg = _avg_value("index_sum", "index_count")
                overhead_bits: list[str] = []
                if dwell_avg and math.isfinite(dwell_avg) and dwell_avg > 0:
                    overhead_bits.append(f"dwell {dwell_avg:.2f} min/hole")
                if index_avg and math.isfinite(index_avg) and index_avg > 0:
                    overhead_bits.append(f"index {index_avg:.2f} min/hole")
                if toolchange_total and math.isfinite(toolchange_total) and toolchange_total > 0:
                    overhead_bits.append(f"toolchange {toolchange_total:.2f} min")

                op_display = str(summary.get("operation") or "drill").title()
                mat_display = str(material_label or "").strip()
                if not mat_display:
                    mat_display = str(summary.get("material") or "").strip()
                if not mat_display:
                    mat_display = "material"
                summary["material"] = mat_display

                expected_group_display = str(
                    summary.get("expected_material_group")
                    or material_group_override
                    or ""
                ).strip().upper()
                row_group_display = str(
                    summary.get("material_group")
                    or summary.get("row_group")
                    or ""
                ).strip().upper()
                group_bits: list[str] = []
                if expected_group_display:
                    group_bits.append(f"group {expected_group_display}")
                if row_group_display:
                    group_bits.append(f"row {row_group_display}")
                if group_bits:
                    mat_segment = f"mat={mat_display} ({', '.join(group_bits)})"
                else:
                    mat_segment = f"mat={mat_display}"

                depth_segment = "depth/hole -"
                if depth_text != "-":
                    depth_segment = f"depth/hole {depth_text} in"

                peck_text = "-"
                if peck_avg and math.isfinite(peck_avg) and peck_avg > 0:
                    peck_text = f"{peck_avg:.2f} min/hole"

                toolchange_text = f"{toolchange_total:.2f} min"

                index_text = "-"
                if index_avg and math.isfinite(index_avg) and index_avg > 0:
                    index_text = f"{index_avg * 60.0:.1f} s/hole"

                line_parts = [
                    "Drill calc → ",
                    f"op={op_display}, {mat_segment}, ",
                    f"SFM={sfm_text}, IPR={ipr_text}; ",
                    f"RPM {rpm_text} IPM {ipm_text}; ",
                    f"{dia_segment}; {depth_segment}; ",
                    f"holes {qty_total}; ",
                    f"index {index_text}; ",
                    f"peck {peck_text}; ",
                    f"toolchange {toolchange_text}; ",
                ]
                if overhead_bits:
                    line_parts.append("overhead: " + ", ".join(overhead_bits) + "; ")
                line_parts.append(f"total hr {total_hours:.2f}.")
                if debug_lines is not None:
                    debug_lines.append("".join(line_parts))
                if debug_summary is not None:
                    debug_summary[op_key] = _jsonify_debug_value(summary)
        if missing_row_messages and warnings is not None:
            for op_display, material_display, dia_val in sorted(missing_row_messages):
                dia_text = f"{dia_val:.3f}".rstrip("0").rstrip(".")
                warning_text = (
                    f"No speeds/feeds row for {op_display}/"
                    f"{material_display.lower()} {dia_text} in — using fallback"
                )
                if warning_text not in warnings:
                    warnings.append(warning_text)
        total_minutes_with_toolchange = total_min + total_toolchange_min

        min_per_hole: float | None = None
        if total_holes > 0:
            min_per_hole = float(total_min) / float(total_holes)
        _update_debug_aggregate(
            hole_count=total_holes,
            avg_diameter=avg_dia_in,
            minutes_per_hole=min_per_hole,
        )
        if total_minutes_with_toolchange > 0:
            return total_minutes_with_toolchange / 60.0

    thickness_for_fallback_mm = thickness_mm_val
    if thickness_for_fallback_mm <= 0:
        depth_candidates = [depth for _, _, depth in group_specs if depth and depth > 0]
        if depth_candidates:
            thickness_for_fallback_mm = max(depth_candidates) * 25.4

    if not fallback_counts or thickness_for_fallback_mm <= 0:
        return 0.0

    def sec_per_hole(d_mm: float) -> float:
        if d_mm <= 3.5:
            return 10.0
        if d_mm <= 6.0:
            return 14.0
        if d_mm <= 10.0:
            return 18.0
        if d_mm <= 13.0:
            return 22.0
        if d_mm <= 20.0:
            return 30.0
        if d_mm <= 32.0:
            return 45.0
        return 60.0

    mfac = 0.8 if "alum" in mat else (1.15 if "stainless" in mat else 1.0)
    tfac = max(0.7, min(2.0, thickness_for_fallback_mm / 6.35))
    toolchange_s = 15.0

    total_sec = 0.0
    total_hole_qty = 0
    weighted_dia_in = 0.0
    for d, qty in fallback_counts.items():
        if qty is None:
            continue
        try:
            qty_int = int(qty)
        except Exception:
            continue
        if qty_int <= 0:
            continue
        total_hole_qty += qty_int
        per = sec_per_hole(float(d)) * mfac * tfac
        total_sec += qty_int * per
        total_sec += toolchange_s
        # aggregate counts and weighted diameter
        weighted_dia_in += (float(d) / 25.4) * qty_int

    holes_fallback = total_hole_qty

    hours = total_sec / 3600.0
    depth_for_bounds = None
    if thickness_for_fallback_mm and thickness_for_fallback_mm > 0:
        depth_for_bounds = float(thickness_for_fallback_mm) / 25.4
    clamped_hours = _apply_drill_minutes_clamp(
        hours,
        total_hole_qty,
        material_group=material_label,
        depth_in=depth_for_bounds,
    )

    avg_dia_for_debug = weighted_dia_in / holes_fallback if holes_fallback else 0.0
    min_per_hole_debug: float | None = None
    if holes_fallback > 0:
        min_per_hole_debug = (clamped_hours * 60.0) / holes_fallback
    _update_debug_aggregate(
        hole_count=total_hole_qty if holes_fallback > 0 else 0,
        avg_diameter=avg_dia_for_debug,
        minutes_per_hole=min_per_hole_debug,
    )

    if debug_summary is not None and debug is not None:
        debug_summary.setdefault("aggregate", {}).update(debug)

    return clamped_hours


legacy_estimate_drilling_hours = _legacy_estimate_drilling_hours

