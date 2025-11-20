"""Speeds/feeds time estimation helpers.

This module converts machining speeds and feeds guidance into a runtime
estimate (in minutes) for a given feature geometry.  The implementation is a
direct translation of the pseudocode supplied by the customer and is designed
to work with rows extracted from the speeds/feeds CSV used throughout the
quoting tool.

The entry point :func:`estimate_time_min` accepts a row, feature geometry,
tool parameters, machine constraints, and non-cutting overhead values.  The
row may be any mapping-like object (for example a ``pandas.Series``) or an
object with attributes matching the CSV column names.  Missing columns are
treated as ``None`` and sensible defaults are used throughout the
calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from cad_quoter.pricing.feed_math import (
    approach_allowance_for_drill,
    ipm_from_feed,
    passes_for_depth,
    rpm_from_sfm,
)

__all__ = [
    "MachineParams",
    "OperationGeometry",
    "OverheadParams",
    "ToolParams",
    "estimate_time_min",
    "estimate_wire_edm_minutes",
    "estimate_face_grind_minutes",
    "estimate_od_grind_minutes",
    "estimate_wet_grind_minutes",
    "estimate_square_up_side_mill_minutes",
    "estimate_square_up_face_mill_minutes",
    "estimate_square_up_grind_minutes",
    "MIN_PER_CUIN_GRIND",
    # Punch time estimation
    "PunchMachineHours",
    "PunchLaborHours",
    "PUNCH_TIME_CONSTANTS",
    "estimate_punch_machine_hours",
    "estimate_punch_labor_hours",
    "convert_punch_to_quote_machine_hours",
    "convert_punch_to_quote_labor_hours",
    "estimate_punch_times",
]


@dataclass(slots=True)
class OperationGeometry:
    """Geometric inputs for an operation."""

    diameter_in: float = 0.0
    depth_in: float = 0.0
    length_in: float = 0.0
    hole_depth_in: float | None = None
    thread_length_in: float | None = None
    pitch_in: float | None = None
    turn_length_in: float = 0.0
    point_angle_deg: float | None = None
    ld_ratio: float | None = None
    pass_count_override: int | None = None
    radial_stock_in: float = 0.0


@dataclass(slots=True)
class ToolParams:
    """Tool attributes relevant to feed calculations."""

    teeth_z: int | None = None


@dataclass(slots=True)
class MachineParams:
    """Machine capability constraints."""

    rapid_ipm: float | None = None
    hp_available: float | None = None
    hp_to_mrr_factor: float | None = None


@dataclass(slots=True)
class OverheadParams:
    """Non-cutting time adjustments."""

    toolchange_min: float | None = None
    approach_retract_in: float | None = None
    peck_penalty_min_per_in_depth: float | None = None
    dwell_min: float | None = None
    peck_min: float | None = None
    # Optional indexing time per hole (seconds). Some callers still populate
    # this; default to None so downstream "to_num" handling can coerce safely.
    index_sec_per_hole: float | None = None


class _RowView:
    """Lightweight adapter that exposes mapping keys as attributes."""

    __slots__ = ("_row",)

    def __init__(self, row: Any) -> None:
        self._row = row

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        row = self._row
        if row is None:
            return None
        if hasattr(row, name):
            return getattr(row, name)
        if isinstance(row, Mapping):
            return row.get(name)
        try:
            return row[name]  # type: ignore[index]
        except Exception:
            return None
def pick_feed_value(row: _RowView, diameter_in: float | None) -> float | None:
    """Choose the appropriate feed value based on the tool diameter."""

    d = to_num(diameter_in, 0.0)
    if d <= 0.1875:
        return to_num(row.fz_ipr_0_125in)
    if d <= 0.3750:
        return to_num(row.fz_ipr_0_25in)
    return to_num(row.fz_ipr_0_5in)


def to_num(value: Any, default: float | None = None) -> float | None:
    """Coerce ``value`` to ``float`` returning ``default`` on failure."""

    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _precomputed_value(precomputed: Mapping[str, Any] | None, key: str) -> float | None:
    """Return a numeric value from ``precomputed`` if available."""

    if not precomputed:
        return None
    try:
        value = precomputed.get(key)
    except AttributeError:  # pragma: no cover - defensive
        return None
    return to_num(value)
def cap_ipm_by_hp(
    ipm: float,
    operation: str,
    woc_in: float,
    doc_in: float,
    machine: MachineParams,
    material_factor: float | None,
) -> float:
    """Cap IPM to keep MRR within the machine's horsepower limits."""

    op = (operation or "").lower()
    if not op.startswith("endmill"):
        return ipm
    mrr = max(woc_in, 1e-6) * max(doc_in, 1e-6) * max(ipm, 0.0)
    hp = to_num(machine.hp_available, 0.0) or 0.0
    mrr_factor = to_num(machine.hp_to_mrr_factor, 1.0) or 1.0
    cap = hp * mrr_factor
    if material_factor is not None:
        cap *= material_factor
    if cap > 0.0 and mrr > cap:
        scale = cap / mrr
        return ipm * scale
    return ipm


def noncut_time_min(overhead: OverheadParams, num_passes: int) -> float:
    """Return non-cutting time in minutes for the operation."""

    minutes = 0.0
    minutes += to_num(overhead.toolchange_min, 0.0) or 0.0
    dwell = to_num(overhead.dwell_min, 0.0) or 0.0
    if dwell:
        minutes += dwell * max(num_passes, 1)
    return minutes


def time_endmill_profile(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
    material_factor: float | None,
) -> float:
    sfm = to_num(row.sfm_start, 0.0) or 0.0
    rpm = rpm_from_sfm(sfm, geom.diameter_in)
    fz = pick_feed_value(row, geom.diameter_in)
    ipm = ipm_from_feed("fz", fz, rpm, tool.teeth_z)

    doc_ax = to_num(row.doc_axial_in, 1.0) or 1.0
    passes, step = passes_for_depth(geom.depth_in, doc_ax, geom.pass_count_override)

    woc_pct = to_num(row.woc_radial_pct, 30.0) or 30.0
    woc_in = max((to_num(geom.diameter_in, 0.0) or 0.0) * (woc_pct / 100.0), 1e-6)

    ipm = cap_ipm_by_hp(ipm, row.operation or "", woc_in, step, machine, material_factor)

    approach = to_num(overhead.approach_retract_in, 0.0) or 0.0
    per_pass_len = max(to_num(geom.length_in, 0.0) or 0.0, 0.0) + 2.0 * approach
    cut_min = (per_pass_len / max(ipm, 1e-6)) * passes
    return cut_min + noncut_time_min(overhead, passes)


def time_endmill_slot(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
    material_factor: float | None,
) -> float:
    return time_endmill_profile(row, geom, tool, machine, overhead, material_factor)


def time_drill(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
    *,
    debug: dict[str, Any] | None = None,
    precomputed: Mapping[str, Any] | None = None,
) -> float:
    sfm = _precomputed_value(precomputed, "sfm")
    if sfm is None:
        sfm = to_num(row.sfm_start, 0.0) or 0.0
    rpm = _precomputed_value(precomputed, "rpm")
    if rpm is None:
        rpm = rpm_from_sfm(sfm, geom.diameter_in)
    ipr = _precomputed_value(precomputed, "ipr")
    if ipr is None:
        ipr = pick_feed_value(row, geom.diameter_in)
    ipm = _precomputed_value(precomputed, "ipm")
    if ipm is None:
        ipm = ipm_from_feed("ipr", ipr, rpm, None)

    axial_depth = (
        max(to_num(geom.hole_depth_in, 0.0) or to_num(geom.depth_in, 0.0) or 0.0, 0.0)
        + approach_allowance_for_drill(geom.diameter_in, geom.point_angle_deg or 118.0)
        + 0.1 * (to_num(geom.diameter_in, 0.0) or 0.0)
    )

    peck = to_num(overhead.peck_min, None)
    if peck is None:
        peck = (to_num(overhead.peck_penalty_min_per_in_depth, 0.0) or 0.0) * axial_depth
    peck = to_num(peck, 0.0) or 0.0
    cut_min = axial_depth / max(ipm, 1e-6)

    approach = to_num(overhead.approach_retract_in, 0.0) or 0.0
    rapid_ipm = to_num(machine.rapid_ipm, 0.0) or 1.0
    rapid_min = (2.0 * approach) / max(rapid_ipm, 1.0)

    noncut = noncut_time_min(overhead, 1)
    index_min = (to_num(getattr(overhead, "index_sec_per_hole", 0.0), 0.0) or 0.0) / 60.0
    total = cut_min + peck + rapid_min + noncut + index_min

    if debug is not None:
        debug.setdefault("sfm", sfm)
        debug.setdefault("ipr", ipr)
        debug.setdefault("rpm", rpm)
        debug.setdefault("ipm", ipm)
        debug.setdefault("axial_depth_in", axial_depth)
        debug.setdefault("peck_min_per_hole", peck)
        debug.setdefault("minutes_per_hole", total)
        debug.setdefault("index_min", index_min)

    return total


def time_deep_drill(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
    *,
    debug: dict[str, Any] | None = None,
    precomputed: Mapping[str, Any] | None = None,
) -> float:
    base_sfm = _precomputed_value(precomputed, "sfm")
    if base_sfm is None:
        base_sfm = to_num(row.sfm_start, 0.0) or 0.0
    deep_sfm = base_sfm * 0.65

    base_ipr = _precomputed_value(precomputed, "ipr")
    if base_ipr is None:
        base_ipr = pick_feed_value(row, geom.diameter_in) or 0.0
    deep_ipr = base_ipr * 0.70

    rpm = rpm_from_sfm(deep_sfm, geom.diameter_in)
    ipm = ipm_from_feed("ipr", deep_ipr, rpm, None)

    precomputed_local: dict[str, Any] = {}
    if precomputed:
        try:
            precomputed_local = dict(precomputed)
        except Exception:  # pragma: no cover - defensive
            precomputed_local = {}
    precomputed_local.update({
        "sfm": deep_sfm,
        "ipr": deep_ipr,
        "rpm": rpm,
        "ipm": ipm,
    })

    minutes = time_drill(
        row,
        geom,
        tool,
        machine,
        overhead,
        debug=debug,
        precomputed=precomputed_local,
    )

    hole_depth = max(to_num(geom.hole_depth_in, 0.0) or to_num(geom.depth_in, 0.0) or 0.0, 0.0)
    axial_depth = hole_depth + approach_allowance_for_drill(
        geom.diameter_in,
        geom.point_angle_deg or 118.0,
    )
    axial_depth += 0.1 * (to_num(geom.diameter_in, 0.0) or 0.0)

    diameter = to_num(geom.diameter_in, 0.0) or 0.0
    ld_ratio = to_num(geom.ld_ratio, None)
    if ld_ratio is None or ld_ratio <= 0.0:
        if diameter > 0.0:
            ld_ratio = axial_depth / diameter
        else:
            ld_ratio = 0.0
    ld_ratio = max(ld_ratio, 0.0)

    penalty_min = 0.05
    penalty_span = 0.08 - penalty_min
    if penalty_span < 0.0:
        penalty_span = 0.0
    scale = 0.0
    if ld_ratio > 3.0 and penalty_span > 0.0:
        scale = min(max((ld_ratio - 3.0) / 9.0, 0.0), 1.0)
    penalty_rate = penalty_min + (penalty_span * scale)
    peck_extra = penalty_rate * axial_depth

    total_minutes = minutes + peck_extra

    if debug is not None:
        existing_peck = to_num(debug.get("peck_min_per_hole"), 0.0) or 0.0
        debug["peck_min_per_hole"] = existing_peck + peck_extra
        existing_minutes = to_num(debug.get("minutes_per_hole"), minutes) or minutes
        debug["minutes_per_hole"] = existing_minutes + peck_extra
        debug["deep_drill_peck_rate_min_per_in"] = penalty_rate

    return total_minutes


def time_ream(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
) -> float:
    sfm = to_num(row.sfm_start, 0.0) or 0.0
    rpm = rpm_from_sfm(sfm, geom.diameter_in)
    ipr = pick_feed_value(row, geom.diameter_in)
    ipm = ipm_from_feed("ipr", ipr, rpm, None)
    axial = max(to_num(geom.hole_depth_in, 0.0) or to_num(geom.depth_in, 0.0) or 0.0, 0.0)
    cut_min = axial / max(ipm, 1e-6)

    approach = to_num(overhead.approach_retract_in, 0.0) or 0.0
    rapid_ipm = to_num(machine.rapid_ipm, 0.0) or 1.0
    rapid_min = (2.0 * approach) / max(rapid_ipm, 1.0)
    index_min = (to_num(getattr(overhead, "index_sec_per_hole", 0.0), 0.0) or 0.0) / 60.0
    return cut_min + rapid_min + noncut_time_min(overhead, 1) + index_min


def time_tap_roll_form(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
) -> float:
    sfm = to_num(row.sfm_start, 0.0) or 0.0
    rpm = rpm_from_sfm(sfm, geom.diameter_in)
    pitch = max(to_num(geom.pitch_in, 0.0) or 0.0, 1e-6)
    ipm = ipm_from_feed("pitch", pitch, rpm, None)

    thread_length = max(
        to_num(geom.thread_length_in, 0.0) or to_num(geom.depth_in, 0.0) or 0.0,
        0.0,
    )
    down_min = thread_length / max(ipm, 1e-6)
    up_min = down_min / 0.7

    approach = to_num(overhead.approach_retract_in, 0.0) or 0.0
    rapid_ipm = to_num(machine.rapid_ipm, 0.0) or 1.0
    rapid_min = (2.0 * approach) / max(rapid_ipm, 1.0)
    index_min = (to_num(getattr(overhead, "index_sec_per_hole", 0.0), 0.0) or 0.0) / 60.0
    return down_min + up_min + rapid_min + noncut_time_min(overhead, 1) + index_min


def time_thread_mill(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
    material_factor: float | None,
) -> float:
    circumference = 3.141592653589793 * max(to_num(geom.diameter_in, 0.0) or 0.0, 1e-6)
    pitch = max(to_num(geom.pitch_in, 0.0) or 0.0, 1e-6)
    helix_per_rev = (circumference**2 + pitch**2) ** 0.5
    revs = max(
        (to_num(geom.thread_length_in, 0.0) or to_num(geom.depth_in, 0.0) or 0.0) / pitch,
        0.0,
    )

    sfm = to_num(row.sfm_start, 0.0) or 0.0
    rpm = rpm_from_sfm(sfm, geom.diameter_in)
    fz = pick_feed_value(row, geom.diameter_in)
    ipm = ipm_from_feed("fz", fz, rpm, tool.teeth_z)

    woc = 0.1 * (to_num(geom.diameter_in, 0.0) or 0.0)
    ipm = cap_ipm_by_hp(ipm, row.operation or "", woc, fz or 0.0, machine, material_factor)

    approach = to_num(overhead.approach_retract_in, 0.0) or 0.0
    path_len = helix_per_rev * revs
    per_pass_len = path_len + 2.0 * approach
    passes = max(int(to_num(geom.pass_count_override, 0.0) or 1), 1)
    cut_min = (per_pass_len / max(ipm, 1e-6)) * passes
    index_min = (to_num(getattr(overhead, "index_sec_per_hole", 0.0), 0.0) or 0.0) / 60.0
    return cut_min + noncut_time_min(overhead, passes) + index_min


def time_turn_rough(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
) -> float:
    sfm = to_num(row.sfm_start, 0.0) or 0.0
    rpm = rpm_from_sfm(sfm, geom.diameter_in)
    fpr = pick_feed_value(row, geom.diameter_in)
    ipm = ipm_from_feed("fpr", fpr, rpm, None)

    passes, _ = passes_for_depth(
        getattr(geom, "radial_stock_in", None),
        row.doc_axial_in,
        geom.pass_count_override,
    )
    approach = to_num(overhead.approach_retract_in, 0.0) or 0.0
    per_pass_len = max(to_num(geom.turn_length_in, 0.0) or 0.0, 0.0) + 2.0 * approach
    cut_min = (per_pass_len / max(ipm, 1e-6)) * passes
    return cut_min + noncut_time_min(overhead, passes)


def time_turn_finish(
    row: _RowView,
    geom: OperationGeometry,
    tool: ToolParams,
    machine: MachineParams,
    overhead: OverheadParams,
) -> float:
    return time_turn_rough(row, geom, tool, machine, overhead)


def time_wire_edm(
    row: _RowView,
    geom: OperationGeometry,
    overhead: OverheadParams,
) -> float:
    lcr = to_num(row.linear_cut_rate_ipm, 0.0) or 0.0
    path_len = max(to_num(geom.length_in, 0.0) or 0.0, 0.0)
    passes = max(int(to_num(geom.pass_count_override, 0.0) or 1), 1)
    cut_min = (path_len / max(lcr, 1e-6)) * passes
    return cut_min + noncut_time_min(overhead, passes)


# ---------- Punch Planner Estimation Stubs --------------------------------
# These functions support time estimation for punch manufacturing processes
# including Wire EDM profiling, face grinding, and OD grinding operations.

MIN_PER_CUIN_GRIND = 3.0  # Global rule for volume-based grinding


def _try_float(x: Any, default: float = 0.0) -> float:
    """Convert value to float with fallback. Handles NaN and inf."""
    try:
        v = float(x)
        # Treat NaN/inf as default
        if v != v or v == float("inf") or v == float("-inf"):
            return default
        return v
    except Exception:
        return default


def _csv_row(
    material: str,
    material_group: str,
    operation_hint: str = "Generic",
    *,
    _get_speeds_feeds_fn: Any = None,
) -> dict[str, Any]:
    """Fetch CSV row for material and operation with fallback chain.

    This is a helper that tries material first, then material_group, then GENERIC.
    The _get_speeds_feeds_fn parameter allows injection for testing/flexibility.
    """
    # Import here to avoid circular dependency
    if _get_speeds_feeds_fn is None:
        try:
            from cad_quoter.planning.process_planner import get_speeds_feeds
            _get_speeds_feeds_fn = get_speeds_feeds
        except ImportError:
            return {}

    # Try material first
    row = _get_speeds_feeds_fn(material=material, operation=operation_hint)
    if row:
        return row

    # Try material group
    if material_group:
        row = _get_speeds_feeds_fn(material=material_group, operation=operation_hint)
        if row:
            return row

    # Fall back to GENERIC
    row = _get_speeds_feeds_fn(material="GENERIC", operation=operation_hint)
    return row or {}


def _wire_mins_per_in(material: str, material_group: str, thickness_in: float) -> float:
    """Calculate wire EDM minutes per inch based on material and thickness.

    Uses banded thickness approach with fallback to conservative defaults.
    """
    row = _csv_row(material, material_group, operation_hint="Wire_EDM")

    # Try generic wire_mins_per_in column first
    base = _try_float(row.get("wire_mins_per_in"), default=None)
    if base is not None and base > 0:
        return max(0.01, base)

    # Use thickness-based banding
    if thickness_in <= 0.125:
        band = "wire_mpi_thin"
    elif thickness_in <= 0.500:
        band = "wire_mpi_med"
    else:
        band = "wire_mpi_thick"

    return max(0.01, _try_float(row.get(band), default=0.9))


def estimate_wire_edm_minutes(
    op: dict[str, Any],
    material: str,
    material_group: str,
) -> float:
    """Estimate Wire EDM time for punch profile cutting.

    Args:
        op: Operation dict containing wire_profile_perimeter_in and thickness_in
        material: Material name
        material_group: Material group (ISO group or similar)

    Returns:
        Estimated minutes for the wire EDM operation
    """
    perim = _try_float(op.get("wire_profile_perimeter_in", 0.0))
    thk = _try_float(op.get("thickness_in", 0.0))
    mpi = _wire_mins_per_in(material, material_group, thk)
    return max(0.0, perim * mpi)


def _grind_factor(material: str, material_group: str) -> float:
    """Get material-specific grinding factor with fallback chain.

    Searches for grinding_time_factor, grind_factor, grind_material_factor, or material_factor.
    Returns 1.0 if not found.
    """
    # Try Grinding operation first, then Generic
    for op_hint in ("Grinding", "Generic"):
        row = _csv_row(material, material_group, operation_hint=op_hint)
        for k in ("grinding_time_factor", "grind_factor", "grind_material_factor", "material_factor"):
            val = row.get(k)
            if val not in (None, ""):
                try:
                    f = float(val)
                    if f > 0:
                        return f
                except Exception:
                    pass
    return 1.0


def _edm_material_factor(material: str, material_group: str) -> float:
    """Get material-specific EDM time factor with fallback chain.

    Searches for edm_time_factor, edm_factor, edm_material_factor, or material_factor.
    Returns 1.0 if not found.
    """
    # Try Wire_EDM operation first, then Generic
    for op_hint in ("Wire_EDM", "Wire_EDM_Rough", "Generic"):
        row = _csv_row(material, material_group, operation_hint=op_hint)
        for k in ("edm_time_factor", "edm_factor", "edm_material_factor", "material_factor"):
            val = row.get(k)
            if val not in (None, ""):
                try:
                    f = float(val)
                    if f > 0:
                        return f
                except Exception:
                    pass
    return 1.0


def estimate_face_grind_minutes(
    length_in: float,
    width_in: float,
    stock_removed_total_in: float,
    material: str,
    material_group: str,
    faces: int = 2,
) -> float:
    """Estimate face/surface grinding time based on volume removal.

    Uses the canonical formula: minutes = volume × 3.0 × grind_factor

    Args:
        length_in: Part length in inches
        width_in: Part width in inches
        stock_removed_total_in: Total stock to remove (both faces combined)
        material: Material name
        material_group: Material group
        faces: Number of faces (default 2 for top+bottom pair)

    Returns:
        Estimated minutes for face grinding
    """
    # Calculate volume: faces parameter represents pairs (2 faces = 1 pair)
    volume_cuin = (length_in * width_in * stock_removed_total_in) * max(1, faces // 2)
    factor = _grind_factor(material, material_group)
    return max(0.0, volume_cuin * MIN_PER_CUIN_GRIND * factor)


def estimate_wet_grind_minutes(
    length_in: float,
    width_in: float,
    stock_removed_total: float = 0.050,
    material: str = "GENERIC",
    material_group: str = "",
    faces: int = 2,
) -> tuple[float, float]:
    """Estimate wet grind time using the agreed formula.

    Formula: minutes = (L * W * stock_removed_total) * 3.0 * grind_factor(material)

    Args:
        length_in: Part length in inches
        width_in: Part width in inches
        stock_removed_total: Total stock to remove (default 0.050 = 0.025 per face × 2)
        material: Material name
        material_group: Material group (ISO group or similar)
        faces: Number of faces (default 2 for top+bottom)

    Returns:
        Tuple of (minutes, grind_material_factor) so renderer can display the factor
    """
    volume_cuin = length_in * width_in * stock_removed_total
    factor = _grind_factor(material, material_group)
    minutes = max(0.0, volume_cuin * MIN_PER_CUIN_GRIND * factor)
    return minutes, factor


def estimate_od_grind_minutes(
    meta: dict[str, Any],
    material: str,
    material_group: str,
    prefer_circ_model: bool = False,
) -> float:
    """Estimate OD grinding time for round pierce punches.

    Two models available:
    1. Volume model (default): minutes = volume_removed × 3.0 × grind_factor
    2. Circumference model (optional): minutes = (circumference × length) × min_per_circ_in × factor

    Args:
        meta: Metadata dict with od_grind_volume_removed_cuin, od_grind_circumference_in,
              od_length_in, diameter, thickness, stock_allow_radial
        material: Material name
        material_group: Material group
        prefer_circ_model: If True and CSV has grind_min_per_circ_in, use circumference model

    Returns:
        Estimated minutes for OD grinding
    """
    from math import pi

    factor = _grind_factor(material, material_group)

    # Check for circumference rate in CSV if user wants it
    row = _csv_row(material, material_group, operation_hint="Grinding")
    circ_rate = _try_float(row.get("grind_min_per_circ_in"), default=None)

    if prefer_circ_model and circ_rate:
        circ = _try_float(meta.get("od_grind_circumference_in"))
        length = _try_float(meta.get("od_length_in"))
        return max(0.0, circ * length * circ_rate * factor)

    # Default volume model
    vol = _try_float(meta.get("od_grind_volume_removed_cuin"))
    if vol <= 0.0:
        # If volume not present, try to infer from D, T, and radial stock
        D = _try_float(meta.get("diameter"))
        T = _try_float(meta.get("thickness"))
        stock = _try_float(meta.get("stock_allow_radial"), default=0.003)
        if D > 0 and T > 0 and stock > 0:
            r = D / 2.0
            vol = pi * (r * r - (r - stock) ** 2) * T

    return max(0.0, vol * MIN_PER_CUIN_GRIND * factor)


def estimate_square_up_side_mill_minutes(
    perimeter_in: float,
    side_stock_in: float,
    tool_diameter_in: float,
    feed_ipm: float,
    axial_depth_per_pass_in: float,
    total_axial_depth_in: float,
    material: str = "GENERIC",
    material_group: str = "",
    debug: dict[str, Any] | None = None,
) -> float:
    """Estimate side milling time for square-up operations.

    Calculates time based on perimeter, radial stock removal, tool diameter,
    feed rate, and number of passes required.

    Args:
        perimeter_in: Part perimeter in inches
        side_stock_in: Radial stock to remove per side in inches
        tool_diameter_in: Tool diameter in inches
        feed_ipm: Feed rate in inches per minute
        axial_depth_per_pass_in: Axial depth of cut per pass in inches
        total_axial_depth_in: Total axial depth to cut (part height)
        material: Material name for factor lookup
        material_group: Material group for factor lookup
        debug: Optional dict to populate with calculation details

    Returns:
        Estimated minutes for side milling operation
    """
    # Calculate number of radial passes needed (stepover-based)
    stepover_in = tool_diameter_in * 0.5  # 50% stepover typical for roughing
    radial_passes = max(1, int((side_stock_in / stepover_in) + 0.999))  # Round up

    # Calculate number of axial passes needed
    axial_passes = max(1, int((total_axial_depth_in / axial_depth_per_pass_in) + 0.999))

    # Total pass count
    total_passes = radial_passes * axial_passes

    # Path length per pass (perimeter with approach/retract allowance)
    path_per_pass_in = perimeter_in + (2.0 * tool_diameter_in)  # Entry/exit allowance

    # Total path length
    total_path_in = path_per_pass_in * total_passes

    # Calculate volume removed for reference
    volume_removed_cuin = perimeter_in * total_axial_depth_in * side_stock_in

    # Calculate time
    if feed_ipm > 0:
        cut_time_min = total_path_in / feed_ipm
    else:
        cut_time_min = 0.0

    # Add non-cutting overhead (tool changes, approach, etc.)
    overhead_min = 0.5  # Minimal overhead for square-up
    total_time_min = cut_time_min + overhead_min

    # Populate debug dict if provided
    if debug is not None:
        debug["sq_perimeter"] = perimeter_in
        debug["sq_side_stock"] = side_stock_in
        debug["sq_tool_dia"] = tool_diameter_in
        debug["sq_feed_ipm"] = feed_ipm
        debug["sq_radial_passes"] = radial_passes
        debug["sq_axial_passes"] = axial_passes
        debug["sq_pass_count"] = total_passes
        debug["sq_path_length"] = total_path_in
        debug["volume_removed_cuin"] = volume_removed_cuin
        debug["sq_time_min"] = total_time_min

    return max(0.0, total_time_min)


def estimate_square_up_face_mill_minutes(
    length_in: float,
    width_in: float,
    top_bottom_stock_in: float,
    tool_diameter_in: float,
    feed_ipm: float,
    pass_count: int = 3,
    material: str = "GENERIC",
    material_group: str = "",
    debug: dict[str, Any] | None = None,
) -> float:
    """Estimate face milling time for square-up top/bottom operations.

    Calculates time based on part dimensions, stock removal, tool diameter,
    feed rate, and number of passes.

    Args:
        length_in: Part length in inches
        width_in: Part width in inches
        top_bottom_stock_in: Total stock to remove from top & bottom in inches
        tool_diameter_in: Face mill diameter in inches
        feed_ipm: Feed rate in inches per minute
        pass_count: Number of passes per face (default 3)
        material: Material name for factor lookup
        material_group: Material group for factor lookup
        debug: Optional dict to populate with calculation details

    Returns:
        Estimated minutes for face milling operation
    """
    # Calculate number of stripes needed to cover width
    stepover_in = width_in / 3.0  # Typically 3 stripes with ~5% overlap
    num_stripes = 3

    # Path length per pass (length with entry/exit allowance)
    path_per_pass_in = length_in + (2.0 * 0.5)  # Small entry/exit allowance

    # Total passes = passes per face × 2 faces
    total_passes = pass_count * 2

    # Total path length
    total_path_in = path_per_pass_in * total_passes

    # Calculate surface area and volume removed
    surface_area_sq_in = length_in * width_in * 2  # Both faces
    volume_removed_cuin = length_in * width_in * top_bottom_stock_in

    # Calculate time
    if feed_ipm > 0:
        cut_time_min = total_path_in / feed_ipm
    else:
        cut_time_min = 0.0

    # Add overhead (5% for tool positioning)
    overhead_factor = 1.05
    total_time_min = cut_time_min * overhead_factor

    # Populate debug dict if provided
    if debug is not None:
        debug["sq_length"] = length_in
        debug["sq_width"] = width_in
        debug["sq_top_bottom_stock"] = top_bottom_stock_in
        debug["sq_tool_dia"] = tool_diameter_in
        debug["sq_feed_ipm"] = feed_ipm
        debug["sq_pass_count"] = total_passes
        debug["sq_path_length"] = total_path_in
        debug["surface_area_sq_in"] = surface_area_sq_in
        debug["volume_removed_cuin"] = volume_removed_cuin
        debug["sq_time_min"] = total_time_min

    return max(0.0, total_time_min)


def estimate_square_up_grind_minutes(
    length_in: float,
    width_in: float,
    top_bottom_stock_in: float,
    material: str = "GENERIC",
    material_group: str = "",
    faces: int = 2,
    debug: dict[str, Any] | None = None,
) -> float:
    """Estimate grinding time for square-up operations.

    Uses volume-based formula consistent with other grinding operations.

    Args:
        length_in: Part length in inches
        width_in: Part width in inches
        top_bottom_stock_in: Total stock to remove from top & bottom in inches
        material: Material name for factor lookup
        material_group: Material group for factor lookup
        faces: Number of faces to grind (default 2)
        debug: Optional dict to populate with calculation details

    Returns:
        Estimated minutes for grinding operation
    """
    # Calculate volume removed
    volume_removed_cuin = length_in * width_in * top_bottom_stock_in
    surface_area_sq_in = length_in * width_in * faces

    # Get material factor
    factor = _grind_factor(material, material_group)

    # Use canonical grinding formula: minutes = volume × 3.0 × factor
    grind_time_min = volume_removed_cuin * MIN_PER_CUIN_GRIND * factor

    # Populate debug dict if provided
    if debug is not None:
        debug["sq_length"] = length_in
        debug["sq_width"] = width_in
        debug["sq_top_bottom_stock"] = top_bottom_stock_in
        debug["surface_area_sq_in"] = surface_area_sq_in
        debug["volume_removed_cuin"] = volume_removed_cuin
        debug["grind_material_factor"] = factor
        debug["grind_time_min"] = grind_time_min

    return max(0.0, grind_time_min)


def estimate_time_min(
    row: Any | None = None,
    geom: OperationGeometry | None = None,
    tool: ToolParams | None = None,
    machine: MachineParams | None = None,
    overhead: OverheadParams | None = None,
    material_factor: float | None = None,
    *,
    operation: str | None = None,
    debug: dict[str, Any] | None = None,
    precomputed: Mapping[str, Any] | None = None,
) -> float:
    """Dispatch to the appropriate time estimator based on the operation."""

    if geom is None or tool is None or machine is None or overhead is None:
        raise TypeError("estimate_time_min requires geometry, tool, machine, and overhead parameters")

    if row is None:
        if operation is None:
            raise TypeError("estimate_time_min requires either a row or an operation name")
        row = {"operation": operation}
    elif isinstance(row, str):
        if operation is None:
            operation = row
        row = {"operation": row}
    elif operation is not None:
        try:
            mapping = dict(row)
        except Exception:
            mapping = {"operation": operation}
        else:
            mapping.setdefault("operation", operation)
        row = mapping

    row_view = _RowView(row)
    op_name = operation or row_view.operation
    op = (op_name or "").lower().replace("-", "_").replace(" ", "_")

    if "wire_edm" in op:
        return time_wire_edm(row_view, geom, overhead)
    if "endmill_profile" in op:
        return time_endmill_profile(row_view, geom, tool, machine, overhead, material_factor)
    if "endmill_slot" in op:
        return time_endmill_slot(row_view, geom, tool, machine, overhead, material_factor)
    if op == "drill":
        if debug is not None:
            debug.setdefault("operation", op)
        return time_drill(
            row_view,
            geom,
            tool,
            machine,
            overhead,
            debug=debug,
            precomputed=precomputed,
        )
    if "deep_drill" in op:
        if debug is not None:
            debug.setdefault("operation", op)
        return time_deep_drill(
            row_view,
            geom,
            tool,
            machine,
            overhead,
            debug=debug,
            precomputed=precomputed,
        )
    if "ream" in op:
        return time_ream(row_view, geom, tool, machine, overhead)
    if "tap" in op:
        return time_tap_roll_form(row_view, geom, tool, machine, overhead)
    if "thread_mill" in op:
        return time_thread_mill(row_view, geom, tool, machine, overhead, material_factor)
    if "turn_rough" in op:
        return time_turn_rough(row_view, geom, tool, machine, overhead)
    if "turn_finish" in op:
        return time_turn_finish(row_view, geom, tool, machine, overhead)
    return 0.0


# ============================================================================
# PUNCH TIME ESTIMATION
# ============================================================================


@dataclass
class PunchMachineHours:
    """Machine hours breakdown for punch manufacturing."""
    rough_turning_min: float = 0.0
    finish_turning_min: float = 0.0
    od_grinding_min: float = 0.0
    id_grinding_min: float = 0.0
    face_grinding_min: float = 0.0
    drilling_min: float = 0.0
    tapping_min: float = 0.0
    chamfer_min: float = 0.0
    polishing_min: float = 0.0
    edm_min: float = 0.0
    sawing_min: float = 0.0
    inspection_min: float = 0.0
    # Critical OD tracking for tolerance-sensitive operations
    critical_od_grinding_min: float = 0.0  # Extra grinding for tight tolerances
    critical_od_inspection_min: float = 0.0  # Extra inspection for critical dims
    critical_od_tolerance_in: float = 0.0  # Tightest OD tolerance on part
    total_minutes: float = 0.0
    total_hours: float = 0.0

    def calculate_totals(self):
        """Calculate total minutes and hours."""
        self.total_minutes = (
            self.rough_turning_min + self.finish_turning_min +
            self.od_grinding_min + self.id_grinding_min + self.face_grinding_min +
            self.drilling_min + self.tapping_min + self.chamfer_min +
            self.polishing_min + self.edm_min + self.sawing_min + self.inspection_min +
            self.critical_od_grinding_min + self.critical_od_inspection_min
        )
        self.total_hours = self.total_minutes / 60.0


@dataclass
class PunchLaborHours:
    """Labor hours breakdown for punch manufacturing."""
    lathe_setup_min: float = 0.0
    grinder_setup_min: float = 0.0
    edm_setup_min: float = 0.0
    cam_programming_min: float = 0.0
    handling_min: float = 0.0
    deburring_min: float = 0.0
    inspection_setup_min: float = 0.0
    first_article_min: float = 0.0
    total_minutes: float = 0.0
    total_hours: float = 0.0

    def calculate_totals(self):
        """Calculate total minutes and hours."""
        self.total_minutes = (
            self.lathe_setup_min + self.grinder_setup_min + self.edm_setup_min +
            self.cam_programming_min + self.handling_min + self.deburring_min +
            self.inspection_setup_min + self.first_article_min
        )
        self.total_hours = self.total_minutes / 60.0


PUNCH_TIME_CONSTANTS = {
    # New turning time model for round parts (formula-based)
    "base_turn_min": 3.0,  # Base turning setup time
    "per_inch_major_dia_min": 2.5,  # Time per inch of shank/major diameter
    "per_inch_minor_dia_min": 3.5,  # Time per inch of pilot/minor diameter (slower, more precision)
    "shoulder_factor": 1.5,  # Additional time per shoulder/diameter transition
    # New grinding time model for round parts (formula-based)
    "base_grind_min": 5.0,  # Base grinding setup time
    "per_inch_grind_min": 3.0,  # Time per inch of OD grinding (pilot + shank)
    "face_grind_min": 4.0,  # Time per face for perpendicular face grinding
    # Legacy turning constants (deprecated, use formula above)
    "rough_turning_per_diam": 8.0,
    "finish_turning_per_diam": 5.0,
    "od_grinding_per_inch": 3.0,
    "id_grinding_per_inch": 6.0,
    "face_grinding_per_face": 4.0,
    "drilling_per_hole": 2.0,
    "tapping_per_hole": 3.0,
    "chamfer_per_edge": 1.5,
    "small_radius_per_edge": 2.0,
    "polish_contour_base": 10.0,  # Reduced base from 30.0 - was too high
    "polish_per_sq_inch": 3.0,    # Reduced from 5.0 - was too aggressive
    "polish_per_radius": 2.0,     # Time per blend radius
    "form_complexity_multiplier": {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0},
    # Material factors for polish time - aluminum polishes faster than hardened steel
    "polish_material_factor": {
        "aluminum": 0.6,
        "6061": 0.6,
        "7075": 0.6,
        "brass": 0.7,
        "bronze": 0.7,
        "copper": 0.7,
        "a2": 1.0,
        "d2": 1.2,
        "m2": 1.2,
        "carbide": 1.5,
        "hss": 1.1,
        "default": 1.0,
    },
    # Cap polish time as fraction of turning time for form_punch parts
    # to prevent unreasonably high polish times on simple contours
    "polish_turning_cap_fraction": 0.30,  # Max 30% of turning time
    "polish_absolute_cap": 30.0,          # Absolute cap of 30 minutes for standard polish
    "polish_high_requirement_cap": 60.0,  # Higher cap for explicit high-polish or EDM burn notes
    "sawing_base": 5.0,
    "inspection_per_diam": 3.0,
    "tight_tolerance_multiplier": {0.0001: 2.0, 0.0002: 1.5, 0.0005: 1.2, 0.001: 1.0},
    "lathe_setup": 30.0,
    "grinder_setup": 20.0,
    "edm_setup": 45.0,
    "cam_programming_base": 30.0,
    "cam_per_operation": 5.0,
    "handling_per_operation": 2.0,
    "deburring_per_edge": 1.0,
    "inspection_setup": 10.0,
    "first_article_base": 20.0,
    # Simple round punch programming (no holes/threads, standard macros)
    "simple_punch_programming_base": 15.0,  # Base for simple round punch
    "simple_punch_per_extra_diam": 3.0,  # Per diameter step beyond first
    "simple_punch_per_chamfer": 2.0,  # Per chamfer (capped at 4)
    "simple_punch_per_radius": 1.0,  # Per small radius (capped at 4)
    "simple_punch_programming_cap": 30.0,  # Max for simple punch
    # Critical OD tolerance handling
    "critical_od_grinding_adder": 5.0,  # Extra grinding per critical diameter
    "critical_od_inspection_adder": 5.0,  # Extra inspection per critical diameter
    "critical_od_tolerance_threshold": 0.0005,  # Total band ≤ this triggers critical
}


def estimate_punch_machine_hours(punch_plan: dict[str, Any], punch_features: dict[str, Any]) -> PunchMachineHours:
    """Estimate machine hours for punch manufacturing."""
    hours = PunchMachineHours()
    tc = PUNCH_TIME_CONSTANTS

    num_diams = punch_features.get("num_ground_diams", 1)
    ground_length = punch_features.get("total_ground_length_in", 0.0)
    tap_count = punch_features.get("tap_count", 0)
    num_chamfers = punch_features.get("num_chamfers", 0)
    num_radii = punch_features.get("num_small_radii", 0)
    has_polish = punch_features.get("has_polish_contour", False)
    has_3d = punch_features.get("has_3d_surface", False)
    has_perp_face = punch_features.get("has_perp_face_grind", False)
    form_level = punch_features.get("form_complexity_level", 0)
    min_dia_tol = punch_features.get("min_dia_tol_in")
    overall_length = punch_features.get("overall_length_in", 0.0)
    max_od = punch_features.get("max_od_or_width_in", 0.0)
    # New turning time model parameters
    shank_length = punch_features.get("shank_length", 0.0)
    pilot_length = punch_features.get("pilot_length", 0.0)
    shoulder_count = punch_features.get("shoulder_count", 0)
    flange_thickness = punch_features.get("flange_thickness", 0.0)
    num_undercuts = punch_features.get("num_undercuts", 0)
    shape_type = punch_features.get("shape_type", "round")
    # New grinding time model parameters
    grind_pilot_len = punch_features.get("grind_pilot_len", 0.0)
    grind_shank_len = punch_features.get("grind_shank_len", 0.0)
    grind_head_faces = punch_features.get("grind_head_faces", 0)

    tol_mult = 1.0
    if min_dia_tol is not None:
        for threshold, mult in sorted(tc["tight_tolerance_multiplier"].items()):
            if min_dia_tol <= threshold:
                tol_mult = mult
                break

    hours.sawing_min = tc["sawing_base"]

    # New turning time model for round parts
    if num_diams > 0 and shape_type == "round":
        # Formula: turning_time_min = base_turn_min + per_inch_major_dia_min * shank_length
        #          + per_inch_minor_dia_min * pilot_length + shoulder_factor * shoulder_count
        base_turn_min = tc["base_turn_min"]
        per_inch_major = tc["per_inch_major_dia_min"]
        per_inch_minor = tc["per_inch_minor_dia_min"]
        shoulder_factor = tc["shoulder_factor"]

        turning_time_min = (
            base_turn_min
            + per_inch_major * shank_length
            + per_inch_minor * pilot_length
            + shoulder_factor * shoulder_count
        )

        # Apply tolerance multiplier for finish turning portion (assume 40% of time is finish)
        rough_portion = turning_time_min * 0.60
        finish_portion = turning_time_min * 0.40 * tol_mult
        total_turning = rough_portion + finish_portion

        hours.rough_turning_min = rough_portion
        hours.finish_turning_min = finish_portion

        # Print transparency information for turning time model
        print(f"  DEBUG [Turning time model]:")
        print(f"    shank_length={shank_length:.3f}\", pilot_length={pilot_length:.3f}\", "
              f"flange_thickness={flange_thickness:.3f}\", shoulder_count={shoulder_count}")
        print(f"    base_turn_min={base_turn_min:.2f}, per_inch_major={per_inch_major:.2f}, "
              f"per_inch_minor={per_inch_minor:.2f}, shoulder_factor={shoulder_factor:.2f}")
        print(f"    turning_time_min={total_turning:.2f} (rough={rough_portion:.2f}, finish={finish_portion:.2f})")

        # Guard check: flag if simple geometry has high turning time
        is_simple_geometry = (shoulder_count <= 1 and num_undercuts <= 1)
        if is_simple_geometry and total_turning > 45.0:
            warning = (f"CHECK: turning time high for simple round punch "
                      f"({total_turning:.1f} min for ≤1 shoulder, ≤1 undercut)")
            print(f"    WARNING: {warning}")
            punch_features.setdefault("warnings", []).append(warning)
    elif num_diams > 0:
        # Fallback to legacy model for non-round parts
        hours.rough_turning_min = num_diams * tc["rough_turning_per_diam"]
        hours.finish_turning_min = num_diams * tc["finish_turning_per_diam"] * tol_mult

    # New grinding time model for round parts
    if shape_type == "round" and (grind_pilot_len > 0 or grind_shank_len > 0 or grind_head_faces > 0):
        # Formula: grind_time_min = base_grind_min + per_inch_grind_min * (grind_pilot_len + grind_shank_len)
        #                          + face_grind_min * num_faces
        base_grind_min = tc["base_grind_min"]
        per_inch_grind = tc["per_inch_grind_min"]
        face_grind_per_face = tc["face_grind_min"]

        # OD grinding time
        od_grind_length = grind_pilot_len + grind_shank_len
        od_grind_time = base_grind_min + per_inch_grind * od_grind_length

        # Apply tolerance multiplier to OD grinding
        od_grind_time *= tol_mult

        hours.od_grinding_min = od_grind_time

        # Face grinding time
        if grind_head_faces > 0:
            hours.face_grinding_min = face_grind_per_face * grind_head_faces

        # Print transparency information for grinding time model
        print(f"  DEBUG [Grinding time model]:")
        print(f"    grind_pilot_len={grind_pilot_len:.3f}\", grind_shank_len={grind_shank_len:.3f}\", "
              f"grind_head_faces={grind_head_faces}")
        print(f"    base_grind_min={base_grind_min:.2f}, per_inch_grind_min={per_inch_grind:.2f}, "
              f"face_grind_min={face_grind_per_face:.2f}")
        print(f"    grind_time_min={od_grind_time + hours.face_grinding_min:.2f} "
              f"(OD={od_grind_time:.2f}, face={hours.face_grinding_min:.2f})")
    elif ground_length > 0:
        # Fallback to legacy model for non-round parts
        hours.od_grinding_min = ground_length * tc["od_grinding_per_inch"] * tol_mult
        if has_perp_face:
            hours.face_grinding_min = tc["face_grinding_per_face"] * 2

    if tap_count > 0:
        hours.drilling_min = tap_count * tc["drilling_per_hole"]
        hours.tapping_min = tap_count * tc["tapping_per_hole"]

    hours.chamfer_min = num_chamfers * tc["chamfer_per_edge"]
    hours.chamfer_min += num_radii * tc["small_radius_per_edge"]

    if has_polish or has_3d:
        # Calculate base polish time from contour area model
        contour_area = max_od * (overall_length * 0.3)
        form_mult = tc["form_complexity_multiplier"].get(form_level, 1.0)

        # Base calculation: reduced base + area contribution + radius contribution
        base_polish = tc["polish_contour_base"] + contour_area * tc["polish_per_sq_inch"]
        base_polish += num_radii * tc["polish_per_radius"]
        base_polish *= form_mult

        # Apply material factor - aluminum and soft metals polish faster
        material_callout = punch_features.get("material_callout", "")
        material_lower = (material_callout or "").lower()
        material_factor = tc["polish_material_factor"].get("default", 1.0)
        for mat_key, factor in tc["polish_material_factor"].items():
            if mat_key != "default" and mat_key in material_lower:
                material_factor = factor
                break

        polish_time = base_polish * material_factor

        # Get family for cap determination
        family = punch_features.get("family", "")

        # Calculate turning time for cap reference
        turning_time = hours.rough_turning_min + hours.finish_turning_min

        # Determine cap based on requirements
        # Check for explicit high-polish requirements that should allow higher time
        has_high_polish_requirement = any([
            punch_features.get("has_no_step_permitted", False),
            "edm" in material_lower or "burn" in material_lower,
            "mirror" in (punch_features.get("text_dump", "") or "").lower(),
            "optical" in (punch_features.get("text_dump", "") or "").lower(),
        ])

        # For form_punch parts, cap polish time relative to turning time
        # unless there's an explicit high-polish requirement
        if family == "form_punch" and not has_high_polish_requirement:
            # Cap at 30% of turning time or absolute cap, whichever is lower
            turning_cap = turning_time * tc["polish_turning_cap_fraction"]
            absolute_cap = tc["polish_absolute_cap"]
            effective_cap = min(turning_cap, absolute_cap) if turning_time > 0 else absolute_cap

            # Ensure minimum polish time of 5 minutes for any polish work
            polish_time = max(5.0, min(polish_time, effective_cap))
        elif has_high_polish_requirement:
            # Higher cap for explicit requirements
            polish_time = min(polish_time, tc["polish_high_requirement_cap"])
        else:
            # Standard cap for non-form_punch parts
            polish_time = min(polish_time, tc["polish_absolute_cap"])

        hours.polishing_min = polish_time

    hours.inspection_min = num_diams * tc["inspection_per_diam"]

    # Critical OD tolerance handling: extra grinding and inspection for gage-pin tolerances
    # Detect very tight diameter tolerances (total band <= 0.0005")
    if min_dia_tol is not None and min_dia_tol <= tc["critical_od_tolerance_threshold"]:
        hours.critical_od_tolerance_in = min_dia_tol
        # Extra fine grinding passes for critical diameters
        hours.critical_od_grinding_min = num_diams * tc["critical_od_grinding_adder"]
        # Extra inspection time for gage-pin checks on critical dimensions
        hours.critical_od_inspection_min = num_diams * tc["critical_od_inspection_adder"]

    hours.calculate_totals()

    return hours


def estimate_punch_labor_hours(punch_plan: dict[str, Any], punch_features: dict[str, Any], machine_hours: PunchMachineHours) -> PunchLaborHours:
    """Estimate labor hours for punch manufacturing."""
    labor = PunchLaborHours()
    tc = PUNCH_TIME_CONSTANTS

    ops = punch_plan.get("ops", [])
    num_ops = len(ops)
    num_chamfers = punch_features.get("num_chamfers", 0)
    num_radii = punch_features.get("num_small_radii", 0)
    tap_count = punch_features.get("tap_count", 0)
    has_polish = punch_features.get("has_polish_contour", False)
    has_3d = punch_features.get("has_3d_surface", False)
    num_diams = punch_features.get("num_ground_diams", 1)
    shape_type = punch_features.get("shape_type", "round")

    needs_lathe = machine_hours.rough_turning_min > 0 or machine_hours.finish_turning_min > 0
    needs_grinder = machine_hours.od_grinding_min > 0 or machine_hours.face_grinding_min > 0
    needs_edm = machine_hours.edm_min > 0

    if needs_lathe:
        labor.lathe_setup_min = tc["lathe_setup"]
    if needs_grinder:
        labor.grinder_setup_min = tc["grinder_setup"]
    if needs_edm:
        labor.edm_setup_min = tc["edm_setup"]

    # Programming time calculation with heuristic for simple round punches
    # A "simple" punch is: round shape, no holes/taps, no 3D surfaces, no polish
    is_simple_punch = (
        shape_type == "round" and
        tap_count == 0 and
        not has_3d and
        not has_polish and
        not needs_edm
    )

    if is_simple_punch:
        # Simple round punch: scale with turned features, not generic op count
        # Base time (standard lathe macros available)
        prog_time = tc["simple_punch_programming_base"]

        # Add time per diameter step beyond first
        if num_diams > 1:
            prog_time += (num_diams - 1) * tc["simple_punch_per_extra_diam"]

        # Add time per chamfer (capped at 4)
        prog_time += min(num_chamfers, 4) * tc["simple_punch_per_chamfer"]

        # Add time per small radius (capped at 4)
        prog_time += min(num_radii, 4) * tc["simple_punch_per_radius"]

        # Cap at maximum for simple punches
        labor.cam_programming_min = min(prog_time, tc["simple_punch_programming_cap"])
    else:
        # Complex punch: use original formula based on operation count
        labor.cam_programming_min = tc["cam_programming_base"] + num_ops * tc["cam_per_operation"]

    labor.handling_min = num_ops * tc["handling_per_operation"]

    total_edges = num_chamfers + num_radii + tap_count * 2
    labor.deburring_min = total_edges * tc["deburring_per_edge"]

    labor.inspection_setup_min = tc["inspection_setup"]
    labor.first_article_min = tc["first_article_base"]

    if has_polish:
        labor.first_article_min *= 1.5

    labor.calculate_totals()
    return labor


def convert_punch_to_quote_machine_hours(machine_hours: PunchMachineHours, labor_hours: PunchLaborHours) -> dict[str, Any]:
    """Convert PunchMachineHours to QuoteDataHelper format."""
    result = {
        "total_drill_minutes": machine_hours.drilling_min,
        "total_tap_minutes": machine_hours.tapping_min,
        "total_cbore_minutes": 0.0,
        "total_cdrill_minutes": 0.0,
        "total_jig_grind_minutes": 0.0,
        "total_milling_minutes": machine_hours.rough_turning_min + machine_hours.finish_turning_min,
        "total_grinding_minutes": machine_hours.od_grinding_min + machine_hours.id_grinding_min + machine_hours.face_grinding_min,
        "total_edm_minutes": machine_hours.edm_min,
        "total_other_minutes": machine_hours.sawing_min + machine_hours.chamfer_min + machine_hours.polishing_min,
        "total_cmm_minutes": machine_hours.inspection_min + machine_hours.critical_od_inspection_min,
        "total_minutes": machine_hours.total_minutes,
        "total_hours": machine_hours.total_hours,
    }

    # Add critical OD tracking for breakdown visibility
    if machine_hours.critical_od_tolerance_in > 0:
        result["critical_od_tolerance_in"] = machine_hours.critical_od_tolerance_in
        result["critical_od_grinding_minutes"] = machine_hours.critical_od_grinding_min
        result["critical_od_inspection_minutes"] = machine_hours.critical_od_inspection_min
        # Include critical OD grinding in total grinding
        result["total_grinding_minutes"] += machine_hours.critical_od_grinding_min

    return result


def convert_punch_to_quote_labor_hours(labor_hours: PunchLaborHours) -> dict[str, Any]:
    """Convert PunchLaborHours to QuoteDataHelper format."""
    return {
        "total_setup_minutes": labor_hours.lathe_setup_min + labor_hours.grinder_setup_min + labor_hours.edm_setup_min,
        "cam_programming_minutes": labor_hours.cam_programming_min,
        "handling_minutes": labor_hours.handling_min,
        "deburring_minutes": labor_hours.deburring_min,
        "inspection_minutes": labor_hours.inspection_setup_min + labor_hours.first_article_min,
        "total_minutes": labor_hours.total_minutes,
        "total_hours": labor_hours.total_hours,
    }


def estimate_punch_times(punch_plan: dict[str, Any], punch_features: dict[str, Any]) -> dict[str, Any]:
    """Main entry point for punch time estimation."""
    machine = estimate_punch_machine_hours(punch_plan, punch_features)
    labor = estimate_punch_labor_hours(punch_plan, punch_features, machine)

    return {
        "machine_hours": convert_punch_to_quote_machine_hours(machine, labor),
        "labor_hours": convert_punch_to_quote_labor_hours(labor),
        "punch_machine_breakdown": machine,
        "punch_labor_breakdown": labor,
    }

