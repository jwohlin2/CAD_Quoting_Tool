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
from math import ceil, radians, tan
from typing import Any, Mapping

__all__ = [
    "MachineParams",
    "OperationGeometry",
    "OverheadParams",
    "ToolParams",
    "estimate_time_min",
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


def rpm_from_sfm(sfm: float | None, diameter_in: float | None) -> float:
    """Convert surface feet per minute to spindle RPM."""

    sfm_val = to_num(sfm, 0.0)
    diameter = max(to_num(diameter_in, 0.0), 1e-6)
    return (sfm_val * 3.82) / diameter


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


def ipm_from_feed(
    feed_type: str,
    feed_val: float | None,
    rpm: float | None,
    teeth_z: int | None,
    *,
    linear_cut_rate_ipm: float | None = None,
) -> float:
    """Convert a chip-load style feed to linear inches per minute."""

    rpm_val = to_num(rpm, 0.0) or 0.0
    if feed_type == "fz":
        z = max(int(teeth_z or 1), 1)
        return (feed_val or 0.0) * z * rpm_val
    if feed_type in {"ipr", "fpr", "pitch"}:
        return (feed_val or 0.0) * rpm_val
    if feed_type == "linear":
        return to_num(linear_cut_rate_ipm, 0.0) or 0.0
    return 0.0


def passes_for_depth(
    depth_in: float | None,
    doc_axial_in: Any,
    pass_override: int | None = None,
) -> tuple[int, float]:
    """Return the number of passes and the per-pass step."""

    depth = to_num(depth_in, 0.0) or 0.0
    if pass_override is not None:
        passes = max(int(pass_override), 1)
        return passes, depth / passes if passes else depth
    doc = max(to_num(doc_axial_in, 0.0) or 0.0, 1e-6)
    passes = max(int(ceil(max(depth, 0.0) / doc)), 1)
    return passes, depth / passes if passes else depth


def approach_allowance_for_drill(
    diameter_in: float | None,
    point_angle_deg: float | None = None,
) -> float:
    """Approximate additional penetration required for the drill point."""

    diameter = to_num(diameter_in, 0.0) or 0.0
    if point_angle_deg is None:
        return 0.3 * diameter
    angle = max(float(point_angle_deg), 1e-6)
    return 0.5 * diameter * tan(radians(90.0 - (angle / 2.0)))


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
    index_min = (to_num(overhead.index_sec_per_hole, 0.0) or 0.0) / 60.0
    total = cut_min + peck + rapid_min + noncut + index_min

    if debug is not None:
        debug.setdefault("sfm", sfm)
        debug.setdefault("ipr", ipr)
        debug.setdefault("rpm", rpm)
        debug.setdefault("ipm", ipm)
        debug.setdefault("axial_depth_in", axial_depth)
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
    return time_drill(
        row,
        geom,
        tool,
        machine,
        overhead,
        debug=debug,
        precomputed=precomputed,
    )


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
    index_min = (to_num(overhead.index_sec_per_hole, 0.0) or 0.0) / 60.0
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
    index_min = (to_num(overhead.index_sec_per_hole, 0.0) or 0.0) / 60.0
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
    index_min = (to_num(overhead.index_sec_per_hole, 0.0) or 0.0) / 60.0
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

