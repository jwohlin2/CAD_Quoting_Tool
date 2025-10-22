"""Shared machining speed and feed math helpers.

These helpers provide canonical conversions between common speed and feed
representations used throughout the quoting application.  They are shared by
both the UI layer and the pricing estimator so the behaviour stays in sync.
"""

from __future__ import annotations

from math import ceil, radians, tan
from typing import Any

__all__ = [
    "rpm_from_sfm",
    "ipm_from_feed",
    "passes_for_depth",
    "approach_allowance_for_drill",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to ``float`` returning ``default`` on failure."""

    if value is None:
        return default
    try:
        number = float(value)
    except Exception:
        return default
    return number


def rpm_from_sfm(sfm: float | None, diameter_in: float | None) -> float:
    """Convert surface feet per minute to spindle RPM."""

    sfm_val = _to_float(sfm, 0.0)
    diameter = max(_to_float(diameter_in, 0.0), 1e-6)
    return (sfm_val * 3.82) / diameter


def ipm_from_feed(
    feed_type: str,
    feed_val: float | None,
    rpm: float | None,
    teeth_z: int | None,
    *,
    linear_cut_rate_ipm: float | None = None,
) -> float:
    """Convert a chip-load style feed to linear inches per minute."""

    rpm_val = _to_float(rpm, 0.0)
    if feed_type == "fz":
        z = max(int(teeth_z or 1), 1)
        return (_to_float(feed_val, 0.0)) * z * rpm_val
    if feed_type in {"ipr", "fpr", "pitch"}:
        return _to_float(feed_val, 0.0) * rpm_val
    if feed_type == "linear":
        return _to_float(linear_cut_rate_ipm, 0.0)
    return 0.0


def passes_for_depth(
    depth_in: float | None,
    doc_axial_in: Any,
    pass_override: int | None = None,
) -> tuple[int, float]:
    """Return the number of passes and the per-pass step."""

    depth = max(_to_float(depth_in, 0.0), 0.0)
    if pass_override is not None:
        passes = max(int(pass_override), 1)
        return passes, depth / passes if passes else depth
    doc = max(_to_float(doc_axial_in, 0.0), 1e-6)
    passes = max(int(ceil(depth / doc)), 1)
    return passes, depth / passes if passes else depth


def approach_allowance_for_drill(
    diameter_in: float | None,
    point_angle_deg: float | None = None,
) -> float:
    """Approximate additional penetration required for the drill point."""

    diameter = _to_float(diameter_in, 0.0)
    if point_angle_deg is None:
        return 0.3 * diameter
    angle = max(_to_float(point_angle_deg, 118.0), 1e-6)
    return 0.5 * diameter * tan(radians(90.0 - (angle / 2.0)))

