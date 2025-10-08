"""Heuristic time estimation helpers for the planner pricing module."""

from __future__ import annotations

from math import ceil
from typing import Any, Iterable, Mapping, Union


Number = Union[float, int]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def minutes_wedm(data: Mapping[str, Any]) -> float:
    """Estimate minutes for a WEDM burn."""

    length_in = _to_float(data.get("length_in"))
    if length_in <= 0.0:
        length_in = _to_float(data.get("length_mm")) / 25.4
    thickness_in = _to_float(data.get("thickness_in"))
    if thickness_in <= 0.0:
        thickness_in = _to_float(data.get("thickness_mm")) / 25.4
    passes = max(_to_int(data.get("passes", 1)), 1)

    setup_min = max(_to_float(data.get("setup_min", 12.0)), 0.0)
    cut_rate_in2_per_hr = max(_to_float(data.get("cut_rate_in2_per_hr", 32.0)), 1e-6)
    thread_min = max(_to_float(data.get("thread_min", 3.0)), 0.0)

    burn_min = (length_in * thickness_in * passes) / cut_rate_in2_per_hr * 60.0
    return setup_min + thread_min + burn_min


def minutes_surface_grind(data: Mapping[str, Any]) -> float:
    """Estimate surface grind minutes based on area and stock removal."""

    area_sq_in = _to_float(data.get("area_sq_in"), 0.0)
    if area_sq_in <= 0.0:
        area_sq_in = _to_float(data.get("area_sq_mm", 0.0)) / 645.16
    area_sq_in = max(area_sq_in, 1.0)

    stock_in = max(_to_float(data.get("stock_total_in", data.get("stock_in")), 0.0), 0.0)
    setup_min = max(_to_float(data.get("setup_min", 8.0)), 0.0)
    feed_ipm = max(_to_float(data.get("feed_ipm", 40.0)), 1e-3)
    doc = max(_to_float(data.get("doc_in", 0.005)), 1e-4)
    passes = max(_to_int(data.get("passes"), ceil(stock_in / doc) if stock_in > 0 else 1), 1)

    wheel_width = max(_to_float(data.get("wheel_width_in", 1.5)), 1e-3)
    path_in = (area_sq_in / wheel_width) * passes
    grind_min = path_in / feed_ipm * 60.0
    spark_out = max(_to_float(data.get("spark_out_min", 2.0)), 0.0)
    return setup_min + spark_out + grind_min


def minutes_blanchard(data: Mapping[str, Any]) -> float:
    diameter_in = _to_float(data.get("diameter_in"), 0.0)
    if diameter_in <= 0.0:
        diameter_in = _to_float(data.get("diameter_mm", 0.0)) / 25.4
    stock_in = max(_to_float(data.get("stock_total_in", data.get("stock_in")), 0.0), 0.0)
    setup_min = max(_to_float(data.get("setup_min", 10.0)), 0.0)
    removal_rate = max(_to_float(data.get("stock_removal_in_per_hr", 0.08)), 1e-6)

    area_factor = max(diameter_in, 6.0) / 6.0
    grind_min = (stock_in * area_factor) / removal_rate * 60.0
    return setup_min + grind_min


def minutes_mill(data: Mapping[str, Any]) -> float:
    volume_in3 = _to_float(data.get("volume_in3"), 0.0)
    if volume_in3 <= 0.0:
        volume_in3 = _to_float(data.get("volume_mm3", 0.0)) / 16387.064
    volume_in3 = max(volume_in3, 0.0)

    setup_min = max(_to_float(data.get("setup_min", 15.0)), 0.0)
    removal_rate = max(_to_float(data.get("removal_rate_in3_per_hr", 18.0)), 1e-6)
    tool_changes = max(_to_int(data.get("tool_changes", 1)), 1)
    tool_change_min = max(_to_float(data.get("tool_change_min", 2.5)), 0.0)

    cut_min = volume_in3 / removal_rate * 60.0
    return setup_min + cut_min + (tool_changes * tool_change_min)


def minutes_drill(features: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> float:
    if isinstance(features, Mapping):
        iterable: Iterable[Mapping[str, Any]] = [features]
    else:
        iterable = features

    total = 0.0
    for feature in iterable:
        qty = max(_to_int(feature.get("qty", feature.get("count", 1)), 1), 1)
        depth_in = _to_float(feature.get("depth_in"), 0.0)
        if depth_in <= 0.0:
            depth_in = _to_float(feature.get("depth_mm", 0.0)) / 25.4
        diameter_in = _to_float(feature.get("diameter_in"), 0.0)
        if diameter_in <= 0.0:
            diameter_in = _to_float(feature.get("diameter_mm", 0.0)) / 25.4
        diameter_in = max(diameter_in, 1e-3)

        feed_ipm = max(_to_float(feature.get("feed_ipm", 12.0)), 1e-3)
        retract_min = max(_to_float(feature.get("retract_min", 0.1)), 0.0)
        default_pecks = ceil(depth_in / max(diameter_in * 4.0, 0.1)) if depth_in > 0 else 1
        pecks = max(_to_int(feature.get("pecks", default_pecks)), 1)

        inches = depth_in * pecks
        total += qty * ((inches / feed_ipm) * 60.0 + retract_min)
    return total


def minutes_tap(count: Number) -> float:
    qty = max(int(_to_float(count, 0.0)), 0)
    return qty * 0.6


def minutes_bores(features: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> float:
    if isinstance(features, Mapping):
        iterable: Iterable[Mapping[str, Any]] = [features]
    else:
        iterable = features

    total = 0.0
    for feature in iterable:
        qty = max(_to_int(feature.get("qty", feature.get("count", 1)), 1), 1)
        diameter_in = _to_float(feature.get("diameter_in"), 0.0)
        if diameter_in <= 0.0:
            diameter_in = _to_float(feature.get("diameter_mm", 0.0)) / 25.4
        diameter_in = max(diameter_in, 1e-3)

        tolerance = max(_to_float(feature.get("tolerance_in", 0.0005)), 1e-5)
        setup_min = max(_to_float(feature.get("setup_min", 6.0)), 0.0)
        removal_rate = max(_to_float(feature.get("removal_rate_in_per_hr", 0.0025)), 1e-6)
        stock_in = max(_to_float(feature.get("stock_in", 0.002)), 0.0)

        grind_min = stock_in / removal_rate * 60.0
        finish_pass = 3.0 if tolerance <= 0.0002 else 1.5
        total += setup_min + qty * (grind_min + finish_pass)
    return total


def minutes_sinker(features: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> float:
    if isinstance(features, Mapping):
        iterable: Iterable[Mapping[str, Any]] = [features]
    else:
        iterable = features

    total = 0.0
    for feature in iterable:
        area_sq_in = max(_to_float(feature.get("area_sq_in", 0.0)), 0.0)
        depth_in = max(_to_float(feature.get("depth_in", 0.0)), 0.0)
        finish_factor = max(_to_float(feature.get("finish_factor", 1.0)), 0.1)
        setup_min = max(_to_float(feature.get("setup_min", 20.0)), 0.0)
        removal_rate = max(_to_float(feature.get("removal_rate_in3_per_hr", 2.5)), 1e-6)

        burn_min = (area_sq_in * depth_in * finish_factor) / removal_rate * 60.0
        orbit_min = max(_to_float(feature.get("orbit_min", 5.0)), 0.0)
        total += setup_min + orbit_min + burn_min
    return total


def minutes_edgebreak(length_ft: Number) -> float:
    length = max(_to_float(length_ft, 0.0), 0.0)
    if length <= 0.0:
        length = _to_float(length_ft, 0.0)
    feed_ft_per_hr = 20.0
    base = (length / max(feed_ft_per_hr, 1e-6)) * 60.0
    return base + (5.0 if length > 0 else 0.0)


def minutes_lap(area_sq_in: Number) -> float:
    area = max(_to_float(area_sq_in, 0.0), 0.0)
    prep_min = 6.0 if area > 0 else 0.0
    lap_rate = 50.0
    return prep_min + (area / max(lap_rate, 1e-6)) * 60.0

