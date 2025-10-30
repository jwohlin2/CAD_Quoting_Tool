"""Utility helpers shared across the desktop quoting application."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping

from cad_quoter.utils.debug_tables import (
    _jsonify_debug_summary as _debug_jsonify_summary,
    _jsonify_debug_value as _debug_jsonify_value,
)
from cad_quoter.utils.numeric import parse_mixed_fraction as _parse_mixed_fraction
from cad_quoter.pricing.feed_math import (
    approach_allowance_for_drill as _approach_allowance_for_drill,
    ipm_from_feed as _ipm_from_feed,
    passes_for_depth as _passes_for_depth,
    rpm_from_sfm as _rpm_from_sfm_shared,
)


def _coerce_float_or_none(value: Any) -> float | None:
    from cad_quoter.domain_models.values import coerce_float_or_none as _coerce

    return _coerce(value)


def _jsonify_debug_value(value: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """Proxy to :func:`cad_quoter.utils.debug_tables._jsonify_debug_value`."""

    return _debug_jsonify_value(value, depth=depth, max_depth=max_depth)


def _jsonify_debug_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Proxy to :func:`cad_quoter.utils.debug_tables._jsonify_debug_summary`."""

    return _debug_jsonify_summary(summary)


def _first_numeric_or_none(*values: Any) -> float | None:
    """Return the first value that can be coerced to a float, or ``None``."""

    for value in values:
        numeric = _coerce_float_or_none(value)
        if numeric is not None:
            return float(numeric)
    return None


def _fmt_rng(vals, prec: int = 2, unit: str | None = None) -> str:
    """Format a sequence of numeric values as a range for display."""

    vs: list[float] = []
    for v in vals or []:
        try:
            f = float(v)
        except Exception:  # pragma: no cover - defensive conversion
            continue
        if math.isfinite(f):
            vs.append(f)
    if not vs:
        return "-"
    lo, hi = min(vs), max(vs)
    if abs(hi - lo) < 10 ** (-prec):
        span = f"{lo:.{prec}f}"
    else:
        span = f"{lo:.{prec}f}-{hi:.{prec}f}"
    return f"{span}{unit}" if unit else span


def _rpm_from_sfm(sfm: float, d_in: float) -> float:
    """Return spindle RPM from surface feet per minute and diameter in inches."""

    return _rpm_from_sfm_shared(sfm, d_in)


_NUMBER_MAJOR = {
    "#0": 0.0600,
    "#1": 0.0730,
    "#2": 0.0860,
    "#3": 0.0990,
    "#4": 0.1120,
    "#5": 0.1250,
    "#6": 0.1380,
    "#8": 0.1640,
    "#10": 0.1900,
    "#12": 0.2160,
}


def _parse_thread_major_in(thread: str) -> float | None:
    """Return major diameter in inches from ``'5/16-18'`` or ``'#10-32'`` style text."""

    s = (thread or "").strip().upper()
    match = re.match(r"^(#\d+)\s*-\s*\d+$", s)
    if match:
        return _NUMBER_MAJOR.get(match.group(1))
    match = re.match(r"^(\d+/\d+|\d+(?:\.\d+)?)\s*-\s*\d+$", s)
    if not match:
        return None
    token = match.group(1)
    if "/" in token:
        num, den = token.split("/")
        return float(num) / float(den)
    return float(token)


def _parse_tpi(thread: str) -> int | None:
    """Extract the threads-per-inch value from a thread designation."""

    match = re.search(r"-(\d+)$", (thread or "").strip())
    return int(match.group(1)) if match else None


_DEFAULT_SFM = {
    "tapping": 60.0,
    "counterbore": 150.0,
    "spot": 200.0,
}

_DEFAULT_IPR = {
    "tapping": None,
    "counterbore": 0.005,
    "spot": 0.004,
}


def _lookup_sfm_ipr(
    op: str,
    diameter_in: float | None,
    material_group: str | None,
    speeds_csv: dict | None,
) -> tuple[float, float | None]:
    """Return default SFM/IPR pairs when a lookup table is unavailable."""

    _ = diameter_in, material_group, speeds_csv  # reserved for future use
    op_key = (op or "").lower()
    return _DEFAULT_SFM.get(op_key, 100.0), _DEFAULT_IPR.get(op_key)


def _rpm_from_sfm_diam(sfm: float, dia_in: float | None) -> float | None:
    """Compute spindle RPM from surface feet per minute and tool diameter."""

    if not dia_in or dia_in <= 0:
        return None
    return _rpm_from_sfm_shared(sfm, dia_in)


def _ipm_from_rpm_ipr(rpm: float | None, ipr: float | None) -> float | None:
    """Compute feed rate (IPM) from RPM and IPR values."""

    if rpm is None or ipr is None:
        return None
    return rpm * ipr


# Backwards-compatible exports for legacy modules.
rpm_from_sfm = _rpm_from_sfm
rpm_from_sfm_diam = _rpm_from_sfm_diam
ipm_from_feed = _ipm_from_feed
passes_for_depth = _passes_for_depth
approach_allowance_for_drill = _approach_allowance_for_drill
parse_thread_major_in = _parse_thread_major_in
parse_tpi = _parse_tpi
lookup_sfm_ipr = _lookup_sfm_ipr
ipm_from_rpm_ipr = _ipm_from_rpm_ipr


def _parse_numeric_text(value: str) -> float | None:
    """Parse a mixed numeric string ("3 1/2") into a floating-point value."""

    if not isinstance(value, str):
        return _coerce_float_or_none(value)
    return _parse_mixed_fraction(value)


def parse_length_to_mm(value: Any) -> float | None:
    """Coerce a numeric or textual length into millimetres."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    unit: str | None = None
    for suffix in ("millimeters", "millimetres", "millimeter", "millimetre", "mm"):
        if lower.endswith(suffix):
            unit = "mm"
            text = text[: -len(suffix)]
            break
    if unit is None:
        for suffix in ("inches", "inch", "in", '"'):
            if lower.endswith(suffix):
                unit = "in"
                text = text[: -len(suffix)]
                break
    if unit is None and '"' in text:
        unit = "in"
        text = text.replace('"', "")
    if unit is None and "mm" in lower:
        unit = "mm"
        text = re.sub(r"mm", "", text, flags=re.IGNORECASE)
    numeric_val = _parse_numeric_text(text)
    if numeric_val is None:
        return None
    if unit == "in":
        return float(numeric_val) * 25.4
    return float(numeric_val)


_parse_length_to_mm = parse_length_to_mm


__all__ = [
    "_jsonify_debug_value",
    "_jsonify_debug_summary",
    "_first_numeric_or_none",
    "_fmt_rng",
    "_rpm_from_sfm",
    "rpm_from_sfm",
    "_ipm_from_feed",
    "ipm_from_feed",
    "_parse_thread_major_in",
    "parse_thread_major_in",
    "_parse_tpi",
    "parse_tpi",
    "_lookup_sfm_ipr",
    "lookup_sfm_ipr",
    "_rpm_from_sfm_diam",
    "rpm_from_sfm_diam",
    "_passes_for_depth",
    "passes_for_depth",
    "_approach_allowance_for_drill",
    "approach_allowance_for_drill",
    "_ipm_from_rpm_ipr",
    "ipm_from_rpm_ipr",
    "_parse_numeric_text",
    "parse_length_to_mm",
]
