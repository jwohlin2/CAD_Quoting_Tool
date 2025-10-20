"""Helpers for handling scrap estimates and stock plans."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterable

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none

SCRAP_DEFAULT_GUESS = 0.15
HOLE_SCRAP_MULT = 1.0  # tune 0.5–1.5 if you want holes to “count” more/less
HOLE_SCRAP_CAP = 0.25


def _coerce_scrap_fraction(val: Any, cap: float = HOLE_SCRAP_CAP) -> float:
    """Return a scrap fraction clamped within ``[0, cap]``.

    The helper accepts common UI inputs such as ``15`` (percent),
    ``0.15`` (fraction), values with a trailing ``%`` and gracefully handles
    ``None``/empty strings by falling back to ``0``.
    """

    cap_val = _coerce_cap(cap, default=HOLE_SCRAP_CAP)

    if val is None:
        raw = 0.0
    elif isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            raw = 0.0
        elif stripped.endswith("%"):
            raw = _float_or_default(stripped.rstrip("%"), default=0.0) / 100.0
        else:
            raw = _float_or_default(stripped, default=0.0)
    else:
        raw = _float_or_default(val, default=0.0)

    if not math.isfinite(raw):
        raw = 0.0
    if raw > 1.0:
        raw = raw / 100.0
    if raw < 0.0:
        raw = 0.0
    return min(cap_val, raw)


def normalize_scrap_pct(val: Any, cap: float = HOLE_SCRAP_CAP) -> float:
    """Backwards-compatible alias for :func:`_coerce_scrap_fraction`."""

    return _coerce_scrap_fraction(val, cap)


def _iter_hole_diams_mm(geo_ctx: Mapping[str, Any] | None) -> list[float]:
    """Return all positive hole diameters from a geometry context."""

    if not isinstance(geo_ctx, Mapping):
        return []

    geo_map = geo_ctx
    derived_obj = geo_map.get("derived")
    derived: Mapping[str, Any] = derived_obj if isinstance(derived_obj, Mapping) else {}

    out: list[float] = []
    dxf_diams = derived.get("hole_diams_mm")
    if isinstance(dxf_diams, Iterable) and not isinstance(dxf_diams, (str, bytes)):
        for d in dxf_diams:
            v = _float_or_default(d)
            if v is not None and v > 0:
                out.append(v)

    step_holes = derived.get("holes")
    if isinstance(step_holes, Iterable) and not isinstance(step_holes, (str, bytes)):
        for h in step_holes:
            if isinstance(h, Mapping):
                v = _float_or_default(h.get("dia_mm"))
            else:
                v = _float_or_default(getattr(h, "dia_mm", None))
            if v is not None and v > 0:
                out.append(v)

    return out


def _plate_bbox_mm2(geo_ctx: Mapping[str, Any] | None) -> float:
    """Return the estimated bounding-box area of a plate in square millimetres."""

    if not isinstance(geo_ctx, Mapping):
        return 0.0

    geo_map = geo_ctx
    derived_obj = geo_map.get("derived")
    derived: Mapping[str, Any] = derived_obj if isinstance(derived_obj, Mapping) else {}

    bbox_mm = derived.get("bbox_mm")
    if (
        isinstance(bbox_mm, (list, tuple))
        and len(bbox_mm) == 2
        and all(isinstance(x, (int, float)) and x > 0 for x in bbox_mm)
    ):
        return float(bbox_mm[0]) * float(bbox_mm[1])

    def _coerce_in(val: Any) -> float | None:
        v = _float_or_default(val)
        if v is not None and v > 0:
            return float(v)
        return None

    try:
        L_in = float(geo_map.get("plate_length_mm")) / 25.4
    except Exception:
        L_in = _coerce_in(geo_map.get("plate_length_in"))
    try:
        W_in = float(geo_map.get("plate_width_mm")) / 25.4
    except Exception:
        W_in = _coerce_in(geo_map.get("plate_width_in"))

    if L_in and W_in:
        return float(L_in * 25.4) * float(W_in * 25.4)
    return 0.0


def _holes_scrap_fraction(
    geo_ctx: Mapping[str, Any] | None,
    *,
    cap: float = HOLE_SCRAP_CAP,
) -> float:
    """Estimate scrap fraction implied by drilled holes in *geo_ctx*."""

    diams = _iter_hole_diams_mm(geo_ctx)
    if not diams:
        return 0.0

    plate_area_mm2 = _plate_bbox_mm2(geo_ctx)
    if plate_area_mm2 <= 0:
        return 0.0

    holes_area_mm2 = 0.0
    for d in diams:
        r = 0.5 * float(d)
        holes_area_mm2 += math.pi * r * r

    frac = HOLE_SCRAP_MULT * (holes_area_mm2 / plate_area_mm2)
    if not math.isfinite(frac) or frac < 0:
        return 0.0

    cap_val = _coerce_cap(cap, default=HOLE_SCRAP_CAP)
    return min(cap_val, float(frac))


def _estimate_scrap_from_stock_plan(
    geo_ctx: Mapping[str, Any] | None,
) -> tuple[float | None, str | None]:
    """Attempt to infer a scrap fraction from stock planning hints."""

    contexts: list[Mapping[str, Any]] = []
    if isinstance(geo_ctx, Mapping):
        contexts.append(geo_ctx)
        inner = geo_ctx.get("geo")
        if isinstance(inner, Mapping):
            contexts.append(inner)

    for ctx in contexts:
        plan = ctx.get("stock_plan_guess") or ctx.get("stock_plan")
        if not isinstance(plan, Mapping):
            continue
        net_vol = _coerce_float_or_none(plan.get("net_volume_in3"))
        stock_vol = _coerce_float_or_none(plan.get("stock_volume_in3"))
        scrap: float | None = None
        if net_vol and net_vol > 0 and stock_vol and stock_vol > 0:
            scrap = max(0.0, (stock_vol - net_vol) / net_vol)
        else:
            part_mass_lb = _coerce_float_or_none(plan.get("part_mass_lb"))
            stock_mass_lb = _coerce_float_or_none(plan.get("stock_mass_lb"))
            if part_mass_lb and part_mass_lb > 0 and stock_mass_lb and stock_mass_lb > 0:
                scrap = max(0.0, (stock_mass_lb - part_mass_lb) / part_mass_lb)
        if scrap is not None:
            return min(HOLE_SCRAP_CAP, float(scrap)), "stock_plan_guess"
    return None, None


def _coerce_cap(cap: Any, *, default: float) -> float:
    """Normalize *cap* to a usable float value."""

    cap_val = _float_or_default(cap, default=default)
    if cap_val is None or not math.isfinite(cap_val):
        return default
    return max(0.0, float(cap_val))


def _float_or_default(val: Any, default: float | None = None) -> float | None:
    """Return ``float(val)`` or ``default`` when conversion fails."""

    try:
        result = float(val)  # type: ignore[arg-type]
    except Exception:
        return default
    return result


__all__ = [
    "SCRAP_DEFAULT_GUESS",
    "HOLE_SCRAP_MULT",
    "HOLE_SCRAP_CAP",
    "_coerce_scrap_fraction",
    "normalize_scrap_pct",
    "_iter_hole_diams_mm",
    "_plate_bbox_mm2",
    "_holes_scrap_fraction",
    "_estimate_scrap_from_stock_plan",
]
