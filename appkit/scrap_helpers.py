from __future__ import annotations

from typing import Mapping, Any, cast
import math

SCRAP_DEFAULT_GUESS = 0.15
HOLE_SCRAP_MULT = 1.0  # tune 0.5–1.5 if you want holes to “count” more/less

def _iter_hole_diams_mm(geo_ctx: Mapping[str, Any] | None) -> list[float]:
    if not isinstance(geo_ctx, Mapping):
        return []
    geo_map = cast(Mapping[str, Any], geo_ctx)
    derived_obj = geo_map.get("derived")
    derived: Mapping[str, Any] = derived_obj if isinstance(derived_obj, Mapping) else {}
    out: list[float] = []
    dxf_diams = derived.get("hole_diams_mm")
    if isinstance(dxf_diams, (list, tuple)):
        for d in dxf_diams:
            try:
                v = float(d)
            except Exception:
                continue
            if v > 0:
                out.append(v)
    step_holes = derived.get("holes")
    if isinstance(step_holes, (list, tuple)):
        for h in step_holes:
            try:
                v = float(h.get("dia_mm"))  # type: ignore[attr-defined]
            except Exception:
                continue
            if v > 0:
                out.append(v)
    return out

def _plate_bbox_mm2(geo_ctx: Mapping[str, Any] | None) -> float:
    if not isinstance(geo_ctx, Mapping):
        return 0.0
    geo_map = cast(Mapping[str, Any], geo_ctx)
    derived_obj = geo_map.get("derived")
    derived: Mapping[str, Any] = derived_obj if isinstance(derived_obj, Mapping) else {}
    bbox_mm = derived.get("bbox_mm")
    if (
        isinstance(bbox_mm, (list, tuple))
        and len(bbox_mm) == 2
        and all(isinstance(x, (int, float)) and x > 0 for x in bbox_mm)
    ):
        return float(bbox_mm[0]) * float(bbox_mm[1])
    # Fallback to UI dims
    def _coerce_in(val: Any) -> float | None:
        try:
            v = float(val)
            return v if v > 0 else None
        except Exception:
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

def _holes_scrap_fraction(geo_ctx: Mapping[str, Any] | None, *, cap: float = 0.25) -> float:
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
    return min(cap, float(frac))

def normalize_scrap_pct(val: Any, cap: float = 0.25) -> float:
    try:
        cap_val = float(cap)
    except Exception:
        cap_val = 0.25
    if not math.isfinite(cap_val):
        cap_val = 0.25
    if val in (None, ""):
        return 0.0
    try:
        f = float(val)
    except Exception:
        return 0.0
    if f > 1.0 + 1e-9:
        f = f / 100.0
    if f < 0:
        f = 0.0
    if not math.isfinite(f):
        return 0.0
    return max(0.0, min(cap_val, f))

__all__ = [
    "SCRAP_DEFAULT_GGUESS" if False else "SCRAP_DEFAULT_GUESS",
    "HOLE_SCRAP_MULT",
    "_iter_hole_diams_mm",
    "_plate_bbox_mm2",
    "_holes_scrap_fraction",
    "normalize_scrap_pct",
]

