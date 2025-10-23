"""Helpers for handling scrap estimates and stock plans."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.material_density import (
    MATERIAL_DENSITY_G_CC_BY_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    normalize_material_key as _normalize_lookup_key,
)
from cad_quoter.utils.numeric import coerce_positive_float as _coerce_positive_float
from cad_quoter.llm_overrides import _plate_mass_properties

SCRAP_DEFAULT_GUESS = 0.15
HOLE_SCRAP_MULT = 1.0  # tune 0.5–1.5 if you want holes to “count” more/less
HOLE_SCRAP_CAP = 0.25
def _holes_removed_mass_g(geo: Mapping[str, Any] | None) -> float | None:
    """Estimate mass removed by holes using geometry context."""

    if not isinstance(geo, Mapping):
        return None

    t_in = _coerce_float_or_none(geo.get("thickness_in"))
    if t_in is None:
        t_mm = _coerce_positive_float(geo.get("thickness_mm"))
        if t_mm is not None:
            t_in = float(t_mm) / 25.4
    if t_in is None or t_in <= 0:
        return None

    hole_diams_mm: list[float] = []
    raw_list = geo.get("hole_diams_mm")
    if isinstance(raw_list, Sequence) and not isinstance(raw_list, (str, bytes, bytearray)):
        for v in raw_list:
            val = _coerce_positive_float(v)
            if val is not None:
                hole_diams_mm.append(val)
    if not hole_diams_mm:
        families = geo.get("hole_diam_families_in")
        if isinstance(families, Mapping):
            for dia_in, qty in families.items():
                d_in = _coerce_positive_float(dia_in)
                q = _coerce_float_or_none(qty)
                if d_in and q and q > 0:
                    hole_diams_mm.extend([d_in * 25.4] * int(round(q)))
    if not hole_diams_mm:
        return None

    density_g_cc = _coerce_float_or_none(geo.get("density_g_cc"))
    if density_g_cc in (None, 0.0):
        material_text = geo.get("material") or geo.get("material_name")
        if isinstance(material_text, str) and material_text.strip():
            norm_key = _normalize_lookup_key(material_text)
            collapsed = norm_key.replace(" ", "")
            density_g_cc = (
                MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(norm_key)
                or MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(collapsed)
                or MATERIAL_DENSITY_G_CC_BY_KEY.get(norm_key)
            )
            if not density_g_cc:
                for token, density in MATERIAL_DENSITY_G_CC_BY_KEYWORD.items():
                    if token and (token in norm_key or token in collapsed):
                        density_g_cc = density
                        break
    if density_g_cc is None or density_g_cc <= 0:
        return None

    plate_len_in = _coerce_float_or_none(geo.get("plate_len_in"))
    plate_wid_in = _coerce_float_or_none(geo.get("plate_wid_in"))
    if plate_len_in is None:
        plate_len_mm = _coerce_positive_float(geo.get("plate_len_mm"))
        if plate_len_mm is not None:
            plate_len_in = float(plate_len_mm) / 25.4
    if plate_wid_in is None:
        plate_wid_mm = _coerce_positive_float(geo.get("plate_wid_mm"))
        if plate_wid_mm is not None:
            plate_wid_in = float(plate_wid_mm) / 25.4

    length_in = plate_len_in if plate_len_in and plate_len_in > 0 else 1.0
    width_in = plate_wid_in if plate_wid_in and plate_wid_in > 0 else 1.0

    _, removed_mass_g = _plate_mass_properties(
        length_in,
        width_in,
        t_in,
        density_g_cc,
        hole_diams_mm,
    )

    if removed_mass_g is None or removed_mass_g <= 0:
        return None

    return removed_mass_g


def build_drill_groups_from_geometry(
    hole_diams_mm: Sequence[Any] | None,
    thickness_in: Any | None,
    ops_claims: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create simple drill groups from hole diameters and plate thickness."""

    try:
        t_in = float(thickness_in) if thickness_in is not None else None
    except Exception:
        t_in = None
    if t_in is not None and (not math.isfinite(t_in) or t_in <= 0):
        t_in = None

    groups: dict[float, dict[str, Any]] = {}
    counts_by_diam: Counter[float] = Counter()
    if isinstance(hole_diams_mm, Sequence) and not isinstance(hole_diams_mm, (str, bytes, bytearray)):
        for raw in hole_diams_mm:
            d_mm = _coerce_positive_float(raw)
            if d_mm is None:
                continue
            d_in = float(d_mm) / 25.4
            if not (d_in > 0 and math.isfinite(d_in)):
                continue
            key = round(d_in, 4)
            bucket = groups.setdefault(
                key,
                {
                    "diameter_in": float(key),
                    "qty": 0,
                    "depth_in": t_in,
                    "op": "deep_drill" if (t_in is not None and t_in >= 3.0 * float(key)) else "drill",
                },
            )
            bucket["qty"] += 1
            counts_by_diam[key] += 1

    if counts_by_diam and isinstance(ops_claims, Mapping):

        def _nearest_bin(val: float, bins: Sequence[float]) -> float | None:
            return min(bins, key=lambda b: abs(b - val)) if bins else None

        bins = sorted(float(d) for d in counts_by_diam.keys())

        claimed = (ops_claims or {}).get("claimed_pilot_diams")
        if claimed:
            claimed_ctr = Counter(
                round(float(d), 4) for d in claimed if d is not None
            )
            for val, qty in claimed_ctr.items():
                if not bins:
                    break
                target = _nearest_bin(val, bins)
                if target is None:
                    continue
                if abs(target - val) <= 0.015:
                    counts_by_diam[target] = max(
                        0,
                        int(counts_by_diam.get(target, 0)) - int(qty),
                    )

        cb_face_counts: Counter[float] = Counter()
        for (diam, _side, _depth), qty in (ops_claims or {}).get("cb_groups", {}).items():
            if diam is None:
                continue
            try:
                cb_face_counts[round(float(diam), 4)] += int(qty)
            except Exception:
                continue

        for face_diam, face_qty in cb_face_counts.items():
            if not bins:
                break
            target = _nearest_bin(face_diam, bins)
            if target is None:
                continue
            if abs(target - face_diam) <= 0.02:
                counts_by_diam[target] = max(
                    0,
                    int(counts_by_diam.get(target, 0)) - int(face_qty),
                )

        for key in list(groups.keys()):
            qty_adj = int(counts_by_diam.get(key, 0))
            if qty_adj <= 0:
                groups.pop(key, None)
            else:
                groups[key]["qty"] = qty_adj

    ordered = [groups[k] for k in sorted(groups.keys())]
    return ordered


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
    "_holes_removed_mass_g",
    "build_drill_groups_from_geometry",
    "_coerce_scrap_fraction",
    "normalize_scrap_pct",
    "_iter_hole_diams_mm",
    "_plate_bbox_mm2",
    "_holes_scrap_fraction",
    "_estimate_scrap_from_stock_plan",
]
