"""Geometry-centric helper utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.llm_overrides import _plate_mass_properties
from cad_quoter.material_density import (
    MATERIAL_DENSITY_G_CC_BY_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    normalize_material_key as _normalize_lookup_key,
)
from cad_quoter.utils.numeric import coerce_positive_float as _coerce_positive_float

SCRAP_DEFAULT_GUESS = 0.15
HOLE_SCRAP_MULT = 1.0  # tune 0.5–1.5 if you want holes to “count” more/less
HOLE_SCRAP_CAP = 0.25


# ---------------------------------------------------------------------------
# Geometry context helpers
# ---------------------------------------------------------------------------


def _iter_geo_contexts(geo_context: Mapping[str, Any] | None) -> Iterable[Mapping[str, Any]]:
    if isinstance(geo_context, Mapping):
        yield geo_context
        inner = geo_context.get("geo")
        if isinstance(inner, Mapping):
            yield inner


def _collection_has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        for candidate in value.values():
            if _collection_has_text(candidate):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_collection_has_text(candidate) for candidate in value)
    return False


def _geo_mentions_outsourced(geo_context: Mapping[str, Any] | None) -> bool:
    for ctx in _iter_geo_contexts(geo_context):
        if _collection_has_text(ctx.get("finishes")):
            return True
        if _collection_has_text(ctx.get("finish_flags")):
            return True
    return False


def _should_include_outsourced_pass(
    outsourced_cost: float, geo_context: Mapping[str, Any] | None
) -> bool:
    try:
        cost_val = float(outsourced_cost)
    except Exception:
        cost_val = 0.0
    if abs(cost_val) > 1e-6:
        return True
    return _geo_mentions_outsourced(geo_context)


def _ensure_geo_context_fields(
    geom: dict[str, Any],
    source: Mapping[str, Any] | None,
    *,
    cfg: Any | None = None,
) -> None:
    """Best-effort backfill of common geometry context fields."""

    del cfg  # cfg reserved for forward compatibility

    try:
        src = source if isinstance(source, Mapping) else {}
        qty = src.get("Quantity") if isinstance(src, Mapping) else None
        try:
            if "qty" not in geom and qty is not None:
                qf = float(qty)
                if math.isfinite(qf) and qf > 0:
                    geom["qty"] = int(round(qf))
        except Exception:
            pass

        L_in = _coerce_positive_float(src.get("Plate Length (in)"))
        W_in = _coerce_positive_float(src.get("Plate Width (in)"))
        T_in = _coerce_positive_float(src.get("Thickness (in)"))
        if L_in is not None and "plate_len_in" not in geom:
            geom["plate_len_in"] = L_in
        if W_in is not None and "plate_wid_in" not in geom:
            geom["plate_wid_in"] = W_in
        if (
            L_in is not None
            and "plate_len_mm" not in geom
            and math.isfinite(float(L_in))
            and float(L_in) > 0
        ):
            geom["plate_len_mm"] = float(L_in) * 25.4
        if (
            W_in is not None
            and "plate_wid_mm" not in geom
            and math.isfinite(float(W_in))
            and float(W_in) > 0
        ):
            geom["plate_wid_mm"] = float(W_in) * 25.4
        if T_in is not None and "thickness_in" not in geom:
            geom["thickness_in"] = T_in

        L_mm = _coerce_positive_float(src.get("Plate Length (mm)"))
        W_mm = _coerce_positive_float(src.get("Plate Width (mm)"))
        T_mm = _coerce_positive_float(src.get("Thickness (mm)"))
        if L_mm is not None and "plate_len_in" not in geom:
            geom["plate_len_in"] = float(L_mm) / 25.4
        if W_mm is not None and "plate_wid_in" not in geom:
            geom["plate_wid_in"] = float(W_mm) / 25.4
        if T_mm is not None and "thickness_in" not in geom:
            geom["thickness_in"] = float(T_mm) / 25.4
        if T_mm is not None and "thickness_mm" not in geom:
            geom["thickness_mm"] = T_mm
        thickness_in_val = geom.get("thickness_in")
        if (
            "thickness_mm" not in geom
            and isinstance(thickness_in_val, (int, float))
            and math.isfinite(thickness_in_val)
            and thickness_in_val > 0
        ):
            geom["thickness_mm"] = float(thickness_in_val) * 25.4

        L_mm_val = _coerce_positive_float(geom.get("plate_len_mm"))
        W_mm_val = _coerce_positive_float(geom.get("plate_wid_mm"))
        if (
            "plate_bbox_area_mm2" not in geom
            and L_mm_val is not None
            and W_mm_val is not None
        ):
            geom["plate_bbox_area_mm2"] = float(L_mm_val) * float(W_mm_val)

        if "hole_sets" not in geom or not geom.get("hole_sets"):
            normalized_sets: list[dict[str, Any]] = []
            raw_groups = geom.get("hole_groups")
            if isinstance(raw_groups, Sequence) and not isinstance(
                raw_groups, (str, bytes, bytearray)
            ):
                for entry in raw_groups:
                    if isinstance(entry, Mapping):
                        dia_mm_val = _coerce_positive_float(
                            entry.get("dia_mm") or entry.get("diameter_mm")
                        )
                        qty_val = _coerce_float_or_none(
                            entry.get("count") or entry.get("qty")
                        )
                    else:
                        dia_mm_val = _coerce_positive_float(
                            getattr(entry, "dia_mm", None)
                            or getattr(entry, "diameter_mm", None)
                        )
                        qty_val = _coerce_float_or_none(
                            getattr(entry, "count", None)
                            or getattr(entry, "qty", None)
                        )
                    if dia_mm_val and qty_val and qty_val > 0:
                        normalized_sets.append(
                            {
                                "dia_mm": float(dia_mm_val),
                                "qty": int(round(float(qty_val))),
                            }
                        )
            if not normalized_sets:
                diams_seq = geom.get("hole_diams_mm")
                thickness_in_guess = geom.get("thickness_in")
                if thickness_in_guess is None:
                    thickness_mm_guess = _coerce_positive_float(
                        geom.get("thickness_mm")
                    )
                    if thickness_mm_guess is not None:
                        thickness_in_guess = float(thickness_mm_guess) / 25.4
                drill_groups = build_drill_groups_from_geometry(
                    diams_seq, thickness_in_guess
                )
                for group in drill_groups:
                    try:
                        dia_in = _coerce_positive_float(group.get("diameter_in"))
                        qty_val = _coerce_float_or_none(group.get("qty"))
                    except Exception:
                        dia_in = None
                        qty_val = None
                    if dia_in and qty_val and qty_val > 0:
                        normalized_sets.append(
                            {
                                "dia_mm": float(dia_in) * 25.4,
                                "qty": int(round(float(qty_val))),
                            }
                        )
            if normalized_sets:
                geom["hole_sets"] = normalized_sets
    except Exception:
        pass


def _apply_drilling_meta_fallback(
    meta: Mapping[str, Any] | None,
    groups: Sequence[Mapping[str, Any]] | None,
) -> tuple[int, int]:
    """Compute deep/std hole counts and backfill helper arrays in meta container."""

    deep = 0
    std = 0
    dia_vals: list[float] = []
    depth_vals: list[float] = []

    if isinstance(groups, Sequence):
        for g in groups:
            try:
                qty = int(_coerce_float_or_none(g.get("qty")) or 0)
            except Exception:
                qty = 0
            op = str(g.get("op") or "").strip().lower()
            if op.startswith("deep"):
                deep += qty
            else:
                std += qty
            d_in = _coerce_float_or_none(g.get("diameter_in"))
            if d_in and d_in > 0:
                dia_vals.append(float(d_in))
            z_in = _coerce_float_or_none(g.get("depth_in"))
            if z_in and z_in > 0:
                depth_vals.append(float(z_in))

    try:
        if isinstance(meta, dict):
            if dia_vals and not meta.get("dia_in_vals"):
                meta["dia_in_vals"] = dia_vals
            if depth_vals and not meta.get("depth_in_vals"):
                meta["depth_in_vals"] = depth_vals
    except Exception:
        pass

    return (deep, std)


# ---------------------------------------------------------------------------
# Scrap estimation helpers
# ---------------------------------------------------------------------------


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
) -> list[dict[str, Any]]:
    """Create simple drill groups from hole diameters and plate thickness."""

    try:
        t_in = float(thickness_in) if thickness_in is not None else None
    except Exception:
        t_in = None
    if t_in is not None and (not math.isfinite(t_in) or t_in <= 0):
        t_in = None

    groups: dict[float, dict[str, Any]] = {}
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

    ordered = [groups[k] for k in sorted(groups.keys())]
    return ordered


def _coerce_scrap_fraction(val: Any, cap: float = HOLE_SCRAP_CAP) -> float:
    """Return a scrap fraction clamped within ``[0, cap]``."""

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

    diams = geo_map.get("hole_diams_mm")
    if isinstance(diams, Iterable) and not isinstance(diams, (str, bytes)):
        for d in diams:
            v = _float_or_default(d)
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
    "_apply_drilling_meta_fallback",
    "_collection_has_text",
    "_ensure_geo_context_fields",
    "_estimate_scrap_from_stock_plan",
    "_geo_mentions_outsourced",
    "_holes_removed_mass_g",
    "_holes_scrap_fraction",
    "_iter_geo_contexts",
    "_iter_hole_diams_mm",
    "_plate_bbox_mm2",
    "_should_include_outsourced_pass",
    "build_drill_groups_from_geometry",
    "normalize_scrap_pct",
    "_coerce_scrap_fraction",
]
