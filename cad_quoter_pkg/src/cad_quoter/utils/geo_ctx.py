"""Helpers for working with geometry context dictionaries."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Mapping as _MappingABC
from typing import Any

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.utils.numeric import coerce_positive_float as _coerce_positive_float
from cad_quoter.utils.scrap import build_drill_groups_from_geometry


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
    """Best-effort backfill of common geometry context fields.

    This is intentionally tolerant and only fills when values are present
    and reasonable. It never raises.
    """

    del cfg  # cfg reserved for forward compatibility

    try:
        src = source if isinstance(source, _MappingABC) else {}
        # Quantity
        qty = src.get("Quantity") if isinstance(src, _MappingABC) else None
        try:
            if "qty" not in geom and qty is not None:
                qf = float(qty)
                if math.isfinite(qf) and qf > 0:
                    geom["qty"] = int(round(qf))
        except Exception:
            pass

        # Dimensions (inches)
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

        # Dimensions (mm) → convert if inch values missing
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
                    if isinstance(entry, _MappingABC):
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
        # Defensive: never let context backfill break pricing
        pass


def _apply_drilling_meta_fallback(
    meta: Mapping[str, Any] | None,
    groups: Sequence[Mapping[str, Any]] | None,
) -> tuple[int, int]:
    """Compute deep/std hole counts and backfill helper arrays in meta container.

    Returns (holes_deep, holes_std).
    """

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

    # Backfill into a mutable container if provided
    try:
        if isinstance(meta, dict):
            if dia_vals and not meta.get("dia_in_vals"):
                meta["dia_in_vals"] = dia_vals
            if depth_vals and not meta.get("depth_in_vals"):
                meta["depth_in_vals"] = depth_vals
    except Exception:
        pass

    return (deep, std)


__all__ = [
    "_iter_geo_contexts",
    "_collection_has_text",
    "_geo_mentions_outsourced",
    "_should_include_outsourced_pass",
    "_ensure_geo_context_fields",
    "_apply_drilling_meta_fallback",
]
