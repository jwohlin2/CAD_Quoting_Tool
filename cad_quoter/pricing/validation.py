"""Pricing validation helpers."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none

__all__ = ["validate_quote_before_pricing"]


def validate_quote_before_pricing(
    geo: Mapping[str, Any] | None,
    process_costs: Mapping[str, float],
    pass_through: Mapping[str, Any],
    process_hours: Mapping[str, float] | None = None,
) -> None:
    """Perform defensive validation of quote inputs prior to pricing."""

    issues: list[str] = []
    geo_ctx: Mapping[str, Any] = geo if isinstance(geo, Mapping) else {}

    has_legacy_buckets = any(key in process_costs for key in ("drilling", "milling"))
    if has_legacy_buckets:
        hole_cost = sum(float(process_costs.get(k, 0.0)) for k in ("drilling", "milling"))
        if geo_ctx.get("hole_diams_mm") and hole_cost < 50:
            issues.append("Unusually low machining time for number of holes.")

    material_cost = float(pass_through.get("Material", 0.0) or 0.0)
    if material_cost < 5.0:
        inner_geo_candidate = geo_ctx.get("geo") if isinstance(geo_ctx, Mapping) else None
        inner_geo: Mapping[str, Any] = (
            inner_geo_candidate if isinstance(inner_geo_candidate, Mapping) else {}
        )

        def _positive(value: Any) -> bool:
            num = _coerce_float_or_none(value)
            return bool(num and num > 0)

        thickness_candidates = [
            geo_ctx.get("thickness_mm"),
            geo_ctx.get("thickness_in"),
            geo_ctx.get("GEO-03_Height_mm"),
            geo_ctx.get("GEO__Stock_Thickness_mm"),
            inner_geo.get("thickness_mm") if isinstance(inner_geo, Mapping) else None,
            inner_geo.get("stock_thickness_mm") if isinstance(inner_geo, Mapping) else None,
        ]
        has_thickness_hint = any(_positive(val) for val in thickness_candidates)

        def _mass_hint(ctx: Mapping[str, Any] | None) -> bool:
            if not isinstance(ctx, Mapping):
                return False
            for key in ("net_mass_kg", "net_mass_kg_est", "mass_kg", "net_mass_g"):
                if key not in ctx:
                    continue
                num = _coerce_float_or_none(ctx.get(key))
                if num and num > 0:
                    return True
            return False

        has_mass_hint = _mass_hint(geo_ctx) or _mass_hint(inner_geo)
        has_material = bool(
            str(geo_ctx.get("material") or "").strip()
            or str(inner_geo.get("material") or "").strip()
        )

        if not (has_thickness_hint or has_mass_hint or has_material):
            issues.append("Material cost is near zero; check material & thickness.")

    try:
        hole_count_val = int(float(geo_ctx.get("hole_count", 0)))
    except Exception:
        holes = geo_ctx.get("hole_diams_mm") if isinstance(geo_ctx, Mapping) else []
        hole_count_val = len(holes) if isinstance(holes, Sequence) else 0
    if hole_count_val <= 0:
        holes = geo_ctx.get("hole_diams_mm") if isinstance(geo_ctx, Mapping) else []
        hole_count_val = len(holes) if isinstance(holes, Sequence) else 0

    if issues:
        allow = str(os.getenv("QUOTE_ALLOW_LOW_MATERIAL", "")).strip().lower()
        if allow not in {"1", "true", "yes", "on"}:
            raise ValueError("Quote blocked:\n- " + "\n- ".join(issues))
