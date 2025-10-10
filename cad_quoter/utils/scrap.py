"""Helpers for handling scrap estimates and stock plans."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none


def _coerce_scrap_fraction(val: Any, cap: float = 0.25) -> float:
    """Return a scrap fraction clamped within ``[0, cap]``.

    The helper accepts common UI inputs such as ``15`` (percent) or ``0.15``
    (fraction) and gracefully handles ``None`` or empty strings by falling
    back to ``0``.
    """

    try:
        cap_val = float(cap)
    except Exception:
        cap_val = 0.25
    if not math.isfinite(cap_val):
        cap_val = 0.25
    cap_val = max(0.0, cap_val)

    if val is None:
        raw = 0.0
    elif isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            raw = 0.0
        elif stripped.endswith("%"):
            try:
                raw = float(stripped.rstrip("%")) / 100.0
            except Exception:
                raw = 0.0
        else:
            try:
                raw = float(stripped)
            except Exception:
                raw = 0.0
    else:
        try:
            raw = float(val)
        except Exception:
            raw = 0.0

    if not math.isfinite(raw):
        raw = 0.0
    if raw > 1.0:
        raw = raw / 100.0
    if raw < 0.0:
        raw = 0.0
    return min(cap_val, raw)


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
            return min(0.25, float(scrap)), "stock_plan_guess"
    return None, None


__all__ = [
    "_coerce_scrap_fraction",
    "_estimate_scrap_from_stock_plan",
]
