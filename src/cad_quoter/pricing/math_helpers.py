"""Utility helpers for pricing and material cost calculations."""

from __future__ import annotations

from collections.abc import Mapping as _MappingABC
from typing import Any, Mapping

import logging
import math

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.domain_models.values import safe_float as _safe_float

from .materials import _material_cost_components

logger = logging.getLogger(__name__)


def _wieland_scrap_usd_per_lb(material_family: str | None) -> float | None:
    """Return the USD/lb scrap price from the Wieland scraper if available."""

    try:
        from cad_quoter.pricing.wieland_scraper import get_scrap_price_per_lb
    except Exception:  # pragma: no cover - external dependency hook
        try:  # pragma: no cover - external dependency hook
            from wieland_scraper import get_scrap_price_per_lb  # type: ignore[import]
        except Exception:
            return None

    fam = str(material_family or "").strip().lower()
    if not fam:
        fam = "aluminum"
    if "alum" in fam:
        fam = "aluminum"
    elif "stainless" in fam:
        fam = "stainless"
    elif "steel" in fam:
        fam = "steel"
    elif "copper" in fam:
        fam = "copper"
    elif "brass" in fam:
        fam = "brass"
    elif "titanium" in fam or fam.startswith("ti"):
        fam = "titanium"

    try:
        price: float | int | str | None = get_scrap_price_per_lb(fam)
    except Exception as exc:  # pragma: no cover - network/HTML failure
        logger.warning("Wieland scrap price lookup failed for %s: %s", fam, exc)
        return None

    if price is None:
        return None

    try:
        price_float = float(price)
    except Exception:
        return None

    if not math.isfinite(price_float) or price_float <= 0:
        return None
    return price_float


def _compute_direct_costs(
    material_total: float | int | str | None,
    scrap_credit: float | int | str | None,
    material_tax: float | int | str | None,
    pass_through: _MappingABC[str, Any] | None,
    *,
    material_detail: Mapping[str, Any] | None = None,
    scrap_price_source: str | None = None,
) -> float:
    """Return the rounded direct-cost total shared by math and rendering."""

    block: dict[str, Any] = {}
    if isinstance(material_detail, _MappingABC):
        block.update(material_detail)

    def _assign_if_missing(key: str, value: Any) -> None:
        if key in block:
            return
        if value in (None, ""):
            return
        coerced = _coerce_float_or_none(value)
        if coerced is None:
            return
        block[key] = float(coerced)

    _assign_if_missing("material_cost_before_credit", material_total)
    _assign_if_missing("material_cost", material_total)
    _assign_if_missing("material_direct_cost", material_total)
    _assign_if_missing("material_cost_pre_credit", material_total)
    _assign_if_missing("material_base_cost", material_total)
    _assign_if_missing("material_scrap_credit", scrap_credit)
    _assign_if_missing("scrap_credit_usd", scrap_credit)
    _assign_if_missing("material_tax", material_tax)
    _assign_if_missing("material_tax_usd", material_tax)

    if scrap_price_source:
        try:
            source_text = str(scrap_price_source).strip()
        except Exception:
            source_text = ""
        if source_text:
            block.setdefault("scrap_price_source", source_text)
            block.setdefault("scrap_credit_source", source_text)

    try:
        components = _material_cost_components(block, overrides=None, cfg=None)
    except Exception:
        components = None

    if isinstance(components, _MappingABC):
        material_total_usd = _coerce_float_or_none(components.get("total_usd")) or 0.0
    else:
        base_val = _coerce_float_or_none(material_total) or 0.0
        tax_val = _coerce_float_or_none(material_tax) or 0.0
        scrap_val = _coerce_float_or_none(scrap_credit) or 0.0
        material_total_usd = float(base_val) + float(tax_val) - float(scrap_val)

    pass_through_total = 0.0
    for key, raw_value in (pass_through or {}).items():
        try:
            if str(key).strip().lower() == "material":
                continue
        except Exception:
            pass
        amount = _coerce_float_or_none(raw_value)
        if amount is not None:
            pass_through_total += float(amount)

    total = round(float(material_total_usd) + float(pass_through_total), 2)
    if total < 0:
        return 0.0
    return total


def _compute_pricing_ladder(
    subtotal: float | int | str | None,
    *,
    expedite_pct: float | int | str | None = 0.0,
    margin_pct: float | int | str | None = 0.0,
) -> dict[str, float]:
    """Return cumulative totals for each step of the pricing ladder."""

    def _pct(value: float | int | str | None) -> float:
        return _safe_float(value, 0.0)

    subtotal_val = round(_safe_float(subtotal, 0.0), 2)

    expedite_pct_val = _pct(expedite_pct)
    margin_pct_val = _pct(margin_pct)

    expedite_cost = round(subtotal_val * expedite_pct_val, 2)

    with_expedite = round(subtotal_val + expedite_cost, 2)
    subtotal_before_margin = with_expedite
    with_margin = round(subtotal_before_margin * (1.0 + margin_pct_val), 2)

    return {
        "subtotal": subtotal_val,
        "with_expedite": with_expedite,
        "with_margin": with_margin,
        "expedite_cost": expedite_cost,
        "subtotal_before_margin": subtotal_before_margin,
    }


def roughly_equal(a: float | int | str | None, b: float | int | str | None, *, eps: float = 0.01) -> bool:
    """Return True when *a* and *b* are approximately equal within ``eps`` dollars."""

    try:
        a_val = float(a or 0.0)
    except Exception:
        return False
    try:
        b_val = float(b or 0.0)
    except Exception:
        return False
    try:
        eps_val = float(eps)
    except Exception:
        eps_val = 0.0
    return math.isclose(a_val, b_val, rel_tol=0.0, abs_tol=abs(eps_val))


__all__ = [
    "_compute_direct_costs",
    "_compute_pricing_ladder",
    "_wieland_scrap_usd_per_lb",
    "roughly_equal",
]
