"""Integration helpers for Wieland material pricing."""
from __future__ import annotations

from typing import Iterable, Tuple
import math


def lookup_price(candidates: Iterable[str]) -> Tuple[float | None, str]:
    """Attempt to look up a USD/kg price using the Wieland scraper."""

    try:
        from .wieland_scraper import get_live_material_price_usd_per_kg
    except Exception:
        return None, ""

    for candidate in candidates:
        label = str(candidate or "").strip()
        if not label:
            continue
        try:
            price, source = get_live_material_price_usd_per_kg(label, fallback_usd_per_kg=-1.0)
        except Exception:
            continue
        if not isinstance(price, (int, float)):
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        if str(source or "").lower().startswith("house_rate"):
            continue
        return float(price), str(source)

    return None, ""


__all__ = ["lookup_price"]
