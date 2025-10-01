"""Integration helpers for Wieland material pricing."""
from __future__ import annotations

from typing import Iterable, Tuple
import math

from cad_quoter.config import logger


def lookup_price(candidates: Iterable[str]) -> Tuple[float | None, str]:
    """Attempt to look up a USD/kg price using the Wieland scraper."""

    try:
        from .wieland_scraper import get_live_material_price_usd_per_kg
    except Exception as exc:  # pragma: no cover - exercised via integration
        logger.exception("Unable to import Wieland scraper: %s", exc)
        return None, ""

    for candidate in candidates:
        label = str(candidate or "").strip()
        if not label:
            continue
        try:
            price, source = get_live_material_price_usd_per_kg(label, fallback_usd_per_kg=-1.0)
        except Exception as exc:
            logger.exception("Wieland lookup failed for %s: %s", label, exc)
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
