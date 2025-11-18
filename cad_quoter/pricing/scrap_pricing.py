# scrap_pricing.py
# -*- coding: utf-8 -*-
"""
Unified scrap metal pricing interface.

Provides a single API that can fetch scrap prices from multiple sources:
- Wieland (primary, more comprehensive)
- ScrapMetalBuyers (fallback)

Configuration via environment variable:
  SCRAP_PRICE_SOURCE = "wieland" | "scrapmetalbuyers" | "auto"
  Default: "auto" (tries Wieland first, falls back to ScrapMetalBuyers)

Public API:
  get_unified_scrap_price_per_lb(material_family, fallback) -> (price, source)
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

# Allow running as standalone script
if __name__ == "__main__":
    from pathlib import Path
    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from cad_quoter.config import logger

# Import both scrapers
from cad_quoter.pricing import wieland_scraper
from cad_quoter.pricing import scrapmetalbuyers_scraper

# Check if Selenium is available in the consolidated scraper
SELENIUM_AVAILABLE = scrapmetalbuyers_scraper.SELENIUM_AVAILABLE


# --------------------------------- config ------------------------------------

SCRAP_PRICE_SOURCE = os.getenv("SCRAP_PRICE_SOURCE", "auto").lower()

# Valid options: "wieland", "scrapmetalbuyers", "auto"
_VALID_SOURCES = {"wieland", "scrapmetalbuyers", "auto"}

if SCRAP_PRICE_SOURCE not in _VALID_SOURCES:
    logger.warning(
        f"Invalid SCRAP_PRICE_SOURCE={SCRAP_PRICE_SOURCE}. Using 'auto'. "
        f"Valid options: {_VALID_SOURCES}"
    )
    SCRAP_PRICE_SOURCE = "auto"


# ----------------------------- unified fetcher -------------------------------

def get_unified_scrap_price_per_lb(
    material_family: Optional[str],
    *,
    fallback: Optional[float] = None
) -> Tuple[Optional[float], str]:
    """
    Get scrap price per lb from configured source(s).

    Args:
        material_family: Material family name (e.g., "copper", "aluminum", "brass")
        fallback: Fallback price if not found (USD/lb)

    Returns:
        Tuple of (price_usd_per_lb, source_description)
        - price can be None if not found and no fallback provided
        - source indicates where the price came from

    Behavior based on SCRAP_PRICE_SOURCE:
        - "wieland": Only use Wieland scraper
        - "scrapmetalbuyers": Only use ScrapMetalBuyers scraper
        - "auto" (default): Material-specific source preference:
            * Aluminum: Wieland first (better AL pricing), then ScrapMetalBuyers
            * All others: ScrapMetalBuyers first (carbide, titanium), then Wieland

    Examples:
        >>> price, source = get_unified_scrap_price_per_lb("copper")
        >>> print(f"Copper: ${price:.2f}/lb from {source}")

        >>> price, source = get_unified_scrap_price_per_lb("aluminum", fallback=0.50)
    """

    if not material_family:
        if fallback is not None:
            return (fallback, "house_rate (no material specified)")
        return (None, "no material specified")

    # Special case: Ceramic has no scrap value
    if material_family == "ceramic":
        return (0.0, "worthless (ceramic has no scrap value)")

    # Wieland-only mode
    if SCRAP_PRICE_SOURCE == "wieland":
        return _get_from_wieland(material_family, fallback)

    # ScrapMetalBuyers-only mode
    if SCRAP_PRICE_SOURCE == "scrapmetalbuyers":
        return _get_from_scrapmetalbuyers(material_family, fallback)

    # Auto mode: material-specific source preference
    if SCRAP_PRICE_SOURCE == "auto":
        # For aluminum: try Wieland first (better pricing for AL)
        # For everything else: use ScrapMetalBuyers first (carbide, titanium, etc.)
        if material_family and "aluminum" in material_family.lower():
            # Aluminum: Wieland first, then ScrapMetalBuyers
            logger.debug(f"Aluminum detected - trying Wieland first")
            price, source = _get_from_wieland(material_family, fallback=None)

            if price is not None and "house_rate" not in source.lower():
                return (price, source)

            logger.debug("Wieland failed for aluminum, trying ScrapMetalBuyers")
            price, source = _get_from_scrapmetalbuyers(material_family, fallback=None)

            if price is not None and "house_rate" not in source.lower():
                return (price, source)

        else:
            # Non-aluminum: ScrapMetalBuyers first (carbide, titanium, steel, etc.)
            logger.debug(f"{material_family} - trying ScrapMetalBuyers first")
            price, source = _get_from_scrapmetalbuyers(material_family, fallback=None)

            if price is not None and "house_rate" not in source.lower():
                return (price, source)

            logger.debug("ScrapMetalBuyers failed, trying Wieland as fallback")
            price, source = _get_from_wieland(material_family, fallback=None)

            if price is not None and "house_rate" not in source.lower():
                return (price, source)

        # Both failed, use fallback
        if fallback is not None:
            return (fallback, "house_rate (not found in Wieland or ScrapMetalBuyers)")

        return (None, "not found in Wieland or ScrapMetalBuyers")

    # Should never reach here due to validation above
    if fallback is not None:
        return (fallback, f"house_rate (invalid source config: {SCRAP_PRICE_SOURCE})")
    return (None, f"invalid source config: {SCRAP_PRICE_SOURCE}")


def _get_from_wieland(
    material_family: str,
    fallback: Optional[float]
) -> Tuple[Optional[float], str]:
    """
    Get scrap price from Wieland scraper.

    Returns (price, source_string) where source indicates it came from Wieland.
    """
    try:
        price = wieland_scraper.get_scrap_price_per_lb(
            material_family,
            fallback=fallback
        )

        if price is not None:
            if fallback is not None and price == fallback:
                return (price, "house_rate (Wieland lookup failed)")
            # Price is from Wieland - but the source string is embedded in their lookup
            # We'll trust it and add "Wieland" prefix if not already present
            return (price, f"Wieland scrap price")

        return (None, "not found in Wieland")

    except Exception as e:
        logger.warning(f"Wieland scraper failed: {e}")
        return (None, f"Wieland error: {e}")


def _get_from_scrapmetalbuyers(
    material_family: str,
    fallback: Optional[float]
) -> Tuple[Optional[float], str]:
    """
    Get scrap price from ScrapMetalBuyers scraper.

    Tries urllib method first (fast), then Selenium (slower but works with JavaScript).

    Returns (price, source_string) where source indicates it came from ScrapMetalBuyers.
    """
    fallback_lb = fallback if fallback is not None else 0.50

    # Try urllib method first (fast, but may not work with JavaScript sites)
    try:
        price, source = scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_lb(
            material_family,
            fallback_usd_per_lb=fallback_lb,
            method="urllib"
        )

        # If we got a real price (not house_rate), use it
        if price is not None and "house_rate" not in source.lower():
            logger.debug(f"Got price from ScrapMetalBuyers (urllib): ${price:.4f}/lb")
            return (price, source)

        logger.debug("urllib method returned house_rate, trying Selenium...")

    except Exception as e:
        logger.debug(f"ScrapMetalBuyers urllib method failed: {e}, trying Selenium...")

    # Try Selenium method if available (slower but works with JavaScript)
    if SELENIUM_AVAILABLE:
        try:
            price, source = scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_lb(
                material_family,
                fallback_usd_per_lb=fallback_lb,
                method="selenium"
            )

            # If we got a real price, use it
            if price is not None and "house_rate" not in source.lower():
                logger.debug(f"Got price from ScrapMetalBuyers (selenium): ${price:.4f}/lb")
                return (price, source)

        except Exception as e:
            logger.warning(f"ScrapMetalBuyers selenium method failed: {e}")
    else:
        logger.debug("Selenium not available (install with: pip install selenium webdriver-manager)")

    # Both methods failed
    return (None, "not found in ScrapMetalBuyers (urllib and selenium both failed)")


# ---------------------------- compatibility alias ----------------------------

def get_scrap_price_per_lb(
    material_family: Optional[str],
    *,
    fallback: Optional[float] = None
) -> Optional[float]:
    """
    Compatibility alias for get_unified_scrap_price_per_lb.

    Returns just the price (not the source string).
    This maintains compatibility with existing code that uses wieland_scraper.get_scrap_price_per_lb.

    Args:
        material_family: Material family name
        fallback: Fallback price if not found

    Returns:
        Price in USD/lb or None
    """
    price, _source = get_unified_scrap_price_per_lb(material_family, fallback=fallback)
    return price


# ----------------------------------- info ------------------------------------

def get_configured_source() -> str:
    """Return the currently configured scrap price source."""
    return SCRAP_PRICE_SOURCE


def get_available_sources() -> list[str]:
    """Return list of available scrap price sources."""
    return ["wieland", "scrapmetalbuyers"]


# ----------------------------------- CLI -------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test unified scrap pricing configuration"
    )
    parser.add_argument(
        "--material",
        type=str,
        default="aluminum",
        help="Material to test (default: aluminum)"
    )
    parser.add_argument(
        "--fallback",
        type=float,
        default=0.50,
        help="Fallback price (default: 0.50)"
    )

    args = parser.parse_args()

    print(f"Unified Scrap Pricing Configuration")
    print("=" * 60)
    print(f"Configured source: {get_configured_source()}")
    print(f"Available sources: {', '.join(get_available_sources())}")
    print(f"Environment: SCRAP_PRICE_SOURCE={os.getenv('SCRAP_PRICE_SOURCE', 'not set (using auto)')}")
    print()

    print(f"Testing material: {args.material}")
    print(f"Fallback price: ${args.fallback:.2f}/lb")
    print()

    try:
        price, source = get_unified_scrap_price_per_lb(
            args.material,
            fallback=args.fallback
        )

        print(f"Result:")
        print(f"  Price: ${price:.4f}/lb" if price else "  Price: N/A")
        print(f"  Source: {source}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
