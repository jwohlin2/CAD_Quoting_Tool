# Quick Start Integration Example

"""
Drop this file into cad_quoter/pricing/ alongside the scrapers
to quickly test the integration.
"""

from typing import Tuple, Optional
import logging

# Import both scrapers
try:
    from . import wieland_scraper
    WIELAND_AVAILABLE = True
except ImportError:
    WIELAND_AVAILABLE = False

try:
    from . import scrapmetalbuyers_scraper
    SMB_AVAILABLE = True
except ImportError:
    SMB_AVAILABLE = False


logger = logging.getLogger(__name__)


def get_material_price_multi_source(
    material_key: str,
    unit: str = "kg",
    strategy: str = "wieland_first"
) -> Tuple[float, str]:
    """
    Get material price from multiple sources with fallback strategy.
    
    Args:
        material_key: Material identifier (e.g., '6061', 'copper', 'brass')
        unit: Output unit ('kg' or 'lb')
        strategy: One of:
            - 'wieland_first': Try Wieland, fallback to SMB
            - 'smb_first': Try SMB, fallback to Wieland
            - 'average': Average of both sources
            - 'wieland_only': Only use Wieland
            - 'smb_only': Only use ScrapMetalBuyers
    
    Returns:
        (price, source_description)
    """
    
    if strategy == "wieland_only":
        if not WIELAND_AVAILABLE:
            return 8.0, "house_rate (Wieland unavailable)"
        if unit == "lb":
            price_kg, src = wieland_scraper.get_live_material_price(material_key, unit="kg")
            return wieland_scraper.usdkg_to_usdlb(price_kg), src
        return wieland_scraper.get_live_material_price(material_key, unit=unit)
    
    if strategy == "smb_only":
        if not SMB_AVAILABLE:
            return 0.50, "house_rate (SMB unavailable)"
        if unit == "kg":
            return scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_kg(material_key)
        return scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_lb(material_key)
    
    if strategy == "wieland_first":
        # Try Wieland first
        if WIELAND_AVAILABLE:
            try:
                if unit == "lb":
                    price_kg, src = wieland_scraper.get_live_material_price(material_key, unit="kg")
                    price = wieland_scraper.usdkg_to_usdlb(price_kg)
                else:
                    price, src = wieland_scraper.get_live_material_price(material_key, unit=unit)
                
                if "house_rate" not in src:
                    return price, src
            except Exception as e:
                logger.warning(f"Wieland lookup failed: {e}")
        
        # Fallback to SMB
        if SMB_AVAILABLE:
            try:
                if unit == "kg":
                    return scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_kg(material_key)
                return scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_lb(material_key)
            except Exception as e:
                logger.warning(f"SMB lookup failed: {e}")
        
        # Final fallback
        fallback = 8.0 if unit == "kg" else 3.63
        return fallback, "house_rate (all sources failed)"
    
    if strategy == "smb_first":
        # Try SMB first
        if SMB_AVAILABLE:
            try:
                if unit == "kg":
                    price, src = scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_kg(material_key)
                else:
                    price, src = scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_lb(material_key)
                
                if "house_rate" not in src:
                    return price, src
            except Exception as e:
                logger.warning(f"SMB lookup failed: {e}")
        
        # Fallback to Wieland
        if WIELAND_AVAILABLE:
            try:
                if unit == "lb":
                    price_kg, src = wieland_scraper.get_live_material_price(material_key, unit="kg")
                    price = wieland_scraper.usdkg_to_usdlb(price_kg)
                else:
                    price, src = wieland_scraper.get_live_material_price(material_key, unit=unit)
                return price, src
            except Exception as e:
                logger.warning(f"Wieland lookup failed: {e}")
        
        # Final fallback
        fallback = 8.0 if unit == "kg" else 3.63
        return fallback, "house_rate (all sources failed)"
    
    if strategy == "average":
        prices = []
        sources = []
        
        # Try Wieland
        if WIELAND_AVAILABLE:
            try:
                if unit == "lb":
                    price_kg, src = wieland_scraper.get_live_material_price(material_key, unit="kg")
                    price = wieland_scraper.usdkg_to_usdlb(price_kg)
                else:
                    price, src = wieland_scraper.get_live_material_price(material_key, unit=unit)
                
                if "house_rate" not in src:
                    prices.append(price)
                    sources.append("Wieland")
            except Exception as e:
                logger.warning(f"Wieland lookup failed: {e}")
        
        # Try SMB
        if SMB_AVAILABLE:
            try:
                if unit == "kg":
                    price, src = scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_kg(material_key)
                else:
                    price, src = scrapmetalbuyers_scraper.get_live_scrap_price_usd_per_lb(material_key)
                
                if "house_rate" not in src:
                    prices.append(price)
                    sources.append("SMB")
            except Exception as e:
                logger.warning(f"SMB lookup failed: {e}")
        
        if prices:
            avg_price = sum(prices) / len(prices)
            source_str = f"Average of {', '.join(sources)}"
            return avg_price, source_str
        
        # Fallback
        fallback = 8.0 if unit == "kg" else 3.63
        return fallback, "house_rate (all sources failed)"
    
    # Unknown strategy
    raise ValueError(f"Unknown strategy: {strategy}")


# Quick test examples
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    materials = ["copper", "aluminum", "6061", "brass", "steel"]
    strategies = ["wieland_first", "smb_first", "average"]
    
    print("=" * 80)
    print("Multi-Source Price Lookup Test")
    print("=" * 80)
    
    for material in materials:
        print(f"\n{material.upper()}:")
        for strategy in strategies:
            try:
                price, source = get_material_price_multi_source(
                    material, 
                    unit="lb", 
                    strategy=strategy
                )
                print(f"  {strategy:20s}: ${price:7.4f}/lb from {source}")
            except Exception as e:
                print(f"  {strategy:20s}: ERROR - {e}")
    
    print("\n" + "=" * 80)
