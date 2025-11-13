# scrapmetalbuyers_scraper_selenium.py
# -*- coding: utf-8 -*-
"""
Enhanced scraper with Selenium support for JavaScript-rendered content.
Falls back to urllib if Selenium is not available.

Install: pip install selenium webdriver-manager
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import tempfile
from typing import Any, Dict, Optional, List, Tuple

# Import base scraper components
from scrapmetalbuyers_scraper import (
    _to_float,
    usdlb_to_usdkg,
    usdkg_to_usdlb,
    _cache_path,
    _read_temp_cache,
    _write_temp_cache,
    _normalize_material_name,
    _find_price_for_material,
    SCRAPMETALBUYERS_URL,
    CACHE_TTL_S,
    _MEM_CACHE,
    MATERIAL_KEYWORDS,
)

# Try to import Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Try to import webdriver-manager for automatic driver management
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

# Try BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


logger = logging.getLogger(__name__)


def _fetch_html_selenium(headless: bool = True, wait_time: int = 10) -> str:
    """Fetch HTML using Selenium with Chrome."""
    if not SELENIUM_AVAILABLE:
        raise RuntimeError("Selenium is not installed. Install with: pip install selenium webdriver-manager")
    
    options = Options()
    if headless:
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
    
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = None
    try:
        # Try to use webdriver-manager for automatic driver management
        if WEBDRIVER_MANAGER_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        else:
            # Fallback to system chromedriver
            driver = webdriver.Chrome(options=options)
        
        logger.info(f"Fetching {SCRAPMETALBUYERS_URL} with Selenium...")
        driver.get(SCRAPMETALBUYERS_URL)
        
        # Wait for the price table to load
        try:
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            logger.info("Price table loaded successfully")
        except TimeoutException:
            logger.warning(f"Timeout waiting for table after {wait_time}s, proceeding anyway")
        
        # Give a bit more time for any dynamic content
        time.sleep(2)
        
        html = driver.page_source
        return html
        
    except WebDriverException as e:
        logger.error(f"Selenium WebDriver error: {e}")
        raise
    finally:
        if driver:
            driver.quit()


def _parse_prices_selenium(html: str) -> Dict[str, float]:
    """Parse prices from Selenium-fetched HTML."""
    prices: Dict[str, float] = {}
    
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup not available, using basic parsing")
        return _parse_prices_fallback(html)
    
    soup = BeautifulSoup(html, 'lxml')
    
    # Find all tables
    tables = soup.find_all('table')
    logger.info(f"Found {len(tables)} table(s)")
    
    for table_idx, table in enumerate(tables):
        rows = table.find_all('tr')
        logger.info(f"Table {table_idx}: {len(rows)} rows")
        
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            
            if len(cells) >= 2:
                material = cells[0].get_text(strip=True)
                price_text = cells[1].get_text(strip=True)
                
                # Skip header rows
                if 'material' in material.lower() or 'price' in price_text.lower():
                    continue
                
                # Extract price
                price_val = _to_float(price_text)
                
                if math.isfinite(price_val) and material and len(material) > 2:
                    logger.debug(f"Found: {material} = ${price_val}")
                    prices[material] = price_val
    
    return prices


def _parse_prices_fallback(html: str) -> Dict[str, float]:
    """Fallback parser without BeautifulSoup."""
    prices: Dict[str, float] = {}
    
    # Extract table content
    table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    tables = table_pattern.findall(html)
    
    for table_html in tables:
        # Extract rows
        row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        rows = row_pattern.findall(table_html)
        
        for row_html in rows:
            # Extract cells
            cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
            cells = cell_pattern.findall(row_html)
            
            if len(cells) >= 2:
                # Clean HTML tags from cells
                material = re.sub(r'<[^>]+>', '', cells[0]).strip()
                price_text = re.sub(r'<[^>]+>', '', cells[1]).strip()
                
                # Skip headers
                if 'material' in material.lower() or 'price' in price_text.lower():
                    continue
                
                price_val = _to_float(price_text)
                if math.isfinite(price_val) and material and len(material) > 2:
                    prices[material] = price_val
    
    return prices


def scrape_scrapmetalbuyers_prices_selenium(
    force: bool = False, 
    debug: bool = False,
    headless: bool = True
) -> Dict[str, Any]:
    """
    Scrape using Selenium for JavaScript-rendered content.
    Returns same structure as base scraper.
    """
    # Check cache first
    mc = _MEM_CACHE.get("data_selenium")
    now = time.time()
    if mc and now - mc[0] < CACHE_TTL_S and not force:
        logger.info("Using in-memory cache")
        return mc[1]
    
    if not force:
        tc = _read_temp_cache()
        if tc:
            logger.info("Using temp file cache")
            _MEM_CACHE["data_selenium"] = (now, tc)
            return tc
    
    # Fetch with Selenium
    try:
        html = _fetch_html_selenium(headless=headless)
    except Exception as e:
        logger.error(f"Failed to fetch with Selenium: {e}")
        raise
    
    if debug:
        snap = os.path.join(tempfile.gettempdir(), "scrapmetalbuyers_selenium_snapshot.html")
        with open(snap, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Saved HTML snapshot to {snap}")
    
    # Parse prices
    prices_lb = _parse_prices_selenium(html)
    
    if not prices_lb:
        logger.warning("No prices found! Check HTML snapshot if debug=True")
    
    # Extract date
    asof = None
    date_patterns = [
        re.compile(r"current as of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I),
        re.compile(r"updated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I),
        re.compile(r"as of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I),
    ]
    for pattern in date_patterns:
        match = pattern.search(html)
        if match:
            asof = match.group(1)
            break
    
    data: Dict[str, Any] = {
        "source": SCRAPMETALBUYERS_URL,
        "asof": asof,
        "prices_usd_per_lb": prices_lb,
        "prices_usd_per_kg": {k: round(usdlb_to_usdkg(v), 4) for k, v in prices_lb.items()},
    }
    
    # Cache results
    _MEM_CACHE["data_selenium"] = (now, data)
    _write_temp_cache(data)
    
    return data


def get_live_scrap_price_usd_per_lb_selenium(
    material_key: str,
    fallback_usd_per_lb: float = 0.50,
    use_cache: bool = True
) -> Tuple[float, str]:
    """Get price using Selenium scraper."""
    data = scrape_scrapmetalbuyers_prices_selenium(force=not use_cache)
    prices_lb = data.get("prices_usd_per_lb", {})
    asof = data.get("asof", "today")
    
    result = _find_price_for_material(prices_lb, material_key)
    if result:
        price, exact_name = result
        return (price, f"ScrapMetalBuyers {exact_name} ({asof})")
    
    return (float(fallback_usd_per_lb), "house_rate")


def _main(argv: List[str]) -> int:
    import argparse
    
    ap = argparse.ArgumentParser(description="ScrapMetalBuyers Selenium scraper")
    ap.add_argument("--json", action="store_true", help="Print full JSON")
    ap.add_argument("--force", action="store_true", help="Bypass cache")
    ap.add_argument("--debug", action="store_true", help="Debug mode with HTML snapshot")
    ap.add_argument("--no-headless", action="store_true", help="Show browser window")
    ap.add_argument("--material", type=str, help="Look up specific material")
    ap.add_argument("--unit", choices=["lb", "kg", "both"], default="lb")
    args = ap.parse_args(argv)
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    if not SELENIUM_AVAILABLE:
        logger.error("Selenium not installed. Install with: pip install selenium webdriver-manager")
        return 1
    
    try:
        data = scrape_scrapmetalbuyers_prices_selenium(
            force=args.force,
            debug=args.debug,
            headless=not args.no_headless
        )
    except Exception as e:
        logger.error(f"Failed to scrape: {e}", exc_info=True)
        return 2
    
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    
    if args.material:
        result = _find_price_for_material(data.get("prices_usd_per_lb", {}), args.material)
        if result:
            price_lb, name = result
            if args.unit == "both":
                price_kg = usdlb_to_usdkg(price_lb)
                logger.info(f"{args.material} ({name}): ${price_lb:.4f}/lb | ${price_kg:.4f}/kg")
            elif args.unit == "kg":
                price_kg = usdlb_to_usdkg(price_lb)
                logger.info(f"{args.material} ({name}): ${price_kg:.4f}/kg")
            else:
                logger.info(f"{args.material} ({name}): ${price_lb:.4f}/lb")
        else:
            logger.warning(f"Material '{args.material}' not found")
        return 0
    
    logger.info(f"Found {len(data['prices_usd_per_lb'])} materials")
    logger.info(f"As of: {data.get('asof', 'unknown')}")
    for material, price in sorted(data['prices_usd_per_lb'].items()):
        logger.info(f"  {material}: ${price:.4f}/lb")
    
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
