# scrapmetalbuyers_scraper.py
# -*- coding: utf-8 -*-
"""
Scrape & normalize metal prices from Scrap Metal Buyers:
  https://scrapmetalbuyers.com/current-prices/

Outputs:
- Scrap metal prices normalized to USD/lb and USD/kg

Public API:
  scrape_scrapmetalbuyers_prices(force=False, debug=False) -> dict
  get_live_scrap_price_usd_per_lb(material_key: str, fallback_usd_per_lb=0.50) -> (float, source)

CLI:
  python scrapmetalbuyers_scraper.py --json
  python scrapmetalbuyers_scraper.py --material copper
  python scrapmetalbuyers_scraper.py --force --debug
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
import urllib.request
from dataclasses import dataclass
from html import unescape as html_unescape
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional, List

import ssl

# --------------------------------- config ------------------------------------

SCRAPMETALBUYERS_URL = "https://scrapmetalbuyers.com/current-prices/"

CACHE_TTL_S = int(os.getenv("SMB_CACHE_TTL_S", 60 * 30))         # 30 minutes
REQUEST_TIMEOUT_S = int(os.getenv("SMB_REQ_TIMEOUT_S", 30))      # seconds
USER_AGENT = os.getenv(
    "SMB_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# --------------------------------- globals -----------------------------------

_NUM_RE = re.compile(r"[+-]?(?:\d[\d\s\u202f\u00a0.,']*)")
_MEM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # key -> (ts, data)

# Conversion constants
LB_PER_KG = 2.20462262185


@dataclass
class ScrapeResult:
    source: str
    asof: Optional[str]
    prices_usd_per_lb: Dict[str, float]
    prices_usd_per_kg: Dict[str, float]


# --------------------------------- utils -------------------------------------


def _to_float(s: str) -> float:
    """Parse first numeric token from a string (handles localized formats)."""

    if s is None:
        return math.nan

    token_match = _NUM_RE.search(str(s).strip())
    if not token_match:
        return math.nan

    token = token_match.group(0)

    # Normalize common thousands separators (space, thin space, apostrophe)
    token = (
        token.replace("\u202f", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .replace("'", "")
    )

    token = token.replace("−", "-")  # minus sign

    # Determine decimal separator heuristically
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            decimal_sep, thousands_sep = ",", "."
        else:
            decimal_sep, thousands_sep = ".", ","
    elif token.count(",") >= 1:
        # If comma present and looks like thousands separator (e.g., 1,234)
        last = token.rfind(",")
        decimals = len(token) - last - 1
        if decimals in (3, 0):
            decimal_sep, thousands_sep = None, ","
        else:
            decimal_sep, thousands_sep = ",", None
    elif token.count(".") >= 2:
        decimal_sep, thousands_sep = ".", None
    else:
        decimal_sep, thousands_sep = ".", None if "." in token else None

    if thousands_sep:
        token = token.replace(thousands_sep, "")
    if decimal_sep and decimal_sep != ".":
        token = token.replace(decimal_sep, ".")

    try:
        return float(token)
    except Exception:
        return math.nan


def usdlb_to_usdkg(usd_per_lb: float) -> float:
    """Convert USD/lb to USD/kg."""
    return usd_per_lb * LB_PER_KG


def usdkg_to_usdlb(usd_per_kg: float) -> float:
    """Convert USD/kg to USD/lb."""
    return usd_per_kg / LB_PER_KG


def _cache_path() -> str:
    return os.path.join(tempfile.gettempdir(), "scrapmetalbuyers_cache.json")


def _read_temp_cache() -> Optional[Dict[str, Any]]:
    p = _cache_path()
    try:
        if not os.path.isfile(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if time.time() - float(payload.get("_ts", 0)) > CACHE_TTL_S:
            return None
        return payload.get("data")
    except Exception:
        return None


def _write_temp_cache(data: Dict[str, Any]) -> None:
    p = _cache_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"_ts": time.time(), "data": data}, f)
    except Exception:
        pass


# --------------------------------- fetching ----------------------------------

try:  # pragma: no cover - optional dependency in production
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - fallback without bs4
    BeautifulSoup = None  # type: ignore


class SoupDocument:
    """Small wrapper that mimics a subset of BeautifulSoup we rely on."""

    __slots__ = ("html", "_soup")

    def __init__(self, html: str) -> None:
        self.html = html
        self._soup = BeautifulSoup(html, "lxml") if BeautifulSoup else None

    def get_text(self, separator: str = "", strip: bool = False) -> str:
        if self._soup is not None:
            try:
                return self._soup.get_text(separator, strip=strip)
            except AttributeError:
                pass
        text = re.sub(r"<[^>]+>", " ", self.html)
        if strip:
            text = " ".join(part for part in text.split() if part)
        return text if separator == "" else separator.join(text.split())

    def find_all(self, *args, **kwargs):
        if self._soup is not None:
            try:
                return self._soup.find_all(*args, **kwargs)
            except AttributeError:
                pass
        return []

    def find(self, *args, **kwargs):
        if self._soup is not None:
            try:
                return self._soup.find(*args, **kwargs)
            except AttributeError:
                pass
        return None


def _fetch_html() -> str:
    context = ssl.create_default_context()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }
    request = urllib.request.Request(SCRAPMETALBUYERS_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S, context=context) as resp:  # type: ignore[arg-type]
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def _get_soup(debug: bool = False) -> SoupDocument:
    html = _fetch_html()
    if debug:
        snap = os.path.join(tempfile.gettempdir(), "scrapmetalbuyers_snapshot.html")
        with open(snap, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved HTML snapshot to {snap}")
    return SoupDocument(html)


# --------------------------------- parsers -----------------------------------

def _parse_prices_from_html(doc: SoupDocument) -> Dict[str, float]:
    """
    Extract price table from HTML.
    Expected structure:
    <table>
      <tr>
        <td>Material Name</td>
        <td>$X.XX per lb</td>
      </tr>
    </table>
    """
    prices: Dict[str, float] = {}

    # Try to find table with BS4
    if doc._soup:
        tables = doc.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    material = cells[0].get_text(strip=True)
                    price_text = cells[1].get_text(strip=True)
                    
                    # Extract price value
                    price_val = _to_float(price_text)
                    if math.isfinite(price_val) and material:
                        # Clean material name
                        material = material.strip()
                        prices[material] = price_val

    # Fallback: regex parsing if BS4 didn't work or found nothing
    if not prices:
        text = doc.get_text(" ", strip=True)
        
        # Pattern: Material name followed by price (with various formats)
        # e.g., "Copper $3.50 per lb" or "Aluminum $0.75/lb"
        patterns = [
            re.compile(r"([A-Za-z][A-Za-z\s/\-()]+?)\s+\$?\s*([0-9.]+)\s*(?:per\s+lb|\/\s*lb)", re.I),
            re.compile(r"([A-Za-z][A-Za-z\s/\-()]+?)\s+\$\s*([0-9.]+)", re.I),
        ]
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                material = match.group(1).strip()
                price_text = match.group(2)
                price_val = _to_float(price_text)
                
                if math.isfinite(price_val) and material and len(material) > 2:
                    # Avoid picking up random text
                    if not any(skip in material.lower() for skip in ["call", "contact", "price", "today", "current"]):
                        prices.setdefault(material, price_val)

    return prices


def _extract_as_of_date(doc: SoupDocument) -> Optional[str]:
    """Try to extract the 'as of' date from the page."""
    text = doc.get_text(" ", strip=True)
    
    # Common patterns
    patterns = [
        re.compile(r"current as of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I),
        re.compile(r"updated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I),
        re.compile(r"as of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})", re.I),
        re.compile(r"(\d{4}-\d{2}-\d{2})"),
        re.compile(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})"),
    ]
    
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)
    
    return None


# ------------------------------ main scrape ----------------------------------

def scrape_scrapmetalbuyers_prices(force: bool = False, debug: bool = False) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "source": URL,
        "asof": "Nov 13, 2025" | None,
        "prices_usd_per_lb": {...},
        "prices_usd_per_kg": {...}
      }
    With in-memory + temp-file caching.
    """
    # in-memory cache
    mc = _MEM_CACHE.get("data")
    now = time.time()
    if mc and now - mc[0] < CACHE_TTL_S and not force:
        return mc[1]

    # temp-file cache
    if not force:
        tc = _read_temp_cache()
        if tc:
            _MEM_CACHE["data"] = (now, tc)
            return tc

    soup = _get_soup(debug=debug)
    
    prices_lb = _parse_prices_from_html(soup)
    asof = _extract_as_of_date(soup)

    data: Dict[str, Any] = {
        "source": SCRAPMETALBUYERS_URL,
        "asof": asof,
        "prices_usd_per_lb": prices_lb,
        "prices_usd_per_kg": {k: round(usdlb_to_usdkg(v), 4) for k, v in prices_lb.items()},
    }

    # cache
    _MEM_CACHE["data"] = (now, data)
    _write_temp_cache(data)
    return data


# --------------------------- material mapping --------------------------------

# Material keyword mapping for fuzzy lookup
MATERIAL_KEYWORDS: Dict[str, List[str]] = {
    "copper": ["copper", "cu", "bare bright", "wire", "#1 copper", "#2 copper"],
    "aluminum": ["aluminum", "aluminium", "al", "alum", "6061", "extrusion"],
    "brass": ["brass", "yellow brass", "red brass"],
    "steel": ["steel", "iron", "ferrous", "scrap steel", "sheet iron"],
    "stainless": ["stainless", "stainless steel", "304", "316", "ss"],
    "lead": ["lead", "pb", "battery", "wheel weight"],
    "zinc": ["zinc", "zn"],
    "nickel": ["nickel", "ni"],
    "tin": ["tin", "sn"],
    "insulated_wire": ["insulated wire", "insulated copper", "wire"],
    "catalytic": ["catalytic", "cat", "converter"],
    "battery": ["battery", "batteries"],
    "carbide": ["carbide", "tungsten carbide"],
    "titanium": ["titanium", "ti"],
}


def _normalize_material_name(name: str) -> str:
    """Normalize material name for matching."""
    return name.lower().strip()


def _find_price_for_material(prices: Dict[str, float], material_key: str) -> Optional[Tuple[float, str]]:
    """
    Find best matching price for a material key.
    Returns (price, exact_material_name) or None.
    """
    normalized_key = _normalize_material_name(material_key)
    
    # Direct match first
    for material_name, price in prices.items():
        if _normalize_material_name(material_name) == normalized_key:
            return (price, material_name)
    
    # Keyword-based matching
    keywords = MATERIAL_KEYWORDS.get(normalized_key, [normalized_key])
    
    for material_name, price in prices.items():
        norm_name = _normalize_material_name(material_name)
        for keyword in keywords:
            if keyword in norm_name or norm_name in keyword:
                return (price, material_name)
    
    # Partial match
    for material_name, price in prices.items():
        norm_name = _normalize_material_name(material_name)
        if normalized_key in norm_name or norm_name in normalized_key:
            return (price, material_name)
    
    return None


def get_live_scrap_price_usd_per_lb(
    material_key: str, 
    fallback_usd_per_lb: float = 0.50
) -> Tuple[float, str]:
    """
    Given a material key like 'copper', 'aluminum', returns (usd_per_lb, source_string).
    Falls back to provided rate if not found.
    """
    data = scrape_scrapmetalbuyers_prices(force=False)
    prices_lb = data.get("prices_usd_per_lb", {})
    asof = data.get("asof", "today")
    
    result = _find_price_for_material(prices_lb, material_key)
    if result:
        price, exact_name = result
        return (price, f"ScrapMetalBuyers {exact_name} ({asof})")
    
    # Fallback
    return (float(fallback_usd_per_lb), "house_rate")


def get_live_scrap_price_usd_per_kg(
    material_key: str, 
    fallback_usd_per_kg: float = 1.10
) -> Tuple[float, str]:
    """
    Given a material key, returns (usd_per_kg, source_string).
    Falls back to provided rate if not found.
    """
    fallback_lb = usdkg_to_usdlb(fallback_usd_per_kg)
    price_lb, source = get_live_scrap_price_usd_per_lb(material_key, fallback_usd_per_lb=fallback_lb)
    price_kg = usdlb_to_usdkg(price_lb)
    return (price_kg, source.replace("USD/lb", "USD/kg") if "USD/lb" in source else source)


# ----------------------------------- CLI -------------------------------------

def _main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Scrap Metal Buyers scraper → USD/lb")
    ap.add_argument("--json", action="store_true", help="Print full JSON result")
    ap.add_argument("--force", action="store_true", help="Bypass cache and re-fetch")
    ap.add_argument("--debug", action="store_true", help="Save HTML snapshot and verbose output")
    ap.add_argument("--material", type=str, default="", help="Lookup price for a material (e.g., copper, aluminum)")
    ap.add_argument("--unit", choices=["lb", "kg", "both"], default="lb",
                    help="Display unit for --material (lb, kg, or both).")
    ap.add_argument("--fallback-lb", type=float, default=0.50, help="Fallback USD/lb if not found")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        data = scrape_scrapmetalbuyers_prices(force=args.force, debug=args.debug)
    except Exception as e:
        logging.error("Failed to fetch ScrapMetalBuyers data: %s", e)
        return 2

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0

    if args.material:
        if args.unit == "both":
            p_lb, src = get_live_scrap_price_usd_per_lb(args.material, fallback_usd_per_lb=args.fallback_lb)
            p_kg = usdlb_to_usdkg(p_lb)
            logging.info(
                "%s: $%.4f / lb   |   $%.4f / kg  (source: %s)",
                args.material,
                p_lb,
                p_kg,
                src,
            )
        elif args.unit == "kg":
            fallback_kg = usdlb_to_usdkg(args.fallback_lb)
            p_kg, src = get_live_scrap_price_usd_per_kg(args.material, fallback_usd_per_kg=fallback_kg)
            logging.info("%s: $%.4f / kg  (source: %s)", args.material, p_kg, src)
        else:
            p_lb, src = get_live_scrap_price_usd_per_lb(args.material, fallback_usd_per_lb=args.fallback_lb)
            logging.info("%s: $%.4f / lb  (source: %s)", args.material, p_lb, src)
        return 0

    asof = data.get("asof", "today")
    logging.info("ScrapMetalBuyers pricing (as of %s)", asof)
    logging.info("Source: %s", data.get("source"))
    logging.info("\nPrices (USD/lb):")
    for material, price in sorted(data.get("prices_usd_per_lb", {}).items()):
        logging.info("  %s: $%.4f", material, price)
    
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
