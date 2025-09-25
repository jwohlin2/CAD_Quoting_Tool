# wieland_scraper.py
# -*- coding: utf-8 -*-
"""
Scrape & normalize metal prices from Wieland:
  https://www.wieland.com/en/resources/metal-information

Outputs:
- FX dict (EURUSD, GBPUSD, EURGBP)
- LME settlement normalized to USD/kg (keys: 'CU','AL','NI','ZN','SN' when present)
- "Metal prices" (EUR/100 KG) normalized to USD/kg
- "Metal prices England" (GBP/t or GBP/100 KG) normalized to USD/kg

Public API:
  scrape_wieland_prices(force=False, debug=False) -> dict
  get_live_material_price_usd_per_kg(material_key: str, fallback_usd_per_kg=8.0) -> (float, source)

CLI:
  python wieland_scraper.py --json
  python wieland_scraper.py --material 6061
  python wieland_scraper.py --force --debug
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import requests
from bs4 import BeautifulSoup


# --------------------------------- config ------------------------------------

WIELAND_URL = "https://www.wieland.com/en/resources/metal-information"

CACHE_TTL_S = int(os.getenv("WIELAND_CACHE_TTL_S", 60 * 30))         # 30 minutes
REQUEST_TIMEOUT_S = int(os.getenv("WIELAND_REQ_TIMEOUT_S", 15))      # seconds
USER_AGENT = os.getenv(
    "WIELAND_USER_AGENT",
    "Mozilla/5.0 (compatible; QuoterBot/1.0; +https://example.local/quote-tool)"
)

# --------------------------------- globals -----------------------------------

_NUM_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
_MEM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # key -> (ts, data)


@dataclass
class ScrapeResult:
    source: str
    fx: Dict[str, float]
    asof: Optional[str]
    lme_usd_per_kg: Dict[str, float]
    wieland_eur100kg: Dict[str, float]
    wieland_usd_per_kg: Dict[str, float]
    england_gbp_t: Dict[str, float]
    england_usd_per_kg: Dict[str, float]


# --------------------------------- utils -------------------------------------
LB_PER_KG = 2.2046226218

def _usdkg_to_usdlb(x: float) -> float:
    return float(x) / LB_PER_KG if x is not None else x


def _to_float(s: str) -> float:
    """Parse first numeric token from a string (handles 9,900.00)."""
    if s is None:
        return math.nan
    m = _NUM_RE.search(s.replace("\xa0", " ").strip())
    if not m:
        return math.nan
    t = m.group(0).replace(",", "")
    try:
        return float(t)
    except Exception:
        return math.nan


def _cache_path() -> str:
    return os.path.join(tempfile.gettempdir(), "wieland_scrape_cache.json")


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


def _get_soup(debug: bool = False) -> BeautifulSoup:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(WIELAND_URL, headers=headers, timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    if debug:
        snap = os.path.join(tempfile.gettempdir(), "wieland_snapshot.html")
        with open(snap, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"[debug] saved HTML snapshot: {snap}", file=sys.stderr)
    return BeautifulSoup(r.text, "lxml")


def _block_text(el) -> str:
    return " ".join(el.stripped_strings)


# ------------------------------- normalization -------------------------------

def _usd_per_kg_from(unit_price: float, unit_str: str, fx: Dict[str, float]) -> float:
    """
    Normalize units → USD/kg
      USD/t or USD/to  → USD/kg   (÷1000)
      EUR/100 KG       → EUR/kg   (÷100) → USD/kg (* EURUSD)
      GBP/t            → GBP/kg   (÷1000) → USD/kg (* GBPUSD)
      GBP/100 KG       → GBP/kg   (÷100)  → USD/kg (* GBPUSD)
      USD/kg           → USD/kg   (as-is)
    """
    unit = (unit_str or "").upper().replace("TONNE", "T").replace("TO", "T").replace("T/", "T")
    if "USD/T" in unit:
        return float(unit_price) / 1000.0
    if "EUR/100 KG" in unit:
        eurusd = fx.get("EURUSD")
        if not eurusd:
            raise RuntimeError("Missing EURUSD for conversion")
        return (float(unit_price) / 100.0) * eurusd
    if "GBP/T" in unit:
        gbpusd = fx.get("GBPUSD")
        if not gbpusd:
            raise RuntimeError("Missing GBPUSD for conversion")
        return (float(unit_price) / 1000.0) * gbpusd
    if "GBP/100 KG" in unit:
        gbpusd = fx.get("GBPUSD")
        if not gbpusd:
            raise RuntimeError("Missing GBPUSD for conversion")
        return (float(unit_price) / 100.0) * gbpusd
    if "USD/KG" in unit:
        return float(unit_price)

    # Fallback heuristics (best effort)
    if "/T" in unit:
        return float(unit_price) / 1000.0
    if "/100 KG" in unit:
        return (float(unit_price) / 100.0) * (fx.get("EURUSD", 1.0))
    return float(unit_price)


# --------------------------------- parsers -----------------------------------

def _parse_fx(soup: BeautifulSoup) -> Dict[str, float]:
    """
    Extract FX from 'Currency' and 'Metal prices England' blocks to avoid cross-matching.
    Examples on page:
      EUR/GBP 0.87071 GBP
      EUR/USD 1.17095 USD
      (England) "GBP £ / USD $ 1.3422"
    """
    fx = {}

    # Scope to 'Currency' heading if present
    cur_heads = soup.find_all(string=re.compile(r"^\s*Currency\s*$", re.I))
    cur_txt = ""
    for h in cur_heads:
        for anc in (getattr(h, "parent", None), getattr(h, "parent", None) and h.parent.parent):
            if getattr(anc, "get_text", None):
                t = anc.get_text(" ", strip=True)
                # simple sanity: contains EUR/
                if "EUR/USD" in t or "EUR / USD" in t or "EUR/GBP" in t:
                    cur_txt = t
                    break
        if cur_txt:
            break
    if not cur_txt:
        cur_txt = soup.get_text(" ", strip=True)

    m = re.search(r"\bEUR\s*/\s*USD\b\s*([0-9.,]+)\s*USD", cur_txt, re.I)
    if m:
        fx["EURUSD"] = _to_float(m.group(1))
    m = re.search(r"\bEUR\s*/\s*GBP\b\s*([0-9.,]+)\s*GBP", cur_txt, re.I)
    if m:
        fx["EURGBP"] = _to_float(m.group(1))

    # England area for GBP/USD number
    eng_heads = soup.find_all(string=re.compile(r"^\s*Metal prices England\s*$", re.I))
    eng_txt = ""
    for h in eng_heads:
        for anc in (getattr(h, "parent", None), getattr(h, "parent", None) and h.parent.parent):
            if getattr(anc, "get_text", None):
                eng_txt = anc.get_text(" ", strip=True)
                if eng_txt:
                    break
        if eng_txt:
            break
    if not eng_txt:
        eng_txt = soup.get_text(" ", strip=True)

    m = re.search(r"\bGBP\b.*?/\s*\bUSD\b.*?([0-9.,]+)", eng_txt, re.I)
    if m:
        fx["GBPUSD"] = _to_float(m.group(1))

    return {k: v for k, v in fx.items() if v and math.isfinite(v)}


def _parse_lme_usd_per_kg(soup: BeautifulSoup, fx: Dict[str, float]) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Scan page text for LME rows like "CU 9,862.00 USD/to" and convert to USD/kg.
    Grab first 'Value from <date>' if present.
    """
    txt = soup.get_text(" ", strip=True)
    out: Dict[str, float] = {}
    asof: Optional[str] = None

    for sym in ("CU", "ZN", "SN", "NI", "AL"):
        m = re.search(rf"\b{sym}\b\s*([0-9.,]+)\s*(USD\s*/\s*(?:to|t|tonne))", txt, re.I)
        if m:
            price = _to_float(m.group(1))
            unit = m.group(2)
            try:
                out[sym] = round(_usd_per_kg_from(price, unit, fx), 4)
            except Exception:
                pass

    d = re.search(r"Value from\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{4})", txt)
    if d:
        asof = d.group(1)

    return out, asof


# ------------------------------ main scrape ----------------------------------

def scrape_wieland_prices(force: bool = False, debug: bool = False) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "source": URL,
        "fx": {...},
        "asof": "Sep 24, 2025" | None,
        "lme_usd_per_kg": {...},
        "wieland_eur100kg": {...},
        "wieland_usd_per_kg": {...},
        "england_gbp_t": {...},
        "england_usd_per_kg": {...}
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
    fx = _parse_fx(soup)

    data: Dict[str, Any] = {
        "source": WIELAND_URL,
        "fx": fx,
        "asof": None,
        "lme_usd_per_kg": {},
        "lme_usd_per_lb": {},          # <--- add
        "wieland_eur100kg": {},
        "wieland_usd_per_kg": {},
        "wieland_usd_per_lb": {},      # <--- add
        "england_gbp_t": {},
        "england_usd_per_kg": {},
        "england_usd_per_lb": {},      # <--- add
    }

    # LME settlement
    lme_map, asof = _parse_lme_usd_per_kg(soup, fx)
    data["lme_usd_per_kg"].update(lme_map)
    data["lme_usd_per_lb"] = {k: round(_usdkg_to_usdlb(v), 6) for k, v in data["lme_usd_per_kg"].items()}
    data["asof"] = asof

    # EUR/100 KG rows (Metal prices) – scan whole page text
    all_txt = soup.get_text(" ", strip=True)
    for row in re.finditer(r"([A-Za-z0-9 \u00ae/()+\-]+?)\s+([0-9.,]+)\s+(EUR\s*/\s*100\s*KG)", all_txt, re.I):
        name = row.group(1).strip()
        val = _to_float(row.group(2))
        unit = row.group(3)
        if math.isfinite(val):
            data["wieland_eur100kg"][name] = val
            try:
                data["wieland_usd_per_kg"][name] = round(_usd_per_kg_from(val, unit, fx), 4)
                data["wieland_usd_per_lb"] = {k: round(_usdkg_to_usdlb(v), 6) for k, v in data["wieland_usd_per_kg"].items()}
            except Exception:
                pass

    # England GBP/t or GBP/100 KG within England section (fallback to whole page)
    eng_heads = soup.find_all(string=re.compile(r"^\s*Metal prices England\s*$", re.I))
    eng_txt = ""
    for h in eng_heads:
        for anc in (getattr(h, "parent", None), getattr(h, "parent", None) and h.parent.parent):
            if getattr(anc, "get_text", None):
                eng_txt = anc.get_text(" ", strip=True)
                if eng_txt:
                    break
        if eng_txt:
            break
    if not eng_txt:
        eng_txt = all_txt

    for row in re.finditer(r"([A-Za-z0-9 ()/\-]+?)\s+([0-9.,]+)\s+(GBP\s*/\s*(?:T|TO|100\s*KG))", eng_txt, re.I):
        name = row.group(1).strip()
        val = _to_float(row.group(2))
        unit = row.group(3)
        if math.isfinite(val):
            data["england_gbp_t"][name] = val
            try:
                data["england_usd_per_kg"][name] = round(_usd_per_kg_from(val, unit, fx), 4)
                data["england_usd_per_lb"] = {k: round(_usdkg_to_usdlb(v), 6) for k, v in data["england_usd_per_kg"].items()}

            except Exception:
                pass

    # cache
    _MEM_CACHE["data"] = (now, data)
    _write_temp_cache(data)
    return data


# --------------------------- material mapping --------------------------------

MATERIAL_MAP: Dict[str, Dict[str, str]] = {
    # Aluminum alloys → LME AL
    "AL": {"bucket": "lme_usd_per_kg", "key": "AL"},
    "ALUMINUM": {"bucket": "lme_usd_per_kg", "key": "AL"},
    "ALUMINIUM": {"bucket": "lme_usd_per_kg", "key": "AL"},
    "6061": {"bucket": "lme_usd_per_kg", "key": "AL"},
    "6061-T6": {"bucket": "lme_usd_per_kg", "key": "AL"},
    "7075": {"bucket": "lme_usd_per_kg", "key": "AL"},

    # Copper
    "CU": {"bucket": "lme_usd_per_kg", "key": "CU"},
    "COPPER": {"bucket": "lme_usd_per_kg", "key": "CU"},
    "C110": {"bucket": "wieland_usd_per_kg", "key": "Wieland Kupfer"},  # list price if present

    # Nickel / stainless (approximate; refine with vendor CSV/premiums)
    "NICKEL": {"bucket": "lme_usd_per_kg", "key": "NI"},
    "304": {"bucket": "lme_usd_per_kg", "key": "NI"},
    "316": {"bucket": "lme_usd_per_kg", "key": "NI"},

    # Brass examples (often in the EUR/100 KG block)
    "MS 58I": {"bucket": "wieland_usd_per_kg", "key": "MS 58I"},
    "CW614N": {"bucket": "wieland_usd_per_kg", "key": "MS 58I"},
}


def get_usd_per_kg(data: Dict[str, Any], bucket: str, key: str) -> Optional[float]:
    d = data.get(bucket) or {}
    val = d.get(key)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def get_live_material_price_usd_per_kg(material_key: str, fallback_usd_per_kg: float = 8.0) -> Tuple[float, str]:
    """
    Given '6061', 'C110', 'Copper', returns (usd_per_kg, source_string).
    1) Try MATERIAL_MAP
    2) Heuristics (family keywords)
    3) Fallback to provided house rate
    """
    data = scrape_wieland_prices(force=False)
    key = (material_key or "").strip().upper()

    m = MATERIAL_MAP.get(key)
    if m:
        p = get_usd_per_kg(data, m["bucket"], m["key"])
        if p:
            return p, f"Wieland {m['bucket']}:{m['key']} ({data.get('asof','today')})"

    # Heuristics by family
    if "AL" in key:
        p = get_usd_per_kg(data, "lme_usd_per_kg", "AL")
        if p:
            return p, f"Wieland LME AL ({data.get('asof','today')})"
    if "CU" in key or "COPPER" in key or "C110" in key:
        p = data.get("wieland_usd_per_kg", {}).get("Wieland Kupfer")
        src = "Wieland Kupfer"
        if p is None:
            p = get_usd_per_kg(data, "lme_usd_per_kg", "CU")
            src = "Wieland LME CU"
        if p:
            return float(p), f"{src} ({data.get('asof','today')})"

    # Fallback
    return float(fallback_usd_per_kg), "house_rate"

def get_live_material_price(material_key: str, unit: str = "kg", fallback_usd_per_kg: float = 8.0) -> Tuple[float, str]:
    """
    Returns (price, source) where price is USD/<unit>, unit in {"kg","lb"}.
    Uses same mapping/heuristics as get_live_material_price_usd_per_kg.
    """
    price_kg, src = get_live_material_price_usd_per_kg(material_key, fallback_usd_per_kg=fallback_usd_per_kg)
    if unit.lower() == "lb":
        return _usdkg_to_usdlb(price_kg), src.replace("USD/kg", "USD/lb") if "USD/kg" in src else src
    return price_kg, src


# ----------------------------------- CLI -------------------------------------

def _main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Wieland metal information → USD/kg")
    ap.add_argument("--json", action="store_true", help="Print full JSON result")
    ap.add_argument("--force", action="store_true", help="Bypass cache and re-fetch")
    ap.add_argument("--debug", action="store_true", help="Save HTML snapshot and verbose logging")
    ap.add_argument("--material", type=str, default="", help="Lookup price for a material key (e.g., 6061, C110, Copper)")
    ap.add_argument("--unit", choices=["kg", "lb", "both"], default="kg",
                help="Display unit for --material (kg, lb, or both).")
    ap.add_argument("--fallback", type=float, default=8.0, help="Fallback USD/kg if not found")
    args = ap.parse_args(argv)

    try:
        data = scrape_wieland_prices(force=args.force, debug=args.debug)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    if args.material:
        if args.unit == "both":
            p_kg, src = get_live_material_price(args.material, unit="kg", fallback_usd_per_kg=args.fallback)
            p_lb, _   = get_live_material_price(args.material, unit="lb", fallback_usd_per_kg=args.fallback)
            print(f"{args.material}: ${p_kg:.4f} / kg   |   ${p_lb:.4f} / lb  (source: {src})")
        else:
            p, src = get_live_material_price(args.material, unit=args.unit, fallback_usd_per_kg=args.fallback)
            print(f"{args.material}: ${p:.4f} / {args.unit}  (source: {src})")
        return 0
    asof = data.get("asof", "today")
    print(f"Wieland metal info (as of {asof})")
    print("FX:", data.get("fx"))
    print("LME: USD/kg:", data.get("lme_usd_per_kg"))
    print("     USD/lb:", data.get("lme_usd_per_lb"))
    def _head(d: Dict[str, float], n=6): 
        items = list(d.items()); return items[:n] + ([("...", "...")],) if len(items) > n else items
    print("Wieland list USD/kg:", _head(data.get("wieland_usd_per_kg", {})))
    print("Wieland list USD/lb:", _head(data.get("wieland_usd_per_lb", {})))
    print("England USD/kg:", _head(data.get("england_usd_per_kg", {})))
    print("England USD/lb:", _head(data.get("england_usd_per_lb", {})))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
