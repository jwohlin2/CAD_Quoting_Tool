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
  python -m cad_quoter.pricing.wieland_scraper --json
  python -m cad_quoter.pricing.wieland_scraper --material 6061
  python -m cad_quoter.pricing.wieland_scraper --force --debug
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

from cad_quoter.config import configure_logging, logger
from cad_quoter.pricing.materials import LB_PER_KG, usdkg_to_usdlb
from cad_quoter.utils import jdump
from cad_quoter.utils.numeric import coerce_positive_float as _coerce_positive_float

try:  # pragma: no cover - optional dependency in production
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - fallback without bs4
    BeautifulSoup = None  # type: ignore


# --------------------------------- config ------------------------------------

WIELAND_URL = "https://www.wieland.com/en/resources/metal-information"

CACHE_TTL_S = int(os.getenv("WIELAND_CACHE_TTL_S", 60 * 30))         # 30 minutes
REQUEST_TIMEOUT_S = int(os.getenv("WIELAND_REQ_TIMEOUT_S", 30))      # seconds
USER_AGENT = os.getenv(
    "WIELAND_USER_AGENT",
    "Mozilla/5.0 (compatible; QuoterBot/1.0; +https://example.local/quote-tool)"
)

# --------------------------------- globals -----------------------------------

_NUM_RE = re.compile(r"[+-]?(?:\d[\d\s\u202f\u00a0.,']*)")
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


class SoupDocument:
    """Small wrapper that mimics a subset of BeautifulSoup we rely on."""

    __slots__ = ("html", "_soup", "_json_cache")

    def __init__(self, html: str) -> None:
        self.html = html
        self._soup = BeautifulSoup(html, "lxml") if BeautifulSoup else None
        self._json_cache: Optional[List[Any]] = None

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

    def find_all(self, *args, **kwargs):  # pragma: no cover - rarely used without bs4
        if self._soup is not None:
            try:
                return self._soup.find_all(*args, **kwargs)
            except AttributeError:
                pass
        return []


def _fetch_html() -> str:
    context = ssl.create_default_context()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }
    request = urllib.request.Request(WIELAND_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S, context=context) as resp:  # type: ignore[arg-type]
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def _get_soup(debug: bool = False) -> SoupDocument:
    html = _fetch_html()
    if debug:
        snap = os.path.join(tempfile.gettempdir(), "wieland_snapshot.html")
        with open(snap, "w", encoding="utf-8") as f:
            f.write(html)
        logger.debug("Saved HTML snapshot to %s", snap)
    return SoupDocument(html)


def _block_text(el) -> str:
    return " ".join(el.stripped_strings)


_JSON_PATTERNS = [
    re.compile(r"<script[^>]+id=\"__NEXT_DATA__\"[^>]*>\s*(\{.*?\})\s*</script>", re.S | re.I),
    re.compile(r"window\.__NUXT__\s*=\s*(\{.*?\});", re.S | re.I),
    re.compile(r"data-props=\"(\{.*?\})\"", re.S | re.I),
    re.compile(r"<script[^>]+type=\"application/json\"[^>]*>\s*(\{.*?\})\s*</script>", re.S | re.I),
]


def _extract_json_payloads(doc: SoupDocument) -> List[Any]:
    if doc._json_cache is not None:
        return doc._json_cache

    payloads: List[Any] = []
    html = doc.html
    for pattern in _JSON_PATTERNS:
        for match in pattern.finditer(html):
            blob = html_unescape(match.group(1))
            blob = blob.strip()
            if blob.endswith(";"):
                blob = blob[:-1]
            if not blob:
                continue
            try:
                payloads.append(json.loads(blob))
            except json.JSONDecodeError:
                # Some payloads are HTML attribute encoded (quotes escaped as &quot;)
                try:
                    payloads.append(json.loads(html_unescape(blob)))
                except Exception:
                    continue
            except Exception:
                continue

    doc._json_cache = payloads
    return payloads


def _iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


def _maybe_to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return math.nan
    if isinstance(value, str):
        return _to_float(value)
    return math.nan


def _compose_unit(row: Dict[str, Any]) -> str:
    currency = str(row.get("currency") or row.get("currencySymbol") or "").strip()
    unit = str(
        row.get("unit")
        or row.get("unitLabel")
        or row.get("unitShort")
        or row.get("unitName")
        or row.get("unitSymbol")
        or ""
    ).strip()
    per = str(row.get("per") or row.get("unitSuffix") or row.get("valuePer") or "").strip()

    currency = currency.upper()
    unit = unit.upper()
    per = per.upper()

    if currency and unit and "/" in unit:
        return f"{currency}/{unit.split('/')[-1]}"
    if currency and unit:
        if per:
            return f"{currency}/{unit} {per}".replace("  ", " ")
        return f"{currency}/{unit}"
    if currency and per:
        return f"{currency}/{per}"
    if currency:
        return currency
    return unit


def _extract_price_rows(doc: SoupDocument) -> Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]], List[Tuple[str, float, str]]]:
    """Return lists of (name, value, unit) for EUR, GBP and USD price tables."""

    eur_rows: List[Tuple[str, float, str]] = []
    gbp_rows: List[Tuple[str, float, str]] = []
    usd_rows: List[Tuple[str, float, str]] = []

    for payload in _extract_json_payloads(doc):
        for entry in _iter_dicts(payload):
            name = entry.get("name") or entry.get("title") or entry.get("label") or entry.get("material")
            value = entry.get("value") or entry.get("price") or entry.get("amount") or entry.get("priceValue")
            if not name or value is None:
                continue

            unit_str = _compose_unit(entry)
            if not unit_str:
                # Try explicit pieces
                currency = entry.get("currency") or entry.get("currencySymbol")
                unit = entry.get("unit") or entry.get("unitLabel") or entry.get("unitShort")
                per = entry.get("per") or entry.get("valuePer") or entry.get("unitSuffix")
                unit_parts = [p for p in [currency, unit, per] if p]
                if unit_parts:
                    currency_part = str(unit_parts[0]).upper()
                    rest = " ".join(str(p).upper() for p in unit_parts[1:])
                    unit_str = currency_part if not rest else f"{currency_part}/{rest}"
            if not unit_str:
                continue

            normalized_unit = (
                unit_str.upper()
                .replace("EUR /", "EUR/")
                .replace("GBP /", "GBP/")
                .replace("USD /", "USD/")
                .replace("  ", " ")
            )

            price_val = _maybe_to_float(value)
            if not math.isfinite(price_val):
                continue

            region_hint = str(entry.get("region") or entry.get("market") or entry.get("area") or "").lower()

            bucket: Optional[List[Tuple[str, float, str]]] = None
            if "GBP" in normalized_unit or "UNITED KINGDOM" in region_hint or "england" in region_hint or "uk" in region_hint:
                bucket = gbp_rows
            elif "EUR" in normalized_unit or "europe" in region_hint or "germany" in region_hint:
                bucket = eur_rows
            elif "USD" in normalized_unit:
                bucket = usd_rows

            if bucket is None:
                # fallback on keywords in unit
                if "GBP" in normalized_unit:
                    bucket = gbp_rows
                elif "EUR" in normalized_unit:
                    bucket = eur_rows
                elif "USD" in normalized_unit:
                    bucket = usd_rows

            if bucket is None:
                continue

            bucket.append((str(name).strip(), price_val, normalized_unit))

    return eur_rows, gbp_rows, usd_rows


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

def _parse_fx(doc: SoupDocument) -> Dict[str, float]:
    """Extract FX pairs from structured payloads with text fallback."""

    fx: Dict[str, float] = {}

    # Structured JSON payloads first
    for payload in _extract_json_payloads(doc):
        for entry in _iter_dicts(payload):
            pair = str(
                entry.get("pair")
                or entry.get("pairName")
                or entry.get("currencyPair")
                or entry.get("pairing")
                or ""
            )
            value = entry.get("value") or entry.get("rate") or entry.get("fxRate") or entry.get("amount")
            base = entry.get("base") or entry.get("baseCurrency") or entry.get("fromCurrency")
            quote = entry.get("quote") or entry.get("targetCurrency") or entry.get("toCurrency")

            candidates: List[Tuple[str, Any]] = []
            if pair:
                candidates.append((pair, value))
            if base and quote:
                pair_name = f"{base}/{quote}".upper()
                candidates.append((pair_name, value or entry.get("price")))

            for key, raw_val in entry.items():
                normalized_key = key.lower().replace("_", "").replace("-", "")
                val = _maybe_to_float(raw_val)
                if not math.isfinite(val):
                    continue
                if "eurusd" in normalized_key:
                    fx.setdefault("EURUSD", val)
                elif "eurgbp" in normalized_key:
                    fx.setdefault("EURGBP", val)
                elif "gbpusd" in normalized_key:
                    fx.setdefault("GBPUSD", val)

            for pair_name, raw in candidates:
                if not pair_name:
                    continue
                val = _maybe_to_float(raw)
                if not math.isfinite(val):
                    continue
                pair_name = pair_name.upper().replace(" ", "")
                if "EUR/USD" in pair_name or pair_name == "EURUSD":
                    fx.setdefault("EURUSD", val)
                elif "EUR/GBP" in pair_name or pair_name == "EURGBP":
                    fx.setdefault("EURGBP", val)
                elif "GBP/USD" in pair_name or pair_name == "GBPUSD":
                    fx.setdefault("GBPUSD", val)

    if len(fx) >= 3:
        return fx

    # Text fallback (old markup)
    text = doc.get_text(" ", strip=True)
    patterns = {
        "EURUSD": re.compile(r"EUR\s*/\s*USD\b\s*([0-9.,\s]+)", re.I),
        "EURGBP": re.compile(r"EUR\s*/\s*GBP\b\s*([0-9.,\s]+)", re.I),
        "GBPUSD": re.compile(r"GBP\s*/\s*USD\b\s*([0-9.,\s]+)", re.I),
    }
    for key, pattern in patterns.items():
        if key in fx and math.isfinite(fx[key]):
            continue
        match = pattern.search(text)
        if match:
            val = _to_float(match.group(1))
            if math.isfinite(val):
                fx[key] = val

    if "GBPUSD" not in fx:
        eurusd = fx.get("EURUSD")
        eurgbp = fx.get("EURGBP")
        if eurusd and eurgbp and math.isfinite(eurusd) and math.isfinite(eurgbp) and eurgbp:
            fx["GBPUSD"] = eurusd / eurgbp

    return {k: v for k, v in fx.items() if math.isfinite(v)}


def _parse_lme_usd_per_kg(doc: SoupDocument, fx: Dict[str, float]) -> Tuple[Dict[str, float], Optional[str]]:
    """Extract LME settlement data and an as-of date from structured payloads."""

    out: Dict[str, float] = {}
    asof: Optional[str] = None

    for payload in _extract_json_payloads(doc):
        for entry in _iter_dicts(payload):
            symbol = (
                entry.get("symbol")
                or entry.get("code")
                or entry.get("metal")
                or entry.get("abbreviation")
                or entry.get("shortName")
                or entry.get("name")
            )
            unit = entry.get("unit") or entry.get("priceUnit") or entry.get("unitLabel")
            currency = entry.get("currency") or entry.get("currencySymbol")
            value = entry.get("value") or entry.get("price") or entry.get("amount")

            unit_str = _compose_unit({"currency": currency, "unit": unit}) if (currency or unit) else str(unit or "")
            unit_str = unit_str.replace("EUR/EUR", "EUR").replace("//", "/").strip()

            if not symbol or not unit_str:
                continue
            symbol_str = str(symbol).strip().upper()
            if len(symbol_str) > 3:
                # Many structured payloads include "Copper" etc. Only keep common LME symbols.
                if symbol_str not in {"COPPER", "NICKEL", "ZINC", "ALUMINUM", "ALUMINIUM", "TIN"}:
                    continue
                symbol_str = {
                    "COPPER": "CU",
                    "NICKEL": "NI",
                    "ZINC": "ZN",
                    "ALUMINUM": "AL",
                    "ALUMINIUM": "AL",
                    "TIN": "SN",
                }[symbol_str]

            if "USD" not in unit_str.upper():
                continue

            price = _maybe_to_float(value)
            if not math.isfinite(price):
                continue

            try:
                out[symbol_str] = round(_usd_per_kg_from(price, unit_str, fx), 4)
            except Exception:
                continue

        if asof is None:
            for text in _iter_strings(payload):
                if not text:
                    continue
                match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
                if match:
                    asof = match.group(1)
                    break
                match = re.search(r"([A-Za-z]{3}\s+\d{1,2},\s+\d{4})", text)
                if match:
                    asof = match.group(1)
                    break

    if out and asof:
        return out, asof

    # Fallback to page text heuristics
    txt = doc.get_text(" ", strip=True)
    for sym in ("CU", "ZN", "SN", "NI", "AL"):
        m = re.search(rf"\b{sym}\b\s*([0-9.,\s]+)\s*(USD\s*/\s*(?:TO|T|TONNE|KG))", txt, re.I)
        if m:
            price = _to_float(m.group(1))
            unit = m.group(2)
            try:
                out.setdefault(sym, round(_usd_per_kg_from(price, unit, fx), 4))
            except Exception:
                continue

    if asof is None:
        m = re.search(r"Value from\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{4})", txt)
        if m:
            asof = m.group(1)

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
    data["lme_usd_per_lb"] = {k: round(usdkg_to_usdlb(v), 6) for k, v in data["lme_usd_per_kg"].items()}
    data["asof"] = asof

    eur_rows, gbp_rows, usd_rows = _extract_price_rows(soup)

    for name, val, unit in eur_rows:
        data["wieland_eur100kg"][name] = val
        try:
            data["wieland_usd_per_kg"][name] = round(_usd_per_kg_from(val, unit, fx), 4)
        except Exception:
            continue

    for name, val, unit in gbp_rows:
        data["england_gbp_t"][name] = val
        try:
            data["england_usd_per_kg"][name] = round(_usd_per_kg_from(val, unit, fx), 4)
        except Exception:
            continue

    # Some entries may already be denominated in USD/kg (direct list prices)
    for name, val, unit in usd_rows:
        try:
            data["wieland_usd_per_kg"].setdefault(name, round(_usd_per_kg_from(val, unit, fx), 4))
        except Exception:
            continue

    # Fallback on plain-text parsing if structured payloads were empty
    if not eur_rows:
        all_txt = soup.get_text(" ", strip=True)
        for row in re.finditer(r"([A-Za-z0-9 \u00ae/()+\-]+?)\s+([0-9.,]+)\s+(EUR\s*/\s*100\s*KG)", all_txt, re.I):
            name = row.group(1).strip()
            val = _to_float(row.group(2))
            unit = row.group(3)
            if math.isfinite(val):
                data["wieland_eur100kg"][name] = val
                try:
                    data["wieland_usd_per_kg"][name] = round(_usd_per_kg_from(val, unit, fx), 4)
                except Exception:
                    continue

    if not gbp_rows:
        all_txt = soup.get_text(" ", strip=True)
        for row in re.finditer(r"([A-Za-z0-9 ()/\-]+?)\s+([0-9.,]+)\s+(GBP\s*/\s*(?:T|TO|100\s*KG))", all_txt, re.I):
            name = row.group(1).strip()
            val = _to_float(row.group(2))
            unit = row.group(3)
            if math.isfinite(val):
                data["england_gbp_t"][name] = val
                try:
                    data["england_usd_per_kg"][name] = round(_usd_per_kg_from(val, unit, fx), 4)
                except Exception:
                    continue

    data["wieland_usd_per_lb"] = {k: round(usdkg_to_usdlb(v), 6) for k, v in data["wieland_usd_per_kg"].items()}
    data["england_usd_per_lb"] = {k: round(usdkg_to_usdlb(v), 6) for k, v in data["england_usd_per_kg"].items()}

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

    # Steel (mild/carbon) → prefer direct USD list if available
    "STEEL": {"bucket": "wieland_usd_per_kg", "key": "Direct USD"},
    "A36": {"bucket": "wieland_usd_per_kg", "key": "Direct USD"},
    "1018": {"bucket": "wieland_usd_per_kg", "key": "Direct USD"},
    "1020": {"bucket": "wieland_usd_per_kg", "key": "Direct USD"},
    "1045": {"bucket": "wieland_usd_per_kg", "key": "Direct USD"},

    # Brass examples (often in the EUR/100 KG block)
    "MS 58I": {"bucket": "wieland_usd_per_kg", "key": "MS 58I"},
    "CW614N": {"bucket": "wieland_usd_per_kg", "key": "MS 58I"},
}


STEEL_KEYWORDS = (
    "STEEL",
    "A36",
    "A-36",
    "MILD",
    "CARBON",
    "HOT ROLLED",
    "COLD ROLLED",
    "CRS",
    "HRS",
    "1018",
    "1020",
    "1045",
)


_STEEL_SCRAP_TOKENS: Tuple[str, ...] = tuple(
    sorted({token.lower() for token in STEEL_KEYWORDS} | {"steel"})
)


_SCRAP_FAMILY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "aluminum": ("alum", "6061", "7075", "2024"),
    "stainless": ("stainless", "304", "316", "17-4", "17 4", "17-7", "17 7"),
    "steel": _STEEL_SCRAP_TOKENS,
    "copper": ("copper", "cu", "c110"),
    "brass": ("brass", "ms 58", "ms58", "bronze"),
    "titanium": ("titanium", "ti"),
}


_SCRAP_LOOKUP_PRIORITY: Dict[
    str, Tuple[Tuple[str, Tuple[str, ...], bool], ...]
] = {
    "aluminum": (
        ("wieland_usd_per_lb", ("Aluminium Scrap", "Aluminum Scrap"), True),
        ("england_usd_per_lb", ("Aluminium Scrap", "Aluminum Scrap"), True),
        ("england_usd_per_lb", (), True),
        ("wieland_usd_per_lb", (), True),
        ("england_usd_per_lb", ("Aluminium", "Aluminum"), False),
        ("wieland_usd_per_lb", ("Aluminium", "Aluminum"), False),
        ("lme_usd_per_lb", ("AL",), False),
    ),
    "stainless": (
        ("england_usd_per_lb", ("Stainless Scrap", "Stainless Steel Scrap"), True),
        ("wieland_usd_per_lb", ("Stainless Scrap",), True),
        ("england_usd_per_lb", (), True),
        ("wieland_usd_per_lb", (), True),
        ("england_usd_per_lb", ("Stainless", "304", "316", "SS"), False),
        ("wieland_usd_per_lb", ("Stainless", "304", "316", "SS"), False),
        ("lme_usd_per_lb", ("NI",), False),
    ),
    "steel": (
        ("england_usd_per_lb", ("Steel Scrap",), True),
        ("wieland_usd_per_lb", ("Steel Scrap",), True),
        ("england_usd_per_lb", ("Steel", "Carbon", "Mild"), True),
        ("wieland_usd_per_lb", ("Steel", "Carbon", "Mild"), True),
        ("england_usd_per_lb", ("Steel", "Carbon", "Mild"), False),
        ("lme_usd_per_lb", ("NI",), False),
    ),
    "copper": (
        ("wieland_usd_per_lb", ("Copper Scrap",), True),
        ("england_usd_per_lb", ("Copper Scrap",), True),
        ("england_usd_per_lb", (), True),
        ("wieland_usd_per_lb", (), True),
        ("england_usd_per_lb", ("Copper", "Kupfer"), False),
        ("wieland_usd_per_lb", ("Copper", "Kupfer"), False),
        ("lme_usd_per_lb", ("CU",), False),
    ),
    "brass": (
        ("england_usd_per_lb", ("Brass Scrap", "MS", "Bronze Scrap"), True),
        ("wieland_usd_per_lb", ("Brass Scrap",), True),
        ("england_usd_per_lb", ("Brass", "MS 58"), False),
        ("wieland_usd_per_lb", ("Brass", "MS 58"), False),
    ),
    "titanium": (
        ("wieland_usd_per_lb", ("Titanium Scrap",), True),
        ("england_usd_per_lb", ("Titanium Scrap",), True),
        ("england_usd_per_lb", ("Titanium",), False),
        ("wieland_usd_per_lb", ("Titanium",), False),
    ),
}


def _lookup_steel_price(data: Dict[str, Any]) -> Optional[Tuple[float, str, str]]:
    """Return (price, bucket, key) for the best available steel proxy."""

    # Prefer explicit Wieland USD list price when present
    direct = get_usd_per_kg(data, "wieland_usd_per_kg", "Direct USD")
    if direct and math.isfinite(direct):
        return direct, "wieland_usd_per_kg", "Direct USD"

    # Some feeds provide steel under the England bucket
    for steel_key in ("Steel", "Steel Scrap", "Steel Billet"):
        val = get_usd_per_kg(data, "england_usd_per_kg", steel_key)
        if val and math.isfinite(val):
            return val, "england_usd_per_kg", steel_key

    # Fall back to LME Nickel as the closest ferrous proxy the scraper exposes
    ni = get_usd_per_kg(data, "lme_usd_per_kg", "NI")
    if ni and math.isfinite(ni):
        return ni, "lme_usd_per_kg", "NI"

    return None


def get_usd_per_kg(data: Dict[str, Any], bucket: str, key: str) -> Optional[float]:
    d = data.get(bucket) or {}
    val = d.get(key)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def _format_source(bucket: str, key: str) -> str:
    if bucket == "wieland_usd_per_kg":
        return f"Wieland {key}"
    if bucket == "lme_usd_per_kg":
        return f"Wieland LME {key}"
    if bucket == "england_usd_per_kg":
        return f"Wieland England {key}"
    return f"Wieland {bucket}:{key}"


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
            return p, f"{_format_source(m['bucket'], m['key'])} ({data.get('asof','today')})"

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
    if any(token in key for token in STEEL_KEYWORDS):
        steel_lookup = _lookup_steel_price(data)
        if steel_lookup:
            price, bucket, bucket_key = steel_lookup
            return float(price), f"{_format_source(bucket, bucket_key)} ({data.get('asof','today')})"

    # Fallback
    return float(fallback_usd_per_kg), "house_rate"

def get_live_material_price(material_key: str, unit: str = "kg", fallback_usd_per_kg: float = 8.0) -> Tuple[float, str]:
    """
    Returns (price, source) where price is USD/<unit>, unit in {"kg","lb"}.
    Uses same mapping/heuristics as get_live_material_price_usd_per_kg.
    """
    price_kg, src = get_live_material_price_usd_per_kg(material_key, fallback_usd_per_kg=fallback_usd_per_kg)
    if unit.lower() == "lb":
        return usdkg_to_usdlb(price_kg), src.replace("USD/kg", "USD/lb") if "USD/kg" in src else src
    return price_kg, src


def _canonical_scrap_family(material_family: Optional[str]) -> str:
    if not material_family:
        return "aluminum"
    family = material_family.strip().lower()
    if not family:
        return "aluminum"
    for canonical, tokens in _SCRAP_FAMILY_KEYWORDS.items():
        if canonical in family:
            return canonical
        if any(token in family for token in tokens):
            return canonical
    return "aluminum"
def _case_insensitive_lookup(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    lowered = key.lower()
    for candidate_key, candidate_value in mapping.items():
        if isinstance(candidate_key, str) and candidate_key.lower() == lowered:
            return candidate_value
    return None


def _lookup_scrap_price(
    data: Mapping[str, Any],
    bucket: str,
    keys: Tuple[str, ...],
    keywords: Iterable[str],
    require_scrap: bool,
) -> Optional[float]:
    bucket_map = data.get(bucket)
    if not isinstance(bucket_map, Mapping):
        return None

    normalized_keywords = tuple(token.lower() for token in keywords)

    if keys:
        for key in keys:
            candidate = _case_insensitive_lookup(bucket_map, key)
            price = _coerce_positive_float(candidate)
            if price is not None:
                return price

    for label, value in bucket_map.items():
        if not isinstance(label, str):
            continue
        normalized = label.lower()
        if require_scrap and "scrap" not in normalized:
            continue
        if normalized_keywords and not any(token in normalized for token in normalized_keywords):
            continue
        price = _coerce_positive_float(value)
        if price is not None:
            return price

    return None


def get_scrap_price_per_lb(material_family: Optional[str], *, fallback: Optional[float] = None) -> Optional[float]:
    """Return the USD/lb scrap price for a material family using Wieland data."""

    data = scrape_wieland_prices(force=False)
    family = _canonical_scrap_family(material_family)
    lookup_plan = _SCRAP_LOOKUP_PRIORITY.get(family) or _SCRAP_LOOKUP_PRIORITY.get("aluminum", ())
    keywords = _SCRAP_FAMILY_KEYWORDS.get(family, ())

    for bucket, keys, require_scrap in lookup_plan:
        price = _lookup_scrap_price(data, bucket, keys, keywords, require_scrap)
        if price is not None:
            return price

    return fallback


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

    configure_logging(logging.DEBUG if args.debug else logging.INFO, force=args.debug)

    try:
        data = scrape_wieland_prices(force=args.force, debug=args.debug)
    except Exception as e:
        logger.error("Failed to fetch Wieland data: %s", e)
        return 2

    if args.json:
        logger.info("Wieland pricing JSON:\n%s", jdump(data, default=None))
        return 0

    if args.material:
        if args.unit == "both":
            p_kg, src = get_live_material_price(args.material, unit="kg", fallback_usd_per_kg=args.fallback)
            p_lb, _   = get_live_material_price(args.material, unit="lb", fallback_usd_per_kg=args.fallback)
            logger.info(
                "%s: $%.4f / kg   |   $%.4f / lb  (source: %s)",
                args.material,
                p_kg,
                p_lb,
                src,
            )
        else:
            p, src = get_live_material_price(args.material, unit=args.unit, fallback_usd_per_kg=args.fallback)
            logger.info("%s: $%.4f / %s  (source: %s)", args.material, p, args.unit, src)
        return 0
    asof = data.get("asof", "today")
    logger.info("Wieland metal info (as of %s)", asof)
    logger.info("FX: %s", data.get("fx"))
    logger.info("LME: USD/kg: %s", data.get("lme_usd_per_kg"))
    logger.info("LME: USD/lb: %s", data.get("lme_usd_per_lb"))
    def _head(d: Dict[str, float], n=6):
        items = list(d.items()); return items[:n] + ([("...", "...")],) if len(items) > n else items
    logger.info("Wieland list USD/kg: %s", _head(data.get("wieland_usd_per_kg", {})))
    logger.info("Wieland list USD/lb: %s", _head(data.get("wieland_usd_per_lb", {})))
    logger.info("England USD/kg: %s", _head(data.get("england_usd_per_kg", {})))
    logger.info("England USD/lb: %s", _head(data.get("england_usd_per_lb", {})))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
