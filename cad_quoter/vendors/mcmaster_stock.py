"""Helpers for working with McMaster-Carr plate stock pricing."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from cad_quoter.resources import default_catalog_csv
from mcmaster_api import McMasterAPI, load_env

__all__ = [
    "StockItem",
    "ALLOW_NEXT_THICKER",
    "load_catalog",
    "choose_item",
    "lookup_sku_and_price_for_mm",
    "main",
]


ALLOW_NEXT_THICKER = False
_CATALOG_CACHE: Optional[Dict[str, Dict[float, List["StockItem"]]]] = None
_ENV_CACHE: Optional[dict] = None
_PRICE_CACHE: dict[Tuple[str, int], Tuple[float, str]] = {}


@dataclass(frozen=True)
class StockItem:
    """Representation of a rectangular plate offered by McMaster-Carr."""

    material: str
    thickness: float  # inches
    length: float  # inches
    width: float  # inches
    part: str


def _normalize_inch_string(value: str) -> str:
    return (
        (value or "")
        .replace("\u201d", '"')
        .replace("\u201c", '"')
        .replace("\u2033", '"')
        .replace("\uff02", '"')
        .replace("\u2044", "/")
        .replace("\xa0", " ")
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )


def _parse_fraction_token(token: str) -> float:
    token = token.strip()
    if not token:
        return 0.0
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        return float(numerator) / float(denominator)
    return float(token)


def parse_inches(value: str) -> float:
    normalised = _normalize_inch_string(str(value or "").strip().lower())
    normalised = (
        normalised.replace('"', "")
        .replace("in.", "")
        .replace("inch", "")
        .replace("in", "")
        .strip()
    )
    normalised = re.sub(r"(?<=\d)-(?=\d)", " ", normalised)
    total = 0.0
    for token in normalised.split():
        total += _parse_fraction_token(token)
    if total == 0.0 and normalised:
        total = _parse_fraction_token(normalised)
    return total


def norm_material(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip().lower())
    replacements = {
        "aluminium": "aluminum",
        "alum": "aluminum",
        "mic 6": "mic6",
        "mic-6": "mic6",
        "tool & jig plate": "aluminum mic6",
    }
    for key, replacement in replacements.items():
        value = value.replace(key, replacement)
    return value


def load_catalog(csv_path: str) -> Dict[str, Dict[float, List[StockItem]]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = {(header or "").strip().lower(): header for header in (reader.fieldnames or [])}
        required = ["material", "thickness_in", "length_in", "width_in", "part"]
        missing = [key for key in required if key not in headers]
        if missing:
            raise ValueError(
                f"CSV missing required headers: {missing}. Found: {list(headers.keys())}"
            )

    catalog: Dict[str, Dict[float, List[StockItem]]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material = norm_material(row[headers["material"]])
            thickness = parse_inches(row[headers["thickness_in"]])
            length = parse_inches(row[headers["length_in"]])
            width = parse_inches(row[headers["width_in"]])
            part = str(row[headers["part"]]).strip()
            if not part or thickness <= 0 or length <= 0 or width <= 0:
                continue
            item = StockItem(material, thickness, length, width, part)
            catalog.setdefault(material, {}).setdefault(thickness, []).append(item)
    return catalog


def _fits(item: StockItem, length: float, width: float) -> bool:
    return (item.length >= length and item.width >= width) or (
        item.length >= width and item.width >= length
    )


def _area(item: StockItem) -> float:
    return item.length * item.width


def _best_fit(items: Iterable[StockItem], length: float, width: float) -> Optional[StockItem]:
    candidates = [item for item in items if _fits(item, length, width)]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (_area(item), min(item.length, item.width), item.part))
    return candidates[0]


def choose_item(
    catalog: Dict[str, Dict[float, List[StockItem]]],
    material: str,
    length: float,
    width: float,
    thickness: float,
) -> Optional[StockItem]:
    normalised_material = norm_material(material)
    by_thickness = catalog.get(normalised_material, {})
    best = _best_fit(by_thickness.get(thickness, []), length, width)
    if best:
        return best
    if ALLOW_NEXT_THICKER:
        for thicker in sorted(t for t in by_thickness.keys() if t > thickness):
            best = _best_fit(by_thickness[thicker], length, width)
            if best:
                return best
    return None


def _mm_to_inches(value_mm: float) -> float:
    return float(value_mm) / 25.4 if value_mm is not None else 0.0


def _best_unit_price_tier(tiers: List[dict], qty: int = 1) -> Optional[dict]:
    if not tiers:
        return None
    tiers_sorted = sorted(tiers, key=lambda tier: tier.get("MinimumQuantity", 10**9))
    eligible = [tier for tier in tiers_sorted if tier.get("MinimumQuantity", 10**9) <= qty]
    return eligible[-1] if eligible else tiers_sorted[0]


def _get_env() -> dict:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = load_env()
    return _ENV_CACHE


def _get_catalog() -> Dict[str, Dict[float, List[StockItem]]]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        env = _get_env()
        csv_path = env.get("CATALOG_CSV_PATH") or os.getenv("CATALOG_CSV_PATH")
        if not csv_path:
            csv_path = str(default_catalog_csv())
        _CATALOG_CACHE = load_catalog(csv_path)
    return _CATALOG_CACHE


def lookup_sku_and_price_for_mm(
    material: str,
    length_mm: float,
    width_mm: float,
    thickness_mm: float,
    qty: int = 1,
) -> (
    Tuple[str, Optional[float], Optional[str], Tuple[float, float, float]]
    | Tuple[None, None, None, Tuple[float, float, float]]
):
    """Resolve a McMaster plate SKU and pricing for the requested dimensions (mm)."""

    catalog = _get_catalog()
    length_in, width_in, thickness_in = map(
        _mm_to_inches, (length_mm, width_mm, thickness_mm)
    )
    item = choose_item(catalog, material, length_in, width_in, thickness_in)
    dims = (length_in, width_in, thickness_in)
    if not item:
        return None, None, None, dims

    cache_key = (item.part, int(qty))
    cached = _PRICE_CACHE.get(cache_key)
    if cached:
        price_each, uom = cached
        return item.part, price_each, uom, (item.length, item.width, item.thickness)

    env = _get_env()
    api = McMasterAPI(
        env["MCMASTER_USER"],
        env["MCMASTER_PASS"],
        env["MCMASTER_PFX_PATH"],
        env["MCMASTER_PFX_PASS"],
    )
    api.login()
    tiers = api.get_price_tiers(item.part)
    tier = _best_unit_price_tier(tiers, qty=qty)
    if not tier:
        return item.part, None, None, (item.length, item.width, item.thickness)

    price_each = float(tier["Amount"])
    uom = str(tier["UnitOfMeasure"])
    _PRICE_CACHE[cache_key] = (price_each, uom)
    return item.part, price_each, uom, (item.length, item.width, item.thickness)


def _get_env_or_prompt() -> tuple[str, str, str, str]:
    env = _get_env()
    return (
        env.get("MCMASTER_USER", ""),
        env.get("MCMASTER_PASS", ""),
        env.get("MCMASTER_PFX_PATH", ""),
        env.get("MCMASTER_PFX_PASS", ""),
    )


def main() -> None:
    """Simple CLI for testing catalog lookups and API integration."""

    csv_path = os.getenv("CATALOG_CSV_PATH") or str(default_catalog_csv())
    print(f"Using catalog CSV: {csv_path}")
    if not csv_path:
        print("No CSV provided.")
        return
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    try:
        catalog = load_catalog(csv_path)
    except Exception as exc:  # pragma: no cover - user interaction
        print(f"Error loading catalog: {exc}")
        return
    if not catalog:
        print("Catalog is empty after load.")
        return

    try:
        length = float(input('Length (in, numeric): ').strip().replace('"', ""))
        width = float(input('Width  (in, numeric): ').strip().replace('"', ""))
        thickness = float(input('Thick  (in, numeric): ').strip().replace('"', ""))
    except Exception:
        print("Please enter numeric inches for L/W/T (e.g., 12, 14, 2).")
        return
    material = input('Material (e.g., "aluminum mic6", "aluminum 5083", "tool steel a2"): ').strip()

    item = choose_item(catalog, material, length, width, thickness)
    if not item:
        print(f'No match found ≥ {length}×{width}×{thickness} in "{norm_material(material)}".')
        if not ALLOW_NEXT_THICKER:
            print("Tip: set ALLOW_NEXT_THICKER=True to allow thicker stock.")
        return

    print(
        "Chosen SKU: "
        f"{item.part}  ({item.length:.3f}\" × {item.width:.3f}\" × {item.thickness:.3f}\"  {item.material})"
    )

    user, password, pfx_path, pfx_password = _get_env_or_prompt()
    api = McMasterAPI(username=user, password=password, pfx_path=pfx_path, pfx_password=pfx_password)
    api.login()
    tiers = api.get_price_tiers(item.part)
    if not tiers:
        print("No price tiers returned.")
        return

    tier = _best_unit_price_tier(tiers, qty=1)
    if tier:
        amount = tier["Amount"]
        uom = tier["UnitOfMeasure"]
        print(f"Price @ qty=1: ${amount:.2f} {uom}")
    else:
        print("No eligible tier for qty=1.")

