"""Material pricing helpers shared between UI and pricing layers."""
from __future__ import annotations

from pathlib import Path
import math
import os
import traceback
from typing import Dict

try:  # Optional dependency: McMaster catalog helpers
    from cad_quoter.vendors.mcmaster_stock import lookup_sku_and_price_for_mm
except Exception:  # pragma: no cover - optional dependency / environment specific
    lookup_sku_and_price_for_mm = None  # type: ignore[assignment]

from cad_quoter.domain_models import (
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    coerce_float_or_none,
    normalize_material_key,
)

LB_PER_KG = 2.2046226218
BACKUP_CSV_NAME = "material_price_backup.csv"

_MCM_DISABLE_ENV = False  # never disable via env; always attempt McMaster
_MCM_DISABLED = False  # do not latch errors; evaluate per-call
_MCM_CACHE: Dict[str, Dict[str, float | str]] = {}

CM3_PER_IN3 = 16.387064
MM_PER_INCH = 25.4

_STANDARD_MCM_REQUESTS: dict[str, tuple[str, float, float, float]] = {
    # material key, length_in, width_in, thickness_in
    "aluminum": ("aluminum 5083", 12.0, 12.0, 0.25),
    "tool_steel": ("tool steel a2", 12.0, 12.0, 0.25),
}

_FAMILY_DENSITY_FALLBACK = {
    "aluminum": 2.70,
    "tool_steel": 7.85,
    "carbide": 15.60,
}


def _density_for_material(normalized_key: str, family: str) -> float | None:
    if normalized_key:
        density = MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(normalized_key)
        if density:
            return float(density)
        for token in normalized_key.split():
            density = MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(token)
            if density:
                return float(density)
    return _FAMILY_DENSITY_FALLBACK.get(family)


def _mcmaster_family_from_key(key: str) -> str:
    if not key:
        return ""
    if "carbide" in key:
        return "carbide"
    if "tool" in key and "steel" in key:
        return "tool_steel"
    if any(token in key for token in ("alum", "6061", "7075", "2024", "5083", "mic6")):
        return "aluminum"
    return ""


def _mcm_log(message: str) -> None:
    try:
        with open("mcmaster_debug.log", "a", encoding="ascii", errors="replace") as _dbg:
            _dbg.write(message.rstrip() + "\n")
    except Exception:
        pass


def _maybe_get_mcmaster_price(display_name: str, normalized_key: str) -> Dict[str, float | str] | None:
    # Always attempt live McMaster pricing for supported families.

    family = _mcmaster_family_from_key(normalized_key)
    if not family:
        return None

    cached = _MCM_CACHE.get(family)
    if cached:
        return cached

    request = _STANDARD_MCM_REQUESTS.get(family)
    if not request:
        return None

    # Ensure the local requests shim proxies to the real package (required by requests-pkcs12)
    os.environ.setdefault("CAD_QUOTER_ALLOW_REQUESTS", "1")

    if lookup_sku_and_price_for_mm is None:
        _mcm_log("[MCM] McMaster catalog helpers unavailable; skipping live price fetch.")
        return None

    try:
        material_key, length_in, width_in, thickness_in = request
        dims_mm = tuple(val * MM_PER_INCH for val in (length_in, width_in, thickness_in))
        sku, price_each, uom, dims_in = lookup_sku_and_price_for_mm(
            material_key,
            dims_mm[0],
            dims_mm[1],
            dims_mm[2],
            qty=1,
        )
    except Exception as exc:
        _mcm_log("[MCM] lookup_sku_and_price_for_mm error: " + repr(exc))
        _mcm_log(traceback.format_exc())
        return None

    if not sku or not price_each or not dims_in:
        return None

    try:
        length_in, width_in, thickness_in = map(float, dims_in)
    except Exception:
        return None

    volume_in3 = length_in * width_in * thickness_in
    if volume_in3 <= 0:
        return None

    density_g_cc = _density_for_material(normalized_key, family)
    if not density_g_cc or density_g_cc <= 0:
        return None

    mass_kg = (volume_in3 * CM3_PER_IN3 * density_g_cc) / 1000.0
    if mass_kg <= 0:
        return None

    usd_per_kg = float(price_each) / mass_kg
    usd_per_lb = usd_per_kg / LB_PER_KG

    record: Dict[str, float | str] = {
        "usd_per_kg": float(usd_per_kg),
        "usd_per_lb": float(usd_per_lb),
        "source": f"mcmaster_api:{sku}",
        "part_number": sku,
        "unit_price": float(price_each),
        "unit_uom": str(uom or "Each"),
    }

    if sku:
        record["mcmaster_part"] = str(sku)
    if price_each not in (None, ""):
        record["price$"] = float(price_each)

    _MCM_CACHE[family] = record
    _mcm_log(
        f"[MCM] Success: {display_name!r} -> sku={sku}, ${price_each}/each, dims_in={dims_in}, usd_per_lb={usd_per_lb:.4f}"
    )
    return record


def get_mcmaster_unit_price(display_name: str, *, unit: str = "kg") -> tuple[float | None, str]:
    """Return a McMaster price for ``display_name`` when the scraper supports it."""

    key = normalize_material_key(display_name)
    record = _maybe_get_mcmaster_price(display_name, key)
    if not record:
        return None, ""

    source = str(record.get("source", "mcmaster"))
    if unit == "lb":
        return float(record["usd_per_lb"]), source
    # default to kg for any unexpected units
    return float(record["usd_per_kg"]), source


def price_value_to_per_gram(value: float, label: str) -> float | None:
    """Normalise user-entered price labels to USD per gram when possible."""

    try:
        base = float(value)
    except Exception:
        return None
    label_lower = str(label or "").lower()
    label_compact = label_lower.replace(" ", "")

    def _has_any(*patterns: str) -> bool:
        return any(p in label_lower or p in label_compact for p in patterns)

    if _has_any("perg", "/g", "$/g", "per gram"):
        return base
    if _has_any("perkg", "/kg", "$/kg", "per kilogram"):
        return base / 1000.0
    if _has_any("perlb", "/lb", "$/lb", "per pound", "/lbs", "$/lbs"):
        return base / 453.59237
    if _has_any("peroz", "/oz", "$/oz", "per ounce"):
        return base / 28.349523125
    return None


def usdkg_to_usdlb(value: float) -> float:
    """Convert a price expressed per-kilogram to per-pound."""

    return float(value) / LB_PER_KG if value is not None else value


def ensure_material_backup_csv(path: str | None = None) -> str:
    """Create a small CSV with dummy prices if it does not already exist."""

    destination = Path(path) if path else Path(__file__).resolve().parents[2] / BACKUP_CSV_NAME
    if destination.exists():
        return str(destination)

    rows = [
        ("steel", 2.20, "", "dummy base"),
        ("stainless steel", 4.00, "", "dummy base"),
        ("aluminum", 2.80, "", "dummy base"),
        ("copper", 9.50, "", "dummy base"),
        ("brass", 7.80, "", "dummy base"),
        ("titanium", 17.00, "", "dummy base"),
    ]

    lines = ["material_key,usd_per_kg,usd_per_lb,notes\n"]
    for material, usdkg, usdlb, notes in rows:
        if usdlb in ("", None):
            usdlb = usdkg / LB_PER_KG
        lines.append(f"{material},{float(usdkg):.6f},{float(usdlb):.6f},{notes}\n")

    destination.write_text("".join(lines), encoding="utf-8")
    return str(destination)


def load_backup_prices_csv(path: str | None = None) -> Dict[str, Dict[str, float | str]]:
    """Load the packaged CSV of fallback material prices."""

    import pandas as pd

    table_path = Path(path) if path else Path(ensure_material_backup_csv())
    df = pd.read_csv(table_path)
    out: Dict[str, Dict[str, float | str]] = {}
    for _, record in df.iterrows():
        key = normalize_material_key(record["material_key"])
        usdkg = coerce_float_or_none(record.get("usd_per_kg"))
        usdlb = coerce_float_or_none(record.get("usd_per_lb"))
        if usdkg and not usdlb:
            usdlb = usdkg_to_usdlb(usdkg)
        if usdlb and not usdkg:
            usdkg = float(usdlb) * LB_PER_KG
        if usdkg and usdlb:
            out[key] = {
                "usd_per_kg": float(usdkg),
                "usd_per_lb": float(usdlb),
                "notes": str(record.get("notes", "")),
            }
    return out


def resolve_material_unit_price(display_name: str, unit: str = "kg") -> tuple[float, str]:
    """Resolve a material price in the requested unit with layered fallbacks."""

    key = normalize_material_key(display_name)

    mcm_record = _maybe_get_mcmaster_price(display_name, key)
    if mcm_record:
        source = str(mcm_record.get("source", "mcmaster"))
        if unit == "lb":
            return float(mcm_record["usd_per_lb"]), source
        # default to kg when unspecified or unexpected units
        return float(mcm_record["usd_per_kg"]), source

    try:
        from .wieland_scraper import get_live_material_price  # type: ignore

        price, source = get_live_material_price(display_name, unit=unit, fallback_usd_per_kg=float("nan"))
        if price is not None and math.isfinite(float(price)):
            return float(price), source
    except Exception as exc:  # pragma: no cover - network/optional dependency
        source = f"wieland_error:{type(exc).__name__}"
    else:
        source = ""

    def _load_prices(path: str | None = None) -> Dict[str, Dict[str, float | str]]:
        from cad_quoter import pricing as pricing_pkg

        package_loader = getattr(pricing_pkg, "load_backup_prices_csv", load_backup_prices_csv)
        if package_loader is load_backup_prices_csv:
            return load_backup_prices_csv(path)
        return package_loader(path)

    try:
        table = _load_prices()
        record = table.get(key)
        if not record:
            if "stainless" in key:
                record = table.get("stainless steel")
            elif "steel" in key:
                record = table.get("steel")
            elif "alum" in key:
                record = table.get("aluminum")
            elif "copper" in key or "cu" in key:
                record = table.get("copper")
            elif "brass" in key:
                record = table.get("brass")
        if record:
            price = record["usd_per_kg"] if unit == "kg" else record["usd_per_lb"]
            return float(price), f"backup_csv:{BACKUP_CSV_NAME}"
    except Exception:  # pragma: no cover - pandas optional path
        pass

    fallback = 2.20 if unit == "kg" else (2.20 / LB_PER_KG)
    return fallback, "hardcoded_default"


__all__ = [
    "BACKUP_CSV_NAME",
    "LB_PER_KG",
    "get_mcmaster_unit_price",
    "ensure_material_backup_csv",
    "load_backup_prices_csv",
    "price_value_to_per_gram",
    "resolve_material_unit_price",
    "usdkg_to_usdlb",
]
