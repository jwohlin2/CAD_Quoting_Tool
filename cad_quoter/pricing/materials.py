"""Material pricing helpers shared between UI and pricing layers."""
from __future__ import annotations

from pathlib import Path
import math
from typing import Dict

from cad_quoter.domain_models import coerce_float_or_none, normalize_material_key

LB_PER_KG = 2.2046226218
BACKUP_CSV_NAME = "material_price_backup.csv"


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

    try:
        from wieland_scraper import get_live_material_price  # type: ignore

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
    "ensure_material_backup_csv",
    "load_backup_prices_csv",
    "price_value_to_per_gram",
    "resolve_material_unit_price",
    "usdkg_to_usdlb",
]
