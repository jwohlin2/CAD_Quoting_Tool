"""Pricing utility wrappers."""

from __future__ import annotations

from appV5 import (
    BACKUP_CSV_NAME,
    LB_PER_KG,
    _price_value_to_per_gram as _price_value_to_per_gram,
    _usdkg_to_usdlb as _usdkg_to_usdlb,
    ensure_material_backup_csv as _ensure_material_backup_csv,
    load_backup_prices_csv as _load_backup_prices_csv,
    resolve_material_unit_price as _resolve_material_unit_price,
)

__all__ = [
    "BACKUP_CSV_NAME",
    "LB_PER_KG",
    "price_value_to_per_gram",
    "usdkg_to_usdlb",
    "ensure_material_backup_csv",
    "load_backup_prices_csv",
    "resolve_material_unit_price",
]


def price_value_to_per_gram(value: float, label: str):
    """Expose :func:`appV5._price_value_to_per_gram`."""

    return _price_value_to_per_gram(value, label)


def usdkg_to_usdlb(value: float) -> float:
    """Expose :func:`appV5._usdkg_to_usdlb`."""

    return _usdkg_to_usdlb(value)


def ensure_material_backup_csv(path: str | None = None) -> str:
    """Expose :func:`appV5.ensure_material_backup_csv`."""

    return _ensure_material_backup_csv(path)


def load_backup_prices_csv(path: str | None = None) -> dict:
    """Expose :func:`appV5.load_backup_prices_csv`."""

    return _load_backup_prices_csv(path)


def resolve_material_unit_price(display_name: str, unit: str = "kg") -> tuple[float, str]:
    """Expose :func:`appV5.resolve_material_unit_price`."""

    return _resolve_material_unit_price(display_name, unit)
