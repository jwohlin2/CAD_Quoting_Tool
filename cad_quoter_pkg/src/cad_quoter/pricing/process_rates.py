"""Canonical per-process rate defaults used by pricing helpers."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

_MACHINE_RATES = MappingProxyType(
    {
        "milling": 90.0,
        "drilling": 95.0,
    }
)

_LABOR_RATES = MappingProxyType(
    {
        "programming": 45.0,
        "milling": 45.0,
        "drilling": 90.0,
        "inspection": 90.0,
    }
)


MACHINE_RATES: Mapping[str, float] = _MACHINE_RATES
"""Read-only view of the default machine rates keyed by process."""

LABOR_RATES: Mapping[str, float] = _LABOR_RATES
"""Read-only view of the default labor rates keyed by process."""

RATES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "machine": MACHINE_RATES,
        "labor": LABOR_RATES,
    }
)
"""Two-bucket representation of the canonical process rates."""


def _lookup(kind: str, process: str) -> float | None:
    mapping = RATES.get(kind)
    if not isinstance(mapping, Mapping):
        return None

    value = mapping.get(process)
    if value is not None:
        return float(value)

    process_norm = str(process or "").strip().lower()
    if not process_norm:
        return None

    for key, amount in mapping.items():
        if key.lower() == process_norm:
            return float(amount)
    return None


def get_rate(kind: str, process: str, *, default: float | None = None) -> float:
    """Return the configured rate for ``process`` under ``kind``."""

    rate = _lookup(kind, process)
    if rate is not None:
        return rate
    if default is not None:
        return float(default)
    return 0.0


def machine_rate(process: str, *, default: float | None = None) -> float:
    """Return the configured machine rate for ``process``."""

    return get_rate("machine", process, default=default)


def labor_rate(process: str, *, default: float | None = None) -> float:
    """Return the configured labor rate for ``process``."""

    return get_rate("labor", process, default=default)


__all__ = [
    "RATES",
    "MACHINE_RATES",
    "LABOR_RATES",
    "get_rate",
    "machine_rate",
    "labor_rate",
]

