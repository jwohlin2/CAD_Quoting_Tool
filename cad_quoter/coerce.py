"""Utility helpers for tolerant numeric coercion."""
from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float | None:
    """Best-effort conversion of ``value`` to a float."""

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    try:
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to an integer via rounding."""

    numeric = to_float(value)
    if numeric is None:
        return None

    try:
        return int(round(numeric))
    except Exception:
        return None


def coerce_float_or_none(value: Any) -> float | None:
    """Compatibility wrapper mirroring :func:`cad_quoter.domain_models.values.coerce_float_or_none`."""

    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except Exception:
            return None
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except Exception:
            return None
    return None
