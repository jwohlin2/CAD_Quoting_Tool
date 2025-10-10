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
