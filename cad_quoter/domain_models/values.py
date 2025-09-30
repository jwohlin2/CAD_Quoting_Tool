"""Domain-level value coercion helpers."""
from __future__ import annotations

from typing import Any


def coerce_float_or_none(value: Any) -> float | None:
    """Attempt to coerce the given value to ``float`` returning ``None`` on failure."""

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


__all__ = ["coerce_float_or_none"]
