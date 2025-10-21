"""Numeric helper utilities for the CAD Quoter project."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["coerce_float", "coerce_int", "coerce_positive_float"]


def coerce_float(value: Any) -> float | None:
    """Best-effort conversion to a finite ``float`` value."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            coerced = float(value)
        except Exception:  # pragma: no cover - defensive guard
            return None
        return coerced if math.isfinite(coerced) else None

    text = str(value).strip()
    if not text:
        return None

    try:
        coerced = float(text)
    except Exception:
        return None

    return coerced if math.isfinite(coerced) else None


def coerce_int(value: Any) -> int | None:
    """Best-effort conversion to an integer via :func:`coerce_float`."""

    number = coerce_float(value)
    if number is None:
        return None

    try:
        return int(round(number))
    except Exception:
        return None


def coerce_positive_float(value: Any) -> float | None:
    """Return *value* as a positive finite ``float`` when possible.

    ``None`` is returned when the input cannot be interpreted as a positive,
    finite floating point number.  This mirrors the tolerant behaviour used in
    several UI helpers where loose user input should not raise exceptions.
    """

    number = coerce_float(value)
    if number is None:
        return None

    return number if number > 0 else None

