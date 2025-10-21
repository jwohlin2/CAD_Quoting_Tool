"""Numeric helper utilities for the CAD Quoter project."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["coerce_positive_float"]


def coerce_positive_float(value: Any) -> float | None:
    """Return *value* as a positive finite ``float`` when possible.

    ``None`` is returned when the input cannot be interpreted as a positive,
    finite floating point number.  This mirrors the tolerant behaviour used in
    several UI helpers where loose user input should not raise exceptions.
    """

    try:
        number = float(value)
    except Exception:
        return None

    try:
        if not math.isfinite(number):
            return None
    except Exception:
        # ``math.isfinite`` can raise ``TypeError`` for duck-typed numbers.
        pass

    return number if number > 0 else None

