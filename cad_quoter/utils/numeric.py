"""Numeric helper utilities for the CAD Quoter project."""

from __future__ import annotations

import math
from typing import Any

from cad_quoter.domain_models.values import (
    coerce_float_or_none as _coerce_float_or_none,
    to_int as _to_int,
    to_positive_float as _to_positive_float,
)

__all__ = ["coerce_float", "coerce_int", "coerce_positive_float"]


def coerce_float(value: Any) -> float | None:
    """Best-effort conversion to a finite ``float`` value."""

    coerced = _coerce_float_or_none(value)
    if coerced is None:
        return None

    return coerced if math.isfinite(coerced) else None


def coerce_int(value: Any) -> int | None:
    """Best-effort conversion to an integer via domain value helpers."""

    return _to_int(value)


def coerce_positive_float(value: Any) -> float | None:
    """Return *value* as a positive finite ``float`` when possible."""

    return _to_positive_float(value)

