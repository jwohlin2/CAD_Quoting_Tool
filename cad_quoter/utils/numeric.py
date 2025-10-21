"""Numeric helper utilities for the CAD Quoter project."""

from __future__ import annotations

import math
import re
from typing import Any

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


def parse_mixed_fraction(value: str) -> float | None:
    """Parse strings like ``"1 1/2"`` or ``"3/4"`` into floats.

    Mixed-fraction strings appear in catalogue CSVs and in user input when
    describing imperial dimensions (e.g. ``"1-1/2"``).  The parser accepts
    optional leading signs, hyphen or space separated whole numbers, and
    fractional components.
    """

    candidate = value.strip()
    if not candidate:
        return None

    sign = 1.0
    if candidate[0] in {"+", "-"}:
        if candidate[0] == "-":
            sign = -1.0
        candidate = candidate[1:].strip()
        if not candidate:
            return None

    candidate = (
        candidate.replace("\u00A0", " ")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    candidate = re.sub(r"(?<=\d)-(?!\d)", " ", candidate)
    candidate = candidate.replace("-", " ")

    total = 0.0
    seen_component = False
    for part in candidate.split():
        if not part:
            continue
        seen_component = True
        if "/" in part:
            num, _, denom = part.partition("/")
            try:
                total += float(num.strip()) / float(denom.strip())
            except Exception:
                return None
        else:
            try:
                total += float(part)
            except Exception:
                return None

    if not seen_component:
        return None

    return sign * total

def _coerce_float_or_none(value: Any) -> float | None:
    from cad_quoter.domain_models.values import coerce_float_or_none as _impl

    return _impl(value)


def _to_int(value: Any) -> int | None:
    from cad_quoter.domain_models.values import to_int as _impl

    return _impl(value)


def _to_positive_float(value: Any) -> float | None:
    from cad_quoter.domain_models.values import to_positive_float as _impl

    return _impl(value)
