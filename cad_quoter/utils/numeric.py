"""Numeric helper utilities for the CAD Quoter project."""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "coerce_float",
    "coerce_int",
    "coerce_positive_float",
    "parse_mixed_fraction",
]


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

