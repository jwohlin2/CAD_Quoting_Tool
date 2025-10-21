"""Domain-level value coercion helpers."""
from __future__ import annotations

import math
import re
from typing import Any


_UNIT_PATTERN = re.compile(
    r"(?i)\b(?:inches?|millimeters?|cm|mm|in)\b\.?")


def parse_mixed_fraction(value: str) -> float | None:
    """Parse strings like ``"1 1/2"`` or ``"3/4"`` into floats.

    The CSV stock catalog that feeds our quoting tooling represents many
    dimensions using imperial style measurements (for example ``1 1/2"`` or
    ``3/4"``).  Prior to this helper ``coerce_float_or_none`` rejected those
    values which meant downstream lookups silently lost legitimate catalog rows.
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

    # Replace separators that sometimes appear between the whole and fractional
    # components with spaces so that ``split`` can isolate each term.
    candidate = (
        candidate.replace("\u00A0", " ")  # non-breaking space
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    # Hyphenated values (e.g. ``1-1/2``) should be treated like spaces between
    # the whole number and the fraction.
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


_parse_mixed_fraction = parse_mixed_fraction


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
        # Remove common unit markers that appear in catalog CSVs.
        cleaned = (
            cleaned.replace("\u2033", "")  # double prime
            .replace("\u2032", "")  # prime
            .replace("\u201D", "")
            .replace("\u201C", "")
            .replace("\u2019", "")
            .replace("\u2018", "")
            .replace("\u00B0", "")
            .replace("\u0022", "")
            .replace("\u0027", "")
            .replace("\u00A0", " ")
        )
        cleaned = _UNIT_PATTERN.sub("", cleaned)
        cleaned = cleaned.strip()
        # Trim trailing punctuation left over from unit removal (e.g. ``"in."``)
        cleaned = cleaned.rstrip(". ")
        try:
            return float(cleaned)
        except Exception:
            parsed_fraction = parse_mixed_fraction(cleaned)
            if parsed_fraction is not None:
                return parsed_fraction
            return None
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except Exception:
            return None
    return None


def to_float(value: Any) -> float | None:
    """Best-effort conversion of ``value`` to a float."""

    if value is None:
        return None

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        coerced = coerce_float_or_none(candidate)
        return coerced

    try:
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to an integer via rounding."""

    numeric = coerce_float_or_none(value)
    if numeric is None:
        return None

    try:
        return int(round(numeric))
    except Exception:
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` coerced to ``float`` with NaN/Inf protection."""

    try:
        coerced = float(value or 0.0)
    except Exception:
        return default
    if not math.isfinite(coerced):
        return default
    return coerced


__all__ = [
    "parse_mixed_fraction",
    "coerce_float_or_none",
    "safe_float",
    "to_float",
    "to_int",
]
