"""Helpers for parsing numeric inch values from free-form text."""

from __future__ import annotations

import re

__all__ = [
    "NUM_DEC_RE",
    "NUM_FRAC_RE",
    "VALUE_PATTERN",
    "_to_inch",
    "first_inch_value",
]

NUM_DEC_RE = re.compile(r"(?<!\d)(?:\d+\.\d+|\.\d+|\d+)(?!\d)")
NUM_FRAC_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")
_NUM_FRAC_PATTERN = r"(?<!\d)(?:\d+\s*/\s*\d+)(?!\d)"
VALUE_PATTERN = rf"(?:{_NUM_FRAC_PATTERN}|{NUM_DEC_RE.pattern})"


def _to_inch(num_text: str) -> float | None:
    """Convert a numeric token to an inch float, normalising leading dots."""

    s = (num_text or "").strip()
    if not s:
        return None
    if "/" in s:
        try:
            from fractions import Fraction

            return float(Fraction(s))
        except Exception:
            return None
    if s.startswith("."):
        s = "0" + s
    try:
        return float(s)
    except Exception:
        return None


def first_inch_value(text: str | None) -> float | None:
    """Return the first recognised inch value from text, if any."""

    if not text:
        return None
    frac = NUM_FRAC_RE.search(text)
    if frac:
        return _to_inch(f"{frac.group(1)}/{frac.group(2)}")
    dec = NUM_DEC_RE.search(text)
    if dec:
        return _to_inch(dec.group(0))
    return None

