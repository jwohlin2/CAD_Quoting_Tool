"""Utilities for parsing diameter tokens from free-form text."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from typing import Iterator

__all__ = [
    "iter_diameter_tokens",
    "first_diameter_token",
    "parse_diameter_value",
    "register_diameter_normalization",
    "capture_diameter_normalizations",
    "drain_diameter_normalizations",
]


_PREFIX_PATTERN = re.compile(r"(Ø|⌀|\u00D8|\bDIA\b\.?)\s*$", re.IGNORECASE)
_SUFFIX_PATTERN = re.compile(r"^\s*(Ø|⌀|\u00D8|\bDIA\b\.?)", re.IGNORECASE)
_TRAILING_ALPHA_RE = re.compile(r'(?i)[A-Z]+$')
_TRAILING_QUOTES_RE = re.compile(r"[\"']+$")
_CANDIDATE_NUMBER_RE = re.compile(r"\d+\s*/\s*\d+|\d*\.\d+|\.\d+|\d+")


_CAPTURE_STACK: ContextVar[set[tuple[str, float]] | None] = ContextVar(
    "_dia_capture",
    default=None,
)
_GLOBAL_EVENTS: set[tuple[str, float]] = set()


def register_diameter_normalization(raw: str | None, value: float | None) -> None:
    """Register a ``raw`` diameter token normalized to ``value`` for debugging."""

    if raw is None or value is None:
        return
    text = str(raw).strip()
    if not text:
        return
    try:
        val = float(value)
    except Exception:
        return
    if not val > 0:
        return
    event = (text, float(val))
    _GLOBAL_EVENTS.add(event)
    sink = _CAPTURE_STACK.get()
    if sink is not None:
        sink.add(event)


@contextmanager
def capture_diameter_normalizations() -> Iterator[set[tuple[str, float]]]:
    """Capture diameter normalizations performed within the context."""

    sink: set[tuple[str, float]] = set()
    token = _CAPTURE_STACK.set(sink)
    try:
        yield sink
    finally:
        _CAPTURE_STACK.reset(token)


def drain_diameter_normalizations() -> list[tuple[str, float]]:
    """Return and clear accumulated global diameter normalizations."""

    if not _GLOBAL_EVENTS:
        return []
    events = sorted(_GLOBAL_EVENTS)
    _GLOBAL_EVENTS.clear()
    return events


def _coerce_number(token: str) -> float | None:
    """Return ``token`` coerced to ``float`` when possible."""

    if not token:
        return None
    cleaned = re.sub(r"\s+", "", token)
    if cleaned.startswith("."):
        cleaned = f"0{cleaned}"
    try:
        if "/" in cleaned:
            return float(Fraction(cleaned))
        return float(cleaned)
    except Exception:
        return None


def iter_diameter_tokens(text: str | None) -> Iterator[tuple[str, float | None]]:
    """Yield ``(raw, value)`` pairs for diameter candidates found in *text*."""

    if not text:
        return
    candidate = str(text)
    for match in _CANDIDATE_NUMBER_RE.finditer(candidate):
        token = match.group(0)
        start, end = match.span()

        before = candidate[:start]
        after = candidate[end:]
        prefix_match = _PREFIX_PATTERN.search(before)
        suffix_match = _SUFFIX_PATTERN.match(after)

        has_decimal = "." in token
        has_fraction = "/" in token
        has_prefix = bool(prefix_match)
        has_suffix = bool(suffix_match)

        # Skip plain integers unless they are annotated with a symbol.
        if not (has_decimal or has_fraction or has_prefix or has_suffix):
            continue

        if has_fraction and not (has_prefix or has_suffix):
            tail = after.lstrip()
            if tail.startswith("-") and re.match(r"-\s*\d", tail):
                # Thread callouts like 1/4-20.
                continue

        raw_start = prefix_match.start() if prefix_match else start
        raw_end = end + (suffix_match.end() if suffix_match else 0)
        raw = candidate[raw_start:raw_end].strip()
        raw = _TRAILING_QUOTES_RE.sub("", raw).strip()
        raw = _TRAILING_ALPHA_RE.sub("", raw).strip()

        value = _coerce_number(token)
        if value is None or not value > 0:
            yield (raw, None)
            continue

        register_diameter_normalization(raw or token, float(value))
        yield (raw, float(value))


def first_diameter_token(text: str | None) -> tuple[str | None, float | None]:
    """Return the first parsed diameter token found in *text*."""

    for raw, value in iter_diameter_tokens(text):
        if value is not None:
            return raw, value
    return (None, None)


def parse_diameter_value(text: str | None) -> float | None:
    """Return the first positive diameter parsed from *text* or ``None``."""

    _, value = first_diameter_token(text)
    return value

