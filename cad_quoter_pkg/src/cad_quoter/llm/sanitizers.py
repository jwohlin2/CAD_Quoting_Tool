"""Shared sanitizers for LLM-derived override payloads."""
from __future__ import annotations

import math
from collections.abc import Mapping as _MappingABC, Sequence
from typing import Any, Callable

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.utils import coerce_bool as _coerce_bool

LLM_MULTIPLIER_MIN = 0.25
LLM_MULTIPLIER_MAX = 4.0
LLM_ADDER_MAX = 8.0


def clamp(x: Any, lo: float, hi: float, default: float | None = None) -> float | None:
    """Clamp *x* to the inclusive range [*lo*, *hi*]."""

    try:
        value = float(x)
    except Exception:
        return default if default is not None else lo
    return max(lo, min(hi, value))


def as_float(value: Any) -> float | None:
    """Best-effort conversion of ``value`` into a float."""

    result = _coerce_float_or_none(value)
    return float(result) if result is not None else None


def as_int(
    value: Any,
    *,
    default: int = 0,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return ``value`` coerced to an integer bounded by ``minimum``/``maximum``."""

    result = as_float(value)
    if result is None:
        return default
    try:
        rounded = int(round(result))
    except Exception:
        return default
    if minimum is not None:
        rounded = max(minimum, rounded)
    if maximum is not None:
        rounded = min(maximum, rounded)
    return rounded


def clean_string(value: Any) -> str | None:
    """Return a stripped string or ``None`` when empty."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clean_string_list(value: Any, *, limit: int | None = None) -> list[str]:
    """Return a list of cleaned, non-empty strings."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        iterable = value
    elif value is None:
        iterable = ()
    else:
        iterable = (value,)

    cleaned: list[str] = []
    for item in iterable:
        text = clean_string(item)
        if not text:
            continue
        cleaned.append(text)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def clean_notes_list(value: Any, *, limit: int = 6, max_length: int = 200) -> list[str]:
    """Return a cleaned list of note strings capped by ``limit`` and ``max_length``."""

    notes = clean_string_list(value, limit=limit)
    return [note[:max_length] for note in notes]


def coerce_bool_flag(value: Any) -> bool | None:
    """Coerce ``value`` into a boolean flag, returning ``None`` when unknown."""

    result = _coerce_bool(value)
    if result is None:
        return None
    return bool(result)


def sanitize_drilling_groups(
    groups: Any,
    *,
    qty_limit: int | None = None,
    note_limit: int = 6,
) -> list[dict[str, Any]]:
    """Return cleaned drilling group definitions."""

    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        return []

    cleaned_groups: list[dict[str, Any]] = []
    for entry in groups:
        if not isinstance(entry, _MappingABC):
            continue
        qty = as_int(entry.get("qty") or entry.get("count"), default=0, minimum=0)
        dia = as_float(entry.get("dia_mm") or entry.get("diameter_mm"))
        if qty <= 0 or dia is None or dia <= 0:
            continue
        depth = as_float(entry.get("depth_mm") or entry.get("depth"))
        peck = clean_string(entry.get("peck") or entry.get("strategy"))
        notes = clean_notes_list(entry.get("notes"), limit=note_limit)

        if qty_limit is not None and qty_limit > 0:
            qty = max(1, min(qty_limit, qty))

        cleaned_entry: dict[str, Any] = {"qty": qty, "dia_mm": round(float(dia), 3)}
        if depth is not None and depth > 0:
            cleaned_entry["depth_mm"] = round(float(depth), 3)
        if peck:
            cleaned_entry["strategy"] = peck[:120]
        if notes:
            cleaned_entry["notes"] = notes
        cleaned_groups.append(cleaned_entry)

    return cleaned_groups


def merge_multiplier(
    container: dict[str, float],
    name: str,
    value: Any,
    *,
    min_bound: float,
    max_bound: float,
    clamp_notes: list[str] | None = None,
    source: str | None = None,
    default: float = 1.0,
) -> dict[str, float]:
    """Merge a multiplier value into ``container`` while respecting bounds."""

    label = clean_string(name)
    val = as_float(value)
    if label is None or val is None:
        return container

    norm = label.lower()
    clamped = clamp(val, min_bound, max_bound, default)
    prev = container.get(norm)
    if prev is None:
        container[norm] = clamped
        return container

    merged_raw = float(prev) * float(clamped)
    merged = clamp(merged_raw, min_bound, max_bound, default)
    if clamp_notes is not None and not math.isclose(merged_raw, merged, abs_tol=1e-6):
        clamp_notes.append(f"{source or norm} multiplier clipped for {norm}")
    container[norm] = merged
    return container


def merge_adder(
    container: dict[str, float],
    name: str,
    value: Any,
    *,
    min_bound: float,
    max_bound: float,
    clamp_notes: list[str] | None = None,
    source: str | None = None,
    limit: float | None = None,
    format_total: Callable[[float], str] | None = None,
    format_limit: Callable[[float], str] | None = None,
) -> dict[str, float]:
    """Merge an adder value into ``container`` while respecting bounds."""

    label = clean_string(name)
    val = as_float(value)
    if label is None or val is None:
        return container

    norm = label.lower()
    cap = max_bound if limit is None else float(limit)
    clamped = clamp(val, min_bound, cap, min_bound)
    if clamped <= 0:
        return container

    prev = float(container.get(norm, 0.0))
    merged_raw = prev + float(clamped)
    merged = clamp(merged_raw, min_bound, cap, min_bound)
    if clamp_notes is not None and not math.isclose(merged_raw, merged, abs_tol=1e-6):
        total_label = format_total(merged_raw) if format_total else f"{merged_raw:.2f}"
        limit_label = format_limit(cap) if format_limit else f"{cap:.2f}"
        clamp_notes.append(
            f"{source or norm} {total_label} clipped to {limit_label} for {norm}"
        )
    container[norm] = merged
    return container
