"""Helpers for working with estimator-style variable sheets."""

from __future__ import annotations

from typing import Any, Callable, Sequence

try:  # pandas is an optional dependency in lightweight environments
    import pandas as pd
except Exception:  # pragma: no cover - pandas is required for full functionality
    pd = None  # type: ignore[assignment]

Matcher = Callable[[Any, str], Any]
PctFunc = Callable[[Any, float | None], float | None]
TimeAggregator = Callable[..., float]


def _as_bool_mask(mask: Any, fallback_length: int) -> Any:
    """Return *mask* coerced to a pandas ``Series`` when possible."""

    if pd is None:
        if isinstance(mask, Sequence):
            return mask
        try:
            return list(mask)
        except Exception:
            return [False] * max(fallback_length, 0)

    if isinstance(mask, pd.Series):
        try:
            return mask.astype(bool)
        except Exception:
            return pd.Series([False] * fallback_length, dtype="bool")

    try:
        return pd.Series(list(mask), dtype="bool")
    except Exception:
        return pd.Series([False] * fallback_length, dtype="bool")


def contains(items: Any, pattern: str, *, matcher: Matcher) -> Any:
    """Return a boolean mask for ``items`` matching ``pattern`` using *matcher*."""

    try:
        length = len(items)
    except Exception:
        length = 0
    result = matcher(items, pattern)
    return _as_bool_mask(result, length)


def _mask_any(mask: Any) -> bool:
    """Return ``True`` if *mask* has any truthy value."""

    any_func = getattr(mask, "any", None)
    if callable(any_func):
        try:
            return bool(any_func())
        except Exception:
            pass
    try:
        return any(bool(v) for v in mask)
    except Exception:
        return False


def first_num(
    items: Any,
    values: Any,
    pattern: str,
    *,
    matcher: Matcher,
    default: float = 0.0,
) -> float:
    """Return the first numeric value for ``pattern`` from ``values``."""

    if pd is None:
        return float(default)
    mask = contains(items, pattern, matcher=matcher)
    if not _mask_any(mask):
        return float(default)
    try:
        series = pd.to_numeric(values[mask].iloc[0], errors="coerce")
    except Exception:
        return float(default)
    return float(series) if pd.notna(series) else float(default)


def num(
    items: Any,
    values: Any,
    pattern: str,
    *,
    matcher: Matcher,
    default: float = 0.0,
) -> float:
    """Return the numeric sum of rows matching ``pattern`` from ``values``."""

    if pd is None:
        return float(default)
    mask = contains(items, pattern, matcher=matcher)
    if not _mask_any(mask):
        return float(default)
    try:
        series = pd.to_numeric(values[mask], errors="coerce").fillna(0.0)
    except Exception:
        return float(default)
    try:
        return float(series.sum())
    except Exception:
        return float(default)


def strv(
    items: Any,
    values: Any,
    pattern: str,
    *,
    matcher: Matcher,
    default: str = "",
) -> str:
    """Return the first string value for ``pattern`` from ``values``."""

    mask = contains(items, pattern, matcher=matcher)
    if not _mask_any(mask):
        return default
    try:
        return str(values[mask].iloc[0])
    except Exception:
        return default


def num_pct(
    items: Any,
    values: Any,
    pattern: str,
    *,
    matcher: Matcher,
    pct_func: PctFunc,
    default: float = 0.0,
) -> float | None:
    """Return the percentage value derived from ``pattern`` using ``pct_func``."""

    base = first_num(
        items,
        values,
        pattern,
        matcher=matcher,
        default=default * 100.0,
    )
    return pct_func(base, default)


def sum_time(
    items: Any,
    values: Any,
    types: Any,
    pattern: str,
    *,
    matcher: Matcher,
    sum_time_func: TimeAggregator,
    default: float = 0.0,
    exclude_pattern: str | None = None,
) -> float:
    """Sum only time-like rows, converting minutes to hours when required."""

    mask = contains(items, pattern, matcher=matcher)
    exclude_mask = (
        contains(items, exclude_pattern, matcher=matcher)
        if exclude_pattern
        else None
    )
    try:
        return float(
            sum_time_func(
                items,
                values,
                types,
                mask,
                default=float(default),
                exclude_mask=exclude_mask,
            )
        )
    except Exception:
        return float(default)


def sheet_num(
    items: Any,
    values: Any,
    pattern: str,
    *,
    matcher: Matcher,
    default: float | None = None,
) -> float | None:
    """Return an optional numeric override from the sheet."""

    value = first_num(items, values, pattern, matcher=matcher, default=float("nan"))
    return value if value == value else default


def sheet_pct(
    items: Any,
    values: Any,
    pattern: str,
    *,
    matcher: Matcher,
    pct_func: PctFunc,
    default: float | None = None,
) -> float | None:
    """Return an optional percentage override from the sheet."""

    value = first_num(items, values, pattern, matcher=matcher, default=float("nan"))
    if value == value:  # NaN check
        pct_value = value / 100.0 if abs(value) > 1.0 else value
        return pct_value
    if default is None:
        return None
    return float(default)


def sheet_text(
    items: Any,
    values: Any,
    pattern: str,
    *,
    matcher: Matcher,
    default: str | None = None,
) -> str | None:
    """Return an optional text override from the sheet."""

    text = strv(items, values, pattern, matcher=matcher, default="").strip()
    return text if text else default


__all__ = [
    "contains",
    "first_num",
    "num",
    "strv",
    "num_pct",
    "sum_time",
    "sheet_num",
    "sheet_pct",
    "sheet_text",
]
