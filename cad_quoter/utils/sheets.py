"""Utilities for processing worksheet-style inputs and related text helpers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

from cad_quoter.resources.loading import load_json

try:  # pragma: no cover - pandas is optional in some environments
    import pandas as pd
except Exception:  # pragma: no cover - pandas is required for full functionality
    pd = None  # type: ignore[assignment]

Matcher = Callable[[Any, str], Any]
PctFunc = Callable[[Any, float | None], float | None]
TimeAggregator = Callable[..., float]

TIME_RE = r"\b(?:hours?|hrs?|hr|time|min(?:ute)?s?)\b"
MONEY_RE = r"(?:rate|/hr|per\s*hour|per\s*hr|price|cost|\$)"


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------


def to_noncapturing(expr: str) -> str:
    """Convert every capturing ``(`` to a non-capturing ``(?:``."""

    out: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        prev = expr[i - 1] if i > 0 else ""
        nxt = expr[i + 1] if i + 1 < len(expr) else ""
        if ch == "(" and prev != "\\" and nxt != "?":
            out.append("(?:")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_to_noncapturing = to_noncapturing


def _match_items_contains(items: "pd.Series", pattern: str) -> "pd.Series":
    """Case-insensitive regex match over Items, with safe fallback."""

    pat = to_noncapturing(pattern)
    try:
        return items.str.contains(pat, case=False, regex=True, na=False)
    except Exception:
        return items.str.contains(re.escape(pattern), case=False, regex=True, na=False)


# ---------------------------------------------------------------------------
# Worksheet helpers
# ---------------------------------------------------------------------------


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


def sum_time_from_series(
    items: Any,
    values: Any,
    data_types: Any,
    mask: Any,
    *,
    default: float = 0.0,
    exclude_mask: Any | None = None,
) -> float:
    """Aggregate hour totals from worksheet-like series, minutes-aware."""

    if pd is None:
        raise RuntimeError("pandas required (install pandas) to aggregate time values")

    try:
        if not mask.any():
            return float(default)
    except Exception:
        return float(default)

    looks_time = items.str.contains(TIME_RE, case=False, regex=True, na=False)
    looks_money = items.str.contains(MONEY_RE, case=False, regex=True, na=False)
    typed_money = data_types.str.contains(r"(?:rate|currency|price|cost)", case=False, na=False)

    excl = looks_money | typed_money
    if exclude_mask is not None:
        try:
            excl = excl | exclude_mask
        except Exception:
            pass

    matched = mask & ~excl & looks_time
    try:
        if not matched.any():
            return float(default)
    except Exception:
        return float(default)

    numeric_candidates = pd.to_numeric(values[matched], errors="coerce")
    mask_numeric = pd.notna(numeric_candidates)
    try:
        has_numeric = mask_numeric.any()
    except Exception:
        has_numeric = any(bool(flag) for flag in mask_numeric)
    if not has_numeric:
        mins_mask = items.str.contains(r"\bmin(?:ute)?s?\b", case=False, regex=True, na=False) & matched
        hrs_mask = matched & ~mins_mask
        hrs_sum = pd.to_numeric(values[hrs_mask], errors="coerce").fillna(0.0).sum()
        mins_sum = pd.to_numeric(values[mins_mask], errors="coerce").fillna(0.0).sum()
        return float(hrs_sum) + float(mins_sum) / 60.0

    numeric_hours = numeric_candidates.fillna(0.0).sum()
    return float(numeric_hours)


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


# ---------------------------------------------------------------------------
# Rule-driven text helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_amortized_rules() -> Mapping[str, Any]:
    return load_json("amortized_label_rules.json")


@lru_cache(maxsize=None)
def amortized_label_pattern() -> re.Pattern[str]:
    """Return the compiled amortized label pattern."""

    rules = _load_amortized_rules()
    pattern_text = str(rules.get("pattern") or "")
    if not pattern_text:
        raise ValueError("amortized_label_rules.json must define a 'pattern' entry")
    flags = 0
    for flag_name in rules.get("pattern_flags", ["IGNORECASE"]):
        flag = getattr(re, str(flag_name), None)
        if flag is None:
            raise ValueError(f"Unsupported regex flag '{flag_name}' in amortized_label_rules.json")
        flags |= flag
    return re.compile(pattern_text, flags)


def _token_sets_contain(token_set: set[str], groups: Iterable[Iterable[str]]) -> bool:
    for group in groups:
        if all(token in token_set for token in group):
            return True
    return False


def canonicalize_amortized_label(label: Any) -> tuple[str, bool]:
    """Return a canonical label and flag for amortized cost rows."""

    text = str(label or "").strip()
    if not text:
        return "", False

    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    normalized = normalized.replace("perpart", "per part")
    normalized = normalized.replace("perpiece", "per piece")
    tokens = normalized.split()
    token_set = set(tokens)

    rules = _load_amortized_rules()
    amortized_tokens = set(map(str, rules.get("amortized_tokens", [])))
    has_amortized = bool(amortized_tokens & token_set)

    if has_amortized:
        per_part_tokens = [
            tuple(map(str, group))
            for group in rules.get("per_part_token_sets", [])
        ]
        per_part_phrases = [str(p) for p in rules.get("per_part_phrases", [])]
        per_part = _token_sets_contain(token_set, per_part_tokens) or any(
            phrase in normalized for phrase in per_part_phrases
        )

        for entry in rules.get("canonical_labels", []):
            tokens_any = {str(token) for token in entry.get("tokens_any", [])}
            if tokens_any & token_set:
                if per_part and entry.get("per_part"):
                    return str(entry["per_part"]), True
                if entry.get("default"):
                    return str(entry["default"]), True
        return text, True

    match = amortized_label_pattern().search(text)
    if match:
        prefix = text[: match.start()].rstrip()
        canonical = f"{prefix} (amortized)" if prefix else match.group(1).lower()
        return canonical, True

    return text, False


@lru_cache(maxsize=None)
def get_proc_mult_targets() -> Mapping[str, tuple[str, float]]:
    """Return process multiplier targets for propagating derived hours."""

    raw = load_json("proc_mult_targets.json")
    result: dict[str, tuple[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        label = str(value.get("label") or "").strip()
        if not label:
            continue
        scale_raw = value.get("scale", 1.0)
        try:
            scale = float(scale_raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Invalid scale for PROC_MULT target '{key}': {scale_raw!r}"
            ) from exc
        result[str(key)] = (label, scale)
    return result


PROC_MULT_TARGETS: Mapping[str, tuple[str, float]] = get_proc_mult_targets()


__all__ = [
    "MONEY_RE",
    "PROC_MULT_TARGETS",
    "TIME_RE",
    "_match_items_contains",
    "_to_noncapturing",
    "amortized_label_pattern",
    "canonicalize_amortized_label",
    "contains",
    "first_num",
    "get_proc_mult_targets",
    "num",
    "num_pct",
    "sheet_num",
    "sheet_pct",
    "sheet_text",
    "strv",
    "sum_time",
    "sum_time_from_series",
    "to_noncapturing",
]
