"""Utilities for normalizing and adjusting vendor lead-time data.

The pricing layer consumes lead-times from a variety of sources.  Some
vendors express the lead-time as a range (``"3-5 business days"``), while
others provide calendar weeks or looser phrases such as ``"rush"``.  The UI
surface needs to derive the same values so we keep a shared implementation
here and allow other packages (such as :mod:`appkit`) to re-use it.
"""
from __future__ import annotations

import math
import re
from typing import Any

from cad_quoter.utils.numeric import coerce_float


_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to|–|—|/)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_TOKEN_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>business\s+days?|business\s+day|days?|day|weeks?|week|wks?|wk|hrs?|hr|hours?|hour|d|w|h)",
    re.IGNORECASE,
)
_COMPACT_UNIT_RE = re.compile(r"(\d)([a-zA-Z])")
def coerce_lead_time_days(value: Any, *, default: int | None = None) -> int | None:
    """Return a conservative day estimate from a vendor lead-time token.

    ``value`` may be a numeric literal, a range such as ``"3-5 business days"``
    or a more relaxed string (``"1 wk"``, ``"rush"``).  We favour the
    upper-bound when ranges are provided so downstream calculations lean
    towards safety.  The function always rounds up to a whole number of days.
    """

    numeric = coerce_float(value)
    if numeric is not None:
        return math.ceil(max(0.0, numeric))

    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    lowered = text.lower()
    if "rush" in lowered and not re.search(r"\d", lowered):
        # Rush entries frequently omit numbers – treat as next-day by default.
        return 1

    working = _COMPACT_UNIT_RE.sub(r"\1 \2", lowered)
    range_match = _RANGE_RE.search(working)
    tokens: list[float] = []

    if range_match:
        upper = coerce_float(range_match.group(2))
        if upper is not None:
            tokens.append(upper)
        # Replace the range with the upper bound so the token regex can pick
        # up the unit that follows (e.g. "3-5 days" -> "5 days").
        working = working[: range_match.start()] + range_match.group(2) + working[range_match.end() :]

    total_days = 0.0
    for match in _TOKEN_RE.finditer(working):
        raw_value = coerce_float(match.group("value"))
        if raw_value is None:
            continue
        unit = match.group("unit").strip().lower()
        if "week" in unit or unit in {"wk", "wks", "w"}:
            factor = 7.0
        elif "hour" in unit or unit in {"hr", "hrs", "h"}:
            # Treat twenty-four hours as one day; values below a day still
            # round up at the end of the function.
            factor = 1.0 / 24.0
        else:
            factor = 1.0
        token_days = raw_value * factor
        tokens.append(token_days)
        total_days += token_days

    if not tokens:
        # Fall back to any digits we can find, taking the largest value.
        values = [
            coerce_float(token)
            for token in re.findall(r"\d+(?:\.\d+)?", working)
        ]
        tokens = [float(val) for val in values if val is not None]

    if not tokens:
        return default

    if total_days > 0:
        days = max(total_days, max(tokens))
    else:
        days = max(tokens)
    if days <= 0:
        return 0

    return max(1, math.ceil(days))


def _business_days_from_calendar(calendar_days: float, weekend_days: float = 2.0) -> float:
    """Convert calendar days to business days by removing weekends."""

    if calendar_days <= 0:
        return 0.0

    weekend_days = max(0.0, min(float(weekend_days), 7.0))
    business_per_week = 7.0 - weekend_days
    if business_per_week <= 0:
        return max(0.0, calendar_days)

    full_weeks = math.floor(calendar_days / 7.0)
    remainder = calendar_days - full_weeks * 7.0
    business_days = full_weeks * business_per_week
    business_days += min(remainder, business_per_week)
    return max(0.0, business_days)


def apply_lead_time_adjustments(
    lead_time_days: Any,
    *,
    includes_weekends: bool = False,
    rush: bool = False,
    weekend_days: float = 2.0,
    rush_discount: float = 2.0,
    minimum_days: float = 1.0,
) -> int | None:
    """Apply calendar weekend and rush adjustments to a lead-time value.

    Parameters
    ----------
    lead_time_days:
        Base lead-time expressed in days (or something coercible via
        :func:`coerce_lead_time_days`).
    includes_weekends:
        When ``True`` the value is considered a calendar-day quantity and we
        remove weekend days to derive business days.
    rush:
        Expedites typically reduce the lead-time by a couple of days.  Set this
        flag to ``True`` when a rush fee has been applied.
    weekend_days:
        Number of non-working days per week (defaults to 2).
    rush_discount:
        Number of days shaved off for rush handling (defaults to two days).
    minimum_days:
        Lower bound for the adjusted lead-time (defaults to one day).
    """

    coerced = coerce_lead_time_days(lead_time_days)
    if coerced is None:
        return None
    if coerced == 0:
        return 0

    days_value = float(coerced)
    if includes_weekends:
        days_value = _business_days_from_calendar(days_value, weekend_days=weekend_days)

    if rush:
        try:
            rush_delta = float(rush_discount)
        except Exception:
            rush_delta = 0.0
        if math.isfinite(rush_delta) and rush_delta > 0:
            days_value = max(days_value - rush_delta, 0.0)

    try:
        minimum = float(minimum_days)
    except Exception:
        minimum = 0.0
    if not math.isfinite(minimum):
        minimum = 0.0
    minimum = max(0.0, minimum)

    adjusted = max(minimum, days_value)
    return max(0, math.ceil(adjusted))


__all__ = [
    "coerce_lead_time_days",
    "apply_lead_time_adjustments",
]
