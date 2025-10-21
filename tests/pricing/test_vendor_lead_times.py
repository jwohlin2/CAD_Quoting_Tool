from __future__ import annotations

import math

import pytest

from cad_quoter.pricing import vendor_lead_times


@pytest.mark.parametrize(
    "raw, expected",
    [
        (3, 3),
        ("3 days", 3),
        ("3-5 business days", 5),
        ("1 wk", 7),
        ("2 weeks 3 days", 17),
        ("rush", 1),
        ("18hrs", 1),
        ("", None),
        (None, None),
    ],
)
def test_coerce_lead_time_days_samples(raw: object, expected: int | None) -> None:
    assert vendor_lead_times.coerce_lead_time_days(raw) == expected


@pytest.mark.parametrize(
    "base, includes_weekends, rush, expected",
    [
        (5, False, False, 5),
        (7, True, False, 5),
        (14, True, True, 8),
        (2, False, True, 1),
        (0, False, False, 0),
    ],
)
def test_apply_lead_time_adjustments(base: object, includes_weekends: bool, rush: bool, expected: int | None) -> None:
    assert (
        vendor_lead_times.apply_lead_time_adjustments(
            base, includes_weekends=includes_weekends, rush=rush
        )
        == expected
    )


def test_apply_lead_time_adjustments_none() -> None:
    assert vendor_lead_times.apply_lead_time_adjustments(None) is None


@pytest.mark.parametrize(
    "calendar_days, weekend_days, expected",
    [
        (7.0, 2.0, 5.0),
        (6.0, 2.0, 5.0),
        (14.0, 2.0, 10.0),
        (7.0, 0.0, 7.0),
    ],
)
def test_business_day_conversion(calendar_days: float, weekend_days: float, expected: float) -> None:
    helper = vendor_lead_times._business_days_from_calendar  # type: ignore[attr-defined]
    assert math.isclose(helper(calendar_days, weekend_days=weekend_days), expected)
