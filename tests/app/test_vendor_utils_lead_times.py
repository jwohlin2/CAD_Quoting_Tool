from __future__ import annotations

import pytest

from appkit import vendor_utils
from cad_quoter.pricing import vendor_lead_times


@pytest.mark.parametrize(
    "raw",
    ["3-5 days", "rush", 10, None],
)
def test_ui_vendor_utils_align_with_pricing(raw: object) -> None:
    assert vendor_utils.coerce_lead_time_days(raw) == vendor_lead_times.coerce_lead_time_days(raw)


@pytest.mark.parametrize(
    "base, includes_weekends, rush",
    [
        (10, False, False),
        (10, True, True),
        ("2 weeks", True, False),
        (None, False, False),
    ],
)
def test_ui_vendor_utils_adjustment_aligns(base: object, includes_weekends: bool, rush: bool) -> None:
    assert vendor_utils.apply_lead_time_adjustments(
        base, includes_weekends=includes_weekends, rush=rush
    ) == vendor_lead_times.apply_lead_time_adjustments(
        base, includes_weekends=includes_weekends, rush=rush
    )
