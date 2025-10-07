import pandas as pd
import pytest

from appV5 import _sum_time_from_series


def _build_series(values: list[str]) -> pd.Series:
    return pd.Series(values)


def test_sum_time_uses_default_when_only_blank_values() -> None:
    items = _build_series(["In-Process Inspection Hours"])
    vals = _build_series([""])
    types = _build_series(["number"])
    mask = items.str.contains(r"In-Process Inspection", case=False, regex=True, na=False)

    result = _sum_time_from_series(items, vals, types, mask, default=1.0)

    assert pytest.approx(result, rel=1e-6) == 1.0


def test_sum_time_respects_explicit_zero_values() -> None:
    items = _build_series(["In-Process Inspection Hours"])
    vals = _build_series(["0"])
    types = _build_series(["number"])
    mask = items.str.contains(r"In-Process Inspection", case=False, regex=True, na=False)

    result = _sum_time_from_series(items, vals, types, mask, default=1.0)

    assert pytest.approx(result, rel=1e-6) == 0.0


def test_sum_time_converts_minutes_to_hours() -> None:
    items = _build_series(["Inspection Minutes"])
    vals = _build_series(["30"])
    types = _build_series(["number"])
    mask = items.str.contains(r"Inspection", case=False, regex=True, na=False)

    result = _sum_time_from_series(items, vals, types, mask, default=0.0)

    assert pytest.approx(result, rel=1e-6) == 0.5
