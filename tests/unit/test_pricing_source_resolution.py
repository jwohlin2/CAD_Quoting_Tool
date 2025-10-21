import pytest

from appkit.planner_adapter import resolve_pricing_source_value


@pytest.mark.parametrize(
    "explicit_value",
    ["Estimator", "Manual", "  Estimator  "]
)
def test_explicit_pricing_source_is_preserved(explicit_value):
    result = resolve_pricing_source_value(
        explicit_value,
        used_planner=True,
        breakdown={"process_minutes": 12},
    )
    assert result == explicit_value.strip()


def test_explicit_planner_value_still_normalizes():
    result = resolve_pricing_source_value("Planner", used_planner=False)
    assert result == "planner"


def test_planner_detected_when_no_explicit_value():
    result = resolve_pricing_source_value(
        None,
        used_planner=False,
        breakdown={"process_minutes": 5},
    )
    assert result == "planner"
