import pytest

from bucketizer import (
    INSPECTION_BASE_MIN,
    INSPECTION_PER_HOLE_MIN,
    bucketize,
)


def test_drilling_machine_cost_moves_to_labor_when_only_machine_present() -> None:
    planner_pricing = {
        "line_items": [
            {
                "op": "drill_ream_bore",
                "minutes": 45.0,
                "machine_cost": 571.0,
                "labor_cost": 0.0,
            }
        ],
        "totals": {"machine_cost": 571.0, "labor_cost": 0.0},
    }
    rates = {"labor": {"Inspector": 95.0}, "machine": {}}

    result = bucketize(planner_pricing, rates, {}, qty=1, geom={})

    drilling = result["buckets"]["Drilling"]
    assert drilling["machine$"] == pytest.approx(0.0)
    assert drilling["labor$"] == pytest.approx(571.0)
    assert drilling["total$"] == pytest.approx(571.0)


def test_inspection_minutes_follow_hole_groups() -> None:
    planner_pricing = {"line_items": []}
    rates = {"labor": {"Inspector": 80.0}, "machine": {}}
    geom = {
        "hole_groups": [
            {"qty": 50},
            {"qty": 10},
        ],
        "tapped_count": 5,
    }

    result = bucketize(planner_pricing, rates, {}, qty=1, geom=geom)

    inspection = result["buckets"]["Inspection"]
    expected_minutes = INSPECTION_BASE_MIN + (60 + 5) * INSPECTION_PER_HOLE_MIN
    assert inspection["minutes"] == pytest.approx(expected_minutes)
