import pytest

from bucketizer import bucketize


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
