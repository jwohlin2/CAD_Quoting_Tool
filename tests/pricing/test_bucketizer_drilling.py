import pytest

from cad_quoter.pricing.process_buckets import (
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


def test_fallback_seeds_feature_buckets_from_geo_summary() -> None:
    planner_pricing = {
        "line_items": [
            {
                "op": "drill_ream_bore",
                "minutes": 90.0,
                "machine_cost": 180.0,
                "labor_cost": 0.0,
            }
        ]
    }
    rates = {"labor": {"Inspector": 80.0}, "machine": {}}
    geom = {
        "tapped_count": 4,
        "counterbore": [{"qty": 2}],
        "ops_summary": {
            "totals": {
                "tap_front": 4,
                "cbore_front": 2,
                "csk_front": 3,
                "jig_grind": 1,
            }
        },
    }

    result = bucketize(planner_pricing, rates, {}, qty=1, geom=geom)
    buckets = result["buckets"]

    tapping = buckets["Tapping"]
    counterbore = buckets["Counterbore"]
    countersink = buckets["Countersink"]
    grinding = buckets["Grinding"]
    drilling = buckets["Drilling"]

    assert tapping["minutes"] == pytest.approx(1.2)
    assert counterbore["minutes"] == pytest.approx(0.3)
    assert countersink["minutes"] == pytest.approx(0.36)
    assert grinding["minutes"] == pytest.approx(15.0)

    assert tapping["labor$"] == pytest.approx(2.4)
    assert counterbore["labor$"] == pytest.approx(0.6)
    assert countersink["labor$"] == pytest.approx(0.72)
    assert grinding["labor$"] == pytest.approx(30.0)

    assert drilling["minutes"] == pytest.approx(90.0 - (1.2 + 0.3 + 0.36 + 15.0))
    assert drilling["labor$"] == pytest.approx(180.0 - (2.4 + 0.6 + 0.72 + 30.0))
