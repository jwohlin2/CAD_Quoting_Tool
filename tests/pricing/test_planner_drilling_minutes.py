import pytest

from cad_quoter.pricing import planner


def test_drilling_minutes_break_out_operations() -> None:
    mins = planner._drilling_minutes({}, thickness_in=0.5, taps=2, cbrores=1, holes=10)

    assert set(mins) == {
        "drill_total_min",
        "tapping_min",
        "counterbore_min",
        "spot_min",
    }

    assert mins["tapping_min"] == pytest.approx((12.0 * 2) / 60.0)
    assert mins["counterbore_min"] == pytest.approx((20.0 * 1) / 60.0)

    pecks = 2
    approach = 10.0
    peck_time = 6.0
    spot_expected = (10 * (approach + pecks * peck_time)) / 60.0
    assert mins["spot_min"] == pytest.approx(spot_expected)

    drill_penetration = max(0.6, 1.2 - 0.3 * 0.5)
    cut_seconds = 60.0 * (0.5 / drill_penetration)
    per_hole_seconds = approach + pecks * peck_time + cut_seconds
    cycle_minutes = (10 * per_hole_seconds) / 60.0
    total_expected = cycle_minutes + ((12.0 * 2) / 60.0) + ((20.0 * 1) / 60.0)
    assert mins["drill_total_min"] == pytest.approx(total_expected)


def test_minutes_die_plate_includes_drill_breakouts() -> None:
    g = {
        "hole_count": 10,
        "tap_qty": 2,
        "cbore_qty": 1,
        "thickness_in": 0.5,
        "pocket_area_in2": 0.0,
        "edge_len_in": 0.0,
        "slot_count": 0,
        "plate_area_in2": 0.0,
    }
    tol = {"profile_tol": None, "flatness_spec": None, "parallelism_spec": None}

    minutes = planner._minutes_die_plate({}, {}, g, tol, material="steel")

    assert "Drilling" in minutes
    assert "Tapping" in minutes
    assert "Counterbore" in minutes
    assert "Spot-Drill" in minutes

    drill_breakdown = planner._drilling_minutes(g, 0.5, 2, 1, 10)

    assert minutes["Drilling"] == pytest.approx(drill_breakdown["drill_total_min"])
    assert minutes["Tapping"] == pytest.approx(drill_breakdown["tapping_min"])
    assert minutes["Counterbore"] == pytest.approx(drill_breakdown["counterbore_min"])
    assert minutes["Spot-Drill"] == pytest.approx(drill_breakdown["spot_min"])
