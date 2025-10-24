import pytest

from cad_quoter.app.geo_helpers import (
    aggregate_ops_from_rows,
    _seed_drill_bins_from_geo,
    _seed_drill_bins_from_geo__local,
)


def test_aggregate_ops_from_rows_counts_drill_and_tap_sides():
    rows = [
        {"hole": "H1", "qty": "2", "desc": "M6 TAP FRONT"},
        {"hole": "H2", "qty": 1, "desc": "1/4-20 TAP FROM BACK"},
        {"hole": "H3", "qty": 3, "desc": "SPOT DRILL"},
        {"hole": "H4", "qty": "4", "desc": ".500 CBORE FRONT & BACK"},
        {"hole": "H5", "qty": "1", "desc": "JIG GRIND"},
    ]

    result = aggregate_ops_from_rows(rows)

    totals = result["totals"]
    assert totals["tap"] == 3
    assert totals["drill"] == 3
    assert totals["tap_front"] == 2
    assert totals["tap_back"] == 1
    assert totals["counterbore"] == 4
    assert totals["counterbore_front"] == 4
    assert totals["counterbore_back"] == 4
    assert totals["spot"] == 3
    assert totals["jig_grind"] == 1

    assert result["actions_total"] == 18
    assert result["back_ops_total"] == 5
    assert result["flip_required"] is True
    assert len(result["rows"]) == 5


@pytest.mark.parametrize(
    "geo_map, expected",
    [
        (
            {"hole_diams_in": [0.257, "0.257", 0.5], "hole_sets": [{"diam_mm": 6.35}]},
            {0.257: 2, 0.5: 1, 0.25: 1},
        ),
        (
            {"hole_diams_mm": [5, "5.0"]},
            {round(5 / 25.4, 3): 2},
        ),
    ],
)
def test_seed_drill_bins_from_geo_local_handles_units(geo_map, expected):
    bins = _seed_drill_bins_from_geo__local(geo_map)
    assert bins == expected


def test_seed_drill_bins_from_geo_prefers_family_counts():
    geo = {
        "hole_diam_families_in": {"0.25": 3, '0.5"': 1},
        "hole_diams_in": [0.25, 0.5, 0.75],
    }

    bins = _seed_drill_bins_from_geo(geo)

    assert bins == {0.25: 3, 0.5: 1}
    # ensure fallback ignored because families already provided
    assert 0.75 not in bins
