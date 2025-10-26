import pytest

from cad_quoter.app.hole_ops import (
    _aggregate_summary_rows,
    _match_summary_operation,
)


@pytest.mark.parametrize(
    "desc, expected",
    [
        ("Ø0.312\" TAP THRU", "Tap"),
        ("C’BORE .500 FROM FRONT", "C'Bore"),
        ("Counter–drill 0.375", "C'Drill/CSink"),
        ("JIG GRIND BACK", "Jig Grind"),
        ("DRILL THRU", "Drill"),
        ("C DRILL THRU", "Drill"),
        ("SPOT DRILL × .125", "C'Drill/CSink"),
    ],
)
def test_match_summary_operation(desc: str, expected: str) -> None:
    label, _ = _match_summary_operation(desc)
    assert label == expected


def test_match_summary_operation_unknown() -> None:
    label, _ = _match_summary_operation("Ream Finish")
    assert label == "Unknown"


def test_aggregate_summary_rows_respects_operation_order() -> None:
    rows = [
        {"qty": 2, "desc": "Tap & C'Bore"},
        {"qty": 3, "desc": "C'Bore"},
        {"qty": 1, "desc": "Counter-Drill"},
        {"qty": 4, "desc": "Jig Grind"},
        {"qty": 5, "desc": "Drill Thru"},
        {"qty": 1, "desc": "Unknown op"},
    ]
    aggregates = _aggregate_summary_rows(rows)
    op_totals = aggregates.get("operation_totals")
    assert op_totals == {
        "Tap": 2,
        "C'Bore": 3,
        "C'Drill/CSink": 1,
        "Jig Grind": 4,
        "Drill": 5,
        "Unknown": 1,
    }

    totals = aggregates.get("totals", {})
    assert totals.get("tap") == 2
    assert totals.get("counterbore") == 3
    assert totals.get("spot") == 1
    assert totals.get("jig_grind") == 4
    assert totals.get("drill") == 5
    assert totals.get("unknown") == 1
