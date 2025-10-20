from __future__ import annotations

import appV5


def test_aggregate_ops_builds_minimal_rows() -> None:
    rows = [
        {
            "hole": "H1",
            "ref": "0.201",
            "qty": "4X",
            "desc": "4X TAP 1/4-20 THRU",
        }
    ]

    summary = appV5.aggregate_ops(rows)

    assert summary["rows"] == [
        {"hole": "H1", "ref": "0.201", "qty": 4, "desc": "4X TAP 1/4-20 THRU"}
    ]
    assert summary["rows_detail"][0]["per_hole"].get("tap_front") == 1
