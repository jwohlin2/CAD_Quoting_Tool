"""Tests for the high level quote renderer helpers."""

from appV5 import render_quote


def test_render_quote_includes_drilling_time_per_hole_section() -> None:
    payload = {
        "summary": {
            "qty": 1,
            "currency": "$",
            "final_price": 100.0,
        },
        "drilling_time_per_hole": {
            "rows": [
                {
                    "diameter_in": 0.125,
                    "qty": 6,
                    "depth_in": 2.04,
                    "sfm": 39.0,
                    "ipr": 0.002,
                    "minutes_per_hole": 1.055,
                    "group_minutes": 6.33,
                }
            ],
            "tool_components": [{"label": "Deep-Drill", "minutes": 8.0}],
            "toolchange_minutes": 8.0,
            "subtotal_minutes": 6.33,
            "total_minutes_with_toolchange": 14.33,
        },
    }

    rendered = render_quote(payload)

    assert "TIME PER HOLE – DRILL GROUPS" in rendered
    assert 'Dia 0.125" × 6  | depth 2.040" | 39 sfm | 0.0020 ipr | t/hole 1.05 min | group 6×1.05 = 6.33 min' in rendered
    assert "Toolchange adders: Deep-Drill 8.00 min = 8.00 min" in rendered
    assert "Subtotal (per-hole × qty) . 6.33 min  (0.11 hr)" in rendered
    assert "TOTAL DRILLING (with toolchange) . 14.33 min  (0.24 hr)" in rendered
