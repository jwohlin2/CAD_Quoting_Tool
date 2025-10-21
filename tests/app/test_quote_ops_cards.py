from __future__ import annotations

import pytest

import appV5


def test_emit_hole_table_ops_cards_renders_sections() -> None:
    lines: list[str] = []
    geo = {
        "ops_summary": {
            "rows": [
                {
                    "qty": 6,
                    "desc": "6X TAP 1/4-20 THRU FROM FRONT",
                    "ref": "",
                },
                {
                    "qty": 3,
                    "desc": "Ø0.500 CBORE × .250 DEEP FROM BACK",
                    "ref": "",
                },
                {
                    "qty": 2,
                    "desc": "CENTER DRILL .187 DIA × .060 DEEP",
                    "ref": "",
                },
                {
                    "qty": 1,
                    "desc": "JIG GRIND Ø.375",
                    "ref": "",
                },
            ]
        }
    }

    appV5._emit_hole_table_ops_cards(lines, geo=geo, material_group="aluminum", speeds_csv=None)

    joined = "\n".join(lines)

    assert "MATERIAL REMOVAL – TAPPING" in joined
    assert "MATERIAL REMOVAL – COUNTERBORE" in joined
    assert "MATERIAL REMOVAL – SPOT (CENTER DRILL)" in joined
    assert "MATERIAL REMOVAL – JIG GRIND" in joined

    tap_line = next(
        line for line in lines if line.startswith("1/4-20 × 6")
    )
    assert "0.05" in tap_line or "0.0500" in tap_line
    assert "rpm" in tap_line and "ipm" in tap_line

    cbore_line = next(line for line in lines if line.startswith("Ø0.5000"))
    assert "rpm" in cbore_line and "ipm" in cbore_line

    spot_line = next(line for line in lines if line.startswith("Spot drill"))
    assert "rpm" in spot_line and "ipm" in spot_line


def test_aggregate_ops_sets_built_rows() -> None:
    rows = [
        {"hole": "A1", "ref": "Ø0.257", "qty": 4, "desc": "Ø0.257 THRU"},
        {"hole": "A2", "ref": "", "qty": 2, "desc": "1/4-20 TAP THRU"},
    ]

    summary = appV5.aggregate_ops(rows)

    assert summary.get("built_rows") == 2


def test_render_ops_cards_from_summary_helper() -> None:
    lines: list[str] = []
    ops_summary = {
        "cards": {
            "drilling": {
                "title": "Time per hole – drill groups",
                "header": ["MATERIAL REMOVAL – DRILLING", "=" * 64],
                "rows": [
                    {
                        "label": "Dia 0.250",
                        "qty": 6,
                        "side": "front",
                        "depth_in": 2.04,
                        "feed_fmt": "39 sfm | 0.0020 ipr",
                        "t_per_hole_min": 1.055,
                    }
                ],
                "footer": ["Subtotal . 6.33 min"],
            }
        }
    }

    rendered = appV5._render_ops_cards_from_summary(lines, ops_summary=ops_summary)

    assert rendered is True
    assert "MATERIAL REMOVAL – DRILLING" in lines
    assert any("DIA 0.250" in line.upper() for line in lines)
    assert ops_summary["cards"]["drilling"].get("total_minutes") == pytest.approx(6.33, rel=1e-6)
