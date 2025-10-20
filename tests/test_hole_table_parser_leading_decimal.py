from __future__ import annotations

import math

from cad_quoter.geometry.hole_table_parser import parse_hole_table_lines


def _find_feature(row, feature_type):
    for feature in row.features:
        if feature.get("type") == feature_type:
            return feature
    raise AssertionError(f"Feature {feature_type!r} not found in row {row.ref}")


def test_parse_hole_table_handles_leading_decimal_tokens() -> None:
    lines = [
        "HOLE   REF Ø   QTY   DESCRIPTION",
        "A      Ø.7500  2     Ø.7500 ±.0001 THRU (JIG GRIND); 1.78Ø C'BORE X .38 DEEP FROM BACK",
        "C              4     1/4-20 TAP (JIG GRIND); .623 C'BORE X .62 DEEP FROM FRONT",
    ]

    rows = parse_hole_table_lines(lines)
    assert len(rows) == 2

    row_a = next(row for row in rows if row.ref == "A")
    drill = _find_feature(row_a, "drill")
    assert math.isclose(drill["dia_mm"], 0.75 * 25.4, rel_tol=1e-6)
    assert drill["thru"] is True
    assert drill["from_face"] == "back"

    cbore_a = _find_feature(row_a, "cbore")
    assert math.isclose(cbore_a["depth_mm"], 0.38 * 25.4, rel_tol=1e-6)
    assert math.isclose(cbore_a["dia_mm"], 1.78 * 25.4, rel_tol=1e-6)

    row_c = next(row for row in rows if row.ref == "C")
    cbore_c = _find_feature(row_c, "cbore")
    assert math.isclose(cbore_c["depth_mm"], 0.62 * 25.4, rel_tol=1e-6)

