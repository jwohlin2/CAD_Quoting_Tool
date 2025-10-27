import pytest

import appV5
from cad_quoter.app.hole_ops import parse_text_table_fragments


def test_merge_wrapped_text_rows_merges_follow_on_lines():
    rows_in = [
        {"hole": "(4)", "ref": "Ø.201", "qty": "", "desc": "TAP THRU"},
        {"hole": "", "ref": "", "qty": "", "desc": ".75 C'BORE AS SHOWN"},
        {"hole": "(6)", "ref": "Ø.312", "qty": "", "desc": "SPOT DRILL"},
        {"hole": "", "ref": "", "qty": "", "desc": "90°"},
    ]

    merged = appV5._merge_wrapped_text_rows(rows_in)

    assert len(merged) == 2
    assert merged[0]["qty"] == "4"
    assert merged[0]["desc"] == "TAP THRU .75 C'BORE AS SHOWN"
    assert merged[1]["qty"] == "6"
    assert merged[1]["desc"] == "SPOT DRILL 90°"


@pytest.mark.parametrize(
    "rows_in, expected_qty",
    [
        ([{"hole": "(2)", "ref": "", "qty": "", "desc": "NPT"}], ["2"]),
        (
            [
                {"hole": "", "ref": "(8)", "qty": "", "desc": "JIG GRIND"},
                {"hole": "", "ref": "", "qty": "", "desc": "Ø.250"},
            ],
            ["8"],
        ),
    ],
)
def test_merge_wrapped_text_rows_backfills_qty(rows_in, expected_qty):
    merged = appV5._merge_wrapped_text_rows(rows_in)
    assert [row.get("qty") for row in merged] == expected_qty


def test_parse_text_table_fragments_with_explicit_qty_column():
    fragments = [
        (0.0, 100.0, 0.12, "HOLE"),
        (1.0, 100.0, 0.12, "REF"),
        (2.0, 100.0, 0.12, "QTY"),
        (3.0, 100.0, 0.12, "DESC"),
        (0.0, 99.0, 0.12, "A1"),
        (1.0, 99.0, 0.12, "Ø.250"),
        (2.0, 99.0, 0.12, "4"),
        (3.0, 99.0, 0.12, "TAP THRU"),
        (0.0, 98.0, 0.12, "A2"),
        (1.0, 98.0, 0.12, "Ø.375"),
        (2.0, 98.0, 0.12, "2"),
        (3.0, 98.0, 0.12, "SPOT DRILL"),
    ]

    rows = parse_text_table_fragments(fragments, min_rows=1)

    assert len(rows) == 2
    assert rows[0]["qty"] == "4"
    assert rows[1]["hole"] == "A2"
    assert "TAP THRU" in rows[0]["desc"].upper()


def test_parse_text_table_fragments_inline_quantity_removes_prefix():
    fragments = [
        (0.0, 100.0, 0.12, "HOLE"),
        (1.0, 100.0, 0.12, "REF"),
        (2.0, 100.0, 0.12, "DESC"),
        (3.0, 100.0, 0.12, "SIDE"),
        (0.0, 99.0, 0.12, "(4) A3"),
        (1.0, 99.0, 0.12, "Ø.500"),
        (2.0, 99.0, 0.12, "C'BORE"),
        (3.0, 99.0, 0.12, "BACK"),
    ]

    rows = parse_text_table_fragments(fragments, min_rows=1)

    assert len(rows) == 1
    assert rows[0]["qty"] == "4"
    assert rows[0]["hole"] == "A3"
    assert "FROM BACK" in rows[0]["desc"].upper()
