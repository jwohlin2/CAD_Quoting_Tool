import pytest

import appV5


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
