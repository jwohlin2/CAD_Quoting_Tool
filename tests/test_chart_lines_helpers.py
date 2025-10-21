import types
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_request_stub():
    sys.modules.setdefault("requests", types.ModuleType("requests"))


from cad_quoter.app.chart_lines import (  # noqa: E402  # pylint: disable=wrong-import-position
    RE_DEPTH,
    RE_TAP,
    build_ops_rows_from_lines_fallback,
)


def test_build_ops_rows_from_lines_fallback_extracts_common_ops() -> None:
    lines = [
        "(2) 1/4-20 TAP",
        "THRU",
        "0.25 DEEP FROM BACK",
        "(3) COUNTERBORE Ø0.750",
        "X 0.25 DEEP FROM FRONT",
        "3/8 - NPT",
    ]

    rows = build_ops_rows_from_lines_fallback(lines)

    assert rows == [
        {"hole": "", "ref": "", "qty": 2, "desc": '1/4-20 TAP THRU × 0.25" FROM BACK'},
        {"hole": "", "ref": "", "qty": 3, "desc": '0.7500 C’BORE × 0.25" FROM FRONT'},
        {"hole": "", "ref": "", "qty": 1, "desc": "3/8 - NPT"},
    ]


@pytest.mark.parametrize(
    "text, thread",
    [
        ("(4) #10-32 TAP", "#10-32"),
        ("1/4-20 TAP", "1/4-20"),
    ],
)
def test_re_tap_exposes_thread_spec(text: str, thread: str) -> None:
    match = RE_TAP.search(text)
    assert match is not None
    assert match.group(2).replace(" ", "") == thread


def test_re_depth_captures_numeric_and_side() -> None:
    match = RE_DEPTH.search("0.50 DEEP FROM BACK")
    assert match is not None
    assert match.group(1) == "0.50"
    assert match.group(2) == "BACK"
