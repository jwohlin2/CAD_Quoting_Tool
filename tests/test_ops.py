"""Tests for summary operation classification helpers."""

from __future__ import annotations

import pytest

from cad_quoter.app import hole_ops


@pytest.mark.parametrize(
    "desc, expected",
    [
        ("1/4-20 TAP THRU", "Tap"),
        ("Ø0.250 DRILL THRU", "Drill"),
        ("Ø0.500 C’BORE FROM BACK", "C'bore"),
        ("SPOT DRILL Ø0.312", "C'drill"),
        ("FINAL JIG GRIND", "Jig Grind"),
    ],
)
def test_match_summary_operation_buckets(desc: str, expected: str) -> None:
    label, normalized = hole_ops._match_summary_operation(desc)

    assert label == expected
    assert normalized == hole_ops._normalize_ops_desc(desc)
