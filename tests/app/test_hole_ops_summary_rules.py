"""Regression tests for summary operation classification helpers."""

from __future__ import annotations

import pytest

from cad_quoter.app.hole_ops import _match_summary_operation


@pytest.mark.parametrize(
    "desc, expected",
    [
        ("1/4-18 N.P.T.", "Tap"),
        ("#10-32 Tap", "Tap"),
        ("M12×1.75 TAP", "Tap"),
        ("Ø.257 Drill", "Drill"),
        ("Letter F Drill", "Drill"),
        ("Counterbore Ø0.375", "Counterbore"),
        ("Counterdrill", "Countersink/Counterdrill"),
        ("C'Sink", "Countersink/Counterdrill"),
        ("Jig Grind", "Jig Grind"),
    ],
)
def test_match_summary_operation(desc: str, expected: str) -> None:
    label, _ = _match_summary_operation(desc)
    assert label == expected
