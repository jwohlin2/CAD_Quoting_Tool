"""Tests for diameter parsing helpers in :mod:`cad_quoter.app.hole_ops`."""

from __future__ import annotations

import math

import pytest

from cad_quoter.app.hole_ops import _parse_ref_to_inch


@pytest.mark.parametrize(
    "raw, expected",
    [
        (".75", 0.75),
        ("Ø.625", 0.625),
        ("3/4", 0.75),
        ("3/4-10", 0.75),
        ("1 1/4 in", 1.25),
    ],
)
def test_parse_ref_to_inch_handles_common_formats(raw: str, expected: float) -> None:
    value = _parse_ref_to_inch(raw)
    assert value is not None
    assert math.isclose(value, expected)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "hello",
        "Ø-",
    ],
)
def test_parse_ref_to_inch_rejects_invalid_input(raw: str | None) -> None:
    assert _parse_ref_to_inch(raw) is None
