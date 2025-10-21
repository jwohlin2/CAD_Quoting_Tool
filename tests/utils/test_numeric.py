"""Tests for :mod:`cad_quoter.utils.numeric`."""

from __future__ import annotations

import math

import pytest

from cad_quoter.utils.numeric import parse_mixed_fraction


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3/4", 0.75),
        ("1 1/2", 1.5),
        ("1-1/2", 1.5),
        ("-2 1/4", -2.25),
        ("+4 3/8", 4.375),
    ],
)
def test_parse_mixed_fraction_handles_catalog_formats(raw: str, expected: float) -> None:
    result = parse_mixed_fraction(raw)
    assert result is not None
    assert math.isclose(result, expected)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "-",
        "1/0",
        "hello",
    ],
)
def test_parse_mixed_fraction_rejects_invalid_input(raw: str) -> None:
    assert parse_mixed_fraction(raw) is None
