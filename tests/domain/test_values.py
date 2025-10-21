"""Tests for value coercion helpers."""
from __future__ import annotations

import math

import pytest

from cad_quoter.domain_models.values import coerce_float_or_none, parse_mixed_fraction


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2\"\"", 2.0),
        ("3/4\"", 0.75),
        ("1 1/2\"", 1.5),
        ("1\u00A01/2\"", 1.5),
        ("1-1/2\"", 1.5),
        ("24\"", 24.0),
        ("0.25 in", 0.25),
    ],
)
def test_coerce_float_or_none_handles_imperial_measurements(raw: str, expected: float) -> None:
    result = coerce_float_or_none(raw)
    assert result is not None
    assert math.isclose(result, expected)


def test_coerce_float_or_none_rejects_invalid_fraction() -> None:
    assert coerce_float_or_none("1/0\"") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3/4", 0.75),
        ("1 1/2", 1.5),
        ("-2 1/4", -2.25),
    ],
)
def test_parse_mixed_fraction_exposes_shared_parser(raw: str, expected: float) -> None:
    result = parse_mixed_fraction(raw)
    assert result is not None
    assert math.isclose(result, expected)
