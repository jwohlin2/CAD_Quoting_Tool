"""Tests for diameter parsing helpers in :mod:`cad_quoter.app.hole_ops`."""

from __future__ import annotations

import math

import pytest

from cad_quoter.app.hole_ops import parse_dim


@pytest.mark.parametrize(
    "raw, expected",
    [
        (".75", 0.75),
        ("Ø.625", 0.625),
        ("3/4", 0.75),
        ("3/4-10", 0.75),
        ("1 1/4 in", 1.25),
        ("Ø6.35mm", 0.25),
        ("10 mm", 10 / 25.4),
    ],
)
def test_parse_dim_handles_common_formats(raw: str, expected: float) -> None:
    value = parse_dim(raw)
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
def test_parse_dim_rejects_invalid_input(raw: str | None) -> None:
    assert parse_dim(raw) is None
