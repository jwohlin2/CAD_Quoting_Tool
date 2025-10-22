"""Tests for environment flag helpers."""

import importlib

import pytest

from cad_quoter.utils import coerce_bool


def _reload_env_utils():
    module = importlib.import_module("cad_quoter.app.env_flags")
    importlib.reload(module)
    return module


@pytest.mark.parametrize(
    ("token", "expected"),
    [(True, True), (False, False), ("t", True), ("on", True), ("f", False), ("off", False)],
)
def test_coerce_bool_tokens(token, expected):
    assert coerce_bool(token) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("t", True),
        ("1", True),
        ("false", False),
        ("f", False),
        ("0", False),
        (None, False),
        ("", False),
        ("maybe", False),
    ],
)
def test_coerce_env_bool_tokens(value, expected):
    env_utils = _reload_env_utils()
    assert env_utils._coerce_env_bool(value) is expected
