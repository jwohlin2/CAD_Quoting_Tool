"""Tests for quote document helpers."""

from __future__ import annotations

import copy
import importlib.machinery
import logging
import sys
import types

import pytest

# Stub optional runtime dependencies so importing ``appV5`` succeeds without
# installing the full desktop requirements.
_STUBBED_MODULES: list[str] = []
for _name in ("requests", "bs4", "lxml"):
    if _name in sys.modules:
        continue
    module = types.ModuleType(_name)
    module.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
    if _name == "requests":
        module.Session = object  # type: ignore[attr-defined]
    sys.modules[_name] = module
    _STUBBED_MODULES.append(_name)

from appV5 import render_quote
from cad_quoter.app.quote_doc import build_quote_doc_from_result
from cad_quoter.utils.render_utils import QuoteDoc, render_quote_doc


@pytest.fixture(autouse=True)
def _silence_logging() -> None:
    """Prevent noisy legacy loggers from polluting test output."""

    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def test_build_quote_doc_matches_wrapper_output() -> None:
    """The structured builder should round-trip to the wrapper text output."""

    result_input: dict = {"breakdown": {}}
    page_width = 40

    doc = build_quote_doc_from_result(result=copy.deepcopy(result_input), page_width=page_width)
    assert isinstance(doc, QuoteDoc)
    assert doc.title.startswith("QUOTE SUMMARY")

    rendered_from_doc = render_quote_doc(doc, divider="-" * page_width)
    rendered_from_wrapper = render_quote(copy.deepcopy(result_input), page_width=page_width)

    assert rendered_from_doc == rendered_from_wrapper


def teardown_module(module: types.ModuleType) -> None:  # type: ignore[unused-argument]
    """Clean up stubbed runtime modules after the tests finish."""

    for name in _STUBBED_MODULES:
        sys.modules.pop(name, None)
