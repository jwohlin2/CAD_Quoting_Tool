"""Tests for DXF text stream extraction helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import pytest

from cad_quoter import geo_extractor
from cad_quoter import geometry


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "Cad Files" / "301_redacted.dwg"


def _dwg_converter_available() -> bool:
    """Return ``True`` when a DWG→DXF converter is accessible for tests."""

    candidates: list[str | None] = [
        os.environ.get("ODA_CONVERTER_EXE"),
        os.environ.get("DWG2DXF_EXE"),
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return True

    default_wrapper = Path(geometry.__file__).with_name("dwg2dxf_wrapper.bat")
    return default_wrapper.exists()


_ENSURE_TEXT_STREAM = getattr(geo_extractor, "ensure_text_stream", None)

if _ENSURE_TEXT_STREAM is None:
    pytest.skip("text stream helper unavailable", allow_module_level=True)

if not FIXTURE_PATH.exists() or not _dwg_converter_available():
    pytest.skip("CAD DWG fixture or converter unavailable", allow_module_level=True)


def _collect_entities(result: object) -> list[object]:
    if isinstance(result, Sequence):
        return list(result)
    if isinstance(result, Iterable):
        return list(result)
    entities = getattr(result, "entities", None)
    if isinstance(entities, Sequence):
        return list(entities)
    return [result]


def noop(*_: object, **__: object) -> None:
    """Simple logging callback used for testing."""


def test_ensure_text_stream_returns_entities() -> None:
    entities_raw = _ENSURE_TEXT_STREAM(str(FIXTURE_PATH), log=noop)

    entities = _collect_entities(entities_raw)

    assert len(entities) > 50
