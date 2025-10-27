"""Tests for DWG layout enumeration helpers."""

from __future__ import annotations

import os
from pathlib import Path

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


_LOAD_DOC = getattr(geo_extractor, "load_doc", None)
_ITER_LAYOUTS = getattr(geo_extractor, "iter_layouts", None)

if _LOAD_DOC is None or _ITER_LAYOUTS is None:
    pytest.skip("layout enumeration helpers unavailable", allow_module_level=True)

if not FIXTURE_PATH.exists() or not _dwg_converter_available():
    pytest.skip("CAD DWG fixture or converter unavailable", allow_module_level=True)


def test_iter_layouts_returns_entries() -> None:
    try:
        doc = _LOAD_DOC(str(FIXTURE_PATH))
    except TypeError:
        doc = _LOAD_DOC(str(FIXTURE_PATH), use_oda=True)

    layouts = list(_ITER_LAYOUTS(doc, None))

    assert isinstance(layouts, list)
    assert layouts, "expected at least one layout entry"
