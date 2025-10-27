"""Integration tests for extracting text rows from CAD fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cad_quoter import geo_extractor
from cad_quoter import geometry


def _dwg_converter_available() -> bool:
    """Return ``True`` when a DWG→DXF converter is accessible."""

    candidates: list[str | None] = [
        os.environ.get("ODA_CONVERTER_EXE"),
        os.environ.get("DWG2DXF_EXE"),
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return True

    default_wrapper = Path(geometry.__file__).with_name("dwg2dxf_wrapper.bat")
    return default_wrapper.exists()


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "Cad Files" / "301_redacted.dwg"

pytestmark = pytest.mark.skipif(
    not FIXTURE_PATH.exists() or not _dwg_converter_available(),
    reason="CAD DWG fixture or converter unavailable",
)


def test_text_table_rows_are_extracted_from_fixture() -> None:
    payload = geo_extractor._read_geo_payload_from_path(FIXTURE_PATH)

    rows = payload.get("rows")
    assert isinstance(rows, list)
    assert len(rows) > 20

    provenance = str(payload.get("provenance_holes") or "")
    assert provenance.upper().startswith("HOLE TABLE")
