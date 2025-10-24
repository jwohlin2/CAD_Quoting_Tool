from __future__ import annotations

from pathlib import Path

import pytest

import cad_quoter.geo_extractor as geo_extractor


class DummyDoc:
    pass


@pytest.fixture
def dummy_doc(monkeypatch):
    doc = DummyDoc()
    monkeypatch.setattr(geo_extractor, "_load_doc_for_path", lambda path, use_oda: doc)
    return doc


def test_extract_geo_prefers_text_table(monkeypatch, tmp_path, dummy_doc):
    rows = [{"qty": 8, "desc": "DRILL THRU", "ref": "Ø.250"} for _ in range(11)]
    rows.append({"qty": 6, "desc": "TAP BACK", "ref": "1/4-20"})
    rows.append({"qty": 4, "desc": "CBORE FRONT", "ref": "Ø.500"})
    text_table = {"rows": rows, "hole_count": 88}

    base_geo = {
        "ops_summary": {},
        "provenance": {"holes": "GEOM baseline"},
        "hole_count_geom": 40,
    }
    monkeypatch.setattr(geo_extractor, "extract_geometry", lambda doc: dict(base_geo))
    monkeypatch.setattr(geo_extractor, "read_acad_table", lambda doc: {})
    monkeypatch.setattr(geo_extractor, "read_text_table", lambda doc: text_table)
    monkeypatch.setattr(geo_extractor, "choose_better_table", lambda a, b: b)

    out_path = tmp_path / "sample.dxf"
    out_path.write_text("DXF")

    geo = geo_extractor.extract_geo_from_path(str(out_path), use_oda=False)

    assert geo["hole_count"] == 88
    ops_summary = geo.get("ops_summary") or {}
    assert ops_summary.get("source") == "text_table"
    assert len(ops_summary.get("rows") or []) == len(rows)


def test_extract_geo_without_table(monkeypatch, tmp_path, dummy_doc):
    base_geo = {
        "ops_summary": {"rows": [{"qty": 3}], "source": "chart_lines"},
        "provenance": {"holes": "GEOM concentric"},
        "hole_count_geom": 5,
    }
    monkeypatch.setattr(geo_extractor, "extract_geometry", lambda doc: dict(base_geo))
    monkeypatch.setattr(geo_extractor, "read_acad_table", lambda doc: {})
    monkeypatch.setattr(geo_extractor, "read_text_table", lambda doc: {})
    monkeypatch.setattr(geo_extractor, "choose_better_table", lambda a, b: {})

    geo = geo_extractor.extract_geo_from_path(str(tmp_path / "blank.dxf"), use_oda=False)

    ops_summary = geo.get("ops_summary") or {}
    assert ops_summary.get("source") == "geom"
    assert ops_summary.get("rows") in (None, [])
    assert geo.get("hole_count") == 5
