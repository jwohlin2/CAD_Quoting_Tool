from __future__ import annotations

import types

import pytest

from cad_quoter import geo_extractor


class _DummyMText:
    def __init__(self, text: str) -> None:
        self._text = text

    def dxftype(self) -> str:
        return "MTEXT"

    def plain_text(self) -> str:
        return self._text


class _DummyText:
    def __init__(self, text: str) -> None:
        self.dxf = types.SimpleNamespace(text=text)

    def dxftype(self) -> str:
        return "TEXT"


class _DummySpace:
    def __init__(self, entities: list[object]) -> None:
        self._entities = entities

    def query(self, _spec: str) -> list[object]:
        return list(self._entities)


class _DummyDoc:
    def __init__(self, entities: list[object]) -> None:
        self._entities = entities

    def modelspace(self) -> _DummySpace:
        return _DummySpace(self._entities)


@pytest.fixture
def fallback_doc() -> _DummyDoc:
    entities = [
        _DummyMText(r"\\A1;(3) %%C0.375\\PTHRU"),
        _DummyMText("FROM BACK"),
        _DummyText("A | Ø0.500 | 2 | (2) DRILL THRU"),
    ]
    return _DummyDoc(entities)


def test_collect_table_text_lines_normalizes_entities(fallback_doc: _DummyDoc) -> None:
    lines = geo_extractor._collect_table_text_lines(fallback_doc)

    assert "(3) Ø0.375 THRU" in lines
    assert "FROM BACK" in lines
    assert "A | Ø0.500 | 2 | (2) DRILL THRU" in lines


def test_read_text_table_uses_internal_fallback(monkeypatch: pytest.MonkeyPatch, fallback_doc: _DummyDoc) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    info = geo_extractor.read_text_table(fallback_doc)

    assert info["hole_count"] == 5
    rows = info["rows"]
    assert len(rows) == 2
    assert rows[0]["qty"] == 3
    assert rows[0]["desc"].startswith("Ø0.375")
    families = info.get("hole_diam_families_in")
    assert families == {"0.375": 3, "0.5": 2}
    assert info.get("provenance_holes") == "HOLE TABLE (TEXT_FALLBACK)"


def test_extract_geo_prefers_text_table_and_sticks(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = object()
    state: dict[str, dict[str, object]] = {"geo": {}}

    def fake_load_doc(_path: object, *, use_oda: bool) -> object:  # pragma: no cover - simple stub
        assert use_oda is False
        return doc

    monkeypatch.setattr(geo_extractor, "_load_doc_for_path", fake_load_doc)
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    def fake_extract_geometry(_doc: object) -> dict[str, object]:
        return state["geo"]

    monkeypatch.setattr(geo_extractor, "extract_geometry", fake_extract_geometry)
    monkeypatch.setattr(geo_extractor, "read_acad_table", lambda _doc: {"rows": []})

    call_count = {"value": 0}

    def fake_read_text_table(_doc: object) -> dict[str, object]:
        if call_count["value"] == 0:
            rows = [{"qty": "2", "desc": "TEXT"}]
        else:
            rows = [{"qty": "0", "desc": "WORSE"}]
        call_count["value"] += 1
        return {"rows": rows}

    monkeypatch.setattr(geo_extractor, "read_text_table", fake_read_text_table)

    geo = geo_extractor.extract_geo_from_path("fake.dxf", use_oda=False)
    ops_summary = geo["ops_summary"]
    rows = ops_summary["rows"]
    assert ops_summary["source"] == "text_table"
    assert rows and rows[0]["desc"] == "TEXT"

    state["geo"] = geo

    geo_again = geo_extractor.extract_geo_from_path("fake.dxf", use_oda=False)
    ops_summary_again = geo_again["ops_summary"]
    assert ops_summary_again["source"] == "text_table"
    assert ops_summary_again["rows"] == rows
