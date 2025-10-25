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

    assert "(3) Ø0.375" in lines
    assert "THRU" in lines
    assert "FROM BACK" in lines
    assert "A | Ø0.500 | 2 | (2) DRILL THRU" in lines


def test_read_text_table_uses_internal_fallback(monkeypatch: pytest.MonkeyPatch, fallback_doc: _DummyDoc) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    info = geo_extractor.read_text_table(fallback_doc)

    assert info["hole_count"] == 5
    rows = info["rows"]
    assert len(rows) == 2
    assert rows[0]["qty"] == 3
    assert rows[0]["desc"] == "(3) Ø0.375 THRU FROM BACK"
    assert rows[0]["ref"] == '0.3750"'
    assert rows[0].get("side") == "back"
    assert rows[1]["qty"] == 2
    assert rows[1]["desc"] == "A | Ø0.500 | 2 | (2) DRILL THRU"
    assert rows[1]["ref"] == '0.5000"'
    families = info.get("hole_diam_families_in")
    assert families == {"0.375": 3, "0.5": 2}
    assert info.get("provenance_holes") == "HOLE TABLE"


def test_read_geo_prefers_text_rows(monkeypatch: pytest.MonkeyPatch, fallback_doc: _DummyDoc) -> None:
    families = {"0.375": 3, "0.5": 2}
    text_rows = [
        {
            "hole": "",
            "qty": 3,
            "ref": '0.3750"',
            "desc": "(3) Ø0.375 THRU FROM BACK",
            "side": "back",
        },
        {
            "hole": "",
            "qty": 2,
            "ref": '0.5000"',
            "desc": "A | Ø0.500 | 2 | (2) DRILL THRU",
        },
    ]

    def fake_acad(_doc: _DummyDoc) -> dict[str, object]:
        return {}

    def fake_text(_doc: _DummyDoc) -> dict[str, object]:
        return {
            "rows": list(text_rows),
            "hole_count": 5,
            "provenance_holes": "HOLE TABLE",
            "hole_diam_families_in": families,
        }

    monkeypatch.setattr(geo_extractor, "read_acad_table", fake_acad)
    monkeypatch.setattr(geo_extractor, "read_text_table", fake_text)

    result = geo_extractor.read_geo(fallback_doc)

    assert result["hole_count"] == 5
    assert result["rows"] == text_rows
    assert result["provenance_holes"] == "HOLE TABLE"
    assert result["hole_diam_families_in"] == families
    expected_lines = [
        'qty=3 ref=0.3750" side=BACK desc=(3) Ø0.375 THRU FROM BACK',
        'qty=2 ref=0.5000" side=- desc=A | Ø0.500 | 2 | (2) DRILL THRU',
    ]
    assert result["chart_lines"] == expected_lines


def test_build_column_table_accepts_wrapped_qty_cells() -> None:
    def _entry(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "x": x, "y": y, "height": 0.1}

    entries = [
        _entry("QTY", 0.0, 300.0),
        _entry("DESC", 60.0, 300.0),
        _entry("REF", 120.0, 300.0),
        _entry("(4)", 0.0, 200.0),
        _entry("Ø0.531 DRILL THRU", 60.0, 200.0),
        _entry("0.531", 120.0, 200.0),
        _entry("2X", 0.0, 180.0),
        _entry("Ø0.201 TAP", 60.0, 180.0),
        _entry("0.201", 120.0, 180.0),
    ]

    table_info, _debug = geo_extractor._build_columnar_table_from_entries(entries)

    assert table_info is not None
    rows = table_info["rows"]
    assert [row["qty"] for row in rows] == [4, 2]
    assert rows[0]["ref"] == '0.5310"'
    assert rows[1]["ref"] == '0.2010"'


def test_build_column_table_falls_back_when_qty_column_missing() -> None:
    def _entry(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "x": x, "y": y, "height": 0.1}

    entries = [
        _entry("ID", 10.0, 320.0),
        _entry("DESCRIPTION", 80.0, 320.0),
        _entry("REF", 150.0, 320.0),
        _entry("A -", 20.0, 220.0),
        _entry("3X Ø0.250 TAP THRU", 90.0, 220.0),
        _entry("Ø0.250", 160.0, 220.0),
        _entry("B -", 20.0, 200.0),
        _entry("(2) Ø0.375 DRILL", 90.0, 200.0),
        _entry("Ø0.375", 160.0, 200.0),
    ]

    table_info, _debug = geo_extractor._build_columnar_table_from_entries(entries)

    assert table_info is not None
    rows = table_info["rows"]
    assert [row["qty"] for row in rows] == [3, 2]
    assert rows[0]["desc"].startswith("Ø0.250 TAP")
    assert rows[0]["ref"] == '0.2500"'
    assert rows[1]["ref"] == '0.3750"'
