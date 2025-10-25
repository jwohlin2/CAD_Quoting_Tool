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
