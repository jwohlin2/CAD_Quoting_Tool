from __future__ import annotations

import types
from pathlib import Path

import pytest

from cad_quoter import geo_extractor


_RTEXT_FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "rtext_fixture.dxf"


def _load_rtext_fixture_text() -> str:
    data = _RTEXT_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    for idx in range(0, len(data) - 1, 2):
        code = data[idx].strip()
        value = data[idx + 1].strip()
        if code in {"1000", "1"} and value:
            return value
    raise AssertionError("rtext fixture did not contain expected text payload")


_RTEXT_FIXTURE_TEXT = _load_rtext_fixture_text()


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


class _DummyRText:
    def __init__(self, text: str) -> None:
        self._text = text
        self.dxf = types.SimpleNamespace(text="")
        payload = [(1000, text)]
        self._xdata = {"RTEXT": list(payload), "ACAD_RTEXT": list(payload)}
        self.raw_content = text
        self.content = text
        self.text = ""

    def dxftype(self) -> str:
        return "RTEXT"

    def get_xdata(self, appid: str) -> list[tuple[int, str]]:
        return list(self._xdata.get(appid, []))

    def has_xdata(self, appid: str) -> bool:
        return appid in self._xdata


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
        _DummyRText(_RTEXT_FIXTURE_TEXT),
    ]
    return _DummyDoc(entities)


def test_iter_entity_text_fragments_handles_rtext() -> None:
    entity = _DummyRText(_RTEXT_FIXTURE_TEXT)

    fragments = list(geo_extractor._iter_entity_text_fragments(entity))

    assert fragments, "expected RTEXT fragments"
    texts = [text for text, _ in fragments]
    assert _RTEXT_FIXTURE_TEXT in texts
    assert all(is_mtext for _, is_mtext in fragments)


def test_collect_table_text_lines_normalizes_entities(fallback_doc: _DummyDoc) -> None:
    lines = geo_extractor._collect_table_text_lines(fallback_doc)

    assert "(3) Ø0.375" in lines
    assert "THRU" in lines
    assert "FROM BACK" in lines
    assert "A | Ø0.500 | 2 | (2) DRILL THRU" in lines
    assert "(4) Ø0.625 DRILL THRU FROM FRONT" in lines


def test_read_text_table_uses_internal_fallback(monkeypatch: pytest.MonkeyPatch, fallback_doc: _DummyDoc) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    info = geo_extractor.read_text_table(fallback_doc)

    assert info["hole_count"] == 9
    rows = info["rows"]
    assert len(rows) == 3
    assert rows[0]["qty"] == 3
    assert rows[0]["desc"] == "Ø0.375 THRU FROM BACK"
    assert rows[0]["ref"] == '0.3750"'
    assert rows[0].get("side") == "back"
    assert rows[1]["qty"] == 2
    assert rows[1]["desc"] == "Ø0.500 | 2 | DRILL THRU"
    assert rows[1]["ref"] == '0.5000"'
    assert rows[2]["qty"] == 4
    assert rows[2]["desc"] == "Ø0.625 DRILL THRU FROM FRONT"
    assert rows[2]["ref"] == '0.6250"'
    assert rows[2].get("side") == "front"
    families = info.get("hole_diam_families_in")
    assert families == {"0.375": 3, "0.5": 2, "0.625": 4}
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


def test_read_geo_promotes_rows_txt_fallback(
    monkeypatch: pytest.MonkeyPatch, fallback_doc: _DummyDoc
) -> None:
    rows_txt_lines = [
        "(4) Ø0.250 DRILL THRU",
        "(12) Ø0.339 TAP FROM FRONT",
    ]

    def fake_acad(_doc: _DummyDoc, **_kwargs: object) -> dict[str, object]:
        return {}

    def fake_text(_doc: _DummyDoc, **_kwargs: object) -> dict[str, object]:
        geo_extractor._LAST_TEXT_TABLE_DEBUG = {
            "rows": [],
            "rows_txt_count": len(rows_txt_lines),
            "rows_txt_lines": list(rows_txt_lines),
        }
        return {}

    monkeypatch.setattr(geo_extractor, "read_acad_table", fake_acad)
    monkeypatch.setattr(geo_extractor, "read_text_table", fake_text)

    result = geo_extractor.read_geo(fallback_doc)

    rows = result["rows"]
    assert len(rows) == 2
    assert sorted(row["qty"] for row in rows) == [4, 12]
    assert result["hole_count"] == 16
    assert result["provenance_holes"] == "HOLE TABLE (fallback)"
    refs = {row.get("ref") for row in rows}
    assert '0.2500"' in refs
    assert '0.3390"' in refs


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


def test_extract_row_quantity_handles_parenthetical_tap() -> None:
    qty, remainder = geo_extractor._extract_row_quantity_and_remainder(
        "(2) 5/8-11 TAP THRU FROM BACK"
    )

    assert qty == 2
    assert remainder == "5/8-11 TAP THRU FROM BACK"


def test_build_column_table_splits_semicolons_and_detects_operations() -> None:
    def _entry(text: str, x: float, y: float) -> dict[str, object]:
        return {"text": text, "x": x, "y": y, "height": 0.1}

    entries = [
        _entry("QTY", 60.0, 360.0),
        _entry("DESC", 120.0, 360.0),
        _entry("REF", 180.0, 360.0),
        _entry("", 60.0, 300.0),
        _entry(
            "(2) 5/8-11 TAP THRU FROM BACK; Ø0.812 C'BORE .5 DEEP; 17/32 DRILL THRU",
            120.0,
            300.0,
        ),
        _entry("5/8-11", 180.0, 300.0),
        _entry("", 60.0, 280.0),
        _entry("(3) 1/4-18 NPT TAP THRU", 120.0, 280.0),
        _entry("1/4-18 NPT", 180.0, 280.0),
    ]

    table_info, _debug = geo_extractor._build_columnar_table_from_entries(entries)

    assert table_info is not None
    rows = table_info["rows"]
    assert len(rows) == 4
    qty_sum = table_info.get("sum_qty")
    if qty_sum is None:
        qty_sum = geo_extractor._sum_qty(rows)
    assert qty_sum == sum(row["qty"] for row in rows) == 9

    desc_map = {row["desc"]: row for row in rows}
    assert "Ø0.812 C'BORE .5 DEEP" in desc_map
    assert any("DRILL THRU" in desc for desc in desc_map)
    tap_desc = next(desc for desc in desc_map if "5/8-11 TAP" in desc)
    drill_desc = next(desc for desc in desc_map if "DRILL THRU" in desc and "C'BORE" not in desc)
    npt_desc = next(desc for desc in desc_map if "NPT TAP" in desc)

    tap_tokens = {token.upper() for token in geo_extractor._CANDIDATE_TOKEN_RE.findall(tap_desc)}
    cbore_tokens = {
        token.upper() for token in geo_extractor._CANDIDATE_TOKEN_RE.findall("Ø0.812 C'BORE .5 DEEP")
    }
    drill_tokens = {
        token.upper() for token in geo_extractor._CANDIDATE_TOKEN_RE.findall(drill_desc)
    }
    npt_tokens = {
        token.upper() for token in geo_extractor._CANDIDATE_TOKEN_RE.findall(npt_desc)
    }

    assert "TAP" in tap_tokens
    assert "C'BORE" in cbore_tokens
    assert "DRILL" in drill_tokens
    assert "NPT" in npt_tokens

    tap_row = desc_map[tap_desc]
    assert tap_row.get("side") == "back"


def test_follow_sheet_layout_processed_even_when_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    class _CountingLayout:
        def __init__(self, entities: list[object]) -> None:
            self._entities = entities
            self.query_calls = 0

        def query(self, _spec: str) -> list[object]:
            self.query_calls += 1
            return list(self._entities)

        def __iter__(self):  # type: ignore[override]
            self.query_calls += 1
            return iter(self._entities)

    class _LayoutManager:
        def __init__(self, mapping: dict[str, _CountingLayout]) -> None:
            self._mapping = mapping

        def names(self) -> list[str]:
            return list(self._mapping.keys())

        def get(self, name: str) -> _CountingLayout | None:
            return self._mapping.get(name)

    model_layout = _CountingLayout([_DummyText("SEE SHT 2 FOR HOLE CHART")])
    sheet_layout = _CountingLayout([_DummyText("(1) HOLE TABLE")])

    class _DocWithLayouts:
        def __init__(self) -> None:
            self.layouts = _LayoutManager({"SHEET 2": sheet_layout})

        def modelspace(self) -> _CountingLayout:
            return model_layout

    doc = _DocWithLayouts()

    geo_extractor.read_text_table(
        doc,
        layout_filters={"all_layouts": False, "patterns": ["Model"]},
    )

    assert sheet_layout.query_calls > 0
