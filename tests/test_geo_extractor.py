from __future__ import annotations

import types
from collections import defaultdict
from pathlib import Path
import re

import pytest

from cad_quoter import geo_extractor
from cad_quoter.utils.chart_buckets import classify_chart_rows


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


class _ToggleLayout(_DummySpace):
    def __init__(self, entities: list[object]) -> None:
        super().__init__(entities)
        self._active = False

    def activate(self) -> None:
        self._active = True

    def query(self, _spec: str) -> list[object]:
        if not self._active:
            return []
        return list(self._entities)

    def __iter__(self):
        return iter(self.query(""))


class _FollowLayoutManager:
    def __init__(self, layout: _ToggleLayout) -> None:
        self._layout = layout
        self._calls: defaultdict[str, int] = defaultdict(int)

    def names(self) -> list[str]:
        return ["SHEET (2)"]

    def get(self, name: str) -> _ToggleLayout:
        self._calls[name] += 1
        if self._calls[name] > 1:
            self._layout.activate()
        return self._layout


class _FollowDoc:
    def __init__(self, model_entities: list[object], layout: _ToggleLayout) -> None:
        self._model = _ToggleLayout(model_entities)
        self._model.activate()
        self.layouts = _FollowLayoutManager(layout)

    def modelspace(self) -> _DummySpace:
        return self._model


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

    result = geo_extractor.read_text_table(fallback_doc)

    rows = result.get("rows") or []
    assert len(rows) == 2
    assert sorted(row.get("qty") for row in rows) == [3, 4]

    debug = geo_extractor.get_last_text_table_debug() or {}
    assert debug.get("rows_txt_count") == 2


def test_read_text_table_raises_when_layout_filter_has_no_match(
    monkeypatch: pytest.MonkeyPatch, fallback_doc: _DummyDoc
) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    with pytest.raises(RuntimeError):
        geo_extractor.read_text_table(
            fallback_doc,
            layout_filters={"all_layouts": False, "patterns": ["^PAPERONLY$"]},
        )


def test_read_text_table_auto_retries_excluded_am_bor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    class _LayeredMText(_DummyMText):
        def __init__(self, text: str, layer: str) -> None:
            super().__init__(text)
            self.dxf = types.SimpleNamespace(layer=layer)

    doc = _DummyDoc(
        [
            _LayeredMText("(2) Ø0.250 DRILL THRU", "AM_BOR"),
            _LayeredMText("(3) Ø0.312 TAP FROM FRONT", "AM_BOR"),
        ]
    )

    result = geo_extractor.read_text_table(doc)

    rows = result.get("rows") or []
    assert len(rows) == 2
    assert sorted(row.get("qty") for row in rows) == [2, 3]
    assert any("Ø0.250" in str(row.get("desc")) for row in rows)


def test_layer_filter_excludes_am_bor_from_post_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    class _LayeredMText(_DummyMText):
        def __init__(self, text: str, layer: str) -> None:
            super().__init__(text)
            self.dxf = types.SimpleNamespace(layer=layer)

    doc = _DummyDoc(
        [
            _LayeredMText("(2) Ø0.250 DRILL THRU", "AM_BOR"),
            _LayeredMText("(3) Ø0.312 TAP FROM FRONT", "AM_BOR"),
        ]
    )

    geo_extractor.read_text_table(doc)
    out = capsys.readouterr().out

    post_lines = [
        line for line in out.splitlines() if "[TEXT-SCAN] kept_by_layer(post)=" in line
    ]
    assert post_lines, "expected kept_by_layer(post) line"
    assert all("AM_BOR" not in line for line in post_lines)


def test_numeric_ladder_line_is_dropped_in_fallback() -> None:
    payload = geo_extractor._publish_fallback_from_rows_txt(
        [
            "(2) Ø0.250 DRILL THRU",
            "1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9",
            "(4) Ø0.375 COUNTERBORE",
        ]
    )

    rows = payload.get("rows") or []
    assert len(rows) == 2
    assert all(row.get("qty") in {2, 4} for row in rows)


def test_admin_note_line_is_ignored() -> None:
    payload = geo_extractor._publish_fallback_from_rows_txt(
        ["(2) Ø0.250 DRILL THRU", "BREAK ALL .12 R"],
    )

    rows = payload.get("rows") or []
    assert len(rows) == 1
    assert rows[0].get("qty") == 2
    assert "BREAK ALL" not in (rows[0].get("desc") or "")


def test_break_all_line_never_forms_row() -> None:
    payload = geo_extractor._fallback_text_table(["BREAK ALL .12 R"])

    rows = payload.get("rows") or []
    assert rows == []

    buckets, row_count, qty_sum = classify_chart_rows(rows)
    assert buckets == {}
    assert row_count == 0
    assert qty_sum == 0


def test_anchor_rows_are_authoritative_over_roi() -> None:
    anchor_rows = [
        {"qty": 2, "desc": "(2) 5/8-11 TAP X 1.00 DEEP"},
        {"qty": 2, "desc": "(2) #10-32 TAP THRU"},
        {
            "qty": 4,
            "desc": "(4) .75Ø C'BORE AS SHOWN; \"R\" (.339Ø) DRILL THRU AS SHOWN; 1/8- N.P.T.",
        },
    ]
    roi_rows = [{"qty": 8, "desc": "ROI noise"}, {"qty": 4, "desc": ".12 R"}]

    merged, dedup, authoritative = geo_extractor._combine_text_rows(
        anchor_rows,
        roi_rows,
        roi_rows,
    )

    assert authoritative
    assert merged == anchor_rows
    assert dedup == 0
    assert sum(row.get("qty", 0) for row in merged) == 8

    buckets, row_count, qty_sum = classify_chart_rows(merged)
    assert row_count == 3
    assert qty_sum == 8
    assert buckets.get("tap") == 8
    assert buckets.get("cbore") == 4
    assert buckets.get("npt") == 4
    assert buckets.get("drill") == 4


def test_classify_op_row_splits_semicolons() -> None:
    desc = "(4) .75Ø C'BORE AS SHOWN; \"R\" (.339Ø) DRILL THRU; 1/8- N.P.T."
    items = geo_extractor.classify_op_row(desc)
    assert items, "expected classifier items"
    totals: dict[str, int] = {}
    for item in items:
        kind = item.get("kind")
        assert isinstance(kind, str)
        totals[kind] = totals.get(kind, 0) + 4
    assert totals.get("cbore") == 4
    assert totals.get("drill") == 4
    assert totals.get("npt") == 4


def test_classify_op_row_counterdrill_synonyms() -> None:
    for text in ("C DRILL", "CTR DRILL", "C' DRILL"):
        items = geo_extractor.classify_op_row(text)
        assert any(item.get("kind") == "cdrill" for item in items)


def test_split_actions_and_classify_action_counts() -> None:
    desc = "(4) .75Ø C'BORE AS SHOWN; \"R\" (.339Ø) DRILL THRU AS SHOWN; 1/8- N.P.T."

    fragments = geo_extractor.split_actions(desc)
    assert len(fragments) == 3

    totals: dict[str, int] = {}
    for fragment in fragments:
        action = geo_extractor.classify_action(fragment)
        kind = action.get("kind")
        assert isinstance(kind, str)
        totals[kind] = totals.get(kind, 0) + 4
        if kind == "tap":
            assert action.get("npt") is True

    assert totals.get("cbore") == 4
    assert totals.get("drill") == 4
    assert totals.get("tap") == 4


def test_ops_manifest_combines_table_and_geom() -> None:
    rows = [
        {"qty": 4, "desc": '\"R\" (.339Ø) DRILL THRU'},
        {"qty": 2, "desc": "(2) 1/4-20 TAP"},
    ]
    geom_holes = {"groups": [{"dia_in": 0.25, "count": 10}], "total": 10}

    manifest = geo_extractor.ops_manifest(rows, geom_holes=geom_holes)

    table_counts = manifest.get("table", {})
    total_counts = manifest.get("total", {})
    geom_counts = manifest.get("geom", {})
    text_info = manifest.get("text", {})

    assert table_counts.get("drill") == 4
    assert table_counts.get("tap") == 2
    assert geom_counts.get("drill") == 10
    assert geom_counts.get("residual_drill") == 6
    assert geom_counts.get("total") == 10
    assert total_counts.get("drill") == 6
    assert text_info.get("estimated_total_drills") == 4


def test_npt_counts_as_tap() -> None:
    rows = [{"qty": 4, "desc": "1/8- N.P.T."}]

    manifest = geo_extractor.ops_manifest(rows, geom_holes={"groups": [], "total": 0})

    table_counts = manifest.get("table", {})
    details = manifest.get("details", {})
    total_counts = manifest.get("total", {})
    text_info = manifest.get("text", {})

    assert table_counts.get("tap") == 4
    assert details.get("npt") == 4
    assert total_counts.get("tap") == 4
    assert text_info.get("estimated_total_drills") == 0


def test_manifest_reconcile_subtracts_sized_drills() -> None:
    rows = [{"qty": 4, "desc": '\"R\" (.339Ø) DRILL THRU'}]
    geom = {"groups": [{"dia_in": 0.339, "count": 77}], "total": 77}

    manifest = geo_extractor.ops_manifest(rows, geom_holes=geom)

    assert manifest.get("details", {}).get("drill_sized") == 4
    assert manifest.get("total", {}).get("drill") == 73
    assert manifest.get("text", {}).get("estimated_total_drills") == 4


def test_merge_table_lines_ignores_numeric_ladder_noise() -> None:
    merged = geo_extractor._merge_table_lines(
        [
            "(2) Ø0.250 DRILL THRU",
            "BREAK ALL .12 R",
            "1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9",
        ]
    )

    assert merged == ["(2) Ø0.250 DRILL THRU"]


def test_fallback_semicolon_row_remains_single_entry() -> None:
    result = geo_extractor._fallback_text_table(
        [
            "(4) .75Ø C'BORE AS SHOWN; \"R\" (.339Ø) DRILL THRU AS SHOWN; 1/8- N.P.T.",
        ]
    )

    rows = result.get("rows") or []
    assert len(rows) == 1
    assert rows[0].get("qty") == 4
    assert "1/8- N.P.T." in rows[0].get("desc", "")

    buckets, row_count, qty_sum = classify_chart_rows(rows)
    assert row_count == 1
    assert qty_sum == 4
    assert buckets.get("cbore") == 4
    assert buckets.get("npt") == 4
    assert buckets.get("tap") == 4


def test_anchor_height_filter_drops_small_text() -> None:
    entries = [
        {"normalized_text": "(2) Ø0.250 DRILL", "height": 0.20},
        {"normalized_text": "FROM FRONT", "height": 0.21},
        {"normalized_text": "(4) TAP 1/4-20", "height": 0.19},
        {"normalized_text": "1 1 2 2 3 3 4 4", "height": 0.06},
    ]

    anchor_height, anchor_count = geo_extractor._compute_anchor_height(entries)

    assert anchor_count == 2
    assert anchor_height == pytest.approx(0.195, rel=1e-3)

    filtered = geo_extractor._filter_entries_by_anchor_height(
        entries, anchor_height=anchor_height
    )
    filtered_texts = {entry.get("normalized_text") for entry in filtered}

    assert "(2) Ø0.250 DRILL" in filtered_texts
    assert "(4) TAP 1/4-20" in filtered_texts
    assert "FROM FRONT" in filtered_texts
    assert "1 1 2 2 3 3 4 4" not in filtered_texts


def test_follow_sheet_layout_scan_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    sheet_entities = [
        _DummyMText("(2) Ø0.250 DRILL THRU"),
        _DummyMText("(3) Ø0.312 TAP FROM FRONT"),
    ]
    follow_layout = _ToggleLayout(sheet_entities)
    doc = _FollowDoc([_DummyMText("SEE SHEET 2 FOR HOLE CHART")], follow_layout)

    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)
    monkeypatch.setattr(
        geo_extractor, "_extract_layer", lambda _entity: "BALLOON", raising=False
    )

    result = geo_extractor.read_text_table(doc)

    rows = result.get("rows") or []
    assert len(rows) == 2
    assert sorted(row.get("qty") for row in rows) == [2, 3]

    debug_snapshot = geo_extractor.get_last_text_table_debug() or {}
    follow_info = debug_snapshot.get("follow_sheet_info") or {}
    assert follow_info.get("texts") == 2


def test_default_text_layer_excludes_do_not_filter_am_bor() -> None:
    patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in geo_extractor.DEFAULT_TEXT_LAYER_EXCLUDE_REGEX
    ]

    assert any(pattern.search("AM_BOR") for pattern in patterns)


def test_normalize_layer_allowlist_adds_defaults() -> None:
    allowlist = geo_extractor._normalize_layer_allowlist(["AM_0"])

    assert allowlist is not None
    assert set(allowlist) == {"BALLOON", "AM_0"}


def test_normalize_layer_allowlist_respects_explicit_default() -> None:
    allowlist = geo_extractor._normalize_layer_allowlist(["BALLOON"])

    assert allowlist is not None
    assert tuple(allowlist) == ("BALLOON",)


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

    def fake_text(_doc: _DummyDoc, **_kwargs: object) -> dict[str, object]:
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


def test_anchor_wins_skips_roi_and_single_publish(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = _DummyDoc(
        [
            _DummyMText("(2) Ø0.250 DRILL THRU"),
            _DummyMText("(3) Ø0.312 TAP FROM FRONT"),
            _DummyMText("(4) Ø0.201 DRILL THRU"),
        ]
    )

    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)
    monkeypatch.setattr(geo_extractor, "read_acad_table", lambda *args, **kwargs: {})
    monkeypatch.setattr(geo_extractor, "extract_geometry", lambda _doc: {})
    monkeypatch.setattr(
        geo_extractor,
        "geom_hole_census",
        lambda _doc: {"groups": [], "total": 0},
    )

    geo_extractor.read_geo(doc)
    captured = capsys.readouterr().out

    assert captured.count("[TEXT-SCAN] pass=roi") == 1
    publish_lines = [
        line
        for line in captured.splitlines()
        if line.startswith("[PATH] publish=")
    ]
    assert len(publish_lines) == 1
    assert publish_lines[0].startswith("[PATH] publish=text_table")


def test_read_geo_raises_when_no_text_rows(
    monkeypatch: pytest.MonkeyPatch, fallback_doc: _DummyDoc
) -> None:
    monkeypatch.setattr(geo_extractor, "read_acad_table", lambda *args, **kwargs: {})
    monkeypatch.setattr(geo_extractor, "read_text_table", lambda *args, **kwargs: {})
    monkeypatch.setattr(geo_extractor, "extract_geometry", lambda _doc: {})
    geo_extractor._LAST_TEXT_TABLE_DEBUG = {"rows_txt_count": 0, "rows_txt_lines": []}

    with pytest.raises(geo_extractor.NoTextRowsError):
        geo_extractor.read_geo(fallback_doc)


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
    assert len(rows) == 2
    qty_sum = table_info.get("sum_qty")
    if qty_sum is None:
        qty_sum = geo_extractor._sum_qty(rows)
    assert qty_sum == sum(row["qty"] for row in rows) == 5

    combined_row = next(row for row in rows if "5/8-11 TAP" in row["desc"])
    assert ";" in combined_row["desc"]
    assert "Ø0.812 C'BORE" in combined_row["desc"]
    assert "17/32 DRILL THRU" in combined_row["desc"]

    npt_row = next(row for row in rows if "NPT TAP" in row["desc"])
    tap_tokens = {
        token.upper()
        for token in geo_extractor._CANDIDATE_TOKEN_RE.findall(combined_row["desc"])
    }
    npt_tokens = {
        token.upper() for token in geo_extractor._CANDIDATE_TOKEN_RE.findall(npt_row["desc"])
    }

    assert tap_tokens >= {"TAP", "FROM BACK", "C'BORE", "DRILL"}
    assert "TAP" in npt_tokens

    assert combined_row.get("side") == "back"


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

    result = geo_extractor.read_text_table(
        doc,
        layout_filters={"all_layouts": False, "patterns": ["Model"]},
    )

    rows = result.get("rows") or []
    assert len(rows) == 1
    assert rows[0].get("qty") == 1

    assert sheet_layout.query_calls >= 1


def test_unique_rows_in_order_dedupes_anchor_and_roi() -> None:
    anchor_rows = [
        {"qty": 2, "desc": "TAP 1/4-20"},
        {"qty": 2, "desc": "TAP 5/16-18"},
        {"qty": 4, "desc": "DRILL + NPT"},
    ]
    roi_rows = [
        {"qty": 2, "desc": "tap 1/4-20"},
        {"qty": 2, "desc": "tap 5/16-18"},
    ]

    merged, dropped = geo_extractor._unique_rows_in_order([anchor_rows, roi_rows])

    assert [row.get("qty") for row in merged] == [2, 2, 4]
    assert dropped == 2


def test_extract_row_quantity_requires_anchor_pattern() -> None:
    qty, remainder = geo_extractor._extract_row_quantity_and_remainder(
        "BREAK ALL EDGES .12 R"
    )

    assert qty is None
    assert "BREAK ALL" in remainder


def test_semicolon_row_remains_single_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geo_extractor, "_resolve_app_callable", lambda name: None)

    doc = _DummyDoc([_DummyMText("(4) COUNTERBORE; DRILL; NPT")])

    result = geo_extractor.read_text_table(doc)

    rows = result.get("rows") or []
    assert len(rows) == 1
    row = rows[0]
    assert row.get("qty") == 4
    desc_text = row.get("desc")
    assert isinstance(desc_text, str)
    assert desc_text.count(";") == 2
