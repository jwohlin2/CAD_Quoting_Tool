from __future__ import annotations

from typing import Any

import pytest

from cad_quoter.geometry import dxf_enrich


@pytest.fixture()
def sample_proxy_rows() -> list[dict[str, Any]]:
    return [
        {
            "layout": "MODEL",
            "layer": "0",
            "etype": "PROXYTEXT",
            "text": "HOLE TABLE HOLE A B REF \\U+22050.2010 \\U+22050.2570 QTY 4 2 DESCRIPTION",
            "block_path": ["BLOCKS", "CHART"],
        },
        {
            "layout": "MODEL",
            "layer": "0",
            "etype": "PROXYTEXT",
            "text": "\\U+22050.2010 THRU FROM BACK",
            "block_path": ["BLOCKS", "CHART"],
        },
        {
            "layout": "MODEL",
            "layer": "0",
            "etype": "PROXYTEXT",
            "text": "\\U+22050.2570 1/4-20 TAP FROM FRONT",
            "block_path": ["BLOCKS", "CHART"],
        },
    ]


def _fake_ops(rows: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "hole": "A",
            "ref": "∅0.2010",
            "qty": 4,
            "desc": "∅0.2010 THRU FROM BACK",
            "type": "drill",
            "thru": True,
            "side": "BACK",
            "diameter_in": 0.201,
        },
        {
            "hole": "B",
            "ref": "∅0.2570",
            "qty": 2,
            "desc": "1/4-20 TAP FROM FRONT",
            "type": "tap",
            "side": "FRONT",
        },
    ]


def test_harvest_hole_table_uses_geo_dump_helpers(monkeypatch: pytest.MonkeyPatch, sample_proxy_rows: list[dict[str, Any]]) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_collect_all_text(_doc: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return sample_proxy_rows

    def _fake_explode(rows: list[str]) -> list[dict[str, Any]]:
        captured["input"] = list(rows)
        return _fake_ops(rows)

    monkeypatch.setattr(dxf_enrich, "collect_all_text", _fake_collect_all_text)
    monkeypatch.setattr(dxf_enrich, "explode_rows_to_operations", _fake_explode)

    result = dxf_enrich.harvest_hole_table(object())

    expected_lines = [
        "HOLE TABLE HOLE A B REF ∅0.2010 ∅0.2570 QTY 4 2 DESCRIPTION",
        "∅0.2010 THRU FROM BACK",
        "∅0.2570 1/4-20 TAP FROM FRONT",
    ]

    assert captured["input"] == expected_lines
    assert result["chart_lines"] == expected_lines
    assert result["holes_from_back"] is True
    assert result["tap_qty"] >= 2
    assert result["structured"] == [
        {"HOLE": "A", "REF_DIAM": "∅0.2010", "QTY": 4, "DESCRIPTION": "THRU FROM BACK"},
        {"HOLE": "B", "REF_DIAM": "∅0.2570", "QTY": 2, "DESCRIPTION": "1/4-20 TAP FROM FRONT"},
    ]


def test_build_geo_from_doc_with_proxy_text(monkeypatch: pytest.MonkeyPatch, sample_proxy_rows: list[dict[str, Any]]) -> None:
    class _Doc:
        def __init__(self) -> None:
            self.header = {"$INSUNITS": 1}

    doc = _Doc()

    monkeypatch.setattr(dxf_enrich, "collect_all_text", lambda _doc, **_kwargs: sample_proxy_rows)
    monkeypatch.setattr(dxf_enrich, "explode_rows_to_operations", lambda rows: _fake_ops(rows))
    monkeypatch.setattr(dxf_enrich, "harvest_plate_dimensions", lambda _doc, _scale: {"plate_len_in": 1.0, "plate_wid_in": 0.5, "prov": "mock"})
    monkeypatch.setattr(dxf_enrich, "harvest_outline_metrics", lambda _doc, _scale: {"edge_len_in": 10.0, "outline_area_in2": 5.0, "prov": "mock"})
    monkeypatch.setattr(dxf_enrich, "harvest_hole_geometry", lambda _doc, _scale: {"hole_count_geom": 0, "hole_diam_families_in": None, "min_hole_in": None, "max_hole_in": None, "prov": "mock"})
    monkeypatch.setattr(dxf_enrich, "harvest_title_notes", lambda _doc: {"material_note": None, "finishes": [], "default_tol": None, "revision": None, "prov": "mock"})

    geo = dxf_enrich.build_geo_from_doc(doc)

    expected_lines = [
        "HOLE TABLE HOLE A B REF ∅0.2010 ∅0.2570 QTY 4 2 DESCRIPTION",
        "∅0.2010 THRU FROM BACK",
        "∅0.2570 1/4-20 TAP FROM FRONT",
    ]

    assert geo["holes_from_back"] is True
    assert geo["hole_table"]["lines"] == expected_lines
    assert geo["hole_table"]["structured"] == [
        {"HOLE": "A", "REF_DIAM": "∅0.2010", "QTY": 4, "DESCRIPTION": "THRU FROM BACK"},
        {"HOLE": "B", "REF_DIAM": "∅0.2570", "QTY": 2, "DESCRIPTION": "1/4-20 TAP FROM FRONT"},
    ]
    ops = geo["hole_table"]["ops"]
    assert any(op.get("hole") == "A" and op.get("side") == "BACK" for op in ops)
    assert any(op.get("hole") == "B" and str(op.get("type")).lower() == "tap" for op in ops)
    assert geo["hole_table"]["summary"]["hole_count_ops"] == 6

