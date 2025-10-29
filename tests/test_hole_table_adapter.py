from __future__ import annotations

from typing import Any, Dict, List


def _build_record(text: str, etype: str = "PROXYTEXT", layout: str = "Model") -> Dict[str, Any]:
    return {"layout": layout, "etype": etype, "text": text}


def test_extract_hole_table_uses_geo_extractor(monkeypatch):
    from cad_quoter.geometry import hole_table_adapter as adapter

    fake_records: List[Dict[str, Any]] = [
        _build_record("  HOLE TABLE  \n"),
        _build_record(
            "HOLE   A    B   REF   Ø0.250   \\U+22050.201   QTY   2   4   DESCRIPTION",
        ),
        _build_record("∅0.201\nDRILL   FROM   BACK"),
        _build_record("Ø0.250  TAP\nFROM  FRONT"),
        _build_record("ignored text", layout="Layout1"),
    ]

    captured_exploded: Dict[str, Any] = {}

    def fake_collect(doc, **_kwargs):  # type: ignore[no-redef]
        fake_collect.called = True  # type: ignore[attr-defined]
        fake_collect.last_doc = doc  # type: ignore[attr-defined]
        return list(fake_records)

    def fake_explode(rows):  # type: ignore[no-redef]
        captured_exploded["rows"] = list(rows)
        return [("A", "Ø0.250", 2, "TAP"), ("B", "∅0.201", 4, "DRILL")]

    monkeypatch.setattr(adapter, "collect_all_text", fake_collect)
    monkeypatch.setattr(adapter, "explode_rows_to_operations", fake_explode)

    structured, ops = adapter.extract_hole_table_from_doc(object())

    assert getattr(fake_collect, "called", False) is True
    assert captured_exploded["rows"] == [
        "HOLE TABLE",
        "HOLE A B REF Ø0.250 ∅0.201 QTY 2 4 DESCRIPTION",
        "∅0.201 DRILL FROM BACK",
        "Ø0.250 TAP FROM FRONT",
    ]

    assert structured == [
        {"HOLE": "A", "REF_DIAM": "Ø0.250", "QTY": "2", "DESCRIPTION": "TAP FROM FRONT"},
        {"HOLE": "B", "REF_DIAM": "∅0.201", "QTY": "4", "DESCRIPTION": "DRILL FROM BACK"},
    ]
    assert ops == [("A", "Ø0.250", 2, "TAP"), ("B", "∅0.201", 4, "DRILL")]

