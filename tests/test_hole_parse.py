"""Unit tests for GEO hole parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from cad_quoter import geo_extractor
from cad_quoter.app import hole_ops
from tools import geo_dump


@pytest.fixture()
def promoted_classifier() -> Callable[[str], str]:
    return geo_extractor._classify_promoted_row  # type: ignore[attr-defined]


@pytest.fixture()
def promoted_row_builder() -> Callable[[object], tuple[list[dict[str, object]], int]]:
    return geo_extractor._prepare_columnar_promoted_rows  # type: ignore[attr-defined]


def test_promoted_row_classifier_handles_core_tokens(promoted_classifier) -> None:
    cases = [
        ("1/4-20 TAP THRU", "tap"),
        ("Ø0.500 COUNTERBORE FROM BACK", "counterbore"),
        ("NOTE: SEE DETAIL", "note"),
        ("Ø0.250 DRILL THRU", "drill"),
    ]
    for desc, expected in cases:
        assert promoted_classifier(desc.upper()) == expected


def test_prepare_columnar_promoted_rows_orders_and_dedupes(promoted_row_builder) -> None:
    table_info = {
        "rows": [
            {"qty": "3", "desc": "(3) 1/4-20 TAP THRU", "ref": "A1", "hole": "H1"},
            {"qty": "3", "desc": "(3) 1/4-20 TAP THRU", "ref": "A1", "hole": "H1"},
            {"qty": 2, "desc": "Ø0.500 C'BORE FROM BACK", "ref": "B2", "side": "BACK"},
            {"qty": 4, "desc": "Ø0.281 DRILL THRU", "ref": "C3"},
            {"qty": 0, "desc": "Ø0.201 DRILL", "ref": "D4"},
        ]
    }

    rows, qty_sum = promoted_row_builder(table_info)

    assert qty_sum == 9
    assert [row["qty"] for row in rows] == [3, 2, 4]
    assert [row["desc"] for row in rows] == [
        "1/4-20 TAP THRU",
        "Ø0.500 C'BORE FROM BACK",
        "Ø0.281 DRILL THRU",
    ]
    assert rows[0]["hole"] == "H1"


def test_geo_dump_artifacts_are_stable(tmp_path: Path) -> None:
    hole_rows = [
        {
            "hole": "A1",
            "qty": 3,
            "ref": "Ø0.201",
            "desc": "1/4-20 TAP FROM FRONT",
        },
        {
            "hole": "B2",
            "qty": 2,
            "ref": "Ø0.500",
            "side": "BACK",
            "desc": "Ø0.500 C'BORE FROM BACK",
        },
        {
            "hole": "C3",
            "qty": 5,
            "ref": "Ø0.281",
            "desc": "Ø0.281 DRILL THRU",
        },
    ]

    geo: dict[str, object] = {}
    summary = hole_ops.update_geo_ops_summary_from_hole_rows(
        geo,
        summary_rows=hole_rows,
        ops_source="unit-test",
    )
    ops_summary = geo.get("ops_summary", {})

    qty_sum = sum(row["qty"] for row in summary["rows"])
    hole_total = sum(row["qty"] for row in summary["rows"])
    rows_payload = geo_dump._build_hole_rows_artifact(  # type: ignore[attr-defined]
        summary["rows"],
        qty_sum=qty_sum,
        hole_count=hole_total,
        provenance="HOLE TABLE",
        source=ops_summary.get("source"),
    )
    assert list(rows_payload.keys()) == [
        "rows",
        "qty_sum",
        "hole_count",
        "provenance",
        "source",
    ]
    assert [list(entry.keys()) for entry in rows_payload["rows"]][:2] == [
        ["hole", "qty", "ref", "desc"],
        ["hole", "qty", "ref", "side", "desc"],
    ]

    totals_payload = geo_dump._build_ops_totals_artifact(ops_summary)  # type: ignore[attr-defined]
    assert totals_payload is not None
    assert list(totals_payload.keys()) == [
        "totals",
        "actions_total",
        "back_ops_total",
        "flip_required",
        "source",
    ]
    assert list(totals_payload["totals"].keys()) == [
        "tap",
        "tap_front",
        "counterbore",
        "counterbore_back",
        "drill",
    ]

    rows_path = tmp_path / "hole_rows.json"
    totals_path = tmp_path / "op_totals.json"
    geo_dump._write_artifact(rows_path, rows_payload)  # type: ignore[attr-defined]
    geo_dump._write_artifact(totals_path, totals_payload)  # type: ignore[attr-defined]

    on_disk_rows = json.loads(rows_path.read_text())
    on_disk_totals = json.loads(totals_path.read_text())
    assert list(on_disk_rows.keys()) == list(rows_payload.keys())
    assert list(on_disk_totals["totals"].keys()) == list(totals_payload["totals"].keys())


def test_geo_dump_main_exits_when_no_text_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dummy_doc: object = object()
    dummy_path = tmp_path / "missing_text.dwg"

    def fake_loader(_path: Path, **_kwargs: object) -> object:
        return dummy_doc

    def fake_read_geo(_doc: object, **_kwargs: object) -> dict[str, object]:
        raise geo_extractor.NoTextRowsError()

    monkeypatch.setattr(geo_extractor, "_load_doc_for_path", fake_loader)
    monkeypatch.setattr(geo_dump, "read_geo", fake_read_geo)

    exit_code = geo_dump.main([str(dummy_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert geo_extractor.NO_TEXT_ROWS_MESSAGE in captured.out
