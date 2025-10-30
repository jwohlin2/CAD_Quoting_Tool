from __future__ import annotations

import json

import pytest

import appV5


def test_emit_hole_table_ops_cards_updates_summary() -> None:
    lines: list[str] = []
    geo = {
        "ops_summary": {
            "rows": [
                {
                    "qty": 6,
                    "desc": "6X TAP 1/4-20 THRU FROM FRONT",
                    "ref": "",
                },
                {
                    "qty": 3,
                    "desc": "Ø0.500 CBORE × .250 DEEP FROM BACK",
                    "ref": "",
                },
                {
                    "qty": 2,
                    "desc": "CENTER DRILL .187 DIA × .060 DEEP",
                    "ref": "",
                },
                {
                    "qty": 1,
                    "desc": "JIG GRIND Ø.375",
                    "ref": "",
                },
            ]
        }
    }

    appV5._emit_hole_table_ops_cards(lines, geo=geo, material_group="aluminum", speeds_csv=None)

    assert lines == []
    summary = geo["ops_summary"]
    assert summary.get("tap_minutes_total") is not None
    assert summary.get("tap_minutes_total", 0.0) > 0


def test_aggregate_ops_sets_built_rows() -> None:
    rows = [
        {"hole": "A1", "ref": "Ø0.257", "qty": 4, "desc": "Ø0.257 THRU"},
        {"hole": "A2", "ref": "", "qty": 2, "desc": "1/4-20 TAP THRU"},
    ]

    summary = appV5.aggregate_ops(rows)

    assert summary.get("built_rows") == 2


def test_emit_hole_table_ops_cards_updates_bucket_view() -> None:
    lines: list[str] = []
    geo = {
        "ops_summary": {
            "rows": [
                {"qty": 3, "desc": "1/4-20 TAP THRU", "ref": ""},
                {"qty": 2, "desc": "Ø0.3125 CBORE × .25 DEEP", "ref": ""},
                {"qty": 4, "desc": "CENTER DRILL .187 DIA × .060 DEEP", "ref": ""},
                {"qty": 1, "desc": "JIG GRIND Ø.375", "ref": ""},
            ],
            "totals": {
                "tap_front": 3,
                "cbore_front": 2,
                "spot_front": 4,
                "jig_grind": 1,
            },
        }
    }
    rates = {
        "TappingRate": 120.0,
        "CounterboreRate": 100.0,
        "DrillingRate": 90.0,
        "GrindingRate": 70.0,
        "LaborRate": 60.0,
    }
    breakdown: dict[str, object] = {"bucket_view": {}, "rates": rates}

    appV5._emit_hole_table_ops_cards(
        lines,
        geo=geo,
        material_group="aluminum",
        speeds_csv=None,
        breakdown=breakdown,
        rates=rates,
    )

    bucket_view = breakdown["bucket_view"]
    assert isinstance(bucket_view, dict)
    buckets = bucket_view.get("buckets", {})
    assert isinstance(buckets, dict)

    tapping = buckets.get("tapping")
    assert isinstance(tapping, dict)
    assert tapping["minutes"] == pytest.approx(0.5)
    assert tapping["machine$"] == pytest.approx(1.0)
    assert tapping["labor$"] == pytest.approx(0.5)
    assert tapping["total$"] == pytest.approx(1.5)

    ops = bucket_view.get("bucket_ops") or {}
    assert isinstance(ops, dict)
    tapping_ops = ops.get("tapping") or []
    if tapping_ops:
        assert any(entry.get("name") == "Tapping ops" for entry in tapping_ops)


def test_collect_ops_entries_uses_explode(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [["A", "Ø0.257", 4, "1/4-20 TAP FROM FRONT"]]

    monkeypatch.setattr(appV5, "_explode_rows_to_operations", lambda rows: entries)

    geo_map = {"hole_table": {"lines": ["HOLE TABLE", "A Ø0.257 4 1/4-20 TAP FROM FRONT"]}}

    result = appV5._collect_ops_entries_for_display(geo_map)

    assert result == [
        {"hole": "A", "ref": "Ø0.257", "qty": 4, "desc": "1/4-20 TAP FROM FRONT"}
    ]


def test_format_hole_table_section_outputs_operations() -> None:
    rows = [
        {"hole": "A", "ref": "Ø1.7500", "qty": 4, "desc": "±.0001 THRU (JIG GRIND)"},
        {"hole": "B", "ref": "Ø.7500", "qty": 2, "desc": "THRU"},
    ]

    lines = appV5._format_hole_table_section(rows)

    assert lines
    assert lines[0] == "HOLE TABLE OPERATIONS"
    assert any(line.strip().startswith("A") for line in lines[1:])
    assert any("THRU" in line for line in lines)


def test_collect_ops_entries_prefers_hole_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    summary_rows = [{"hole": "", "ref": "Ø.3320", "qty": 1, "desc": "THRU"}]
    exploded_rows = [
        ["G", "Ø.3320", 1, "THRU"],
        ["G", "Ø.3320", 1, "3/8-24 TAP X .38 DEEP FROM BACK"],
    ]

    monkeypatch.setattr(appV5, "_explode_rows_to_operations", lambda rows: exploded_rows)

    geo_map = {
        "ops_summary": {"rows": summary_rows},
        "hole_table": {"lines": ["HOLE TABLE", "G Ø.3320 1 THRU"]},
    }

    result = appV5._collect_ops_entries_for_display(geo_map)

    assert {"hole": "G", "ref": "Ø.3320", "qty": 1, "desc": "THRU"} in result
    assert {"hole": "", "ref": "Ø.3320", "qty": 1, "desc": "THRU"} not in result


def test_collect_ops_entries_matches_gold_sample() -> None:
    with open("tests/gold/hole_table_sample.jsonl", "r", encoding="utf-8") as fh:
        sample = json.loads(fh.readline())

    geo_map = {"hole_table": {"lines": sample["raw_lines"]}}

    expected = [
        {"hole": hole, "ref": ref, "qty": int(qty), "desc": desc}
        for hole, ref, qty, desc in sample["expected_ops"]
    ]

    result = appV5._collect_ops_entries_for_display(geo_map)

    assert sorted(result, key=lambda row: (row["hole"], row["ref"], row["desc"])) == sorted(
        expected, key=lambda row: (row["hole"], row["ref"], row["desc"])
    )
