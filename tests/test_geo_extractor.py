from __future__ import annotations

from collections import Counter
import sys
import types

_geometry_stub = types.ModuleType("cad_quoter.geometry")
_geometry_stub.convert_dwg_to_dxf = lambda *_args, **_kwargs: None
_geometry_stub.detect_units_scale = lambda *_args, **_kwargs: (1.0, "inch")

sys.modules.setdefault("cad_quoter.geometry", _geometry_stub)
sys.modules.setdefault("cad_quoter.geometry.dxf_enrich", _geometry_stub)

from cad_quoter import geo_extractor


def _collect_rows(lines: list[str]):
    fallback = geo_extractor._fallback_text_table(lines)
    rows = fallback.get("rows") or []
    total = int(fallback.get("hole_count") or 0)
    return rows, total


def test_text_fallback_rows_stitched_and_split() -> None:
    lines = [
        "(12) 1/4-20 TAP FROM BACK",
        "; .201 DRILL FROM BACK",
        "(8) 1/2-13 TAP FROM FRONT",
        "; .421 DRILL FROM FRONT",
        "(6) COUNTERBORE Ø0.750 FROM FRONT",
        "; .500 DRILL FROM FRONT",
        "(5) COUNTERDRILL .562 FROM BACK",
        "; .281 DRILL FROM BACK",
        "(4) 3/8-16 TAP FROM FRONT",
        "; .312 DRILL FROM FRONT",
    ]

    rows, total = _collect_rows(lines)

    assert len(rows) >= 10
    assert total == geo_extractor._sum_qty(rows) == 70

    kind_counts: Counter[str] = Counter()
    sides = {row.get("side") for row in rows}

    for row in rows:
        action = geo_extractor.classify_action(row["desc"])
        kind = str(action.get("kind") or "").lower()
        kind_counts[kind] += 1
        assert row.get("qty") > 0
        assert row.get("side") in {"front", "back"}

    assert kind_counts["tap"] == 3
    assert kind_counts["drill"] == 5
    assert kind_counts["counterbore"] == 1
    assert kind_counts["counterdrill"] == 1
    assert sides == {"front", "back"}


def test_text_fallback_full_table() -> None:
    lines = [
        "(8) 1/4-20 TAP THRU FROM FRONT",
        "(8) #7 DRILL THRU FROM FRONT",
        "(6) 3/8-16 TAP THRU FROM BACK",
        "(6) 5/16 DRILL THRU FROM BACK",
        "(10) JIG GRIND Ø.375 THRU",
        "(4) Ø0.750 C'BORE X .38 DEEP FROM BACK",
        "(4) Ø.500 DRILL THRU FROM BACK",
        "(12) Ø.281 DRILL THRU FROM FRONT",
        "(5) C'DRILL .562 FROM FRONT",
        "(5) Ø.201 DRILL THRU FROM FRONT",
        "(9) SPOT DRILL .125 DEEP",
        "(8) #30 DRILL THRU",
        "(4) JIG GRIND Ø.500",
    ]

    rows, total = _collect_rows(lines)

    assert len(rows) >= 12
    assert 80 <= geo_extractor._sum_qty(rows) <= 90
    assert 80 <= total <= 90

    manifest = geo_extractor.ops_manifest(rows)
    table_totals = manifest["table"]

    assert table_totals["tap"] > 0
    assert table_totals["jig_grind"] > 0
    assert table_totals["counterdrill"] > 0


def test_geom_residual_and_pilot_logic() -> None:
    rows = [
        {"qty": 5, "desc": "1/4-20 TAP FROM FRONT"},
        {"qty": 2, "desc": "3/8-16 TAP FROM BACK"},
    ]

    geom_payload = {
        "total": 10,
        "groups": [],
        "residual_centers": [{"center": (float(idx), 0.0)} for idx in range(6)],
    }

    manifest = geo_extractor.ops_manifest(rows, geom_holes=geom_payload)

    table_tap = manifest["table"]["tap"]
    geom_residual = manifest["geom"]["drill_residual"]

    assert manifest["table"]["drill_only"] == 0
    assert geom_residual == 6
    assert table_tap == 7
    assert manifest["total"]["drill"] == geom_residual + table_tap


def test_table_authoritative_no_pilot_add() -> None:
    rows = [
        {"qty": 4, "desc": "1/4-20 TAP THRU", "ref": "A"},
        {"qty": 4, "desc": "#7 DRILL THRU", "ref": "A"},
    ]

    manifest = geo_extractor.ops_manifest(rows)

    assert manifest["table"]["drill"] == 4
    assert manifest["details"].get("drill_implied_from_taps") == 0
    assert manifest["total"]["drill"] == manifest["table"]["drill"]
