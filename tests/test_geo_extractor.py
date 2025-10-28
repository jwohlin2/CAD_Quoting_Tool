from __future__ import annotations

from collections import Counter
import importlib
import sys
import types

import pytest


@pytest.fixture()
def geo_extractor() -> types.ModuleType:
    """Load ``cad_quoter.geo_extractor`` with a geometry stub and restore afterwards."""

    geometry_stub = types.ModuleType("cad_quoter.geometry")
    geometry_stub.convert_dwg_to_dxf = lambda *_args, **_kwargs: None
    geometry_stub.detect_units_scale = lambda *_args, **_kwargs: (1.0, "inch")

    original_geometry = sys.modules.get("cad_quoter.geometry")
    original_geometry_enrich = sys.modules.get("cad_quoter.geometry.dxf_enrich")
    original_geo_extractor = sys.modules.get("cad_quoter.geo_extractor")

    sys.modules.pop("cad_quoter.geo_extractor", None)

    sys.modules["cad_quoter.geometry"] = geometry_stub
    sys.modules["cad_quoter.geometry.dxf_enrich"] = geometry_stub

    module = importlib.import_module("cad_quoter.geo_extractor")

    try:
        yield module
    finally:
        if original_geometry is not None:
            sys.modules["cad_quoter.geometry"] = original_geometry
        else:
            sys.modules.pop("cad_quoter.geometry", None)

        if original_geometry_enrich is not None:
            sys.modules["cad_quoter.geometry.dxf_enrich"] = original_geometry_enrich
        else:
            sys.modules.pop("cad_quoter.geometry.dxf_enrich", None)

        if original_geo_extractor is not None:
            sys.modules["cad_quoter.geo_extractor"] = original_geo_extractor
        else:
            sys.modules.pop("cad_quoter.geo_extractor", None)


def _collect_rows(geo_extractor_module: types.ModuleType, lines: list[str]):
    fallback = geo_extractor_module._fallback_text_table(lines)
    rows = fallback.get("rows") or []
    total = int(fallback.get("hole_count") or 0)
    return rows, total


def test_text_fallback_rows_stitched_and_split(geo_extractor: types.ModuleType) -> None:
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

    rows, total = _collect_rows(geo_extractor, lines)

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



def test_geom_residual_and_pilot_logic(geo_extractor: types.ModuleType) -> None:
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
