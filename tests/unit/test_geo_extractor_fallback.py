from __future__ import annotations

import sys
import types

_geometry_stub = types.ModuleType("cad_quoter.geometry")
_geometry_stub.convert_dwg_to_dxf = lambda *_args, **_kwargs: None
_geometry_stub.detect_units_scale = lambda *_args, **_kwargs: (1.0, "inch")

sys.modules.setdefault("cad_quoter.geometry", _geometry_stub)
sys.modules.setdefault("cad_quoter.geometry.dxf_enrich", _geometry_stub)

from cad_quoter import geo_extractor


def _collect_rows(helper, lines: list[str]):
    result = helper(lines)
    assert result, "expected fallback helper to return rows"
    rows = result.get("rows") or []
    return rows, int(result.get("hole_count") or 0)


def test_publish_fallback_from_rows_txt_multiaction() -> None:
    lines = [
        "(4) 1/4-20 TAP FROM BACK",
        "; .201 DRILL FROM BACK",
        "(6) COUNTERBORE Ø0.750",
        "X 0.25 DEEP FROM FRONT",
        "(3) COUNTERDRILL .562 FROM FRONT",
        "BREAK ALL .12 R",
        "1 1 2 2 3 3",
        "(2) 3/8 - NPT TAP FROM FRONT",
    ]

    publish_rows, publish_total = _collect_rows(
        geo_extractor._publish_fallback_from_rows_txt, lines
    )
    table_rows, table_total = _collect_rows(geo_extractor._fallback_text_table, lines)

    for rows, total in ((publish_rows, publish_total), (table_rows, table_total)):
        assert len(rows) == 5
        assert total == 19

        classified = {row["desc"]: geo_extractor.classify_action(row["desc"]) for row in rows}

        tap_row = next(row for row in rows if classified[row["desc"]]["kind"] == "tap" and not classified[row["desc"]]["npt"])
        drill_row = next(row for row in rows if classified[row["desc"]]["kind"] == "drill")
        cbore_row = next(
            row for row in rows if classified[row["desc"]]["kind"] == "counterbore"
        )
        cdrill_row = next(
            row for row in rows if classified[row["desc"]]["kind"] == "counterdrill"
        )
        npt_row = next(row for row in rows if classified[row["desc"]]["kind"] == "tap" and classified[row["desc"]]["npt"])

        assert tap_row["qty"] == 4
        assert tap_row.get("side") == "back"

        assert drill_row["qty"] == 4
        assert drill_row.get("side") == "back"

        assert cbore_row["qty"] == 6
        assert cbore_row.get("side") == "front"

        assert cdrill_row["qty"] == 3
        assert cdrill_row.get("side") == "front"

        assert npt_row["qty"] == 2
        assert npt_row.get("side") == "front"
        assert npt_row.get("npt") is True
        assert classified[npt_row["desc"]].get("tap_type") == "pipe"

        notes = [row for row in rows if "BREAK" in row["desc"].upper()]
        assert not notes, "notes should be ignored"

