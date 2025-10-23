import types

import pytest

from cad_quoter.geometry import dxf_enrich


@pytest.mark.parametrize(
    "lines, expected",
    [
        (
            [
                "(4) .75 C'BORE AS SHOWN",
                "(6) SPOT DRILL 90°",
                "(2) JIG GRIND Ø.250",
            ],
            {"(4) .75 C'BORE AS SHOWN", "(6) SPOT DRILL 90°", "(2) JIG GRIND Ø.250"},
        ),
    ],
)
def test_harvest_hole_table_includes_spot_and_jig(monkeypatch, lines, expected):
    def fake_iter_table_text(_doc):
        return iter(lines)

    monkeypatch.setattr(dxf_enrich, "iter_table_text", fake_iter_table_text)

    result = dxf_enrich.harvest_hole_table(types.SimpleNamespace())
    assert set(result["chart_lines"]) == expected
