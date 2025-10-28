from __future__ import annotations

from types import ModuleType

import pytest

from cad_quoter.render import RenderState, render_material, render_nre
from cad_quoter.utils.render_state import RenderState as LegacyRenderState


def _make_render_state() -> RenderState:
    payload = {
        "qty": 10,
        "breakdown": {
            "qty": 10,
            "totals": {},
            "rates": {},
            "nre": {"programming_hr": 2.0},
            "nre_detail": {
                "programming": {"prog_hr": 1.5, "eng_hr": 0.5, "prog_rate": 150.0},
                "fixture": {"build_hr": 1.0, "build_rate": 60.0},
            },
            "nre_cost_details": {},
            "labor_costs": {},
        },
    }
    state = RenderState(payload, page_width=72)
    state.lines = []
    state.recorder = None
    state.separate_labor_cfg = False
    state.cfg_labor_rate_value = 0.0
    return state


def _make_legacy_state() -> LegacyRenderState:
    result: dict[str, object] = {"breakdown": {}}
    breakdown = result["breakdown"]
    return LegacyRenderState(
        result=result,
        breakdown=breakdown,
        currency="$",
        show_zeros=False,
        page_width=72,
        divider="-" * 72,
    )


def test_render_material_returns_empty_list() -> None:
    state = _make_render_state()

    lines = render_material(state)

    assert lines == []


def test_render_nre_computes_programming_totals() -> None:
    state = _make_render_state()

    lines = render_nre(state)

    assert lines
    assert lines[0] == "NRE / Setup Costs (per lot)"
    assert any("Programming" in line for line in lines[1:])
    assert state.programming_rate == pytest.approx(150.0)
    assert state.programming_per_part > 0


def test_appv5_render_material_smoke(appv5_module: ModuleType) -> None:
    state = _make_legacy_state()

    lines = appv5_module.render_material(
        state,
        material={},
        material_stock_block={},
        material_selection={},
        material_display_label="Test Alloy",
        normalized_material_key="test-alloy",
        drilling_meta=None,
        ui_vars={},
        baseline=None,
        g={},
        geo_context=None,
        pricing_geom=None,
        pricing=None,
        material_detail_for_breakdown={},
        material_cost_components=None,
        material_overrides=None,
        show_material_shipping=False,
        shipping_total=0.0,
        shipping_source=None,
        material_warning_summary=False,
    )

    assert isinstance(lines, list)
    assert state.material_warning_needed is False
