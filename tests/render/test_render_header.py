from __future__ import annotations

from cad_quoter.app.quote_doc import build_quote_header_lines
from cad_quoter.render import RenderState
from cad_quoter.render.header import apply_pricing_source, render_header


def _create_state(payload: dict) -> RenderState:
    state = RenderState(payload, page_width=74)
    state.lines = []
    return state


def test_render_header_matches_quote_doc_builder() -> None:
    payload: dict[str, object] = {"qty": 1, "breakdown": {}}
    state = _create_state(payload)

    expected_lines, expected_value = build_quote_header_lines(
        qty=state.qty,
        result=state.result,
        breakdown=state.breakdown,
        page_width=state.page_width,
        divider=state.divider,
        process_meta=state.process_meta,
        process_meta_raw=state.process_meta_raw,
        hour_summary_entries=state.hour_summary_entries,
        cfg=state.cfg,
    )

    header_lines = render_header(state)

    assert header_lines == expected_lines
    assert state.pricing_source_value == expected_value


def test_render_header_applies_planner_metadata() -> None:
    payload: dict[str, object] = {
        "qty": 3,
        "breakdown": {"pricing_source": "planner"},
        "app_meta": {},
        "decision_state": {"baseline": {}},
    }
    state = _create_state(payload)

    header_lines = render_header(state)

    assert any(line == "Pricing Source: Planner" for line in header_lines)
    assert state.pricing_source_value == "planner"
    assert state.breakdown.get("pricing_source") == "planner"

    app_meta = state.result.get("app_meta")
    assert isinstance(app_meta, dict)
    assert app_meta.get("used_planner") is True

    decision_state = state.result.get("decision_state")
    assert isinstance(decision_state, dict)
    baseline = decision_state.get("baseline")
    assert isinstance(baseline, dict)
    assert baseline.get("pricing_source") == "planner"


def test_apply_pricing_source_removes_metadata_when_missing() -> None:
    payload: dict[str, object] = {
        "qty": 2,
        "breakdown": {"pricing_source": "legacy"},
        "decision_state": {"baseline": {"pricing_source": "legacy"}},
    }
    state = _create_state(payload)
    state.pricing_source_value = None

    apply_pricing_source(state, state.breakdown)
    apply_pricing_source(state, state.result)

    assert "pricing_source" not in state.breakdown

    decision_state = state.result.get("decision_state")
    assert isinstance(decision_state, dict)
    baseline = decision_state.get("baseline")
    assert isinstance(baseline, dict)
    assert "pricing_source" not in baseline
