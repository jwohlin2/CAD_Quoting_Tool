"""Shared fixtures for render tests."""

from __future__ import annotations

import copy
from typing import Any, Iterable

import pytest

from cad_quoter.render.state import RenderState as ModernRenderState
from cad_quoter.utils.render_state import RenderState as LegacyRenderState
from cad_quoter.utils.render_utils import QuoteDocRecorder
from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload

_DEFAULT_CURRENCY = "$"
_DEFAULT_PAGE_WIDTH = 74
_DEFAULT_DIVIDER = "-" * _DEFAULT_PAGE_WIDTH


@pytest.fixture
def dummy_quote_payload_components() -> dict[str, Any]:
    """Provide an isolated dummy quote payload and frequently used shortcuts."""

    payload = copy.deepcopy(_dummy_quote_payload())
    breakdown = payload.setdefault("breakdown", {})
    decision_state = payload.setdefault("decision_state", {})
    baseline = decision_state.setdefault("baseline", {})
    baseline_breakdown = baseline.setdefault("breakdown", {})

    material_block = breakdown.setdefault("material_block", {})
    material_detail = breakdown.setdefault("material_detail", {})
    material_cost_components = breakdown.setdefault("material_cost_components", {})
    material_overrides = breakdown.setdefault("material_overrides", {})
    material_selection = breakdown.setdefault("material_selected", {})
    nre_detail = breakdown.setdefault("nre_detail", {})

    pricing = payload.get("pricing")
    if not isinstance(pricing, dict):
        pricing = dict(breakdown.get("pricing") or {})
    breakdown.setdefault("pricing", pricing)
    payload["pricing"] = pricing

    ui_vars = payload.setdefault("ui_vars", {})

    return {
        "payload": payload,
        "breakdown": breakdown,
        "baseline": baseline,
        "baseline_breakdown": baseline_breakdown,
        "material_block": material_block,
        "material_detail": material_detail,
        "material_cost_components": material_cost_components,
        "material_overrides": material_overrides,
        "material_selection": material_selection,
        "nre_detail": nre_detail,
        "pricing": pricing,
        "ui_vars": ui_vars,
    }


@pytest.fixture
def legacy_render_state(dummy_quote_payload_components: dict[str, Any]) -> LegacyRenderState:
    """Build a legacy render state mirroring the v5 pipeline behaviour."""

    payload = copy.deepcopy(dummy_quote_payload_components["payload"])
    breakdown = payload.get("breakdown", {})
    lines: list[str] = []

    state = LegacyRenderState(
        result=payload,
        breakdown=breakdown,
        currency=_DEFAULT_CURRENCY,
        show_zeros=False,
        page_width=_DEFAULT_PAGE_WIDTH,
        divider=_DEFAULT_DIVIDER,
        lines=lines,
    )

    def append_line(value: Any) -> None:
        lines.append("" if value is None else str(value))

    def append_lines(values: Iterable[str]) -> None:
        for value in values:
            append_line(value)

    def write_wrapped(text: str, indent: str = "") -> None:
        if text in (None, ""):
            return
        for chunk in str(text).splitlines():
            append_line(f"{indent}{chunk}" if indent else chunk)

    state.append_line = append_line
    state.append_lines = append_lines
    state.write_wrapped = write_wrapped

    yield state


@pytest.fixture
def modern_render_state(dummy_quote_payload_components: dict[str, Any]) -> ModernRenderState:
    """Build a modern render state ready for section renderers."""

    payload = copy.deepcopy(dummy_quote_payload_components["payload"])

    state = ModernRenderState(
        payload=payload,
        currency=_DEFAULT_CURRENCY,
        show_zeros=False,
        page_width=_DEFAULT_PAGE_WIDTH,
    )
    state.lines = []
    state.recorder = QuoteDocRecorder(state.divider)

    yield state
