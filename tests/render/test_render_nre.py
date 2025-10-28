from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from cad_quoter.render import RenderState as ModernRenderState
from cad_quoter.render.nre import render_nre as modern_render_nre
from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import QuoteDocRecorder
from cad_quoter.utils.text_rules import canonicalize_amortized_label
from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload
from tests.test_render_material_section import _load_appv5


def _ensure_legacy_dependencies(monkeypatch: pytest.MonkeyPatch, appV5: Any) -> None:
    """Install helpers that the legacy renderer expects at runtime."""

    from cad_quoter.render import detect_planner_drilling, has_planner_drilling
    from cad_quoter.render.config import apply_render_overrides, ensure_mutable_breakdown
    from cad_quoter.render.guards import render_drilling_guard
    from cad_quoter.render.writer import QuoteWriter as LegacyWriter
    from cad_quoter.utils.render_utils import QuoteDocRecorder as LegacyRecorder

    monkeypatch.setattr(appV5, "apply_render_overrides", apply_render_overrides, raising=False)
    monkeypatch.setattr(appV5, "ensure_mutable_breakdown", ensure_mutable_breakdown, raising=False)
    monkeypatch.setattr(appV5, "QuoteWriter", LegacyWriter, raising=False)
    monkeypatch.setattr(appV5, "QuoteDocRecorder", LegacyRecorder, raising=False)
    monkeypatch.setattr(appV5, "detect_planner_drilling", detect_planner_drilling, raising=False)
    monkeypatch.setattr(appV5, "has_planner_drilling", has_planner_drilling, raising=False)
    monkeypatch.setattr(appV5, "render_drilling_guard", render_drilling_guard, raising=False)


def _extract_nre_section(rendered: str) -> list[str]:
    lines = rendered.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == "NRE / Setup Costs (per lot)":
            start = index
            break
    if start is None:
        return []
    collected: list[str] = []
    for line in lines[start:]:
        collected.append(line)
        if line.strip() == "" and len(collected) > 1:
            break
    return collected


def _build_modern_state(payload: dict[str, Any], *, show_zeros: bool) -> ModernRenderState:
    state = ModernRenderState(payload, page_width=74, show_zeros=show_zeros)
    recorder = QuoteDocRecorder(state.divider)
    writer = QuoteWriter(
        divider=state.divider,
        page_width=state.page_width,
        currency=state.currency,
        recorder=recorder,
    )
    state.recorder = recorder
    state.lines = writer.lines
    setattr(state, "writer", writer)
    return state


def _mutate_engineer_fallback(payload: dict[str, Any]) -> dict[str, Any]:
    programming = payload["breakdown"]["nre_detail"]["programming"]
    programming["prog_hr"] = 0.0
    programming["eng_hr"] = 1.5
    programming.pop("prog_rate", None)
    rates = payload["breakdown"].setdefault("rates", {})
    rates["ProgrammingRate"] = 62.5
    nre = payload["breakdown"]["nre"]
    nre["programming_per_lot"] = 0.0
    nre["programming_per_part"] = 0.0
    nre["programming_hr"] = 0.0
    nre["programming_cost"] = 0.0
    return payload


def _mutate_zero_programming(payload: dict[str, Any]) -> dict[str, Any]:
    programming = payload["breakdown"]["nre_detail"]["programming"]
    programming["prog_hr"] = 0.0
    programming["eng_hr"] = 0.0
    programming.pop("prog_rate", None)
    programming.pop("per_lot", None)
    nre = payload["breakdown"]["nre"]
    nre["programming_per_lot"] = 0.0
    nre["programming_per_part"] = 0.0
    nre["programming_hr"] = 0.0
    nre["programming_cost"] = 0.0
    return payload


def _mutate_missing_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    payload["breakdown"]["labor_costs"]["Fixture Build (amortized)"] = 9.5
    payload["breakdown"]["nre_detail"].pop("fixture", None)
    payload["breakdown"]["nre"].pop("fixture_per_part", None)
    return payload


@pytest.mark.parametrize(
    "scenario_name, mutate_payload, show_zeros",
    [
        ("baseline", lambda p: p, False),
        ("engineer_fallback", _mutate_engineer_fallback, False),
        ("zero_programming", _mutate_zero_programming, False),
        ("missing_fixture", _mutate_missing_fixture, False),
    ],
)
def test_render_nre_matches_legacy(
    monkeypatch: pytest.MonkeyPatch,
    scenario_name: str,
    mutate_payload: Callable[[dict[str, Any]], dict[str, Any]],
    show_zeros: bool,
) -> None:
    appV5 = _load_appv5(monkeypatch)
    _ensure_legacy_dependencies(monkeypatch, appV5)

    base_payload = _dummy_quote_payload()
    mutated_payload = mutate_payload(copy.deepcopy(base_payload))

    legacy_payload = copy.deepcopy(mutated_payload)
    modern_payload = copy.deepcopy(mutated_payload)

    legacy_rendered = appV5.render_quote(legacy_payload, currency="$", show_zeros=show_zeros)
    legacy_section = _extract_nre_section(legacy_rendered)

    modern_state = _build_modern_state(modern_payload, show_zeros=show_zeros)
    modern_section = modern_render_nre(modern_state)

    assert modern_section == legacy_section

    legacy_nre = legacy_payload["breakdown"]["nre"]
    programming_hours = float(legacy_nre.get("programming_hr") or 0.0)
    programming_per_lot = float(legacy_nre.get("programming_per_lot") or 0.0)
    expected_programming_rate = 0.0
    if programming_hours > 0 and programming_per_lot > 0:
        expected_programming_rate = programming_per_lot / programming_hours
    else:
        rates_map = legacy_payload["breakdown"].get("rates", {}) or {}
        programmer_rate = 0.0
        for key in ("ProgrammerRate", "ProgrammingRate"):
            try:
                programmer_rate = float(rates_map.get(key) or 0.0)
            except Exception:
                programmer_rate = 0.0
            if programmer_rate > 0:
                break
        if programmer_rate <= 0:
            programmer_rate = modern_state.default_labor_rate
        expected_programming_rate = programmer_rate

    expected_programming_per_part = float(legacy_nre.get("programming_per_part") or 0.0)
    expected_fixture_per_part = float(legacy_nre.get("fixture_per_part") or 0.0)

    assert modern_state.programming_rate == pytest.approx(expected_programming_rate)
    assert modern_state.programming_per_part == pytest.approx(expected_programming_per_part)
    assert modern_state.fixture_per_part == pytest.approx(expected_fixture_per_part)

    legacy_labor_raw = legacy_payload["breakdown"].get("labor_costs", {}) or {}
    expected_labor_totals: dict[str, float] = {}
    expected_amortized_totals: dict[str, float] = {}
    for label, value in legacy_labor_raw.items():
        try:
            numeric = float(value)
        except Exception:
            continue
        canonical_label, is_amortized = canonicalize_amortized_label(label)
        key = canonical_label or str(label)
        expected_labor_totals[key] = expected_labor_totals.get(key, 0.0) + numeric
        if is_amortized and numeric > 0:
            expected_amortized_totals[key] = expected_amortized_totals.get(key, 0.0) + numeric

    assert set(modern_state.labor_cost_totals) == set(expected_labor_totals)
    for key, value in expected_labor_totals.items():
        assert modern_state.labor_cost_totals[key] == pytest.approx(value)

    assert set(modern_state.amortized_totals) == set(expected_amortized_totals)
    for key, value in expected_amortized_totals.items():
        assert modern_state.amortized_totals[key] == pytest.approx(value)

    if expected_amortized_totals:
        expected_amortized_total = sum(expected_amortized_totals.values())
    else:
        expected_amortized_total = expected_programming_per_part + expected_fixture_per_part
    assert modern_state.amortized_nre_total == pytest.approx(expected_amortized_total)
