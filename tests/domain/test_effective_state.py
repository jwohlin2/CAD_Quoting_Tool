from __future__ import annotations

import pytest

from cad_quoter.domain import (
    QuoteState,
    compute_effective_state,
    effective_to_overrides,
    merge_effective,
    reprice_with_effective,
)
from appV5 import apply_suggestions


def test_merge_effective_clamps_and_tracks_sources() -> None:
    baseline = {
        "process_hours": {"drilling": 2.0},
        "scrap_pct": 0.05,
        "_bounds": {"mult_max": 2.0, "adder_max_hr": 1.0},
    }
    suggestions = {
        "process_hour_multipliers": {"drilling": 3.5},
        "process_hour_adders": {"inspection": 2.5},
        "scrap_pct": 0.3,
    }
    overrides = {
        "process_hour_multipliers": {"drilling": 0.25},
        "process_hour_adders": {"inspection": 5.0},
        "scrap_pct": 0.4,
    }

    merged = merge_effective(baseline, suggestions, overrides)

    assert merged["process_hour_multipliers"]["drilling"] == pytest.approx(0.25)
    assert merged["process_hours"]["drilling"] == pytest.approx(0.5)
    assert merged["process_hour_adders"]["inspection"] == pytest.approx(1.0)
    assert merged["process_hours"]["inspection"] == pytest.approx(1.0)
    assert merged["scrap_pct"] == pytest.approx(0.25)

    clamp_notes = merged.get("_clamp_notes", [])
    assert any("adder[inspection]" in note for note in clamp_notes)
    assert any("scrap_pct" in note for note in clamp_notes)

    sources = merged.get("_source_tags", {})
    assert sources["process_hour_multipliers"]["drilling"] == "user"
    assert sources["process_hour_adders"]["inspection"] == "user"
    assert sources["scrap_pct"] == "user"


def test_compute_effective_state_respects_accept_flags() -> None:
    state = QuoteState()
    state.baseline = {"process_hours": {"milling": 2.0}, "scrap_pct": 0.05}
    state.suggestions = {
        "process_hour_multipliers": {"milling": 1.5, "turning": 2.0},
        "process_hour_adders": {"inspection": 0.75},
        "scrap_pct": 0.2,
    }
    state.accept_llm = {
        "process_hour_multipliers": {"milling": True, "turning": False},
        "process_hour_adders": {},
        "scrap_pct": True,
    }
    state.user_overrides = {}

    merged, sources = compute_effective_state(state)

    assert merged["process_hours"]["milling"] == pytest.approx(3.0)
    assert "turning" not in merged["process_hour_multipliers"]
    assert merged["scrap_pct"] == pytest.approx(0.2)
    assert sources["process_hour_multipliers"]["milling"] == "llm"
    assert sources["scrap_pct"] == "llm"


def test_merge_effective_caps_extreme_values() -> None:
    baseline = {"process_hours": {"milling": 2.0}}
    suggestions = {
        "process_hour_multipliers": {"milling": 12.0},
        "process_hour_adders": {"milling": 24.0},
    }

    merged = merge_effective(baseline, suggestions, {})

    assert merged["process_hour_multipliers"]["milling"] == pytest.approx(4.0)
    assert merged["process_hour_adders"]["milling"] == pytest.approx(8.0)
    assert merged["process_hours"]["milling"] == pytest.approx((2.0 * 4.0) + 8.0)
    clamp_notes = merged.get("_clamp_notes", [])
    assert any("multiplier[milling]" in note for note in clamp_notes)
    assert any("adder[milling]" in note for note in clamp_notes)


def test_reprice_with_effective_applies_drilling_floor() -> None:
    state = QuoteState()
    state.geo = {"hole_count": 8}
    state.baseline = {"process_hours": {"drilling": 0.001}}
    state.suggestions = {}
    state.user_overrides = {}
    state.accept_llm = {}

    reprice_with_effective(state)

    floor = (8 * 9.0) / 3600.0
    assert state.effective["process_hours"]["drilling"] == pytest.approx(floor)
    assert state.guard_context["hole_count"] == 8


def test_merge_effective_tracks_new_fields() -> None:
    baseline = {
        "fixture_build_hr": 0.5,
        "_bounds": {"scrap_max": 0.25},
    }
    suggestions = {
        "fixture_build_hr": 1.2,
        "soft_jaw_hr": 0.3,
        "packaging_flat_cost": 12.0,
        "shipping_cost": 30.0,
        "shipping_hint": "Foam inserts",
    }
    overrides = {"fai_required": True, "shipping_cost": 20.0}

    merged = merge_effective(baseline, suggestions, overrides)

    assert merged["fixture_build_hr"] == pytest.approx(1.2)
    assert merged["soft_jaw_hr"] == pytest.approx(0.3)
    assert merged["packaging_flat_cost"] == pytest.approx(12.0)
    assert merged["shipping_cost"] == pytest.approx(20.0)
    assert merged["fai_required"] is True
    assert merged["shipping_hint"] == "Foam inserts"


def test_effective_to_overrides_emits_new_keys() -> None:
    effective = {
        "fixture_build_hr": 1.0,
        "soft_jaw_hr": 0.25,
        "cmm_minutes": 18.0,
        "packaging_hours": 0.2,
        "shipping_cost": 18.0,
        "shipping_hint": "Double box",
    }
    overrides = effective_to_overrides(effective, {})

    assert overrides["fixture_build_hr"] == pytest.approx(1.0)
    assert overrides["soft_jaw_hr"] == pytest.approx(0.25)
    assert overrides["cmm_minutes"] == pytest.approx(18.0)
    assert overrides["packaging_hours"] == pytest.approx(0.2)
    assert overrides["shipping_cost"] == pytest.approx(18.0)
    assert overrides["shipping_hint"] == "Double box"


def test_apply_suggestions_round_trips_through_merge_effective() -> None:
    baseline = {
        "process_hours": {"milling": 2.0, "drilling": 1.0},
        "scrap_pct": 0.05,
        "setups": 1,
        "fixture": "standard",
    }
    suggestions = {
        "process_hour_multipliers": {"milling": 1.5},
        "process_hour_adders": {"drilling": 0.2},
        "scrap_pct": 0.12,
        "setups": 2,
        "fixture": "soft jaws",
        "notes": ["Increase inspection"],
        "no_change_reason": "baseline ok",
    }

    applied = apply_suggestions(baseline, suggestions)

    expected = merge_effective(baseline, suggestions, {})
    expected.pop("_source_tags", None)
    expected.pop("_clamp_notes", None)
    expected["_llm_notes"] = ["Increase inspection", "no_change: baseline ok"]

    assert applied == expected
