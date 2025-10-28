from __future__ import annotations

import pytest

from cad_quoter.render.payloads import (
    _render_as_float,
    build_cost_breakdown_payload,
    build_price_drivers_payload,
    build_summary_payload,
)


def test_build_summary_payload_returns_expected_structure() -> None:
    summary_payload, metrics = build_summary_payload(
        quote_qty=10,
        subtotal_before_margin="100.456",
        price="150.789",
        applied_pcts={"MarginPct": "12.5", "ExpeditePct": 15},
        expedite_cost="7.4",
        breakdown={"total_labor_cost": "45.1"},
        ladder_labor="55.5",
        total_direct_costs_value="30.75",
        currency="$",
    )

    assert summary_payload == {
        "qty": 10,
        "final_price": 150.79,
        "unit_price": 150.79,
        "subtotal_before_margin": 100.46,
        "margin_pct": 12.5,
        "margin_amount": 50.33,
        "expedite_pct": 15.0,
        "expedite_amount": 7.4,
        "currency": "$",
    }

    assert metrics.keys() == {
        "subtotal_before_margin",
        "final_price",
        "margin_amount",
        "expedite_amount",
        "labor_total_amount",
        "direct_total_amount",
    }
    assert metrics["subtotal_before_margin"] == pytest.approx(100.456)
    assert metrics["final_price"] == pytest.approx(150.789)
    assert metrics["margin_amount"] == pytest.approx(50.333)
    assert metrics["expedite_amount"] == pytest.approx(7.4)
    assert metrics["labor_total_amount"] == pytest.approx(45.1)
    assert metrics["direct_total_amount"] == pytest.approx(30.75)


def test_build_summary_payload_handles_fallbacks() -> None:
    summary_payload, metrics = build_summary_payload(
        quote_qty="2.5",
        subtotal_before_margin=None,
        price=None,
        applied_pcts={},
        expedite_cost=None,
        breakdown={},
        ladder_labor="12.5",
        total_direct_costs_value=None,
        currency="€",
    )

    assert summary_payload["qty"] == 2.5
    assert summary_payload["margin_pct"] == 0.0
    assert summary_payload["expedite_pct"] == 0.0
    assert summary_payload["currency"] == "€"
    assert metrics["labor_total_amount"] == 12.5
    assert metrics["direct_total_amount"] == 0.0


def test_build_price_drivers_payload_deduplicates_entries() -> None:
    payload = build_price_drivers_payload(
        [" First reason ", "second reason", "first reason"],
        ["second reason", "Third"],
    )

    assert payload == [
        {"detail": "First reason"},
        {"detail": "second reason"},
        {"detail": "Third"},
    ]


def test_build_cost_breakdown_payload_matches_legacy_structure() -> None:
    payload = build_cost_breakdown_payload(
        labor_total_amount="45.10",
        direct_total_amount="30.75",
        expedite_amount="7.40",
        subtotal_before_margin="100.456",
        margin_amount="50.333",
        final_price="150.789",
    )

    assert payload == [
        ("Machine & Labor", 45.1),
        ("Direct Costs", 30.75),
        ("Expedite", 7.4),
        ("Subtotal before Margin", 100.46),
        ("Margin", 50.33),
        ("Final Price", 150.79),
    ]


def test_build_cost_breakdown_payload_omits_zero_expedite() -> None:
    payload = build_cost_breakdown_payload(
        labor_total_amount=0,
        direct_total_amount=0,
        expedite_amount=0,
        subtotal_before_margin=0,
        margin_amount=0,
        final_price=0,
    )

    assert payload == [
        ("Machine & Labor", 0.0),
        ("Direct Costs", 0.0),
        ("Subtotal before Margin", 0.0),
        ("Margin", 0.0),
        ("Final Price", 0.0),
    ]


def test_render_as_float_handles_invalid_input() -> None:
    assert _render_as_float("not-a-number", 1.23) == 1.23
