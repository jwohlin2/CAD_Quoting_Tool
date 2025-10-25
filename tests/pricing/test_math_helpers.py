from __future__ import annotations

from typing import Any

import pytest

from cad_quoter.pricing import math_helpers
from cad_quoter.pricing.math_helpers import (
    _compute_direct_costs,
    _compute_pricing_ladder,
    _wieland_scrap_usd_per_lb,
    roughly_equal,
)


def test_compute_direct_costs_fallback_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(math_helpers, "_material_cost_components", lambda *_args, **_kwargs: None)
    total = _compute_direct_costs(
        100.0,
        10.0,
        5.0,
        {"material": 1.0, "packaging": "7.5"},
    )
    assert total == 102.5


def test_compute_direct_costs_uses_components(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_block: dict[str, Any] = {}

    def fake_components(block: dict[str, Any], overrides: Any = None, cfg: Any = None) -> dict[str, Any]:
        captured_block.update(block)
        return {"total_usd": 123.456}

    monkeypatch.setattr(math_helpers, "_material_cost_components", fake_components)
    total = _compute_direct_costs(
        50.0,
        5.0,
        2.0,
        {"misc": 10},
        material_detail={"scrap_credit_mass_lb": 2.0},
        scrap_price_source="Wieland",
    )
    assert total == 133.46
    assert captured_block["scrap_credit_source"] == "Wieland"
    assert captured_block["material_cost"] == 50.0


def test_compute_pricing_ladder_applies_percentages() -> None:
    totals = _compute_pricing_ladder(100.0, expedite_pct=0.1, margin_pct=0.2)
    assert totals == {
        "subtotal": 100.0,
        "with_expedite": 110.0,
        "with_margin": 132.0,
        "expedite_cost": 10.0,
        "subtotal_before_margin": 110.0,
    }


def test_roughly_equal_handles_tolerance() -> None:
    assert roughly_equal(10.0, 10.005, eps=0.01)
    assert not roughly_equal(10.0, 10.5, eps=0.01)


def test_wieland_scrap_normalizes_family(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_price(family: str) -> float:
        calls.append(family)
        prices = {
            "aluminum": 1.23,
            "stainless": 2.34,
        }
        return prices[family]

    monkeypatch.setattr("cad_quoter.pricing.wieland_scraper.get_scrap_price_per_lb", fake_price)

    assert _wieland_scrap_usd_per_lb("Stainless Steel") == pytest.approx(2.34)
    assert calls[-1] == "stainless"
    assert _wieland_scrap_usd_per_lb(None) == pytest.approx(1.23)
