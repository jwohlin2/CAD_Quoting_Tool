from __future__ import annotations

from types import MappingProxyType
from types import ModuleType
from cad_quoter.render.config import apply_render_overrides, ensure_mutable_breakdown
from cad_quoter.ui.services import QuoteConfiguration


def test_apply_render_overrides_sets_expected_defaults() -> None:
    params = {"foo": "bar"}

    cfg = apply_render_overrides(None, default_params=params)

    assert isinstance(cfg, QuoteConfiguration)
    assert cfg.prefer_removal_drilling_hours is True
    assert cfg.separate_machine_labor is True
    assert cfg.machine_rate_per_hr == 45.0
    assert cfg.labor_rate_per_hr == 45.0
    assert cfg.milling_attended_fraction == 1.0
    assert cfg.default_params is not params
    assert cfg.default_params == params


def test_apply_render_overrides_falls_back_when_setattr_fails() -> None:
    params = {"foo": "bar"}

    class FrozenConfig:
        def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover - simple stub
            raise AttributeError(name)

    cfg = apply_render_overrides(FrozenConfig(), default_params=params)

    assert isinstance(cfg, QuoteConfiguration)
    assert cfg.machine_rate_per_hr == 45.0


def test_ensure_mutable_breakdown_returns_original_mapping_when_mutable() -> None:
    original: dict[str, int] = {"a": 1}

    view, mutable = ensure_mutable_breakdown(original)

    assert view is original
    assert mutable is original


def test_ensure_mutable_breakdown_clones_when_immutable() -> None:
    proxy = MappingProxyType({"a": 1})

    view, mutable = ensure_mutable_breakdown(proxy)

    assert view is mutable
    assert isinstance(mutable, dict)
    mutable["b"] = 2
    assert "b" not in proxy


def test_render_quote_applies_overrides_and_mutable_breakdown(
    appv5_module: ModuleType,
) -> None:
    breakdown_proxy = MappingProxyType({"totals": {"labor_cost": 0.0}})
    payload = {
        "summary": {
            "qty": 1,
            "final_price": 10.0,
            "unit_price": 10.0,
            "subtotal_before_margin": 9.0,
            "margin_pct": 0.1,
        },
        "price_drivers": [],
        "cost_breakdown": {},
        "breakdown": breakdown_proxy,
    }

    cfg = QuoteConfiguration(default_params={})
    cfg.machine_rate_per_hr = 100.0
    cfg.separate_machine_labor = False

    text = appv5_module.render_quote(payload, currency="$", cfg=cfg)

    assert "$" in text
    assert cfg.machine_rate_per_hr == 45.0
    assert cfg.separate_machine_labor is True
    assert isinstance(payload["breakdown"], dict)
