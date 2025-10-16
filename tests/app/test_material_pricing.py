from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest


for _module_name in ("requests", "bs4", "lxml"):
    if _module_name not in sys.modules:
        stub = types.ModuleType(_module_name)
        stub.__spec__ = importlib.machinery.ModuleSpec(_module_name, loader=None)
        sys.modules[_module_name] = stub

import appV5


class _FailingPricingEngine:
    def get_usd_per_kg(self, *args, **kwargs):  # noqa: D401 - simple stub
        raise RuntimeError("metals api unavailable")


def test_compute_material_cost_prefers_mcmaster_before_wieland(monkeypatch):
    def _fake_mcmaster(name: str, *, unit: str = "kg") -> tuple[float, str]:
        assert unit == "kg"
        return 123.45, "mcmaster:aluminum"

    def _fail_wieland(_keys):
        raise AssertionError("wieland lookup should not run when McMaster succeeds")

    monkeypatch.setattr(appV5, "_get_mcmaster_unit_price", _fake_mcmaster)
    monkeypatch.setattr(appV5, "lookup_wieland_price", _fail_wieland)

    cost, detail = appV5.compute_material_cost(
        material_name="Aluminum",
        mass_kg=1.0,
        scrap_frac=0.0,
        overrides={},
        vendor_csv=None,
        pricing=_FailingPricingEngine(),
    )

    assert cost == pytest.approx(123.45)
    assert detail["unit_price_usd_per_kg"] == pytest.approx(123.45)
    assert detail["unit_price_source"] == "mcmaster:aluminum"
    assert detail["source"] == "mcmaster:aluminum"


def test_compute_material_cost_uses_resolver_when_providers_fail(monkeypatch):
    monkeypatch.setattr(appV5, "lookup_wieland_price", lambda _keys: (None, None))

    fallback_calls: list[tuple[str, str]] = []

    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        fallback_calls.append((name, unit))
        return 321.0, "backup_csv"

    monkeypatch.setattr(appV5, "_resolve_material_unit_price", _fake_resolver)

    cost, detail = appV5.compute_material_cost(
        material_name="6061",
        mass_kg=1.0,
        scrap_frac=0.0,
        overrides={},
        vendor_csv=None,
        pricing=_FailingPricingEngine(),
    )

    assert fallback_calls == [("6061", "kg")]
    assert cost == pytest.approx(321.0)
    assert detail["unit_price_usd_per_kg"] == pytest.approx(321.0)
    assert detail["unit_price_source"] == "backup_csv"
    assert detail["source"] == "backup_csv"


def test_material_price_helper_returns_fallback_price(monkeypatch):
    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        assert unit == "kg"
        return 400.0, "backup_csv"

    monkeypatch.setattr(appV5, "_resolve_material_unit_price", _fake_resolver)

    price_per_g, source = appV5._material_price_per_g_from_choice("Stainless Steel", {})

    assert price_per_g == pytest.approx(0.4)
    assert source == "backup_csv"


def test_compute_material_block_uses_price_resolver(monkeypatch):
    module = types.ModuleType("metals_api")

    def _raise(_material_key: str) -> float:
        raise RuntimeError("metals api offline")

    module.price_per_lb_for_material = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    calls: list[tuple[str, str]] = []

    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        calls.append((name, unit))
        assert unit == "lb"
        return 10.0, "resolver"

    monkeypatch.setattr(appV5, "_resolve_material_unit_price", _fake_resolver)

    geo_ctx = {
        "material_display": "Aluminum 6061",
        "thickness_in": 1.0,
        "outline_bbox": {"plate_len_in": 2.0, "plate_wid_in": 3.0},
    }

    block = appV5._compute_material_block(geo_ctx, "aluminum", 2.70, 0.1)

    assert calls == [("Aluminum 6061", "lb")]
    assert block["price_per_lb"] == pytest.approx(10.0)
    assert block["price_source"] == "resolver"

    start_lb = block["start_lb"]
    scrap_lb = block["scrap_lb"]
    assert scrap_lb >= start_lb * 0.1 - 1e-6
    assert block["net_lb"] == pytest.approx(start_lb - scrap_lb)
    assert block["total_material_cost"] == pytest.approx((start_lb - scrap_lb) * 10.0)


def test_compute_material_block_applies_supplier_min(monkeypatch):
    module = types.ModuleType("metals_api")
    module.price_per_lb_for_material = lambda _material_key: 2.0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    geo_ctx = {
        "material_display": "Steel",
        "thickness_in": 0.5,
        "outline_bbox": {"plate_len_in": 1.0, "plate_wid_in": 1.0},
        "supplier_min$": 75.0,
    }

    block = appV5._compute_material_block(geo_ctx, "steel", 7.85, 0.2)

    assert block["supplier_min$"] == pytest.approx(75.0)
    assert block["price_per_lb"] == pytest.approx(2.0)
    assert block["total_material_cost"] == pytest.approx(75.0)
