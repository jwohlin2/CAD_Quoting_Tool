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


def test_material_breakdown_uses_supplier_min_and_price(monkeypatch):
    def _fake_material_block(_geo, _material_key, _density, scrap_pct):
        return {
            "material": _geo.get("material_display") or _material_key,
            "stock_L_in": 10.0,
            "stock_W_in": 5.0,
            "stock_T_in": 1.0,
            "start_lb": 10.0,
            "net_lb": 8.0,
            "scrap_lb": 2.0,
            "scrap_pct": float(scrap_pct),
            "price_per_lb": 0.0,
            "source": "test-stock",
            "total_material_cost": 0.0,
            "supplier_min$": 75.0,
        }

    monkeypatch.setattr(appV5, "_compute_material_block", _fake_material_block)
    monkeypatch.setattr(appV5, "_resolve_price_per_lb", lambda _key, _display=None: (4.0, "spot"))

    df_rows = [
        {"Item": "Material Name", "Example Values / Options": "Tool Steel"},
        {"Item": "Scrap Percent (%)", "Example Values / Options": 0.2},
    ]

    result = appV5.compute_quote_from_df(
        df_rows,
        params={},
        rates={},
        geo={},
        ui_vars={},
    )

    material = result["breakdown"]["material"]
    assert material["starting_weight_lb"] == pytest.approx(10.0)
    assert material["scrap_pct"] == pytest.approx(0.2)
    assert material["price_per_lb$"] == pytest.approx(4.0)
    assert material["supplier_min$"] == pytest.approx(75.0)
    assert material["total_material_cost"] == pytest.approx(75.0)

    assert result["breakdown"]["total_direct_costs"] == pytest.approx(75.0)
