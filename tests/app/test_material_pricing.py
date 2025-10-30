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

import cad_quoter.pricing.materials as materials


def _fail_live_price(*_args, **_kwargs):
    raise RuntimeError("providers unavailable")


def test_resolve_price_per_lb_prefers_mcmaster_before_resolver(monkeypatch):
    module = types.ModuleType("metals_api")

    def _raise(_material_key: str) -> float:
        raise RuntimeError("metals api unavailable")

    module.price_per_lb_for_material = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    def _fake_mcmaster(name: str, *, unit: str = "kg") -> tuple[float, str]:
        assert unit == "lb"
        return 123.45, "mcmaster:aluminum"

    def _fail_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        raise AssertionError("resolver should not run when McMaster succeeds")

    monkeypatch.setattr(materials, "get_mcmaster_unit_price", _fake_mcmaster, raising=False)
    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fail_resolver, raising=False)

    price, source = materials._resolve_price_per_lb("aluminum", "Aluminum")

    assert price == pytest.approx(123.45)
    assert source == "mcmaster:aluminum"


def test_resolve_price_per_lb_uses_resolver_when_providers_fail(monkeypatch):
    module = types.ModuleType("metals_api")
    module.price_per_lb_for_material = lambda _material_key: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    wieland_module = types.ModuleType("cad_quoter.pricing.wieland_scraper")
    wieland_module.get_live_material_price = _fail_live_price  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cad_quoter.pricing.wieland_scraper", wieland_module)

    def _failing_mcmaster(_name: str, *, unit: str = "kg") -> tuple[float | None, str]:
        return None, ""

    fallback_calls: list[tuple[str, str]] = []

    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        fallback_calls.append((name, unit))
        return 321.0, "backup_csv"

    monkeypatch.setattr(materials, "get_mcmaster_unit_price", _failing_mcmaster, raising=False)
    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fake_resolver, raising=False)

    price, source = materials._resolve_price_per_lb("6061", "6061")

    assert fallback_calls == [("6061", "lb")]
    assert price == pytest.approx(321.0)
    assert source == "backup_csv"

def test_material_price_helper_returns_fallback_price(monkeypatch):
    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        assert unit == "kg"
        return 400.0, "backup_csv"

    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fake_resolver, raising=False)

    price_per_g, source = materials._material_price_per_g_from_choice("Stainless Steel", {})

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

    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fake_resolver, raising=False)

    geo_ctx = {
        "material_display": "Aluminum 6061",
        "thickness_in": 1.0,
        "outline_bbox": {"plate_len_in": 2.0, "plate_wid_in": 3.0},
    }

    block = materials._compute_material_block(geo_ctx, "aluminum", 2.70, 0.1)

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

    block = materials._compute_material_block(geo_ctx, "steel", 7.85, 0.2)

    assert block["supplier_min$"] == pytest.approx(75.0)
    assert block["price_per_lb"] == pytest.approx(2.0)
    assert block["total_material_cost"] == pytest.approx(75.0)


def test_material_cost_components_prefers_stock_piece():
    block = {
        "starting_mass_g": 1000.0,
        "scrap_mass_g": 200.0,
        "unit_price_per_lb_usd": 5.0,
        "unit_price_per_lb_source": "resolver",
        "stock_piece_price_usd": 20.0,
        "stock_piece_source": "McMaster API (qty=1, part=4936K451)",
        "material_tax_usd": 1.5,
        "scrap_price_usd_per_lb": 0.8,
    }
    overrides = {"scrap_recovery_pct": 0.9}

    components = materials._material_cost_components(block, overrides=overrides, cfg=None)

    assert components["base_usd"] == pytest.approx(20.0)
    assert components["base_source"] == "McMaster API (qty=1, part=4936K451)"
    assert components["stock_piece_usd"] == pytest.approx(20.0)
    assert components["stock_source"] == "McMaster API (qty=1, part=4936K451)"
    assert components["tax_usd"] == pytest.approx(1.5)
    assert components["scrap_credit_usd"] == pytest.approx(0.32)
    assert components["scrap_rate_text"] == "$0.80/lb × 90%"
    assert components["total_usd"] == pytest.approx(21.18)


def test_material_cost_components_handles_per_lb_pricing():
    block = {
        "start_mass_g": 500.0,
        "unit_price_per_lb_usd": 6.0,
        "unit_price_source": "resolver",
        "material_tax": 0.75,
    }

    components = materials._material_cost_components(block, overrides={}, cfg=None)

    start_lb = 500.0 * 0.00220462262
    expected_base = round(start_lb * 6.0, 2)
    expected_total = round(expected_base + 0.75, 2)

    assert components["base_usd"] == pytest.approx(expected_base)
    assert components["base_source"] == "resolver @ $6.00/lb"
    assert components["tax_usd"] == pytest.approx(0.75)
    assert components["scrap_credit_usd"] == 0.0
    assert components["scrap_rate_text"] is None
    assert components["total_usd"] == pytest.approx(expected_total)


def test_material_cost_components_respects_explicit_credit():
    block = {
        "starting_mass_g": 1000.0,
        "material_tax_usd": 0.0,
        "stock_piece_price_usd": 50.0,
        "stock_piece_source": "Stock piece",
        "material_scrap_credit": 12.34,
    }

    components = materials._material_cost_components(block, overrides={}, cfg=None)

    assert components["scrap_credit_usd"] == pytest.approx(12.34)
    assert components["total_usd"] == pytest.approx(37.66)


def test_plan_stock_blank_respects_rounding_tolerance():
    plan = materials.plan_stock_blank(12.0, 6.0, 2.03, None, None)

    assert plan["need_len_in"] == pytest.approx(12.05)
    assert plan["need_wid_in"] == pytest.approx(6.05)
    assert plan["stock_len_in"] == pytest.approx(12.0)
    assert plan["stock_wid_in"] == pytest.approx(6.0)
    assert plan["stock_thk_in"] == pytest.approx(2.0)


def test_plan_stock_blank_uses_configured_tolerance():
    cfg = types.SimpleNamespace(round_tol_in=0.1)

    plan = materials.plan_stock_blank(18.0, 8.0, 2.0, None, None, cfg=cfg)

    assert plan["round_tol_in"] == pytest.approx(0.1)
    assert plan["need_len_in"] == pytest.approx(18.1)
    assert plan["stock_len_in"] == pytest.approx(18.0)


def test_vendor_catalog_prefers_exact_thickness(monkeypatch):
    from cad_quoter.pricing import vendor_csv

    rows = [
        {
            "material": "Aluminum MIC6",
            "thickness_in": "3.5",
            "length_in": "12",
            "width_in": "24",
            "vendor": "McMaster",
            "part": "86825K626",
        },
        {
            "material": "Aluminum MIC6",
            "thickness_in": "2",
            "length_in": "12",
            "width_in": "24",
            "vendor": "McMaster",
            "part": "86825K954",
        },
    ]

    calls: list[float] = []

    def fake_pick(L, W, T, *, material_key, catalog_rows):
        calls.append(T)
        assert catalog_rows is rows
        if pytest.approx(T, abs=1e-6) == pytest.approx(2.0):
            return {
                "len_in": 24.0,
                "wid_in": 12.0,
                "thk_in": T,
                "mcmaster_part": "86825K954",
            }
        return None

    monkeypatch.setattr(vendor_csv, "load_mcmaster_catalog_rows", lambda _path=None: rows)
    monkeypatch.setattr(vendor_csv, "pick_mcmaster_plate_sku", fake_pick)

    picked = vendor_csv.pick_plate_from_mcmaster(
        "Aluminum MIC6",
        12.0,
        12.0,
        2.0,
        scrap_fraction=0.0,
        allow_thickness_upsize=False,
        thickness_tolerance=0.02,
    )

    assert calls, "helper should be consulted"
    assert picked is not None
    assert picked["thk_in"] == pytest.approx(2.0)
    assert picked["part_no"] == "86825K954"
    assert picked["len_in"] == pytest.approx(24.0)
    assert picked["wid_in"] == pytest.approx(12.0)
