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
