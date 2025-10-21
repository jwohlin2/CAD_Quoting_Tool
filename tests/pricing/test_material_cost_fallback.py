from __future__ import annotations

import sys
import types

import pytest

import cad_quoter.pricing.materials as materials


def test_resolve_material_unit_price_uses_csv_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(materials, "_maybe_get_mcmaster_price", lambda *_args, **_kwargs: None)

    wieland_module = types.ModuleType("cad_quoter.pricing.wieland_scraper")

    def _fail_live_price(*_args, **_kwargs):
        raise RuntimeError("providers unavailable")

    wieland_module.get_live_material_price = _fail_live_price  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cad_quoter.pricing.wieland_scraper", wieland_module)

    def _fake_loader(_path: str | None = None) -> dict[str, dict[str, float | str]]:
        key = materials.normalize_material_key("Aluminum 6061")
        return {key: {"usd_per_kg": 200.0, "usd_per_lb": 200.0 / materials.LB_PER_KG}}

    monkeypatch.setattr(materials, "load_backup_prices_csv", _fake_loader)

    material_name = "Aluminum 6061"
    price, source = materials.resolve_material_unit_price(material_name, unit="kg")

    assert price == pytest.approx(200.0)
    assert source.startswith("backup_csv")
    assert source.endswith(materials.BACKUP_CSV_NAME)

    mass_kg = 2.0
    expected_cost = mass_kg * price
    assert expected_cost == pytest.approx(400.0)
