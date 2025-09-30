from __future__ import annotations

import types
import sys

import pytest

from cad_quoter import pricing


def test_price_value_to_per_gram_converts_common_units() -> None:
    assert pricing.price_value_to_per_gram(2.5, "$/kg") == pytest.approx(0.0025)
    assert pricing.price_value_to_per_gram(8.0, "per lb") == pytest.approx(8.0 / 453.59237)
    assert pricing.price_value_to_per_gram(12.0, "Per Ounce") == pytest.approx(12.0 / 28.349523125)
    assert pricing.price_value_to_per_gram(1.2, "per gram") == pytest.approx(1.2)
    assert pricing.price_value_to_per_gram(10.0, "each") is None


def test_resolve_material_unit_price_prefers_wieland(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("cad_quoter.pricing.wieland_scraper")
    module.get_live_material_price = lambda *args, **kwargs: (17.5, "wieland-live")
    monkeypatch.setitem(sys.modules, "cad_quoter.pricing.wieland_scraper", module)

    price, source = pricing.resolve_material_unit_price("6061 aluminum", unit="kg")
    assert price == pytest.approx(17.5)
    assert source == "wieland-live"


def test_resolve_material_unit_price_falls_back_to_csv(
    monkeypatch: pytest.MonkeyPatch, sample_pricing_table: dict
) -> None:
    module = types.ModuleType("cad_quoter.pricing.wieland_scraper")
    module.get_live_material_price = lambda *args, **kwargs: (None, "wieland-offline")
    monkeypatch.setitem(sys.modules, "cad_quoter.pricing.wieland_scraper", module)

    monkeypatch.setattr(pricing, "load_backup_prices_csv", lambda path=None: sample_pricing_table)

    import appV5

    monkeypatch.setattr(appV5, "load_backup_prices_csv", lambda path=None: sample_pricing_table)

    price, source = pricing.resolve_material_unit_price("Stainless Steel 304", unit="lb")
    expected = sample_pricing_table["stainless steel"]["usd_per_lb"]
    assert price == pytest.approx(expected)
    assert source == f"backup_csv:{pricing.BACKUP_CSV_NAME}"


def test_usdkg_to_usdlb_round_trip() -> None:
    kg_price = 4.4
    lb_price = pricing.usdkg_to_usdlb(kg_price)
    assert lb_price == pytest.approx(kg_price / pricing.LB_PER_KG)
    assert pricing.usdkg_to_usdlb(lb_price * pricing.LB_PER_KG) == pytest.approx(lb_price)
