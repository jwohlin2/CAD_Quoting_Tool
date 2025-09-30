import pytest

import appV5


class _FailingPricingEngine:
    def get_usd_per_kg(self, *args, **kwargs):
        raise RuntimeError("providers unavailable")


def test_compute_material_cost_uses_csv_fallback(monkeypatch):
    monkeypatch.setattr(appV5, "lookup_wieland_price", lambda _c: (None, ""))

    mass_kg = 2.0
    scrap = 0.0
    overrides = {}
    vendor_csv = None

    material_name = "Aluminum 6061"
    cost, detail = appV5.compute_material_cost(
        material_name,
        mass_kg,
        scrap,
        overrides,
        vendor_csv,
        pricing=_FailingPricingEngine(),
    )

    expected_unit_price, expected_source = appV5._resolve_material_unit_price(
        material_name,
        unit="kg",
    )
    assert detail["unit_price_usd_per_kg"] == pytest.approx(expected_unit_price)
    assert detail["source"].startswith("backup_csv")
    assert detail["source"].endswith(appV5.BACKUP_CSV_NAME)
    assert cost == pytest.approx(mass_kg * expected_unit_price)
