import math

from cad_quoter.app.quote_doc import build_material_detail_lines
from cad_quoter.pricing.materials import LB_PER_KG


def test_build_material_detail_lines_formats_mass_scrap_and_credit() -> None:
    material = {
        "mass_g": 1000.0,
        "mass_g_net": 800.0,
        "material_removed_mass_g_est": 200.0,
        "scrap_pct": 0.25,
        "material_scrap_credit_entered": "true",
        "material_scrap_credit": 12.0,
        "scrap_credit_unit_price_usd_per_lb": 1.5,
        "unit_price_usd_per_lb": 10.0,
        "unit_price_source": "catalog",
        "unit_price_asof": "2023-01-01",
        "supplier_min_charge": 5.0,
    }
    scrap_context = {"scrap_pct": material["scrap_pct"]}

    updates, lines = build_material_detail_lines(
        material,
        scrap_context=scrap_context,
        currency="$",
        show_zeros=False,
        show_material_shipping=True,
        shipping_total=7.5,
        shipping_source="material",
    )

    assert math.isclose(updates.get("scrap_credit_mass_lb", 0.0), 200.0 / 1000.0 * LB_PER_KG)
    assert any(line.startswith("  Starting Weight:") for line in lines)
    assert any("Scrap Weight:" in line for line in lines)
    assert any("Scrap Percentage:" in line for line in lines)
    assert any(line.startswith("  Scrap Credit: -$12.00") for line in lines)
    assert any(line.startswith("  Shipping: $7.50") for line in lines)
    assert any("Material Price: $10.00 / lb" in line for line in lines)


def test_build_material_detail_lines_skips_credit_without_flag() -> None:
    material = {
        "mass_g": 500.0,
        "mass_g_net": 500.0,
        "material_scrap_credit_entered": False,
        "material_scrap_credit": 15.0,
        "unit_price_usd_per_lb": 8.0,
    }
    updates, lines = build_material_detail_lines(
        material,
        scrap_context={"scrap_pct": None},
        currency="$",
        show_zeros=True,
        show_material_shipping=False,
        shipping_total=0.0,
        shipping_source=None,
    )

    assert updates.get("scrap_credit_mass_lb") is None
    assert any(line.startswith("  Starting Weight:") for line in lines)
    assert any(line.startswith("  Net Weight:") for line in lines)
    assert "Scrap Credit" not in "\n".join(lines)
