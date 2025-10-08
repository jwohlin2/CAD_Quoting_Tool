import appV5
from cad_quoter.pricing.materials import LB_PER_KG


def _format_weight_lb_oz_for_test(mass_g: float | None) -> str:
    grams = max(0.0, float(mass_g or 0.0))
    if grams <= 0:
        return "0 oz"
    pounds_total = grams / 1000.0 * LB_PER_KG
    total_ounces = pounds_total * 16.0
    pounds = int(total_ounces // 16)
    ounces = total_ounces - pounds * 16
    precision = 1 if pounds > 0 or ounces >= 1.0 else 2
    ounces = round(ounces, precision)
    if ounces >= 16.0:
        pounds += 1
        ounces = 0.0
    parts: list[str] = []
    if pounds > 0:
        parts.append(f"{pounds} lb" if pounds != 1 else "1 lb")
    if ounces > 0 or pounds == 0:
        ounce_text = f"{ounces:.{precision}f}".rstrip("0").rstrip(".")
        if not ounce_text:
            ounce_text = "0"
        parts.append(f"{ounce_text} oz")
    return " ".join(parts)


def _base_totals() -> dict:
    return {
        "labor_cost": 0.0,
        "direct_costs": 0.0,
        "subtotal": 0.0,
        "with_overhead": 0.0,
        "with_ga": 0.0,
        "with_contingency": 0.0,
        "with_expedite": 0.0,
    }


def test_render_quote_shows_net_mass_when_scrap_present() -> None:
    result = {
        "price": 10.0,
        "breakdown": {
            "qty": 1,
            "totals": _base_totals(),
            "nre_detail": {},
            "nre": {},
            "material": {
                "mass_g": 120.0,
                "effective_mass_g": 120.0,
                "mass_g_net": 100.0,
                "scrap_pct": 0.2,
            },
            "process_costs": {},
            "process_meta": {},
            "pass_through": {},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "nre_cost_details": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert "Mass:" not in rendered
    assert "Starting Weight: 4.2 oz" in rendered
    assert "Scrap Weight: 0.71 oz" in rendered
    assert "Net Weight: 3.5 oz" in rendered
    assert "Scrap Percentage: 20.0%" in rendered


def test_render_quote_uses_net_mass_g_fallback() -> None:
    start_g = 14866.489927078912
    net_g = 11149.867445309184
    material = {
        "mass_g": start_g,
        "effective_mass_g": start_g,
        "net_mass_g": net_g,
        "scrap_pct": 0.25,
    }

    result = _base_material_quote(material)

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert "Starting Weight: 32 lb 12.4 oz" in rendered
    assert "Scrap Weight: 8 lb 3.1 oz" in rendered
    assert "Net Weight: 24 lb 9.3 oz" in rendered


def test_render_quote_omits_unlabeled_scrap_adjusted_mass() -> None:
    start_g = 14866.489927078912
    net_g = 11149.867445309184
    material = {
        "mass_g": start_g,
        "effective_mass_g": start_g,
        "net_mass_g": net_g,
        "scrap_pct": 0.25,
    }

    rendered = appV5.render_quote(
        _base_material_quote(material), currency="$", show_zeros=False
    )

    scrap_adjusted_mass = net_g * (1.0 - material["scrap_pct"])
    unexpected_line = _format_weight_lb_oz_for_test(scrap_adjusted_mass)

    assert unexpected_line not in rendered.splitlines()


def _base_material_quote(material: dict) -> dict:
    return {
        "price": 10.0,
        "breakdown": {
            "qty": 1,
            "totals": _base_totals(),
            "material": material,
            "nre": {},
            "nre_detail": {},
            "process_costs": {},
            "process_meta": {},
            "pass_through": {},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "nre_cost_details": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }


def test_render_quote_unit_price_prefers_lb_over_metric() -> None:
    result = _base_material_quote(
        {
            "mass_g": 120.0,
            "effective_mass_g": 120.0,
            "unit_price_usd_per_kg": 20.0,
            "unit_price_asof": "2024-01-01",
        }
    )

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)
    unit_line = next(line for line in rendered.splitlines() if "Material Price:" in line)

    assert unit_line.strip().startswith("Material Price: $9.07 / lb")
    assert "/ kg" not in unit_line
    assert "/ g" not in unit_line


def test_render_quote_unit_price_converts_from_per_gram() -> None:
    result = _base_material_quote(
        {
            "mass_g": 120.0,
            "effective_mass_g": 120.0,
            "unit_price_per_g": 0.012,
        }
    )

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)
    unit_line = next(line for line in rendered.splitlines() if "Material Price:" in line)

    assert unit_line.strip().startswith("Material Price: $5.44 / lb")
    assert "/ kg" not in unit_line
    assert "/ g" not in unit_line


def test_render_quote_does_not_duplicate_detail_lines() -> None:
    result = {
        "price": 10.0,
        "breakdown": {
            "qty": 1,
            "totals": _base_totals(),
            "material": {},
            "nre_detail": {
                "programming": {
                    "per_lot": 150.0,
                    "prog_hr": 1.0,
                    "prog_rate": 75.0,
                },
                "fixture": {
                    "per_lot": 80.0,
                    "build_hr": 0.5,
                    "build_rate": 60.0,
                    "mat_cost": 20.0,
                    "labor_cost": 30.0,
                },
            },
            "nre_cost_details": {
                "Programming & Eng (per lot)": "Programmer 1.00 hr @ $75.00/hr",
                "Fixturing (per lot)": "Build 0.50 hr @ $60.00/hr; Material $20.00",
            },
            "process_costs": {"grinding": 300.0},
            "process_meta": {
                "grinding": {"hr": 1.5, "rate": 120.0, "base_extra": 200.0},
            },
            "labor_cost_details": {
                "Grinding": "1.50 hr @ $120.00/hr; includes $200.00 extras",

            },
            "pass_through": {},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "direct_cost_details": {},
            "labor_costs": {
                "Programming (amortized)": 150.0,
                "Fixture Build (amortized)": 30.0,
            },
            "nre": {
                "programming_per_part": 150.0,
                "fixture_per_part": 30.0,
            },
        },
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert rendered.count("- Programmer: 1.00 hr @ $75.00/hr") == 1
    assert rendered.count("Programmer 1.00 hr @ $75.00/hr") == 0
    assert "Programming (amortized)" in rendered
    assert "Fixture Build (amortized)" in rendered
    assert "- Programmer (lot): 1.00 hr @ $75.00/hr" in rendered
    assert "- Build labor (lot): 0.50 hr @ $60.00/hr" in rendered
    assert "includes $200.00 extras" not in rendered
    assert rendered.count("1.50 hr @ $120.00/hr") == 1


def test_render_quote_shows_flat_extras_when_no_hours() -> None:
    result = {
        "price": 10.0,
        "breakdown": {
            "qty": 1,
            "totals": _base_totals(),
            "material": {},
            "nre": {},
            "nre_detail": {},
            "nre_cost_details": {},
            "process_costs": {"grinding": 200.0},
            "process_meta": {
                "grinding": {"hr": 0.0, "rate": 90.0, "base_extra": 200.0},
            },
            "labor_cost_details": {},
            "pass_through": {},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert "includes $200.00 extras" not in rendered
    assert rendered.count("2.22 hr @ $90.00/hr") == 1
