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

    expected_scrap = _format_weight_lb_oz_for_test(24.0)

    assert "Mass:" not in rendered
    assert "Starting Weight: 4.2 oz" in rendered
    assert f"Scrap Weight: {expected_scrap}" in rendered
    assert "Net Weight: 3.5 oz" in rendered
    assert "Scrap Percentage: 20.0%" in rendered


def test_render_quote_prefers_removed_mass_estimate_for_scrap() -> None:
    removal_mass_g = 125.0
    material = {
        "mass_g": 1000.0,
        "effective_mass_g": 1000.0,
        "net_mass_g": 900.0,
        "material_removed_mass_g_est": removal_mass_g,
        "scrap_pct": 0.4,
    }

    rendered = appV5.render_quote(
        _base_material_quote(material), currency="$", show_zeros=False
    )

    expected_scrap = _format_weight_lb_oz_for_test(removal_mass_g)

    assert f"Scrap Weight: {expected_scrap}" in rendered


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

    expected_scrap = _format_weight_lb_oz_for_test(start_g * material["scrap_pct"])

    assert "Starting Weight: 32 lb 12.4 oz" in rendered
    assert f"Scrap Weight: {expected_scrap}" in rendered
    assert "Net Weight: 24 lb 9.3 oz" in rendered


def test_render_quote_prefers_mass_g_for_starting_weight() -> None:
    start_g = 14866.489927078912
    effective_g = 18583.1124084375
    net_g = 11149.867445309184
    material = {
        "mass_g": start_g,
        "effective_mass_g": effective_g,
        "net_mass_g": net_g,
        "scrap_pct": 0.25,
    }

    result = _base_material_quote(material)

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    expected_scrap = _format_weight_lb_oz_for_test(
        effective_g * material["scrap_pct"]
    )

    assert "Starting Weight: 32 lb 12.4 oz" in rendered
    assert f"Scrap Weight: {expected_scrap}" in rendered
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


def test_render_quote_mentions_scrap_source_from_holes() -> None:
    material = {
        "mass_g": 120.0,
        "effective_mass_g": 120.0,
        "net_mass_g": 100.0,
        "scrap_pct": 0.25,
        "scrap_pct_from_holes": 0.05,
    }

    rendered = appV5.render_quote(
        _base_material_quote(material), currency="$", show_zeros=False
    )

    assert "Scrap Percentage: 25.0% (holes)" in rendered


def test_render_quote_mentions_scrap_source_label() -> None:
    material = {
        "mass_g": 120.0,
        "effective_mass_g": 120.0,
        "net_mass_g": 100.0,
        "scrap_pct": 0.18,
        "scrap_source_label": "entered+holes",
    }

    rendered = appV5.render_quote(
        _base_material_quote(material), currency="$", show_zeros=False
    )

    assert "Scrap Percentage: 18.0% (entered + holes)" in rendered


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


def _amortized_breakdown(qty: int, *, config_flags: dict | None = None) -> dict:
    labor_costs: dict[str, float] = {}
    if qty > 1:
        labor_costs = {
            "Programming (amortized)": 150.0,
            "Fixture Build (amortized)": 30.0,
        }

    bucket_view = {
        "buckets": {
            "grinding": {
                "minutes": 90.0,
                "labor$": 300.0,
                "machine$": 0.0,
            },
        },
        "order": ["grinding"],
    }
    if qty > 1:
        bucket_view["buckets"]["programming_amortized"] = {
            "minutes": 60.0,
            "labor$": 150.0,
            "machine$": 0.0,
        }
        bucket_view["buckets"]["fixture_build_amortized"] = {
            "minutes": 30.0,
            "labor$": 30.0,
            "machine$": 0.0,
        }
        bucket_view.setdefault("order", []).extend(
            ["programming_amortized", "fixture_build_amortized"]
        )

    breakdown = {
        "qty": qty,
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
        "labor_costs": labor_costs,
        "nre": {
            "programming_per_part": 150.0,
            "fixture_per_part": 30.0,
        },
        "bucket_view": bucket_view,
        "process_plan": {"bucket_view": bucket_view},
        "process_plan_summary": {"bucket_view": bucket_view},
    }
    if config_flags is not None:
        breakdown["config_flags"] = dict(config_flags)
    return breakdown


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
        "breakdown": _amortized_breakdown(5),
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


def test_render_quote_hides_amortized_nre_for_single_qty() -> None:
    result = {
        "price": 10.0,
        "breakdown": _amortized_breakdown(1),
    }

    assert result["breakdown"]["labor_costs"] == {}

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert "Programming (amortized)" not in rendered
    assert "Fixture Build (amortized)" not in rendered
    assert "Programming & Eng:" in rendered
    assert "Fixturing:" in rendered


def test_render_quote_uses_programming_per_part_for_single_qty() -> None:
    breakdown = _amortized_breakdown(1)
    breakdown["nre_detail"]["programming"]["per_lot"] = 0.0
    breakdown["nre"]["programming_per_part"] = 1462.0

    result = {"price": 10.0, "breakdown": breakdown}

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    programming_line = next(
        line for line in rendered.splitlines() if "Programming & Eng:" in line
    )
    assert "$1,462.00" in programming_line


def test_render_quote_skips_amortized_labor_totals_for_single_qty() -> None:
    breakdown = _amortized_breakdown(1)
    breakdown["labor_costs"] = {
        "Programming (amortized)": 150.0,
        "Fixture Build (amortized)": 30.0,
        "Grinding": 25.0,
    }
    breakdown["process_costs"] = {}
    breakdown["process_meta"] = {}
    breakdown["labor_cost_details"] = {}

    result = {"price": 10.0, "breakdown": breakdown}

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert "Programming (amortized)" not in rendered
    assert "Fixture Build (amortized)" not in rendered
    assert "Grinding" in rendered


def test_render_quote_ignores_force_amortized_flag_for_single_qty() -> None:
    result = {
        "price": 10.0,
        "breakdown": _amortized_breakdown(
            1, config_flags={"show_amortized_nre": True}
        ),
    }

    assert result["breakdown"]["labor_costs"] == {}

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert "Programming (amortized)" not in rendered
    assert "Fixture Build (amortized)" not in rendered
    assert "Amortized across" not in rendered


def test_render_quote_shows_flat_extras_when_no_hours() -> None:
    bucket_view = {
        "buckets": {
            "grinding": {
                "minutes": 133.2,
                "labor$": 200.0,
                "machine$": 0.0,
            },
        }
    }

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
            "bucket_view": bucket_view,
            "process_plan": {"bucket_view": bucket_view},
            "process_plan_summary": {"bucket_view": bucket_view},
        },
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert "includes $200.00 extras" not in rendered
    assert rendered.count("2.22 hr @ $90.00/hr") == 1
