import appV5


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
    mass_line = next(line for line in rendered.splitlines() if "Mass:" in line)

    assert "0.22 lb net" in mass_line
    assert "scrap-adjusted 0.26 lb" in mass_line


def test_render_quote_does_not_duplicate_detail_lines() -> None:
    result = {
        "price": 10.0,
        "breakdown": {
            "qty": 1,
            "totals": _base_totals(),
            "material": {},
            "nre": {},
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
        },
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert rendered.count("- Programmer: 1.00 hr @ $75.00/hr") == 1
    assert rendered.count("Programmer 1.00 hr @ $75.00/hr") == 0
    assert rendered.count("includes $200.00 extras") == 1
    assert rendered.count("includes 1.67 hr extras") == 1
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

    assert rendered.count("includes $200.00 extras") == 1
    assert "hr extras" not in rendered
    process_total_line = next(
        line for line in rendered.splitlines() if line.startswith("  Total")
    )
    assert process_total_line.strip().endswith("$200.00")
