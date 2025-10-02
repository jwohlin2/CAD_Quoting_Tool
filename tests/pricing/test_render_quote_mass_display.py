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

    assert "100.0 g net" in mass_line
    assert "scrap-adjusted 120.0 g" in mass_line
