import appV5


def test_render_quote_places_why_after_pricing_ladder_and_llm_adjustments() -> None:
    result = {
        "price": 54.19,
        "narrative": "Tight tolerance adds inspection time.",
        "llm_notes": ["LLM suggested fixture optimization."],
        "breakdown": {
            "qty": 3,
            "totals": {
                "labor_cost": 30.0,
                "direct_costs": 10.0,
                "subtotal": 40.0,
                "with_overhead": 44.0,
                "with_ga": 46.2,
                "with_contingency": 47.124,
                "with_expedite": 47.124,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {"machining": 25.0},
            "process_meta": {},
            "pass_through": {"material": 5.0},
            "applied_pcts": {
                "OverheadPct": 0.10,
                "GA_Pct": 0.05,
                "ContingencyPct": 0.02,
                "MarginPct": 0.15,
            },
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    assert "Pricing Ladder" in lines
    assert "LLM Adjustments" in lines
    assert "Why this price" in lines

    total_labor_idx = next(i for i, line in enumerate(lines) if "Total Labor Cost" in line)
    assert set(lines[total_labor_idx - 1]) == {"-"}

    total_direct_idx = next(i for i, line in enumerate(lines) if "Total Direct Costs" in line)
    assert set(lines[total_direct_idx - 1]) == {"-"}

    pricing_idx = lines.index("Pricing Ladder")
    llm_idx = lines.index("LLM Adjustments")
    why_idx = lines.index("Why this price")

    assert pricing_idx < llm_idx < why_idx
    assert lines[why_idx - 1] == ""
    assert rendered.endswith("\n")


def test_render_quote_includes_hour_summary() -> None:
    result = {
        "price": 120.0,
        "breakdown": {
            "qty": 2,
            "totals": {
                "labor_cost": 80.0,
                "direct_costs": 20.0,
                "subtotal": 100.0,
                "with_overhead": 110.0,
                "with_ga": 115.0,
                "with_contingency": 117.3,
                "with_expedite": 117.3,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {"milling": 60.0, "deburr": 20.0},
            "process_meta": {
                "milling": {"hr": 4.0},
                "deburr": {"hr": 1.5},
                "inspection": {"hr": 0.5},
            },
            "pass_through": {"material": 20.0},
            "applied_pcts": {
                "OverheadPct": 0.10,
                "GA_Pct": 0.05,
                "ContingencyPct": 0.02,
                "MarginPct": 0.15,
            },
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    assert "Labor Hour Summary" in lines
    summary_idx = lines.index("Labor Hour Summary")
    divider_idx = summary_idx + 1
    assert lines[divider_idx].startswith("-")
    summary_block = lines[summary_idx:summary_idx + 7]
    assert any("Milling" in line and "4.00 hr" in line for line in summary_block)
    assert any("Deburr" in line and "1.50 hr" in line for line in summary_block)
    assert any("Inspection" in line and "0.50 hr" in line for line in summary_block)

    total_hours_idx = next(i for i, line in enumerate(lines) if "Total Hours" in line)
    assert set(lines[total_hours_idx - 1]) == {"-"}
    assert "6.00 hr" in lines[total_hours_idx]
