import appV5


def test_render_quote_places_why_after_pricing_ladder_and_llm_adjustments() -> None:
    result = {
        "price": 54.19,
        "narrative": "Tight tolerance adds inspection time.",
        "llm_notes": ["LLM suggested fixture optimization."],
        "breakdown": {
            "qty": 3,
            "totals": {
                "labor_cost": 25.0,
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
    assert set(lines[total_labor_idx - 1].strip()) == {"-"}

    total_direct_idx = next(i for i, line in enumerate(lines) if "Total Direct Costs" in line)
    assert set(lines[total_direct_idx - 1].strip()) == {"-"}

    pricing_idx = lines.index("Pricing Ladder")
    llm_idx = lines.index("LLM Adjustments")
    why_idx = lines.index("Why this price")

    assert pricing_idx < llm_idx < why_idx
    assert lines[why_idx - 1] == ""
    assert rendered.endswith("\n")


def test_render_quote_planner_why_section_uses_final_bucket_view() -> None:
    breakdown = {
        "qty": 1,
        "totals": {
            "labor_cost": 900.0,
            "direct_costs": 175.0,
            "subtotal": 1075.0,
            "with_overhead": 1182.5,
            "with_ga": 1241.625,
            "with_contingency": 1241.625,
            "with_expedite": 1241.625,
        },
        "nre_detail": {},
        "nre": {},
        "material": {"scrap_pct": 0.12},
        "process_costs": {},
        "process_meta": {},
        "pass_through": {"Material": 150.0, "Shipping": 25.0},
        "applied_pcts": {
            "OverheadPct": 0.10,
            "GA_Pct": 0.05,
            "ContingencyPct": 0.0,
            "MarginPct": 0.15,
        },
        "rates": {},
        "params": {},
        "labor_cost_details": {},
        "direct_cost_details": {},
        "pricing_source": "planner",
        "bucket_view": {
            "buckets": {
                "milling": {
                    "total$": 500.0,
                    "machine$": 350.0,
                    "labor$": 150.0,
                    "minutes": 120.0,
                },
                "drilling": {
                    "total$": 300.0,
                    "machine$": 200.0,
                    "labor$": 100.0,
                    "minutes": 90.0,
                },
                "inspection": {
                    "total$": 100.0,
                    "labor$": 100.0,
                    "minutes": 60.0,
                },
                "finishing": {
                    "total$": 50.0,
                    "labor$": 50.0,
                    "minutes": 30.0,
                },
            }
        },
        "decision_state": {"effective": {"setups": 2, "part_count": 4}},
    }

    result = {"price": 1450.0, "breakdown": breakdown}

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    heading = "Planner operations (final bucket view):"
    assert any(line.strip() == heading for line in lines)
    heading_idx = next(i for i, line in enumerate(lines) if line.strip() == heading)
    bucket_lines = lines[heading_idx + 1 : heading_idx + 4]

    assert len(bucket_lines) == 3
    assert bucket_lines[0].strip().startswith("Milling:")
    assert "$500.00" in bucket_lines[0]
    assert "hr" in bucket_lines[0]
    assert all("overrides" not in line for line in bucket_lines)

    trailing_block = [line.strip() for line in lines[heading_idx + 1 : heading_idx + 10]]
    assert any(line.startswith("Parts: 4") for line in trailing_block)
    assert any(line.startswith("Setups: 2") for line in trailing_block)
    assert any("Scrap: 12.0%" in line for line in trailing_block)
    assert any("Pass-Through: $175.00" in line for line in trailing_block)


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
    summary_block = lines[summary_idx:summary_idx + 6]
    assert any("Milling" in line and "4.00 hr" in line for line in summary_block)
    assert any(
        "Finishing/Deburr" in line and "1.50 hr" in line for line in summary_block
        for line in summary_block
    )
    assert any("Total Hours" in line and "5.50 hr" in line for line in summary_block)


def test_render_quote_merges_deburr_variants() -> None:
    result = {
        "price": 150.0,
        "breakdown": {
            "qty": 2,
            "totals": {
                "labor_cost": 55.0,
                "direct_costs": 10.0,
                "subtotal": 70.0,
                "with_overhead": 77.0,
                "with_ga": 80.85,
                "with_contingency": 82.4715,
                "with_expedite": 82.4715,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {
                "milling": 40.0,
                "Deburr": 10.0,
                "Finishing/Deburr": 5.0,
            },
            "process_meta": {
                "milling": {"hr": 3.0},
                "deburr": {"hr": 1.0},
                "Finishing/Deburr": {"hr": 0.5},
            },
            "pass_through": {},
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

    labor_idx = lines.index("Process & Labor Costs")
    end_idx = next(i for i in range(labor_idx, len(lines)) if lines[i] == "")
    labor_block = lines[labor_idx:end_idx]

    deburr_lines = [line for line in labor_block if "Finishing/Deburr" in line]
    assert len(deburr_lines) == 1
    assert "$15.00" in deburr_lines[0]


def test_render_quote_skips_duplicate_programming_amortized_row() -> None:
    result = {
        "price": 180.0,
        "breakdown": {
            "qty": 3,
            "totals": {
                "labor_cost": 90.0,
                "direct_costs": 20.0,
                "subtotal": 110.0,
                "with_overhead": 121.0,
                "with_ga": 127.05,
                "with_contingency": 129.591,
                "with_expedite": 129.591,
            },
            "nre_detail": {"programming": {"prog_hr": 2.0, "prog_rate": 60.0}},
            "nre": {"programming_per_part": 15.0},
            "material": {},
            "process_costs": {
                "milling": 60.0,
                "Programming (amortized)": 15.0,
            },
            "process_meta": {
                "milling": {"hr": 2.5},
                "Programming (amortized)": {"hr": 0.75},
            },
            "pass_through": {},
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
    labor_idx = lines.index("Process & Labor Costs")
    end_idx = next(i for i in range(labor_idx, len(lines)) if lines[i] == "")
    labor_block = lines[labor_idx:end_idx]

    programming_lines = [line for line in labor_block if "Programming (amortized)" in line]
    assert len(programming_lines) == 1


def test_render_quote_includes_amortized_and_misc_rows() -> None:
    result = {
        "price": 420.0,
        "breakdown": {
            "qty": 5,
            "totals": {
                "labor_cost": 255.0,
                "direct_costs": 40.0,
                "subtotal": 295.0,
                "with_overhead": 324.5,
                "with_ga": 340.725,
                "with_contingency": 347.5395,
                "with_expedite": 347.5395,
            },
            "nre_detail": {
                "programming": {
                    "prog_hr": 2.0,
                    "prog_rate": 60.0,
                    "eng_hr": 1.0,
                    "eng_rate": 55.0,
                },
                "fixture": {
                    "build_hr": 1.5,
                    "labor_cost": 90.0,
                    "build_rate": 50.0,
                },
            },
            "nre": {},
            "material": {},
            "process_costs": {"milling": 120.0, "deburr": 80.0},
            "process_meta": {"milling": {"hr": 2.0}, "deburr": {"hr": 1.0}},
            "pass_through": {"Material": 40.0},
            "labor_costs": {
                "Milling": 120.0,
                "Finishing/Deburr": 80.0,
                "Programming (amortized)": 30.0,
                "Fixture Build (amortized)": 20.0,
                "Misc (LLM deltas)": 5.0,
            },
            "applied_pcts": {
                "OverheadPct": 0.10,
                "GA_Pct": 0.05,
                "ContingencyPct": 0.02,
                "MarginPct": 0.15,
            },
            "rates": {"FixtureBuildRate": 50.0},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    labor_idx = lines.index("Process & Labor Costs")
    next_blank = next(i for i in range(labor_idx + 1, len(lines)) if lines[i] == "")
    labor_section = lines[labor_idx:next_blank]

    assert any("Programming (amortized)" in line for line in labor_section)
    assert any("Fixture Build (amortized)" in line for line in labor_section)
    assert any("Misc (LLM deltas)" in line for line in labor_section)

    section_total_line = next(
        line for line in labor_section if line.strip().startswith("Total") and "$" in line
    )
    section_total = float(section_total_line.split("$")[-1].replace(",", ""))

    total_labor_line = next(line for line in lines if line.startswith("Total Labor Cost:"))
    total_labor_amount = float(total_labor_line.split("$")[-1].replace(",", ""))

    assert abs(section_total - 255.0) < 1e-6
    assert abs(total_labor_amount - section_total) < 1e-6


def test_render_quote_hour_summary_adds_programming_hours() -> None:
    result = {
        "price": 150.0,
        "breakdown": {
            "qty": 5,
            "totals": {
                "labor_cost": 78.0,
                "direct_costs": 30.0,
                "subtotal": 120.0,
                "with_overhead": 132.0,
                "with_ga": 138.6,
                "with_contingency": 142.758,
                "with_expedite": 142.758,
            },
            "nre_detail": {
                "programming": {"prog_hr": 2.0, "amortized": True},
                "fixture": {"build_hr": 1.5, "labor_cost": 90.0, "build_rate": 60.0},
            },
            "nre": {},
            "material": {},
            "process_costs": {"milling": 60.0},
            "process_meta": {"milling": {"hr": 3.0}},
            "pass_through": {"material": 30.0},
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
    summary_block = lines[summary_idx:summary_idx + 10]
    assert any("Programming" in line and "2.00 hr" in line for line in summary_block)
    assert any(
        "Programming (amortized per part)" in line and "0.40 hr" in line
        for line in summary_block
    )
    assert any("Fixture Build" in line and "1.50 hr" in line for line in summary_block)
    assert any(
        "Fixture Build (amortized per part)" in line and "0.30 hr" in line
        for line in summary_block
    )
    assert any("Total Hours" in line and "6.50 hr" in line for line in summary_block)


def test_render_quote_hides_drill_debug_without_flag() -> None:
    result = {
        "price": 54.19,
        "drill_debug": ["CSV drill feeds"],
        "app": {"llm_debug_enabled": False},
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 0.0,
                "direct_costs": 0.0,
                "subtotal": 0.0,
                "with_overhead": 0.0,
                "with_ga": 0.0,
                "with_contingency": 0.0,
                "with_expedite": 0.0,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {},
            "process_meta": {},
            "pass_through": {},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
            "app": {"llm_debug_enabled": False},
        },
    }

    rendered = appV5.render_quote(result, currency="$")

    assert "Drill Debug" not in rendered


def test_render_quote_clamps_single_piece_hours_and_warns() -> None:
    result = {
        "price": 220.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 150.0,
                "direct_costs": 30.0,
                "subtotal": 180.0,
                "with_overhead": 198.0,
                "with_ga": 207.9,
                "with_contingency": 212.058,
                "with_expedite": 212.058,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {"milling": 150.0},
            "process_meta": {"milling": {"hr": 42.0}},
            "pass_through": {},
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

    summary_idx = lines.index("Labor Hour Summary")
    summary_block = lines[summary_idx:summary_idx + 6]
    assert any("Milling" in line and "24.00 hr" in line for line in summary_block)

    if "Red Flags" in lines:
        flag_idx = lines.index("Red Flags")
        flag_section = lines[flag_idx: flag_idx + 5]
        assert any("capped at 24 hr" in line for line in flag_section)


def test_render_quote_dedupes_planner_rollup_cost_rows() -> None:
    result = {
        "price": 200.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 30.0,
                "direct_costs": 30.0,
                "subtotal": 150.0,
                "with_overhead": 165.0,
                "with_ga": 173.25,
                "with_contingency": 178.4475,
                "with_expedite": 178.4475,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {
                "Planner Total": 120.0,
                "Planner Machine": 70.0,
                "Planner Labor": 55.0,
                "planner_labor": 50.0,
                "milling": 30.0,
            },
            "process_meta": {
                "planner_total": {"minutes": 240.0},
                "planner_machine": {"minutes": 120.0},
                "planner_labor": {"minutes": 120.0},
                "milling": {"hr": 1.5},
            },
            "pass_through": {},
            "pricing_source": "planner",
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
    summary_block = lines[summary_idx: summary_idx + 10]
    assert any("Planner Total" in line for line in summary_block)
    assert any("Planner Machine" in line for line in summary_block)
    assert any("Planner Labor" in line for line in summary_block)

    labor_idx = lines.index("Process & Labor Costs")
    next_blank = next(i for i in range(labor_idx + 1, len(lines)) if lines[i] == "")
    labor_section = lines[labor_idx:next_blank]
    assert not any("Planner" in line for line in labor_section)


def test_total_labor_cost_matches_displayed_rows_and_pass_through_labor() -> None:
    result = {
        "price": 110.0,
        "breakdown": {
            "qty": 4,
            "totals": {
                "labor_cost": 50.0,
                "direct_costs": 22.0,
                "subtotal": 72.0,
                "with_overhead": 79.2,
                "with_ga": 83.16,
                "with_contingency": 84.8232,
                "with_expedite": 84.8232,
            },
            "process_costs": {"milling": 40.0, "deburr": 10.0},
            "process_meta": {},
            "pass_through": {
                "Material": 12.0,
                "outside labor": 15.0,
                "Shipping": 7.0,
            },
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

    total_labor_line = next(line for line in lines if line.startswith("Total Labor Cost:"))
    total_labor_amount = float(total_labor_line.split("$")[-1].replace(",", ""))

    process_start = lines.index("Process & Labor Costs")
    process_end = next(i for i in range(process_start, len(lines)) if lines[i] == "")
    process_lines = lines[process_start + 2:process_end]

    process_amounts: list[float] = []
    table_total_amount = 0.0
    for line in process_lines:
        if "$" not in line:
            continue
        amount = float(line.split("$")[-1].replace(",", ""))
        label_text = line.split("$")[0].strip()
        if label_text.lower().startswith("total"):
            table_total_amount = amount
        else:
            process_amounts.append(amount)

    assert sum(process_amounts) == table_total_amount
    assert total_labor_amount == table_total_amount + 15.0
