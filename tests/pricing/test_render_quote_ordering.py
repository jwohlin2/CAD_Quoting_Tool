import math
import re

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
                "direct_costs": 15.0,
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
            "pass_through": {"Material": 15.0},
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


def test_total_direct_costs_uses_pricing_breakdown() -> None:
    result = {
        "price": 42.0,
        "breakdown": {
            "qty": 2,
            "totals": {
                "labor_cost": 10.0,
                "direct_costs": 99.0,
                "subtotal": 25.0,
                "with_overhead": 25.0,
                "with_ga": 25.0,
                "with_contingency": 25.0,
                "with_expedite": 25.0,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {"milling": 10.0},
            "process_meta": {},
            "pass_through": {"Material": 12.5, "Shipping": 5.0},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
            "pricing": {
                "direct_costs": {"material": 12.5, "shipping": 5.0},
            },
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    direct_line = next(line for line in lines if line.startswith("Total Direct Costs:"))
    assert direct_line.endswith("$17.50")

    pass_section_start = next(
        idx for idx, line in enumerate(lines) if line.startswith("Pass-Through & Direct Costs")
    )
    pass_section_end = next(
        idx
        for idx in range(pass_section_start, len(lines))
        if lines[idx] == ""
    )
    pass_section = lines[pass_section_start:pass_section_end]
    assert any(line.strip().startswith("Total") and line.strip().endswith("$17.50") for line in pass_section)


def test_render_quote_renders_bucket_table_without_planner_why_section() -> None:
    bucket_view = {
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
    }

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
        "bucket_view": bucket_view,
        "process_plan": {"bucket_view": bucket_view},
        "decision_state": {"effective": {"setups": 2, "part_count": 4}},
    }

    result = {"price": 1450.0, "breakdown": breakdown}

    previous_override = appV5.SHOW_BUCKET_DIAGNOSTICS_OVERRIDE
    appV5.SHOW_BUCKET_DIAGNOSTICS_OVERRIDE = True
    try:
        rendered = appV5.render_quote(result, currency="$")
    finally:
        appV5.SHOW_BUCKET_DIAGNOSTICS_OVERRIDE = previous_override
    lines = rendered.splitlines()

    assert all("Planner operations" not in line for line in lines)

    header_idx = next(
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("Bucket") and "Hours" in line and "Labor $" in line
    )
    separator_idx = header_idx + 1
    separator = lines[separator_idx].replace("|", "").replace(" ", "")
    assert separator
    assert set(separator) == {"-"}

    bucket_rows: list[str] = []
    for line in lines[header_idx + 2 :]:
        if not line.strip():
            break
        bucket_rows.append(line)

    assert any("Milling" in line and "$500.00" in line and "2.00" in line for line in bucket_rows)
    assert any("Drilling" in line and "$300.00" in line and "1.50" in line for line in bucket_rows)
    assert any("Inspection" in line and "$100.00" in line and "1.00" in line for line in bucket_rows)


def test_planner_rollup_hours_reconcile_with_bucket_view() -> None:
    bucket_view = {
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
        }
    }

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
        "bucket_view": bucket_view,
        "process_plan": {"bucket_view": bucket_view},
        "decision_state": {"effective": {"setups": 2, "part_count": 4}},
    }

    result = {"price": 1450.0, "breakdown": breakdown}

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    def _extract_hour(label: str) -> float:
        for line in lines:
            if line.strip().startswith(label):
                match = re.search(r"([0-9]+(?:\.[0-9]+)?) hr", line)
                if match:
                    return float(match.group(1))
        raise AssertionError(f"Missing hour line for {label!r}")

    planner_total = _extract_hour("Planner Total")
    planner_labor = _extract_hour("Planner Labor")
    planner_machine = _extract_hour("Planner Machine")

    assert math.isclose(planner_labor + planner_machine, planner_total, abs_tol=0.05)

    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")

    laborish = {
        "finishing_deburr",
        "inspection",
        "assembly",
        "toolmaker_support",
        "ehs_compliance",
        "fixture_build_amortized",
        "programming_amortized",
    }

    expected_machine = 0.0
    expected_labor = 0.0
    for key, info in bucket_view["buckets"].items():
        minutes = float(info.get("minutes", 0.0) or 0.0)
        hours = minutes / 60.0 if minutes else 0.0
        if hours <= 0.0:
            continue
        if _norm(key) in laborish:
            expected_labor += hours
        else:
            expected_machine += hours

    assert math.isclose(planner_machine, expected_machine, abs_tol=0.05)
    assert math.isclose(planner_labor, expected_labor, abs_tol=0.05)


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
    bucket_view = {
        "buckets": {
            "milling": {
                "minutes": 180.0,
                "labor$": 40.0,
                "machine$": 0.0,
            },
            "Deburr": {
                "minutes": 30.0,
                "labor$": 10.0,
                "machine$": 0.0,
            },
            "Finishing/Deburr": {
                "minutes": 15.0,
                "labor$": 5.0,
                "machine$": 0.0,
            },
        }
    }
    bucket_view.setdefault("order", ["milling", "Deburr", "Finishing/Deburr"])
    bucket_view["buckets"]["programming_amortized"] = {
        "minutes": 45.0,
        "labor$": 15.0,
        "machine$": 0.0,
    }
    bucket_view["order"].append("programming_amortized")

    result = {
        "price": 150.0,
        "breakdown": {
            "qty": 2,
            "totals": {
                "labor_cost": 55.0,
                "direct_costs": 15.0,
                "subtotal": 70.0,
                "with_overhead": 77.0,
                "with_ga": 80.85,
                "with_contingency": 82.467,
                "with_expedite": 82.467,
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
            "pass_through": {"Material": 15.0},
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
            "bucket_view": bucket_view,
            "process_plan": {"bucket_view": bucket_view},
            "process_plan_summary": {"bucket_view": bucket_view},
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
    bucket_view = {
        "buckets": {
            "milling": {
                "minutes": 180.0,
                "labor$": 40.0,
                "machine$": 0.0,
            },
            "Deburr": {
                "minutes": 30.0,
                "labor$": 10.0,
                "machine$": 0.0,
            },
            "Finishing/Deburr": {
                "minutes": 15.0,
                "labor$": 5.0,
                "machine$": 0.0,
            },
        }
    }
    bucket_view["buckets"]["programming_amortized"] = {
        "minutes": 45.0,
        "labor$": 15.0,
        "machine$": 0.0,
    }
    bucket_view["order"] = [
        "milling",
        "Deburr",
        "Finishing/Deburr",
        "programming_amortized",
    ]

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
            "nre": {"programming_per_lot": 15.0},
            "material": {},
            "process_costs": {
                "milling": 60.0,
                "Programming (amortized)": 15.0,
            },
            "process_meta": {
                "milling": {"hr": 2.5},
                "Programming (amortized)": {"hr": 0.75},
            },
            "pass_through": {"Material": 20.0},
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
            "bucket_view": bucket_view,
            "process_plan": {"bucket_view": bucket_view},
            "process_plan_summary": {"bucket_view": bucket_view},
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
    bucket_view = {
        "buckets": {
            "milling": {
                "minutes": 120.0,
                "labor$": 120.0,
                "machine$": 0.0,
            },
            "deburr": {
                "minutes": 60.0,
                "labor$": 80.0,
                "machine$": 0.0,
            },
        },
        "order": ["milling", "deburr"],
    }
    bucket_view["buckets"]["programming_amortized"] = {
        "minutes": 120.0,
        "labor$": 30.0,
        "machine$": 0.0,
    }
    bucket_view["buckets"]["fixture_build_amortized"] = {
        "minutes": 90.0,
        "labor$": 20.0,
        "machine$": 0.0,
    }
    bucket_view["order"].extend(["programming_amortized", "fixture_build_amortized"])

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
            "bucket_view": bucket_view,
            "process_plan": {"bucket_view": bucket_view},
            "process_plan_summary": {"bucket_view": bucket_view},
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
                "direct_costs": 42.0,
                "subtotal": 120.0,
                "with_overhead": 132.0,
                "with_ga": 138.6,
                "with_contingency": 141.372,
                "with_expedite": 141.372,
            },
            "nre_detail": {
                "programming": {"prog_hr": 2.0, "amortized": True},
                "fixture": {"build_hr": 1.5, "labor_cost": 90.0, "build_rate": 60.0},
            },
            "nre": {},
            "material": {},
            "process_costs": {"milling": 60.0},
            "process_meta": {"milling": {"hr": 3.0}},
            "pass_through": {"Material": 30.0, "Shipping": 12.0},
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
        "Programming (amortized)" in line and "0.40 hr" in line
        for line in summary_block
    )
    assert any("Fixture Build" in line and "1.50 hr" in line for line in summary_block)
    assert any(
        "Fixture Build (amortized)" in line and "0.30 hr" in line
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
                "direct_costs": 30.0,
                "subtotal": 30.0,
                "with_overhead": 30.0,
                "with_ga": 30.0,
                "with_contingency": 30.0,
                "with_expedite": 30.0,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {},
            "process_meta": {},
            "pass_through": {"Material": 30.0},
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
            "pass_through": {"Material": 30.0},
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
    bucket_view = {
        "buckets": {
            "milling": {
                "total$": 120.0,
                "machine$": 70.0,
                "labor$": 50.0,
                "minutes": 180.0,
            },
            "deburr": {
                "total$": 60.0,
                "machine$": 30.0,
                "labor$": 30.0,
                "minutes": 60.0,
            },
        }
    }

    result = {
        "price": 200.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 30.0,
                "direct_costs": 120.0,
                "subtotal": 150.0,
                "with_overhead": 165.0,
                "with_ga": 173.25,
                "with_contingency": 176.715,
                "with_expedite": 176.715,
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
            "bucket_view": bucket_view,
            "process_plan": {"bucket_view": bucket_view},
            "pass_through": {"Material": 120.0},
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


def test_total_labor_cost_matches_process_rows() -> None:
    result = {
        "price": 110.0,
        "breakdown": {
            "qty": 4,
            "totals": {
                "labor_cost": 50.0,
                "direct_costs": 34.0,
                "subtotal": 84.0,
                "with_overhead": 92.4,
                "with_ga": 97.02,
                "with_contingency": 98.9604,
                "with_expedite": 98.9604,
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
    assert total_labor_amount == table_total_amount


def test_render_quote_displays_single_shipping_entry_and_reconciles_ladder() -> None:
    result = {
        "price": 100.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 50.0,
                "direct_costs": 50.0,
                "subtotal": 100.0,
                "with_overhead": 100.0,
                "with_ga": 100.0,
                "with_contingency": 100.0,
                "with_expedite": 100.0,
            },
            "material": {
                "material": "Aluminum 6061",
                "material_cost": 40.0,
                "mass_g": 1200.0,
                "net_mass_g": 1000.0,
            },
            "process_costs": {"milling": 50.0},
            "process_meta": {"milling": {"hr": 1.0, "rate": 50.0}},
            "pass_through": {"Material": 40.0, "Shipping": 10.0},
            "applied_pcts": {
                "OverheadPct": 0.0,
                "GA_Pct": 0.0,
                "ContingencyPct": 0.0,
                "MarginPct": 0.0,
            },
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    material_idx = lines.index("Material & Stock")
    material_end = next(idx for idx in range(material_idx, len(lines)) if lines[idx] == "")
    material_section = lines[material_idx:material_end]
    assert all("Shipping" not in line for line in material_section)

    pass_idx = next(
        idx for idx, line in enumerate(lines) if line.startswith("Pass-Through & Direct Costs")
    )
    pass_end = next(idx for idx in range(pass_idx, len(lines)) if lines[idx] == "")
    pass_section = lines[pass_idx:pass_end]
    shipping_lines = [line for line in pass_section if "Shipping" in line]
    assert len(shipping_lines) == 1

    ladder_line = next(
        line for line in lines if line.startswith("Subtotal (Labor + Directs):")
    )
    ladder_total = float(ladder_line.split("$")[-1].replace(",", ""))
    assert ladder_total == 100.0


def test_render_quote_final_price_tracks_pricing_ladder_math() -> None:
    result = {
        "price": 0.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 40.0,
                "direct_costs": 60.0,
                "subtotal": 100.0,
            },
            "material": {},
            "process_costs": {"milling": 40.0},
            "process_meta": {"milling": {"hr": 1.0, "rate": 40.0}},
            "pass_through": {"Material": 60.0},
            "applied_pcts": {
                "OverheadPct": 0.10,
                "GA_Pct": 0.05,
                "ContingencyPct": 0.02,
                "ExpeditePct": 0.03,
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

    def _extract_amount(prefix: str) -> float:
        line = next(line for line in lines if line.startswith(prefix))
        return float(line.split("$")[-1].replace(",", ""))

    subtotal = _extract_amount("Subtotal (Labor + Directs):")
    final_price_display = _extract_amount("Final Price with Margin")

    expected = round(subtotal, 2)
    for pct in (0.10, 0.05, 0.02, 0.03, 0.15):
        expected = round(expected * (1.0 + pct), 2)

    assert math.isclose(final_price_display, expected, abs_tol=0.01)
    assert math.isclose(result["price"], expected, abs_tol=0.01)

    totals = result["breakdown"]["totals"]
    assert math.isclose(totals["with_margin"], expected, abs_tol=0.01)
    assert math.isclose(totals["price"], expected, abs_tol=0.01)

def test_render_quote_promotes_planner_pricing_source() -> None:
    result = {
        "price": 0.0,
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
            "process_costs": {"milling": 0.0},
            "process_meta": {
                "planner_total": {"minutes": 120.0},
                "milling": {"hr": 0.0},
            },
            "pass_through": {"Material": 0.0},
            "pricing_source": "legacy",
            "applied_pcts": {
                "OverheadPct": 0.0,
                "GA_Pct": 0.0,
                "ContingencyPct": 0.0,
                "MarginPct": 0.0,
            },
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    assert "Pricing Source: Planner" in lines
    assert all("Pricing Source: Legacy" not in line for line in lines)


def test_render_quote_header_is_canonical() -> None:
    result = {
        "price": 0.0,
        "app_meta": {"used_planner": True},
        "speeds_feeds_path": "/mnt/speeds_feeds.csv",
        "speeds_feeds_loaded": True,
        "breakdown": {
            "qty": 2,
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
            "pricing_source": "legacy",
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
            "red_flags": [],
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()

    speeds_lines = [line for line in lines if line.startswith("Speeds/Feeds CSV:")]
    pricing_lines = [line for line in lines if line.startswith("Pricing Source:")]

    assert len(speeds_lines) == 1
    assert speeds_lines[0].endswith("(loaded)")
    assert len(pricing_lines) == 1
    assert pricing_lines[0] == "Pricing Source: Planner"


def test_render_quote_hour_summary_uses_final_hours() -> None:
    result = {
        "price": 160.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 160.0,
                "direct_costs": 0.0,
                "subtotal": 160.0,
                "with_overhead": 160.0,
                "with_ga": 160.0,
                "with_contingency": 160.0,
                "with_expedite": 160.0,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {"milling": 100.0, "drilling": 60.0},
            "process_meta": {
                "milling": {"hr": 2.0, "rate": 50.0},
                "drilling": {"hr": 1.5, "final_hr": 0.75, "rate": 80.0},
            },
            "labor_costs": {"Milling": 100.0, "Drilling": 60.0},
            "labor_cost_details": {
                "Milling": "2.00 hr @ $50.00/hr",
                "Drilling": "0.75 hr @ $80.00/hr",
            },
            "pass_through": {"Material": 0.0},
            "pricing_source": "legacy",
            "applied_pcts": {
                "OverheadPct": 0.0,
                "GA_Pct": 0.0,
                "ContingencyPct": 0.0,
                "MarginPct": 0.0,
            },
            "rates": {"MillingRate": 50.0, "DrillingRate": 80.0},
            "params": {},
            "direct_cost_details": {},
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()
    summary_idx = lines.index("Labor Hour Summary")
    summary_block = lines[summary_idx: summary_idx + 8]
    drilling_line = next(line for line in summary_block if "Drilling" in line)

    assert "0.75 hr" in drilling_line


def test_render_quote_direct_costs_match_displayed_pass_through() -> None:
    result = {
        "price": 385.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 160.0,
                "direct_costs": 225.0,
                "subtotal": 385.0,
                "with_overhead": 385.0,
                "with_ga": 385.0,
                "with_contingency": 385.0,
                "with_expedite": 385.0,
            },
            "nre_detail": {},
            "nre": {},
            "material": {
                "material_cost_before_credit": 200.0,
                "material_scrap_credit": 10.0,
                "material_tax": 5.0,
            },
            "process_costs": {"milling": 100.0, "drilling": 60.0},
            "process_meta": {
                "milling": {"hr": 2.0, "rate": 50.0},
                "drilling": {"hr": 0.75, "rate": 80.0},
            },
            "labor_costs": {"Milling": 100.0, "Drilling": 60.0},
            "labor_cost_details": {
                "Milling": "2.00 hr @ $50.00/hr",
                "Drilling": "0.75 hr @ $80.00/hr",
            },
            "pass_through": {
                "Material": 200.0,
                "Consumables": 30.0,
                "Hidden Fee": -10.0,
            },
            "pricing_source": "legacy",
            "applied_pcts": {
                "OverheadPct": 0.0,
                "GA_Pct": 0.0,
                "ContingencyPct": 0.0,
                "MarginPct": 0.0,
            },
            "rates": {"MillingRate": 50.0, "DrillingRate": 80.0},
            "params": {},
            "direct_cost_details": {"Consumables": "Shop supplies"},
        },
    }

    rendered = appV5.render_quote(result, currency="$")
    lines = rendered.splitlines()
    pass_idx = next(
        idx for idx, line in enumerate(lines) if line.startswith("Pass-Through & Direct Costs")
    )
    pass_end = next(idx for idx in range(pass_idx, len(lines)) if lines[idx] == "")
    pass_section = lines[pass_idx:pass_end]

    assert all("Hidden Fee" not in line for line in pass_section)
    total_line = next(line for line in pass_section if line.strip().startswith("Total"))
    assert total_line.endswith("$225.00")


def test_render_quote_backfills_programming_and_inspection_rates() -> None:
    result = {
        "price": 0.0,
        "breakdown": {
            "qty": 5,
            "totals": {
                "labor_cost": 0.0,
                "direct_costs": 0.0,
                "subtotal": 0.0,
                "with_overhead": 0.0,
                "with_ga": 0.0,
                "with_contingency": 0.0,
                "with_expedite": 0.0,
            },
            "nre_detail": {"programming": {"prog_hr": 2.0}},
            "nre": {"programming_hr": 3.0},
            "material": {},
            "process_costs": {},
            "process_meta": {},
            "pass_through": {},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
            "bucket_view": {
                "buckets": {
                    "inspection": {
                        "minutes": 60.0,
                        "labor$": 0.0,
                        "machine$": 0.0,
                    }
                },
                "order": ["inspection"],
            },
        },
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)
    lines = rendered.splitlines()

    assert any("Programming Cost:" in line and "$270" in line for line in lines)
    assert any("Programmer:" in line and "$90.00/hr" in line for line in lines)
    assert any("Programming Hrs:" in line and "3.00 hr" in line for line in lines)

    inspection_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip().startswith("Inspection")),
        None,
    )
    assert inspection_idx is not None
    assert "hr @" in lines[inspection_idx + 1]
    assert "/hr" in lines[inspection_idx + 1]
    assert "—/hr" not in lines[inspection_idx + 1]
