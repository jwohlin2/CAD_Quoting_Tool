import copy
import math
import re

import appV5
from cad_quoter.domain_models import normalize_material_key
from cad_quoter.pricing.speeds_feeds_selector import material_group_for_speeds_feeds


_PLANNER_LINE_ITEMS = [
    {
        "op": "Milling Rough",
        "minutes": 240.0,
        "machine_cost": 360.0,
        "labor_cost": 240.0,
    },
    {
        "op": "Drilling Ops",
        "minutes": 90.0,
        "machine_cost": 120.0,
        "labor_cost": 90.0,
    },
    {
        "op": "Surface Grind",
        "minutes": 120.0,
        "machine_cost": 180.0,
        "labor_cost": 150.0,
    },
    {
        "op": "Deburr & Finish",
        "minutes": 60.0,
        "labor_cost": 80.0,
    },
    {
        "op": "Inspection",
        "minutes": 45.0,
        "labor_cost": 40.0,
    },
]


_QUICK_CHECK_BREAKDOWN = {
    "material_block": {
        "stock_L_in": 12.0,
        "total_material_cost": 120.0,
    },
    "total_direct_costs": 150.0,
    "bucket_view": {"totals": {"total$": 200.0}},
    "nre": {"programming_per_part": 50.0},
    "price": 400.0,
    "process_plan": {"drilling": {}, "inspection": {}},
    "nre_detail": {"programming": {"auto_prog_hr": 1.2}},
}


DUMMY_QUOTE_RESULT = {
    "price": 1414.875,
    "qty": 12,
    "speeds_feeds_path": "/mnt/planner/feeds.csv",
    "drill_debug": [
        (
            "Drill calc → op=Planner Drill, mat=Aluminum MIC6, "
            "SFM=280–320, IPR=0.0030–0.0040; RPM 3400–3900 IPM 10.2–13.3; "
            "Ø 0.500\"; depth/hole 1.50 in; holes 48; index 8.0 s/hole; "
            "peck 0.08 min/hole; toolchange 0.50 min; total hr 1.50."
        )
    ],
    "ui_vars": {"Material": "Aluminum MIC6"},
    "decision_state": {
        "baseline": {
            "normalized_quote_material": "aluminum mic6",
            "drill_params": {"material": "aluminum mic6", "group": "N1"},
            "process_hours": {
                "milling": 6.0,
                "drilling": 1.5,
                "grinding": 2.0,
                "finishing_deburr": 1.0,
                "inspection": 0.75,
            },
            "used_planner": True,
            "pricing_source": "planner",
        },
        "suggestions": {},
        "user_overrides": {},
        "effective": {},
        "effective_sources": {},
    },
    "breakdown": {
        "qty": 12,
        "pricing_source": "planner",
        "pricing_source_text": "/mnt/planner/feeds.csv",
        "drilling_meta": {
            "material": "aluminum mic6",
            "material_display": "Aluminum MIC6",
            "speeds_feeds_path": "/mnt/planner/feeds.csv",
            "speeds_feeds_loaded": True,
        },
        "material": {
            "material": "Aluminum MIC6",
            "material_name": "Aluminum MIC6",
            "material_cost": 420.0,
            "mass_g": 98500.0,
            "net_mass_g": 91500.0,
            "scrap_pct": 0.07,
        },
        "material_selected": {"canonical": "Aluminum MIC6"},
        "process_costs": {
            "Machine": 660.0,
            "Labor": 600.0,
        },
        "process_meta": {
            "milling": {
                "minutes": 360.0,
                "hr": 6.0,
                "rate": 100.0,
                "labor$": 240.0,
                "machine$": 360.0,
                "$": 600.0,
                "total$": 600.0,
            },
            "drilling": {
                "minutes": 90.0,
                "hr": 1.5,
                "rate": 140.0,
                "labor$": 90.0,
                "machine$": 120.0,
                "$": 210.0,
                "total$": 210.0,
            },
            "grinding": {
                "minutes": 120.0,
                "hr": 2.0,
                "rate": 165.0,
                "labor$": 150.0,
                "machine$": 180.0,
                "$": 330.0,
                "total$": 330.0,
            },
            "finishing_deburr": {
                "minutes": 60.0,
                "hr": 1.0,
                "rate": 80.0,
                "labor$": 80.0,
                "machine$": 0.0,
                "$": 80.0,
                "total$": 80.0,
            },
            "inspection": {
                "minutes": 45.0,
                "hr": 0.75,
                "rate": 53.333333333333336,
                "labor$": 40.0,
                "machine$": 0.0,
                "$": 40.0,
                "total$": 40.0,
            },
            "planner_total": {
                "minutes": 675.0,
                "hr": 11.25,
                "cost": 1260.0,
                "labor_cost": 600.0,
                "machine_cost": 660.0,
                "line_items": copy.deepcopy(_PLANNER_LINE_ITEMS),
            },
            "planner_machine": {
                "minutes": 420.0,
                "hr": 7.0,
                "cost": 660.0,
            },
            "planner_labor": {
                "minutes": 675.0,
                "hr": 11.25,
                "cost": 600.0,
            },
        },
        "bucket_view": {
            "buckets": {
                "milling": {
                    "minutes": 360.0,
                    "labor$": 240.0,
                    "machine$": 360.0,
                    "total$": 600.0,
                },
                "drilling": {
                    "minutes": 90.0,
                    "labor$": 90.0,
                    "machine$": 120.0,
                    "total$": 210.0,
                },
                "grinding": {
                    "minutes": 120.0,
                    "labor$": 150.0,
                    "machine$": 180.0,
                    "total$": 330.0,
                },
                "finishing_deburr": {
                    "minutes": 60.0,
                    "labor$": 80.0,
                    "machine$": 0.0,
                    "total$": 80.0,
                },
                "inspection": {
                    "minutes": 45.0,
                    "labor$": 40.0,
                    "machine$": 0.0,
                    "total$": 40.0,
                },
            }
        },
        "hour_summary": {
            "order": [
                "milling",
                "drilling",
                "grinding",
                "finishing_deburr",
                "inspection",
            ],
            "buckets": {
                "milling": {
                    "label": "Milling",
                    "hr": 6.0,
                    "rate": 100.0,
                    "$": 600.0,
                    "labor$": 240.0,
                    "machine$": 360.0,
                },
                "drilling": {
                    "label": "Drilling",
                    "hr": 1.5,
                    "rate": 140.0,
                    "$": 210.0,
                    "labor$": 90.0,
                    "machine$": 120.0,
                },
                "grinding": {
                    "label": "Grinding",
                    "hr": 2.0,
                    "rate": 165.0,
                    "$": 330.0,
                    "labor$": 150.0,
                    "machine$": 180.0,
                },
                "finishing_deburr": {
                    "label": "Finishing/Deburr",
                    "hr": 1.0,
                    "rate": 80.0,
                    "$": 80.0,
                    "labor$": 80.0,
                    "machine$": 0.0,
                },
                "inspection": {
                    "label": "Inspection",
                    "hr": 0.75,
                    "rate": 53.333333333333336,
                    "$": 40.0,
                    "labor$": 40.0,
                    "machine$": 0.0,
                },
            },
            "total_hours": 11.25,
        },
        "labor_costs": {
            "Milling": 600.0,
            "Drilling": 210.0,
            "Grinding": 330.0,
            "Finishing/Deburr": 80.0,
            "Inspection": 40.0,
        },
        "labor_cost_details": {
            "Milling": "6.00 hr @ $100.00/hr",
            "Drilling": "1.50 hr @ $80.00/hr",
            "Grinding": "2.00 hr @ $90.00/hr",
            "Finishing/Deburr": "1.00 hr @ $80.00/hr",
            "Inspection": "0.75 hr @ $95.00/hr",
        },
        "nre_detail": {
            "programming": {"prog_hr": 4.0, "prog_rate": 95.0, "amortized": False},
            "fixture": {"build_hr": 3.0, "build_rate": 85.0},
        },
        "nre": {
            "programming_per_lot": 0.0,
            "fixture_per_part": 0.0,
            "extra_nre_cost": 0.0,
        },
        "nre_cost_details": {},
        "pass_through": {
            "Material": 420.0,
            "Shipping": 65.0,
            "Consumables": 35.0,
        },
        "pass_meta": {},
        "totals": {
            "labor_cost": 1260.0,
            "direct_costs": 520.0,
            "subtotal": 1780.0,
            "with_overhead": 1993.6,
            "with_ga": 2093.28,
            "with_contingency": 2093.28,
            "with_expedite": 2093.28,
            "price": 2511.94,
        },
        "applied_pcts": {
            "OverheadPct": 0.12,
            "GA_Pct": 0.05,
            "ContingencyPct": 0.0,
            "MarginPct": 0.2,
        },
        "rates": {
            "MillingRate": 100.0,
            "DrillingRate": 80.0,
            "SurfaceGrindRate": 90.0,
            "DeburrRate": 80.0,
            "InspectionRate": 95.0,
        },
        "params": {},
        "labor_cost_details_input": {},
        "drill_debug": [
            "Planner drill analysis: material Aluminum MIC6 with matched feeds",
        ],
        "direct_cost_details": {},
        "app_meta": {"used_planner": True},
    },
}


_EXPECTED_BUCKET_ROWS = {
    "Milling": (6.0, 240.0, 360.0, 600.0),
    "Drilling": (1.5, 90.0, 120.0, 210.0),
    "Grinding": (2.0, 150.0, 180.0, 330.0),
    "Finishing/Deburr": (1.0, 80.0, 0.0, 80.0),
    "Inspection": (0.75, 40.0, 0.0, 40.0),
}


def _dummy_quote_payload(*, debug_enabled: bool = False) -> dict:
    payload = copy.deepcopy(DUMMY_QUOTE_RESULT)
    if not debug_enabled:
        payload.pop("drill_debug", None)
        breakdown = payload.get("breakdown")
        if isinstance(breakdown, dict):
            breakdown.pop("drill_debug", None)
    return payload


def _render_lines(payload: dict, *, drop_planner_display: bool = False) -> list[str]:
    if drop_planner_display:
        payload = copy.deepcopy(payload)
        breakdown = payload.get("breakdown", {})
        if isinstance(breakdown, dict):
            breakdown.pop("bucket_view", None)
            breakdown.pop("planner_bucket_rollup", None)
            breakdown.pop("planner_bucket_display_map", None)
            breakdown.pop("process_plan", None)
    rendered = appV5.render_quote(payload, currency="$")
    return rendered.splitlines()


def _extract_currency(line: str) -> float:
    match = re.search(r"\$([0-9,]+\.[0-9]{2})", line)
    assert match, f"expected currency value in line: {line!r}"
    return float(match.group(1).replace(",", ""))


def test_dummy_quote_material_consistency() -> None:
    payload = _dummy_quote_payload(debug_enabled=True)
    baseline = payload["decision_state"]["baseline"]
    drilling_meta = payload["breakdown"]["drilling_meta"]
    drill_debug = payload["breakdown"]["drill_debug"]

    normalized = baseline["normalized_quote_material"]
    drill_material = baseline["drill_params"]["material"]
    drill_group = baseline["drill_params"].get("group")
    rendered_material = drilling_meta["material"]

    normalized_key = normalize_material_key(normalized)
    drill_key = normalize_material_key(drill_material)
    rendered_key = normalize_material_key(rendered_material)
    expected_group = material_group_for_speeds_feeds(normalized_key)

    assert normalized_key == drill_key == rendered_key
    assert drill_group == expected_group
    assert drill_group
    assert any(normalized_key in entry.lower() for entry in drill_debug)


def test_dummy_quote_pricing_source_reflects_planner_usage() -> None:
    payload = _dummy_quote_payload()
    baseline = payload["decision_state"]["baseline"]
    assert baseline["used_planner"] is True

    breakdown = payload["breakdown"]
    assert breakdown["pricing_source"].lower() == "planner"
    assert breakdown.get("pricing_source_text", "").lower() != "legacy"


def test_quick_breakdown_sanity_checks() -> None:
    breakdown = copy.deepcopy(_QUICK_CHECK_BREAKDOWN)

    material_block = breakdown.get("material_block", {})
    assert material_block.get("stock_L_in") is not None

    total_material_cost = float(material_block.get("total_material_cost", 0.0))
    assert breakdown["total_direct_costs"] >= total_material_cost >= 0.0

    ops = set((breakdown.get("process_plan") or {}).keys())
    programming_detail = breakdown["nre_detail"]["programming"]
    if ops.issubset({"drilling", "inspection"}):
        auto_prog_hr = float(programming_detail["auto_prog_hr"])
        assert 0.4 <= auto_prog_hr <= 3.0

    calc = (
        float(breakdown["bucket_view"]["totals"]["total$"])
        + float(breakdown["nre"]["programming_per_part"])
        + float(breakdown["total_direct_costs"])
    )
    assert math.isclose(float(breakdown["price"]), calc, abs_tol=0.01)


def test_dummy_quote_process_table_matches_planner_totals() -> None:
    payload = _dummy_quote_payload()
    bucket_view = payload["breakdown"]["bucket_view"]["buckets"]
    table_rows: dict[str, tuple[float, float, float, float]] = {}
    for key, metrics in bucket_view.items():
        label = {
            "milling": "Milling",
            "drilling": "Drilling",
            "grinding": "Grinding",
            "finishing_deburr": "Finishing/Deburr",
            "inspection": "Inspection",
        }[key]
        hours_val = float(metrics.get("minutes", 0.0)) / 60.0
        labor_val = float(metrics.get("labor$", 0.0))
        machine_val = float(metrics.get("machine$", 0.0))
        total_val = float(metrics.get("total$", labor_val + machine_val))
        table_rows[label] = (round(hours_val, 2), labor_val, machine_val, total_val)

    assert table_rows == _EXPECTED_BUCKET_ROWS

    total_hours = sum(row[0] for row in table_rows.values())
    total_labor = sum(row[1] for row in table_rows.values())
    total_machine = sum(row[2] for row in table_rows.values())
    total_cost = sum(row[3] for row in table_rows.values())

    process_meta = payload["breakdown"]["process_meta"]
    planner_total = process_meta["planner_total"]
    planner_machine = process_meta["planner_machine"]
    planner_labor = process_meta["planner_labor"]

    assert math.isclose(total_cost, planner_total["cost"], abs_tol=0.01)
    assert math.isclose(total_machine, planner_machine["cost"], abs_tol=0.01)
    assert math.isclose(total_labor, planner_labor["cost"], abs_tol=0.01)
    assert math.isclose(total_hours, planner_total["minutes"] / 60.0, abs_tol=0.01)


def test_dummy_quote_hour_summary_aligns_with_planner_buckets() -> None:
    payload = _dummy_quote_payload()
    bucket_view = payload["breakdown"]["bucket_view"]["buckets"]
    baseline_hours = payload["decision_state"]["baseline"]["process_hours"]
    process_meta = payload["breakdown"]["process_meta"]
    hour_summary = payload["breakdown"]["hour_summary"]
    summary_buckets = hour_summary["buckets"]

    for key, expected_label in (
        ("milling", "milling"),
        ("drilling", "drilling"),
        ("grinding", "grinding"),
        ("finishing_deburr", "finishing_deburr"),
        ("inspection", "inspection"),
    ):
        metrics = bucket_view[key]
        bucket_hours = float(metrics["minutes"]) / 60.0
        total_cost = float(metrics.get("total$", 0.0))
        if total_cost == 0.0:
            total_cost = float(metrics.get("labor$", 0.0)) + float(metrics.get("machine$", 0.0))
        assert math.isclose(bucket_hours, baseline_hours[expected_label], abs_tol=0.01)

        meta_entry = process_meta[key]
        summary_entry = summary_buckets[key]

        assert math.isclose(float(meta_entry["hr"]), bucket_hours, abs_tol=0.01)
        assert math.isclose(float(meta_entry["$"]), total_cost, abs_tol=0.01)
        assert math.isclose(float(summary_entry["hr"]), bucket_hours, abs_tol=0.01)
        assert math.isclose(float(summary_entry["$"]), total_cost, abs_tol=0.01)

        if bucket_hours > 0:
            expected_rate = total_cost / bucket_hours
            assert math.isclose(float(meta_entry["rate"]), expected_rate, abs_tol=0.01)
            assert math.isclose(float(summary_entry["rate"]), expected_rate, abs_tol=0.01)

    programming_meta = payload["breakdown"]["nre_detail"]["programming"]
    fixture_meta = payload["breakdown"]["nre_detail"]["fixture"]
    assert programming_meta["prog_hr"] == 4.0
    assert fixture_meta["build_hr"] == 3.0


def test_dummy_quote_has_no_planner_red_flags() -> None:
    payload = _dummy_quote_payload()
    assert "red_flags" not in payload["breakdown"]


def test_dummy_quote_pricing_source_header() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]
    drill_meta = breakdown["drilling_meta"]

    assert str(breakdown["pricing_source"]).lower() == "planner"

    path = str(breakdown.get("pricing_source_text") or "").strip()
    speeds_path = str(drill_meta.get("speeds_feeds_path") or "").strip()

    assert path
    assert speeds_path
    assert path == speeds_path


def test_dummy_quote_drill_debug_material_alignment() -> None:
    payload = _dummy_quote_payload(debug_enabled=True)
    breakdown = payload["breakdown"]
    debug_lines = payload.get("drill_debug", [])

    assert debug_lines, "expected drill debug sample"

    drill_line = next(line for line in debug_lines if line.startswith("Drill calc"))
    material_block = breakdown["material"]["material"]

    match = re.search(r"mat=([^,]+)", drill_line)
    assert match is not None
    debug_material = match.group(1).strip()

    debug_key = normalize_material_key(debug_material)
    breakdown_key = normalize_material_key(material_block)
    drilling_meta_key = normalize_material_key(breakdown["drilling_meta"]["material"])

    assert debug_key == breakdown_key == drilling_meta_key
    assert breakdown["drilling_meta"].get("material_display") == material_block


def test_dummy_quote_drill_debug_ranges_and_index() -> None:
    payload = _dummy_quote_payload(debug_enabled=True)
    debug_line = next(line for line in payload.get("drill_debug", []) if line.startswith("Drill calc"))

    index_match = re.search(r"index\s+([0-9]+(?:\.[0-9]+)?)\s*s/hole", debug_line)
    assert index_match is not None
    index_seconds = float(index_match.group(1))
    assert 6.0 <= index_seconds <= 10.0

    sfm_match = re.search(r"SFM=([0-9.]+)–([0-9.]+)", debug_line)
    ipr_match = re.search(r"IPR=([0-9.]+)–([0-9.]+)", debug_line)

    assert sfm_match is not None
    assert ipr_match is not None

    sfm_low, sfm_high = map(float, sfm_match.groups())
    ipr_low, ipr_high = map(float, ipr_match.groups())

    assert sfm_low < sfm_high
    assert ipr_low < ipr_high


def test_dummy_quote_drilling_bucket_matches_summary() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]

    bucket_minutes = breakdown["bucket_view"]["buckets"]["drilling"]["minutes"]
    summary_hr = breakdown["process_meta"]["drilling"]["hr"]

    assert abs(bucket_minutes / 60.0 - summary_hr) <= 1e-6

    red_flags = [flag.lower() for flag in breakdown.get("red_flags", [])]
    assert all("drift" not in flag for flag in red_flags)


def test_dummy_quote_has_no_csv_debug_duplicates() -> None:
    payload = _dummy_quote_payload(debug_enabled=True)

    assert not any(
        line.strip().startswith("CSV drill feeds (N1)")
        for line in payload.get("drill_debug", [])
    )


def test_dummy_quote_render_avoids_duplicate_planner_tables() -> None:
    lines = _render_lines(_dummy_quote_payload())
    assert all("Planner diagnostics (not billed)" not in line for line in lines)


def test_dummy_quote_render_has_no_planner_drift_note() -> None:
    lines = [line.lower() for line in _render_lines(_dummy_quote_payload())]
    assert all("drifted by" not in line for line in lines)


def test_dummy_quote_bucket_hours_and_costs_align() -> None:
    payload = _dummy_quote_payload()
    bucket_view = payload["breakdown"]["bucket_view"]["buckets"]

    bucket_label_map = {
        "milling": "Milling",
        "drilling": "Drilling",
        "grinding": "Grinding",
        "finishing_deburr": "Finishing/Deburr",
        "inspection": "Inspection",
    }

    buckets_with_costs: set[str] = set()
    for key, metrics in bucket_view.items():
        label = bucket_label_map.get(
            key,
            key.replace("_", " ").replace(" deburr", "/Deburr").title(),
        )
        minutes = float(metrics.get("minutes", 0.0) or 0.0)
        labor_cost = float(metrics.get("labor$", 0.0) or 0.0)
        machine_cost = float(metrics.get("machine$", 0.0) or 0.0)
        total_cost = float(metrics.get("total$", labor_cost + machine_cost))
        if minutes <= 0 and total_cost <= 0:
            continue
        buckets_with_costs.add(label)

    labor_costs = payload["breakdown"]["labor_costs"]
    labor_rows = {
        label
        for label, amount in labor_costs.items()
        if float(amount or 0.0) > 0.01
    }

    assert buckets_with_costs == labor_rows

    baseline_hours = payload["decision_state"]["baseline"]["process_hours"]
    summary_labels = {
        bucket_label_map.get(
            key,
            key.replace("_", " ").replace(" deburr", "/Deburr").title(),
        )
        for key, hours in baseline_hours.items()
        if float(hours or 0.0) > 0.0
    }

    assert summary_labels == buckets_with_costs


def test_dummy_quote_direct_costs_match_across_sections() -> None:
    payload = _dummy_quote_payload()
    lines = _render_lines(payload)

    material_block = payload["breakdown"].get("material", {})
    material_total = (
        material_block.get("material_cost_before_credit")
        or material_block.get("material_cost")
        or material_block.get("material_direct_cost")
        or 0.0
    )
    direct_costs = appV5._compute_direct_costs(
        material_total,
        material_block.get("material_scrap_credit"),
        material_block.get("material_tax"),
        payload["breakdown"].get("pass_through"),
    )

    top_direct_line = next(line for line in lines if "Total Direct Costs:" in line)
    top_direct = _extract_currency(top_direct_line)
    assert math.isclose(top_direct, direct_costs, abs_tol=0.01)

    pass_section_start = next(
        idx for idx, line in enumerate(lines) if line.startswith("Pass-Through & Direct Costs")
    )
    pass_section: list[str] = []
    for line in lines[pass_section_start + 2 :]:
        if not line.strip():
            break
        pass_section.append(line)
    pass_header_line = lines[pass_section_start]
    assert math.isclose(_extract_currency(pass_header_line), direct_costs, abs_tol=0.01)
    material_line = next(line for line in pass_section if "Material & Stock" in line)
    assert math.isclose(_extract_currency(material_line), material_total, abs_tol=0.01)
    material_detail_line = next(
        line for line in pass_section if "contributes" in line and "Material & Stock" in line
    )
    assert math.isclose(_extract_currency(material_detail_line), material_total, abs_tol=0.01)

    pass_through_block = payload["breakdown"].get("pass_through", {}) or {}
    shipping_amount = float(pass_through_block.get("Shipping", 0.0))
    shipping_line = next(line for line in pass_section if "Shipping" in line)
    assert math.isclose(_extract_currency(shipping_line), shipping_amount, abs_tol=0.01)

    for label, amount in pass_through_block.items():
        if label in {"Material", "Shipping"}:
            continue
        amount_val = float(amount or 0.0)
        if amount_val <= 0 and all(label not in entry for entry in pass_section):
            continue
        detail_line = next(line for line in pass_section if label in line)
        assert math.isclose(_extract_currency(detail_line), amount_val, abs_tol=0.01)

    pass_total_line = next(line for line in pass_section if line.strip().startswith("Total"))
    pass_total = _extract_currency(pass_total_line)
    assert math.isclose(pass_total, direct_costs, abs_tol=0.01)

    ladder_line = next(line for line in lines if "Subtotal (Labor + Directs):" in line)
    ladder_subtotal = _extract_currency(ladder_line)

    assert lines.count("Process & Labor Costs") == 1
    process_idx = lines.index("Process & Labor Costs")
    process_end = next(
        (i for i in range(process_idx, len(lines)) if lines[i] == ""), len(lines)
    )
    process_block = lines[process_idx + 2 : process_end]
    total_line = next(line for line in process_block if line.strip().startswith("Total"))
    labor_amount = _extract_currency(total_line)
    ladder_direct = ladder_subtotal - labor_amount

    expected_ladder_direct = direct_costs
    assert math.isclose(ladder_direct, expected_ladder_direct, abs_tol=0.01)
    assert math.isclose(ladder_subtotal, labor_amount + expected_ladder_direct, abs_tol=0.01)


def test_compute_direct_costs_adds_wieland_scrap_credit() -> None:
    material_detail = {
        "scrap_credit_mass_lb": 5.0,
        "scrap_credit_unit_price_usd_per_lb": 2.5,
    }

    total = appV5._compute_direct_costs(
        100.0,
        0.0,
        0.0,
        {},
        material_detail=material_detail,
        scrap_price_source="wieland",
    )

    expected_credit = 5.0 * 2.5 * appV5.SCRAP_RECOVERY_DEFAULT
    expected_total = round(100.0 - expected_credit, 2)
    assert math.isclose(total, expected_total, abs_tol=1e-6)


def test_dummy_quote_ladder_uses_pricing_direct_costs_when_available() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]
    direct_costs = float(breakdown["totals"].get("direct_costs", 0.0))
    machine_cost = float(breakdown.get("process_costs", {}).get("Machine", 0.0))
    pricing_directs = direct_costs + machine_cost + 12.34
    payload["pricing"] = {"total_direct_costs": pricing_directs}

    lines = _render_lines(payload)

    process_idx = lines.index("Process & Labor Costs")
    process_end = next(
        (i for i in range(process_idx, len(lines)) if lines[i] == ""), len(lines)
    )
    process_block = lines[process_idx + 2 : process_end]
    total_line = next(line for line in process_block if line.strip().startswith("Total"))
    ladder_labor = _extract_currency(total_line)

    ladder_line = next(line for line in lines if "Subtotal (Labor + Directs):" in line)
    ladder_subtotal = _extract_currency(ladder_line)
    ladder_direct = ladder_subtotal - ladder_labor

    direct_costs_total = float(breakdown["totals"].get("direct_costs", 0.0))

    assert math.isclose(ladder_direct, direct_costs_total, abs_tol=0.01)
    assert math.isclose(ladder_subtotal, ladder_labor + direct_costs_total, abs_tol=0.01)


def test_render_omits_amortized_rows_for_single_quantity() -> None:
    payload = _dummy_quote_payload()
    payload["qty"] = 1
    payload["breakdown"]["qty"] = 1
    lines = _render_lines(payload)

    assert any("programming (amortized)" in line.lower() for line in lines)

    assert lines.count("Process & Labor Costs") == 1
    process_idx = lines.index("Process & Labor Costs")
    process_end = next(
        (i for i in range(process_idx, len(lines)) if lines[i] == ""), len(lines)
    )
    process_rows = lines[process_idx + 2 : process_end]
    assert any("programming (amortized)" in line.lower() for line in process_rows)
