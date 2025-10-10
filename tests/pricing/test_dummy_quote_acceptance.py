import copy
import re


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
            "normalized_quote_material": "Aluminum MIC6",
            "drill_params": {"material": "Aluminum MIC6"},
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
            "material": "Aluminum MIC6",
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
            "milling": {"minutes": 360.0, "hr": 6.0, "rate": 100.0},
            "drilling": {"minutes": 90.0, "hr": 1.5, "rate": 80.0},
            "grinding": {"minutes": 120.0, "hr": 2.0, "rate": 90.0},
            "finishing_deburr": {"minutes": 60.0, "hr": 1.0, "rate": 80.0},
            "inspection": {"minutes": 45.0, "hr": 0.75, "rate": 95.0},
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
                    "labor$": 600.0,
                    "machine$": 0.0,
                    "total$": 600.0,
                },
                "drilling": {
                    "minutes": 90.0,
                    "labor$": 120.0,
                    "machine$": 0.0,
                    "total$": 120.0,
                },
                "finishing": {
                    "minutes": 90.0,
                    "labor$": 90.0,
                    "machine$": 0.0,
                    "total$": 90.0,
                },
                "inspection": {
                    "minutes": 60.0,
                    "labor$": 95.0,
                    "machine$": 0.0,
                    "total$": 95.0,
                },
            }
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
            "programming_per_part": 0.0,
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
        "red_flags": [],
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


def _dummy_quote_payload() -> dict:
    return copy.deepcopy(DUMMY_QUOTE_RESULT)


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


def test_dummy_quote_material_consistency() -> None:
    payload = _dummy_quote_payload()
    baseline = payload["decision_state"]["baseline"]
    drilling_meta = payload["breakdown"]["drilling_meta"]
    drill_debug = payload["breakdown"]["drill_debug"]

    normalized = baseline["normalized_quote_material"]
    drill_material = baseline["drill_params"]["material"]
    rendered_material = drilling_meta["material"]

    assert normalized == drill_material == rendered_material
    assert any(normalized in entry for entry in drill_debug)


def test_dummy_quote_pricing_source_reflects_planner_usage() -> None:
    payload = _dummy_quote_payload()
    baseline = payload["decision_state"]["baseline"]
    assert baseline["used_planner"] is True

    breakdown = payload["breakdown"]
    assert breakdown["pricing_source"].lower() == "planner"
    assert breakdown.get("pricing_source_text", "").lower() != "legacy"


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

    for key, expected_label in (
        ("milling", "milling"),
        ("drilling", "drilling"),
        ("grinding", "grinding"),
        ("finishing_deburr", "finishing_deburr"),
        ("inspection", "inspection"),
    ):
        bucket_hours = float(bucket_view[key]["minutes"]) / 60.0
        assert math.isclose(bucket_hours, baseline_hours[expected_label], abs_tol=0.01)

    programming_meta = payload["breakdown"]["nre_detail"]["programming"]
    fixture_meta = payload["breakdown"]["nre_detail"]["fixture"]
    assert programming_meta["prog_hr"] == 4.0
    assert fixture_meta["build_hr"] == 3.0


def test_dummy_quote_has_no_planner_red_flags() -> None:
    payload = _dummy_quote_payload()
    assert "red_flags" not in payload["breakdown"]

    assert abs((labor + direct) - subtotal) <= 0.01


def test_dummy_quote_pricing_source_header() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]
    drill_meta = breakdown["drilling_meta"]

    assert breakdown["pricing_source"] == "planner"

    path = str(breakdown.get("pricing_source_text") or "").strip()
    speeds_path = str(drill_meta.get("speeds_feeds_path") or "").strip()

    assert path
    assert speeds_path
    assert path == speeds_path


def test_dummy_quote_drill_debug_material_alignment() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]
    debug_lines = payload.get("drill_debug", [])

    assert debug_lines, "expected drill debug sample"

    drill_line = next(line for line in debug_lines if line.startswith("Drill calc"))
    material_block = breakdown["material"]["material"]

    match = re.search(r"mat=([^,]+)", drill_line)
    assert match is not None
    debug_material = match.group(1).strip()

    assert debug_material == material_block == breakdown["drilling_meta"]["material"]


def test_dummy_quote_drill_debug_ranges_and_index() -> None:
    payload = _dummy_quote_payload()
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
    payload = _dummy_quote_payload()

    assert not any(
        line.strip().startswith("CSV drill feeds (N1)")
        for line in payload.get("drill_debug", [])
    )
