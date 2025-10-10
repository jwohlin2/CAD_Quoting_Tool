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
                "finishing_deburr": 1.5,
                "inspection": 1.0,
            },
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
            "material_cost": 400.0,
            "mass_g": 125000.0,
            "net_mass_g": 112500.0,
            "scrap_pct": 0.1,
        },
        "material_selected": {"canonical": "Aluminum MIC6"},
        "process_costs": {
            "milling": 600.0,
            "drilling": 120.0,
            "finishing_deburr": 90.0,
            "inspection": 95.0,
        },
        "process_meta": {
            "milling": {"hr": 6.0, "rate": 100.0},
            "drilling": {"hr": 1.5, "rate": 80.0},
            "finishing_deburr": {"hr": 1.5, "rate": 60.0},
            "inspection": {"hr": 1.0, "rate": 95.0},
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
            "Drilling": 120.0,
            "Finishing/Deburr": 90.0,
            "Inspection": 95.0,
        },
        "labor_cost_details": {
            "Milling": "6.00 hr @ $100.00/hr",
            "Drilling": "1.50 hr @ $80.00/hr",
            "Finishing/Deburr": "1.50 hr @ $60.00/hr",
            "Inspection": "1.00 hr @ $95.00/hr",
        },
        "direct_cost_details": {
            "Consumables": "Shop supplies",
            "Shipping": "Ground service",
        },
        "pass_through": {
            "Consumables": 30.0,
            "Shipping": 45.0,
            "Material": 400.0,
        },
        "red_flags": [],
        "applied_pcts": {
            "OverheadPct": 0.10,
            "GA_Pct": 0.05,
            "ContingencyPct": 0.00,
            "MarginPct": 0.20,
        },
        "rates": {
            "MillingRate": 100.0,
            "DrillingRate": 80.0,
            "DeburrRate": 60.0,
            "InspectionRate": 95.0,
        },
        "params": {},
        "nre_detail": {},
        "nre": {},
        "nre_cost_details": {},
        "labor_cost_details_input": {},
        "pass_meta": {},
        "totals": {
            "labor_cost": 905.0,
            "direct_costs": 75.0,
            "subtotal": 980.0,
            "with_overhead": 1078.0,
            "with_ga": 1131.9,
            "with_contingency": 1131.9,
            "with_expedite": 1131.9,
            "price": 1414.875,
        },
    },
}


def _dummy_quote_payload() -> dict:
    return copy.deepcopy(DUMMY_QUOTE_RESULT)


def test_dummy_quote_material_consistency() -> None:
    payload = _dummy_quote_payload()
    baseline = payload["decision_state"]["baseline"]
    drilling_meta = payload["breakdown"]["drilling_meta"]

    normalized = baseline["normalized_quote_material"]
    drill_material = baseline["drill_params"]["material"]
    rendered_material = drilling_meta["material"]

    assert normalized == drill_material == rendered_material


def test_dummy_quote_has_no_planner_rows() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]

    for mapping in (breakdown["process_costs"], breakdown["process_meta"], breakdown["labor_costs"]):
        for key in mapping:
            assert not str(key).strip().lower().startswith("planner ")


def test_dummy_quote_deburr_rows_are_merged() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]

    process_keys = [key.lower() for key in breakdown["process_costs"].keys()]
    labor_labels = [label.lower() for label in breakdown["labor_costs"].keys()]

    assert process_keys.count("finishing_deburr") == 1
    assert "deburr" not in {key for key in process_keys if key != "finishing_deburr"}
    assert all("deburr" not in label or "finishing/deburr" in label for label in labor_labels)


def test_dummy_quote_hour_buckets_align_with_costs() -> None:
    payload = _dummy_quote_payload()
    breakdown = payload["breakdown"]
    process_costs = breakdown["process_costs"]
    process_meta = breakdown["process_meta"]

    for key, meta in process_meta.items():
        hr = float(meta.get("hr", 0.0) or 0.0)
        if hr <= 0:
            continue
        canon_key = str(key).lower()
        assert canon_key in process_costs, f"Missing cost row for {key!r}"


def test_dummy_quote_pricing_ladder_reconciles() -> None:
    payload = _dummy_quote_payload()
    totals = payload["breakdown"]["totals"]
    labor = float(totals["labor_cost"])
    direct = float(totals["direct_costs"])
    subtotal = float(totals["subtotal"])

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
