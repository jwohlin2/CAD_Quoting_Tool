import copy

import appV5


DUMMY_QUOTE_RESULT = {
    "price": 1414.875,
    "qty": 12,
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
        "pricing_source": "legacy",
        "drilling_meta": {"material": "Aluminum MIC6"},
        "material": {
            "material": "Aluminum MIC6",
            "material_name": "Aluminum MIC6",
            "material_cost": 400.0,
            "mass_g": 125000.0,
            "net_mass_g": 112500.0,
            "scrap_pct": 0.1,
        },
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
