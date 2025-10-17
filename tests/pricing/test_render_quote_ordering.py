import math
from collections.abc import Mapping

import appV5


def _render_payload(result: Mapping) -> dict:
    rendered = appV5.render_quote(result, currency="$")
    assert "QUOTE SUMMARY" in rendered
    breakdown = result.get("breakdown", {}) if isinstance(result, dict) else {}
    payload = breakdown.get("render_payload") if isinstance(breakdown, dict) else None
    if payload is None and isinstance(result, dict):
        payload = result.get("render_payload")
    assert isinstance(payload, dict), "expected render payload attached to result"
    return payload


def test_render_quote_emits_structured_sections() -> None:
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
                "with_expedite": 47.124,
            },
            "nre_detail": {},
            "nre": {},
            "material": {},
            "process_costs": {"machining": 25.0},
            "process_meta": {},
            "pass_through": {"Material": 15.0},
            "applied_pcts": {
                "MarginPct": 0.15,
            },
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    payload = _render_payload(result)

    summary = payload["summary"]
    assert summary["qty"] == 3
    assert math.isclose(summary["margin_pct"], 0.15, rel_tol=1e-6)
    assert math.isclose(summary["final_price"], result["price"], rel_tol=1e-6)

    drivers = payload.get("price_drivers", [])
    assert any("Tight tolerance" in driver.get("detail", "") for driver in drivers)
    assert any("fixture" in driver.get("detail", "").lower() for driver in drivers)

    cost_breakdown = dict(payload.get("cost_breakdown", []))
    assert math.isclose(cost_breakdown["Direct Costs"], 15.0, rel_tol=1e-6)
    assert "Machine & Labor" in cost_breakdown


def test_render_quote_cost_breakdown_prefers_pricing_totals() -> None:
    result = {
        "price": 42.0,
        "breakdown": {
            "qty": 2,
            "totals": {
                "labor_cost": 10.0,
                "direct_costs": 99.0,
                "subtotal": 25.0,
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

    payload = _render_payload(result)
    cost_breakdown = dict(payload.get("cost_breakdown", []))
    assert math.isclose(cost_breakdown["Direct Costs"], 17.5, rel_tol=1e-6)

    materials = payload.get("materials", [])
    assert any(entry.get("label") == "Shipping" for entry in materials)


def test_render_quote_process_payload_tracks_bucket_view() -> None:
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
        },
        "order": ["milling", "drilling", "inspection", "finishing"],
    }

    breakdown = {
        "qty": 1,
        "totals": {
            "labor_cost": 900.0,
            "direct_costs": 175.0,
            "subtotal": 1075.0,
            "with_expedite": 1241.625,
        },
        "nre_detail": {},
        "nre": {},
        "material": {"scrap_pct": 0.12},
        "process_costs": {},
        "process_meta": {},
        "pass_through": {"Material": 150.0, "Shipping": 25.0},
        "applied_pcts": {
            "MarginPct": 0.15,
        },
        "rates": {},
        "params": {},
        "labor_cost_details": {},
        "direct_cost_details": {},
        "pricing_source": "planner",
        "bucket_view": bucket_view,
        "process_plan": {"bucket_view": bucket_view},
        "process_plan_summary": {"bucket_view": bucket_view},
    }

    result = {"price": 1450.0, "breakdown": breakdown}

    payload = _render_payload(result)
    processes = payload.get("processes", [])
    labels = [entry["label"] for entry in processes]
    assert "Milling" in labels
    assert any(label.startswith("Finishing") for label in labels)

    drilling_rows = [entry for entry in processes if entry["label"].lower().startswith("drill")]
    if drilling_rows:
        drilling = drilling_rows[0]
        assert math.isclose(drilling.get("hours", 0.0), 1.5, rel_tol=1e-6)
        assert math.isclose(drilling.get("amount", 0.0), 300.0, rel_tol=1e-6)


def test_render_payload_obeys_pricing_math_guards() -> None:
    bucket_view = {
        "buckets": {
            "milling": {
                "total$": 480.0,
                "machine$": 320.0,
                "labor$": 160.0,
                "minutes": 96.0,
            },
            "finishing": {
                "total$": 240.0,
                "labor$": 240.0,
                "minutes": 60.0,
            },
        },
        "order": ["milling", "finishing"],
    }

    breakdown = {
        "qty": 2,
        "totals": {
            "labor_cost": 720.0,
            "direct_costs": 260.0,
            "subtotal": 980.0,
            "with_expedite": 980.0,
        },
        "nre_detail": {},
        "nre": {},
        "material": {"total_cost": 260.0},
        "process_costs": {},
        "process_meta": {},
        "pass_through": {"Material": 260.0},
        "applied_pcts": {
            "MarginPct": 0.1,
        },
        "rates": {},
        "params": {},
        "labor_cost_details": {},
        "direct_cost_details": {},
        "pricing_source": "planner",
        "bucket_view": bucket_view,
        "process_plan": {"bucket_view": bucket_view},
        "process_plan_summary": {"bucket_view": bucket_view},
    }

    result = {"price": 1078.0, "breakdown": breakdown}

    payload = _render_payload(result)
    summary = payload["summary"]
    subtotal_before_margin = summary.get("subtotal_before_margin")
    assert subtotal_before_margin is not None

    materials_direct = payload.get("materials_direct")
    assert materials_direct is not None

    processes = payload.get("processes", [])
    labor_sum = sum(float(entry.get("amount", 0.0) or 0.0) for entry in processes)
    assert math.isclose(subtotal_before_margin, materials_direct + labor_sum, abs_tol=0.01)

    margin_pct = summary.get("margin_pct")
    final_price = summary.get("final_price")
    assert margin_pct is not None and final_price is not None
    expected_final = round(subtotal_before_margin * (1 + margin_pct), 2)
    assert math.isclose(final_price, expected_final, abs_tol=0.01)

    reported_labor_total = payload.get("labor_total_amount")
    assert reported_labor_total is not None
    assert math.isclose(reported_labor_total, labor_sum, abs_tol=0.01)



def test_render_quote_promotes_planner_pricing_source() -> None:
    result = {
        "price": 0.0,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 0.0,
                "direct_costs": 0.0,
                "subtotal": 0.0,
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

    assert "Pricing Source: Estimator" in lines
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
    assert pricing_lines[0] == "Pricing Source: Estimator"
