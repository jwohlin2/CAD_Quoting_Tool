import math
from collections.abc import Mapping

import appV5


def _render_payload(result: Mapping) -> dict:
    rendered = appV5.render_quote(result, currency="$")
    assert "Quote Summary" in rendered
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
