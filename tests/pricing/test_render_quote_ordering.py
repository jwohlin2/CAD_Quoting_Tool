import math
from collections.abc import Mapping

import appV5

from cad_quoter.llm import explain_quote


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

    ladder = payload.get("ladder", {})
    labor_total = ladder.get("labor_total")
    direct_total = ladder.get("direct_total")
    ladder_subtotal = ladder.get("subtotal_before_margin")
    assert labor_total is not None
    assert direct_total is not None
    assert ladder_subtotal is not None
    assert math.isclose(ladder_subtotal, labor_total + direct_total, abs_tol=0.01)
    assert math.isclose(materials_direct, direct_total, abs_tol=0.01)

    margin_pct = summary.get("margin_pct")
    final_price = summary.get("final_price")
    assert margin_pct is not None and final_price is not None
    expected_final = round(subtotal_before_margin * (1 + margin_pct), 2)
    assert math.isclose(final_price, expected_final, abs_tol=0.01)

    reported_labor_total = payload.get("labor_total_amount")
    assert reported_labor_total is not None
    assert math.isclose(reported_labor_total, labor_total, abs_tol=0.01)


def test_explain_quote_reports_drilling_minutes_from_removal_card() -> None:
    breakdown = {
        "totals": {"price": 120.0, "qty": 1, "labor_cost": 40.0},
        "material_direct_cost": 30.0,
    }
    render_state = {"extra": {"drill_total_minutes": 30.0}}

    explanation = explain_quote(breakdown, render_state=render_state)

    assert "Drilling time comes from removal-card math (0.50 hr total)." in explanation
    assert "No drilling accounted." not in explanation


def test_explain_quote_reports_no_drilling_when_minutes_absent() -> None:
    breakdown = {"totals": {"price": 75.0, "qty": 2, "labor_cost": 0.0}}

    explanation = explain_quote(breakdown, render_state={"extra": {}})

    assert "No drilling accounted." in explanation
    assert "Drilling time comes from removal-card math" not in explanation


def test_render_quote_includes_quick_whatifs_section() -> None:
    result = {
        "price": 230.0,
        "qty": 1,
        "breakdown": {
            "qty": 1,
            "totals": {
                "labor_cost": 120.0,
                "direct_costs": 80.0,
                "subtotal": 200.0,
                "with_expedite": 200.0,
                "with_margin": 230.0,
            },
            "total_direct_costs": 80.0,
            "nre_detail": {},
            "nre": {"programming_per_part": 30.0},
            "material": {"material_cost": 80.0, "total_material_cost": 80.0},
            "process_costs": {"machining": 90.0},
            "process_meta": {},
            "pass_through": {"Material": 80.0},
            "applied_pcts": {"MarginPct": 0.15},
            "rates": {},
            "params": {},
            "labor_cost_details": {"Programming (amortized)": 30.0},
            "direct_cost_details": {"Material": "$80"},
        },
    }

    rendered = appV5.render_quote(result, currency="$")

    assert "QUICK WHAT-IFS (INTERNAL KNOBS)" in rendered
    assert "A) Margin slider (Qty = 1)" in rendered
    assert "10% margin" in rendered
    assert "B) Qty break (assumes same ops; programming amortized; 15% margin)" in rendered
    assert "2,      $105.00,      $80.00,     $185.00,     $212.75" in rendered

    breakdown = result["breakdown"]
    payload = breakdown["render_payload"]
    quick = payload["quick_whatifs"]

    slider_prices = [entry["final_price"] for entry in quick["margin_slider"]]
    assert slider_prices == [220.0, 230.0, 240.0, 250.0]

    qty_breaks = quick["qty_breaks"]
    assert [entry["label"] for entry in qty_breaks] == ["1", "2", "5", "10"]
    assert math.isclose(qty_breaks[1]["final_price"], 212.75, rel_tol=1e-6)
    assert math.isclose(qty_breaks[2]["labor_per_part"], 96.0, rel_tol=1e-6)
    assert all(math.isclose(entry["expedite_per_part"], 0.0, rel_tol=1e-6) for entry in qty_breaks)
