import math
from collections.abc import Mapping

import appV5

from cad_quoter.llm import explain_quote


def _render_payload(result: Mapping) -> tuple[str, dict]:
    rendered = appV5.render_quote(result, currency="$")
    assert "QUOTE SUMMARY" in rendered
    breakdown = result.get("breakdown", {}) if isinstance(result, dict) else {}
    payload = breakdown.get("render_payload") if isinstance(breakdown, dict) else None
    if payload is None and isinstance(result, dict):
        payload = result.get("render_payload")
    assert isinstance(payload, dict), "expected render payload attached to result"
    return rendered, payload


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

    rendered_text, payload = _render_payload(result)

    summary = payload["summary"]
    assert summary["qty"] == 3
    assert math.isclose(summary["margin_pct"], 0.15, rel_tol=1e-6)
    assert math.isclose(summary["final_price"], result["price"], rel_tol=1e-6)

    assert "QUICK WHAT-IFS (INTERNAL KNOBS)" in rendered_text
    assert "Margin slider (Qty =" in rendered_text
    assert "Qty break (assumes same ops" in rendered_text

    quick_entries = payload.get("quick_what_ifs")
    assert isinstance(quick_entries, list) and quick_entries

    slider_payload = payload.get("margin_slider")
    assert isinstance(slider_payload, Mapping)
    assert slider_payload.get("points")
    assert math.isclose(
        float(slider_payload.get("current_pct", 0.0)),
        summary["margin_pct"],
        rel_tol=1e-6,
    )

    qty_breaks = payload.get("qty_breaks")
    assert isinstance(qty_breaks, list) and qty_breaks
    first_break = qty_breaks[0]
    assert {"qty", "labor_per_part", "final_price"}.issubset(first_break.keys())

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

    _, payload = _render_payload(result)
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

    _, payload = _render_payload(result)
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

    _, payload = _render_payload(result)
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
        "bucket_view": {"buckets": {"drilling": {"total$": 90.0}}},
    }
    render_state = {"extra": {"drill_total_minutes": 30.0}}

    explanation = explain_quote(breakdown, render_state=render_state)

    assert "Main cost drivers: Drilling $90.00." in explanation
    assert "Main cost drivers derive from planner buckets; none dominate." not in explanation


def test_explain_quote_skips_legacy_drilling_text_when_bucket_present() -> None:
    breakdown = {
        "totals": {"price": 180.0, "qty": 1, "labor_cost": 60.0},
        "material_direct_cost": 45.0,
        "bucket_view": {"buckets": {"drilling": {"total$": 180.0}}},
    }
    render_state = {"extra": {"drill_total_minutes": 30.0}}
    plan_info = {"bucket_view": {"buckets": {"drilling": {"total$": 180.0}}}}

    explanation = explain_quote(
        breakdown,
        render_state=render_state,
        plan_info=plan_info,
    )

    assert "Main cost drivers: Drilling $180.00." in explanation


def test_explain_quote_reports_no_drilling_when_minutes_absent() -> None:
    breakdown = {"totals": {"price": 75.0, "qty": 2, "labor_cost": 0.0}}

    explanation = explain_quote(breakdown, render_state={"extra": {}})

    assert "Main cost drivers derive from bucket totals; none dominate." in explanation
