import math
import re
from collections.abc import Mapping

import appV5

from cad_quoter.llm import explain_quote
from cad_quoter.utils.render_utils import QuoteDoc, QuoteDocRecorder


def _quote_doc_from_text(text: str, divider: str = "-" * 74) -> QuoteDoc:
    recorder = QuoteDocRecorder(divider)
    previous = None
    for index, line in enumerate(text.splitlines()):
        recorder.observe_line(index, line, previous)
        previous = line
    return recorder.build_doc()


def _render_text_and_doc(result: Mapping) -> tuple[str, QuoteDoc]:
    rendered = appV5.render_quote(result, currency="$")
    return rendered, _quote_doc_from_text(rendered)


def _quote_doc_sections(doc: QuoteDoc) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    for section in doc.sections:
        title = section.title or ""
        sections[title] = [row.text for row in section.rows]
    return sections


_MONEY_RE = re.compile(r"\$\s*([0-9,]+\.[0-9]{2})")


def _parse_money_lines(lines: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in lines:
        match = _MONEY_RE.search(line)
        if not match:
            continue
        amount = float(match.group(1).replace(",", ""))
        label = line[: match.start()].strip()
        if label.endswith(":"):
            label = label[:-1].strip()
        if label.startswith("="):
            label = label.lstrip("= ").strip()
        if label:
            result[label] = amount
    return result


def _section_lines(sections: Mapping[str, list[str]], prefix: str) -> list[str]:
    for title, rows in sections.items():
        if title.startswith(prefix):
            return rows
    return []


def _summary_section(sections: Mapping[str, list[str]]) -> tuple[str, list[str]]:
    for title, rows in sections.items():
        if title.startswith("QUOTE SUMMARY - Qty"):
            return title, rows
    raise AssertionError("expected QUOTE SUMMARY section")


def test_render_quote_emits_structured_sections() -> None:
    result = {
        "price": 46.0,
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

    rendered, doc = _render_text_and_doc(result)
    assert "QUOTE SUMMARY" in rendered

    sections = _quote_doc_sections(doc)
    summary_title, summary_lines = _summary_section(sections)

    qty_match = re.search(r"Qty\s+([0-9.]+)", summary_title)
    assert qty_match is not None
    qty_value = float(qty_match.group(1))
    assert math.isclose(qty_value, 3.0, rel_tol=1e-6)

    summary_amounts = _parse_money_lines(summary_lines)
    final_price = summary_amounts.get("Final Price per Part")
    assert final_price is not None
    assert math.isclose(final_price, result["price"], rel_tol=1e-6)

    why_lines = [
        line.strip() for line in _section_lines(sections, "Why this price") if line.strip()
    ]
    assert any("Tight tolerance" in line for line in why_lines)

    pass_through_lines = _section_lines(sections, "Pass-Through & Direct Costs")
    pass_through_amounts = _parse_money_lines(pass_through_lines)
    assert math.isclose(pass_through_amounts.get("Total", 0.0), 15.0, rel_tol=1e-6)


def test_render_quote_does_not_attach_direct_costs_struct() -> None:
    result: dict[str, object] = {
        "price": 41.0,
        "breakdown": {
            "qty": 2,
            "totals": {
                "labor_cost": 12.0,
                "direct_costs": 8.0,
                "subtotal": 20.0,
            },
            "nre_detail": {},
            "nre": {},
            "material": {"material_cost": 5.0},
            "process_costs": {"milling": 12.0},
            "process_meta": {},
            "pass_through": {"Material": 5.0, "Vendor": 3.0},
            "applied_pcts": {},
            "rates": {},
            "params": {},
            "labor_cost_details": {},
            "direct_cost_details": {},
        },
    }

    _render_text_and_doc(result)

    assert "direct_costs_struct" not in result

    breakdown = result.get("breakdown")
    assert isinstance(breakdown, dict)
    assert "direct_costs_struct" not in breakdown

    pricing = breakdown.get("pricing")
    if isinstance(pricing, dict):
        assert "direct_costs_struct" not in pricing

    totals = breakdown.get("totals")
    if isinstance(totals, dict):
        assert "direct_costs_struct" not in totals


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

    rendered, doc = _render_text_and_doc(result)
    assert "QUOTE SUMMARY" in rendered

    sections = _quote_doc_sections(doc)
    pass_through_lines = _section_lines(sections, "Pass-Through & Direct Costs")
    pass_through_amounts = _parse_money_lines(pass_through_lines)
    assert math.isclose(pass_through_amounts.get("Total", 0.0), 17.5, rel_tol=1e-6)
    assert any("Shipping" in line for line in pass_through_lines)


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
            "labor_cost": 950.0,
            "direct_costs": 25.0,
            "subtotal": 975.0,
            "with_expedite": 975.0,
        },
        "nre_detail": {},
        "nre": {},
        "material": {"scrap_pct": 0.12},
        "process_costs": {},
        "process_meta": {},
        "pass_through": {"Shipping": 25.0},
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

    result = {"price": 1121.25, "breakdown": breakdown}

    rendered, doc = _render_text_and_doc(result)
    assert "QUOTE SUMMARY" in rendered

    sections = _quote_doc_sections(doc)
    process_lines = _section_lines(sections, "Process & Labor Costs")
    process_amounts = _parse_money_lines(process_lines)
    labels = list(process_amounts.keys())
    assert any(label.startswith("Milling") for label in labels)
    assert any(label.lower().startswith("finishing") for label in labels)

    drilling_amount = None
    for label, amount in process_amounts.items():
        if label.lower().startswith("drilling"):
            drilling_amount = amount
            break
    if drilling_amount is not None:
        assert math.isclose(drilling_amount, 300.0, rel_tol=1e-6)


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

    rendered, doc = _render_text_and_doc(result)
    assert "QUOTE SUMMARY" in rendered

    sections = _quote_doc_sections(doc)
    _, summary_lines = _summary_section(sections)
    summary_amounts = _parse_money_lines(summary_lines)
    final_price_per_part = summary_amounts.get("Final Price per Part")
    assert final_price_per_part is not None
    subtotal_before_margin = (
        summary_amounts.get("Total Labor Cost", 0.0)
        + summary_amounts.get("Total Direct Costs", 0.0)
    )

    cost_breakdown = _parse_money_lines(_section_lines(sections, "Pass-Through & Direct Costs"))
    direct_costs_reported = cost_breakdown.get("Total")
    assert direct_costs_reported is not None

    process_amounts = _parse_money_lines(_section_lines(sections, "Process & Labor Costs"))
    labor_sum = sum(
        amount for label, amount in process_amounts.items() if label.lower() != "total"
    )
    assert math.isclose(subtotal_before_margin, direct_costs_reported + labor_sum, abs_tol=0.01)

    totals_declared = breakdown.get("totals", {})
    subtotal_from_totals = float(totals_declared.get("with_expedite", 0.0))
    direct_total_declared = float(totals_declared.get("direct_costs", 0.0))
    labor_total_declared = float(totals_declared.get("labor_cost", 0.0))
    margin_pct = float(breakdown.get("applied_pcts", {}).get("MarginPct", 0.0))
    final_price_per_part = summary_amounts.get("Final Price per Part", 0.0)

    assert math.isclose(subtotal_from_totals, subtotal_before_margin, abs_tol=0.01)
    assert math.isclose(direct_total_declared, direct_costs_reported, abs_tol=0.01)
    assert math.isclose(labor_total_declared, labor_sum, abs_tol=0.01)

    expected_final = round(subtotal_before_margin * (1 + margin_pct), 2)
    assert math.isclose(final_price_per_part, expected_final, abs_tol=0.01)
    assert math.isclose(float(result.get("price", 0.0)), expected_final, abs_tol=0.01)


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

    assert "Main cost drivers derive from planner buckets; none dominate." in explanation
