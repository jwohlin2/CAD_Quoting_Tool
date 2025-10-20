from appV5 import render_quote

WARNING_LABEL = "⚠ MATERIALS MISSING"


def _base_payload() -> dict:
    return {
        "summary": {
            "qty": 1,
            "subtotal_before_margin": 120.0,
            "margin_pct": 0.25,
            "final_price": 150.0,
        },
        "materials": [
            {"label": "Aluminum 6061", "detail": "Plate", "amount": 0.0},
        ],
        "materials_direct": 0.0,
        "cost_breakdown": [
            ("Direct Costs", 0.0),
            ("Machine & Labor", 120.0),
        ],
        "processes": [
            {"label": "Machining", "hours": 2.0, "rate": 60.0, "amount": 120.0},
        ],
        "labor_total_amount": 120.0,
    }


def test_render_quote_highlights_missing_materials() -> None:
    payload = _base_payload()
    text = render_quote(payload)
    assert text.count(WARNING_LABEL) == 2
    assert "Materials & Stock" in text
    assert "Cost Breakdown" in text


def test_render_quote_omits_material_warning_when_cost_present() -> None:
    payload = _base_payload()
    payload["materials_direct"] = 260.0
    payload["materials"][0]["amount"] = 260.0
    text = render_quote(payload)
    assert WARNING_LABEL not in text
