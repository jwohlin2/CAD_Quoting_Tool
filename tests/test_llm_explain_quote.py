from cad_quoter import llm


def test_explain_quote_notes_drilling_from_plan_info() -> None:
    """Ensure drilling minutes from plan info are acknowledged in explanations."""

    breakdown = {
        "totals": {"price": 100, "qty": 1},
        "process_costs": {},
    }
    plan_info = {"bucket_state_extra": {"drill_total_minutes": 18.0}}

    explanation = llm.explain_quote(breakdown, plan_info=plan_info)

    assert "Drilling time comes from removal-card math" in explanation
    assert "No drilling accounted" not in explanation


def test_explain_quote_ignores_non_numeric_drilling_minutes() -> None:
    breakdown = {"totals": {"price": 120}}
    render_state = {"extra": {"removal_drilling_hours": "NaN"}}

    explanation = llm.explain_quote(breakdown, render_state=render_state)

    assert "No drilling accounted" in explanation
    assert "removal-card math" not in explanation


def test_explain_quote_uses_qty_string_for_piece_label() -> None:
    breakdown = {"totals": {"price": 250, "qty": "3"}}

    explanation = llm.explain_quote(breakdown)

    assert "for 3 pieces" in explanation
