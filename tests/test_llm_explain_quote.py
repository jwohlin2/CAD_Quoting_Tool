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
