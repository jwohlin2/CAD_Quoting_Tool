from __future__ import annotations

import pandas as pd


def test_compute_quote_uses_planner_minutes(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 25.4,
    }

    plan_called: dict[str, bool] = {"plan": False}
    price_called: dict[str, bool] = {"price": False}

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        plan_called["plan"] = True
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "Roughing"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        price_called["price"] = True
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {"op": "Machine Time", "minutes": 30.0, "machine_cost": 90.0, "labor_cost": 0.0},
                {"op": "Handwork", "minutes": 15.0, "machine_cost": 0.0, "labor_cost": 45.0},
            ],
            "totals": {"minutes": 45.0, "machine_cost": 90.0, "labor_cost": 45.0},
        }

    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9},
        geo=geo,
        ui_vars={},
    )

    assert plan_called["plan"] is True
    assert price_called["price"] is True

    baseline = result["decision_state"]["baseline"]
    assert baseline["pricing_source"] == "planner", baseline["pricing_source"]
    assert baseline["process_plan_pricing"]["totals"]["machine_cost"] == 90.0

    breakdown = result["breakdown"]
    plan_pricing = breakdown.get("process_plan_pricing")
    assert plan_pricing is not None, "expected planner pricing to be present"
    assert plan_pricing.get("line_items"), "expected planner pricing line items"
    assert isinstance(plan_pricing["line_items"][0], dict)
    assert breakdown["pricing_source"] == "planner", breakdown["pricing_source"]
    assert breakdown["process_costs"] == {"Machine": 90.0, "Labor": 45.0}
    assert breakdown["process_minutes"] == 45.0
    planner_meta = breakdown["process_meta"]
    assert "planner_machine" in planner_meta
    assert "planner_labor" in planner_meta
    assert plan_pricing["totals"]["minutes"] == 45.0
