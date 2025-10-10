from __future__ import annotations

import pandas as pd
import pytest


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


def test_planner_drilling_bucket_uses_estimator(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
        "hole_diams_mm": [6.0, 8.0, 10.0, 6.0],
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "drill_ream_bore"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {"op": "drill_ream_bore", "minutes": 28.0, "machine_cost": 280.0, "labor_cost": 0.0},
                {"op": "cnc_rough_mill", "minutes": 15.0, "machine_cost": 225.0, "labor_cost": 0.0},
            ],
            "totals": {"minutes": 43.0, "machine_cost": 505.0, "labor_cost": 0.0},
        }

    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9, "DrillingRate": 75.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert breakdown["pricing_source"] == "planner"

    drill_meta = breakdown["drilling_meta"]
    estimator_hours = drill_meta.get("estimator_hours_for_planner", 0.0)
    assert estimator_hours > 0

    bucket_view = breakdown["bucket_view"]
    drilling_bucket = bucket_view.get("drilling")
    assert drilling_bucket is not None, "expected drilling bucket from planner override"

    expected_minutes = estimator_hours * 60.0
    assert drilling_bucket["minutes"] == pytest.approx(expected_minutes, abs=0.05)

    process_meta = breakdown["process_meta"]
    drilling_meta_entry = process_meta.get("drilling")
    assert drilling_meta_entry is not None, "expected drilling meta entry"
    assert drilling_meta_entry.get("hr") == pytest.approx(estimator_hours, abs=0.01)
    rate_used = drilling_meta_entry.get("rate", 0.0)
    expected_cost = estimator_hours * rate_used
    assert drilling_bucket["machine_cost"] == pytest.approx(expected_cost, abs=0.05)
    assert drilling_bucket.get("labor_cost", 0.0) == pytest.approx(0.0, abs=1e-6)
    basis = drilling_meta_entry.get("basis") or []
    assert any("planner_drilling_override" in str(item) for item in basis)
