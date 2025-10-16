from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

@pytest.fixture(autouse=True)
def _disable_speeds_feeds_loader(monkeypatch):
    import appV5

    monkeypatch.setattr(appV5, "_load_speeds_feeds_table_from_path", lambda _path: (None, False))


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
    assert str(baseline["pricing_source"]).lower() == "planner", baseline["pricing_source"]
    assert baseline["process_plan_pricing"]["totals"]["machine_cost"] == 90.0

    breakdown = result["breakdown"]
    plan_pricing = breakdown.get("process_plan_pricing")
    assert plan_pricing is not None, "expected planner pricing to be present"
    assert plan_pricing.get("line_items"), "expected planner pricing line items"
    assert isinstance(plan_pricing["line_items"][0], dict)
    assert str(breakdown["pricing_source"]).lower() == "planner", breakdown["pricing_source"]
    assert breakdown["process_costs"] == {"Machine": 90.0, "Labor": 45.0}
    assert breakdown["process_minutes"] == 45.0
    planner_meta = breakdown["process_meta"]
    assert "planner_machine" in planner_meta
    assert "planner_labor" in planner_meta
    assert plan_pricing["totals"]["minutes"] == 45.0


def test_planner_total_mismatch_records_red_flag(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 25.4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
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
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {
                    "op": "Machine Time",
                    "minutes": 30.0,
                    "machine_cost": 90.0,
                    "labor_cost": 0.0,
                },
                {
                    "op": "Handwork",
                    "minutes": 15.0,
                    "machine_cost": 0.0,
                    "labor_cost": 45.0,
                },
            ],
            "totals": {"minutes": 45.0, "machine_cost": 90.0, "labor_cost": 45.0},
        }

    original_roughly_equal = appV5.roughly_equal

    def fake_roughly_equal(a, b, *, eps=0.01):
        if eps == appV5._PLANNER_BUCKET_ABS_EPSILON:
            return False
        return original_roughly_equal(a, b, eps=eps)

    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)
    monkeypatch.setattr(appV5, "roughly_equal", fake_roughly_equal)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    red_flags = breakdown.get("red_flags") or []
    assert any("Planner totals drifted" in flag for flag in red_flags)

    plan_pricing = breakdown.get("process_plan_pricing") or {}
    planner_totals = plan_pricing.get("totals", {})
    planner_labor_cost = float(planner_totals.get("labor_cost", 0.0) or 0.0)

    process_costs = breakdown["process_costs"]
    assert "Labor" in process_costs

    totals = breakdown["totals"]
    labor_rendered = float(breakdown.get("labor_cost_rendered", 0.0) or 0.0)
    assert totals["labor_cost"] == pytest.approx(labor_rendered)
    assert labor_rendered > 0.0


def test_planner_bucket_minutes_from_line_items_when_bucketize_empty(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "drill_ream_bore"}, {"op": "cnc_rough_mill"}, {"op": "deburr"}]}

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
                {"op": "drill_ream_bore", "minutes": 30.0, "machine_cost": 300.0, "labor_cost": 0.0},
                {"op": "cnc_rough_mill", "minutes": 20.0, "machine_cost": 200.0, "labor_cost": 0.0},
                {"op": "deburr", "minutes": 10.0, "machine_cost": 0.0, "labor_cost": 100.0},
            ],
            "totals": {"minutes": 60.0, "machine_cost": 500.0, "labor_cost": 100.0},
        }

    def fake_bucketize(*_args, **_kwargs):
        return {"buckets": {}, "totals": {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0, "total$": 0.0}}

    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "bucketize", fake_bucketize)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.95},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert str(breakdown["pricing_source"]).lower() == "planner"

    bucket_view = breakdown.get("bucket_view") or {}
    buckets = bucket_view.get("buckets") or {}
    assert buckets, "expected bucket minutes derived from planner line items"
    assert buckets["drilling"]["minutes"] == pytest.approx(30.0, abs=0.01)
    assert buckets["milling"]["minutes"] == pytest.approx(20.0, abs=0.01)
    assert buckets["finishing_deburr"]["minutes"] == pytest.approx(10.0, abs=0.01)

    planner_meta = breakdown["process_meta"]
    assert planner_meta["planner_total"]["minutes"] == pytest.approx(60.0, abs=0.01)
    assert planner_meta["planner_machine"]["minutes"] == pytest.approx(50.0, abs=0.01)
    assert planner_meta["planner_labor"]["minutes"] == pytest.approx(10.0, abs=0.01)


def test_planner_does_not_emit_legacy_buckets_when_line_items_present(monkeypatch):
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
    assert str(breakdown["pricing_source"]).lower() == "planner"

    bucket_view = breakdown["bucket_view"]
    assert "drilling" not in bucket_view
    assert "milling" not in bucket_view

    drill_meta = breakdown.get("drilling_meta") or {}
    assert "estimator_hours_for_planner" not in drill_meta


def test_planner_fallback_when_no_line_items(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
        "hole_diams_mm": [6.0, 8.0],
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": []}

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
        return {"line_items": [], "totals": {}}

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
    assert breakdown["pricing_source"] == "legacy"
    red_flags = breakdown.get("red_flags") or []
    assert any("Planner recognized no operations" in flag for flag in red_flags)

    bucket_view = breakdown.get("bucket_view") or {}
    drilling_bucket = bucket_view.get("drilling")
    assert drilling_bucket is not None, "expected drilling bucket for fallback"

    estimator_hours = drilling_bucket["minutes"] / 60.0
    assert estimator_hours > 0
    expected_minutes = estimator_hours * 60.0
    assert drilling_bucket["minutes"] == pytest.approx(expected_minutes, abs=0.05)

    process_meta = breakdown["process_meta"]
    drilling_meta_entry = process_meta.get("drilling") or {}
    assert drilling_meta_entry.get("hr") == pytest.approx(
        expected_minutes / 60.0, abs=1e-6
    )
    assert drilling_bucket["machine_cost"] == pytest.approx(
        (expected_minutes / 60.0) * 75.0, abs=0.05
    )
    assert drilling_bucket.get("labor_cost", 0.0) == pytest.approx(0.0, abs=1e-6)


def test_planner_zero_totals_logs_flag_and_falls_back(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
        "hole_diams_mm": [8.0] * 4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "drill"}]}

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
            "recognized_line_items": 1,
            "line_items": [
                {"op": "Machine Time", "minutes": 30.0, "machine_cost": 0.0, "labor_cost": 0.0}
            ],
            "totals": {"minutes": 30.0, "machine_cost": 0.0, "labor_cost": 0.0},
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
    assert breakdown["pricing_source"] == "legacy"
    red_flags = breakdown.get("red_flags") or []
    assert any("Planner produced zero machine/labor cost" in flag for flag in red_flags)
    quote_log = breakdown.get("quote_log") or []
    assert any("Planner produced zero machine/labor cost" in entry for entry in quote_log)

    app_meta = result.get("app_meta") or {}
    assert not app_meta.get("used_planner", False)


def test_planner_milling_bucket_backfills_from_estimator_when_planner_falls_back(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(
        [
            {
                "Item": "Roughing Cycle Time (hr)",
                "Example Values / Options": 1.65,
                "Data Type / Input Method": "Number",
            },
            {
                "Item": "Setup Time per Setup (hr)",
                "Example Values / Options": 0.0,
                "Data Type / Input Method": "Number",
            },
        ]
    )

    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 25.4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": []}

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
        return {"line_items": [], "totals": {}}

    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 1.0},
        rates={"MillingRate": 100.0, "DrillingRate": 75.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert breakdown["pricing_source"] == "legacy"
    red_flags = breakdown.get("red_flags") or []
    assert any("Planner recognized no operations" in flag for flag in red_flags)
    bucket_view = breakdown["bucket_view"]
    milling_bucket = bucket_view.get("milling")
    assert milling_bucket is not None, "expected milling bucket backfill"
    assert milling_bucket["minutes"] == pytest.approx(99.0, abs=0.1)
    assert milling_bucket["machine_cost"] == pytest.approx(165.0, abs=0.1)
    assert milling_bucket.get("labor_cost", 0.0) == pytest.approx(0.0, abs=1e-6)

    process_meta = breakdown["process_meta"]
    milling_meta = process_meta.get("milling")
    assert milling_meta is not None, "expected milling meta entry"
    assert milling_meta.get("hr") == pytest.approx(1.65, abs=0.01)
    basis = milling_meta.get("basis") or []
    assert any("planner_milling_backfill" in str(item) for item in basis)


def test_die_plate_163_holes_has_planner_totals(monkeypatch):
    import appV5
    import planner_pricing

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 50.8,
        "hole_diams_mm": [13.8] * 163,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "Deep_Drill"}]}

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
            "recognized_line_items": 2,
            "line_items": [
                {
                    "op": "Planner Machine",
                    "minutes": 420.0,
                    "machine_cost": 2400.0,
                    "labor_cost": 600.0,
                },
                {
                    "op": "Programming (amortized)",
                    "minutes": 0.0,
                    "machine_cost": 0.0,
                    "labor_cost": 320.0,
                },
            ],
            "totals": {"minutes": 420.0, "machine_cost": 2400.0, "labor_cost": 920.0},
        }

    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9, "DrillingRate": 90.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert str(breakdown["pricing_source"]).lower() == "planner"
    process_costs = breakdown["process_costs"]
    assert process_costs["Machine"] > 0
    assert process_costs["Labor"] > 0

    planner_meta = breakdown["process_meta"]
    planner_total = planner_meta.get("planner_total") or {}
    assert planner_total.get("machine_cost", 0.0) > 0
    assert planner_total.get("labor_cost", 0.0) > 0
    assert planner_total.get("amortized_programming", 0.0) > 0

    planner_machine = planner_meta.get("planner_machine") or {}
    assert planner_machine.get("cost", 0.0) > 0

    planner_labor = planner_meta.get("planner_labor") or {}
    assert planner_labor.get("cost", 0.0) > 0
    assert planner_labor.get("amortized_programming", 0.0) > 0
