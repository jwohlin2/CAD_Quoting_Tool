from __future__ import annotations

import pytest

from cad_quoter.app import planner_support as module


@pytest.fixture(autouse=True)
def _reset_bucketize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure bucketize helpers are predictable for each test."""

    def _fake_bucketize(pricing: dict, *_args, **_kwargs):
        return {"buckets": {}, "order": []}

    def _identity_prepare(raw):
        return {"prepared": raw}

    monkeypatch.setattr(module, "bucketize", _fake_bucketize)
    monkeypatch.setattr(module, "_prepare_bucket_view", _identity_prepare)


def test_apply_planner_result_updates_totals_and_amortized(monkeypatch: pytest.MonkeyPatch) -> None:
    bucket_view = {"buckets": {"milling": {"minutes": 10, "machine$": 20, "labor$": 0}}}

    def _bucketize_override(pricing: dict, *_args, **_kwargs):
        return bucket_view

    monkeypatch.setattr(module, "bucketize", _bucketize_override)

    breakdown: dict[str, object] = {"red_flags": []}
    baseline: dict[str, object] = {}
    process_plan_summary: dict[str, object] = {}
    process_costs: dict[str, float] = {"Machine": 25.0, "Labor": 15.0}
    process_meta: dict[str, object] = {}
    totals_block: dict[str, float] = {}

    planner_result = {
        "totals": {"machine_cost": 100, "labor_cost": 50, "minutes": 120},
        "line_items": [
            {"op": "Milling", "minutes": 120, "machine_cost": 100, "labor_cost": 50},
            {"op": "Programming per part", "labor_cost": 10},
            {"op": "Fixture per part", "labor_cost": 5},
        ],
    }

    result = module.apply_planner_result(
        planner_result,
        breakdown=breakdown,
        baseline=baseline,
        process_plan_summary=process_plan_summary,
        process_costs=process_costs,
        process_meta=process_meta,
        totals_block=totals_block,
        bucketize_rates={"labor": {}, "machine": {}},
        bucketize_nre={},
        geom_for_bucketize={},
        qty_for_bucketize=1,
    )

    assert result.use_planner is True
    assert result.planner_used is True
    assert pytest.approx(result.planner_machine_cost_total) == 100.0
    assert pytest.approx(result.planner_labor_cost_total) == 35.0
    assert pytest.approx(result.amortized_programming) == 10.0
    assert pytest.approx(result.amortized_fixture) == 5.0
    assert result.bucket_view_prepared == {"prepared": bucket_view}

    assert process_costs == {"Machine": 100.0, "Labor": 35.0}
    assert baseline["pricing_source"] == "Planner"
    assert process_plan_summary["used_planner"] is True
    assert process_meta["planner_total"]["amortized_programming"] == 10.0
    assert totals_block["machine_cost"] == 100.0

    milling_metrics = result.aggregated_bucket_minutes["milling"]
    assert pytest.approx(milling_metrics["minutes"]) == 120.0
    assert pytest.approx(milling_metrics["machine$"]) == 165.0
    assert pytest.approx(milling_metrics["labor$"]) == 0.0


def test_apply_planner_result_flags_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    breakdown: dict[str, object] = {"red_flags": []}
    baseline: dict[str, object] = {}
    process_plan_summary: dict[str, object] = {}
    process_costs: dict[str, float] = {"Machine": 200.0, "Labor": 80.0}
    process_meta: dict[str, object] = {}
    totals_block: dict[str, float] = {}

    planner_result = {
        "totals": {"machine_cost": 10, "labor_cost": 20, "minutes": 60},
        "line_items": [{"op": "Milling", "minutes": 60, "machine_cost": 10, "labor_cost": 20}],
    }

    module.apply_planner_result(
        planner_result,
        breakdown=breakdown,
        baseline=baseline,
        process_plan_summary=process_plan_summary,
        process_costs=process_costs,
        process_meta=process_meta,
        totals_block=totals_block,
        bucketize_rates={"labor": {}, "machine": {}},
        bucketize_nre={},
        geom_for_bucketize={},
        qty_for_bucketize=1,
        planner_bucket_abs_epsilon=0.1,
    )

    assert any("drifted" in flag for flag in breakdown["red_flags"])


def test_apply_planner_result_handles_fallback() -> None:
    breakdown: dict[str, object] = {"red_flags": []}
    baseline: dict[str, object] = {}
    process_plan_summary: dict[str, object] = {}
    process_costs: dict[str, float] = {}
    process_meta: dict[str, object] = {}
    totals_block: dict[str, float] = {}

    planner_result = {"line_items": []}

    result = module.apply_planner_result(
        planner_result,
        breakdown=breakdown,
        baseline=baseline,
        process_plan_summary=process_plan_summary,
        process_costs=process_costs,
        process_meta=process_meta,
        totals_block=totals_block,
        bucketize_rates={"labor": {}, "machine": {}},
        bucketize_nre={},
        geom_for_bucketize={},
        qty_for_bucketize=1,
    )

    assert result.use_planner is False
    assert "Planner recognized no operations" in result.fallback_reason
    assert totals_block == {}
    assert breakdown["process_plan_pricing"] == planner_result
