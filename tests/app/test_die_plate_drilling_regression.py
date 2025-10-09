from __future__ import annotations

import re

import pandas as pd


def test_steel_plate_drilling_regression(monkeypatch, tmp_path) -> None:
    import appV5
    import planner_pricing

    csv_path = tmp_path / "speeds_feeds.csv"
    csv_path.write_text("Operation,Material\nMill,Aluminum\n", encoding="utf-8")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        return {"ops": [{"op": "Drilling"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        return {
            "line_items": [
                {"op": "Drilling", "minutes": 150.0, "machine_cost": 300.0, "labor_cost": 0.0},
                {"op": "Deburr", "minutes": 45.0, "machine_cost": 0.0, "labor_cost": 90.0},
            ],
            "totals": {"minutes": 195.0, "machine_cost": 300.0, "labor_cost": 90.0},
        }

    class _FakeSpeedsFeedsTable:
        columns = ["operation", "material"]

        def __init__(self) -> None:
            self._records: list[dict[str, object]] = []

        @property
        def empty(self) -> bool:
            return False

        def to_dict(self, orient: str) -> list[dict[str, object]]:
            if orient != "records":
                raise ValueError("unsupported orient")
            return list(self._records)

        def __len__(self) -> int:
            return len(self._records)

    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "_load_speeds_feeds_table", lambda path: _FakeSpeedsFeedsTable())
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    geo = {
        "material": "steel",
        "thickness_mm": 50.8,
        "hole_diams_mm": [13.815] * 163,
        "GEO_Hole_Groups": [
            {"dia_mm": 13.815, "depth_mm": 50.8, "count": 163, "through": True},
        ],
        "process_planner_family": "die_plate",
    }

    ui_vars = {"Material": "Steel", "Thickness (in)": 2.0, "Hole Count (override)": 163}

    params = {
        "PlannerMode": "auto",
        "SpeedsFeedsCSVPath": str(csv_path),
        "OEE_EfficiencyPct": 0.9,
    }

    result = appV5.compute_quote_from_df(
        df,
        params=params,
        geo=geo,
        ui_vars=ui_vars,
        llm_enabled=False,
    )

    baseline = result["decision_state"]["baseline"]
    breakdown = result["breakdown"]

    assert baseline["pricing_source"] == "planner"
    assert breakdown["pricing_source"] == "planner"

    drill_hours = baseline["process_hours"].get("drilling")
    assert drill_hours is not None
    assert 2.0 <= float(drill_hours) <= 4.0

    narrative = result["narrative"]
    for value in re.findall(r"[↑↓](\d+\.\d+)", narrative):
        assert float(value) <= 24.0 + 1e-6
