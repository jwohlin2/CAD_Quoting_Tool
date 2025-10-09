import pandas as pd
import pytest

import appV5
from appV5 import estimate_drilling_hours
from cad_quoter.pricing.time_estimator import (
    MachineParams,
    OverheadParams,
)


def test_estimate_drilling_hours_uses_speeds_feeds_table() -> None:
    table = pd.DataFrame(
        [
            {
                "operation": "Drill",
                "material": "Aluminum",
                "sfm_start": 120,
                "fz_ipr_0_125in": 0.002,
                "fz_ipr_0_25in": 0.004,
                "fz_ipr_0_5in": 0.008,
            }
        ]
    )

    machine = MachineParams(rapid_ipm=200)
    overhead = OverheadParams(
        toolchange_min=0.5,
        approach_retract_in=0.25,
        peck_penalty_min_per_in_depth=0.02,
    )

    hours = estimate_drilling_hours(
        [6.35, 6.35, 12.7],
        12.7,
        "Aluminum",
        hole_groups=[
            {"dia_mm": 6.35, "depth_mm": 12.7, "count": 2},
            {"dia_mm": 12.7, "depth_mm": 12.7, "count": 1},
        ],
        speeds_feeds_table=table,
        machine_params=machine,
        overhead_params=overhead,
    )

    assert hours == pytest.approx(0.021743675, rel=1e-6)


def test_estimate_drilling_hours_fallback_without_table() -> None:
    hours = estimate_drilling_hours([5.0, 5.0], 6.0, "Steel")
    assert hours > 0.0


def test_apply_drilling_per_hole_bounds_limits_hours() -> None:
    floor_hours = appV5._apply_drilling_per_hole_bounds(0.0001, hole_count_hint=10)
    assert floor_hours == pytest.approx((10 * 0.10) / 60.0)

    capped_hours = appV5._apply_drilling_per_hole_bounds(10.0, hole_count_hint=5)
    assert capped_hours == pytest.approx((5 * 2.0) / 60.0)


def test_estimate_drilling_hours_csv_respects_per_hole_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    table = pd.DataFrame(
        [
            {
                "operation": "Drill",
                "material": "Aluminum",
                "sfm_start": 120,
                "fz_ipr_0_125in": 0.002,
                "fz_ipr_0_25in": 0.004,
                "fz_ipr_0_5in": 0.008,
            }
        ]
    )

    machine = MachineParams(rapid_ipm=200)
    overhead = OverheadParams(
        toolchange_min=0.0,
        approach_retract_in=0.0,
        peck_penalty_min_per_in_depth=0.0,
    )

    monkeypatch.setattr(appV5, "_estimate_time_min", lambda *args, **kwargs: 0.001)

    hours = estimate_drilling_hours(
        [],
        12.7,
        "Aluminum",
        hole_groups=[
            {"dia_mm": 6.35, "depth_mm": 12.7, "count": 2},
            {"dia_mm": 12.7, "depth_mm": 12.7, "count": 1},
        ],
        speeds_feeds_table=table,
        machine_params=machine,
        overhead_params=overhead,
    )

    expected_floor = (3 * 0.10) / 60.0
    assert hours == pytest.approx(expected_floor)


def test_estimate_drilling_hours_fallback_passes_hole_count(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, int | None] = {}
    original = appV5._apply_drilling_per_hole_bounds

    def _wrapper(hours: float, *, hole_count_hint: int | None = None) -> float:
        recorded["hint"] = hole_count_hint
        return original(hours, hole_count_hint=hole_count_hint)

    monkeypatch.setattr(appV5, "_apply_drilling_per_hole_bounds", _wrapper)

    estimate_drilling_hours([5.0, 5.0], 6.0, "Steel")

    assert recorded.get("hint") == 2


def test_estimate_drilling_hours_group_counts_used_for_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, int | None] = {}
    original = appV5._apply_drilling_per_hole_bounds

    def _wrapper(hours: float, *, hole_count_hint: int | None = None) -> float:
        recorded["hint"] = hole_count_hint
        return original(hours, hole_count_hint=hole_count_hint)

    monkeypatch.setattr(appV5, "_apply_drilling_per_hole_bounds", _wrapper)

    estimate_drilling_hours(
        [],
        12.7,
        "Aluminum",
        hole_groups=[
            {"dia_mm": 6.35, "depth_mm": 12.7, "count": 2},
            {"dia_mm": 12.7, "depth_mm": 12.7, "count": 1},
        ],
    )

    assert recorded.get("hint") == 3
