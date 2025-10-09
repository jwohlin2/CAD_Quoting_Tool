import pandas as pd
import pytest

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
        12.7 / 25.4,
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
    hours = estimate_drilling_hours([5.0, 5.0], 6.0 / 25.4, "Steel")
    assert hours > 0.0


def test_estimate_drilling_hours_debug_details() -> None:
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

    debug: dict[str, float | int | None] = {}
    hours = estimate_drilling_hours(
        [6.35, 6.35, 12.7],
        12.7 / 25.4,
        "Aluminum",
        hole_groups=[
            {"dia_mm": 6.35, "depth_mm": 12.7, "count": 2},
            {"dia_mm": 12.7, "depth_mm": 12.7, "count": 1},
        ],
        speeds_feeds_table=table,
        machine_params=MachineParams(rapid_ipm=200),
        overhead_params=OverheadParams(
            toolchange_min=0.5,
            approach_retract_in=0.25,
            peck_penalty_min_per_in_depth=0.02,
        ),
        debug=debug,
    )

    assert debug["hole_count"] == 3
    assert debug["thickness_in"] == pytest.approx(12.7 / 25.4, rel=1e-6)
    assert debug["avg_dia_in"] == pytest.approx((0.25 * 2 + 0.5) / 3, rel=1e-6)
    assert debug["sfm"] == pytest.approx(120, rel=1e-6)
    assert debug["ipr"] == pytest.approx((0.004 * 2 + 0.008) / 3, rel=1e-6)
    assert debug["rpm"] == pytest.approx((1833.6 * 2 + 916.8) / 3, rel=1e-6)
    assert debug["ipm"] == pytest.approx(7.3344, rel=1e-6)
    assert debug["min_per_hole"] == pytest.approx(hours * 60 / 3, rel=1e-6)
