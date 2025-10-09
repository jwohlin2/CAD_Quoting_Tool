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


def test_estimate_drilling_hours_uses_deep_drill_for_high_ld() -> None:
    table = pd.DataFrame(
        [
            {
                "operation": "Deep_Drill",
                "material": "Aluminum 6061-T6",
                "material_group": "N1",
                "sfm_start": 80,
                "fz_ipr_0_125in": 0.001,
                "fz_ipr_0_25in": 0.0015,
                "fz_ipr_0_5in": 0.002,
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
        [6.35],
        19.05,
        "Aluminum",
        hole_groups=[{"dia_mm": 6.35, "depth_mm": 19.05, "count": 1}],
        speeds_feeds_table=table,
        machine_params=machine,
        overhead_params=overhead,
    )

    assert hours == pytest.approx(0.016385496, rel=1e-6)
