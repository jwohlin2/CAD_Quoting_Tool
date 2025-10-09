from __future__ import annotations

import pytest

from cad_quoter.pricing.time_estimator import (
    MachineParams,
    OperationGeometry,
    OverheadParams,
    ToolParams,
    estimate_time_min,
)


def test_endmill_profile_time_basic() -> None:
    row = {
        "operation": "Endmill_Profile",
        "sfm_start": 400,
        "fz_ipr_0_5in": 0.003,
        "doc_axial_in": 0.25,
        "woc_radial_pct": 50,
    }
    geom = OperationGeometry(diameter_in=0.5, depth_in=0.5, length_in=10.0)
    tool = ToolParams(teeth_z=4)
    machine = MachineParams(rapid_ipm=400, hp_available=10.0, hp_to_mrr_factor=1.0)
    overhead = OverheadParams(toolchange_min=0.5, approach_retract_in=0.1)

    minutes = estimate_time_min(row, geom, tool, machine, overhead, material_factor=1.0)
    assert minutes == pytest.approx(1.0562827225, rel=1e-6)


def test_drill_time_includes_peck_and_breakthrough() -> None:
    row = {
        "operation": "drill",
        "sfm_start": 100,
        "fz_ipr_0_25in": 0.004,
    }
    geom = OperationGeometry(diameter_in=0.25, hole_depth_in=1.0, point_angle_deg=118.0)
    tool = ToolParams(teeth_z=1)
    machine = MachineParams(rapid_ipm=200)
    overhead = OverheadParams(
        toolchange_min=0.5,
        approach_retract_in=0.25,
        peck_penalty_min_per_in_depth=0.03,
    )

    minutes = estimate_time_min(row, geom, tool, machine, overhead)
    assert minutes == pytest.approx(0.8155, rel=1e-4)


def test_endmill_profile_caps_by_machine_hp() -> None:
    row = {
        "operation": "Endmill_Profile",
        "sfm_start": 500,
        "fz_ipr_0_5in": 0.02,
        "doc_axial_in": 0.5,
        "woc_radial_pct": 50,
    }
    geom = OperationGeometry(diameter_in=1.0, depth_in=1.0, length_in=8.0)
    tool = ToolParams(teeth_z=4)
    machine = MachineParams(rapid_ipm=400, hp_available=5.0, hp_to_mrr_factor=1.0)
    overhead = OverheadParams(approach_retract_in=0.2)

    minutes = estimate_time_min(row, geom, tool, machine, overhead, material_factor=1.0)
    assert minutes == pytest.approx(0.84, rel=1e-6)

