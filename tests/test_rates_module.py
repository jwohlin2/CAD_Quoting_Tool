from __future__ import annotations

from cad_quoter.costing_glue import op_cost
from cad_quoter import rates


def test_migrate_flat_to_two_bucket_handles_aliases() -> None:
    flat = {
        "ProgrammingRate": 125.0,
        "CAMRate": 120.0,
        "WireEDMRate": 140.0,
        "SurfaceGrindRate": 115.0,
    }

    migrated = rates.migrate_flat_to_two_bucket(flat)

    assert migrated["labor"]["Programmer"] == 125.0
    assert migrated["machine"]["WireEDM"] == 140.0
    assert migrated["machine"]["Blanchard"] == 115.0


def test_migrate_flat_to_two_bucket_handles_cam_rate_only() -> None:
    flat = {"CAMRate": 110.0}

    migrated = rates.migrate_flat_to_two_bucket(flat)

    assert migrated["labor"]["Programmer"] == 110.0


def test_two_bucket_to_flat_prefers_known_keys() -> None:
    two_bucket = {
        "labor": {"Programmer": 130.0, "Machinist": 90.0},
        "machine": {"WireEDM": 150.0},
    }

    flat = rates.two_bucket_to_flat(two_bucket)

    assert flat["ProgrammingRate"] == 130.0
    assert flat["WireEDMRate"] == 150.0
    assert flat["Machinist"] == 90.0
    assert "CAMRate" not in flat


def test_op_cost_combines_machine_and_labor_minutes() -> None:
    op = {"op": "wire_edm_windows"}
    two_bucket = {
        "labor": {"EDMOperator": 60.0},
        "machine": {"WireEDM": 120.0},
    }

    cost = op_cost(op, two_bucket, minutes=30.0)

    assert cost == 60.0
