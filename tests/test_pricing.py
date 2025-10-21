"""Aggregated pricing and sourcing tests."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

_requests_stub = sys.modules.setdefault("requests", types.ModuleType("requests"))
_requests_stub.Session = getattr(
    _requests_stub,
    "Session",
    type("Session", (), {"__init__": lambda self, *args, **kwargs: None}),
)

from cad_quoter import rates  # noqa: E402  # pylint: disable=wrong-import-position
from cad_quoter.costing_glue import op_cost  # noqa: E402  # pylint: disable=wrong-import-position
import cad_quoter.pricing.materials as materials  # noqa: E402  # pylint: disable=wrong-import-position
from cad_quoter.pricing.materials import pick_stock_from_mcmaster  # noqa: E402  # pylint: disable=wrong-import-position
from planner_pricing import price_with_planner  # noqa: E402  # pylint: disable=wrong-import-position


@pytest.mark.parametrize(
    "flat, expected",
    [
        (
            {
                "ProgrammingRate": 125.0,
                "CAMRate": 120.0,
                "WireEDMRate": 140.0,
                "SurfaceGrindRate": 115.0,
            },
            {
                "labor": {"Programmer": 125.0},
                "machine": {"WireEDM": 140.0, "Blanchard": 115.0},
            },
        ),
        (
            {"CAMRate": 110.0},
            {"labor": {"Programmer": 110.0}},
        ),
        (
            {"ShopRate": 82.5},
            {
                "labor": {"ProgrammingRate": pytest.approx(90.0)},
                "machine": {
                    "MillingRate": pytest.approx(90.0),
                    "WireEDMRate": pytest.approx(130.0),
                },
            },
        ),
        (
            {},
            {
                "labor": {
                    "Programmer": pytest.approx(90.0),
                    "InspectionRate": pytest.approx(85.0),
                },
                "machine": {
                    "MillingRate": pytest.approx(90.0),
                    "DrillingRate": pytest.approx(95.0),
                    "WireEDMRate": pytest.approx(130.0),
                },
            },
        ),
        (
            {"FinishingRate": 47.0},
            {
                "labor": {
                    "Finisher": pytest.approx(47.0),
                    "DeburrRate": pytest.approx(47.0),
                },
            },
        ),
    ],
)
def test_migrate_flat_to_two_bucket_handles_common_inputs(flat: dict[str, float], expected: dict[str, object]) -> None:
    migrated = rates.migrate_flat_to_two_bucket(flat)

    for bucket, expectations in expected.items():
        for key, value in expectations.items():
            assert migrated[bucket][key] == value


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


def test_ensure_two_bucket_defaults_preserves_and_backfills() -> None:
    two_bucket = {
        "labor": {"Toolmaker": 70.0},
        "machine": {"CNC_Mill": 105.0},
    }

    ensured = rates.ensure_two_bucket_defaults(two_bucket)

    assert ensured["labor"]["Programmer"] == pytest.approx(90.0)
    assert ensured["labor"]["Toolmaker"] == pytest.approx(70.0)
    assert ensured["machine"]["CNC_Mill"] == pytest.approx(105.0)
    assert ensured["machine"]["DrillingRate"] == pytest.approx(95.0)


@pytest.mark.parametrize(
    "length,width,thickness,expected_part",
    [
        (12.0, 24.0, 3.5, "86825K626"),
    ],
)
def test_pick_stock_prefers_smallest_plate_when_scrap_blocks_exact(
    length: float, width: float, thickness: float, expected_part: str
) -> None:
    cfg = SimpleNamespace(enforce_exact_thickness=True, allow_thickness_upsize=False)
    result = pick_stock_from_mcmaster("Aluminum MIC6", length, width, thickness, cfg=cfg)
    assert result is not None
    assert pytest.approx(24.0) == result["len_in"]
    assert pytest.approx(12.0) == result["wid_in"]
    assert pytest.approx(thickness) == result["thk_in"]
    assert result.get("mcmaster_part") == expected_part
    assert pytest.approx(24.0) == result["required_blank_len_in"]
    assert pytest.approx(12.0) == result["required_blank_wid_in"]


def test_pick_stock_rejects_api_thickness_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    if materials._mc is None:  # pragma: no cover - optional dependency missing
        pytest.skip("McMaster helpers not available")

    cfg = SimpleNamespace(enforce_exact_thickness=True, allow_thickness_upsize=False)
    need_thickness = 2.0

    def fake_lookup(material: str, length_mm: float, width_mm: float, thickness_mm: float, qty: int = 1):
        length_in = length_mm / 25.4
        width_in = width_mm / 25.4
        return (
            "WRONGSKU",
            99.0,
            "Each",
            (length_in, width_in, need_thickness + 1.5),
        )

    monkeypatch.setattr(materials._mc, "lookup_sku_and_price_for_mm", fake_lookup)

    result = pick_stock_from_mcmaster(
        "Aluminum MIC6",
        12.0,
        12.0,
        need_thickness,
        cfg=cfg,
    )

    assert result is not None
    assert pytest.approx(need_thickness) == result["thk_in"]
    assert result.get("stock_piece_api_price") is None
    assert result.get("stock_piece_price_usd") is None
    assert result.get("stock_piece_source") is None


def test_price_with_planner_uses_geometry_minutes() -> None:
    rates_config = {
        "machine": {
            "WireEDMRate": 120.0,
            "MillingRate": 95.0,
            "DrillingRate": 80.0,
            "SurfaceGrindRate": 85.0,
        },
        "labor": {
            "InspectionRate": 55.0,
            "FixtureBuildRate": 60.0,
            "ProgrammingRate": 70.0,
            "DeburrRate": 42.0,
        },
    }

    params = {
        "material": "tool_steel_annealed",
        "profile_tol": 0.0005,
        "flatness_spec": 0.0008,
        "parallelism_spec": 0.0012,
    }

    geom = {
        "hole_count": 16,
        "tap_qty": 4,
        "cbore_qty": 2,
        "slot_count": 3,
        "edge_len_in": 24.0,
        "pocket_area_total_in2": 12.0,
        "plate_area_in2": 60.0,
        "thickness_in": 1.5,
        "setups": 3,
    }

    result = price_with_planner("die_plate", params, geom, rates_config, oee=0.9)

    assert "ops" in result["plan"]
    assert "ops_seen" in result["assumptions"]
    assert "wire_edm_outline" in result["assumptions"]["ops_seen"]
    assert result["totals"]["minutes"] > 0.0
    assert result["totals"]["machine_cost"] > 0.0

    wire_items = [item for item in result["line_items"] if item["op"] == "Wire EDM"]
    assert wire_items, "expected Wire EDM bucket in planner pricing"
    assert wire_items[0]["minutes"] > 1.0
    assert "machine_cost" in wire_items[0]
    assert "labor_cost" in wire_items[0]
    assert wire_items[0]["labor_cost"] == 0.0


def test_price_with_planner_reads_geom_fallbacks() -> None:
    rates_config = {
        "machine": {
            "MillingRate": 95.0,
            "DrillingRate": 80.0,
        },
        "labor": {
            "InspectionRate": 55.0,
            "DeburrRate": 40.0,
        },
    }

    params = {"material": "aluminum"}

    geom = {
        "derived": {
            "hole_count_geom": 24,
            "edge_length_mm": 760.0,
            "plate_bbox_area_mm2": 25000.0,
            "thickness_mm": 25.4,
        },
        "feature_counts": {"tap_qty": 6, "cbore_qty": 2},
        "hole_diams_mm": [6.0] * 24,
    }

    result = price_with_planner("die_plate", params, geom, rates_config, oee=0.9)

    totals = result["totals"]
    assert totals["minutes"] > 0.0
    assert totals["machine_cost"] > 0.0
    drilling = [item for item in result["line_items"] if item["op"] == "Drilling"]
    assert drilling, "expected drilling minutes from derived geometry"
    assert drilling[0]["minutes"] > 0.0
    assert drilling[0]["machine_cost"] > 0.0
    assert drilling[0]["labor_cost"] == 0.0
