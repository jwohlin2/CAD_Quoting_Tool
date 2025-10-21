from cad_quoter.pricing.planner import price_with_planner


def test_price_with_planner_uses_geometry_minutes() -> None:
    rates = {
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

    result = price_with_planner("die_plate", params, geom, rates, oee=0.9)

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
    rates = {
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

    result = price_with_planner("die_plate", params, geom, rates, oee=0.9)

    totals = result["totals"]
    assert totals["minutes"] > 0.0
    assert totals["machine_cost"] > 0.0
    drilling = [item for item in result["line_items"] if item["op"] == "Drilling"]
    assert drilling, "expected drilling minutes from derived geometry"
    assert drilling[0]["minutes"] > 0.0
    assert drilling[0]["machine_cost"] > 0.0
    assert drilling[0]["labor_cost"] == 0.0
