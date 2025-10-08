from planner_pricing import price_with_planner


def test_price_with_planner_uses_geometry_minutes() -> None:
    rates = {
        "machine": {
            "WireEDM": 120.0,
            "CNC_Mill": 95.0,
            "SurfaceGrind": 85.0,
        },
        "labor": {
            "Machinist": 60.0,
            "Finisher": 45.0,
            "Assembler": 40.0,
            "Inspector": 55.0,
            "Grinder": 58.0,
            "Engineer": 70.0,
        },
    }

    params = {
        "material": "tool_steel_annealed",
        "overall_length": 3.0,
        "min_feature_width": 0.5,
        "min_inside_radius": 0.05,
        "profile_tol": 0.0005,
        "blind_relief": False,
        "edge_condition": "sharp",
    }

    geom = {
        "wedm": {
            "perimeter_in": 12.0,
            "starts": 1,
            "tabs": 0,
            "passes": 2,
            "wire_in": 0.010,
        },
        "milling": {"volume_cuin": 8.0},
        "sg": {"area_sq_in": 18.0, "stock_in": 0.002},
        "drill": [{"dia_in": 0.25, "depth_in": 0.75}],
        "length_ft_edges": 3.0,
        "lap_area_sq_in": 1.5,
    }

    result = price_with_planner("punch", params, geom, rates, oee=1.0)

    assert "ops" in result["plan"]
    assert result["totals"]["minutes"] > 0.0
    assert result["totals"]["machine_cost"] > 0.0

    wire_items = [item for item in result["line_items"] if item["op"] == "wire_edm_outline"]
    assert wire_items, "expected WEDM operation in planner output"
    assert wire_items[0]["minutes"] > 1.0
