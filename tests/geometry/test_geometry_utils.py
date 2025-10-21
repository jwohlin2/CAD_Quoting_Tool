from __future__ import annotations

import math

import pandas as pd
import pytest

from cad_quoter.geometry import (
    collect_geo_features_from_df,
    _hole_groups_from_cylinders,
    map_geo_to_double_underscore,
    update_variables_df_with_geo,
)


def test_map_geo_to_double_underscore_transforms_dimensions(sample_geo_metrics: dict) -> None:
    mapped = map_geo_to_double_underscore(sample_geo_metrics)

    assert mapped["GEO__BBox_X_mm"] == 120.0
    assert mapped["GEO__BBox_Y_mm"] == 60.0
    assert mapped["GEO__BBox_Z_mm"] == 25.0
    assert mapped["GEO__MaxDim_mm"] == 120.0
    assert mapped["GEO__MinDim_mm"] == 25.0
    assert mapped["GEO__WEDM_PathLen_mm"] == 95.0
    assert mapped["GEO__Area_to_Volume"] == pytest.approx(42000.0 / 180000.0)


def test_update_variables_df_with_geo_updates_and_inserts(sample_geo_metrics: dict) -> None:
    sample_geo_dataframe = pd.DataFrame(
        [
            {
                "Item": "GEO__BBox_X_mm",
                "Example Values / Options": 0.0,
                "Data Type / Input Method": "number",
            },
            {
                "Item": "Existing",
                "Example Values / Options": 1.0,
                "Data Type / Input Method": "number",
            },
        ]
    )
    mapped = map_geo_to_double_underscore(sample_geo_metrics)
    updated = update_variables_df_with_geo(sample_geo_dataframe.copy(), mapped)

    items = updated["Item"].astype(str)
    mask_bbox = items.str.fullmatch("GEO__BBox_X_mm")
    bbox_series = updated.loc[(mask_bbox, "Example Values / Options")]
    assert bbox_series[0] == 120.0

    features = collect_geo_features_from_df(updated)
    assert features["GEO__WEDM_PathLen_mm"] == pytest.approx(95.0)


def test_collect_geo_features_from_df_round_trips(sample_geo_metrics: dict) -> None:
    mapped = map_geo_to_double_underscore(sample_geo_metrics)
    df = pd.DataFrame(
        {"Item": key, "Example Values / Options": value, "Data Type / Input Method": "number"}
        for key, value in mapped.items()
    )

    extracted = collect_geo_features_from_df(df)

    for key, value in mapped.items():
        assert extracted[key] == pytest.approx(float(value))


def test_hole_groups_do_not_double_count_through_holes() -> None:
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    except Exception:  # pragma: no cover - pythonocc fallback
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir

    block_builder = BRepPrimAPI_MakeBox(40.0, 40.0, 20.0)
    if not hasattr(block_builder, "Shape"):
        pytest.skip("OCC geometry kernel unavailable for hole detection test")
    block = block_builder.Shape()
    axis = gp_Ax2(gp_Pnt(20.0, 20.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    cutter_builder = BRepPrimAPI_MakeCylinder(axis, 5.0, 20.0)
    if not hasattr(cutter_builder, "Shape"):
        pytest.skip("OCC geometry kernel unavailable for hole detection test")
    cutter = cutter_builder.Shape()
    solid_builder = BRepAlgoAPI_Cut(block, cutter)
    if not hasattr(solid_builder, "Shape"):
        pytest.skip("OCC geometry kernel unavailable for hole detection test")
    solid = solid_builder.Shape()

    holes = _hole_groups_from_cylinders(solid)

    matching = [h for h in holes if math.isclose(h["dia_mm"], 10.0, abs_tol=1e-3)]

    assert matching, "Expected to find a 10 mm through hole"
    assert matching[0]["count"] == 1
