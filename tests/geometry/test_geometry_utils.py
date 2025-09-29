from __future__ import annotations

import pandas as pd
import pytest

from cad_quoter.geometry import (
    collect_geo_features_from_df,
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


def test_update_variables_df_with_geo_updates_and_inserts(
    sample_geo_metrics: dict, sample_geo_dataframe: pd.DataFrame
) -> None:
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
