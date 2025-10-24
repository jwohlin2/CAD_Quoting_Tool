from __future__ import annotations

import pandas as pd
import pytest

from cad_quoter.utils.geo_fallbacks import (
    _append_row,
    _assign_cell,
    _coerce_float,
    _ensure_column,
    _get_cell,
    _iter_indexed_rows,
    _match_column,
    _normalise_geo_map,
    _normalise_label,
    _resolve_column_names,
    collect_geo_features_from_df,
    map_geo_to_double_underscore,
    update_variables_df_with_geo,
)


def test_coerce_float_filters_non_finite_values() -> None:
    assert _coerce_float("42.5") == pytest.approx(42.5)
    assert _coerce_float("nan") is None
    assert _coerce_float(float("inf")) is None


def test_label_and_column_helpers_normalise_inputs() -> None:
    assert _normalise_label("  GEO__Example  ") == "GEO__Example"
    assert _normalise_label(123) == "123"

    columns = [" Item ", "Example Values / Options", "Other"]
    assert _match_column(columns, "item") == " Item "

    df = pd.DataFrame(columns=["Custom Item", "Value"])
    item_col, value_col, dtype_col = _resolve_column_names(df)
    assert item_col == "Custom Item"
    assert value_col == "Example Values / Options"
    assert dtype_col == "Data Type / Input Method"


def test_iter_indexed_rows_handles_iterables_and_mappings() -> None:
    rows = [{"Item": "A"}, {"Item": "B"}]
    indexed = list(_iter_indexed_rows(rows))
    assert indexed[0] == (0, {"Item": "A"})

    mapping = {"Item": "C"}
    assert list(_iter_indexed_rows(mapping)) == [(0, mapping)]


def test_update_variables_df_with_geo_writes_numeric_values() -> None:
    df = pd.DataFrame(
        [
            {
                "Item": "Existing",
                "Example Values / Options": 1.0,
                "Data Type / Input Method": "number",
            }
        ]
    )
    geo = {" GEO__BBox_X_mm ": "120", "GEO__Ignored": "nan"}

    updated = update_variables_df_with_geo(df.copy(), geo)

    mask = updated["Item"].astype(str).str.fullmatch("GEO__BBox_X_mm")
    assert mask.any()
    bbox_rows = updated.loc[mask]
    bbox_value = bbox_rows["Example Values / Options"].iloc[0]
    assert bbox_value == pytest.approx(120.0)
    dtype_value = bbox_rows["Data Type / Input Method"].iloc[0]
    assert dtype_value == "number"


def test_collect_geo_features_from_df_filters_invalid_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "Item": "GEO__Valid",
                "Example Values / Options": "42",
                "Data Type / Input Method": "number",
            },
            {
                "Item": "GEO__Invalid",
                "Example Values / Options": "foo",
                "Data Type / Input Method": "text",
            },
            {
                "Item": "Other",
                "Example Values / Options": 1,
                "Data Type / Input Method": "number",
            },
        ]
    )

    features = collect_geo_features_from_df(df)

    assert features == {"GEO__Valid": pytest.approx(42.0)}


def test_map_geo_to_double_underscore_fallback_extracts_metrics() -> None:
    geo = {
        "GEO-01_Length_mm": "120",
        "GEO-02_Width_mm": 60,
        "GEO-03_Height_mm": "25",
        "GEO_Volume_mm3": "180000",
        "GEO-SurfaceArea_mm2": "42000",
        "Feature_Face_Count": "6",
        "GEO_WEDM_PathLen_mm": "95",
    }

    mapped = map_geo_to_double_underscore(geo)

    assert mapped["GEO__BBox_X_mm"] == pytest.approx(120.0)
    assert mapped["GEO__BBox_Y_mm"] == pytest.approx(60.0)
    assert mapped["GEO__BBox_Z_mm"] == pytest.approx(25.0)
    assert mapped["GEO__Area_to_Volume"] == pytest.approx(42000.0 / 180000.0)
    assert mapped["GEO__Face_Count"] == pytest.approx(6.0)


def test_geo_row_helpers_work_with_row_container() -> None:
    class StubFrame:
        def __init__(self) -> None:
            self._rows: list[dict[str, object]] = [{}]
            self.columns = ["Item", "Example Values / Options"]

        def __len__(self) -> int:
            return len(self._rows)

    frame = StubFrame()
    _ensure_column(frame, "Item")
    index = _append_row(frame, "Item", "Example Values / Options", None, "GEO__Foo", 3.14)
    _assign_cell(frame, index, "Example Values / Options", 6.28)

    assert index == 1
    assert frame._rows[index]["Item"] == "GEO__Foo"
    assert _get_cell(frame, index, "Example Values / Options") == 6.28

    geo_map = _normalise_geo_map({"GEO__Foo": "6.28", "Bad": "text"})
    assert geo_map == {"GEO__Foo": pytest.approx(6.28)}
