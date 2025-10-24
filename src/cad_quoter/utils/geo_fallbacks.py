"""Shared geometry dataframe fallbacks used across the package.

These helpers were previously duplicated in multiple modules.  Centralising
them here keeps the behaviour consistent and makes it easier to extend the
fallback logic without touching several call sites.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = [
    "_coerce_float",
    "_normalise_label",
    "_match_column",
    "_resolve_column_names",
    "_iter_indexed_rows",
    "_ensure_column",
    "_assign_cell",
    "_get_cell",
    "_append_row",
    "_normalise_geo_map",
    "map_geo_to_double_underscore",
    "collect_geo_features_from_df",
    "update_variables_df_with_geo",
    "upsert_var_row",
]


def _coerce_float(value: Any) -> float | None:
    """Convert ``value`` to a finite float or return ``None`` on failure."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _normalise_label(value: Any) -> str:
    """Normalise dataframe column labels to stripped strings."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _match_column(columns: Iterable[str], target: str) -> str | None:
    """Locate a column name that exactly or partially matches ``target``."""

    target_lower = target.lower()
    for column in columns:
        if column.lower() == target_lower:
            return column
    for column in columns:
        if target_lower in column.lower():
            return column
    return None


def _resolve_column_names(df: Any) -> tuple[str, str, str | None]:
    columns = [str(col) for col in getattr(df, "columns", [])]
    item_col = _match_column(columns, "item") or "Item"
    value_col = _match_column(columns, "example values / options") or "Example Values / Options"
    dtype_col = _match_column(columns, "data type / input method") or "Data Type / Input Method"
    return item_col, value_col, dtype_col


def _iter_indexed_rows(df: Any) -> Iterable[tuple[int, Mapping[str, Any]]]:
    if df is None:
        return []
    iterrows = getattr(df, "iterrows", None)
    if callable(iterrows):
        return ((idx, dict(row)) for idx, row in iterrows())
    if isinstance(df, Mapping):
        return [(0, df)]
    if isinstance(df, Iterable):
        return ((idx, dict(row)) for idx, row in enumerate(df))
    return []


def _ensure_column(df: Any, column: str) -> None:
    columns = list(getattr(df, "columns", []))
    if column in columns:
        return
    length = len(df) if hasattr(df, "__len__") else 0
    try:
        df[column] = [None] * length
    except Exception:
        pass


def _assign_cell(df: Any, index: int, column: str, value: Any) -> None:
    if not column:
        return
    if hasattr(df, "_rows"):
        rows = getattr(df, "_rows")
        while len(rows) <= index:
            rows.append({})
        rows[index][column] = value
        columns = list(getattr(df, "columns", []))
        if column not in columns:
            columns.append(column)
            df.columns = columns  # type: ignore[attr-defined]
        return
    if hasattr(df, "loc"):
        df.loc[index, column] = value
    elif hasattr(df, "at"):
        df.at[index, column] = value


def _get_cell(df: Any, index: int, column: str) -> Any:
    if not column:
        return None
    if hasattr(df, "_rows"):
        rows = getattr(df, "_rows")
        if 0 <= index < len(rows):
            return rows[index].get(column)
        return None
    if hasattr(df, "loc"):
        return df.loc[index, column]
    if hasattr(df, "at"):
        return df.at[index, column]
    return None


def _append_row(
    df: Any,
    item_col: str,
    value_col: str,
    dtype_col: str | None,
    label: str,
    value: float,
) -> int:
    new_index = len(df) if hasattr(df, "__len__") else 0
    row = {item_col: label, value_col: value}
    if dtype_col:
        row[dtype_col] = "number"
    if hasattr(df, "_rows"):
        columns = list(getattr(df, "columns", []))
        if item_col not in columns:
            columns.append(item_col)
        if value_col not in columns:
            columns.append(value_col)
        if dtype_col and dtype_col not in columns:
            columns.append(dtype_col)
        rows = getattr(df, "_rows")
        while len(rows) <= new_index:
            rows.append({})
        rows[new_index].update(row)
        df.columns = columns  # type: ignore[attr-defined]
    elif hasattr(df, "loc"):
        df.loc[new_index] = row
    else:
        raise TypeError("Unsupported dataframe type for GEO__ updates")
    return new_index


def _normalise_geo_map(geo: Mapping[str, Any] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if not geo:
        return result
    for key, value in geo.items():
        label = _normalise_label(key)
        if not label:
            continue
        coerced = _coerce_float(value)
        if coerced is None:
            continue
        result[label] = coerced
    return result


def _map_geo_to_double_underscore_fallback(geo: Mapping[str, Any] | None) -> dict[str, float]:
    data: dict[str, Any] = dict(geo or {})

    def _as_float(key: str) -> float | None:
        return _coerce_float(data.get(key))

    mapped: dict[str, float] = {}

    length = _as_float("GEO-01_Length_mm")
    width = _as_float("GEO-02_Width_mm")
    height = _as_float("GEO-03_Height_mm")

    if length is not None:
        mapped["GEO__BBox_X_mm"] = length
    if width is not None:
        mapped["GEO__BBox_Y_mm"] = width
    if height is not None:
        mapped["GEO__BBox_Z_mm"] = height

    dims = [value for value in (length, width, height) if value is not None]
    if dims:
        max_dim = max(dims)
        min_dim = min(dims)
        mapped["GEO__MaxDim_mm"] = max_dim
        mapped["GEO__MinDim_mm"] = min_dim
        mapped["GEO__Stock_Thickness_mm"] = min_dim

    volume = _as_float("GEO-Volume_mm3")
    if volume is None:
        volume = _as_float("GEO_Volume_mm3")
    if volume is not None:
        mapped["GEO__Volume_mm3"] = volume

    surface_area = _as_float("GEO-SurfaceArea_mm2")
    if surface_area is not None:
        mapped["GEO__SurfaceArea_mm2"] = surface_area

    face_count = _as_float("Feature_Face_Count")
    if face_count is None:
        face_count = _as_float("GEO_Face_Count")
    if face_count is not None:
        mapped["GEO__Face_Count"] = face_count

    wedm_path = _as_float("GEO_WEDM_PathLen_mm")
    if wedm_path is not None:
        mapped["GEO__WEDM_PathLen_mm"] = wedm_path

    if surface_area is not None and volume not in (None, 0.0):
        try:
            mapped["GEO__Area_to_Volume"] = surface_area / float(volume)
        except Exception:
            pass

    return mapped


def map_geo_to_double_underscore(
    geo: Mapping[str, Any] | None,
    mapper: Callable[[Mapping[str, Any] | None], Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    if callable(mapper):
        result = mapper(geo)
        if result is None:
            return {}
        return dict(result)
    return _map_geo_to_double_underscore_fallback(geo)


def _collect_geo_features_from_df_fallback(df: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    item_col, value_col, _ = _resolve_column_names(df)
    for _, row in _iter_indexed_rows(df):
        label = _normalise_label(row.get(item_col))
        if not label or not label.startswith("GEO__"):
            continue
        value = _coerce_float(row.get(value_col))
        if value is None:
            continue
        result[label] = value
    return result


def collect_geo_features_from_df(
    df: Any,
    collector: Callable[[Any], Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    if callable(collector):
        result = collector(df)
        if result is None:
            return {}
        return dict(result)
    return _collect_geo_features_from_df_fallback(df)


def _update_variables_df_with_geo_fallback(
    df: Any,
    geo: Mapping[str, Any] | None,
) -> Any:
    if df is None:
        return df

    geo_map = _normalise_geo_map(geo)
    if not geo_map:
        return df

    item_col, value_col, dtype_col = _resolve_column_names(df)
    _ensure_column(df, item_col)
    _ensure_column(df, value_col)
    if dtype_col:
        _ensure_column(df, dtype_col)

    label_to_index: dict[str, int] = {}
    for idx, row in _iter_indexed_rows(df):
        label = _normalise_label(row.get(item_col))
        if label:
            label_to_index[label] = idx

    for label, value in geo_map.items():
        index = label_to_index.get(label)
        if index is not None:
            _assign_cell(df, index, value_col, value)
            if dtype_col:
                existing = _get_cell(df, index, dtype_col)
                if not existing:
                    _assign_cell(df, index, dtype_col, "number")
        else:
            index = _append_row(df, item_col, value_col, dtype_col, label, value)
            label_to_index[label] = index

    return df


def update_variables_df_with_geo(
    df: Any,
    geo: Mapping[str, Any] | None,
    updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> Any:
    if callable(updater):
        return updater(df, geo)
    return _update_variables_df_with_geo_fallback(df, geo)


def upsert_var_row(df, item, value, dtype="number"):
    """Upsert one row by ``Item`` name (case-insensitive)."""

    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    if isinstance(item, pd.Series):
        item = item.iloc[0]
    item = str(item)

    if isinstance(value, pd.Series):
        value = value.iloc[0]
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            pass
        else:
            coerced = _coerce_float(value)
            if coerced is not None:
                value = coerced
    except Exception:
        pass

    cols = list(df.columns)
    row = {c: "" for c in cols}
    if "Item" in row:
        row["Item"] = item
    if "Example Values / Options" in row:
        row["Example Values / Options"] = value
    if "Data Type / Input Method" in row:
        row["Data Type / Input Method"] = dtype

    mask = df["Item"].astype(str).str.casefold() == item.casefold()
    if mask.any():
        df.loc[mask, cols] = [row[c] for c in cols]
        return df

    new_row = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    return pd.concat([df, new_row], ignore_index=True)
