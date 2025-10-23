"""Lightweight geometry fallbacks used when optional CAD dependencies are absent."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping

EZDXF_VERSION = "unknown"
HAS_EZDXF = False
HAS_ODAFC = False


class GeometryService:
    """Placeholder service exposing no-op geometry helpers."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple default
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop


def _return_none(*_args: Any, **_kwargs: Any) -> None:
    return None


def _return_empty_list(*_args: Any, **_kwargs: Any) -> list[Any]:
    return []


def _return_empty_dict(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {}


def convert_dwg_to_dxf(path: str, *, out_ver: str = "ACAD2018") -> str:
    """Return the original path when DWG conversion is unavailable."""

    return path


def get_dwg_converter_path() -> str | None:
    return ""


def require_ezdxf() -> None:
    raise RuntimeError("DXF operations require ezdxf, which is not installed")


def extract_text_lines_from_dxf(_path: str, *, include_tables: bool = False) -> list[str]:
    return []


def harvest_text_lines(_doc: Any, *, include_tables: bool = False) -> list[str]:
    return []


HOLE_TOKENS = None


def parse_hole_table_lines(_lines: Iterable[str], **_kwargs: Any) -> list[Mapping[str, Any]]:
    return []


def extract_features_with_occ(_path: str | Any, **_kwargs: Any) -> Mapping[str, Any] | None:
    return None


def enrich_geo_stl(_path: str | Any) -> Mapping[str, Any] | None:
    return None


def read_step_shape(_path: str | Any) -> Any:
    return None


def read_step_or_iges_or_brep(_path: str | Any) -> Any:
    return None


def read_cad_any(_path: str | Any) -> Any:
    return None


def safe_bbox(_shape: Any) -> tuple[float, float, float, float, float, float]:
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def enrich_geo_occ(_shape: Any) -> Mapping[str, Any]:
    return {}


def get_import_diagnostics_text() -> str:
    return ""


def _hole_groups_from_cylinders(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    return []


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalise_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _match_column(columns: Iterable[str], target: str) -> str | None:
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


def _iter_indexed_rows(df: Any) -> Iterator[tuple[int, Mapping[str, Any]]]:
    if df is None:
        return iter(())
    iterrows = getattr(df, "iterrows", None)
    if callable(iterrows):
        return ((idx, dict(row)) for idx, row in iterrows())
    if isinstance(df, Mapping):
        return iter(((0, df),))
    if isinstance(df, Iterable):
        return ((idx, dict(row)) for idx, row in enumerate(df))
    return iter(())


def _ensure_column(df: Any, column: str) -> None:
    if not column:
        return
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


def _append_row(
    df: Any,
    item_col: str,
    value_col: str,
    dtype_col: str | None,
    label: str,
    value: Any,
    dtype: str,
) -> int:
    new_index = len(df) if hasattr(df, "__len__") else 0
    row = {item_col: label, value_col: value}
    if dtype_col:
        row[dtype_col] = dtype
    if hasattr(df, "_rows"):
        rows = getattr(df, "_rows")
        columns = list(getattr(df, "columns", []))
        if item_col not in columns:
            columns.append(item_col)
        if value_col not in columns:
            columns.append(value_col)
        if dtype_col and dtype_col not in columns:
            columns.append(dtype_col)
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


def map_geo_to_double_underscore(geo: Mapping[str, Any] | None) -> dict[str, float]:
    """Return a subset of GEO metrics using the double-underscore naming scheme."""

    data = dict(geo or {})

    def _geo_value(*keys: str) -> float | None:
        for key in keys:
            value = _coerce_float(data.get(key))
            if value is not None:
                return value
        return None

    mapped: dict[str, float] = {}

    length = _geo_value("GEO-01_Length_mm")
    width = _geo_value("GEO-02_Width_mm")
    height = _geo_value("GEO-03_Height_mm")

    if length is not None:
        mapped["GEO__BBox_X_mm"] = length
    if width is not None:
        mapped["GEO__BBox_Y_mm"] = width
    if height is not None:
        mapped["GEO__BBox_Z_mm"] = height

    dims = [value for value in (length, width, height) if value is not None]
    if dims:
        mapped["GEO__MaxDim_mm"] = max(dims)
        mapped["GEO__MinDim_mm"] = min(dims)
        mapped["GEO__Stock_Thickness_mm"] = min(dims)

    volume = _geo_value("GEO-Volume_mm3", "GEO_Volume_mm3")
    if volume is not None:
        mapped["GEO__Volume_mm3"] = volume

    surface_area = _geo_value("GEO-SurfaceArea_mm2")
    if surface_area is not None:
        mapped["GEO__SurfaceArea_mm2"] = surface_area

    face_count = _geo_value("Feature_Face_Count", "GEO_Face_Count")
    if face_count is not None:
        mapped["GEO__Face_Count"] = face_count

    wedm_path = _geo_value("GEO_WEDM_PathLen_mm")
    if wedm_path is not None:
        mapped["GEO__WEDM_PathLen_mm"] = wedm_path

    if surface_area is not None and volume not in (None, 0.0):
        try:
            mapped["GEO__Area_to_Volume"] = surface_area / float(volume)
        except Exception:
            pass

    return mapped


def collect_geo_features_from_df(df: Any) -> dict[str, float]:
    """Extract GEO__ rows from a dataframe-like object into a mapping."""

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


def update_variables_df_with_geo(df: Any, geo: Mapping[str, Any] | None) -> Any:
    """Update or insert GEO__ rows within a dataframe-like object."""

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
            label_to_index[label.casefold()] = idx

    for label, value in geo_map.items():
        lookup = label.casefold()
        if lookup in label_to_index:
            index = label_to_index[lookup]
            _assign_cell(df, index, item_col, label)
            _assign_cell(df, index, value_col, value)
            if dtype_col and not _normalise_label(_get_cell(df, index, dtype_col)):
                _assign_cell(df, index, dtype_col, "number")
        else:
            index = _append_row(df, item_col, value_col, dtype_col, label, value, "number")
            label_to_index[lookup] = index
    return df


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


def upsert_var_row(df: Any, key: str, value: Any, *, dtype: str = "number") -> Any:
    """Insert or update a row in a dataframe-like object by Item label."""

    if df is None:
        return df

    item_col, value_col, dtype_col = _resolve_column_names(df)
    _ensure_column(df, item_col)
    _ensure_column(df, value_col)
    if dtype_col:
        _ensure_column(df, dtype_col)

    label = _normalise_label(key)
    coerced_value = _coerce_float(value)
    stored_value: Any = coerced_value if coerced_value is not None else value

    label_to_index: dict[str, int] = {}
    for idx, row in _iter_indexed_rows(df):
        existing_label = _normalise_label(row.get(item_col))
        if existing_label:
            label_to_index[existing_label.casefold()] = idx

    lookup = label.casefold()
    if lookup in label_to_index:
        index = label_to_index[lookup]
        _assign_cell(df, index, item_col, label)
        _assign_cell(df, index, value_col, stored_value)
        if dtype_col:
            _assign_cell(df, index, dtype_col, dtype)
        return df

    _append_row(df, item_col, value_col, dtype_col, label, stored_value, dtype)
    return df


__all__ = [
    "GeometryService",
    "EZDXF_VERSION",
    "HAS_EZDXF",
    "HAS_ODAFC",
    "convert_dwg_to_dxf",
    "get_dwg_converter_path",
    "require_ezdxf",
    "extract_text_lines_from_dxf",
    "parse_hole_table_lines",
    "extract_features_with_occ",
    "enrich_geo_stl",
    "read_step_shape",
    "read_step_or_iges_or_brep",
    "read_cad_any",
    "safe_bbox",
    "enrich_geo_occ",
    "get_import_diagnostics_text",
    "_hole_groups_from_cylinders",
    "map_geo_to_double_underscore",
    "collect_geo_features_from_df",
    "update_variables_df_with_geo",
    "upsert_var_row",
]
