"""Local aggregate package for development without installation."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Mapping

_package_root = Path(__file__).resolve().parent
__path__ = [str(_package_root)]

_extra_src = _package_root.parent / "cad_quoter_pkg" / "src" / "cad_quoter"
if _extra_src.exists():
    extra_src_text = str(_extra_src)
    if extra_src_text not in __path__:
        __path__.append(extra_src_text)


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


def _iter_indexed_rows(df: Any) -> Iterable[tuple[int, Mapping[str, Any]]]:
    iterrows = getattr(df, "iterrows", None)
    if callable(iterrows):
        return ((idx, dict(row)) for idx, row in iterrows())
    if isinstance(df, Mapping):
        return [(0, df)]
    if isinstance(df, Iterable):
        return ((idx, dict(row)) for idx, row in enumerate(df))
    return []


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


def _fallback_map_geo_to_double_underscore(geo: Mapping[str, Any] | None) -> dict[str, float]:
    data = dict(geo or {})

    def _geo_value(*keys: str) -> float | None:
        for key in keys:
            coerced = _coerce_float(data.get(key))
            if coerced is not None:
                return coerced
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


def _fallback_collect_geo_features_from_df(df: Any) -> dict[str, float]:
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


def _fallback_update_variables_df_with_geo(df: Any, geo: Mapping[str, Any] | None) -> Any:
    if df is None:
        return df

    geo_map: dict[str, float] = {}
    for key, value in (geo or {}).items():
        label = _normalise_label(key)
        coerced = _coerce_float(value)
        if label and coerced is not None:
            geo_map[label] = coerced

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


def _load_module_from_path(name: str, path: Path) -> types.ModuleType | None:
    if not path.exists():
        return None

    try:
        from importlib.machinery import SourceFileLoader
        from importlib.util import module_from_spec, spec_from_loader

        loader = SourceFileLoader(name, str(path))
        spec = spec_from_loader(name, loader)
        if spec is None or spec.loader is None:
            return None
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    except Exception:
        return None


def _load_real_geometry_module() -> types.ModuleType | None:
    candidates = [
        _extra_src / "geometry" / "__init__.py",
        _package_root / "geometry" / "__init__.py",
    ]
    for path in candidates:
        module = _load_module_from_path("_cad_quoter_geometry_real", path)
        if module is not None:
            return module
    return None


def _load_geometry_submodule(name: str) -> types.ModuleType | None:
    candidates = [
        _extra_src / "geometry" / f"{name}.py",
        _package_root / "geometry" / f"{name}.py",
    ]
    for path in candidates:
        module = _load_module_from_path(f"_cad_quoter_geometry_{name}", path)
        if module is not None:
            sys.modules[f"cad_quoter.geometry.{name}"] = module
            return module
    return None


def _ensure_geometry_stub() -> None:
    module = sys.modules.get("cad_quoter.geometry")
    if module is None:
        module = types.ModuleType("cad_quoter.geometry")
        module.__spec__ = None  # type: ignore[attr-defined]
        sys.modules["cad_quoter.geometry"] = module

    fallbacks = {
        "map_geo_to_double_underscore": _fallback_map_geo_to_double_underscore,
        "collect_geo_features_from_df": _fallback_collect_geo_features_from_df,
        "update_variables_df_with_geo": _fallback_update_variables_df_with_geo,
        "_hole_groups_from_cylinders": lambda *_args, **_kwargs: [],
    }
    for name, value in fallbacks.items():
        if not hasattr(module, name):
            setattr(module, name, value)

    real_module = _load_real_geometry_module()
    if real_module is not None:
        exported = getattr(real_module, "__all__", [])
        if not exported:
            exported = [name for name in dir(real_module) if not name.startswith("_")]
        for name in exported:
            if hasattr(real_module, name):
                setattr(module, name, getattr(real_module, name))
        for submodule in ("dxf_enrich", "dxf_text", "hole_table_parser", "occ_compat"):
            _load_geometry_submodule(submodule)


_ensure_geometry_stub()

__all__ = []
