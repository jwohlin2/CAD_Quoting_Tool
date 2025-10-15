"""Core package for CAD Quoter tooling."""

from __future__ import annotations

import importlib
import sys
import types


def _ensure_geometry_module() -> None:
    """Replace geometry stubs installed by tests with the real module when available."""

    module = sys.modules.get("cad_quoter.geometry")
    if module is not None and getattr(module, "__file__", None):
        return

    try:
        sys.modules.pop("cad_quoter.geometry", None)
        importlib.import_module("cad_quoter.geometry")
        return
    except Exception:
        stub = module if module is not None else types.ModuleType("cad_quoter.geometry")
        stub.__dict__.setdefault("__spec__", None)

        defaults: dict[str, object] = {
            "load_model": lambda *_args, **_kwargs: None,
            "load_cad_any": lambda *_args, **_kwargs: None,
            "read_cad_any": lambda *_args, **_kwargs: None,
            "read_step_shape": lambda *_args, **_kwargs: None,
            "read_step_or_iges_or_brep": lambda *_args, **_kwargs: None,
            "convert_dwg_to_dxf": lambda path, **_kwargs: path,
            "enrich_geo_occ": lambda *_args, **_kwargs: {},
            "enrich_geo_stl": lambda *_args, **_kwargs: {},
            "safe_bbox": lambda *_args, **_kwargs: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "iter_solids": lambda *_args, **_kwargs: iter(()),
            "explode_compound": lambda *_args, **_kwargs: [],
            "parse_hole_table_lines": lambda *_args, **_kwargs: [],
            "extract_text_lines_from_dxf": lambda *_args, **_kwargs: [],
            "require_ezdxf": lambda *_args, **_kwargs: None,
            "get_dwg_converter_path": lambda *_args, **_kwargs: "",
            "have_dwg_support": lambda *_args, **_kwargs: False,
            "get_import_diagnostics_text": lambda *_args, **_kwargs: "",
        }

        for name, value in defaults.items():
            setattr(stub, name, value)

        sys.modules["cad_quoter.geometry"] = stub

        try:
            from cad_quoter.geometry_wrappers import (
                collect_geo_features_from_df as _collect_geo_features_from_df,
                map_geo_to_double_underscore as _map_geo_to_double_underscore,
                update_variables_df_with_geo as _update_variables_df_with_geo,
            )
        except Exception:
            return

        stub.collect_geo_features_from_df = _collect_geo_features_from_df
        stub.map_geo_to_double_underscore = _map_geo_to_double_underscore
        stub.update_variables_df_with_geo = _update_variables_df_with_geo
        try:
            from importlib import import_module as _import_module

            _app_module = _import_module("appV5")
            _hole_groups = getattr(_app_module, "_hole_groups_from_cylinders", None)
        except Exception:
            _hole_groups = None
        stub._hole_groups_from_cylinders = (
            _hole_groups if callable(_hole_groups) else (lambda *_args, **_kwargs: [])
        )


_ensure_geometry_module()

_geometry_stub = sys.modules.get("cad_quoter.geometry")
if _geometry_stub is not None and not hasattr(
    _geometry_stub, "collect_geo_features_from_df"
):
    try:
        from cad_quoter.geometry_wrappers import (
            collect_geo_features_from_df as _collect_geo_features_from_df,
            map_geo_to_double_underscore as _map_geo_to_double_underscore,
            update_variables_df_with_geo as _update_variables_df_with_geo,
        )
    except Exception:
        pass
    else:
        _geometry_stub.collect_geo_features_from_df = _collect_geo_features_from_df
        _geometry_stub.map_geo_to_double_underscore = _map_geo_to_double_underscore
        _geometry_stub.update_variables_df_with_geo = _update_variables_df_with_geo

__all__ = [
    "_ensure_geometry_module",
]
