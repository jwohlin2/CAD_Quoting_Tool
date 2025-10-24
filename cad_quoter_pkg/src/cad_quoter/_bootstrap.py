"""Helpers for bootstrapping :mod:`cad_quoter.geometry` in different contexts."""

from __future__ import annotations

from importlib import import_module, util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Iterable, Iterator, Sequence
import sys
import types

_GEOMETRY_MODULE = "cad_quoter.geometry"
_GEOMETRY_SUBMODULES = ("dxf_text", "dxf_enrich")

_STUB_DEFAULTS: dict[str, object] = {
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


def ensure_geometry_module(
    *, extra_search_paths: Iterable[Path | str] | None = None,
) -> ModuleType:
    """Ensure :mod:`cad_quoter.geometry` is importable and fully bootstrapped."""

    search_paths = list(_normalize_search_paths(extra_search_paths))
    _extend_sys_path(search_paths)

    existing = sys.modules.get(_GEOMETRY_MODULE)
    if existing is not None and getattr(existing, "__file__", None):
        module = existing
    else:
        module = _load_real_geometry_module(existing)
        if module is None:
            module = _ensure_geometry_stub(existing)

    _load_geometry_submodules(module, search_paths)
    return module


def _normalize_search_paths(
    extra_search_paths: Iterable[Path | str] | None,
) -> Iterator[Path]:
    if not extra_search_paths:
        return iter(())
    paths: list[Path] = []
    for raw_path in extra_search_paths:
        try:
            path = Path(raw_path)
        except TypeError:
            continue
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved.is_dir():
            paths.append(resolved)
    return iter(paths)


def _extend_sys_path(search_paths: Sequence[Path]) -> None:
    for path in search_paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_real_geometry_module(existing: ModuleType | None) -> ModuleType | None:
    sys.modules.pop(_GEOMETRY_MODULE, None)
    try:
        module = import_module(_GEOMETRY_MODULE)
    except Exception:
        if existing is not None:
            sys.modules[_GEOMETRY_MODULE] = existing
        return None
    return module


def _ensure_geometry_stub(existing: ModuleType | None) -> ModuleType:
    stub = existing if existing is not None else types.ModuleType(_GEOMETRY_MODULE)
    stub.__dict__.setdefault("__spec__", None)

    for name, value in _STUB_DEFAULTS.items():
        setattr(stub, name, value)

    sys.modules[_GEOMETRY_MODULE] = stub

    try:
        from cad_quoter.utils.geo_fallbacks import (
            collect_geo_features_from_df as _collect_geo_features_from_df,
            map_geo_to_double_underscore as _map_geo_to_double_underscore,
            update_variables_df_with_geo as _update_variables_df_with_geo,
        )
    except Exception:
        pass
    else:
        stub.collect_geo_features_from_df = _collect_geo_features_from_df
        stub.map_geo_to_double_underscore = _map_geo_to_double_underscore
        stub.update_variables_df_with_geo = _update_variables_df_with_geo

    try:
        from cad_quoter.geometry import _hole_groups_from_cylinders as _hole_groups
    except Exception:
        _hole_groups = None

    stub._hole_groups_from_cylinders = (
        _hole_groups if callable(_hole_groups) else (lambda *_args, **_kwargs: [])
    )

    return stub


def _load_geometry_submodules(module: ModuleType, search_paths: Sequence[Path]) -> None:
    try:
        geometry_file = Path(getattr(module, "__file__", "")).resolve()
    except Exception:
        geometry_file = None

    candidates: list[Path] = []
    if geometry_file is not None and geometry_file.exists():
        candidates.append(geometry_file.parent)

    module_path = getattr(module, "__path__", None)
    if module_path:
        for entry in module_path:
            try:
                candidate = Path(entry)
            except Exception:
                continue
            if candidate not in candidates:
                candidates.append(candidate)

    for extra in search_paths:
        for candidate in _possible_geometry_dirs(extra):
            if candidate not in candidates:
                candidates.append(candidate)

    for submodule in _GEOMETRY_SUBMODULES:
        _load_geometry_submodule(module, submodule, candidates)


def _possible_geometry_dirs(base: Path) -> Iterator[Path]:
    potential = [
        base,
        base / "cad_quoter" / "geometry",
        base / "geometry",
    ]
    seen: set[Path] = set()
    for candidate in potential:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            yield resolved


def _load_geometry_submodule(module: ModuleType, name: str, candidates: Sequence[Path]) -> None:
    module_name = f"{_GEOMETRY_MODULE}.{name}"
    if module_name in sys.modules:
        setattr(module, name, sys.modules[module_name])
        return

    for candidate in candidates:
        location = candidate / f"{name}.py"
        if not location.is_file():
            continue
        loaded = _load_module_from_path(module_name, location)
        if loaded is not None:
            setattr(module, name, loaded)
            return


def _load_module_from_path(module_name: str, location: Path) -> ModuleType | None:
    spec = importlib_util.spec_from_file_location(module_name, location)
    if spec is None or spec.loader is None:
        return None

    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        sys.modules.pop(module_name, None)
        return None
    return module


__all__ = ["ensure_geometry_module"]

