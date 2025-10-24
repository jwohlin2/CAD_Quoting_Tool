from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable
from types import ModuleType


CAPABILITY_NAMES = [
    "load_model",
    "load_cad_any",
    "read_cad_any",
    "read_step_shape",
    "read_step_or_iges_or_brep",
    "convert_dwg_to_dxf",
    "enrich_geo_occ",
    "enrich_geo_stl",
    "safe_bbox",
    "iter_solids",
    "explode_compound",
    "parse_hole_table_lines",
    "extract_text_lines_from_dxf",
    "require_ezdxf",
    "get_dwg_converter_path",
    "have_dwg_support",
    "get_import_diagnostics_text",
    "collect_geo_features_from_df",
    "map_geo_to_double_underscore",
    "update_variables_df_with_geo",
    "_hole_groups_from_cylinders",
]

GEOMETRY_SUBMODULES = ("dxf_text", "dxf_enrich")


def _clear_geometry_modules() -> None:
    for name in [key for key in sys.modules if key == "cad_quoter" or key.startswith("cad_quoter.")]:
        sys.modules.pop(name, None)


def _temporarily_load_geometry(paths: Iterable[Path], *, drop_repo_root: bool) -> object:
    repo_root = Path(__file__).resolve().parents[2]
    added_paths: list[str] = []
    removed_entries: list[tuple[int, str]] = []
    try:
        if drop_repo_root:
            for index in range(len(sys.path) - 1, -1, -1):
                entry = sys.path[index]
                try:
                    resolved = Path(entry or ".").resolve()
                except Exception:
                    continue
                if resolved == repo_root:
                    removed_entries.append((index, entry))
                    sys.path.pop(index)

        for path in paths:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
                added_paths.append(text)

        _clear_geometry_modules()
        importlib.invalidate_caches()
        geometry = importlib.import_module("cad_quoter.geometry")
        return geometry
    finally:
        for text in added_paths:
            try:
                sys.path.remove(text)
            except ValueError:
                pass
        for index, entry in reversed(removed_entries):
            sys.path.insert(index, entry)


def _capability_signature(obj: object) -> tuple[str | None, str | None, bool]:
    if obj is None:
        return (None, None, False)
    module_name = getattr(obj, "__module__", None)
    name = getattr(obj, "__name__", None)
    return (module_name, name, callable(obj))


def _snapshot_capabilities(geometry: object) -> dict[str, tuple[str | None, str | None, bool]]:
    snapshot: dict[str, tuple[str | None, str | None, bool]] = {}
    for name in CAPABILITY_NAMES:
        snapshot[name] = _capability_signature(getattr(geometry, name, None))
    for submodule in GEOMETRY_SUBMODULES:
        member = getattr(geometry, submodule, None)
        module_name = getattr(member, "__name__", None)
        module_file = getattr(member, "__file__", None)
        snapshot[f"submodule:{submodule}"] = (
            module_name,
            module_file,
            isinstance(member, ModuleType),
        )
    return snapshot


def test_geometry_capabilities_match_between_installed_and_repo_checkout():
    repo_root = Path(__file__).resolve().parents[2]
    pkg_src = repo_root / "src"

    installed_geometry = _temporarily_load_geometry([pkg_src], drop_repo_root=True)
    installed_snapshot = _snapshot_capabilities(installed_geometry)

    repo_geometry = _temporarily_load_geometry([repo_root / "src"], drop_repo_root=False)
    repo_snapshot = _snapshot_capabilities(repo_geometry)

    assert repo_snapshot == installed_snapshot
