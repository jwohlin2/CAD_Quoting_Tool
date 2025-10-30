"""Utilities that lazily access heavy geometry helpers.

This module provides lightweight wrappers around :mod:`cad_quoter.geometry`
objects. Importing :mod:`cad_quoter.geometry` at module import time pulls in a
large OCCT dependency stack, so these helpers make the commonly used entry
points available without eagerly importing every symbol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

from cad_quoter import geometry

STACK = getattr(geometry, "STACK", "pythonocc")
STACK_GPROP = getattr(geometry, "STACK_GPROP", STACK)


def _missing_geo_helper(name: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{name} is unavailable (OCCT bindings required)")

    return _raise


BND_ADD_FALLBACK: Callable[..., Any] = lambda *_args, **_kwargs: None
bnd_add = getattr(geometry, "bnd_add", BND_ADD_FALLBACK)
BRepTools_UVBounds = getattr(
    geometry, "uv_bounds", _missing_geo_helper("BRepTools.UVBounds")
)
BRepCheck_Analyzer = getattr(
    geometry, "BRepCheck_Analyzer", _missing_geo_helper("BRepCheck_Analyzer")
)

_read_step_or_iges_or_brep_impl = cast(
    Callable[[str | Path], Any],
    getattr(
        geometry,
        "read_step_or_iges_or_brep",
        _missing_geo_helper("read_step_or_iges_or_brep"),
    ),
)

_require_ezdxf_impl = cast(
    Callable[[], Any],
    getattr(geometry, "require_ezdxf", _missing_geo_helper("require_ezdxf")),
)

_convert_dwg_to_dxf_impl = cast(
    Callable[[str], str],
    getattr(
        geometry,
        "convert_dwg_to_dxf",
        _missing_geo_helper("convert_dwg_to_dxf"),
    ),
)

_get_dwg_converter_path_impl = cast(
    Callable[[], str | None],
    getattr(geometry, "get_dwg_converter_path", lambda: None),
)


def read_step_or_iges_or_brep(path: str | Path) -> Any:
    return _read_step_or_iges_or_brep_impl(path)


def require_ezdxf() -> Any:
    return _require_ezdxf_impl()


def convert_dwg_to_dxf(path: str) -> str:
    return _convert_dwg_to_dxf_impl(path)


def get_dwg_converter_path() -> str | None:
    return _get_dwg_converter_path_impl()


__all__ = [
    "STACK",
    "STACK_GPROP",
    "BND_ADD_FALLBACK",
    "bnd_add",
    "BRepTools_UVBounds",
    "BRepCheck_Analyzer",
    "read_step_or_iges_or_brep",
    "require_ezdxf",
    "convert_dwg_to_dxf",
    "get_dwg_converter_path",
]
