from __future__ import annotations

"""OCCT tooling helpers (BRepTools, BRepGProp, etc.)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ._occt_base import PREFIX, STACK, load_module, missing
from .occt_core import BRep_Builder, TopoDS_Shape

__all__ = ["STACK_GPROP", "BRepTools", "bnd_add", "brep_read", "uv_bounds", "BRepGProp"]

# Default to "missing" implementations.  These provide stable module level
# attributes even when only the light-weight test doubles from ``tests.conftest``
# are present.
STACK_GPROP = "missing"
BRepTools = SimpleNamespace()


def bnd_add(*_: Any, **__: Any) -> Any:
    return missing("BRepBndLib.Add")


def brep_read(_: str | Path) -> Any:
    return missing("BRepTools.Read")


def uv_bounds(_: Any) -> tuple[float, float, float, float]:
    return missing("BRepTools.UVBounds")


BRepGProp = SimpleNamespace(  # type: ignore
    LinearProperties_s=lambda *_a, **_k: missing("BRepGProp.LinearProperties"),
    SurfaceProperties_s=lambda *_a, **_k: missing("BRepGProp.SurfaceProperties"),
    VolumeProperties_s=lambda *_a, **_k: missing("BRepGProp.VolumeProperties"),
)

if PREFIX is not None:  # pragma: no branch - exercised via feature detection
    try:
        breptools = load_module("BRepTools")
    except ImportError:
        breptools = None
    else:
        BRepTools = getattr(breptools, "BRepTools", breptools)

        _brep_read = getattr(breptools, "Read_s", None) or getattr(breptools, "Read", None)
        if _brep_read is None and hasattr(breptools, "breptools_Read"):
            _brep_read = getattr(breptools, "breptools_Read")

        def brep_read(path: str | Path) -> Any:  # type: ignore[no-redef]
            if _brep_read is None:
                raise RuntimeError("BRepTools.Read is unavailable in this build")
            shape = TopoDS_Shape()
            ok = _brep_read(shape, str(path), BRep_Builder())
            if ok is False:
                raise RuntimeError("BREP read failed")
            return shape

        _uv_bounds = getattr(breptools, "UVBounds", None)
        if _uv_bounds is None and hasattr(breptools, "breptools_UVBounds"):
            _uv_bounds = getattr(breptools, "breptools_UVBounds")

        def uv_bounds(face: Any) -> tuple[float, float, float, float]:  # type: ignore[no-redef]
            if _uv_bounds is None:
                raise RuntimeError("BRepTools.UVBounds is unavailable in this build")
            return _uv_bounds(face)

    if 'breptools' in locals() and breptools is not None:
        try:
            bnd_mod = load_module("BRepBndLib")
        except ImportError:
            _bnd_funcs: list[Any] = []
        else:
            _bnd_funcs = [
                getattr(bnd_mod, attr)
                for attr in ("Add", "Add_s", "BRepBndLib_Add", "brepbndlib_Add")
                if hasattr(bnd_mod, attr)
            ]

        if _bnd_funcs:
            def bnd_add(shape: Any, box: Any, use_triangulation: bool = True) -> Any:  # type: ignore[no-redef]
                last_error: Exception | None = None
                for fn in _bnd_funcs:
                    try:
                        return fn(shape, box, use_triangulation)
                    except TypeError as exc:
                        last_error = exc
                        try:
                            return fn(shape, box)
                        except TypeError as exc2:
                            last_error = exc2
                if last_error:
                    raise last_error
                raise RuntimeError("No usable BRepBndLib.Add variant")

    try:
        module = load_module("BRepGProp")
    except ImportError:
        pass
    else:
        try:
            BRepGProp = module.BRepGProp  # type: ignore[attr-defined]
            STACK_GPROP = STACK
        except AttributeError:
            BRepGProp = SimpleNamespace(  # type: ignore
                LinearProperties_s=getattr(module, "brepgprop_LinearProperties"),
                SurfaceProperties_s=getattr(module, "brepgprop_SurfaceProperties"),
                VolumeProperties_s=getattr(module, "brepgprop_VolumeProperties"),
            )
            STACK_GPROP = f"{STACK}-shim"
