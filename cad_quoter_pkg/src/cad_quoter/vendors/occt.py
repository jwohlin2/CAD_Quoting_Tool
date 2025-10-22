from __future__ import annotations

"""Public OCCT shim that combines the legacy helper modules."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ._occt_base import BACKEND, PREFIX, STACK, load_module, missing

__all__ = [
    "STACK",
    "BACKEND",
    "STEPControl_Reader",
    "IGESControl_Reader",
    "IFSelect_RetDone",
    "TopoDS",
    "TopoDS_Shape",
    "TopoDS_Compound",
    "TopoDS_Face",
    "TopExp",
    "TopExp_Explorer",
    "TopAbs_FACE",
    "TopAbs_EDGE",
    "TopAbs_SOLID",
    "TopAbs_SHELL",
    "TopAbs_COMPOUND",
    "TopAbs_ShapeEnum",
    "TopTools_IndexedDataMapOfShapeListOfShape",
    "BRep_Builder",
    "BRep_Tool",
    "BRepAlgoAPI_Section",
    "BRepAdaptor_Curve",
    "ShapeFix_Shape",
    "BRepCheck_Analyzer",
    "Bnd_Box",
    "GeomAdaptor_Surface",
    "ShapeAnalysis_Surface",
    "GProp_GProps",
    "GeomAbs_Plane",
    "GeomAbs_Cylinder",
    "GeomAbs_Torus",
    "GeomAbs_Cone",
    "GeomAbs_BSplineSurface",
    "GeomAbs_BezierSurface",
    "GeomAbs_Circle",
    "gp_Pnt",
    "gp_Dir",
    "gp_Pln",
    "gp_Vec",
    "STACK_GPROP",
    "BRepTools",
    "bnd_add",
    "brep_read",
    "uv_bounds",
    "BRepGProp",
]


def _sentinel(name: str) -> Any:
    def _raise(*_: Any, **__: Any) -> Any:
        return missing(name)

    return _raise


if PREFIX is None:  # pragma: no cover - bindings missing entirely
    STEPControl_Reader = _sentinel("STEPControl_Reader")  # type: ignore
    IGESControl_Reader = _sentinel("IGESControl_Reader")  # type: ignore
    IFSelect_RetDone = _sentinel("IFSelect_RetDone")  # type: ignore
    TopoDS = _sentinel("TopoDS")  # type: ignore
    TopoDS_Shape = _sentinel("TopoDS_Shape")  # type: ignore
    TopoDS_Compound = _sentinel("TopoDS_Compound")  # type: ignore
    TopoDS_Face = _sentinel("TopoDS_Face")  # type: ignore
    TopExp = _sentinel("TopExp")  # type: ignore
    TopExp_Explorer = _sentinel("TopExp_Explorer")  # type: ignore
    TopAbs_FACE = _sentinel("TopAbs_FACE")  # type: ignore
    TopAbs_EDGE = _sentinel("TopAbs_EDGE")  # type: ignore
    TopAbs_SOLID = _sentinel("TopAbs_SOLID")  # type: ignore
    TopAbs_SHELL = _sentinel("TopAbs_SHELL")  # type: ignore
    TopAbs_COMPOUND = _sentinel("TopAbs_COMPOUND")  # type: ignore
    TopAbs_ShapeEnum = _sentinel("TopAbs_ShapeEnum")  # type: ignore
    TopTools_IndexedDataMapOfShapeListOfShape = _sentinel(
        "TopTools_IndexedDataMapOfShapeListOfShape"
    )  # type: ignore
    BRep_Builder = _sentinel("BRep_Builder")  # type: ignore
    BRep_Tool = _sentinel("BRep_Tool")  # type: ignore
    BRepAlgoAPI_Section = _sentinel("BRepAlgoAPI_Section")  # type: ignore
    BRepAdaptor_Curve = _sentinel("BRepAdaptor_Curve")  # type: ignore
    ShapeFix_Shape = _sentinel("ShapeFix_Shape")  # type: ignore
    BRepCheck_Analyzer = _sentinel("BRepCheck_Analyzer")  # type: ignore
    Bnd_Box = _sentinel("Bnd_Box")  # type: ignore
    GeomAdaptor_Surface = _sentinel("GeomAdaptor_Surface")  # type: ignore
    ShapeAnalysis_Surface = _sentinel("ShapeAnalysis_Surface")  # type: ignore
    GProp_GProps = _sentinel("GProp_GProps")  # type: ignore
    GeomAbs_Plane = _sentinel("GeomAbs_Plane")  # type: ignore
    GeomAbs_Cylinder = _sentinel("GeomAbs_Cylinder")  # type: ignore
    GeomAbs_Torus = _sentinel("GeomAbs_Torus")  # type: ignore
    GeomAbs_Cone = _sentinel("GeomAbs_Cone")  # type: ignore
    GeomAbs_BSplineSurface = _sentinel("GeomAbs_BSplineSurface")  # type: ignore
    GeomAbs_BezierSurface = _sentinel("GeomAbs_BezierSurface")  # type: ignore
    GeomAbs_Circle = _sentinel("GeomAbs_Circle")  # type: ignore
    gp_Pnt = _sentinel("gp_Pnt")  # type: ignore
    gp_Dir = _sentinel("gp_Dir")  # type: ignore
    gp_Pln = _sentinel("gp_Pln")  # type: ignore
    gp_Vec = _sentinel("gp_Vec")  # type: ignore
else:
    _map = {
        "STEPControl": ("STEPControl_Reader",),
        "IGESControl": ("IGESControl_Reader",),
        "IFSelect": ("IFSelect_RetDone",),
        "TopoDS": ("TopoDS", "TopoDS_Shape", "TopoDS_Compound", "TopoDS_Face"),
        "TopExp": ("TopExp", "TopExp_Explorer"),
        "TopAbs": (
            "TopAbs_FACE",
            "TopAbs_EDGE",
            "TopAbs_SOLID",
            "TopAbs_SHELL",
            "TopAbs_COMPOUND",
            "TopAbs_ShapeEnum",
        ),
        "TopTools": ("TopTools_IndexedDataMapOfShapeListOfShape",),
        "BRep": ("BRep_Builder", "BRep_Tool"),
        "BRepAlgoAPI": ("BRepAlgoAPI_Section",),
        "BRepAdaptor": ("BRepAdaptor_Curve",),
        "ShapeFix": ("ShapeFix_Shape",),
        "BRepCheck": ("BRepCheck_Analyzer",),
        "Bnd": ("Bnd_Box",),
        "GeomAdaptor": ("GeomAdaptor_Surface",),
        "ShapeAnalysis": ("ShapeAnalysis_Surface",),
        "GProp": ("GProp_GProps",),
    }
    for module, names in _map.items():
        try:
            mod = load_module(module)
        except ImportError:
            for name in names:
                globals()[name] = _sentinel(name)
            continue
        for name in names:
            globals()[name] = getattr(mod, name, _sentinel(name))

    try:
        geomabs = load_module("GeomAbs")
    except ImportError:
        for name in (
            "GeomAbs_Plane",
            "GeomAbs_Cylinder",
            "GeomAbs_Torus",
            "GeomAbs_Cone",
            "GeomAbs_BSplineSurface",
            "GeomAbs_BezierSurface",
            "GeomAbs_Circle",
        ):
            globals()[name] = _sentinel(name)
    else:
        for name in (
            "GeomAbs_Plane",
            "GeomAbs_Cylinder",
            "GeomAbs_Torus",
            "GeomAbs_Cone",
            "GeomAbs_BSplineSurface",
            "GeomAbs_BezierSurface",
            "GeomAbs_Circle",
        ):
            globals()[name] = getattr(geomabs, name, _sentinel(name))

    try:
        gp_mod = load_module("gp")
    except ImportError:
        for name in ("gp_Pnt", "gp_Dir", "gp_Pln", "gp_Vec"):
            globals()[name] = _sentinel(name)
    else:
        for name in ("gp_Pnt", "gp_Dir", "gp_Pln", "gp_Vec"):
            globals()[name] = getattr(gp_mod, name, _sentinel(name))

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

    if "breptools" in locals() and breptools is not None:
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
