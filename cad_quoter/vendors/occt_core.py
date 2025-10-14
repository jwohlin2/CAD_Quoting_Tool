from __future__ import annotations

"""Core OCCT exports shared across the codebase."""

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
]

if PREFIX is None:  # pragma: no cover - bindings missing entirely
    def _sentinel(name: str) -> Any:
        def _raise(*_: Any, **__: Any) -> Any:
            return missing(name)

        return _raise

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
        mod = load_module(module)
        for name in names:
            globals()[name] = getattr(mod, name)
