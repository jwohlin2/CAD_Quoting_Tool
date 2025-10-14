from __future__ import annotations

"""Thin wrapper that re-exports the shared OCCT shim for appkit."""

from cad_quoter.vendors.occt import (  # noqa: F401
    BRepTools,
    BRep_Tool,
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_ShapeEnum,
    TopExp,
    TopExp_Explorer,
    TopoDS,
    TopoDS_Face,
    TopoDS_Shape,
    TopTools_IndexedDataMapOfShapeListOfShape,
)

__all__ = [name for name in globals() if not name.startswith("_")]
