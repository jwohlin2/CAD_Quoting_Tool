from __future__ import annotations

"""Public OCCT shim that re-exports submodules."""

from .occt_core import (  # noqa: F401
    BACKEND,
    STACK,
    STEPControl_Reader,
    IGESControl_Reader,
    IFSelect_RetDone,
    TopoDS,
    TopoDS_Shape,
    TopoDS_Compound,
    TopoDS_Face,
    TopExp,
    TopExp_Explorer,
    TopAbs_FACE,
    TopAbs_EDGE,
    TopAbs_SOLID,
    TopAbs_SHELL,
    TopAbs_COMPOUND,
    TopAbs_ShapeEnum,
    TopTools_IndexedDataMapOfShapeListOfShape,
    BRep_Builder,
    BRep_Tool,
    BRepAlgoAPI_Section,
    BRepAdaptor_Curve,
    ShapeFix_Shape,
    BRepCheck_Analyzer,
    Bnd_Box,
    GeomAdaptor_Surface,
    ShapeAnalysis_Surface,
    GProp_GProps,
)
from .occt_geom import (  # noqa: F401
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Torus,
    GeomAbs_Cone,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_Circle,
    gp_Pnt,
    gp_Dir,
    gp_Pln,
    gp_Vec,
)
from .occt_tools import (  # noqa: F401
    STACK_GPROP,
    BRepTools,
    bnd_add,
    brep_read,
    uv_bounds,
    BRepGProp,
)

__all__ = [name for name in globals() if not name.startswith("_")]
