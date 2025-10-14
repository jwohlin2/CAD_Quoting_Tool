from __future__ import annotations

"""Surface / vector helpers from OCCT."""

from typing import Any

from ._occt_base import PREFIX, load_module, missing

__all__ = [
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
]

if PREFIX is None:  # pragma: no cover - bindings missing entirely
    def _sentinel(name: str) -> Any:
        def _raise(*_: Any, **__: Any) -> Any:
            return missing(name)

        return _raise

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
    geomabs = load_module("GeomAbs")
    for name in (
        "GeomAbs_Plane",
        "GeomAbs_Cylinder",
        "GeomAbs_Torus",
        "GeomAbs_Cone",
        "GeomAbs_BSplineSurface",
        "GeomAbs_BezierSurface",
        "GeomAbs_Circle",
    ):
        globals()[name] = getattr(geomabs, name)

    gp_mod = load_module("gp")
    for name in ("gp_Pnt", "gp_Dir", "gp_Pln", "gp_Vec"):
        if hasattr(gp_mod, name):
            globals()[name] = getattr(gp_mod, name)
