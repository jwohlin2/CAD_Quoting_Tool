from __future__ import annotations

from typing import Any, cast, Type

_missing_msg = (
    "OCCT bindings are required for geometry operations. "
    "Please install pythonocc-core or the OCP wheels."
)

try:
    # Prefer OCP
    from OCP.BRep import BRep_Tool  # type: ignore
    from OCP.TopAbs import (  # type: ignore
        TopAbs_COMPOUND,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_ShapeEnum,
    )
    from OCP.TopExp import TopExp, TopExp_Explorer  # type: ignore
    from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape  # type: ignore
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape  # type: ignore
    from OCP.BRepTools import BRepTools  # type: ignore
except Exception:
    try:
        # Fallback to OCC.Core
        from OCC.Core.BRep import BRep_Tool  # type: ignore
        from OCC.Core.TopAbs import (  # type: ignore
            TopAbs_COMPOUND,
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_SHELL,
            TopAbs_SOLID,
            TopAbs_ShapeEnum,
        )
        from OCC.Core.TopExp import TopExp, TopExp_Explorer  # type: ignore
        from OCC.Core.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape  # type: ignore
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape  # type: ignore
        from OCC.Core.BRepTools import BRepTools  # type: ignore
    except Exception:
        class _MissingOCCTSentinelMeta(type):
            def __call__(cls, *args: Any, **kwargs: Any):
                raise ImportError(_missing_msg)

        class _MissingOCCTSentinel(metaclass=_MissingOCCTSentinelMeta):
            @classmethod
            def __getattr__(cls, item: str):
                raise ImportError(_missing_msg)

        _missing_type = cast(Type[Any], _MissingOCCTSentinel)
        _missing_any = cast(Any, _MissingOCCTSentinel)
        BRep_Tool = _missing_any
        TopAbs_EDGE = TopAbs_FACE = TopAbs_SHELL = TopAbs_SOLID = TopAbs_COMPOUND = TopAbs_ShapeEnum = _missing_any  # type: ignore
        TopExp = TopExp_Explorer = _missing_any
        TopoDS = TopoDS_Face = TopoDS_Shape = _missing_type
        TopTools_IndexedDataMapOfShapeListOfShape = _missing_type

        class _MissingBRepTools:
            def __getattr__(self, _):
                raise ImportError(_missing_msg)

        BRepTools = cast(Any, _MissingBRepTools())

__all__ = [
    "BRep_Tool",
    "TopAbs_COMPOUND",
    "TopAbs_EDGE",
    "TopAbs_FACE",
    "TopAbs_SHELL",
    "TopAbs_SOLID",
    "TopAbs_ShapeEnum",
    "TopExp",
    "TopExp_Explorer",
    "TopoDS",
    "TopoDS_Face",
    "TopoDS_Shape",
    "TopTools_IndexedDataMapOfShapeListOfShape",
    "BRepTools",
]

