from __future__ import annotations

"""Shared OCCT helpers that smooth over backend differences for appkit."""

from importlib import import_module
from typing import Any, Callable, Iterator

_MISSING_HELP = (
    "OCCT bindings are required for geometry operations. "
    "Install pythonocc-core or the OCP wheels."
)

try:
    from cad_quoter.vendors.occt import (  # noqa: F401
        BACKEND,
        STACK,
        STACK_GPROP,
        BRepAdaptor_Curve,
        BRepAlgoAPI_Section,
        BRep_Builder,
        BRepCheck_Analyzer,
        BRepGProp,
        BRepTools,
        BRep_Tool,
        GProp_GProps,
        TopAbs_COMPOUND,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_ShapeEnum,
        TopExp,
        TopExp_Explorer,
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopoDS,
        TopoDS_Compound,
        TopoDS_Face,
        TopoDS_Shape,
    )
except Exception as _OCCT_ERROR:  # pragma: no cover - environment without OCCT
    class _MissingProxy:
        def __init__(self, name: str) -> None:
            self._name = name

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(f"{self._name} requires OCCT bindings. {_MISSING_HELP}") from _OCCT_ERROR

        def __getattr__(self, attr: str) -> Any:
            raise ImportError(
                f"{self._name}.{attr} requires OCCT bindings. {_MISSING_HELP}"
            ) from _OCCT_ERROR

        def __iter__(self):
            raise ImportError(f"{self._name} requires OCCT bindings. {_MISSING_HELP}") from _OCCT_ERROR

        def __bool__(self) -> bool:  # pragma: no cover - sentinel helper
            return False

    def _missing_func(name: str) -> Callable[..., Any]:
        def _impl(*_: Any, **__: Any) -> Any:
            raise ImportError(f"{name} requires OCCT bindings. {_MISSING_HELP}") from _OCCT_ERROR

        return _impl

    BACKEND = STACK = STACK_GPROP = "missing"  # type: ignore[assignment]
    for _name in [
        "BRepAdaptor_Curve",
        "BRepAlgoAPI_Section",
        "BRep_Builder",
        "BRepCheck_Analyzer",
        "BRepGProp",
        "BRepTools",
        "BRep_Tool",
        "GProp_GProps",
        "TopAbs_COMPOUND",
        "TopAbs_EDGE",
        "TopAbs_FACE",
        "TopAbs_SHELL",
        "TopAbs_SOLID",
        "TopAbs_ShapeEnum",
        "TopExp",
        "TopExp_Explorer",
        "TopTools_IndexedDataMapOfShapeListOfShape",
        "TopoDS",
        "TopoDS_Compound",
        "TopoDS_Face",
        "TopoDS_Shape",
    ]:
        globals()[_name] = _MissingProxy(_name)

    FACE_OF = _missing_func("FACE_OF")
    as_face = _missing_func("as_face")
    ensure_shape = _missing_func("ensure_shape")
    ensure_face = _missing_func("ensure_face")
    iter_faces = _missing_func("iter_faces")
    face_surface = _missing_func("face_surface")
    to_edge_safe = _missing_func("to_edge_safe")
    to_edge = _missing_func("to_edge")
    to_solid = _missing_func("to_solid")
    to_shell = _missing_func("to_shell")
    map_size = _missing_func("map_size")
    list_iter = _missing_func("list_iter")
    linear_properties = _missing_func("linear_properties")
    map_shapes_and_ancestors = _missing_func("map_shapes_and_ancestors")

    __all__ = [name for name in globals() if not name.startswith("_")]
else:

    def _resolve_cast(name: str) -> Callable[[Any], Any]:
        """Best-effort resolver for TopoDS casting helpers."""

        # Try modern static helpers exposed on the TopoDS namespace first.
        topods_any = TopoDS
        for attr in (f"{name}_s", name):
            fn = getattr(topods_any, attr, None)
            if callable(fn):
                return fn

        def _module_candidates() -> Iterator[str]:
            if STACK == "ocp":
                yield "OCP.TopoDS"
            yield "OCC.Core.TopoDS"

        for module_name in _module_candidates():
            try:
                module = import_module(module_name)
            except Exception:
                continue

            for attr in (f"topods_{name}", name):
                fn = getattr(module, attr, None)
                if callable(fn):
                    return fn

            topods_ns = getattr(module, "topods", None)
            if topods_ns is not None:
                fn = getattr(topods_ns, name, None)
                if callable(fn):
                    return fn

        def _raise(obj: Any) -> Any:
            raise TypeError(f"Cannot cast {type(obj).__name__} to TopoDS_{name}")

        return _raise

    FACE_OF = _resolve_cast("Face")
    _TO_EDGE = _resolve_cast("Edge")
    _TO_SOLID = _resolve_cast("Solid")
    _TO_SHELL = _resolve_cast("Shell")

    def _unwrap_value(obj: Any) -> Any:
        """Unwrap OCCT iterator nodes that expose a ``Value()`` accessor."""

        return obj.Value() if hasattr(obj, "Value") and callable(obj.Value) else obj

    def _is_named(obj: Any, names: tuple[str, ...]) -> bool:
        try:
            return type(obj).__name__ in names
        except Exception:
            return False

    def _is_instance(obj: Any, qualnames: tuple[str, ...]) -> bool:
        try:
            return type(obj).__name__ in qualnames
        except Exception:
            return False

    def ensure_shape(obj: Any) -> Any:
        obj = _unwrap_value(obj)
        if obj is None:
            raise TypeError("Expected TopoDS_Shape, got None")
        if hasattr(obj, "IsNull") and obj.IsNull():
            raise TypeError("Expected non-null TopoDS_Shape")
        return obj

    def ensure_face(obj: Any) -> Any:
        obj = _unwrap_value(obj)
        if obj is None:
            raise TypeError("Expected a face, got None")
        if isinstance(obj, TopoDS_Face) or type(obj).__name__ == "TopoDS_Face":
            return obj
        st = obj.ShapeType() if hasattr(obj, "ShapeType") else None
        if st == TopAbs_FACE:
            return FACE_OF(obj)
        raise TypeError(f"Not a face: {type(obj).__name__}")

    def as_face(obj: Any) -> Any:
        """Cast any face-like object (tuples, handles, shapes) to ``TopoDS_Face``."""

        if obj is None:
            raise TypeError("Expected a shape, got None")
        if isinstance(obj, tuple) and obj:
            obj = obj[0]
        return ensure_face(obj)

    def iter_faces(shape: Any) -> Iterator[Any]:
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            yield ensure_face(exp.Current())
            exp.Next()

    def face_surface(face_like: Any) -> tuple[Any, Any | None]:
        f = ensure_face(face_like)
        tool = BRep_Tool
        surface_s = getattr(tool, "Surface_s", None)
        if callable(surface_s):
            s = surface_s(f)
        else:
            s = tool.Surface(f)
        location_fn = getattr(tool, "Location", None)
        if callable(location_fn):
            loc = location_fn(f)
        else:
            face_location = getattr(f, "Location", None)
            loc = face_location() if callable(face_location) else None
        if isinstance(s, tuple):
            s, loc2 = s
            loc = loc or loc2
        if hasattr(s, "Surface"):
            s = s.Surface()
        return s, loc

    def to_edge_safe(obj: Any) -> Any:
        obj = _unwrap_value(obj)
        if _is_named(obj, ("TopoDS_Edge", "Edge")):
            return obj
        return _TO_EDGE(ensure_shape(obj))

    def to_edge(obj: Any) -> Any:
        if _is_instance(obj, ("TopoDS_Edge", "Edge")):
            return obj
        return _TO_EDGE(obj)

    def to_solid(obj: Any) -> Any:
        if _is_instance(obj, ("TopoDS_Solid", "Solid")):
            return obj
        return _TO_SOLID(obj)

    def to_shell(obj: Any) -> Any:
        if _is_instance(obj, ("TopoDS_Shell", "Shell")):
            return obj
        return _TO_SHELL(obj)

    def map_size(obj: Any) -> int:
        for name in ("Size", "Extent", "Length"):
            if hasattr(obj, name):
                return getattr(obj, name)()
        raise AttributeError(f"No size method on {type(obj)}")

    def list_iter(lst: Any) -> Iterator[Any]:
        if hasattr(lst, "cbegin"):
            it = lst.cbegin()
            while it.More():
                yield it.Value()
                it.Next()
        else:
            for value in list(lst):
                yield value

    def linear_properties(edge: Any, gprops: Any) -> Any:
        """Linear properties across OCP/pythonocc names."""

        fn = getattr(BRepGProp, "LinearProperties", None)
        if fn is None:
            fn = getattr(BRepGProp, "LinearProperties_s", None)
        if fn is None:
            try:
                from OCC.Core.BRepGProp import (  # type: ignore[import-not-found]
                    brepgprop_LinearProperties as _legacy,
                )
            except Exception:
                raise
            return _legacy(edge, gprops)
        return fn(edge, gprops)

    def map_shapes_and_ancestors(root_shape: Any, sub_enum: Any, anc_enum: Any) -> Any:
        """Return ``TopTools_IndexedDataMapOfShapeListOfShape`` for (sub → ancestors)."""

        if root_shape is None:
            raise TypeError("root_shape is None")
        if hasattr(root_shape, "IsNull") and root_shape.IsNull():
            raise TypeError("root_shape is null")

        amap = TopTools_IndexedDataMapOfShapeListOfShape()
        fn = getattr(TopExp, "MapShapesAndAncestors", None) or getattr(
            TopExp, "MapShapesAndAncestors_s", None
        )
        if fn is None:
            raise RuntimeError("TopExp.MapShapesAndAncestors is unavailable")
        fn(root_shape, sub_enum, anc_enum, amap)
        return amap

    __all__ = [name for name in globals() if not name.startswith("_")]
