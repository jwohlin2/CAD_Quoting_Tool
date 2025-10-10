from __future__ import annotations

from typing import Any as _Any, Protocol


class TopoDS_Shape(Protocol):
    ...


class TopoDS_Face(TopoDS_Shape, Protocol):
    ...


class TopoDS_Edge(TopoDS_Shape, Protocol):
    ...


class TopoDS_Solid(TopoDS_Shape, Protocol):
    ...


class TopoDS_Shell(TopoDS_Shape, Protocol):
    ...


class TopoDS_Compound(TopoDS_Shape, Protocol):
    ...


TopoDS: _Any
topods: _Any
