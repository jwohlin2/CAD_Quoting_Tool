from __future__ import annotations

from typing import Any as _Any

from .TopAbs import TopAbs_ShapeEnum
from .TopoDS import TopoDS_Shape


TopExp: _Any


class TopExp_Explorer:
    def __init__(
        self,
        S: TopoDS_Shape,
        ToFind: TopAbs_ShapeEnum,
        ToAvoid: TopAbs_ShapeEnum | None = None,
    ) -> None: ...

    def More(self) -> bool: ...

    def Next(self) -> None: ...

    def Current(self) -> TopoDS_Shape: ...
