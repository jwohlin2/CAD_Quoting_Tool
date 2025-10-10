from __future__ import annotations

from enum import IntEnum
from typing import Any as _Any


class TopAbs_ShapeEnum(IntEnum):
    ...


TopAbs_EDGE: TopAbs_ShapeEnum
TopAbs_FACE: TopAbs_ShapeEnum
TopAbs_SOLID: TopAbs_ShapeEnum
TopAbs_SHELL: TopAbs_ShapeEnum
TopAbs_COMPOUND: TopAbs_ShapeEnum

# Backwards compatibility – some code still expects `_Any` fallbacks.
TopAbs_ShapeEnum_like = TopAbs_ShapeEnum | _Any
