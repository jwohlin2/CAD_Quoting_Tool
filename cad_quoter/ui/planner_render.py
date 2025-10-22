"""Runtime shim exposing the packaged planner render helpers."""
from __future__ import annotations

from cad_quoter_pkg.src.cad_quoter.ui import planner_render as _planner_render_impl

globals().update(
    {
        name: getattr(_planner_render_impl, name)
        for name in dir(_planner_render_impl)
        if not name.startswith("__")
    }
)

__all__ = getattr(
    _planner_render_impl,
    "__all__",
    [name for name in globals().keys() if not name.startswith("__")],
)

del _planner_render_impl
