"""Core package for CAD Quoter tooling."""

from __future__ import annotations

from ._bootstrap import ensure_geometry_module

ensure_geometry_module()

__all__ = [
    "ensure_geometry_module",
]
