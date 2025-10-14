"""Core package for CAD Quoter tooling."""

from __future__ import annotations

import importlib
import sys


def _ensure_geometry_module() -> None:
    """Replace geometry stubs installed by tests with the real module when available."""

    module = sys.modules.get("cad_quoter.geometry")
    if module is not None and getattr(module, "__file__", None) is None:
        sys.modules.pop("cad_quoter.geometry", None)
        importlib.import_module("cad_quoter.geometry")


_ensure_geometry_module()

__all__ = [
    "_ensure_geometry_module",
]
