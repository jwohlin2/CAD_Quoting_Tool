"""Thin compatibility package for legacy appkit imports."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["ui"]


def __getattr__(name: str) -> ModuleType:
    if name == "ui":
        module = import_module("appkit.ui")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
