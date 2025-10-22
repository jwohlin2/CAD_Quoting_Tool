"""UI helper shims for the lightweight appkit namespace."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["editor_controls"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f"appkit.ui.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
