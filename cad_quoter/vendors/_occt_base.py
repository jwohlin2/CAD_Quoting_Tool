from __future__ import annotations

"""Backend detection helpers for OCCT shims."""

import importlib
from typing import Any, Final

_MISSING_MSG = (
    "OCCT bindings are required for geometry operations. "
    "Install pythonocc-core or the OCP wheels."
)

PREFIX: Final[str | None]
STACK: Final[str]
BACKEND: Final[str]
_IMPORT_ERROR: Exception | None = None

_prefix = None
_stack = "missing"
for candidate, label in (("OCP", "ocp"), ("OCC.Core", "pythonocc")):
    try:
        importlib.import_module(f"{candidate}.TopoDS")
    except Exception as exc:  # pragma: no cover - environment dependent
        _IMPORT_ERROR = exc
        continue
    _prefix = candidate
    _stack = label
    break

PREFIX = _prefix
STACK = BACKEND = _stack if _prefix else "missing"


def load_module(name: str) -> Any:
    if PREFIX is None:
        raise ImportError(_MISSING_MSG) from _IMPORT_ERROR
    return importlib.import_module(f"{PREFIX}.{name}")


def get_symbol(module: str, *candidates: str) -> Any:
    mod = load_module(module)
    for attr in candidates:
        if hasattr(mod, attr):
            return getattr(mod, attr)
    raise AttributeError(f"{module} lacks any of {candidates}")


def missing(name: str) -> Any:
    raise ImportError(f"{name} requires OCCT bindings. {_MISSING_MSG}")


__all__ = ["PREFIX", "STACK", "BACKEND", "load_module", "get_symbol", "missing"]
