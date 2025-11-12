"""Root-level test configuration ensuring optional stubs load for standalone tests."""

from __future__ import annotations

import importlib
import sys
import types
from importlib.machinery import ModuleSpec


def pytest_configure() -> None:
    """Install lightweight stubs for optional dependencies before collection."""

    if "requests" not in sys.modules:
        try:  # pragma: no cover - best effort import
            importlib.import_module("requests")
        except ModuleNotFoundError:
            stub = types.ModuleType("requests")
            stub.__spec__ = ModuleSpec("requests", loader=None)
            sys.modules["requests"] = stub
