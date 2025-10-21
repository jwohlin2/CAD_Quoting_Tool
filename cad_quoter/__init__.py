"""Convenience namespace for local CAD Quoter modules.

This repository ships two different locations that contribute modules to the
``cad_quoter`` namespace:

* ``cad_quoter_pkg/src/cad_quoter`` contains the bulk of the core business
  logic that is distributed as a standalone package.
* ``cad_quoter/ui`` holds Tkinter UI helpers that are used only when running
  ``appV5.py`` directly from the repository.

Python (and tools such as Pylance/Pyright) normally discover packages by
walking ``sys.path``.  When the repository is used without installing the
``cad_quoter`` package, the core modules live outside of the default search
path, which results in missing-import diagnostics.  This module stitches the
two locations together into a single package and eagerly loads the packaged
``cad_quoter`` module so that its side effects (e.g. geometry fallbacks) run
as expected.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

__all__: list[str] = []

_PACKAGE_ROOT = Path(__file__).resolve().parent
__path__ = [str(_PACKAGE_ROOT)]

_CORE_MODULE: ModuleType | None = None


def _iter_core_locations() -> Iterable[Path]:
    """Yield candidate directories that may contain the packaged modules."""

    repo_root = _PACKAGE_ROOT.parent
    packaged = repo_root / "cad_quoter_pkg" / "src" / "cad_quoter"
    if packaged.is_dir():
        yield packaged


def _load_core_package() -> ModuleType | None:
    """Load the packaged ``cad_quoter`` module under a private alias."""

    global _CORE_MODULE
    if _CORE_MODULE is not None:
        return _CORE_MODULE

    for core_path in _iter_core_locations():
        if str(core_path) not in __path__:
            __path__.append(str(core_path))

        init_file = core_path / "__init__.py"
        if not init_file.is_file():
            continue

        spec = importlib.util.spec_from_file_location(
            f"{__name__}._core", init_file
        )
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        _CORE_MODULE = module

        core_all = getattr(module, "__all__", None)
        if core_all:
            __all__.extend(item for item in core_all if item not in __all__)

        return module

    return None


_load_core_package()


def __getattr__(name: str) -> ModuleType | object:
    """Provide attribute access to symbols defined in the packaged module."""

    module = _CORE_MODULE or _load_core_package()
    if module is not None and hasattr(module, name):
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
