"""Compatibility shim re-exporting :mod:`cad_quoter` under ``cad_quoter_pkg.src``."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from types import ModuleType
from pathlib import Path

_TARGET = "cad_quoter"
_ALIAS = __name__


def _load_target() -> ModuleType:
    spec = importlib.machinery.PathFinder.find_spec(
        _TARGET,
        [str(Path(__file__).resolve().parents[3] / "src")],
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            "The cad_quoter package is not available. Install cad-quoter or keep src on the import path."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[_ALIAS] = module
    sys.modules.setdefault(_TARGET, module)
    spec.loader.exec_module(module)
    module.__name__ = _TARGET
    module.__package__ = _TARGET

    try:
        repo_root = Path(__file__).resolve().parents[3]
    except Exception:  # pragma: no cover - defensive bootstrap
        extra_paths: tuple[Path, ...] = ()
    else:
        candidate = repo_root / "src"
        extra_paths = (candidate,) if candidate.is_dir() else ()

    ensure_geometry = getattr(module, "ensure_geometry_module", None)
    if callable(ensure_geometry):
        try:
            ensure_geometry(extra_search_paths=extra_paths)
        except Exception:  # pragma: no cover - import should remain best-effort
            pass
    return module


def __getattr__(name: str):  # pragma: no cover - defers to the real package
    module = _load_target()
    return getattr(module, name)


def __dir__() -> list[str]:  # pragma: no cover - mirrors real package metadata
    module = _load_target()
    return sorted(set(dir(module)))


_module = _load_target()
