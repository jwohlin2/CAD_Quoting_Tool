"""Development-time alias for the vendored :mod:`cad_quoter` package."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
import sys

_TARGET = "cad_quoter"


def _ensure_src_on_path() -> Path | None:
    try:
        repo_root = Path(__file__).resolve().parent.parent
    except Exception:  # pragma: no cover - defensive bootstrap
        return None

    src_root = repo_root / "src"
    if not src_root.is_dir():
        return None

    src_text = str(src_root)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return src_root


def _import_target() -> ModuleType:
    """Load the real :mod:`cad_quoter` package from the vendored source tree."""

    previous = sys.modules.pop(__name__, None)
    try:
        module = import_module(_TARGET)
    except Exception as exc:  # pragma: no cover - defensive bootstrap
        src_root = _ensure_src_on_path()
        if src_root is not None:
            try:
                module = import_module(_TARGET)
            except Exception:
                pass
            else:
                return module
        raise ModuleNotFoundError(
            "The cad_quoter package is not available. Install cad-quoter or keep src "
            "on the import path."
        ) from exc
    finally:
        if __name__ not in sys.modules and previous is not None:
            sys.modules[__name__] = previous
    return module


_module = _import_target()
_module.__name__ = __name__
_module.__package__ = __name__

# Re-export the imported package so consumers see the real module object.
sys.modules[__name__] = _module

# Populate the current module namespace with the real module's attributes for
# introspection tools that read directly from ``globals()``.
_globals = globals()
for _key, _value in _module.__dict__.items():
    if _key.startswith("__") and _key not in {"__all__", "__path__"}:
        continue
    _globals.setdefault(_key, _value)

__all__ = getattr(_module, "__all__", [])
__doc__ = _module.__doc__
__loader__ = getattr(_module, "__loader__", None)
__path__ = getattr(_module, "__path__", [])
__package__ = __name__
__spec__ = getattr(_module, "__spec__", None)
if __spec__ is not None:
    try:
        __spec__.name = __name__
    except Exception:  # pragma: no cover - importlib implementation detail
        pass
__file__ = getattr(_module, "__file__", None)


try:
    _repo_root = Path(__file__).resolve().parent.parent
except Exception:  # pragma: no cover - defensive guard for exotic environments
    _EXTRA_PATHS: tuple[Path, ...] = ()
else:
    candidate = _repo_root / "src"
    _EXTRA_PATHS = (candidate,) if candidate.is_dir() else ()

try:
    _ensure_geometry_module = _module.ensure_geometry_module
except AttributeError:  # pragma: no cover - vendored module missing helper
    _ensure_geometry_module = None

if callable(_ensure_geometry_module):
    try:
        _ensure_geometry_module(extra_search_paths=_EXTRA_PATHS)
    except Exception:  # pragma: no cover - bootstrap should never block import
        pass
