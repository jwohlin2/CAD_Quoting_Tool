"""Development-time alias for the vendored :mod:`cad_quoter` package."""
from __future__ import annotations

from importlib import import_module, util as importlib_util
from pathlib import Path
from types import ModuleType
import sys

_TARGET = "cad_quoter_pkg.src.cad_quoter"


def _import_target() -> ModuleType:
    """Load the real :mod:`cad_quoter` package from the vendored source tree."""

    try:
        module = import_module(_TARGET)
    except Exception as exc:  # pragma: no cover - defensive bootstrap
        raise ModuleNotFoundError(
            "The cad_quoter package is not available. Install cad-quoter or keep "
            "cad_quoter_pkg/src on the import path."
        ) from exc
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


def _bootstrap_geometry_modules() -> None:
    """Load geometry helpers when the package installed stub fallbacks."""

    geometry_pkg = sys.modules.get("cad_quoter.geometry")
    if geometry_pkg is None:
        return
    if getattr(geometry_pkg, "__path__", None):
        return

    module_file = Path(getattr(_module, "__file__", ""))
    try:
        geometry_dir = module_file.resolve().parent / "geometry"
    except Exception:  # pragma: no cover - defensive guard
        return
    if not geometry_dir.is_dir():
        return

    for name in ("dxf_text", "dxf_enrich"):
        module_name = f"cad_quoter.geometry.{name}"
        if module_name in sys.modules:
            continue
        location = geometry_dir / f"{name}.py"
        if not location.is_file():
            continue
        spec = importlib_util.spec_from_file_location(module_name, location)
        if not spec or not spec.loader:
            continue
        module = importlib_util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - import-time failure propagates lazily
            sys.modules.pop(module_name, None)
            continue
        setattr(geometry_pkg, name, module)


_bootstrap_geometry_modules()
