"""Development shim that exposes the vendored :mod:`cad_quoter` package."""

from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path
from pkgutil import extend_path
from types import ModuleType
import sys

__path__ = extend_path(__path__, __name__)


def _load_src_package() -> ModuleType | None:
    """Load the real :mod:`cad_quoter` package from the repository."""

    try:
        repo_root = Path(__file__).resolve().parent.parent
    except Exception:  # pragma: no cover - defensive bootstrap
        return None

    package_root = repo_root / "src" / "cad_quoter"
    init_file = package_root / "__init__.py"
    if not init_file.is_file():
        return None

    spec = importlib_util.spec_from_file_location(
        __name__,
        init_file,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib_util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_module = _load_src_package()
if _module is not None:
    globals().update(
        {
            key: value
            for key, value in _module.__dict__.items()
            if not (key.startswith("__") and key not in {"__all__", "__doc__", "__path__"})
        }
    )
    __all__ = getattr(_module, "__all__", [])
    __doc__ = _module.__doc__
    __path__ = getattr(_module, "__path__", __path__)
