"""Legacy wrappers that are no longer part of the supported surface area.

This module is intentionally not imported anywhere in the application.  The
helpers are preserved here solely for external scripts that might still import
symbols from :mod:`appV5` which have since been removed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cad_quoter.app import runtime as _runtime

try:  # pragma: no cover - optional OCC geometry stack may be unavailable
    import cad_quoter.geometry as _geometry
except Exception:  # pragma: no cover - geometry is optional in tests
    _geometry = None  # type: ignore[assignment]

# Backwards compatibility constants mirrored from the previous wrappers
_DEFAULT_VL_MODEL_NAMES = _runtime.DEFAULT_VL_MODEL_NAMES
_DEFAULT_MM_PROJ_NAMES = _runtime.DEFAULT_MM_PROJ_NAMES
VL_MODEL = str(_runtime.LEGACY_VL_MODEL)
MM_PROJ = str(_runtime.LEGACY_MM_PROJ)
LEGACY_VL_MODEL = str(_runtime.LEGACY_VL_MODEL)
LEGACY_MM_PROJ = str(_runtime.LEGACY_MM_PROJ)
PREFERRED_MODEL_DIRS: list[str | Path] = [str(p) for p in _runtime.PREFERRED_MODEL_DIRS]


def _geometry_helper(name: str) -> Any:
    if _geometry is None or not hasattr(_geometry, name):
        raise RuntimeError(f"geometry helper '{name}' is unavailable in this environment")
    return getattr(_geometry, name)


def discover_qwen_vl_assets(
    *, model_path: str | None = None, mmproj_path: str | None = None
) -> tuple[str, str]:
    """Legacy wrapper for :func:`cad_quoter.app.runtime.discover_qwen_vl_assets`."""

    preferred: list[Path] = []
    for value in PREFERRED_MODEL_DIRS:
        try:
            preferred.append(Path(value).expanduser())
        except Exception:
            continue

    original_preferred = getattr(_runtime, "PREFERRED_MODEL_DIRS", ())
    try:
        if preferred:
            _runtime.PREFERRED_MODEL_DIRS = tuple(preferred)
        return _runtime.discover_qwen_vl_assets(
            model_path=model_path,
            mmproj_path=mmproj_path,
        )
    finally:
        try:
            _runtime.PREFERRED_MODEL_DIRS = original_preferred
        except Exception:
            pass


def read_step_or_iges_or_brep(path: str) -> Any:
    """Delegate to :mod:`cad_quoter.geometry` for STEP/IGES/BREP import."""

    reader = _geometry_helper("read_step_or_iges_or_brep")
    return reader(path)


def explode_compound(shape: Any) -> list[Any]:
    """Delegate to :mod:`cad_quoter.geometry` to explode compound shapes."""

    helper = _geometry_helper("explode_compound")
    result = helper(shape)
    # The historical wrapper always returned a list; normalise here as well.
    if isinstance(result, list):
        return result
    return list(result)
