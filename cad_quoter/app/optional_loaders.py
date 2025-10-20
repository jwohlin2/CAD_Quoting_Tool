"""Optional third-party integrations used by the quoting UI."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import typing

try:  # pragma: no cover - optional dependency
    import pandas as _pd  # type: ignore[import]
except Exception:  # pragma: no cover - pandas is optional
    _pd = None  # type: ignore[assignment]

pd = typing.cast(typing.Any, _pd)

try:  # pragma: no cover - optional DXF enrichment hook
    from geo_read_more import build_geo_from_dxf as _build_geo_from_dxf_path
except Exception:  # pragma: no cover - defensive fallback
    _build_geo_from_dxf_path = None  # type: ignore[assignment]

_build_geo_from_dxf_hook: Optional[Callable[[str], Dict[str, Any]]] = None


def build_geo_from_dxf(path: str) -> Dict[str, Any]:
    """Return auxiliary DXF metadata via the configured loader."""

    loader: Optional[Callable[[str], Dict[str, Any]]]
    loader = _build_geo_from_dxf_hook or _build_geo_from_dxf_path
    if loader is None:
        raise RuntimeError(
            "DXF metadata loader is unavailable; install geo_read_more or register a hook."
        )
    result = loader(path)
    if not isinstance(result, dict):
        raise TypeError("DXF metadata loader must return a dictionary")
    return result


def set_build_geo_from_dxf_hook(loader: Optional[Callable[[str], Dict[str, Any]]]) -> None:
    """Register a callable used by :func:`build_geo_from_dxf`."""

    if loader is not None and not callable(loader):
        raise TypeError("DXF metadata hook must be callable or ``None``")

    global _build_geo_from_dxf_hook
    _build_geo_from_dxf_hook = loader


__all__ = ["pd", "build_geo_from_dxf", "set_build_geo_from_dxf_hook"]
