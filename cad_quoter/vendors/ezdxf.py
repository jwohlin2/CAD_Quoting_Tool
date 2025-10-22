from __future__ import annotations

"""Optional ezdxf / ODA File Converter bindings."""

from pathlib import Path
from typing import Any, Callable, Final

_EZDXF_ERROR: Exception | None = None
_ODAFC_ERROR: Exception | None = None
_ezdxf: Any | None
_odafc: Any | None = None

try:  # pragma: no cover - exercised indirectly
    import ezdxf as _ezdxf  # type: ignore
except Exception as exc:  # pragma: no cover - platform specific
    _ezdxf = None  # type: ignore[assignment]
    _EZDXF_ERROR = exc
else:
    _EZDXF_ERROR = None
    try:  # pragma: no cover - optional dependency
        from ezdxf.addons import odafc as _odafc  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        _odafc = None  # type: ignore[assignment]
        _ODAFC_ERROR = exc
    else:
        _ODAFC_ERROR = None

EZDXF_VERSION: Final[str] = getattr(_ezdxf, "__version__", "unknown") if _ezdxf else "unknown"
HAS_EZDXF: Final[bool] = _ezdxf is not None
HAS_ODAFC: Final[bool] = "_odafc" in globals() and _odafc is not None

_ezdxf_readfile: Callable[[str], Any] | None = None
if HAS_EZDXF:
    _ezdxf_readfile = getattr(_ezdxf, "readfile", None)

_odafc_readfile: Callable[[str], Any] | None = None
if HAS_ODAFC:
    _odafc_readfile = getattr(_odafc, "readfile", None)


def require_ezdxf(action: str = "DXF operations") -> Any:
    """Return the ``ezdxf`` module or raise a friendly error."""

    if _ezdxf is None:
        msg = f"{action} requires ezdxf, which is unavailable."
        if _EZDXF_ERROR is not None:
            raise RuntimeError(f"{msg} ({_EZDXF_ERROR})") from _EZDXF_ERROR
        raise RuntimeError(msg)
    return _ezdxf


def require_odafc(action: str = "DWG conversion") -> Any:
    """Return the ``ezdxf.addons.odafc`` module or raise a friendly error."""

    if not HAS_ODAFC or _odafc_readfile is None:
        if _ODAFC_ERROR is not None:
            raise RuntimeError(
                f"{action} requires ezdxf.addons.odafc (ODA File Converter)."
                f" It failed to load: {_ODAFC_ERROR}"
            ) from _ODAFC_ERROR
        raise RuntimeError(
            f"{action} requires ezdxf.addons.odafc with the ODA File Converter binaries installed."
        )
    return _odafc


def read_document(path: str | Path) -> Any:
    """Return an ezdxf drawing, falling back to odafc when available."""

    doc_path = str(path)
    if _ezdxf_readfile is not None:
        return _ezdxf_readfile(doc_path)
    if _odafc_readfile is not None:
        return _odafc_readfile(doc_path)
    raise RuntimeError("No DXF/DWG reader available. Install ezdxf or the ODA File Converter.")


__all__ = [
    "HAS_EZDXF",
    "EZDXF_VERSION",
    "HAS_ODAFC",
    "require_ezdxf",
    "require_odafc",
    "read_document",
]
