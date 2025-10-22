"""Optional ezdxf / ODA File Converter bindings."""

import logging
from pathlib import Path
from typing import Any, Callable, Final

log = logging.getLogger(__name__)

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
_ezdxf_recover_readfile: Callable[[str], Any] | None = None
if HAS_EZDXF:
    _ezdxf_readfile = getattr(_ezdxf, "readfile", None)
    recover_mod = getattr(_ezdxf, "recover", None)
    if recover_mod is not None:
        _ezdxf_recover_readfile = getattr(recover_mod, "readfile", None)

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


def _recover_document(doc_path: str, *, error: Exception) -> Any | None:
    """Attempt to recover a DXF document after ``error``."""

    if _ezdxf_recover_readfile is None:
        return None
    try:
        result = _ezdxf_recover_readfile(doc_path)
    except Exception as recover_exc:  # pragma: no cover - depends on dxfs supplied
        log.debug("ezdxf recover.readfile failed", exc_info=recover_exc)
        return None

    # ezdxf.recover.readfile returns (doc, auditor)
    if isinstance(result, tuple) and len(result) == 2:
        doc, auditor = result
        if getattr(auditor, "has_errors", False):  # pragma: no cover - diagnostic only
            log.warning(
                "Recovered DXF %s with auditor errors after %s: %s",
                doc_path,
                type(error).__name__,
                auditor.errors,
            )
        else:
            log.info(
                "Recovered DXF %s after %s", doc_path, type(error).__name__
            )
        return doc

    return result


def read_document(path: str | Path) -> Any:
    """Return an ezdxf drawing, falling back to odafc when available."""

    doc_path = str(path)
    last_error: Exception | None = None
    if _ezdxf_readfile is not None:
        try:
            return _ezdxf_readfile(doc_path)
        except Exception as exc:  # pragma: no cover - depends on dxfs supplied
            last_error = exc
            recovered = _recover_document(doc_path, error=exc)
            if recovered is not None:
                return recovered
            raise
    if _odafc_readfile is not None:
        try:
            return _odafc_readfile(doc_path)
        except Exception as exc:  # pragma: no cover - depends on dwgs supplied
            last_error = exc
            recovered = _recover_document(doc_path, error=exc)
            if recovered is not None:
                return recovered
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(
        "No DXF/DWG reader available. Install ezdxf or the ODA File Converter."
    )


__all__ = [
    "HAS_EZDXF",
    "EZDXF_VERSION",
    "HAS_ODAFC",
    "require_ezdxf",
    "require_odafc",
    "read_document",
]
