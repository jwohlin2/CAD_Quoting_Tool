"""Optional ezdxf / ODA File Converter bindings."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
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


def _configured_dwg_converter() -> Path | None:
    """Return the configured DWG→DXF converter if available."""

    exe = (
        os.environ.get("ODA_CONVERTER_EXE")
        or os.environ.get("DWG2DXF_EXE")
        or str(Path(__file__).with_name("dwg2dxf_wrapper.bat"))
        or r"D:\CAD_Quoting_Tool\dwg2dxf_wrapper.bat"
    )
    if not exe:
        return None
    path = Path(exe)
    if not path.exists():
        return None
    return path


def _convert_dwg_to_dxf(path: Path) -> Path | None:
    """Convert ``path`` to DXF using the configured converter."""

    converter = _configured_dwg_converter()
    if converter is None:
        return None

    out_dir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
    out_dxf = out_dir / (path.stem + ".dxf")

    exe_lower = converter.name.lower()
    if exe_lower.endswith((".bat", ".cmd")):
        cmd = ["cmd", "/c", str(converter), str(path), str(out_dxf)]
    elif "odafileconverter" in exe_lower:
        cmd = [
            str(converter),
            str(path.parent),
            str(out_dir),
            "ACAD2018",
            "DXF",
            "0",
            "0",
            path.name,
        ]
    else:
        cmd = [str(converter), str(path), str(out_dxf)]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - user environment
        raise RuntimeError(f"DWG converter not found: {converter}") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - converter runtime
        raise RuntimeError(
            "DWG→DXF conversion failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    produced = out_dxf if out_dxf.exists() else out_dir / (path.stem + ".dxf")
    if not produced.exists():  # pragma: no cover - unexpected converter behaviour
        raise RuntimeError(
            "Converter reported success but DXF was not produced.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"checked: {out_dxf} | {produced}"
        )
    return produced


def read_document(path: str | Path) -> Any:
    """Return an ezdxf drawing, falling back to odafc when available."""

    path_obj = Path(path)
    original_dwg: Path | None = None
    converted_path: Path | None = None
    last_error: Exception | None = None

    if path_obj.suffix.lower() == ".dwg":
        original_dwg = path_obj
        try:
            converted_path = _convert_dwg_to_dxf(path_obj)
        except Exception as exc:  # pragma: no cover - depends on env/converter
            last_error = exc

    candidate_paths: list[Path] = []
    if converted_path is not None:
        candidate_paths.append(converted_path)
    if original_dwg is None:
        candidate_paths.append(path_obj)
    elif converted_path is None and _odafc_readfile is None:
        # If we cannot convert and do not have odafc, still attempt ezdxf for diagnostic error
        candidate_paths.append(path_obj)

    if _ezdxf_readfile is not None:
        for candidate in candidate_paths:
            doc_path = str(candidate)
            try:
                return _ezdxf_readfile(doc_path)
            except Exception as exc:  # pragma: no cover - depends on dxfs supplied
                last_error = exc
                recovered = _recover_document(doc_path, error=exc)
                if recovered is not None:
                    return recovered

    if original_dwg is not None and _odafc_readfile is not None:
        doc_path = str(original_dwg)
        try:
            return _odafc_readfile(doc_path)
        except Exception as exc:  # pragma: no cover - depends on dwgs supplied
            last_error = exc
            recovered = _recover_document(doc_path, error=exc)
            if recovered is not None:
                return recovered

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
