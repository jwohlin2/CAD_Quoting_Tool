"""Optional ezdxf / ODA File Converter bindings."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Final

log = logging.getLogger(__name__)

_DWG_CONVERTER_TIP: Final[str] = (
    "Set ODA_CONVERTER_EXE or DWG2DXF_EXE to a converter that accepts <input.dwg> <output.dxf>."
)

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

    candidates: list[Path] = []

    for env_var in ("ODA_CONVERTER_EXE", "DWG2DXF_EXE"):
        configured = os.environ.get(env_var)
        if configured:
            candidates.append(Path(configured))

    wrapper = Path(__file__).with_name("dwg2dxf_wrapper.bat")
    if wrapper.exists():
        candidates.append(wrapper)

    vendor_dir = Path(__file__).resolve().parent
    for name in ("ODAFileConverter.exe", "dwg2dxf.exe", "dwg2dxf.sh"):
        bundled = vendor_dir / name
        if bundled.exists():
            candidates.append(bundled)

    program_dirs: list[Path] = []
    for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        raw = os.environ.get(env_var)
        if raw:
            program_dirs.append(Path(raw))

    for base in program_dirs:
        oda_root = base / "ODA"
        for root in (base, oda_root):
            if not root.exists():
                continue
            for exe in sorted(root.glob("ODAFileConverter*/ODAFileConverter.exe"), reverse=True):
                candidates.append(exe)

    path_env = os.environ.get("PATH", "")
    if path_env:
        for part in path_env.split(os.pathsep):
            if not part:
                continue
            for name in ("ODAFileConverter.exe", "dwg2dxf.exe", "dwg2dxf"):
                exe = Path(part) / name
                if exe.exists():
                    candidates.append(exe)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _convert_dwg_to_dxf(path: Path, *, target_version: str = "ACAD2018") -> tuple[Path, bool]:
    """Convert ``path`` to DXF using the configured converter.

    Returns a tuple of ``(produced_path, supports_versions)`` where
    ``supports_versions`` indicates if the converter honours ``target_version``.
    """

    converter = _configured_dwg_converter()
    if converter is None:
        raise RuntimeError(
            "DWG support requires a DWG→DXF converter executable (e.g. ODAFileConverter.exe).\n"
            f"{_DWG_CONVERTER_TIP}"
        )

    out_dir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
    out_dxf = out_dir / (path.stem + ".dxf")

    exe_lower = converter.name.lower()
    supports_versions = False
    if exe_lower.endswith((".bat", ".cmd")):
        cmd = ["cmd", "/c", str(converter), str(path), str(out_dxf)]
    elif "odafileconverter" in exe_lower:
        cmd = [
            str(converter),
            str(path.parent),
            str(out_dir),
            target_version,
            "DXF",
            "0",
            "0",
            path.name,
        ]
        supports_versions = True
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

    if produced.suffix.lower() == ".zip" or _looks_like_zip(produced):
        produced = _extract_converted_zip(produced)

    return produced, supports_versions


def _looks_like_zip(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            signature = fh.read(4)
    except OSError:  # pragma: no cover - filesystem permissions
        return False
    return signature.startswith(b"PK\x03\x04")


def _extract_converted_zip(path: Path) -> Path:
    """Extract the primary DXF from ``path`` and return its location."""

    with zipfile.ZipFile(path) as archive:
        dxf_names = [name for name in archive.namelist() if name.lower().endswith(".dxf")]
        if not dxf_names:
            raise RuntimeError(
                "DWG converter produced a ZIP archive without a DXF payload.\n"
                f"archive: {path}"
            )
        # Prefer the entry that matches the archive stem, but fall back to the first DXF.
        stem = path.stem.lower()
        target_name = next((name for name in dxf_names if Path(name).stem.lower() == stem), dxf_names[0])
        extracted_path = path.with_suffix("") if path.suffix else path
        extracted_path = extracted_path.with_suffix(".dxf")
        with archive.open(target_name) as src, extracted_path.open("wb") as dst:
            data = src.read()
            dst.write(data)
    try:
        path.unlink()
    except OSError:  # pragma: no cover - cleanup best-effort
        pass
    return extracted_path


def _load_with_ezdxf(doc_path: Path) -> tuple[Any | None, Exception | None]:
    """Attempt to load ``doc_path`` with ezdxf, returning (doc, error)."""

    if _ezdxf_readfile is None:
        return None, None

    try:
        return _ezdxf_readfile(str(doc_path)), None
    except Exception as exc:  # pragma: no cover - depends on dxfs supplied
        recovered = _recover_document(str(doc_path), error=exc)
        if recovered is not None:
            return recovered, None
        return None, exc


def read_document(path: str | Path) -> Any:
    """Return an ezdxf drawing, falling back to odafc when available."""

    path_obj = Path(path)
    last_error: Exception | None = None

    if path_obj.suffix.lower() == ".dwg":
        conversion_versions: tuple[str, ...] = ("ACAD2018", "ACAD2013", "ACAD2010", "ACAD2007")
        supports_versions: bool | None = None
        attempted_versions: list[str] = []

        for idx, version in enumerate(conversion_versions):
            if supports_versions is False and idx > 0:
                break
            try:
                converted_path, supports_versions = _convert_dwg_to_dxf(path_obj, target_version=version)
            except Exception as exc:  # pragma: no cover - depends on env/converter
                last_error = exc
                if supports_versions is False:
                    break
                continue

            attempted_versions.append(version)
            doc, error = _load_with_ezdxf(converted_path)
            if doc is not None:
                return doc
            if error is not None:
                last_error = error
                continue

        if _odafc_readfile is not None:
            try:
                return _odafc_readfile(str(path_obj))
            except Exception as exc:  # pragma: no cover - depends on dwgs supplied
                last_error = exc
                recovered = _recover_document(str(path_obj), error=exc)
                if recovered is not None:
                    return recovered

        if last_error is not None:
            if attempted_versions:
                attempted = ", ".join(attempted_versions)
                raise RuntimeError(
                    "Failed to load DWG after converting with ODAFileConverter versions "
                    f"{attempted}. Last error: {last_error}"
                ) from last_error
            raise last_error

        if _odafc_readfile is None:
            raise RuntimeError(
                "DWG support is unavailable because ezdxf could not find a converter executable and the odafc backend is missing.\n"
                "Install the ODA File Converter and expose it via ODA_CONVERTER_EXE/DWG2DXF_EXE, or install ezdxf.addons.odafc.\n"
                f"{_DWG_CONVERTER_TIP}"
            )

        raise RuntimeError("DWG support failed without a specific error. Please retry the import.")

    doc, error = _load_with_ezdxf(path_obj)
    if doc is not None:
        return doc
    if error is not None:
        raise error

    raise RuntimeError("No DXF/DWG reader available. Install ezdxf or the ODA File Converter.")


__all__ = [
    "HAS_EZDXF",
    "EZDXF_VERSION",
    "HAS_ODAFC",
    "require_ezdxf",
    "require_odafc",
    "read_document",
]
