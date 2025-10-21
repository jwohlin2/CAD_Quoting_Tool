"""Local aggregate package for development without installation."""

from __future__ import annotations

from pathlib import Path

_package_root = Path(__file__).resolve().parent
__path__ = [str(_package_root)]

_extra_src = _package_root.parent / "cad_quoter_pkg" / "src" / "cad_quoter"
if _extra_src.exists():
    extra_src_text = str(_extra_src)
    if extra_src_text not in __path__:
        __path__.append(extra_src_text)

__all__ = []
