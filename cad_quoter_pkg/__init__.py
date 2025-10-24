"""Helper package enabling direct imports from ``cad_quoter_pkg`` during type checking."""
from __future__ import annotations

from pathlib import Path
import sys

_src_dir = Path(__file__).resolve().parent.parent / "src"
_src_text = str(_src_dir)
if _src_dir.is_dir() and _src_text not in sys.path:
    sys.path.append(_src_text)

__all__: list[str] = []
