"""Helper package enabling direct imports from ``cad_quoter_pkg`` during type checking."""
from __future__ import annotations

from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
_repo_text = str(_repo_root)
if _repo_root.is_dir() and _repo_text not in sys.path:
    sys.path.append(_repo_text)

__all__: list[str] = []
