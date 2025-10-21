"""Helper package initializer for local development.

This repository keeps the main ``cad_quoter`` sources within
``cad_quoter_pkg/src`` so that they can be packaged for distribution.
When developing directly from the repo (for example when running
``appV5.py`` or ``pytest``) we still want ``import cad_quoter`` to work
without needing to install the package.  We therefore extend the
package search path to include the packaged source directory so that
``cad_quoter.ui`` (which lives next to this file) continues to be
available alongside the rest of the modules.
"""

from __future__ import annotations

from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_PKG_SRC = _ROOT / "cad_quoter_pkg" / "src" / "cad_quoter"

_extended_path = [str(_THIS_DIR)]
if _PKG_SRC.exists():
    _extended_path.append(str(_PKG_SRC))

# ``__path__`` controls which directories Python searches when looking for
# ``cad_quoter`` submodules.  By including the packaged sources we make the
# rest of the project importable without installing the distribution.
__path__ = _extended_path  # type: ignore[name-defined]

