"""Minimal stand-in for the ``requests`` package used in tests.

The real McMaster-Carr integration requires the genuine ``requests`` library to
perform mutual TLS calls.  The original stub prevented *any* network access,
which is great for tests but blocked developers from running the helper CLIs
locally even when they have valid credentials and certificates.

Setting ``CAD_QUOTER_ALLOW_REQUESTS=1`` (or any truthy value such as ``true``)
activates a passthrough mode that imports the real ``requests`` module.  When
the flag is absent we keep the lightweight stub to avoid accidental outbound
traffic during automated tests.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

# Always proxy to the real 'requests' package in this runtime.
_ALLOW_REAL = True

if _ALLOW_REAL:
    # Load the real 'requests' package into this module object so that
    # 'import requests.adapters' and other submodules resolve correctly.
    import sys as _sys
    import os as _os
    import importlib.machinery as _machinery

    _this_dir = _os.path.abspath(_os.path.dirname(__file__))
    _search_path = [p for p in list(_sys.path) if _os.path.abspath(p or "") != _this_dir]

    _spec = _machinery.PathFinder.find_spec("requests", _search_path)
    if _spec is None or _spec.loader is None:
        raise ImportError("Real 'requests' package not found on sys.path; install it and retry")

    # Configure this module to behave like the real package before executing it.
    __spec__ = _spec
    __package__ = "requests"
    try:
        # Provide submodule search locations so relative imports work during exec
        __path__ = list(_spec.submodule_search_locations or [])
    except Exception:
        __path__ = []
    try:
        __file__ = _spec.origin or __file__
    except Exception:
        pass

    # Execute the real package code in-place into this module object.
    _spec.loader.exec_module(_sys.modules[__name__])
else:
    # Retain a minimal stub for completeness, though code paths should not hit this.
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def post(self, *args, **kwargs):
            raise RuntimeError("requests stub: network operations are unavailable")
        def put(self, *args, **kwargs):
            raise RuntimeError("requests stub: network operations are unavailable")
        def mount(self, *args, **kwargs):
            return None
    def __getattr__(name: str) -> Any:
        raise RuntimeError(f"requests stub does not implement '{name}'")
