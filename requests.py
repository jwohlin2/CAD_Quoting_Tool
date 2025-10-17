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

_ALLOW_REAL = os.environ.get("CAD_QUOTER_ALLOW_REQUESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if _ALLOW_REAL:
    _real_requests = importlib.import_module("requests")
    globals().update({name: getattr(_real_requests, name) for name in dir(_real_requests) if not name.startswith("__")})
else:

    class Session:
        def __init__(self, *args, **kwargs):  # noqa: D401 - intentionally empty stub
            pass

        def post(self, *args, **kwargs):
            raise RuntimeError("requests stub: network operations are unavailable in tests")

        def put(self, *args, **kwargs):
            raise RuntimeError("requests stub: network operations are unavailable in tests")

        def mount(self, *args, **kwargs):
            return None

    def __getattr__(name: str) -> Any:
        raise RuntimeError(f"requests stub does not implement '{name}'")
