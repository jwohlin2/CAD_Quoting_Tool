"""Compatibility wrapper for the packaged McMaster-Carr helpers."""

from __future__ import annotations

from cad_quoter.vendors.mcmaster_stock import *  # noqa: F401,F403

if __name__ == "__main__":  # pragma: no cover - manual invocation
    from cad_quoter.vendors.mcmaster_stock import main

    main()
