"""Module entry-point for ``python -m deploy``."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":  # pragma: no cover - invoked via module execution
    raise SystemExit(main())
