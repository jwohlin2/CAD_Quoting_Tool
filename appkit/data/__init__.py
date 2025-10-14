"""Helpers for loading static data files bundled with the appkit package."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _read_text(name: str) -> str:
    path = _DATA_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def load_text(name: str) -> str:
    """Return the contents of ``name`` from the data directory."""

    return _read_text(name)


def load_json(name: str) -> Any:
    """Parse ``name`` (relative to the data directory) as JSON."""

    return json.loads(_read_text(name))


__all__ = ["load_json", "load_text"]
