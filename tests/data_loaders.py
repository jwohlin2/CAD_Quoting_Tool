from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _read_text(name: str) -> str:
    path = _DATA_DIR / name
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_geometry_samples() -> dict[str, Any]:
    """Return cached geometry sample data loaded from JSON."""

    return json.loads(_read_text("geometry_samples.json"))


@lru_cache(maxsize=None)
def load_speeds_feeds_samples() -> tuple[dict[str, str], ...]:
    """Return cached speeds/feeds samples parsed from CSV rows."""

    path = _DATA_DIR / "speeds_feeds_samples.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader: Iterable[dict[str, str]] = csv.DictReader(handle)
        return tuple(dict(row) for row in reader)


__all__ = ["load_geometry_samples", "load_speeds_feeds_samples"]
