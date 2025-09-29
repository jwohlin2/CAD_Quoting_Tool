"""Geometry ingest helpers for the CAD Quoter application.

The real project performs sophisticated CAD parsing.  For the purposes of
this refactor we expose a small service with a clear API so the UI can be
wired up without bundling all of the heavy logic into ``appV5.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeometrySummary:
    """Light-weight representation of extracted geometry metrics."""

    source_path: Path
    notes: str

    def as_display_text(self) -> str:
        """Return a human-readable summary suitable for the UI."""

        return f"Loaded geometry from {self.source_path.name}\n{self.notes}"


class GeometryService:
    """Facade around geometry loading/parsing for the UI."""

    def load_geometry(self, source: Path) -> GeometrySummary:
        """Load geometry from ``source`` returning a friendly summary."""

        if not source.exists():
            raise FileNotFoundError(source)
        notes = "Geometry inspection placeholder – detailed parsing occurs in dedicated modules."
        return GeometrySummary(source_path=source, notes=notes)

    def try_load(self, source: str | Path) -> Optional[GeometrySummary]:
        """Attempt to load geometry, swallowing ``FileNotFoundError`` for the UI."""

        path = Path(source)
        try:
            return self.load_geometry(path)
        except FileNotFoundError:
            return None


__all__ = ["GeometryService", "GeometrySummary"]
