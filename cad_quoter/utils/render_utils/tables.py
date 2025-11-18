"""Table rendering utilities (stubs for backwards compatibility)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_WIDTH = 80


@dataclass
class ColumnSpec:
    """Specification for a table column."""
    name: str
    width: int = 10
    align: str = "left"


def ascii_table(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[str] | None = None,
    columns: Sequence[ColumnSpec] | None = None,
    width: int = DEFAULT_WIDTH,
) -> str:
    """Render data as an ASCII table."""
    lines: list[str] = []
    if headers:
        lines.append(" | ".join(str(h) for h in headers))
        lines.append("-" * width)
    for row in rows:
        lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)


def draw_boxed_table(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[str] | None = None,
    width: int = DEFAULT_WIDTH,
) -> str:
    """Render data as a boxed ASCII table."""
    return ascii_table(rows, headers=headers, width=width)


def draw_kv_table(
    items: Sequence[tuple[str, Any]],
    *,
    width: int = DEFAULT_WIDTH,
) -> str:
    """Render key-value pairs as a table."""
    lines: list[str] = []
    for key, value in items:
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


__all__ = [
    "ColumnSpec",
    "DEFAULT_WIDTH",
    "ascii_table",
    "draw_boxed_table",
    "draw_kv_table",
]
