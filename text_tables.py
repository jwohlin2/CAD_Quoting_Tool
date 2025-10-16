"""Utility helpers for rendering fixed-width ASCII tables.

All tables produced by this module are deterministic, rely on spaces for
alignment, and keep line widths well below the 114 character hard limit used
by downstream email clients.  The functions are intentionally tiny so other
renderers can build complex documents without having to worry about layout
math in every caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

ELLIPSIS = "\u2026"
DEFAULT_WIDTH = 114


@dataclass(frozen=True)
class ColumnSpec:
    """Describe a table column.

    Attributes
    ----------
    width:
        Number of characters allocated for the column content (not including
        the surrounding pipes).
    align:
        Default alignment for body rows – "L", "C", or "R".
    header_align:
        Optional override for header alignment.  When unset the header uses the
        same alignment as the body rows.
    """

    width: int
    align: str = "L"
    header_align: str | None = None


def money(value: float | int | None, currency: str = "$") -> str:
    """Format ``value`` as a money string with thousands separators."""

    if value is None:
        return "—"
    number = float(value)
    sign = "-" if number < 0 else ""
    magnitude = abs(number)
    return f"{sign}{currency}{magnitude:,.2f}"


def pct(value: float | int | None) -> str:
    """Format a ratio (0.0–1.0) or percentage (0–100) with one decimal place."""

    if value is None:
        return "—"
    number = float(value)
    if abs(number) > 1.0:
        number /= 100.0
    return f"{number * 100.0:.1f}%"


def ellipsize(text: str, width: int) -> str:
    """Clamp ``text`` to ``width`` characters using a single ellipsis if needed."""

    if width <= 0:
        return ""
    clean = text if isinstance(text, str) else str(text)
    if len(clean) <= width:
        return clean
    if width == 1:
        return ELLIPSIS
    return f"{clean[: width - 1]}{ELLIPSIS}"


def _coerce_alignment(value: str) -> str:
    upper = (value or "L").upper()
    if upper not in {"L", "C", "R"}:
        upper = "L"
    return upper


def _pad(text: str, width: int, align: str) -> str:
    truncated = ellipsize(text, width)
    pad = max(width - len(truncated), 0)
    if align == "R":
        return " " * pad + truncated
    if align == "C":
        left = pad // 2
        right = pad - left
        return " " * left + truncated + " " * right
    return truncated + " " * pad


def draw_boxed_table(
    headers: Sequence[str] | None,
    rows: Sequence[Sequence[str]],
    colspecs: Sequence[ColumnSpec],
) -> str:
    """Render a fixed-width ASCII table with box-drawing borders."""

    if any(spec.width <= 0 for spec in colspecs):
        raise ValueError("column widths must be positive")
    column_count = len(colspecs)
    if headers and len(headers) != column_count:
        raise ValueError("header count must match column specification")
    for row in rows:
        if len(row) != column_count:
            raise ValueError("row does not match column specification")

    horizontal = "+" + "+".join("-" * spec.width for spec in colspecs) + "+"

    def _render_row(cells: Sequence[str], *, header: bool = False) -> str:
        formatted: list[str] = []
        for idx, cell in enumerate(cells):
            spec = colspecs[idx]
            align = spec.header_align if header and spec.header_align else spec.align
            align = _coerce_alignment(align)
            formatted.append(_pad(str(cell), spec.width, align))
        return "|" + "|".join(formatted) + "|"

    output: list[str] = [horizontal]
    if headers:
        output.append(_render_row(headers, header=True))
        output.append(horizontal)
    for row in rows:
        output.append(_render_row(row))
    output.append(horizontal)
    return "\n".join(output)


def draw_kv_table(
    pairs: Iterable[tuple[str, str]],
    left_width: int,
    right_width: int,
    *,
    left_align: str = "L",
    right_align: str = "R",
) -> str:
    """Convenience wrapper for two-column key/value tables."""

    colspecs = (
        ColumnSpec(left_width, left_align),
        ColumnSpec(right_width, right_align),
    )
    rows = list(pairs)
    return draw_boxed_table(None, rows, colspecs)


__all__ = [
    "ColumnSpec",
    "DEFAULT_WIDTH",
    "draw_boxed_table",
    "draw_kv_table",
    "ellipsize",
    "money",
    "pct",
]

