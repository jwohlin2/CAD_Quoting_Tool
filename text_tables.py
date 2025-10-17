"""Utility helpers for rendering fixed-width ASCII tables.

All tables produced by this module are deterministic, rely on spaces for
alignment, and keep line widths well below the 114 character hard limit used
by downstream email clients.  The functions are intentionally tiny so other
renderers can build complex documents without having to worry about layout
math in every caller.
"""

from __future__ import annotations

import textwrap
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


def ascii_table(
    headers: Sequence[str] | None,
    rows: Sequence[Sequence[object]],
    *,
    col_widths: Sequence[int],
    col_aligns: Sequence[str] | None = None,
    header_aligns: Sequence[str] | None = None,
) -> str:
    """Render an ASCII table with optional headers and automatic wrapping."""

    column_count = len(col_widths)
    if column_count == 0:
        raise ValueError("at least one column is required")
    if any(width <= 0 for width in col_widths):
        raise ValueError("column widths must be positive")

    def _normalize_alignments(values: Sequence[str] | None, fallback: str) -> list[str]:
        result: list[str] = []
        for idx in range(column_count):
            raw = fallback
            if values and idx < len(values) and values[idx]:
                raw = str(values[idx])
            token = raw.strip().upper()[0] if raw else "L"
            result.append(token if token in {"L", "C", "R"} else "L")
        return result

    body_aligns = _normalize_alignments(col_aligns, "L")
    header_aligns = _normalize_alignments(header_aligns, "C")

    def _wrap_cell(value: object, width: int) -> list[str]:
        text = "" if value is None else str(value)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        segments = text.split("\n") or [""]
        for segment in segments:
            clean = segment.strip()
            wrapped = textwrap.wrap(
                clean,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
                drop_whitespace=False,
            )
            if not wrapped:
                lines.append("")
            else:
                lines.extend(wrapped)
        return lines or [""]

    def _pad(text: str, width: int, align: str) -> str:
        pad = max(width - len(text), 0)
        if align == "R":
            return " " * pad + text
        if align == "C":
            left = pad // 2
            right = pad - left
            return " " * left + text + " " * right
        return text + " " * pad

    def _render_row(cells: Sequence[object], aligns: Sequence[str]) -> list[str]:
        wrapped_cells = [_wrap_cell(cell, col_widths[idx]) for idx, cell in enumerate(cells)]
        height = max((len(cell_lines) for cell_lines in wrapped_cells), default=1)
        lines: list[str] = []
        for line_idx in range(height):
            pieces: list[str] = []
            for col_idx in range(column_count):
                cell_lines = wrapped_cells[col_idx]
                segment = cell_lines[line_idx] if line_idx < len(cell_lines) else ""
                pieces.append(_pad(segment, col_widths[col_idx], aligns[col_idx]))
            lines.append("|" + "|".join(pieces) + "|")
        return lines

    horizontal = "+" + "+".join("-" * width for width in col_widths) + "+"
    output: list[str] = [horizontal]

    if headers:
        if len(headers) != column_count:
            raise ValueError("header count must match column specification")
        output.extend(_render_row(headers, header_aligns))
        output.append(horizontal)

    for row in rows:
        if len(row) != column_count:
            raise ValueError("row does not match column specification")
        output.extend(_render_row(row, body_aligns))

    output.append(horizontal)
    return "\n".join(output)


__all__ = [
    "ColumnSpec",
    "DEFAULT_WIDTH",
    "ascii_table",
    "draw_boxed_table",
    "draw_kv_table",
    "ellipsize",
    "money",
    "pct",
]

