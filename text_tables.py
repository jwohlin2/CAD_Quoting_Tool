"""Compatibility shim for legacy imports of the ASCII table utilities."""

from __future__ import annotations

from cad_quoter.utils.render_utils import (
    ColumnSpec,
    DEFAULT_WIDTH,
    ascii_table,
    draw_boxed_table,
    draw_kv_table,
    ellipsize,
    money,
    pct,
)

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
