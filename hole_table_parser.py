"""Backwards-compatible wrapper for :mod:`cad_quoter.geometry.hole_table_parser`."""

from cad_quoter.geometry.hole_table_parser import (
    HoleRow,
    parse_drill_token,
    parse_hole_table_lines,
)

__all__ = [
    "HoleRow",
    "parse_drill_token",
    "parse_hole_table_lines",
]

