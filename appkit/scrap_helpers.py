"""Compatibility layer for scrap helpers.

The canonical implementations live in :mod:`cad_quoter.utils.scrap`.  This
module re-exports those helpers for backwards compatibility with callers that
still import from :mod:`appkit.scrap_helpers`.
"""

from cad_quoter.utils.scrap import *  # noqa: F401,F403

__all__ = [
    "SCRAP_DEFAULT_GUESS",
    "HOLE_SCRAP_MULT",
    "HOLE_SCRAP_CAP",
    "_coerce_scrap_fraction",
    "normalize_scrap_pct",
    "_iter_hole_diams_mm",
    "_plate_bbox_mm2",
    "_holes_scrap_fraction",
    "_estimate_scrap_from_stock_plan",
]
