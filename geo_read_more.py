"""Compatibility wrapper for the shared DXF enrichment helpers."""

from __future__ import annotations

from cad_quoter.geometry.dxf_enrich import (
    build_geo_from_doc as _build_geo_from_doc,
    build_geo_from_dxf as _build_geo_from_dxf,
    detect_units_scale as _detect_units_scale,
    harvest_hole_geometry as _harvest_hole_geometry,
    harvest_hole_table,
    harvest_outline_metrics as _harvest_outline_metrics,
    harvest_plate_dimensions as _harvest_plate_dimensions,
    harvest_title_notes as _harvest_title_notes,
    iter_spaces as _iter_spaces,
    iter_table_text,
)

__all__ = [
    "build_geo_from_dxf",
    "build_geo_from_doc",
    "detect_units_scale",
    "harvest_plate_dims",
    "harvest_outline",
    "harvest_holes_geometry",
    "harvest_hole_table",
    "harvest_title_notes",
    "iter_table_text",
]


def detect_units_scale(doc):
    """Backward compatible alias for :func:`cad_quoter.geometry.dxf_enrich.detect_units_scale`."""

    return _detect_units_scale(doc)


def build_geo_from_doc(doc):
    """Backward compatible alias for the shared DXF enrichment routine."""

    return _build_geo_from_doc(doc)


def build_geo_from_dxf(path: str):
    """Return GEO metadata harvested from ``path``."""

    return _build_geo_from_dxf(path)


def _units_scale(doc):  # pragma: no cover - legacy alias
    return detect_units_scale(doc)


def _spaces(doc):  # pragma: no cover - legacy alias
    return _iter_spaces(doc)


def harvest_plate_dims(doc, to_in):  # pragma: no cover - legacy alias
    return _harvest_plate_dimensions(doc, to_in)


def harvest_outline(doc, to_in):  # pragma: no cover - legacy alias
    return _harvest_outline_metrics(doc, to_in)


def harvest_holes_geometry(doc, to_in):  # pragma: no cover - legacy alias
    return _harvest_hole_geometry(doc, to_in)


def harvest_title_notes(doc):  # pragma: no cover - legacy alias
    return _harvest_title_notes(doc)

