"""Smoke tests for the geometry module.

These ensure heavy OCC/trimesh dependencies can be imported without
initialising the GUI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # pragma: no cover - exercised only when OCC is installed
    from cad_quoter import geometry
except ModuleNotFoundError as exc:  # pragma: no cover - skip when OCC missing
    pytest.skip(f"geometry module unavailable: {exc}", allow_module_level=True)


def test_geometry_service_instantiation() -> None:
    service = geometry.GeometryService()
    assert isinstance(service, geometry.GeometryService)


def test_module_exports() -> None:
    # The module should expose convenience helpers used by the UI.
    assert callable(geometry.extract_features)
    assert callable(geometry.enrich_stl)
    assert hasattr(geometry, "HAS_TRIMESH")
