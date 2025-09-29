"""Smoke-test for cad_quoter.geometry imports."""


import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from cad_quoter import geometry
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    pytest.skip(str(exc), allow_module_level=True)
except ImportError as exc:  # pragma: no cover - environment specific
    pytest.skip(str(exc), allow_module_level=True)


def test_geometry_service_imports() -> None:
    service = geometry.GeometryService()
    assert hasattr(service, "load_model")
    assert isinstance(geometry.HAS_TRIMESH, bool)

