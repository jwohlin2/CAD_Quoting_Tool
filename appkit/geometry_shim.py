"""Geometry helper re-exports for backward compatibility."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol

import cad_quoter.geometry as geometry
from cad_quoter.vendors import ezdxf as _ezdxf_vendor


def _missing_geom_fn(name: str):
    def _fn(*_a, **_k):
        raise RuntimeError(f"geometry helper '{name}' is unavailable in this build")

    return _fn


def _export(name: str):
    return getattr(geometry, name, _missing_geom_fn(name))


read_cad_any = _export("read_cad_any")
read_step_shape = _export("read_step_shape")
read_dxf_as_occ_shape = _export("read_dxf_as_occ_shape")
convert_dwg_to_dxf = _export("convert_dwg_to_dxf")
enrich_geo_occ = _export("enrich_geo_occ")
enrich_geo_stl = _export("enrich_geo_stl")
safe_bbox = _export("safe_bbox")
iter_solids = _export("iter_solids")
parse_hole_table_lines = _export("parse_hole_table_lines")
extract_text_lines_from_dxf = _export("extract_text_lines_from_dxf")
text_harvest = _export("text_harvest")
upsert_var_row = _export("upsert_var_row")
require_ezdxf = _export("require_ezdxf")
get_dwg_converter_path = _export("get_dwg_converter_path")
get_import_diagnostics_text = _export("get_import_diagnostics_text")
extract_features_with_occ = _export("extract_features_with_occ")

_HAS_TRIMESH = getattr(geometry, "HAS_TRIMESH", False)
_HAS_EZDXF = getattr(geometry, "HAS_EZDXF", False)
_HAS_ODAFC = getattr(geometry, "HAS_ODAFC", False)
_EZDXF_VER = geometry.EZDXF_VERSION
_HAS_PYMUPDF = getattr(geometry, "_HAS_PYMUPDF", getattr(geometry, "HAS_PYMUPDF", False))
fitz = getattr(geometry, "fitz", None)

if _HAS_EZDXF:
    try:
        ezdxf = _ezdxf_vendor.require_ezdxf()
    except Exception:
        ezdxf = None  # type: ignore[assignment]
else:
    ezdxf = None  # type: ignore[assignment]

if _HAS_ODAFC:
    try:
        odafc = _ezdxf_vendor.require_odafc()
    except Exception:
        odafc = None  # type: ignore[assignment]
else:
    odafc = None  # type: ignore[assignment]

if TYPE_CHECKING:
    class _EzdxfLayouts(Protocol):
        def names_in_taborder(self) -> Iterable[str]: ...
