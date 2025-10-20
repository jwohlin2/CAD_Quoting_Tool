"""Service helpers and configuration containers for the Tkinter UI."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cad_quoter.geometry as _geometry


def _missing_geom_fn(name: str):
    def _fn(*_a, **_k):
        raise RuntimeError(f"geometry helper '{name}' is unavailable in this build")

    return _fn


def _export_geom(name: str):
    return getattr(_geometry, name, _missing_geom_fn(name))


enrich_geo_occ = _export_geom("enrich_geo_occ")
enrich_geo_stl = _export_geom("enrich_geo_stl")
extract_features_with_occ = _export_geom("extract_features_with_occ")
read_cad_any = _export_geom("read_cad_any")
read_step_shape = _export_geom("read_step_shape")
safe_bbox = _export_geom("safe_bbox")
from cad_quoter.app import runtime as _runtime
from cad_quoter.resources import default_app_settings_json
from cad_quoter.domain_models import DEFAULT_MATERIAL_DISPLAY
from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none

from appkit.utils import _parse_length_to_mm


@dataclass(slots=True)
class GeometryLoader:
    """Facade around geometry helper functions used by the UI layer."""

    extract_pdf_all_fn: Callable[[Path, int], dict] | None = None
    extract_pdf_vector_fn: Callable[[str | Path], dict] | None = None
    extract_dxf_or_dwg_fn: Callable[[str | Path], dict] | None = None
    occ_feature_fn: Callable[[str | Path], Any] = extract_features_with_occ
    stl_enricher: Callable[[str | Path], Any] = enrich_geo_stl
    step_reader: Callable[[str | Path], Any] = read_step_shape
    cad_reader: Callable[[str | Path], Any] = read_cad_any
    bbox_fn: Callable[[Any], Any] = safe_bbox
    occ_enricher: Callable[[Any], Any] = enrich_geo_occ

    def extract_pdf_all(self, path: str | Path, dpi: int = 300) -> dict:
        if self.extract_pdf_all_fn is None:
            raise RuntimeError("extract_pdf_all function not provided")
        return self.extract_pdf_all_fn(Path(path), dpi)

    def extract_2d_features_from_pdf_vector(self, path: str | Path) -> dict:
        if self.extract_pdf_vector_fn is None:
            raise RuntimeError("extract_2d_features_from_pdf_vector function not provided")
        return self.extract_pdf_vector_fn(path)

    def extract_2d_features_from_dxf_or_dwg(self, path: str | Path) -> dict:
        if self.extract_dxf_or_dwg_fn is None:
            raise RuntimeError("extract_2d_features_from_dxf_or_dwg function not provided")
        return self.extract_dxf_or_dwg_fn(path)

    def extract_features_with_occ(self, path: str | Path):
        return self.occ_feature_fn(path)

    def enrich_geo_stl(self, path: str | Path):
        return self.stl_enricher(path)

    def read_step_shape(self, path: str | Path) -> Any:
        return self.step_reader(path)

    def read_cad_any(self, path: str | Path) -> Any:
        return self.cad_reader(path)

    def safe_bbox(self, shape: Any):
        return self.bbox_fn(shape)

    def enrich_geo_occ(self, shape: Any):
        return self.occ_enricher(shape)


@dataclass(slots=True)
class UIConfiguration:
    """Aggregate user-interface defaults for the desktop application."""

    title: str = "Compos-AI"
    window_geometry: str = "1260x900"
    llm_enabled_default: bool = True
    apply_llm_adjustments_default: bool = True
    settings_path: Path | None = None
    default_llm_model_path: str | None = None
    default_params: dict[str, Any] | None = None
    default_material_display: str = DEFAULT_MATERIAL_DISPLAY

    def __post_init__(self) -> None:
        if self.settings_path is None:
            self.settings_path = default_app_settings_json()
        if self.default_params is None:
            self.default_params = {}

    def create_params(self) -> dict[str, Any]:
        return copy.deepcopy(self.default_params)


@dataclass(slots=True)
class PricingRegistry:
    """Provide mutable copies of editable pricing defaults to the UI."""

    default_params: dict[str, Any] | None = None
    default_rates: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.default_params is None:
            self.default_params = {}
        if self.default_rates is None:
            self.default_rates = {}

    def create_params(self) -> dict[str, Any]:
        return copy.deepcopy(self.default_params)

    def create_rates(self) -> dict[str, float]:
        return copy.deepcopy(self.default_rates)


@dataclass(slots=True)
class LLMServices:
    """Helper hooks for locating default models and loading vision LLMs."""

    default_model_locator: Callable[[], str] = _runtime.find_default_qwen_model
    vision_loader: Callable[..., Any] = _runtime.load_qwen_vl

    def default_model_path(self) -> str:
        try:
            return self.default_model_locator() or ""
        except Exception:
            return ""

    def load_vision_model(
        self,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = 20,
        n_threads: int | None = None,
    ):
        return self.vision_loader(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads)


def infer_geo_override_defaults(geo_data: dict[str, Any] | None) -> dict[str, Any]:
    """Infer UI override defaults from geometry payloads."""

    if not isinstance(geo_data, dict):
        return {}

    sources: list[dict[str, Any]] = []
    seen: set[int] = set()

    def collect(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        sources.append(obj)
        for val in obj.values():
            collect(val)

    collect(geo_data)
    if not sources:
        return {}

    def find_value(*keys: str) -> Any:
        for key in keys:
            for src in sources:
                if key in src:
                    val = src.get(key)
                    if val not in (None, ""):
                        return val
        return None

    overrides: dict[str, Any] = {}

    plate_len_in = _coerce_float_or_none(find_value("plate_len_in", "plate_length_in"))
    if plate_len_in is None:
        plate_len_mm = _parse_length_to_mm(find_value("plate_len_mm", "plate_length_mm"))
        if plate_len_mm is not None:
            plate_len_in = float(plate_len_mm) / 25.4
    if plate_len_in is not None and plate_len_in > 0:
        overrides["Plate Length (in)"] = float(plate_len_in)

    plate_wid_in = _coerce_float_or_none(find_value("plate_wid_in", "plate_width_in"))
    if plate_wid_in is None:
        plate_wid_mm = _parse_length_to_mm(find_value("plate_wid_mm", "plate_width_mm"))
        if plate_wid_mm is not None:
            plate_wid_in = float(plate_wid_mm) / 25.4
    if plate_wid_in is not None and plate_wid_in > 0:
        overrides["Plate Width (in)"] = float(plate_wid_in)

    thickness_in = _coerce_float_or_none(
        find_value("thickness_in_guess", "thickness_in", "deepest_hole_in")
    )
    if thickness_in is None:
        thickness_mm = _parse_length_to_mm(find_value("thickness_mm", "thickness_mm_guess"))
        if thickness_mm is not None:
            thickness_in = float(thickness_mm) / 25.4
    if thickness_in is not None and thickness_in > 0:
        overrides["Thickness (in)"] = float(thickness_in)

    scrap_pct_val = _coerce_float_or_none(find_value("scrap_pct", "scrap_percent"))
    if scrap_pct_val is not None and scrap_pct_val > 0:
        overrides["Scrap Percent (%)"] = float(scrap_pct_val) * 100.0

    from_back = any(
        bool(src.get(key))
        for key in ("holes_from_back", "needs_back_face", "from_back")
        for src in sources
        if isinstance(src, dict)
    )

    setups_val = _coerce_float_or_none(find_value("setups", "milling_setups", "number_of_setups"))
    setups_int = int(round(setups_val)) if setups_val and setups_val > 0 else None
    if setups_int and setups_int > 0:
        overrides["Number of Milling Setups"] = max(1, setups_int)
    elif from_back:
        overrides["Number of Milling Setups"] = 2

    material_value = None
    for key in ("material", "material_note", "stock_guess", "stock_material", "material_name"):
        candidate = find_value(key)
        if isinstance(candidate, str) and candidate.strip():
            material_value = candidate.strip()
            break
    if material_value:
        overrides["Material"] = material_value

    fai_flag = find_value("fai_required", "fai")
    if fai_flag is not None:
        overrides["FAIR Required"] = 1 if bool(fai_flag) else 0

    def _coerce_count(label: str, *keys: str) -> None:
        val = _coerce_float_or_none(find_value(*keys))
        if val is None:
            return
        try:
            count = int(round(val))
        except Exception:
            return
        if count > 0:
            overrides[label] = count

    _coerce_count("Tap Qty (LLM/GEO)", "tap_qty", "tap_count")
    _coerce_count("Cbore Qty (LLM/GEO)", "cbore_qty", "counterbore_qty")
    _coerce_count("Csk Qty (LLM/GEO)", "csk_qty", "countersink_qty")

    hole_sum = 0.0
    hole_total = 0

    raw_holes_mm = find_value("hole_diams_mm", "hole_diams_mm_precise")
    if isinstance(raw_holes_mm, (list, tuple)) and raw_holes_mm:
        for entry in raw_holes_mm:
            val = _coerce_float_or_none(entry)
            if val is None and entry is not None:
                val = _parse_length_to_mm(entry)
            if val is None or val <= 0:
                continue
            hole_sum += float(val)
            hole_total += 1

    if hole_total == 0:
        raw_holes_in = find_value("hole_diams_in")
        if isinstance(raw_holes_in, (list, tuple)):
            for entry in raw_holes_in:
                val = _coerce_float_or_none(entry)
                if val is None and entry is not None:
                    try:
                        val = float(entry)
                    except Exception:
                        val = None
                if val is None or val <= 0:
                    continue
                hole_sum += float(val * 25.4)
                hole_total += 1

    if hole_total == 0:
        bins_data = find_value("hole_bins", "hole_bins_top")
        if isinstance(bins_data, dict):
            for diam_key, count_val in bins_data.items():
                count = _coerce_float_or_none(count_val)
                if count is None or count <= 0:
                    continue
                diam_mm = _parse_length_to_mm(diam_key)
                if diam_mm is None or diam_mm <= 0:
                    continue
                hole_sum += float(diam_mm) * float(count)
                hole_total += int(round(count))

    if hole_total > 0:
        overrides["Avg Hole Diameter (mm)"] = hole_sum / float(hole_total)
        overrides["Hole Count (override)"] = int(hole_total)

    hole_count_val = _coerce_float_or_none(find_value("hole_count", "hole_count_geom"))
    if hole_count_val is not None and hole_count_val > 0:
        overrides["Hole Count (override)"] = int(round(hole_count_val))

    pocket_count_val = _coerce_float_or_none(find_value("pocket_count", "pocket_count_geom"))
    if pocket_count_val is not None and pocket_count_val > 0:
        overrides["Pocket Count"] = int(round(pocket_count_val))

    return overrides


__all__ = [
    "GeometryLoader",
    "UIConfiguration",
    "PricingRegistry",
    "LLMServices",
    "infer_geo_override_defaults",
]
