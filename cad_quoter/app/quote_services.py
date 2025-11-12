"""Quote configuration helpers used outside the legacy Tk interface."""

from __future__ import annotations

import copy
from typing import Any

from cad_quoter.domain_models import (
    DEFAULT_MATERIAL_DISPLAY,
    coerce_float_or_none as _coerce_float_or_none,
    normalize_material_key as _normalize_lookup_key,
)
from cad_quoter.utils.machining import parse_length_to_mm

__all__ = [
    "QuoteConfiguration",
    "infer_geo_override_defaults",
]


class QuoteConfiguration:
    """Container for default parameter configuration used by quote rendering."""

    default_params: dict[str, Any] | None = None
    default_material_display: str = DEFAULT_MATERIAL_DISPLAY
    prefer_removal_drilling_hours: bool = True
    stock_price_source: str = "mcmaster_api"
    scrap_price_source: str = "wieland"
    enforce_exact_thickness: bool = True
    allow_thickness_upsize: bool = False
    round_tol_in: float = 0.05
    stock_rounding_mode: str = "per_axis_min_area"
    separate_machine_labor: bool = True
    machine_rate_per_hr: float = 45.0
    labor_rate_per_hr: float = 45.0
    hole_source_preference: str = "table"  # "table" | "geometry" | "auto"
    hole_merge_tol_diam_in: float = 0.001
    hole_merge_tol_depth_in: float = 0.01

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.default_params is None:
            self.default_params = {}

    def copy_default_params(self) -> dict[str, Any]:
        """Return a deep copy of the default parameter set."""

        return copy.deepcopy(self.default_params)

    @property
    def default_material_key(self) -> str:
        return _normalize_lookup_key(self.default_material_display)


def infer_geo_override_defaults(geo_data: dict[str, Any] | None) -> dict[str, Any]:
    """Infer override defaults from geometry payloads."""

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
        plate_len_mm = parse_length_to_mm(find_value("plate_len_mm", "plate_length_mm"))
        if plate_len_mm is not None:
            plate_len_in = float(plate_len_mm) / 25.4
    if plate_len_in is not None and plate_len_in > 0:
        overrides["Plate Length (in)"] = float(plate_len_in)

    plate_wid_in = _coerce_float_or_none(find_value("plate_wid_in", "plate_width_in"))
    if plate_wid_in is None:
        plate_wid_mm = parse_length_to_mm(find_value("plate_wid_mm", "plate_width_mm"))
        if plate_wid_mm is not None:
            plate_wid_in = float(plate_wid_mm) / 25.4
    if plate_wid_in is not None and plate_wid_in > 0:
        overrides["Plate Width (in)"] = float(plate_wid_in)

    thickness_in = _coerce_float_or_none(
        find_value("thickness_in_guess", "thickness_in", "deepest_hole_in")
    )
    if thickness_in is None:
        thickness_mm = parse_length_to_mm(find_value("thickness_mm", "thickness_mm_guess"))
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
                val = parse_length_to_mm(entry)
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
                diam_mm = parse_length_to_mm(diam_key)
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
