# -*- coding: utf-8 -*-
# app_gui_occ_flow_v8_single_autollm.py
r"""
Single-file CAD Quoter (v8)
- LLM (Qwen via llama-cpp) is ENABLED by default.
- Auto-loads a GGUF model from:
    1) QWEN_GGUF_PATH env var (if a file)
    2) D:\CAD_Quoting_Tool\models\*.gguf  (your path)
    3) <script_dir>\models\*.gguf
    4) .\models\*.gguf (cwd)
- STEP/IGES via OCP, STL via trimesh
- Variables auto-detect, Overrides tab, LLM tab
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import time
import typing
import tkinter as tk
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

from cad_quoter.app import runtime as _runtime
from cad_quoter.app.container import ServiceContainer, create_default_container
from cad_quoter.config import (
    AppEnvironment,
    ConfigError,
    configure_logging,
    logger,
)
from cad_quoter.config import (
    describe_runtime_environment as _describe_runtime_environment,
)

APP_ENV = AppEnvironment.from_env()


EXTRA_DETAIL_RE = re.compile(r"^includes\b.*extras\b", re.IGNORECASE)


def _coerce_env_bool(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip().lower()
    return text in {"1", "true", "yes", "on"}


FORCE_PLANNER = _coerce_env_bool(os.environ.get("FORCE_PLANNER"))


def jdump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


def describe_runtime_environment() -> dict[str, str]:
    """Return a redacted snapshot of runtime configuration for auditors."""

    info = _describe_runtime_environment()
    info["llm_debug_enabled"] = str(APP_ENV.llm_debug_enabled)
    info["llm_debug_dir"] = str(APP_ENV.llm_debug_dir)
    return info


def roughly_equal(a: float | int | str | None, b: float | int | str | None, *, eps: float = 0.01) -> bool:
    """Return True when *a* and *b* are approximately equal within ``eps`` dollars."""

    try:
        a_val = float(a or 0.0)
    except Exception:
        return False
    try:
        b_val = float(b or 0.0)
    except Exception:
        return False
    try:
        eps_val = float(eps)
    except Exception:
        eps_val = 0.0
    return math.isclose(a_val, b_val, rel_tol=0.0, abs_tol=abs(eps_val))


import copy
import sys
import textwrap
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeAlias, TypeVar, cast


T = TypeVar("T")


def resolve_planner(
    params: Mapping[str, Any] | None,
    signals: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Determine whether planner pricing should be used and the mode."""

    default_mode = "auto"
    if FORCE_PLANNER:
        planner_mode = "planner"
    elif isinstance(params, _MappingABC):
        try:
            raw_mode = params.get("PlannerMode", default_mode)
        except Exception:
            raw_mode = default_mode
        try:
            planner_mode = str(raw_mode).strip().lower() or default_mode
        except Exception:
            planner_mode = default_mode
    else:
        planner_mode = default_mode

    signals_map: Mapping[str, Any]
    if isinstance(signals, _MappingABC):
        signals_map = signals
    else:
        signals_map = {}

    has_line_items = bool(signals_map.get("line_items"))
    has_pricing = bool(signals_map.get("pricing_result"))
    has_totals = bool(signals_map.get("totals_present"))
    try:
        recognized_raw = signals_map.get("recognized_line_items", 0)
        recognized_count = int(recognized_raw)
    except Exception:
        recognized_count = 0
    has_recognized = recognized_count > 0

    base_signal = has_line_items or has_pricing or has_totals or has_recognized

    if planner_mode == "planner":
        used_planner = base_signal or has_totals
    elif planner_mode == "legacy":
        used_planner = has_line_items or has_recognized
    else:  # auto / default
        used_planner = base_signal or has_recognized

    if has_line_items:
        used_planner = True
    return used_planner, planner_mode


def _normalize_item_text(value: Any) -> str:
    """Return a normalized key for matching variables rows."""

    if value is None:
        text = ""
    else:
        text = str(value)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_item(value: Any) -> str:
    """Public wrapper for item normalization used across the editor."""

    return _normalize_item_text(value)

import cad_quoter.geometry as geometry
from cad_quoter.geometry import read_dxf_as_occ_shape
from cad_quoter.geo2d import (
    apply_2d_features_to_variables,
    to_noncapturing as _to_noncapturing,
)
from bucketizer import bucketize

# Guardrails for LLM-generated process adjustments.
LLM_MULTIPLIER_MIN = 0.25
LLM_MULTIPLIER_MAX = 4.0
LLM_ADDER_MAX = 8.0

# Tolerance for invariant checks that guard against silent drift when rendering
# cost sections.
_LABOR_SECTION_ABS_EPSILON = 0.51
_PLANNER_BUCKET_ABS_EPSILON = 0.51


ensure_runtime_dependencies = _runtime.ensure_runtime_dependencies
find_default_qwen_model = _runtime.find_default_qwen_model
load_qwen_vl = _runtime.load_qwen_vl

ensure_runtime_dependencies()

# Backwards compatibility: legacy module-level names expected by tests/scripts
_DEFAULT_VL_MODEL_NAMES = _runtime.DEFAULT_VL_MODEL_NAMES
_DEFAULT_MM_PROJ_NAMES = _runtime.DEFAULT_MM_PROJ_NAMES
VL_MODEL = str(_runtime.LEGACY_VL_MODEL)
MM_PROJ = str(_runtime.LEGACY_MM_PROJ)
LEGACY_VL_MODEL = str(_runtime.LEGACY_VL_MODEL)
LEGACY_MM_PROJ = str(_runtime.LEGACY_MM_PROJ)
PREFERRED_MODEL_DIRS: list[str | Path] = [str(p) for p in _runtime.PREFERRED_MODEL_DIRS]


from cad_quoter.domain_models import (
    DEFAULT_MATERIAL_DISPLAY,
    DEFAULT_MATERIAL_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    MATERIAL_DISPLAY_BY_KEY,
    MATERIAL_DROPDOWN_OPTIONS,
    MATERIAL_KEYWORDS,
    MATERIAL_MAP,
    MATERIAL_OTHER_KEY,
)
from cad_quoter.domain_models import (
    coerce_float_or_none as _coerce_float_or_none,
)
from cad_quoter.domain_models import (
    normalize_material_key as _normalize_lookup_key,
)
from cad_quoter.coerce import to_float, to_int
from cad_quoter.utils import compact_dict, sdict
from cad_quoter.pricing import (
    LB_PER_KG,
    PricingEngine,
    create_default_registry,
)

_DEFAULT_MATERIAL_DENSITY_G_CC = MATERIAL_DENSITY_G_CC_BY_KEY.get(
    DEFAULT_MATERIAL_KEY, 7.85
)
from cad_quoter.pricing import (
    get_mcmaster_unit_price as _get_mcmaster_unit_price,
)
from cad_quoter.pricing import (
    price_value_to_per_gram as _price_value_to_per_gram,
)
from cad_quoter.pricing import (
    resolve_material_unit_price as _resolve_material_unit_price,
)
from cad_quoter.pricing import time_estimator as _time_estimator
from cad_quoter.pricing.speeds_feeds_selector import (
    normalize_material as _normalize_material,
)
from cad_quoter.pricing.speeds_feeds_selector import (
    pick_speeds_row as _pick_speeds_row,
)
from cad_quoter.pricing.speeds_feeds_selector import (
    unit_hp_cap as _unit_hp_cap,
)
from cad_quoter.pricing.time_estimator import (
    MachineParams as _TimeMachineParams,
)
from cad_quoter.pricing.time_estimator import (
    OperationGeometry as _TimeOperationGeometry,
)
from cad_quoter.pricing.time_estimator import (
    OverheadParams as _TimeOverheadParams,
)
from cad_quoter.pricing.time_estimator import (
    ToolParams as _TimeToolParams,
)
from cad_quoter.pricing.time_estimator import (
    estimate_time_min as _estimate_time_min,
)
from cad_quoter.pricing.wieland import lookup_price as lookup_wieland_price
from cad_quoter.rates import migrate_flat_to_two_bucket, two_bucket_to_flat
from cad_quoter.vendors.mcmaster_stock import lookup_sku_and_price_for_mm
from cad_quoter.llm import (
    EDITOR_FROM_UI,
    EDITOR_TO_SUGG,
    LLMClient,
    SUGG_TO_EDITOR,
    infer_hours_and_overrides_from_geo as _infer_hours_and_overrides_from_geo,
    parse_llm_json,
    explain_quote,
)

_DEFAULT_MATERIAL_DENSITY_G_CC = MATERIAL_DENSITY_G_CC_BY_KEY.get(
    DEFAULT_MATERIAL_KEY,
    7.85,
)

try:
    from process_planner import (
        PLANNERS as _PROCESS_PLANNERS,
    )
    from process_planner import (
        choose_skims as _planner_choose_skims,
    )
    from process_planner import (
        choose_wire_size as _planner_choose_wire_size,
    )
    from process_planner import (
        needs_wedm_for_windows as _planner_needs_wedm_for_windows,
    )
    from process_planner import (
        plan_job as _process_plan_job,
    )
except Exception:  # pragma: no cover - planner is optional at runtime
    _process_plan_job = None
    _PROCESS_PLANNERS = {}
    _planner_choose_wire_size = None
    _planner_choose_skims = None
    _planner_needs_wedm_for_windows = None
else:  # pragma: no cover - defensive guard for unexpected exports
    if not isinstance(_PROCESS_PLANNERS, dict):
        _PROCESS_PLANNERS = {}


_PROCESS_PLANNER_HELPERS: dict[str, Callable[..., Any]] = {}
if "_planner_choose_wire_size" in globals() and callable(_planner_choose_wire_size):
    _PROCESS_PLANNER_HELPERS["choose_wire_size"] = _planner_choose_wire_size  # type: ignore[index]
if "_planner_choose_skims" in globals() and callable(_planner_choose_skims):
    _PROCESS_PLANNER_HELPERS["choose_skims"] = _planner_choose_skims  # type: ignore[index]
if "_planner_needs_wedm_for_windows" in globals() and callable(
    _planner_needs_wedm_for_windows
):
    _PROCESS_PLANNER_HELPERS["needs_wedm_for_windows"] = _planner_needs_wedm_for_windows  # type: ignore[index]


# Mapping of process keys to editor labels for propagating derived hours from
# LLM suggestions. The scale term allows lightweight conversions if we need to
# express the hour totals in another unit for a given field.
PROC_MULT_TARGETS: dict[str, tuple[str, float]] = {
    # Processes that have direct counterparts in the default variables sheet.
    "inspection": ("In-Process Inspection Hours", 1.0),
    "finishing_deburr": ("Deburr Hours", 1.0),
    "deburr": ("Deburr Hours", 1.0),
    "saw_waterjet": ("Sawing Hours", 1.0),
    "assembly": ("Assembly Hours", 1.0),
    "packaging": ("Packaging Labor Hours", 1.0),
}

HARDWARE_PASS_LABEL = "Hardware"
LEGACY_HARDWARE_PASS_LABEL = "Hardware / BOM"
_HARDWARE_LABEL_ALIASES = {
    HARDWARE_PASS_LABEL.lower(),
    LEGACY_HARDWARE_PASS_LABEL.lower(),
    "hardware/bom",
    "hardware bom",
}

# --- In-process inspection curve knobs ---
INPROC_REF_TOL_IN = 0.002   # ref tolerance where no premium starts
INPROC_BASE_HR    = 0.30    # hours at/looser than ref
INPROC_SCALE_HR   = 1.60    # how much to add as tolerance tightens
INPROC_EXP        = 0.60    # curve shape: lower=flatter

# small, bounded adders for “many tight callouts”
INPROC_TIGHT_PER   = 0.15   # +hr per extra tight tol (≤0.0015")
INPROC_TIGHT_MAX   = 0.60
INPROC_SUBTHOU_PER = 0.20   # +hr per sub-thou tol (≤0.0005")
INPROC_SUBTHOU_MAX = 0.40
INPROC_MENTION_PER = 0.10   # “tight tolerance” textual mentions
INPROC_MENTION_MAX = 0.30


def _canonical_pass_label(label: str | None) -> str:
    name = str(label or "").strip()
    if name.lower() in _HARDWARE_LABEL_ALIASES:
        return HARDWARE_PASS_LABEL
    return name


def _canonicalize_pass_through_map(data: Any) -> dict[str, float]:
    """Normalize a pass-through dictionary.

    Accepts flexible shapes (mapping, list of pairs, list of dicts) and returns
    a clean ``{canonical_label: float_amount}`` map. Unknown or non-numeric
    amounts are ignored. Labels are normalized via ``_canonical_pass_label``.
    """
    result: dict[str, float] = {}

    def _add(label: Any, amount: Any) -> None:
        key = _canonical_pass_label(label)
        try:
            val = to_float(amount)
        except Exception:
            val = None
        if key and val is not None and math.isfinite(val):
            result[key] = result.get(key, 0.0) + float(val)

    if isinstance(data, Mapping):
        for k, v in data.items():
            if isinstance(v, Mapping):
                # common shapes: {label: {amount|value|cost: x}}
                inner = v
                amt = inner.get("amount", inner.get("value", inner.get("cost", inner.get("price"))))
                _add(k, amt)
            else:
                _add(k, v)
        return result

    if isinstance(data, (list, tuple)):
        for entry in data:
            if isinstance(entry, Mapping):
                lbl = entry.get("label") or entry.get("name") or entry.get("key") or entry.get("type")
                amt = entry.get("amount", entry.get("value", entry.get("cost", entry.get("price"))))
                if lbl is None and len(entry) == 1:
                    # single-key dict
                    k = next(iter(entry.keys()))
                    _add(k, entry.get(k))
                else:
                    _add(lbl, amt)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                _add(entry[0], entry[1])
        return result

    # Fallback: nothing recognized
    return result


_AMORTIZED_LABEL_PATTERN = re.compile(
    r"\s*\((amortized|amortised)(?:\s+(?:per\s+(?:part|piece|pc|unit)|each|ea))?\)\s*$",
    re.IGNORECASE,
)


def _canonical_amortized_label(label: Any) -> tuple[str, bool]:
    """Return a canonical label and flag for amortized cost rows."""

    text = str(label or "").strip()
    if not text:
        return "", False

    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    normalized = normalized.replace("perpart", "per part")
    normalized = normalized.replace("perpiece", "per piece")
    tokens = normalized.split()
    token_set = set(tokens)

    def _has(*want: str) -> bool:
        return all(token in token_set for token in want)

    amortized_tokens = {"amortized", "amortised"}
    has_amortized = any(token in token_set for token in amortized_tokens)

    if has_amortized:
        per_part = (
            _has("per", "part")
            or _has("per", "pc")
            or _has("per", "piece")
            or "per piece" in normalized
            or "per unit" in normalized
        )
        if "programming" in token_set:
            canonical = (
                "Programming (amortized per part)"
                if per_part
                else "Programming (amortized)"
            )
            return canonical, True
        if "fixture" in token_set or "fixturing" in token_set:
            canonical = (
                "Fixture Build (amortized per part)"
                if per_part
                else "Fixture Build (amortized)"
            )
            return canonical, True
        return text, True

    match = _AMORTIZED_LABEL_PATTERN.search(text)
    if match:
        prefix = text[: match.start()].rstrip()
        canonical = f"{prefix} (amortized)" if prefix else match.group(1).lower()
        return canonical, True

    return text, False

import pandas as pd
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    try:
        from OCP.TopoDS import TopoDS_Shape as _TopoDSShape  # type: ignore[import]
    except ImportError:
        try:
            from OCC.Core.TopoDS import TopoDS_Shape as _TopoDSShape  # type: ignore[import]
        except ImportError:  # pragma: no cover - typing-only fallback
            from typing import Any as _TopoDSShape  # type: ignore[assignment]
    TopoDSShapeT: TypeAlias = _TopoDSShape
else:
    TopoDSShapeT: TypeAlias = Any

try:  # Prefer pythonocc-core's OCP bindings when available
    from OCP.BRep import BRep_Tool  # type: ignore[import]
    from OCP.TopAbs import (  # type: ignore[import]
        TopAbs_COMPOUND,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_ShapeEnum,
    )
    from OCP.TopExp import TopExp, TopExp_Explorer  # type: ignore[import]
    from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape  # type: ignore[import]
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape  # type: ignore[import]
    from OCP.BRepTools import BRepTools  # type: ignore[import]
except ImportError:
    try:  # Fallback to pythonocc-core
        from OCC.Core.BRep import BRep_Tool  # type: ignore[import]
        from OCC.Core.TopAbs import (  # type: ignore[import]
            TopAbs_COMPOUND,
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_SHELL,
            TopAbs_SOLID,
            TopAbs_ShapeEnum,
        )
        from OCC.Core.TopExp import TopExp, TopExp_Explorer  # type: ignore[import]
        from OCC.Core.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape  # type: ignore[import]
        from OCC.Core.TopTools import (  # type: ignore[import]
            TopTools_IndexedDataMapOfShapeListOfShape,
        )
    except ImportError:  # pragma: no cover - only hit in environments without OCCT
        if TYPE_CHECKING:  # Keep type checkers happy without introducing runtime deps
            from typing import Any

            BRep_Tool = TopAbs_COMPOUND = TopAbs_EDGE = TopAbs_FACE = TopAbs_SHELL = TopAbs_SOLID = TopAbs_ShapeEnum = TopExp = TopExp_Explorer = BRepTools = Any  # type: ignore[assignment]
            TopoDS = TopoDS_Face = TopoDS_Shape = Any  # type: ignore[assignment]
            TopTools_IndexedDataMapOfShapeListOfShape = Any  # type: ignore[assignment]
        else:
            class _MissingOCCTModule:
                def __getattr__(self, item):
                    raise ImportError(
                        "OCCT bindings are required for geometry operations. "
                        "Please install pythonocc-core or the OCP wheels."
                    )

            BRep_Tool = _MissingOCCTModule()
            TopAbs_EDGE = TopAbs_FACE = TopAbs_SHELL = TopAbs_SOLID = TopAbs_COMPOUND = TopAbs_ShapeEnum = _MissingOCCTModule()
            TopExp = TopExp_Explorer = _MissingOCCTModule()
            BRepTools = _MissingOCCTModule()
            TopoDS = TopoDS_Face = TopoDS_Shape = _MissingOCCTModule()
            TopTools_IndexedDataMapOfShapeListOfShape = _MissingOCCTModule()

if TYPE_CHECKING:
    from typing import cast

    BRep_Tool = cast(_BRepToolProto, BRep_Tool)
    TopoDS = cast(_TopoDSProto, TopoDS)

try:
    from geo_read_more import build_geo_from_dxf as build_geo_from_dxf_path
except Exception:
    build_geo_from_dxf_path = None  # type: ignore[assignment]


_MappingABC = Mapping

_build_geo_from_dxf_hook: Optional[Callable[[str], Dict[str, Any]]] = None


# Geometry helpers (re-exported for backward compatibility)
def _missing_geom_fn(name: str):
    def _fn(*_a, **_k):
        raise RuntimeError(f"geometry helper '{name}' is unavailable in this build")
    return _fn

def _export(name: str):
    return getattr(geometry, name, _missing_geom_fn(name))

load_model = _export("load_model")
load_cad_any = _export("load_cad_any")
read_cad_any = _export("read_cad_any")
read_step_shape = _export("read_step_shape")
read_step_or_iges_or_brep = _export("read_step_or_iges_or_brep")
convert_dwg_to_dxf = _export("convert_dwg_to_dxf")
enrich_geo_occ = _export("enrich_geo_occ")
enrich_geo_stl = _export("enrich_geo_stl")
safe_bbox = _export("safe_bbox")
safe_bounding_box = _export("safe_bounding_box")
iter_solids = _export("iter_solids")
explode_compound = _export("explode_compound")
parse_hole_table_lines = _export("parse_hole_table_lines")
extract_text_lines_from_dxf = _export("extract_text_lines_from_dxf")
contours_from_polylines = _export("contours_from_polylines")
holes_from_circles = _export("holes_from_circles")
text_harvest = _export("text_harvest")
extract_entities = _export("extract_entities")
read_dxf = _export("read_dxf")
upsert_var_row = _export("upsert_var_row")
get_dwg_converter_path = _export("get_dwg_converter_path")
have_dwg_support = _export("have_dwg_support")
get_import_diagnostics_text = _export("get_import_diagnostics_text")

_geometry_require_ezdxf = getattr(geometry, "require_ezdxf", None)
_HAS_TRIMESH = getattr(geometry, "HAS_TRIMESH", False)
_HAS_EZDXF = getattr(geometry, "HAS_EZDXF", False)
_HAS_ODAFC = getattr(geometry, "HAS_ODAFC", False)
_EZDXF_VER = geometry.EZDXF_VERSION

if _HAS_EZDXF:
    try:
        import ezdxf  # type: ignore[import]
    except Exception:
        ezdxf = None  # type: ignore[assignment]
else:
    ezdxf = None  # type: ignore[assignment]

if _HAS_ODAFC:
    try:
        from ezdxf.addons import odafc  # type: ignore[import]
    except Exception:
        odafc = None  # type: ignore[assignment]
else:
    odafc = None  # type: ignore[assignment]

if TYPE_CHECKING:
    try:
        from ezdxf.ezdxf import Drawing as _EzdxfDrawing
    except Exception:  # pragma: no cover - optional dependency
        _EzdxfDrawing = Any  # type: ignore[assignment]
else:
    _EzdxfDrawing = Any  # type: ignore[assignment]

Drawing = _EzdxfDrawing

SCRAP_DEFAULT_GUESS = 0.15


# --- holes-based scrap helpers ---
HOLE_SCRAP_MULT = 1.0  # tune 0.5–1.5 if you want holes to “count” more/less than area ratio


def _iter_hole_diams_mm(geo_ctx: Mapping[str, Any] | None) -> list[float]:
    """Collect hole diameters (mm) from DXF and STEP-derived context."""

    if not isinstance(geo_ctx, Mapping):
        return []
    geo_map = cast(Mapping[str, Any], geo_ctx)
    derived_obj = geo_map.get("derived")
    if isinstance(derived_obj, Mapping):
        derived: Mapping[str, Any] = derived_obj
    else:
        derived = {}
    out: list[float] = []

    # DXF path: app already stores list of diameters as millimetres.
    # See holes_from_circles → "hole_diams_mm".
    # (DXF radius→dia, full arcs treated as circles.)
    dxf_diams = derived.get("hole_diams_mm")
    if isinstance(dxf_diams, (list, tuple)):
        for d in dxf_diams:
            try:
                v = float(d)
            except Exception:
                continue
            if v > 0:
                out.append(v)

    # STEP path: cylindrical faces → list of dicts with "dia_mm".
    step_holes = derived.get("holes")
    if isinstance(step_holes, (list, tuple)):
        for h in step_holes:
            try:
                v = float(h.get("dia_mm"))
            except Exception:
                continue
            if v > 0:
                out.append(v)
    return out


def _plate_bbox_mm2(geo_ctx: Mapping[str, Any] | None) -> float:
    """Approximate plate area from bbox (mm^2); fall back to UI plate L×W (in)."""

    if not isinstance(geo_ctx, Mapping):
        return 0.0
    # Preferred: derived bbox from DXF polylines
    geo_map = cast(Mapping[str, Any], geo_ctx)
    derived_obj = geo_map.get("derived")
    if isinstance(derived_obj, Mapping):
        derived: Mapping[str, Any] = derived_obj
    else:
        derived = {}
    bbox_mm = derived.get("bbox_mm")
    if (
        isinstance(bbox_mm, (list, tuple))
        and len(bbox_mm) == 2
        and all(isinstance(x, (int, float)) and x > 0 for x in bbox_mm)
    ):
        return float(bbox_mm[0]) * float(bbox_mm[1])

    # Fallback: UI plate Length/Width (inches) already present in sheet/vars
    try:
        L_in = float(geo_map.get("plate_length_mm") or 0.0) / 25.4  # if caller put mm there
    except Exception:
        L_in = None
    try:
        W_in = float(geo_map.get("plate_width_mm") or 0.0) / 25.4
    except Exception:
        W_in = None

    # If the mm fields aren't set, check UI vars stuffed into geo_ctx
    if not (L_in and W_in):
        try:
            L_in = L_in or float(geo_map.get("plate_length_in"))
        except Exception:
            pass
        try:
            W_in = W_in or float(geo_map.get("plate_width_in"))
        except Exception:
            pass

    if L_in and W_in and L_in > 0 and W_in > 0:
        return float(L_in * 25.4) * float(W_in * 25.4)
    return 0.0


def _holes_scrap_fraction(geo_ctx: Mapping[str, Any] | None, *, cap: float = 0.25) -> float:
    """Scrap ~= sum(hole areas) / plate area (clamped)."""

    diams = _iter_hole_diams_mm(geo_ctx)
    if not diams:
        return 0.0
    plate_area_mm2 = _plate_bbox_mm2(geo_ctx)
    if plate_area_mm2 <= 0:
        return 0.0
    holes_area_mm2 = 0.0
    for d in diams:
        r = 0.5 * float(d)
        holes_area_mm2 += math.pi * r * r
    frac = HOLE_SCRAP_MULT * (holes_area_mm2 / plate_area_mm2)
    if not math.isfinite(frac) or frac < 0:
        return 0.0
    return min(cap, float(frac))


def _holes_removed_mass_g(geo_ctx: Mapping[str, Any] | None) -> float | None:
    """
    Optional: estimate grams removed by (through) holes = Σ(πr^2 * thickness) * density.
    Uses thickness_mm from geo_ctx and material density map; safe to skip if unknown.
    """

    if not isinstance(geo_ctx, Mapping):
        return None
    # thickness
    thickness_mm = None
    for key in ("thickness_mm", ("geo", "thickness_mm")):
        raw: Any | None = None
        try:
            if isinstance(key, tuple):
                nested = geo_ctx.get(key[0])
                if isinstance(nested, Mapping):
                    raw = nested.get(key[1])
            else:
                raw = geo_ctx.get(key)
        except Exception:
            raw = None
        if raw in (None, ""):
            continue
        try:
            thickness_mm = float(raw)
        except (TypeError, ValueError):
            thickness_mm = None
        if thickness_mm and thickness_mm > 0:
            break
    if not thickness_mm or thickness_mm <= 0:
        return None

    diams = _iter_hole_diams_mm(geo_ctx)
    if not diams:
        return None
    vol_mm3 = 0.0
    for d in diams:
        r = 0.5 * float(d)
        vol_mm3 += math.pi * r * r * float(thickness_mm)

    # material density lookup (g/cc)
    name = str((geo_ctx.get("material") or "")).strip().lower()
    key = _normalize_lookup_key(name) if name else None
    dens = None
    if key and key in MATERIAL_DENSITY_G_CC_BY_KEY:
        dens = MATERIAL_DENSITY_G_CC_BY_KEY[key]
    if not dens and name:
        dens = MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(name)
    if not dens:
        dens = 7.85  # steel-ish default
    grams = (vol_mm3 / 1000.0) * float(dens)
    return grams if grams > 0 else None


def normalize_scrap_pct(val: Any, cap: float = 0.25) -> float:
    """Return a clamped scrap fraction within ``[0, cap]``.

    The helper accepts common UI inputs such as ``15`` (percent) or ``0.15``
    (fraction) and gracefully handles ``None`` or empty strings by falling
    back to ``0``.
    """

    try:
        cap_val = float(cap)
    except Exception:
        cap_val = 0.25
    if not math.isfinite(cap_val):
        cap_val = 0.25
    cap_val = max(0.0, cap_val)

    if val is None:
        raw = 0.0
    elif isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            raw = 0.0
        elif stripped.endswith("%"):
            try:
                raw = float(stripped.rstrip("%")) / 100.0
            except Exception:
                raw = 0.0
        else:
            try:
                raw = float(stripped)
            except Exception:
                raw = 0.0
    else:
        try:
            raw = float(val)
        except Exception:
            raw = 0.0

    if not math.isfinite(raw):
        raw = 0.0
    if raw > 1.0:
        raw = raw / 100.0
    raw = max(raw, 0.0)
    return min(cap_val, raw)


def _scrap_value_provided(val: Any) -> bool:
    """Return ``True`` when ``val`` looks like a real scrap entry."""

    if val is None:
        return False
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return False
        if stripped.endswith("%"):
            stripped = stripped.rstrip("% ").strip()
        try:
            parsed = float(stripped)
        except Exception:
            return False
        return math.isfinite(parsed)
    try:
        parsed = float(val)
    except Exception:
        return False
    return math.isfinite(parsed)


def _auto_accept_suggestions(suggestions: dict[str, Any] | None) -> dict[str, Any]:
    accept: dict[str, Any] = {}
    if not isinstance(suggestions, dict):
        return accept
    meta = suggestions.get("_meta") if isinstance(suggestions.get("_meta"), dict) else {}

    def _confidence_for(path: tuple[str, ...]) -> float | None:
        node: Any = meta
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        if isinstance(node, dict):
            conf_val = node.get("confidence")
            if isinstance(conf_val, (int, float)):
                return float(conf_val)
        return None

    for bucket in ("process_hour_multipliers", "process_hour_adders", "add_pass_through"):
        data = suggestions.get(bucket)
        if isinstance(data, dict) and data:
            bucket_accept: dict[str, bool] = {}
            for subkey in data.keys():
                conf = _confidence_for((bucket, str(subkey)))
                bucket_accept[str(subkey)] = True if conf is None else conf >= 0.6
            accept[bucket] = bucket_accept
    scalar_keys = (
        "scrap_pct",
        "contingency_pct",
        "setups",
        "fixture",
        "fixture_build_hr",
        "soft_jaw_hr",
        "soft_jaw_material_cost",
        "handling_adder_hr",
        "cmm_minutes",
        "in_process_inspection_hr",
        "fai_required",
        "fai_prep_hr",
        "packaging_hours",
        "packaging_flat_cost",
        "shipping_hint",
    )
    for scalar_key in scalar_keys:
        if scalar_key in suggestions:
            conf = _confidence_for((scalar_key,))
            accept[scalar_key] = True if conf is None else conf >= 0.6
    if suggestions.get("notes"):
        accept["notes"] = True
    if isinstance(suggestions.get("operation_sequence"), list):
        conf = _confidence_for(("operation_sequence",))
        accept["operation_sequence"] = True if conf is None else conf >= 0.6
    if isinstance(suggestions.get("drilling_strategy"), dict):
        conf = _confidence_for(("drilling_strategy",))
        accept["drilling_strategy"] = True if conf is None else conf >= 0.6
    return accept
from cad_quoter.domain import QuoteState


def _as_float_or_none(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            return float(cleaned)
    except Exception:
        return None
    return None


def coerce_bounds(bounds: Mapping | None) -> dict[str, Any]:
    """Normalize LLM bounds into a canonical structure."""

    bounds = bounds or {}

    mult_min = _as_float_or_none(bounds.get("mult_min"))
    if mult_min is None:
        mult_min = LLM_MULTIPLIER_MIN
    else:
        mult_min = max(LLM_MULTIPLIER_MIN, float(mult_min))

    mult_max = _as_float_or_none(bounds.get("mult_max"))
    if mult_max is None:
        mult_max = LLM_MULTIPLIER_MAX
    else:
        mult_max = min(LLM_MULTIPLIER_MAX, float(mult_max))
    mult_max = max(mult_max, mult_min)

    adder_max = _as_float_or_none(bounds.get("adder_max_hr"))
    add_hr_cap = _as_float_or_none(bounds.get("add_hr_max"))
    if adder_max is None and add_hr_cap is not None:
        adder_max = float(add_hr_cap)
    elif adder_max is not None and add_hr_cap is not None:
        adder_max = min(float(adder_max), float(add_hr_cap))
    if adder_max is None:
        adder_max = LLM_ADDER_MAX
    adder_max = max(0.0, min(LLM_ADDER_MAX, float(adder_max)))

    scrap_min = _as_float_or_none(bounds.get("scrap_min"))
    scrap_min = max(0.0, float(scrap_min)) if scrap_min is not None else 0.0

    scrap_max = _as_float_or_none(bounds.get("scrap_max"))
    scrap_max = float(scrap_max) if scrap_max is not None else 0.25

    bucket_caps_raw = bounds.get("adder_bucket_max") or bounds.get("add_hr_bucket_max")
    bucket_caps: dict[str, float] = {}
    if isinstance(bucket_caps_raw, Mapping):
        for key, raw in bucket_caps_raw.items():
            cap_val = _as_float_or_none(raw)
            if cap_val is None:
                continue
            bucket_caps[str(key).lower()] = max(0.0, float(cap_val))

    return {
        "mult_min": mult_min,
        "mult_max": mult_max,
        "adder_max_hr": adder_max,
        "scrap_min": scrap_min,
        "scrap_max": scrap_max,
        "adder_bucket_max": bucket_caps,
    }


def build_suggest_payload(
    geo: dict | None,
    baseline: dict | None,
    rates: dict | None,
    bounds: dict | None,
) -> dict:
    """Assemble the JSON payload passed to the suggestion LLM."""

    geo = geo or {}
    baseline = baseline or {}
    rates = rates or {}
    bounds = bounds or {}

    derived = geo.get("derived") or {}

    def _clean_nested(value: Any, depth: int = 0, max_depth: int = 3, limit: int = 24):
        if depth >= max_depth:
            return None
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for idx, (key, val) in enumerate(value.items()):
                if idx >= limit:
                    break
                cleaned_val = _clean_nested(val, depth + 1, max_depth, limit)
                if cleaned_val is not None:
                    cleaned[str(key)] = cleaned_val
            return cleaned
        if isinstance(value, (list, tuple, set)):
            cleaned_list: list[Any] = []
            for idx, item in enumerate(value):
                if idx >= limit:
                    break
                cleaned_val = _clean_nested(item, depth + 1, max_depth, limit)
                if cleaned_val is not None:
                    cleaned_list.append(cleaned_val)
            return cleaned_list
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:
                return None
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            return value.strip()
        coerced = to_float(value)
        if coerced is not None:
            return coerced
        try:
            return str(value)
        except Exception:
            return None

    hole_bins = derived.get("hole_bins") or {}
    hole_bins_top: dict[str, int] = {}
    if isinstance(hole_bins, dict):
        sorted_bins = sorted(
            ((str(k), to_int(v) or 0) for k, v in hole_bins.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        hole_bins_top = {k: int(v) for k, v in sorted_bins[:8] if v}

    raw_thickness = geo.get("thickness_mm")
    thickness_mm = to_float(raw_thickness)
    if thickness_mm is None and isinstance(raw_thickness, dict):
        for key in ("value", "mm", "thickness_mm"):
            candidate = to_float(raw_thickness.get(key))
            if candidate is not None:
                thickness_mm = candidate
                break
    if thickness_mm is None:
        thickness_mm = to_float(geo.get("thickness"))

    material_val = geo.get("material")
    if isinstance(material_val, dict):
        material_name = (
            material_val.get("name")
            or material_val.get("display")
            or material_val.get("material")
        )
    else:
        material_name = material_val
    material_name = str(material_name).strip() if material_name else "Steel"

    hole_count_val = to_float(geo.get("hole_count"))
    if hole_count_val is None:
        hole_count_val = to_float(derived.get("hole_count"))
    hole_count = int(hole_count_val or 0)

    tap_qty = to_int(derived.get("tap_qty")) or 0
    cbore_qty = to_int(derived.get("cbore_qty")) or 0
    csk_qty = to_int(derived.get("csk_qty")) or 0

    finish_flags: list[str] = []
    raw_finish = derived.get("finish_flags") or geo.get("finish_flags")
    if isinstance(raw_finish, (list, tuple, set)):
        finish_flags = [str(flag).strip().upper() for flag in raw_finish if str(flag).strip()]

    needs_back_face = bool(
        derived.get("needs_back_face")
        or geo.get("needs_back_face")
        or geo.get("from_back")
    )

    derived_summary: dict[str, Any] = {}
    for key in (
        "tap_minutes_hint",
        "cbore_minutes_hint",
        "csk_minutes_hint",
        "tap_class_counts",
        "tap_details",
        "npt_qty",
        "max_hole_depth_in",
        "plate_area_in2",
        "finish_flags",
        "stock_guess",
        "has_ldr_notes",
        "has_tight_tol",
        "dfm_geo",
        "tolerance_inputs",
        "default_tolerance_note",
        "stock_catalog",
        "machine_limits",
        "fixture_plan",
        "fai_required",
        "pocket_area_total_in2",
        "slot_count",
        "edge_len_in",
        "hole_table_source",
    ):
        value = derived.get(key)
        if value in (None, ""):
            continue
        if key in {"tap_minutes_hint", "cbore_minutes_hint", "csk_minutes_hint"}:
            cleaned = to_float(value)
        elif key in {"npt_qty", "slot_count"}:
            cleaned = to_int(value)
        elif key in {"has_ldr_notes"}:
            cleaned = bool(value)
        else:
            cleaned = _clean_nested(value, limit=12)
        if cleaned not in (None, "", [], {}):
            derived_summary[key] = cleaned

    if finish_flags and "finish_flags" not in derived_summary:
        derived_summary["finish_flags"] = finish_flags

    bbox_mm = _clean_nested(geo.get("bbox_mm"), limit=6)

    hole_groups: list[dict[str, Any]] = []
    raw_groups = geo.get("hole_groups")
    if isinstance(raw_groups, (list, tuple)):
        for idx, entry in enumerate(raw_groups):
            if idx >= 12:
                break
            if not isinstance(entry, dict):
                continue
            cleaned_entry = {
                "dia_mm": to_float(entry.get("dia_mm")),
                "depth_mm": to_float(entry.get("depth_mm")),
                "through": bool(entry.get("through")) if entry.get("through") is not None else None,
                "count": to_int(entry.get("count")),
            }
            hole_groups.append(compact_dict(cleaned_entry, drop_values=(None, "")))

    geo_notes: list[str] = []
    raw_notes = geo.get("notes")
    if isinstance(raw_notes, (list, tuple, set)):
        geo_notes = [str(note).strip() for note in raw_notes if str(note).strip()][:8]

    gdt_counts: dict[str, int] = {}
    raw_gdt = geo.get("gdt")
    if isinstance(raw_gdt, dict):
        for key, value in raw_gdt.items():
            val = to_int(value)
            if val:
                gdt_counts[str(key)] = val

    baseline_hours_raw = baseline.get("process_hours") if isinstance(baseline.get("process_hours"), dict) else {}
    baseline_hours: dict[str, float] = {}
    for proc, hours in (baseline_hours_raw or {}).items():
        val = to_float(hours)
        if val is not None:
            baseline_hours[str(proc)] = float(val)

    baseline_pass_raw = baseline.get("pass_through") if isinstance(baseline.get("pass_through"), dict) else {}
    baseline_pass = _canonicalize_pass_through_map(baseline_pass_raw)

    top_process_hours = sorted(baseline_hours.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    top_pass_through = sorted(baseline_pass.items(), key=lambda kv: (-kv[1], kv[0]))[:6]

    baseline_summary = {
        "scrap_pct": to_float(baseline.get("scrap_pct")) or 0.0,
        "setups": to_int(baseline.get("setups")) or 1,
        "fixture": baseline.get("fixture"),
        "process_hours": baseline_hours,
        "pass_through": baseline_pass,
        "top_process_hours": top_process_hours,
        "top_pass_through": top_pass_through,
    }

    rates_of_interest = {
        key: to_float(rates.get(key))
        for key in (
            "MillingRate",
            "TurningRate",
            "WireEDMRate",
            "SinkerEDMRate",
            "SurfaceGrindRate",
            "InspectionRate",
            "FixtureBuildRate",
            "AssemblyRate",
            "PackagingRate",
            "DeburrRate",
            "DrillingRate",
        )
        if rates.get(key) is not None and to_float(rates.get(key)) is not None
    }

    signals: dict[str, Any] = {
        "hole_bins_top": hole_bins_top,
        "tap_qty": tap_qty,
        "cbore_qty": cbore_qty,
        "csk_qty": csk_qty,
        "needs_back_face": needs_back_face,
    }

    for key in (
        "tap_minutes_hint",
        "cbore_minutes_hint",
        "csk_minutes_hint",
        "tap_class_counts",
        "tap_details",
        "npt_qty",
        "stock_guess",
        "has_tight_tol",
        "dfm_geo",
        "tolerance_inputs",
        "default_tolerance_note",
        "stock_catalog",
        "machine_limits",
        "fixture_plan",
        "fai_required",
        "pocket_area_total_in2",
        "slot_count",
        "edge_len_in",
    ):
        if key in derived_summary:
            signals[key] = derived_summary[key]

    if gdt_counts:
        signals["gdt_counts"] = gdt_counts

    geo_summary = {
        "material": material_name,
        "thickness_mm": thickness_mm,
        "hole_count": hole_count,
        "finish_flags": finish_flags,
        "needs_back_face": needs_back_face,
        "bbox_mm": bbox_mm,
        "hole_groups": hole_groups,
        "notes": geo_notes,
        "derived": derived_summary,
        "gdt": gdt_counts,
    }

    if geo.get("meta"):
        geo_summary["meta"] = _clean_nested(geo.get("meta"), limit=12)
    if geo.get("provenance"):
        geo_summary["provenance"] = _clean_nested(geo.get("provenance"), limit=12)

    coerced_bounds = coerce_bounds(bounds)
    bounds_summary = {
        "mult_min": coerced_bounds["mult_min"],
        "mult_max": coerced_bounds["mult_max"],
        "adder_max_hr": coerced_bounds["adder_max_hr"],
        "scrap_min": coerced_bounds["scrap_min"],
        "scrap_max": coerced_bounds["scrap_max"],
    }

    seed_extra: dict[str, Any] = {}
    dfm_geo_summary = derived_summary.get("dfm_geo")
    if isinstance(dfm_geo_summary, dict) and dfm_geo_summary:
        dfm_bits: list[str] = []
        min_wall = to_float(dfm_geo_summary.get("min_wall_mm"))
        if min_wall is not None:
            dfm_bits.append(f"min_wall≈{min_wall:.1f}mm")
        if dfm_geo_summary.get("thin_wall"):
            dfm_bits.append("thin_walls")
        unique_normals = to_int(dfm_geo_summary.get("unique_normals"))
        if unique_normals:
            dfm_bits.append(f"{unique_normals} normals")
        deburr_edge = to_float(dfm_geo_summary.get("deburr_edge_len_mm"))
        if deburr_edge and deburr_edge > 0:
            dfm_bits.append(f"deburr_edge≈{deburr_edge:.0f}mm")
        face_count = to_int(dfm_geo_summary.get("face_count"))
        if face_count and face_count > 0:
            dfm_bits.append(f"{face_count} faces")
        if dfm_bits:
            seed_extra["dfm_summary"] = dfm_bits[:5]

    tol_inputs_summary = derived_summary.get("tolerance_inputs")
    if isinstance(tol_inputs_summary, dict) and tol_inputs_summary:
        tol_labels = [str(k).strip() for k in tol_inputs_summary.keys() if str(k).strip()]
        if tol_labels:
            seed_extra["tolerance_focus"] = sorted(tol_labels)[:6]

    has_tight_tol_seed = derived_summary.get("has_tight_tol")
    if has_tight_tol_seed is not None:
        seed_extra["has_tight_tol"] = bool(has_tight_tol_seed)

    default_tol_note = derived_summary.get("default_tolerance_note")
    if isinstance(default_tol_note, str) and default_tol_note.strip():
        seed_extra["default_tolerance_note"] = default_tol_note.strip()[:160]

    if derived_summary.get("fai_required") is not None:
        seed_extra["fai_required"] = bool(derived_summary.get("fai_required"))

    stock_catalog_summary = derived_summary.get("stock_catalog")
    stock_focus: list[str] = []
    if isinstance(stock_catalog_summary, dict):
        for entry in stock_catalog_summary.values():
            label = None
            if isinstance(entry, dict):
                label = entry.get("item") or entry.get("name") or entry.get("stock")
            elif isinstance(entry, str):
                label = entry
            if label:
                label = str(label).strip()
            if label:
                stock_focus.append(label)
        if not stock_focus:
            stock_focus = [str(k).strip() for k in stock_catalog_summary.keys() if str(k).strip()]
    elif isinstance(stock_catalog_summary, (list, tuple, set)):
        for entry in stock_catalog_summary:
            label = str(entry).strip()
            if label:
                stock_focus.append(label)
    if stock_focus:
        seed_extra["stock_focus"] = stock_focus[:6]

    seed = {
        "top_process_hours": top_process_hours,
        "top_pass_through": top_pass_through,
        "hole_count": hole_count,
        "setups": baseline_summary["setups"],
        "finish_flags": finish_flags,
    }
    if seed_extra:
        seed.update(seed_extra)

    payload = {
        "geo": geo_summary,
        "baseline": baseline_summary,
        "signals": signals,
        "rates": rates_of_interest,
        "bounds": bounds_summary,
        "seed": seed,
    }

    return payload







def sanitize_suggestions(s: dict, bounds: dict) -> dict:
    coerced_bounds = coerce_bounds(bounds)

    mult_min = coerced_bounds["mult_min"]
    mult_max = coerced_bounds["mult_max"]
    base_adder_max = coerced_bounds["adder_max_hr"]
    scrap_min = coerced_bounds["scrap_min"]
    scrap_max = coerced_bounds["scrap_max"]

    bucket_caps = coerced_bounds.get("adder_bucket_max", {})

    meta_info: dict[str, Any] = {}

    def _normalize_conf(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            mapping = {
                "low": 0.3,
                "medium": 0.6,
                "med": 0.6,
                "mid": 0.6,
                "high": 0.85,
                "very high": 0.95,
                "certain": 0.98,
            }
            key = value.strip().lower()
            if key in mapping:
                return mapping[key]
        conf = to_float(value)
        if conf is None:
            return None
        return max(0.0, min(1.0, float(conf)))

    def _extract_detail(raw: Any) -> tuple[Any, dict[str, Any]]:
        detail: dict[str, Any] = {}
        value = raw
        if isinstance(raw, dict):
            if "value" in raw:
                value = raw.get("value")
            reason = str(raw.get("reason") or "").strip()
            if reason:
                detail["reason"] = reason[:160]
            conf = _normalize_conf(raw.get("confidence"))
            if conf is not None:
                detail["confidence"] = conf
            source_raw = raw.get("source") or raw.get("sources")
            sources: list[str] = []
            if isinstance(source_raw, str):
                sources = [source_raw]
            elif isinstance(source_raw, (list, tuple, set)):
                sources = list(source_raw)
            cleaned_sources = []
            for src in sources:
                if not src:
                    continue
                cleaned_sources.append(str(src).strip().upper()[:24])
            if cleaned_sources:
                detail["source"] = cleaned_sources
        return value, detail

    def _store_meta(path: tuple[str, ...], detail: dict[str, Any], value: Any) -> None:
        cleaned = compact_dict(detail)
        if not cleaned:
            return
        cleaned["value"] = value
        node = meta_info
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = cleaned

    def _extract_float_field(
        raw: Any, lo: float | None, hi: float | None, path: tuple[str, ...]
    ) -> float | None:
        if raw is None:
            return None
        value, detail = _extract_detail(raw)
        num = to_float(value)
        if num is None:
            return None
        if lo is not None:
            num = max(lo, num)
        if hi is not None:
            num = min(hi, num)
        _store_meta(path, detail, num)
        return num

    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"y", "yes", "true", "1"}:
                return True
            if lowered in {"n", "no", "false", "0"}:
                return False
        return None

    def _extract_bool_field(raw: Any, path: tuple[str, ...]) -> bool | None:
        if raw is None:
            return None
        value, detail = _extract_detail(raw)
        flag = _coerce_bool(value)
        if flag is None:
            return None
        _store_meta(path, detail, flag)
        return flag

    # Flatten optional setup block into primary keys for backward compatibility.
    setup_block: dict[str, Any] | None = None
    setup_reco = s.get("setup_recommendation")
    if isinstance(setup_reco, Mapping):
        setup_block = dict(setup_reco)
    else:
        setup_plan = s.get("setup_plan")
        if isinstance(setup_plan, Mapping):
            setup_block = dict(setup_plan)
    if setup_block:
        for key in ("setups", "fixture", "notes"):
            if key not in s and setup_block.get(key) is not None:
                s[key] = setup_block.get(key)
        if "fixture_build_hr" in setup_block and "fixture_build_hr" not in s:
            s["fixture_build_hr"] = setup_block.get("fixture_build_hr")
    mults: dict[str, float] = {}
    for proc, raw_val in (s.get("process_hour_multipliers") or {}).items():
        value, detail = _extract_detail(raw_val)
        num = to_float(value)
        if num is None:
            continue
        num = max(mult_min, min(mult_max, num))
        proc_key = str(proc)
        mults[proc_key] = num
        _store_meta(("process_hour_multipliers", proc_key), detail, num)

    adders: dict[str, float] = {}
    for proc, raw_val in (s.get("process_hour_adders") or {}).items():
        value, detail = _extract_detail(raw_val)
        num = to_float(value)
        if num is None:
            continue
        proc_key = str(proc)
        bucket_cap = bucket_caps.get(proc_key.lower())
        limit = bucket_cap if bucket_cap is not None else base_adder_max
        if limit is None:
            limit = LLM_ADDER_MAX
        limit = max(0.0, float(limit))
        num = max(0.0, min(limit, num))
        adders[proc_key] = num
        _store_meta(("process_hour_adders", proc_key), detail, num)

    raw_scrap = s.get("scrap_pct", 0.0)
    scrap_val, scrap_detail = _extract_detail(raw_scrap)
    scrap_float = to_float(scrap_val)
    if scrap_float is None:
        scrap_float = 0.0
    scrap_float = max(scrap_min, min(scrap_max, scrap_float))
    _store_meta(("scrap_pct",), scrap_detail, scrap_float)

    raw_setups = s.get("setups", 1)
    setups_val, setups_detail = _extract_detail(raw_setups)
    try:
        setups_int = int(round(float(setups_val)))
    except Exception:
        setups_int = 1
    setups_int = max(1, min(4, setups_int))
    _store_meta(("setups",), setups_detail, setups_int)

    raw_fixture = s.get("fixture", "standard")
    fixture_val, fixture_detail = _extract_detail(raw_fixture)
    fixture_str = str(fixture_val).strip() if fixture_val is not None else "standard"
    if not fixture_str:
        fixture_str = "standard"
    fixture_str = fixture_str[:120]
    _store_meta(("fixture",), fixture_detail, fixture_str)

    notes_raw = s.get("notes") or []
    notes: list[str] = []
    for note in notes_raw:
        if isinstance(note, dict):
            value, detail = _extract_detail(note)
            text = str(value).strip()
            if text:
                trimmed = text[:160]
                notes.append(trimmed)
                _store_meta(("notes", str(len(notes) - 1)), detail, trimmed)
            continue
        if not isinstance(note, str):
            continue
        cleaned = note.strip()
        if cleaned:
            notes.append(cleaned[:160])

    raw_no_change = s.get("no_change_reason")
    no_change_val, no_change_detail = _extract_detail(raw_no_change)
    no_change_str = str(no_change_val or "")
    if no_change_str.strip():
        _store_meta(("no_change_reason",), no_change_detail, no_change_str.strip())

    extra: dict[str, Any] = {}

    fixture_build_raw = s.get("fixture_build_hr")
    if fixture_build_raw is None and setup_block:
        fixture_build_raw = setup_block.get("fixture_build_hr")
    fixture_build_hr = _extract_float_field(fixture_build_raw, 0.0, 2.0, ("fixture_build_hr",))
    if fixture_build_hr is not None:
        extra["fixture_build_hr"] = fixture_build_hr

    soft_block = s.get("soft_jaw_plan") if isinstance(s.get("soft_jaw_plan"), dict) else None
    soft_hr_raw = s.get("soft_jaw_hr")
    soft_cost_raw = s.get("soft_jaw_material_cost")
    if soft_block:
        if soft_hr_raw is None:
            soft_hr_raw = soft_block.get("hours") or soft_block.get("hr")
        if soft_cost_raw is None:
            soft_cost_raw = soft_block.get("stock_cost") or soft_block.get("material_cost")
    soft_hr = _extract_float_field(soft_hr_raw, 0.0, 1.0, ("soft_jaw_hr",))
    if soft_hr is not None:
        extra["soft_jaw_hr"] = soft_hr
    soft_cost = _extract_float_field(soft_cost_raw, 0.0, 60.0, ("soft_jaw_material_cost",))
    if soft_cost is not None:
        extra["soft_jaw_material_cost"] = soft_cost

    op_block = s.get("operation_sequence")
    op_steps_raw: Any = None
    op_handling_raw: Any = None
    if isinstance(op_block, dict):
        op_steps_raw = op_block.get("ops") or op_block.get("sequence")
        op_handling_raw = op_block.get("handling_adder_hr") or op_block.get("handling_hr")
    elif isinstance(op_block, (list, tuple, set)):
        op_steps_raw = op_block
    if op_handling_raw is None:
        op_handling_raw = s.get("handling_adder_hr")
    if isinstance(op_steps_raw, (list, tuple, set)):
        op_steps_clean = [str(step).strip()[:80] for step in op_steps_raw if str(step).strip()]
        if op_steps_clean:
            extra["operation_sequence"] = op_steps_clean[:12]
    handling_hr = _extract_float_field(op_handling_raw, 0.0, 0.2, ("handling_adder_hr",))
    if handling_hr is not None:
        extra["handling_adder_hr"] = handling_hr

    drilling_block = s.get("drilling_strategy") or s.get("drilling_plan")
    if isinstance(drilling_block, dict):
        drilling_clean: dict[str, Any] = {}
        mult_val, mult_detail = _extract_detail(drilling_block.get("multiplier"))
        mult = to_float(mult_val)
        if mult is not None:
            mult = max(0.8, min(1.5, mult))
            drilling_clean["multiplier"] = mult
            _store_meta(("drilling_strategy", "multiplier"), mult_detail, mult)
        floor_val, floor_detail = _extract_detail(
            drilling_block.get("per_hole_floor_sec") or drilling_block.get("floor_sec_per_hole")
        )
        floor = to_float(floor_val)
        if floor is not None:
            floor = max(0.0, floor)
            drilling_clean["per_hole_floor_sec"] = floor
            _store_meta(("drilling_strategy", "per_hole_floor_sec"), floor_detail, floor)
        if drilling_block.get("note") or drilling_block.get("reason"):
            note_text = str(drilling_block.get("note") or drilling_block.get("reason")).strip()
            if note_text:
                drilling_clean["note"] = note_text[:160]
        if drilling_clean:
            extra["drilling_strategy"] = drilling_clean

    cmm_raw = s.get("cmm_minutes") or s.get("cmm_min")
    cmm_minutes = _extract_float_field(cmm_raw, 0.0, 60.0, ("cmm_minutes",))
    if cmm_minutes is not None:
        extra["cmm_minutes"] = cmm_minutes

    inproc_raw = s.get("in_process_inspection_hr")
    inproc_hr = _extract_float_field(inproc_raw, 0.0, 0.5, ("in_process_inspection_hr",))
    if inproc_hr is not None:
        extra["in_process_inspection_hr"] = inproc_hr

    fai_flag = _extract_bool_field(s.get("fai_required"), ("fai_required",))
    if fai_flag is not None:
        extra["fai_required"] = fai_flag

    fai_prep = _extract_float_field(s.get("fai_prep_hr"), 0.0, 1.0, ("fai_prep_hr",))
    if fai_prep is not None:
        extra["fai_prep_hr"] = fai_prep

    packaging_hr = _extract_float_field(s.get("packaging_hours"), 0.0, 0.5, ("packaging_hours",))
    if packaging_hr is not None:
        extra["packaging_hours"] = packaging_hr

    packaging_cost = _extract_float_field(s.get("packaging_flat_cost"), 0.0, 25.0, ("packaging_flat_cost",))
    if packaging_cost is not None:
        extra["packaging_flat_cost"] = packaging_cost
    shipping_override_val = _extract_float_field(s.get("shipping_cost"), 0.0, None, ("shipping_cost",))
    if shipping_override_val is not None:
        extra["shipping_cost"] = shipping_override_val

    shipping_hint = s.get("shipping_hint") or s.get("shipping_class")
    if isinstance(shipping_hint, dict):
        value, detail = _extract_detail(shipping_hint)
        hint = str(value).strip()
        if hint:
            extra["shipping_hint"] = hint[:80]
            _store_meta(("shipping_hint",), detail, hint[:80])
    elif isinstance(shipping_hint, str) and shipping_hint.strip():
        extra["shipping_hint"] = shipping_hint.strip()[:80]

    sanitized = {
        "process_hour_multipliers": mults or {"drilling": 1.0, "milling": 1.0},
        "process_hour_adders": adders or {"inspection": 0.0},
        "scrap_pct": scrap_float,
        "setups": setups_int,
        "fixture": fixture_str,
        "notes": notes,
        "no_change_reason": no_change_str,
    }

    if extra:
        sanitized.update(extra)

    if meta_info:
        sanitized["_meta"] = meta_info

    return sanitized


def apply_suggestions(baseline: dict, s: dict) -> dict:
    eff = copy.deepcopy(baseline or {})
    raw_process_hours = eff.get("process_hours")
    base_hours: dict[Any, Any] = raw_process_hours if isinstance(raw_process_hours, dict) else {}
    ph = {k: float(to_float(v) or 0.0) for k, v in base_hours.items()}

    for proc, mult in (s.get("process_hour_multipliers") or {}).items():
        if proc in ph:
            try:
                ph[proc] = ph[proc] * float(mult)
            except Exception:
                ph[proc] = ph[proc]

    for proc, add in (s.get("process_hour_adders") or {}).items():
        base = ph.get(proc, 0.0)
        try:
            ph[proc] = float(base) + float(add)
        except Exception:
            ph[proc] = float(base)

    eff["process_hours"] = ph
    eff["scrap_pct"] = s.get("scrap_pct", eff.get("scrap_pct"))
    eff["setups"] = s.get("setups", eff.get("setups", 1))
    eff["fixture"] = s.get("fixture", eff.get("fixture", "standard"))
    notes = list(s.get("notes") or [])
    if s.get("no_change_reason"):
        notes.append(f"no_change: {s['no_change_reason']}")
    eff["_llm_notes"] = notes
    return eff

def _coerce_user_value(raw: Any, kind: str) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None
    try:
        if kind in {"float", "currency", "percent", "multiplier", "hours"}:
            return float(raw)
        if kind == "int":
            return int(round(float(raw)))
    except Exception:
        return None
    if kind == "text":
        return str(raw)
    return raw


def _format_value(value: Any, kind: str) -> str:
    if value is None:
        return "–"
    try:
        if kind == "percent":
            return f"{float(value) * 100:.1f}%"
        if kind in {"float", "hours", "multiplier"}:
            return f"{float(value):.3f}"
        if kind == "currency":
            return f"${float(value):,.2f}"
        if kind == "int":
            return f"{int(round(float(value)))}"
    except Exception:
        return str(value)
    return str(value)


def _format_entry_value(value: Any, kind: str) -> str:
    if value is None:
        return ""
    try:
        if kind == "int":
            return str(int(round(float(value))))
        if kind == "currency":
            return f"{float(value):.2f}"
        if kind in {"percent", "multiplier", "hours", "float"}:
            return f"{float(value):.3f}"
    except Exception:
        return str(value)
    return str(value)


def _collect_process_keys(*dicts: Mapping[str, Any] | None) -> set[str]:
    keys: set[str] = set()
    for d in dicts:
        if isinstance(d, Mapping):
            keys.update(str(k) for k in d.keys())
    return keys


def merge_effective(
    baseline: dict | None,
    suggestions: dict | None,
    overrides: dict | None,
    *,
    guard_ctx: dict | None = None,
) -> dict:
    """Tri-state merge for baseline vs LLM suggestions vs user overrides."""

    baseline = copy.deepcopy(baseline or {})
    suggestions = dict(suggestions or {})
    overrides = dict(overrides or {})
    guard_ctx = dict(guard_ctx or {})

    bounds = baseline.get("_bounds") if isinstance(baseline, dict) else None
    if isinstance(bounds, dict):
        bounds = {str(k): v for k, v in bounds.items()}
    else:
        bounds = {}

    def _clamp(value: float, kind: str, label: str, source: str) -> tuple[float, bool]:
        clamped = value
        changed = False
        source_norm = str(source).strip().lower()
        if kind == "multiplier":
            mult_min_val = to_float(bounds.get("mult_min"))
            mult_max_val = to_float(bounds.get("mult_max"))
            mult_min = max(
                LLM_MULTIPLIER_MIN,
                mult_min_val if mult_min_val is not None else LLM_MULTIPLIER_MIN,
            )
            mult_max = min(
                LLM_MULTIPLIER_MAX,
                mult_max_val if mult_max_val is not None else LLM_MULTIPLIER_MAX,
            )
            mult_max = max(mult_max, mult_min)
            clamped = max(mult_min, min(mult_max, float(value)))
        elif kind == "adder":
            orig_val = float(value)
            add_hr_max_raw = to_float(bounds.get("add_hr_max"))
            adder_max_raw = to_float(bounds.get("adder_max_hr"))
            adder_max_candidates = [
                LLM_ADDER_MAX,
                adder_max_raw if adder_max_raw is not None else None,
                add_hr_max_raw if add_hr_max_raw is not None else None,
            ]
            bucket_caps_raw = bounds.get("adder_bucket_max") or bounds.get("add_hr_bucket_max")
            bucket_caps: dict[str, float] = {}
            if isinstance(bucket_caps_raw, dict):
                for key, raw in bucket_caps_raw.items():
                    cap_val = to_float(raw)
                    if cap_val is not None:
                        bucket_caps[str(key).lower()] = float(cap_val)
            adder_min = to_float(bounds.get("add_hr_min"))
            if adder_min is None:
                adder_min = to_float(bounds.get("adder_min_hr"))
            raw_val = orig_val
            if source_norm == "llm" and raw_val > 240:
                raw_val = raw_val / 60.0
            lower_bound = adder_min if adder_min is not None else 0.0
            bucket_name = None
            if "[" in label and "]" in label:
                bucket_name = label.split("[", 1)[-1].split("]", 1)[0].strip().lower()
            bucket_max = None
            if bucket_name:
                bucket_max = bucket_caps.get(bucket_name)
            if bucket_max is not None:
                adder_max_effective = bucket_max
            else:
                filtered = [cand for cand in adder_max_candidates if cand is not None]
                adder_max_effective = min(filtered) if filtered else LLM_ADDER_MAX
            adder_max_effective = max(adder_max_effective, lower_bound)
            clamped = max(lower_bound, min(adder_max_effective, raw_val))
        elif kind == "scrap":
            scrap_min = max(0.0, to_float(bounds.get("scrap_min")) or 0.0)
            scrap_max = to_float(bounds.get("scrap_max")) or 0.25
            clamped = max(scrap_min, min(scrap_max, float(value)))
        elif kind == "setups":
            clamped = int(max(1, min(4, round(float(value)))))
        if not math.isclose(float(clamped), float(value), rel_tol=1e-6, abs_tol=1e-6):
            note = f"{label} {float(value):.3f} → {float(clamped):.3f} ({source})"
            clamp_notes.append(note)
            changed = True
        return clamped, changed

    def _clamp_range(value: float, lo: float | None, hi: float | None, label: str, source: str) -> tuple[float, bool]:
        num = float(value)
        changed = False
        orig = num
        if lo is not None and num < lo:
            num = lo
            changed = True
        if hi is not None and num > hi:
            num = hi
            changed = True
        if changed:
            clamp_notes.append(f"{label} {orig:.3f} → {num:.3f} ({source})")
        return num, changed

    def _merge_numeric_field(key: str, lo: float | None, hi: float | None, label: str) -> None:
        base_val = to_float(baseline.get(key)) if baseline.get(key) is not None else None
        value = base_val
        source = "baseline"
        if overrides.get(key) is not None:
            cand = to_float(overrides.get(key))
            if cand is not None:
                value, _ = _clamp_range(cand, lo, hi, label, "user override")
                source = "user"
        elif suggestions.get(key) is not None:
            cand = to_float(suggestions.get(key))
            if cand is not None:
                value, _ = _clamp_range(cand, lo, hi, label, "LLM")
                source = "llm"
        if value is not None:
            eff[key] = float(value)
        elif key in eff:
            eff.pop(key, None)
        source_tags[key] = source

    def _coerce_bool_value(raw: Any) -> bool | None:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"y", "yes", "true", "1"}:
                return True
            if lowered in {"n", "no", "false", "0"}:
                return False
        return None

    def _merge_bool_field(key: str) -> None:
        base_val = baseline.get(key) if isinstance(baseline.get(key), bool) else None
        value = base_val
        source = "baseline"
        if key in overrides:
            cand = _coerce_bool_value(overrides.get(key))
            if cand is not None:
                value = cand
                source = "user"
        elif key in suggestions:
            cand = _coerce_bool_value(suggestions.get(key))
            if cand is not None:
                value = cand
                source = "llm"
        if value is not None:
            eff[key] = bool(value)
        elif key in eff:
            eff.pop(key, None)
        source_tags[key] = source

    def _merge_text_field(key: str, *, max_len: int = 160) -> None:
        base_val = baseline.get(key) if isinstance(baseline.get(key), str) else None
        value = base_val
        source = "baseline"
        override_val = overrides.get(key)
        if isinstance(override_val, str):
            override_stripped = override_val.strip()
            if override_stripped:
                value = override_stripped[:max_len]
                source = "user"
        else:
            suggestion_val = suggestions.get(key)
            if isinstance(suggestion_val, str):
                suggestion_stripped = suggestion_val.strip()
                if suggestion_stripped:
                    value = suggestion_stripped[:max_len]
                    source = "llm"
        if value is not None:
            eff[key] = value
        elif key in eff:
            eff.pop(key, None)
        source_tags[key] = source

    def _merge_list_field(key: str) -> None:
        base_val = baseline.get(key) if isinstance(baseline.get(key), list) else None
        value = base_val
        source = "baseline"
        override_val = overrides.get(key)
        if isinstance(override_val, list):
            cleaned_override: list[str] = []
            for item in override_val:
                text = str(item).strip()
                if text:
                    cleaned_override.append(text[:80])
            value = cleaned_override
            source = "user"
        else:
            suggestion_list = suggestions.get(key)
            if isinstance(suggestion_list, list):
                cleaned_suggestion: list[str] = []
                for item in suggestion_list:
                    text = str(item).strip()
                    if text:
                        cleaned_suggestion.append(text[:80])
                value = cleaned_suggestion
                source = "llm"
        if value:
            eff[key] = value
        elif key in eff:
            eff.pop(key, None)
        source_tags[key] = source

    def _merge_dict_field(key: str) -> None:
        base_val = baseline.get(key) if isinstance(baseline.get(key), dict) else None
        value = base_val
        source = "baseline"
        override_val = overrides.get(key)
        if isinstance(override_val, dict):
            value = copy.deepcopy(override_val)
            source = "user"
        else:
            suggestion_dict = suggestions.get(key)
            if isinstance(suggestion_dict, dict):
                value = copy.deepcopy(suggestion_dict)
                source = "llm"
        if value is not None:
            eff[key] = value
        elif key in eff:
            eff.pop(key, None)
        source_tags[key] = source

    eff = copy.deepcopy(baseline)
    eff.pop("_bounds", None)
    clamp_notes: list[str] = []
    source_tags: dict[str, Any] = {}

    raw_baseline_hours = baseline.get("process_hours")
    baseline_hours_raw = raw_baseline_hours if isinstance(raw_baseline_hours, dict) else {}
    baseline_hours: dict[str, float] = {}
    for proc, hours in (baseline_hours_raw or {}).items():
        val = to_float(hours)
        if val is not None:
            baseline_hours[str(proc)] = float(val)

    raw_sugg_mult = suggestions.get("process_hour_multipliers")
    sugg_mult: dict[str, Any] = raw_sugg_mult if isinstance(raw_sugg_mult, dict) else {}
    raw_over_mult = overrides.get("process_hour_multipliers")
    over_mult: dict[str, Any] = raw_over_mult if isinstance(raw_over_mult, dict) else {}
    mult_keys = sorted(_collect_process_keys(baseline_hours, sugg_mult, over_mult))
    final_hours: dict[str, float] = dict(baseline_hours)
    final_mults: dict[str, float] = {}
    mult_sources: dict[str, str] = {}
    for proc in mult_keys:
        base_hr = baseline_hours.get(proc, 0.0)
        source = "baseline"
        val = 1.0
        if proc in over_mult and over_mult[proc] is not None:
            cand = to_float(over_mult.get(proc))
            if cand is not None:
                val = float(cand)
                val, _ = _clamp(val, "multiplier", f"multiplier[{proc}]", "user override")
                source = "user"
        elif proc in sugg_mult and sugg_mult[proc] is not None:
            cand = to_float(sugg_mult.get(proc))
            if cand is not None:
                val = float(cand)
                val, _ = _clamp(val, "multiplier", f"multiplier[{proc}]", "LLM")
                source = "llm"
        final_mults[proc] = float(val)
        mult_sources[proc] = source
        final_hours[proc] = float(base_hr) * float(val)

    raw_sugg_add = suggestions.get("process_hour_adders")
    sugg_add: dict[str, Any] = raw_sugg_add if isinstance(raw_sugg_add, dict) else {}
    raw_over_add = overrides.get("process_hour_adders")
    over_add: dict[str, Any] = raw_over_add if isinstance(raw_over_add, dict) else {}
    add_keys = sorted(_collect_process_keys(sugg_add, over_add))
    final_adders: dict[str, float] = {}
    add_sources: dict[str, str] = {}
    for proc in add_keys:
        source = "baseline"
        add_val = 0.0
        if proc in over_add and over_add[proc] is not None:
            cand = to_float(over_add.get(proc))
            if cand is not None:
                add_val = float(cand)
                add_val, _ = _clamp(add_val, "adder", f"adder[{proc}]", "user override")
                source = "user"
        elif proc in sugg_add and sugg_add[proc] is not None:
            cand = to_float(sugg_add.get(proc))
            if cand is not None:
                add_val = float(cand)
                add_val, _ = _clamp(add_val, "adder", f"adder[{proc}]", "LLM")
                source = "llm"
        if not math.isclose(add_val, 0.0, abs_tol=1e-9):
            final_adders[proc] = add_val
            final_hours[proc] = final_hours.get(proc, 0.0) + add_val
        add_sources[proc] = source

    raw_sugg_pass_candidate = suggestions.get("add_pass_through")
    raw_sugg_pass = raw_sugg_pass_candidate if isinstance(raw_sugg_pass_candidate, dict) else {}
    raw_over_pass_candidate = overrides.get("add_pass_through")
    raw_over_pass = raw_over_pass_candidate if isinstance(raw_over_pass_candidate, dict) else {}
    sugg_pass = _canonicalize_pass_through_map(raw_sugg_pass)
    over_pass = _canonicalize_pass_through_map(raw_over_pass)
    pass_keys = sorted(set(sugg_pass) | set(over_pass))
    final_pass: dict[str, float] = {}
    pass_sources: dict[str, str] = {}
    for key in pass_keys:
        source = "baseline"
        val = 0.0
        if key in over_pass:
            val = float(over_pass[key])
            source = "user"
        elif key in sugg_pass:
            val = float(sugg_pass[key])
            source = "llm"
        if not math.isclose(val, 0.0, abs_tol=1e-9):
            final_pass[key] = val
        pass_sources[key] = source

    hole_count_guard = _coerce_float_or_none(guard_ctx.get("hole_count"))
    try:
        hole_count_guard_int = int(round(float(hole_count_guard))) if hole_count_guard is not None else 0
    except Exception:
        hole_count_guard_int = 0
    min_sec_per_hole = to_float(guard_ctx.get("min_sec_per_hole"))
    min_sec_per_hole = float(min_sec_per_hole) if min_sec_per_hole is not None else 9.0
    if hole_count_guard_int > 0 and "drilling" in final_hours:
        current_drill = to_float(final_hours.get("drilling"))
        if current_drill is not None:
            drill_floor_hr = (hole_count_guard_int * min_sec_per_hole) / 3600.0
            if current_drill < drill_floor_hr - 1e-6:
                clamp_notes.append(
                    f"process_hours[drilling] {current_drill:.3f} → {drill_floor_hr:.3f} (guardrail)"
                )
                final_hours["drilling"] = drill_floor_hr

    tap_qty_guard = _coerce_float_or_none(guard_ctx.get("tap_qty"))
    try:
        tap_qty_guard_int = int(round(float(tap_qty_guard))) if tap_qty_guard is not None else 0
    except Exception:
        tap_qty_guard_int = 0
    min_min_per_tap = to_float(guard_ctx.get("min_min_per_tap"))
    min_min_per_tap = float(min_min_per_tap) if min_min_per_tap is not None else 0.2
    if tap_qty_guard_int > 0 and "tapping" in final_hours:
        current_tap = to_float(final_hours.get("tapping"))
        if current_tap is not None:
            tap_floor_hr = (tap_qty_guard_int * min_min_per_tap) / 60.0
            if current_tap < tap_floor_hr - 1e-6:
                clamp_notes.append(
                    f"process_hours[tapping] {current_tap:.3f} → {tap_floor_hr:.3f} (guardrail)"
                )
                final_hours["tapping"] = tap_floor_hr

    eff["process_hour_multipliers"] = final_mults
    if mult_sources:
        source_tags["process_hour_multipliers"] = mult_sources
    eff["process_hour_adders"] = final_adders
    if add_sources:
        source_tags["process_hour_adders"] = add_sources
    if final_pass:
        eff["add_pass_through"] = final_pass
    if pass_sources:
        source_tags["add_pass_through"] = pass_sources
    eff["process_hours"] = {k: float(v) for k, v in final_hours.items() if abs(float(v)) > 1e-9}

    scrap_base = baseline.get("scrap_pct")
    scrap_user = overrides.get("scrap_pct") or overrides.get("scrap_pct_override")
    scrap_sugg = suggestions.get("scrap_pct")
    scrap_source = "baseline"
    scrap_val = scrap_base if scrap_base is not None else 0.0
    if scrap_user is not None:
        cand = to_float(scrap_user)
        if cand is not None:
            scrap_val = float(cand)
            scrap_val, _ = _clamp(scrap_val, "scrap", "scrap_pct", "user override")
            scrap_source = "user"
    elif scrap_sugg is not None:
        cand = to_float(scrap_sugg)
        if cand is not None:
            scrap_val = float(cand)
            scrap_val, _ = _clamp(scrap_val, "scrap", "scrap_pct", "LLM")
            scrap_source = "llm"
    eff["scrap_pct"] = float(scrap_val)
    source_tags["scrap_pct"] = scrap_source

    contingency_base = baseline.get("contingency_pct")
    contingency_user = overrides.get("contingency_pct") or overrides.get("contingency_pct_override")
    contingency_sugg = suggestions.get("contingency_pct")
    contingency_source = "baseline"
    contingency_val = contingency_base if contingency_base is not None else None
    if contingency_user is not None:
        cand = to_float(contingency_user)
        if cand is not None:
            contingency_val = float(cand)
            contingency_val, _ = _clamp(contingency_val, "scrap", "contingency_pct", "user override")
            contingency_source = "user"
    elif contingency_sugg is not None:
        cand = to_float(contingency_sugg)
        if cand is not None:
            contingency_val = float(cand)
            contingency_val, _ = _clamp(contingency_val, "scrap", "contingency_pct", "LLM")
            contingency_source = "llm"
    if contingency_val is not None:
        eff["contingency_pct"] = float(contingency_val)
        source_tags["contingency_pct"] = contingency_source

    setups_base = baseline.get("setups") or 1
    setups_user = overrides.get("setups")
    setups_sugg = suggestions.get("setups")
    setups_source = "baseline"
    setups_val = setups_base
    if setups_user is not None:
        cand = to_float(setups_user)
        if cand is not None:
            setups_val, _ = _clamp(cand, "setups", "setups", "user override")
            setups_source = "user"
    elif setups_sugg is not None:
        cand = to_float(setups_sugg)
        if cand is not None:
            setups_val, _ = _clamp(cand, "setups", "setups", "LLM")
            setups_source = "llm"
    eff["setups"] = int(setups_val)
    source_tags["setups"] = setups_source

    fixture_base = baseline.get("fixture")
    fixture_user = overrides.get("fixture")
    fixture_sugg = suggestions.get("fixture")
    fixture_source = "baseline"
    fixture_val = fixture_base
    if isinstance(fixture_user, str) and fixture_user.strip():
        fixture_val = fixture_user.strip()
        fixture_source = "user"
    elif isinstance(fixture_sugg, str) and fixture_sugg.strip():
        fixture_val = fixture_sugg.strip()
        fixture_source = "llm"
    if fixture_val is not None:
        eff["fixture"] = fixture_val
    source_tags["fixture"] = fixture_source

    notes_val = []
    if isinstance(suggestions.get("notes"), list):
        notes_val.extend([str(n).strip() for n in suggestions["notes"] if isinstance(n, str) and n.strip()])
    if isinstance(overrides.get("notes"), list):
        notes_val.extend([str(n).strip() for n in overrides["notes"] if isinstance(n, str) and n.strip()])
    if notes_val:
        eff["notes"] = notes_val

    _merge_numeric_field("fixture_build_hr", 0.0, 2.0, "fixture_build_hr")
    _merge_numeric_field("soft_jaw_hr", 0.0, 1.0, "soft_jaw_hr")
    _merge_numeric_field("soft_jaw_material_cost", 0.0, 60.0, "soft_jaw_material_cost")
    _merge_numeric_field("handling_adder_hr", 0.0, 0.2, "handling_adder_hr")
    _merge_numeric_field("cmm_minutes", 0.0, 60.0, "cmm_minutes")
    _merge_numeric_field("in_process_inspection_hr", 0.0, 0.5, "in_process_inspection_hr")
    _merge_numeric_field("inspection_total_hr", 0.0, 12.0, "inspection_total_hr")
    _merge_bool_field("fai_required")
    _merge_numeric_field("fai_prep_hr", 0.0, 1.0, "fai_prep_hr")
    _merge_numeric_field("packaging_hours", 0.0, 0.5, "packaging_hours")
    _merge_numeric_field("packaging_flat_cost", 0.0, 25.0, "packaging_flat_cost")
    _merge_numeric_field("shipping_cost", 0.0, None, "shipping_cost")
    _merge_text_field("shipping_hint", max_len=80)
    _merge_list_field("operation_sequence")
    _merge_dict_field("drilling_strategy")

    if guard_ctx.get("needs_back_face"):
        current_setups = eff.get("setups")
        setups_int = to_int(current_setups) or 0
        if setups_int < 2:
            eff["setups"] = 2
            clamp_notes.append(f"setups {setups_int} → 2 (back-side guardrail)")
            source_tags["setups"] = "guardrail"

    finish_flags_ctx = guard_ctx.get("finish_flags")
    baseline_pass_ctx_raw = (
        guard_ctx.get("baseline_pass_through")
        if isinstance(guard_ctx.get("baseline_pass_through"), dict)
        else {}
    )
    baseline_pass_ctx = _canonicalize_pass_through_map(baseline_pass_ctx_raw)
    finish_pass_key = _canonical_pass_label(
        guard_ctx.get("finish_pass_key") or "Outsourced Vendors"
    )
    finish_floor = _as_float_or_none(guard_ctx.get("finish_cost_floor"))
    finish_floor = float(finish_floor) if finish_floor is not None else 50.0
    if finish_flags_ctx and finish_floor > 0:
        combined_pass: dict[str, float] = dict(baseline_pass_ctx)
        for key, value in (final_pass.items() if isinstance(final_pass, dict) else []):
            val = to_float(value)
            if val is not None:
                combined_pass[key] = combined_pass.get(key, 0.0) + float(val)
        current_finish_cost = combined_pass.get(finish_pass_key, 0.0)
        if current_finish_cost < finish_floor - 1e-6:
            needed = finish_floor - current_finish_cost
            if needed > 0:
                if not isinstance(final_pass, dict):
                    final_pass = {}
                final_pass[finish_pass_key] = float(final_pass.get(finish_pass_key, 0.0) or 0.0) + needed
                clamp_notes.append(
                    f"add_pass_through[{finish_pass_key}] {current_finish_cost:.2f} → {finish_floor:.2f} (finish guardrail)"
                )
                pass_sources[finish_pass_key] = "guardrail"
                eff["add_pass_through"] = final_pass
                source_tags["add_pass_through"] = pass_sources

    if clamp_notes:
        eff["_clamp_notes"] = clamp_notes
    eff["_source_tags"] = source_tags
    return eff


def compute_effective_state(state: QuoteState) -> tuple[dict, dict]:
    baseline = state.baseline or {}
    suggestions = state.suggestions or {}
    overrides = state.user_overrides or {}
    accept = state.accept_llm or {}

    bounds = state.bounds or baseline.get("_bounds") or {}

    applied: dict[str, Any] = {}

    def include_bucket(bucket: str) -> dict:
        data = suggestions.get(bucket)
        if not isinstance(data, dict):
            return {}
        acc_map = accept.get(bucket)
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(acc_map, dict):
                if not acc_map.get(key):
                    continue
            elif not accept.get(bucket):
                continue
            result[str(key)] = value
        return result

    for bucket in ("process_hour_multipliers", "process_hour_adders", "add_pass_through"):
        selected = include_bucket(bucket)
        if selected:
            applied[bucket] = selected

    scalar_keys = (
        "scrap_pct",
        "contingency_pct",
        "setups",
        "fixture",
        "fixture_build_hr",
        "soft_jaw_hr",
        "soft_jaw_material_cost",
        "handling_adder_hr",
        "cmm_minutes",
        "in_process_inspection_hr",
        "fai_required",
        "fai_prep_hr",
        "packaging_hours",
        "packaging_flat_cost",
        "shipping_hint",
    )
    for scalar_key in scalar_keys:
        if scalar_key in suggestions and accept.get(scalar_key):
            applied[scalar_key] = suggestions.get(scalar_key)

    if "notes" in suggestions:
        applied["notes"] = suggestions.get("notes")
    if accept.get("operation_sequence") and isinstance(suggestions.get("operation_sequence"), list):
        applied["operation_sequence"] = suggestions.get("operation_sequence")
    if accept.get("drilling_strategy") and isinstance(suggestions.get("drilling_strategy"), dict):
        applied["drilling_strategy"] = suggestions.get("drilling_strategy")

    baseline_for_merge = copy.deepcopy(baseline)
    if bounds:
        baseline_for_merge["_bounds"] = dict(bounds)

    merged = merge_effective(
        baseline_for_merge,
        applied,
        overrides,
        guard_ctx=getattr(state, "guard_context", None),
    )
    sources = merged.pop("_source_tags", {})
    clamp_notes = merged.pop("_clamp_notes", None)
    if clamp_notes:
        log = state.llm_raw.setdefault("clamp_notes", [])
        for note in clamp_notes:
            if note not in log:
                log.append(note)
    state.effective = merged
    state.effective_sources = sources
    return merged, sources

def reprice_with_effective(state: QuoteState) -> QuoteState:
    """Recompute effective values and enforce guardrails before pricing."""

    geo_ctx = state.geo or {}
    inner_geo_raw = geo_ctx.get("geo")
    inner_geo_ctx: dict[str, Any] = inner_geo_raw if isinstance(inner_geo_raw, dict) else {}
    hole_count_guard = _coerce_float_or_none(geo_ctx.get("hole_count"))
    if hole_count_guard is None:
        hole_count_guard = _coerce_float_or_none(inner_geo_ctx.get("hole_count"))
    tap_qty_guard = _coerce_float_or_none(geo_ctx.get("tap_qty"))
    if tap_qty_guard is None:
        tap_qty_guard = _coerce_float_or_none(inner_geo_ctx.get("tap_qty"))
    finish_flags_guard: set[str] = set()
    finishes_geo = geo_ctx.get("finishes") or inner_geo_ctx.get("finishes")
    if isinstance(finishes_geo, (list, tuple, set)):
        finish_flags_guard.update(str(flag).strip().upper() for flag in finishes_geo if isinstance(flag, str) and flag.strip())
    explicit_finish_flags = geo_ctx.get("finish_flags") or inner_geo_ctx.get("finish_flags")
    if isinstance(explicit_finish_flags, (list, tuple, set)):
        finish_flags_guard.update(str(flag).strip().upper() for flag in explicit_finish_flags if isinstance(flag, str) and flag.strip())
    guard_ctx: dict[str, Any] = {
        "hole_count": hole_count_guard,
        "tap_qty": tap_qty_guard,
        "min_sec_per_hole": 9.0,
        "min_min_per_tap": 0.2,
        "needs_back_face": bool(
            geo_ctx.get("needs_back_face")
            or geo_ctx.get("from_back")
            or inner_geo_ctx.get("needs_back_face")
            or inner_geo_ctx.get("from_back")
        ),
        "baseline_pass_through": (state.baseline.get("pass_through") if isinstance(state.baseline.get("pass_through"), dict) else {}),
    }
    if finish_flags_guard:
        guard_ctx["finish_flags"] = sorted(finish_flags_guard)
        guard_ctx.setdefault("finish_cost_floor", 50.0)
    state.guard_context = guard_ctx

    ensure_accept_flags(state)
    merged, sources = compute_effective_state(state)
    state.effective = merged
    state.effective_sources = sources

    # drilling floor guard
    eff_hours = state.effective.get("process_hours") if isinstance(state.effective.get("process_hours"), dict) else {}
    if eff_hours:
        try:
            hole_count_geo = int(float(state.geo.get("hole_count", 0) or 0))
        except Exception:
            hole_count_geo = 0
        hole_count = hole_count_geo
        if hole_count <= 0:
            holes = state.geo.get("hole_diams_mm")
            if isinstance(holes, (list, tuple)):
                hole_count = len(holes)
        if hole_count > 0 and "drilling" in eff_hours:
            current = to_float(eff_hours.get("drilling")) or 0.0
            min_sec_per_hole = 9.0
            floor_hr = (hole_count * min_sec_per_hole) / 3600.0
            if current < floor_hr:
                eff_hours["drilling"] = floor_hr
                state.effective["process_hours"] = eff_hours
                note = f"Raised drilling to floor for {hole_count} holes"
                notes = state.effective.setdefault("notes", [])
                if note not in notes:
                    notes.append(note)
    return state


def get_why_text(
    state: QuoteState,
    *,
    pricing_source: str,
    process_meta: Mapping[str, Any] | None = None,
    final_hours: Mapping[str, Any] | None = None,
) -> str:
    g = state.geo or {}
    b = state.baseline or {}
    e = state.effective or {}

    def _as_int(val: Any) -> int:
        try:
            return int(float(val))
        except Exception:
            return 0

    def _friendly_label(name: str) -> str:
        text = str(name or "").replace("_", " ").strip()
        return text.title() if text else ""

    def _as_float(val: Any) -> float:
        try:
            return float(val)
        except Exception:
            return 0.0

    def _hours_from_any(value: Any) -> float:
        if isinstance(value, Mapping):
            hr_val = to_float(value.get("hr"))
            if hr_val is None:
                minutes_val = to_float(value.get("minutes"))
                if minutes_val is not None:
                    hr_val = float(minutes_val) / 60.0
            if hr_val is None and "value" in value:
                hr_val = to_float(value.get("value"))
            if hr_val is not None:
                try:
                    return float(hr_val)
                except Exception:
                    return 0.0
            return 0.0
        num = to_float(value)
        return float(num) if num is not None else 0.0

    holes = _as_int(g.get("hole_count"))
    if holes <= 0:
        holes = _as_int(g.get("hole_count_override"))
    if holes <= 0:
        diam_list = g.get("hole_diams_mm")
        if isinstance(diam_list, (list, tuple)):
            holes = len(diam_list)

    thick_mm = to_float(g.get("thickness_mm"))
    if thick_mm is None:
        thick_mm = to_float(state.ui_vars.get("Thickness (in)"))
        if thick_mm is not None:
            thick_mm *= 25.4
    mat = g.get("material") or state.ui_vars.get("Material") or "material"
    mat = str(mat).title()

    setups = int(to_float(e.get("setups")) or to_float(b.get("setups")) or 1)
    scrap_pct = round(100.0 * float(to_float(e.get("scrap_pct")) or 0.0), 1)

    pricing_source_clean = str(pricing_source or "").strip().lower()
    meta_map: dict[str, Any] = {}
    if isinstance(process_meta, Mapping):
        meta_map = {str(k): v for k, v in process_meta.items()}

    final_hours_map: dict[str, float] = {}
    if isinstance(final_hours, Mapping):
        for key, value in final_hours.items():
            final_hours_map[str(key)] = max(0.0, _hours_from_any(value))
    else:
        proc_hours_raw = e.get("process_hours") if isinstance(e, Mapping) else None
        if not isinstance(proc_hours_raw, Mapping):
            proc_hours_raw = {}
        for key, value in proc_hours_raw.items():
            final_hours_map[str(key)] = max(0.0, _hours_from_any(value))

    top_text = ""
    if pricing_source_clean == "planner" and meta_map:
        planner_total_meta = None
        for cand_key in ("planner_total", "planner total"):
            if cand_key in meta_map:
                planner_total_meta = meta_map[cand_key]
                break
        if isinstance(planner_total_meta, Mapping):
            raw_items = planner_total_meta.get("line_items")
            if isinstance(raw_items, list):
                planner_entries: list[tuple[str, float]] = []
                for item in raw_items:
                    if not isinstance(item, Mapping):
                        continue
                    name = str(item.get("op") or "").strip()
                    if not name:
                        continue
                    minutes_val = _as_float(item.get("minutes"))
                    if minutes_val <= 0:
                        continue
                    planner_entries.append((name, minutes_val))
                planner_entries.sort(key=lambda pair: pair[1], reverse=True)
                top_bits: list[str] = []
                for name, minutes_val in planner_entries[:3]:
                    top_bits.append(f"{name} {minutes_val / 60.0:.2f} h")
                if top_bits:
                    top_text = ", ".join(top_bits)
        if not top_text and final_hours_map:
            bucket_candidates: list[tuple[str, float]] = []
            for key, hours in final_hours_map.items():
                key_lower = key.lower()
                if hours <= 0:
                    continue
                if key_lower.startswith("planner_") or key_lower in {"planner_total"}:
                    continue
                bucket_candidates.append((key, hours))
            bucket_candidates.sort(key=lambda pair: pair[1], reverse=True)
            top_bits: list[str] = []
            for name, hours in bucket_candidates[:3]:
                label = _friendly_label(name) or name
                top_bits.append(f"{label} {hours:.2f} h")
            if top_bits:
                top_text = ", ".join(top_bits)

    if not top_text:
        if not final_hours_map:
            proc_hours_raw = e.get("process_hours") if isinstance(e, Mapping) else None
            if not isinstance(proc_hours_raw, Mapping):
                proc_hours_raw = {}
            for key, value in proc_hours_raw.items():
                try:
                    final_hours_map[str(key)] = max(0.0, float(value or 0.0))
                except Exception:
                    continue
        top_candidates = sorted(final_hours_map.items(), key=lambda kv: kv[1], reverse=True)
        top_bits: list[str] = []
        for name, hours in top_candidates[:3]:
            if hours <= 0:
                continue
            label = _friendly_label(name) or name
            top_bits.append(f"{label} {hours:.2f} h")
        top_text = ", ".join(top_bits) if top_bits else "Baseline machining"

    base_hours_raw = b.get("process_hours") if isinstance(b.get("process_hours"), Mapping) else {}
    base_hours_map: dict[str, float] = {}
    if isinstance(base_hours_raw, Mapping):
        for proc, base_hr in base_hours_raw.items():
            base_hours_map[str(proc)] = max(0.0, _hours_from_any(base_hr))
    delta_lines: list[str] = []
    for proc_key in sorted(set(base_hours_map) | set(final_hours_map)):
        base_hr = base_hours_map.get(proc_key, 0.0) or 0.0
        adj_hr = final_hours_map.get(proc_key, 0.0) or 0.0
        try:
            diff = float(adj_hr) - float(base_hr)
        except Exception:
            continue
        if not math.isfinite(diff):
            continue
        diff = max(-24.0, min(24.0, diff))
        if abs(diff) < 1e-3:
            continue
        label = _friendly_label(proc_key) or proc_key
        arrow = "↑" if diff > 0 else "↓"
        delta_lines.append(f"  - {label}: {arrow}{abs(diff):.2f} h")
    delta_text = "\n".join(delta_lines)

    parts: list[str] = []
    if holes and thick_mm:
        parts.append(f"Plate in {mat}, thickness {thick_mm/25.4:.2f} in with {holes} holes.")
    elif holes:
        parts.append(f"Part with {holes} holes in {mat}.")
    else:
        parts.append(f"Part in {mat}.")
    if setups > 1:
        fixture = e.get("fixture") or b.get("fixture") or "standard"
        parts.append(f"Estimated {setups} setups; fixture: {fixture}.")
    parts.append(f"Major labor: {top_text}.")
    if delta_text:
        parts.append(f"Adjustments vs baseline:\n{delta_text}")

    material_source = state.material_source or "shop defaults"
    parts.append(f"Material priced via {material_source}; scrap {scrap_pct}% applied.")

    guard_ctx = state.guard_context if isinstance(state.guard_context, Mapping) else None
    if isinstance(guard_ctx, Mapping):
        extras_raw = guard_ctx.get("narrative_extra")
        if isinstance(extras_raw, str):
            extra_lines = [extras_raw]
        elif isinstance(extras_raw, Sequence):
            extra_lines = [str(line) for line in extras_raw]
        else:
            extra_lines = []
        for line in extra_lines:
            text = str(line).strip()
            if text:
                parts.append(text)

    return " ".join(parts)


def effective_to_overrides(effective: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or {}
    out: dict[str, Any] = {}
    mults = effective.get("process_hour_multipliers") if isinstance(effective.get("process_hour_multipliers"), dict) else {}
    if mults:
        cleaned = {k: float(v) for k, v in mults.items() if v is not None and not math.isclose(float(v), 1.0, rel_tol=1e-6, abs_tol=1e-6)}
        if cleaned:
            out["process_hour_multipliers"] = cleaned
    adders = effective.get("process_hour_adders") if isinstance(effective.get("process_hour_adders"), dict) else {}
    if adders:
        cleaned_add = {k: float(v) for k, v in adders.items() if v is not None and not math.isclose(float(v), 0.0, abs_tol=1e-6)}
        if cleaned_add:
            out["process_hour_adders"] = cleaned_add
    passes = (
        effective.get("add_pass_through")
        if isinstance(effective.get("add_pass_through"), dict)
        else {}
    )
    if passes:
        canonical_passes = _canonicalize_pass_through_map(passes)
        cleaned_pass = {
            k: float(v)
            for k, v in canonical_passes.items()
            if not math.isclose(float(v), 0.0, abs_tol=1e-6)
        }
        if cleaned_pass:
            out["add_pass_through"] = cleaned_pass
    scrap_eff = effective.get("scrap_pct")
    scrap_base = baseline.get("scrap_pct")
    if scrap_eff is not None and (scrap_base is None or not math.isclose(float(scrap_eff), float(scrap_base or 0.0), abs_tol=1e-6)):
        out["scrap_pct_override"] = float(scrap_eff)
    contingency_eff = effective.get("contingency_pct")
    contingency_base = baseline.get("contingency_pct")
    if contingency_eff is not None and (contingency_base is None or not math.isclose(float(contingency_eff), float(contingency_base or 0.0), abs_tol=1e-6)):
        out["contingency_pct_override"] = float(contingency_eff)
    setups_eff = effective.get("setups")
    fixture_eff = effective.get("fixture")
    if setups_eff is not None or fixture_eff is not None:
        out["setup_recommendation"] = {}
        if setups_eff is not None:
            out["setup_recommendation"]["setups"] = setups_eff
        if fixture_eff is not None:
            out["setup_recommendation"]["fixture"] = fixture_eff
    numeric_keys = {
        "fixture_build_hr": (0.0, None),
        "soft_jaw_hr": (0.0, None),
        "soft_jaw_material_cost": (0.0, None),
        "handling_adder_hr": (0.0, None),
        "cmm_minutes": (0.0, None),
        "in_process_inspection_hr": (0.0, None),
        "fai_prep_hr": (0.0, None),
        "packaging_hours": (0.0, None),
        "packaging_flat_cost": (0.0, None),
        "shipping_cost": (0.0, None),
    }
    for key, (_default, _) in numeric_keys.items():
        eff_val = effective.get(key)
        base_val = baseline.get(key) if isinstance(baseline, dict) else None
        if eff_val is None:
            continue
        if base_val is None or not math.isclose(float(eff_val), float(base_val or 0.0), rel_tol=1e-6, abs_tol=1e-6):
            out[key] = float(eff_val)
    bool_keys = ["fai_required"]
    for key in bool_keys:
        eff_val = effective.get(key)
        base_val = baseline.get(key) if isinstance(baseline, dict) else None
        if eff_val is None:
            continue
        if base_val is None or bool(eff_val) != bool(base_val):
            out[key] = bool(eff_val)
    text_keys = ["shipping_hint"]
    for key in text_keys:
        eff_val = effective.get(key)
        base_val = baseline.get(key) if isinstance(baseline, dict) else None
        if eff_val is None:
            continue
        if (base_val or "") != (eff_val or ""):
            out[key] = eff_val
    if effective.get("operation_sequence"):
        out["operation_sequence"] = list(effective["operation_sequence"])
    if isinstance(effective.get("drilling_strategy"), dict):
        out["drilling_strategy"] = copy.deepcopy(effective["drilling_strategy"])
    return out


def ensure_accept_flags(state: QuoteState) -> None:
    suggestions = state.suggestions or {}
    accept = state.accept_llm
    if not isinstance(accept, dict):
        state.accept_llm = {}
        accept = state.accept_llm

    for key in ("process_hour_multipliers", "process_hour_adders", "add_pass_through"):
        sugg = suggestions.get(key)
        if not isinstance(sugg, dict):
            continue
        bucket = accept.setdefault(key, {})
        for subkey in sugg.keys():
            if subkey not in bucket or not isinstance(bucket.get(subkey), bool):
                bucket[subkey] = False
        # remove stale keys
        for stale in list(bucket.keys()):
            if stale not in sugg:
                bucket.pop(stale, None)

    for key in (
        "scrap_pct",
        "contingency_pct",
        "setups",
        "fixture",
        "fixture_build_hr",
        "soft_jaw_hr",
        "soft_jaw_material_cost",
        "handling_adder_hr",
        "cmm_minutes",
        "in_process_inspection_hr",
        "fai_required",
        "fai_prep_hr",
        "packaging_hours",
        "packaging_flat_cost",
        "shipping_hint",
    ):
        if key in suggestions and not isinstance(accept.get(key), bool):
            accept[key] = False
        if key not in suggestions and key in accept and not isinstance(accept.get(key), dict):
            # keep user toggles if overrides exist even without suggestions
            continue
    if isinstance(suggestions.get("operation_sequence"), list) and not isinstance(accept.get("operation_sequence"), bool):
        accept["operation_sequence"] = False
    if isinstance(suggestions.get("drilling_strategy"), dict) and not isinstance(accept.get("drilling_strategy"), bool):
        accept["drilling_strategy"] = False


if sys.platform == 'win32':
    occ_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(occ_bin):
        os.add_dll_directory(occ_bin)
import importlib
import subprocess
import tempfile
from tkinter import filedialog, messagebox, ttk

try:
    from hole_table_parser import parse_hole_table_lines as _parse_hole_table_lines
except Exception:
    _parse_hole_table_lines = None

try:
    from dxf_text_extract import extract_text_lines_from_dxf as _extract_text_lines_from_dxf
except Exception:
    _extract_text_lines_from_dxf = None

# ---------- OCC / OCP compatibility ----------
STACK = None


def _import_optional(module_name: str):
    """Safely import *module_name* and return ``None`` if it is unavailable."""

    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _make_bnd_add_ocp():
    # Try every known symbol name in OCP builds
    candidates = []
    module = _import_optional("OCP.BRepBndLib")
    if module is None:
        raise ImportError("OCP.BRepBndLib is not available")

    for attr in (
        "Add",
        "Add_s",
        "BRepBndLib_Add",
        "brepbndlib_Add",
    ):
        fn = getattr(module, attr, None)
        if fn:
            candidates.append(fn)

    klass = getattr(module, "BRepBndLib", None)
    if klass:
        for attr in ("Add", "Add_s", "AddClose_s", "AddOptimal_s", "AddOBB_s"):
            fn = getattr(klass, attr, None)
            if fn:
                candidates.append(fn)

    if not candidates:
        raise ImportError("No BRepBndLib.Add symbol found in OCP")

    # return a dispatcher that tries different call signatures
    def _bnd_add(shape, box, use_triangulation=True):
        last_err = None
        for fn in candidates:
            # Try 3-arg form
            try:
                return fn(shape, box, use_triangulation)
            except TypeError as e:
                last_err = e
            # Try 2-arg form
            try:
                return fn(shape, box)
            except TypeError as e:
                last_err = e
            except Exception as e:
                last_err = e
        # If we get here, none of the variants worked
        raise last_err or RuntimeError("No callable BRepBndLib.Add variant succeeded")

    return _bnd_add

_ocp_modules = {
    "Bnd": _import_optional("OCP.Bnd"),
    "BRep": _import_optional("OCP.BRep"),
    "BRepCheck": _import_optional("OCP.BRepCheck"),
    "IFSelect": _import_optional("OCP.IFSelect"),
    "ShapeFix": _import_optional("OCP.ShapeFix"),
    "STEPControl": _import_optional("OCP.STEPControl"),
    "TopoDS": _import_optional("OCP.TopoDS"),
}

bnd_add: Callable[[Any, Any, bool], Any]

if all(module is not None for module in _ocp_modules.values()):
    # Prefer OCP (CadQuery/ocp bindings)
    bnd_module = cast(Any, _ocp_modules["Bnd"])
    brep_module = cast(Any, _ocp_modules["BRep"])
    brepcheck_module = cast(Any, _ocp_modules["BRepCheck"])
    ifselect_module = cast(Any, _ocp_modules["IFSelect"])
    shapefix_module = cast(Any, _ocp_modules["ShapeFix"])
    step_module = cast(Any, _ocp_modules["STEPControl"])
    topods_module = cast(Any, _ocp_modules["TopoDS"])

    Bnd_Box = bnd_module.Bnd_Box
    BRep_Builder = brep_module.BRep_Builder
    BRepCheck_Analyzer = brepcheck_module.BRepCheck_Analyzer
    IFSelect_RetDone = ifselect_module.IFSelect_RetDone
    ShapeFix_Shape = shapefix_module.ShapeFix_Shape
    STEPControl_Reader = step_module.STEPControl_Reader
    TopoDS_Compound = topods_module.TopoDS_Compound
    TopoDS_Shape = topods_module.TopoDS_Shape

    try:
        bnd_add = _make_bnd_add_ocp()
        STACK = "ocp"
    except Exception:
        # Fallback: try dynamic dispatch at call time for OCP builds missing Add

        def _bnd_add_ocp_fallback(shape, box, use_triangulation=True):
            candidates = [
                ("OCP.BRepBndLib", "Add"),
                ("OCP.BRepBndLib", "Add_s"),
                ("OCP.BRepBndLib", "BRepBndLib_Add"),
                ("OCP.BRepBndLib", "brepbndlib_Add"),
                ("OCC.Core.BRepBndLib", "Add"),
                ("OCC.Core.BRepBndLib", "Add_s"),
                ("OCC.Core.BRepBndLib", "BRepBndLib_Add"),
                ("OCC.Core.BRepBndLib", "brepbndlib_Add"),
            ]
            for mod_name, attr in candidates:
                module = _import_optional(mod_name)
                if module and hasattr(module, attr):
                    return getattr(module, attr)(shape, box, use_triangulation)
            raise RuntimeError("No BRepBndLib.Add available in this build")

        bnd_add = _bnd_add_ocp_fallback
        STACK = "ocp"
else:
    # Fallback to pythonocc-core
    _occ_modules = {
        "Bnd": _import_optional("OCC.Core.Bnd"),
        "BRep": _import_optional("OCC.Core.BRep"),
        "BRepBndLib": _import_optional("OCC.Core.BRepBndLib"),
        "BRepCheck": _import_optional("OCC.Core.BRepCheck"),
        "IFSelect": _import_optional("OCC.Core.IFSelect"),
        "ShapeFix": _import_optional("OCC.Core.ShapeFix"),
        "STEPControl": _import_optional("OCC.Core.STEPControl"),
        "TopoDS": _import_optional("OCC.Core.TopoDS"),
    }

    missing = [name for name, module in _occ_modules.items() if module is None]
    if missing:
        raise ImportError(
            "Required OCC.Core modules are unavailable: {}".format(
                ", ".join(sorted(missing))
            )
        )

    bnd_module = cast(Any, _occ_modules["Bnd"])
    brep_module = cast(Any, _occ_modules["BRep"])
    brepbndlib_module = cast(Any, _occ_modules["BRepBndLib"])
    brepcheck_module = cast(Any, _occ_modules["BRepCheck"])
    ifselect_module = cast(Any, _occ_modules["IFSelect"])
    shapefix_module = cast(Any, _occ_modules["ShapeFix"])
    step_module = cast(Any, _occ_modules["STEPControl"])
    topods_module = cast(Any, _occ_modules["TopoDS"])

    Bnd_Box = bnd_module.Bnd_Box
    BRep_Builder = brep_module.BRep_Builder
    _brep_add = brepbndlib_module.brepbndlib_Add
    BRepCheck_Analyzer = brepcheck_module.BRepCheck_Analyzer
    IFSelect_RetDone = ifselect_module.IFSelect_RetDone
    ShapeFix_Shape = shapefix_module.ShapeFix_Shape
    STEPControl_Reader = step_module.STEPControl_Reader
    TopoDS_Compound = topods_module.TopoDS_Compound
    TopoDS_Shape = topods_module.TopoDS_Shape

    def _bnd_add_pythonocc(shape, box, use_triangulation=True):
        _brep_add(shape, box, use_triangulation)

    bnd_add = _bnd_add_pythonocc
    STACK = "pythonocc"
# ---------- end shim ----------
# ----- one-backend imports -----
_backend_modules = {
    "BRep": _import_optional("OCP.BRep"),
    "TopAbs": _import_optional("OCP.TopAbs"),
    "TopExp": _import_optional("OCP.TopExp"),
}

_brep_mod = _backend_modules["BRep"]
_topabs_mod = _backend_modules["TopAbs"]
_topexp_mod = _backend_modules["TopExp"]

if _brep_mod and _topabs_mod and _topexp_mod:
    BRep_Tool = cast(Any, _brep_mod).BRep_Tool
    TopAbs_FACE = cast(Any, _topabs_mod).TopAbs_FACE
    TopExp_Explorer = cast(Any, _topexp_mod).TopExp_Explorer
    BACKEND = "ocp"
else:
    _backend_modules = {
        "BRep": _import_optional("OCC.Core.BRep"),
        "TopAbs": _import_optional("OCC.Core.TopAbs"),
        "TopExp": _import_optional("OCC.Core.TopExp"),
    }

    missing = [name for name, module in _backend_modules.items() if module is None]
    if missing:
        raise ImportError(
            "Required backend modules are unavailable: {}".format(
                ", ".join(sorted(missing))
            )
        )

    _brep_mod = _backend_modules["BRep"]
    _topabs_mod = _backend_modules["TopAbs"]
    _topexp_mod = _backend_modules["TopExp"]

    assert _brep_mod is not None
    assert _topabs_mod is not None
    assert _topexp_mod is not None

    BRep_Tool = cast(Any, _brep_mod).BRep_Tool
    TopAbs_FACE = cast(Any, _topabs_mod).TopAbs_FACE
    TopExp_Explorer = cast(Any, _topexp_mod).TopExp_Explorer
    BACKEND = "pythonocc"


def _resolve_face_of():
    """Return a callable that casts a shape-like object to a TopoDS_Face."""

    # Prefer helpers exposed by cad_quoter.geometry when available
    fn = getattr(geometry, "FACE_OF", None)
    if callable(fn):
        return fn

    # Try OCP's modern `topods.Face` helper first
    try:  # pragma: no cover - depends on optional OCC bindings
        from OCP.TopoDS import topods as _topods  # type: ignore[import-not-found]

        if hasattr(_topods, "Face"):
            return _topods.Face  # type: ignore[return-value]
    except Exception:
        pass

    # pythonocc-core exposes either topods_Face or topods.Face
    try:  # pragma: no cover - depends on optional OCC bindings
        from OCC.Core.TopoDS import topods_Face  # type: ignore[import-not-found]

        return topods_Face  # type: ignore[return-value]
    except Exception:
        pass
    try:  # pragma: no cover - depends on optional OCC bindings
        from OCC.Core.TopoDS import topods as _occ_topods  # type: ignore[import-not-found]

        face_fn = getattr(_occ_topods, "Face", None)
        if callable(face_fn):
            return face_fn
    except Exception:
        pass

    # Fall back to methods on the TopoDS namespace (OCP variants expose Face_s)
    try:  # pragma: no cover - depends on optional OCC bindings
        from OCP.TopoDS import TopoDS as _TopoDS  # type: ignore[import-not-found]

        for attr in ("Face_s", "Face"):
            face_fn = getattr(_TopoDS, attr, None)
            if callable(face_fn):
                return face_fn
    except Exception:
        pass

    def _raise(obj):
        raise TypeError(f"Cannot cast {type(obj).__name__} to TopoDS_Face")

    return _raise


FACE_OF = _resolve_face_of()

def as_face(obj: Any) -> TopoDS_Face:
    """
    Return a TopoDS_Face for the *current backend*.
    - If already a Face, return it (don't re-cast).
    - If a tuple (some APIs return (surf, loc) or similar), take first elem.
    - If null/None, raise cleanly.
    - Otherwise cast with FACE_OF.
    """
    if obj is None:
        raise TypeError("Expected a shape, got None")

    # unwrap tuples (defensive)
    if isinstance(obj, tuple) and obj:
        obj = obj[0]
    return ensure_face(obj)


def iter_faces(shape: Any) -> Iterator[TopoDS_Face]:
    explorer_cls = cast(type[Any], TopExp_Explorer)
    exp = explorer_cls(shape, cast(Any, TopAbs_FACE))
    while exp.More():
        yield ensure_face(exp.Current())
        exp.Next()


def face_surface(face_like: Any) -> tuple[Any, Any | None]:
    f = ensure_face(face_like)
    tool = cast(Any, BRep_Tool)
    surface_s = getattr(tool, "Surface_s", None)
    if callable(surface_s):
        s = surface_s(f)
    else:
        s = tool.Surface(f)
    location_fn = getattr(tool, "Location", None)
    if callable(location_fn):
        loc = location_fn(f)
    else:
        face_location = getattr(f, "Location", None)
        loc = face_location() if callable(face_location) else None
    if isinstance(s, tuple):
        s, loc2 = s
        loc = loc or loc2
    if hasattr(s, "Surface"):
        s = s.Surface()  # unwrap handle
    return s, loc



# ---------- OCCT compat (OCP or pythonocc-core) ----------
# ---------- Robust casters that work on OCP and pythonocc ----------
# Lock topods casters to the active backend
if STACK == "ocp":

    def _TO_EDGE(s):
        if type(s).__name__ in ("TopoDS_Edge", "Edge"):
            return s
        if hasattr(TopoDS, "Edge_s"):
            return TopoDS.Edge_s(s)
        try:
            from OCP.TopoDS import topods as _topods  # type: ignore[import]
            return _topods.Edge(s)
        except Exception as e:
            raise TypeError(f"Cannot cast to Edge from {type(s).__name__}") from e

    def _TO_SOLID(s):
        if type(s).__name__ in ("TopoDS_Solid", "Solid"):
            return s
        if hasattr(TopoDS, "Solid_s"):
            return TopoDS.Solid_s(s)
        from OCP.TopoDS import topods as _topods  # type: ignore[import]
        return _topods.Solid(s)

    def _TO_SHELL(s):
        if type(s).__name__ in ("TopoDS_Shell", "Shell"):
            return s
        if hasattr(TopoDS, "Shell_s"):
            return TopoDS.Shell_s(s)
        from OCP.TopoDS import topods as _topods  # type: ignore[import]
        return _topods.Shell(s)
else:
    # Resolve within OCC.Core only
    _occ_topods_module = _import_optional("OCC.Core.TopoDS")
    if _occ_topods_module is None:
        raise ImportError("OCC.Core.TopoDS is required for pythonocc backend")

    def _resolve_topods_callable(*names: str):
        for candidate in names:
            attr = getattr(_occ_topods_module, candidate, None)
            if attr:
                return attr
        nested = getattr(_occ_topods_module, "topods", None)
        if nested is not None:
            for candidate in names:
                attr = getattr(nested, candidate, None)
                if attr:
                    return attr
        raise AttributeError(
            f"None of {names!r} are available on OCC.Core.TopoDS for casting"
        )

    _TO_EDGE = _resolve_topods_callable("topods_Edge", "Edge")
    _TO_SOLID = _resolve_topods_callable("topods_Solid", "Solid")
    _TO_SHELL = _resolve_topods_callable("topods_Shell", "Shell")

# Type guards
def _is_named(obj, names: tuple[str, ...]) -> bool:
    try:
        return type(obj).__name__ in names
    except Exception:
        return False

def _unwrap_value(obj):
    # Some containers return nodes with .Value()
    return obj.Value() if hasattr(obj, "Value") and callable(obj.Value) else obj

def ensure_shape(obj):
    obj = _unwrap_value(obj)
    if obj is None:
        raise TypeError("Expected TopoDS_Shape, got None")
    is_null = getattr(obj, "IsNull", None)
    if callable(is_null) and is_null():
        raise TypeError("Expected non-null TopoDS_Shape")
    return obj

# Safe casters: no-ops if already cast; unwrap list nodes; check kind
# Choose stack
_BRepGProp_mod = None
STACK_GPROP = "pythonocc"
_ocp_brepgprop = _import_optional("OCP.BRepGProp")
if _ocp_brepgprop is not None and hasattr(_ocp_brepgprop, "BRepGProp"):
    _BRepGProp_mod = getattr(_ocp_brepgprop, "BRepGProp")
    STACK_GPROP = "ocp"
else:
    _occ_brepgprop = _import_optional("OCC.Core.BRepGProp")
    if _occ_brepgprop is None:
        raise ImportError("OCC.Core.BRepGProp is required for pythonocc backend")
    if hasattr(_occ_brepgprop, "BRepGProp"):
        _BRepGProp_mod = getattr(_occ_brepgprop, "BRepGProp")
    else:
        from types import SimpleNamespace

        from OCC.Core.BRepGProp import (
            brepgprop_LinearProperties as _lp,  # type: ignore[attr-defined]
        )
        from OCC.Core.BRepGProp import (
            brepgprop_SurfaceProperties as _sp,  # type: ignore[attr-defined]
        )
        from OCC.Core.BRepGProp import (
            brepgprop_VolumeProperties as _vp,  # type: ignore[attr-defined]
        )
        _BRepGProp_mod = SimpleNamespace(
            LinearProperties=getattr(_occ_brepgprop, "brepgprop_LinearProperties"),
            SurfaceProperties=getattr(_occ_brepgprop, "brepgprop_SurfaceProperties"),
            VolumeProperties=getattr(_occ_brepgprop, "brepgprop_VolumeProperties"),
        )

    def _to_edge_occ(s):
        try:
            from OCC.Core.TopoDS import topods_Edge as _fn  # type: ignore[attr-defined]
        except Exception:
            from OCC.Core.TopoDS import Edge as _fn  # type: ignore[attr-defined]
        return _fn(s)

    _TO_EDGE = _to_edge_occ

# Resolve topods casters across bindings


# ---- modern wrappers (no deprecation warnings)
# ---- modern wrappers (no deprecation warnings)
def linear_properties(edge, gprops):
    """Linear properties across OCP/pythonocc names."""
    fn = getattr(_BRepGProp_mod, "LinearProperties", None)
    if fn is None:
        fn = getattr(_BRepGProp_mod, "LinearProperties_s", None)
    if fn is None:
        try:
            from OCC.Core.BRepGProp import brepgprop_LinearProperties as _old  # type: ignore
            return _old(edge, gprops)
        except Exception:
            raise
    return fn(edge, gprops)

def map_shapes_and_ancestors(root_shape, sub_enum, anc_enum):
    """Return TopTools_IndexedDataMapOfShapeListOfShape for (sub → ancestors)."""
    # Ensure we pass a *Shape*, not a Face
    if root_shape is None:
        raise TypeError("root_shape is None")
    if not hasattr(root_shape, "IsNull") or root_shape.IsNull():
        # If someone handed us a Face, try to grab its TShape parent; else fail.
        # Safer: require a real TopoDS_Shape from STEP/IGES root.
        pass

    amap = cast(
        "TopTools_IndexedDataMapOfShapeListOfShape",
        TopTools_IndexedDataMapOfShapeListOfShape(),  # type: ignore[call-overload]
    )
    # static/instance variants across wheels
    fn = getattr(TopExp, "MapShapesAndAncestors", None) or getattr(TopExp, "MapShapesAndAncestors_s", None)
    if fn is None:
        raise RuntimeError("TopExp.MapShapesAndAncestors not available in this OCP wheel")
    fn(root_shape, sub_enum, anc_enum, amap)
    return amap

# modern topods casters: topods.Edge(shape) / topods.Face(shape)
# ---- Robust topods casters that are no-ops for already-cast objects ----
def _is_instance(obj, qualnames):
    """True if obj.__class__.__name__ matches any in qualnames across OCP/OCC."""
    try:
        name = type(obj).__name__
    except Exception:
        return False
    return name in qualnames  # e.g. ["TopoDS_Face", "Face"]

def ensure_face(obj: Any) -> TopoDS_Face:
    if obj is None:
        raise TypeError("Expected a face, got None")
    face_type = cast(type, TopoDS_Face)
    if isinstance(obj, face_type) or type(obj).__name__ == "TopoDS_Face":
        return obj
    st = obj.ShapeType() if hasattr(obj, "ShapeType") else None
    if st == TopAbs_FACE:
        return FACE_OF(obj)
    raise TypeError(f"Not a face: {type(obj).__name__}")
def to_edge(s):
    if _is_instance(s, ["TopoDS_Edge", "Edge"]):
        return s
    return _TO_EDGE(s)

def to_solid(s):
    if _is_instance(s, ["TopoDS_Solid", "Solid"]):
        return s
    return _TO_SOLID(s)

def to_shell(s):
    if _is_instance(s, ["TopoDS_Shell", "Shell"]):
        return s
    return _TO_SHELL(s)

# Generic size helpers so we never call Extent() on lists/maps again
def map_size(m):
    for name in ("Size", "Extent", "Length"):
        if hasattr(m, name):
            return getattr(m, name)()
    raise AttributeError(f"No size method on {type(m)}")

def list_iter(lst):
    """
    Iterate TopTools_ListOfShape across bindings.
    Use cbegin()/More()/Value()/Next() if available; otherwise fall back to python list conversion.
    """
    # OCP + pythonocc expose cbegin iterator on TopTools_ListOfShape
    if hasattr(lst, "cbegin"):
        it = lst.cbegin()
        while it.More():
            yield it.Value()
            it.Next()
    else:
        # last-resort: try Python iteration (some wheels support it)
        for v in list(lst):
            yield v
# ---------- end compat ----------


# ---- tiny helpers you can use elsewhere --------------------------------------
def require_ezdxf():
    """Raise a clear error if ezdxf is missing and return the module."""

    if callable(_geometry_require_ezdxf):
        return _geometry_require_ezdxf()

    if not geometry.HAS_EZDXF:
        raise RuntimeError("ezdxf not installed. Install with pip/conda (package name: 'ezdxf').")

    try:
        import ezdxf as _ezdxf
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError("Failed to import ezdxf even though HAS_EZDXF is true.") from exc

    return _ezdxf

def get_dwg_converter_path() -> str:
    """Resolve a DWG?DXF converter path (.bat/.cmd/.exe)."""
    exe = os.environ.get("ODA_CONVERTER_EXE") or os.environ.get("DWG2DXF_EXE")
    # Fall back to a local wrapper next to this script if it exists.
    local = str(Path(__file__).with_name("dwg2dxf_wrapper.bat"))
    if not exe and Path(local).exists():
        exe = local
    return exe or ""

def get_import_diagnostics_text() -> str:
    import os
    import sys
    lines = []
    lines.append(f"Python: {sys.executable}")
    try:
        lines.append("PyMuPDF: OK")
    except Exception as e:
        lines.append(f"PyMuPDF: MISSING ({e})")

    try:
        import ezdxf
        lines.append(f"ezdxf: {getattr(ezdxf, '__version__', 'unknown')}")
        try:
            lines.append("ezdxf.addons.odafc: OK")
        except Exception as e:
            lines.append(f"ezdxf.addons.odafc: not available ({e})")
    except Exception as e:
        lines.append(f"ezdxf: MISSING ({e})")

    oda = os.environ.get("ODA_CONVERTER_EXE") or "(not set)"
    d2d = os.environ.get("DWG2DXF_EXE") or "(not set)"
    lines.append(f"ODA_CONVERTER_EXE: {oda}")
    lines.append(f"DWG2DXF_EXE: {d2d}")

    # local wrapper presence
    from pathlib import Path
    wrapper = Path(__file__).with_name("dwg2dxf_wrapper.bat")
    lines.append(f"Local wrapper present: {wrapper.exists()} ({wrapper})")
    return "\n".join(lines)
# Optional PDF stack
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore[assignment]
    _HAS_PYMUPDF = False

def upsert_var_row(df, item, value, dtype="number"):
    """
    Upsert one row by Item name (case-insensitive).
    - Forces `item` and `value` to scalars.
    - Works on the sanitized 3-column df (Item, Example..., Data Type...).
    """
    import numpy as np

    # force scalars
    if isinstance(item, pd.Series):
        item = item.iloc[0]
    item = str(item)

    if isinstance(value, pd.Series):
        value = value.iloc[0]
    # try to make numeric if it looks numeric; otherwise keep as-is
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            pass
        else:
            value = float(value)
    except Exception:
        # leave non-numerics (e.g., text) as-is
        pass

    # build a row with the exact df schema
    cols = list(df.columns)
    row: Dict[str, Any] = {c: "" for c in cols}
    if "Item" in row:
        row["Item"] = item
    if "Example Values / Options" in row:
        row["Example Values / Options"] = value
    if "Data Type / Input Method" in row:
        row["Data Type / Input Method"] = dtype

    # case-insensitive exact match on Item
    m = df["Item"].astype(str).str.casefold() == item.casefold()
    if m.any():
        df.loc[m, cols] = [row[c] for c in cols]
        return df

    # append
    new_row = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    return pd.concat([df, new_row], ignore_index=True)

DIM_RE = re.compile(r"(?:ï¿½|DIAM|DIA)\s*([0-9.+-]+)|R\s*([0-9.+-]+)|([0-9.+-]+)\s*[xX]\s*([0-9.+-]+)")

def load_drawing(path: Path) -> Drawing:
    ezdxf = require_ezdxf()
    if path.suffix.lower() == ".dwg":
        # Prefer explicit converter/wrapper if configured (works even if ODA isnï¿½t on PATH)
        exe = get_dwg_converter_path()
        if exe:
            dxf_path = convert_dwg_to_dxf(str(path))
            return ezdxf.readfile(dxf_path)
        # Fallback: odafc (requires ODAFileConverter on PATH)
        if geometry.HAS_ODAFC:
            return odafc.readfile(str(path))
        raise RuntimeError(
            "DWG import needs ODA File Converter. Set ODA_CONVERTER_EXE to the exe "
            "or place dwg2dxf_wrapper.bat next to the script."
        )
    return ezdxf.readfile(str(path))  # DXF directly


# ==== OpenCascade compat (works with OCP OR OCC.Core) ====
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Tuple


def _missing_uv_bounds(_: Any) -> Tuple[float, float, float, float]:
    raise RuntimeError("BRepTools_UVBounds is unavailable")


def _missing_brep_read(_: str):
    raise RuntimeError("BREP read is unavailable")


BRepTools_UVBounds: Callable[[Any], Tuple[float, float, float, float]] = _missing_uv_bounds
_brep_read = _missing_brep_read

try:
    # ---- OCP branch ----
    from OCP.Bnd import Bnd_Box  # type: ignore[import]
    from OCP.BRep import (  # type: ignore[import]
        BRep_Builder,
        BRep_Tool,  # OCP version
    )  # type: ignore[import]
    from OCP.BRepAdaptor import BRepAdaptor_Curve  # type: ignore[import]
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section  # type: ignore[import]
    from OCP.BRepCheck import BRepCheck_Analyzer  # type: ignore[import]
    from OCP.BRepGProp import BRepGProp  # type: ignore[import]
    from OCP.GeomAbs import (  # type: ignore[import]
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_Circle,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Torus,
    )
    from OCP.GeomAdaptor import GeomAdaptor_Surface  # type: ignore[import]
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt  # type: ignore[import]
    from OCP.GProp import GProp_GProps  # type: ignore[import]
    from OCP.IFSelect import IFSelect_RetDone  # type: ignore[import]
    from OCP.IGESControl import IGESControl_Reader  # type: ignore[import]
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface  # type: ignore[import]
    from OCP.ShapeFix import ShapeFix_Shape  # type: ignore[import]
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID  # type: ignore[import]
    from OCP.TopExp import TopExp, TopExp_Explorer  # type: ignore[import]
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Face, TopoDS_Shape  # type: ignore[import]

    # ADD THESE TWO IMPORTS
    from OCP.BRepTools import BRepTools  # type: ignore[import]
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape  # type: ignore[import]
    BACKEND_OCC = "OCP"

    def _ocp_uv_bounds(face: TopoDS_Face) -> Tuple[float, float, float, float]:
        return BRepTools.UVBounds(face)


    def _ocp_brep_read(path: str) -> TopoDS_Shape:
        s = TopoDS_Shape()
        builder = BRep_Builder()  # type: ignore[call-arg]
        if hasattr(BRepTools, "Read_s"):
            ok = BRepTools.Read_s(s, str(path), builder)
        else:
            ok = BRepTools.Read(s, str(path), builder)
        if ok is False:
            raise RuntimeError("BREP read failed")
        return s


    BRepTools_UVBounds = _ocp_uv_bounds
    _brep_read = _ocp_brep_read



except Exception:
    # ---- OCC.Core branch ----
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import (
        BRep_Builder,
        BRep_Tool,  # ? OCC version
    )
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.GeomAbs import (
        GeomAbs_BezierSurface,
        GeomAbs_BSplineSurface,
        GeomAbs_Circle,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Torus,
    )
    from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import (
        TopAbs_COMPOUND,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Face, TopoDS_Shape

    # ADD TopTools import and TopoDS_Face for the fix below
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    BACKEND_OCC = "OCC.Core"

    # BRepGProp shim (pythonocc uses free functions)
    import OCC.Core.BRepGProp as _occ_brepgprop  # type: ignore[import]

    brepgprop_LinearProperties = _occ_brepgprop.brepgprop_LinearProperties
    brepgprop_SurfaceProperties = _occ_brepgprop.brepgprop_SurfaceProperties
    brepgprop_VolumeProperties = _occ_brepgprop.brepgprop_VolumeProperties
    class _BRepGPropShim:
        @staticmethod
        def SurfaceProperties_s(shape_or_face, gprops): brepgprop_SurfaceProperties(shape_or_face, gprops)
        @staticmethod
        def LinearProperties_s(edge, gprops):          brepgprop_LinearProperties(edge, gprops)
        @staticmethod
        def VolumeProperties_s(shape, gprops):         brepgprop_VolumeProperties(shape, gprops)
    BRepGProp = _BRepGPropShim



    # UV bounds and brep read are free functions
    from OCC.Core.BRepTools import BRepTools
    from OCC.Core.BRepTools import breptools_Read as _occ_breptools_read

    def _occ_uv_bounds(face: TopoDS_Face) -> Tuple[float, float, float, float]:
        fn = getattr(BRepTools, "UVBounds", None)
        if fn is None:
            from OCC.Core.BRepTools import breptools_UVBounds as _legacy
            return _legacy(face)
        return fn(face)


    def _occ_brep_read(path: str) -> TopoDS_Shape:
        s = TopoDS_Shape()
        ok = _occ_breptools_read(s, str(path), BRep_Builder())
        if not ok:
            raise RuntimeError("BREP read failed")
        return s


    BRepTools_UVBounds = _occ_uv_bounds
    _brep_read = _occ_brep_read

def _shape_from_reader(reader):
    """Return a healed TopoDS_Shape from a STEP/IGES reader."""
    transfer_count = 0
    if hasattr(reader, "NbShapes"):
        try:
            transfer_count = reader.NbShapes()
        except Exception:
            transfer_count = 0
    if not transfer_count and hasattr(reader, "NbRootsForTransfer"):
        try:
            transfer_count = reader.NbRootsForTransfer()
        except Exception:
            transfer_count = 0
    if transfer_count <= 0:
        raise RuntimeError("Reader produced zero shapes")

    if transfer_count == 1:
        shape = reader.Shape(1)
    else:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        added = 0
        for i in range(1, transfer_count + 1):
            s = reader.Shape(i)
            if s is None or s.IsNull():
                continue
            builder.Add(compound, s)
            added += 1
        if added == 0:
            raise RuntimeError("Reader produced only null sub-shapes")
        shape = compound

    if shape is None or shape.IsNull():
        raise RuntimeError("Reader produced a null TopoDS_Shape")

    fixer = cast(Any, ShapeFix_Shape)(shape)
    fixer.Perform()
    healed = fixer.Shape()
    if healed is None or healed.IsNull():
        raise RuntimeError("Shape healing failed (null shape)")

    try:
        analyzer = BRepCheck_Analyzer(healed)
        # we do not require validity, but invoking the analyzer surfaces issues early
        analyzer.IsValid()
    except Exception:
        pass

    return healed

def read_step_shape(path: str) -> TopoDSShapeT:
    rdr = STEPControl_Reader()
    if rdr.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("STEP read failed (file unreadable or unsupported).")
    rdr.TransferRoots()
    n = rdr.NbShapes()
    if n == 0:
        raise RuntimeError("STEP read produced zero shapes (no transferable roots).")

    if n == 1:
        shape = rdr.Shape(1)
    else:
        builder = BRep_Builder()
        comp = TopoDS_Compound()
        builder.MakeCompound(comp)
        for i in range(1, n + 1):
            s = rdr.Shape(i)
            if not s.IsNull():
                builder.Add(comp, s)
        shape = comp

    if shape.IsNull():
        raise RuntimeError("STEP produced a null TopoDS_Shape.")
    # Verify we truly pass a Shape to MapShapesAndAncestors
    logger.debug(
        "Shape type: %s IsNull: %s",
        type(shape).__name__,
        getattr(shape, "IsNull", lambda: True)(),
    )
    amap = map_shapes_and_ancestors(shape, TopAbs_EDGE, TopAbs_FACE)
    logger.debug("Shape ancestor map size: %d", amap.Size())
    # DEBUG: sanity probe for STEP faces
    if os.environ.get("STEP_PROBE", "0") == "1":
        cnt = 0
        try:
            for f in iter_faces(shape):
                _surf, _loc = face_surface(f)
                cnt += 1
        except Exception:
            # Keep debug non-fatal; report and continue
            logger.exception("STEP_PROBE face probe failed")
        else:
            logger.debug("STEP_PROBE faces=%d", cnt)

    fx = cast(Any, ShapeFix_Shape)(shape)
    fx.Perform()
    return fx.Shape()

def safe_bbox(shape: TopoDSShapeT):
    if shape is None or shape.IsNull():
        raise ValueError("Cannot compute bounding box of a null shape.")
    box = Bnd_Box()
    bnd_add(shape, box, True)  # <- uses whichever binding is available
    return box
def read_step_or_iges_or_brep(path: str) -> TopoDSShapeT:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".step", ".stp"):
        return read_step_shape(str(p))
    if ext in (".iges", ".igs"):
        ig = IGESControl_Reader()
        if ig.ReadFile(str(p)) != IFSelect_RetDone:
            raise RuntimeError("IGES read failed")
        ig.TransferRoots()
        return _shape_from_reader(ig)
    if ext == ".brep":
        return _brep_read(str(p))
    raise RuntimeError(f"Unsupported OCC format: {ext}")

def convert_dwg_to_dxf(dwg_path: str, *, out_ver="ACAD2018") -> str:
    """
    Robust DWG?DXF wrapper.
    Works with:
      - A .bat/.cmd wrapper that accepts:  <input.dwg> <output.dxf>
      - A custom exe that accepts:         <input.dwg> <output.dxf>
      - ODAFileConverter.exe (7-arg form): <in_dir> <out_dir> <ver> DXF 0 0 <filter>
    Looks for the converter via env var or common local paths and prints
    the exact command used on failure.
    """
    # 1) find converter
    exe = (os.environ.get("ODA_CONVERTER_EXE")
           or os.environ.get("DWG2DXF_EXE")
           or str(Path(__file__).with_name("dwg2dxf_wrapper.bat"))
           or r"D:\CAD_Quoting_Tool\dwg2dxf_wrapper.bat")

    if not exe or not Path(exe).exists():
        raise RuntimeError(
            "DWG import needs a DWG?DXF converter.\n"
            "Set ODA_CONVERTER_EXE (recommended) or DWG2DXF_EXE to a .bat/.cmd/.exe.\n"
            "Expected .bat signature:  <input.dwg> <output.dxf>"
        )

    dwg = Path(dwg_path)
    out_dir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
    out_dxf = out_dir / (dwg.stem + ".dxf")

    exe_lower = Path(exe).name.lower()
    cmd: list[str] = []
    try:
        if exe_lower.endswith(".bat") or exe_lower.endswith(".cmd"):
            # ? run batch via cmd.exe so it actually executes
            cmd = ["cmd", "/c", exe, str(dwg), str(out_dxf)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elif "odafileconverter" in exe_lower:
            # ? official ODAFileConverter CLI
            cmd = [exe, str(dwg.parent), str(out_dir), out_ver, "DXF", "0", "0", dwg.name]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            # ? generic exe that accepts <in> <out>
            cmd = [exe, str(dwg), str(out_dxf)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "DWG?DXF conversion failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e

    # 2) resolve the produced DXF
    produced = out_dxf if out_dxf.exists() else (out_dir / (dwg.stem + ".dxf"))
    if not produced.exists():
        raise RuntimeError(
            "Converter returned success but DXF not found.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"checked: {out_dxf} | {produced}"
        )
    return str(produced)



ANG_TOL = math.radians(5.0)
DOT_TOL = math.cos(ANG_TOL)
SMALL = 1e-7



def iter_solids(shape: TopoDS_Shape):
    explorer = cast(type[Any], TopExp_Explorer)
    exp = explorer(shape, cast(TopAbs_ShapeEnum, TopAbs_SOLID))
    while exp.More():
        yield to_solid(exp.Current())
        exp.Next()

def explode_compound(shape: TopoDS_Shape):
    """If the file is a big COMPOUND, break it into shapes (parts/bodies)."""
    explorer = cast(type[Any], TopExp_Explorer)
    exp = explorer(shape, cast(TopAbs_ShapeEnum, TopAbs_COMPOUND))
    if exp.More():
        # Itï¿½s a compound ï¿½ return its shells/solids/faces as needed
        solids = list(iter_solids(shape))
        if solids:
            return solids
        # fallback to shells
        sh = explorer(shape, cast(TopAbs_ShapeEnum, TopAbs_SHELL))
        shells = []
        while sh.More():
            shells.append(to_shell(sh.Current()))
            sh.Next()
        return shells
    return [shape]



def _bbox(shape):
    box = safe_bbox(shape)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (xmin, ymin, zmin, xmax, ymax, zmax)

def _area_of_face(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass())

def _length_of_edge(edge) -> float:
    props = GProp_GProps()
    linear_properties(edge, props)
    return float(props.Mass())

def _face_midpoint_uv(face):
    umin, umax, vmin, vmax = BRepTools_UVBounds(face)
    return (0.5*(umin+umax), 0.5*(vmin+vmax))


def _face_normal(face):
    try:
        try:
            from OCP.BRepLProp import BRepLProp_SLProps  # type: ignore[import]
        except ImportError:
            from OCC.Core.BRepLProp import BRepLProp_SLProps
        u, v = _face_midpoint_uv(face)
        props = BRepLProp_SLProps(face_surface(face)[0], u, v, 1, SMALL)
        if props.IsNormalDefined():
            n = props.Normal()
            if face.Orientation().Name() == "REVERSED":
                n.Reverse()
            return gp_Dir(n.X(), n.Y(), n.Z())
    except Exception:
        pass
    return None

def _face_type(face) -> str:
    surf = face_surface(face)[0]
    ga = GeomAdaptor_Surface(surf)
    st = ga.GetType()
    if st == GeomAbs_Plane: return "planar"
    if st in (GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone): return "cylindrical"
    if st in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface): return "freeform"
    return "other"



def _cluster_normals(normals):
    clusters = []
    for n in normals:
        added = False
        for c in clusters:
            if abs(n.Dot(c)) > DOT_TOL:
                added = True; break
        if not added:
            clusters.append(n)
    return clusters

def _sum_edge_length_sharp(shape, angle_thresh_deg=175.0) -> float:
    angle_thresh = math.radians(angle_thresh_deg)
    edge2faces = map_shapes_and_ancestors(shape, TopAbs_EDGE, TopAbs_FACE)
    total = 0.0
    for i in range(1, map_size(edge2faces) + 1):
        edge = to_edge(edge2faces.FindKey(i))
        face_list = edge2faces.FindFromIndex(i)
        faces = [ensure_face(shp) for shp in list_iter(face_list)]
        if len(faces) < 2:
            continue
        f1, f2 = faces[0], faces[-1]
        n1 = _face_normal(f1)
        n2 = _face_normal(f2)
        if not n1 or not n2:
            continue
        ang = math.acos(max(-1.0, min(1.0, abs(n1.Dot(n2)))))
        if ang < (math.pi - angle_thresh):
            total += _length_of_edge(edge)
    return total

def _largest_planar_faces_and_normals(shape):
    largest_area = 0.0
    normals = []
    for f in iter_faces(shape):
        if _face_type(f) == "planar":
            a = _area_of_face(f)
            largest_area = max(largest_area, a)
            n = _face_normal(f)
            if n: normals.append(n)
    clusters = _cluster_normals(normals)
    return largest_area, clusters

def _surface_areas_by_type(shape):
    from collections import defaultdict
    areas = defaultdict(float)
    for f in iter_faces(shape):
        t = _face_type(f)
        areas[t] += _area_of_face(f)
    return areas

def _section_perimeter_len(shape, z_values):
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
    total = 0.0
    explorer = cast(type[Any], TopExp_Explorer)
    for z in z_values:
        plane = gp_Pln(gp_Pnt(0,0,z), gp_Dir(0,0,1))
        sec = BRepAlgoAPI_Section(shape, plane, False); sec.Build()
        if not sec.IsDone(): continue
        w = sec.Shape()
        it = explorer(w, cast(TopAbs_ShapeEnum, TopAbs_EDGE))
        while it.More():
            e = to_edge(it.Current())
            total += _length_of_edge(e)
            it.Next()
    return total

def _min_wall_between_parallel_planes(shape):
    planes = []
    for f in iter_faces(shape):
        if _face_type(f) == "planar":
            n = _face_normal(f)
            if n:
                umin, umax, vmin, vmax = BRepTools_UVBounds(f)
                surf, _ = face_surface(f)
                sas = ShapeAnalysis_Surface(surf)
                pnt = sas.Value(0.5*(umin+umax), 0.5*(vmin+vmax))
                d = n.X()*pnt.X() + n.Y()*pnt.Y() + n.Z()*pnt.Z()
                planes.append((f, n, d, _bbox(f)))
    def overlap(a1, a2, b1, b2): return not (a2 < b1 or b2 < a1)
    min_th = None
    for i in range(len(planes)):
        f1, n1, d1, b1 = planes[i]
        for j in range(i+1, len(planes)):
            f2, n2, d2, b2 = planes[j]
            if abs(abs(n1.Dot(n2)) - 1.0) < 1e-3:
                nd = (abs(n1.X()), abs(n1.Y()), abs(n1.Z()))
                normal_axis = nd.index(max(nd))
                if normal_axis == 2:
                    ok = overlap(b1[0], b1[3], b2[0], b2[3]) and overlap(b1[1], b1[4], b2[1], b2[4])
                elif normal_axis == 1:
                    ok = overlap(b1[0], b1[3], b2[0], b2[3]) and overlap(b1[2], b1[5], b2[2], b2[5])
                else:
                    ok = overlap(b1[1], b1[4], b2[1], b2[4]) and overlap(b1[2], b1[5], b2[2], b2[5])
                if ok:
                    th = abs(d1 - d2)
                    if (min_th is None) or (th < min_th): min_th = th
    return min_th

def _hole_face_identifier(face, cylinder, face_bbox):
    centers = []
    try:
        exp = cast(Any, TopExp_Explorer)(face, TopAbs_EDGE)
    except Exception:
        exp = None
    while exp and exp.More():
        try:
            edge = to_edge(cast(Any, exp.Current()))
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() == GeomAbs_Circle:
                circ = curve.Circle()
                loc = circ.Location()
                centers.append((round(loc.X(), 3), round(loc.Y(), 3), round(loc.Z(), 3)))
        except Exception:
            pass
        finally:
            exp.Next()
    if centers:
        return ("edges", tuple(sorted(centers)))
    loc = cylinder.Axis().Location()
    center = (
        round(0.5 * (face_bbox[0] + face_bbox[3]), 3),
        round(0.5 * (face_bbox[1] + face_bbox[4]), 3),
        round(0.5 * (face_bbox[2] + face_bbox[5]), 3),
    )
    return (
        "fallback",
        (
            round(loc.X(), 3),
            round(loc.Y(), 3),
            round(loc.Z(), 3),
            *center,
        ),
    )


def _hole_groups_from_cylinders(shape, bbox=None):
    if bbox is None:
        bbox = _bbox(shape)
    groups = {}
    for f in iter_faces(shape):
        if _face_type(f) != "cylindrical":
            continue
        ga = GeomAdaptor_Surface(face_surface(f)[0])
        try:
            cyl = ga.Cylinder()
            r = abs(cyl.Radius())
            ax = cyl.Axis().Direction()
            fb = _bbox(f)

            def proj(x, y, z):
                return x * ax.X() + y * ax.Y() + z * ax.Z()

            span = abs(proj(fb[3], fb[4], fb[5]) - proj(fb[0], fb[1], fb[2]))
            dia = 2.0 * r
            bmin = proj(*bbox[:3])
            bmax = proj(*bbox[3:])
            bspan = abs(bmax - bmin)
            through = span > 0.9 * bspan
            key = (round(dia, 2), round(span, 2), through)
            hole_id = _hole_face_identifier(f, cyl, fb)
        except Exception:
            continue

        entry = groups.setdefault(
            key,
            {
                "dia_mm": round(dia, 2),
                "depth_mm": round(span, 2),
                "through": through,
                "count": 0,
                "_ids": set(),
            },
        )
        if hole_id in entry["_ids"]:
            continue
        entry["_ids"].add(hole_id)
        entry["count"] += 1
    for entry in groups.values():
        entry.pop("_ids", None)
    return list(groups.values())

def _turning_score(shape, areas_by_type):
    total = sum(areas_by_type.values()) or 1.0
    cyl_ratio = areas_by_type.get("cylindrical", 0.0)/total
    axes = []
    for f in iter_faces(shape):
        if _face_type(f) == "cylindrical":
            ga = GeomAdaptor_Surface(face_surface(f)[0])
            try: axes.append(ga.Cylinder().Axis().Direction())
            except Exception: pass
    if axes:
        ax = gp_Dir(sum(a.X() for a in axes) or 1e-9, sum(a.Y() for a in axes) or 1e-9, sum(a.Z() for a in axes) or 1e-9)
        align = sum(abs(ax.Dot(a)) for a in axes)/len(axes)
    else:
        align = 0.0
    score = max(0.0, min(1.0, 0.5*cyl_ratio + 0.5*align))
    xmin,ymin,zmin,xmax,ymax,zmax = _bbox(shape)
    Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin
    length = max(Lx, Ly, Lz)
    maxod = sorted([Lx, Ly, Lz])[-2]
    return {"GEO_Turning_Score_0to1": round(score,3), "GEO_MaxOD_mm": round(maxod,3), "GEO_Length_mm": round(length,3)}

def enrich_geo_occ(shape):
    xmin,ymin,zmin,xmax,ymax,zmax = _bbox(shape)
    L,W,H = xmax-xmin, ymax-ymin, zmax-zmin

    props = GProp_GProps()
    try:
        BRepGProp.VolumeProperties_s(shape, props); volume = float(props.Mass())
        com = props.CentreOfMass(); center = [com.X(), com.Y(), com.Z()]
    except Exception:
        volume, center = 0.0, [0.0,0.0,0.0]
    try:
        BRepGProp.SurfaceProperties_s(shape, props); area_total = float(props.Mass())
    except Exception:
        area_total = 0.0

    faces = 0
    for _ in iter_faces(shape): faces += 1

    areas_by_type = _surface_areas_by_type(shape)
    largest_planar, normal_clusters = _largest_planar_faces_and_normals(shape)
    unique_normals = len(normal_clusters)

    min_wall = _min_wall_between_parallel_planes(shape)
    thinwall_present = (min_wall is not None) and (min_wall < 1.0)

    z_vals = [zmin + 0.25*(zmax-zmin), zmin + 0.50*(zmax-zmin), zmin + 0.75*(zmax-zmin)]
    wedm_len = _section_perimeter_len(shape, z_vals)

    deburr_len = _sum_edge_length_sharp(shape, angle_thresh_deg=175.0)

    holes = _hole_groups_from_cylinders(shape, (xmin,ymin,zmin,xmax,ymax,zmax))

    bbox_vol = max(L*W*H, 1e-9)
    free_ratio = areas_by_type.get("freeform", 0.0)/max(area_total, 1e-9)
    complexity = (faces / (volume if volume>0 else bbox_vol))*100.0 + 30.0*free_ratio

    axis_dirs = [gp_Dir(1,0,0), gp_Dir(-1,0,0), gp_Dir(0,1,0), gp_Dir(0,-1,0), gp_Dir(0,0,1), gp_Dir(0,0,-1)]
    hit=0; tot=0
    for f in iter_faces(shape):
        n = _face_normal(f)
        if n:
            tot += 1
            if any(abs(n.Dot(a))>DOT_TOL for a in axis_dirs): hit += 1
    access = (hit/tot) if tot else 0.0

    geo = {
        "GEO-01_Length_mm": round(L,3),
        "GEO-02_Width_mm": round(W,3),
        "GEO-03_Height_mm": round(H,3),
        "GEO-Volume_mm3": round(volume,2),
        "GEO-SurfaceArea_mm2": round(area_total,2),
        "Feature_Face_Count": int(faces),
        "GEO_Area_Planar_mm2": round(areas_by_type.get("planar",0.0),2),
        "GEO_Area_Cyl_mm2": round(areas_by_type.get("cylindrical",0.0),2),
        "GEO_Area_Freeform_mm2": round(areas_by_type.get("freeform",0.0),2),
        "GEO_LargestPlane_Area_mm2": round(largest_planar,2),
        "GEO_Setup_UniqueNormals": int(unique_normals),
        "GEO_MinWall_mm": round(min_wall,3) if min_wall is not None else None,
        "GEO_ThinWall_Present": bool(thinwall_present),
        "GEO_WEDM_PathLen_mm": round(wedm_len,2),
        "GEO_Deburr_EdgeLen_mm": round(deburr_len,2),
        "GEO_Hole_Groups": holes,
        "GEO_Complexity_0to100": round(complexity,2),
        "GEO_3Axis_Accessible_Pct": round(access,3),
        "GEO_CenterOfMass": center,
        "OCC_Backend": "OCC"
    }
    geo.update(_turning_score(shape, areas_by_type))
    return geo

def enrich_geo_stl(path):
    import time
    start_time = time.time()
    logger.info("[%.2fs] Starting enrich_geo_stl for %s", time.time() - start_time, path)
    if not geometry.HAS_TRIMESH:
        raise RuntimeError("trimesh not available to process STL")

    logger.info("[%.2fs] Loading mesh...", time.time() - start_time)
    trimesh_mod = getattr(geometry, "trimesh", None)
    if trimesh_mod is None:
        raise RuntimeError("trimesh is unavailable despite HAS_TRIMESH flag")
    mesh = trimesh_mod.load(path, force="mesh")
    logger.info(
        "[%.2fs] Mesh loaded. Faces: %d",
        time.time() - start_time,
        len(mesh.faces),
    )

    if mesh.is_empty:
        raise RuntimeError("Empty STL mesh")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    L, W, H = float(xmax-xmin), float(ymax-ymin), float(zmax-zmin)
    area_total = float(mesh.area)
    volume = float(mesh.volume) if mesh.is_volume else 0.0
    faces = int(len(mesh.faces))

    logger.info("[%.2fs] Calculating WEDM length...", time.time() - start_time)
    z_vals = [zmin + 0.25*(zmax-zmin), zmin + 0.50*(zmax-zmin), zmin + 0.75*(zmax-zmin)]
    wedm_len = 0.0
    for z in z_vals:
        sec = mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1])
        if sec is None:
            continue
        planar = sec.to_2D() # Changed to_planar() to to_2D()
        try:
            wedm_len += float(planar.length)
        except Exception:
            pass
    logger.info("[%.2fs] WEDM length calculated.", time.time() - start_time)

    logger.info("[%.2fs] Calculating deburr length...", time.time() - start_time)
    try:
        import numpy as np
        angles = mesh.face_adjacency_angles
        edges = mesh.face_adjacency_edges
        if len(angles) and len(edges):
            sharp = angles > math.radians(15.0)
            vec = mesh.vertices[edges[:,0]] - mesh.vertices[edges[:,1]]
            lengths = np.linalg.norm(vec, axis=1)
            deburr_len = float(lengths[sharp].sum())
        else:
            deburr_len = 0.0
    except Exception:
        deburr_len = 0.0
    logger.info("[%.2fs] Deburr length calculated.", time.time() - start_time)

    bbox_vol = max(L*W*H, 1e-9)
    complexity = (faces / max(volume, bbox_vol)) * 100.0

    logger.info("[%.2fs] Calculating 3-axis accessibility...", time.time() - start_time)
    try:
        n = mesh.face_normals
        area_faces = mesh.area_faces.reshape(-1,1)
        import numpy as np
        axes = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float).T
        dots = np.abs(n @ axes)  # (F,6)
        close = (dots > DOT_TOL).any(axis=1)
        access = float((area_faces[close].sum() / area_faces.sum())) if area_faces.sum() > 0 else 0.0
    except Exception:
        access = 0.0
    logger.info("[%.2fs] 3-axis accessibility calculated.", time.time() - start_time)

    try:
        center = list(map(float, mesh.center_mass))
    except Exception:
        center = [0.0,0.0,0.0]

    logger.info("[%.2fs] Finished enrich_geo_stl.", time.time() - start_time)
    return {
        "GEO-01_Length_mm": round(L,3),
        "GEO-02_Width_mm": round(W,3),
        "GEO-03_Height_mm": round(H,3),
        "GEO-Volume_mm3": round(volume,2),
        "GEO-SurfaceArea_mm2": round(area_total,2),
        "Feature_Face_Count": faces,
        "GEO_LargestPlane_Area_mm2": None,
        "GEO_Setup_UniqueNormals": None,
        "GEO_MinWall_mm": None,
        "GEO_ThinWall_Present": False,
        "GEO_WEDM_PathLen_mm": round(wedm_len,2),
        "GEO_Deburr_EdgeLen_mm": round(deburr_len,2),
        "GEO_Hole_Groups": [],
        "GEO_Complexity_0to100": round(complexity,2),
        "GEO_3Axis_Accessible_Pct": round(access,3),
        "GEO_CenterOfMass": center,
        "OCC_Backend": "trimesh (STL)"
    }


def read_cad_any(path: str):
    from OCP.IFSelect import IFSelect_RetDone  # type: ignore[import]
    from OCP.IGESControl import IGESControl_Reader  # type: ignore[import]
    from OCP.TopoDS import TopoDS_Shape  # type: ignore[import]

    ext = Path(path).suffix.lower()
    if ext in (".step", ".stp"):
        return read_step_shape(path)
    if ext in (".iges", ".igs"):
        ig = IGESControl_Reader()
        if ig.ReadFile(path) != IFSelect_RetDone:
            raise RuntimeError("IGES read failed")
        ig.TransferRoots()
        return _shape_from_reader(ig)
    if ext == ".brep":
        shape_cls = cast(type[Any], TopoDS_Shape)
        s = shape_cls()
        if not cast(Any, BRepTools).Read(s, path, None):
            raise RuntimeError("BREP read failed")
        return s
    if ext == ".dxf":
        return geometry.read_dxf_as_occ_shape(path)
    if ext == ".dwg":
        conv = os.environ.get("ODA_CONVERTER_EXE") or os.environ.get("DWG2DXF_EXE")
        logger.info("Using DWG converter: %s", conv)
        dxf_path = convert_dwg_to_dxf(path)
        return geometry.read_dxf_as_occ_shape(dxf_path)
    raise RuntimeError(f"Unsupported CAD format: {ext}")

# ---- LLM hours inference ----
def infer_hours_and_overrides_from_geo(
    geo: dict,
    params: dict | None = None,
    rates: dict | None = None,
    *,
    client: LLMClient | None = None,
) -> dict:
    """Delegate to the shared LLM module for hour estimation."""

    return _infer_hours_and_overrides_from_geo(
        geo,
        params=params,
        rates=rates,
        client=client,
    )


_LLM_HOUR_ITEM_MAP: dict[str, str] = {
    "Programming_Hours": "Programming Hours",
    "CAM_Programming_Hours": "CAM Programming Hours",
    "Engineering_Hours": "Engineering (Docs/Fixture Design) Hours",
    "Fixture_Build_Hours": "Fixture Build Hours",
    "Roughing_Cycle_Time_hr": "Roughing Cycle Time",
    "Semi_Finish_Cycle_Time_hr": "Semi-Finish Cycle Time",
    "Finishing_Cycle_Time_hr": "Finishing Cycle Time",
    "InProcess_Inspection_Hours": "In-Process Inspection Hours",
    "Final_Inspection_Hours": "Final Inspection Hours",
    "CMM_Programming_Hours": "CMM Programming Hours",
    "CMM_RunTime_min": "CMM Run Time min",
    "Deburr_Hours": "Deburr Hours",
    "Tumble_Hours": "Tumbling Hours",
    "Blast_Hours": "Bead Blasting Hours",
    "Laser_Mark_Hours": "Laser Mark Hours",
    "Masking_Hours": "Masking Hours",
    "Saw_Waterjet_Hours": "Sawing Hours",
    "Assembly_Hours": "Assembly Hours",
    "Packaging_Labor_Hours": "Packaging Labor Hours",
}

_LLM_SETUP_ITEM_MAP: dict[str, str] = {
    "Milling_Setups": "Number of Milling Setups",
    "Setup_Hours_per_Setup": "Setup Hours / Setup",
}

_LLM_INSPECTION_ITEM_MAP: dict[str, str] = {
    "FAIR_Required": "FAIR Required",
    "Source_Inspection_Required": "Source Inspection Requirement",
}


def clamp_llm_hours(
    raw: Mapping[str, Any] | None,
    geo: Mapping[str, Any] | None,
    *,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sanitize LLM-derived hour estimates before applying them to the UI."""

    cleaned: dict[str, Any] = {}
    raw_map = cast(Mapping[str, Any], raw or {})

    hours_out: dict[str, float] = {}
    hours_val = raw_map.get("hours")
    if isinstance(hours_val, _MappingABC):
        hours_src: Mapping[str, Any] = cast(Mapping[str, Any], hours_val)
    else:
        hours_src = {}
    for key, value in hours_src.items():
        val = _coerce_float_or_none(value)
        if val is None:
            continue
        upper = 48.0
        if str(key).endswith("_min"):
            upper = 2400.0
        hours_out[str(key)] = clamp(float(val), 0.0, upper, 0.0)
    if hours_out:
        cleaned["hours"] = hours_out

    setups_out: dict[str, Any] = {}
    setups_val = raw_map.get("setups")
    if isinstance(setups_val, _MappingABC):
        setups_src: Mapping[str, Any] = cast(Mapping[str, Any], setups_val)
    else:
        setups_src = {}
    if setups_src:
        count_raw = setups_src.get("Milling_Setups")
        if count_raw is not None:
            try:
                setups_out["Milling_Setups"] = max(1, min(6, int(round(float(count_raw)))))
            except Exception:
                pass
        setup_hours = _coerce_float_or_none(setups_src.get("Setup_Hours_per_Setup"))
        if setup_hours is not None:
            setups_out["Setup_Hours_per_Setup"] = clamp(float(setup_hours), 0.0, LLM_ADDER_MAX, 0.0)
    if setups_out:
        cleaned["setups"] = setups_out

    inspection_out: dict[str, bool] = {}
    inspection_val = raw_map.get("inspection")
    if isinstance(inspection_val, _MappingABC):
        inspection_src: Mapping[str, Any] = cast(Mapping[str, Any], inspection_val)
    else:
        inspection_src = {}
    for key in _LLM_INSPECTION_ITEM_MAP:
        if key in inspection_src:
            inspection_out[key] = bool(inspection_src.get(key))
    if inspection_out:
        cleaned["inspection"] = inspection_out

    notes_raw = raw_map.get("notes")
    if isinstance(notes_raw, list):
        cleaned["notes"] = [str(n).strip() for n in notes_raw if str(n).strip()][:8]

    meta_raw = raw_map.get("_meta")
    if isinstance(meta_raw, _MappingABC):
        cleaned["_meta"] = dict(meta_raw)

    for key, value in raw_map.items():
        if key in {"hours", "setups", "inspection", "notes", "_meta"}:
            continue
        cleaned.setdefault(str(key), value)

    return cleaned


def apply_llm_hours_to_variables(
    df: pd.DataFrame | None,
    estimates: Mapping[str, Any] | None,
    *,
    allow_overwrite_nonzero: bool = False,
    log: dict | None = None,
) -> pd.DataFrame | None:
    """Apply sanitized LLM hour estimates to a variables dataframe."""

    if not _HAS_PANDAS or df is None:
        return df

    estimates_map = cast(Mapping[str, Any], estimates or {})
    df_out = df.copy(deep=True)
    normalized_items = df_out["Item"].astype(str).apply(_normalize_item_text)
    index_lookup = {norm: idx for idx, norm in zip(df_out.index, normalized_items)}

    def _write_value(label: str, value: Any, *, dtype: str = "number") -> None:
        nonlocal df_out, normalized_items, index_lookup
        if value is None:
            return
        normalized = _normalize_item_text(label)
        idx = index_lookup.get(normalized)
        new_value = value
        if idx is None:
            df_out = upsert_var_row(df_out, label, new_value, dtype=dtype)
            normalized_items = df_out["Item"].astype(str).apply(_normalize_item_text)
            index_lookup = {norm: idx for idx, norm in zip(df_out.index, normalized_items)}
            idx = index_lookup.get(normalized)
            previous = None
        else:
            previous = df_out.at[idx, "Example Values / Options"]
            if not allow_overwrite_nonzero:
                existing_val = _coerce_float_or_none(previous)
                if existing_val is not None and abs(existing_val) > 1e-9:
                    return
            df_out.at[idx, "Example Values / Options"] = new_value
            df_out.at[idx, "Data Type / Input Method"] = dtype
        if idx is None:
            return
        df_out.at[idx, "Example Values / Options"] = new_value
        df_out.at[idx, "Data Type / Input Method"] = dtype
        if log is not None:
            log.setdefault("llm_hours", []).append({
                "item": label,
                "value": new_value,
                "previous": previous,
            })

    hours_val = estimates_map.get("hours")
    if isinstance(hours_val, _MappingABC):
        hours_src: Mapping[str, Any] = cast(Mapping[str, Any], hours_val)
    else:
        hours_src = {}
    for key, value in hours_src.items():
        label = _LLM_HOUR_ITEM_MAP.get(str(key))
        if not label:
            continue
        val = _coerce_float_or_none(value)
        if val is None:
            continue
        _write_value(label, float(val), dtype="number")

    setups_val = estimates_map.get("setups")
    if isinstance(setups_val, _MappingABC):
        setups_src: Mapping[str, Any] = cast(Mapping[str, Any], setups_val)
    else:
        setups_src = {}
    for key, value in setups_src.items():
        label = _LLM_SETUP_ITEM_MAP.get(str(key))
        if not label:
            continue
        if key == "Milling_Setups":
            try:
                numeric = max(1, min(6, int(round(float(value)))))
            except Exception:
                continue
            _write_value(label, numeric, dtype="number")
        else:
            val = _coerce_float_or_none(value)
            if val is None:
                continue
            _write_value(label, float(val), dtype="number")

    inspection_val = estimates_map.get("inspection")
    if isinstance(inspection_val, _MappingABC):
        inspection_src: Mapping[str, Any] = cast(Mapping[str, Any], inspection_val)
    else:
        inspection_src = {}
    for key, value in inspection_src.items():
        label = _LLM_INSPECTION_ITEM_MAP.get(str(key))
        if not label:
            continue
        _write_value(label, "True" if bool(value) else "False", dtype="Checkbox")

    return df_out


def infer_shop_overrides_from_geo(
    geo: Mapping[str, Any] | None,
    *,
    params: Mapping[str, Any] | None = None,
    rates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return sanitized LLM output for the manual LLM tab."""

    estimates_raw = infer_hours_and_overrides_from_geo(
        dict(geo or {}),
        params=dict(params or {}),
        rates=dict(rates or {}),
    )
    cleaned = clamp_llm_hours(estimates_raw, geo or {}, params=params)
    return {
        "estimates": cleaned,
        "LLM_Adjustments": {},
    }

# --- WHICH SHEET ROWS MATTER TO THE ESTIMATOR --------------------------------
def _estimator_patterns():
    pats = [
        r"\b(Qty|Lot Size|Quantity)\b",
        r"\b(Overhead|Shop Overhead)\b", r"\b(Margin|Profit Margin)\b",
        r"\b(G&A|General\s*&\s*Admin)\b", r"\b(Contingency|Risk\s*Adder)\b",
        r"\b(Expedite|Rush\s*Fee)\b",
        r"\b(Net\s*Volume|Volume_net|Volume\s*\(cm\^?3\))\b",
        r"\b(Density|Material\s*Density)\b", r"\b(Scrap\s*%|Expected\s*Scrap)\b",
        r"\b(Material\s*Price.*(per\s*g|/g)|Unit\s*Price\s*/\s*g)\b",
        r"\b(Supplier\s*Min\s*Charge|min\s*charge)\b",
        r"\b(Material\s*MOQ)\b",
        r"\b(Material\s*Cost|Raw\s*Material\s*Cost)\b",

        r"(Programming|CAM\s*Programming|2D\s*CAM|3D\s*CAM|Simulation|Verification|DFM\s*Review|Tool\s*Library|Setup\s*Sheets)",
        r"\b(CAM\s*Programming|CAM\s*Sim|Post\s*Processing)\b",
        r"(Fixture\s*Design|Process\s*Sheet|Traveler|Documentation|Complex\s*Assembly\s*Doc)",

        r"(Fixture\s*Build|Custom\s*Fixture\s*Build)", r"(Fixture\s*Material\s*Cost|Fixture\s*Hardware)",
        r"(Project\s*(?:Manager|Management|Mgmt)|Program\s*Manager|Project\s*Coordination)",
        r"(Tool\s*(?:&|and)\s*Die\s*Maker|Toolmaker|Tool\s*(?:&|and)\s*Die\s*Support|Tooling\s*Support|Tool\s*Room)",

        r"(Roughing\s*Cycle\s*Time|Adaptive|HSM)", r"(Semi[- ]?Finish|Rest\s*Milling)", r"(Finishing\s*Cycle\s*Time)",
        r"(Number\s*of\s*Milling\s*Setups|Milling\s*Setups)", r"(Setup\s*Time\s*per\s*Setup|Setup\s*Hours\s*/\s*Setup)",
        r"(Thin\s*Wall\s*Factor|Thin\s*Wall\s*Multiplier)", r"(Tolerance\s*Multiplier|Tight\s*Tolerance\s*Factor)",
        r"(Finish\s*Multiplier|Surface\s*Finish\s*Factor)",

        r"(OD\s*Turning|OD\s*Rough/Finish|Outer\s*Diameter)", r"(ID\s*Boring|Drilling|Reaming)",
        r"(Threading|Tapping|Single\s*Point)", r"(Cut[- ]?Off|Parting)",

        r"(WEDM\s*Hours|Wire\s*EDM\s*Hours|EDM\s*Burn\s*Time)",
        r"(EDM\s*Length_mm|WEDM\s*Length_mm|EDM\s*Perimeter_mm)",
        r"(EDM\s*Thickness_mm|Stock\s*Thickness_mm)", r"(EDM\s*Passes|WEDM\s*Passes)",
        r"(EDM\s*Cut\s*Rate_mm\^?2/min|WEDM\s*Cut\s*Rate)", r"(EDM\s*Edge\s*Factor)",
        r"(WEDM\s*Wire\s*Cost\s*/\s*m|Wire\s*Cost\s*/m)", r"(Wire\s*Usage\s*m\s*/\s*mm\^?2)",

        r"(Sinker\s*EDM\s*Hours|Ram\s*EDM\s*Hours|Burn\s*Time)", r"(Sinker\s*Burn\s*Volume_mm3|EDM\s*Volume_mm3)",
        r"(Sinker\s*MRR_mm\^?3/min|EDM\s*MRR)", r"(Electrode\s*Count)", r"(Electrode\s*(Cost|Material).*)",

        r"(Surface\s*Grind|Pre[- ]?Op\s*Grinding|Blank\s*Squaring)", r"(Jig\s*Grind)",
        r"(OD/ID\s*Grind|Cylindrical\s*Grind)", r"(Grind\s*Volume_mm3|Grinding\s*Volume)",
        r"(Grind\s*MRR_mm\^?3/min|Grinding\s*MRR)", r"(Grinding\s*Passes)",
        r"(Dress\s*Frequency\s*passes)", r"(Dress\s*Time\s*min(\s*/\s*pass)?)",
        r"(Grinding\s*Wheel\s*Cost)",

        r"(Lapping|Honing|Polishing)",

        r"(Deburr|Edge\s*Break)", r"(Tumbling|Vibratory)", r"(Bead\s*Blasting|Sanding)",
        r"(Laser\s*Mark|Engraving)", r"(Masking\s*for\s*Plating|Masking)",

        r"(In[- ]?Process\s*Inspection)", r"(Final\s*Inspection|Manual\s*Inspection)",
        r"(CMM\s*Programming)", r"(CMM\s*Run\s*Time\s*min)", r"(CMM\s*Run\s*Time)\b",
        r"(FAIR|ISIR|PPAP)", r"(Source\s*Inspection)",

        r"(Sawing|Waterjet|Blank\s*Prep)",

        r"(Assembly|Manual\s*Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)", r"(Hardware|BOM\s*Cost|Fasteners|Bearings|CFM\s*Hardware)",

        r"(Outsourced\s*Heat\s*Treat|Heat\s*Treat\s*Cost)", r"(Plating|Coating|Anodize|Black\s*Oxide|DLC|PVD|CVD)",
        r"(Passivation|Cleaning\s*Vendor)",

        r"(Packaging|Boxing|Crating\s*Labor)", r"(Custom\s*Crate\s*NRE)", r"(Packaging\s*Materials|Foam|Trays)",
        r"(Freight|Shipping\s*Cost)", r"(Insurance|Liability\s*Adder)",

        r"(EHS|Compliance|Training|Waste\s*Handling)", r"(Gauge|Check\s*Fixture\s*NRE)",
    ]
    import re
    return [re.compile(p, re.I) for p in pats]

# --- APPLY LLM OUTPUT ---------------------------------------------------------
def apply_sheet_edits_to_df(df, sheet_edits: list[dict]):
    if not sheet_edits:
        return df, []
    items = df["Item"].astype(str)
    applied = []
    for e in sheet_edits:
        item = e["item"]; value = e["value"]
        m = items.str.fullmatch(item, case=False, na=False)
        if m.any():
            i = df.index[m][0]
            df.at[i, "Example Values / Options"] = value
            df.at[i, "Data Type / Input Method"] = "number"
            applied.append((item, value))
    return df, applied

# ================== LLM DECISION LOG / AUDIT ==================
import json as _json_audit
import time as _time_audit

LOGS_DIR = Path(r"D:\\CAD_Quoting_Tool\\Logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _coerce_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return x

# =============================================================

# ----------------- Variables & quote -----------------
try:
    pd  # type: ignore[name-defined]
    _HAS_PANDAS = True
except NameError:
    _HAS_PANDAS = False

CORE_COLS = ["Item", "Example Values / Options", "Data Type / Input Method"]

_MASTER_VARIABLES_CACHE: dict[str, Any] = {
    "loaded": False,
    "core": None,
    "full": None,
}

_SPEEDS_FEEDS_CACHE: dict[str, pd.DataFrame | None] = {}


def _coerce_core_types(df_core: pd.DataFrame) -> pd.DataFrame:
    """Light normalization for estimator expectations."""
    core = df_core.copy()
    core["Item"] = core["Item"].astype(str)
    core["Data Type / Input Method"] = core["Data Type / Input Method"].astype(str).str.lower()
    # Leave "Example Values / Options" as-is (can be text or number); estimator coerces later.
    return core


@dataclass(frozen=True)
class EditorControlSpec:
    """Instruction for rendering a Quote Editor control."""

    control: str
    entry_value: str = ""
    options: tuple[str, ...] = ()
    base_text: str = ""
    checkbox_state: bool = False
    checkbox_label: str = "Enabled"
    display_label: str = ""
    guessed_dropdown: bool = False


def _coerce_checkbox_state(value: Any, default: bool = False) -> bool:
    """Best-effort conversion from spreadsheet text to a boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            if math.isnan(value):
                return default
        except Exception:
            pass
        return bool(value)

    text = str(value).strip().lower()
    if not text:
        return default
    if text in _TRUTHY_TOKENS or text.startswith("y"):
        return True
    if text in _FALSY_TOKENS or text.startswith("n"):
        return False

    for part in re.split(r"[/|,\s]+", text):
        if part in _TRUTHY_TOKENS or part.startswith("y"):
            return True
        if part in _FALSY_TOKENS or part.startswith("n"):
            return False

    return default


_BOOL_PAIR_RE = re.compile(
    r"^\s*(true|false|yes|no|on|off)\s*(?:/|\||,|\s+or\s+)\s*(true|false|yes|no|on|off)\s*$",
    re.IGNORECASE,
)
_TRUTHY_TOKENS = {"true", "1", "yes", "y", "on"}
_FALSY_TOKENS = {"false", "0", "no", "n", "off"}


def _split_editor_options(text: str) -> list[str]:
    if not text:
        return []
    if _BOOL_PAIR_RE.match(text):
        parts = re.split(r"[/|,]|\s+or\s+", text, flags=re.IGNORECASE)
    else:
        parts = re.split(r"[,\n;|]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _looks_like_bool_options(options: Sequence[str]) -> bool:
    if not options or len(options) > 4:
        return False
    normalized = {opt.lower() for opt in options}
    return bool(normalized & _TRUTHY_TOKENS) and bool(normalized & _FALSY_TOKENS)


def _is_numeric_token(token: str) -> bool:
    token = str(token).strip()
    if not token:
        return False
    try:
        float(token.replace(",", ""))
        return True
    except Exception:
        return False


def _format_numeric_entry_value(raw: str) -> tuple[str, bool]:
    parsed = _coerce_float_or_none(raw)
    if parsed is None:
        return str(raw).strip(), False
    txt = f"{float(parsed):.6f}".rstrip("0").rstrip(".")
    return (txt if txt else "0", True)


def derive_editor_control_spec(dtype_source: str, example_value: Any) -> EditorControlSpec:
    """Classify a spreadsheet row into a UI control plan."""

    dtype_raw = re.sub(
        r"\s+",
        " ",
        str((dtype_source or "")).replace("\u00A0", " "),
    ).strip().lower()
    raw_value = ""
    if example_value is not None and not (isinstance(example_value, float) and math.isnan(example_value)):
        raw_value = str(example_value)
    initial_value = raw_value.strip()

    options = _split_editor_options(initial_value)
    looks_like_bool = _looks_like_bool_options(options)
    is_checkbox = "checkbox" in dtype_raw or looks_like_bool
    declared_dropdown = "dropdown" in dtype_raw or "select" in dtype_raw
    is_formula_like = any(term in dtype_raw for term in ("lookup", "calculated")) or (
        "value" in dtype_raw and ("lookup" in dtype_raw or "calculated" in dtype_raw)
    )
    if not is_formula_like and "value" in dtype_raw and not declared_dropdown:
        is_formula_like = True
    is_numeric_dtype = any(term in dtype_raw for term in ("number", "numeric", "decimal", "integer", "float"))

    has_formula_chars = bool(re.search(r"[=*()+{}]", initial_value))
    non_numeric_options = [opt for opt in options if not _is_numeric_token(opt)]
    guessed_dropdown = False

    if is_checkbox:
        normalized = initial_value.lower()
        if normalized in _TRUTHY_TOKENS:
            state = True
        elif normalized in _FALSY_TOKENS:
            state = False
        elif options:
            first = options[0].lower()
            state = first in _TRUTHY_TOKENS or first.startswith("y")
        else:
            state = _coerce_checkbox_state(initial_value, False)

        truthy_label = next(
            (
                opt.strip()
                for opt in options
                if opt and (opt.lower() in _TRUTHY_TOKENS or opt.lower().startswith("y"))
            ),
            "",
        )
        checkbox_label = truthy_label.title() if truthy_label else "Enabled"
        if checkbox_label.lower() in {"true", "1"}:
            checkbox_label = "Enabled"

        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Checkbox"
        return EditorControlSpec(
            control="checkbox",
            entry_value="True" if state else "False",
            options=tuple(options),
            checkbox_state=state,
            checkbox_label=checkbox_label,
            display_label=display,
        )

    if declared_dropdown or (not is_formula_like and not is_numeric_dtype and non_numeric_options and len(options) >= 2 and not has_formula_chars):
        guessed_dropdown = not declared_dropdown
        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Dropdown"
        selected = initial_value or (options[0] if options else "")
        if options and selected not in options:
            selected = options[0]
        return EditorControlSpec(
            control="dropdown",
            entry_value=selected,
            options=tuple(options),
            display_label=(display + " (auto)" if guessed_dropdown else display),
            guessed_dropdown=guessed_dropdown,
        )

    if is_formula_like or has_formula_chars:
        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Lookup / Calculated"
        entry_value, parsed_ok = _format_numeric_entry_value(initial_value)
        if not parsed_ok:
            entry_value = ""
        base_text = initial_value if initial_value else ""
        return EditorControlSpec(
            control="formula",
            entry_value=entry_value,
            base_text=base_text,
            display_label=display,
        )

    if is_numeric_dtype or (_is_numeric_token(initial_value) and not non_numeric_options):
        display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Number"
        entry_value = ""
        if initial_value:
            entry_value, parsed_ok = _format_numeric_entry_value(initial_value)
            if not parsed_ok:
                entry_value = initial_value
        return EditorControlSpec(
            control="number",
            entry_value=entry_value,
            display_label=display,
        )

    display = dtype_source.strip() if isinstance(dtype_source, str) and dtype_source.strip() else "Text"
    return EditorControlSpec(
        control="text",
        entry_value=initial_value,
        display_label=display,
        options=tuple(options),
    )

def sanitize_vars_df(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy containing only the 3 core columns the estimator needs.
    - Does NOT mutate or overwrite the original file.
    - Creates missing core columns as blanks.
    - Normalizes types.
    """
    # Try to map any variant header names to our canon names (case/space tolerant)
    canon = {str(c).strip().lower(): c for c in df_full.columns}

    # Build list of the *actual* columns that correspond to CORE_COLS (if present)
    actual = []
    for want in CORE_COLS:
        key = want.strip().lower()
        # allow loose matches for common variants
        candidates = [
            canon.get(key),
            canon.get(key.replace(" / ", " ").replace("/", " ")),
            canon.get(key.replace(" ", "")),
        ]
        col = next((c for c in candidates if c in df_full.columns), None)
        actual.append(col)

    # Start with what we can find; add any missing columns as empty
    core = pd.DataFrame()
    for want, col in zip(CORE_COLS, actual):
        if col is not None:
            core[want] = df_full[col]
        else:
            core[want] = "" if want != "Example Values / Options" else None

    return _coerce_core_types(core)

def read_variables_file(
    path: str, return_full: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read .xlsx/.csv, keep original data intact, and return a sanitized copy for the estimator.
    - If return_full=True, returns (core_df, full_df); otherwise returns core_df only.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas required (conda/pip install pandas)")

    lp = path.lower()
    if lp.endswith(".xlsx"):
        # Prefer a sheet named "Variables" if it exists; else first sheet
        xl = pd.ExcelFile(path)
        sheet_name = "Variables" if "Variables" in xl.sheet_names else xl.sheet_names[0]
        df_full = pd.read_excel(path, sheet_name=sheet_name)
    elif lp.endswith(".csv"):
        delimiter = ","
        try:
            with open(path, encoding="utf-8-sig") as sniff:
                header_line = sniff.readline()
            if "\t" in header_line:
                delimiter = "\t"
        except Exception:
            delimiter = ","

        read_csv_kwargs: dict[str, Any] = {"encoding": "utf-8-sig"}
        if delimiter == "\t":
            read_csv_kwargs["sep"] = "\t"

        try:
            df_full = pd.read_csv(path, **read_csv_kwargs)
        except Exception as csv_err:
            # When pandas' default engine hits irregular commas (extra columns
            # from free-form notes, etc.) fall back to a more forgiving parser
            # that collapses spill-over cells into the final column so that the
            # sheet still loads for the estimator instead of forcing users back
            # through the file picker.
            import csv as _csv

            csv_delimiter = "\t" if delimiter == "\t" else ","

            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(_csv.reader(f, delimiter=csv_delimiter))

            if not rows:
                raise csv_err

            header = rows[0]
            if not header:
                raise csv_err
            normalized_dicts: list[dict[str, str]] = []
            for row in rows[1:]:
                if len(row) > len(header):
                    keep = len(header) - 1
                    merged_tail = csv_delimiter.join(row[keep:]) if keep >= 0 else ""
                    row = row[:keep] + [merged_tail]
                elif len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                normalized_dicts.append(dict(zip(header, row)))

            try:
                df_full = pd.DataFrame(normalized_dicts)
            except Exception:
                raise csv_err
    else:
        raise ValueError("Variables must be .xlsx or .csv")

    core = sanitize_vars_df(df_full)

    return (core, df_full) if return_full else core


def _load_master_variables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the packaged master variables sheet once and serve cached copies."""
    if not _HAS_PANDAS:
        return (None, None)

    global _MASTER_VARIABLES_CACHE
    cache = _MASTER_VARIABLES_CACHE

    if cache.get("loaded"):
        core_cached = cache.get("core")
        full_cached = cache.get("full")
        core_copy = (
            core_cached.copy()
            if _HAS_PANDAS and isinstance(core_cached, pd.DataFrame)
            else None
        )
        full_copy = (
            full_cached.copy()
            if _HAS_PANDAS and isinstance(full_cached, pd.DataFrame)
            else None
        )
        return (core_copy, full_copy)

    master_path = Path(__file__).with_name("Master_Variables.csv")
    fallback = Path(r"D:\CAD_Quoting_Tool\Master_Variables.csv")
    if not master_path.exists() and fallback.exists():
        master_path = fallback
    if not master_path.exists():
        cache["loaded"] = True
        cache["core"] = None
        cache["full"] = None
        return (None, None)

    try:
        core_df, full_df = read_variables_file(str(master_path), return_full=True)
        core_df = cast(pd.DataFrame, core_df)
        full_df = cast(pd.DataFrame, full_df)
    except Exception:
        logger.warning("Failed to load master variables CSV from %s", master_path, exc_info=True)
        cache["loaded"] = True
        cache["core"] = None
        cache["full"] = None
        return (None, None)

    cache["loaded"] = True
    cache["core"] = core_df
    cache["full"] = full_df
    return (core_df.copy(), full_df.copy())


def find_variables_near(cad_path: str):
    """Look for variables.* in the same folder, then one level up."""
    import os
    folder = os.path.dirname(cad_path)
    names = ["variables.xlsx", "variables.csv"]
    subs  = ["variables", "vars"]

    def _scan(dirpath):
        try:
            listing = os.listdir(dirpath)
        except Exception:
            return None
        low = {e.lower(): e for e in listing}
        for n in names:
            if n in low:
                return os.path.join(dirpath, low[n])
        for e in listing:
            le = e.lower()
            if le.endswith((".xlsx", ".csv")) and any(s in le for s in subs):
                return os.path.join(dirpath, e)
        return None

    hit = _scan(folder)
    if hit:
        return hit
    parent = os.path.dirname(folder)
    if os.path.isdir(parent):
        return _scan(parent)
    return None


def parse_pct(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v/100.0 if v > 1.0 else v

# ---------- Pretty printer for quote results ----------

PROCESS_LABEL_OVERRIDES: dict[str, str] = {
    "finishing_deburr": "Finishing/Deburr",
    "saw_waterjet": "Saw/Waterjet",
}

PREFERRED_PROCESS_BUCKET_ORDER: tuple[str, ...] = (
    "milling",
    "drilling",
    "counterbore",
    "countersink",
    "tapping",
    "grinding",
    "finishing_deburr",
    "saw_waterjet",
    "inspection",
    "assembly",
    "packaging",
    "misc",
)


CANON_MAP: dict[str, str] = {
    "deburr": "finishing_deburr",
    "deburring": "finishing_deburr",
    "finish_deburr": "finishing_deburr",
    "finishing_deburr": "finishing_deburr",
    "finishing": "finishing_deburr",
    "finishing/deburr": "finishing_deburr",
    "inspection": "inspection",
    "milling": "milling",
    "drilling": "drilling",
    "counterbore": "counterbore",
    "tapping": "tapping",
    "grinding": "grinding",
    "saw_waterjet": "saw_waterjet",
    "fixture_build_amortized": "fixture_build_amortized",
    "programming_amortized": "programming_amortized",
    "misc": "misc",
}

PLANNER_META: frozenset[str] = frozenset({"planner_labor", "planner_machine", "planner_total"})
_HIDE_IN_BUCKET_VIEW: frozenset[str] = frozenset({*PLANNER_META, "misc"})
_PREFERRED_BUCKET_VIEW_ORDER: tuple[str, ...] = (
    "programming",
    "programming_amortized",
    "fixture_build",
    "fixture_build_amortized",
    "milling",
    "drilling",
    "counterbore",
    "countersink",
    "tapping",
    "grinding",
    "finishing_deburr",
    "saw_waterjet",
    "wire_edm",
    "sinker_edm",
    "inspection",
    "assembly",
    "toolmaker_support",
    "packaging",
    "ehs_compliance",
    "turning",
    "lapping_honing",
)


def _normalize_bucket_key(name: str | None) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    aliases = {
        "finishing": "finishing_deburr",
        "deburring": "finishing_deburr",
        "finish_deburr": "finishing_deburr",
        "finishing_deburr": "finishing_deburr",
        "deburr": "finishing_deburr",
        "saw": "saw_waterjet",
        "saw_waterjet": "saw_waterjet",
        "waterjet": "saw_waterjet",
        "counter_bore": "counterbore",
        "counter_bores": "counterbore",
        "counterbore": "counterbore",
        "counter_sink": "countersink",
        "counter_sinks": "countersink",
        "countersink": "countersink",
        "thread_mill": "tapping",
    }
    return aliases.get(text, text)


def _rate_key_for_bucket(bucket: str | None) -> str | None:
    canon = _normalize_bucket_key(bucket)
    mapping = {
        "milling": "MillingRate",
        "drilling": "DrillingRate",
        "counterbore": "DrillingRate",
        "countersink": "DrillingRate",
        "tapping": "TappingRate",
        "grinding": "SurfaceGrindRate",
        "inspection": "InspectionRate",
        "finishing_deburr": "DeburrRate",
        "assembly": "AssemblyRate",
        "packaging": "PackagingRate",
        "saw_waterjet": "SawWaterjetRate",
        "misc": "MillingRate",
    }
    return mapping.get(canon)


def _canonical_bucket_key(name: str | None) -> str:
    normalized = _normalize_bucket_key(name)
    if not normalized:
        return ""
    canon = CANON_MAP.get(normalized)
    if canon:
        return canon

    tokens = [tok for tok in normalized.split("_") if tok]
    if any(tok.startswith("deburr") or tok.startswith("debur") for tok in tokens):
        return "finishing_deburr"
    if any(tok.startswith("finish") for tok in tokens):
        return "finishing_deburr"

    return normalized


def _preferred_order_then_alpha(keys: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    remaining = {key for key in keys if key}
    ordered: list[str] = []

    for preferred in _PREFERRED_BUCKET_VIEW_ORDER:
        if preferred in remaining:
            ordered.append(preferred)
            seen.add(preferred)
    remaining -= seen

    if remaining:
        ordered.extend(sorted(remaining))

    return ordered


def _coerce_bucket_metric(data: Mapping[str, Any] | None, *candidates: str) -> float:
    if not isinstance(data, Mapping):
        return 0.0
    for key in candidates:
        if key in data:
            try:
                return float(data.get(key) or 0.0)
            except Exception:
                continue
    return 0.0


def _prepare_bucket_view(raw_view: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a canonicalized bucket view suitable for cost table rendering."""

    prepared: dict[str, Any] = {}
    if isinstance(raw_view, Mapping):
        for key, value in raw_view.items():
            if key == "buckets":
                continue
            prepared[key] = copy.deepcopy(value)

    source = raw_view.get("buckets") if isinstance(raw_view, Mapping) else None
    if not isinstance(source, Mapping):
        source = raw_view if isinstance(raw_view, Mapping) else {}

    folded: dict[str, dict[str, float]] = {}

    for raw_key, raw_info in source.items():
        canon = _canonical_bucket_key(raw_key)
        if not canon:
            continue
        if canon in _HIDE_IN_BUCKET_VIEW or canon.startswith("planner_"):
            continue
        info_map = raw_info if isinstance(raw_info, Mapping) else {}
        bucket = folded.setdefault(
            canon,
            {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0},
        )

        minutes = _coerce_bucket_metric(info_map, "minutes")
        labor = _coerce_bucket_metric(info_map, "labor$", "labor_cost", "labor")
        machine = _coerce_bucket_metric(info_map, "machine$", "machine_cost", "machine")
        total = _coerce_bucket_metric(info_map, "total$", "total_cost", "total")

        if math.isclose(total, 0.0, abs_tol=1e-9):
            total = labor + machine

        bucket["minutes"] += minutes
        bucket["labor$"] += labor
        bucket["machine$"] += machine
        bucket["total$"] += total

    totals = {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0}

    for canon, metrics in list(folded.items()):
        minutes = round(float(metrics.get("minutes", 0.0)), 2)
        labor = round(float(metrics.get("labor$", 0.0)), 2)
        machine = round(float(metrics.get("machine$", 0.0)), 2)
        total = round(float(metrics.get("total$", labor + machine)), 2)

        if (
            math.isclose(minutes, 0.0, abs_tol=0.01)
            and math.isclose(labor, 0.0, abs_tol=0.01)
            and math.isclose(machine, 0.0, abs_tol=0.01)
            and math.isclose(total, 0.0, abs_tol=0.01)
        ):
            folded.pop(canon, None)
            continue

        metrics["minutes"] = minutes
        metrics["labor$"] = labor
        metrics["machine$"] = machine
        metrics["total$"] = total

        totals["minutes"] += minutes
        totals["labor$"] += labor
        totals["machine$"] += machine
        totals["total$"] += total

    prepared["buckets"] = folded
    prepared["order"] = _preferred_order_then_alpha(folded.keys())
    prepared["totals"] = {key: round(value, 2) for key, value in totals.items()}

    return prepared


def canonicalize_costs(process_costs: Mapping[str, Any] | None) -> dict[str, float]:
    items: Iterable[tuple[Any, Any]]
    if isinstance(process_costs, Mapping):
        items = process_costs.items()
    else:
        try:
            items = dict(process_costs or {}).items()  # type: ignore[arg-type]
        except Exception:
            items = []

    out: dict[str, float] = {}
    for raw_key, raw_value in items:
        key = str(raw_key).strip().lower()
        if not key:
            continue
        key = key.replace(" ", "_").replace("-", "_").replace("/", "_")
        if not key:
            continue
        if key.startswith("planner_") or key in PLANNER_META:
            continue
        canon_key = CANON_MAP.get(key, key)
        try:
            amount = float(raw_value or 0.0)
        except Exception:
            amount = 0.0
        out[canon_key] = out.get(canon_key, 0.0) + amount

    if out.get("misc", 0.0) <= 1.0:
        out.pop("misc", None)
    return out


def _process_label(key: str | None) -> str:
    raw = str(key or "").strip().lower().replace(" ", "_")
    canon = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    alias = {
        "finishing_deburr": "finishing/deburr",
        "deburr": "finishing/deburr",
        "deburring": "finishing/deburr",
        "finish_deburr": "finishing/deburr",
        "saw_waterjet": "saw / waterjet",
        "counter_bore": "counterbore",
        "counter_sink": "countersink",
        "prog_amortized": "programming (amortized)",
        "programming_amortized": "programming (amortized)",
        "fixture_build_amortized": "fixture build (amortized)",
    }.get(canon, canon)
    if alias == "saw / waterjet":
        return "Saw / Waterjet"
    return alias.replace("_", " ").title()


def _display_bucket_label(
    canon_key: str,
    label_overrides: Mapping[str, str] | None = None,
) -> str:
    overrides = sdict(label_overrides)
    if canon_key in overrides:
        return overrides[canon_key]
    return _process_label(canon_key)


def _format_planner_bucket_line(
    canon_key: str,
    amount: float,
    meta: Mapping[str, Any] | None,
    *,
    planner_bucket_display_map: Mapping[str, Mapping[str, Any]] | None = None,
    label_overrides: Mapping[str, str] | None = None,
    currency_formatter: Callable[[float], str] | None = None,
) -> tuple[str | None, float, float, float]:
    if not planner_bucket_display_map:
        return (None, amount, 0.0, 0.0)
    info = planner_bucket_display_map.get(canon_key)
    if not isinstance(info, Mapping):
        return (None, amount, 0.0, 0.0)

    try:
        minutes_val = float(info.get("minutes", 0.0) or 0.0)
    except Exception:
        minutes_val = 0.0

    hr_val = 0.0
    if isinstance(meta, Mapping):
        try:
            hr_val = float(meta.get("hr", 0.0) or 0.0)
        except Exception:
            hr_val = 0.0
    if hr_val <= 0 and minutes_val > 0:
        hr_val = minutes_val / 60.0

    total_cost = amount
    for key_option in ("total_cost", "total$", "total"):
        if key_option in info:
            try:
                candidate = float(info.get(key_option) or 0.0)
            except Exception:
                continue
            if candidate:
                total_cost = candidate
                break

    rate_val = 0.0
    if isinstance(meta, Mapping):
        try:
            rate_val = float(meta.get("rate", 0.0) or 0.0)
        except Exception:
            rate_val = 0.0
    if rate_val <= 0 and hr_val > 0 and total_cost > 0:
        rate_val = total_cost / hr_val

    if currency_formatter is None:
        currency_formatter = lambda x: f"${float(x):,.2f}"  # pragma: no cover

    hours_text = f"{max(hr_val, 0.0):.2f} hr"
    if hr_val <= 0:
        hours_text = "0.00 hr"
    if rate_val > 0:
        rate_text = f"{currency_formatter(rate_val)}/hr"
    else:
        rate_text = "—"

    display_override = (
        f"{_display_bucket_label(canon_key, label_overrides)}: {hours_text} × {rate_text} →"
    )
    return (display_override, float(total_cost), hr_val, rate_val)


def _extract_bucket_map(source: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    bucket_map: dict[str, dict[str, Any]] = {}
    if not isinstance(source, Mapping):
        return bucket_map
    struct: Mapping[str, Any] = source
    buckets_obj = source.get("buckets") if isinstance(source, Mapping) else None
    if isinstance(buckets_obj, Mapping):
        struct = buckets_obj
    for raw_key, raw_value in struct.items():
        canon = _canonical_bucket_key(raw_key)
        if not canon:
            continue
        if isinstance(raw_value, Mapping):
            bucket_map[canon] = {str(k): v for k, v in raw_value.items()}
        else:
            bucket_map[canon] = {}
    return bucket_map


@dataclass(frozen=True)
class ProcessDisplayEntry:
    process_key: str
    canonical_key: str
    label: str
    amount: float
    detail_bits: tuple[str, ...]
    display_override: str | None = None


class PlannerBucketOp(TypedDict):
    op: str
    minutes: float
    machine: float
    labor: float
    total: float


def _iter_ordered_process_entries(
    process_costs: Mapping[str, Any] | None,
    *,
    process_meta: Mapping[str, Any] | None,
    applied_process: Mapping[str, Any] | None,
    show_zeros: bool,
    planner_bucket_display_map: Mapping[str, Mapping[str, Any]] | None,
    label_overrides: Mapping[str, str] | None,
    currency_formatter: Callable[[float], str],
) -> Iterable[ProcessDisplayEntry]:
    costs = list((process_costs or {}).items())
    seen_keys: set[str] = set()
    ordered_items: list[tuple[str, Any]] = []

    for bucket in PREFERRED_PROCESS_BUCKET_ORDER:
        for key, value in costs:
            if _normalize_bucket_key(key) != bucket:
                continue
            include = False
            try:
                include = (float(value) > 0) or show_zeros
            except Exception:
                include = show_zeros
            if not include:
                continue
            ordered_items.append((key, value))
            seen_keys.add(key)

    remaining_items: list[tuple[str, Any]] = []
    for key, value in costs:
        if key in seen_keys:
            continue
        include = False
        try:
            include = (float(value) > 0) or show_zeros
        except Exception:
            include = show_zeros
        if include:
            remaining_items.append((key, value))

    remaining_items.sort(key=lambda kv: _normalize_bucket_key(kv[0]))
    ordered_items.extend(remaining_items)

    meta_lookup: dict[str, Mapping[str, Any]] = {}
    if isinstance(process_meta, Mapping):
        for raw_key, raw_meta in process_meta.items():
            if not isinstance(raw_meta, Mapping):
                continue
            key_lower = str(raw_key).lower()
            meta_lookup[key_lower] = raw_meta
            canon = _canonical_bucket_key(raw_key)
            if canon and canon not in meta_lookup:
                meta_lookup[canon] = raw_meta

    applied_lookup: dict[str, Mapping[str, Any]] = {}
    if isinstance(applied_process, Mapping):
        for raw_key, raw_meta in applied_process.items():
            if not isinstance(raw_meta, Mapping):
                continue
            key_lower = str(raw_key).lower()
            applied_lookup[key_lower] = raw_meta
            canon = _canonical_bucket_key(raw_key)
            if canon and canon not in applied_lookup:
                applied_lookup[canon] = raw_meta

    for key, raw_value in ordered_items:
        canon_key = _canonical_bucket_key(key)
        meta_key = str(key).lower()
        meta = meta_lookup.get(meta_key, {})
        if not meta:
            meta = meta_lookup.get(canon_key, {})

        try:
            value_float = float(raw_value or 0.0)
        except Exception:
            value_float = 0.0

        try:
            extra_val = float(meta.get("base_extra", 0.0) or 0.0)
        except Exception:
            extra_val = 0.0
        try:
            meta_hr = float(meta.get("hr", 0.0) or 0.0)
        except Exception:
            meta_hr = 0.0
        try:
            meta_rate = float(meta.get("rate", 0.0) or 0.0)
        except Exception:
            meta_rate = 0.0

        display_override, amount_override, display_hr, display_rate = _format_planner_bucket_line(
            canon_key,
            value_float,
            meta,
            planner_bucket_display_map=planner_bucket_display_map,
            label_overrides=label_overrides,
            currency_formatter=currency_formatter,
        )
        use_display = display_override is not None
        label = (
            _display_bucket_label(canon_key, label_overrides)
            if canon_key
            else _process_label(key)
        )

        detail_bits: list[str] = []
        if not use_display:
            if display_hr > 0:
                rate_for_detail = display_rate if display_rate > 0 else meta_rate
                if rate_for_detail > 0:
                    detail_bits.append(
                        f"{display_hr:.2f} hr @ {currency_formatter(rate_for_detail)}/hr"
                    )
                else:
                    detail_bits.append(f"{display_hr:.2f} hr")
            elif meta_hr > 0:
                if meta_rate > 0:
                    detail_bits.append(
                        f"{meta_hr:.2f} hr @ {currency_formatter(meta_rate)}/hr"
                    )
                else:
                    detail_bits.append(f"{meta_hr:.2f} hr")

        rate_for_extra = display_rate if display_rate > 0 else meta_rate
        if abs(extra_val) > 1e-6:
            if (not use_display and display_hr <= 1e-6 and rate_for_extra > 0) or (
                use_display and display_hr <= 1e-6 and rate_for_extra > 0
            ):
                extra_hours = extra_val / rate_for_extra
                detail_bits.append(f"{extra_hours:.2f} hr @ {currency_formatter(rate_for_extra)}/hr")

        notes_entry = applied_lookup.get(meta_key, {})
        if not notes_entry:
            notes_entry = applied_lookup.get(canon_key, {})
        proc_notes = notes_entry.get("notes") if isinstance(notes_entry, Mapping) else None
        if proc_notes:
            detail_bits.append("LLM: " + ", ".join(proc_notes))

        amount_to_use = amount_override if use_display else value_float

        yield ProcessDisplayEntry(
            process_key=str(key),
            canonical_key=canon_key,
            label=label,
            amount=amount_to_use,
            detail_bits=tuple(detail_bits),
            display_override=display_override,
        )


def render_quote(
    result: dict,
    currency: str = "$",
    show_zeros: bool = False,
    llm_explanation: str = "",
    page_width: int = 74,
) -> str:
    """Pretty printer for a full quote with auto-included non-zero lines."""
    breakdown    = result.get("breakdown", {}) or {}
    totals       = breakdown.get("totals", {}) or {}
    declared_labor_total = float(totals.get("labor_cost", 0.0) or 0.0)
    nre_detail   = breakdown.get("nre_detail", {}) or {}
    nre          = breakdown.get("nre", {}) or {}
    material_raw = breakdown.get("material", {}) or {}
    if isinstance(material_raw, Mapping):
        material_block = dict(material_raw)
    else:  # tolerate legacy iterables or unexpected values
        try:
            material_block = dict(material_raw or {})
        except Exception:
            material_block = {}
    material_selection_raw = breakdown.get("material_selected") or {}
    if isinstance(material_selection_raw, Mapping):
        material_selection = dict(material_selection_raw)
    else:
        try:
            material_selection = dict(material_selection_raw or {})
        except Exception:
            material_selection = {}
    material = material_block
    drilling_meta = breakdown.get("drilling_meta", {}) or {}
    process_costs_raw = breakdown.get("process_costs", {}) or {}
    process_costs = (
        dict(process_costs_raw)
        if isinstance(process_costs_raw, Mapping)
        else dict(process_costs_raw or {})
    )
    pass_through_raw = breakdown.get("pass_through", {}) or {}
    if isinstance(pass_through_raw, Mapping):
        pass_through = dict(pass_through_raw)
    else:
        try:
            pass_through = dict(pass_through_raw or {})
        except Exception:
            pass_through = {}
    applied_pcts = breakdown.get("applied_pcts", {}) or {}
    process_meta_raw = breakdown.get("process_meta", {}) or {}
    applied_process_raw = breakdown.get("applied_process", {}) or {}
    process_meta: dict[str, Any] = {}
    bucket_alias_map: dict[str, str] = {}
    applied_process: dict[str, Any] = {}
    rates        = breakdown.get("rates", {}) or {}
    params       = breakdown.get("params", {}) or {}
    nre_cost_details = breakdown.get("nre_cost_details", {}) or {}
    labor_cost_details_input_raw = breakdown.get("labor_cost_details", {}) or {}

    # Shipping is displayed in exactly one section of the quote to avoid
    # conflicting totals.  Prefer the pass-through value when available and
    # otherwise fall back to a material-specific entry before rendering.
    shipping_pipeline = "pass_through"  # pipeline (a) – display under Pass-Through
    shipping_source = "pass_through"
    shipping_raw_value: Any = pass_through.get("Shipping")
    if not shipping_raw_value:
        shipping_raw_value = material_block.get("shipping")
        if shipping_raw_value:
            shipping_source = "material"
        else:
            shipping_source = None
    shipping_total = float(_coerce_float_or_none(shipping_raw_value) or 0.0)
    if shipping_pipeline == "pass_through":
        pass_through["Shipping"] = shipping_total
        material_block.pop("shipping", None)
        show_material_shipping = False
    else:
        pass_through.pop("Shipping", None)
        if shipping_source:
            material_block["shipping"] = shipping_total
        show_material_shipping = (
            (shipping_total > 0)
            or (shipping_total == 0 and bool(shipping_source) and show_zeros)
        )

    def _merge_detail_text(existing: str | None, new_value: Any) -> str:
        segments: list[str] = []
        seen: set[str] = set()
        for candidate in (existing, new_value):
            if candidate is None:
                continue
            for segment in re.split(r";\s*", str(candidate)):
                seg = segment.strip()
                if not seg:
                    continue
                if seg not in seen:
                    segments.append(seg)
                    seen.add(seg)
        if segments:
            return "; ".join(segments)
        if existing is None and new_value is None:
            return ""
        if new_value is None:
            return existing or ""
        return str(new_value)

    labor_cost_details_input: dict[str, str] = {}
    for raw_label, raw_detail in labor_cost_details_input_raw.items():
        canonical_label, _ = _canonical_amortized_label(raw_label)
        if not canonical_label:
            canonical_label = str(raw_label)
        merged = _merge_detail_text(labor_cost_details_input.get(canonical_label), raw_detail)
        labor_cost_details_input[canonical_label] = merged

    labor_cost_details: dict[str, str] = dict(labor_cost_details_input)
    labor_cost_details_seed: dict[str, str] = dict(labor_cost_details_input)

    labor_cost_totals_raw = breakdown.get("labor_costs", {}) or {}
    labor_cost_totals: dict[str, float] = {}
    for key, value in labor_cost_totals_raw.items():
        canonical_label, _ = _canonical_amortized_label(key)
        if not canonical_label:
            canonical_label = str(key)
        try:
            labor_cost_totals[canonical_label] = labor_cost_totals.get(canonical_label, 0.0) + float(value)
        except Exception:
            continue
    labor_costs_display: dict[str, float] = {}
    hour_summary_entries: dict[str, tuple[float, bool]] = {}
    prog_hr: float = 0.0
    direct_cost_details = breakdown.get("direct_cost_details", {}) or {}
    material_net_cost: float | None = None
    pass_through_total = 0.0
    display_machine = 0.0
    display_labor_for_ladder = 0.0
    hour_summary_entries: dict[str, tuple[float, bool]] = {}
    qty_raw = result.get("qty")
    if qty_raw in (None, ""):
        qty_raw = breakdown.get("qty")
    qty = int(qty_raw or 1)
    price        = float(result.get("price", totals.get("price", 0.0)))

    g = breakdown.get("geo_context") or breakdown.get("geo") or result.get("geo") or {}
    if not isinstance(g, dict):
        g = {}

    # Optional: LLM decision bullets can be placed either on result or breakdown
    llm_notes = (result.get("llm_notes") or breakdown.get("llm_notes") or [])[:8]
    notes_order = [
        str(note).strip() for note in llm_notes if str(note).strip()
    ]

    # ---- helpers -------------------------------------------------------------
    divider = "-" * int(page_width)

    red_flags: list[str] = []
    hour_summary_entries: dict[str, tuple[float, bool]] = {}
    for source in (result, breakdown):
        flags_raw = source.get("red_flags") if isinstance(source, Mapping) else None
        if isinstance(flags_raw, (list, tuple, set)):
            for flag in flags_raw:
                text = str(flag).strip()
                if text and text not in red_flags:
                    red_flags.append(text)

    def _m(x) -> str:
        return f"{currency}{float(x):,.2f}"

    def _h(x) -> str:
        return f"{float(x):.2f} hr"

    def _hours_with_rate_text(hours: Any, rate: Any) -> str:
        try:
            hours_val = float(hours or 0.0)
        except Exception:
            hours_val = 0.0
        hours_text = f"{hours_val:.2f} hr"
        try:
            rate_val = float(rate or 0.0)
        except Exception:
            rate_val = 0.0
        if rate_val > 0:
            return f"{hours_text} @ {_m(rate_val)}/hr"
        return hours_text

    def _pct(x) -> str:
        return f"{float(x or 0.0) * 100:.1f}%"

    def _fmt_dim(val) -> str:
        if isinstance(val, (int, float)):
            return f"{float(val):.3f}".rstrip("0").rstrip(".")
        if val is None:
            return "—"
        text = str(val).strip()
        return text if text else "—"

    def _format_weight_lb_decimal(mass_g: float | None) -> str:
        grams = max(0.0, float(mass_g or 0.0))
        pounds = grams / 1000.0 * LB_PER_KG
        if pounds <= 0:
            return "0.00 lb"
        text = f"{pounds:.2f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return f"{text} lb"

    def _format_weight_lb_oz(mass_g: float | None) -> str:
        grams = max(0.0, float(mass_g or 0.0))
        if grams <= 0:
            return "0 oz"
        pounds_total = grams / 1000.0 * LB_PER_KG
        total_ounces = pounds_total * 16.0
        pounds = int(total_ounces // 16)
        ounces = total_ounces - pounds * 16
        precision = 1 if pounds > 0 or ounces >= 1.0 else 2
        ounces = round(ounces, precision)
        if ounces >= 16.0:
            pounds += 1
            ounces = 0.0
        parts: list[str] = []
        if pounds > 0:
            parts.append(f"{pounds} lb" if pounds != 1 else "1 lb")
        if ounces > 0 or pounds == 0:
            ounce_text = f"{ounces:.{precision}f}".rstrip("0").rstrip(".")
            if not ounce_text:
                ounce_text = "0"
            parts.append(f"{ounce_text} oz")
        return " ".join(parts) if parts else "0 oz"

    def _is_truthy_flag(value) -> bool:
        """Return True only for explicit truthy values.

        Material scrap credit overrides are stored as flags that may round-trip
        through JSON/CSV layers. Those conversions can turn ``False`` into the
        string "false", which would previously evaluate truthy and cause the
        scrap credit line to render even when no override was entered. Treat
        only well-known truthy strings/numbers as True; unknown or falsy inputs
        default to False so that the credit row is hidden unless a user-supplied
        override is present.
        """

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if lowered in {"", "0", "false", "f", "no", "n", "off"}:
                return False
            return False
        return False

    def _lookup_config_flag(*keys: str) -> bool:
        """Return True if any mapping contains a truthy value for ``keys``.

        Configuration toggles can be supplied via several payload containers.
        Check the common locations so callers can opt-in to optional behaviours.
        """

        potential_sources: Sequence[Mapping[str, Any] | None] = (
            result,
            breakdown,
            breakdown.get("config_flags") if isinstance(breakdown, Mapping) else None,
            result.get("config_flags") if isinstance(result, Mapping) else None,
            breakdown.get("config") if isinstance(breakdown, Mapping) else None,
            result.get("config") if isinstance(result, Mapping) else None,
            breakdown.get("flags") if isinstance(breakdown, Mapping) else None,
            result.get("flags") if isinstance(result, Mapping) else None,
            breakdown.get("ui_flags") if isinstance(breakdown, Mapping) else None,
            result.get("ui_flags") if isinstance(result, Mapping) else None,
            breakdown.get("ui_vars") if isinstance(breakdown, Mapping) else None,
            result.get("ui_vars") if isinstance(result, Mapping) else None,
            params if isinstance(params, Mapping) else None,
        )

        for source in potential_sources:
            if not isinstance(source, Mapping):
                continue
            for key in keys:
                if key in source:
                    try:
                        candidate = source.get(key)
                    except Exception:
                        candidate = None
                    if candidate is not None:
                        return _is_truthy_flag(candidate)
        return False

    def write_line(s: str, indent: str = ""):
        lines.append(f"{indent}{s}")

    def write_wrapped(text: str, indent: str = ""):
        if text is None:
            return
        txt = str(text).strip()
        if not txt:
            return
        wrapper = textwrap.TextWrapper(width=max(10, page_width - len(indent)))
        for chunk in wrapper.wrap(txt):
            write_line(chunk, indent)

    def write_detail(detail: str, indent: str = "    "):
        if not detail:
            return
        for segment in re.split(r";\s*", str(detail)):
            write_wrapped(segment, indent)

    def render_bucket_table(rows: Sequence[tuple[str, float, float, float, float]]):
        if not rows:
            return

        headers = ("Bucket", "Hours", "Labor $", "Machine $", "Total $")

        display_rows: list[tuple[str, str, str, str, str]] = []
        for bucket, hours, labor_val, machine_val, total_val in rows:
            display_rows.append(
                (
                    str(bucket),
                    f"{float(hours):.2f}",
                    _m(labor_val),
                    _m(machine_val),
                    _m(total_val),
                )
            )

        col_widths: list[int] = []
        for idx, header in enumerate(headers):
            width = len(header)
            for row_values in display_rows:
                width = max(width, len(row_values[idx]))
            col_widths.append(width)

        def _fmt(value: str, idx: int) -> str:
            if idx == 0:
                return f"{value:<{col_widths[idx]}}"
            return f"{value:>{col_widths[idx]}}"

        if lines and lines[-1] != "":
            lines.append("")

        header_line = " | ".join(_fmt(header, idx) for idx, header in enumerate(headers))
        separator_line = " | ".join("-" * width for width in col_widths)
        lines.append(header_line)
        lines.append(separator_line)
        for row_values in display_rows:
            lines.append(" | ".join(_fmt(value, idx) for idx, value in enumerate(row_values)))
        lines.append("")

    def _is_total_label(label: str) -> bool:
        clean = str(label or "").strip()
        if not clean:
            return False
        clean = clean.rstrip(":")
        clean = clean.lstrip("= ")
        return clean.lower().startswith("total")

    def _ensure_total_separator(width: int) -> None:
        if not lines:
            return
        width = max(0, int(width))
        if width <= 0:
            return
        if lines[-1] == divider:
            return
        pad = max(0, page_width - width)
        short_divider = " " * pad + "-" * width
        if lines[-1] == short_divider:
            return
        lines.append(short_divider)

    def _format_row(label: str, val: float, indent: str = "") -> str:
        left = f"{indent}{label}"
        right = _m(val)
        pad = max(1, page_width - len(left) - len(right))
        return f"{left}{' ' * pad}{right}"

    def row(label: str, val: float, indent: str = ""):
        # left-label, right-amount aligned to page_width
        if _is_total_label(label):
            _ensure_total_separator(len(_m(val)))
        lines.append(_format_row(label, val, indent))

    def hours_row(label: str, val: float, indent: str = ""):
        left = f"{indent}{label}"
        right = _h(val)
        if _is_total_label(label):
            _ensure_total_separator(len(right))
        pad = max(1, page_width - len(left) - len(right))
        lines.append(f"{left}{' ' * pad}{right}")

    def _is_extra_segment(segment: str) -> bool:
        try:
            return bool(EXTRA_DETAIL_RE.match(str(segment)))
        except Exception:
            return False

    def _merge_detail(existing: str | None, new_bits: list[str]) -> str | None:
        segments: list[str] = []
        seen: set[str] = set()
        for bit in new_bits:
            seg = str(bit).strip()
            if not seg or _is_extra_segment(seg):
                continue
            if seg not in seen:
                segments.append(seg)
                seen.add(seg)
        if existing:
            for segment in re.split(r";\s*", str(existing)):
                seg = segment.strip()
                if not seg or _is_extra_segment(seg):
                    continue
                if seg not in seen:
                    segments.append(seg)
                    seen.add(seg)
        if segments:
            return "; ".join(segments)
        if existing:
            filtered_existing: list[str] = []
            for segment in re.split(r";\s*", str(existing)):
                seg = segment.strip()
                if not seg or _is_extra_segment(seg):
                    continue
                filtered_existing.append(seg)
            if filtered_existing:
                return "; ".join(filtered_existing)
            return str(existing).strip()
        return None

    def add_process_notes(key: str, indent: str = "    "):
        meta = _lookup_process_meta(key) or {}
        try:
            hr_val = float(meta.get("hr", 0.0) or 0.0)
        except Exception:
            hr_val = 0.0
        if hr_val <= 0:
            try:
                minutes_val = float(meta.get("minutes", 0.0) or 0.0)
            except Exception:
                minutes_val = 0.0
            if minutes_val > 0:
                hr_val = minutes_val / 60.0
        rate_val = meta.get("rate")
        try:
            rate_float = float(rate_val or 0.0)
        except Exception:
            rate_float = 0.0
        if rate_float <= 0:
            rate_key = _rate_key_for_bucket(str(key))
            if rate_key:
                try:
                    rate_float = float(rates.get(rate_key, 0.0) or 0.0)
                except Exception:
                    rate_float = 0.0
        if hr_val > 0:
            write_line(_hours_with_rate_text(hr_val, rate_float), indent)

    def add_pass_basis(key: str, indent: str = "    "):
        basis_map = breakdown.get("pass_basis", {}) or {}
        info = basis_map.get(key) or {}
        txt = info.get("basis") or info.get("note")
        if txt:
            write_line(str(txt), indent)

    # ---- header --------------------------------------------------------------
    lines: list[str] = []
    hour_summary_entries: dict[str, tuple[float, bool]] = {}
    ui_vars = result.get("ui_vars") or {}
    if not isinstance(ui_vars, dict):
        ui_vars = {}
    g = result.get("geo") or {}
    if not isinstance(g, dict):
        g = {}
    drill_debug_entries: list[str] = []
    for source in (result, breakdown):
        if not isinstance(source, Mapping):
            continue
        entries = source.get("drill_debug")
        if isinstance(entries, (list, tuple, set)):
            for entry in entries:
                text = str(entry).strip()
                if text and text not in drill_debug_entries:
                    drill_debug_entries.append(text)
    # Canonical QUOTE SUMMARY header (legacy variants removed in favour of this
    # block so the Speeds/Feeds status + Drill Debug output stay consistent).
    lines.append(f"QUOTE SUMMARY - Qty {qty}")
    lines.append(divider)
    speeds_feeds_display = (
        result.get("speeds_feeds_path")
        or breakdown.get("speeds_feeds_path")
    )
    path_text = str(speeds_feeds_display).strip() if speeds_feeds_display else ""
    speeds_feeds_loaded_display: bool | None = None
    for source in (result, breakdown):
        if not isinstance(source, Mapping):
            continue
        if "speeds_feeds_loaded" in source:
            raw_flag = source.get("speeds_feeds_loaded")
            speeds_feeds_loaded_display = None if raw_flag is None else bool(raw_flag)
            break
    if speeds_feeds_loaded_display is True:
        status_suffix = " (loaded)"
    elif speeds_feeds_loaded_display is False:
        status_suffix = " (not loaded)"
    else:
        status_suffix = ""
    if path_text:
        write_wrapped(f"Speeds/Feeds CSV: {path_text}{status_suffix}")
    elif status_suffix:
        write_wrapped(f"Speeds/Feeds CSV: (not set){status_suffix}")
    else:
        write_line("Speeds/Feeds CSV: (not set)")
    lines.append("")
    def _drill_debug_enabled() -> bool:
        """Return True when drill debug output should be rendered."""

        def _coerce_bool(value: object) -> bool | None:
            try:
                return bool(value)
            except Exception:
                return None

        for source in (result, breakdown):
            if not isinstance(source, Mapping):
                continue
            for key in ("app", "app_state", "app_meta"):
                candidate = source.get(key)
                if isinstance(candidate, Mapping) and "llm_debug_enabled" in candidate:
                    coerced = _coerce_bool(candidate.get("llm_debug_enabled"))
                    if coerced is not None:
                        return coerced
        return bool(APP_ENV.llm_debug_enabled)

    if drill_debug_entries and _drill_debug_enabled():
        lines.append("Drill Debug")
        lines.append(divider)
        for entry in drill_debug_entries:
            write_wrapped(entry, "  ")
        lines.append("")
    row("Final Price per Part:", price)
    total_labor_label = "Total Labor Cost:"
    row(total_labor_label, float(totals.get("labor_cost", 0.0)))
    total_labor_row_index = len(lines) - 1
    row("Total Direct Costs:", float(totals.get("direct_costs", 0.0)))
    pricing_source_value = breakdown.get("pricing_source")
    pricing_source_text = str(
        result.get("pricing_source_text")
        or breakdown.get("pricing_source_text")
        or ""
    )
    # If planner produced hours, treat source as planner for display consistency.
    if not pricing_source_value:
        try:
            hs_entries = dict(hour_summary_entries or {})
        except UnboundLocalError:
            hs_entries = {}

        def _has_positive_planner_hours(value: object) -> bool:
            base_value: object
            if isinstance(value, (list, tuple)) and value:
                base_value = value[0]
            else:
                base_value = value
            if isinstance(base_value, (int, float)):
                return float(base_value) > 0.0
            if isinstance(base_value, str):
                text = base_value.strip()
                if not text:
                    return False
                try:
                    return float(text) > 0.0
                except Exception:
                    return False
            return False

        if any(
            str(label).lower().startswith("planner")
            and _has_positive_planner_hours(value)
            for label, value in hs_entries.items()
        ):
            pricing_source_value = "planner"
    if pricing_source_value:
        lines.append(f"Pricing Source: {pricing_source_value}")
    pricing_source_lower = (
        str(pricing_source_value).strip().lower()
        if pricing_source_value is not None
        else ""
    )
    pricing_source_text: str | None = None
    if pricing_source_text:
        lines.append(f"Pricing Source: {pricing_source_text}")
        pricing_source_lower = pricing_source_text.lower()
    if red_flags:
        lines.append("")
        lines.append("Red Flags")
        lines.append(divider)
        for flag in red_flags:
            write_wrapped(f"⚠️ {flag}", "  ")
    lines.append("")

    narrative = result.get("narrative") or breakdown.get("narrative")
    why_parts: list[str] = []
    if narrative:
        if isinstance(narrative, str):
            parts = [seg.strip() for seg in re.split(r"(?<=\.)\s+", narrative) if seg.strip()]
            if not parts:
                parts = [narrative.strip()]
        else:
            parts = [str(line).strip() for line in narrative if str(line).strip()]
        why_parts.extend(parts)
    if llm_explanation:
        if isinstance(llm_explanation, str):
            text = llm_explanation.strip()
            if text:
                why_parts.append(text)
        else:
            why_parts.extend([str(item).strip() for item in llm_explanation if str(item).strip()])

    hour_trace_data = None
    if isinstance(result, Mapping):
        hour_trace_data = result.get("hour_trace")
    if hour_trace_data is None and isinstance(breakdown, Mapping):
        hour_trace_data = breakdown.get("hour_trace")
    explanation_lines: list[str] = []
    try:
        explanation_text = explain_quote(breakdown, hour_trace=hour_trace_data)
    except Exception:
        explanation_text = ""
    if explanation_text:
        for line in str(explanation_text).splitlines():
            text = line.strip()
            if text:
                explanation_lines.append(text)
    if explanation_lines:
        why_parts = explanation_lines + why_parts

    def _is_planner_meta(key: str) -> bool:
        canonical_key = _canonical_bucket_key(key)
        if not canonical_key:
            return False
        return canonical_key.startswith("planner_") or canonical_key == "planner_total"

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return default

    def _merge_process_meta(
        existing: Mapping[str, Any] | None, incoming: Mapping[str, Any] | Any
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
        incoming_map: Mapping[str, Any]
        if isinstance(incoming, Mapping):
            incoming_map = incoming
        else:
            incoming_map = {}

        if not incoming_map:
            return merged

        existing_hr = _safe_float(merged.get("hr"))
        existing_minutes = _safe_float(merged.get("minutes"))
        existing_extra = _safe_float(merged.get("base_extra"))
        existing_cost = _safe_float(merged.get("cost"))
        existing_rate = _safe_float(merged.get("rate"))

        incoming_minutes = _safe_float(incoming_map.get("minutes"))
        incoming_hr = _safe_float(incoming_map.get("hr"))
        if incoming_hr <= 0 and incoming_minutes > 0:
            incoming_hr = incoming_minutes / 60.0

        incoming_extra = _safe_float(incoming_map.get("base_extra"))
        incoming_cost = _safe_float(incoming_map.get("cost"))
        incoming_rate = _safe_float(incoming_map.get("rate"))

        if incoming_cost <= 0 and incoming_rate > 0 and incoming_hr > 0:
            incoming_cost = incoming_rate * incoming_hr

        total_minutes = existing_minutes + incoming_minutes
        if total_minutes > 0:
            merged["minutes"] = total_minutes
        elif "minutes" in merged:
            merged.pop("minutes", None)

        total_hr = existing_hr + incoming_hr
        if total_hr > 0:
            merged["hr"] = total_hr
        elif "hr" in merged:
            merged.pop("hr", None)

        total_extra = existing_extra + incoming_extra
        if abs(total_extra) > 1e-9:
            merged["base_extra"] = total_extra
        elif "base_extra" in merged:
            merged.pop("base_extra", None)

        if existing_cost <= 0 and existing_rate > 0 and existing_hr > 0:
            existing_cost = existing_rate * existing_hr
        total_cost = existing_cost + incoming_cost
        if total_cost > 0:
            merged["cost"] = total_cost
        elif "cost" in merged:
            merged.pop("cost", None)

        if total_hr > 0:
            if total_cost > 0:
                merged["rate"] = total_cost / total_hr
            elif incoming_rate > 0:
                merged["rate"] = incoming_rate
            elif existing_rate > 0:
                merged["rate"] = existing_rate
            else:
                merged.pop("rate", None)
        elif incoming_rate > 0:
            merged["rate"] = incoming_rate

        def _collect_notes(value: Any, dest: list[str], seen: set[str]) -> None:
            if isinstance(value, str):
                text = value.strip()
                if text and text not in seen:
                    dest.append(text)
                    seen.add(text)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    text = str(item).strip()
                    if text and text not in seen:
                        dest.append(text)
                        seen.add(text)

        notes: list[str] = []
        seen_notes: set[str] = set()
        _collect_notes(merged.get("notes"), notes, seen_notes)
        _collect_notes(incoming_map.get("notes"), notes, seen_notes)
        if notes:
            merged["notes"] = notes
        elif "notes" in merged:
            merged.pop("notes", None)

        special_keys = {"hr", "minutes", "base_extra", "cost", "rate", "notes"}
        for key, value in incoming_map.items():
            if key in special_keys:
                continue
            merged[key] = value

        return merged

    def _fold_process_meta(
        meta_source: Mapping[str, Any] | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        folded: dict[str, dict[str, Any]] = {}
        alias_map: dict[str, str] = {}
        if not isinstance(meta_source, Mapping):
            return {}, {}

        for raw_key, raw_meta in meta_source.items():
            alias_key = str(raw_key).lower().strip()
            if not alias_key:
                continue
            if _is_planner_meta(alias_key):
                folded[alias_key] = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
                continue

            canon_key = _canonical_bucket_key(raw_key) or alias_key
            alias_map.setdefault(alias_key, canon_key)
            existing = folded.get(canon_key)
            folded[canon_key] = _merge_process_meta(existing, raw_meta)

        result: dict[str, dict[str, Any]] = {key: value for key, value in folded.items()}
        for alias_key, canon_key in alias_map.items():
            result[alias_key] = result.get(canon_key, {})

        return result, alias_map

    def _merge_applied_process_entries(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        notes: list[str] = []
        seen_notes: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            value_notes = entry.get("notes")
            if isinstance(value_notes, str):
                text = value_notes.strip()
                if text and text not in seen_notes:
                    notes.append(text)
                    seen_notes.add(text)
            elif isinstance(value_notes, (list, tuple, set)):
                for item in value_notes:
                    text = str(item).strip()
                    if text and text not in seen_notes:
                        notes.append(text)
                        seen_notes.add(text)
            for key, value in entry.items():
                if key == "notes":
                    continue
                merged.setdefault(key, value)
        if notes:
            merged["notes"] = notes
        return merged

    def _fold_applied_process(
        applied_source: Mapping[str, Any] | None, alias_map: Mapping[str, str]
    ) -> dict[str, Any]:
        base: dict[str, Any] = {}
        if isinstance(applied_source, Mapping):
            base = {str(k).lower().strip(): (v or {}) for k, v in applied_source.items()}
        if not alias_map:
            return base

        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for alias_key, canon_key in alias_map.items():
            entry = base.get(alias_key)
            if isinstance(entry, Mapping):
                grouped.setdefault(canon_key, []).append(entry)

        for canon_key, entries in grouped.items():
            merged_entry = _merge_applied_process_entries(entries)
            base[canon_key] = merged_entry
            for alias_key, alias_canon in alias_map.items():
                if alias_canon == canon_key:
                    base[alias_key] = merged_entry

        return base

    process_meta, bucket_alias_map = _fold_process_meta(process_meta_raw)
    applied_process = _fold_applied_process(applied_process_raw, bucket_alias_map)

    def _planner_bucket_for_op(name: str | None) -> str:
        text = str(name or "").lower()
        if not text:
            return "milling"
        if any(token in text for token in ("c'bore", "counterbore")):
            return "counterbore"
        if any(token in text for token in ("csk", "countersink")):
            return "countersink"
        if any(
            token in text
            for token in (
                "tap",
                "thread mill",
                "thread_mill",
                "rigid tap",
                "rigid_tap",
            )
        ):
            return "tapping"
        if any(token in text for token in ("drill", "ream", "bore")):
            return "drilling"
        if any(
            token in text
            for token in (
                "grind",
                "od grind",
                "id grind",
                "surface grind",
                "jig grind",
            )
        ):
            return "grinding"
        if "wire" in text or "wedm" in text:
            return "wire_edm"
        if "edm" in text:
            return "sinker_edm"
        if any(token in text for token in ("saw", "waterjet")):
            return "saw_waterjet"
        if any(token in text for token in ("deburr", "finish")):
            return "finishing_deburr"
        if "inspect" in text or "cmm" in text or "fai" in text:
            return "inspection"
        return "milling"

    def _bucket_cost(info: Mapping[str, Any] | None, *keys: str) -> float:
        if not isinstance(info, Mapping):
            return 0.0
        for key in keys:
            if key in info:
                try:
                    return float(info.get(key) or 0.0)
                except Exception:
                    continue
        return 0.0

    process_plan_breakdown_raw = breakdown.get("process_plan")
    process_plan_breakdown: Mapping[str, Any] | None
    if isinstance(process_plan_breakdown_raw, Mapping):
        process_plan_breakdown = process_plan_breakdown_raw
    else:
        process_plan_breakdown = None

    planner_bucket_from_plan = _extract_bucket_map(
        process_plan_breakdown.get("bucket_view") if isinstance(process_plan_breakdown, Mapping) else None
    )

    use_planner_bucket_display = bool(planner_bucket_from_plan) and pricing_source_lower == "planner"
    if use_planner_bucket_display and set(planner_bucket_from_plan.keys()) == {"misc"}:
        use_planner_bucket_display = False
    planner_bucket_display_map: dict[str, dict[str, Any]] = (
        dict(planner_bucket_from_plan) if use_planner_bucket_display else {}
    )

    if use_planner_bucket_display:
        existing_canon_keys = {_canonical_bucket_key(key) for key in list(process_costs.keys())}
        bucket_canon_keys = set(planner_bucket_from_plan.keys())
        has_only_machine_labor = existing_canon_keys and existing_canon_keys <= {"machine", "labor"}
        replace_machine_labor = has_only_machine_labor or not bucket_canon_keys.issubset(existing_canon_keys)
        if has_only_machine_labor and bucket_canon_keys == {"misc"}:
            replace_machine_labor = False
        if replace_machine_labor:
            for key in list(process_costs.keys()):
                if _canonical_bucket_key(key) in {"machine", "labor"}:
                    process_costs.pop(key, None)
            for canon_key, info in planner_bucket_from_plan.items():
                total_cost = 0.0
                for key_option in ("total_cost", "total$", "total"):
                    if key_option in info:
                        try:
                            total_cost = float(info.get(key_option) or 0.0)
                        except Exception:
                            continue
                        if total_cost:
                            break
                if total_cost <= 0:
                    try:
                        total_cost = float(process_costs.get(canon_key, 0.0) or 0.0)
                    except Exception:
                        total_cost = 0.0
                process_costs[canon_key] = float(total_cost)
                existing_meta = process_meta.get(canon_key) if isinstance(process_meta, dict) else None
                meta_update = dict(existing_meta) if isinstance(existing_meta, Mapping) else {}
                try:
                    minutes_val = float(info.get("minutes", meta_update.get("minutes", 0.0)) or 0.0)
                except Exception:
                    minutes_val = float(meta_update.get("minutes", 0.0) or 0.0)
                if minutes_val > 0:
                    meta_update["minutes"] = round(minutes_val, 1)
                    meta_update["hr"] = round(minutes_val / 60.0, 3)
                elif "hr" not in meta_update:
                    try:
                        meta_update["hr"] = float(meta_update.get("hr", 0.0) or 0.0)
                    except Exception:
                        meta_update["hr"] = 0.0
                try:
                    hr_for_rate = float(meta_update.get("hr", 0.0) or 0.0)
                except Exception:
                    hr_for_rate = 0.0
                if hr_for_rate > 0 and total_cost > 0:
                    meta_update["rate"] = round(total_cost / hr_for_rate, 2)
                else:
                    try:
                        meta_update["rate"] = float(meta_update.get("rate", 0.0) or 0.0)
                    except Exception:
                        meta_update["rate"] = 0.0
                meta_update["cost"] = round(total_cost, 2)
                process_meta[canon_key] = meta_update

    bucket_rollup_map: dict[str, dict[str, Any]] = {}
    raw_rollup = breakdown.get("planner_bucket_rollup")
    if isinstance(raw_rollup, Mapping):
        for key, value in raw_rollup.items():
            canon = _canonical_bucket_key(key)
            if canon:
                if isinstance(value, Mapping):
                    bucket_rollup_map[canon] = {str(k): v for k, v in value.items()}
                else:
                    bucket_rollup_map[canon] = {}
    if not bucket_rollup_map and planner_bucket_from_plan:
        bucket_rollup_map = dict(planner_bucket_from_plan)
    if not bucket_rollup_map:
        bucket_struct = breakdown.get("bucket_view")
        if isinstance(bucket_struct, Mapping):
            if isinstance(bucket_struct.get("buckets"), Mapping):
                bucket_struct = bucket_struct.get("buckets")
            if isinstance(bucket_struct, Mapping):
                for key, value in bucket_struct.items():
                    canon = _canonical_bucket_key(key)
                    if canon:
                        if isinstance(value, Mapping):
                            bucket_rollup_map[canon] = {str(k): v for k, v in value.items()}
                        else:
                            bucket_rollup_map[canon] = {}

    planner_total_meta = process_meta.get("planner_total", {}) if isinstance(process_meta, dict) else {}
    planner_line_items_meta = planner_total_meta.get("line_items") if isinstance(planner_total_meta, Mapping) else None
    bucket_ops_map: dict[str, list[PlannerBucketOp]] = {}
    if isinstance(planner_line_items_meta, list):
        for entry in planner_line_items_meta:
            if not isinstance(entry, Mapping):
                continue
            op_name = entry.get("op")
            bucket_key = _planner_bucket_for_op(op_name)
            minutes_val = _bucket_cost(entry, "minutes")
            machine_val = _bucket_cost(entry, "machine_cost", "machine$")
            labor_val = _bucket_cost(entry, "labor_cost", "labor$")
            total_val = machine_val + labor_val
            bucket_ops = bucket_ops_map.setdefault(bucket_key, [])
            bucket_ops.append(
                {
                    "op": str(op_name or "").strip(),
                    "minutes": minutes_val,
                    "machine": machine_val,
                    "labor": labor_val,
                    "total": total_val,
                }
            )

    if bucket_ops_map:
        for ops in bucket_ops_map.values():
            ops.sort(key=lambda item: (-item.get("minutes", 0.0), item.get("op", "")))

    process_costs_canon = canonicalize_costs(process_costs)

    label_overrides = {
        "finishing_deburr": "Finishing/Deburr",
        "saw_waterjet": "Saw/Waterjet",
    }

    def _lookup_process_meta(key: str | None) -> Mapping[str, Any] | None:
        if not isinstance(process_meta, dict):
            return None
        candidates: list[str] = []
        base = str(key or "").lower()
        if base:
            candidates.append(base)
        canon = _canonical_bucket_key(key)
        if canon and canon not in candidates:
            candidates.append(canon)
        variants: list[str] = []
        for candidate in list(candidates):
            if "_" in candidate:
                variants.append(candidate.replace("_", " "))
            if " " in candidate:
                variants.append(candidate.replace(" ", "_"))
        seen: set[str] = set()
        for candidate in candidates + variants:
            candidate_key = candidate.strip()
            if not candidate_key or candidate_key in seen:
                continue
            seen.add(candidate_key)
            meta_entry = process_meta.get(candidate_key)
            if isinstance(meta_entry, Mapping):
                return meta_entry
        return None

    def _display_bucket_label(canon_key: str, overrides: Mapping[str, str] | None = None) -> str:
        try:
            _ovr = overrides if isinstance(overrides, Mapping) else label_overrides
        except Exception:
            _ovr = label_overrides
        if isinstance(_ovr, Mapping) and canon_key in _ovr:
            return str(_ovr[canon_key])
        return _process_label(canon_key)

    def _format_planner_bucket_line(
        canon_key: str,
        amount: float,
        meta: Mapping[str, Any] | None,
    ) -> tuple[str | None, float, float, float]:
        if not planner_bucket_display_map:
            return (None, amount, 0.0, 0.0)
        info = planner_bucket_display_map.get(canon_key)
        if not isinstance(info, Mapping):
            return (None, amount, 0.0, 0.0)
        try:
            minutes_val = float(info.get("minutes", 0.0) or 0.0)
        except Exception:
            minutes_val = 0.0
        hr_val = 0.0
        if isinstance(meta, Mapping):
            try:
                hr_val = float(meta.get("hr", 0.0) or 0.0)
            except Exception:
                hr_val = 0.0
        if hr_val <= 0 and minutes_val > 0:
            hr_val = minutes_val / 60.0
        total_cost = amount
        for key_option in ("total_cost", "total$", "total"):
            if key_option in info:
                try:
                    candidate = float(info.get(key_option) or 0.0)
                except Exception:
                    continue
                if candidate:
                    total_cost = candidate
                    break
        rate_val = 0.0
        if isinstance(meta, Mapping):
            try:
                rate_val = float(meta.get("rate", 0.0) or 0.0)
            except Exception:
                rate_val = 0.0
        if rate_val <= 0 and hr_val > 0 and total_cost > 0:
            rate_val = total_cost / hr_val
        hours_text = f"{max(hr_val, 0.0):.2f} hr"
        if hr_val <= 0:
            hours_text = "0.00 hr"
        if rate_val > 0:
            rate_text = f"{_m(rate_val)}/hr"
        else:
            rate_text = "—"
        display_override = f"{_display_bucket_label(canon_key)}: {hours_text} × {rate_text} →"
        return (display_override, float(total_cost), hr_val, rate_val)

    bucket_order = [
        "milling",
        "drilling",
        "counterbore",
        "countersink",
        "tapping",
        "grinding",
        "finishing_deburr",
        "saw_waterjet",
        "inspection",
        "assembly",
        "packaging",
        "misc",
    ]
    def _planner_bucket_info(bucket_key: str) -> Mapping[str, Any]:
        rollup_info = bucket_rollup_map.get(bucket_key)
        display_info = (
            planner_bucket_display_map.get(bucket_key)
            if isinstance(planner_bucket_display_map, Mapping)
            else None
        )
        if isinstance(display_info, Mapping):
            merged: dict[str, Any] = {}
            if isinstance(rollup_info, Mapping):
                merged.update(rollup_info)
            merged.update(display_info)
            if isinstance(rollup_info, Mapping):
                for extra_key in ("machine_cost", "machine$", "labor_cost", "labor$"):
                    if extra_key not in merged and extra_key in rollup_info:
                        merged[extra_key] = rollup_info[extra_key]
            return merged
        if isinstance(rollup_info, Mapping):
            return rollup_info
        return {}

    bucket_keys = []
    seen_buckets: set[str] = set()
    for key in bucket_order:
        if key in bucket_rollup_map or key in bucket_ops_map:
            bucket_keys.append(key)
            seen_buckets.add(key)
    extra_source = set(bucket_rollup_map.keys()) | set(bucket_ops_map.keys())
    extra_keys = sorted(key for key in extra_source if key not in seen_buckets)
    bucket_keys.extend(extra_keys)

    planner_why_lines: list[str] = []
    if bucket_keys:
        planner_why_lines.append("Planner operations (hours & cost):")
        for bucket_key in bucket_keys:
            info = _planner_bucket_info(bucket_key)
            minutes_val = _bucket_cost(info, "minutes")
            plan_hr = minutes_val / 60.0 if minutes_val else 0.0
            machine_val = _bucket_cost(info, "machine_cost", "machine$")
            labor_val = _bucket_cost(info, "labor_cost", "labor$")
            plan_total = _bucket_cost(info, "total_cost", "total$", "total")
            if plan_total == 0.0:
                plan_total = machine_val + labor_val
            final_meta = _lookup_process_meta(bucket_key) or {}
            try:
                final_hr = float(final_meta.get("hr", 0.0) or 0.0)
            except Exception:
                final_hr = 0.0
            final_cost = process_costs_canon.get(bucket_key, plan_total)
            label = _display_bucket_label(bucket_key, label_overrides)
            summary_bits: list[str] = []
            if plan_hr > 0:
                summary_bits.append(f"{plan_hr:.2f} hr plan")
            if machine_val > 0 or labor_val > 0:
                cost_parts: list[str] = []
                if machine_val > 0:
                    cost_parts.append(f"{_m(machine_val)} machine")
                if labor_val > 0:
                    cost_parts.append(f"{_m(labor_val)} labor")
                if cost_parts:
                    summary_bits.append(" + ".join(cost_parts))
            if not summary_bits:
                summary_bits.append("0.00 hr plan")

            line = f"{label}: {'; '.join(summary_bits)}"
            delta_line = ""
            if plan_hr > 1e-6:
                ratio = final_hr / plan_hr if plan_hr else 0.0
                if math.isfinite(ratio) and ratio <= 100 and abs(final_hr - plan_hr) >= 0.01:
                    delta_line = f" → overrides {final_hr:.2f} hr ({_m(final_cost)})"
            elif final_hr > 0:
                delta_line = ""
            elif abs(final_cost - plan_total) >= 0.01:
                delta_line = f" → overrides {_m(final_cost)}"
            if not delta_line and abs(final_cost - plan_total) >= 0.01 and plan_hr > 1e-6:
                delta_line = f" → overrides {_m(final_cost)}"
            if delta_line:
                line += delta_line
            planner_why_lines.append(line)

            op_entries = bucket_ops_map.get(bucket_key, [])
            for op_entry in op_entries:
                minutes = op_entry.get("minutes", 0.0)
                if minutes <= 0 and op_entry.get("total", 0.0) <= 0:
                    continue
                op_label = op_entry.get("op", "") or "Operation"
                canonical_label = _canonical_bucket_key(str(op_label or ""))
                friendly = _process_label(canonical_label)
                raw_label = str(op_label or "").strip()
                if raw_label:
                    friendly = raw_label.replace("_", " ").strip().title()
                op_line = f"  - {friendly}: {minutes:.1f} min"
                detail_bits: list[str] = []
                machine_cost = op_entry.get("machine", 0.0)
                labor_cost = op_entry.get("labor", 0.0)
                if machine_cost:
                    detail_bits.append(f"{_m(machine_cost)} machine")
                if labor_cost:
                    detail_bits.append(f"{_m(labor_cost)} labor")
                if not detail_bits and op_entry.get("total", 0.0):
                    detail_bits.append(f"{_m(op_entry.get('total', 0.0))}")
                if detail_bits:
                    op_line += " (" + ", ".join(detail_bits) + ")"
                planner_why_lines.append(op_line)

    if planner_why_lines:
        why_parts.extend(planner_why_lines)

    material_display_for_debug: str = ""

    # ---- material & stock (compact; shown only if we actually have data) -----
    if material:
        mass_g = material.get("mass_g")
        net_mass_g = material.get("mass_g_net")
        if _coerce_float_or_none(net_mass_g) is None:
            fallback_net_mass = material.get("net_mass_g")
            if _coerce_float_or_none(fallback_net_mass) is not None:
                net_mass_g = fallback_net_mass
        upg    = material.get("unit_price_per_g")
        minchg = material.get("supplier_min_charge")
        matcost= material.get("material_cost")
        scrap  = material.get("scrap_pct", None)  # will show only if present in breakdown
        scrap_credit_entered = _is_truthy_flag(
            material.get("material_scrap_credit_entered")
        )
        scrap_credit = float(material.get("material_scrap_credit") or 0.0)
        unit_price_kg = material.get("unit_price_usd_per_kg")
        unit_price_lb = material.get("unit_price_usd_per_lb")
        price_source  = material.get("unit_price_source") or material.get("source")
        price_asof    = material.get("unit_price_asof")

        have_any = any(
            v
            for v in [
                mass_g,
                net_mass_g,
                upg,
                minchg,
                matcost,
                scrap,
                scrap_credit if scrap_credit_entered else 0.0,
            ]
        )

        detail_lines: list[str] = []
        total_material_cost: float | None = None

        if have_any:
            lines.append("Material & Stock")
            lines.append(divider)
            material_name_display = ""
            if isinstance(material_selection, Mapping):
                material_name_display = (
                    material_selection.get("canonical_material")
                    or material_selection.get("material_display")
                    or material_selection.get("input_material")
                    or material_selection.get("material")
                    or ""
                )
            if not material_name_display and isinstance(drilling_meta, Mapping):
                drill_display = (
                    drilling_meta.get("material")
                    or drilling_meta.get("material_display")
                )
                if not drill_display:
                    drill_display = (
                        drilling_meta.get("material_key")
                        or drilling_meta.get("material_lookup")
                    )
                    if drill_display:
                        normalized_key = _normalize_lookup_key(drill_display)
                        drill_display = MATERIAL_DISPLAY_BY_KEY.get(
                            normalized_key,
                            drill_display,
                        )
                if drill_display:
                    material_name_display = str(drill_display)
            if not material_name_display:
                material_name_display = (
                    material.get("material_name")
                    or material.get("material")
                    or g.get("material")
                    or ui_vars.get("Material")
                    or ""
                )
            if isinstance(material_name_display, str):
                material_name_display = material_name_display.strip()
            else:
                material_name_display = str(material_name_display).strip()
            if material_name_display:
                material_display_for_debug = material_name_display
                lines.append(f"  Material used:  {material_name_display}")

            if matcost or show_zeros:
                total_material_cost = float(matcost or 0.0)
                material_net_cost = total_material_cost
            scrap_credit_lines: list[str] = []
            if scrap_credit_entered and scrap_credit:
                credit_display = _m(scrap_credit)
                if credit_display.startswith(currency):
                    credit_display = f"-{credit_display}"
                else:
                    credit_display = f"-{currency}{float(scrap_credit):,.2f}"
                scrap_credit_lines.append(f"  Scrap Credit: {credit_display}")
                scrap_credit_mass_lb = _coerce_float_or_none(
                    material.get("scrap_credit_mass_lb")
                )
                scrap_credit_unit_price_lb = _coerce_float_or_none(
                    material.get("scrap_credit_unit_price_usd_per_lb")
                )
                if scrap_credit_mass_lb and scrap_credit_unit_price_lb:
                    scrap_credit_mass_g = (
                        float(scrap_credit_mass_lb) / LB_PER_KG * 1000.0
                    )
                    scrap_credit_lines.append(
                        "    based on "
                        f"{_format_weight_lb_oz(scrap_credit_mass_g)} × {currency}{float(scrap_credit_unit_price_lb):,.2f} / lb"
                    )
            net_mass_val = _coerce_float_or_none(net_mass_g)
            effective_mass_val = _coerce_float_or_none(mass_g)
            removal_mass_val = None
            for removal_key in ("material_removed_mass_g", "material_removed_mass_g_est"):
                removal_mass_val = _coerce_float_or_none(material.get(removal_key))
                if removal_mass_val:
                    break
            scrap_fraction_val = normalize_scrap_pct(scrap)
            if scrap_fraction_val <= 0:
                scrap_fraction_val = None
            base_mass_for_scrap = None
            if net_mass_val and net_mass_val > 0:
                base_mass_for_scrap = float(net_mass_val)
            elif effective_mass_val and effective_mass_val > 0:
                base_mass_for_scrap = float(effective_mass_val)
            scrap_adjusted_mass_val: float | None = None
            if base_mass_for_scrap:
                if removal_mass_val and removal_mass_val > 0:
                    scrap_adjusted_mass_val = max(0.0, base_mass_for_scrap - float(removal_mass_val))
                elif scrap_fraction_val is not None:
                    scrap_adjusted_mass_val = max(0.0, base_mass_for_scrap * (1.0 - scrap_fraction_val))
                elif (
                    effective_mass_val is not None
                    and net_mass_val is not None
                ):
                    diff_mass = abs(float(effective_mass_val) - float(net_mass_val))
                    base_candidate = max(float(effective_mass_val), float(net_mass_val))
                    scrap_adjusted_mass_val = max(0.0, base_candidate - diff_mass)
            starting_mass_val = _coerce_float_or_none(material.get("mass_g"))
            if starting_mass_val is None:
                starting_mass_val = _coerce_float_or_none(
                    material.get("effective_mass_g")
                )
            if starting_mass_val is None:
                starting_mass_val = effective_mass_val
            if (
                net_mass_val is None
                and removal_mass_val is not None
                and removal_mass_val >= 0
            ):
                base_for_net = starting_mass_val
                if base_for_net is None:
                    base_for_net = effective_mass_val
                if base_for_net is not None:
                    net_mass_val = max(0.0, float(base_for_net) - float(removal_mass_val))
            if net_mass_val is None:
                net_mass_val = effective_mass_val
            show_mass_line = (
                (net_mass_val and net_mass_val > 0)
                or (effective_mass_val and effective_mass_val > 0)
                or show_zeros
            )
            if show_mass_line:
                net_display = _format_weight_lb_decimal(net_mass_val)
                mass_desc: list[str] = [f"{net_display} net"]
                scrap_desc_mass = scrap_adjusted_mass_val
                if scrap_desc_mass is None:
                    scrap_desc_mass = effective_mass_val
                if (
                    scrap_desc_mass is not None
                    and (
                        not net_mass_val
                        or abs(float(scrap_desc_mass) - float(net_mass_val)) > 0.05
                    )
                ):
                    mass_desc.append(
                        f"scrap-adjusted {_format_weight_lb_decimal(scrap_desc_mass)}"
                    )
                elif effective_mass_val and not net_mass_val:
                    mass_desc.append(
                        f"scrap-adjusted {_format_weight_lb_decimal(effective_mass_val)}"
                    )

            scrap_mass_val: float | None = None
            if (
                starting_mass_val is not None
                and net_mass_val is not None
            ):
                scrap_mass_val = max(0.0, float(starting_mass_val) - float(net_mass_val))
            if scrap_mass_val is None and removal_mass_val is not None:
                scrap_mass_val = max(0.0, float(removal_mass_val))
            if (
                scrap_mass_val is None
                and scrap_fraction_val is not None
                and starting_mass_val is not None
            ):
                scrap_mass_val = max(
                    0.0, float(starting_mass_val) * float(scrap_fraction_val)
                )
            if (
                scrap_mass_val is None
                and scrap_adjusted_mass_val is not None
                and net_mass_val is not None
            ):
                scrap_mass_val = max(
                    0.0,
                    abs(float(scrap_adjusted_mass_val) - float(net_mass_val)),
                )

            weight_lines: list[str] = []
            if (starting_mass_val and starting_mass_val > 0) or show_zeros:
                weight_lines.append(
                    f"  Starting Weight: {_format_weight_lb_oz(starting_mass_val)}"
                )
            if (net_mass_val and net_mass_val > 0) or show_zeros:
                weight_lines.append(
                    f"  Net Weight: {_format_weight_lb_oz(net_mass_val)}"
                )
            if scrap_mass_val is not None:
                if scrap_mass_val > 0 or show_zeros:
                    weight_lines.append(
                        f"  Scrap Weight: {_format_weight_lb_oz(scrap_mass_val)}"
                    )
            elif show_zeros:
                weight_lines.append("  Scrap Weight: 0 oz")
            if scrap is not None:
                weight_lines.append(f"  Scrap Percentage: {_pct(scrap)}")
            # Historically the renderer would emit an extra weight-only line here when
            # ``scrap_adjusted_mass`` was available.  The value was the computed "with
            # scrap" mass, but because it lacked a label it rendered as a stray line like
            # ``9 lb 6.4 oz`` in the middle of the material section.  That formatting is
            # confusing and does not provide any additional context to the reader, so we
            # intentionally skip adding that line.  The net, starting, and scrap weights
            # already convey the information a customer needs.

            detail_lines.extend(weight_lines)
            if scrap_credit_lines:
                detail_lines.extend(scrap_credit_lines)

            shipping_tax_lines: list[str] = []
            base_cost_before_scrap = _coerce_float_or_none(
                material.get("material_cost_before_credit")
            )
            if base_cost_before_scrap is None:
                net_mass_for_base = _coerce_float_or_none(net_mass_val)
                if net_mass_for_base is None:
                    net_mass_for_base = _coerce_float_or_none(
                        material.get("net_mass_g")
                    )
                if net_mass_for_base is not None and net_mass_for_base > 0:
                    per_lb_value = _coerce_float_or_none(
                        material.get("unit_price_usd_per_lb")
                    )
                    if per_lb_value is None:
                        per_g_value = _coerce_float_or_none(
                            material.get("unit_price_per_g")
                        )
                        if per_g_value is not None:
                            per_lb_value = per_g_value * (1000.0 / LB_PER_KG)
                    if per_lb_value is not None:
                        base_cost_before_scrap = (
                            float(net_mass_for_base) / 1000.0 * LB_PER_KG
                        ) * float(per_lb_value)

            if base_cost_before_scrap is not None or show_zeros:
                base_val = float(base_cost_before_scrap or 0.0)
                if show_material_shipping and (shipping_total > 0 or show_zeros):
                    if shipping_source:
                        shipping_display = shipping_total
                    else:
                        shipping_display = base_val * 0.15
                    shipping_tax_lines.append(f"  Shipping: {_m(shipping_display)}")
                tax_cost = base_val * 0.065
                if tax_cost > 0 or show_zeros:
                    shipping_tax_lines.append(f"  Material Tax: {_m(tax_cost)}")

            if shipping_tax_lines:
                detail_lines.extend(shipping_tax_lines)

            if upg or unit_price_kg or unit_price_lb or show_zeros:
                grams_per_lb = 1000.0 / LB_PER_KG
                per_lb_value = _coerce_float_or_none(unit_price_lb)
                if per_lb_value is None:
                    per_kg_value = _coerce_float_or_none(unit_price_kg)
                    if per_kg_value is not None:
                        per_lb_value = per_kg_value / LB_PER_KG
                if per_lb_value is None:
                    per_g_value = _coerce_float_or_none(upg)
                    if per_g_value is not None:
                        per_lb_value = per_g_value * grams_per_lb
                if per_lb_value is None and show_zeros:
                    per_lb_value = 0.0

                if per_lb_value is not None:
                    display_line = f"{_m(per_lb_value)} / lb"
                    extras: list[str] = []
                    if price_asof:
                        extras.append(f"as of {price_asof}")
                    extra = f" ({', '.join(extras)})" if extras else ""
                    price_lines = [f"  Material Price: {display_line}{extra}"]
                else:
                    price_lines = []
            else:
                price_lines = []
            if price_source:
                price_lines.append(f"  Source: {price_source}")
            if minchg or show_zeros:
                price_lines.append(f"  Supplier Min Charge: {_m(minchg or 0)}")
            if price_lines:
                last_line = detail_lines[-1] if detail_lines else ""
                if (
                    detail_lines
                    and last_line != ""
                    and not last_line.lstrip().startswith("Scrap Credit:")
                    and not last_line.lstrip().startswith("based on ")
                ):
                    detail_lines.append("")
                detail_lines.extend(price_lines)
            stock_L = _fmt_dim(ui_vars.get("Plate Length (in)"))
            stock_W = _fmt_dim(ui_vars.get("Plate Width (in)"))
            th_in = ui_vars.get("Thickness (in)")
            if th_in in (None, ""):
                th_in = g.get("thickness_in_guess")
            if not th_in:
                th_in = 1.0
            stock_T = _fmt_dim(th_in)
            lines.append(f"  Stock used: {stock_L} × {stock_W} × {stock_T} in")
            if detail_lines:
                lines.extend(detail_lines)
            if total_material_cost is not None:
                row("Total Material Cost :", total_material_cost, indent="  ")
            lines.append("")

    # ---- NRE / Setup costs ---------------------------------------------------
    lines.append("NRE / Setup Costs (per lot)")
    lines.append(divider)
    prog = nre_detail.get("programming") or {}
    fix  = nre_detail.get("fixture") or {}

    # Programming & Eng (auto-hide if zero unless show_zeros)
    if (prog.get("per_lot", 0.0) > 0) or show_zeros or any(prog.get(k) for k in ("prog_hr", "eng_hr")):
        row("Programming & Eng:", float(prog.get("per_lot", 0.0)))
        has_detail = False
        if prog.get("prog_hr"):
            has_detail = True
            write_line(
                f"- Programmer: {_hours_with_rate_text(prog.get('prog_hr'), prog.get('prog_rate'))}",
                "    ",
            )
        if prog.get("eng_hr"):
            has_detail = True
            write_line(
                f"- Engineering: {_hours_with_rate_text(prog.get('eng_hr'), prog.get('eng_rate'))}",
                "    ",
            )
        if not has_detail:
            prog_detail = nre_cost_details.get("Programming & Eng (per lot)")
            if prog_detail not in (None, ""):
                write_detail(str(prog_detail))

    # Fixturing (with renamed subline)
    if (fix.get("per_lot", 0.0) > 0) or show_zeros or fix.get("build_hr"):
        row("Fixturing:", float(fix.get("per_lot", 0.0)))
        has_detail = False
        if fix.get("build_hr"):
            has_detail = True
            write_line(
                f"- Build Labor: {_hours_with_rate_text(fix.get('build_hr'), fix.get('build_rate'))}",
                "    ",
            )
        if not has_detail:
            fix_detail = nre_cost_details.get("Fixturing (per lot)")
            if fix_detail not in (None, ""):
                write_detail(str(fix_detail))

    # Any other NRE numeric keys (auto include)
    other_nre_total = 0.0
    for k, v in (nre or {}).items():
        if k in ("programming_per_part", "fixture_per_part"):  # these are per-part rollups
            continue
        if isinstance(v, (int, float)) and (v > 0 or show_zeros):
            row(k.replace("_", " ").title() + ":", float(v))
            other_nre_total += float(v)
    if (prog or fix or other_nre_total > 0) and not lines[-1].strip() == "":
        lines.append("")

    # ---- Process & Labor (auto include non-zeros; sorted desc) ---------------
    bucket_table_rows: list[tuple[str, float, float, float, float]] = []
    bucket_table_totals: dict[str, float] | None = None

    process_plan_summary_local = locals().get("process_plan_summary")
    if not isinstance(process_plan_summary_local, Mapping):
        process_plan_summary_local = (
            breakdown.get("process_plan") if isinstance(breakdown, Mapping) else None
        )
    bucket_view = (
        process_plan_summary_local.get("bucket_view")
        if isinstance(process_plan_summary_local, Mapping)
        else None
    )
    if isinstance(bucket_view, Mapping):
        order = bucket_view.get("order")
        buckets = bucket_view.get("buckets")
        if isinstance(order, Sequence) and isinstance(buckets, Mapping):
            bucket_table_totals = {"hours": 0.0, "labor": 0.0, "machine": 0.0, "total": 0.0}
            for bucket_key in order:
                info = buckets.get(bucket_key)
                if not isinstance(info, Mapping):
                    continue
                try:
                    minutes_val = float(info.get("minutes", 0.0) or 0.0)
                except Exception:
                    minutes_val = 0.0
                hours_val = minutes_val / 60.0
                try:
                    labor_val = float(info.get("labor$", 0.0) or 0.0)
                except Exception:
                    labor_val = 0.0
                try:
                    machine_val = float(info.get("machine$", 0.0) or 0.0)
                except Exception:
                    machine_val = 0.0
                total_val = labor_val + machine_val

                if total_val <= 0.01 and hours_val <= 0.01:
                    continue

                bucket_table_rows.append(
                    (
                        bucket_key,
                        round(hours_val, 2),
                        round(labor_val, 2),
                        round(machine_val, 2),
                        round(total_val, 2),
                    )
                )

                bucket_table_totals["hours"] += hours_val
                bucket_table_totals["labor"] += labor_val
                bucket_table_totals["machine"] += machine_val
                bucket_table_totals["total"] += total_val

    if bucket_table_rows:
        render_bucket_table(bucket_table_rows)
    else:
        bucket_table_totals = None

    lines.append("Process & Labor Costs")
    lines.append(divider)
    proc_total = 0.0

    labor_row_order: list[str] = []
    labor_row_data: dict[str, dict[str, Any]] = {}

    def _labor_storage_key(label: str, process_key: str | None) -> str:
        canonical_label, _ = _canonical_amortized_label(label)
        normalized_label = canonical_label or str(label)
        label_lower = normalized_label.lower()
        is_amortized_label = "amortized" in label_lower

        if process_key is not None:
            bucket_key = _canonical_bucket_key(process_key)
            if bucket_key:
                return bucket_key

        if not is_amortized_label:
            bucket_from_label = _canonical_bucket_key(normalized_label)
            if bucket_from_label:
                return bucket_from_label

        return normalized_label

    _HOUR_LABEL_CANON = {
        "finishing deburr": "Finishing/Deburr",
        "finishing/deburr": "Finishing/Deburr",
        "deburr": "Finishing/Deburr",
    }

    def _canonical_hour_label(label: str) -> str:
        key = str(label or "").strip().lower()
        return _HOUR_LABEL_CANON.get(key, label)

    def _add_labor_cost_line(
        label: str,
        amount: float,
        *,
        process_key: str | None = None,
        detail_bits: list[str] | None = None,
        fallback_detail: str | None = None,
        force: bool = False,
        display_override: str | None = None,
    ) -> None:
        nonlocal proc_total
        amount_val = float(amount or 0.0)
        if not force and not ((amount_val > 0) or show_zeros):
            return

        storage_key = _labor_storage_key(str(label), process_key)
        display_text = _canonical_hour_label(display_override or str(label))

        entry = labor_row_data.get(storage_key)
        if entry is None:
            entry = {
                "label": display_text,
                "amount": 0.0,
                "process_keys": [],
                "has_override": bool(display_override),
            }
            labor_row_data[storage_key] = entry
            labor_row_order.append(storage_key)
        elif display_override:
            entry["label"] = display_text
            entry["has_override"] = True
        elif not entry.get("has_override"):
            entry["label"] = display_text

        entry["amount"] += amount_val
        if process_key is not None:
            proc_key_str = str(process_key)
            if proc_key_str and proc_key_str not in entry["process_keys"]:
                entry["process_keys"].append(proc_key_str)

        existing_detail = labor_cost_details.get(storage_key)
        merged_detail = _merge_detail(existing_detail, detail_bits or [])
        if merged_detail:
            labor_cost_details[storage_key] = merged_detail
        elif fallback_detail:
            fallback_segments = [
                seg for seg in re.split(r";\s*", str(fallback_detail)) if seg.strip()
            ]
            sanitized_fallback = _merge_detail(None, fallback_segments)
            if sanitized_fallback:
                labor_cost_details.setdefault(storage_key, sanitized_fallback)

        labor_costs_display[storage_key] = labor_costs_display.get(storage_key, 0.0) + amount_val
        proc_total += amount_val

    def _emit_labor_cost_lines() -> None:
        merged_entries: dict[str, dict[str, Any]] = {}
        merged_order: list[str] = []

        for storage_key in labor_row_order:
            entry = labor_row_data[storage_key]
            canonical_label = _canonical_hour_label(entry["label"])
            merged_entry = merged_entries.get(canonical_label)
            if merged_entry is None:
                merged_entry = {
                    "label": canonical_label,
                    "amount": 0.0,
                    "detail_texts": [],
                    "process_keys_no_detail": [],
                }
                merged_entries[canonical_label] = merged_entry
                merged_order.append(canonical_label)

            merged_entry["amount"] += entry["amount"]

            detail_text = labor_cost_details.get(storage_key)
            if detail_text:
                merged_entry["detail_texts"].append(detail_text)
            else:
                for proc_key in entry["process_keys"]:
                    if proc_key not in merged_entry["process_keys_no_detail"]:
                        merged_entry["process_keys_no_detail"].append(proc_key)

        for canonical_label in merged_order:
            merged_entry = merged_entries[canonical_label]
            row(merged_entry["label"], merged_entry["amount"], indent="  ")
            for detail_text in merged_entry["detail_texts"]:
                if detail_text not in (None, ""):
                    write_detail(str(detail_text), indent="    ")
            if not merged_entry["detail_texts"] or merged_entry["process_keys_no_detail"]:
                for proc_key in merged_entry["process_keys_no_detail"]:
                    add_process_notes(proc_key, indent="    ")

    def _fold_process_costs(values: Mapping[str, Any] | None) -> dict[str, float]:
        if not isinstance(values, Mapping):
            return {}

        # Preserve the first-seen label for each canonical bucket key while
        # summing duplicate entries together.
        folded: dict[str, float] = {}
        label_map: dict[str, str] = {}
        order: list[str] = []

        for raw_key, raw_val in values.items():
            canon_key = _canonical_bucket_key(raw_key)
            try:
                numeric_val = float(raw_val or 0.0)
            except Exception:
                numeric_val = 0.0

            existing_label = label_map.get(canon_key)
            if existing_label is None:
                label_map[canon_key] = raw_key
                order.append(raw_key)
                folded[raw_key] = numeric_val
            else:
                folded[existing_label] += numeric_val

        return {label: folded[label] for label in order}

    process_costs = _fold_process_costs(process_costs)

    process_cost_items_all = list((process_costs or {}).items())
    display_process_cost_items = [
        (key, value)
        for key, value in process_cost_items_all
        if not _is_planner_meta(key)
    ]

    # Maintain a stable ordering of process cost buckets that prioritizes the
    # preferred buckets while appending any "extra" buckets after them.  The
    # `bucket_keys` list already contains the preferred order followed by any
    # additional keys sourced from the rollup/ops maps; we reuse that ordering
    # here and extend it with any further keys that appear only in the display
    # data so that everything receives a deterministic slot.
    bucket_sort_order: list[str] = list(bucket_keys) if bucket_keys else list(bucket_order)

    existing_bucket_keys = set(bucket_sort_order)
    for key, _ in display_process_cost_items:
        canon = _normalize_bucket_key(key)
        if canon.startswith("planner_"):
            continue
        if canon and canon not in existing_bucket_keys:
            bucket_sort_order.append(canon)
            existing_bucket_keys.add(canon)

    bucket_sort_rank = {canon: idx for idx, canon in enumerate(bucket_sort_order)}

    def _is_planner_rollup_key(name: str | None) -> bool:
        if pricing_source_lower != "planner":
            return False
        canon = _canonical_bucket_key(name or "")
        if not canon:
            return False
        if canon in {"planner_total", "planner_machine", "planner_labor"}:
            return True
        norm = _normalize_bucket_key(name)
        return norm.startswith("planner_")

    sortable_process_items: list[tuple[int, int, str, float]] = []
    for original_index, (key, value) in enumerate(display_process_cost_items):
        norm = _normalize_bucket_key(key)
        if norm.startswith("planner_"):
            continue
        if not ((value > 0) or show_zeros):
            continue

        sort_rank = bucket_sort_rank.get(norm)
        if sort_rank is None:
            # Fallback for any stray keys: place them after the known buckets in
            # their original order to keep the output deterministic.
            sort_rank = len(bucket_sort_rank) + original_index

        sortable_process_items.append((sort_rank, original_index, str(key), float(value or 0.0)))

    sortable_process_items.sort(key=lambda item: (item[0], item[1]))
    ordered_process_items: list[tuple[str, float]] = [
        (key, value) for _, _, key, value in sortable_process_items
    ]

    display_process_costs: dict[str, Any] = {}
    for key, value in (process_costs or {}).items():
        if _is_planner_meta(key):
            continue
        display_process_costs[key] = value

    def _append_unique(bits: list[str], value: Any) -> None:
        text = str(value).strip()
        if not text:
            return
        if text not in bits:
            bits.append(text)

    aggregated_process_rows: dict[str, dict[str, Any]] = {}
    aggregated_order: list[str] = []
    process_entries_for_display: list[ProcessDisplayEntry] = []

    for entry in _iter_ordered_process_entries(
        display_process_costs,
        process_meta=process_meta,
        applied_process=applied_process,
        show_zeros=show_zeros,
        planner_bucket_display_map=planner_bucket_display_map,
        label_overrides=label_overrides,
        currency_formatter=_m,
    ):
        process_entries_for_display.append(entry)
        canon_key = entry.canonical_key or _canonical_bucket_key(entry.process_key)
        if canon_key in {"planner_labor", "planner_machine", "planner_total"}:
            continue
        if canon_key.startswith("planner_"):
            continue
        canonical_label, _ = _canonical_amortized_label(entry.label)
        fallback_detail_raw = labor_cost_details_input.get(canonical_label or entry.label)
        fallback_detail = (
            str(fallback_detail_raw).strip()
            if fallback_detail_raw not in (None, "")
            else None
        )

        bucket = aggregated_process_rows.get(canon_key)
        if bucket is None:
            bucket = {
                "amount": 0.0,
                "detail_bits": [],
                "process_keys": [],
                "fallback_detail": fallback_detail,
                "display_override": entry.display_override,
                "representative_key": str(entry.process_key or canon_key),
            }
            aggregated_process_rows[canon_key] = bucket
            aggregated_order.append(canon_key)
        else:
            if not bucket.get("fallback_detail") and fallback_detail:
                bucket["fallback_detail"] = fallback_detail
            if entry.display_override:
                bucket["display_override"] = entry.display_override

        try:
            amount_val = float(entry.amount or 0.0)
        except Exception:
            amount_val = 0.0
        bucket["amount"] += amount_val

        proc_key_text = str(entry.process_key or "").strip()
        if proc_key_text and proc_key_text not in bucket["process_keys"]:
            bucket["process_keys"].append(proc_key_text)

        for bit in entry.detail_bits:
            _append_unique(bucket["detail_bits"], bit)

    hidden_canon_keys = {"planner_labor", "planner_machine", "planner_total"}
    remaining_costs = {canon: data["amount"] for canon, data in aggregated_process_rows.items()}

    for canon_key in aggregated_order:
        if canon_key in hidden_canon_keys or canon_key.startswith("planner_"):
            continue

        amount = remaining_costs.get(canon_key, 0.0)
        bucket = aggregated_process_rows[canon_key]
        process_keys = bucket.get("process_keys") or []
        representative_key = bucket.get("representative_key")
        if not process_keys and representative_key:
            text_key = str(representative_key).strip()
            if text_key:
                process_keys = [text_key]

        label = _display_bucket_label(canon_key, label_overrides)
        primary_process_key = process_keys[0] if process_keys else canon_key

        detail_bits = list(bucket.get("detail_bits", []))
        if canon_key == "misc":
            _append_unique(detail_bits, "LLM adjustments")

        _add_labor_cost_line(
            label,
            amount,
            process_key=primary_process_key,
            detail_bits=detail_bits,
            fallback_detail=bucket.get("fallback_detail"),
            display_override=bucket.get("display_override"),
        )

        storage_key = _labor_storage_key(label, primary_process_key)
        entry = labor_row_data.get(storage_key)
        if entry and len(process_keys) > 1:
            for extra_key in process_keys[1:]:
                if extra_key not in entry["process_keys"]:
                    entry["process_keys"].append(extra_key)


    show_amortized_single_qty = _lookup_config_flag(
        "show_amortized_nre_single_qty",
        "show_amortized_nre_for_single_qty",
        "show_single_qty_amortized_nre",
    )

    try:
        amortized_qty = int(result.get("qty") or breakdown.get("qty") or qty or 1)
    except Exception:
        amortized_qty = qty if qty > 0 else 1
    show_amortized = amortized_qty > 1 or (show_amortized_single_qty and amortized_qty > 0)

    programming_per_part_cost = labor_cost_totals.get("Programming (amortized)")
    if programming_per_part_cost is None:
        programming_per_part_cost = float(nre.get("programming_per_part", 0.0) or 0.0)
    programming_detail = (nre_detail or {}).get("programming") or {}
    prog_bits: list[str] = []
    try:
        prog_hr = float(programming_detail.get("prog_hr", 0.0) or 0.0)
    except Exception:
        prog_hr = 0.0
    try:
        prog_rate = float(programming_detail.get("prog_rate", 0.0) or 0.0)
    except Exception:
        prog_rate = 0.0
    if prog_hr > 0:
        prog_bits.append(f"- Programmer (lot): {_hours_with_rate_text(prog_hr, prog_rate)}")
    try:
        eng_hr = float(programming_detail.get("eng_hr", 0.0) or 0.0)
    except Exception:
        eng_hr = 0.0
    try:
        eng_rate = float(programming_detail.get("eng_rate", 0.0) or 0.0)
    except Exception:
        eng_rate = 0.0
    if eng_hr > 0:
        prog_bits.append(f"- Engineering (lot): {_hours_with_rate_text(eng_hr, eng_rate)}")
    programming_detail_bits = list(prog_bits)
    if show_amortized and qty > 1 and programming_per_part_cost > 0:
        programming_detail_bits.append(f"Amortized across {qty} pcs")
    if show_amortized and programming_per_part_cost > 0:
        programming_detail_bits.append("amortized")

    fixture_detail = (nre_detail or {}).get("fixture") or {}
    fixture_labor_per_part_cost = labor_cost_totals.get("Fixture Build (amortized)")
    if fixture_labor_per_part_cost is None:
        try:
            fixture_labor_total = float(fixture_detail.get("labor_cost", 0.0) or 0.0)
        except Exception:
            fixture_labor_total = 0.0
        fixture_labor_per_part_cost = (
            fixture_labor_total / qty if qty > 0 else fixture_labor_total
        )
    fixture_bits: list[str] = []
    try:
        fixture_hr = float(fixture_detail.get("build_hr", 0.0) or 0.0)
    except Exception:
        fixture_hr = 0.0
    try:
        fixture_rate = float(
            fixture_detail.get("build_rate", rates.get("FixtureBuildRate", 0.0)) or 0.0
        )
    except Exception:
        fixture_rate = 0.0
    if fixture_hr > 0:
        fixture_bits.append(
            f"- Build labor (lot): {_hours_with_rate_text(fixture_hr, fixture_rate)}"
        )
    try:
        soft_jaw_hr = float(fixture_detail.get("soft_jaw_hr", 0.0) or 0.0)
    except Exception:
        soft_jaw_hr = 0.0
    if soft_jaw_hr > 0:
        fixture_bits.append(f"Soft jaw prep {soft_jaw_hr:.2f} hr")
    fixture_detail_bits = list(fixture_bits)
    if show_amortized and qty > 1 and fixture_labor_per_part_cost > 0:
        fixture_detail_bits.append(f"Amortized across {qty} pcs")
    if show_amortized and fixture_labor_per_part_cost > 0:
        fixture_detail_bits.append("amortized")

    amortized_rows: list[tuple[str, float, list[str]]] = []
    if show_amortized and programming_per_part_cost > 0:
        amortized_rows.append(
            (
                "Programming (amortized)",
                float(programming_per_part_cost),
                programming_detail_bits,
            )
        )
    if show_amortized and fixture_labor_per_part_cost > 0:
        amortized_rows.append(
            (
                "Fixture Build (amortized)",
                float(fixture_labor_per_part_cost),
                fixture_detail_bits,
            )
        )

    for label, amount, detail_bits in amortized_rows:
        if label not in labor_costs_display:
            _add_labor_cost_line(
                label,
                amount,
                detail_bits=detail_bits,
            )

    _emit_labor_cost_lines()

    machine_in_labor_section = any(
        _canonical_bucket_key(str(key)) == "machine"
        for key in labor_costs_display.keys()
    )

    displayed_process_total = sum(float(value or 0.0) for value in labor_costs_display.values())
    if not math.isclose(proc_total, displayed_process_total, rel_tol=1e-6, abs_tol=0.05):
        try:
            logger.warning(
                "Labor section drift: accumulated=%s vs displayed=%s",
                f"{proc_total:.2f}", f"{displayed_process_total:.2f}",
            )
        except Exception:
            pass
        # Prefer what we actually render
        proc_total = displayed_process_total

    expected_labor_total = float(totals.get("labor_cost", 0.0) or 0.0)
    if abs(displayed_process_total - expected_labor_total) >= 0.01:
        raise AssertionError("bucket sum mismatch")

    proc_total = displayed_process_total
    row("Total", proc_total, indent="  ")

    hour_summary_entries.clear()

    def _record_hour_entry(label: str, value: float, *, include_in_total: bool = True) -> None:
        try:
            numeric_value = float(value or 0.0)
        except Exception:
            numeric_value = 0.0

        if not ((numeric_value > 0) or show_zeros):
            return

        single_piece_qty = False
        try:
            single_piece_qty = qty_for_hours > 0 and (qty_for_hours == 1 or math.isclose(qty_for_hours, 1.0))
        except Exception:
            single_piece_qty = False

        if single_piece_qty and numeric_value > 24.0:
            warning = (
                f"{label} hours capped at 24 hr for single-piece quote (was {numeric_value:.2f} hr)."
            )
            if warning not in red_flags:
                red_flags.append(warning)
            numeric_value = 24.0

        hour_summary_entries[label] = (numeric_value, include_in_total)

    programming_meta = (nre_detail or {}).get("programming") or {}
    try:
        programming_hours = float(programming_meta.get("prog_hr", 0.0) or 0.0)
    except Exception:
        programming_hours = 0.0

    fixture_meta = (nre_detail or {}).get("fixture") or {}
    try:
        fixture_hours = float(fixture_meta.get("build_hr", 0.0) or 0.0)
    except Exception:
        fixture_hours = 0.0

    try:
        qty_for_hours = float(qty)
    except Exception:
        qty_for_hours = 0.0
    if qty_for_hours <= 0:
        qty_for_hours = 0.0

    try:
        programming_per_part_amount = float(programming_per_part_cost or 0.0)
    except Exception:
        programming_per_part_amount = 0.0
    programming_is_amortized = bool(programming_meta.get("amortized")) or (
        qty_for_hours > 1 and programming_per_part_amount > 0
    )

    try:
        fixture_per_part_amount = float(fixture_labor_per_part_cost or 0.0)
    except Exception:
        fixture_per_part_amount = 0.0
    fixture_is_amortized = qty_for_hours > 1 and fixture_per_part_amount > 0

    if str(pricing_source_value).lower() == "planner":
        planner_total_meta = process_meta.get("planner_total") or {}
        planner_items = planner_total_meta.get("line_items") if isinstance(planner_total_meta, dict) else None
        total_minutes = 0.0
        if isinstance(planner_items, list):
            for item in planner_items:
                try:
                    total_minutes += max(0.0, float(item.get("minutes", 0.0) or 0.0))
                except Exception:
                    continue
        if total_minutes <= 0:
            try:
                total_minutes = float(planner_total_meta.get("minutes", 0.0) or 0.0)
            except Exception:
                total_minutes = 0.0
        try:
            planner_total_hr = float(planner_total_meta.get("hr", 0.0) or 0.0)
        except Exception:
            planner_total_hr = 0.0
        if total_minutes > 0:
            planner_total_hr = total_minutes / 60.0

        def _planner_hours_for(key: str) -> float:
            meta = process_meta.get(key) or {}
            try:
                minutes_val = float(meta.get("minutes", 0.0) or 0.0)
            except Exception:
                minutes_val = 0.0
            if minutes_val > 0:
                return minutes_val / 60.0
            try:
                return float(meta.get("hr", 0.0) or 0.0)
            except Exception:
                return 0.0

        planner_labor_hr = _planner_hours_for("planner_labor")
        planner_machine_hr = _planner_hours_for("planner_machine")

        def _emit_hour_row(label: str, value: float, *, include_in_total: bool = True) -> None:
            _record_hour_entry(label, value, include_in_total=include_in_total)

        def _bucket_minutes(info: Mapping[str, Any] | None) -> float:
            if not isinstance(info, Mapping):
                return 0.0
            try:
                return float(info.get("minutes", 0.0) or 0.0)
            except Exception:
                return 0.0

        def _hours_for_bucket(canon_key: str) -> float:
            info: Mapping[str, Any] | None = None
            if isinstance(planner_bucket_display_map, Mapping):
                info = planner_bucket_display_map.get(canon_key)
            if not isinstance(info, Mapping):
                info = bucket_rollup_map.get(canon_key)
            minutes_val = _bucket_minutes(info)
            if minutes_val > 0:
                return minutes_val / 60.0
            meta = _lookup_process_meta(canon_key) or {}
            try:
                hr_val = float(meta.get("hr", 0.0) or 0.0)
            except Exception:
                hr_val = 0.0
            if hr_val > 0:
                return hr_val
            try:
                minutes_meta = float(meta.get("minutes", 0.0) or 0.0)
            except Exception:
                minutes_meta = 0.0
            if minutes_meta > 0:
                return minutes_meta / 60.0
            return 0.0

        _emit_hour_row("Planner Total", round(planner_total_hr, 2))
        _emit_hour_row("Planner Labor", round(planner_labor_hr, 2), include_in_total=False)
        _emit_hour_row("Planner Machine", round(planner_machine_hr, 2), include_in_total=False)

        skip_hour_canon_keys = {
            "programming",
            "fixture_build",
            "programming_amortized",
            "fixture_build_amortized",
        }

        for canon_key in aggregated_order:
            if not canon_key or canon_key in skip_hour_canon_keys:
                continue
            if canon_key in {"planner_labor", "planner_machine", "planner_total"}:
                continue
            if canon_key.startswith("planner_"):
                continue
            hours_val = _hours_for_bucket(canon_key)
            if hours_val <= 0.01:
                continue
            label = _display_bucket_label(canon_key, label_overrides)
            _emit_hour_row(label, round(hours_val, 2))

        _emit_hour_row("Programming (lot)", round(programming_hours, 2))
        if programming_is_amortized and qty_for_hours > 0:
            per_part_prog_hr = programming_hours / qty_for_hours
            _emit_hour_row(
                "Programming (amortized per part)",
                round(per_part_prog_hr, 2),
                include_in_total=False,
            )
        _emit_hour_row("Fixture Build (lot)", round(fixture_hours, 2))
        if fixture_is_amortized and qty_for_hours > 0:
            per_part_fixture_hr = fixture_hours / qty_for_hours
            _emit_hour_row(
                "Fixture Build (amortized per part)",
                round(per_part_fixture_hr, 2),
                include_in_total=False,
            )
    else:
        for key, meta in sorted((process_meta or {}).items()):
            meta = meta or {}
            try:
                hr_val = float(meta.get("hr", 0.0) or 0.0)
            except Exception:
                hr_val = 0.0
            canon_key = _canonical_bucket_key(key)
            if canon_key:
                display_label = _display_bucket_label(canon_key)
            else:
                display_label = _process_label(key)
            _record_hour_entry(display_label, hr_val)

        _record_hour_entry("Programming", programming_hours)
        if programming_is_amortized and qty_for_hours > 0:
            per_part_prog_hr = programming_hours / qty_for_hours
            _record_hour_entry(
                "Programming (amortized per part)",
                per_part_prog_hr,
                include_in_total=False,
            )
        _record_hour_entry("Fixture Build", fixture_hours)
        if fixture_is_amortized and qty_for_hours > 0:
            per_part_fixture_hr = fixture_hours / qty_for_hours
            _record_hour_entry(
                "Fixture Build (amortized per part)",
                per_part_fixture_hr,
                include_in_total=False,
            )

    if hour_summary_entries:
        lines.append("")
        lines.append("Labor Hour Summary")
        lines.append(divider)
        if str(pricing_source_value).lower() == "planner":
            entries_iter = list(hour_summary_entries.items())
        else:
            entries_iter = list(
                sorted(hour_summary_entries.items(), key=lambda kv: kv[1][0], reverse=True)
            )
        folded_entries: dict[str, list[Any]] = {}
        folded_order: list[str] = []
        for label, (hr_val, include_in_total) in entries_iter:
            canonical_label = _canonical_hour_label(label)
            folded = folded_entries.get(canonical_label)
            if folded is None:
                folded_entries[canonical_label] = [hr_val, bool(include_in_total)]
                folded_order.append(canonical_label)
            else:
                folded[0] += hr_val
                folded[1] = folded[1] or bool(include_in_total)
        total_hours = 0.0
        for canonical_label in folded_order:
            hr_val, include_in_total = folded_entries[canonical_label]
            hours_row(canonical_label, hr_val, indent="  ")
            if include_in_total and hr_val:
                total_hours += hr_val
        hours_row("Total Hours", total_hours, indent="  ")
    lines.append("")

    # ---- Pass-Through & Direct (auto include non-zeros; sorted desc) --------
    lines.append("Pass-Through & Direct Costs")
    lines.append(divider)
    pass_total = 0.0
    pass_through_labor_total = 0.0
    for key, value in sorted((pass_through or {}).items(), key=lambda kv: kv[1], reverse=True):
        canonical_label = _canonical_pass_label(key)
        if canonical_label.lower() == "material":
            continue
        if (value > 0) or show_zeros:
            # cosmetic: "consumables_hr_cost" → "Consumables /Hr Cost"
            label = key.replace("_", " ").replace("hr", "/hr").title()
            row(label, float(value), indent="  ")
            add_pass_basis(key, indent="    ")
            detail_value = direct_cost_details.get(key)
            if detail_value not in (None, ""):
                write_detail(str(detail_value), indent="    ")
            amount_val = float(value or 0.0)
            pass_total += amount_val
            canonical_pass_label = canonical_label
            if "labor" in canonical_pass_label.lower():
                pass_through_labor_total += amount_val
    row("Total", pass_total, indent="  ")
    pass_through_total = float(pass_total)

    computed_total_labor_cost = proc_total + pass_through_labor_total
    expected_labor_total = computed_total_labor_cost
    if declared_labor_total > computed_total_labor_cost + 0.01:
        expected_labor_total = declared_labor_total
    components_total = display_labor_for_ladder + pass_through_labor_total + display_machine
    machine_gap = expected_labor_total - components_total
    if machine_gap >= 0.01:
        if not machine_in_labor_section and abs(display_machine) <= 0.01:
            display_machine = machine_gap
        else:
            raise AssertionError("bucket sum mismatch")
    elif machine_gap <= -0.01:
        raise AssertionError("bucket sum mismatch")
    if isinstance(totals, dict):
        totals["labor_cost"] = computed_total_labor_cost
    if 0 <= total_labor_row_index < len(lines):
        lines[total_labor_row_index] = _format_row(
            total_labor_label,
            computed_total_labor_cost,
        )

    computed_subtotal = proc_total + pass_total
    declared_subtotal = float(totals.get("subtotal", computed_subtotal))
    if material_net_cost is None:
        try:
            material_key = next(
                (
                    key
                    for key in (pass_through or {})
                    if str(key).strip().lower() == "material"
                ),
                None,
            )
            if material_key is not None:
                material_net_cost = float(pass_through.get(material_key) or 0.0)
            else:
                material_net_cost = 0.0
        except Exception:
            material_net_cost = 0.0
    directs = float(material_net_cost) + float(pass_through_total) + float(display_machine)
    ladder_subtotal = float(display_labor_for_ladder) + directs
    subtotal = ladder_subtotal
    printed_subtotal = subtotal
    assert roughly_equal(ladder_subtotal, printed_subtotal, eps=0.01)
    lines.append("")

    # ---- Pricing ladder ------------------------------------------------------
    lines.append("Pricing Ladder")
    lines.append(divider)
    overhead_pct    = float(applied_pcts.get("OverheadPct", 0.0) or 0.0)
    ga_pct          = float(applied_pcts.get("GA_Pct", 0.0) or 0.0)
    contingency_pct = float(applied_pcts.get("ContingencyPct", 0.0) or 0.0)
    expedite_pct    = float(applied_pcts.get("ExpeditePct", 0.0) or 0.0)

    with_overhead    = subtotal * (1.0 + overhead_pct)
    with_ga          = with_overhead * (1.0 + ga_pct)
    with_contingency = with_ga * (1.0 + contingency_pct)
    with_expedite    = with_contingency * (1.0 + expedite_pct)

    row("Subtotal (Labor + Directs):", subtotal)
    row(f"+ Overhead ({_pct(applied_pcts.get('OverheadPct'))}):",     with_overhead - subtotal)
    row(f"+ G&A ({_pct(applied_pcts.get('GA_Pct'))}):",               with_ga - with_overhead)
    row(f"+ Contingency ({_pct(applied_pcts.get('ContingencyPct'))}):", with_contingency - with_ga)
    if applied_pcts.get("ExpeditePct"):
        row(f"+ Expedite ({_pct(applied_pcts.get('ExpeditePct'))}):", with_expedite - with_contingency)
    row("= Subtotal before Margin:", with_expedite)
    row(f"Final Price with Margin ({_pct(applied_pcts.get('MarginPct'))}):", price)
    lines.append("")

    # ---- LLM adjustments bullets (optional) ---------------------------------
    if llm_notes:
        lines.append("LLM Adjustments")
        lines.append(divider)
        import textwrap as _tw
        for n in llm_notes:
            for w in _tw.wrap(str(n), width=page_width):
                lines.append(f"- {w}")
        lines.append("")

    if why_parts:
        if lines and lines[-1]:
            lines.append("")
        lines.append("Why this price")
        lines.append(divider)
        for part in why_parts:
            write_wrapped(part, "  ")
        if lines[-1]:
            lines.append("")

    return "\n".join(lines)
# ===== QUOTE CONFIG (edit-friendly) ==========================================
CONFIG_INIT_ERRORS: list[str] = []

try:
    SERVICE_CONTAINER = create_default_container()
except Exception as exc:
    CONFIG_INIT_ERRORS.append(f"Service container initialisation error: {exc}")

    def _empty_params() -> dict[str, Any]:
        return {}

    SERVICE_CONTAINER = ServiceContainer(
        load_params=_empty_params,
        load_rates=lambda: {"labor": {}, "machine": {}},
        pricing_engine_factory=lambda: PricingEngine(create_default_registry()),
    )

def _coerce_two_bucket_rates(value: Any) -> dict[str, dict[str, float]]:
    if isinstance(value, dict):
        labor_raw = value.get("labor")
        machine_raw = value.get("machine")
        if isinstance(labor_raw, dict) and isinstance(machine_raw, dict):
            labor: dict[str, float] = {}
            for key, raw in labor_raw.items():
                try:
                    labor[str(key)] = float(raw)
                except Exception:
                    continue
            machine: dict[str, float] = {}
            for key, raw in machine_raw.items():
                try:
                    machine[str(key)] = float(raw)
                except Exception:
                    continue
            return {"labor": labor, "machine": machine}

        flat: dict[str, float] = {}
        for key, raw in value.items():
            try:
                flat[str(key)] = float(raw)
            except Exception:
                continue
        if flat:
            return migrate_flat_to_two_bucket(flat)

    return {"labor": {}, "machine": {}}


try:
    _rates_raw = SERVICE_CONTAINER.load_rates()
except ConfigError as exc:
    RATES_TWO_BUCKET_DEFAULT = {"labor": {}, "machine": {}}
    CONFIG_INIT_ERRORS.append(f"Rates configuration error: {exc}")
except Exception as exc:
    RATES_TWO_BUCKET_DEFAULT = {"labor": {}, "machine": {}}
    CONFIG_INIT_ERRORS.append(f"Unexpected rates configuration error: {exc}")
else:
    RATES_TWO_BUCKET_DEFAULT = _coerce_two_bucket_rates(_rates_raw)

RATES_DEFAULT = two_bucket_to_flat(RATES_TWO_BUCKET_DEFAULT)

LABOR_RATE_KEYS: set[str] = {
    "ProgrammingRate",
    "EngineerRate",
    "InspectionRate",
    "FinishingRate",
    "FixtureBuildRate",
    "AssemblyRate",
    "ProjectManagementRate",
    "ToolmakerSupportRate",
    "PackagingRate",
    "DeburrRate",
}

MACHINE_RATE_KEYS: set[str] = {
    "MillingRate",
    "DrillingRate",
    "TurningRate",
    "WireEDMRate",
    "SinkerEDMRate",
    "SurfaceGrindRate",
    "ODIDGrindRate",
    "JigGrindRate",
    "LappingRate",
    "SawWaterjetRate",
}

try:
    PARAMS_DEFAULT = SERVICE_CONTAINER.load_params()
except ConfigError as exc:
    PARAMS_DEFAULT = {}
    CONFIG_INIT_ERRORS.append(f"Parameter configuration error: {exc}")
except Exception as exc:
    PARAMS_DEFAULT = {}
    CONFIG_INIT_ERRORS.append(f"Unexpected parameter configuration error: {exc}")

# ---- Service containers -----------------------------------------------------


@dataclass
class QuoteConfiguration:
    """Container for default parameter configuration used by the UI."""

    default_params: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(PARAMS_DEFAULT))
    default_material_display: str = DEFAULT_MATERIAL_DISPLAY

    def copy_default_params(self) -> Dict[str, Any]:
        """Return a deep copy of the default parameter set."""

        return copy.deepcopy(self.default_params)

    @property
    def default_material_key(self) -> str:
        return _normalize_lookup_key(self.default_material_display)


# Common regex pieces (kept non-capturing to avoid pandas warnings)
TIME_RE = r"\b(?:hours?|hrs?|hr|time|min(?:ute)?s?)\b"


_TOLERANCE_VALUE_RE = re.compile(
    r"(?:±|\+/-|\+-)?\s*(?P<value>(?:\d+)?\.?\d+)\s*(?P<unit>mm|millimeters?|µm|um|in|inch(?:es)?|\"|thou|thousandths)?",
    re.IGNORECASE,
)
_TIGHT_TOL_TRIGGER_RE = re.compile(r"(±\s*0\.000[12])|(tight\s*tolerance)", re.IGNORECASE)


def _tolerance_values_from_any(value: Any) -> list[float]:
    """Return tolerance magnitudes (inches) parsed from an arbitrary input value."""

    results: list[float] = []
    if value is None:
        return results
    if isinstance(value, (list, tuple, set)):
        for entry in value:
            results.extend(_tolerance_values_from_any(entry))
        return results
    if isinstance(value, (int, float)):
        try:
            num = abs(float(value))
        except Exception:
            return results
        if 0.0 < num <= 0.25:
            results.append(num)
        return results

    text = str(value or "").strip()
    if not text:
        return results

    for match in _TOLERANCE_VALUE_RE.finditer(text):
        raw_value = match.group("value")
        unit = (match.group("unit") or "").lower()
        try:
            magnitude = abs(float(raw_value))
        except Exception:
            continue
        if magnitude <= 0.0:
            continue
        if unit in {"mm", "millimeter", "millimeters"}:
            magnitude /= 25.4
        elif unit in {"µm", "um"}:
            magnitude /= 1000.0  # to mm
            magnitude /= 25.4
        elif unit in {"thou", "thousandths"}:
            magnitude /= 1000.0
        elif unit in {"in", "inch", "inches", '"'}:
            pass
        elif magnitude > 0.25:
            continue
        if magnitude <= 0.0:
            continue
        if magnitude <= 0.25:
            results.append(magnitude)
    return results


def _sum_time_from_series(
    items: pd.Series,
    values: pd.Series,
    data_types: pd.Series,
    mask: pd.Series,
    *,
    default: float = 0.0,
    exclude_mask: pd.Series | None = None,
) -> float:
    """Shared implementation for extracting hour totals from sheet rows."""

    try:
        if not mask.any():
            return float(default)
    except Exception:
        return float(default)

    looks_time = items.str.contains(TIME_RE, case=False, regex=True, na=False)
    looks_money = items.str.contains(MONEY_RE, case=False, regex=True, na=False)
    typed_money = data_types.str.contains(r"(?:rate|currency|price|cost)", case=False, na=False)

    excl = looks_money | typed_money
    if exclude_mask is not None:
        try:
            excl = excl | exclude_mask
        except Exception:
            pass

    matched = mask & ~excl & looks_time
    try:
        if not matched.any():
            return float(default)
    except Exception:
        return float(default)

    numeric_candidates = pd.to_numeric(values[matched], errors="coerce")
    mask_numeric = pd.notna(numeric_candidates)
    try:
        has_numeric = mask_numeric.any()
    except Exception:
        has_numeric = any(bool(flag) for flag in mask_numeric)
    if not has_numeric:
        return float(default)

    mins_mask = items.str.contains(r"\bmin(?:ute)?s?\b", case=False, regex=True, na=False) & matched
    hrs_mask = matched & ~mins_mask

    hrs_sum = pd.to_numeric(values[hrs_mask], errors="coerce").fillna(0.0).sum()
    mins_sum = pd.to_numeric(values[mins_mask], errors="coerce").fillna(0.0).sum()
    return float(hrs_sum) + float(mins_sum) / 60.0
MONEY_RE = r"(?:rate|/hr|per\s*hour|per\s*hr|price|cost|\$)"

# ===== QUOTE HELPERS ========================================================

def alt(*terms: str) -> str:
    """Build a non-capturing (?:a|b|c) regex; terms are already escaped/regex-ready."""
    return r"(?:%s)" % "|".join(terms)

def pct(value: Any, default: float | None = 0.0) -> float | None:
    """Accept 0-1 or 0-100 and return 0-1."""
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v / 100.0 if v > 1.0 else v


_DEFAULT_PRICING_ENGINE = SERVICE_CONTAINER.get_pricing_engine()


def compute_mass_and_scrap_after_removal(
    net_mass_g: float | None,
    scrap_frac: float | None,
    removal_mass_g: float | None,
    *,
    scrap_min: float = 0.0,
    scrap_max: float = 0.25,
) -> tuple[float, float, float]:
    """Return updated net mass, scrap fraction and effective mass after removal.

    ``net_mass_g`` and ``scrap_frac`` represent the current part state while
    ``removal_mass_g`` is the expected mass machined away from the finished
    part.  The removal is treated as additional scrap, so the effective mass
    that must be purchased increases accordingly even though the finished
    part becomes lighter.

    Scrap bounds mirror the UI constraints, clamping the resulting fraction
    into ``[scrap_min, scrap_max]``.
    """

    base_net = max(0.0, float(net_mass_g or 0.0))
    base_scrap = normalize_scrap_pct(scrap_frac)
    removal = max(0.0, float(removal_mass_g or 0.0))

    if base_net <= 0 or removal <= 0:
        effective_mass = base_net * (1.0 + base_scrap)
        return base_net, base_scrap, effective_mass

    removal = min(removal, base_net)
    net_after = max(1e-6, base_net - removal)

    scrap_min = max(0.0, float(scrap_min))
    scrap_max = max(scrap_min, float(scrap_max))

    scrap_mass = base_net * base_scrap + removal
    scrap_after = scrap_mass / net_after if net_after > 0 else scrap_max
    scrap_after = max(scrap_min, min(scrap_max, scrap_after))

    effective_mass = net_after * (1.0 + scrap_after)
    return net_after, scrap_after, effective_mass


def _material_price_from_choice(choice: str, material_lookup: dict[str, float]) -> float | None:
    """Resolve a material price per-gram for the editor helpers."""

    choice = str(choice or "").strip()
    if not choice:
        return None

    norm_choice = _normalize_lookup_key(choice)
    if norm_choice == MATERIAL_OTHER_KEY:
        return None

    price = material_lookup.get(norm_choice)
    if price is None:
        try:
            price_per_kg, _src = _resolve_material_unit_price(choice, unit="kg")
        except Exception:
            return None
        if not price_per_kg:
            return None
        price = float(price_per_kg) / 1000.0
    return float(price)


def _update_material_price_field(
    material_choice_var: Any,
    material_price_var: Any,
    material_lookup: dict[str, float],
) -> bool:
    """Update the UI material price field, returning ``True`` when changed."""

    if material_choice_var is None or material_price_var is None:
        return False

    price = _material_price_from_choice(material_choice_var.get(), material_lookup)
    if price is None:
        return False

    current_val = _coerce_float_or_none(material_price_var.get())
    if current_val is not None and abs(current_val - price) < 1e-6:
        return False

    material_price_var.set(f"{price:.4f}")
    return True


def compute_material_cost(
    material_name: str,
    mass_kg: float,
    scrap_frac: float,
    overrides: dict[str, Any] | None,
    vendor_csv: str | None,
    *,
    default_material_display: str = DEFAULT_MATERIAL_DISPLAY,
    pricing: PricingEngine | None = None,
) -> tuple[float, dict[str, Any]]:

    overrides = overrides or {}
    pricing_engine = pricing or _DEFAULT_PRICING_ENGINE

    key = (material_name or "").strip().upper()
    meta = MATERIAL_MAP.get(key)
    if meta is None:
        if "AL" in key or "6061" in key:
            meta = MATERIAL_MAP["6061"].copy()
        elif "C110" in key or "COPPER" in key:
            meta = MATERIAL_MAP["C110"].copy()
        else:
            meta = {"symbol": key or "XAL", "basis": "usd_per_kg"}
    else:
        meta = meta.copy()

    symbol = str(meta.get("symbol", key or "XAL"))
    basis = str(meta.get("basis", "index_usd_per_tonne"))

    vendor_csv = vendor_csv or ""
    usd_per_kg: float | None = None
    source = ""
    basis_used = basis
    price_candidates: list[str] = []

    if vendor_csv:
        try:
            vendor_quote = pricing_engine.get_usd_per_kg(
                symbol,
                basis,
                vendor_csv=vendor_csv,
                providers=("vendor_csv",),
            )
        except Exception:
            vendor_quote = None
        if vendor_quote:
            usd_per_kg = vendor_quote.usd_per_kg
            source = vendor_quote.source
            basis_used = vendor_quote.basis

    if usd_per_kg is None:
        # If overrides include a stock plan with explicit dims, derive USD/kg
        try:
            stock_plan = overrides.get("stock_recommendation") if isinstance(overrides, dict) else None
        except Exception:
            stock_plan = None

        if isinstance(stock_plan, dict) and (usd_per_kg is None):
            L_mm = _coerce_float_or_none(stock_plan.get("length_mm"))
            W_mm = _coerce_float_or_none(stock_plan.get("width_mm"))
            T_mm = _coerce_float_or_none(stock_plan.get("thickness_mm"))

            if L_mm and W_mm and T_mm and (L_mm > 0) and (W_mm > 0) and (T_mm > 0):
                try:
                    sku, price_each, uom, dims_in = lookup_sku_and_price_for_mm(
                        material_name or symbol,
                        float(L_mm),
                        float(W_mm),
                        float(T_mm),
                        qty=1,
                    )
                except Exception:
                    sku, price_each, uom, dims_in = None, None, None, (None, None, None)
                if sku and price_each and isinstance(uom, str) and uom.strip().lower() == "each":
                    norm_key = _normalize_lookup_key(material_name or symbol)
                    density_g_cc = (
                        MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(norm_key)
                        or MATERIAL_DENSITY_G_CC_BY_KEY.get(symbol)
                        or MATERIAL_DENSITY_G_CC_BY_KEY.get(norm_key)
                    )
                    dims_tuple = tuple(float(x) if x is not None else 0.0 for x in (dims_in or ()))
                    if len(dims_tuple) != 3 or not all(val > 0 for val in dims_tuple):
                        dims_tuple = (float(L_mm) / 25.4, float(W_mm) / 25.4, float(T_mm) / 25.4)
                    if density_g_cc and density_g_cc > 0:
                        L_stock_mm, W_stock_mm, T_stock_mm = (val * 25.4 for val in dims_tuple)
                        volume_cm3 = (L_stock_mm * W_stock_mm * T_stock_mm) / 1000.0
                        stock_mass_kg = (volume_cm3 * float(density_g_cc)) / 1000.0
                        if stock_mass_kg and stock_mass_kg > 0:
                            usd_per_kg = float(price_each) / float(stock_mass_kg)
                            source = f"mcmaster_stock:{sku}"
                            basis_used = "usd_per_kg"

    if usd_per_kg is None:
        lookup_label = material_name or key or symbol
        if lookup_label:
            try:
                mcm_price, mcm_source = _get_mcmaster_unit_price(lookup_label, unit="kg")
            except Exception:
                mcm_price, mcm_source = None, ""
            if mcm_price and math.isfinite(float(mcm_price)):
                usd_per_kg = float(mcm_price)
                source = mcm_source or "mcmaster"
                basis_used = "usd_per_kg"

    if usd_per_kg is None:
        wieland_key = meta.get("wieland_key")
        if wieland_key:
            price_candidates.append(str(wieland_key))
        if material_name:
            price_candidates.append(str(material_name))
        if key:
            price_candidates.append(str(key))
        if symbol:
            price_candidates.append(str(symbol))

        usd_wieland, source_wieland = lookup_wieland_price(price_candidates)
        if usd_wieland is not None:
            usd_per_kg = usd_wieland
            source = source_wieland or "wieland"
            basis_used = "usd_per_kg"

    provider_error: Exception | None = None
    if usd_per_kg is None:
        try:
            provider_quote = pricing_engine.get_usd_per_kg(
                symbol,
                basis,
                vendor_csv=vendor_csv if vendor_csv else None,
            )
        except Exception as err:
            provider_quote = None
            provider_error = err
        else:
            usd_per_kg = provider_quote.usd_per_kg
            source = provider_quote.source
            basis_used = provider_quote.basis

    if (usd_per_kg is None) or (not math.isfinite(float(usd_per_kg))) or (usd_per_kg <= 0):
        resolver_name = material_name or ""
        if not resolver_name:
            fallback_key = _normalize_lookup_key(default_material_display)
            resolver_name = MATERIAL_DISPLAY_BY_KEY.get(fallback_key, default_material_display)
        try:
            resolved_price, resolver_source = _resolve_material_unit_price(
                resolver_name,
                unit="kg",
            )

        except Exception:
            resolved_price, resolver_source = None, ""
        if resolved_price and math.isfinite(float(resolved_price)):
            usd_per_kg = float(resolved_price)
            source = resolver_source or source or "resolver"
            basis_used = "usd_per_kg"

    if usd_per_kg is None:
        usd_per_kg = 0.0
        source = source or (str(provider_error) if provider_error else "price_unavailable")

    premium = float(meta.get("premium_usd_per_kg", 0.0))
    premium_override = overrides.get("premium_usd_per_kg")
    if premium_override is not None:
        try:
            premium = float(premium_override)
        except Exception:
            pass

    scrap_override = overrides.get("scrap_pct_override")
    if scrap_override is not None:
        try:
            scrap_frac = float(scrap_override)
        except Exception:
            pass
    scrap_frac = max(0.0, float(scrap_frac))

    loss_factor = float(meta.get("loss_factor", 0.0))
    effective_scrap = max(0.0, scrap_frac + max(0.0, loss_factor))
    effective_kg = float(mass_kg) * (1.0 + effective_scrap)

    unit_price = usd_per_kg + premium
    cost = effective_kg * unit_price

    effective_mass_g = effective_kg * 1000.0
    net_mass_g = float(mass_kg) * 1000.0

    detail = {
        "material_name": material_name,
        "symbol": symbol,
        "basis": basis_used,
        "source": source,
        "mass_g_net": net_mass_g,
        "net_mass_g": net_mass_g,
        "mass_g": effective_mass_g,
        "effective_mass_g": effective_mass_g,
        "scrap_pct": scrap_frac,
        "loss_factor": loss_factor,
        "unit_price_usd_per_kg": unit_price,
        "unit_price_usd_per_lb": unit_price / LB_PER_KG if unit_price else 0.0,
        "unit_price_source": source,
        "vendor_premium_usd_per_kg": premium,
        "material_cost": cost,
    }
    if source:
        m = re.search(r"\(([^)]+)\)\s*$", source)
        if m:
            detail["unit_price_asof"] = m.group(1)
    if price_candidates:
        detail["price_lookup_keys"] = price_candidates
    if provider_error and "error" not in detail:
        detail.setdefault("provider_error", str(provider_error))

    return cost, detail


def _material_family(material: str) -> str:
    name = _normalize_lookup_key(material)
    if not name:
        return "steel"
    if any(tag in name for tag in ("alum", "6061", "7075", "2024", "5052", "5083")):
        return "alum"
    if any(tag in name for tag in ("stainless", "17 4", "316", "304", "ss")):
        return "stainless"
    if any(tag in name for tag in ("titanium", "ti-6al-4v", "ti64", "grade 5")):
        return "titanium"
    if any(tag in name for tag in ("copper", "c110", "cu")):
        return "copper"
    if any(tag in name for tag in ("brass", "c360", "c260")):
        return "brass"
    if any(
        tag in name
        for tag in ("plastic", "uhmw", "delrin", "acetal", "peek", "abs", "nylon")
    ):
        return "plastic"

    return "steel"


def _density_for_material(material: str, default: float = _DEFAULT_MATERIAL_DENSITY_G_CC) -> float:
    """Return a rough density guess (g/cc) for the requested material."""

    raw = (material or "").strip()
    if not raw:
        return default

    normalized = _normalize_lookup_key(raw)
    collapsed = normalized.replace(" ", "")

    for token in (normalized, collapsed):
        if token and token in MATERIAL_DENSITY_G_CC_BY_KEYWORD:
            return MATERIAL_DENSITY_G_CC_BY_KEYWORD[token]

    for token, density in MATERIAL_DENSITY_G_CC_BY_KEYWORD.items():
        if not token:
            continue
        if token in normalized or token in collapsed:
            return density

    lower = raw.lower()
    if any(tag in lower for tag in ("plastic", "uhmw", "delrin", "acetal", "peek", "abs", "nylon")):
        return 1.45
    if any(tag in lower for tag in ("foam", "poly", "composite")):
        return 1.10
    if any(tag in lower for tag in ("magnesium", "az31", "az61")):
        return 1.80
    if "graphite" in lower:
        return 1.85

    return default


def require_plate_inputs(geo: dict, ui_vars: dict[str, Any] | None) -> None:
    ui_vars = ui_vars or {}
    thickness_val = ui_vars.get("Thickness (in)")
    thickness_in = _coerce_float_or_none(thickness_val)
    if thickness_in is None and thickness_val is not None:
        try:
            thickness_in = float(thickness_val)
        except Exception:
            thickness_in = 0.0
    elif thickness_in is None:
        thickness_in = 0.0
    material = str(ui_vars.get("Material") or "").strip()
    geo_missing: list[str] = []

    if not material:
        geo_missing.append("material")
    if not geo.get("thickness_mm") and (thickness_in or 0.0) <= 0:
        geo_missing.append("thickness")

    if geo_missing:
        raise ValueError(
            "Missing required inputs for plate quote: "
            + ", ".join(geo_missing)
            + ". Set Material and Thickness in the Quote Editor."
        )

    thickness_in = float(thickness_in or 0.0)
    if thickness_in > 0 and not geo.get("thickness_mm"):
        geo["thickness_mm"] = thickness_in * 25.4
    if material and not geo.get("material"):
        geo["material"] = material


def net_mass_kg(
    plate_L_in,
    plate_W_in,
    t_in,
    hole_d_mm,
    density_g_cc: float = 7.85,
    *,
    return_removed_mass: bool = False,
):
    """Estimate the net mass of a rectangular plate and optional removed material.

    When ``return_removed_mass`` is true the function returns a tuple
    ``(net_mass_kg, removed_mass_g)`` where ``removed_mass_g`` is the grams of
    material evacuated by the provided circular holes.  Otherwise the previous
    behaviour of returning only the net mass in kilograms is preserved.
    """

    if not (plate_L_in and plate_W_in and t_in):
        return (None, None) if return_removed_mass else None

    L_val = _coerce_float_or_none(plate_L_in)
    W_val = _coerce_float_or_none(plate_W_in)
    T_val = _coerce_float_or_none(t_in)
    if L_val is None or W_val is None or T_val is None:
        return (None, None) if return_removed_mass else None

    L = float(L_val)
    W = float(W_val)
    T = float(T_val)
    if L <= 0 or W <= 0 or T <= 0:
        return (None, None) if return_removed_mass else None
    vol_plate_in3 = L * W * T
    try:
        t_mm = T * 25.4
    except Exception:
        t_mm = None
    if t_mm is None:
        return None
    holes_mm = []
    for d in hole_d_mm or []:
        val = _coerce_float_or_none(d)
        if val and val > 0:
            holes_mm.append(float(val))
    vol_holes_mm3 = sum(math.pi * (d / 2.0) ** 2 * t_mm for d in holes_mm)
    vol_holes_in3 = vol_holes_mm3 / 16387.064 if vol_holes_mm3 else 0.0
    vol_net_in3 = max(0.0, vol_plate_in3 - vol_holes_in3)
    vol_cc = vol_net_in3 * 16.387064
    mass_g = vol_cc * float(density_g_cc or 0.0)
    removed_mass_g = vol_holes_in3 * 16.387064 * float(density_g_cc or 0.0)

    net_mass = mass_g / 1000.0 if mass_g else 0.0
    if return_removed_mass:
        return net_mass, removed_mass_g
    return net_mass


def _normalize_speeds_feeds_df(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        normalized = re.sub(r"[^0-9a-z]+", "_", str(col).strip().lower())
        normalized = normalized.strip("_")
        rename[col] = normalized
    normalized_df = df.rename(columns=rename)
    return normalized_df


def _coerce_speeds_feeds_records(table: Any | None) -> list[Mapping[str, Any]]:
    if table is None:
        return []
    records: list[Mapping[str, Any]] = []
    if hasattr(table, "to_dict"):
        try:
            raw_records = table.to_dict("records")  # type: ignore[attr-defined]
        except Exception:
            raw_records = None
        if isinstance(raw_records, list):
            records = [row for row in raw_records if isinstance(row, Mapping)]
    if not records:
        stub_rows = getattr(table, "_rows", None)
        if isinstance(stub_rows, list):
            records = [row for row in stub_rows if isinstance(row, Mapping)]
    if not records and isinstance(table, Sequence):
        records = [row for row in table if isinstance(row, Mapping)]  # type: ignore[arg-type]
    return records


def _record_key_map(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        re.sub(r"[^0-9a-z]+", "_", str(key).strip().lower()).strip("_"): key
        for key in record.keys()
    }


def _lookup_material_group_from_table(
    table: Any | None,
    normalized_material_key: str,
) -> str | None:
    if not normalized_material_key:
        return None
    for record in _coerce_speeds_feeds_records(table):
        key_map = _record_key_map(record)
        mat_field = next(
            (key_map[name] for name in ("material", "material_name", "canonical_material") if name in key_map),
            None,
        )
        group_field = next(
            (key_map[name] for name in ("material_group", "iso_group", "group") if name in key_map),
            None,
        )
        if mat_field is None or group_field is None:
            continue
        material_value = record.get(mat_field)
        if _normalize_lookup_key(str(material_value or "")) != normalized_material_key:
            continue
        group_value = record.get(group_field)
        if not group_value:
            continue
        return str(group_value).strip().upper()
    return None


def _material_label_from_table(
    table: Any | None,
    material_key: str | None,
    normalized_lookup: str,
) -> str | None:
    records = _coerce_speeds_feeds_records(table)
    if not records:
        return None
    target_group = str(material_key or "").strip().upper()
    for record in records:
        key_map = _record_key_map(record)
        mat_field = next(
            (key_map[name] for name in ("material", "material_name", "canonical_material") if name in key_map),
            None,
        )
        group_field = next(
            (key_map[name] for name in ("material_group", "iso_group", "group") if name in key_map),
            None,
        )
        if mat_field is None:
            continue
        row_material = record.get(mat_field)
        if normalized_lookup:
            if _normalize_lookup_key(str(row_material or "")) == normalized_lookup:
                label = str(row_material).strip()
                if label:
                    return label
        if group_field and target_group:
            group_value = record.get(group_field)
            if group_value and str(group_value).strip().upper() == target_group:
                label = str(row_material).strip()
                if label:
                    return label
    return None


def _load_speeds_feeds_table(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        resolved = Path(str(path)).expanduser()
    except Exception:
        return None
    key = str(resolved)
    if key in _SPEEDS_FEEDS_CACHE:
        return _SPEEDS_FEEDS_CACHE[key]
    if not resolved.is_file():
        _SPEEDS_FEEDS_CACHE[key] = None
        return None
    try:
        df = pd.read_csv(resolved)
    except Exception as exc:
        logger.warning("Failed to read speeds/feeds CSV at %s: %s", resolved, exc)
        _SPEEDS_FEEDS_CACHE[key] = None
        return None
    normalized = _normalize_speeds_feeds_df(df)
    _SPEEDS_FEEDS_CACHE[key] = normalized
    return normalized


def _select_speeds_feeds_row(
    table: pd.DataFrame | None,
    operation: str,
    material_key: str | None = None,
) -> Mapping[str, Any] | None:
    if table is None:
        return None
    try:
        if getattr(table, "empty"):
            return None
    except Exception:
        try:
            if len(table) == 0:
                return None
        except Exception:
            pass
    op_col = next((col for col in ("operation", "op", "process") if col in table.columns), None)
    if op_col is None:
        return None
    op_raw = str(operation or "").strip().lower()
    op_target = op_raw.replace("-", "_").replace(" ", "_")
    op_variants: set[str] = set()
    if op_target:
        op_variants.add(op_target)
        trimmed = op_target.rstrip("_")
        if trimmed:
            op_variants.add(trimmed)
        if op_target.endswith("ing") and len(op_target) > 3:
            op_variants.add(op_target[:-3])
        op_variants.add(op_target.replace("_", " "))
    drill_synonyms = {
        "drill": {"drill", "drilling"},
        "deep_drill": {
            "deep_drill",
            "deep drilling",
            "deepdrill",
            "deep drill",
        },
    }
    for key, synonyms in drill_synonyms.items():
        key_norm = key.replace("-", "_").replace(" ", "_")
        if key_norm in op_variants:
            for syn in synonyms:
                syn_norm = syn.strip().lower().replace("-", "_").replace(" ", "_")
                if syn_norm:
                    op_variants.add(syn_norm)
    op_variants = {variant for variant in op_variants if variant}
    try:
        records = table.to_dict("records")  # type: ignore[attr-defined]
    except Exception:
        records = getattr(table, "_rows", None)
        if records is None:
            records = []
    if not records:
        return None
    def _normalize(text: Any) -> str:
        raw = "" if text is None else str(text)
        return raw.strip().lower().replace("-", "_").replace(" ", "_")

    candidates = [
        (idx, row, _normalize(row.get(op_col)))
        for idx, row in enumerate(records)
    ]
    matches = [row for _, row, norm in candidates if norm in op_variants]
    if not matches:
        matches = [
            row
            for _, row, norm in candidates
            if any(variant and variant in norm for variant in op_variants)
        ]
    if not matches:
        return None
    if material_key and matches:
        preferred_columns = ("material_group", "material_family", "material")
        candidate_columns: list[str] = []
        try:
            table_columns = list(getattr(table, "columns", []))
        except Exception:
            table_columns = []
        for col in preferred_columns:
            if col in table_columns and col not in candidate_columns:
                candidate_columns.append(col)
        if not candidate_columns:
            seen: set[str] = set()
            for row in matches:
                if isinstance(row, Mapping):
                    for col in preferred_columns:
                        if col in row and col not in seen:
                            candidate_columns.append(col)
                            seen.add(col)
        mat_target = str(material_key).strip().lower()
        for mat_col in candidate_columns:
            exact = [row for row in matches if _normalize(row.get(mat_col)) == mat_target]
            if exact:
                matches = exact
                break
            partial = [row for row in matches if mat_target in _normalize(row.get(mat_col))]
            if partial:
                matches = partial
    return cast(Mapping[str, Any], matches[0]) if matches else None


def _clean_path_text(text: str) -> str:
    raw = text.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1].strip()
    return os.path.expandvars(raw)


def _stringify_resolved_path(path_obj: Path) -> str:
    try:
        try:
            return str(path_obj.resolve(strict=False))
        except TypeError:
            return str(path_obj.resolve())
    except Exception:
        return str(path_obj)


def _resolve_speeds_feeds_path(
    params: Mapping[str, Any] | None,
    ui_vars: Mapping[str, Any] | None = None,
) -> str | None:
    candidates: list[str] = []
    seen: set[str] = set()

    def _push(value: Any) -> None:
        if not value:
            return
        try:
            text = str(value).strip()
        except Exception:
            return
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(text)

    ui_mapping = ui_vars if isinstance(ui_vars, Mapping) else None
    if ui_mapping:
        for key in (
            "SpeedsFeedsCSVPath",
            "Speeds Feeds CSV",
            "Speeds/Feeds CSV",
            "Speeds & Feeds CSV",
        ):
            _push(ui_mapping.get(key))

    if isinstance(params, Mapping):
        for key in (
            "SpeedsFeedsCSVPath",
            "Speeds Feeds CSV",
            "Speeds/Feeds CSV",
            "Speeds & Feeds CSV",
        ):
            _push(params.get(key))

    for env_key in ("CADQ_SF_CSV", "SPEEDS_FEEDS_CSV", "CAD_QUOTER_SPEEDS_FEEDS"):
        _push(os.environ.get(env_key))

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    fallback_paths: Sequence[Path | str] = (
        script_dir / "speeds_feeds_merged.csv",
        repo_root / "speeds_feeds_merged.csv",
        repo_root / "cad_quoter" / "pricing" / "resources" / "speeds_feeds_merged.csv",
        r"D:\\CAD_Quoting_Tool\\speeds_feeds_merged.csv",
    )
    for candidate in fallback_paths:
        _push(candidate)

    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _machine_params_from_params(params: Mapping[str, Any] | None) -> _TimeMachineParams:
    rapid = _coerce_float_or_none(params.get("MachineRapidIPM")) if isinstance(params, Mapping) else None
    hp = _coerce_float_or_none(params.get("MachineHorsepower")) if isinstance(params, Mapping) else None
    mrr_factor = (
        _coerce_float_or_none(params.get("MachineHpToMrrFactor"))
        if isinstance(params, Mapping)
        else None
    )
    return _TimeMachineParams(
        rapid_ipm=float(rapid) if rapid and rapid > 0 else 300.0,
        hp_available=float(hp) if hp and hp > 0 else None,
        hp_to_mrr_factor=float(mrr_factor) if mrr_factor and mrr_factor > 0 else None,
    )


def _drill_overhead_from_params(params: Mapping[str, Any] | None) -> _TimeOverheadParams:
    toolchange = (
        _coerce_float_or_none(params.get("DrillToolchangeMinutes"))
        if isinstance(params, Mapping)
        else None
    )
    approach = (
        _coerce_float_or_none(params.get("DrillApproachRetractIn"))
        if isinstance(params, Mapping)
        else None
    )
    peck = (
        _coerce_float_or_none(params.get("DrillPeckPenaltyMinPerIn"))
        if isinstance(params, Mapping)
        else None
    )
    dwell = (
        _coerce_float_or_none(params.get("DrillDwellMinutes"))
        if isinstance(params, Mapping)
        else None
    )
    index_sec = (
        _coerce_float_or_none(params.get("DrillIndexSecondsPerHole"))
        if isinstance(params, Mapping)
        else None
    )
    return _TimeOverheadParams(
        toolchange_min=float(toolchange) if toolchange and toolchange >= 0 else 0.5,
        approach_retract_in=float(approach) if approach and approach >= 0 else 0.25,
        peck_penalty_min_per_in_depth=float(peck) if peck and peck >= 0 else 0.03,
        dwell_min=float(dwell) if dwell and dwell >= 0 else None,
        index_sec_per_hole=float(index_sec) if index_sec is not None and index_sec >= 0 else None,
    )


def _clean_hole_groups(raw: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        dia = _coerce_float_or_none(entry.get("dia_mm"))
        depth = _coerce_float_or_none(entry.get("depth_mm"))
        count = _coerce_float_or_none(entry.get("count"))
        if dia is None or dia <= 0:
            continue
        qty = int(round(count)) if count is not None else 0
        if qty <= 0:
            qty = 1
        cleaned.append(
            {
                "dia_mm": float(dia),
                "depth_mm": float(depth) if depth is not None else None,
                "count": qty,
                "through": bool(entry.get("through")),
            }
        )
    return cleaned if cleaned else None

MIN_DRILL_MIN_PER_HOLE = 0.10
DEFAULT_MAX_DRILL_MIN_PER_HOLE = 2.00

DEEP_DRILL_SFM_FACTOR = 0.65
DEEP_DRILL_IPR_FACTOR = 0.70
DEEP_DRILL_PECK_PENALTY_MIN_PER_IN = 0.07

DEFAULT_DRILL_INDEX_SEC_PER_HOLE = 5.3746248
DEFAULT_DEEP_DRILL_INDEX_SEC_PER_HOLE = 4.3038756


def _default_drill_index_seconds(operation: str | None) -> float:
    """Return the default indexing time (seconds) for a drill operation."""

    op_name = (operation or "").strip().lower()
    if op_name == "deep_drill":
        return DEFAULT_DEEP_DRILL_INDEX_SEC_PER_HOLE
    return DEFAULT_DRILL_INDEX_SEC_PER_HOLE


def _drill_minutes_per_hole_bounds(
    material_group: str | None = None,
    *,
    depth_in: float | None = None,
) -> tuple[float, float]:
    """Return the (min, max) minutes-per-hole bounds for drilling."""

    min_minutes = MIN_DRILL_MIN_PER_HOLE
    max_minutes = DEFAULT_MAX_DRILL_MIN_PER_HOLE
    depth_value = None
    if depth_in is not None:
        try:
            depth_value = float(depth_in)
        except (TypeError, ValueError):
            depth_value = None
    if depth_value is not None and depth_value <= 0:
        depth_value = None

    if material_group:
        raw_key = str(material_group).strip()
        key_lower = raw_key.lower()
        key_upper = raw_key.upper()

        def _starts_with(prefixes: tuple[str, ...]) -> bool:
            return any(key_upper.startswith(prefix) for prefix in prefixes)

        if (
            "inconel" in key_lower
            or "titanium" in key_lower
            or key_upper.startswith("TI")
            or _starts_with(("S", "H"))
        ):
            max_minutes = 6.0
        elif (
            "steel" in key_lower
            or "stainless" in key_lower
            or _starts_with(("P", "M"))
        ):
            max_minutes = 5.0
        elif (
            "alum" in key_lower
            or "copper" in key_lower
            or "brass" in key_lower
            or "bronze" in key_lower
            or _starts_with(("N", "C"))
        ):
            max_minutes = 2.0
        else:
            max_minutes = DEFAULT_MAX_DRILL_MIN_PER_HOLE

    if depth_value is not None and depth_value > 1.0:
        max_minutes += 0.2 * (depth_value - 1.0)

    max_minutes = max(max_minutes, min_minutes)
    return min_minutes, max_minutes


def _apply_drill_minutes_clamp(
    hours: float,
    hole_count: int,
    *,
    material_group: str | None = None,
    depth_in: float | None = None,
) -> float:
    if hours <= 0.0 or hole_count <= 0:
        return hours
    min_min_per_hole, max_min_per_hole = _drill_minutes_per_hole_bounds(
        material_group,
        depth_in=depth_in,
    )
    min_hr = (hole_count * min_min_per_hole) / 60.0
    max_hr = (hole_count * max_min_per_hole) / 60.0
    return max(min(hours, max_hr), min_hr)


def estimate_drilling_hours(
    hole_diams_mm: list[float],
    thickness_in: float,
    mat_key: str,
    *,
    hole_groups: list[Mapping[str, Any]] | None = None,
    speeds_feeds_table: pd.DataFrame | None = None,
    machine_params: _TimeMachineParams | None = None,
    overhead_params: _TimeOverheadParams | None = None,
    warnings: list[str] | None = None,
    debug_lines: list[str] | None = None,
    debug_summary: dict[str, dict[str, Any]] | None = None,
) -> float:
    """
    Conservative plate-drilling model with floors so 100+ holes don't collapse to minutes.

    ``hole_diams_mm`` is measured in millimetres; ``thickness_in`` is the plate thickness in inches.
    """
    material_lookup = _normalize_lookup_key(mat_key) if mat_key else ""
    material_label = MATERIAL_DISPLAY_BY_KEY.get(material_lookup, mat_key)
    if (
        speeds_feeds_table is not None
        and (not material_label or material_label == mat_key)
    ):
        alt_label = _material_label_from_table(
            speeds_feeds_table,
            mat_key,
            material_lookup,
        )
        if alt_label:
            material_label = alt_label
    mat = str(material_label or mat_key or "").lower()
    material_factor = _unit_hp_cap(material_label)
    debug: dict[str, Any] | None
    # Create a local debug aggregate only when caller requested debug output.
    # Previously referenced an undefined 'debug_meta'; use available signals instead.
    if (debug_lines is not None) or (debug_summary is not None):
        debug = {}
    else:
        debug = None
    debug_list = debug_lines if debug_lines is not None else None
    if debug_summary is not None:
        debug_summary.clear()
    avg_dia_in = 0.0
    seen_debug: set[str] = set()
    chosen_material_label: str = ""

    def _log_debug(entry: str) -> None:
        if debug_list is None:
            return
        text = str(entry or "").strip()
        if not text or text in seen_debug:
            return
        debug_list.append(text)
        seen_debug.add(text)

    def _update_range(target: dict[str, Any], min_key: str, max_key: str, value: Any) -> None:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val):
            return
        current_min = target.get(min_key)
        if current_min is None or val < current_min:
            target[min_key] = val
        current_max = target.get(max_key)
        if current_max is None or val > current_max:
            target[max_key] = val

    if debug_list is not None and speeds_feeds_table is None:
        _log_debug("MISS table: using heuristic fallback")

    group_specs: list[tuple[float, int, float]] = []
    # Optional aggregate debug dict for callers that want structured details
    fallback_counts: Counter[float] | None = None

    if hole_groups:
        fallback_counts = Counter()
        for entry in hole_groups:
            if not isinstance(entry, Mapping):
                continue
            dia_mm = _coerce_float_or_none(entry.get("dia_mm"))
            count = _coerce_float_or_none(entry.get("count"))
            depth_mm = _coerce_float_or_none(entry.get("depth_mm"))
            if dia_mm is None or dia_mm <= 0:
                continue
            qty = int(round(count)) if count is not None else 0
            if qty <= 0:
                qty = int(max(1, round(count or 1)))
            depth_in = 0.0
            diameter_in = float(dia_mm) / 25.4
            if depth_mm and depth_mm > 0:
                depth_in = float(depth_mm) / 25.4
            elif thickness_in and thickness_in > 0:
                depth_in = float(thickness_in)
            breakthrough_in = max(0.04, 0.2 * diameter_in)
            if depth_in > 0:
                depth_in += breakthrough_in
            else:
                depth_in = breakthrough_in
            group_specs.append((diameter_in, qty, depth_in))
            fallback_counts[round(float(dia_mm), 3)] += qty
    if not group_specs:
        if not hole_diams_mm or thickness_in <= 0:
            return 0.0
        depth_in = float(thickness_in)
        counts = Counter(round(float(d), 3) for d in hole_diams_mm if d and math.isfinite(d))
        for dia_mm, qty in counts.items():
            if qty <= 0:
                continue
            diameter_in = float(dia_mm) / 25.4
            breakthrough_in = max(0.04, 0.2 * diameter_in)
            total_depth_in = depth_in + breakthrough_in if depth_in > 0 else breakthrough_in
            group_specs.append((diameter_in, int(qty), total_depth_in))
        fallback_counts = counts
    elif fallback_counts is None:
        fallback_counts = Counter()
        for dia_in, qty, _ in group_specs:
            fallback_counts[round(dia_in * 25.4, 3)] += qty

    if group_specs:
        base_machine = machine_params or _machine_params_from_params(None)
        overhead = overhead_params or _drill_overhead_from_params(None)
        overhead_for_calc = overhead
        per_hole_overhead = replace(overhead, toolchange_min=0.0)
        total_min = 0.0
        total_toolchange_min = 0.0
        total_holes = 0
        material_cap_val = to_float(material_factor)
        if material_cap_val is not None and material_cap_val <= 0:
            material_cap_val = None

        total_qty_for_avg = 0
        weighted_dia_sum = 0.0
        depth_candidates: list[float] = []
        for diameter_in, qty, depth_val in group_specs:
            try:
                qty_int = int(qty)
            except Exception:
                qty_int = 0
            if qty_int <= 0:
                continue
            total_qty_for_avg += qty_int
            weighted_dia_sum += float(diameter_in) * qty_int
            if depth_val and depth_val > 0:
                depth_candidates.append(float(depth_val))
        if total_qty_for_avg > 0:
            avg_dia_in = weighted_dia_sum / total_qty_for_avg
        depth_for_bounds = max(depth_candidates) if depth_candidates else None

        hp_cap_val = to_float(getattr(base_machine, "hp_to_mrr_factor", None))
        combined_cap = None
        if material_cap_val is not None and hp_cap_val is not None:
            combined_cap = min(hp_cap_val, material_cap_val)
        elif material_cap_val is not None:
            combined_cap = material_cap_val
        elif hp_cap_val is not None:
            combined_cap = hp_cap_val
        if combined_cap is not None:
            machine_for_cut = _TimeMachineParams(
                rapid_ipm=base_machine.rapid_ipm,
                hp_available=base_machine.hp_available,
                hp_to_mrr_factor=float(combined_cap),
            )
        else:
            machine_for_cut = base_machine

        row_cache: dict[tuple[str, float], tuple[Mapping[str, Any], _TimeToolParams] | None] = {}
        missing_row_messages: set[tuple[str, str, float]] = set()
        debug_summary_entries: dict[str, dict[str, Any]] = {}

        def _build_tool_params(row: Mapping[str, Any]) -> _TimeToolParams:
            key_map = {
                str(k).strip().lower().replace("-", "_").replace(" ", "_"): k
                for k in row.keys()
            }

            def _row_float(*names: str) -> float | None:
                for name in names:
                    actual = key_map.get(name)
                    if actual is None:
                        continue
                    val = to_float(row.get(actual))
                    if val is not None:
                        return float(val)
                return None

            teeth_val = _row_float("teeth_z", "flutes", "flute_count", "teeth")
            teeth_int: int | None = None
            if teeth_val is not None and teeth_val > 0:
                try:
                    teeth_int = int(round(teeth_val))
                except Exception:
                    teeth_int = None
            if teeth_int is None or teeth_int <= 0:
                teeth_int = 1
            return _TimeToolParams(teeth_z=teeth_int)

        # Use the function's input thickness (inches) for L/D heuristics
        try:
            thickness_in_val = float(thickness_in) if thickness_in and float(thickness_in) > 0 else None
        except Exception:
            thickness_in_val = None

        for diameter_in, qty, depth_in in group_specs:
            qty_i = int(qty)
            if qty_i <= 0 or diameter_in <= 0 or depth_in <= 0:
                continue
            tool_dia_in = float(diameter_in)
            thickness_for_ratio = thickness_in_val if thickness_in_val and thickness_in_val > 0 else depth_in
            l_over_d = 0.0
            if tool_dia_in > 0 and thickness_for_ratio and thickness_for_ratio > 0:
                l_over_d = float(thickness_for_ratio) / float(tool_dia_in)
            op_name = "deep_drill" if l_over_d >= 3.0 else "drill"
            cache_key = (op_name, round(float(diameter_in), 4))
            row: Mapping[str, Any] | None = None
            cache_entry = row_cache.get(cache_key)
            if cache_entry is None:
                material_for_lookup: str | None = None
                for candidate in (material_label, mat_key, material_lookup):
                    if candidate:
                        material_for_lookup = str(candidate)
                        break

                row: Mapping[str, Any] | None = None
                if speeds_feeds_table is not None:
                    row = _select_speeds_feeds_row(
                        speeds_feeds_table,
                        operation=op_name,
                        material_key=material_for_lookup,
                    )
                    if not row and op_name.lower() == "deep_drill":
                        row = _select_speeds_feeds_row(
                            speeds_feeds_table,
                            operation="Drill",
                            material_key=material_for_lookup,
                        )

                if not row:
                    row = _pick_speeds_row(
                        material_label=material_label,
                        operation=op_name,
                        tool_diameter_in=float(diameter_in),
                        table=speeds_feeds_table,
                    )
                if not row and op_name.lower() == "deep_drill":
                    row = _pick_speeds_row(
                        material_label=material_label,
                        operation="drill",
                        tool_diameter_in=float(diameter_in),
                        table=speeds_feeds_table,
                    )
                if row and isinstance(row, Mapping):
                    cache_entry = (row, _build_tool_params(row))
                    # Always use one material label for both Debug and Calc.
                    chosen_material_label = str(
                        row.get("material")
                        or row.get("material_family")
                        or material_label
                        or mat_key
                        or material_lookup
                        or ""
                    ).strip()
                    if debug_list is not None:
                        row_material = chosen_material_label
                        qty_display = qty
                        try:
                            qty_display = int(qty)
                        except Exception:
                            pass
                        depth_display = depth_in
                        try:
                            depth_display = float(depth_in)
                        except Exception:
                            depth_display = depth_in
                        sfm_val = to_float(row.get("sfm_start") or row.get("sfm"))
                        feed_val = to_float(
                            row.get("fz_ipr_0_25in")
                            or row.get("feed_ipr")
                            or row.get("fz")
                            or row.get("ipr")
                        )
                        info_bits: list[str] = [
                            f"OK {op_name}",
                            f"{float(diameter_in):.3f}\"",
                        ]
                        if depth_display:
                            try:
                                info_bits.append(f"depth {float(depth_display):.3f}\"")
                            except Exception:
                                info_bits.append(f"depth {depth_display}")
                        info_bits.append(f"qty {qty_display}")
                        if row_material:
                            info_bits.append(row_material)
                        if sfm_val is not None:
                            info_bits.append(f"{sfm_val:g} sfm")
                        if feed_val is not None:
                            info_bits.append(f"{feed_val:g} ipr")
                        _log_debug(" | ".join(info_bits))
                else:
                    cache_entry = None
                    material_display = str(material_label or mat_key or material_lookup or "material").strip()
                    if not material_display:
                        material_display = "material"
                    op_display = "deep drilling" if op_name.lower() == "deep_drill" else "drilling"
                    missing_row_messages.add(
                        (
                            op_display,
                            material_display,
                            round(float(diameter_in), 4),
                        )
                    )
                    _log_debug(
                        f"MISS {op_display} {material_display.lower()} {round(float(diameter_in), 4):.3f}\""
                    )
                row_cache[cache_key] = cache_entry
            else:
                try:
                    row = cache_entry[0]
                except Exception:
                    row = None
            geom = _TimeOperationGeometry(
                diameter_in=float(diameter_in),
                hole_depth_in=float(depth_in),
                point_angle_deg=118.0,
                ld_ratio=l_over_d,
            )
            diameter_float = to_float(diameter_in)
            if diameter_float is None:
                try:
                    diameter_float = float(diameter_in)
                except Exception:
                    diameter_float = None
            precomputed_speeds: dict[str, float] = {}
            row_view = _time_estimator._RowView(row)
            sfm_candidate = getattr(row_view, "sfm_start", None)
            if sfm_candidate is None:
                sfm_candidate = getattr(row_view, "sfm", None)
            sfm_val = _time_estimator.to_num(sfm_candidate)
            if sfm_val is not None and math.isfinite(sfm_val):
                precomputed_speeds["sfm"] = float(sfm_val)
            ipr_val = _time_estimator.pick_feed_value(row_view, float(diameter_float or 0.0))
            if ipr_val is not None and math.isfinite(ipr_val):
                precomputed_speeds["ipr"] = float(ipr_val)
            rpm_val: float | None = None
            if diameter_float and diameter_float > 0 and "sfm" in precomputed_speeds:
                rpm_val = (precomputed_speeds["sfm"] * 12.0) / (math.pi * float(diameter_float))
                if math.isfinite(rpm_val):
                    precomputed_speeds["rpm"] = float(rpm_val)
                else:
                    rpm_val = None
            ipm_val: float | None = None
            if rpm_val is not None and "ipr" in precomputed_speeds:
                ipm_val = float(rpm_val) * precomputed_speeds["ipr"]
                if math.isfinite(ipm_val):
                    precomputed_speeds["ipm"] = float(ipm_val)
                else:
                    ipm_val = None
            is_deep_drill = op_name.lower() == "deep_drill"
            if is_deep_drill:
                sfm_pre = precomputed_speeds.get("sfm")
                if sfm_pre is not None and math.isfinite(sfm_pre):
                    new_sfm = float(sfm_pre) * DEEP_DRILL_SFM_FACTOR
                    precomputed_speeds["sfm"] = new_sfm
                    if diameter_float and diameter_float > 0:
                        rpm_val = (new_sfm * 12.0) / (math.pi * float(diameter_float))
                        if math.isfinite(rpm_val):
                            precomputed_speeds["rpm"] = float(rpm_val)
                elif "rpm" in precomputed_speeds:
                    rpm_only = precomputed_speeds.get("rpm")
                    if rpm_only is not None and math.isfinite(rpm_only):
                        precomputed_speeds["rpm"] = float(rpm_only) * DEEP_DRILL_SFM_FACTOR
                ipr_pre = precomputed_speeds.get("ipr")
                if ipr_pre is not None and math.isfinite(ipr_pre):
                    precomputed_speeds["ipr"] = float(ipr_pre) * DEEP_DRILL_IPR_FACTOR
                if "rpm" in precomputed_speeds and "ipr" in precomputed_speeds:
                    rpm_calc = precomputed_speeds["rpm"]
                    ipr_calc = precomputed_speeds["ipr"]
                    if math.isfinite(rpm_calc) and math.isfinite(ipr_calc):
                        precomputed_speeds["ipm"] = float(rpm_calc) * float(ipr_calc)
            debug_payload: dict[str, Any] | None = None
            tool_params: _TimeToolParams
            minutes: float
            overhead_for_calc = per_hole_overhead
            if is_deep_drill:
                peck_rate_val = to_float(
                    per_hole_overhead.peck_penalty_min_per_in_depth
                )
                adjusted_peck = max(
                    DEEP_DRILL_PECK_PENALTY_MIN_PER_IN,
                    float(peck_rate_val) if peck_rate_val and peck_rate_val > 0 else 0.0,
                )
                overhead_for_calc = replace(
                    per_hole_overhead,
                    peck_penalty_min_per_in_depth=adjusted_peck,
                )
            if cache_entry:
                row, tool_params = cache_entry
                if debug_lines is not None:
                    debug_payload = {}
                minutes = _estimate_time_min(
                    row,
                    geom,
                    tool_params,
                    machine_for_cut,
                    overhead_for_calc,
                    material_factor=material_cap_val,
                    debug=debug_payload,
                    precomputed=precomputed_speeds,
                )
                overhead_for_calc = per_hole_overhead
            else:
                peck_rate = to_float(
                    overhead_for_calc.peck_penalty_min_per_in_depth
                )
                peck_min = None
                if peck_rate and depth_in and depth_in > 0:
                    peck_min = float(peck_rate) * float(depth_in)
                dwell_val = to_float(overhead_for_calc.dwell_min)
                legacy_overhead = _TimeOverheadParams(
                    toolchange_min=0.0,
                    approach_retract_in=overhead_for_calc.approach_retract_in,
                    peck_penalty_min_per_in_depth=None,
                    dwell_min=dwell_val,
                    peck_min=peck_min,
                    index_sec_per_hole=overhead_for_calc.index_sec_per_hole,
                )
                overhead_for_calc = legacy_overhead
                tool_params = _TimeToolParams(teeth_z=1)
                if debug_lines is not None:
                    debug_payload = {}
                effective_index_sec = to_float(overhead_for_calc.index_sec_per_hole)
                if effective_index_sec is None or not math.isfinite(effective_index_sec):
                    effective_index_sec = _default_drill_index_seconds(op_name)
                legacy_overhead = replace(
                    legacy_overhead,
                    index_sec_per_hole=effective_index_sec,
                )
                overhead_for_calc = legacy_overhead
                minutes = _estimate_time_min(
                    geom=geom,
                    tool=tool_params,
                    machine=machine_for_cut,
                    overhead=overhead_for_calc,
                    material_factor=material_cap_val,
                    operation=op_name,
                    debug=debug_payload,
                    precomputed=precomputed_speeds,
                )
                overhead_for_calc = legacy_overhead
                if minutes <= 0:
                    continue
                overhead_for_calc = legacy_overhead
            if minutes <= 0:
                continue
            try:
                qty_int = int(qty)
            except Exception:
                continue
            if qty_int <= 0:
                continue
            total_holes += qty_int
            total_min += minutes * qty_int
            toolchange_added = 0.0
            if overhead.toolchange_min and qty_int > 0:
                toolchange_added = float(overhead.toolchange_min)
                total_toolchange_min += toolchange_added
            if debug_payload is not None:
                try:
                    operation_name = str(debug_payload.get("operation") or op_name).lower()
                except Exception:
                    operation_name = op_name.lower()
                if precomputed_speeds:
                    for key, value in precomputed_speeds.items():
                        debug_payload.setdefault(key, value)
                sfm_val = precomputed_speeds.get("sfm") if precomputed_speeds else None
                if sfm_val is None:
                    sfm_val = debug_payload.get("sfm")
                ipr_val = precomputed_speeds.get("ipr") if precomputed_speeds else None
                if ipr_val is None:
                    ipr_val = debug_payload.get("ipr")
                rpm_val = precomputed_speeds.get("rpm") if precomputed_speeds else None
                if rpm_val is None:
                    rpm_val = debug_payload.get("rpm")
                ipm_val = precomputed_speeds.get("ipm") if precomputed_speeds else None
                if ipm_val is None:
                    ipm_val = debug_payload.get("ipm")
                depth_val = debug_payload.get("axial_depth_in")
                minutes_per = debug_payload.get("minutes_per_hole")
                qty_for_debug = int(qty) if qty else 0
                mat_display = chosen_material_label or str(
                    material_label or mat_key or material_lookup or ""
                ).strip()
                if not mat_display:
                    mat_display = "material"
                if debug_lines is not None:
                    summary = debug_summary_entries.setdefault(
                        operation_name,
                        {
                            "operation": operation_name,
                            "material": mat_display,
                            "qty": 0,
                            "total_minutes": 0.0,
                            "toolchange_total": 0.0,
                            "sfm_sum": 0.0,
                            "sfm_count": 0,
                            "ipr_sum": 0.0,
                            "ipr_count": 0,
                            "rpm_sum": 0.0,
                            "rpm_count": 0,
                            "ipm_sum": 0.0,
                            "ipm_count": 0,
                            "rpm_min": None,
                            "rpm_max": None,
                            "ipm_min": None,
                            "ipm_max": None,
                            "ipr_min": None,
                            "ipr_max": None,
                            "ipr_effective_min": None,
                            "ipr_effective_max": None,
                            "bins": {},
                            "diameter_weight_sum": 0.0,
                            "diameter_qty_sum": 0,
                            "diam_min": None,
                            "diam_max": None,
                            "depth_weight_sum": 0.0,
                            "depth_qty_sum": 0,
                            "depth_min": None,
                            "depth_max": None,
                            "peck_sum": 0.0,
                            "peck_count": 0,
                            "dwell_sum": 0.0,
                            "dwell_count": 0,
                            "index_sum": 0.0,
                            "index_count": 0,
                        },
                    )
                    if chosen_material_label:
                        summary["material"] = chosen_material_label
                    elif mat_display and (
                        not summary.get("material")
                        or summary.get("material") == "material"
                    ):
                        summary["material"] = mat_display
                    minutes_val = to_float(minutes_per)
                    minutes_per_hole = minutes_val if minutes_val is not None else float(minutes)
                    summary["qty"] += qty_for_debug
                    summary["total_minutes"] += minutes_per_hole * qty_for_debug
                    summary["toolchange_total"] += toolchange_added
                    sfm_float = to_float(sfm_val)
                    if sfm_float is not None and math.isfinite(sfm_float):
                        summary["sfm_sum"] += sfm_float * qty_for_debug
                        summary["sfm_count"] += qty_for_debug
                    rpm_float = to_float(rpm_val)
                    ipr_float = to_float(ipr_val)
                    ipm_float = to_float(ipm_val)
                    if (
                        (ipm_float is None or not math.isfinite(ipm_float))
                        and rpm_float is not None
                        and math.isfinite(rpm_float)
                        and ipr_float is not None
                        and math.isfinite(ipr_float)
                    ):
                        ipm_float = float(rpm_float) * float(ipr_float)
                    ipr_effective_float: float | None = None
                    if (
                        rpm_float is not None
                        and math.isfinite(rpm_float)
                        and ipm_float is not None
                        and math.isfinite(ipm_float)
                        and abs(float(rpm_float)) > 1e-9
                    ):
                        ipr_effective_float = float(ipm_float) / float(rpm_float)
                    elif ipr_float is not None and math.isfinite(ipr_float):
                        ipr_effective_float = float(ipr_float)
                    if debug_payload is not None:
                        if rpm_float is not None and math.isfinite(rpm_float):
                            debug_payload["rpm"] = float(rpm_float)
                        if ipm_float is not None and math.isfinite(ipm_float):
                            debug_payload["ipm"] = float(ipm_float)
                        if ipr_effective_float is not None and math.isfinite(ipr_effective_float):
                            debug_payload["ipr_effective"] = float(ipr_effective_float)
                            debug_payload["ipr"] = float(ipr_effective_float)
                        elif ipr_float is not None and math.isfinite(ipr_float):
                            debug_payload["ipr"] = float(ipr_float)
                    if rpm_float is not None and math.isfinite(rpm_float):
                        summary["rpm_sum"] += rpm_float * qty_for_debug
                        summary["rpm_count"] += qty_for_debug
                        _update_range(summary, "rpm_min", "rpm_max", rpm_float)
                    if ipm_float is not None and math.isfinite(ipm_float):
                        summary["ipm_sum"] += ipm_float * qty_for_debug
                        summary["ipm_count"] += qty_for_debug
                        _update_range(summary, "ipm_min", "ipm_max", ipm_float)
                    if ipr_effective_float is not None and math.isfinite(ipr_effective_float):
                        summary["ipr_sum"] += ipr_effective_float * qty_for_debug
                        summary["ipr_count"] += qty_for_debug
                        _update_range(summary, "ipr_min", "ipr_max", ipr_effective_float)
                        _update_range(
                            summary,
                            "ipr_effective_min",
                            "ipr_effective_max",
                            ipr_effective_float,
                        )
                    bins = summary.setdefault("bins", {})
                    bin_key = f"{float(tool_dia_in):.4f}"
                    bin_summary = bins.setdefault(
                        bin_key,
                        {
                            "diameter_in": float(tool_dia_in),
                            "qty": 0,
                            "rpm_min": None,
                            "rpm_max": None,
                            "ipm_min": None,
                            "ipm_max": None,
                            "ipr_min": None,
                            "ipr_max": None,
                            "ipr_effective_min": None,
                            "ipr_effective_max": None,
                        },
                    )
                    bin_summary["qty"] += qty_for_debug
                    if rpm_float is not None and math.isfinite(rpm_float):
                        _update_range(bin_summary, "rpm_min", "rpm_max", rpm_float)
                    if ipm_float is not None and math.isfinite(ipm_float):
                        _update_range(bin_summary, "ipm_min", "ipm_max", ipm_float)
                    if ipr_effective_float is not None and math.isfinite(ipr_effective_float):
                        _update_range(bin_summary, "ipr_min", "ipr_max", ipr_effective_float)
                        _update_range(
                            bin_summary,
                            "ipr_effective_min",
                            "ipr_effective_max",
                            ipr_effective_float,
                        )
                    summary["diameter_weight_sum"] += float(tool_dia_in) * qty_for_debug
                    summary["diameter_qty_sum"] += qty_for_debug
                    diam_min = summary.get("diam_min")
                    diam_max = summary.get("diam_max")
                    if diam_min is None or float(tool_dia_in) < diam_min:
                        summary["diam_min"] = float(tool_dia_in)
                    if diam_max is None or float(tool_dia_in) > diam_max:
                        summary["diam_max"] = float(tool_dia_in)
                    depth_float = to_float(depth_val)
                    if depth_float is None:
                        try:
                            depth_float = float(depth_in)
                        except Exception:
                            depth_float = None
                    if depth_float is not None:
                        summary["depth_weight_sum"] += float(depth_float) * qty_for_debug
                        summary["depth_qty_sum"] += qty_for_debug
                        depth_min = summary.get("depth_min")
                        depth_max = summary.get("depth_max")
                        if depth_min is None or float(depth_float) < depth_min:
                            summary["depth_min"] = float(depth_float)
                        if depth_max is None or float(depth_float) > depth_max:
                            summary["depth_max"] = float(depth_float)
                    peck_rate = to_float(
                        overhead_for_calc.peck_penalty_min_per_in_depth
                    )
                    if depth_float is not None and peck_rate is not None and peck_rate > 0:
                        peck_total = float(peck_rate) * float(depth_float)
                        if math.isfinite(peck_total) and peck_total > 0:
                            summary["peck_sum"] += peck_total * qty_for_debug
                            summary["peck_count"] += qty_for_debug
                    dwell_val_float = to_float(overhead_for_calc.dwell_min)
                    if dwell_val_float is not None and dwell_val_float > 0:
                        summary["dwell_sum"] += float(dwell_val_float) * qty_for_debug
                        summary["dwell_count"] += qty_for_debug
                    index_min_val = None
                    if debug_payload is not None:
                        index_min_val = to_float(
                            debug_payload.get("index_min")
                        )
                    if index_min_val is None:
                        index_sec_val = to_float(
                            overhead_for_calc.index_sec_per_hole
                        )
                        if index_sec_val is not None and index_sec_val > 0:
                            index_min_val = float(index_sec_val) / 60.0
                    if index_min_val is not None and index_min_val > 0:
                        summary["index_sum"] += float(index_min_val) * qty_for_debug
                        summary["index_count"] += qty_for_debug
                    if not summary.get("material"):
                        summary["material"] = "material"
                qty_int = qty_for_debug
        hole_count_for_clamp = total_holes
        if hole_count_for_clamp <= 0 and fallback_counts:
            hole_count_for_clamp = sum(
                max(0, int(qty)) for qty in fallback_counts.values() if qty
            )

        clamp_ratio = 1.0
        if total_min > 0 and hole_count_for_clamp > 0:
            uncapped_minutes = total_min
            clamped_hours = _apply_drill_minutes_clamp(
                total_min / 60.0,
                hole_count_for_clamp,
                material_group=material_label,
                depth_in=depth_for_bounds,
            )
            total_min = clamped_hours * 60.0
            if uncapped_minutes > 1e-9:
                clamp_ratio = total_min / uncapped_minutes

        if clamp_ratio != 1.0 and debug_summary_entries:
            for summary in debug_summary_entries.values():
                minutes_total = summary.get("total_minutes", 0.0) or 0.0
                summary["total_minutes"] = minutes_total * clamp_ratio

        if debug_lines is not None and debug_summary_entries:
            for op_key, summary in sorted(debug_summary_entries.items()):
                qty_total = summary.get("qty", 0)
                if qty_total <= 0:
                    continue
                minutes_total = summary.get("total_minutes", 0.0) or 0.0
                minutes_avg = minutes_total / qty_total if qty_total else 0.0
                toolchange_total = summary.get("toolchange_total", 0.0) or 0.0
                total_hours = (minutes_total + toolchange_total) / 60.0

                def _avg_value(sum_key: str, count_key: str) -> float | None:
                    total = summary.get(sum_key, 0.0) or 0.0
                    count = summary.get(count_key, 0) or 0
                    if count <= 0:
                        return None
                    return float(total) / float(count)

                def _format_avg(value: float | None, fmt: str) -> str:
                    coerced = _coerce_float_or_none(value)
                    if coerced is None or not math.isfinite(float(coerced)):
                        return "-"
                    return fmt.format(float(coerced))

                def _format_range(
                    min_val: float | None,
                    max_val: float | None,
                    fmt: str,
                    *,
                    tolerance: float = 0.0,
                ) -> str:
                    min_f = _coerce_float_or_none(min_val)
                    max_f = _coerce_float_or_none(max_val)
                    if min_f is None and max_f is None:
                        return "-"
                    if min_f is None:
                        min_f = max_f
                    if max_f is None:
                        max_f = min_f
                    if min_f is None or max_f is None:
                        return "-"
                    source_min = min_val if min_val is not None else max_val
                    source_max = max_val if max_val is not None else min_val
                    if source_min is None or source_max is None:
                        return "-"
                    try:
                        min_f = float(source_min)
                        max_f = float(source_max)
                    except (TypeError, ValueError):
                        return "-"
                    if not math.isfinite(min_float) or not math.isfinite(max_float):
                        return "-"
                    if tolerance and abs(max_float - min_float) <= tolerance:
                        return fmt.format(max_float)
                    if abs(max_float - min_float) <= 1e-12:
                        return fmt.format(max_float)
                    return f"{fmt.format(min_float)}–{fmt.format(max_float)}"

                sfm_avg = _avg_value("sfm_sum", "sfm_count")
                rpm_avg = _avg_value("rpm_sum", "rpm_count")
                ipm_avg = _avg_value("ipm_sum", "ipm_count")
                summary["rpm"] = rpm_avg
                summary["ipm"] = ipm_avg
                summary["minutes_per_hole"] = minutes_avg
                sfm_text = _format_avg(sfm_avg, "{:.0f}")
                ipr_min_val = summary.get("ipr_effective_min")
                if ipr_min_val is None:
                    ipr_min_val = summary.get("ipr_min")
                ipr_max_val = summary.get("ipr_effective_max")
                if ipr_max_val is None:
                    ipr_max_val = summary.get("ipr_max")
                ipr_text = _format_range(ipr_min_val, ipr_max_val, "{:.4f}", tolerance=5e-5)
                rpm_text = _format_range(summary.get("rpm_min"), summary.get("rpm_max"), "{:.0f}", tolerance=0.5)
                ipm_text = _format_range(summary.get("ipm_min"), summary.get("ipm_max"), "{:.1f}", tolerance=0.05)

                diam_qty = summary.get("diameter_qty_sum", 0) or 0
                dia_segment = "Ø -"
                if diam_qty > 0:
                    diam_sum = summary.get("diameter_weight_sum", 0.0) or 0.0
                    avg_dia = diam_sum / diam_qty if diam_qty else 0.0
                    diam_min = summary.get("diam_min")
                    diam_max = summary.get("diam_max")
                    dia_range_text = _format_range(diam_min, diam_max, "{:.3f}\"", tolerance=5e-4)
                    if dia_range_text == "-":
                        dia_range_text = f"{float(avg_dia):.3f}\""
                    dia_segment = f"Ø {dia_range_text}"

                depth_qty = summary.get("depth_qty_sum", 0) or 0
                depth_text = "-"
                if depth_qty > 0:
                    depth_sum = summary.get("depth_weight_sum", 0.0) or 0.0
                    avg_depth = depth_sum / depth_qty if depth_qty else 0.0
                    depth_min = summary.get("depth_min")
                    depth_max = summary.get("depth_max")
                    depth_range_text = _format_range(
                        depth_min, depth_max, "{:.2f}", tolerance=5e-3
                    )
                    if depth_range_text == "-":
                        depth_range_text = f"{float(avg_depth):.2f}"
                    depth_text = depth_range_text

                peck_avg = _avg_value("peck_sum", "peck_count")
                dwell_avg = _avg_value("dwell_sum", "dwell_count")
                index_avg = _avg_value("index_sum", "index_count")
                overhead_bits: list[str] = []
                if dwell_avg and math.isfinite(dwell_avg) and dwell_avg > 0:
                    overhead_bits.append(f"dwell {dwell_avg:.2f} min/hole")
                if index_avg and math.isfinite(index_avg) and index_avg > 0:
                    overhead_bits.append(f"index {index_avg:.2f} min/hole")
                if toolchange_total and math.isfinite(toolchange_total) and toolchange_total > 0:
                    overhead_bits.append(f"toolchange {toolchange_total:.2f} min")

                op_display = str(summary.get("operation") or "drill").title()
                mat_display = str(summary.get("material") or "material").strip()
                if not mat_display:
                    mat_display = "material"
                summary["material"] = mat_display

                depth_segment = "depth/hole -"
                if depth_text != "-":
                    depth_segment = f"depth/hole {depth_text} in"

                peck_text = "-"
                if peck_avg and math.isfinite(peck_avg) and peck_avg > 0:
                    peck_text = f"{peck_avg:.2f} min/hole"

                toolchange_text = f"{toolchange_total:.2f} min"

                index_text = "-"
                if minutes_avg and math.isfinite(minutes_avg) and minutes_avg > 0:
                    index_text = f"{minutes_avg * 60.0:.1f} s/hole"

                line_parts = [
                    "Drill calc → ",
                    f"op={op_display}, mat={mat_display}, ",
                    f"SFM={sfm_text}, IPR={ipr_text}; ",
                    f"RPM {rpm_text} IPM {ipm_text}; ",
                    f"{dia_segment}; {depth_segment}; ",
                    f"holes {qty_total}; ",
                    f"index {index_text}; ",
                    f"peck {peck_text}; ",
                    f"toolchange {toolchange_text}; ",
                ]
                if overhead_bits:
                    line_parts.append("overhead: " + ", ".join(overhead_bits) + "; ")
                line_parts.append(f"total hr {total_hours:.2f}.")
                debug_lines.append("".join(line_parts))
                if debug_summary is not None:
                    debug_summary[op_key] = dict(summary)
        if missing_row_messages and warnings is not None:
            for op_display, material_display, dia_val in sorted(missing_row_messages):
                dia_text = f"{dia_val:.3f}".rstrip("0").rstrip(".")
                warning_text = (
                    f"No speeds/feeds row for {op_display}/"
                    f"{material_display.lower()} {dia_text} in — using fallback"
                )
                if warning_text not in warnings:
                    warnings.append(warning_text)
        total_minutes_with_toolchange = total_min + total_toolchange_min

        if debug is not None and total_holes > 0:
            try:
                avg_dia_in = float(avg_dia_in)
            except Exception:
                avg_dia_in = 0.0
            debug.update(
                {
                    "thickness_in": float(thickness_in or 0.0),
                    "avg_dia_in": float(avg_dia_in),
                    "sfm": None,
                    "ipr": None,
                    "rpm": None,
                    "ipm": None,
                    "min_per_hole": (float(total_min) / float(total_holes)) if total_holes else 0.0,
                    "hole_count": int(total_holes),
                }
            )
        if total_minutes_with_toolchange > 0:
            return total_minutes_with_toolchange / 60.0

    thickness_for_fallback_mm = float(thickness_in or 0.0) * 25.4
    if thickness_for_fallback_mm <= 0:
        depth_candidates = [depth for _, _, depth in group_specs if depth and depth > 0]
        if depth_candidates:
            thickness_for_fallback_mm = max(depth_candidates) * 25.4

    if not fallback_counts or thickness_for_fallback_mm <= 0:
        return 0.0

    def sec_per_hole(d_mm: float) -> float:
        if d_mm <= 3.5:
            return 10.0
        if d_mm <= 6.0:
            return 14.0
        if d_mm <= 10.0:
            return 18.0
        if d_mm <= 13.0:
            return 22.0
        if d_mm <= 20.0:
            return 30.0
        if d_mm <= 32.0:
            return 45.0
        return 60.0

    mfac = 0.8 if "alum" in mat else (1.15 if "stainless" in mat else 1.0)
    tfac = max(0.7, min(2.0, thickness_for_fallback_mm / 6.35))
    toolchange_s = 15.0

    total_sec = 0.0
    holes_fallback = 0
    weighted_dia_in = 0.0
    for d, qty in fallback_counts.items():
        if qty is None:
            continue
        try:
            qty_int = int(qty)
        except Exception:
            continue
        if qty_int <= 0:
            continue
        holes_fallback += qty_int
        per = sec_per_hole(float(d)) * mfac * tfac
        total_sec += qty_int * per
        total_sec += toolchange_s
        # aggregate counts and weighted diameter
        # total_qty was not previously initialized; use holes_fallback as the count
        weighted_dia_in += (float(d) / 25.4) * int(qty)

    if debug is not None and holes_fallback > 0:
        avg_dia_in = weighted_dia_in / holes_fallback if holes_fallback else 0.0
        debug.update(
            {
            "thickness_in": float(thickness_in or 0.0),
            "avg_dia_in": float(avg_dia_in),
            "sfm": None,
            "ipr": None,
            "rpm": None,
            "ipm": None,
            "min_per_hole": (total_sec / 60.0) / holes_fallback if holes_fallback else None,
            "hole_count": int(holes_fallback),
            }
        )
    elif debug is not None:
        debug.update(
            {
                "thickness_in": float(thickness_in or 0.0),
                "avg_dia_in": 0.0,
                "sfm": None,
                "ipr": None,
                "rpm": None,
                "ipm": None,
                "min_per_hole": None,
                "hole_count": 0,
            }
        )

    hours = total_sec / 3600.0
    depth_for_bounds = None
    if thickness_for_fallback_mm and thickness_for_fallback_mm > 0:
        depth_for_bounds = float(thickness_for_fallback_mm) / 25.4
    return _apply_drill_minutes_clamp(
        hours,
        holes_fallback,
        material_group=material_label,
        depth_in=depth_for_bounds,
    )


def _drilling_floor_hours(hole_count: int) -> float:
    min_sec_per_hole = 9.0
    return max((float(hole_count or 0) * min_sec_per_hole) / 3600.0, 0.0)


def validate_drilling_reasonableness(hole_count: int, drill_hr_after_overrides: float) -> tuple[bool, str]:
    floor_hr = _drilling_floor_hours(hole_count)
    ok = drill_hr_after_overrides >= floor_hr
    msg = f"drilling hours ({drill_hr_after_overrides:.2f} h) below floor ({floor_hr:.2f} h) for {hole_count} holes"
    return ok, msg


def validate_process_floor_hours(
    label: str,
    process_hr_after_overrides: float,
    floor_hr: float,
    context: str = "",
) -> tuple[bool, str]:
    floor_val = max(0.0, float(floor_hr or 0.0))
    if floor_val <= 0.0:
        return True, ""
    hr_val = max(0.0, float(process_hr_after_overrides or 0.0))
    ok = hr_val + 1e-6 >= floor_val
    friendly = label.replace("_", " ").strip().title()
    message = f"{friendly} hours ({hr_val:.2f} h) below floor ({floor_val:.2f} h)"
    if context:
        message = f"{message} {context}"
    return ok, message


def _estimate_programming_hours_auto(
    geo: Mapping[str, Any] | None,
    process_meta: Mapping[str, Mapping[str, Any] | Any] | None,
    params: Mapping[str, Any] | None,
    *,
    setups_hint: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Heuristic programming-hour estimate driven by process + geometry context."""

    params = params or {}
    geo_outer = dict(geo or {})
    inner_geo_raw = geo_outer.get("geo")
    inner_geo = dict(inner_geo_raw) if isinstance(inner_geo_raw, Mapping) else {}

    def _geo_lookup(*keys: str) -> Any:
        for key in keys:
            if key in geo_outer and geo_outer[key] is not None:
                return geo_outer[key]
            if key in inner_geo and inner_geo[key] is not None:
                return inner_geo[key]
        return None

    def _float(value: Any, default: float = 0.0) -> float:
        num = _coerce_float_or_none(value)
        return float(num) if num is not None else float(default)

    def _int(value: Any, default: int = 0) -> int:
        num = _coerce_float_or_none(value)
        if num is None:
            return default
        try:
            return int(round(num))
        except Exception:
            return default

    faces = max(4.0, _float(_geo_lookup("GEO__Face_Count", "face_count"), 6.0))
    unique_normals = max(1.0, _float(_geo_lookup("GEO__Unique_Normal_Count", "unique_normals"), faces / 2.0))
    hole_count = _int(_geo_lookup("hole_count", "GEO__Hole_Count"), 0)
    if hole_count <= 0:
        hole_list = _geo_lookup("hole_diams_mm", "holes")
        if isinstance(hole_list, Sequence) and not isinstance(hole_list, (str, bytes)):
            hole_count = len(hole_list)
    pocket_count = max(0, _int(_geo_lookup("pocket_count", "GEO__Pocket_Count"), 0))
    slot_count = max(0, _int(_geo_lookup("slot_count", "GEO__Slot_Count"), 0))

    complexity_raw = _float(_geo_lookup("GEO_Complexity_0to100", "complexity_score"), 25.0)
    complexity = max(0.0, min(1.0, complexity_raw / 100.0))
    max_dim_mm = _float(_geo_lookup("GEO__MaxDim_mm", "max_dim_mm", "bounding_box_max_mm"), 100.0)
    thin_wall = bool(_geo_lookup("GEO__ThinWall_Present", "thin_wall_present", "thin_walls"))

    setups_val = None
    if setups_hint is not None:
        try:
            setups_val = float(setups_hint)
        except Exception:
            setups_val = None
    if setups_val is None:
        setups_val = _float(
            _geo_lookup(
                "setups_hint",
                "baseline_setups",
                "Milling_Setups",
                "setups",
                "milling_setups",
            ),
            0.0,
        )
    if setups_val <= 0.0:
        if unique_normals <= 4.0:
            setups_val = 1.0
        elif unique_normals <= 6.0:
            setups_val = 2.0
        else:
            setups_val = 3.0
    setups = max(1, min(int(round(setups_val)), 5))

    def _norm_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    normalized_meta: dict[str, Mapping[str, Any]] = {}
    if isinstance(process_meta, Mapping):
        for key, meta in process_meta.items():
            if isinstance(meta, Mapping):
                normalized_meta[_norm_key(key)] = meta

    def _hours_for(*keys: str) -> float:
        total = 0.0
        for key in keys:
            meta = normalized_meta.get(_norm_key(key))
            if not isinstance(meta, Mapping):
                continue
            try:
                total += float(meta.get("hr", 0.0) or 0.0)
            except Exception:
                continue
        return total

    milling_hr = _hours_for("milling")
    turning_hr = _hours_for("turning")
    wire_edm_hr = _hours_for("wire_edm", "wireedm")
    sinker_hr = _hours_for("sinker_edm", "ram_edm")
    grinding_hr = _hours_for("grinding", "surface_grind", "jig_grind", "odid_grind")
    drilling_like_hr = _hours_for("drilling", "tapping", "counterbore", "countersink")
    saw_hr = _hours_for("saw_waterjet", "sawing", "waterjet")
    finishing_hr = _hours_for("finishing_deburr", "deburr") + _hours_for("lapping_honing", "lapping", "honing")
    inspection_hr = _hours_for("inspection")

    total_cutting_hr = milling_hr + turning_hr + wire_edm_hr + sinker_hr + grinding_hr + drilling_like_hr + saw_hr
    special_processes = sum(
        1 for hr in (turning_hr, wire_edm_hr, sinker_hr, grinding_hr) if hr > 1e-3
    )

    contributions: dict[str, float] = {}
    contributions["setups"] = 0.25 + 0.35 * setups
    feature_term = max(0.0, (faces - 6.0) * 0.02) + hole_count * 0.015 + pocket_count * 0.03 + slot_count * 0.02
    if feature_term:
        contributions["features"] = feature_term
    complexity_term = complexity * 0.9 + max(0.0, (unique_normals - 3.0) * 0.05)
    if complexity_term:
        contributions["complexity"] = complexity_term
    size_term = min(0.5, max_dim_mm / 400.0)
    if size_term:
        contributions["size"] = size_term
    machining_term = min(2.5, math.log1p(max(0.0, total_cutting_hr) * 60.0) / 3.2)
    if machining_term:
        contributions["machining"] = machining_term
    finishing_term = min(0.5, finishing_hr * 0.2)
    if finishing_term:
        contributions["finishing"] = finishing_term
    inspection_term = min(0.3, inspection_hr * 0.15)
    if inspection_term:
        contributions["inspection"] = inspection_term
    if thin_wall:
        contributions["thin_wall"] = 0.2
    if special_processes:
        contributions["multi_process"] = 0.18 * special_processes

    raw_total = sum(contributions.values())

    total = raw_total
    prog_cap = _coerce_float_or_none(params.get("ProgCapHr")) if isinstance(params, Mapping) else None
    prog_simple_dim = _coerce_float_or_none(params.get("ProgSimpleDim_mm")) if isinstance(params, Mapping) else None
    if prog_cap and prog_simple_dim and max_dim_mm and max_dim_mm <= prog_simple_dim and setups <= 2:
        total = min(total, float(prog_cap))
    prog_ratio = _coerce_float_or_none(params.get("ProgMaxToMillingRatio")) if isinstance(params, Mapping) else None
    if prog_ratio and milling_hr > 0:
        total = min(total, float(prog_ratio) * milling_hr)

    floor_hr = _coerce_float_or_none(params.get("ProgAutoFloorHr")) if isinstance(params, Mapping) else None
    if floor_hr is None or floor_hr <= 0:
        floor_hr = 0.35
    total = max(float(floor_hr), total)

    components = {key: round(val, 4) for key, val in contributions.items() if val}
    components["_raw_total"] = round(raw_total, 4)
    if abs(total - raw_total) > 1e-6:
        components["_capped_total"] = round(total, 4)

    return float(total), components


def validate_quote_before_pricing(
    geo: Mapping[str, Any] | None,
    process_costs: dict[str, float],
    pass_through: dict[str, Any],
    process_hours: dict[str, float] | None = None,
) -> None:
    issues: list[str] = []
    geo_ctx: Mapping[str, Any] = geo if isinstance(geo, Mapping) else {}
    has_legacy_buckets = any(key in process_costs for key in ("drilling", "milling"))
    if has_legacy_buckets:
        hole_cost = sum(float(process_costs.get(k, 0.0)) for k in ("drilling", "milling"))
        if geo_ctx.get("hole_diams_mm") and hole_cost < 50:
            issues.append("Unusually low machining time for number of holes.")
    material_cost = float(pass_through.get("Material", 0.0) or 0.0)
    if material_cost < 5.0:
        inner_geo_candidate = geo_ctx.get("geo") if isinstance(geo_ctx, Mapping) else None
        inner_geo: Mapping[str, Any] = (
            inner_geo_candidate if isinstance(inner_geo_candidate, Mapping) else {}
        )

        def _positive(value: Any) -> bool:
            num = _coerce_float_or_none(value)
            return bool(num and num > 0)

        thickness_candidates = [
            geo_ctx.get("thickness_mm"),
            geo_ctx.get("thickness_in"),
            geo_ctx.get("GEO-03_Height_mm"),
            geo_ctx.get("GEO__Stock_Thickness_mm"),
            inner_geo.get("thickness_mm") if isinstance(inner_geo, Mapping) else None,
            inner_geo.get("stock_thickness_mm") if isinstance(inner_geo, Mapping) else None,
        ]
        has_thickness_hint = any(_positive(val) for val in thickness_candidates)

        def _mass_hint(ctx: Mapping[str, Any] | None) -> bool:
            if not isinstance(ctx, Mapping):
                return False
            for key in ("net_mass_kg", "net_mass_kg_est", "mass_kg", "net_mass_g"):
                if key not in ctx:
                    continue
                num = _coerce_float_or_none(ctx.get(key))
                if num and num > 0:
                    return True
            return False

        has_mass_hint = _mass_hint(geo_ctx) or _mass_hint(inner_geo)
        has_material = bool(
            str(geo_ctx.get("material") or "").strip()
            or str(inner_geo.get("material") or "").strip()
        )

        if not (has_thickness_hint or has_mass_hint or has_material):
            issues.append("Material cost is near zero; check material & thickness.")
    try:
        hole_count_val = int(float(geo_ctx.get("hole_count", 0)))
    except Exception:
        hole_count_val = len(geo_ctx.get("hole_diams_mm") or [])
    if hole_count_val <= 0:
        hole_count_val = len(geo_ctx.get("hole_diams_mm") or [])
    if issues:
        allow = str(os.getenv("QUOTE_ALLOW_LOW_MATERIAL", "")).strip().lower()
        if allow in {"1", "true", "yes", "on"}:
            issues.clear()
        else:
            raise ValueError("Quote blocked:\n- " + "\n- ".join(issues))


# pyright: ignore[reportGeneralTypeIssues]
def compute_quote_from_df(df: pd.DataFrame,
                          params: Dict[str, Any] | None = None,
                          rates: Dict[str, float] | None = None,
                          *,
                          default_params: Dict[str, Any] | None = None,
                          default_rates: Dict[str, float] | None = None,
                          default_material_display: str | None = None,
                          material_vendor_csv: str | None = None,
                          llm_enabled: bool = True,
                          llm_model_path: str | None = None,
                          llm_client: LLMClient | None = None,
                          geo: dict[str, Any] | None = None,
                          ui_vars: dict[str, Any] | None = None,
                          quote_state: QuoteState | None = None,
                          reuse_suggestions: bool = False,
                          llm_suggest: Any | None = None,
                          pricing: PricingEngine | None = None) -> Dict[str, Any]:
    """
    Estimator that consumes variables from the sheet (Item, Example Values / Options, Data Type / Input Method).

    Steps:
      1) Tolerant lookup (regex on 'Item') for numbers/percents/strings
      2) Aggregate hours ï¿½ rates per process (milling/turning/EDM/grinding/ï¿½)
      3) Geometry-derived estimates (optional)
      4) NRE, vendors, packaging/shipping/insurance
      5) Overhead ? G&A ? Contingency ? Expedite ? Margin
      6) Return total and a transparent breakdown
    """
    # ---- merge configs (easy to edit) ---------------------------------------
    # Default pricing source; updated to 'planner' later if planner path is used
    pricing_source = "legacy"

    params_defaults = default_params if default_params is not None else QuoteConfiguration().default_params
    rates_defaults = default_rates if default_rates is not None else PricingRegistry().default_rates
    if not isinstance(default_material_display, str) or not default_material_display.strip():
        default_material_display = DEFAULT_MATERIAL_DISPLAY
    params = {**params_defaults, **(params or {})}
    rates = {**rates_defaults, **(rates or {})}
    rates.setdefault("DrillingRate", rates.get("MillingRate", 0.0))
    rates.setdefault("ProjectManagementRate", rates.get("EngineerRate", 0.0))
    rates.setdefault("ToolmakerSupportRate", rates.get("FixtureBuildRate", rates.get("EngineerRate", 0.0)))
    if llm_model_path is None:
        llm_model_path = str(params.get("LLMModelPath", "") or "")
    else:
        params["LLMModelPath"] = llm_model_path
    if ui_vars is None:
        try:
            ui_vars = {
                str(row["Item"]): row["Example Values / Options"]
                for _, row in df.iterrows()
            }
        except Exception:
            ui_vars = {}
    else:
        ui_vars = dict(ui_vars)
    if quote_state is None:
        quote_state = QuoteState()
    pricing_engine = pricing or _DEFAULT_PRICING_ENGINE
    quote_state.ui_vars = dict(ui_vars)
    quote_state.rates = dict(rates)
    geo_context = dict(geo or {})
    inner_geo_raw = geo_context.get("geo")
    inner_geo = dict(inner_geo_raw) if isinstance(inner_geo_raw, dict) else {}
    geo_context["geo"] = inner_geo

    plate_length_in_val: float | None = None
    plate_width_in_val: float | None = None
    plate_thickness_in_val: float | None = None
    removal_mass_g_est: float | None = None
    hole_scrap_frac_est: float = 0.0
    hole_scrap_clamped_val: float = 0.0
    material_selection: dict[str, Any] = {}

    def _int_from(value: Any) -> int:
        num = _coerce_float_or_none(value)
        if num is None:
            return 0
        try:
            return int(round(num))
        except Exception:
            return 0

    def _max_int(*values: Any) -> int:
        best = 0
        for value in values:
            num = _coerce_float_or_none(value)
            if num is None:
                continue
            if num > best:
                try:
                    best = int(round(num))
                except Exception:
                    continue
        return max(0, best)

    hole_count_canonical = _int_from(geo_context.get("hole_count"))
    if hole_count_canonical <= 0 and inner_geo:
        hole_count_canonical = _int_from(inner_geo.get("hole_count"))
    hole_count_canonical = max(0, hole_count_canonical)
    geo_context["hole_count"] = hole_count_canonical
    if inner_geo:
        inner_geo["hole_count"] = hole_count_canonical

    feature_counts_geo_raw = geo_context.get("feature_counts")
    if isinstance(feature_counts_geo_raw, dict):
        feature_counts_geo = feature_counts_geo_raw
    else:
        feature_counts_geo = {}
        if feature_counts_geo_raw:
            try:
                feature_counts_geo.update(dict(feature_counts_geo_raw))
            except Exception:
                feature_counts_geo = {}
        geo_context["feature_counts"] = feature_counts_geo

    MM_PER_IN = 25.4

    def _read_editor_num(df, label: str):
        """Read a numeric value from the Quote Editor table by label; returns float|None."""

        try:
            mask = df["Item"].astype(str).str.fullmatch(label, case=False, na=False)
            val = str(df.loc[mask, "Example Values / Options"].iloc[0]).strip()
            return float(val)
        except Exception:
            return None

    # --- thickness for LLM --------------------------------------------------------
    thickness_for_llm: float | None = None
    geo_for_thickness = geo_context
    try:
        t_geo = geo_for_thickness.get("thickness_mm")
        if isinstance(t_geo, dict):
            thickness_for_llm = float(t_geo.get("value")) if t_geo.get("value") is not None else None
        elif t_geo is not None:
            thickness_for_llm = float(t_geo)
    except Exception:
        thickness_for_llm = None

    if thickness_for_llm is None:
        try:
            t_geo_raw = geo_for_thickness.get("thickness")
            if t_geo_raw is not None:
                thickness_for_llm = float(t_geo_raw)
        except Exception:
            thickness_for_llm = None

    # Fallback: Quote Editor "Thickness (in)"
    if thickness_for_llm is None:
        t_in = _read_editor_num(df, r"Thickness \(in\)")
        if t_in is not None:
            thickness_for_llm = t_in * MM_PER_IN

    # Last-ditch: Z dimension of bbox if present (treat as thickness for plates)
    bbox_for_llm: list[float] | None = None
    try:
        bb = geo_for_thickness.get("bbox_mm")
        if isinstance(bb, (list, tuple)) and len(bb) == 3:
            bbox_for_llm = [float(bb[0]), float(bb[1]), float(bb[2])]
            if thickness_for_llm is None:
                thickness_for_llm = float(bb[2])
    except Exception:
        bbox_for_llm = None

    thickness_for_llm = None if thickness_for_llm is None else float(thickness_for_llm)
    chart_ops_geo = geo_context.get("chart_ops") if isinstance(geo_context.get("chart_ops"), list) else None
    chart_reconcile_geo = geo_context.get("chart_reconcile") if isinstance(geo_context.get("chart_reconcile"), dict) else None
    chart_source_geo = geo_context.get("chart_source") if isinstance(geo_context.get("chart_source"), str) else None
    is_plate_2d = str(geo_context.get("kind", "")).lower() == "2d"
    if is_plate_2d:
        require_plate_inputs(geo_context, ui_vars)

    chart_tap_qty = _int_from(chart_reconcile_geo.get("tap_qty")) if chart_reconcile_geo else 0
    chart_cbore_qty = _int_from(chart_reconcile_geo.get("cbore_qty")) if chart_reconcile_geo else 0
    chart_csk_qty = _int_from(chart_reconcile_geo.get("csk_qty")) if chart_reconcile_geo else 0
    tap_qty_override_geo = _coerce_float_or_none(ui_vars.get("Tap Qty (LLM/GEO)"))
    tap_qty_geo = _max_int(
        chart_tap_qty,
        feature_counts_geo.get("tap_qty") if isinstance(feature_counts_geo, dict) else None,
        inner_geo.get("tap_qty") if inner_geo else None,
        geo_context.get("tap_qty"),
    )
    if tap_qty_override_geo and tap_qty_override_geo > 0:
        tap_qty_geo = int(round(tap_qty_override_geo))
    cbore_qty_geo = _max_int(
        chart_cbore_qty,
        feature_counts_geo.get("cbore_qty") if isinstance(feature_counts_geo, dict) else None,
        inner_geo.get("cbore_qty") if inner_geo else None,
        geo_context.get("cbore_qty"),
    )
    csk_qty_geo = _max_int(
        chart_csk_qty,
        feature_counts_geo.get("csk_qty") if isinstance(feature_counts_geo, dict) else None,
        inner_geo.get("csk_qty") if inner_geo else None,
        geo_context.get("csk_qty"),
    )
    tap_qty_seed = max(chart_tap_qty, tap_qty_geo)
    cbore_qty_seed = max(chart_cbore_qty, cbore_qty_geo)
    csk_qty_seed = max(chart_csk_qty, csk_qty_geo)
    geo_context["tap_qty"] = tap_qty_geo
    inner_geo["tap_qty"] = tap_qty_geo
    geo_context["cbore_qty"] = cbore_qty_geo
    inner_geo["cbore_qty"] = cbore_qty_geo
    geo_context["csk_qty"] = csk_qty_geo
    inner_geo["csk_qty"] = csk_qty_geo
    if isinstance(feature_counts_geo, dict):
        if tap_qty_geo:
            feature_counts_geo["tap_qty"] = tap_qty_geo
        if cbore_qty_geo:
            feature_counts_geo["cbore_qty"] = cbore_qty_geo
        if csk_qty_geo:
            feature_counts_geo["csk_qty"] = csk_qty_geo
    # ---- sheet views ---------------------------------------------------------
    items = df["Item"].astype(str)
    vals  = df["Example Values / Options"]
    dtt   = df["Data Type / Input Method"].astype(str).str.lower()
    # --- lookups --------------------------------------------------------------
    def contains(pattern: str):
        return _match_items_contains(items, pattern)

    def first_num(pattern: str, default: float = 0.0) -> float:
        m = contains(pattern)
        if not m.any():
            return float(default)
        s = pd.to_numeric(vals[m].iloc[0], errors="coerce")
        return float(s) if pd.notna(s) else float(default)

    def num(pattern: str, default: float = 0.0) -> float:
        m = contains(pattern)
        if not m.any():
            return float(default)
        return float(pd.to_numeric(vals[m], errors="coerce").fillna(0.0).sum())

    def strv(pattern: str, default: str = "") -> str:
        m = contains(pattern)
        return (str(vals[m].iloc[0]) if m.any() else default)

    def num_pct(pattern: str, default: float = 0.0) -> float:
        return pct(first_num(pattern, default * 100.0), default)

    def sum_time(pattern: str, default: float = 0.0, exclude_pattern: str | None = None) -> float:
        """Sum only time-like rows; minutes are converted to hours."""

        mask = contains(pattern)
        exclude_mask = contains(exclude_pattern) if exclude_pattern else None
        return _sum_time_from_series(
            items,
            vals,
            dtt,
            mask,
            default=float(default),
            exclude_mask=exclude_mask,
        )

    # --- OPTIONAL: let sheet override params/rates if those rows exist ---
    def sheet_num(pat, default=None):
        v = first_num(pat, float('nan'))
        return v if v == v else default  # NaN check

    def sheet_pct(pat, default=None):
        v = first_num(pat, float('nan'))
        if v == v:  # not NaN
            return (v/100.0) if abs(v) > 1.0 else v
        return default

    def sheet_text(pat, default=None):
        t = strv(pat, "").strip()
        return t if t else default

    # Params overrides
    params["OEE_EfficiencyPct"]        = sheet_pct(r"\b(OEE\s*Efficiency\s*%|OEE)\b", params.get("OEE_EfficiencyPct"))
    params["VendorMarkupPct"]          = sheet_pct(r"\b(Vendor\s*Markup\s*%)\b", params.get("VendorMarkupPct"))
    params["MinLotCharge"]             = sheet_num(r"\b(Min\s*Lot\s*Charge)\b", params.get("MinLotCharge"))
    params["ProgCapHr"]                = sheet_num(r"\b(Programming\s*Cap\s*Hr)\b", params.get("ProgCapHr"))
    params["ProgSimpleDim_mm"]         = sheet_num(r"\b(Prog\s*Simple\s*Dim_mm)\b", params.get("ProgSimpleDim_mm"))
    params["ProgMaxToMillingRatio"]    = sheet_num(r"\b(Prog\s*Max\s*To\s*Milling\s*Ratio)\b", params.get("ProgMaxToMillingRatio"))
    prog_hr_override = sheet_num(r"\b(Programming\s*(?:Override|Manual)(?:\s*H(?:ou)?rs?)?)\b", None)
    params["MillingConsumablesPerHr"]  = sheet_num(r"(?i)Milling\s*Consumables\s*/\s*Hr", params.get("MillingConsumablesPerHr"))
    params["TurningConsumablesPerHr"]  = sheet_num(r"(?i)Turning\s*Consumables\s*/\s*Hr", params.get("TurningConsumablesPerHr"))
    params["EDMConsumablesPerHr"]      = sheet_num(r"(?i)EDM\s*Consumables\s*/\s*Hr", params.get("EDMConsumablesPerHr"))
    params["GrindingConsumablesPerHr"] = sheet_num(r"(?i)Grinding\s*Consumables\s*/\s*Hr", params.get("GrindingConsumablesPerHr"))
    params["InspectionConsumablesPerHr"]= sheet_num(r"(?i)Inspection\s*Consumables\s*/\s*Hr", params.get("InspectionConsumablesPerHr"))
    params["UtilitiesPerSpindleHr"]    = sheet_num(r"(?i)Utilities\s*Per\s*Spindle\s*Hr", params.get("UtilitiesPerSpindleHr"))
    params["ConsumablesFlat"]          = sheet_num(r"(?i)Consumables\s*Flat", params.get("ConsumablesFlat"))

    # Rates overrides
    def rate_from_sheet(label, key):
        v = sheet_num(label, None)
        if v is not None:
            rates[key] = v

    rate_from_sheet(r"Rate\s*-\s*Programming",   "ProgrammingRate")
    rate_from_sheet(r"Rate\s*-\s*Engineer",      "EngineerRate")
    rate_from_sheet(r"Rate\s*-\s*Milling",       "MillingRate")
    rate_from_sheet(r"Rate\s*-\s*Turning",       "TurningRate")
    rate_from_sheet(r"Rate\s*-\s*Wire\s*EDM",    "WireEDMRate")
    rate_from_sheet(r"Rate\s*-\s*Sinker\s*EDM",  "SinkerEDMRate")
    rate_from_sheet(r"Rate\s*-\s*Surface\s*Grind","SurfaceGrindRate")
    rate_from_sheet(r"Rate\s*-\s*ODID\s*Grind",  "ODIDGrindRate")
    rate_from_sheet(r"Rate\s*-\s*Jig\s*Grind",   "JigGrindRate")
    rate_from_sheet(r"Rate\s*-\s*Lapping",       "LappingRate")
    rate_from_sheet(r"Rate\s*-\s*Inspection",    "InspectionRate")
    rate_from_sheet(r"Rate\s*-\s*Finishing",     "FinishingRate")
    rate_from_sheet(r"Rate\s*-\s*Saw/Waterjet",  "SawWaterjetRate")
    rate_from_sheet(r"Rate\s*-\s*Fixture\s*Build","FixtureBuildRate")
    rate_from_sheet(r"Rate\s*-\s*Assembly",      "AssemblyRate")
    rate_from_sheet(r"Rate\s*-\s*Packaging",     "PackagingRate")
    rate_from_sheet(r"Rate\s*-\s*Project\s*Mgmt", "ProjectManagementRate")
    rate_from_sheet(r"Rate\s*-\s*Project\s*Management", "ProjectManagementRate")
    rate_from_sheet(r"Rate\s*-\s*Project\s*Manager", "ProjectManagementRate")
    rate_from_sheet(r"Rate\s*-\s*Toolmaker",     "ToolmakerSupportRate")
    rate_from_sheet(r"Rate\s*-\s*Tool\s*&\s*Die", "ToolmakerSupportRate")

    # ---- knobs & qty ---------------------------------------------------------
    OverheadPct    = num_pct(r"\b" + alt('Overhead','Shop Overhead') + r"\b", params["OverheadPct"])
    MarginPct      = num_pct(r"\b" + alt('Margin','Profit Margin') + r"\b", params["MarginPct"])
    GA_Pct         = num_pct(r"\b" + alt('G&A',r'General\s*&\s*Admin') + r"\b", params["GA_Pct"])
    ContingencyPct = num_pct(r"\b" + alt('Contingency',r'Risk\s*Adder') + r"\b",  params["ContingencyPct"])
    ExpeditePct    = num_pct(r"\b" + alt('Expedite',r'Rush\s*Fee') + r"\b",      params["ExpeditePct"])

    priority = strv(alt('PM-01_Quote_Priority',r'Quote\s*Priority'), "").strip().lower()
    if priority not in ("expedite", "critical"):
        ExpeditePct = 0.0

    qty_override_val = None
    if isinstance(params, Mapping):
        qty_override_val = _coerce_float_or_none(params.get("Quantity"))

    sheet_qty_raw = first_num(r"\b" + alt('Qty','Lot Size','Quantity') + r"\b", float("nan"))
    sheet_qty_val = sheet_qty_raw if sheet_qty_raw == sheet_qty_raw else None

    if qty_override_val is not None:
        Qty = max(1, int(round(qty_override_val)))
    elif sheet_qty_val is not None:
        Qty = max(1, int(round(sheet_qty_val)))
    else:
        Qty = 1
    amortize_programming = True  # single switch if you want to expose later

    # ---- geometry (optional) -------------------------------------------------
    GEO_wedm_len_mm = first_num(r"\bGEO__?(?:WEDM_PathLen_mm|EDM_Length_mm)\b", 0.0)
    GEO_vol_mm3     = first_num(r"\bGEO__?(?:Volume|Volume_mm3|Net_Volume_mm3)\b", 0.0)

    # ---- material ------------------------------------------------------------
    vol_cm3       = first_num(r"\b(?:Net\s*Volume|Volume_net|Volume\s*\(cm\^?3\))\b", GEO_vol_mm3 / 1000.0)
    density_g_cc  = first_num(r"\b(?:Density|Material\s*Density)\b", 0.0)

    scrap_pattern = r"\b(?:Scrap\s*%|Expected\s*Scrap)\b"
    scrap_sheet_fraction: float | None = None
    try:
        sheet_has_scrap = contains(scrap_pattern)
    except Exception:
        sheet_has_scrap = pd.Series(dtype=bool)
    if hasattr(sheet_has_scrap, "any") and sheet_has_scrap.any():
        scrap_candidate = first_num(scrap_pattern, float("nan"))
        if scrap_candidate == scrap_candidate:  # not NaN
            scrap_sheet_fraction = normalize_scrap_pct(scrap_candidate)

    scrap_auto: float | None = None
    scrap_source_label: str | None = None
    if scrap_sheet_fraction is not None:
        scrap_auto = scrap_sheet_fraction
        scrap_source_label = "sheet"
    else:
        scrap_from_stock, stock_source = _estimate_scrap_from_stock_plan(geo_context)
        if scrap_from_stock is not None:
            scrap_auto = scrap_from_stock
            scrap_source_label = stock_source or "stock_plan"
    if scrap_auto is None:
        scrap_auto = SCRAP_DEFAULT_GUESS
        scrap_source_label = "default_guess"

    try:
        ui_scrap_raw = ui_vars.get("Scrap Percent (%)") if "ui_vars" in locals() else None
    except Exception:
        ui_scrap_raw = None
    ui_scrap_val = normalize_scrap_pct(ui_scrap_raw)
    ui_has_scrap = _scrap_value_provided(ui_scrap_raw)
    if not ui_has_scrap:
        try:
            mask = df["Item"].astype(str).str.fullmatch(r"Scrap Percent \(%\)", case=False, na=False)
            candidate = df.loc[mask, "Example Values / Options"].iloc[0]
            if _scrap_value_provided(candidate):
                ui_scrap_val = normalize_scrap_pct(candidate)
                ui_has_scrap = True
        except Exception:
            ui_has_scrap = False

    if ui_has_scrap:
        scrap_pct = normalize_scrap_pct(ui_scrap_val)
        scrap_pct_baseline = scrap_pct
        scrap_source_label = "ui"
    else:
        fallback_scrap = SCRAP_DEFAULT_GUESS if scrap_auto is None else scrap_auto
        scrap_pct = normalize_scrap_pct(fallback_scrap)
        scrap_pct_baseline = scrap_pct

    holes_scrap = _holes_scrap_fraction(geo_context)
    if holes_scrap and holes_scrap > 0:
        # Take the more conservative (larger) scrap figure so we don’t underquote.
        scrap_pct = normalize_scrap_pct(max(scrap_pct_baseline, holes_scrap))
        scrap_pct_baseline = scrap_pct
        scrap_source_label = (scrap_source_label or "auto") + "+holes"
    else:
        scrap_pct = scrap_pct_baseline
    material_name_raw = sheet_text(r"(?i)Material\s*(?:Name|Grade|Alloy|Type)")
    if not material_name_raw:
        fallback_name = strv(r"(?i)^Material$", "")
        if fallback_name and not re.fullmatch(r"\s*[$0-9.,]+\s*", str(fallback_name)):
            material_name_raw = fallback_name
    material_name = str(material_name_raw or "").strip()
    if not material_name and geo_context.get("material"):
        material_name = str(geo_context.get("material"))
    if material_name:
        geo_context.setdefault("material", material_name)
    material_name = str(geo_context.get("material") or material_name or "").strip()

    material_lookup_key = _normalize_lookup_key(material_name) if material_name else ""
    if material_lookup_key:
        material_selection["material_lookup"] = material_lookup_key

    normalized_material_row = _normalize_material(material_name) if material_name else None
    canonical_material = ""
    if normalized_material_row:
        canonical_material = str(
            normalized_material_row.get("canonical_material") or ""
        ).strip()
        iso_group = str(normalized_material_row.get("iso_group") or "").strip()
        if iso_group:
            material_selection["material_group"] = iso_group.upper()
    if not canonical_material:
        canonical_material = MATERIAL_DISPLAY_BY_KEY.get(
            material_lookup_key,
            material_name or default_material_display or "",
        )
    if canonical_material:
        material_selection["canonical_material"] = canonical_material

    if material_name:
        material_selection["input_material"] = material_name
    if material_selection.get("material_group"):
        geo_context.setdefault("material_group", material_selection["material_group"])

    density_guess_from_material = _density_for_material(material_name, _DEFAULT_MATERIAL_DENSITY_G_CC)
    if not density_g_cc or density_g_cc <= 0:
        density_g_cc = density_guess_from_material

    if is_plate_2d:
        length_in_val = _coerce_float_or_none(ui_vars.get("Plate Length (in)"))
        width_in_val = _coerce_float_or_none(ui_vars.get("Plate Width (in)"))
        thickness_mm_val = _coerce_float_or_none(geo_context.get("thickness_mm"))
        if thickness_mm_val is None:
            t_in = _coerce_float_or_none(ui_vars.get("Thickness (in)"))
            if t_in is not None:
                thickness_mm_val = float(t_in) * 25.4
        thickness_mm_val = float(thickness_mm_val or 0.0)
        if length_in_val and length_in_val > 0:
            plate_length_in_val = float(length_in_val)
        if width_in_val and width_in_val > 0:
            plate_width_in_val = float(width_in_val)
        if thickness_mm_val > 0:
            geo_context["thickness_mm"] = thickness_mm_val
        length_mm = float(length_in_val or 0.0) * 25.4
        width_mm = float(width_in_val or 0.0) * 25.4
        if length_mm > 0:
            geo_context.setdefault("plate_length_mm", length_mm)
        if width_mm > 0:
            geo_context.setdefault("plate_width_mm", width_mm)
        vol_mm3_plate = max(1.0, length_mm * width_mm * max(thickness_mm_val, 0.0))
        vol_cm3 = max(vol_cm3, vol_mm3_plate / 1000.0)
        GEO_vol_mm3 = max(GEO_vol_mm3, vol_mm3_plate)
        density_lookup = {
            "steel": 7.85,
            "stainless": 7.90,
            "alum": 2.70,
            "titanium": 4.43,
            "copper": 8.96,
            "brass": 8.40,
            "plastic": 1.45,
        }

        def _context_density(ctx: dict[str, Any] | None) -> float | None:
            if not isinstance(ctx, dict):
                return None
            val = _coerce_float_or_none(ctx.get("density_g_cc"))
            if val and val > 0:
                return float(val)
            return None

        inner_geo_ctx = geo_context.get("geo") if isinstance(geo_context.get("geo"), dict) else None
        density_candidates: list[float] = []
        for ctx in (geo_context, inner_geo_ctx):
            dens = _context_density(ctx)
            if dens:
                density_candidates.append(dens)
        if not density_candidates:
            density_candidates.append(
                _density_for_material(material_name or default_material_display, default=0.0)
            )
        density_candidates.append(
            density_lookup.get(
                _material_family(geo_context.get("material") or material_name),
                density_lookup["steel"],
            )
        )
        density_g_cc = next((d for d in density_candidates if d and d > 0), density_lookup["steel"])
        holes_for_mass: list[float] = []
        hole_sources: list[Any] = []
        for key in ("hole_diams_mm_precise", "hole_diams_mm"):
            raw = geo_context.get(key)
            if isinstance(raw, (list, tuple)):
                hole_sources.extend(raw)
        if not hole_sources:
            inner_geo = geo_context.get("geo")
            if isinstance(inner_geo, dict):
                raw_inner = inner_geo.get("hole_diams_mm")
                if isinstance(raw_inner, (list, tuple)):
                    hole_sources.extend(raw_inner)
        for hv in hole_sources:
            val = _coerce_float_or_none(hv)
            if val and val > 0:
                holes_for_mass.append(float(val))
        thickness_in_for_mass = (thickness_mm_val / 25.4) if thickness_mm_val else _coerce_float_or_none(ui_vars.get("Thickness (in)"))
        net_mass_calc_tuple = net_mass_kg(
            length_in_val,
            width_in_val,
            thickness_in_for_mass,
            holes_for_mass,
            density_g_cc,
            return_removed_mass=True,
        )
        if isinstance(net_mass_calc_tuple, tuple):
            net_mass_calc, removed_mass_candidate = net_mass_calc_tuple
        else:
            net_mass_calc, removed_mass_candidate = net_mass_calc_tuple, None

        if net_mass_calc:
            geo_context.setdefault("net_mass_kg_est", float(net_mass_calc))
            try:
                vol_cm3_net = (float(net_mass_calc) * 1000.0) / float(density_g_cc or 0.0)
            except Exception:
                vol_cm3_net = None
            if vol_cm3_net and vol_cm3_net > 0:
                vol_cm3 = max(vol_cm3, vol_cm3_net)
                GEO_vol_mm3 = max(GEO_vol_mm3, vol_cm3_net * 1000.0)
                geo_context.setdefault("net_volume_cm3", float(vol_cm3_net))

        if removed_mass_candidate and removed_mass_candidate > 0:
            removal_mass_g_est = float(removed_mass_candidate)
            if net_mass_calc:
                try:
                    ratio = removal_mass_g_est / max(1e-6, float(net_mass_calc) * 1000.0)
                except Exception:
                    ratio = 0.0
            else:
                ratio = 0.0
            if ratio > 0:
                hole_scrap_frac_est = max(hole_scrap_frac_est, float(ratio))

        if thickness_mm_val > 0:
            plate_thickness_in_val = float(thickness_mm_val) / 25.4
        elif thickness_in_for_mass and thickness_in_for_mass > 0:
            plate_thickness_in_val = float(thickness_in_for_mass)
        else:
            t_in_hint = _coerce_float_or_none(ui_vars.get("Thickness (in)"))
            if t_in_hint and t_in_hint > 0:
                plate_thickness_in_val = float(t_in_hint)

    if removal_mass_g_est and removal_mass_g_est > 0:
        geo_context.setdefault("material_removed_mass_g_est", float(removal_mass_g_est))
        if inner_geo is not None:
            inner_geo.setdefault("material_removed_mass_g_est", float(removal_mass_g_est))

    if hole_scrap_frac_est and hole_scrap_frac_est > 0:
        hole_scrap_clamped_val = max(0.0, min(0.25, float(hole_scrap_frac_est)))
        if scrap_source_label == "default_guess":
            scrap_pct = hole_scrap_clamped_val
            scrap_pct_baseline = hole_scrap_clamped_val
        else:
            scrap_pct = max(scrap_pct, hole_scrap_clamped_val)
            scrap_pct_baseline = max(scrap_pct_baseline, hole_scrap_clamped_val)
        geo_context.setdefault("scrap_pct_from_holes", float(hole_scrap_frac_est))
        geo_context.setdefault("scrap_pct_from_holes_clamped", hole_scrap_clamped_val)
        if inner_geo is not None:
            inner_geo.setdefault("scrap_pct_from_holes", float(hole_scrap_frac_est))
            inner_geo.setdefault("scrap_pct_from_holes_clamped", hole_scrap_clamped_val)

    if density_g_cc and density_g_cc > 0:
        geo_context.setdefault("density_g_cc", float(density_g_cc))

    net_mass_g = max(0.0, vol_cm3 * density_g_cc)
    mass_basis = "volume"
    fallback_meta: dict[str, Any] = {}
    if scrap_source_label:
        fallback_meta.setdefault("scrap_source", scrap_source_label)

    def _mass_hint_kg(ctx: dict[str, Any] | None) -> float:
        if not isinstance(ctx, dict):
            return 0.0
        hints: list[float] = []
        for key in ("net_mass_kg", "net_mass_kg_est", "mass_kg"):
            val = _coerce_float_or_none(ctx.get(key))
            if val and val > 0:
                hints.append(float(val))
        mass_g_hint = _coerce_float_or_none(ctx.get("net_mass_g"))
        if mass_g_hint and mass_g_hint > 0:
            hints.append(float(mass_g_hint) / 1000.0)
        return max(hints) if hints else 0.0

    if net_mass_g <= 0:
        inner_geo = geo_context.get("geo") if isinstance(geo_context.get("geo"), dict) else {}
        mass_hint_kg = max(_mass_hint_kg(geo_context), _mass_hint_kg(inner_geo))
        if mass_hint_kg > 0:
            net_mass_g = mass_hint_kg * 1000.0
            mass_basis = "geo_mass_hint"
            fallback_meta["mass_hint_kg"] = mass_hint_kg
        else:
            def _volume_hint_cm3(ctx: dict[str, Any] | None) -> float:
                if not isinstance(ctx, dict):
                    return 0.0
                vol_candidates = [
                    ctx.get("net_volume_cm3"),
                    ctx.get("volume_cm3"),
                ]
                for key in ("GEO-Volume_mm3", "volume_mm3", "GEO__Volume_mm3"):
                    mm3_val = _coerce_float_or_none(ctx.get(key))
                    if mm3_val and mm3_val > 0:
                        vol_candidates.append(float(mm3_val) / 1000.0)
                vol_candidates = [
                    _coerce_float_or_none(val)
                    for val in vol_candidates
                    if val is not None
                ]
                vols = [float(v) for v in vol_candidates if v and v > 0]
                if vols:
                    return max(vols)
                dims = []
                for key in ("GEO-01_Length_mm", "GEO-02_Width_mm", "GEO-03_Height_mm"):
                    val = _coerce_float_or_none(ctx.get(key))
                    if val and val > 0:
                        dims.append(float(val))
                if len(dims) == 3:
                    return (dims[0] * dims[1] * dims[2]) / 1000.0
                return 0.0

            inner_geo = geo_context.get("geo") if isinstance(geo_context.get("geo"), dict) else {}
            bbox_volume_cm3 = max(_volume_hint_cm3(geo_context), _volume_hint_cm3(inner_geo))
            if bbox_volume_cm3 > 0:
                density_guess = density_g_cc if density_g_cc > 0 else _density_for_material(material_name or default_material_display)
                net_mass_g = bbox_volume_cm3 * max(density_guess, 0.5)
                mass_basis = "bbox_volume"
                fallback_meta["fallback_volume_cm3"] = bbox_volume_cm3
                fallback_meta["fallback_density_g_cc"] = density_guess
            else:
                density_guess = density_g_cc if density_g_cc > 0 else _density_for_material(material_name or default_material_display)
                min_mass_kg = 0.05
                net_mass_g = min_mass_kg * 1000.0
                mass_basis = "minimum_mass"
                fallback_meta["fallback_density_g_cc"] = density_guess
                fallback_meta["minimum_mass_kg"] = min_mass_kg

    effective_mass_g = net_mass_g * (1.0 + scrap_pct)
    mass_kg = net_mass_g / 1000.0

    unit_price_per_g  = first_num(r"\b(?:Material\s*Price.*(?:per\s*g|/g)|Unit\s*Price\s*/\s*g)\b", 0.0)
    supplier_min_charge = first_num(r"\b(?:Supplier\s*Min\s*Charge|min\s*charge)\b", 0.0)
    explicit_mat  = num(r"\b(?:Material\s*Cost|Raw\s*Material\s*Cost)\b", 0.0)
    scrap_credit_pattern = r"(?:Material\s*Scrap(?:\s*Credit)?|Remnant(?:\s*Credit)?)"
    scrap_credit_mask = contains(scrap_credit_pattern)
    material_scrap_credit_entered = False
    if scrap_credit_mask.any():
        scrap_credit_values = pd.to_numeric(vals[scrap_credit_mask], errors="coerce")
        if hasattr(scrap_credit_values, "tolist"):
            iter_values = scrap_credit_values.tolist()
        elif isinstance(scrap_credit_values, (list, tuple)):
            iter_values = list(scrap_credit_values)
        else:
            try:
                iter_values = list(scrap_credit_values)
            except TypeError:
                iter_values = [scrap_credit_values]
        for raw_value in iter_values:
            coerced = _coerce_float_or_none(raw_value)
            if coerced is not None and abs(float(coerced)) > 0:
                material_scrap_credit_entered = True
                break
    material_scrap_credit_raw = (
        num(scrap_credit_pattern, 0.0) if material_scrap_credit_entered else 0.0
    )
    material_scrap_credit = (
        abs(float(material_scrap_credit_raw or 0.0)) if material_scrap_credit_entered else 0.0
    )
    material_scrap_credit_applied = 0.0

    if material_vendor_csv is None:
        material_vendor_csv = params.get("MaterialVendorCSVPath", "") if isinstance(params, dict) else ""
    material_vendor_csv = str(material_vendor_csv or "")
    material_overrides: dict[str, Any] = {}
    provider_cost: float | None = None
    material_detail: dict[str, Any] = {}

    try:
        provider_cost, material_detail = compute_material_cost(
            material_name,
            mass_kg,
            scrap_pct,
            material_overrides,
            material_vendor_csv,
            default_material_display=default_material_display,
            pricing=pricing_engine,
        )
    except Exception as err:
        material_detail = {
            "material_name": material_name,
            "mass_g": effective_mass_g,
            "effective_mass_g": effective_mass_g,
            "mass_g_net": net_mass_g,
            "net_mass_g": net_mass_g,
            "scrap_pct": scrap_pct,
            "error": str(err),
        }
        provider_cost = None

    manual_baseline = unit_price_per_g * effective_mass_g
    base_cost = provider_cost if provider_cost is not None else manual_baseline
    base_cost = float(base_cost or 0.0)

    cost_with_min = max(base_cost, float(supplier_min_charge or 0.0))
    material_cost = max(cost_with_min, float(explicit_mat or 0.0))

    if material_detail:
        material_detail.update({
            "supplier_min_charge": float(supplier_min_charge or 0.0),
            "explicit_cost_override": float(explicit_mat or 0.0),
            "material_cost_before_overrides": base_cost,
            "material_cost": float(material_cost),
            "mass_basis": mass_basis,
        })
        unit_price_from_detail = material_detail.get("unit_price_usd_per_kg")
        if unit_price_from_detail is not None:
            try:
                unit_price_per_g = float(unit_price_from_detail) / 1000.0
            except Exception:
                pass

    material_direct_cost = float(material_cost)
    material_cost_before_credit = float(material_direct_cost)

    material_detail_for_breakdown = dict(material_detail)
    if material_selection.get("canonical_material"):
        material_detail_for_breakdown.setdefault(
            "canonical_material", material_selection["canonical_material"]
        )
    if material_selection.get("material_group"):
        material_detail_for_breakdown.setdefault(
            "material_group", material_selection["material_group"]
        )
    if material_selection.get("material_lookup"):
        material_detail_for_breakdown.setdefault(
            "material_lookup", material_selection["material_lookup"]
        )
    material_detail_for_breakdown.setdefault("material_name", material_name)
    material_detail_for_breakdown.setdefault("mass_g", effective_mass_g)
    material_detail_for_breakdown.setdefault("effective_mass_g", effective_mass_g)
    material_detail_for_breakdown.setdefault("mass_g_net", net_mass_g)
    material_detail_for_breakdown.setdefault("net_mass_g", net_mass_g)
    material_detail_for_breakdown.setdefault("scrap_pct", scrap_pct)
    if removal_mass_g_est and removal_mass_g_est > 0:
        material_detail_for_breakdown.setdefault("material_removed_mass_g_est", float(removal_mass_g_est))
    if hole_scrap_clamped_val > 0:
        material_detail_for_breakdown.setdefault("scrap_pct_from_holes", float(hole_scrap_frac_est))
        material_detail_for_breakdown.setdefault("scrap_pct_from_holes_clamped", float(hole_scrap_clamped_val))
    if mass_basis:
        material_detail_for_breakdown.setdefault("mass_basis", mass_basis)
    for key, value in fallback_meta.items():
        material_detail_for_breakdown.setdefault(key, value)
    material_detail_for_breakdown["unit_price_per_g"] = float(unit_price_per_g or 0.0)
    if "unit_price_usd_per_kg" not in material_detail_for_breakdown and unit_price_per_g:
        material_detail_for_breakdown["unit_price_usd_per_kg"] = float(unit_price_per_g) * 1000.0
    material_detail_for_breakdown["material_direct_cost"] = material_direct_cost
    if provider_cost is not None:
        material_detail_for_breakdown["provider_cost"] = float(provider_cost)

    if is_plate_2d:
        mat_for_price = geo_context.get("material") or material_name
        try:
            mat_usd_per_kg, mat_src = _resolve_material_unit_price(
                mat_for_price,
                unit="kg",
            )

        except Exception:
            mat_usd_per_kg, mat_src = 0.0, ""
        mat_usd_per_kg = float(mat_usd_per_kg or 0.0)
        effective_scrap_source = scrap_pct if scrap_pct else scrap_pct_baseline
        effective_scrap = normalize_scrap_pct(effective_scrap_source)
        material_cost = mass_kg * (1.0 + effective_scrap) * mat_usd_per_kg
        material_direct_cost = round(material_cost, 2)
        material_detail_for_breakdown.update({
            "material_name": mat_for_price,
            "mass_g_net": mass_kg * 1000.0,
            "net_mass_g": mass_kg * 1000.0,
            "mass_g": mass_kg * 1000.0 * (1.0 + effective_scrap),
            "effective_mass_g": mass_kg * 1000.0 * (1.0 + effective_scrap),
            "scrap_pct": effective_scrap,
            "unit_price_usd_per_kg": mat_usd_per_kg,
            "unit_price_per_g": mat_usd_per_kg / 1000.0 if mat_usd_per_kg else material_detail_for_breakdown.get("unit_price_per_g", 0.0),
            "unit_price_usd_per_lb": (mat_usd_per_kg / LB_PER_KG) if mat_usd_per_kg else material_detail_for_breakdown.get("unit_price_usd_per_lb"),
            "source": mat_src or material_detail_for_breakdown.get("source", ""),
        })
        material_detail_for_breakdown["material_cost"] = material_direct_cost
        material_detail_for_breakdown["material_direct_cost"] = material_direct_cost
        material_cost_before_credit = float(material_direct_cost)
        scrap_pct = effective_scrap

    scrap_credit_from_weight = 0.0
    scrap_credit_mass_lb = 0.0
    scrap_credit_unit_price_lb = None
    eff_mass_for_credit = _coerce_float_or_none(material_detail_for_breakdown.get("mass_g"))
    if eff_mass_for_credit is None:
        eff_mass_for_credit = _coerce_float_or_none(
            material_detail_for_breakdown.get("effective_mass_g")
        )
    net_mass_for_credit = _coerce_float_or_none(
        material_detail_for_breakdown.get("net_mass_g")
    )
    if net_mass_for_credit is None:
        net_mass_for_credit = _coerce_float_or_none(
            material_detail_for_breakdown.get("mass_g_net")
        )
    if eff_mass_for_credit is not None and net_mass_for_credit is not None:
        scrap_mass_g = max(0.0, float(eff_mass_for_credit) - float(net_mass_for_credit))
        if scrap_mass_g > 0:
            scrap_credit_mass_lb = scrap_mass_g / 1000.0 * LB_PER_KG
            scrap_credit_unit_price_lb = _coerce_float_or_none(
                material_detail_for_breakdown.get("unit_price_usd_per_lb")
            )
            if scrap_credit_unit_price_lb is None:
                scrap_credit_unit_price_kg = _coerce_float_or_none(
                    material_detail_for_breakdown.get("unit_price_usd_per_kg")
                )
                if scrap_credit_unit_price_kg is not None:
                    scrap_credit_unit_price_lb = float(scrap_credit_unit_price_kg) / LB_PER_KG
            if scrap_credit_unit_price_lb is None:
                unit_price_per_g_val = _coerce_float_or_none(
                    material_detail_for_breakdown.get("unit_price_per_g")
                )
                if unit_price_per_g_val is not None:
                    scrap_credit_unit_price_lb = float(unit_price_per_g_val) * (1000.0 / LB_PER_KG)
            if scrap_credit_unit_price_lb is not None and scrap_credit_unit_price_lb > 0:
                scrap_credit_from_weight = round(
                    scrap_credit_mass_lb * float(scrap_credit_unit_price_lb), 2
                )
                material_detail_for_breakdown["material_scrap_credit_calculated"] = (
                    scrap_credit_from_weight
                )
                material_detail_for_breakdown["scrap_credit_mass_lb"] = scrap_credit_mass_lb
                material_detail_for_breakdown[
                    "scrap_credit_unit_price_usd_per_lb"
                ] = float(scrap_credit_unit_price_lb)

    if scrap_credit_from_weight > 0:
        material_scrap_credit = scrap_credit_from_weight
        material_scrap_credit_entered = True

    material_detail_for_breakdown["material_scrap_credit_entered"] = (
        material_scrap_credit_entered
    )

    if material_scrap_credit:
        credit_cap = min(material_scrap_credit, float(material_cost_before_credit))
        if credit_cap > 0:
            material_scrap_credit_applied = float(credit_cap)
            material_direct_cost = max(0.0, float(material_cost_before_credit) - material_scrap_credit_applied)
            material_detail_for_breakdown["material_scrap_credit"] = material_scrap_credit_applied
            material_detail_for_breakdown["material_cost_before_credit"] = float(material_cost_before_credit)
            material_detail_for_breakdown["material_cost"] = material_direct_cost
            material_detail_for_breakdown["material_direct_cost"] = material_direct_cost

    # ---- programming / cam / dfm --------------------------------------------
    eng_hr  = sum_time(r"(?:Fixture\s*Design|Process\s*Sheet|Traveler|Documentation)")

    # ---- milling hours & caps ------------------------------------------------
    rough_hr   = sum_time(r"(?:Roughing\s*Cycle\s*Time|Adaptive|HSM)")
    semi_hr    = sum_time(r"(?:Semi[- ]?Finish|Rest\s*Milling)")
    finish_hr  = sum_time(r"(?:Finishing\s*Cycle\s*Time)")
    setups     = first_num(r"(?:Number\s*of\s*Milling\s*Setups|Milling\s*Setups)", 0.0)
    setup_each = first_num(r"(?:Setup\s*Time\s*per\s*Setup|Setup\s*Hours\s*/\s*Setup)", 0.5)

    setups = max(1, min(int(round(setups or 1)), 3))
    from_back_geo_flag = bool(geo_context.get("needs_back_face") or geo_context.get("from_back"))
    if from_back_geo_flag and setups < 2:
        setups = 2
        geo_context["needs_back_face"] = True
    setup_hr  = setups * setup_each
    milling_hr = rough_hr + semi_hr + finish_hr + setup_hr

    max_dim   = first_num(r"\bGEO__MaxDim_mm\b", 0.0)
    is_simple = (max_dim and max_dim <= params["ProgSimpleDim_mm"] and setups <= 2)

    # ---- fixture -------------------------------------------------------------
    fixture_sheet_hr = sum_time(r"(?:Fixture\s*Build|Custom\s*Fixture\s*Build)")
    setups = max(1, min(int(round(setups or 1)), 3))  # you already computed setups above

    if fixture_sheet_hr and fixture_sheet_hr > 0:
        # honor an explicit entry from the sheet/UI if present
        fixture_build_hr = float(fixture_sheet_hr)
    else:
        # gentle, setup-driven curve (no hard cap)
        base = float(params.get("FixtureBaseHr", 0.70))   # ~0.7 hr base
        per_setup = float(params.get("FixturePerSetupHr", 0.15))  # small add per extra setup
        fixture_build_hr = base + per_setup * max(0, setups - 1)
        # (Optional) tiny tolerance bump without caps:
        try:
            min_tol_in = float(min_tol_for_fixture) if min_tol_for_fixture else None
        except Exception:
            min_tol_in = None
        if min_tol_in and min_tol_in > 0:
            norm = max(0.0, min(1.0, (INPROC_REF_TOL_IN - min_tol_in) / INPROC_REF_TOL_IN))
            fixture_build_hr *= (1.0 + 0.20 * (norm ** 0.6))  # ≤ ~+20% at very tight

    # Fixture material cost is no longer applied; only labor is considered.
    fixture_labor_cost    = float(fixture_build_hr) * float(rates["FixtureBuildRate"])
    fixture_cost          = fixture_labor_cost
    fixture_labor_per_part = (fixture_labor_cost / Qty) if Qty > 1 else fixture_labor_cost
    fixture_per_part       = (fixture_cost / Qty) if Qty > 1 else fixture_cost

    nre_detail = {
        "fixture": {
            "build_hr": float(fixture_build_hr), "build_rate": rates["FixtureBuildRate"],
            "labor_cost": float(fixture_labor_cost),
            "per_lot": fixture_cost, "per_part": fixture_per_part
        }
    }

    process_plan_summary: dict[str, Any] = {}
    family_for_breakdown: str | None = None
    planner_pricing_result: dict[str, Any] | None = None
    planner_bucket_view: dict[str, Any] | None = None
    planner_two_bucket_rates: dict[str, Any] | None = None
    planner_geom_payload: dict[str, Any] | None = None
    planner_pricing_error: str | None = None
    planner_process_minutes: float | None = None
    planner_drill_minutes: float | None = None
    planner_drilling_override: dict[str, float] | None = None
    used_planner = False

    red_flag_messages: list[str] = []
    _red_flag_seen: set[str] = set()

    def _record_red_flag(message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        if text not in _red_flag_seen:
            _red_flag_seen.add(text)
            red_flag_messages.append(text)

    force_legacy_pricing = False

    # ---- primary processes ---------------------------------------------------
    # OEE-adjusted helper
    oee = max(1e-6, params["OEE_EfficiencyPct"])
    def eff(hr: float) -> float:
        try: return float(hr) / oee
        except Exception: return hr

    # Turning
    od_turn_hr   = sum_time(r"(?:OD\s*Turning|OD\s*Rough/Finish|Outer\s*Diameter)")
    id_bore_hr   = sum_time(r"(?:ID\s*Boring|Drilling|Reaming)")
    threading_hr = sum_time(r"(?:Threading|Tapping|Single\s*Point)")
    cutoff_hr    = sum_time(r"(?:Cut[- ]?Off|Parting)")
    turning_hr   = eff(od_turn_hr + id_bore_hr + threading_hr + cutoff_hr)

    # Wire EDM
    wedm_hr_dir  = sum_time(r"(?:WEDM\s*Hours|Wire\s*EDM\s*Hours|EDM\s*Burn\s*Time)")
    wedm_len_mm  = first_num(r"(?:EDM\s*Length_mm|WEDM\s*Length_mm|EDM\s*Perimeter_mm)", GEO_wedm_len_mm)
    wedm_thk_mm  = first_num(r"(?:EDM\s*Thickness_mm|Stock\s*Thickness_mm)", 0.0)
    wedm_passes  = max(1, int(first_num(r"(?:EDM\s*Passes|WEDM\s*Passes)", 1)))
    cut_rate     = max(0.0001, first_num(r"(?:EDM\s*Cut\s*Rate_mm\^?2/min|WEDM\s*Cut\s*Rate)", 2.0))
    edge_factor  = max(1.0, first_num(r"(?:EDM\s*Edge\s*Factor)", 1.0))
    wedm_mm2     = wedm_len_mm * max(0.0, wedm_thk_mm) * wedm_passes
    wedm_hr_calc = (wedm_mm2 / cut_rate) / 60.0 * edge_factor if wedm_mm2 > 0 else 0.0
    wedm_hr      = eff(wedm_hr_dir if wedm_hr_dir > 0 else wedm_hr_calc)

    wire_cost_m  = first_num(r"(?:WEDM\s*Wire\s*Cost\s*/\s*m|Wire\s*Cost\s*/m)", 0.0)
    wire_per_mm2 = first_num(r"(?:Wire\s*Usage\s*m\s*/\s*mm\^?2)", 0.0)
    wire_cost    = (wedm_mm2 * wire_per_mm2 * wire_cost_m) if (wire_per_mm2 > 0 and wire_cost_m > 0) else 0.0

    # Sinker EDM
    sinker_dir   = sum_time(r"(?:Sinker\s*EDM\s*Hours|Ram\s*EDM\s*Hours|Burn\s*Time)")
    sinker_vol   = first_num(r"(?:Sinker\s*Burn\s*Volume_mm3|EDM\s*Volume_mm3)", 0.0)
    sinker_mrr   = max(0.0001, first_num(r"(?:Sinker\s*MRR_mm\^?3/min|EDM\s*MRR)", 8.0))
    sinker_calc  = (sinker_vol / sinker_mrr) / 60.0 if sinker_vol > 0 else 0.0
    sinker_hr    = eff(sinker_dir if sinker_dir > 0 else sinker_calc)

    electrodes_n = max(0, int(first_num(r"(?:Electrode\s*Count)", 0)))
    electrode_each = first_num(r"(?:Electrode\s*(?:Cost|Material)\s*(?:/ea|per\s*ea)?)", 0.0)
    electrodes_cost = electrodes_n * electrode_each

    # Grinding
    surf_grind_hr = eff(sum_time(r"(?:Surface\s*Grind|Pre[- ]?Op\s*Grinding|Blank\s*Squaring)"))
    jig_grind_hr  = eff(sum_time(r"(?:Jig\s*Grind)"))
    odid_grind_hr = eff(sum_time(r"(?:OD/ID\s*Grind|Cylindrical\s*Grind)"))

    grind_vol   = first_num(r"(?:Grind\s*Volume_mm3|Grinding\s*Volume)", 0.0)
    grind_mrr   = max(0.0001, first_num(r"(?:Grind\s*MRR_mm\^?3/min|Grinding\s*MRR)", 300.0))
    grind_pass  = max(0, int(first_num(r"(?:Grinding\s*Passes)", 0)))
    dress_freq  = max(1, int(first_num(r"(?:Dress\s*Frequency\s*passes)", 4)))
    dress_min   = first_num(r"(?:Dress\s*Time\s*min(?:\s*/\s*pass)?)", 0.0)
    grind_calc  = (grind_vol / grind_mrr) / 60.0 if grind_vol > 0 else 0.0
    dresses     = math.floor(grind_pass / max(1, dress_freq))
    dressing_hr = (dress_min * dresses) / 60.0
    grind_hr_calc = grind_calc  # for meta
    grinding_hr  = surf_grind_hr + jig_grind_hr + odid_grind_hr + grind_hr_calc + dressing_hr

    # Apply multipliers
    five_axis_mult = 1.0 + num_pct(r"(?:5[- ]?Axis|Five[- ]Axis|Multi[- ]Axis)", max(0.0, params["FiveAxisMultiplier"] - 1.0))
    tight_tol_mult = 1.0 + num_pct(r"(?:Tight\s*Tolerance|Very\s*Tight)",        max(0.0, params["TightToleranceMultiplier"] - 1.0))

    # Costs
    milling_cost = eff(milling_hr) * rates["MillingRate"] * five_axis_mult * tight_tol_mult
    turning_cost = turning_hr * rates["TurningRate"]
    wedm_cost    = wedm_hr * rates["WireEDMRate"] + wire_cost
    sinker_cost  = sinker_hr * rates["SinkerEDMRate"] + electrodes_cost

    grinding_cost = (
        surf_grind_hr * rates["SurfaceGrindRate"] +
        jig_grind_hr  * rates["JigGrindRate"]     +
        odid_grind_hr * rates["ODIDGrindRate"]    +
        (grind_hr_calc + dressing_hr) * rates["SurfaceGrindRate"] +
        first_num(r"(?:Grinding\s*Wheel\s*Cost)", 0.0)
    )

    # Lapping / Finishing
    lap_hr   = sum_time(r"(?:Lapping|Honing|Polishing)")
    lap_cost = lap_hr * rates["LappingRate"]

    deburr_manual_hr = sum_time(r"(?:Deburr|Edge\s*Break)")
    tumble_hr        = sum_time(r"(?:Tumbling|Vibratory)")
    blast_hr         = sum_time(r"(?:Bead\s*Blasting|Sanding)")
    laser_mark_hr    = sum_time(r"(?:Laser\s*Mark|Engraving)")
    masking_hr       = sum_time(r"(?:Masking\s*for\s*Plating|Masking)")
    finishing_rate   = float(rates.get("FinishingRate", 0.0))
    finishing_misc_hr = tumble_hr + blast_hr + laser_mark_hr + masking_hr
    finishing_cost   = finishing_misc_hr * finishing_rate

    # Inspection & docs
    inproc_default = sheet_num(r"\bIn-Process\s*Inspection\s*Hours\b", 1.0)
    final_default  = sheet_num(r"\bFinal\s*Inspection\s*Hours\b", 0.0)
    inproc_hr   = sum_time(r"(?:In[- ]?Process\s*Inspection)", default=inproc_default)
    final_hr    = sum_time(r"(?:Final\s*Inspection|Manual\s*Inspection)", default=final_default)
    cmm_prog_hr = sum_time(r"(?:CMM\s*Programming)")
    cmm_run_hr  = sum_time(r"(?:CMM\s*Run\s*Time)\b") + sum_time(r"(?:CMM\s*Run\s*Time\s*min)")
    fair_hr     = sum_time(r"(?:FAIR|ISIR|PPAP)")
    srcinsp_hr  = sum_time(r"(?:Source\s*Inspection)")

    inspection_components: dict[str, float] = {
        "in_process": float(inproc_hr or 0.0),
        "final": float(final_hr or 0.0),
        "cmm_programming": float(cmm_prog_hr or 0.0),
        "cmm_run": float(cmm_run_hr or 0.0),
        "fair": float(fair_hr or 0.0),
        "source": float(srcinsp_hr or 0.0),
    }
    inspection_adjustments: dict[str, float] = {}
    inspection_hr_total = sum(inspection_components.values())

    inspection_cost = inspection_hr_total * rates["InspectionRate"]

    # Consumables & utilities
    spindle_hr = (eff(milling_hr) + turning_hr + wedm_hr + sinker_hr + surf_grind_hr + jig_grind_hr + odid_grind_hr)
    consumables_hr_cost = (
        params["MillingConsumablesPerHr"]    * eff(milling_hr) +
        params["TurningConsumablesPerHr"]    * turning_hr +
        params["EDMConsumablesPerHr"]        * (wedm_hr + sinker_hr) +
        params["GrindingConsumablesPerHr"]   * (surf_grind_hr + jig_grind_hr + odid_grind_hr) +
        params["InspectionConsumablesPerHr"] * inspection_hr_total
    )
    utilities_cost   = params["UtilitiesPerSpindleHr"] * spindle_hr
    consumables_flat = float(params["ConsumablesFlat"] or 0.0)

    # Saw / Waterjet
    sawing_hr = sum_time(r"(?:Sawing|Waterjet|Blank\s*Prep)")
    saw_cost  = sawing_hr * rates["SawWaterjetRate"]

    # Assembly / hardware
    assembly_hr   = sum_time(r"(?:Assembly|Manual\s*Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)")
    assembly_cost = assembly_hr * rates["AssemblyRate"]
    hardware_cost = num(r"(?:Hardware|BOM\s*Cost|Fasteners|Bearings|CFM\s*Hardware)")

    toolmaker_support_hr = sum_time(r"(?:Tool\s*(?:&|and)\s*Die\s*Maker|Toolmaker|Tool\s*(?:&|and)\s*Die\s*Support|Tooling\s*Support|Tool\s*Room)")
    toolmaker_support_rate = float(rates.get("ToolmakerSupportRate", rates.get("FixtureBuildRate", rates.get("EngineerRate", 0.0))))
    toolmaker_support_cost = toolmaker_support_hr * toolmaker_support_rate

    # Outsourced vendors
    heat_treat_cost  = num(r"(?:Outsourced\s*Heat\s*Treat|Heat\s*Treat\s*Cost)")
    coating_cost     = num(r"(?:Plating|Coating|Anodize|Black\s*Oxide|DLC|PVD|CVD)")
    passivation_cost = num(r"(?:Passivation|Cleaning\s*Vendor)")
    outsourced_costs = heat_treat_cost + coating_cost + passivation_cost
    include_outsourced_pass = _should_include_outsourced_pass(outsourced_costs, geo_context)

    # Packaging / logistics
    packaging_hr      = sum_time(r"(?:Packaging|Boxing|Crating\s*Labor)")
    crate_nre_cost    = num(r"(?:Custom\s*Crate\s*NRE)")
    packaging_mat     = num(r"(?:Packaging\s*Materials|Foam|Trays)")
    shipping_pct_of_material = float(params.get("ShippingPctOfMaterial", 0.15) or 0.0)
    shipping_cost_default = round(material_direct_cost * shipping_pct_of_material, 2) if shipping_pct_of_material else 0.0
    shipping_cost     = float(shipping_cost_default)
    insurance_pct     = num_pct(r"(?:Insurance|Liability\s*Adder)", params["InsurancePct"])
    packaging_cost    = packaging_hr * rates["AssemblyRate"] + crate_nre_cost + packaging_mat
    packaging_flat_base = float((crate_nre_cost or 0.0) + (packaging_mat or 0.0))
    shipping_basis_desc = "Outbound freight & logistics"
    shipping_cost_base = float(shipping_cost)

    # EHS / compliance
    ehs_hr   = sum_time(r"(?:EHS|Compliance|Training|Waste\s*Handling)")
    ehs_cost = ehs_hr * rates["InspectionRate"]

    raw_holes = geo_context.get("hole_diams_mm") or []
    hole_diams_list: list[float] = []
    for _d in raw_holes:
        val = _coerce_float_or_none(_d)
        if val is None:
            try:
                val = float(_d)
            except Exception:
                continue
        hole_diams_list.append(float(val))
    if hole_diams_list:
        geo_context["hole_diams_mm"] = hole_diams_list

    hole_groups_geo = _clean_hole_groups(geo_context.get("GEO_Hole_Groups"))
    if hole_groups_geo is not None:
        geo_context["GEO_Hole_Groups"] = hole_groups_geo

    hole_count_override = _coerce_float_or_none(ui_vars.get("Hole Count (override)"))
    avg_hole_diam_override = _coerce_float_or_none(ui_vars.get("Avg Hole Diameter (mm)"))
    if (not hole_diams_list) and hole_count_override and hole_count_override > 0:
        avg_val = float(avg_hole_diam_override or 0.0)
        if avg_val <= 0:
            avg_val = float(_coerce_float_or_none(geo_context.get("avg_hole_diameter_mm")) or 5.0)
        hole_diams_list = [float(avg_val)] * int(min(1000, max(1, round(hole_count_override))))
        geo_context.setdefault("hole_diams_mm", hole_diams_list)
        geo_context.setdefault("hole_count", int(max(1, round(hole_count_override))))

    thickness_for_drill = _coerce_float_or_none(geo_context.get("thickness_mm")) or 0.0
    if thickness_for_drill <= 0:
        thickness_in_ui = _coerce_float_or_none(ui_vars.get("Thickness (in)"))
        if thickness_in_ui and thickness_in_ui > 0:
            thickness_for_drill = float(thickness_in_ui) * 25.4

    selected_material_name: str | None = None
    if isinstance(material_detail_for_breakdown, Mapping):
        for key in ("material_name", "material", "material_key"):
            raw_value = material_detail_for_breakdown.get(key)
            if raw_value is None:
                continue
            candidate = str(raw_value).strip()
            if candidate:
                selected_material_name = candidate
                break
    if selected_material_name is None and isinstance(getattr(quote_state, "effective", None), Mapping):
        try:
            effective_material = quote_state.effective.get("material")  # type: ignore[union-attr]
        except Exception:
            effective_material = None
        if effective_material:
            candidate = str(effective_material).strip()
            if candidate:
                selected_material_name = candidate

    selected_lookup = str(material_selection.get("material_lookup") or "").strip()

    drill_material_source = (
        selected_material_name
        or geo_context.get("material")
        or material_name
        or "Steel"
    )
    drill_material_source = str(drill_material_source).strip()
    drill_material_lookup = (
        selected_lookup
        or (
            _normalize_lookup_key(drill_material_source)
            if drill_material_source
            else ""
        )
    )
    drill_material_group: str | None = None
    if material_selection.get("material_group"):
        group_val = str(material_selection["material_group"]).strip()
        if group_val:
            drill_material_group = group_val
    drill_material_key = (
        drill_material_group
        or drill_material_lookup
        or (drill_material_source or "")
    )
    drill_material_key = drill_material_lookup or drill_material_source
    speeds_feeds_raw = _resolve_speeds_feeds_path(params, ui_vars)
    speeds_feeds_path: str | None = None
    speeds_feeds_table: pd.DataFrame | None = None
    raw_path_text = str(speeds_feeds_raw).strip() if speeds_feeds_raw else ""
    if raw_path_text:
        try:
            expanded_text = _clean_path_text(raw_path_text)
            resolved_candidate = Path(expanded_text).expanduser()
        except Exception as exc:
            logger.warning(
                "Invalid speeds/feeds CSV path %r: %s; using legacy drilling heuristics for this quote.",
                raw_path_text,
                exc,
            )
            speeds_feeds_path = raw_path_text
            _record_red_flag(
                f"Speeds/feeds CSV path invalid ({raw_path_text}) — using legacy drilling heuristics."
            )
        else:
            speeds_feeds_path = _stringify_resolved_path(resolved_candidate)
            if resolved_candidate.is_file():
                speeds_feeds_table = _load_speeds_feeds_table(str(resolved_candidate))
                if speeds_feeds_table is None:
                    logger.warning(
                        "Failed to load speeds/feeds CSV at %s; using legacy drilling heuristics for this quote.",
                        resolved_candidate,
                    )
                    _record_red_flag(
                        f"Failed to load speeds/feeds CSV at {resolved_candidate} — using legacy drilling heuristics."
                    )
                else:
                    speeds_feeds_path = _stringify_resolved_path(resolved_candidate)
            else:
                logger.warning(
                    "Speeds/feeds CSV not found at %s; using legacy drilling heuristics for this quote.",
                    resolved_candidate,
                )
                _record_red_flag(
                    f"Speeds/feeds CSV not found at {resolved_candidate} — using legacy drilling heuristics."
                )
    else:
        logger.warning(
            "Speeds/feeds CSV path not configured; using legacy drilling heuristics for this quote."
        )
        _record_red_flag("Speeds/feeds CSV not configured — using legacy drilling heuristics.")
    speeds_feeds_loaded_flag = speeds_feeds_table is not None

    if speeds_feeds_table is not None and drill_material_lookup:
        table_group = _lookup_material_group_from_table(
            speeds_feeds_table,
            drill_material_lookup,
        )
        if table_group:
            drill_material_group = table_group
    if drill_material_group:
        drill_material_key = drill_material_group
        geo_context.setdefault("material_group", drill_material_group)
        material_selection["material_group"] = drill_material_group
    drill_material_lookup_final = (
        _normalize_lookup_key(drill_material_key) if drill_material_key else ""
    )
    if drill_material_lookup_final and not material_selection.get("material_lookup"):
        material_selection["material_lookup"] = drill_material_lookup_final
    drill_material_display = material_selection.get("canonical_material") or MATERIAL_DISPLAY_BY_KEY.get(
        drill_material_lookup_final,
        drill_material_key or drill_material_source or "",
    )
    if speeds_feeds_table is not None and (
        not drill_material_display or drill_material_display == drill_material_key
    ):
        lookup_for_table = drill_material_lookup_final or drill_material_lookup
        alt_label = _material_label_from_table(
            speeds_feeds_table,
            drill_material_key,
            lookup_for_table,
        )
        if alt_label:
            drill_material_display = alt_label
    if not drill_material_display:
        drill_material_display = str(material_name or drill_material_source or "")
    drill_material_display = str(drill_material_display or "").strip()
    if drill_material_display:
        material_selection["canonical_material"] = drill_material_display
    machine_params_default = _machine_params_from_params(params)
    drill_overhead_default = _drill_overhead_from_params(params)
    speeds_feeds_warnings: list[str] = []
    thickness_mm = float(thickness_for_drill or 0.0)
    thickness_in = thickness_mm / 25.4 if thickness_mm else 0.0
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Estimating drilling hours with thickness %.3f mm (%.3f in)",
            thickness_mm,
            thickness_mm / 25.4 if thickness_mm else 0.0,
        )
    drill_debug_lines: list[str] = []
    drill_debug_summary: dict[str, dict[str, Any]] = {}
    speeds_feeds_row: Mapping[str, Any] | None = None  # ensure defined in outer scope
    selected_op_name: str = "drill"  # default for debug display
    avg_dia_in = 0.0
    material_display_for_debug: str = ""

    if not material_display_for_debug:
        candidate_display = (
            material_selection.get("canonical_material")
            or material_selection.get("material_display")
            or material_selection.get("input_material")
            or material_selection.get("material")
        )
        if candidate_display:
            material_display_for_debug = str(candidate_display).strip()
    if not material_display_for_debug:
        material_display_for_debug = str(drill_material_display or "").strip()

    drill_hr = estimate_drilling_hours(
        hole_diams_mm=hole_diams_list,
        thickness_in=thickness_in,
        mat_key=drill_material_key or "",
        hole_groups=hole_groups_geo,
        speeds_feeds_table=speeds_feeds_table,
        machine_params=machine_params_default,
        overhead_params=drill_overhead_default,
        warnings=speeds_feeds_warnings,
        debug_lines=drill_debug_lines,
        debug_summary=drill_debug_summary,
    )
    if not math.isfinite(drill_hr) or drill_hr < 0:
        drill_hr = 0.0

    drill_hr = min(drill_hr, 500.0)

    # Expose drilling estimator hours to downstream planner override logic.
    drill_estimator_hours_for_planner = float(drill_hr)

    if material_display_for_debug:
        canonical_material_text = material_display_for_debug
        updated_debug_lines: list[str] = []
        for line in drill_debug_lines:
            if line.startswith("Drill calc") and "mat=" in line:
                updated_debug_lines.append(
                    re.sub(
                        r"(mat=)[^,]+",
                        rf"\1{canonical_material_text}",
                        line,
                        count=1,
                    )
                )
            else:
                updated_debug_lines.append(line)
        drill_debug_lines[:] = updated_debug_lines

    drill_debug_line: str | None = None
    if speeds_feeds_row and avg_dia_in > 0:
        key_map = {
            str(k).strip().lower().replace("-", "_").replace(" ", "_"): k
            for k in speeds_feeds_row.keys()
        }

        def _row_float(*names: str) -> float | None:
            for name in names:
                actual = key_map.get(name)
                if actual is None:
                    continue
                val = _coerce_float_or_none(speeds_feeds_row.get(actual))
                if val is not None:
                    return float(val)
            return None

        sfm_val = _row_float(
            "sfm_start",
            "sfm",
            "surface_ft_min",
            "surface_feet_per_min",
        )
        ipr_val: float | None = None
        ipr_candidates: list[tuple[float, float]] = []
        for norm_key, actual in key_map.items():
            if not norm_key.startswith("fz_ipr_"):
                continue
            raw_val = _coerce_float_or_none(speeds_feeds_row.get(actual))
            if raw_val is None:
                continue
            suffix = norm_key[len("fz_ipr_") :]
            if suffix.endswith("in"):
                suffix = suffix[:-2]
            suffix = suffix.replace("__", "_")
            try:
                dia_val = float(suffix.replace("_", "."))
            except Exception:
                continue
            ipr_candidates.append((dia_val, float(raw_val)))
        if ipr_candidates:
            ipr_val = min(ipr_candidates, key=lambda pair: abs(pair[0] - avg_dia_in))[1]
        if ipr_val is None:
            ipr_val = _row_float("ipr", "feed_per_rev", "feed_rev", "fz_ipr")

        rpm_val: float | None = None
        if sfm_val and avg_dia_in > 0:
            rpm_val = (float(sfm_val) * 12.0) / (math.pi * avg_dia_in)

        ipm_val = None
        if rpm_val and ipr_val:
            ipm_val = rpm_val * ipr_val
        if ipm_val is None:
            ipm_val = _row_float("linear_cut_rate_ipm", "feed_ipm", "feed_rate_ipm")

        debug_bits: list[str] = []
        if rpm_val and rpm_val > 0:
            debug_bits.append(f"{rpm_val:.0f} RPM")
        if ipm_val and ipm_val > 0:
            debug_bits.append(f"{ipm_val:.1f} IPM")
        elif ipr_val and ipr_val > 0:
            debug_bits.append(f"{ipr_val:.4f} IPR")

        if debug_bits:
            avg_mm = avg_dia_in * 25.4
            op_display = selected_op_name.replace("_", " ")
            drill_debug_line = (
                f"CSV drill feeds ({op_display} {material_group}): "
                f"{', '.join(debug_bits)} @ Ø{avg_mm:.1f} mm"
            )

    if drill_debug_line:
        drill_debug_lines.append(drill_debug_line)

    drilling_meta: dict[str, Any] = {
        "material_key": drill_material_key or "",
        "material_source": material_selection.get("input_material")
        or drill_material_source
        or "",
        "material_lookup": drill_material_lookup_final or selected_lookup,
        "speeds_feeds_loaded": speeds_feeds_loaded_flag,
    }
    final_group = material_selection.get("material_group") or drill_material_group
    if final_group:
        drilling_meta["material_group"] = final_group
    final_material_display = material_selection.get("canonical_material") or drill_material_display
    if final_material_display:
        drilling_meta["material"] = final_material_display
        drilling_meta["material_display"] = final_material_display
    if speeds_feeds_path:
        drilling_meta["speeds_feeds_path"] = speeds_feeds_path
    if drill_debug_line:
        drilling_meta["speeds_feeds_debug"] = drill_debug_line
    if drill_debug_summary:
        drilling_meta["debug_summary"] = {
            key: dict(value) for key, value in drill_debug_summary.items()
        }
        if not drill_material_display:
            for summary in drill_debug_summary.values():
                mat_val = str(summary.get("material") or "").strip()
                if mat_val:
                    drilling_meta["material"] = mat_val
                    drilling_meta["material_display"] = mat_val
                    break

    for warning_text in speeds_feeds_warnings:
        _record_red_flag(warning_text)
    if quote_state is not None:
        try:
            guard_ctx = quote_state.guard_context
        except Exception:
            guard_ctx = None
        if isinstance(guard_ctx, dict):
            if speeds_feeds_path:
                guard_ctx.setdefault("speeds_feeds_path", speeds_feeds_path)
            try:
                loaded_flag = bool(
                    (speeds_feeds_table is not None)
                    and (getattr(speeds_feeds_table, "empty", False) is False)
                )
            except Exception:
                try:
                    loaded_flag = speeds_feeds_table is not None and len(speeds_feeds_table) > 0
                except Exception:
                    loaded_flag = False
            guard_ctx["speeds_feeds_loaded"] = loaded_flag
            if red_flag_messages:
                ctx_flags = guard_ctx.setdefault("red_flags", [])
                for msg in red_flag_messages:
                    if msg not in ctx_flags:
                        ctx_flags.append(msg)
    drill_rate = float(rates.get("DrillingRate") or rates.get("MillingRate", 0.0) or 0.0)
    if drill_estimator_hours_for_planner > 0:
        override_minutes = drill_estimator_hours_for_planner * 60.0
        override_cost = drill_estimator_hours_for_planner * drill_rate
        planner_drilling_override = {
            "minutes": float(override_minutes),
            "hours": float(drill_estimator_hours_for_planner),
            "rate": float(drill_rate),
            "machine_cost": float(override_cost),
            "labor_cost": 0.0,
            "total_cost": float(override_cost),
        }
    hole_count_geo = _coerce_float_or_none(geo_context.get("hole_count"))
    hole_count_for_tripwire = 0
    if hole_count_geo and hole_count_geo > 0:
        hole_count_for_tripwire = int(round(hole_count_geo))
    elif hole_diams_list:
        hole_count_for_tripwire = len(hole_diams_list)
    elif hole_count_override and hole_count_override > 0:
        hole_count_for_tripwire = int(round(hole_count_override))
    geo_context["hole_count"] = hole_count_for_tripwire
    avg_tap_min = 0.20
    avg_cbore_min = 0.15
    avg_csk_min = 0.12
    tap_class_counts_geo = geo_context.get("tap_class_counts") if isinstance(geo_context.get("tap_class_counts"), dict) else {}
    tapping_minutes_total = 0.0
    if tap_class_counts_geo:
        for cls_key, qty_val in tap_class_counts_geo.items():
            try:
                qty_int = int(qty_val)
            except Exception:
                continue
            if qty_int <= 0:
                continue
            key = str(cls_key).lower()
            if key == "npt":
                minutes_per = TAP_MINUTES_BY_CLASS.get("pipe", TAP_MINUTES_BY_CLASS.get("medium", 0.3))
            else:
                minutes_per = TAP_MINUTES_BY_CLASS.get(key, TAP_MINUTES_BY_CLASS.get("medium", 0.3))
            tapping_minutes_total += qty_int * float(minutes_per)
    if tapping_minutes_total > 0:
        tapping_hr = tapping_minutes_total / 60.0
    else:
        tapping_hr = (tap_qty_geo * avg_tap_min) / 60.0
    cbore_hr = (cbore_qty_geo * avg_cbore_min) / 60.0
    csk_hr = (csk_qty_geo * avg_csk_min) / 60.0
    tap_minutes_hint_val = _coerce_float_or_none(geo_context.get("tap_minutes_hint"))
    if tap_minutes_hint_val and tap_minutes_hint_val > 0:
        tapping_hr = max(tapping_hr, float(tap_minutes_hint_val) / 60.0)
    cbore_minutes_hint_val = _coerce_float_or_none(geo_context.get("cbore_minutes_hint"))
    if cbore_minutes_hint_val and cbore_minutes_hint_val > 0:
        cbore_hr = max(cbore_hr, float(cbore_minutes_hint_val) / 60.0)
    csk_minutes_hint_val = _coerce_float_or_none(geo_context.get("csk_minutes_hint"))
    if csk_minutes_hint_val and csk_minutes_hint_val > 0:
        csk_hr = max(csk_hr, float(csk_minutes_hint_val) / 60.0)
    tapping_hr_floor = float(tapping_hr)
    cbore_hr_floor = float(cbore_hr)
    csk_hr_floor = float(csk_hr)
    tapping_hr_rounded = round(tapping_hr, 3)
    cbore_hr_rounded = round(cbore_hr, 3)
    csk_hr_rounded = round(csk_hr, 3)

    edge_len_in = _coerce_float_or_none(geo_context.get("edge_len_in"))
    if edge_len_in is None:
        edge_len_in = _coerce_float_or_none(geo_context.get("edge_length_in"))
    if edge_len_in is None:
        edge_len_mm = _coerce_float_or_none(geo_context.get("profile_length_mm"))
        if edge_len_mm is not None:
            edge_len_in = float(edge_len_mm) / 25.4
    edge_len_in = float(edge_len_in or 0.0)
    deburr_ipm_edge = 1000.0
    deburr_edge_hr_raw = edge_len_in / deburr_ipm_edge if edge_len_in else 0.0
    sec_per_hole = 5.0
    hole_count_total = hole_count_for_tripwire if hole_count_for_tripwire else _int_from(hole_count_geo)
    deburr_holes_hr_raw = (float(hole_count_total) * sec_per_hole) / 3600.0 if hole_count_total else 0.0
    deburr_auto_hr_raw = deburr_edge_hr_raw + deburr_holes_hr_raw
    deburr_manual_hr_float = float(deburr_manual_hr or 0.0)
    deburr_total_hr_raw = deburr_manual_hr_float + deburr_auto_hr_raw
    deburr_total_hr = round(deburr_total_hr_raw, 3)
    deburr_auto_hr = round(deburr_auto_hr_raw, 3)
    deburr_edge_hr = round(deburr_edge_hr_raw, 3)
    deburr_holes_hr = round(deburr_holes_hr_raw, 3)
    deburr_manual_hr_rounded = round(deburr_manual_hr_float, 3)
    deburr_total_hr_floor = max(0.0, float(deburr_total_hr_raw))

    # ---- roll-ups ------------------------------------------------------------
    inspection_hr_total = inproc_hr + final_hr + cmm_prog_hr + cmm_run_hr + fair_hr + srcinsp_hr

    process_costs = {
        "milling": milling_cost,
        "turning": turning_cost,
        "wire_edm": wedm_cost,
        "sinker_edm": sinker_cost,
        "grinding": grinding_cost,
        "lapping_honing": lap_cost,
        "finishing_deburr": finishing_cost,
        "inspection": inspection_cost,
        "saw_waterjet": saw_cost,
        "assembly": assembly_cost,
        "toolmaker_support": toolmaker_support_cost,
        "packaging": packaging_cost,
        "ehs_compliance": ehs_cost,
    }

    existing_drill_cost = float(process_costs.get("drilling", 0.0) or 0.0)
    existing_drill_hr = existing_drill_cost / drill_rate if drill_rate else 0.0
    baseline_drill_hr = max(existing_drill_hr, float(drill_hr or 0.0))
    try:
        min_sec_per_hole = 9.0
        drill_hr_floor = (float(hole_count_for_tripwire or 0) * min_sec_per_hole) / 3600.0
        baseline_drill_hr = max(baseline_drill_hr, drill_hr_floor)
    except Exception:
        pass
    baseline_drill_hr = round(baseline_drill_hr, 3)
    deburr_cost = deburr_total_hr * finishing_rate

    legacy_process_costs = {
        "drilling": baseline_drill_hr * drill_rate,
        "tapping": tapping_hr_rounded * drill_rate,
        "counterbore": cbore_hr_rounded * drill_rate,
        "countersink": csk_hr_rounded * drill_rate,
        "deburr": deburr_cost,
    }

    process_meta = {
        "milling":          {"hr": eff(milling_hr), "rate": rates.get("MillingRate", 0.0)},
        "turning":          {"hr": eff(turning_hr), "rate": rates.get("TurningRate", 0.0)},
        "wire_edm":         {"hr": eff(wedm_hr),    "rate": rates.get("WireEDMRate", 0.0)},
        "sinker_edm":       {"hr": eff(sinker_hr),  "rate": rates.get("SinkerEDMRate", 0.0)},
        "grinding":         {"hr": eff(grinding_hr),"rate": rates.get("SurfaceGrindRate", 0.0)},
        "lapping_honing":   {"hr": lap_hr,          "rate": rates.get("LappingRate", 0.0)},
        "finishing_deburr": {"hr": finishing_misc_hr, "rate": finishing_rate},
        "inspection":       {
            "hr": inspection_hr_total,
            "rate": rates.get("InspectionRate", 0.0),
            "baseline_hr": inspection_hr_total,
            "components": inspection_components,
            "adjustments": inspection_adjustments,
        },
        "saw_waterjet":     {"hr": sawing_hr,       "rate": rates.get("SawWaterjetRate", 0.0)},
        "assembly":         {"hr": assembly_hr,     "rate": rates.get("AssemblyRate", 0.0)},
        "toolmaker_support":  {"hr": toolmaker_support_hr, "rate": toolmaker_support_rate},
        "packaging":        {"hr": packaging_hr,    "rate": rates.get("PackagingRate", rates.get("AssemblyRate", 0.0))},
        "ehs_compliance":   {"hr": ehs_hr,          "rate": rates.get("InspectionRate", 0.0)},
    }
    legacy_per_process_keys: set[str] = set(process_meta.keys())
    legacy_process_meta = {
        "drilling": {"hr": baseline_drill_hr, "rate": drill_rate},
        "tapping": {"hr": tapping_hr_rounded, "rate": drill_rate},
        "counterbore": {"hr": cbore_hr_rounded, "rate": drill_rate},
        "countersink": {"hr": csk_hr_rounded, "rate": drill_rate},
        "deburr": {"hr": deburr_total_hr, "rate": finishing_rate},
    }
    if deburr_auto_hr:
        legacy_process_meta["deburr"]["auto_calc_hr"] = deburr_auto_hr
    if deburr_manual_hr_rounded:
        legacy_process_meta["deburr"]["manual_hr"] = deburr_manual_hr_rounded
    if deburr_edge_hr:
        legacy_process_meta["deburr"]["edge_hr"] = deburr_edge_hr
    if deburr_holes_hr:
        legacy_process_meta["deburr"]["hole_touch_hr"] = deburr_holes_hr

    legacy_baseline_had_values = any(
        float(value or 0.0) > 0.0 for value in legacy_process_costs.values()
    ) or any(
        float(meta.get("hr", 0.0) or 0.0) > 0.0 for meta in legacy_process_meta.values()
    )

    planner_meta_keys: set[str] = set()

    meta_lookup: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in process_meta.items() if isinstance(value, Mapping)
    }

    planner_meta_keys: set[str] = set()
    if not used_planner:
        for key, value in legacy_process_costs.items():
            process_costs[key] = float(value)
        for key, meta in legacy_process_meta.items():
            process_meta[key] = dict(meta)

    legacy_baseline_ignored = False

    inspection_meta_entry = process_meta.get("inspection")
    if isinstance(inspection_meta_entry, Mapping):
        inspection_meta_entry["components"] = inspection_components
        inspection_meta_entry["adjustments"] = inspection_adjustments
        inspection_meta_entry.setdefault("baseline_hr", float(inspection_hr_total))
        if os.environ.get("DEBUG_INSPECTION_META") == "1":
            import pdb

            pdb.set_trace()

    for key, meta in list(process_meta.items()):
        if not isinstance(meta, dict):
            continue
        rate_val = float(meta.get("rate", 0.0) or 0.0)
        hr_val = float(meta.get("hr", 0.0) or 0.0)
        meta["base_extra"] = process_costs.get(key, 0.0) - hr_val * rate_val

    # Track process keys populated by the planner so we don't double-count
    planner_meta_keys: set[str] = set()
    allowed_process_hour_keys: set[str] = set()
    for key, meta in process_meta.items():
        if not isinstance(meta, dict):
            continue
        if used_planner and key in legacy_per_process_keys and key not in planner_meta_keys:
            continue
        allowed_process_hour_keys.add(key)

    process_hours_baseline = {
        key: float(process_meta.get(key, {}).get("hr", 0.0) or 0.0)
        for key in allowed_process_hour_keys
    }
    process_costs_baseline = {k: float(v) for k, v in process_costs.items()}

    def _default_pass_meta(shipping_basis_desc: str) -> dict[str, dict[str, str]]:
        return {
            "Material": {"basis": "Stock / raw material"},
            HARDWARE_PASS_LABEL: {"basis": "Pass-through hardware / BOM"},
            "Outsourced Vendors": {"basis": "Outside processing vendors"},
            "Shipping": {"basis": shipping_basis_desc},
            "Consumables /Hr": {"basis": "Machine & inspection hours $/hr"},
            "Utilities": {"basis": "Spindle/inspection hours $/hr"},
            "Consumables": {"basis": "Fixed shop supplies"},
            "Packaging Flat": {"basis": "Packaging materials & crates"},
        }

    pass_meta = _default_pass_meta(shipping_basis_desc)

    def _build_pass_through(
        planner_directs,
        *,
        material_direct_cost,
        shipping_cost,
        hardware_cost,
        outsourced_costs,
        include_outsourced_pass,
        utilities_cost,
        consumables_flat,
        packaging_flat_base,
        material_scrap_credit_applied,
    ) -> dict[str, float]:
        def _on(k: str) -> bool:
            return bool((planner_directs or {}).get(k))

        pass_through: dict[str, float] = {
            _canonical_pass_label("Shipping"): float(shipping_cost),
        }
        if _on("hardware") and float(hardware_cost) > 0:
            pass_through[_canonical_pass_label(LEGACY_HARDWARE_PASS_LABEL)] = float(
                hardware_cost
            )
        if (
            _on("outsourced")
            and include_outsourced_pass
            and float(outsourced_costs) > 0
        ):
            pass_through["Outsourced Vendors"] = float(outsourced_costs)
        if _on("utilities") and float(utilities_cost) > 0:
            pass_through["Utilities"] = float(utilities_cost)
        if _on("consumables_flat") and float(consumables_flat) > 0:
            pass_through["Consumables"] = float(consumables_flat)
        if _on("packaging_flat") and float(packaging_flat_base) > 0:
            pass_through["Packaging Flat"] = float(packaging_flat_base)

        pass_through = _canonicalize_pass_through_map(pass_through)

        credit = float(material_scrap_credit_applied or 0.0)
        if credit > 0:
            pass_through["Material Scrap Credit"] = -credit
        return pass_through
    if material_scrap_credit_applied:
        pass_meta["Material Scrap Credit"] = {"basis": "Scrap / remnant credit"}

    mat_source = material_detail_for_breakdown.get("source")
    if mat_source:
        mat_symbol = material_detail_for_breakdown.get("symbol")
        if mat_symbol:
            pass_meta["Material"]["basis"] = f"{mat_symbol} via {mat_source}"
        else:
            pass_meta["Material"]["basis"] = f"Source: {mat_source}"
    quote_state.material_source = pass_meta.get("Material", {}).get("basis") or mat_source or "shop defaults"

    planner_directs = (
        (process_plan_summary.get("plan") or {}).get("directs") or {}
        if isinstance(process_plan_summary, dict)
        else {}
    )
    pass_through = _build_pass_through(
        planner_directs,
        material_direct_cost=material_direct_cost,
        shipping_cost=shipping_cost,
        hardware_cost=hardware_cost,
        outsourced_costs=outsourced_costs,
        include_outsourced_pass=include_outsourced_pass,
        utilities_cost=utilities_cost,
        consumables_flat=consumables_flat,
        packaging_flat_base=packaging_flat_base,
        material_scrap_credit_applied=material_scrap_credit_applied,
    )
    pass_through[_canonical_pass_label("Material")] = float(material_direct_cost)
    pass_through_baseline = {k: float(v) for k, v in pass_through.items()}

    fix_detail = nre_detail.get("fixture", {})
    fixture_plan_desc = None
    try:
        fb = float(fixture_build_hr)
    except Exception:
        fb = 0.0
    if fb:
        fixture_plan_desc = f"{fb:.2f} hr build"
    strategy = fix_detail.get("strategy") if isinstance(fix_detail, dict) else None
    if isinstance(strategy, str) and strategy.strip():
        if fixture_plan_desc:
            fixture_plan_desc = f"{fixture_plan_desc} ({strategy.strip()})"
        else:
            fixture_plan_desc = strategy.strip()

    fixture_build_hr_base = float(fixture_build_hr or 0.0)
    cmm_minutes_base = float((cmm_run_hr or 0.0) * 60.0)
    inproc_hr_base = float(inproc_hr or 0.0)
    packaging_hr_base = float(packaging_hr or 0.0)
    fai_flag_base = False
    if isinstance(ui_vars, dict):
        raw_fai = ui_vars.get("FAIR Required")
        fai_flag_base = _coerce_checkbox_state(raw_fai, False)

    def _clean_stock_entry(entry: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for k, v in entry.items():
            key = str(k)
            if isinstance(v, (str, int, float, bool)) or v is None:
                cleaned[key] = v
            else:
                try:
                    cleaned[key] = float(v)
                except Exception:
                    cleaned[key] = str(v)
        return cleaned

    stock_catalog_param = params.get("StockCatalog")
    stock_catalog: list[dict[str, Any]] = []
    if isinstance(stock_catalog_param, (list, tuple)):
        for entry in list(stock_catalog_param)[:12]:
            if isinstance(entry, dict):
                stock_catalog.append(_clean_stock_entry(entry))

    machine_limits = {
        "max_rpm": float(_coerce_float_or_none(params.get("MachineMaxRPM")) or 0.0),
        "max_torque_nm": float(_coerce_float_or_none(params.get("MachineMaxTorqueNm")) or 0.0),
        "max_z_travel_mm": float(_coerce_float_or_none(params.get("MachineMaxZTravel_mm")) or 0.0),
    }

    bbox_info: dict[str, float] = {}
    bbox_src = (
        ("GEO-01_Length_mm", "length_mm"),
        ("GEO-02_Width_mm", "width_mm"),
        ("GEO-03_Height_mm", "height_mm"),
    )
    dims_for_stock: list[float] = []
    for src_key, dest_key in bbox_src:
        val = _coerce_float_or_none(geo_context.get(src_key))
        if val is not None:
            fval = float(val)
            bbox_info[dest_key] = fval
            if fval > 0:
                dims_for_stock.append(fval)
    if dims_for_stock:
        bbox_info["max_dim_mm"] = max(dims_for_stock)
        bbox_info["min_dim_mm"] = min(dims_for_stock)

    vol_cm3 = float(_coerce_float_or_none(geo_context.get("volume_cm3")) or 0.0)
    density_g_cc = float(_coerce_float_or_none(geo_context.get("density_g_cc")) or 0.0)
    part_mass_g_est = 0.0
    if vol_cm3 > 0 and density_g_cc > 0:
        part_mass_g_est = vol_cm3 * density_g_cc

    hole_groups_geo = geo_context.get("GEO_Hole_Groups")

    dfm_geo = {
        "min_wall_mm": _coerce_float_or_none(geo_context.get("GEO_MinWall_mm")),
        "thin_wall": bool(geo_context.get("GEO_ThinWall_Present")),
        "largest_plane_mm2": _coerce_float_or_none(geo_context.get("GEO_LargestPlane_Area_mm2")),
        "unique_normals": int(_coerce_float_or_none(geo_context.get("GEO_Setup_UniqueNormals")) or 0),
        "face_count": int(_coerce_float_or_none(geo_context.get("Feature_Face_Count")) or 0),
        "deburr_edge_len_mm": _coerce_float_or_none(geo_context.get("GEO_Deburr_EdgeLen_mm")),
    }

    features = {
        "qty": Qty,
        "max_dim_mm": max_dim,
        "setups": setups,
        "rough_hr": rough_hr,
        "semi_hr": semi_hr,
        "finish_hr": finish_hr,
        "milling_hr": milling_hr,
        "wedm_len_mm": GEO_wedm_len_mm,
        "volume_cm3": vol_cm3,
        "density_g_cc": density_g_cc,
        "scrap_pct": scrap_pct,
        "scrap_source": scrap_source_label,
        "material_cost_baseline": material_cost,
        "bbox_mm": bbox_info,
        "machine_limits": machine_limits,
        "stock_catalog": stock_catalog,
        "part_mass_g_est": part_mass_g_est,
        "dfm_geo": dfm_geo,
        "fixture_build_hr": float(fixture_build_hr or 0.0),
        "cmm_minutes": float((cmm_run_hr or 0.0) * 60.0),
        "in_process_inspection_hr": float(inproc_hr or 0.0),
        "packaging_hours": float(packaging_hr or 0.0),
        "packaging_flat_cost": float((crate_nre_cost or 0.0) + (packaging_mat or 0.0)),
        "fai_required": bool(fai_flag_base),
    }
    try:
        removal_g = _holes_removed_mass_g(geo_context)
    except Exception:
        removal_g = None
    if removal_g and removal_g > 0:
        features["material_removed_mass_g"] = float(removal_g)
        material_detail_for_breakdown.setdefault("material_removed_mass_g", float(removal_g))
    if removal_mass_g_est and removal_mass_g_est > 0:
        features["material_removed_mass_g_est"] = float(removal_mass_g_est)
    if hole_scrap_frac_est and hole_scrap_frac_est > 0:
        features["scrap_pct_holes_est"] = float(hole_scrap_frac_est)
        if hole_scrap_clamped_val:
            features["scrap_pct_holes_clamped"] = float(hole_scrap_clamped_val)
    # Safely include tap/cbore counts even if later sections define seeds
    try:
        _tq = int(tap_qty_geo)
    except Exception:
        _tq = int(_coerce_float_or_none(geo_context.get("tap_qty")) or 0)
    if _tq > 0:
        features["tap_qty_geo"] = _tq
    try:
        _cq = int(cbore_qty_geo)
    except Exception:
        _cq = int(_coerce_float_or_none(geo_context.get("cbore_qty")) or 0)
    if _cq > 0:
        features["cbore_qty_geo"] = _cq
    try:
        _sk = int(csk_qty_geo)
    except Exception:
        _sk = int(_coerce_float_or_none(geo_context.get("csk_qty")) or 0)
    if _sk > 0:
        features["csk_qty_geo"] = _sk
    if from_back_geo_flag:
        features["needs_back_face_geo"] = True
    if chart_source_geo:
        features["chart_source"] = chart_source_geo
    if chart_ops_geo:
        features["chart_ops"] = chart_ops_geo
    if chart_reconcile_geo:
        features["hole_chart_reconcile"] = chart_reconcile_geo
        features["hole_chart_agreement"] = bool(chart_reconcile_geo.get("agreement"))
        features["hole_chart_tap_qty"] = int(chart_reconcile_geo.get("tap_qty") or 0)
        features["hole_chart_cbore_qty"] = int(chart_reconcile_geo.get("cbore_qty") or 0)
        if chart_reconcile_geo.get("csk_qty") is not None:
            features["hole_chart_csk_qty"] = int(chart_reconcile_geo.get("csk_qty") or 0)
    if geo_context.get("inference_knobs"):
        features["inference_knobs"] = geo_context.get("inference_knobs")
    hole_tool_sizes = sorted({round(x, 2) for x in hole_diams_list}) if hole_diams_list else []
    features.update({
        "is_2d": bool(is_plate_2d),
        "hole_count": hole_count_for_tripwire,
        "hole_tool_sizes": hole_tool_sizes,
        "hole_groups": hole_groups_geo,
        "profile_length_mm": float(_coerce_float_or_none(geo_context.get("profile_length_mm")) or 0.0),
        "thickness_mm": float(_coerce_float_or_none(geo_context.get("thickness_mm")) or 0.0),
        "material_key": drill_material_lookup or (drill_material_source or material_name),
        "drilling_hr_baseline": baseline_drill_hr,
    })
    # Normalize tolerance inputs (optional). Prefer explicit geo_context input; otherwise
    # lightly harvest likely tolerance fields from UI labels.
    tolerance_inputs = None
    try:
        _tol_in = geo_context.get("tolerance_inputs")
        if isinstance(_tol_in, Mapping):
            tolerance_inputs = dict(_tol_in)
    except Exception:
        tolerance_inputs = None
    if tolerance_inputs is None and isinstance(ui_vars, Mapping):
        try:
            tol_like: dict[str, Any] = {}
            tol_key_re = re.compile(r"flatness|parallel|profile|runout|\btir\b|\bra\b|surface\s*finish|id\s*(ra|finish)|od\s*(ra|finish)", re.IGNORECASE)
            for label, value in ui_vars.items():
                key = str(label)
                if tol_key_re.search(key):
                    tol_like[key] = value
            if tol_like:
                tolerance_inputs = tol_like
        except Exception:
            pass
    if hole_count_override and hole_count_override > 0:
        features["hole_count_override"] = int(round(hole_count_override))
    if avg_hole_diam_override and avg_hole_diam_override > 0:
        features["avg_hole_diam_override_mm"] = float(avg_hole_diam_override)
    if tolerance_inputs:
        features["tolerance_inputs"] = tolerance_inputs
    if fix_detail:
        setup_hint = {}
        if fix_detail.get("strategy"):
            setup_hint["fixture_strategy"] = fix_detail.get("strategy")
        if fix_detail.get("build_hr"):
            setup_hint["fixture_build_hr"] = _coerce_float_or_none(fix_detail.get("build_hr"))
        if setup_hint:
            features["fixture_plan"] = setup_hint


    if _process_plan_job is not None:
        def _first_from_sources(
            patterns: tuple[str, ...],
            *sources: Mapping[str, Any],
            transform: Callable[[Any], T | None],
        ) -> T | None:
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                for label, raw_value in source.items():
                    key = str(label)
                    if any(re.search(pattern, key, re.IGNORECASE) for pattern in patterns):
                        try:
                            value = transform(raw_value)
                        except Exception:
                            value = None
                        if value is not None:
                            return value
            return None

        def _first_numeric(patterns: tuple[str, ...], *sources: Mapping[str, Any]) -> float | None:
            return _first_from_sources(
                patterns,
                *sources,
                transform=_coerce_float_or_none,
            )

        def _first_text(patterns: tuple[str, ...], *sources: Mapping[str, Any]) -> str | None:
            return _first_from_sources(
                patterns,
                *sources,
                transform=lambda raw: (str(raw).strip() or None),
            )

        def _count_from_ui(patterns: tuple[str, ...]) -> int:
            val = _first_numeric(patterns, ui_vars)
            if val is None:
                return 0
            try:
                qty = int(round(val))
            except Exception:
                return 0
            return qty if qty > 0 else 0

        def _mm_to_in(val: float | None) -> float | None:
            if val is None:
                return None
            try:
                return float(val) / 25.4
            except Exception:
                return None

        _known_planner_families = set(_PROCESS_PLANNERS.keys() if _PROCESS_PLANNERS else [])
        if not _known_planner_families:
            _known_planner_families = {
                "die_plate",
                "punch",
                "pilot_punch",
                "bushing_id_critical",
                "cam_or_hemmer",
                "flat_die_chaser",
                "pm_compaction_die",
                "shear_blade",
                "extrude_hone",
            }

        def _normalize_family(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip().lower()
            if not text:
                return None
            text = re.sub(r"[^a-z0-9]+", "_", text)
            synonyms = {
                "dieplates": "die_plate",
                "die_plate_plate": "die_plate",
                "plate": "die_plate",
                "punches": "punch",
                "pilot": "pilot_punch",
                "pilot_punches": "pilot_punch",
                "bushing": "bushing_id_critical",
                "ring_gauge": "bushing_id_critical",
                "ring_gauges": "bushing_id_critical",
                "cam": "cam_or_hemmer",
                "hemmer": "cam_or_hemmer",
                "hemming": "cam_or_hemmer",
                "flat_die_chasers": "flat_die_chaser",
                "pm_die": "pm_compaction_die",
                "pm_compaction": "pm_compaction_die",
                "shear": "shear_blade",
                "shear_blades": "shear_blade",
                "extrudehone": "extrude_hone",
                "extrude_hone_process": "extrude_hone",
            }
            text = synonyms.get(text, text)
            if text not in _known_planner_families and text.endswith("s"):
                singular = text[:-1]
                if singular in _known_planner_families:
                    text = singular
            return text if text in _known_planner_families else None

        def _collect_text_hints() -> list[str]:
            hints: list[str] = []

            def _push(val: Any) -> None:
                if val is None:
                    return
                if isinstance(val, str):
                    stripped = val.strip()
                    if stripped:
                        hints.append(stripped)
                elif isinstance(val, Mapping):
                    for sub_val in val.values():
                        _push(sub_val)
                elif isinstance(val, (list, tuple, set)):
                    for sub_val in val:
                        _push(sub_val)

            _push(geo_context.get("part_name"))
            _push(geo_context.get("title"))
            _push(geo_context.get("description"))
            _push(geo_context.get("notes"))
            _push(geo_context.get("chart_lines"))
            _push(geo_context.get("process_notes"))
            _push(geo_context.get("process_family"))
            if isinstance(ui_vars, Mapping):
                for label, value in ui_vars.items():
                    _push(label)
                    _push(value)
            return hints

        def _build_die_plate_inputs() -> dict[str, Any]:
            inputs: dict[str, Any] = {
                "material": material_name or default_material_display,
                "qty": int(Qty),
            }
            if plate_length_in_val and plate_width_in_val:
                inputs["plate_LxW"] = [float(plate_length_in_val), float(plate_width_in_val)]
            if plate_thickness_in_val and plate_thickness_in_val > 0:
                inputs["thickness"] = float(plate_thickness_in_val)

            flatness_spec = _first_numeric((r"flatness",), tolerance_inputs, ui_vars)
            if flatness_spec is not None:
                inputs["flatness_spec"] = float(flatness_spec)

            parallelism_spec = _first_numeric((r"parallel",), tolerance_inputs, ui_vars)
            if parallelism_spec is not None:
                inputs["parallelism_spec"] = float(parallelism_spec)

            profile_tol = _first_numeric((r"profile",), tolerance_inputs, ui_vars)
            if profile_tol is not None:
                inputs["profile_tol"] = float(profile_tol)

            corner_radius = _first_numeric((r"(corner|inside).*(radius|r)",), tolerance_inputs, ui_vars)
            if corner_radius is not None:
                inputs["window_corner_radius_req"] = float(corner_radius)

            hole_qty = max(
                _coerce_float_or_none(geo_context.get("hole_count")) or 0.0,
                _coerce_float_or_none(geo_context.get("hole_count_geo")) or 0.0,
                float(hole_count_for_tripwire or 0),
            )
            if hole_qty > 0:
                inputs["hole_count"] = int(round(hole_qty))

            windows_need_sharp = bool(geo_context.get("windows_need_sharp"))
            if not windows_need_sharp and GEO_wedm_len_mm:
                try:
                    windows_need_sharp = float(GEO_wedm_len_mm) > 0
                except Exception:
                    windows_need_sharp = bool(windows_need_sharp)
            if not windows_need_sharp:
                window_text = _first_text((r"sharp", r"wedm"), tolerance_inputs, ui_vars)
                if window_text:
                    lowered = window_text.lower()
                    if "sharp" in lowered or "wedm" in lowered:
                        windows_need_sharp = True
            inputs["windows_need_sharp"] = bool(windows_need_sharp)

            incoming_cut = str(geo_context.get("incoming_cut") or "").strip().lower()
            if not incoming_cut:
                incoming_text = _first_text(
                    (r"(incoming|blank).*(saw|water|flame|plasma|laser)", r"(saw|waterjet|flame|plasma|laser)"),
                    ui_vars,
                )
                if incoming_text:
                    lowered = incoming_text.lower()
                    if "water" in lowered:
                        incoming_cut = "waterjet"
                    elif any(token in lowered for token in ("flame", "oxy", "plasma")):
                        incoming_cut = "flame"
                    elif "laser" in lowered:
                        incoming_cut = "waterjet"
                    elif "saw" in lowered or "band" in lowered:
                        incoming_cut = "saw"
            if incoming_cut not in {"saw", "waterjet", "flame"}:
                incoming_cut = "saw"
            inputs["incoming_cut"] = incoming_cut

            stress_relief_risk = str(geo_context.get("stress_relief_risk") or "").strip().lower()
            if not stress_relief_risk:
                sr_text = _first_text((r"stress\s*relief", r"stress\s*risk"), ui_vars)
                if sr_text:
                    lowered = sr_text.lower()
                    if "high" in lowered:
                        stress_relief_risk = "high"
                    elif "med" in lowered:
                        stress_relief_risk = "med"
                    elif "low" in lowered:
                        stress_relief_risk = "low"
            if stress_relief_risk == "medium":
                stress_relief_risk = "med"
            if stress_relief_risk not in {"low", "med", "high"}:
                stress_relief_risk = "low"
            inputs["stress_relief_risk"] = stress_relief_risk

            marking_value: str | None = None
            mark_text = _first_text((r"mark", r"engrave"), ui_vars)
            if mark_text:
                lowered = mark_text.lower()
                if any(token in lowered for token in ("laser", "engrave", "etch")):
                    marking_value = "laser"
                elif "stamp" in lowered:
                    marking_value = "stamp"
                elif "none" in lowered:
                    marking_value = "none"
            if marking_value is None:
                mark_hours = _coerce_float_or_none(ui_vars.get("Laser Marking / Engraving Time"))
                if mark_hours and mark_hours > 0:
                    marking_value = "laser"
            if marking_value is None and isinstance(ui_vars, Mapping):
                for label, value in ui_vars.items():
                    if re.search(r"(laser\s*mark|engrave)", str(label), re.IGNORECASE):
                        if _coerce_checkbox_state(value, False):
                            marking_value = "laser"
                            break
            if marking_value is None:
                marking_value = "none"
            inputs["marking"] = marking_value

            hole_sets: list[dict[str, Any]] = []
            total_taps = 0
            if isinstance(tap_class_counts_geo, dict):
                for qty in tap_class_counts_geo.values():
                    num = _coerce_float_or_none(qty)
                    if num and num > 0:
                        try:
                            total_taps += int(round(num))
                        except Exception:
                            continue
            if total_taps <= 0 and tap_qty_geo:
                try:
                    total_taps = int(round(float(tap_qty_geo)))
                except Exception:
                    total_taps = 0
            if total_taps > 0:
                hole_sets.append({"type": "tapped", "qty": total_taps})

            dowel_press_qty = _count_from_ui((r"dowel.*press", r"press.*dowel"))
            if dowel_press_qty:
                hole_sets.append({"type": "dowel_press", "qty": dowel_press_qty})

            dowel_slip_qty = _count_from_ui((r"dowel.*slip", r"slip.*dowel"))
            if dowel_slip_qty:
                hole_sets.append({"type": "dowel_slip", "qty": dowel_slip_qty})

            if hole_sets:
                inputs["hole_sets"] = hole_sets

            return compact_dict(inputs)

        def _bbox_inches() -> tuple[float | None, float | None, float | None]:
            length = _mm_to_in(_coerce_float_or_none(geo_context.get("plate_length_mm")) or bbox_info.get("max_dim_mm"))
            width = _mm_to_in(_coerce_float_or_none(geo_context.get("plate_width_mm")) or bbox_info.get("min_dim_mm"))
            thickness = _mm_to_in(_coerce_float_or_none(geo_context.get("thickness_mm")) or bbox_info.get("min_dim_mm"))
            return length, width, thickness

        def _build_punch_inputs() -> dict[str, Any]:
            length_in, width_in, thickness_in = _bbox_inches()
            feature_candidates = [val for val in (width_in, thickness_in) if val and val > 0]
            min_feature = min(feature_candidates) if feature_candidates else None
            inputs: dict[str, Any] = {
                "material": material_name or default_material_display,
                "overall_length": length_in,
                "min_feature_width": min_feature,
                "min_inside_radius": _mm_to_in(_coerce_float_or_none(geo_context.get("min_inside_radius_mm"))),
                "profile_tol": _first_numeric((r"profile",), tolerance_inputs, ui_vars),
            }

            blind_relief_flag = bool(geo_context.get("blind_relief"))
            if not blind_relief_flag:
                blind_text = _first_text((r"blind\s*relief", r"blind\s*pocket"), ui_vars)
                if blind_text:
                    blind_relief_flag = True
            inputs["blind_relief"] = blind_relief_flag

            edge_text = _first_text((r"edge\s*(condition|prep)",), ui_vars)
            if edge_text:
                inputs["edge_condition"] = edge_text

            coating_text = _first_text((r"coat(ing)?",), ui_vars)
            if coating_text:
                inputs["coating"] = coating_text

            runout_val = _first_numeric((r"runout", r"tir"), tolerance_inputs, ui_vars)
            if runout_val is None:
                runout_val = _coerce_float_or_none(geo_context.get("runout_to_shank"))
            inputs["runout_to_shank"] = runout_val

            bearing_width = _first_numeric((r"bearing\s*(land|width)",), ui_vars)
            bearing_ra = _first_numeric((r"bearing.*ra", r"surface\s*finish"), tolerance_inputs, ui_vars)
            if bearing_width or bearing_ra:
                inputs["bearing_land_spec"] = compact_dict({"width": bearing_width, "Ra": bearing_ra})

            return compact_dict(inputs)

        def _build_bushing_inputs() -> dict[str, Any]:
            inputs: dict[str, Any] = {
                "material": material_name or default_material_display,
                "tight_od": not _coerce_checkbox_state(ui_vars.get("OD Not Critical"), False)
                if isinstance(ui_vars, Mapping)
                else True,
                "target_id_Ra": _first_numeric((r"ID\s*Ra", r"ID\s*finish"), tolerance_inputs, ui_vars),
            }
            wire_text = _first_text((r"wire", r"wedm"), ui_vars)
            if wire_text:
                lowered = wire_text.lower()
                inputs["create_id_by_wire_first"] = "wire" in lowered or "wedm" in lowered
            return compact_dict(inputs)

        def _build_cam_inputs() -> dict[str, Any]:
            inputs: dict[str, Any] = {
                "material": material_name or default_material_display,
                "profile_tol": _first_numeric((r"profile",), tolerance_inputs, ui_vars),
            }
            if geo_context.get("windows_need_sharp"):
                inputs["windows_need_sharp"] = True
            else:
                sharp_text = _first_text((r"sharp", r"wedm"), tolerance_inputs, ui_vars)
                if sharp_text and any(word in sharp_text.lower() for word in ("sharp", "wedm")):
                    inputs["windows_need_sharp"] = True
            return compact_dict(inputs)

        def _build_extrude_hone_inputs() -> dict[str, Any]:
            return compact_dict(
                {
                    "target_Ra": _first_numeric((r"target\s*ra", r"surface\s*finish"), tolerance_inputs, ui_vars)
                    or _coerce_float_or_none(geo_context.get("target_Ra")),
                }
            )

        def _derive_planner_hints(
            family: str | None, inputs: Mapping[str, Any] | None
        ) -> dict[str, Any]:
            if not family or not inputs or not _PROCESS_PLANNER_HELPERS:
                return {}

            choose_wire = _PROCESS_PLANNER_HELPERS.get("choose_wire_size")
            choose_skims = _PROCESS_PLANNER_HELPERS.get("choose_skims")
            needs_wedm = _PROCESS_PLANNER_HELPERS.get("needs_wedm_for_windows")

            def _float(value: Any) -> float | None:
                try:
                    if value is None:
                        return None
                    return float(value)
                except (TypeError, ValueError):
                    return None

            hints: dict[str, Any] = {}

            if family == "die_plate":
                windows_need_sharp = bool(inputs.get("windows_need_sharp"))
                corner_r = _float(inputs.get("window_corner_radius_req"))
                profile_tol = _float(inputs.get("profile_tol"))
                wedm_needed: bool | None = None
                if needs_wedm:
                    wedm_needed = bool(needs_wedm(windows_need_sharp, corner_r, profile_tol))
                    hints["windows_wedm_recommended"] = wedm_needed
                if choose_wire and (wedm_needed or wedm_needed is None):
                    hints["recommended_wire_in"] = float(choose_wire(corner_r, None))
                if choose_skims and profile_tol is not None:
                    skims = int(choose_skims(profile_tol))
                    hints["recommended_wire_passes"] = f"R+{skims}S"
                    hints["recommended_wire_skims"] = skims

            if family in {"punch", "pilot_punch"}:
                min_feature = _float(inputs.get("min_feature_width"))
                min_inside = _float(inputs.get("min_inside_radius"))
                profile_tol = _float(inputs.get("profile_tol"))
                if choose_wire:
                    hints["recommended_wire_in"] = float(choose_wire(min_inside, min_feature))
                if choose_skims and profile_tol is not None:
                    skims = int(choose_skims(profile_tol))
                    hints["recommended_wire_passes"] = f"R+{skims}S"
                    hints["recommended_wire_skims"] = skims

            if family == "cam_or_hemmer":
                profile_tol = _float(inputs.get("profile_tol"))
                windows_need_sharp = bool(inputs.get("windows_need_sharp"))
                wedm_needed: bool | None = None
                if needs_wedm:
                    wedm_needed = bool(needs_wedm(windows_need_sharp, None, profile_tol))
                    hints["wedm_preferred"] = wedm_needed
                if choose_wire and (wedm_needed or wedm_needed is None):
                    hints["recommended_wire_in"] = float(choose_wire(None, None))
                if choose_skims and profile_tol is not None:
                    skims = int(choose_skims(profile_tol))
                    hints["recommended_wire_passes"] = f"R+{skims}S"
                    hints["recommended_wire_skims"] = skims

            return compact_dict(hints)

        planner_family: str | None = None
        planner_inputs: dict[str, Any] | None = None

        explicit_family_sources = [
            geo_context.get("process_planner_family"),
            geo_context.get("planner_family"),
            geo_context.get("process_family"),
        ]
        if isinstance(ui_vars, Mapping):
            for label, value in ui_vars.items():
                normalized_label = str(label).strip().lower()
                if any(token in normalized_label for token in ("planner family", "process family")):
                    explicit_family_sources.append(value)
        for candidate in explicit_family_sources:
            planner_family = _normalize_family(candidate)
            if planner_family:
                break

        if planner_family == "die_plate" or (planner_family is None and is_plate_2d):
            planner_family = "die_plate"
            planner_inputs = _build_die_plate_inputs()

        if planner_family is None:
            text_hints = "\n".join(_collect_text_hints()).lower()
            keyword_map = [
                ("pilot_punch", ("pilot punch", "piloted punch")),
                ("punch", ("punch", "insert", "pierce insert", "form insert")),
                ("bushing_id_critical", ("bushing", "ring gauge", "guide bushing")),
                ("cam_or_hemmer", ("cam", "hemmer", "hemming")),
                ("flat_die_chaser", ("die chaser", "chaser")),
                ("pm_compaction_die", ("pm compaction", "compaction die")),
                ("shear_blade", ("shear blade", "shear knife", "shear die")),
                ("extrude_hone", ("extrude hone", "abrasive flow")),
            ]
            for family_key, keywords in keyword_map:
                if any(keyword in text_hints for keyword in keywords):
                    planner_family = family_key if family_key in _known_planner_families else None
                    if planner_family:
                        break

        if planner_family == "punch":
            planner_inputs = _build_punch_inputs()
        elif planner_family == "pilot_punch":
            planner_inputs = _build_punch_inputs()
            if planner_inputs is None:
                planner_inputs = {}
            planner_inputs = compact_dict(
                {**planner_inputs, "runout_to_shank": planner_inputs.get("runout_to_shank", 0.0003)}
            )
        elif planner_family == "bushing_id_critical":
            planner_inputs = _build_bushing_inputs()
        elif planner_family == "cam_or_hemmer":
            planner_inputs = _build_cam_inputs()
        elif planner_family == "flat_die_chaser" or planner_family == "pm_compaction_die":
            planner_inputs = {}
        elif planner_family == "shear_blade":
            planner_inputs = {"material": material_name or default_material_display}
        elif planner_family == "extrude_hone":
            planner_inputs = _build_extrude_hone_inputs()

        planner_hints: dict[str, Any] = {}
        if planner_family and planner_inputs is None:
            planner_inputs = {}

        if planner_family and planner_inputs is not None:
            planner_hints = _derive_planner_hints(planner_family, planner_inputs)
            process_plan_summary = {
                "family": planner_family,
                "inputs": planner_inputs,
            }
            if planner_hints:
                process_plan_summary["derived"] = planner_hints

            try:
                planner_result = _process_plan_job(planner_family, planner_inputs)
            except Exception as planner_err:  # pragma: no cover - defensive logging
                logger.debug("Process planner failed for %s: %s", planner_family, planner_err)
                process_plan_summary["error"] = str(planner_err)
                if FORCE_PLANNER:
                    raise ConfigError(
                        "FORCE_PLANNER enabled but process planner execution failed"
                    ) from planner_err
            else:
                process_plan_summary["plan"] = planner_result
                if planner_result.get("ops"):
                    try:
                        from planner_pricing import price_with_planner  # type: ignore
                    except Exception as pricing_import_err:  # pragma: no cover - optional dependency
                        planner_pricing_error = str(pricing_import_err)
                        if FORCE_PLANNER:
                            raise ConfigError(
                                "FORCE_PLANNER enabled but planner pricing module is unavailable"
                            ) from pricing_import_err
                    else:
                        try:
                            two_bucket_rates = migrate_flat_to_two_bucket(rates)
                            planner_two_bucket_rates = two_bucket_rates
                        except Exception as rate_err:
                            planner_pricing_error = str(rate_err)
                            if FORCE_PLANNER:
                                raise ConfigError(
                                    "FORCE_PLANNER enabled but planner pricing could not derive rates"
                                ) from rate_err
                        else:
                            geom_payload: dict[str, Any] = {"source": "appV5"}
                            geom_candidates = [
                                geo_context.get("planner_geom"),
                                geo_context.get("time_models"),
                                geo_context.get("geom") if isinstance(geo_context.get("geom"), dict) else None,
                                inner_geo,
                            ]
                            for candidate in geom_candidates:
                                if isinstance(candidate, dict):
                                    geom_payload.update(candidate)
                            planner_geom_payload = dict(geom_payload)
                            try:
                                planner_pricing_result = price_with_planner(
                                    planner_family,
                                    planner_inputs or {},
                                    geom_payload,
                                    two_bucket_rates,
                                    oee=float(params.get("OEE_EfficiencyPct", 0.85) or 0.85),
                                )
                            except Exception as price_err:  # pragma: no cover - defensive logging
                                planner_pricing_error = str(price_err)
                            else:
                                process_plan_summary["pricing"] = copy.deepcopy(planner_pricing_result)
                else:
                    if FORCE_PLANNER:
                        raise ConfigError("FORCE_PLANNER enabled but process planner returned no operations")
                    if planner_pricing_error is None:
                        planner_pricing_error = "Planner returned no operations"

    elif FORCE_PLANNER:
        raise ConfigError("FORCE_PLANNER enabled but process planner is unavailable")

    if process_plan_summary:
        features["process_plan"] = copy.deepcopy(process_plan_summary)
        if planner_pricing_error and "pricing" not in process_plan_summary:
            process_plan_summary["pricing_error"] = planner_pricing_error

    planner_line_items: list[dict[str, Any]] = []
    recognized_line_items = 0
    planner_machine_cost_total = 0.0
    planner_labor_cost_total = 0.0
    planner_total_minutes = 0.0
    planner_bucket_view: dict[str, Any] | None = None
    planner_bucket_rollup: dict[str, dict[str, float]] | None = None

    if planner_pricing_result is not None:
        # Canonical planner integration + bucketization path
        quote_state.process_plan.setdefault("pricing", copy.deepcopy(planner_pricing_result))
        totals = planner_pricing_result.get("totals", {}) if isinstance(planner_pricing_result, dict) else {}
        line_items_raw = planner_pricing_result.get("line_items", []) if isinstance(planner_pricing_result, dict) else []
        if isinstance(line_items_raw, list):
            for entry in line_items_raw:
                if isinstance(entry, _MappingABC):
                    planner_line_items.append({k: entry.get(k) for k in ("op", "minutes", "machine_cost", "labor_cost")})
                    recognized_line_items += 1

        planner_machine_cost_total = float(totals.get("machine_cost", 0.0) or 0.0)
        planner_labor_cost_total = float(totals.get("labor_cost", 0.0) or 0.0)
        planner_total_minutes = float(totals.get("minutes", 0.0) or 0.0)
        planner_totals_cost = planner_machine_cost_total + planner_labor_cost_total

        if planner_line_items:
            display_machine_cost = 0.0
            display_labor_cost = 0.0
            for entry in planner_line_items:
                try:
                    machine_val = float(entry.get("machine_cost", 0.0) or 0.0)
                except Exception:
                    machine_val = 0.0
                try:
                    labor_val = float(entry.get("labor_cost", 0.0) or 0.0)
                except Exception:
                    labor_val = 0.0
                display_machine_cost += machine_val
                display_labor_cost += labor_val

            display_total_cost = display_machine_cost + display_labor_cost
            if abs(display_total_cost - planner_totals_cost) >= _PLANNER_BUCKET_ABS_EPSILON:
                raise AssertionError(
                    "Planner line items do not reconcile with planner totals."
                )

        def _planner_bucketize(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
            def _resolve_bucket(name: str) -> str:
                text = (name or "").lower()
                if not text:
                    return "milling"

                def _contains(*tokens: str) -> bool:
                    return any(token in text for token in tokens)

                if _contains("countersink", "counter sink", "csk", "c-sink", "c sink"):
                    return "countersink"
                if _contains("counterbore", "c'bore", "c bore", "cbore", "spotface"):
                    return "counterbore"
                if "tap" in text or _contains("thread mill", "threadmill"):
                    return "tapping"
                if _contains("grind", "od grind", "id grind", "surface grind", "jig grind"):
                    return "grinding"
                if "wire" in text or "wedm" in text:
                    return "wire_edm"
                if "edm" in text:
                    return "sinker_edm"
                if _contains("saw", "waterjet", "water jet"):
                    return "saw_waterjet"
                if _contains("deburr", "finish", "polish", "blend"):
                    return "finishing_deburr"
                if _contains("inspect", "cmm", "fai", "first article", "qc", "quality"):
                    return "inspection"
                if _contains("assembly", "fit", "bench", "install"):
                    return "assembly"
                if _contains("toolmaker", "tooling support"):
                    return "toolmaker_support"
                if _contains("package", "pack-out", "pack out", "boxing"):
                    return "packaging"
                if _contains("ehs", "environment", "safety"):
                    return "ehs_compliance"
                if _contains("turn", "lathe"):
                    return "turning"
                if _contains("lap", "hone"):
                    return "lapping_honing"
                if _contains("drill", "ream", "bore", "hole"):
                    return "drilling"
                if _contains("milling", "mill", "t-slot", "pocket", "profile"):
                    return "milling"
                return "misc"

            bucket_view: dict[str, dict[str, float]] = {}
            for entry in entries:
                bucket = _resolve_bucket(str(entry.get("op", "")))
                minutes_val = float(entry.get("minutes") or 0.0)
                machine_cost = float(entry.get("machine_cost") or 0.0)
                labor_cost = float(entry.get("labor_cost") or 0.0)
                bucket_totals = bucket_view.setdefault(
                    bucket,
                    {
                        "minutes": 0.0,
                        "labor_cost": 0.0,
                        "machine_cost": 0.0,
                        "total_cost": 0.0,
                    },
                )
                bucket_totals["minutes"] += minutes_val
                bucket_totals["labor_cost"] += labor_cost
                bucket_totals["machine_cost"] += machine_cost
                bucket_totals["total_cost"] += labor_cost + machine_cost
            return bucket_view

        bucket_view = _planner_bucketize(planner_line_items)

        if bucket_view:
            display_machine_cost = 0.0
            display_labor_cost = 0.0
            for info in bucket_view.values():
                if not isinstance(info, Mapping):
                    continue
                try:
                    machine_val = float(
                        info.get("machine_cost")
                        or info.get("machine$")
                        or 0.0
                    )
                except Exception:
                    machine_val = 0.0
                try:
                    labor_val = float(
                        info.get("labor_cost") or info.get("labor$") or 0.0
                    )
                except Exception:
                    labor_val = 0.0
                display_machine_cost += machine_val
                display_labor_cost += labor_val

            display_total_cost = display_machine_cost + display_labor_cost
            if abs(display_total_cost - planner_totals_cost) >= _PLANNER_BUCKET_ABS_EPSILON:
                raise AssertionError(
                    "Planner bucket rows do not reconcile with planner totals."
                )

        if bucket_view:
            try:
                planner_bucket_total = 0.0
                for info in bucket_view.values():
                    if not isinstance(info, Mapping):
                        continue
                    planner_bucket_total += float(info.get("machine_cost", 0.0) or 0.0)
                    planner_bucket_total += float(info.get("labor_cost", 0.0) or 0.0)
                planner_totals_expected = planner_machine_cost_total + planner_labor_cost_total
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Planner bucket invariant check failed")
            else:
                if not math.isclose(
                    planner_bucket_total,
                    planner_totals_expected,
                    rel_tol=0.0,
                    abs_tol=_PLANNER_BUCKET_ABS_EPSILON,
                ):
                    logger.warning(
                        "Planner bucket totals drifted: %.2f vs %.2f",
                        planner_bucket_total,
                        planner_totals_expected,
                    )
            planner_bucket_rollup = copy.deepcopy(bucket_view)
            planner_bucket_view = _prepare_bucket_view(bucket_view)
            planner_bucket_view_copy = copy.deepcopy(planner_bucket_view)
            process_plan_summary["bucket_view"] = planner_bucket_view_copy
            quote_state.process_plan.setdefault("bucket_view", planner_bucket_view_copy)
            drill_bucket = bucket_view.get("drilling") if isinstance(bucket_view, _MappingABC) else None
            if drill_bucket:
                try:
                    minutes_val = float(drill_bucket.get("minutes") or 0.0)
                except Exception:
                    minutes_val = 0.0
                if minutes_val > 0.0:
                    planner_drill_minutes = minutes_val

        machine_minutes = 0.0
        labor_minutes = 0.0
        for entry in planner_line_items:
            minutes_val = float(entry.get("minutes", 0.0) or 0.0)
            if float(entry.get("machine_cost", 0.0) or 0.0) > 0.0:
                machine_minutes += minutes_val
            if float(entry.get("labor_cost", 0.0) or 0.0) > 0.0:
                labor_minutes += minutes_val
            if planner_drill_minutes in (None, 0.0):
                name = str(entry.get("op") or "").lower()
                if any(token in name for token in ("drill", "ream", "bore")):
                    if minutes_val > 0.0:
                        planner_drill_minutes = minutes_val

        planner_totals_present = any(
            float(val) > 0.0 for val in (planner_machine_cost_total, planner_labor_cost_total, planner_total_minutes)
        )

        planner_signals = {
            "line_items": planner_line_items,
            "pricing_result": planner_pricing_result,
            "recognized_line_items": recognized_line_items,
            "totals_present": planner_totals_present,
        }

        used_planner, planner_mode = resolve_planner(
            params=params if isinstance(params, _MappingABC) else None,
            signals=planner_signals,
        )
        force_planner_for_recognized = recognized_line_items > 0

        if force_planner_for_recognized:
            used_planner = True

        if recognized_line_items == 0:
            if planner_totals_present and planner_total_minutes > 0.0:
                if planner_machine_cost_total > 0.0 and planner_labor_cost_total <= 0.0:
                    machine_minutes = planner_total_minutes
                elif planner_labor_cost_total > 0.0 and planner_machine_cost_total <= 0.0:
                    labor_minutes = planner_total_minutes
            if FORCE_PLANNER:
                raise ConfigError("FORCE_PLANNER enabled but planner pricing returned no line items")
            if planner_pricing_error is None:
                planner_pricing_error = "Planner pricing returned no line items"

        if (not used_planner) and planner_drill_minutes and planner_drill_minutes > 0.0:
            planner_drill_hr = planner_drill_minutes / 60.0
            if planner_drill_hr > 0.0 and drill_hr > 10.0 * planner_drill_hr:
                _record_red_flag("Legacy drilling estimate outlier — using planner drilling hours.")
                drill_hr = planner_drill_hr
                baseline_drill_hr = max(existing_drill_hr, float(drill_hr or 0.0))
                baseline_drill_hr = round(baseline_drill_hr, 3)
                drill_meta = legacy_process_meta.setdefault("drilling", {})
                drill_meta["hr"] = baseline_drill_hr
                drill_meta.setdefault("rate", drill_rate)
                legacy_process_costs["drilling"] = baseline_drill_hr * drill_rate
                process_costs["drilling"] = legacy_process_costs["drilling"]
                process_costs_baseline["drilling"] = legacy_process_costs["drilling"]
                process_hours_baseline["drilling"] = baseline_drill_hr
                features["drilling_hr_baseline"] = baseline_drill_hr

        if used_planner:
            pricing_source = "planner"
            if legacy_baseline_had_values:
                legacy_baseline_ignored = True
            planner_process_minutes = planner_total_minutes
            process_costs = {
                "Machine": round(planner_machine_cost_total, 2),
                "Labor": round(planner_labor_cost_total, 2),
            }

            # Using planner pricing should mark the overall pricing source accordingly,
            # even if drilling falls back to legacy heuristics due to CSV issues.
            pricing_source = "planner"

            existing_process_meta = {key: dict(value) for key, value in process_meta.items()}
            process_meta = existing_process_meta
            process_meta["planner_machine"] = {
                "hr": round(machine_minutes / 60.0, 3),
                "minutes": round(machine_minutes, 1),
                "rate": 0.0,
                "cost": round(planner_machine_cost_total, 2),
            }
            planner_meta_keys.add("planner_machine")
            process_meta["planner_labor"] = {
                "hr": round(labor_minutes / 60.0, 3),
                "minutes": round(labor_minutes, 1),
                "rate": 0.0,
                "cost": round(planner_labor_cost_total, 2),
            }
            planner_meta_keys.add("planner_labor")
            total_cost = planner_machine_cost_total + planner_labor_cost_total
            process_meta["planner_total"] = {
                "hr": round(planner_total_minutes / 60.0, 3),
                "minutes": round(planner_total_minutes, 1),
                "rate": 0.0,
                "cost": round(total_cost, 2),
            }
            planner_meta_keys.add("planner_total")
            if planner_line_items:
                process_meta["planner_total"]["line_items"] = copy.deepcopy(planner_line_items)
                for entry in planner_line_items:
                    name = str(entry.get("op") or "").strip()
                    if not name:
                        continue
                    minutes_val = float(entry.get("minutes", 0.0) or 0.0)
                    hours_val = minutes_val / 60.0 if minutes_val else 0.0
                    cost_val = float(entry.get("machine_cost") or entry.get("labor_cost") or 0.0)
                    rate_val = (cost_val / hours_val) if hours_val > 0 else 0.0
                    key_norm = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                    process_meta[key_norm] = {
                        "minutes": round(minutes_val, 2),
                        "hr": round(hours_val, 3),
                        "cost": round(cost_val, 2),
                        "rate": round(rate_val, 2) if rate_val else 0.0,
                    }
                    planner_meta_keys.add(key_norm)

            if planner_drilling_override:
                override_minutes = float(planner_drilling_override.get("minutes") or 0.0)
                override_hours = float(planner_drilling_override.get("hours") or 0.0)
                override_machine_cost = float(planner_drilling_override.get("machine_cost") or 0.0)
                override_labor_cost = float(planner_drilling_override.get("labor_cost") or 0.0)
                override_total_cost = float(
                    planner_drilling_override.get("total_cost")
                    or (override_machine_cost + override_labor_cost)
                )
                override_rate = float(planner_drilling_override.get("rate") or 0.0)

                def _apply_override_to_bucket_map(bucket_map: Mapping[str, Any] | None) -> None:
                    if not isinstance(bucket_map, dict):
                        return
                    entry = bucket_map.get("drilling")
                    if isinstance(entry, Mapping):
                        entry = dict(entry)
                    else:
                        entry = {}
                    entry.update(
                        {
                            "minutes": round(override_minutes, 2),
                            "machine_cost": round(override_machine_cost, 2),
                            "labor_cost": round(override_labor_cost, 2),
                            "total_cost": round(override_total_cost, 2),
                        }
                    )
                    bucket_map["drilling"] = entry

                for mapping in (bucket_view, planner_bucket_view, planner_bucket_rollup):
                    _apply_override_to_bucket_map(mapping)
                plan_bucket_view = process_plan_summary.get("bucket_view")
                _apply_override_to_bucket_map(plan_bucket_view)
                existing_plan = getattr(quote_state, "process_plan", None)
                if isinstance(existing_plan, dict):
                    _apply_override_to_bucket_map(existing_plan.get("bucket_view"))
                if override_minutes > 0:
                    planner_drill_minutes = override_minutes
                process_meta["drilling"] = {
                    "hr": round(override_hours, 3),
                    "minutes": round(override_minutes, 1),
                    "rate": round(override_rate, 2) if override_rate else 0.0,
                    "cost": round(override_total_cost, 2),
                    "basis": ["planner_drilling_override"],
                }
                planner_meta_keys.add("drilling")

            if bucket_view:
                for b, info in bucket_view.items():
                    hr = info["minutes"] / 60.0 if info["minutes"] else 0.0
                    cost = info["total_cost"]
                    rate = (cost / hr) if hr > 0 else 0.0
                    update_payload = {
                        "minutes": round(info["minutes"], 1),
                        "hr": round(hr, 3),
                        "rate": round(rate, 2),
                        "cost": round(cost, 2),
                    }
                    existing_meta = process_meta.get(b)
                    if isinstance(existing_meta, Mapping):
                        existing_meta.update(update_payload)
                    else:
                        process_meta[b] = update_payload
                    planner_meta_keys.add(b)

        if bucket_view:
            for b, info in bucket_view.items():
                hr = info["minutes"] / 60.0 if info["minutes"] else 0.0
                cost = info["total_cost"]
                rate = (cost / hr) if hr > 0 else 0.0
                update_payload = {
                    "minutes": round(info["minutes"], 1),
                    "hr": round(hr, 3),
                    "rate": round(rate, 2),
                    "cost": round(cost, 2),
                }
                existing_meta = process_meta.get(b)
                if isinstance(existing_meta, Mapping):
                    existing_meta.update(update_payload)
                else:
                    process_meta[b] = update_payload
                planner_meta_keys.add(b)

        meta_lookup = {
            key: dict(value) for key, value in process_meta.items() if isinstance(value, Mapping)
        }
        allowed_process_hour_keys = {
            key
            for key in planner_meta_keys
            if key not in legacy_per_process_keys or key.startswith("planner_")
        }
        if not allowed_process_hour_keys:
            allowed_process_hour_keys = set(planner_meta_keys)
        if "drilling" in legacy_process_meta:
            allowed_process_hour_keys.add("drilling")
        process_hours_baseline = {
            key: float(process_meta.get(key, {}).get("hr", 0.0) or 0.0)
            for key in allowed_process_hour_keys
        }
        process_costs_baseline = {k: float(v) for k, v in process_costs.items()}

    if family_for_breakdown is None:
        family_hint_candidates = (
            planner_family,
            geo_context.get("process_planner_family"),
            geo_context.get("planner_family"),
            geo_context.get("process_family"),
        )
        for candidate in family_hint_candidates:
            if candidate is None:
                continue
            normalized_family = _normalize_family(candidate)
            if normalized_family:
                family_for_breakdown = normalized_family
                break
            text = str(candidate).strip().lower()
            if text:
                family_for_breakdown = text
                break

    die_plate_families = {"die", "die_plate", "die plates", "shoe"}
    if str(family_for_breakdown or "").strip().lower() in die_plate_families:
        setups_for_fixture = 0
        try:
            setups_for_fixture = int(round(float(setups)))
        except Exception:
            setups_for_fixture = 0
        setups_for_fixture = max(1, setups_for_fixture)
        fixture_build_hr = max(0.75, 0.33 * setups_for_fixture)
        fixture_rate = float(rates.get("FixtureBuildRate", 0.0))
        fixture_labor_cost = float(fixture_build_hr) * fixture_rate
        fixture_cost = float(fixture_labor_cost)
        fixture_labor_per_part = (
            fixture_labor_cost / Qty if Qty > 1 else fixture_labor_cost
        )
        fixture_per_part = (
            fixture_cost / Qty if Qty > 1 else fixture_cost
        )
        fixture_entry = nre_detail.setdefault("fixture", {})
        fixture_entry.update(
            {
                "build_hr": float(fixture_build_hr),
                "build_rate": fixture_rate,
                "labor_cost": float(fixture_labor_cost),
                "per_lot": float(fixture_cost),
                "per_part": float(fixture_per_part),
            }
        )
        features["fixture_build_hr"] = float(fixture_build_hr)
        fixture_build_hr_base = float(fixture_build_hr or 0.0)
        fixture_plan_desc = None
        strategy_text = None
        if isinstance(fixture_entry, Mapping):
            strategy_value = fixture_entry.get("strategy")
            if isinstance(strategy_value, str):
                strategy_text = strategy_value.strip()
        if fixture_build_hr > 0:
            fixture_plan_desc = f"{fixture_build_hr:.2f} hr build"
        if strategy_text:
            if fixture_plan_desc:
                fixture_plan_desc = f"{fixture_plan_desc} ({strategy_text})"
            else:
                fixture_plan_desc = strategy_text

    baseline_data = {
        "process_hours": process_hours_baseline,
        "pass_through": pass_through_baseline,
        "scrap_pct": scrap_pct_baseline,
        "setups": int(setups),
        "contingency_pct": ContingencyPct,
        "fixture_build_hr": fixture_build_hr_base,
        "soft_jaw_hr": 0.0,
        "soft_jaw_material_cost": 0.0,
        "handling_adder_hr": 0.0,
        "cmm_minutes": cmm_minutes_base,
        "in_process_inspection_hr": inproc_hr_base,
        "inspection_total_hr": inspection_hr_total,
        "inspection_components": {k: float(v) for k, v in inspection_components.items()},
        "fai_required": fai_flag_base,
        "fai_prep_hr": 0.0,
        "packaging_hours": packaging_hr_base,
        "packaging_flat_cost": packaging_flat_base,
        "shipping_cost": shipping_cost_base,
        "shipping_hint": "",
    }
    if red_flag_messages:
        baseline_data["red_flags"] = list(red_flag_messages)
    baseline_data["speeds_feeds_path"] = speeds_feeds_path
    baseline_data["pricing_source"] = pricing_source
    if legacy_baseline_ignored:
        baseline_data["legacy_baseline_ignored"] = True
    if fixture_plan_desc:
        baseline_data["fixture"] = fixture_plan_desc
    if process_plan_summary:
        baseline_data["process_plan"] = copy.deepcopy(process_plan_summary)
    if planner_process_minutes is not None:
        baseline_data["process_minutes"] = round(float(planner_process_minutes), 1)
    if planner_pricing_result is not None:
        baseline_data["process_plan_pricing"] = copy.deepcopy(planner_pricing_result)
    if planner_bucket_view is not None:
        baseline_data["process_plan_bucket_view"] = copy.deepcopy(planner_bucket_view)
    quote_state.baseline = baseline_data
    quote_state.process_plan = copy.deepcopy(process_plan_summary)

    base_costs = {
        "process_costs": process_costs_baseline,
        "pass_through": pass_through_baseline,
        "nre_detail": copy.deepcopy(nre_detail),
        "rates": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in rates.items()},
        "params": dict(params),
    }

    def _jsonable(val):
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        if isinstance(val, (list, tuple)):
            return [_jsonable(v) for v in val]
        if isinstance(val, dict):
            return {str(k): _jsonable(v) for k, v in val.items()}
        try:
            return float(val)
        except Exception:
            try:
                return val.item()  # numpy scalar
            except Exception:
                return str(val)

    hole_diams_ctx = sorted([round(x, 3) for x in hole_diams_list])[:200] if hole_diams_list else []
    quote_inputs_ctx = {str(k): _jsonable(v) for k, v in ui_vars.items()}

    geo_payload = _jsonable(dict(geo_context))
    if hole_diams_ctx:
        geo_payload["hole_diams_mm"] = hole_diams_ctx
    geo_payload["hole_count"] = hole_count_for_tripwire
    if hole_count_override and hole_count_override > 0:
        geo_payload["hole_count_override"] = int(round(hole_count_override))
    if avg_hole_diam_override and avg_hole_diam_override > 0:
        geo_payload["avg_hole_diam_override_mm"] = float(avg_hole_diam_override)
    quote_state.geo = dict(geo_payload)


    baseline_setups = int(to_float(setups) or to_float(baseline_data.get("setups")) or 1)
    baseline_fixture = baseline_data.get("fixture")

    llm_baseline_payload = {
        "process_hours": process_hours_baseline,
        "pass_through": pass_through_baseline,
        "scrap_pct": scrap_pct_baseline,
        "setups": baseline_setups,
        "fixture": baseline_fixture,
    }
    llm_bounds = {
        "mult_min": LLM_MULTIPLIER_MIN,
        "mult_max": LLM_MULTIPLIER_MAX,
        "adder_max_hr": LLM_ADDER_MAX,
        "scrap_min": 0.0,
        "scrap_max": 0.25,
    }
    baseline_bounds = baseline_data.get("_bounds") if isinstance(baseline_data.get("_bounds"), dict) else None
    if baseline_bounds:
        bucket_caps_raw = baseline_bounds.get("adder_bucket_max") or baseline_bounds.get("add_hr_bucket_max")
        if isinstance(bucket_caps_raw, dict):
            bucket_caps_clean: dict[str, float] = {}
            for key, raw in bucket_caps_raw.items():
                cap_val = to_float(raw)
                if cap_val is not None:
                    bucket_caps_clean[str(key)] = float(cap_val)
            if bucket_caps_clean:
                llm_bounds["adder_bucket_max"] = bucket_caps_clean

    tap_qty_seed = int(max(tap_qty_seed, tap_qty_geo))
    cbore_qty_seed = int(max(cbore_qty_seed, cbore_qty_geo))
    geo_context["tap_qty"] = int(tap_qty_geo or 0)
    geo_context["cbore_qty"] = int(cbore_qty_geo or 0)
    hole_bins_for_seed: dict[str, int] = {}
    if chart_reconcile_geo:
        bins_raw = chart_reconcile_geo.get("chart_bins") or {}
        hole_bins_for_seed = {str(k): int(v) for k, v in bins_raw.items()}

    raw_geo_block = geo_context.get("raw") if isinstance(geo_context.get("raw"), dict) else {}
    leader_texts: list[str] = []
    if isinstance(raw_geo_block, dict):
        leaders = raw_geo_block.get("leaders")
        if isinstance(leaders, (list, tuple)):
            leader_texts.extend(str(txt) for txt in leaders if txt)
        leader_entries = raw_geo_block.get("leader_entries")
        if isinstance(leader_entries, (list, tuple)):
            for entry in leader_entries:
                if isinstance(entry, dict):
                    raw_txt = entry.get("raw")
                    if raw_txt:
                        leader_texts.append(str(raw_txt))
    has_leader_notes = False
    leader_keywords = ("FROM BACK", "BACK SIDE", "OPP SIDE", "JIG GRIND")
    for text in leader_texts:
        upper = text.upper()
        if any(keyword in upper for keyword in leader_keywords):
            has_leader_notes = True
            break

    def _first_float(*values: Any) -> float | None:
        for value in values:
            cand = _coerce_float_or_none(value)
            if cand is not None:
                return float(cand)
        return None

    max_hole_depth_in = _first_float(
        geo_context.get("deepest_hole_in"),
        (geo_context.get("chart_summary") or {}).get("deepest_hole_in") if isinstance(geo_context.get("chart_summary"), dict) else None,
        (geo_context.get("geo_read_more") or {}).get("deepest_hole_in") if isinstance(geo_context.get("geo_read_more"), dict) else None,
        inner_geo.get("deepest_hole_in") if inner_geo else None,
    )

    plate_len_val = _coerce_float_or_none(
        geo_context.get("plate_len_in")
        or inner_geo.get("plate_len_in")
    )
    plate_wid_val = _coerce_float_or_none(
        geo_context.get("plate_wid_in")
        or inner_geo.get("plate_wid_in")
    )
    outline_area_val = _coerce_float_or_none(
        geo_context.get("outline_area_in2")
        or inner_geo.get("outline_area_in2")
    )
    plate_area_in2 = None
    if plate_len_val and plate_wid_val:
        try:
            plate_area_in2 = float(plate_len_val) * float(plate_wid_val)
        except Exception:
            plate_area_in2 = None
    elif outline_area_val:
        plate_area_in2 = float(outline_area_val)

    text_candidates: list[str] = []
    if isinstance(geo_context.get("notes"), (list, tuple)):
        text_candidates.extend(str(val) for val in geo_context.get("notes") if val)
    if isinstance(geo_context.get("chart_lines"), (list, tuple)):
        text_candidates.extend(str(val) for val in geo_context.get("chart_lines") if val)
    if leader_texts:
        text_candidates.extend(leader_texts)
    default_tol_text = geo_context.get("default_tol") or geo_context.get("default_tolerance")
    if default_tol_text:
        text_candidates.append(str(default_tol_text))
    material_note_txt = geo_context.get("material_note")
    if material_note_txt:
        text_candidates.append(str(material_note_txt))

    has_tight_tol_flag = False
    for snippet in text_candidates:
        upper = snippet.upper()
        if RE_TIGHT_TOL.search(upper):
            has_tight_tol_flag = True
            break
        for match in re.findall(r"0\.000\d+", upper):
            try:
                if float(match) <= 0.0003:
                    has_tight_tol_flag = True
                    break
            except Exception:
                continue
        if has_tight_tol_flag:
            break

    finish_flags_set: set[str] = set()
    finishes_geo = geo_context.get("finishes") or inner_geo.get("finishes") if inner_geo else None
    if isinstance(finishes_geo, (list, tuple, set)):
        for fin in finishes_geo:
            if isinstance(fin, str) and fin.strip():
                finish_flags_set.add(fin.strip().upper())
    finish_keywords = {
        "BLACK OXIDE": ("BLACK OXIDE", "BLK OX"),
        "ANODIZE": ("ANODIZE", "ANODIZED", "ANODISE"),
        "PASSIVATION": ("PASSIVATE", "PASSIVATION"),
    }
    for snippet in text_candidates:
        upper = snippet.upper()
        for canonical, tokens in finish_keywords.items():
            if any(token in upper for token in tokens):
                finish_flags_set.add(canonical)
    finish_flags_list = sorted(finish_flags_set)

    stock_guess = None
    material_family = geo_context.get("material_family") or (inner_geo.get("material_family") if inner_geo else None)
    if isinstance(material_family, str) and material_family.strip():
        lower = material_family.lower()
        if "alum" in lower:
            stock_guess = "Aluminum"
        elif "brass" in lower:
            stock_guess = "Brass"
        elif "copper" in lower:
            stock_guess = "Copper"
        elif "steel" in lower:
            stock_guess = "Steel"
    material_text_parts: list[str] = []
    for key in ("material_note", "material", "material_family"):
        val = geo_context.get(key)
        if not val and inner_geo:
            val = inner_geo.get(key)
        if val:
            material_text_parts.append(str(val))
    combined_material_text = " ".join(material_text_parts).upper()
    if not stock_guess and combined_material_text:
        if any(tok in combined_material_text for tok in ("6061", "7075", "ALUMIN", "ALUM.")):
            stock_guess = "Aluminum"
        elif any(tok in combined_material_text for tok in ("A2", "D2", "H13", "STEEL", "17-4", "15-5", "S7", "O1", "4140", "4340")):
            stock_guess = "Steel"
        elif any(tok in combined_material_text for tok in ("C360", "BRASS", "C260")):
            stock_guess = "Brass"
        elif any(tok in combined_material_text for tok in ("C110", "COPPER")):
            stock_guess = "Copper"

    needs_back_face_flag = bool(
        geo_context.get("GEO_Setup_NeedsBackOps")
        or geo_context.get("needs_back_face")
        or (bool(is_plate_2d) and baseline_setups > 1)
    )

    bbox_payload = bbox_for_llm

    geo_for_suggest = {
        "meta": {"is_2d_plate": bool(is_plate_2d)},
        "hole_count": hole_count_for_tripwire,
        "derived": {
            "tap_qty": tap_qty_seed,
            "cbore_qty": cbore_qty_seed,
            "csk_qty": csk_qty_seed,
            "hole_bins": hole_bins_for_seed,
            "needs_back_face": needs_back_face_flag,
            "tap_minutes_hint": _coerce_float_or_none(geo_context.get("tap_minutes_hint")),
            "cbore_minutes_hint": _coerce_float_or_none(geo_context.get("cbore_minutes_hint")),
            "csk_minutes_hint": _coerce_float_or_none(geo_context.get("csk_minutes_hint")),
            "tap_class_counts": geo_context.get("tap_class_counts"),
            "tap_details": geo_context.get("tap_details"),
            "npt_qty": int(_coerce_float_or_none(geo_context.get("npt_qty")) or 0),
            "inference_knobs": geo_context.get("inference_knobs"),
        },
        "thickness_mm": float(thickness_for_llm) if thickness_for_llm is not None else None,
        "material": material_name,
        "bbox_mm": bbox_payload,
    }
    derived_block = geo_for_suggest["derived"]
    derived_block["has_ldr_notes"] = has_leader_notes
    derived_block["has_tight_tol"] = has_tight_tol_flag
    if default_tol_text:
        derived_block["default_tolerance_note"] = str(default_tol_text)
    if max_hole_depth_in is not None:
        derived_block["max_hole_depth_in"] = max_hole_depth_in
    if plate_area_in2 is not None:
        derived_block["plate_area_in2"] = plate_area_in2
    if finish_flags_list:
        derived_block["finish_flags"] = finish_flags_list
    if stock_guess:
        derived_block["stock_guess"] = stock_guess
    dfm_geo_block = features.get("dfm_geo") if isinstance(features.get("dfm_geo"), dict) else None
    if dfm_geo_block and any(val not in (None, "", []) for val in dfm_geo_block.values()):
        derived_block["dfm_geo"] = dfm_geo_block
    tol_inputs_block = tolerance_inputs if isinstance(tolerance_inputs, dict) else None
    if tol_inputs_block:
        derived_block["tolerance_inputs"] = tol_inputs_block
    for extra_key in ("stock_catalog", "machine_limits", "fixture_plan"):
        extra_val = features.get(extra_key)
        if extra_val not in (None, "", [], {}):
            derived_block[extra_key] = extra_val
    if "fai_required" in features:
        derived_block["fai_required"] = bool(features.get("fai_required"))

    payload = build_suggest_payload(geo_for_suggest, llm_baseline_payload, base_costs["rates"], llm_bounds)

    suggestions_struct: dict[str, Any] = {}
    overrides_meta: dict[str, Any] = {}
    sanitized_struct: dict[str, Any] = {}
    if reuse_suggestions and quote_state and quote_state.suggestions:
        suggestions_struct = copy.deepcopy(quote_state.suggestions)
        overrides_meta = dict(quote_state.llm_raw or {})
        sanitized_struct = suggestions_struct if isinstance(suggestions_struct, dict) else {}
    else:
        llm_obj = None
        s_raw: dict[str, Any] = {}
        s_text = ""
        s_usage: dict[str, Any] = {}
        sanitized_struct = {}
        if llm_enabled:
            if llm_suggest is not None:
                try:
                    # Prepare a safe JSON body for the LLM message; default=str guards non-serializable entries
                    _payload_body = jdump(payload)
                    chat_out = llm_suggest.create_chat_completion(
                        messages=[
                            {"role": "system", "content": SYSTEM_SUGGEST},
                            {"role": "user", "content": _payload_body},
                        ],
                        temperature=0.3,
                        top_p=0.9,
                        max_tokens=512,
                    )
                    s_usage = chat_out.get("usage", {}) or {}
                    choice = (chat_out.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    s_text = str(message.get("content") or "")
                    s_raw = parse_llm_json(s_text) or {}
                    sanitized_struct = sanitize_suggestions(s_raw, llm_bounds)
                    overrides_meta["model"] = "qwen2.5-vl"
                    overrides_meta["n_ctx"] = getattr(llm_suggest, "n_ctx", None)
                except Exception as exc:
                    fallback_struct = {
                        "process_hour_multipliers": {"drilling": 1.0, "milling": 1.0},
                        "process_hour_adders": {"inspection": 0.0},
                        "scrap_pct": llm_baseline_payload.get("scrap_pct", 0.0),
                        "setups": llm_baseline_payload.get("setups", baseline_setups),
                        "fixture": llm_baseline_payload.get("fixture") or baseline_fixture or "standard",
                        "notes": [f"llm error: {exc}"],
                        "no_change_reason": "exception",
                    }
                    sanitized_struct = sanitize_suggestions(fallback_struct, llm_bounds)
                    overrides_meta["error"] = repr(exc)
                    s_raw = {}
                    s_text = ""
                    s_usage = {}
            elif llm_model_path:
                try:
                    client = llm_client
                    created_client = False
                    if client is None or not client.available or client.model_path != llm_model_path:
                        client = LLMClient(
                            llm_model_path,
                            debug_enabled=APP_ENV.llm_debug_enabled,
                            debug_dir=APP_ENV.llm_debug_dir,
                        )
                        created_client = True
                    llm_obj = client
                    s_raw, s_text, s_usage = run_llm_suggestions(client, payload)
                    sanitized_struct = sanitize_suggestions(s_raw, llm_bounds)
                except Exception as exc:
                    fallback_struct = {
                        "process_hour_multipliers": {"drilling": 1.0, "milling": 1.0},
                        "process_hour_adders": {"inspection": 0.0},
                        "scrap_pct": llm_baseline_payload.get("scrap_pct", 0.0),
                        "setups": llm_baseline_payload.get("setups", baseline_setups),
                        "fixture": llm_baseline_payload.get("fixture") or baseline_fixture or "standard",
                        "notes": [f"llm error: {exc}"],
                        "no_change_reason": "exception",
                    }
                    sanitized_struct = sanitize_suggestions(fallback_struct, llm_bounds)
                    overrides_meta["error"] = repr(exc)
                    s_raw = {}
                    s_text = ""
                    s_usage = {}
                finally:
                    if created_client:
                        try:
                            client.close()
                        except Exception:
                            pass
            else:
                reason = "model_missing"
                fallback_struct = {
                    "process_hour_multipliers": {"drilling": 1.0, "milling": 1.0},
                    "process_hour_adders": {"inspection": 0.0},
                    "scrap_pct": llm_baseline_payload.get("scrap_pct", 0.0),
                    "setups": llm_baseline_payload.get("setups", baseline_setups),
                    "fixture": llm_baseline_payload.get("fixture") or baseline_fixture or "standard",
                    "notes": ["LLM model missing"],
                    "no_change_reason": reason,
                }
                sanitized_struct = sanitize_suggestions(fallback_struct, llm_bounds)
                overrides_meta["error"] = reason
        else:
            fallback_struct = {
                "process_hour_multipliers": {"drilling": 1.0, "milling": 1.0},
                "process_hour_adders": {"inspection": 0.0},
                "scrap_pct": llm_baseline_payload.get("scrap_pct", 0.0),
                "setups": llm_baseline_payload.get("setups", baseline_setups),
                "fixture": llm_baseline_payload.get("fixture") or baseline_fixture or "standard",
                "notes": ["LLM disabled"],
                "no_change_reason": "llm_disabled",
            }
            sanitized_struct = sanitize_suggestions(fallback_struct, llm_bounds)
            overrides_meta["error"] = "llm_disabled"
        suggestions_struct = copy.deepcopy(sanitized_struct)
        applied_effective = apply_suggestions(baseline_data, sanitized_struct)
        overrides_meta.setdefault("model", getattr(llm_obj, "model_path", None) if llm_obj is not None else (llm_model_path or "unavailable"))
        overrides_meta.setdefault("n_ctx", getattr(llm_obj, "n_ctx", None) if llm_obj is not None else overrides_meta.get("n_ctx"))
        overrides_meta.update({
            "raw_response_text": s_text,
            "parsed_response": s_raw,
            "sanitized": sanitized_struct,
            "payload": payload,
            "usage": s_usage,
            "applied_effective": applied_effective,
        })
        if APP_ENV.llm_debug_enabled:
            snap = {
                "model": overrides_meta.get("model"),
                "n_ctx": overrides_meta.get("n_ctx"),
                "messages": [
                    {"role": "system", "content": SYSTEM_SUGGEST},
                    {"role": "user", "content": jdump(payload)},
                ],
                "params": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 512},
                "context_payload": payload,
                "raw_response_text": s_text,
                "parsed_response": s_raw,
                "sanitized": sanitized_struct,
                "usage": s_usage,
            }
            snap_path = APP_ENV.llm_debug_dir / f"llm_snapshot_{int(time.time())}.json"
            snap_path.write_text(jdump(snap), encoding="utf-8")

    quote_state.llm_raw = dict(overrides_meta)
    quote_state.suggestions = sanitized_struct if isinstance(sanitized_struct, dict) else {}
    quote_state.accept_llm = _auto_accept_suggestions(quote_state.suggestions)
    quote_state.bounds = dict(llm_bounds)
    reprice_with_effective(quote_state)
    effective_struct = quote_state.effective
    effective_sources = quote_state.effective_sources
    overrides = effective_to_overrides(effective_struct, quote_state.baseline)
    effective_scrap_val = scrap_pct_baseline
    if isinstance(effective_struct, dict) and effective_struct.get("scrap_pct") is not None:
        effective_scrap_val = effective_struct.get("scrap_pct")
    elif isinstance(quote_state.suggestions, dict) and quote_state.suggestions.get("scrap_pct") is not None:
        effective_scrap_val = quote_state.suggestions.get("scrap_pct")
    scrap_pct = normalize_scrap_pct(effective_scrap_val)

    llm_notes: list[str] = []
    suggestion_notes = []
    if isinstance(quote_state.suggestions, dict):
        suggestion_notes = quote_state.suggestions.get("notes") or []
    for note in list(suggestion_notes) + list(overrides.get("notes") or []):
        if isinstance(note, str) and note.strip():
            llm_notes.append(note.strip())
    notes_from_clamps = list(overrides_meta.get("clamp_notes", [])) if overrides_meta else []
    if overrides_meta.get("warnings"):
        for warn in overrides_meta["warnings"]:
            if isinstance(warn, str) and warn.strip():
                notes_from_clamps.append(warn.strip())

    if used_planner and baseline_data.get("legacy_baseline_ignored"):
        llm_notes.append("Legacy baseline ignored (planner active)")

    drilling_groups = overrides.get("drilling_groups") if isinstance(overrides, dict) else None
    if isinstance(drilling_groups, list) and drilling_groups:
        bits: list[str] = []
        for grp in drilling_groups[:3]:
            if not isinstance(grp, dict):
                continue
            qty = grp.get("qty")
            dia = grp.get("dia_mm")
            try:
                qty_i = int(qty)
            except Exception:
                qty_i = None
            dia_v = None
            try:
                dia_v = float(dia) if dia is not None else None
            except Exception:
                dia_v = None
            if qty_i and dia_v:
                label = f"{qty_i}×{dia_v:.1f}mm"
            elif qty_i:
                label = f"{qty_i} holes"
            elif dia_v:
                label = f"Ø{dia_v:.1f}mm"
            else:
                continue
            peck = grp.get("peck")
            if isinstance(peck, str) and peck.strip():
                label += f" ({peck.strip()[:24]})"
            bits.append(label)
        if bits:
            llm_notes.append("Drilling groups: " + ", ".join(bits))

    stock_plan = overrides.get("stock_recommendation") if isinstance(overrides, dict) else None
    if isinstance(stock_plan, dict) and stock_plan:
        item = stock_plan.get("stock_item") or stock_plan.get("form")
        L = stock_plan.get("length_mm")
        W = stock_plan.get("width_mm")
        T = stock_plan.get("thickness_mm")
        dims: list[str] = []
        for val in (L, W, T):
            try:
                dims.append(f"{float(val):.0f}")
            except Exception:
                dims.append(None)
        dims_clean = [d for d in dims if d is not None]
        dim_str = " × ".join(dims_clean) + " mm" if dims_clean else ""
        label = f"Stock: {item}" if item else "Stock plan applied"
        if dim_str.strip():
            label += f" ({dim_str.strip()})"
        cut = stock_plan.get("cut_count")
        try:
            cut_i = int(cut)
        except Exception:
            cut_i = 0
        if cut_i:
            label += f", {cut_i} cuts"
        llm_notes.append(label)

    setup_plan = overrides.get("setup_recommendation") if isinstance(overrides, dict) else None
    if isinstance(setup_plan, dict) and setup_plan:
        setups_val = setup_plan.get("setups")
        try:
            setups_i = int(setups_val)
        except Exception:
            setups_i = None
        fixture = setup_plan.get("fixture")
        if setups_i or fixture:
            piece = f"Setups: {setups_i}" if setups_i else "Setups updated"
            if isinstance(fixture, str) and fixture.strip():
                piece += f" ({fixture.strip()[:60]})"
            suffix = ""
            if isinstance(quote_state.effective_sources, dict):
                if setups_i is not None:
                    src_tag = quote_state.effective_sources.get("setups")
                    if src_tag == "user":
                        suffix = " (user override)"
                    elif src_tag == "llm":
                        suffix = " (LLM)"
                elif fixture:
                    src_tag = quote_state.effective_sources.get("fixture")
                    if src_tag == "user":
                        suffix = " (user override)"
                    elif src_tag == "llm":
                        suffix = " (LLM)"
            llm_notes.append(piece + suffix)

    dfm_risks = overrides.get("dfm_risks") if isinstance(overrides, dict) else None
    if isinstance(dfm_risks, list) and dfm_risks:
        risks_clean = [str(r).strip() for r in dfm_risks if str(r).strip()]
        if risks_clean:
            llm_notes.append("DFM risks: " + "; ".join(risks_clean[:4]))

    dfm_context_bits: list[str] = []
    derived_for_notes = geo_for_suggest.get("derived") if isinstance(geo_for_suggest, dict) else {}
    dfm_geo_notes = derived_for_notes.get("dfm_geo") if isinstance(derived_for_notes, dict) else None
    if not isinstance(dfm_geo_notes, dict) and isinstance(features.get("dfm_geo"), dict):
        dfm_geo_notes = features.get("dfm_geo")
    if isinstance(dfm_geo_notes, dict):
        min_wall_val = dfm_geo_notes.get("min_wall_mm")
        try:
            min_wall_num = float(min_wall_val) if min_wall_val is not None else None
        except Exception:
            min_wall_num = None
        if min_wall_num:
            dfm_context_bits.append(f"min wall ≈ {min_wall_num:.1f} mm")
        if dfm_geo_notes.get("thin_wall"):
            dfm_context_bits.append("thin walls flagged")
        unique_normals_val = dfm_geo_notes.get("unique_normals")
        try:
            unique_normals_int = int(unique_normals_val) if unique_normals_val is not None else None
        except Exception:
            unique_normals_int = None
        if unique_normals_int:
            dfm_context_bits.append(f"{unique_normals_int} unique normals")
        face_count_val = dfm_geo_notes.get("face_count")
        try:
            face_count_int = int(face_count_val) if face_count_val is not None else None
        except Exception:
            face_count_int = None
        if face_count_int and face_count_int > 0:
            dfm_context_bits.append(f"{face_count_int} faces")
        deburr_edge_val = dfm_geo_notes.get("deburr_edge_len_mm")
        try:
            deburr_edge_num = float(deburr_edge_val) if deburr_edge_val is not None else None
        except Exception:
            deburr_edge_num = None
        if deburr_edge_num and deburr_edge_num > 0:
            dfm_context_bits.append(f"deburr edge ≈ {deburr_edge_num:.0f} mm")

    if has_tight_tol_flag:
        dfm_context_bits.append("tight tolerance callouts")

    tol_inputs_for_notes = tolerance_inputs if isinstance(tolerance_inputs, dict) else None
    if tol_inputs_for_notes:
        tol_labels = [str(k).strip() for k in tol_inputs_for_notes.keys() if str(k).strip()]
        if tol_labels:
            dfm_context_bits.append("tolerance inputs: " + ", ".join(sorted(tol_labels)[:3]))

    default_tol_note_for_notes = derived_for_notes.get("default_tolerance_note")
    if isinstance(default_tol_note_for_notes, str) and default_tol_note_for_notes.strip():
        dfm_context_bits.append(default_tol_note_for_notes.strip()[:120])

    if dfm_context_bits:
        unique_bits = list(dict.fromkeys(dfm_context_bits))
        llm_notes.append("DFM factors: " + "; ".join(unique_bits[:4]))

    tol_plan = overrides.get("tolerance_impacts") if isinstance(overrides, dict) else None
    if isinstance(tol_plan, dict) and tol_plan:
        tol_bits: list[str] = []
        for key in ("in_process_inspection_hr", "final_inspection_hr", "finishing_hr"):
            if key in tol_plan:
                try:
                    tol_bits.append(f"{key.replace('_',' ')} +{float(tol_plan[key]):.2f}h")
                except Exception:
                    continue
        surface = tol_plan.get("suggested_finish")
        if isinstance(surface, str) and surface.strip():
            tol_bits.append(surface.strip()[:80])
        if tol_bits:
            llm_notes.append("Tolerance/finish: " + "; ".join(tol_bits))

    op_sequence = overrides.get("operation_sequence") if isinstance(overrides, dict) else None
    if isinstance(op_sequence, list) and op_sequence:
        cleaned_ops = [str(step).strip() for step in op_sequence if str(step).strip()]
        if cleaned_ops:
            llm_notes.append("Ops: " + " → ".join(cleaned_ops[:8]))

    drilling_strategy = overrides.get("drilling_strategy") if isinstance(overrides, dict) else None
    if isinstance(drilling_strategy, dict) and drilling_strategy:
        strat_bits: list[str] = []
        mult_val = drilling_strategy.get("multiplier")
        if isinstance(mult_val, (int, float)):
            strat_bits.append(f"mult ×{float(mult_val):.2f}")
        floor_val = drilling_strategy.get("per_hole_floor_sec")
        if isinstance(floor_val, (int, float)):
            strat_bits.append(f">={float(floor_val):.0f}s/hole")
        note_val = drilling_strategy.get("note") or drilling_strategy.get("reason")
        if isinstance(note_val, str) and note_val.strip():
            strat_bits.append(note_val.strip()[:80])
        if strat_bits:
            llm_notes.append("Drilling strategy: " + "; ".join(strat_bits))

    shipping_hint = overrides.get("shipping_hint") if isinstance(overrides, dict) else None
    if isinstance(shipping_hint, str) and shipping_hint.strip():
        llm_notes.append(f"Shipping: {shipping_hint.strip()[:80]}")

    fai_flag_override = overrides.get("fai_required") if isinstance(overrides, dict) else None
    if fai_flag_override:
        llm_notes.append("FAI required")

    applied_process: dict[str, dict[str, Any]] = {}
    applied_pass: dict[str, dict[str, Any]] = {}

    def _clamp_override(value: Any, lo: float | None, hi: float | None) -> float | None:
        if value is None:
            return None
        try:
            num = float(value)
        except Exception:
            return None
        if lo is not None:
            num = max(lo, num)
        if hi is not None:
            num = min(hi, num)
        return num

    def _ensure_process_entry(proc_key: str) -> dict[str, Any]:
        entry = applied_process.setdefault(
            proc_key,
            {
                "old_hr": float(process_meta.get(proc_key, {}).get("hr", 0.0)),
                "old_cost": float(process_costs.get(proc_key, 0.0)),
                "notes": [],
            },
        )
        entry.setdefault("notes", [])
        return entry

    def _update_process_hours(proc_key: str, new_hr: float, note: str | None = None) -> None:
        meta = process_meta.get(proc_key)
        if not meta:
            return
        old_hr = float(meta.get("hr", 0.0))
        if math.isclose(new_hr, old_hr, abs_tol=1e-6):
            return
        entry = _ensure_process_entry(proc_key)
        if note:
            entry["notes"].append(note)
        meta["hr"] = new_hr
        rate = float(meta.get("rate", 0.0))
        base_extra = float(meta.get("base_extra", 0.0))
        new_cost = new_hr * rate + base_extra
        process_costs[proc_key] = round(new_cost, 2)
        entry["new_hr"] = new_hr
        entry["new_cost"] = process_costs[proc_key]

    source_lookup = quote_state.effective_sources if isinstance(quote_state.effective_sources, dict) else {}

    def _source_suffix(key: str) -> str:
        tag = source_lookup.get(key)
        if tag == "user":
            return " (user override)"
        if tag == "llm":
            return " (LLM)"
        return ""

    def _normalize_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    process_key_map = {_normalize_key(k): k for k in process_costs.keys()}
    pass_key_map = {
        _normalize_key(_canonical_pass_label(k)): _canonical_pass_label(k)
        for k in pass_through.keys()
    }

    def _friendly_process(name: str) -> str:
        return name.replace("_", " ").title()

    def _cost_of(proc_key: str, hours: float) -> float:
        parts = proc_key.replace("_", " ").split()
        rate_key = "".join(part.title() for part in parts) + "Rate"
        return float(hours) * float(rates.get(rate_key, rates.get("MillingRate", 120.0)))

    fixture_notes: list[str] = []
    fixture_build_override = _clamp_override((overrides or {}).get("fixture_build_hr"), 0.0, 2.0)
    soft_jaw_hr_override = _clamp_override((overrides or {}).get("soft_jaw_hr"), 0.0, 1.0) or 0.0
    soft_jaw_cost_override = _clamp_override((overrides or {}).get("soft_jaw_material_cost"), 0.0, 60.0) or 0.0

    total_fixture_hr = fixture_build_override if fixture_build_override is not None else fixture_build_hr_base
    if soft_jaw_hr_override > 0:
        total_fixture_hr += soft_jaw_hr_override
        fixture_notes.append(f"Soft jaw prep +{soft_jaw_hr_override:.2f} h{_source_suffix('soft_jaw_hr')}")
    if fixture_build_override is not None:
        fixture_notes.append(f"Fixture build set to {fixture_build_override:.2f} h{_source_suffix('fixture_build_hr')}")

    if soft_jaw_cost_override > 0:
        fixture_notes.append(f"Soft jaw stock +${soft_jaw_cost_override:,.2f}{_source_suffix('soft_jaw_material_cost')}")

    if total_fixture_hr != fixture_build_hr_base or soft_jaw_cost_override > 0:
        fixture_build_hr = total_fixture_hr
        fixture_labor_cost = fixture_build_hr * float(rates.get("FixtureBuildRate", 0.0))
        fixture_cost = fixture_labor_cost
        fixture_labor_per_part = (fixture_labor_cost / Qty) if Qty > 1 else fixture_labor_cost
        fixture_per_part = (fixture_cost / Qty) if Qty > 1 else fixture_cost
        nre_detail.setdefault("fixture", {})
        nre_detail["fixture"].update(
            {
                "build_hr": float(fixture_build_hr),
                "labor_cost": float(fixture_labor_cost),
                "per_lot": float(fixture_cost),
                "per_part": float(fixture_per_part),
                "soft_jaw_hr": float(soft_jaw_hr_override),
                "soft_jaw_mat": float(soft_jaw_cost_override),
            }
        )
        features["fixture_build_hr"] = float(fixture_build_hr)

    handling_override = _clamp_override((overrides or {}).get("handling_adder_hr"), 0.0, 0.2)
    if handling_override and handling_override > 0:
        base_milling_hr = float(process_meta.get("milling", {}).get("hr", 0.0))
        _update_process_hours(
            "milling",
            base_milling_hr + handling_override,
            f"+{handling_override:.2f} h handling{_source_suffix('handling_adder_hr')}",
        )
        llm_notes.append(f"Handling adder +{handling_override:.2f} h{_source_suffix('handling_adder_hr')}")

    packaging_hours_override = _clamp_override((overrides or {}).get("packaging_hours"), 0.0, 0.5)
    if packaging_hours_override is not None:
        _update_process_hours(
            "packaging",
            packaging_hours_override,
            f"Packaging set to {packaging_hours_override:.2f} h{_source_suffix('packaging_hours')}",
        )

    packaging_flat_override = _clamp_override((overrides or {}).get("packaging_flat_cost"), 0.0, 25.0)
    if packaging_flat_override is not None:
        baseline_val = float(pass_through_baseline.get("Packaging Flat", 0.0))
        entry = applied_pass.setdefault("Packaging Flat", {"old_value": baseline_val, "notes": []})
        entry["notes"].append(f"set to ${packaging_flat_override:,.2f}{_source_suffix('packaging_flat_cost')}")
        entry["new_value"] = float(packaging_flat_override)
        pass_through["Packaging Flat"] = float(packaging_flat_override)

    shipping_override = _clamp_override((overrides or {}).get("shipping_cost"), 0.0, None)
    if shipping_override is not None:
        baseline_val = float(pass_through_baseline.get("Shipping", shipping_cost_base))
        entry = applied_pass.setdefault("Shipping", {"old_value": baseline_val, "notes": []})
        entry["notes"].append(f"set to ${shipping_override:,.2f}{_source_suffix('shipping_cost')}")
        entry["new_value"] = float(shipping_override)
        pass_through["Shipping"] = float(shipping_override)
        pass_key_map[_normalize_key("Shipping")] = "Shipping"
        pass_meta.setdefault("Shipping", {})["basis"] = "Outbound freight & logistics (user override)"

    cmm_minutes_override = _clamp_override((overrides or {}).get("cmm_minutes"), 0.0, 60.0)
    if cmm_minutes_override is not None:
        base_cmm_hr = float(cmm_run_hr or 0.0)
        target_cmm_hr = float(cmm_minutes_override) / 60.0
        total_inspection_hr = float(process_meta.get("inspection", {}).get("hr", 0.0))
        adjusted_hr = max(0.0, total_inspection_hr - base_cmm_hr + target_cmm_hr)
        _update_process_hours(
            "inspection",
            adjusted_hr,
            f"CMM {target_cmm_hr:.2f} h{_source_suffix('cmm_minutes')}",
        )
        llm_notes.append(f"CMM runtime {target_cmm_hr:.2f} h{_source_suffix('cmm_minutes')}")
        delta_cmm = target_cmm_hr - base_cmm_hr
        if abs(delta_cmm) > 1e-9:
            inspection_adjustments["cmm_run"] = inspection_adjustments.get("cmm_run", 0.0) + delta_cmm
        cmm_run_hr = target_cmm_hr

    inproc_override = _clamp_override((overrides or {}).get("in_process_inspection_hr"), 0.0, 0.5)
    if inproc_override and inproc_override > 0:
        current_inspection_hr = float(process_meta.get("inspection", {}).get("hr", 0.0))
        _update_process_hours(
            "inspection",
            current_inspection_hr + inproc_override,
            f"+{inproc_override:.2f} h in-process{_source_suffix('in_process_inspection_hr')}",
        )
        llm_notes.append(f"In-process inspection +{inproc_override:.2f} h{_source_suffix('in_process_inspection_hr')}")
        inspection_adjustments["in_process"] = (
            inspection_adjustments.get("in_process", 0.0) + inproc_override
        )

    fai_prep_override = _clamp_override((overrides or {}).get("fai_prep_hr"), 0.0, 1.0)
    if fai_prep_override and fai_prep_override > 0:
        current_inspection_hr = float(process_meta.get("inspection", {}).get("hr", 0.0))
        _update_process_hours(
            "inspection",
            current_inspection_hr + fai_prep_override,
            f"+{fai_prep_override:.2f} h FAI prep{_source_suffix('fai_prep_hr')}",
        )
        llm_notes.append(f"FAI prep +{fai_prep_override:.2f} h{_source_suffix('fai_prep_hr')}")
        inspection_adjustments["fai_prep"] = (
            inspection_adjustments.get("fai_prep", 0.0) + fai_prep_override
        )

    inspection_total_override = _clamp_override((overrides or {}).get("inspection_total_hr"), 0.0, 12.0)
    if inspection_total_override is not None:
        current_inspection_hr = float(process_meta.get("inspection", {}).get("hr", 0.0))
        _update_process_hours(
            "inspection",
            inspection_total_override,
            f"Inspection set to {inspection_total_override:.2f} h{_source_suffix('inspection_total_hr')}",
        )
        llm_notes.append(
            f"Inspection total set to {inspection_total_override:.2f} h{_source_suffix('inspection_total_hr')}"
        )
        delta_total = inspection_total_override - current_inspection_hr
        if abs(delta_total) > 1e-9:
            inspection_adjustments["total_override"] = (
                inspection_adjustments.get("total_override", 0.0) + delta_total
            )

    if fixture_notes:
        llm_notes.extend(fixture_notes)

    material_direct_cost_base = material_direct_cost

    old_scrap = normalize_scrap_pct(features.get("scrap_pct", scrap_pct))
    base_net_mass_g = _coerce_float_or_none(material_detail_for_breakdown.get("net_mass_g")) or 0.0

    removal_mass_g: float | None = None
    removal_lb: float | None = None
    if isinstance(overrides, dict):
        for key in ("material_removed_mass_g", "material_removed_g"):
            val = overrides.get(key)
            num = _coerce_float_or_none(val)
            if num is not None and num > 0:
                removal_mass_g = float(num)
                break
        if removal_mass_g is None:
            removal_lb_val = _coerce_float_or_none(overrides.get("material_removed_lb"))
            if removal_lb_val is not None and removal_lb_val > 0:
                removal_lb = float(removal_lb_val)
                removal_mass_g = removal_lb / LB_PER_KG * 1000.0
    if removal_mass_g is not None and removal_mass_g > 0 and removal_lb is None:
        removal_lb = removal_mass_g / 1000.0 * LB_PER_KG

    bounds = quote_state.bounds if isinstance(quote_state.bounds, dict) else {}
    scrap_min_bound = max(0.0, _coerce_float_or_none(bounds.get("scrap_min")) or 0.0)
    scrap_max_bound = _coerce_float_or_none(bounds.get("scrap_max")) or 0.25

    net_after = base_net_mass_g
    scrap_after = old_scrap
    effective_mass_after = base_net_mass_g * (1.0 + old_scrap)
    mass_scale = 1.0
    removal_applied = False

    if removal_mass_g and removal_mass_g > 0 and base_net_mass_g > 0:
        net_after, scrap_after, effective_mass_after = compute_mass_and_scrap_after_removal(
            base_net_mass_g,
            old_scrap,
            removal_mass_g,
            scrap_min=scrap_min_bound,
            scrap_max=scrap_max_bound,
        )
        mass_scale = net_after / max(1e-6, base_net_mass_g)
        removal_applied = True

    scrap_override_raw = overrides.get("scrap_pct_override") if isinstance(overrides, dict) else None
    scrap_override_applied = False
    if scrap_override_raw is not None:
        override_scrap = normalize_scrap_pct(scrap_override_raw)
        if not math.isclose(override_scrap, scrap_after, abs_tol=1e-6):
            scrap_override_applied = True
        scrap_after = override_scrap
        effective_mass_after = net_after * (1.0 + scrap_after)

    scrap_scale = (1.0 + scrap_after) / max(1e-6, (1.0 + old_scrap))
    total_scale = mass_scale * scrap_scale

    scrap_changed = not math.isclose(scrap_after, old_scrap, abs_tol=1e-6)
    change_triggered = removal_applied or scrap_override_applied or scrap_changed or not math.isclose(total_scale, 1.0, abs_tol=1e-6)

    if change_triggered:
        baseline = float(features.get("material_cost_baseline", material_direct_cost_base))
        scaled = round(baseline * total_scale, 2)
        pass_through["Material"] = scaled
        entry = applied_pass.setdefault("Material", {"old_value": float(material_direct_cost_base), "notes": []})
        if removal_applied and removal_lb is not None:
            entry["notes"].append(f"material removal -{removal_lb:.2f} lb")
        elif removal_applied:
            entry["notes"].append(f"material removal -{removal_mass_g:.1f} g")
        if scrap_changed:
            src_tag = (
                quote_state.effective_sources.get("scrap_pct")
                if quote_state and isinstance(quote_state.effective_sources, dict)
                else None
            )
            suffix = ""
            if scrap_override_applied:
                if src_tag == "user":
                    suffix = " (user override)"
                elif src_tag == "llm":
                    suffix = " (LLM)"
            elif removal_applied:
                suffix = " (material removal)"
            entry["notes"].append(
                f"scrap {old_scrap * 100:.1f}% → {scrap_after * 100:.1f}%{suffix}"
            )
        entry["new_value"] = scaled

        if removal_applied:
            removal_note = (
                f"Material removal -{removal_lb:.2f} lb → net {net_after / 1000.0 * LB_PER_KG:.2f} lb"
                if removal_lb is not None
                else f"Material removal -{removal_mass_g:.1f} g"
            )
            llm_notes.append(removal_note)
        if scrap_changed:
            src_tag = (
                quote_state.effective_sources.get("scrap_pct")
                if quote_state and isinstance(quote_state.effective_sources, dict)
                else None
            )
            suffix = ""
            if scrap_override_applied:
                if src_tag == "user":
                    suffix = " (user override)"
                elif src_tag == "llm":
                    suffix = " (LLM)"
            elif removal_applied:
                suffix = " (material removal)"
            llm_notes.append(
                f"Scrap {old_scrap * 100:.1f}% → {scrap_after * 100:.1f}%{suffix}"
            )

        features["scrap_pct"] = scrap_after
        if removal_applied:
            features["material_removed_mass_g"] = removal_mass_g
            features["net_mass_g"] = net_after

    scrap_pct = scrap_after
    material_detail_for_breakdown["mass_g_net"] = net_after
    material_detail_for_breakdown["net_mass_g"] = net_after
    material_detail_for_breakdown["effective_mass_g"] = effective_mass_after
    material_detail_for_breakdown["mass_g"] = effective_mass_after
    material_detail_for_breakdown["scrap_pct"] = scrap_pct
    if change_triggered:
        material_detail_for_breakdown["material_cost"] = pass_through.get(
            "Material", material_detail_for_breakdown.get("material_cost")
        )
        material_detail_for_breakdown["material_direct_cost"] = pass_through.get(
            "Material", material_detail_for_breakdown.get("material_direct_cost")
        )

    material_detail_for_breakdown["scrap_pct"] = scrap_pct

    src_mult_map = {}
    if isinstance(quote_state.effective_sources, dict):
        src_mult_map = quote_state.effective_sources.get("process_hour_multipliers") or {}
    ph_mult = overrides.get("process_hour_multipliers") if overrides else {}
    for key, mult in (ph_mult or {}).items():
        if not isinstance(mult, (int, float)):
            continue
        actual = process_key_map.get(_normalize_key(key))
        if not actual:
            continue
        mult = clamp(mult, LLM_MULTIPLIER_MIN, LLM_MULTIPLIER_MAX, 1.0)
        old_cost = float(process_costs.get(actual, 0.0))
        if old_cost <= 0:
            continue
        entry = applied_process.setdefault(
            actual,
            {
                "old_hr": float(process_meta.get(actual, {}).get("hr", 0.0)),
                "old_cost": old_cost,
                "notes": [],
            },
        )
        process_costs[actual] = round(old_cost * mult, 2)
        entry["notes"].append(f"x{mult:.2f}")
        meta = process_meta.get(actual)
        if meta:
            rate = float(meta.get("rate", 0.0))
            base_extra = float(meta.get("base_extra", 0.0))
            if rate > 0:
                meta["hr"] = max(0.0, (process_costs[actual] - base_extra) / rate)
        entry["new_hr"] = float(process_meta.get(actual, {}).get("hr", entry["old_hr"]))
        entry["new_cost"] = process_costs[actual]
        src_tag = src_mult_map.get(key)
        suffix = ""
        if src_tag == "user":
            suffix = " (user override)"
        elif src_tag == "llm":
            suffix = " (LLM)"
        llm_notes.append(f"{_friendly_process(actual)} hours x{mult:.2f}{suffix}")

    src_add_map = {}
    if isinstance(quote_state.effective_sources, dict):
        src_add_map = quote_state.effective_sources.get("process_hour_adders") or {}
    ph_add = overrides.get("process_hour_adders") if overrides else {}
    for key, add_hr in (ph_add or {}).items():
        if not isinstance(add_hr, (int, float)):
            continue
        add_hr = clamp(add_hr, 0.0, LLM_ADDER_MAX, 0.0)
        if add_hr <= 0:
            continue
        actual = process_key_map.get(_normalize_key(key))
        if not actual:
            continue
        entry = applied_process.setdefault(
            actual,
            {
                "old_hr": float(process_meta.get(actual, {}).get("hr", 0.0)),
                "old_cost": float(process_costs.get(actual, 0.0)),
                "notes": [],
            },
        )
        delta_cost = _cost_of(actual, add_hr)
        process_costs[actual] = round(float(process_costs.get(actual, 0.0)) + delta_cost, 2)
        meta = process_meta.get(actual)
        if meta:
            meta["hr"] = float(meta.get("hr", 0.0)) + float(add_hr)
        entry["notes"].append(f"+{float(add_hr):.2f} hr")
        entry["new_hr"] = float(process_meta.get(actual, {}).get("hr", entry["old_hr"]))
        entry["new_cost"] = process_costs[actual]
        src_tag = src_add_map.get(key)
        suffix = ""
        if src_tag == "user":
            suffix = " (user override)"
        elif src_tag == "llm":
            suffix = " (LLM)"
        llm_notes.append(f"{_friendly_process(actual)} +{float(add_hr):.2f} hr{suffix}")

    src_pass_map: dict[str, Any] = {}
    if isinstance(quote_state.effective_sources, dict):
        raw_src_pass = quote_state.effective_sources.get("add_pass_through") or {}
        if isinstance(raw_src_pass, Mapping):
            for key, value in raw_src_pass.items():
                canon_key = _canonical_pass_label(key)
                if canon_key:
                    src_pass_map[canon_key] = value
    add_pass_raw = overrides.get("add_pass_through") if overrides else {}
    add_pass = _canonicalize_pass_through_map(add_pass_raw)
    for label, add_val in (add_pass or {}).items():
        if not isinstance(add_val, (int, float)):
            continue
        add_val = clamp(add_val, 0.0, 200.0, 0.0)
        if add_val <= 0:
            continue
        actual_label = pass_key_map.get(_normalize_key(label), str(label))
        old_val = float(pass_through.get(actual_label, 0.0))
        new_val = round(old_val + float(add_val), 2)
        pass_through[actual_label] = new_val
        pass_key_map[_normalize_key(actual_label)] = actual_label
        entry = applied_pass.setdefault(actual_label, {"old_value": old_val, "notes": []})
        entry["notes"].append(f"+${float(add_val):,.2f}")
        entry["new_value"] = new_val
        if actual_label not in pass_meta:
            pass_meta[actual_label] = {"basis": "LLM override"}
        src_tag = src_pass_map.get(actual_label)
        suffix = ""
        if src_tag == "user":
            suffix = " (user override)"
        elif src_tag == "llm":
            suffix = " (LLM)"
        llm_notes.append(f"{actual_label}: +${float(add_val):,.0f}{suffix}")

    cont_override = overrides.get("contingency_pct_override") if overrides else None
    if cont_override is not None:
        cont_val = clamp(cont_override, 0.0, 0.25, ContingencyPct)
        if cont_val is not None:
            ContingencyPct = float(cont_val)
            params["ContingencyPct"] = ContingencyPct
            src_tag = None
            if isinstance(quote_state.effective_sources, dict):
                src_tag = quote_state.effective_sources.get("contingency_pct")
            suffix = ""
            if src_tag == "user":
                suffix = " (user override)"
            elif src_tag == "llm":
                suffix = " (LLM)"
            llm_notes.append(f"Contingency set to {ContingencyPct * 100:.1f}%{suffix}")

    for label, value in list(pass_through.items()):
        try:
            pass_through[label] = round(float(value), 2)
        except Exception:
            pass_through[label] = 0.0

    try:
        setups_for_cap = float(setups)
    except Exception:
        setups_for_cap = 1.0
    if not math.isfinite(setups_for_cap) or setups_for_cap <= 0:
        setups_for_cap = 1.0
    hour_ceiling = max(0.0, 24.0 * setups_for_cap)

    def _collect_process_hours() -> dict[str, float]:
        bounded: dict[str, float] = {}
        for key, meta in process_meta.items():
            raw_val: Any = meta
            if isinstance(meta, Mapping):
                raw_val = meta.get("hr", 0.0)
            try:
                hr_val = float(raw_val or 0.0)
            except Exception:
                hr_val = 0.0
            if not math.isfinite(hr_val):
                hr_val = 0.0
            hr_val = max(0.0, min(hour_ceiling, hr_val))
            if isinstance(meta, dict):
                meta["hr"] = hr_val
            bounded[key] = hr_val
        return bounded

    process_hours_final = _collect_process_hours()

    if pricing_source != "planner":
        hole_count_for_guard = 0
        try:
            hole_count_for_guard = int(float(geo_context.get("hole_count", 0) or 0))
        except Exception:
            pass
        if hole_count_for_guard <= 0:
            hole_count_for_guard = len(geo_context.get("hole_diams_mm", []) or [])

        def _hours_for_guard(key: str) -> float:
            if key in process_hours_final:
                return float(process_hours_final.get(key, 0.0) or 0.0)
            try:
                return float(process_meta.get(key, {}).get("hr", 0.0) or 0.0)
            except Exception:
                return 0.0

        def _apply_floor_guard(
            proc_key: str,
            *,
            floor_hr: float,
            validator: typing.Callable[[float], tuple[bool, str]],
        ) -> None:
            floor_val = max(0.0, float(floor_hr or 0.0))
            if floor_val <= 0.0:
                return
            current_hr = _hours_for_guard(proc_key)
            ok, reason = validator(current_hr)
            if ok:
                return
            meta = process_meta.setdefault(proc_key, {})
            try:
                rate = float(meta.get("rate", 0.0) or 0.0)
            except Exception:
                rate = 0.0
            try:
                base_extra = float(meta.get("base_extra", 0.0) or 0.0)
            except Exception:
                base_extra = 0.0
            try:
                old_hr = float(meta.get("hr", 0.0) or 0.0)
            except Exception:
                old_hr = current_hr
            old_cost = float(process_costs.get(proc_key, 0.0) or 0.0)
            entry = applied_process.setdefault(
                proc_key,
                {
                    "old_hr": old_hr,
                    "old_cost": old_cost,
                    "notes": [],
                },
            )
            entry.setdefault("notes", []).append(f"Raised to floor {floor_val:.2f} h")
            meta["hr"] = floor_val
            process_costs[proc_key] = round(floor_val * rate + base_extra, 2)
            process_hours_final[proc_key] = floor_val
            llm_notes.append(
                f"Raised {_friendly_process(proc_key)} to floor: {reason}"
            )

        if hole_count_for_guard > 0:
            floor_hr = _drilling_floor_hours(hole_count_for_guard)
            _apply_floor_guard(
                "drilling",
                floor_hr=floor_hr,
                validator=lambda hr, count=hole_count_for_guard: validate_drilling_reasonableness(
                    count, hr
                ),
            )

        tap_context = ""
        if tap_qty_geo:
            tap_context = f"for {int(tap_qty_geo)} taps"
        elif tap_minutes_hint_val and tap_minutes_hint_val > 0:
            tap_context = "(tap minutes hint)"
        elif tapping_minutes_total > 0:
            tap_context = "(tap class estimate)"
        if tapping_hr_floor > 0:
            _apply_floor_guard(
                "tapping",
                floor_hr=tapping_hr_floor,
                validator=lambda hr, floor=tapping_hr_floor, ctx=tap_context: validate_process_floor_hours(
                    "tapping", hr, floor, ctx
                ),
            )

        cbore_context = ""
        if cbore_qty_geo:
            cbore_context = f"for {int(cbore_qty_geo)} counterbores"
        elif cbore_minutes_hint_val and cbore_minutes_hint_val > 0:
            cbore_context = "(counterbore minutes hint)"
        if cbore_hr_floor > 0:
            _apply_floor_guard(
                "counterbore",
                floor_hr=cbore_hr_floor,
                validator=lambda hr, floor=cbore_hr_floor, ctx=cbore_context: validate_process_floor_hours(
                    "counterbore", hr, floor, ctx
                ),
            )

        csk_context = ""
        if csk_qty_geo:
            csk_context = f"for {int(csk_qty_geo)} countersinks"
        elif csk_minutes_hint_val and csk_minutes_hint_val > 0:
            csk_context = "(countersink minutes hint)"
        if csk_hr_floor > 0:
            _apply_floor_guard(
                "countersink",
                floor_hr=csk_hr_floor,
                validator=lambda hr, floor=csk_hr_floor, ctx=csk_context: validate_process_floor_hours(
                    "countersink", hr, floor, ctx
                ),
            )

        if deburr_total_hr_floor > 0:
            _apply_floor_guard(
                "deburr",
                floor_hr=deburr_total_hr_floor,
                validator=lambda hr, floor=deburr_total_hr_floor: validate_process_floor_hours(
                    "deburr", hr, floor, "(auto/manual estimate)"
                ),
            )
        process_hours_final = _collect_process_hours()
        baseline_total_hours = sum(
            float(process_hours_baseline.get(k, 0.0)) for k in process_hours_baseline
        )
        final_total_hours = sum(
            float(process_hours_final.get(k, 0.0)) for k in process_hours_final
        )
        if baseline_total_hours > 1e-6:
            ratio = final_total_hours / baseline_total_hours if baseline_total_hours else 1.0
            if ratio < 0.6 or ratio > 3.0:
                msg = (
                    f"Process hours shift {baseline_total_hours:.2f}h → {final_total_hours:.2f}h "
                    f"({ratio:.2f}×); confirm change"
                )
                overrides_meta.setdefault("alerts", []).append(msg)
                overrides_meta["confirm_required"] = True
                llm_notes.append(f"Requires confirm: {msg}")

        validate_quote_before_pricing(
            geo_context, process_costs, pass_through, process_hours_final
        )
    else:
        process_hours_final = {}
        for key, meta in process_meta.items():
            if key not in allowed_process_hour_keys or not isinstance(meta, dict):
                continue
            try:
                hr_val = float(meta.get("hr", 0.0) or 0.0)
            except Exception:
                hr_val = 0.0
            process_hours_final[key] = hr_val
            if key in process_costs:
                rate_val = float(meta.get("rate", 0.0) or 0.0)
                base_extra_val = float(meta.get("base_extra", 0.0) or 0.0)
                process_costs[key] = round(hr_val * rate_val + base_extra_val, 2)
        process_hours_final = _collect_process_hours()

    meta_lookup = process_meta if isinstance(process_meta, dict) else {}
    for proc_key, final_hr in process_hours_final.items():
        if pricing_source == "planner" and proc_key not in process_costs:
            continue
        meta = meta_lookup.get(proc_key) or {}
        try:
            rate = float(meta.get("rate", 0.0) or 0.0)
        except Exception:
            rate = 0.0
        try:
            base_extra = float(meta.get("base_extra", 0.0) or 0.0)
        except Exception:
            base_extra = 0.0
        hr_clean = max(0.0, float(final_hr or 0.0))
        if rate > 0 or base_extra != 0.0:
            new_cost = hr_clean * rate + base_extra
            process_costs[proc_key] = round(new_cost, 2)
        elif "cost" in meta:
            try:
                process_costs[proc_key] = round(float(meta.get("cost", 0.0) or 0.0), 2)
            except Exception:
                pass

    for key, entry in applied_process.items():
        final_hr = process_hours_final.get(key)
        if final_hr is None:
            try:
                final_hr = float(process_meta.get(key, {}).get("hr", entry.get("new_hr", entry["old_hr"])))
            except Exception:
                final_hr = entry.get("new_hr", entry["old_hr"])
        entry["new_hr"] = float(final_hr or 0.0)
        entry["new_cost"] = float(process_costs.get(key, entry.get("new_cost", entry["old_cost"])))
        entry["delta_hr"] = entry["new_hr"] - entry["old_hr"]
        entry["delta_cost"] = entry["new_cost"] - entry["old_cost"]

    for label, entry in applied_pass.items():
        entry["new_value"] = float(pass_through.get(label, entry.get("new_value", entry["old_value"])))
        entry["delta_value"] = entry["new_value"] - entry["old_value"]

    auto_prog_hr, auto_components = _estimate_programming_hours_auto(
        geo_context,
        process_meta,
        params,
        setups_hint=float(setups),
    )
    total_prog_hr = float(auto_prog_hr)
    prog_override_applied = False
    if prog_hr_override is not None:
        try:
            override_val = max(0.0, float(prog_hr_override))
        except Exception:
            override_val = None
        else:
            total_prog_hr = override_val
            prog_override_applied = True

    # Apply programming caps after auto/override is known
    try:
        if is_simple:
            total_prog_hr = min(float(total_prog_hr), float(params.get("ProgCapHr", total_prog_hr)))
        if milling_hr and float(milling_hr) > 0:
            ratio_cap = float(params.get("ProgMaxToMillingRatio", 1.0)) * float(milling_hr)
            total_prog_hr = min(float(total_prog_hr), ratio_cap)
    except Exception:
        # Defensive: do not fail rendering if caps are malformed; keep computed hours
        pass

    programming_cost = total_prog_hr * float(rates.get("ProgrammingRate", 0.0)) + eng_hr * float(rates.get("EngineerRate", 0.0))
    prog_per_lot = programming_cost
    programming_per_part = (prog_per_lot / Qty) if (amortize_programming and Qty > 1) else prog_per_lot

    programming_detail = {
        "prog_hr": float(total_prog_hr),
        "prog_rate": float(rates.get("ProgrammingRate", 0.0)),
        "eng_hr": float(eng_hr),
        "eng_rate": float(rates.get("EngineerRate", 0.0)),
        "per_lot": float(prog_per_lot),
        "per_part": float(programming_per_part),
        "amortized": bool(amortize_programming and Qty > 1),
        "auto_prog_hr": float(auto_prog_hr),
    }
    if auto_components:
        programming_detail["auto_components"] = {k: float(v) for k, v in auto_components.items()}
    if prog_override_applied:
        programming_detail["override_applied"] = True

    nre_detail["programming"] = programming_detail

    if (
        planner_bucket_view is None
        and planner_pricing_result is not None
        and planner_two_bucket_rates is not None
    ):
        programming_min = float(programming_detail.get("prog_hr", 0.0) or 0.0) * 60.0
        fixture_min = float(fixture_build_hr or 0.0) * 60.0
        nre_minutes = {
            "programming_min": programming_min,
            "fixture_min": fixture_min,
        }
        try:
            qty_for_bucket = int(round(float(Qty)))
        except Exception:
            qty_for_bucket = 1
        qty_for_bucket = max(1, qty_for_bucket)
        try:
            raw_bucket_view = bucketize(
                planner_pricing_result,
                planner_two_bucket_rates,
                nre_minutes,
                qty=qty_for_bucket,
                geom=planner_geom_payload or {},
            )
        except Exception as bucketize_err:
            if planner_pricing_error is None:
                planner_pricing_error = str(bucketize_err)
            if FORCE_PLANNER:
                raise ConfigError(
                    "FORCE_PLANNER enabled but planner bucketization failed"
                ) from bucketize_err
        else:
            planner_bucket_view = _prepare_bucket_view(raw_bucket_view)
            process_plan_summary.setdefault("bucket_view", copy.deepcopy(planner_bucket_view))
            existing_plan = getattr(quote_state, "process_plan", None)
            if isinstance(existing_plan, dict):
                existing_plan.setdefault("bucket_view", copy.deepcopy(planner_bucket_view))

    material_direct_cost = float(
        pass_through.get(_canonical_pass_label("Material"), material_direct_cost_base)
    )
    hardware_cost = float(pass_through.get(HARDWARE_PASS_LABEL, hardware_cost))
    outsourced_costs = float(pass_through.get("Outsourced Vendors", outsourced_costs))
    shipping_cost = float(pass_through.get("Shipping", shipping_cost_base))
    consumables_hr_cost = float(pass_through.get("Consumables /Hr", consumables_hr_cost))
    utilities_cost = float(pass_through.get("Utilities", utilities_cost))
    consumables_flat = float(pass_through.get("Consumables", consumables_flat))

    if used_planner:
        machine_cost_final = float(
            process_meta.get("planner_machine", {}).get("cost", planner_machine_cost_total)
        )
        labor_cost_final = float(
            process_meta.get("planner_labor", {}).get("cost", planner_labor_cost_total)
        )
        process_costs = {
            "Machine": round(machine_cost_final, 2),
            "Labor": round(labor_cost_final, 2),
        }

    labor_cost = programming_per_part + fixture_labor_per_part + sum(process_costs.values())

    base_direct_costs = sum(pass_through.values())
    vendor_markup = params["VendorMarkupPct"]
    insurance_cost = insurance_pct * (labor_cost + base_direct_costs)
    vendor_marked_add = vendor_markup * (outsourced_costs + shipping_cost)

    base_directs_excluding_ship_mat = sum(
        float(value)
        for label, value in pass_through.items()
        if label not in ("Material", "Shipping", "Material Scrap Credit")
    )

    if base_directs_excluding_ship_mat > 0:
        pass_meta.setdefault("Insurance", {"basis": "Applied at insurance pct"})
        pass_meta.setdefault("Vendor Markup Added", {"basis": "Vendor + freight markup"})
        pass_meta.setdefault("Shipping", {"basis": "Outbound freight & logistics"})
        insurance_cost = round(insurance_cost, 2)
        vendor_marked_add = round(vendor_marked_add, 2)
        pass_through["Insurance"] = insurance_cost
        pass_through["Vendor Markup Added"] = vendor_marked_add
        total_direct_costs = base_direct_costs + insurance_cost + vendor_marked_add
    else:
        insurance_cost = 0.0
        vendor_marked_add = 0.0
        total_direct_costs = base_direct_costs
    direct_costs = total_direct_costs

    subtotal = labor_cost + direct_costs

    with_overhead = subtotal * (1.0 + OverheadPct)
    with_ga = with_overhead * (1.0 + GA_Pct)
    with_cont = with_ga * (1.0 + ContingencyPct)
    with_expedite = with_cont * (1.0 + ExpeditePct)

    price_before_margin = with_expedite
    price = price_before_margin * (1.0 + MarginPct)

    min_lot = float(params["MinLotCharge"] or 0.0)
    price = max(price, min_lot)

    # Initialize labor detail seed for display/summary merging in this scope.
    # In other processing paths we derive this from breakdown input; here we
    # start with an empty seed so downstream merges are safe.
    labor_cost_details_seed: dict[str, str] = {}
    labor_cost_details_input: dict[str, str] = dict(labor_cost_details_seed)
    labor_cost_details: dict[str, str] = dict(labor_cost_details_seed)
    labor_costs_display: dict[str, float] = {}

    def _is_extra_segment(segment: str) -> bool:
        try:
            return bool(EXTRA_DETAIL_RE.match(str(segment)))
        except Exception:
            return False

    def _merge_labor_detail(label: str, amount: float, detail_bits: list[str]) -> None:
        canonical_label, _ = _canonical_amortized_label(label)
        if not canonical_label:
            canonical_label = str(label)
        labor_costs_display[canonical_label] = float(amount)
        existing_detail = labor_cost_details_input.get(canonical_label)
        merged_bits: list[str] = []
        seen: set[str] = set()

        def _append_segment(segment: str) -> None:
            seg = segment.strip()
            if not seg or _is_extra_segment(seg):
                return
            if seg not in seen:
                merged_bits.append(seg)
                seen.add(seg)

        for bit in detail_bits:
            _append_segment(str(bit))

        if existing_detail:
            for segment in re.split(r";\s*", existing_detail):
                _append_segment(segment)

        if merged_bits:
            merged_text = "; ".join(merged_bits)
            labor_cost_details_input[canonical_label] = merged_text
            labor_cost_details[canonical_label] = merged_text
        else:
            labor_cost_details_input.pop(canonical_label, None)
            labor_cost_details.pop(canonical_label, None)
    # Build a minimal process display list if not already available in this scope.
    # Earlier rendering helpers construct a richer list; here we only need
    # label/amount/detail_bits to merge labor detail text for the quote summary.
    try:
        _ = process_entries_for_display  # type: ignore[name-defined]
    except NameError:
        tmp_entries: list[ProcessDisplayEntry] = []
        for _key, _val in (process_costs or {}).items():
            try:
                amount_val = float(_val or 0.0)
            except Exception:
                amount_val = 0.0
            if amount_val <= 0.0:
                continue
            canon = _canonical_bucket_key(_key)
            label = _display_bucket_label(canon, None) if canon else _process_label(_key)
            tmp_entries.append(
                ProcessDisplayEntry(
                    process_key=str(_key),
                    canonical_key=canon,
                    label=label,
                    amount=amount_val,
                    detail_bits=tuple(),
                    display_override=None,
                )
            )
        process_entries_for_display = tmp_entries  # type: ignore[no-redef]

    for entry in process_entries_for_display:
        try:
            amount_val = float(entry.amount or 0.0)
        except Exception:
            amount_val = 0.0
        if amount_val <= 0.0:
            continue
        label = entry.display_override or entry.label
        _merge_labor_detail(label, amount_val, list(entry.detail_bits))

    show_programming_amortized = Qty > 1 and programming_per_part > 0
    if show_programming_amortized:
        programming_bits: list[str] = []
        prog_hr_detail = float(programming_detail.get("prog_hr", 0.0) or 0.0)
        prog_rate_detail = float(programming_detail.get("prog_rate", 0.0) or 0.0)
        if prog_hr_detail > 0:
            programming_bits.append(
                f"- Programmer (lot): {_hours_with_rate_text(prog_hr_detail, prog_rate_detail)}"
            )
        eng_hr_detail = float(programming_detail.get("eng_hr", 0.0) or 0.0)
        eng_rate_detail = float(programming_detail.get("eng_rate", 0.0) or 0.0)
        if eng_hr_detail > 0:
            programming_bits.append(
                f"- Engineering (lot): {_hours_with_rate_text(eng_hr_detail, eng_rate_detail)}"
            )
        programming_bits.append(f"Amortized across {Qty} pcs")

        _merge_labor_detail("Programming (amortized)", programming_per_part, programming_bits)

    show_fixture_amortized = Qty > 1 and fixture_labor_per_part > 0
    if show_fixture_amortized:
        fixture_bits: list[str] = []
        fixture_detail = nre_detail.get("fixture", {}) if isinstance(nre_detail, dict) else {}
        fixture_build_hr_detail = float(fixture_detail.get("build_hr", 0.0) or 0.0)
        fixture_rate_detail = float(
            fixture_detail.get("build_rate", rates.get("FixtureBuildRate", 0.0))
        )
        if fixture_build_hr_detail > 0:
            fixture_bits.append(
                f"- Build labor (lot): {_hours_with_rate_text(fixture_build_hr_detail, fixture_rate_detail)}"
            )
        soft_jaw_hr = float(fixture_detail.get("soft_jaw_hr", 0.0) or 0.0)
        if soft_jaw_hr > 0:
            fixture_bits.append(f"Soft jaw prep {soft_jaw_hr:.2f} hr")
        fixture_bits.append(f"Amortized across {Qty} pcs")

        _merge_labor_detail("Fixture Build (amortized)", fixture_labor_per_part, fixture_bits)

    try:
        labor_display_total = sum(float(value or 0.0) for value in labor_costs_display.values())
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Labor section invariant calculation failed")
    else:
        expected_labor_total = float(labor_cost or 0.0)
        diff = abs(labor_display_total - expected_labor_total)
        if diff > 0.5:
            flag_message = (
                f"Labor totals drifted by ${diff:,.2f}: "
                f"rendered ${labor_display_total:,.2f} vs expected ${expected_labor_total:,.2f}"
            )
            if flag_message not in red_flags:
                red_flags.append(flag_message)
            logger.warning(
                "Labor section totals drifted beyond threshold: %.2f vs %.2f",
                labor_display_total,
                expected_labor_total,
            )
        elif not math.isclose(
            labor_display_total,
            expected_labor_total,
            rel_tol=0.0,
            abs_tol=_LABOR_SECTION_ABS_EPSILON,
        ):
            logger.warning(
                "Labor section totals drifted: %.2f vs %.2f",
                labor_display_total,
                expected_labor_total,
            )

    direct_costs_display: dict[str, float] = {label: float(value) for label, value in pass_through.items()}
    direct_cost_details: dict[str, str] = {}
    for label, value in direct_costs_display.items():
        detail_bits: list[str] = []
        basis = pass_meta.get(label, {}).get("basis")
        if basis and label != "Material":
            detail_bits.append(f"Basis: {basis}")
        if label == "Insurance":
            detail_bits.append(f"Applied {insurance_pct:.1%} of labor + directs")
        elif label == "Vendor Markup Added":
            detail_bits.append(f"Markup {vendor_markup:.1%} on vendors + shipping")
        elif label == "Material":
            pass
        pass_notes = applied_pass.get(label, {}).get("notes")
        if pass_notes:
            detail_bits.append("LLM: " + ", ".join(pass_notes))
        if detail_bits:
            direct_cost_details[label] = "; ".join(detail_bits)

    prog_detail = nre_detail.get("programming", {})
    fix_detail = nre_detail.get("fixture", {})
    nre_costs_display: dict[str, float] = {}
    nre_cost_details: dict[str, str] = {}

    prog_per_lot = float(prog_detail.get("per_lot", 0.0) or 0.0)
    if prog_per_lot:
        label = "Programming & Eng (per lot)"
        nre_costs_display[label] = prog_per_lot
        details = []
        if prog_detail.get("prog_hr"):
            details.append(f"Programmer {prog_detail['prog_hr']:.2f} hr @ ${prog_detail.get('prog_rate',0):,.2f}/hr")
        if prog_detail.get("eng_hr"):
            details.append(f"Engineer {prog_detail['eng_hr']:.2f} hr @ ${prog_detail.get('eng_rate',0):,.2f}/hr")
        if details:
            nre_cost_details[label] = "; ".join(details)

    fix_per_lot = float(fix_detail.get("per_lot", 0.0) or 0.0)
    if fix_per_lot:
        label = "Fixturing (per lot)"
        nre_costs_display[label] = fix_per_lot
        details = []
        if fix_detail.get("build_hr"):
            details.append(f"Build {fix_detail['build_hr']:.2f} hr @ ${fix_detail.get('build_rate',0):,.2f}/hr")
        if details:
            nre_cost_details[label] = "; ".join(details)

    applied_multipliers_log: dict[str, float] = {}
    for key, value in (overrides.get("process_hour_multipliers") or {}).items():
        actual = process_key_map.get(_normalize_key(key), key)
        try:
            applied_multipliers_log[actual] = float(value)
        except Exception:
            applied_multipliers_log[actual] = value

    applied_adders_log: dict[str, float] = {}
    for key, value in (overrides.get("process_hour_adders") or {}).items():
        actual = process_key_map.get(_normalize_key(key), key)
        try:
            applied_adders_log[actual] = float(value)
        except Exception:
            applied_adders_log[actual] = value

    llm_cost_log = {
        "overrides": overrides,
        "applied_process": applied_process,
        "applied_pass": applied_pass,
        "scrap_pct": scrap_pct,
        "baseline_process_hours": process_hours_baseline,
        "baseline_pass_through": pass_through_baseline,
        "baseline_scrap_pct": scrap_pct_baseline,
        "final_process_hours": process_hours_final,
    }
    if llm_notes:
        llm_cost_log["notes"] = llm_notes
    if overrides_meta.get("raw") is not None:
        llm_cost_log["raw_response"] = overrides_meta["raw"]
    if overrides_meta.get("raw_text"):
        llm_cost_log["raw_response_text"] = overrides_meta["raw_text"]
    if overrides_meta.get("usage"):
        llm_cost_log["usage"] = overrides_meta["usage"]
    if notes_from_clamps:
        llm_cost_log["clamp_notes"] = notes_from_clamps
    if process_plan_summary:
        llm_cost_log["process_plan"] = copy.deepcopy(process_plan_summary)
    if applied_multipliers_log:
        llm_cost_log["applied_multipliers"] = applied_multipliers_log
    if applied_adders_log:
        llm_cost_log["applied_adders"] = applied_adders_log
    if isinstance(quote_state.effective_sources, dict) and quote_state.effective_sources:
        llm_cost_log["effective_sources"] = quote_state.effective_sources
    if isinstance(quote_state.user_overrides, dict) and quote_state.user_overrides:
        llm_cost_log["user_overrides"] = quote_state.user_overrides

    if pricing_source == "planner":
        process_costs = {
            "Machine": round(planner_machine_cost_total, 2),
            "Labor": round(planner_labor_cost_total, 2),
        }

    app_meta = {"llm_debug_enabled": bool(APP_ENV.llm_debug_enabled)}

    breakdown = {
        "qty": Qty,
        "setups": int(setups) if isinstance(setups, (int, float)) else setups,
        "family": family_for_breakdown,
        "scrap_pct": scrap_pct,
        "material_direct_cost": material_direct_cost,
        "total_direct_costs": round(total_direct_costs, 2),
        "material": material_detail_for_breakdown,
        "material_selected": dict(material_selection),
        "nre": {
            "programming_per_part": programming_per_part,
            "fixture_per_part": fixture_per_part,
            "extra_nre_cost": 0.0,
        },
        "nre_detail": nre_detail,
        "nre_costs": nre_costs_display,
        "nre_cost_details": nre_cost_details,
        "process_costs": process_costs,
        "process_meta": process_meta,
        "labor_costs": labor_costs_display,
        "labor_cost_details": labor_cost_details,
        "pass_through": pass_through,
        "direct_costs": direct_costs_display,
        "direct_cost_details": direct_cost_details,
        "speeds_feeds_path": speeds_feeds_path,
        "drill_debug": list(drill_debug_lines),
        "drilling_meta": drilling_meta,
        "pass_meta": pass_meta,
        "totals": {
            "labor_cost": labor_cost,
            "direct_costs": direct_costs,
            "subtotal": subtotal,
            "with_overhead": with_overhead,
            "with_ga": with_ga,
            "with_contingency": with_cont,
            "with_expedite": with_expedite,
            "price": price,
        },
        "applied_pcts": {
            "OverheadPct": OverheadPct,
            "GA_Pct": GA_Pct,
            "ContingencyPct": ContingencyPct,
            "ExpeditePct": ExpeditePct,
            "MarginPct": MarginPct,
            "InsurancePct": insurance_pct,
            "VendorMarkupPct": vendor_markup,
        },
        "llm_overrides": overrides,
        "llm_notes": llm_notes,
        "llm_cost_log": llm_cost_log,
    }

    breakdown["pricing_source"] = pricing_source
    if red_flag_messages:
        breakdown["red_flags"] = list(red_flag_messages)
    if planner_bucket_view is not None:
        breakdown["bucket_view"] = copy.deepcopy(planner_bucket_view)
    if planner_bucket_rollup is not None:
        breakdown["planner_bucket_rollup"] = copy.deepcopy(planner_bucket_rollup)

    if process_plan_summary:
        breakdown["process_plan"] = copy.deepcopy(process_plan_summary)
    if planner_process_minutes is not None:
        breakdown["process_minutes"] = round(float(planner_process_minutes), 1)
    if planner_pricing_result is not None:
        breakdown["process_plan_pricing"] = copy.deepcopy(planner_pricing_result)

    decision_state = {
        "baseline": copy.deepcopy(quote_state.baseline),
        "suggestions": copy.deepcopy(quote_state.suggestions),
        "user_overrides": copy.deepcopy(quote_state.user_overrides),
        "effective": copy.deepcopy(quote_state.effective),
        "effective_sources": copy.deepcopy(quote_state.effective_sources),
        "app": dict(app_meta),
    }
    breakdown["decision_state"] = decision_state

    if os.environ.get("DEBUG_INSPECTION_META") == "1":
        import pdb

        pdb.set_trace()

    narrative_text = get_why_text(
        quote_state,
        pricing_source=str(pricing_source),
        process_meta=process_meta,
        final_hours=process_hours_final,
    )
    breakdown["narrative"] = narrative_text

    if APP_ENV.llm_debug_enabled and overrides_meta and (
        overrides_meta.get("raw") is not None or overrides_meta.get("raw_text")
    ):
        try:
            files = sorted(APP_ENV.llm_debug_dir.glob("llm_snapshot_*.json"))
            if files:
                latest = files[-1]
                snap = json.loads(latest.read_text(encoding="utf-8"))
                snap["applied"] = {
                    "process_hour_multipliers": applied_multipliers_log,
                    "process_hour_adders": applied_adders_log,
                    "scrap_pct": scrap_pct,
                    "notes": llm_notes,

                    "clamped": notes_from_clamps,
                    "pass_through": {k: v for k, v in applied_pass.items()},
                }
                latest.write_text(jdump(snap), encoding="utf-8")
        except Exception:
            pass

    material_source_final = (
        quote_state.material_source
        or material_detail_for_breakdown.get("source")
        or (pass_meta.get("Material", {}) or {}).get("basis")
        or "shop defaults"
    )

    if isinstance(getattr(quote_state, "guard_context", None), dict) and red_flag_messages:
        ctx_flags = quote_state.guard_context.setdefault("red_flags", [])
        for msg in red_flag_messages:
            if msg not in ctx_flags:
                ctx_flags.append(msg)

    result_payload = {
        "price": price,
        "labor": labor_cost,
        "with_overhead": with_overhead,
        "breakdown": breakdown,
        "narrative": narrative_text,
        "decision_state": decision_state,
        "geo": copy.deepcopy(quote_state.geo),
        "ui_vars": copy.deepcopy(quote_state.ui_vars),
        "process_plan": copy.deepcopy(quote_state.process_plan),
        "material_source": material_source_final,
        "speeds_feeds_path": speeds_feeds_path,
        "speeds_feeds_loaded": speeds_feeds_loaded_flag,
        "drill_debug": list(drill_debug_lines),
        "drilling_meta": drilling_meta,
        "red_flags": list(red_flag_messages),
        "app": dict(app_meta),
    }

    breakdown["speeds_feeds_path"] = speeds_feeds_path
    breakdown["speeds_feeds_loaded"] = speeds_feeds_loaded_flag
    result_payload["speeds_feeds_path"] = speeds_feeds_path
    result_payload["speeds_feeds_loaded"] = speeds_feeds_loaded_flag

    return result_payload


# ---------- Tracing ----------
def trace_hours_from_df(df):
    out={}
    def grab(regex,label):
        items=df["Item"].astype(str)
        m=items.str.contains(regex, flags=re.I, regex=True, na=False)
        looks_time=items.str.contains(r"(hours?|hrs?|hr|time|min(ute)?s?)\b", flags=re.I, regex=True, na=False)
        mm=m & looks_time
        if mm.any():
            sub=df.loc[mm, ["Item","Example Values / Options"]].copy()
            sub["Example Values / Options"]=pd.to_numeric(sub["Example Values / Options"], errors="coerce").fillna(0.0)
            out[label]=[
                (str(r["Item"]), float(r["Example Values / Options"]))
                for _, r in sub.iterrows()
            ]
    grab(r"(Fixture\s*Design|Process\s*Sheet|Traveler|Documentation)","Engineering")
    grab(r"(CAM\s*Programming|CAM\s*Sim|Post\s*Processing)","CAM")
    grab(r"(Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)","Assembly")
    return out

# ---------- Redaction ----------
# ---------- Process router ----------
# ---------- Simple Tk editor ----------
def extract_2d_features_from_pdf_vector(pdf_path: str) -> dict:
    if not _HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) not installed. pip install pymupdf")

    import math
    assert fitz is not None
    fitz_mod = cast(Any, fitz)
    doc = fitz_mod.open(pdf_path)
    page_any = cast(Any, doc[0])
    text = str(page_any.get_text("text") or "").lower()
    drawings = page_any.get_drawings()

    # perimeter from vector segments (points are in PostScript points; 1 pt = 0.352777ï¿½ mm)
    pt_to_mm = 0.352777778
    per_pts = 0.0

    def _as_xy(obj):
        if hasattr(obj, "x") and hasattr(obj, "y"):
            return float(obj.x), float(obj.y)
        if isinstance(obj, (tuple, list)) and len(obj) >= 2:
            try:
                return float(obj[0]), float(obj[1])
            except Exception:
                return None
        return None

    def _add_polyline(points):
        nonlocal per_pts
        if len(points) < 2:
            return
        for a, b in zip(points, points[1:]):
            if a is None or b is None:
                continue
            per_pts += math.hypot(b[0] - a[0], b[1] - a[1])

    for d in drawings:
        for item in d["items"]:
            kind, *data = item
            if kind == "l":  # line
                p1 = _as_xy(data[0]) if len(data) >= 1 else None
                p2 = _as_xy(data[1]) if len(data) >= 2 else None
                if p1 and p2:
                    per_pts += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            elif kind in {"c", "b", "v", "y"}:  # bezier / curves
                pts = [_as_xy(part) for part in data]
                _add_polyline([pt for pt in pts if pt is not None])
            elif kind == "re" and data:
                rect = data[0]
                if isinstance(rect, (tuple, list)) and len(rect) >= 4:
                    w = float(rect[2])
                    h = float(rect[3])
                    per_pts += 2.0 * (abs(w) + abs(h))
            elif kind == "qu":  # quadratic curve (3 points)
                pts = [_as_xy(part) for part in data]
                _add_polyline([pt for pt in pts if pt is not None])

    # scrape thickness/material from text
    import re
    thickness_mm = None
    m = re.search(r"(thk|thickness)\s*[:=]?\s*([0-9.]+)\s*(mm|in|in\.|\")", text)
    if m:
        val = float(m.group(2))
        unit = m.group(3)
        thickness_mm = val * (25.4 if (unit.startswith("in") or unit == '"') else 1.0)

    material = None
    mm = re.search(r"(matl|material)\s*[:=]?\s*([a-z0-9 \-\+]+)", text)
    if mm:
        material = mm.group(2).strip()

    return {
        "kind": "2D", "source": "pdf",
        "profile_length_mm": round(per_pts * pt_to_mm, 2),
        "hole_diams_mm": [],
        "hole_count": 0,
        "thickness_mm": thickness_mm,
        "material": material,
    }
# ---------- PDF-driven variables + inference ----------
REQUIRED_COLS = ["Item", "Example Values / Options", "Data Type / Input Method"]

def default_variables_template() -> pd.DataFrame:
    if _HAS_PANDAS:
        core_df, _ = _load_master_variables()
        if core_df is not None:
            updated = core_df.copy()
            columns = list(updated.columns)
            for required in REQUIRED_COLS:
                if required not in columns:
                    columns.append(required)
            normalized_targets = {
                "fair required": "FAIR Required",
                "source inspection requirement": "Source Inspection Requirement",
            }
            seen_items: set[str] = set()
            adjusted_rows: list[dict[str, Any]] = []
            for _, row in updated.iterrows():
                row_dict: dict[str, Any] = dict(row)
                item_text = str(row_dict.get("Item", "") or "")
                normalized = item_text.strip().lower()
                seen_items.add(normalized)
                if normalized in normalized_targets:
                    row_dict["Data Type / Input Method"] = "Checkbox"
                    row_dict["Example Values / Options"] = "False / True"
                adjusted_rows.append(row_dict)
            for normalized, display in normalized_targets.items():
                if normalized not in seen_items:
                    new_row: dict[str, Any] = {col: "" for col in columns}
                    new_row.update(
                        {
                            "Item": display,
                            "Data Type / Input Method": "Checkbox",
                            "Example Values / Options": "False / True",
                        }
                    )
                    adjusted_rows.append(new_row)
            return pd.DataFrame(adjusted_rows, columns=columns)
    rows = [
        ("Overhead %", 0.15, "number"),
        ("G&A %", 0.08, "number"),
        ("Contingency %", 0.0, "number"),
        ("Profit Margin %", 0.0, "number"),
        ("Programmer $/hr", 120.0, "number"),
        ("CAM Programmer $/hr", 125.0, "number"),
        ("Milling $/hr", 100.0, "number"),
        ("Inspection $/hr", 95.0, "number"),
        ("Deburr $/hr", 60.0, "number"),
        ("Packaging $/hr", 55.0, "number"),
        ("Programming Hours", 0.0, "number"),
        ("CAM Programming Hours", 0.0, "number"),
        ("Engineering (Docs/Fixture Design) Hours", 0.0, "number"),
        ("Fixture Build Hours", 0.0, "number"),
        ("Roughing Cycle Time", 0.0, "number"),
        ("Semi-Finish Cycle Time", 0.0, "number"),
        ("Finishing Cycle Time", 0.0, "number"),
        ("In-Process Inspection Hours", 1.0, "number"),
        ("Final Inspection Hours", 0.0, "number"),
        ("CMM Programming Hours", 0.0, "number"),
        ("CMM Run Time min", 0.0, "number"),
        ("Deburr Hours", 0.0, "number"),
        ("Tumbling Hours", 0.0, "number"),
        ("Bead Blasting Hours", 0.0, "number"),
        ("Laser Mark Hours", 0.0, "number"),
        ("Masking Hours", 0.0, "number"),
        ("Sawing Hours", 0.0, "number"),
        ("Assembly Hours", 0.0, "number"),
        ("Packaging Labor Hours", 0.0, "number"),
        ("Number of Milling Setups", 1, "number"),
        ("Setup Hours / Setup", 0.3, "number"),
        ("FAIR Required", "False / True", "Checkbox"),
        ("Source Inspection Requirement", "False / True", "Checkbox"),
        ("Quantity", 1, "number"),
        ("Material", "", "text"),
        ("Thickness (in)", 0.0, "number"),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLS)

def coerce_or_make_vars_df(df: pd.DataFrame | None) -> pd.DataFrame:
    """Ensure the variables dataframe has the required columns with tolerant matching."""
    if df is None:
        return default_variables_template().copy()

    import re

    def _norm_col(s: str) -> str:
        s = str(s).replace("\u00A0", " ")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return re.sub(r"[^a-z0-9]", "", s)

    canon_map = {
        "item": "Item",
        "examplevaluesoptions": "Example Values / Options",
        "datatypeinputmethod": "Data Type / Input Method",
    }

    rename: dict[str, str] = {}
    for col in list(df.columns):
        key = _norm_col(col)
        if key in canon_map:
            rename[col] = canon_map[key]
    if rename:
        df = df.rename(columns=rename)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""

    return df

def extract_pdf_all(pdf_path: Path, dpi: int = 300) -> dict:
    if not _HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) not installed. pip install pymupdf")
    pdf_path = Path(pdf_path)
    assert fitz is not None
    fitz_mod = cast(Any, fitz)
    doc = fitz_mod.open(pdf_path)
    pages = []
    for idx, page in enumerate(doc):
        page_any = cast(Any, page)
        text = str(page_any.get_text("text") or "")
        blocks = page_any.get_text("blocks")
        tables = []
        try:
            found = page_any.find_tables()
            for table in getattr(found, "tables", []):
                tables.append(table.extract())
        except Exception:
            pass
        zoom = dpi / 72.0
        pix = page_any.get_pixmap(matrix=fitz_mod.Matrix(zoom, zoom), alpha=False)
        png_path = pdf_path.with_suffix(f".p{idx}.png")
        pix.save(png_path)
        pages.append({
            "index": idx,
            "text": text,
            "blocks": blocks,
            "tables": tables,
            "image_path": str(png_path),
            "char_count": len(text),
            "table_count": len(tables),
        })
    if pages:
        best = max(
            pages,
            key=lambda page: (
                (page.get("table_count", 0) or 0) > 0,
                page.get("char_count", 0) or 0,
            ),
        )
    else:
        best = {"text": "", "image_path": ""}
    doc.close()
    return {"pages": pages, "best": best}

JSON_SCHEMA = {
    "part_name": "str|null",
    "material": {"name": "str|null", "thickness_in": "float|null"},
    "quantity": "int|null",
    "estimates": {
        "Programming Hours": "float|null",
        "CAM Programming Hours": "float|null",
        "Engineering (Docs/Fixture Design) Hours": "float|null",
        "Fixture Build Hours": "float|null",
        "Roughing Cycle Time": "float|null",
        "Semi-Finish Cycle Time": "float|null",
        "Finishing Cycle Time": "float|null",
        "In-Process Inspection Hours": "float|null",
        "Final Inspection Hours": "float|null",
        "Deburr Hours": "float|null",
        "Sawing Hours": "float|null",
        "Assembly Hours": "float|null",
        "Packaging Labor Hours": "float|null"
    },
    "flags": {"FAIR Required": "0|1", "Source Inspection Requirement": "0|1"},
    "reasoning_brief": "str"
}

def _truncate_text(text: str, max_chars: int = 5000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return f"{head}\n...\n{tail}"

def build_llm_prompt(best_page: dict) -> dict:
    text = _truncate_text(best_page.get("text", ""))
    schema = jdump(JSON_SCHEMA)
    system = (
        "You are a manufacturing estimator. Read the drawing text and image and return JSON only. "
        "Estimate hours conservatively and do not invent dimensions."
    )
    user = (
        f"TEXT FROM PDF PAGE:\n{text}\n\n"
        f"REQUIRED JSON SHAPE:\n{schema}\n\n"
        "Return strictly JSON. If a field is unknown use null, and use 0/1 for flags."
    )
    return {"system": system, "user": user, "image_path": best_page.get("image_path", "")}

def run_llm_json(system: str, user: str, image_path: str) -> dict:
    """Adapter for the local VL model; override with real llama.cpp call."""
    # Placeholder: integrate with your Qwen VL llama.cpp wrapper and return parsed JSON.
    result_text = "{}"
    try:
        match = re.search(r"\{.*\}", result_text, flags=re.S)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}

def infer_pdf_estimate(structured: dict) -> dict:
    best_page = structured.get("best", {})
    prompt = build_llm_prompt(best_page)
    return run_llm_json(prompt["system"], prompt["user"], prompt["image_path"])

MAP_KEYS = {
    "Quantity": "quantity",
    "Material": ("material", "name"),
    "Thickness (in)": ("material", "thickness_in"),
    "Programming Hours": ("estimates", "Programming Hours"),
    "CAM Programming Hours": ("estimates", "CAM Programming Hours"),
    "Engineering (Docs/Fixture Design) Hours": ("estimates", "Engineering (Docs/Fixture Design) Hours"),
    "Fixture Build Hours": ("estimates", "Fixture Build Hours"),
    "Roughing Cycle Time": ("estimates", "Roughing Cycle Time"),
    "Semi-Finish Cycle Time": ("estimates", "Semi-Finish Cycle Time"),
    "Finishing Cycle Time": ("estimates", "Finishing Cycle Time"),
    "In-Process Inspection Hours": ("estimates", "In-Process Inspection Hours"),
    "Final Inspection Hours": ("estimates", "Final Inspection Hours"),
    "Deburr Hours": ("estimates", "Deburr Hours"),
    "Sawing Hours": ("estimates", "Sawing Hours"),
    "Assembly Hours": ("estimates", "Assembly Hours"),
    "Packaging Labor Hours": ("estimates", "Packaging Labor Hours"),
    "FAIR Required": ("flags", "FAIR Required"),
    "Source Inspection Requirement": ("flags", "Source Inspection Requirement"),
}

def _deep_get(d: dict, path):
    if isinstance(path, str):
        return d.get(path)
    cur = d
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

def merge_estimate_into_vars(vars_df: pd.DataFrame, estimate: dict) -> pd.DataFrame:
    for item, src in MAP_KEYS.items():
        value = _deep_get(estimate, src)
        if value is None:
            continue
        mask = vars_df["Item"].astype(str).str.fullmatch(re.escape(item), case=False, na=False)
        if not mask.any():
            dtype = "number" if isinstance(value, (int, float)) else "text"
            new_row = pd.DataFrame([{ "Item": item, "Example Values / Options": "", "Data Type / Input Method": dtype }])
            vars_df = pd.concat([vars_df, new_row], ignore_index=True)
            mask = vars_df["Item"].astype(str).str.fullmatch(re.escape(item), case=False, na=False)
        vars_df.loc[mask, "Example Values / Options"] = value
    return vars_df
# ---- 2D: DXF / DWG (ezdxf) ---------------------------------------------------
RE_TAP    = re.compile(r"(\(\d+\)\s*)?(#\s*\d{1,2}-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?)\s*TAP", re.I)
RE_NPT    = re.compile(r"(\d+\/\d+)\s*-\s*N\.?P\.?T\.?", re.I)
RE_THRU   = re.compile(r"\bTHRU\b", re.I)
RE_CBORE  = re.compile(r"C[’']?BORE|CBORE|COUNTERBORE", re.I)
RE_CSK    = re.compile(r"CSK|C'SINK|COUNTERSINK", re.I)
RE_DEPTH  = re.compile(r"(\d+(?:\.\d+)?)\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?", re.I)
RE_DIA    = re.compile(r"[Ø⌀\u00D8]?\s*(\d+(?:\.\d+)?)", re.I)
RE_THICK  = re.compile(r"(\d+(?:\.\d+)?)\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?", re.I)
RE_MAT    = re.compile(r"\b(MATL?|MATERIAL)\b\s*[:=\-]?\s*([A-Z0-9 \-\+/\.]+)", re.I)
RE_HARDNESS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[-–]\s*(\d+(?:\.\d+)?))?\s*HRC", re.I)
RE_HEAT_TREAT = re.compile(r"HEAT\s*TREAT(?:ED|\s+TO)?|\bQUENCH\b|\bTEMPER\b", re.I)
RE_COAT   = re.compile(
    r"\b(ANODIZE(?:\s*(?:CLR|BLACK|BLK))?|BLACK OXIDE|ZINC PLATE|NICKEL PLATE|PASSIVATE|CHEM FILM|IRIDITE|ALODINE|POWDER COAT|E-?COAT|PAINT)\b",
    re.I,
)
RE_TOL    = re.compile(r"\bUNLESS OTHERWISE SPECIFIED\b.*?([±\+\-]\s*\d+\.\d+)", re.I | re.S)
RE_FRONT_BACK = re.compile(r"FRONT\s*&\s*BACK|FRONT\s+AND\s+BACK|BOTH\s+SIDES|TWO\s+SIDES|2\s+SIDES|OPPOSITE\s+SIDE", re.I)
RE_FLIP_CALL  = re.compile(r"OP\.\s*\d+\s*FLIP|FLIP\s+PART", re.I)
RE_JIG_GRIND  = re.compile(r"\bJIG\s*GRIND\b", re.I)
RE_REAM       = re.compile(r"\bREAM(?:ING)?\b", re.I)
RE_TIGHT_TOL  = re.compile(r"±\s*0\.000[12]", re.I)
RE_FIT_CLASS  = re.compile(r"\bH\d\s*/\s*G\d\b", re.I)
RE_CSK_ANGLE  = re.compile(r"(60|82|90|100|110|120)\s*°", re.I)
RE_SLOT_NOTE  = re.compile(r"SLOT[^\n]*?(\d+(?:\.\d+)?)\s*(?:WIDE|WIDTH)?", re.I)
RE_HARDWARE_LINE = re.compile(r"\((\d+)\)\s*([A-Z0-9][A-Z0-9 \-/#\.]+)", re.I)

RE_COORD_HDR = re.compile(r"\bLIST\s+OF\s+COORDINATES\b", re.I)
RE_COORD_ROW = re.compile(r"([A-Z]\d+)\s+([\-+]?\d+(?:\.\d+)?)\s+([\-+]?\d+(?:\.\d+)?)")
RE_FINISH_STRONG = re.compile(r"\b(ANODIZE(?:\s+BLACK|\s+CLEAR)?|BLACK OXIDE|ZINC PLATE|NICKEL PLATE|PASSIVATE|PHOSPHATE|ECOAT|E-?COAT)\b", re.I)
RE_HEAT_TREAT_STRONG = re.compile(r"\b(A2|D2|O1|H13|4140|4340|S7|A36)\b.*?(HRC\s*\d{2}|\d{2}[-–]\d{2}\s*HRC)?", re.I)
RE_MAT_STRONG = re.compile(r"\b(MATERIAL|MAT)\b[:\s]*([A-Z0-9\-\s/\.]+)")
RE_TAP_TOKEN = re.compile(
    r"(#\d{1,2}-\d+|\d+/\d+-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?|\d+/\d+\s*-\s*NPT|N\.?P\.?T)",
    re.I,
)

LETTER_DRILL_IN = {
    "A": 0.234,
    "B": 0.238,
    "C": 0.242,
    "Q": 0.332,
    "R": 0.339,
    "S": 0.348,
    "T": 0.358,
}

SKIP_LAYERS = {"DEFPOINTS", "DIM", "AM_0", "CENTER", "TEXT", "NOTES", "TITLE", "BORDER"}

WIRE_GAGE_MAJOR_DIA_IN = {
    0: 0.06,
    1: 0.073,
    2: 0.086,
    3: 0.099,
    4: 0.112,
    5: 0.125,
    6: 0.138,
    7: 0.151,
    8: 0.164,
    9: 0.177,
    10: 0.19,
    11: 0.203,
    12: 0.216,
}

TAP_MINUTES_BY_CLASS = {
    "small": 0.22,
    "medium": 0.3,
    "large": 0.38,
    "xl": 0.45,
    "pipe": 0.55,
}

CBORE_MIN_PER_SIDE_MIN = 0.15
CSK_MIN_PER_SIDE_MIN = 0.12
NPT_INSPECTION_MIN_PER_HOLE = 2.5
JIG_GRIND_MIN_PER_FEATURE = 15.0
REAM_MIN_PER_FEATURE = 6.0
TIGHT_TOL_INSPECTION_MIN = 4.0
TIGHT_TOL_CMM_MIN = 6.0
HANDLING_ADDER_RANGE_HR = (0.1, 0.3)
LB_PER_IN3_PER_GCC = 0.036127292


def detect_units_scale(doc) -> dict[str, float | int]:
    try:
        u = int(doc.header.get("$INSUNITS", 1))
    except Exception:
        u = 1
    to_in = 1.0 if u == 1 else (1 / 25.4) if u in (4, 13) else 1.0
    return {"insunits": u, "to_in": float(to_in)}


def harvest_ordinates(doc, to_in: float) -> dict[str, Any]:
    vals_x: list[float] = []
    vals_y: list[float] = []
    spaces: list[Any] = []
    try:
        spaces.append(doc.modelspace())
    except Exception:
        pass
    try:
        for name in doc.layouts.names_in_taborder():
            if name.lower() in {"model", "defpoints"}:
                continue
            try:
                spaces.append(doc.layouts.get(name).entity_space)
            except Exception:
                continue
    except Exception:
        pass

    for space in spaces:
        try:
            dims = space.query("DIMENSION")
        except Exception:
            continue
        for dim in dims:
            try:
                dimtype = int(dim.dxf.dimtype)
            except Exception:
                continue
            try:
                is_ord = (dimtype & 6) == 6 or dimtype == 6
            except Exception:
                is_ord = False
            if not is_ord:
                continue
            try:
                m = float(dim.get_measurement()) * to_in
            except Exception:
                continue
            if not (0.05 <= m <= 1000):
                continue
            try:
                angle = float(getattr(dim.dxf, "text_rotation", 0.0)) % 180.0
            except Exception:
                angle = 0.0
            target = vals_y if 60.0 <= angle <= 120.0 else vals_x
            target.append(m)

    def pick_pair(vals: Sequence[float]) -> tuple[float | None, float | None]:
        if not vals:
            return (None, None)
        uniq = sorted({round(v, 3) for v in vals if v > 0.2})
        if not uniq:
            return (None, None)
        if len(uniq) == 1:
            return (uniq[0], None)
        return (uniq[-1], uniq[-2])

    L1, L2 = pick_pair(vals_x)
    W1, W2 = pick_pair(vals_y)
    return {
        "plate_len_in": round(W1, 3) if W1 else None,
        "plate_wid_in": round(L1, 3) if L1 else None,
        "candidates": {"x": vals_x, "y": vals_y, "x2": L2, "y2": W2},
        "provenance": "ORDINATE DIMENSIONS" if (W1 or L1) else None,
    }


def harvest_outline_bbox(doc, to_in: float) -> dict[str, Any]:
    biggest: tuple[float, Any] | None = None
    vertices: list[tuple[float, float]] = []
    for sp in _spaces(doc):
        try:
            polylines = list(sp.query("LWPOLYLINE"))
        except Exception:
            polylines = []
        for pl in polylines:
            if not getattr(pl, "closed", False):
                continue
            try:
                area = float(pl.area()) * (to_in ** 2)
            except Exception:
                continue
            if biggest is None or area > biggest[0]:
                biggest = (area, pl)
    if biggest:
        pl = biggest[1]
        try:
            get_points = getattr(pl, "get_points", None)
            raw_pts = list(get_points()) if callable(get_points) else []
            pts = [
                (float(x) * to_in, float(y) * to_in)
                for x, y, *_ in raw_pts
            ]
        except Exception:
            pts = []
        vertices.extend(pts)

    if not vertices:
        return {"plate_len_in": None, "plate_wid_in": None, "prov": "OUTLINE BBOX (none)"}

    xs, ys = zip(*vertices)
    L = max(ys) - min(ys)
    W = max(xs) - min(xs)
    prov = "OUTLINE AABB"

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    if dx and dy and max(dx, dy) / max(L, W) > 1.2:
        best_len = 0.0
        best_ang = 0.0
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            vx, vy = x2 - x1, y2 - y1
            el = (vx * vx + vy * vy) ** 0.5
            if el > best_len:
                best_len = el
                best_ang = math.atan2(vy, vx)
        ca = math.cos(-best_ang)
        sa = math.sin(-best_ang)
        rot = [(x * ca - y * sa, x * sa + y * ca) for x, y in vertices]
        rx, ry = zip(*rot)
        L = max(ry) - min(ry)
        W = max(rx) - min(rx)
        prov = "OUTLINE PCA AABB"

    return {"plate_len_in": round(L, 3), "plate_wid_in": round(W, 3), "prov": prov}


def harvest_coordinate_table(doc, to_in: float) -> dict[str, Any]:
    lines: list[str] = []
    for sp in _spaces(doc):
        try:
            entities = sp.query("MTEXT,TEXT")
        except Exception:
            entities = []
        for e in entities:
            try:
                if e.dxftype() == "MTEXT":
                    text = e.plain_text()
                else:
                    text = e.dxf.text
            except Exception:
                continue
            if not text:
                continue
            for frag in str(text).splitlines():
                frag = frag.strip()
                if frag:
                    lines.append(frag)

    start_idx = next((i for i, t in enumerate(lines) if RE_COORD_HDR.search(t)), None)
    if start_idx is None:
        return {
            "coord_count": None,
            "coord_ids": [],
            "x_span_in": None,
            "y_span_in": None,
            "prov": "COORD LIST (none)",
        }

    window = lines[start_idx : start_idx + 2000]
    xs: list[float] = []
    ys: list[float] = []
    ids: set[str] = set()
    for row in window:
        match = RE_COORD_ROW.search(row)
        if not match:
            continue
        ids.add(match.group(1))
        try:
            xs.append(float(match.group(2)) * to_in)
            ys.append(float(match.group(3)) * to_in)
        except Exception:
            continue

    if not ids:
        return {"coord_count": 0, "coord_ids": [], "prov": "COORD LIST (empty)"}

    return {
        "coord_count": len(ids),
        "coord_ids": sorted(ids),
        "x_span_in": round(max(xs) - min(xs), 3) if xs else None,
        "y_span_in": round(max(ys) - min(ys), 3) if ys else None,
        "prov": "COORD LIST",
    }


def _circle_candidates(doc, to_in: float, plate_bbox: tuple[float, float, float, float] | None = None) -> list[tuple[float, float, float, float]]:
    """Collect circle data, preferring model space to avoid duplicate layout views."""

    if doc is None:
        return []

    try:
        modelspace = doc.modelspace()
        modelspace_id = id(modelspace)
    except Exception:
        modelspace = None
        modelspace_id = None

    per_space: list[tuple[bool, list[tuple[float, float, float, float]]]] = []
    for sp in _spaces(doc):
        is_model = modelspace_id is not None and id(sp) == modelspace_id
        local: list[tuple[float, float, float, float]] = []
        try:
            entities = sp.query("CIRCLE")
        except Exception:
            entities = []
        for c in entities:
            layer = (getattr(getattr(c, "dxf", object()), "layer", "") or "").upper()
            if layer in SKIP_LAYERS:
                continue
            try:
                radius_in = float(c.dxf.radius) * float(to_in)
            except Exception:
                continue
            diameter_in = round(2.0 * radius_in, 4)
            if not (0.06 <= diameter_in <= 2.5):
                continue
            try:
                x, y = c.dxf.center[:2]
            except Exception:
                continue
            if plate_bbox:
                xmin, ymin, xmax, ymax = plate_bbox
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    continue
            local.append((x, y, diameter_in, radius_in))
        if local:
            per_space.append((is_model, local))

    if not per_space:
        return []

    if any(is_model and entries for is_model, entries in per_space):
        circles: list[tuple[float, float, float, float]] = []
        for is_model, entries in per_space:
            if is_model:
                circles.extend(entries)
        return circles

    circles: list[tuple[float, float, float, float]] = []
    for _, entries in per_space:
        circles.extend(entries)
    return circles


def filtered_circles(doc, to_in: float, plate_bbox: tuple[float, float, float, float] | None = None) -> list[tuple[float, float, float, int]]:
    circles_raw = _circle_candidates(doc, to_in, plate_bbox=plate_bbox)

    clustered: list[tuple[float, float, float, int]] = []
    tol = 0.01
    for x, y, d, _ in circles_raw:
        for idx, (cx, cy, cd, count) in enumerate(clustered):
            if abs(cx - x) < tol and abs(cy - y) < tol and abs(cd - d) < 1e-3:
                clustered[idx] = (cx, cy, cd, count + 1)
                break
        else:
            clustered.append((x, y, d, 1))
    return clustered


def classify_concentric(doc, to_in: float, tol_center: float = 0.005, min_gap_in: float = 0.03) -> dict[str, Any]:
    pts: list[tuple[float, float, float]] = []
    for x, y, _, radius_in in _circle_candidates(doc, to_in):
        dia = 2.0 * radius_in
        if not (0.04 <= dia <= 6.0):
            continue
        pts.append((x, y, radius_in))
    pts.sort()
    cbore_like = 0
    csk_like = 0
    for i, (x1, y1, r1) in enumerate(pts):
        for j in range(i + 1, min(i + 15, len(pts))):
            x2, y2, r2 = pts[j]
            if abs(x1 - x2) < tol_center and abs(y1 - y2) < tol_center:
                gap = abs(r2 - r1)
                if gap >= min_gap_in:
                    if gap < 0.15:
                        cbore_like += 1
                    else:
                        csk_like += 1
    return {"cbore_pairs_geom": cbore_like, "csk_pairs_geom": csk_like, "prov": "CONCENTRIC CIRCLES"}


def tap_classes_from_row_text(u: str, qty: int) -> dict[str, int]:
    cls = {"small": 0, "medium": 0, "large": 0, "npt": 0}
    if qty <= 0:
        return cls
    text = (u or "")
    if not text:
        return cls
    upper = text.upper()
    for m in RE_TAP_TOKEN.finditer(upper):
        tok = m.group(1).upper().replace(" ", "").replace(".", "") if m.group(1) else ""
        if not tok:
            continue
        if "NPT" in tok:
            cls["npt"] += qty
            continue
        dia_in = None
        if tok.startswith("#"):
            nums = re.findall(r"\d+", tok)
            if nums:
                num = int(nums[0])
                dia_in = {6: 0.138, 8: 0.164, 10: 0.19, 12: 0.216}.get(num, 0.19)
        elif tok.startswith("M"):
            nums = re.findall(r"\d+(?:\.\d+)?", tok)
            if nums:
                dia_in = float(nums[0]) / 25.4
        elif "/" in tok:
            try:
                num, den = map(int, tok.split("-", 1)[0].split("/"))
                dia_in = num / den
            except Exception:
                dia_in = None
        else:
            frac = re.findall(r"([0-9.]+)", tok)
            if frac:
                try:
                    dia_in = float(frac[0])
                except Exception:
                    dia_in = None
        if dia_in is None:
            continue
        if dia_in < 0.250:
            cls["small"] += qty
        elif dia_in < 0.500:
            cls["medium"] += qty
        else:
            cls["large"] += qty
        break
    return cls


def tap_classes_from_lines(lines: Sequence[str] | None) -> dict[str, int]:
    classes = {"small": 0, "medium": 0, "large": 0, "npt": 0}
    if not lines:
        return classes
    for raw in lines:
        if not raw:
            continue
        text = str(raw)
        qty = 1
        try:
            upper_text = text.upper()
            m_qty = re.search(r"\bQTY\b[:\s]*(\d+)", upper_text)
            if m_qty:
                qty = int(m_qty.group(1))
        except Exception:
            qty = 1
        row_cls = tap_classes_from_row_text(text, qty)
        for key, val in row_cls.items():
            if val:
                classes[key] = classes.get(key, 0) + int(val)
    return classes


def harvest_gdt(doc) -> dict[str, Any]:
    texts: list[str] = []
    for sp in _spaces(doc):
        try:
            entities = sp.query("MTEXT,TEXT")
        except Exception:
            entities = []
        for e in entities:
            try:
                if e.dxftype() == "MTEXT":
                    txt = e.plain_text()
                else:
                    txt = e.dxf.text
            except Exception:
                continue
            if txt:
                texts.append(txt)
    upper = "\n".join(texts).upper()
    return {
        "gdt": {
            "true_position": len(re.findall(r"\bTRUE\s*POSITION\b|⌀\s*POS", upper)),
            "flatness": len(re.findall(r"\bFLATNESS\b", upper)),
            "parallelism": len(re.findall(r"\bPARALLELISM\b", upper)),
            "profile": len(re.findall(r"\bPROFILE\b", upper)),
        },
        "prov": "TEXT GD&T",
    }


def harvest_finish_ht_material(all_text_upper: str) -> dict[str, Any]:
    finishes = sorted({m.group(1).upper() for m in RE_FINISH_STRONG.finditer(all_text_upper)})
    heat_treat = None
    m_ht = RE_HEAT_TREAT_STRONG.search(all_text_upper)
    if m_ht:
        heat_treat = m_ht.group(1).upper()
        if m_ht.group(2):
            heat_treat = f"{heat_treat} {m_ht.group(2).upper()}".strip()
    material_note = None
    m_mat = RE_MAT_STRONG.search(all_text_upper)
    if m_mat:
        material_note = m_mat.group(2).strip().upper()
    return {"finishes": finishes, "heat_treat": heat_treat, "material_note": material_note}


def harvest_hardware_notes(all_text_upper: str) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for match in re.finditer(r"\((\d+)\)\s*(DOWEL|SHCS|FHCS|BHCS|NUT|WASHER|PIN)\b.*?(\d+[-/]\d+|#\d+|\d+-\d+)", all_text_upper, re.I):
        qty = int(match.group(1))
        hardware_type = match.group(2).upper()
        size = match.group(3)
        items.append({"qty": qty, "type": hardware_type, "size": size})
    return {"hardware_items": items, "prov": "NOTES HW"}


def quick_deburr_estimates(edge_len_in: float | None, hole_count: int | None) -> dict[str, Any]:
    deburr_ipm_edge = 1000.0
    sec_per_hole = 5.0
    edge_hours = (edge_len_in or 0.0) / deburr_ipm_edge
    hole_hours = ((hole_count or 0) * sec_per_hole) / 3600.0
    return {
        "deburr_edge_hr_suggest": round(edge_hours, 3) if edge_hours else 0.0,
        "deburr_hole_hr_suggest": round(hole_hours, 3) if hole_hours else 0.0,
        "prov": "GEOM (edge & holes)",
    }


def _spaces(doc) -> list[Any]:
    spaces: list[Any] = []
    if doc is None:
        return spaces
    seen: set[int] = set()
    try:
        msp = doc.modelspace()
    except Exception:
        msp = None
    if msp is not None and id(msp) not in seen:
        seen.add(id(msp))
        spaces.append(msp)
    try:
        layout_names = list(doc.layouts.names_in_taborder())
    except Exception:
        layout_names = []
    for layout_name in layout_names:
        if layout_name.lower() in ("model", "defpoints"):
            continue
        try:
            es = doc.layouts.get(layout_name).entity_space
        except Exception:
            continue
        if es is None or id(es) in seen:
            continue
        seen.add(id(es))
        spaces.append(es)
    return spaces


def _iter_table_text(doc):
    if doc is None:
        return
    for sp in _spaces(doc):
        try:
            for t in sp.query("TABLE"):
                try:
                    for r in range(t.dxf.n_rows):
                        row = []
                        for c in range(t.dxf.n_cols):
                            try:
                                cell = t.get_cell(r, c)
                            except Exception:
                                cell = None
                            row.append(cell.get_text() if cell else "")
                        line = " | ".join(s.strip() for s in row if s)
                        if line:
                            yield line
                except Exception:
                    continue
        except Exception:
            pass
        try:
            for e in sp.query("MTEXT,TEXT"):
                try:
                    if e.dxftype() == "MTEXT":
                        text = e.plain_text()
                    else:
                        text = e.dxf.text
                except Exception:
                    text = None
                if text:
                    yield text.strip()
        except Exception:
            continue


def hole_count_from_acad_table(doc) -> dict[str, Any]:
    """Extract hole and tap data from an AutoCAD TABLE entity."""

    result: dict[str, Any] = {}
    if doc is None:
        return result

    for sp in _spaces(doc):
        try:
            tables = sp.query("TABLE")
        except Exception:
            tables = []
        for t in tables:
            try:
                hdr = [
                    " ".join((t.get_cell(0, c).get_text() or "").split()).upper()
                    if t.get_cell(0, c)
                    else ""
                    for c in range(t.dxf.n_cols)
                ]
            except Exception:
                continue
            if "HOLE" not in " ".join(hdr):
                continue

            def find_col(name: str) -> int | None:
                for i, txt in enumerate(hdr):
                    if name in txt:
                        return i
                return None

            c_qty = find_col("QTY")
            c_desc = find_col("DESCRIPTION")
            c_ref = find_col("REF")

            total = 0
            families: dict[float, int] = {}
            row_taps = 0
            tap_classes = {"small": 0, "medium": 0, "large": 0, "npt": 0}

            for r in range(1, t.dxf.n_rows):
                cell = t.get_cell(r, c_qty) if c_qty is not None else None
                try:
                    qty_text = (cell.get_text() if cell else "0") or "0"
                except Exception:
                    qty_text = "0"
                try:
                    qty = int(float(qty_text.strip() or "0"))
                except Exception:
                    digits = re.findall(r"\d+", qty_text)
                    qty = int(digits[0]) if digits else 0
                if qty <= 0:
                    continue
                total += qty

                ref_txt = ""
                desc = ""
                if c_ref is not None:
                    ref_cell = t.get_cell(r, c_ref)
                    try:
                        ref_txt = (ref_cell.get_text() if ref_cell else "") or ""
                    except Exception:
                        ref_txt = ""
                if c_desc is not None:
                    desc_cell = t.get_cell(r, c_desc)
                    try:
                        desc = (desc_cell.get_text() if desc_cell else "") or ""
                    except Exception:
                        desc = ""
                u = (ref_txt + " " + desc).upper()
                m = re.search(r"[Ø⌀]\s*(\d+(?:\.\d+)?)", u)
                if m:
                    try:
                        d = round(float(m.group(1)), 4)
                        families[d] = families.get(d, 0) + qty
                    except Exception:
                        pass

                tap_cls = tap_classes_from_row_text(u, qty)
                tap_sum = sum(tap_cls.values())
                if tap_sum:
                    for key, val in tap_cls.items():
                        if val:
                            tap_classes[key] = tap_classes.get(key, 0) + int(val)
                    row_taps += tap_sum
                elif "TAP" in u or "N.P.T" in u or "NPT" in u:
                    row_taps += qty

            if total > 0:
                filtered_classes = {k: int(v) for k, v in tap_classes.items() if v}
                result = {
                    "hole_count": total,
                    "hole_diam_families_in": families,
                    "tap_qty_from_table": row_taps,
                    "tap_class_counts": filtered_classes,
                    "provenance_holes": "HOLE TABLE (ACAD_TABLE)",
                }
                return result

    return result


def hole_count_from_text_table(doc, lines: Sequence[str] | None = None) -> tuple[int, dict] | tuple[None, None]:
    if lines is None:
        source_lines = _iter_table_text(doc) or []
    else:
        source_lines = lines
    cleaned = [str(raw).strip() for raw in source_lines if raw]
    if not cleaned:
        return None, None

    idx = None
    for i, s in enumerate(cleaned):
        if "HOLE TABLE" in s.upper():
            idx = i
            break
    if idx is None:
        return None, None

    total = 0
    fam: dict[float, int] = {}
    for s in cleaned[idx + 1 :]:
        u = s.upper()
        if not u or "DESCRIPTION" in u:
            continue
        if "LIST OF COORDINATES" in u or "SEE SHEET" in u:
            break

        mqty = re.search(r"\bQTY\b[:\s]*([0-9]+)", u)
        if not mqty:
            cols = [c.strip() for c in u.split("|")]
            for c in cols:
                if c.isdigit():
                    mqty = re.match(r"(\d+)$", c)
                    if mqty:
                        break
        if mqty:
            q = int(mqty.group(1))
            total += q
        else:
            q = None

        mref = re.search(r"\bREF\s*[Ø⌀]?\s*(\d+(?:\.\d+)?)", u)
        if not mref:
            mref = re.search(r"[Ø⌀]\s*(\d+(?:\.\d+)?)", u)
        if mref and mqty:
            d = round(float(mref.group(1)), 4)
            fam[d] = fam.get(d, 0) + int(mqty.group(1))

    return (total, fam) if total else (None, None)


def hole_count_from_geometry(doc, to_in, plate_bbox=None) -> tuple[int, dict]:
    clustered = filtered_circles(doc, to_in, plate_bbox=plate_bbox)
    fam: dict[float, int] = {}
    for _, _, d, _ in clustered:
        fam[d] = fam.get(d, 0) + 1
    return len(clustered), fam


def _major_diameter_from_thread(spec: str) -> float | None:
    if not spec:
        return None
    spec_clean = spec.strip().upper().replace(" ", "")
    if "NPT" in spec_clean:
        # treat nominal pipe size as fractional portion before NPT
        parts = spec_clean.split("NPT", 1)[0]
        parts = parts.rstrip("-")
        if not parts:
            return None
        try:
            return float(Fraction(parts))
        except Exception:
            pass
        try:
            return float(parts)
        except Exception:
            return None
    if spec_clean.startswith("#"):
        try:
            gauge = int(re.sub(r"[^0-9]", "", spec_clean.split("-", 1)[0]))
        except Exception:
            return None
        return WIRE_GAGE_MAJOR_DIA_IN.get(gauge)
    if spec_clean.startswith("M"):
        try:
            mm_val = float(re.findall(r"M(\d+(?:\.\d+)?)", spec_clean)[0])
        except Exception:
            return None
        return mm_val / 25.4
    # fractional or decimal imperial (e.g., 5/16-18 or 0.375-16)
    lead = spec_clean.split("-", 1)[0]
    try:
        if "/" in lead:
            return float(Fraction(lead))
        return float(lead)
    except Exception:
        return None


def _classify_thread_spec(spec: str) -> tuple[str, float, bool]:
    major = _major_diameter_from_thread(spec)
    if spec and "NPT" in spec.upper():
        return "pipe", TAP_MINUTES_BY_CLASS["pipe"], True
    if major is None:
        return "unknown", TAP_MINUTES_BY_CLASS["medium"], False
    if major <= 0.2:
        return "small", TAP_MINUTES_BY_CLASS["small"], False
    if major <= 0.3125:
        return "medium", TAP_MINUTES_BY_CLASS["medium"], False
    if major <= 0.5:
        return "large", TAP_MINUTES_BY_CLASS["large"], False
    return "xl", TAP_MINUTES_BY_CLASS["xl"], False


def _normalize_hole_text(text: str | None) -> str:
    """Return a normalized representation of a hole note.

    The goal is to aggressively collapse whitespace and case differences so
    that duplicated callouts from leaders and table rows can be identified even
    if their formatting differs slightly.
    """

    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip().upper()
    # Normalize the most common diameter symbols so that a leader that uses
    # "Ø" matches a table row that omits it or substitutes "⌀".
    cleaned = cleaned.replace("Ø", "").replace("⌀", "")
    return cleaned


def _dedupe_hole_entries(
    existing_entries: Iterable[dict[str, Any]],
    new_entries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter ``new_entries`` to those that are not duplicates of ``existing_entries``.

    Duplicate detection is based on the normalized ``raw`` text attached to each
    entry.  This keeps leader annotations that simply restate a table row from
    inflating the hole counts while still allowing genuinely new callouts to be
    considered.
    """

    seen: set[str] = {
        _normalize_hole_text(entry.get("raw"))
        for entry in existing_entries
        if isinstance(entry, dict)
    }
    unique_entries: list[dict[str, Any]] = []
    for entry in new_entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = _normalize_hole_text(entry.get("raw"))
        if fingerprint and fingerprint in seen:
            continue
        if fingerprint:
            seen.add(fingerprint)
        unique_entries.append(entry)
    return unique_entries


def _parse_hole_line(line: str, to_in: float, *, source: str | None = None) -> dict[str, Any] | None:
    if not line:
        return None
    U = line.upper()
    if not any(k in U for k in ("HOLE", "TAP", "THRU", "CBORE", "C'BORE", "DRILL")):
        return None

    entry: dict[str, Any] = {
        "qty": None,
        "tap": None,
        "tap_class": None,
        "tap_minutes_per": None,
        "tap_is_npt": False,
        "thru": False,
        "cbore": False,
        "csk": False,
        "ref_dia_in": None,
        "depth_in": None,
        "side": None,
        "double_sided": False,
        "raw": line,
    }
    if source:
        entry["source"] = source

    m_qty = re.search(r"\bQTY\b[:\s]*(\d+)", U)
    if m_qty:
        entry["qty"] = int(m_qty.group(1))

    mt = RE_TAP.search(U)
    if mt:
        thread_spec = None
        if mt.lastindex and mt.lastindex >= 2:
            thread_spec = mt.group(2)
        else:
            thread_spec = mt.group(1)
        if thread_spec:
            cleaned = thread_spec.replace(" ", "")
        else:
            cleaned = None
        entry["tap"] = cleaned
        if cleaned:
            cls, minutes_per, is_npt = _classify_thread_spec(cleaned)
            entry["tap_class"] = cls
            entry["tap_minutes_per"] = minutes_per
            entry["tap_is_npt"] = is_npt
    else:
        m_npt = RE_NPT.search(U)
        if m_npt:
            cleaned = m_npt.group(0).replace(" ", "")
            entry["tap"] = cleaned
            cls, minutes_per, is_npt = _classify_thread_spec(cleaned)
            entry["tap_class"] = cls
            entry["tap_minutes_per"] = minutes_per
            entry["tap_is_npt"] = is_npt
        else:
            entry["tap"] = None
    entry["thru"] = bool(RE_THRU.search(U))
    entry["cbore"] = bool(RE_CBORE.search(U))
    entry["csk"] = bool(RE_CSK.search(U))
    if RE_FRONT_BACK.search(U):
        entry["double_sided"] = True

    md = RE_DEPTH.search(U)
    if md:
        try:
            depth = float(md.group(1)) * float(to_in)
        except Exception:
            depth = None
        side = (md.group(2) or "").upper() or None
        if depth is not None:
            entry["depth_in"] = depth
        if side:
            entry["side"] = side

    mref = re.search(r"REF\s*[Ø⌀]\s*(\d+(?:\.\d+)?)", U)
    if mref:
        try:
            entry["ref_dia_in"] = float(mref.group(1)) * float(to_in)
        except Exception:
            entry["ref_dia_in"] = None

    if entry.get("ref_dia_in") is None:
        mdia = RE_DIA.search(U)
        if mdia and ("Ø" in U or "⌀" in U or " REF" in U):
            try:
                entry["ref_dia_in"] = float(mdia.group(1)) * float(to_in)
            except Exception:
                entry["ref_dia_in"] = None

    return entry


def _aggregate_hole_entries(entries: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    hole_count = 0
    tap_qty = 0
    cbore_qty = 0
    csk_qty = 0
    max_depth_in = 0.0
    back_ops = False
    tap_details: dict[str, dict[str, Any]] = {}
    tap_minutes_total = 0.0
    tap_class_counter: Counter[str] = Counter()
    npt_qty = 0
    double_cbore = False
    double_csk = False
    cbore_minutes_total = 0.0
    csk_minutes_total = 0.0
    if entries:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            qty_val = entry.get("qty")
            if qty_val is None:
                qty = 0
            else:
                try:
                    qty = int(qty_val)
                except Exception:
                    try:
                        qty = int(round(float(qty_val)))
                    except Exception:
                        qty = 0
            qty = qty if qty > 0 else 1
            hole_count += qty
            if entry.get("tap"):
                tap_qty += qty
                spec = str(entry.get("tap") or "").strip()
                cls = entry.get("tap_class") or "unknown"
                minutes_per = entry.get("tap_minutes_per") or TAP_MINUTES_BY_CLASS.get("medium", 0.3)
                tap_minutes_total += qty * float(minutes_per)
                tap_class_counter[str(cls)] += qty
                detail = tap_details.setdefault(spec, {
                    "spec": spec,
                    "qty": 0,
                    "class": cls,
                    "minutes_per_hole": float(minutes_per),
                    "total_minutes": 0.0,
                    "is_npt": bool(entry.get("tap_is_npt")),
                })
                detail["qty"] += qty
                detail["total_minutes"] += qty * float(minutes_per)
                if entry.get("tap_is_npt"):
                    npt_qty += qty
            if entry.get("cbore"):
                ops_qty = qty * (2 if entry.get("double_sided") else 1)
                cbore_qty += ops_qty
                cbore_minutes_total += ops_qty * CBORE_MIN_PER_SIDE_MIN
                if entry.get("double_sided"):
                    double_cbore = True
            if entry.get("csk"):
                ops_qty = qty * (2 if entry.get("double_sided") else 1)
                csk_qty += ops_qty
                csk_minutes_total += ops_qty * CSK_MIN_PER_SIDE_MIN
                if entry.get("double_sided"):
                    double_csk = True
            depth = entry.get("depth_in")
            try:
                if depth and float(depth) > max_depth_in:
                    max_depth_in = float(depth)
            except Exception:
                continue
            side = str(entry.get("side") or "").upper()
            if side == "BACK" or (entry.get("raw") and "BACK" in str(entry.get("raw")).upper()) or entry.get("double_sided") and (entry.get("cbore") or entry.get("csk")):
                back_ops = True
    tap_details_list = []
    for spec, detail in tap_details.items():
        detail["qty"] = int(detail.get("qty", 0) or 0)
        detail["total_minutes"] = round(float(detail.get("total_minutes", 0.0)), 3)
        detail["minutes_per_hole"] = round(float(detail.get("minutes_per_hole", 0.0)), 3)
        tap_details_list.append(detail)
    tap_details_list.sort(key=lambda d: -d.get("qty", 0))
    tap_class_counts = {cls: int(qty) for cls, qty in tap_class_counter.items() if qty}
    return {
        "hole_count": hole_count if hole_count else None,
        "tap_qty": tap_qty,
        "cbore_qty": cbore_qty,
        "csk_qty": csk_qty,
        "deepest_hole_in": max_depth_in if max_depth_in > 0 else None,
        "provenance": "HOLE TABLE / NOTES" if hole_count else None,
        "from_back": back_ops,
        "tap_details": tap_details_list,
        "tap_minutes_hint": round(tap_minutes_total, 3) if tap_minutes_total else None,
        "tap_class_counts": tap_class_counts,
        "npt_qty": npt_qty,
        "cbore_minutes_hint": round(cbore_minutes_total, 3) if cbore_minutes_total else None,
        "csk_minutes_hint": round(csk_minutes_total, 3) if csk_minutes_total else None,
        "double_sided_cbore": double_cbore,
        "double_sided_csk": double_csk,
    }


def summarize_hole_chart_lines(lines: Iterable[str] | None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for raw in lines or []:
        text = str(raw or "")
        if not text.strip():
            continue
        entry = _parse_hole_line(text, 1.0, source="CHART")
        if not entry:
            continue
        if not entry.get("qty"):
            mqty = re.search(r"\((\d+)\)", text)
            if mqty:
                try:
                    entry["qty"] = int(mqty.group(1))
                except Exception:
                    pass
        entries.append(entry)
    agg = _aggregate_hole_entries(entries)
    return {
        "tap_qty": int(agg.get("tap_qty") or 0),
        "cbore_qty": int(agg.get("cbore_qty") or 0),
        "csk_qty": int(agg.get("csk_qty") or 0),
        "deepest_hole_in": agg.get("deepest_hole_in"),
        "from_back": bool(agg.get("from_back")),
        "tap_details": agg.get("tap_details") or [],
        "tap_minutes_hint": agg.get("tap_minutes_hint"),
        "tap_class_counts": agg.get("tap_class_counts") or {},
        "npt_qty": int(agg.get("npt_qty") or 0),
        "cbore_minutes_hint": agg.get("cbore_minutes_hint"),
        "csk_minutes_hint": agg.get("csk_minutes_hint"),
        "double_sided_cbore": bool(agg.get("double_sided_cbore")),
        "double_sided_csk": bool(agg.get("double_sided_csk")),
    }


def derive_inference_knobs(
    tokens_text: str,
    combined_agg: dict[str, Any],
    *,
    hole_families: dict | None = None,
    geom_families: dict | None = None,
    material_info: dict | None = None,
    thickness_guess: float | None = None,
    thickness_provenance: str | None = None,
    pocket_metrics: dict | None = None,
    stock_plan: dict | None = None,
    table_hole_count: int | None = None,
    geometry_hole_count: int | None = None,
) -> dict[str, Any]:
    tokens_upper = tokens_text.upper() if isinstance(tokens_text, str) else ""
    token_lines = [ln.strip() for ln in (tokens_text.splitlines() if tokens_text else []) if ln.strip()]
    knobs: dict[str, Any] = {}

    # --- Setups / handling ---------------------------------------------------
    setup_signals: list[str] = []
    if combined_agg.get("from_back"):
        setup_signals.append("Hole table flags BACK operations")
    if RE_FRONT_BACK.search(tokens_upper):
        setup_signals.append("Text includes FRONT & BACK")
    if re.search(r"FROM\s+BACK", tokens_upper):
        setup_signals.append("Note: FROM BACK callout")
    if "BACK SIDE" in tokens_upper:
        setup_signals.append("Note references BACK SIDE")
    if RE_FLIP_CALL.search(tokens_upper):
        setup_signals.append("Operation flip callout present")
    if setup_signals:
        signal_count = len(setup_signals)
        handling = HANDLING_ADDER_RANGE_HR[0] + max(0, signal_count - 1) * 0.05
        handling = min(HANDLING_ADDER_RANGE_HR[1], handling)
        knobs["setups"] = {
            "confidence": "high",
            "signals": setup_signals,
            "recommended": {
                "setups_min": 2,
                "fixture_plan": "pins + toe clamps",
                "handling_adder_hr": round(handling, 3),
                "handling_bounds_hr": [round(HANDLING_ADDER_RANGE_HR[0], 3), round(HANDLING_ADDER_RANGE_HR[1], 3)],
            },
            "targets": [
                "setups",
                "fixture",
                "process_hour_adders.handling",
            ],
        }

    # --- Fixture build hours -------------------------------------------------
    fixture_signals: list[str] = []
    fixture_components: dict[str, float] = {}
    fixture_hours = 0.0

    hole_count_hint = max(
        int(combined_agg.get("hole_count") or 0),
        int(table_hole_count or 0),
        int(geometry_hole_count or 0),
    )
    if hole_count_hint:
        fixture_signals.append(f"Hole count ≈ {hole_count_hint}")
        hole_component = max(0.0, min(8.0, hole_count_hint / 16.0))
        if hole_component > 0:
            fixture_components["hole_population_hr"] = round(hole_component, 3)
            fixture_hours += hole_component

    if combined_agg.get("from_back"):
        fixture_signals.append("Back-side ops flagged")
        back_component = 1.0
        fixture_components["second_op_setup_hr"] = round(back_component, 3)
        fixture_hours += back_component

    thickness_in = thickness_guess
    if thickness_in is None and stock_plan:
        thickness_in = stock_plan.get("stock_thk_in")
    if isinstance(thickness_in, (int, float)) and thickness_in > 0:
        fixture_signals.append(f"Thickness ≈ {float(thickness_in):.2f} in")
        if thickness_in >= 2.0:
            thick_component = 1.0
        elif thickness_in >= 1.25:
            thick_component = 0.6
        else:
            thick_component = 0.0
        if thick_component > 0:
            fixture_components["thickness_penalty_hr"] = round(thick_component, 3)
            fixture_hours += thick_component

    stock_len = float(stock_plan.get("stock_len_in") or 0.0) if stock_plan else 0.0
    stock_wid = float(stock_plan.get("stock_wid_in") or 0.0) if stock_plan else 0.0
    if stock_len and stock_wid:
        fixture_signals.append(f"Stock blank ≈ {stock_len:.1f}×{stock_wid:.1f} in")
        blank_area = stock_len * stock_wid
        if blank_area >= 400:
            area_component = 1.2
        elif blank_area >= 225:
            area_component = 0.6
        else:
            area_component = 0.0
        if area_component > 0:
            fixture_components["blank_envelope_hr"] = round(area_component, 3)
            fixture_hours += area_component

    part_mass_lb = float(stock_plan.get("part_mass_lb") or 0.0) if stock_plan else 0.0
    if part_mass_lb:
        fixture_signals.append(f"Part mass ≈ {part_mass_lb:.1f} lb")
        if part_mass_lb >= 45:
            mass_component = 1.5
        elif part_mass_lb >= 25:
            mass_component = 0.9
        elif part_mass_lb >= 15:
            mass_component = 0.5
        else:
            mass_component = 0.0
        if mass_component > 0:
            fixture_components["mass_handling_hr"] = round(mass_component, 3)
            fixture_hours += mass_component

    if fixture_hours > 0:
        base_component = 0.5
        fixture_components["base_setup_hr"] = round(base_component, 3)
        fixture_hours += base_component

    if fixture_hours > 0:
        fixture_hours = max(0.0, min(20.0, fixture_hours))
        confidence = "medium"
        if len(fixture_signals) >= 4:
            confidence = "high"
        elif len(fixture_signals) <= 1:
            confidence = "low"
        knobs["fixture_build"] = {
            "confidence": confidence,
            "signals": fixture_signals,
            "recommended": {
                "build_hours": round(fixture_hours, 3),
                "bounds_hr": [0.0, 20.0],
                "components": fixture_components,
            },
            "targets": ["fixture_build_hr"],
        }

    # --- Tapping --------------------------------------------------------------
    tap_details = combined_agg.get("tap_details") or []
    tap_minutes_total = float(combined_agg.get("tap_minutes_hint") or 0.0)
    npt_qty = int(combined_agg.get("npt_qty") or 0)
    if tap_details or tap_minutes_total:
        tap_signals = []
        for detail in tap_details:
            try:
                qty = int(detail.get("qty", 0) or 0)
            except Exception:
                qty = 0
            spec = detail.get("spec") or ""
            if qty and spec:
                tap_signals.append(f"{qty}×{spec}")
        if not tap_signals and combined_agg.get("tap_qty"):
            tap_signals.append(f"Tap qty {combined_agg.get('tap_qty')}")
        npt_inspection_hr = 0.0
        if npt_qty:
            npt_inspection_hr = npt_qty * (NPT_INSPECTION_MIN_PER_HOLE / 60.0)
            tap_signals.append(f"Pipe taps detected ({npt_qty})")
        knobs["tapping"] = {
            "confidence": "high" if tap_signals else "medium",
            "signals": tap_signals,
            "recommended": {
                "tapping_minutes": round(tap_minutes_total, 3) if tap_minutes_total else None,
                "tapping_hours": round(tap_minutes_total / 60.0, 3) if tap_minutes_total else None,
                "tap_details": tap_details,
                "npt_qty": npt_qty or None,
                "npt_inspection_hr": round(npt_inspection_hr, 3) if npt_inspection_hr else None,
            },
            "targets": [
                "process_hour_multipliers.drilling",
                "process_hour_adders.inspection",
            ],
        }

    # --- Counterbores ---------------------------------------------------------
    cbore_qty = int(combined_agg.get("cbore_qty") or 0)
    if cbore_qty:
        cbore_minutes = float(combined_agg.get("cbore_minutes_hint") or (cbore_qty * CBORE_MIN_PER_SIDE_MIN))
        cbore_signals = [f"Counterbores qty {cbore_qty}"]
        if combined_agg.get("double_sided_cbore"):
            cbore_signals.append("Counterbore FRONT/BACK callout")
        knobs["counterbore"] = {
            "confidence": "high",
            "signals": cbore_signals,
            "recommended": {
                "minutes": round(cbore_minutes, 3),
                "hours": round(cbore_minutes / 60.0, 3),
                "double_sided": bool(combined_agg.get("double_sided_cbore")),
            },
            "targets": ["process_hour_multipliers.drilling"],
        }

    # --- Countersinks --------------------------------------------------------
    csk_qty = int(combined_agg.get("csk_qty") or 0)
    if csk_qty:
        csk_minutes = float(combined_agg.get("csk_minutes_hint") or (csk_qty * CSK_MIN_PER_SIDE_MIN))
        csk_signals = [f"Countersinks qty {csk_qty}"]
        if combined_agg.get("double_sided_csk"):
            csk_signals.append("Countersink FRONT/BACK callout")
        angle_matches = sorted({match.group(1) for match in RE_CSK_ANGLE.finditer(tokens_upper)})
        if angle_matches:
            csk_signals.append(f"Countersink angles {', '.join(angle_matches)}°")
        knobs["countersink"] = {
            "confidence": "high",
            "signals": csk_signals,
            "recommended": {
                "minutes": round(csk_minutes, 3),
                "hours": round(csk_minutes / 60.0, 3),
                "double_sided": bool(combined_agg.get("double_sided_csk")),
                "angles": angle_matches or None,
            },
            "targets": ["process_hour_multipliers.drilling"],
        }

    # --- Precision finishing (ream / jig grind) -----------------------------
    jig_matches = list(RE_JIG_GRIND.finditer(tokens_upper))
    ream_matches = list(RE_REAM.finditer(tokens_upper))
    tight_tol_matches = list(RE_TIGHT_TOL.finditer(tokens_upper))
    fit_matches = list(RE_FIT_CLASS.finditer(tokens_upper))
    precision_signals: list[str] = []
    if jig_matches:
        precision_signals.append(f"JIG GRIND callouts ×{len(jig_matches)}")
    if ream_matches:
        precision_signals.append(f"REAM instructions ×{len(ream_matches)}")
    if tight_tol_matches:
        precision_signals.append(f"±0.0002 tolerance mentions ×{len(tight_tol_matches)}")
    if fit_matches:
        precision_signals.append(f"Fit classes (H/G) ×{len(fit_matches)}")
    if precision_signals:
        jig_hr = len(jig_matches) * (JIG_GRIND_MIN_PER_FEATURE / 60.0)
        ream_hr = len(ream_matches) * (REAM_MIN_PER_FEATURE / 60.0)
        inspection_hr = (len(tight_tol_matches) + len(fit_matches) + len(ream_matches)) * (TIGHT_TOL_INSPECTION_MIN / 60.0)
        cmm_min = (len(tight_tol_matches) + len(fit_matches)) * TIGHT_TOL_CMM_MIN
        knobs["precision_finish"] = {
            "confidence": "high",
            "signals": precision_signals,
            "recommended": {
                "jig_grind_hr": round(jig_hr, 3) if jig_hr else None,
                "ream_hr": round(ream_hr, 3) if ream_hr else None,
                "inspection_hr": round(inspection_hr, 3) if inspection_hr else None,
                "cmm_minutes": round(cmm_min, 1) if cmm_min else None,
            },
            "targets": [
                "process_hour_adders.inspection",
                "process_hour_multipliers.milling",
            ],
        }

    # --- Material & heat treat -----------------------------------------------
    material_signals: list[str] = []
    mat_family = None
    density = None
    price_class = None
    hardness_range = None
    heat_treat_required = False
    if isinstance(material_info, dict):
        mat_note = material_info.get("material_note")
        if mat_note:
            material_signals.append(mat_note)
        mat_family = material_info.get("material_family")
        density = material_info.get("density_g_cc")
        price_class = material_info.get("material_price_class")
        hardness_range = material_info.get("hardness_hrc_range")
        heat_treat_required = bool(material_info.get("heat_treat_required"))
        for line in material_info.get("heat_treat_notes", []) or []:
            material_signals.append(line)
        if hardness_range:
            lo, hi = hardness_range
            material_signals.append(f"Hardness target {lo:.0f}–{hi:.0f} HRC" if lo != hi else f"Hardness target {lo:.0f} HRC")
        if heat_treat_required and "HEAT TREAT" not in material_signals:
            material_signals.append("HEAT TREAT callout")
    if material_signals:
        wear_multiplier = 1.0
        if hardness_range:
            hi = hardness_range[1]
            wear_multiplier = 1.15 if hi >= 55 else 1.1
        elif heat_treat_required:
            wear_multiplier = 1.1
        heat_treat_pass = 0.0
        if heat_treat_required:
            heat_treat_pass = 120.0
        knobs["material_heat_treat"] = {
            "confidence": "high",
            "signals": material_signals,
            "recommended": {
                "material_family": mat_family,
                "density_g_cc": density,
                "price_class": price_class,
                "heat_treat_pass_through": round(heat_treat_pass, 2) if heat_treat_pass else None,
                "milling_tool_wear_multiplier": round(wear_multiplier, 3),
            },
            "targets": [
                "density_g_cc",
                "add_pass_through.Outsourced Vendors",
                "process_hour_multipliers.milling",
            ],
        }

    # --- Finish / coating ----------------------------------------------------
    finishes = []
    mask_required = False
    if isinstance(material_info, dict):
        finishes = material_info.get("finishes") or []
        mask_required = bool(material_info.get("finish_masking_required"))
    if finishes:
        finish_signals = list(finishes)
        if mask_required:
            finish_signals.append("MASK callout")
        handling_hr = 0.2 + 0.05 * max(len(finishes) - 1, 0)
        vendor_cost = 90.0 + 20.0 * max(len(finishes) - 1, 0)
        masking_hr = 0.3 if (mask_required and any("ANODIZE" in f for f in finishes)) else 0.0
        knobs["finish_coating"] = {
            "confidence": "high",
            "signals": finish_signals,
            "recommended": {
                "vendor_pass_through": round(vendor_cost, 2),
                "handling_hr": round(handling_hr, 3),
                "masking_hr": round(masking_hr, 3) if masking_hr else None,
            },
            "targets": [
                "add_pass_through.Outsourced Vendors",
                "process_hour_adders.handling",
                "process_hour_adders.masking",
            ],
        }

    # --- Thickness fallback --------------------------------------------------
    if thickness_guess and thickness_provenance:
        thickness_signals = [f"Blind feature depth {thickness_guess:.3f} in"]
        knobs["thickness_fallback"] = {
            "confidence": "medium",
            "signals": thickness_signals,
            "recommended": {
                "thickness_in": round(thickness_guess, 3),
                "provenance": thickness_provenance,
            },
            "targets": ["material.thickness_in"],
        }

    # --- Toolchange / diameter families -------------------------------------
    hole_family_count = len(hole_families or geom_families or {})
    tap_family_count = len({(detail.get("spec") or "").upper() for detail in combined_agg.get("tap_details") or [] if detail.get("spec")})
    csk_angles = sorted({match.group(1) for match in RE_CSK_ANGLE.finditer(tokens_upper)})
    if hole_family_count or tap_family_count or csk_angles:
        total_families = hole_family_count + tap_family_count + len(csk_angles)
        toolchange_hours = total_families * 0.03
        tool_signals = []
        if hole_family_count:
            tool_signals.append(f"Hole diameter families {hole_family_count}")
        if tap_family_count:
            tool_signals.append(f"Tap sizes {tap_family_count}")
        if csk_angles:
            tool_signals.append(f"CSK angles {', '.join(csk_angles)}°")
        knobs["toolchange_families"] = {
            "confidence": "high",
            "signals": tool_signals,
            "recommended": {
                "families_total": total_families,
                "toolchange_hours": round(toolchange_hours, 3),
            },
            "targets": [
                "process_hour_adders.setup_toolchanges",
                "process_hour_multipliers.drilling",
                "process_hour_multipliers.milling",
            ],
        }

    # --- Pocketing / slotting ------------------------------------------------
    pocket_area = 0.0
    pocket_count = 0
    if pocket_metrics:
        pocket_area = float(pocket_metrics.get("pocket_area_total_in2") or 0.0)
        pocket_count = int(pocket_metrics.get("pocket_count") or 0)
    slot_matches = [match for match in RE_SLOT_NOTE.finditer(tokens_upper)]
    slot_signals = [match.group(0) for match in slot_matches]
    if pocket_area or slot_signals:
        depth_guess = thickness_guess or 0.25
        milling_adder = pocket_area * depth_guess * 0.02
        if slot_signals:
            milling_adder += max(0.05, 0.03 * len(slot_signals))
        milling_adder = max(0.1 if pocket_area or slot_signals else 0.0, min(milling_adder, 2.0))
        knob_signals = []
        if pocket_area:
            knob_signals.append(f"Pocket area ~{pocket_area:.2f} in² across {pocket_count} loops")
        knob_signals.extend(slot_signals)
        knobs["pocketing_slotting"] = {
            "confidence": "medium",
            "signals": knob_signals,
            "recommended": {
                "milling_hour_adder": round(milling_adder, 3),
                "depth_basis_in": round(depth_guess, 3),
            },
            "targets": ["process_hour_adders.milling"],
        }

    # --- Hardware / assembly -------------------------------------------------
    hardware_items: list[tuple[int, str]] = []
    for line in token_lines:
        m = RE_HARDWARE_LINE.search(line)
        if not m:
            continue
        qty = int(m.group(1))
        item = m.group(2).upper()
        if any(keyword in item for keyword in ("DOWEL", "PIN", "SHCS", "SOCKET", "SCREW", "BOLT", "INSERT", "BUSH")):
            hardware_items.append((qty, item))
    if hardware_items:
        total_qty = sum(q for q, _ in hardware_items)
        hardware_signals = [f"({qty}) {item}" for qty, item in hardware_items]
        assembly_hours = total_qty * 0.05
        hardware_cost = total_qty * 5.0
        knobs["hardware_assembly"] = {
            "confidence": "medium",
            "signals": hardware_signals,
            "recommended": {
                "hardware_pass_through": round(hardware_cost, 2),
                "assembly_hours": round(assembly_hours, 3),
            },
            "targets": [
                f"add_pass_through.{HARDWARE_PASS_LABEL}",
                "process_hour_adders.assembly",
            ],
        }

    # --- Stock selection -----------------------------------------------------
    if stock_plan:
        stock_signals = []
        stock_len = stock_plan.get("stock_len_in")
        stock_wid = stock_plan.get("stock_wid_in")
        stock_thk = stock_plan.get("stock_thk_in")
        if stock_len and stock_wid and stock_thk:
            stock_signals.append(f"Stock {stock_len}×{stock_wid}×{stock_thk} in")
        if stock_plan.get("part_mass_lb"):
            stock_signals.append(f"Net mass ≈ {stock_plan['part_mass_lb']:.2f} lb")
        if stock_signals:
            knobs["stock_selection"] = {
                "confidence": "high",
                "signals": stock_signals,
                "recommended": {
                    "stock_plan": stock_plan,
                },
                "targets": ["material.stock"],
            }

    # --- Risk / contingency --------------------------------------------------
    risk_signals: list[str] = []
    tolerance_mentions = re.findall(r"±\s*0\.00\d", tokens_upper)
    gd_t_mentions = re.findall(r"\b(MMC|LMC|TP|POSITION|PROFILE|FLATNESS)\b", tokens_upper)
    if len(tolerance_mentions) >= 3:
        risk_signals.append(f"Multiple tight tolerances ({len(tolerance_mentions)})")
    if len(gd_t_mentions) >= 2:
        risk_signals.append(f"GD&T callouts ({len(gd_t_mentions)})")
    if "CRITICAL" in tokens_upper:
        risk_signals.append("CRITICAL note present")
    if isinstance(material_info, dict) and material_info.get("material_family") in {"tool steel", "copper", "brass"}:
        risk_signals.append(f"Material family {material_info.get('material_family')}")
    if risk_signals:
        contingency = 0.03 + 0.01 * max(len(risk_signals) - 1, 0)
        contingency = min(contingency, 0.05)
        knobs["risk_contingency"] = {
            "confidence": "medium",
            "signals": risk_signals,
            "recommended": {
                "contingency_pct": round(contingency, 3),
            },
            "targets": ["contingency_pct"],
        }

    # --- Packaging / shipping ------------------------------------------------
    packaging_signals: list[str] = []
    part_mass_lb = None
    if stock_plan:
        part_mass_lb = stock_plan.get("part_mass_lb")
        if stock_plan.get("stock_len_in") and stock_plan.get("stock_wid_in"):
            max_dim = max(stock_plan["stock_len_in"], stock_plan["stock_wid_in"], stock_plan.get("stock_thk_in", 0.0))
        else:
            max_dim = None
    else:
        max_dim = None
    if part_mass_lb and part_mass_lb > 20:
        packaging_signals.append(f"Mass ~{part_mass_lb:.1f} lb")
    if max_dim and max_dim > 18:
        packaging_signals.append(f"Largest dimension ~{max_dim:.1f} in")
    if "FRAGILE" in tokens_upper or "HANDLE WITH CARE" in tokens_upper:
        packaging_signals.append("Fragility note present")
    if packaging_signals:
        packaging_hours = 0.25 + (0.1 if part_mass_lb and part_mass_lb > 40 else 0.0)
        packaging_flat = 25.0 + (10.0 if part_mass_lb and part_mass_lb > 40 else 0.0)
        knobs["packaging_shipping"] = {
            "confidence": "medium",
            "signals": packaging_signals,
            "recommended": {
                "packaging_hours": round(packaging_hours, 3),
                "packaging_flat_cost": round(packaging_flat, 2),
                "shipping_weight_lb": round(part_mass_lb, 2) if part_mass_lb else None,
            },
            "targets": [
                "process_hour_adders.packaging",
                "add_pass_through.Shipping",
            ],
        }

    return knobs


def harvest_hole_specs(doc, to_in: float, table_lines: Sequence[str] | None = None):
    holes: list[dict[str, Any]] = []
    lines = list(table_lines) if table_lines is not None else list(_iter_table_text(doc) or [])
    for line in lines:
        entry = _parse_hole_line(line, to_in, source="TABLE")
        if entry:
            holes.append(entry)
    agg = _aggregate_hole_entries(holes)
    notes: list[str] = []
    if agg.get("deepest_hole_in"):
        notes.append(f"Max hole depth detected: {agg['deepest_hole_in']:.3f} in")
    return holes, agg, notes


def harvest_leaders(doc) -> list[str]:
    texts: list[str] = []
    spaces: list[Any] = []
    try:
        spaces.append(doc.modelspace())
    except Exception:
        pass
    try:
        for name in doc.layouts.names_in_taborder():
            if name.lower() in {"model", "defpoints"}:
                continue
            try:
                spaces.append(doc.layouts.get(name).entity_space)
            except Exception:
                continue
    except Exception:
        pass
    for space in spaces:
        try:
            for ml in space.query("MLEADER"):
                try:
                    texts.append(ml.get_mtext().plain_text())
                except Exception:
                    continue
        except Exception:
            continue
        try:
            for _ld in space.query("LEADER"):
                # Many LEADER entities lack attached text; skip
                pass
        except Exception:
            continue
    return [t for t in texts if t]


def _infer_material_family_details(material_note: str | None) -> tuple[str | None, float | None, str | None]:
    if not material_note:
        return (None, None, None)
    note = material_note.strip().upper()
    families = [
        ("aluminum", ["6061", "5052", "5083", "2024", "7050", "7075"], 2.7, "commodity"),
        ("stainless steel", ["17-4", "15-5", "316", "304", "303", "410", "420"], 7.9, "premium"),
        ("tool steel", ["A2", "D2", "S7", "O1", "H13", "AISI A2", "AISI D2"], 7.8, "premium"),
        ("alloy steel", ["4140", "4340", "8620", "1045", "4130"], 7.85, "commodity"),
        ("copper", ["C110", "COPPER"], 8.9, "premium"),
        ("brass", ["BRASS", "C260", "C360"], 8.5, "premium"),
        ("plastic", ["DELRIN", "ACETAL", "NYLON", "UHMW", "PEEK", "PVC", "ABS"], 1.4, "commodity"),
    ]
    for family, tokens, density, price_class in families:
        for token in tokens:
            if token in note:
                return (family, density, price_class)
    if "AL" in note or "ALUMIN" in note:
        return ("aluminum", 2.7, "commodity")
    if "STAINLESS" in note:
        return ("stainless steel", 7.9, "premium")
    if any(key in note for key in ("TOOL", "A2", "D2", "S7")):
        return ("tool steel", 7.8, "premium")
    if "STEEL" in note:
        return ("alloy steel", 7.85, "commodity")
    return (None, None, None)


def harvest_material_finish(tokens_text: str) -> dict[str, Any]:
    text = tokens_text or ""
    U = text.upper()
    mat = None
    hardness_range: tuple[float, float] | None = None
    heat_treat_required = False
    heat_treat_notes: list[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        if not mat:
            m = RE_MAT.search(line)
            if m:
                mat = m.group(2).strip()
        if RE_HEAT_TREAT.search(line):
            heat_treat_required = True
            heat_treat_notes.append(line.strip())
        hmatch = RE_HARDNESS.search(line)
        if hmatch:
            try:
                lo = float(hmatch.group(1))
                hi = float(hmatch.group(2)) if hmatch.group(2) else lo
                hardness_range = (min(lo, hi), max(lo, hi))
            except Exception:
                pass
    if not mat:
        m = RE_MAT.search(U)
        if m:
            mat = m.group(2).strip()
    if mat and " HT" in mat.upper():
        heat_treat_required = True
    if RE_HEAT_TREAT.search(U):
        heat_treat_required = True
    coats = sorted({match.group(0).upper() for match in RE_COAT.finditer(U)})
    tol = None
    mt = RE_TOL.search(U)
    if mt:
        tol = mt.group(1).replace(" ", "")
    family, density, price_class = _infer_material_family_details(mat)
    finish_masking = "MASK" in U
    result: dict[str, Any] = {
        "material_note": mat,
        "material_family": family,
        "density_g_cc": density,
        "material_price_class": price_class,
        "finishes": coats,
        "default_tol": tol,
        "heat_treat_required": bool(heat_treat_required),
        "heat_treat_notes": heat_treat_notes,
        "hardness_hrc_range": hardness_range,
        "finish_masking_required": finish_masking,
    }
    return result


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def detect_pockets_and_islands(doc, to_in: float) -> dict[str, Any]:
    if doc is None:
        return {}
    areas_in2: list[float] = []
    pocket_details: list[float] = []
    for sp in _spaces(doc):
        try:
            polylines = list(sp.query("LWPOLYLINE"))
        except Exception:
            polylines = []
        for pl in polylines:
            try:
                get_points = getattr(pl, "get_points", None)
                pts = list(get_points("xy")) if callable(get_points) else []
            except Exception:
                continue
            if len(pts) < 3:
                continue
            flags = getattr(getattr(pl, "dxf", object()), "flags", 0)
            is_closed = bool(getattr(pl, "closed", False)) or bool(flags & 1)
            if not is_closed:
                continue
            xy_pts = [(float(x) * to_in, float(y) * to_in) for x, y in pts]
            area = abs(_polygon_area(xy_pts))
            if area < 0.05:
                continue
            areas_in2.append(area)
        try:
            lwps = list(sp.query("POLYLINE"))
        except Exception:
            lwps = []
        for poly in lwps:
            vertices = []
            for v in getattr(poly, "vertices", []):
                try:
                    vertices.append((float(v.dxf.location.x) * to_in, float(v.dxf.location.y) * to_in))
                except Exception:
                    continue
            if len(vertices) < 3:
                continue
            if vertices[0] != vertices[-1]:
                vertices.append(vertices[0])
            area = abs(_polygon_area(vertices))
            if area < 0.05:
                continue
            areas_in2.append(area)
    if not areas_in2:
        return {}
    largest = max(areas_in2)
    for area in areas_in2:
        if area < 0.9 * largest:
            pocket_details.append(area)
    pocket_total = sum(pocket_details)
    return {
        "closed_loop_areas_in2": [round(a, 3) for a in areas_in2],
        "pocket_candidate_areas_in2": [round(a, 3) for a in pocket_details],
        "pocket_area_total_in2": round(pocket_total, 3) if pocket_total else 0.0,
        "pocket_count": len(pocket_details),
    }


def plan_stock_blank(
    plate_len_in: float | None,
    plate_wid_in: float | None,
    thickness_in: float | None,
    density_g_cc: float | None,
    hole_families: dict | None,
) -> dict[str, Any]:
    if not plate_len_in or not plate_wid_in or not thickness_in:
        return {}
    stock_lengths = [6, 8, 10, 12, 18, 24, 36, 48]
    stock_widths = [6, 8, 10, 12, 18, 24, 36]
    stock_thicknesses = [0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    def pick_size(value: float, options: Sequence[float]) -> float:
        for opt in options:
            if value <= opt:
                return opt
        return math.ceil(value)

    stock_len = pick_size(float(plate_len_in) * 1.05, stock_lengths)
    stock_wid = pick_size(float(plate_wid_in) * 1.05, stock_widths)
    stock_thk = pick_size(float(thickness_in) * 1.05, stock_thicknesses)
    volume_in3 = float(plate_len_in) * float(plate_wid_in) * float(thickness_in)
    hole_area = 0.0
    if hole_families:
        for dia_in, qty in hole_families.items():
            try:
                d = float(dia_in)
                q = float(qty)
            except Exception:
                continue
            hole_area += math.pi * (d / 2.0) ** 2 * q
    net_volume_in3 = max(volume_in3 - hole_area * float(thickness_in), 0.0)
    density = float(density_g_cc or 0.0)
    part_mass_lb = density * LB_PER_IN3_PER_GCC * net_volume_in3 if density > 0 else None
    stock_volume_in3 = stock_len * stock_wid * stock_thk
    stock_mass_lb = density * LB_PER_IN3_PER_GCC * stock_volume_in3 if density > 0 else None
    return {
        "stock_len_in": round(stock_len, 3),
        "stock_wid_in": round(stock_wid, 3),
        "stock_thk_in": round(stock_thk, 3),
        "part_volume_in3": round(volume_in3, 3),
        "stock_volume_in3": round(stock_volume_in3, 3),
        "net_volume_in3": round(net_volume_in3, 3),
        "part_mass_lb": round(part_mass_lb, 3) if part_mass_lb else None,
        "stock_mass_lb": round(stock_mass_lb, 3) if stock_mass_lb else None,
    }


def _build_geo_from_ezdxf_doc(doc) -> dict[str, Any]:
    units = detect_units_scale(doc)
    to_in = units.get("to_in", 1.0) or 1.0
    ords = harvest_ordinates(doc, float(to_in))
    table_lines = list(_iter_table_text(doc) or [])
    coord_info = harvest_coordinate_table(doc, float(to_in))
    outline_hint = harvest_outline_bbox(doc, float(to_in))
    holes, hole_agg, hole_notes = harvest_hole_specs(doc, float(to_in), table_lines=table_lines)
    leaders = harvest_leaders(doc)

    plate_len = ords.get("plate_len_in")
    plate_wid = ords.get("plate_wid_in")
    plate_prov = ords.get("provenance")
    if (plate_len is None or plate_wid is None) and outline_hint:
        if plate_len is None and outline_hint.get("plate_len_in"):
            plate_len = outline_hint.get("plate_len_in")
            plate_prov = outline_hint.get("prov")
        if plate_wid is None and outline_hint.get("plate_wid_in"):
            plate_wid = outline_hint.get("plate_wid_in")
            plate_prov = outline_hint.get("prov")
    if (plate_len is None or plate_wid is None) and coord_info:
        if plate_wid is None and coord_info.get("x_span_in"):
            plate_wid = coord_info.get("x_span_in")
            plate_prov = coord_info.get("prov")
        if plate_len is None and coord_info.get("y_span_in"):
            plate_len = coord_info.get("y_span_in")
            plate_prov = coord_info.get("prov")
    ords["plate_len_in"] = plate_len
    ords["plate_wid_in"] = plate_wid
    if plate_prov:
        ords["provenance"] = plate_prov

    leader_entries: list[dict[str, Any]] = []
    for text in leaders:
        entry = _parse_hole_line(text, float(to_in), source="LEADER")
        if entry:
            leader_entries.append(entry)
    if leader_entries:
        unique_leaders = _dedupe_hole_entries(holes, leader_entries)
        if unique_leaders:
            holes.extend(unique_leaders)

    combined_agg = _aggregate_hole_entries(holes)
    if combined_agg.get("provenance") is None and isinstance(hole_agg, dict):
        combined_agg["provenance"] = hole_agg.get("provenance")
    max_depth = combined_agg.get("deepest_hole_in")
    notes = list(hole_notes)
    if max_depth and (not notes or not any("Max hole depth" in note for note in notes)):
        notes.append(f"Max hole depth detected: {max_depth:.3f} in")
    if combined_agg.get("from_back") and not any("back" in str(note).lower() for note in notes):
        notes.append("Hole chart references BACK operations.")

    tokens_parts: list[str] = []
    try:
        tokens_parts.extend(text_harvest(doc))
    except Exception:
        pass
    for line in table_lines:
        tokens_parts.append(line)
    tokens_parts.extend(leaders)
    tokens_blob = "\n".join(p for p in tokens_parts if p)
    tokens_upper = tokens_blob.upper()
    material_info = harvest_material_finish(tokens_blob)
    strong_material = harvest_finish_ht_material(tokens_upper)
    if strong_material.get("material_note") and not material_info.get("material_note"):
        material_info["material_note"] = strong_material.get("material_note")
    if strong_material.get("finishes"):
        combined_fin = set(material_info.get("finishes") or []) | set(strong_material.get("finishes") or [])
        material_info["finishes"] = sorted(combined_fin)
    if strong_material.get("heat_treat"):
        notes = material_info.get("heat_treat_notes") or []
        if strong_material["heat_treat"] not in notes:
            notes.append(strong_material["heat_treat"])
        material_info["heat_treat_notes"] = notes
        material_info["heat_treat_required"] = True
    hardware_notes = harvest_hardware_notes(tokens_upper)
    gdt_summary = harvest_gdt(doc)
    concentric_info = classify_concentric(doc, float(to_in))
    if concentric_info.get("cbore_pairs_geom"):
        existing = int(combined_agg.get("cbore_qty") or 0)
        combined_agg["cbore_qty"] = max(existing, int(concentric_info.get("cbore_pairs_geom") or 0))
    if concentric_info.get("csk_pairs_geom"):
        existing_csk = int(combined_agg.get("csk_qty") or 0)
        combined_agg["csk_qty"] = max(existing_csk, int(concentric_info.get("csk_pairs_geom") or 0))

    max_depth = combined_agg.get("deepest_hole_in")
    thickness_guess = None
    thickness_provenance = None
    if max_depth:
        try:
            depth_val = float(max_depth)
        except Exception:
            depth_val = 0.0
        if depth_val > 0.0:
            clamped = min(3.0, max(0.125, depth_val))
            thickness_guess = clamped
            thickness_provenance = "GEO hole depth fallback"
            if clamped != depth_val:
                msg = f"Thickness inferred from hole depth {depth_val:.3f} in (clamped to {clamped:.3f} in)"
            else:
                msg = f"Thickness inferred from hole depth {clamped:.3f} in"
            if msg not in notes:
                notes.append(msg)

    # unified hole counting with fallbacks
    table_info = hole_count_from_acad_table(doc)
    cnt = None
    fam: dict | None = None
    tap_classes_from_table: dict[str, int] | None = None
    tap_qty_from_table = 0
    source = "HOLE TABLE (ACAD_TABLE)"
    if table_info:
        cnt = table_info.get("hole_count")
        fam = table_info.get("hole_diam_families_in")
        raw_classes = table_info.get("tap_class_counts") if isinstance(table_info.get("tap_class_counts"), dict) else None
        if raw_classes:
            tap_classes_from_table = {k: int(v) for k, v in raw_classes.items() if v}
        tap_qty_from_table = int(table_info.get("tap_qty_from_table") or 0)
        if table_info.get("provenance_holes"):
            source = str(table_info.get("provenance_holes"))
    if not cnt:
        text_cnt, text_fam = hole_count_from_text_table(doc, table_lines)
        if text_cnt:
            cnt, fam = text_cnt, text_fam
            source = "HOLE TABLE (TEXT)"

    geom_cnt, geom_fam = hole_count_from_geometry(doc, float(to_in))
    if not cnt and geom_cnt:
        cnt, fam = geom_cnt, geom_fam
        source = "GEOMETRY CIRCLE COUNT"

    flags: list[str] | None = None
    if cnt and geom_cnt:
        if abs(cnt - geom_cnt) / max(cnt, geom_cnt) > 0.15:
            flags = [f"hole_count_conflict: table={cnt}, geom={geom_cnt}"]
    coord_count = coord_info.get("coord_count") if isinstance(coord_info, dict) else None
    if coord_count and cnt:
        if abs(coord_count - cnt) / max(cnt, coord_count) > 0.15:
            if not flags:
                flags = []
            flags.append(f"hole_count_coord_conflict: table={cnt}, coord={coord_count}")

    if cnt:
        combined_agg["hole_count"] = int(cnt)
    if tap_classes_from_table:
        combined_agg["tap_class_counts"] = dict(tap_classes_from_table)
    if tap_qty_from_table:
        combined_agg["tap_qty"] = max(int(tap_qty_from_table), int(combined_agg.get("tap_qty") or 0))
    if not combined_agg.get("provenance") and table_info and table_info.get("provenance_holes"):
        combined_agg["provenance"] = table_info.get("provenance_holes")

    pocket_metrics = detect_pockets_and_islands(doc, float(to_in))
    stock_plan = plan_stock_blank(
        ords.get("plate_len_in"),
        ords.get("plate_wid_in"),
        thickness_guess,
        material_info.get("density_g_cc") if isinstance(material_info, dict) else None,
        fam or geom_fam,
    )

    inference_knobs = derive_inference_knobs(
        tokens_blob,
        combined_agg,
        hole_families=fam,
        geom_families=geom_fam,
        material_info=material_info,
        thickness_guess=thickness_guess,
        thickness_provenance=thickness_provenance,
        pocket_metrics=pocket_metrics,
        stock_plan=stock_plan,
        table_hole_count=cnt,
        geometry_hole_count=geom_cnt,
    )

    geo = {
        "plate_len_in": ords.get("plate_len_in"),
        "plate_wid_in": ords.get("plate_wid_in"),
        "thickness_in_guess": thickness_guess,
        "hole_count": combined_agg.get("hole_count") or 0,
        "tap_qty": combined_agg.get("tap_qty") or 0,
        "cbore_qty": combined_agg.get("cbore_qty") or 0,
        "csk_qty": combined_agg.get("csk_qty") or 0,
        "tap_details": combined_agg.get("tap_details") or [],
        "tap_minutes_hint": combined_agg.get("tap_minutes_hint"),
        "tap_class_counts": combined_agg.get("tap_class_counts") or {},
        "npt_qty": combined_agg.get("npt_qty") or 0,
        "cbore_minutes_hint": combined_agg.get("cbore_minutes_hint"),
        "csk_minutes_hint": combined_agg.get("csk_minutes_hint"),
        "from_back": bool(combined_agg.get("from_back")),
        "needs_back_face": bool(combined_agg.get("from_back")),
        "provenance": {
            "plate_size": ords.get("provenance"),
            "thickness": thickness_provenance,
            "holes": combined_agg.get("provenance"),
        },
        "notes": notes,
        "raw": {"holes": holes, "leaders": leaders, "leader_entries": leader_entries},
        "units": units,
    }
    geo.update(material_info)
    if hardware_notes.get("hardware_items") is not None:
        geo["hardware_items"] = hardware_notes.get("hardware_items")
        if hardware_notes.get("prov"):
            geo.setdefault("provenance", {})["hardware"] = hardware_notes.get("prov")
    if gdt_summary.get("gdt"):
        geo["gdt"] = gdt_summary.get("gdt")
    if gdt_summary.get("prov"):
        geo.setdefault("provenance", {})["gdt"] = gdt_summary.get("prov")
    if table_lines:
        geo["chart_lines"] = list(table_lines)
    if inference_knobs:
        geo["inference_knobs"] = inference_knobs
    geo["hole_count"] = int(cnt or 0)
    geo["hole_diam_families_in"] = fam or {}
    geo.setdefault("provenance", {})["holes"] = source
    geo["hole_count_geom"] = geom_cnt or 0
    if geom_fam:
        geo["hole_diam_families_geom_in"] = geom_fam
    if coord_info:
        if coord_info.get("coord_count") is not None:
            geo["coord_count"] = coord_info.get("coord_count")
        if coord_info.get("coord_ids"):
            geo["coord_ids"] = coord_info.get("coord_ids")
        if coord_info.get("x_span_in") is not None:
            geo.setdefault("outline_hint", {})["x_span_in"] = coord_info.get("x_span_in")
            geo["x_span_in"] = coord_info.get("x_span_in")
        if coord_info.get("y_span_in") is not None:
            geo.setdefault("outline_hint", {})["y_span_in"] = coord_info.get("y_span_in")
            geo["y_span_in"] = coord_info.get("y_span_in")
        if coord_info.get("prov"):
            geo.setdefault("provenance", {})["coord_list"] = coord_info.get("prov")
    if outline_hint:
        geo["outline_bbox"] = outline_hint
        if outline_hint.get("prov"):
            geo.setdefault("provenance", {})["outline_bbox"] = outline_hint.get("prov")
    if concentric_info:
        if concentric_info.get("cbore_pairs_geom") is not None:
            geo["cbore_pairs_geom"] = concentric_info.get("cbore_pairs_geom")
        if concentric_info.get("csk_pairs_geom") is not None:
            geo["csk_pairs_geom"] = concentric_info.get("csk_pairs_geom")
        if concentric_info.get("prov"):
            geo.setdefault("provenance", {})["concentric"] = concentric_info.get("prov")
    if thickness_provenance:
        geo.setdefault("provenance", {})["thickness"] = thickness_provenance
    if material_info.get("density_g_cc"):
        geo["density_g_cc"] = material_info.get("density_g_cc")
    if material_info.get("material_family"):
        geo["material_family"] = material_info.get("material_family")
    if material_info.get("material_price_class"):
        geo["material_price_class"] = material_info.get("material_price_class")
    if material_info.get("heat_treat_required"):
        geo["heat_treat_required"] = True
    if material_info.get("hardness_hrc_range"):
        geo["hardness_hrc_range"] = material_info.get("hardness_hrc_range")
    if material_info.get("finish_masking_required"):
        geo["finish_masking_required"] = bool(material_info.get("finish_masking_required"))
    if pocket_metrics:
        geo["pocket_metrics"] = pocket_metrics
        if pocket_metrics.get("pocket_area_total_in2"):
            geo["pocket_area_total_in2"] = pocket_metrics.get("pocket_area_total_in2")
            geo["pocket_count"] = pocket_metrics.get("pocket_count")
    if stock_plan:
        geo["stock_plan_guess"] = stock_plan
    geo["hole_family_count"] = len(fam or geom_fam or {})
    if tap_classes_from_table:
        geo["tap_classes"] = dict(tap_classes_from_table)
    else:
        geo["tap_classes"] = tap_classes_from_lines(geo.get("chart_lines"))
    deburr_hints = quick_deburr_estimates(geo.get("edge_len_in"), geo.get("hole_count"))
    deburr_prov = deburr_hints.pop("prov", None)
    geo.update(deburr_hints)
    if deburr_prov:
        geo.setdefault("provenance", {})["deburr"] = deburr_prov
    if flags:
        geo["flags"] = flags
    return geo


def _extract_entity_text(entity: Any) -> str:
    """Return the textual content for TEXT/MTEXT entities.

    ezdxf exposes ``plain_text`` on MTEXT entities only, so we guard the
    attribute access at runtime and keep static type-checkers happy by using
    ``getattr`` instead of attribute access directly on ``DXFEntity``.
    """

    if entity is None:
        return ""

    plain_callable = getattr(entity, "plain_text", None)
    text_value: Any
    if callable(plain_callable):
        try:
            text_value = plain_callable()
        except Exception:
            text_value = None
    else:
        text_value = None

    if not text_value:
        dxf_attr = getattr(entity, "dxf", None)
        text_value = getattr(dxf_attr, "text", None) if dxf_attr is not None else None

    return str(text_value).strip() if text_value else ""


def _coerce_int_or_zero(value: Any) -> int:
    """Coerce ``value`` to an integer, returning ``0`` on failure."""

    coerced = _coerce_float_or_none(value)
    if coerced is None:
        return 0
    try:
        return int(coerced)
    except Exception:
        return 0


def extract_2d_features_from_dxf_or_dwg(path: str) -> dict:
    ezdxf_mod = require_ezdxf()


    # --- load doc ---
    dxf_text_path: str | None = None
    doc = None
    lower_path = path.lower()
    if lower_path.endswith(".dwg"):
        if geometry.HAS_ODAFC:
            # uses ODAFileConverter through ezdxf, no env var needed
            from ezdxf.addons import odafc  # type: ignore

            doc = odafc.readfile(path)
        else:
            dxf_path = convert_dwg_to_dxf(path, out_ver="ACAD2018")
            dxf_text_path = dxf_path
            doc = ezdxf_mod.readfile(dxf_path)
    else:
        doc = ezdxf_mod.readfile(path)
        dxf_text_path = path

    sp = doc.modelspace()
    units = detect_units_scale(doc)
    to_in = float(units.get("to_in", 1.0) or 1.0)
    u2mm = to_in * 25.4

    geo = _build_geo_from_ezdxf_doc(doc)

    geo_read_more: dict[str, Any] | None = None
    extra_geo_loader = _build_geo_from_dxf_hook or build_geo_from_dxf_path
    if extra_geo_loader and dxf_text_path:
        try:
            geo_read_more = extra_geo_loader(dxf_text_path)
        except Exception as exc:  # pragma: no cover - diagnostic only
            geo_read_more = {"ok": False, "error": str(exc)}

    if geo_read_more:
        geo["geo_read_more"] = geo_read_more
        if geo_read_more.get("ok"):
            def _should_replace(current: Any) -> bool:
                if current is None:
                    return True
                if isinstance(current, str):
                    stripped = current.strip()
                    if not stripped:
                        return True
                    try:
                        return float(stripped) == 0.0
                    except Exception:
                        return False
                if isinstance(current, (int, float)):
                    try:
                        return float(current) == 0.0
                    except Exception:
                        return False
                return False

            def _adopt(target: str, source_key: str) -> None:
                if target not in geo or _should_replace(geo.get(target)):
                    value = geo_read_more.get(source_key)
                    if value not in (None, ""):
                        geo[target] = value

            _adopt("plate_len_in", "plate_len_in")
            _adopt("plate_wid_in", "plate_wid_in")
            _adopt("thickness_in_guess", "deepest_hole_in")
            _adopt("material_note", "material_note")
            if not geo.get("finishes") and geo_read_more.get("finishes"):
                geo["finishes"] = list(geo_read_more.get("finishes") or [])
            if not geo.get("default_tol") and geo_read_more.get("default_tol"):
                geo["default_tol"] = geo_read_more.get("default_tol")
            if not geo.get("revision") and geo_read_more.get("revision"):
                geo["revision"] = geo_read_more.get("revision")

            for qty_key in ("tap_qty", "cbore_qty", "csk_qty"):
                try:
                    current = int(float(geo.get(qty_key, 0) or 0))
                except Exception:
                    current = 0
                try:
                    candidate = int(float(geo_read_more.get(qty_key, 0) or 0))
                except Exception:
                    candidate = 0
                if candidate > current:
                    geo[qty_key] = candidate

            if geo_read_more.get("holes_from_back"):
                geo["from_back"] = True
                geo["needs_back_face"] = True

            for key in (
                "edge_len_in",
                "outline_area_in2",
                "hole_diam_families_in",
                "hole_table_families_in",
                "hole_count_geom",
                "min_hole_in",
                "max_hole_in",
            ):
                value = geo_read_more.get(key)
                if value not in (None, ""):
                    geo[key] = value

            more_prov = geo_read_more.get("provenance")
            if isinstance(more_prov, dict):
                existing_prov = geo.get("provenance")
                merged: dict[str, Any]
                if isinstance(existing_prov, dict):
                    merged = existing_prov.copy()
                else:
                    merged = {}
                for key, value in more_prov.items():
                    if key not in merged and value:
                        merged[key] = value
                geo["provenance"] = merged

            if geo_read_more.get("chart_lines"):
                existing_lines = list(geo.get("chart_lines") or []) if isinstance(geo.get("chart_lines"), list) else []
                for line in geo_read_more.get("chart_lines") or []:
                    if line not in existing_lines:
                        existing_lines.append(line)
                if existing_lines:
                    geo["chart_lines"] = existing_lines
        else:
            geo.setdefault("geo_read_more_error", geo_read_more.get("error"))

    # perimeter from lightweight polylines, polylines, arcs
    import math
    per = 0.0
    for e in sp.query("LWPOLYLINE"):
        get_points = getattr(e, "get_points", None)
        pts: list[tuple[float, float, Any]] = []
        if callable(get_points):
            try:
                pts = list(get_points("xyb"))
            except Exception:
                pts = []
        for i in range(len(pts)):
            x1, y1, _ = pts[i]
            x2, y2, _ = pts[(i + 1) % len(pts)]
            per += math.hypot(x2 - x1, y2 - y1)
    for e in sp.query("POLYLINE"):
        raw_vertices = getattr(e, "vertices", None)
        vs: list[tuple[float, float]] = []
        if raw_vertices is not None:
            try:
                vs = [
                    (float(v.dxf.location.x), float(v.dxf.location.y))
                    for v in raw_vertices
                ]
            except Exception:
                vs = []
        for i in range(len(vs) - 1):
            x1, y1 = vs[i]
            x2, y2 = vs[i + 1]
            per += math.hypot(x2 - x1, y2 - y1)
    for e in sp.query("ARC"):
        per += abs(e.dxf.end_angle - e.dxf.start_angle) * math.pi/180.0 * e.dxf.radius

    # holes from circles
    holes = list(sp.query("CIRCLE"))
    entity_holes_mm = [float(2.0 * c.dxf.radius * u2mm) for c in holes]
    hole_diams_mm = [round(val, 2) for val in entity_holes_mm]

    chart_lines: list[str] = []
    chart_ops: list[dict[str, Any]] = []
    chart_reconcile: dict[str, Any] | None = None
    chart_source: str | None = None
    chart_summary: dict[str, Any] | None = None

    extractor = _extract_text_lines_from_dxf or geometry.extract_text_lines_from_dxf
    if extractor and dxf_text_path:
        try:
            chart_lines = extractor(dxf_text_path)
        except Exception:
            chart_lines = []
    if not chart_lines:
        chart_lines = _extract_text_lines_from_ezdxf_doc(doc)

    if chart_lines:
        chart_summary = summarize_hole_chart_lines(chart_lines)
    parser = _parse_hole_table_lines or geometry.parse_hole_table_lines
    if chart_lines and parser:
        try:
            hole_rows = parser(chart_lines)
        except Exception:
            hole_rows = []
        if hole_rows:
            chart_ops = hole_rows_to_ops(hole_rows)
            chart_source = "dxf_text_regex"
            chart_reconcile = reconcile_holes(entity_holes_mm, chart_ops)
    if chart_summary:
        geo.setdefault("chart_summary", chart_summary)
        if chart_summary.get("tap_qty"):
            geo["tap_qty"] = max(
                _coerce_int_or_zero(geo.get("tap_qty")),
                _coerce_int_or_zero(chart_summary.get("tap_qty")),
            )
        if chart_summary.get("cbore_qty"):
            geo["cbore_qty"] = max(
                _coerce_int_or_zero(geo.get("cbore_qty")),
                _coerce_int_or_zero(chart_summary.get("cbore_qty")),
            )
        if chart_summary.get("csk_qty"):
            geo["csk_qty"] = max(
                _coerce_int_or_zero(geo.get("csk_qty")),
                _coerce_int_or_zero(chart_summary.get("csk_qty")),
            )
        deepest_chart = _coerce_float_or_none(chart_summary.get("deepest_hole_in"))
        existing_thickness = _coerce_float_or_none(geo.get("thickness_in_guess"))
        if deepest_chart is not None and (
            existing_thickness is None or deepest_chart > existing_thickness
        ):
            geo["thickness_in_guess"] = deepest_chart
            provenance_value = geo.get("provenance")
            if isinstance(provenance_value, dict):
                provenance_value["thickness"] = "HOLE TABLE depth max"
            else:
                geo["provenance"] = {"thickness": "HOLE TABLE depth max"}
        if chart_summary.get("from_back"):
            geo["from_back"] = True
            geo["needs_back_face"] = True
            notes = geo.get("notes")
            if isinstance(notes, list) and not any("back" in str(n).lower() for n in notes):
                notes.append("Hole chart references BACK operations.")

    # scrape text for thickness/material
    text_fragments: list[str] = []
    for text_entity in sp.query("TEXT"):
        fragment = _extract_entity_text(text_entity)
        if fragment:
            text_fragments.append(fragment)
    for mtext_entity in sp.query("MTEXT"):
        fragment = _extract_entity_text(mtext_entity)
        if fragment:
            text_fragments.append(fragment)
    txt = " ".join(text_fragments).lower()
    import re
    thickness_mm = None
    m = re.search(r"(thk|thickness)\s*[:=]?\s*([0-9.]+)\s*(mm|in|in\.|\")", txt)
    if m:
        thickness_mm = float(m.group(2)) * (25.4 if (m.group(3).startswith("in") or m.group(3)=='"') else 1.0)
    material = None
    mm = re.search(r"(matl|material)\s*[:=]?\s*([a-z0-9 \-\+]+)", txt)
    if mm:
        material = mm.group(2).strip()

    if not thickness_mm:
        thickness_guess_in = _coerce_float_or_none(geo.get("thickness_in_guess"))
        if thickness_guess_in is not None:
            thickness_mm = thickness_guess_in * 25.4
    if not material:
        material = geo.get("material_note")

    result: dict[str, Any] = {
        "kind": "2D",
        "source": Path(path).suffix.lower().lstrip("."),
        "profile_length_mm": round(per * u2mm, 2),
        "hole_diams_mm": hole_diams_mm,
        "hole_count": len(hole_diams_mm),
        "thickness_mm": thickness_mm,
        "material": material,
        "geo": geo,
    }
    if geo.get("hole_count"):
        result["hole_count_table"] = geo.get("hole_count")
    if geo.get("tap_qty") or geo.get("cbore_qty") or geo.get("csk_qty"):
        result["feature_counts"] = {
            "tap_qty": geo.get("tap_qty", 0),
            "cbore_qty": geo.get("cbore_qty", 0),
            "csk_qty": geo.get("csk_qty", 0),
        }
    if entity_holes_mm:
        result["hole_diams_mm_precise"] = entity_holes_mm
    if chart_ops:
        result["chart_ops"] = chart_ops
    if chart_source:
        result["chart_source"] = chart_source
    if chart_lines:
        result["chart_lines"] = chart_lines
    if chart_summary:
        result["chart_summary"] = chart_summary
    if chart_reconcile:
        result["chart_reconcile"] = chart_reconcile
        result["hole_chart_agreement"] = bool(chart_reconcile.get("agreement"))
    if geo.get("finishes"):
        result["finishes"] = geo.get("finishes")
    if geo.get("default_tol"):
        result["default_tolerance"] = geo.get("default_tol")
    if geo.get("notes"):
        result["notes"] = list(geo.get("notes") or [])
    if geo_read_more:
        result["geo_read_more"] = geo_read_more
        if geo_read_more.get("ok"):
            edge_len_in_val = _coerce_float_or_none(geo_read_more.get("edge_len_in"))
            if edge_len_in_val is not None and edge_len_in_val > 0:
                result["edge_length_in"] = edge_len_in_val
                result["edge_len_in"] = edge_len_in_val
                result["profile_length_mm"] = round(edge_len_in_val * 25.4, 2)
            outline_area_in2_val = _coerce_float_or_none(geo_read_more.get("outline_area_in2"))
            if outline_area_in2_val is not None and outline_area_in2_val > 0:
                result["outline_area_in2"] = outline_area_in2_val
                result["outline_area_mm2"] = round(outline_area_in2_val * (25.4 ** 2), 2)
            if geo_read_more.get("hole_diam_families_in"):
                result["hole_diam_families_in"] = dict(geo_read_more.get("hole_diam_families_in") or {})
            if geo_read_more.get("hole_table_families_in"):
                result["hole_table_families_in"] = dict(geo_read_more.get("hole_table_families_in") or {})
            hole_count_geom_val = _coerce_float_or_none(geo_read_more.get("hole_count_geom"))
            hole_count_geom = int(hole_count_geom_val) if hole_count_geom_val is not None else None
            current_hole_count = _coerce_int_or_zero(result.get("hole_count"))
            if hole_count_geom is not None and hole_count_geom > current_hole_count:
                result["hole_count"] = hole_count_geom
            if geo_read_more.get("holes_from_back"):
                result["holes_from_back"] = True
            if geo_read_more.get("material_note"):
                result.setdefault("material_note", geo_read_more.get("material_note"))
            if geo_read_more.get("material_note") and not result.get("material"):
                result["material"] = geo_read_more.get("material_note")
            if geo_read_more.get("chart_lines") and not result.get("chart_lines"):
                result["chart_lines"] = list(geo_read_more.get("chart_lines") or [])
            if geo_read_more.get("tap_qty") or geo_read_more.get("cbore_qty") or geo_read_more.get("csk_qty"):
                feature_counts_raw = result.get("feature_counts") if isinstance(result.get("feature_counts"), dict) else {}
                feature_counts: dict[str, Any] = {str(k): v for k, v in dict(feature_counts_raw).items()}
                if geo_read_more.get("tap_qty"):
                    feature_counts["tap_qty"] = max(int(feature_counts.get("tap_qty", 0) or 0), int(geo_read_more.get("tap_qty") or 0))
                if geo_read_more.get("cbore_qty"):
                    feature_counts["cbore_qty"] = max(int(feature_counts.get("cbore_qty", 0) or 0), int(geo_read_more.get("cbore_qty") or 0))
                if geo_read_more.get("csk_qty"):
                    feature_counts["csk_qty"] = max(int(feature_counts.get("csk_qty", 0) or 0), int(geo_read_more.get("csk_qty") or 0))
                if feature_counts:
                    result["feature_counts"] = feature_counts
            for qty_key in ("tap_qty", "cbore_qty", "csk_qty"):
                candidate = _coerce_int_or_zero(geo_read_more.get(qty_key, 0))
                if candidate:
                    current = _coerce_int_or_zero(result.get(qty_key))
                    if candidate > current:
                        result[qty_key] = candidate
    result["units"] = units
    return result
def _extract_text_lines_from_ezdxf_doc(doc: Any) -> list[str]:
    """Harvest TEXT / MTEXT strings from an ezdxf Drawing."""

    if doc is None:
        return []

    try:
        msp = doc.modelspace()
    except Exception:
        return []

    lines: list[str] = []
    for entity in msp:
        try:
            kind = entity.dxftype()
        except Exception:
            continue
        if kind in ("TEXT", "MTEXT"):
            text = _extract_entity_text(entity)
            if text:
                lines.append(text)
        elif kind == "INSERT":
            try:
                for sub in entity.virtual_entities():
                    try:
                        sub_kind = sub.dxftype()
                    except Exception:
                        continue
                    if sub_kind in ("TEXT", "MTEXT"):
                        text = _extract_entity_text(sub)
                        if text:
                            lines.append(text)
            except Exception:
                continue

    joined = "\n".join(lines)
    if "HOLE TABLE" in joined.upper():
        upper = joined.upper()
        start = upper.index("HOLE TABLE")
        joined = joined[start:]
    return [ln for ln in joined.splitlines() if ln.strip()]


def hole_rows_to_ops(rows: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Flatten parsed HoleRow objects into estimator-friendly operations."""

    ops: list[dict[str, Any]] = []
    if not rows:
        return ops

    for row in rows:
        if row is None:
            continue
        try:
            features = list(getattr(row, "features", []) or [])
        except Exception:
            features = []
        try:
            qty_val = int(getattr(row, "qty", 0) or 0)
        except Exception:
            qty_val = 0
        ref_val = getattr(row, "ref", "")
        for feature in features:
            if not isinstance(feature, dict):
                continue
            op = dict(feature)
            op.setdefault("qty", qty_val or op.get("qty") or 0)
            op.setdefault("ref", ref_val)
            ops.append(op)
    return ops


def reconcile_holes(entity_holes_mm: Iterable[Any] | None, chart_ops: Iterable[dict] | None) -> dict[str, Any]:
    """Compare entity-detected hole sizes with chart-derived operations."""

    def _as_qty(value: Any) -> int:
        try:
            qty = int(round(float(value)))
        except Exception:
            qty = 0
        return qty if qty > 0 else 1

    bin_key = lambda dia: round(float(dia), 1)

    ent_bins: Counter[float] = Counter()
    if entity_holes_mm:
        for value in entity_holes_mm:
            try:
                dia = float(value)
            except Exception:
                continue
            if dia > 0:
                ent_bins[bin_key(dia)] += 1

    chart_bins: Counter[float] = Counter()
    tap_qty = 0
    cbore_qty = 0
    csk_qty = 0
    if chart_ops:
        for op in chart_ops:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type") or "").lower()
            qty = _as_qty(op.get("qty"))
            if op_type == "tap":
                tap_qty += qty
            if op_type == "cbore":
                cbore_qty += qty
            if op_type in {"csk", "countersink"}:
                csk_qty += qty
            if op_type != "drill":
                continue
            dia_raw = op.get("dia_mm")
            if dia_raw is None:
                continue
            try:
                dia_val = float(dia_raw)
            except Exception:
                continue
            if dia_val <= 0:
                continue
            chart_bins[bin_key(dia_val)] += qty

    entity_total = sum(ent_bins.values())
    chart_total = sum(chart_bins.values())
    max_total = max(entity_total, chart_total)
    tolerance = max(5, 0.1 * max_total) if max_total else 5
    agreement = abs(entity_total - chart_total) <= tolerance

    return {
        "entity_bins": {float(k): int(v) for k, v in ent_bins.items()},
        "chart_bins": {float(k): int(v) for k, v in chart_bins.items()},
        "tap_qty": int(tap_qty),
        "cbore_qty": int(cbore_qty),
        "csk_qty": int(csk_qty),
        "agreement": bool(agreement),
        "entity_total": int(entity_total),
        "chart_total": int(chart_total),
    }


def get_llm_quote_explanation(result: dict, model_path: str) -> str:
    """
    Returns a single-paragraph, friendly explanation of the main cost drivers.
    Works with the local LLMClient.ask_json(), and has a safe fallback if no model.
    """
    import json
    import os

    breakdown = result.get("breakdown", {}) or {}
    totals = breakdown.get("totals", {}) or {}
    process_costs = breakdown.get("process_costs", {}) or {}
    material_detail = breakdown.get("material", {}) or {}
    pass_meta = breakdown.get("pass_meta", {}) or {}

    geo = result.get("geo") or {}
    ui_vars = result.get("ui_vars") or {}

    final_price = float(result.get("price", totals.get("price", 0.0) or 0.0))
    labor_cost = float(totals.get("labor_cost", 0.0) or 0.0)
    direct_costs = float(totals.get("direct_costs", 0.0) or 0.0)
    subtotal = labor_cost + direct_costs

    def _to_float(value):
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return None
                return float(cleaned)
        except Exception:
            return None
        return None

    proc_pairs = []
    for key, value in (process_costs or {}).items():
        val = _to_float(value)
        if val is None:
            continue
        proc_pairs.append((str(key), val))
    proc_pairs.sort(key=lambda kv: kv[1], reverse=True)
    proc_top = proc_pairs[:3]

    def _to_int(value):
        try:
            if isinstance(value, bool):
                return int(value)
            return int(float(value))
        except Exception:
            return None

    hole_count = _to_int((geo or {}).get("hole_count"))

    thickness_in_val = _to_float(geo.get("thickness_in"))
    thickness_mm = _to_float(geo.get("thickness_mm"))
    if thickness_in_val is None and thickness_mm is not None:
        try:
            thickness_in_val = float(thickness_mm) / 25.4
        except Exception:
            thickness_in_val = None
    ui_thickness_in = _to_float(ui_vars.get("Thickness (in)"))
    if thickness_in_val is None and ui_thickness_in is not None:
        thickness_in_val = float(ui_thickness_in)
    thickness_in = round(float(thickness_in_val), 2) if thickness_in_val is not None else None

    material_name = geo.get("material") or ui_vars.get("Material")
    if isinstance(material_name, str):
        material_name = material_name.strip() or None

    material_display = material_name or "Steel"

    hole_count_val = hole_count if isinstance(hole_count, int) else None
    geo_notes_default: list[str] = []
    try:
        thickness_note_val = float(thickness_in_val) if thickness_in_val is not None else None
    except Exception:
        thickness_note_val = None
    if hole_count_val and thickness_note_val and material_display:
        geo_notes_default = [f"{hole_count_val} holes in {thickness_note_val:.2f} in {material_display}"]

    material_source = (
        result.get("material_source")
        or material_detail.get("source")
        or (pass_meta.get("Material", {}) or {}).get("basis")
        or "shop defaults"
    )

    effective_scrap = material_detail.get("scrap_pct")
    if effective_scrap is None:
        effective_scrap = breakdown.get("scrap_pct")
    effective_scrap = normalize_scrap_pct(effective_scrap)
    scrap_pct_percent = round(100.0 * effective_scrap, 1)

    ctx: dict[str, Any] = {
        "purpose": "quote_explanation",
        "rollup": {
            "final_price": final_price,
            "labor_cost": labor_cost,
            "direct_costs": direct_costs,
            "subtotal": subtotal,
            "labor_pct": round(100.0 * labor_cost / subtotal, 1) if subtotal else 0.0,
            "directs_pct": round(100.0 * direct_costs / subtotal, 1) if subtotal else 0.0,
            "top_processes": [
                {"name": name.replace("_", " "), "usd": val}
                for name, val in proc_top
            ],
        },
        "geo_summary": {
            "hole_count": hole_count,
            "thickness_in": thickness_in,
            "material": material_name,
        },
        "geo_notes": geo_notes_default,
        "material_source": material_source,
        "scrap_pct": scrap_pct_percent,
    }

    def _render_explanation(parsed: dict | None = None, raw_text: str = "") -> str:
        data = parsed if isinstance(parsed, dict) else {}
        if not data and raw_text:
            data = parse_llm_json(raw_text) or {}

        fallback_drivers = [
            {
                "label": "Labor",
                "usd": labor_cost,
                "pct_of_subtotal": round(100.0 * labor_cost / subtotal, 1) if subtotal else 0.0,
            },
            {
                "label": "Directs",
                "usd": direct_costs,
                "pct_of_subtotal": round(100.0 * direct_costs / subtotal, 1) if subtotal else 0.0,
            },
        ]

        drivers_raw = data.get("drivers") if isinstance(data.get("drivers"), list) else []
        drivers: list[dict[str, float | str]] = []
        for idx, default in enumerate(fallback_drivers):
            entry = drivers_raw[idx] if idx < len(drivers_raw) and isinstance(drivers_raw[idx], dict) else {}
            label = str(entry.get("label") or default["label"]).strip() or default["label"]
            usd = _to_float(entry.get("usd"))
            pct = _to_float(entry.get("pct_of_subtotal"))
            drivers.append(
                {
                    "label": label,
                    "usd": usd if usd is not None else float(default["usd"]),
                    "pct_of_subtotal": pct if pct is not None else float(default["pct_of_subtotal"]),
                }
            )
        while len(drivers) < 2:
            drivers.append({"label": f"Bucket {len(drivers) + 1}", "usd": 0.0, "pct_of_subtotal": 0.0})

        driver_primary = drivers[0] if drivers else {"label": "Labor", "usd": 0.0, "pct_of_subtotal": 0.0}
        driver_secondary = drivers[1] if len(drivers) > 1 else driver_primary

        geo_notes_raw = data.get("geo_notes") if isinstance(data.get("geo_notes"), list) else []
        geo_notes = [str(note).strip() for note in geo_notes_raw if str(note).strip()]
        if not geo_notes:
            default_notes = ctx.get("geo_notes") if isinstance(ctx.get("geo_notes"), list) else []
            if default_notes:
                geo_notes = [str(note).strip() for note in default_notes if str(note).strip()]
        if not geo_notes:
            hc = ctx.get("geo_summary", {}).get("hole_count")
            thk = ctx.get("geo_summary", {}).get("thickness_in")
            mat = ctx.get("geo_summary", {}).get("material")
            if hc and thk and mat:
                try:
                    geo_notes = [f"{int(hc)} holes in {float(thk):.2f} in {mat}"]
                except Exception:
                    geo_notes = []

        class _TopProcessEntry(TypedDict):
            name: str
            usd: float

        top_processes_raw = data.get("top_processes") if isinstance(data.get("top_processes"), list) else []
        top_processes: list[dict[str, float | str]] = []
        for entry in top_processes_raw:
            if not isinstance(entry, dict):
                continue
            name_val = str(entry.get("name") or "").strip()
            usd_val = _to_float(entry.get("usd"))
            if name_val and usd_val is not None:
                top_processes.append({"name": name_val, "usd": usd_val})
        if not top_processes:
            rollup = ctx.get("rollup")
            rollup_processes = rollup.get("top_processes") if isinstance(rollup, dict) else None
            if isinstance(rollup_processes, list):
                for proc in rollup_processes:
                    if not isinstance(proc, dict):
                        continue
                    name_val = str(proc.get("name") or "").strip()
                    usd_val = _to_float(proc.get("usd"))
                    if name_val and usd_val is not None:
                        top_processes.append({"name": name_val, "usd": usd_val})

        material_section = data.get("material") if isinstance(data.get("material"), dict) else {}
        scrap_val = _to_float(material_section.get("scrap_pct"))
        if scrap_val is None:
            scrap_fallback = ctx.get("scrap_pct")
            scrap_val = _to_float(scrap_fallback)
            if scrap_val is None and isinstance(scrap_fallback, (int, float)):
                scrap_val = float(scrap_fallback)
            if scrap_val is None:
                scrap_val = 0.0
        material_struct = {
            "source": str(material_section.get("source") or material_source).strip() or material_source,
            "scrap_pct": round(float(scrap_val), 1),
        }

        explanation = data.get("explanation") if isinstance(data.get("explanation"), str) else ""
        explanation = explanation.strip()

        if not explanation:
            top_text = ""
            if top_processes:
                top_bits = [
                    f"{proc['name']} ${proc['usd']:.0f}"
                    for proc in top_processes
                    if proc.get("name") and _to_float(proc.get("usd")) is not None
                ]
                if top_bits:
                    top_text = "Top processes: " + ", ".join(top_bits) + ". "

            geo_text = ""
            if geo_notes:
                geo_text = "Geometry: " + ", ".join(geo_notes) + ". "

            scrap_display = material_struct.get("scrap_pct")
            if scrap_display is None:
                scrap_display = ctx.get("scrap_pct")
            try:
                scrap_str = f"{float(scrap_display):.1f}"
            except Exception:
                scrap_str = str(scrap_display)

            material_text = ""
            if material_struct.get("source"):
                material_text = f"Material via {material_struct['source']}; scrap {scrap_str}% applied."
            else:
                material_text = f"Scrap {scrap_str}% applied."

            explanation = (
                f"Labor ${labor_cost:.2f} ({float(driver_primary['pct_of_subtotal']):.1f}%) and directs "
                f"${direct_costs:.2f} ({float(driver_secondary['pct_of_subtotal']):.1f}%) drive cost. "
                + top_text
                + geo_text
                + material_text
            ).strip()

        return explanation

    if not model_path or not os.path.isfile(model_path):
        return _render_explanation()

    system_prompt = (
        "You are a manufacturing estimator. Using ONLY the provided fields, produce a concise JSON explanation.\n"
        "Do not invent numbers. Mention the biggest cost buckets and key geometry drivers if present.\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "explanation": "…1–3 sentences…",\n'
        '  "drivers": [\n'
        '    {"label":"Labor","usd": <number>,"pct_of_subtotal": <number>},\n'
        '    {"label":"Directs","usd": <number>,"pct_of_subtotal": <number>}\n'
        "  ],\n"
        '  "top_processes": [{"name":"drilling","usd": <number>}],\n'
        '  "geo_notes": ["<e.g., 163 holes in 0.25 in steel>"],\n'
        '  "material": {"source":"<string>","scrap_pct": <number>}\n'
        "}"
    )

    user_prompt = "```json\n" + json.dumps(ctx, indent=2, sort_keys=True, default=str) + "\n```"

    client: LLMClient | None = None
    try:
        client = LLMClient(
            model_path,
            debug_enabled=APP_ENV.llm_debug_enabled,
            debug_dir=APP_ENV.llm_debug_dir,
        )
        parsed, raw_text, _usage = client.ask_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=256,
            context=ctx,
        )
        return _render_explanation(parsed, raw_text)
    except Exception:
        return _render_explanation()
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


# ==== LLM DECISION ENGINE =====================================================
def clamp(x, lo, hi, default=None):
    try:
        v = float(x)
    except Exception:
        return default if default is not None else lo
    return max(lo, min(hi, v))


def _safe_get(d, k, typ, default=None):
    try:
        v = d.get(k, default)
        return v if isinstance(v, typ) else default
    except Exception:
        return default


def get_llm_overrides(
    model_path: str,
    features: dict,
    base_costs: dict,
    *,
    context_payload: dict | None = None,
) -> tuple[dict, dict]:
    """
    Ask the local LLM for cost overrides based on CAD features and base costs.
    Returns (overrides_dict, meta) where overrides_dict may contain:
      {
        "scrap_pct_override": 0.00-0.25,          # fraction, not %
        "process_hour_multipliers": {"milling": 1.10, "turning": 0.95, ...},
        "process_hour_adders": {"milling": 0.25, "inspection": 0.10},   # hours
        "add_pass_through": {"Material": 12.0, "Tooling": 30.0},        # dollars
        "contingency_pct_override": 0.00-0.25,    # optional
        "notes": ["short human-readable bullets"]
      }
    and meta captures the raw response, usage stats, and clamp notes.
    """
    import json, os

    def _meta(raw=None, raw_text="", usage=None, clamp_notes=None):
        return {
            "raw": raw,
            "raw_text": raw_text or "",
            "usage": usage or {},
            "clamp_notes": clamp_notes or [],
        }

    def _fallback(meta=None):
        return {"notes": ["LLM disabled or unavailable; using base costs"]}, (meta or _meta())

    if not model_path or not os.path.isfile(model_path):
        return _fallback()

    try:
        llm = LLMClient(
            model_path,
            debug_enabled=APP_ENV.llm_debug_enabled,
            debug_dir=APP_ENV.llm_debug_dir,
        )
    except Exception:
        return _fallback()

    clamp_notes: list[str] = []
    out: dict[str, Any] = {}

    def _as_float(value):
        res = _coerce_float_or_none(value)
        return float(res) if res is not None else None

    def _as_int(value, default: int = 0) -> int:
        res = _as_float(value)
        if res is None:
            return default
        try:
            return int(round(res))
        except Exception:
            return default

    hole_count_feature = max(0, _as_int(features.get("hole_count"), 0))
    thickness_feature = _as_float(features.get("thickness_mm")) or 0.0
    density_feature = _as_float(features.get("density_g_cc")) or 0.0
    volume_feature = _as_float(features.get("volume_cm3")) or 0.0
    part_mass_est = _as_float(features.get("part_mass_g_est")) or 0.0
    if part_mass_est <= 0 and density_feature > 0 and volume_feature > 0:
        part_mass_est = density_feature * volume_feature
    density_for_stock = density_feature if density_feature > 0 else 7.85

    bbox_feature = features.get("bbox_mm") if isinstance(features.get("bbox_mm"), dict) else {}
    part_dims: list[float] = []
    for key in ("length_mm", "width_mm", "height_mm"):
        val = _as_float(bbox_feature.get(key))
        if val and val > 0:
            part_dims.append(val)
    if thickness_feature and thickness_feature > 0:
        part_dims.append(thickness_feature)
    part_dims_sorted = sorted([d for d in part_dims if d > 0], reverse=True)

    stock_catalog_raw = features.get("stock_catalog")
    stock_catalog = stock_catalog_raw if isinstance(stock_catalog_raw, (list, tuple)) else []
    catalog_dims_sorted: list[list[float]] = []
    for entry in stock_catalog:
        if not isinstance(entry, dict):
            continue
        dims = []
        for key in ("length_mm", "width_mm", "height_mm", "thickness_mm"):
            val = _as_float(entry.get(key))
            if val and val > 0:
                dims.append(val)
        if dims:
            dims = sorted(dims, reverse=True)
            catalog_dims_sorted.append(dims[:3])

    part_fits_catalog = True
    if part_dims_sorted and catalog_dims_sorted:
        part_fits_catalog = any(
            all(
                part_dims_sorted[i] <= dims[i] + 1e-6
                for i in range(min(len(part_dims_sorted), len(dims)))
            )
            for dims in catalog_dims_sorted
        )

    task_meta: dict[str, dict[str, Any]] = {}
    task_outputs: dict[str, dict[str, Any]] = {}
    combined_usage: dict[str, float] = {}

    def _merge_usage(usage: dict | None):
        if not isinstance(usage, dict):
            return
        for key, value in usage.items():
            try:
                combined_usage[key] = combined_usage.get(key, 0.0) + float(value)
            except Exception:
                continue

    ctx = copy.deepcopy(context_payload or {})
    ctx.setdefault("geo", {})
    ctx.setdefault("quote_vars", {})
    catalogs_ctx = dict(ctx.get("catalogs") or {})
    catalogs_ctx["stock"] = stock_catalog
    ctx["catalogs"] = catalogs_ctx
    bounds_ctx = dict(ctx.get("bounds") or {})
    bounds_ctx.update(
        {
            "mult_min": LLM_MULTIPLIER_MIN,
            "mult_max": LLM_MULTIPLIER_MAX,
            "add_hr_min": 0.0,
            "add_hr_max": LLM_ADDER_MAX,
            "scrap_min": 0.0,
            "scrap_max": 0.25,
        }
    )
    ctx["bounds"] = bounds_ctx
    baseline_ctx = dict(ctx.get("baseline") or {})
    baseline_ctx.setdefault("scrap_pct", float(features.get("scrap_pct") or 0.0))
    baseline_ctx.setdefault("pass_through", base_costs.get("pass_through", {}))
    ctx["baseline"] = baseline_ctx
    ctx.setdefault("rates", base_costs.get("rates", {}))

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        try:
            return float(obj)
        except Exception:
            return str(obj)

    def _run_task(name: str, system_prompt: str, payload: dict, *, temperature: float = 0.2, max_tokens: int = 256):
        entry = {"system_prompt": system_prompt, "payload": payload}
        task_meta[name] = entry
        try:
            try:
                prompt_body = json.dumps(payload, indent=2, default=str)
            except TypeError:
                prompt_body = json.dumps(_jsonify(payload), indent=2)
            prompt = "```json\n" + prompt_body + "\n```"
            parsed, raw_text, usage = llm.ask_json(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                context=ctx,
            )
            entry["raw"] = parsed
            entry["raw_text"] = raw_text or ""
            entry["usage"] = usage or {}
            _merge_usage(usage)
            if not isinstance(parsed, dict):
                parsed = parse_llm_json(raw_text)
            if isinstance(parsed, dict):
                task_outputs[name] = parsed
                return parsed
            return {}
        except Exception as exc:
            entry["error"] = repr(exc)
            return {}

    # ---- Capability prompts -------------------------------------------------
    hole_payload = {
        "hole_count": hole_count_feature,
        "hole_groups": features.get("hole_groups"),
        "hole_diams_mm": features.get("hole_tool_sizes") or features.get("hole_diams_mm"),
        "thickness_mm": thickness_feature,
        "material": features.get("material_key"),
        "baseline_drilling_hr": _as_float(features.get("drilling_hr_baseline")),
        "machine_limits": features.get("machine_limits"),
    }
    if hole_count_feature >= 1:
        drilling_system = (
            "You are a manufacturing estimator. Cluster similar holes by tool diameter/depth,"
            " suggest a pecking strategy, and recommend bounded drilling time tweaks."
            " Return JSON only with optional keys: {\"drilling_groups\":[{...}],"
            " \"process_hour_multipliers\":{\"drilling\":float},"
            " \"process_hour_adders\":{\"drilling\":float}, \"notes\":[""...""]}."
            " If hole_count < 5 or data is insufficient, return {}."
            " Respect multipliers in [0.25,4.0] and adders in [0,8] hours."
            " When hole_count ≥ 50 and thickness_mm ≥ 3, consider multipliers up to 2.0."
        )
        _run_task("drilling", drilling_system, hole_payload, max_tokens=384)

    stock_payload = {
        "bbox_mm": bbox_feature,
        "part_mass_g_est": part_mass_est,
        "material": features.get("material_key"),
        "scrap_pct_baseline": float(features.get("scrap_pct") or 0.0),
    }
    if stock_catalog and bbox_feature:
        stock_system = (
            "You are a manufacturing estimator. Choose a stock item from `catalogs.stock` that"
            " minimally encloses bbox LxWxT (mm). Return JSON only: {\"stock_recommendation\":{...},"
            " \"scrap_pct\":0.14, \"process_hour_adders\":{\"sawing\":0.2,\"handling\":0.1},"
            " \"notes\":[""...""]}. Keep scrap_pct within [0,0.25] and hour adders within [0,8]."
            " Do not invent SKUs. If none fit, return {\"needs_user_input\":\"no stock fits\"}."
        )
        _run_task("stock", stock_system, stock_payload, max_tokens=384)

    setup_payload = {
        "baseline_setups": _as_int(features.get("setups"), 0),
        "unique_normals": features.get("dfm_geo", {}).get("unique_normals"),
        "face_count": features.get("dfm_geo", {}).get("face_count"),
        "fixture_plan": features.get("fixture_plan"),
        "qty": features.get("qty"),
    }
    setup_system = (
        "You are a manufacturing estimator. Suggest the number of milling setups and fixture"
        " approach for the feature summary. Return JSON only: {\"setups\":int,"
        " \"fixture\":\"...\", \"setup_adders_hr\":0.0, \"notes\":[""...""]}."
        " Do not exceed 4 setups without explicit approval; if unsure, return {}."
    )
    _run_task("setups", setup_system, setup_payload, max_tokens=256)

    dfm_payload = {
        "dfm_geo": features.get("dfm_geo"),
        "material": features.get("material_key"),
        "tall_features": features.get("tall_features"),
        "hole_groups": features.get("hole_groups"),
    }
    dfm_system = (
        "You are a manufacturing estimator. Flag genuine DFM risks (thin walls, deep pockets,"
        " tiny radii, thread density) only when thresholds you define are exceeded."
        " Return JSON only: {\"dfm_risks\":[""thin walls <2mm""],"
        " \"process_hour_multipliers\":{...}, \"process_hour_adders\":{...}, \"notes\":[...]}"
        " or {} if no issues. Keep multipliers within [0.25,4.0] and adders within [0,8]."
    )
    _run_task("dfm", dfm_system, dfm_payload, max_tokens=320)

    baseline_proc_hours = ctx.get("baseline", {}).get("process_hours") if isinstance(ctx.get("baseline"), dict) else {}
    if not isinstance(baseline_proc_hours, dict):
        baseline_proc_hours = {}
    tol_payload = {
        "tolerance_inputs": features.get("tolerance_inputs"),
        "baseline_inspection_hr": {
            "in_process": _as_float(baseline_proc_hours.get("inspection")),
            "final": _as_float(baseline_proc_hours.get("final_inspection")),
        },
    }
    if tol_payload["tolerance_inputs"]:
        tol_system = (
            "You are a manufacturing estimator. When tolerances/finishes are tight, add bounded"
            " inspection or finishing time and suggest surface-finish ops. Return JSON only:"
            " {\"tolerance_impacts\":{\"in_process_inspection_hr\":0.2,"
            " \"final_inspection_hr\":0.1, \"finishing_hr\":0.1, \"suggested_surface_finish\":\"...\","
            " \"notes\":[...]}}. Keep hour adders within [0,8] and return {} if no change is needed."
        )
        _run_task("tolerance", tol_system, tol_payload, max_tokens=320)

    raw_text_combined = "\n\n".join(
        f"{name}: {meta.get('raw_text', '').strip()}".strip()
        for name, meta in task_meta.items()
        if meta.get("raw_text")
    ).strip()
    raw_by_task = {name: meta.get("raw") for name, meta in task_meta.items() if "raw" in meta}

    merged: dict[str, Any] = {}

    def _merge_result(dest: dict, src: dict | None):
        if not isinstance(src, dict):
            return
        for key, value in src.items():
            if key in {"notes", "dfm_risks", "risks"} and isinstance(value, list):
                dest.setdefault(key, [])
                for item in value:
                    if isinstance(item, str):
                        dest[key].append(item)
            elif key in {"process_hour_multipliers", "process_hour_adders", "add_pass_through"} and isinstance(value, dict):
                dest.setdefault(key, {})
                dest[key].update(value)
            elif key == "scrap_pct":
                dest.setdefault("scrap_pct_override", value)
            elif key == "drilling_groups" and isinstance(value, list):
                dest.setdefault(key, [])
                for grp in value:
                    if isinstance(grp, dict):
                        dest[key].append(grp)
            else:
                dest[key] = value

    for data in task_outputs.values():
        _merge_result(merged, data)

    parsed = merged
    needs_input_msg = parsed.get("needs_user_input") if isinstance(parsed, dict) else None
    if needs_input_msg:
        clamp_notes.append(f"needs_user_input: {needs_input_msg}")

    clean_mults: dict[str, float] = {}
    clean_adders: dict[str, float] = {}

    def _ensure_mults_dict() -> dict[str, float]:
        if "process_hour_multipliers" in out:
            return out["process_hour_multipliers"]
        out["process_hour_multipliers"] = clean_mults
        return clean_mults

    def _ensure_adders_dict() -> dict[str, float]:
        if "process_hour_adders" in out:
            return out["process_hour_adders"]
        out["process_hour_adders"] = clean_adders
        return clean_adders

    def _merge_multiplier(name: str, value, source: str) -> None:
        val = _as_float(value)
        if val is None:
            return
        clamped = clamp(val, LLM_MULTIPLIER_MIN, LLM_MULTIPLIER_MAX, 1.0)
        container = _ensure_mults_dict()
        norm = str(name).lower()
        prev = container.get(norm)
        if prev is None:
            container[norm] = clamped
            return
        new_val = clamp(prev * clamped, LLM_MULTIPLIER_MIN, LLM_MULTIPLIER_MAX, 1.0)
        if not math.isclose(prev * clamped, new_val, abs_tol=1e-6):
            clamp_notes.append(f"{source} multiplier clipped for {norm}")
        container[norm] = new_val

    def _merge_adder(name: str, value, source: str) -> None:
        val = _as_float(value)
        if val is None:
            return
        clamped = clamp(val, 0.0, LLM_ADDER_MAX, 0.0)
        if clamped <= 0:
            return
        container = _ensure_adders_dict()
        norm = str(name).lower()
        prev = float(container.get(norm, 0.0))
        new_val = clamp(prev + clamped, 0.0, LLM_ADDER_MAX, 0.0)
        if not math.isclose(prev + clamped, new_val, abs_tol=1e-6):
            clamp_notes.append(
                f"{source} {prev + clamped:.2f} hr clipped to {LLM_ADDER_MAX:.1f} for {norm}"
            )
        container[norm] = new_val

    def _clean_notes_list(values, limit: int = 6) -> list[str]:
        clean: list[str] = []
        if not isinstance(values, list):
            return clean
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            clean.append(text[:200])
            if len(clean) >= limit:
                break
        return clean

    scr = parsed.get("scrap_pct_override", None)
    if scr is not None:
        try:
            orig = float(scr)
        except Exception:
            orig = None
        clamped_scrap = clamp(scr, 0.0, 0.25, None)
        if clamped_scrap is not None:
            out["scrap_pct_override"] = clamped_scrap
            if orig is None:
                clamp_notes.append("scrap_pct_override non-numeric → default applied")
            elif not math.isclose(orig, clamped_scrap, abs_tol=1e-6):
                clamp_notes.append(
                    f"scrap_pct_override {orig:.3f} → {clamped_scrap:.3f}"
                )

    mults = _safe_get(parsed, "process_hour_multipliers", dict, {})
    for k, v in (mults or {}).items():
        if isinstance(v, (int, float)):
            orig = float(v)
            clamped_val = clamp(v, LLM_MULTIPLIER_MIN, LLM_MULTIPLIER_MAX, 1.0)
            clean_mults[k.lower()] = clamped_val
            if not math.isclose(orig, clamped_val, abs_tol=1e-6):
                clamp_notes.append(
                    f"process_hour_multipliers[{k}] {orig:.2f} → {clamped_val:.2f}"
                )
        else:
            clamp_notes.append(f"process_hour_multipliers[{k}] non-numeric")

    adds = _safe_get(parsed, "process_hour_adders", dict, {})
    for k, v in (adds or {}).items():
        if isinstance(v, (int, float)):
            orig = float(v)
            clamped_val = clamp(v, 0.0, LLM_ADDER_MAX, 0.0)
            clean_adders[k.lower()] = clamped_val
            if not math.isclose(orig, clamped_val, abs_tol=1e-6):
                clamp_notes.append(
                    f"process_hour_adders[{k}] {orig:.2f} → {clamped_val:.2f}"
                )
        else:
            clamp_notes.append(f"process_hour_adders[{k}] non-numeric")

    addpt = _safe_get(parsed, "add_pass_through", dict, {})
    clean_pass: dict[str, float] = {}
    for k, v in (addpt or {}).items():
        if isinstance(v, (int, float)):
            orig = float(v)
            clamped_val = clamp(v, 0.0, 200.0, 0.0)
            canon_key = _canonical_pass_label(k)
            if not canon_key:
                continue
            clean_pass[canon_key] = clamped_val
            if not math.isclose(orig, clamped_val, abs_tol=1e-6):
                clamp_notes.append(
                    f"add_pass_through[{k}] {orig:.2f} → {clamped_val:.2f}"
                )
        else:
            clamp_notes.append(f"add_pass_through[{k}] non-numeric")
    if clean_pass:
        out["add_pass_through"] = clean_pass

    cont = parsed.get("contingency_pct_override", None)
    if cont is not None:
        try:
            orig = float(cont)
        except Exception:
            orig = None
        clamped_val = clamp(cont, 0.0, 0.25, None)
        if clamped_val is not None:
            out["contingency_pct_override"] = clamped_val
            if orig is None:
                clamp_notes.append("contingency_pct_override non-numeric → default applied")
            elif not math.isclose(orig, clamped_val, abs_tol=1e-6):
                clamp_notes.append(
                    f"contingency_pct_override {orig:.3f} → {clamped_val:.3f}"
                )

    drill_groups_raw = _safe_get(parsed, "drilling_groups", list, [])
    if drill_groups_raw:
        if hole_count_feature < 5:
            clamp_notes.append("ignored drilling_groups; hole_count < 5")
        else:
            drill_groups_clean: list[dict[str, Any]] = []
            for grp in drill_groups_raw:
                if not isinstance(grp, dict):
                    continue
                dia = _as_float(grp.get("dia_mm") or grp.get("diameter_mm"))
                qty = _as_int(grp.get("qty") or grp.get("count"), 0)
                depth = _as_float(grp.get("depth_mm") or grp.get("depth"))
                peck = grp.get("peck") or grp.get("strategy")
                notes = grp.get("notes")
                if dia is None or qty <= 0:
                    continue
                qty = max(1, min(hole_count_feature, qty))
                cleaned_group: dict[str, Any] = {
                    "dia_mm": round(dia, 3),
                    "qty": qty,
                }
                if depth is not None and depth > 0:
                    cleaned_group["depth_mm"] = round(depth, 2)
                if isinstance(peck, str) and peck.strip():
                    cleaned_group["peck"] = peck.strip()[:40]
                group_notes = _clean_notes_list(notes, limit=3)
                if group_notes:
                    cleaned_group["notes"] = group_notes
                drill_groups_clean.append(cleaned_group)
            if drill_groups_clean:
                out["drilling_groups"] = drill_groups_clean

    stock_plan_raw = (
        parsed.get("stock_recommendation")
        or parsed.get("stock_plan")
        or parsed.get("stock")
    )
    if isinstance(stock_plan_raw, dict):
        length = _as_float(stock_plan_raw.get("length_mm"))
        width = _as_float(stock_plan_raw.get("width_mm"))
        thickness = _as_float(stock_plan_raw.get("thickness_mm"))
        dims_field = stock_plan_raw.get("size_mm") or stock_plan_raw.get("dimensions_mm")
        if isinstance(dims_field, dict):
            length = length or _as_float(dims_field.get("length")) or _as_float(dims_field.get("length_mm"))
            width = width or _as_float(dims_field.get("width")) or _as_float(dims_field.get("width_mm"))
            thickness = thickness or _as_float(dims_field.get("thickness")) or _as_float(dims_field.get("height")) or _as_float(dims_field.get("thickness_mm"))
        elif isinstance(dims_field, (list, tuple)):
            dims_nums = [d for d in (_as_float(x) for x in dims_field) if d and d > 0]
            if len(dims_nums) >= 1 and length is None:
                length = dims_nums[0]
            if len(dims_nums) >= 2 and width is None:
                width = dims_nums[1]
            if len(dims_nums) >= 3 and thickness is None:
                thickness = dims_nums[2]

        dims_plan = [d for d in (length, width, thickness) if d and d > 0]
        dims_plan_sorted = sorted(dims_plan, reverse=True)
        fits_part = True
        if part_dims_sorted and dims_plan_sorted:
            fits_part = all(
                part_dims_sorted[i] <= dims_plan_sorted[i] + 1e-6
                for i in range(min(len(part_dims_sorted), len(dims_plan_sorted)))
            )
        mass_ratio_ok = True
        stock_mass_g = None
        if length and width and thickness and density_for_stock > 0 and part_mass_est > 0:
            stock_volume_cm3 = (length * width * thickness) / 1000.0
            stock_mass_g = stock_volume_cm3 * density_for_stock
            if stock_mass_g > 3.0 * part_mass_est + 1e-6:
                mass_ratio_ok = False

        if not fits_part:
            clamp_notes.append("stock_recommendation ignored: stock smaller than part bbox")
        elif not part_fits_catalog and catalog_dims_sorted:
            clamp_notes.append("stock_recommendation ignored: part exceeds stock catalog")
        elif not mass_ratio_ok:
            clamp_notes.append("stock_recommendation ignored: stock mass >3× part mass")
        elif not (length and width and thickness):
            clamp_notes.append("stock_recommendation ignored: missing stock dimensions")
        else:
            clean_stock_plan: dict[str, Any] = {}
            label = stock_plan_raw.get("stock_item") or stock_plan_raw.get("item")
            if label:
                clean_stock_plan["stock_item"] = str(label)
            material_label = stock_plan_raw.get("material")
            if material_label:
                clean_stock_plan["material"] = str(material_label)
            form_label = stock_plan_raw.get("form") or stock_plan_raw.get("shape")
            if form_label:
                clean_stock_plan["form"] = str(form_label)
            clean_stock_plan["length_mm"] = round(float(length), 3)
            clean_stock_plan["width_mm"] = round(float(width), 3)
            clean_stock_plan["thickness_mm"] = round(float(thickness), 3)
            clean_stock_plan["count"] = max(1, _as_int(stock_plan_raw.get("count") or stock_plan_raw.get("quantity"), 1))
            clean_stock_plan["cut_count"] = max(0, _as_int(stock_plan_raw.get("cut_count") or stock_plan_raw.get("cuts"), 0))
            if stock_mass_g is not None:
                clean_stock_plan["stock_mass_g_est"] = round(stock_mass_g, 1)

            scrap_val = stock_plan_raw.get("scrap_pct")
            if scrap_val is None:
                scrap_val = stock_plan_raw.get("scrap_fraction")
            scrap_frac = None
            if scrap_val is not None:
                frac = pct(scrap_val, None)
                if frac is None:
                    frac = _as_float(scrap_val)
                if frac is not None:
                    scrap_frac = clamp(frac, 0.0, 0.25, None)
            if scrap_frac is not None:
                clean_stock_plan["scrap_pct"] = scrap_frac
                if "scrap_pct_override" not in out:
                    out["scrap_pct_override"] = scrap_frac

            plan_notes = _clean_notes_list(stock_plan_raw.get("notes"))
            if plan_notes:
                clean_stock_plan["notes"] = plan_notes

            saw_hr = _as_float(stock_plan_raw.get("sawing_hr") or stock_plan_raw.get("saw_hr"))
            if saw_hr and saw_hr > 0:
                saw_hr_clamped = clamp(saw_hr, 0.0, LLM_ADDER_MAX, 0.0)
                clean_stock_plan["sawing_hr"] = saw_hr_clamped
                _merge_adder("saw_waterjet", saw_hr_clamped, "stock_plan.sawing_hr")
            handling_hr = _as_float(stock_plan_raw.get("handling_hr"))
            if handling_hr and handling_hr > 0:
                handling_hr_clamped = clamp(handling_hr, 0.0, LLM_ADDER_MAX, 0.0)
                clean_stock_plan["handling_hr"] = handling_hr_clamped
                _merge_adder("assembly", handling_hr_clamped, "stock_plan.handling_hr")

            plan_adders = _safe_get(stock_plan_raw, "process_hour_adders", dict, {})
            for key, val in (plan_adders or {}).items():
                _merge_adder(key, val, "stock_plan.process_hour_adders")
            plan_mults = _safe_get(stock_plan_raw, "process_hour_multipliers", dict, {})
            for key, val in (plan_mults or {}).items():
                _merge_multiplier(key, val, "stock_plan.process_hour_multipliers")

            out["stock_recommendation"] = clean_stock_plan

    setup_plan_raw = parsed.get("setup_recommendation") or parsed.get("setup_plan")
    if isinstance(setup_plan_raw, dict):
        clean_setup: dict[str, Any] = {}
        setups_val = _as_int(setup_plan_raw.get("setups") or setup_plan_raw.get("count"), 0)
        if setups_val > 0:
            if setups_val > 4:
                clamp_notes.append(f"setup_recommendation setups {setups_val} → 4")
                setups_val = 4
            clean_setup["setups"] = setups_val
        fixture = setup_plan_raw.get("fixture") or setup_plan_raw.get("fixture_type")
        if isinstance(fixture, str) and fixture.strip():
            clean_setup["fixture"] = fixture.strip()[:120]
        setup_hr = _as_float(setup_plan_raw.get("setup_adders_hr") or setup_plan_raw.get("setup_hours"))
        if setup_hr and setup_hr > 0:
            clean_setup["setup_adders_hr"] = clamp(setup_hr, 0.0, LLM_ADDER_MAX, 0.0)
        setup_notes = _clean_notes_list(setup_plan_raw.get("notes"))
        if setup_notes:
            clean_setup["notes"] = setup_notes
        if clean_setup:
            out["setup_recommendation"] = clean_setup

    risks_raw = parsed.get("dfm_risks") or parsed.get("risks")
    risk_notes = _clean_notes_list(risks_raw, limit=8)
    if risk_notes:
        out["dfm_risks"] = risk_notes

    tol_raw = parsed.get("tolerance_impacts")
    if isinstance(tol_raw, dict):
        clean_tol: dict[str, Any] = {}
        inproc_hr = _as_float(tol_raw.get("in_process_inspection_hr") or tol_raw.get("in_process_hr"))
        if inproc_hr and inproc_hr > 0:
            inproc_clamped = clamp(inproc_hr, 0.0, LLM_ADDER_MAX, 0.0)
            clean_tol["in_process_inspection_hr"] = inproc_clamped
            _merge_adder("inspection", inproc_clamped, "tolerance_in_process_hr")
        final_hr = _as_float(tol_raw.get("final_inspection_hr") or tol_raw.get("final_hr"))
        if final_hr and final_hr > 0:
            final_clamped = clamp(final_hr, 0.0, LLM_ADDER_MAX, 0.0)
            clean_tol["final_inspection_hr"] = final_clamped
            _merge_adder("inspection", final_clamped, "tolerance_final_hr")
        finish_hr = _as_float(tol_raw.get("finishing_hr") or tol_raw.get("finish_hr"))
        if finish_hr and finish_hr > 0:
            finish_clamped = clamp(finish_hr, 0.0, LLM_ADDER_MAX, 0.0)
            clean_tol["finishing_hr"] = finish_clamped
            _merge_adder("finishing_deburr", finish_clamped, "tolerance_finishing_hr")
        surface = tol_raw.get("surface_finish") or tol_raw.get("suggested_surface_finish")
        if isinstance(surface, str) and surface.strip():
            clean_tol["suggested_finish"] = surface.strip()[:160]
        tol_notes = _clean_notes_list(tol_raw.get("notes"))
        if tol_notes:
            clean_tol["notes"] = tol_notes
        if clean_tol:
            out["tolerance_impacts"] = clean_tol

    if clean_mults:
        out["process_hour_multipliers"] = clean_mults
    elif "process_hour_multipliers" in out:
        out.pop("process_hour_multipliers", None)

    if clean_adders:
        out["process_hour_adders"] = clean_adders
    elif "process_hour_adders" in out:
        out.pop("process_hour_adders", None)

    notes = _safe_get(parsed, "notes", list, [])
    out["notes"] = [str(n)[:200] for n in notes][:6]

    meta = _meta(raw=raw_by_task, raw_text=raw_text_combined, usage=combined_usage, clamp_notes=clamp_notes)
    meta["tasks"] = task_meta
    meta["task_outputs"] = task_outputs
    meta["context"] = ctx
    try:
        llm.close()
    except Exception:
        pass
    return out, meta
# ----------------- GUI -----------------
# ---- service containers ----------------------------------------------------


@dataclass(slots=True)
class UIConfiguration:
    """Aggregate user-interface defaults for the desktop application."""

    title: str = "Compos-AI"
    window_geometry: str = "1260x900"
    llm_enabled_default: bool = True
    apply_llm_adjustments_default: bool = True
    settings_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("app_settings.json")
    )
    default_llm_model_path: str | None = None
    default_params: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(PARAMS_DEFAULT))
    default_material_display: str = DEFAULT_MATERIAL_DISPLAY

    def create_params(self) -> dict[str, Any]:
        return copy.deepcopy(self.default_params)


class GeometryLoader:
    """Facade around geometry helper functions used by the UI layer."""

    def extract_pdf_all(self, path: str | Path, dpi: int = 300) -> dict:
        return extract_pdf_all(Path(path), dpi=dpi)

    def extract_2d_features_from_pdf_vector(self, path: str | Path) -> dict:
        return extract_2d_features_from_pdf_vector(str(path))

    def extract_2d_features_from_dxf_or_dwg(self, path: str | Path) -> dict:
        return extract_2d_features_from_dxf_or_dwg(str(path))

    def extract_features_with_occ(self, path: str | Path):
        return geometry.extract_features_with_occ(str(path))

    def enrich_geo_stl(self, path: str | Path):
        return geometry.enrich_geo_stl(str(path))

    def read_step_shape(self, path: str | Path) -> Any:
        return geometry.read_step_shape(str(path))

    def read_cad_any(self, path: str | Path) -> Any:
        return geometry.read_cad_any(str(path))

    def safe_bbox(self, shape: Any):
        return geometry.safe_bbox(shape)

    def enrich_geo_occ(self, shape: Any):
        return geometry.enrich_geo_occ(shape)


@dataclass(slots=True)
class PricingRegistry:
    """Provide mutable copies of editable pricing defaults to the UI."""

    default_params: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(PARAMS_DEFAULT))
    default_rates: dict[str, float] = field(default_factory=lambda: copy.deepcopy(RATES_DEFAULT))

    def create_params(self) -> dict[str, Any]:
        return copy.deepcopy(self.default_params)

    def create_rates(self) -> dict[str, float]:
        return copy.deepcopy(self.default_rates)


@dataclass(slots=True)
class LLMServices:
    """Helper hooks for locating default models and loading vision LLMs."""

    default_model_locator: Callable[[], str] = find_default_qwen_model
    vision_loader: Callable[..., Any] = load_qwen_vl

    def default_model_path(self) -> str:
        return self.default_model_locator() or ""

    def load_vision_model(
        self,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = 20,
        n_threads: int | None = None,
    ):
        return self.vision_loader(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads)


# ---- GEO defaults helpers ---------------------------------------------------

def _parse_numeric_text(value: str) -> float | None:
    if not isinstance(value, str):
        return _coerce_float_or_none(value)
    text = value.strip()
    if not text:
        return None
    cleaned = re.sub(r"[^0-9./\s-]", " ", text)
    parts = [part.strip() for part in cleaned.split() if part.strip()]
    if not parts:
        return None
    total = 0.0
    for part in parts:
        try:
            total += float(Fraction(part))
            continue
        except Exception:
            pass
        try:
            total += float(part)
        except Exception:
            return None
    return total


def _parse_length_to_mm(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    unit = None
    for suffix in ("millimeters", "millimetres", "millimeter", "millimetre", "mm"):
        if lower.endswith(suffix):
            unit = "mm"
            text = text[: -len(suffix)]
            break
    if unit is None:
        for suffix in ("inches", "inch", "in", "\""):
            if lower.endswith(suffix):
                unit = "in"
                text = text[: -len(suffix)]
                break
    if unit is None and "\"" in text:
        unit = "in"
        text = text.replace("\"", "")
    if unit is None and "mm" in lower:
        unit = "mm"
        text = re.sub(r"mm", "", text, flags=re.IGNORECASE)
    numeric_val = _parse_numeric_text(text)
    if numeric_val is None:
        return None
    if unit == "in":
        return float(numeric_val) * 25.4
    return float(numeric_val)


def infer_geo_override_defaults(geo_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(geo_data, dict):
        return {}

    sources: list[dict[str, Any]] = []
    seen: set[int] = set()

    def collect(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
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
                        val = float(Fraction(str(entry)))
                    except Exception:
                        val = None
                if val is None or val <= 0:
                    continue
                hole_sum += float(val) * 25.4
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

    hole_count_val = _coerce_float_or_none(find_value("hole_count", "hole_count_geom"))
    if hole_count_val is not None and hole_count_val > 0:
        overrides["Hole Count (override)"] = int(round(hole_count_val))
    elif hole_total:
        overrides["Hole Count (override)"] = int(hole_total)

    if hole_total and hole_sum > 0:
        overrides["Avg Hole Diameter (mm)"] = hole_sum / float(hole_total)

    return overrides


# ---- tk tooltip helper -----------------------------------------------------


class CreateToolTip:
    """Attach a lightweight tooltip to a Tk widget."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        *,
        delay: int = 500,
        wraplength: int = 320,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id: str | None = None
        self._tip_window: tk.Toplevel | None = None
        self._label: tk.Label | ttk.Label | None = None
        self._pinned = False

        self.widget.bind("<Enter>", self._schedule_show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<FocusIn>", self._schedule_show, add="+")
        self.widget.bind("<FocusOut>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._on_button_press, add="+")
        self.widget.bind("<Button-1>", self._toggle_pin, add="+")

    def update_text(self, text: str) -> None:
        self.text = text
        if self._label is not None:
            self._label.configure(text=text)

    def _on_button_press(self, event: tk.Event | None = None) -> None:
        if event is not None and getattr(event, "num", None) == 1:
            return
        self._hide()

    def _toggle_pin(self, _event: tk.Event | None = None) -> None:
        if self._pinned:
            self._pinned = False
            self._hide()
        else:
            self._pinned = True
            self._cancel_scheduled()
            self._show()

    def _schedule_show(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled()
        if not self.text:
            return
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel_scheduled(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            finally:
                self._after_id = None

    def _show(self) -> None:
        if self._tip_window is not None or not self.text:
            return

        bbox = None
        try:
            bbox = self.widget.bbox("insert")
        except Exception:
            bbox = None
        if bbox:
            x, y, width, height = bbox
        else:
            x = y = 0
            width = self.widget.winfo_width()
            height = self.widget.winfo_height()

        root_x = self.widget.winfo_rootx()
        root_y = self.widget.winfo_rooty()
        x = root_x + x + width + 12
        y = root_y + y + height + 12

        master = self.widget.winfo_toplevel()
        tip = tk.Toplevel(master)
        tip.wm_overrideredirect(True)
        try:
            tip.transient(master)
        except Exception:
            tip.wm_transient(master)
        tip.wm_geometry(f"+{x}+{y}")
        tip.lift()
        try:
            tip.attributes("-topmost", True)
        except Exception:
            pass

        label = tk.Label(
            tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", 8, "normal"),
            wraplength=self.wraplength,
        )
        label.pack(ipadx=4, ipady=2)

        self._tip_window = tip
        self._label = label

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled()
        if self._pinned:
            return
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None
        self._label = None


# ---- scrollable frame helper -----------------------------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # Bind wheel only while cursor is over this widget
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    # Windows & macOS wheel
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-int(event.delta/120), "units")

    # Linux wheel
    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>",  self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>",  self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


class App(tk.Tk):
    def __init__(
        self,
        pricing: PricingEngine | None = None,
        *,
        configuration: UIConfiguration | None = None,
        geometry_loader: GeometryLoader | None = None,
        pricing_registry: PricingRegistry | None = None,
        llm_services: LLMServices | None = None,
        geometry_service: geometry.GeometryService | None = None,
    ):

        super().__init__()

        self.configuration = configuration or UIConfiguration()
        self.geometry_loader = geometry_loader or GeometryLoader()
        self.pricing_registry = pricing_registry or PricingRegistry()
        self.llm_services = llm_services or LLMServices()
        self.pricing: PricingEngine = pricing or _DEFAULT_PRICING_ENGINE

        default_material_display = getattr(
            self.configuration,
            "default_material_display",
            DEFAULT_MATERIAL_DISPLAY,
        )
        if not isinstance(default_material_display, str) or not default_material_display.strip():
            default_material_display = DEFAULT_MATERIAL_DISPLAY
        self.default_material_display = default_material_display

        if getattr(self.configuration, "title", None):
            self.title(self.configuration.title)
        if getattr(self.configuration, "window_geometry", None):
            self.geometry(self.configuration.window_geometry)


        self.geometry_service = geometry_service or geometry.GeometryService()

        self.vars_df = None
        self.vars_df_full = None
        self.geo = None
        self.geo_context: dict[str, Any] = {}
        if hasattr(self.configuration, "create_params"):
            try:
                params = self.configuration.create_params()
            except Exception:
                params = copy.deepcopy(PARAMS_DEFAULT)
            if not isinstance(params, dict):
                params = copy.deepcopy(PARAMS_DEFAULT)
            self.params = params
            self.default_params_template = copy.deepcopy(params)
        else:
            base_params = getattr(self.configuration, "default_params", PARAMS_DEFAULT)
            template_params = base_params if isinstance(base_params, dict) else PARAMS_DEFAULT
            self.default_params_template = copy.deepcopy(template_params)
            self.params = copy.deepcopy(self.default_params_template)

        if hasattr(self.pricing_registry, "create_rates"):
            try:
                rates = self.pricing_registry.create_rates()
            except Exception:
                rates = copy.deepcopy(RATES_DEFAULT)
            if not isinstance(rates, dict):
                rates = copy.deepcopy(RATES_DEFAULT)
            self.rates = rates
            self.default_rates_template = copy.deepcopy(rates)
        else:
            base_rates = getattr(self.pricing_registry, "default_rates", RATES_DEFAULT)
            template_rates = base_rates if isinstance(base_rates, dict) else RATES_DEFAULT
            self.default_rates_template = copy.deepcopy(template_rates)
            self.rates = copy.deepcopy(self.default_rates_template)
        self.config_errors = list(CONFIG_INIT_ERRORS)

        self.quote_state = QuoteState()
        self.llm_events: list[dict[str, Any]] = []
        self.llm_errors: list[dict[str, Any]] = []
        self._llm_client_cache: LLMClient | None = None
        self.settings_path = Path(__file__).with_name("app_settings.json")

        self.settings = self._load_settings()
        if not isinstance(self.settings, dict):
            self.settings = {}

        saved_rate_mode = str(self.settings.get("rate_mode", "") or "").strip().lower()
        if saved_rate_mode not in {"simple", "detailed"}:
            saved_rate_mode = "detailed"
        self.rate_mode = tk.StringVar(value=saved_rate_mode)
        self.simple_labor_rate_var = tk.StringVar()
        self.simple_machine_rate_var = tk.StringVar()
        self._updating_simple_rates = False

        if hasattr(self.rate_mode, "trace_add"):
            self.rate_mode.trace_add("write", self._on_rate_mode_changed)
        elif hasattr(self.rate_mode, "trace"):
            self.rate_mode.trace("w", lambda *_: self._on_rate_mode_changed())

        def _bind_simple(var: tk.StringVar, kind: str) -> None:
            if hasattr(var, "trace_add"):
                var.trace_add("write", lambda *_: self._on_simple_rate_changed(kind))
            elif hasattr(var, "trace"):
                var.trace("w", lambda *_: self._on_simple_rate_changed(kind))

        _bind_simple(self.simple_labor_rate_var, "labor")
        _bind_simple(self.simple_machine_rate_var, "machine")

        thread_setting = str(self.settings.get("llm_thread_limit", "") or "").strip()
        env_thread_setting = os.environ.get("QWEN_N_THREADS", "").strip()
        initial_thread_setting = thread_setting or env_thread_setting
        self.llm_thread_limit = tk.StringVar(value=initial_thread_setting)
        self._llm_thread_limit_applied: int | None = None

        vendor_csv = str(self.settings.get("material_vendor_csv", "") or "")
        if vendor_csv:
            self.params["MaterialVendorCSVPath"] = vendor_csv

        saved_vars_path = self._get_last_variables_path()
        if saved_vars_path:
            path_obj = Path(saved_vars_path)
            if path_obj.exists() and path_obj.suffix.lower() in (".csv", ".xlsx"):
                try:
                    core_df, full_df = read_variables_file(saved_vars_path, return_full=True)
                except Exception:
                    logger.warning("Failed to preload variables from %s", saved_vars_path, exc_info=True)
                else:
                    if isinstance(core_df, pd.DataFrame) and isinstance(full_df, pd.DataFrame):
                        self._refresh_variables_cache(core_df, full_df)
                    else:
                        logger.warning(
                            "Variables preload returned unexpected types: %s, %s",
                            type(core_df),
                            type(full_df),
                        )

        # LLM defaults: ON + auto model discovery
        default_model = (
            self.configuration.default_llm_model_path
            if getattr(self.configuration, "default_llm_model_path", None)
            else self.llm_services.default_model_path()
        )
        if default_model:
            os.environ["QWEN_GGUF_PATH"] = default_model
            self.params["LLMModelPath"] = default_model

        self.llm_enabled = tk.BooleanVar(value=self.configuration.llm_enabled_default)
        self.apply_llm_adj = tk.BooleanVar(value=self.configuration.apply_llm_adjustments_default)
        self.llm_model_path = tk.StringVar(value=default_model)

        if hasattr(self.llm_thread_limit, "trace_add"):
            self.llm_thread_limit.trace_add("write", self._on_llm_thread_limit_changed)
        elif hasattr(self.llm_thread_limit, "trace"):
            self.llm_thread_limit.trace("w", lambda *_: self._on_llm_thread_limit_changed())
        self._apply_llm_thread_limit_env(persist=False)

        # Create a Menu Bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Overrides...", command=self.load_overrides)
        file_menu.add_command(label="Save Overrides...", command=self.save_overrides)
        file_menu.add_separator()
        file_menu.add_command(label="Import Quote Session...", command=self.import_quote_session)
        file_menu.add_command(label="Export Quote Session...", command=self.export_quote_session)
        file_menu.add_separator()
        file_menu.add_command(label="Set Material Vendor CSV...", command=self.set_material_vendor_csv)
        file_menu.add_command(label="Clear Material Vendor CSV", command=self.clear_material_vendor_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="Diagnostics",
            command=lambda: messagebox.showinfo("Diagnostics", get_import_diagnostics_text())
        )
        # Simplified top toolbar
        top = ttk.Frame(self); top.pack(fill="x", pady=(6,8))
        ttk.Button(top, text="1. Load CAD & Vars", command=self.open_flow).pack(side="left", padx=5)
        ttk.Button(top, text="2. Generate Quote", command=self.gen_quote).pack(side="left", padx=5)
        ttk.Button(top, text="LLM Inspector", command=self.open_llm_inspector).pack(side="left", padx=5)

        # Tabs
        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True)
        self.tab_geo = ttk.Frame(self.nb); self.nb.add(self.tab_geo, text="GEO")
        self.tab_editor = ttk.Frame(self.nb); self.nb.add(self.tab_editor, text="Quote Editor")
        self.editor_scroll = ScrollableFrame(self.tab_editor)
        self.editor_scroll.pack(fill="both", expand=True)
        self.tab_out = ttk.Frame(self.nb); self.nb.add(self.tab_out, text="Output")
        self.tab_llm = ttk.Frame(self)

        # Dictionaries to hold editor variables
        self.quote_vars = {}
        self.param_vars = {}
        self.rate_vars = {}
        self.editor_widgets_frame = None
        self.editor_vars: dict[str, tk.Variable] = {}
        self.editor_label_widgets: dict[str, ttk.Label] = {}
        self.editor_label_base: dict[str, str] = {}
        self.editor_value_sources: dict[str, str] = {}
        self._editor_set_depth = 0
        self._building_editor = False
        self._reprice_in_progress = False
        self.auto_reprice_enabled = False
        self._quote_dirty = False
        self.effective_process_hours: dict[str, float] = {}
        self.effective_scrap: float = 0.0
        self.effective_setups: int = 1
        self.effective_fixture: str = "standard"

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5)
        status_bar.pack(side="bottom", fill="x")

        if self.config_errors:
            message = "Configuration errors detected:\n- " + "\n- ".join(self.config_errors)
            messagebox.showerror("Configuration", message)
            self.status_var.set("Configuration error: see dialog for details.")

        # GEO (single pane; CAD open handled by top bar)
        self.geo_txt = tk.Text(self.tab_geo, wrap="word"); self.geo_txt.pack(fill="both", expand=True)

        # LLM (hidden frame is still built to keep functionality without a visible tab)
        if hasattr(self, "_build_llm"):
            try:
                self._build_llm(self.tab_llm)
            except Exception:
                pass

        # Output
        self.output_nb = ttk.Notebook(self.tab_out)
        self.output_nb.pack(fill="both", expand=True)

        self.output_tab_simplified = ttk.Frame(self.output_nb)
        self.output_nb.add(self.output_tab_simplified, text="Simplified")
        self.output_tab_full = ttk.Frame(self.output_nb)
        self.output_nb.add(self.output_tab_full, text="Full Detail")

        self.output_text_widgets: dict[str, tk.Text] = {}

        simplified_txt = tk.Text(self.output_tab_simplified, wrap="word")
        simplified_txt.pack(fill="both", expand=True)
        full_txt = tk.Text(self.output_tab_full, wrap="word")
        full_txt.pack(fill="both", expand=True)

        for name, widget in {"simplified": simplified_txt, "full": full_txt}.items():
            try:
                widget.tag_configure("rcol", tabs=("4.8i right",), tabstyle="tabular")
            except tk.TclError:
                widget.tag_configure("rcol", tabs=("4.8i right",))
            self.output_text_widgets[name] = widget

        self.output_nb.select(self.output_tab_simplified)

        self.LLM_SUGGEST = None
        self._llm_load_attempted = False
        self._llm_load_error: Exception | None = None


    def _reset_llm_logs(self) -> None:
        self.llm_events.clear()
        self.llm_errors.clear()
        if isinstance(self.quote_state, QuoteState):
            self.quote_state.llm_events = []
            self.quote_state.llm_errors = []

    def _llm_log_event(self, kind: str, payload: dict[str, Any]) -> None:
        entry = {"kind": kind, "payload": payload, "ts": time.time()}
        self.llm_events.append(entry)
        if len(self.llm_events) > 100:
            self.llm_events = self.llm_events[-100:]
        if isinstance(self.quote_state, QuoteState):
            self.quote_state.llm_events = list(self.llm_events)

    def _llm_handle_error(self, exc: Exception, context: dict[str, Any]) -> None:
        entry = {"error": repr(exc), "context": context, "ts": time.time()}
        self.llm_errors.append(entry)
        if len(self.llm_errors) > 50:
            self.llm_errors = self.llm_errors[-50:]
        if isinstance(self.quote_state, QuoteState):
            self.quote_state.llm_errors = list(self.llm_errors)
        try:
            self.status_var.set(f"LLM error: {exc}")
        except Exception:
            pass

    def get_llm_client(self, model_path: str | None = None) -> LLMClient | None:
        path = (model_path or "").strip()
        if not path and hasattr(self, "llm_model_path"):
            path = (self.llm_model_path.get().strip() if self.llm_model_path.get() else "")
        if not path:
            path = os.environ.get("QWEN_GGUF_PATH", "")
        path = path.strip()
        if not path:
            return None
        self._apply_llm_thread_limit_env(persist=False)
        cached = getattr(self, "_llm_client_cache", None)
        if cached and cached.model_path == path:
            return cached
        if cached:
            try:
                cached.close()
            except Exception:
                pass
        client = LLMClient(
            path,
            debug_enabled=APP_ENV.llm_debug_enabled,
            debug_dir=APP_ENV.llm_debug_dir,
            on_event=self._llm_log_event,
            on_error=self._llm_handle_error,
        )
        self._llm_client_cache = client
        return client


    def _load_settings(self) -> dict[str, Any]:
        path = getattr(self, "settings_path", None)
        if not isinstance(path, Path):
            return {}
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _save_settings(self) -> None:
        path = getattr(self, "settings_path", None)
        if not isinstance(path, Path):
            return
        try:
            path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _get_last_variables_path(self) -> str:
        if isinstance(self.settings, dict):
            return str(self.settings.get("last_variables_path", "") or "").strip()
        return ""

    def _set_last_variables_path(self, path: str | None) -> None:
        if not isinstance(self.settings, dict):
            self.settings = {}
        value = str(path) if path else ""
        self.settings["last_variables_path"] = value
        self._save_settings()

    def _validate_thread_limit(self, proposed: str) -> bool:
        text = str(proposed).strip()
        return text.isdigit() or text == ""

    def _current_llm_thread_limit(self) -> int | None:
        try:
            raw = self.llm_thread_limit.get()
        except Exception:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            value = int(text, 10)
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    def _invalidate_llm_client_cache(self) -> None:
        cached = getattr(self, "_llm_client_cache", None)
        if cached is not None:
            try:
                cached.close()
            except Exception:
                pass
        self._llm_client_cache = None

    def _apply_llm_thread_limit_env(self, *, persist: bool = True) -> int | None:
        limit = self._current_llm_thread_limit()
        prior = getattr(self, "_llm_thread_limit_applied", None)

        if limit is None:
            os.environ.pop("QWEN_N_THREADS", None)
        else:
            os.environ["QWEN_N_THREADS"] = str(limit)

        if persist:
            if not isinstance(self.settings, dict):
                self.settings = {}
            self.settings["llm_thread_limit"] = str(limit) if limit is not None else ""
            self._save_settings()

        if limit != prior:
            self._llm_thread_limit_applied = limit
            self._invalidate_llm_client_cache()

        return limit

    def _on_llm_thread_limit_changed(self, *_: object) -> None:
        self._apply_llm_thread_limit_env(persist=True)

    def _variables_dialog_defaults(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        saved = self._get_last_variables_path()
        if not saved:
            return defaults
        saved_path = Path(saved)
        if saved_path.exists():
            defaults["initialdir"] = str(saved_path.parent)
            defaults["initialfile"] = saved_path.name
            return defaults

        if saved_path.name:
            defaults["initialfile"] = saved_path.name
        parent = saved_path.parent
        try:
            if parent and str(parent):
                if parent.exists():
                    defaults.setdefault("initialdir", str(parent))
        except Exception:
            pass
        self._set_last_variables_path("")
        return defaults

    def _refresh_variables_cache(self, core_df: pd.DataFrame, full_df: pd.DataFrame) -> None:
        self.vars_df = core_df
        self.vars_df_full = full_df

    def set_material_vendor_csv(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Material Vendor CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        if not isinstance(self.settings, dict):
            self.settings = {}
        self.settings["material_vendor_csv"] = path
        self.params["MaterialVendorCSVPath"] = path
        self._save_settings()
        try:
            self.pricing.clear_cache()
        except Exception:
            pass
        self.status_var.set(f"Material vendor CSV set to {path}")

    def clear_material_vendor_csv(self) -> None:
        if not isinstance(self.settings, dict):
            self.settings = {}
        self.settings["material_vendor_csv"] = ""
        self.params["MaterialVendorCSVPath"] = ""
        self._save_settings()
        try:
            self.pricing.clear_cache()
        except Exception:
            pass
        self.status_var.set("Material vendor CSV cleared.")

    def _ensure_llm_loaded(self):
        """Load the optional vision LLM on-demand.

        The original implementation performed this work during ``__init__`` of
        the Tk application, which meant we blocked the UI event loop for
        several seconds (or longer if large models needed to be paged in).
        Loading lazily keeps the first paint of the window snappy while still
        providing feedback the moment the user requests LLM features.
        """

        if self.LLM_SUGGEST is not None:
            return self.LLM_SUGGEST
        if self._llm_load_attempted and self._llm_load_error is not None:
            return None

        self._llm_load_attempted = True
        start = time.perf_counter()

        limit: int | None = None
        try:
            limit = self._apply_llm_thread_limit_env(persist=False)
            status = "Loading Vision LLM (GPU)…"
            if limit:
                status = f"Loading Vision LLM (GPU, {limit} CPU threads)…"
            self.status_var.set(status)
            self.update_idletasks()
        except Exception:
            pass

        try:
            self.LLM_SUGGEST = self.llm_services.load_vision_model(
                n_ctx=8192,
                n_gpu_layers=20,
                n_threads=limit,
            )
        except Exception as exc:
            self._llm_load_error = exc
            try:
                limit = self._apply_llm_thread_limit_env(persist=False)
                msg = f"Vision LLM GPU load failed ({exc}); retrying CPU mode…"
                if limit:
                    msg = f"{msg[:-1]} with {limit} CPU threads…)"
                self.status_var.set(msg)
                self.update_idletasks()
            except Exception:
                pass
            try:
                self.LLM_SUGGEST = self.llm_services.load_vision_model(
                    n_ctx=4096,
                    n_gpu_layers=0,
                    n_threads=limit,
                )
            except Exception as exc2:
                self._llm_load_error = exc2
                try:
                    self.status_var.set(f"Vision LLM unavailable: {exc2}")
                except Exception:
                    pass
                return None
        else:
            self._llm_load_error = None

        duration = time.perf_counter() - start
        try:
            self.status_var.set(f"Vision LLM ready in {duration:.1f}s")
        except Exception:
            pass
        return self.LLM_SUGGEST

    def _populate_editor_tab(self, df: pd.DataFrame) -> None:
        df = coerce_or_make_vars_df(df)
        if self.vars_df_full is None:
            _, master_full = _load_master_variables()
            if master_full is not None:
                self.vars_df_full = master_full
        """Rebuild the Quote Editor tab using the latest variables dataframe."""
        def _ensure_row(dataframe: pd.DataFrame, item: str, value: Any, dtype: str = "number") -> pd.DataFrame:
            mask = dataframe["Item"].astype(str).str.fullmatch(item, case=False)
            if mask.any():
                return dataframe
            return upsert_var_row(dataframe, item, value, dtype=dtype)

        df = _ensure_row(df, "Scrap Percent (%)", 15.0, dtype="number")
        df = _ensure_row(df, "Plate Length (in)", 12.0, dtype="number")
        df = _ensure_row(df, "Plate Width (in)", 14.0, dtype="number")
        df = _ensure_row(df, "Thickness (in)", 0.0, dtype="number")
        df = _ensure_row(df, "Hole Count (override)", 0, dtype="number")
        df = _ensure_row(df, "Avg Hole Diameter (mm)", 0.0, dtype="number")
        df = _ensure_row(df, "Material", "", dtype="text")
        self.vars_df = df
        parent = self.editor_scroll.inner
        for child in parent.winfo_children():
            child.destroy()

        self.quote_vars.clear()
        self.param_vars.clear()
        self.rate_vars.clear()
        self.editor_vars.clear()
        self.editor_label_widgets.clear()
        self.editor_label_base.clear()
        self.editor_value_sources.clear()

        self._building_editor = True

        self.editor_widgets_frame = parent
        self.editor_widgets_frame.grid_columnconfigure(0, weight=1)

        items_series = df["Item"].astype(str)
        normalized_items = items_series.apply(_normalize_item_text)
        qty_mask = normalized_items.isin({"quantity", "qty", "lot size"})
        if qty_mask.any():
            qty_raw = df.loc[qty_mask, "Example Values / Options"].iloc[0]
            try:
                qty_value = float(str(qty_raw).strip())
            except Exception:
                qty_value = float(self.params.get("Quantity", 1) or 1)
            if math.isnan(qty_value):
                qty_value = float(self.params.get("Quantity", 1) or 1)
            self.params["Quantity"] = max(1, int(round(qty_value)))

        raw_skip_items = {
            "Overhead %", "Overhead",
            "G&A %", "G&A", "GA %", "GA",
            "Contingency %", "Contingency",
            "Profit Margin %", "Profit Margin", "Margin %", "Margin",
            "Expedite %", "Expedite",
            "Insurance %", "Insurance",
            "Vendor Markup %", "Vendor Markup",
            "Min Lot Charge",
            "Programmer $/hr",
            "CAM Programmer $/hr",
            "Milling $/hr",
            "Inspection $/hr",
            "Deburr $/hr",
            "Packaging $/hr",
            "Quantity", "Qty", "Lot Size",
        }
        skip_items = {_normalize_item_text(item) for item in raw_skip_items}


        material_lookup: Dict[str, float] = {}
        for _, row_data in df.iterrows():
            item_label = str(row_data.get("Item", ""))
            raw_value = _coerce_float_or_none(row_data.get("Example Values / Options"))
            if raw_value is None:
                continue
            per_g = _price_value_to_per_gram(raw_value, item_label)
            if per_g is None:
                continue
            normalized_label = _normalize_lookup_key(item_label)
            for canonical_key, keywords in MATERIAL_KEYWORDS.items():
                if canonical_key == MATERIAL_OTHER_KEY:
                    continue
                if any((kw and kw in normalized_label) for kw in keywords):
                    material_lookup[canonical_key] = per_g
                    break

        current_row = 0

        quote_frame = ttk.Labelframe(self.editor_widgets_frame, text="Quote-Specific Variables", padding=(10, 5))
        quote_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1

        row_index = 0
        material_choice_var: tk.StringVar | None = None
        material_price_var: tk.StringVar | None = None
        self.var_material: tk.StringVar | None = None

        def update_material_price(*_):
            _update_material_price_field(
                material_choice_var,
                material_price_var,
                material_lookup,
            )


        # Prefer the headers from the original dataframe if available so that
        # we can surface the richer context ("Why it Matters", formulas, etc.).
        def _resolve_column(name: str) -> str:
            import re

            def _norm_col(s: str) -> str:
                s = str(s).replace("\u00A0", " ")
                s = re.sub(r"\s+", " ", s).strip().lower()
                return re.sub(r"[^a-z0-9]", "", s)

            target = _norm_col(name)
            column_sources: list[pd.Index] = []
            if self.vars_df_full is not None:
                column_sources.append(self.vars_df_full.columns)
            column_sources.append(df.columns)
            for columns in column_sources:
                for col in columns:
                    if _norm_col(col) == target:
                        return col
            return name

        dtype_col_name = _resolve_column("Data Type / Input Method")
        value_col_name = _resolve_column("Example Values / Options")

        # Build a lookup so each row can pull the descriptive columns from the
        # original spreadsheet while still operating on the sanitized df copy.
        full_lookup: dict[str, pd.Series] = {}
        if self.vars_df_full is not None and "Item" in self.vars_df_full.columns:
            full_items = self.vars_df_full["Item"].astype(str)
            for idx, normalized in enumerate(full_items.apply(normalize_item)):
                if normalized and normalized not in full_lookup:
                    full_lookup[normalized] = self.vars_df_full.iloc[idx]

        for _, row_data in df.iterrows():
            item_name = str(row_data["Item"])
            normalized_name = normalize_item(item_name)
            if normalized_name in skip_items:
                continue

            full_row = full_lookup.get(normalized_name)

            dtype_source = row_data.get(dtype_col_name, "")
            if full_row is not None:
                dtype_source = full_row.get(dtype_col_name, dtype_source)

            initial_raw = row_data.get(value_col_name, "")
            if full_row is not None:
                initial_raw = full_row.get(value_col_name, initial_raw)
            initial_value = "" if pd.isna(initial_raw) else str(initial_raw)

            control_spec = derive_editor_control_spec(dtype_source, initial_raw)
            label_text = item_name
            if full_row is not None and "Variable ID" in full_row:
                var_id = str(full_row.get("Variable ID", "") or "").strip()
                if var_id:
                    label_text = f"{var_id} • {label_text}"
            display_hint = control_spec.display_label.strip()
            if display_hint and display_hint.lower() not in {"number", "text"}:
                label_text = f"{label_text}\n[{display_hint}]"


            row_container = ttk.Frame(quote_frame)
            row_container.grid(row=row_index, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
            row_container.grid_columnconfigure(1, weight=1)

            label_widget = ttk.Label(row_container, text=label_text, wraplength=400)

            label_widget.grid(row=0, column=0, sticky="w", padx=(0, 6))

            control_row = 0
            info_indicator: ttk.Label | None = None
            info_tooltip: CreateToolTip | None = None
            info_text_parts: list[str] = []

            control_container = ttk.Frame(row_container)
            control_container.grid(row=control_row, column=1, sticky="ew", padx=5)
            control_container.grid_columnconfigure(0, weight=1)
            control_container.grid_columnconfigure(1, weight=1)
            control_container.grid_columnconfigure(2, weight=0)


            def _add_info_label(text: str) -> None:
                nonlocal info_indicator, info_tooltip

                text = text.strip()
                if not text:
                    return
                info_text_parts.append(text)
                combined_text = "\n\n".join(info_text_parts)

                if info_indicator is None:
                    info_indicator = ttk.Label(
                        control_container,
                        text="ⓘ",
                        padding=(4, 0),
                        cursor="question_arrow",  # show question cursor while keeping tooltip
                        takefocus=0,
                    )
                    info_indicator.grid(row=control_row, column=2, sticky="nw", padx=(6, 0))
                    info_tooltip = CreateToolTip(info_indicator, combined_text, wraplength=360)
                elif info_tooltip is not None:
                    info_tooltip.update_text(combined_text)

            if normalized_name in {"material"}:
                var = tk.StringVar(value=self.default_material_display)
                if initial_value:
                    var.set(initial_value)
                normalized_initial = _normalize_lookup_key(var.get())
                for canonical_key, keywords in MATERIAL_KEYWORDS.items():
                    if canonical_key == MATERIAL_OTHER_KEY:
                        continue
                    if any(kw and kw in normalized_initial for kw in keywords):
                        display = MATERIAL_DISPLAY_BY_KEY.get(canonical_key)
                        if display:
                            var.set(display)
                        break
                combo = ttk.Combobox(
                    control_container,
                    textvariable=var,
                    values=MATERIAL_DROPDOWN_OPTIONS,
                    width=32,
                )
                combo.grid(row=0, column=0, sticky="ew")
                combo.bind("<<ComboboxSelected>>", update_material_price)
                var.trace_add("write", update_material_price)
                material_choice_var = var
                self.var_material = var
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            elif re.search(
                r"(Material\s*Price.*(per\s*gram|per\s*g|/g)|Unit\s*Price\s*/\s*g)",
                item_name,
                flags=re.IGNORECASE,
            ):
                var = tk.StringVar(value=initial_value)
                ttk.Entry(control_container, textvariable=var, width=30).grid(row=0, column=0, sticky="w")
                material_price_var = var
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            elif control_spec.control == "dropdown":
                options = list(control_spec.options) or ([control_spec.entry_value] if control_spec.entry_value else [])
                selected = control_spec.entry_value or (options[0] if options else "")
                var = tk.StringVar(value=selected)
                combo = ttk.Combobox(
                    control_container,
                    textvariable=var,
                    values=options,
                    width=28,
                    state="readonly" if options else "normal",
                )
                combo.grid(row=0, column=0, sticky="ew")
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            elif control_spec.control == "checkbox":
                initial_bool = control_spec.checkbox_state
                bool_var = tk.BooleanVar(value=bool(initial_bool))
                var = tk.StringVar(value="True" if initial_bool else "False")

                def _sync_string_from_bool(*_args: Any) -> None:
                    value = "True" if bool_var.get() else "False"
                    if var.get() != value:
                        var.set(value)

                def _sync_bool_from_string(*_args: Any) -> None:
                    current = var.get()
                    parsed = _coerce_checkbox_state(current, bool_var.get())
                    if bool_var.get() != parsed:
                        bool_var.set(parsed)

                if hasattr(bool_var, "trace_add"):
                    bool_var.trace_add("write", _sync_string_from_bool)
                else:  # pragma: no cover - legacy Tk fallback
                    bool_var.trace("w", lambda *_: _sync_string_from_bool())

                if hasattr(var, "trace_add"):
                    var.trace_add("write", _sync_bool_from_string)
                else:  # pragma: no cover - legacy Tk fallback
                    var.trace("w", lambda *_: _sync_bool_from_string())

                ttk.Checkbutton(
                    control_container,
                    variable=bool_var,
                    text=control_spec.checkbox_label or "Enabled",
                ).grid(row=0, column=0, sticky="w")
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            else:
                entry_value = control_spec.entry_value
                if not entry_value and control_spec.control != "formula":
                    entry_value = initial_value
                var = tk.StringVar(value=entry_value)
                ttk.Entry(control_container, textvariable=var, width=30).grid(row=0, column=0, sticky="w")
                base_text = control_spec.base_text.strip() if isinstance(control_spec.base_text, str) else ""
                if base_text:
                    _add_info_label(f"Based on: {base_text}")
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)

            why_text = ""
            if full_row is not None and "Why it Matters" in full_row:
                why_text = str(full_row.get("Why it Matters", "") or "").strip()
            elif "Why it Matters" in row_data:
                why_text = str(row_data.get("Why it Matters", "") or "").strip()
            if why_text:
                _add_info_label(why_text)

            swing_text = ""
            if full_row is not None and "Typical Price Swing*" in full_row:
                swing_text = str(full_row.get("Typical Price Swing*", "") or "").strip()
            elif "Typical Price Swing*" in row_data:
                swing_text = str(row_data.get("Typical Price Swing*", "") or "").strip()
            if swing_text:
                _add_info_label(f"Typical swing: {swing_text}")

            row_index += 1

        if material_choice_var is not None and material_price_var is not None:
            existing = _coerce_float_or_none(material_price_var.get())
            if existing is None or abs(existing) < 1e-9:
                update_material_price()

        def create_global_entries(
            parent_frame: tk.Widget,
            keys,
            data_source,
            var_dict,
            columns: int = 2,
        ) -> None:
            for i, key in enumerate(keys):
                row, col = divmod(i, columns)
                label_widget = ttk.Label(parent_frame, text=key)
                label_widget.grid(row=row, column=col * 2, sticky="e", padx=5, pady=2)
                var = tk.StringVar(value=str(data_source.get(key, "")))
                entry = ttk.Entry(parent_frame, textvariable=var, width=15)
                if "Path" in key:
                    entry.config(width=50)
                entry.grid(row=row, column=col * 2 + 1, sticky="w", padx=5, pady=2)
                var_dict[key] = var
                self._register_editor_field(key, var, label_widget)

        comm_frame = ttk.Labelframe(self.editor_widgets_frame, text="Global Overrides: Commercial & General", padding=(10, 5))
        comm_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1
        comm_keys = [
            "OverheadPct", "GA_Pct", "MarginPct", "ContingencyPct",
            "ExpeditePct", "VendorMarkupPct", "InsurancePct", "MinLotCharge", "Quantity",
        ]
        create_global_entries(comm_frame, comm_keys, self.params, self.param_vars)

        rates_frame = ttk.Labelframe(self.editor_widgets_frame, text="Global Overrides: Hourly Rates ($/hr)", padding=(10, 5))
        rates_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1

        mode_frame = ttk.Frame(rates_frame)
        mode_frame.grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Radiobutton(mode_frame, text="Simple", value="simple", variable=self.rate_mode).grid(
            row=0, column=1, sticky="w", padx=(0, 8)
        )
        ttk.Radiobutton(mode_frame, text="Detailed", value="detailed", variable=self.rate_mode).grid(
            row=0, column=2, sticky="w"
        )

        rates_container = ttk.Frame(rates_frame)
        rates_container.grid(row=1, column=0, sticky="ew")
        rates_container.grid_columnconfigure(0, weight=1)

        if self._simple_rate_mode_active():
            self._build_simple_rate_entries(rates_container)
        else:
            create_global_entries(
                rates_container,
                sorted(self.rates.keys()),
                self.rates,
                self.rate_vars,
                columns=3,
            )

        self._building_editor = False
        try:
            self._apply_geo_defaults(self.geo)
        except Exception:
            pass

    def _simple_rate_mode_active(self) -> bool:
        try:
            mode = str(self.rate_mode.get() or "").strip().lower()
        except Exception:
            return False
        return mode == "simple"

    def _format_rate_value(self, value: Any) -> str:
        if value in (None, ""):
            return ""
        try:
            num = float(value)
        except Exception:
            return str(value)
        text = f"{num:.3f}".rstrip("0").rstrip(".")
        return text or "0"

    def _build_simple_rate_entries(self, parent: tk.Widget) -> None:
        known_keys = set(self.rates.keys()) | LABOR_RATE_KEYS | MACHINE_RATE_KEYS
        for key in sorted(known_keys):
            formatted = self._format_rate_value(self.rates.get(key, ""))
            self.rate_vars[key] = tk.StringVar(value=formatted)

        container = ttk.Frame(parent)
        container.grid(row=0, column=0, sticky="w")

        ttk.Label(container, text="Labor Rate").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(container, textvariable=self.simple_labor_rate_var, width=15).grid(
            row=0, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Label(container, text="Machine Rate").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(container, textvariable=self.simple_machine_rate_var, width=15).grid(
            row=1, column=1, sticky="w", padx=5, pady=2
        )

        self._sync_simple_rate_fields()

    def _sync_simple_rate_fields(self) -> None:
        if not self._simple_rate_mode_active():
            return

        self._updating_simple_rates = True
        try:
            for key, var in self.rate_vars.items():
                var.set(self._format_rate_value(self.rates.get(key, "")))

            labor_value = next((self.rates.get(k) for k in LABOR_RATE_KEYS if k in self.rates), None)
            machine_value = next((self.rates.get(k) for k in MACHINE_RATE_KEYS if k in self.rates), None)

            self.simple_labor_rate_var.set(self._format_rate_value(labor_value))
            self.simple_machine_rate_var.set(self._format_rate_value(machine_value))
        finally:
            self._updating_simple_rates = False

    def _update_rate_group(self, keys: Iterable[str], value: float) -> bool:
        changed = False
        formatted = self._format_rate_value(value)
        for key in keys:
            previous = _coerce_float_or_none(self.rates.get(key))
            if previous is None or abs(previous - value) > 1e-6:
                changed = True
            self.rates[key] = float(value)
            var = self.rate_vars.get(key)
            if var is None:
                self.rate_vars[key] = tk.StringVar(value=formatted)
            elif var.get() != formatted:
                var.set(formatted)
        return changed

    def _apply_simple_rates(self, hint: str | None = None, *, trigger_reprice: bool = True) -> None:
        if not self._simple_rate_mode_active() or self._updating_simple_rates:
            return

        labor_value = _coerce_float_or_none(self.simple_labor_rate_var.get())
        machine_value = _coerce_float_or_none(self.simple_machine_rate_var.get())

        changed = False
        if labor_value is not None:
            changed |= self._update_rate_group(LABOR_RATE_KEYS, float(labor_value))
        if machine_value is not None:
            changed |= self._update_rate_group(MACHINE_RATE_KEYS, float(machine_value))

        if changed and trigger_reprice:
            self.reprice(hint=hint or "Updated hourly rates.")

    def _on_simple_rate_changed(self, kind: str) -> None:
        hint = None
        if kind == "labor":
            hint = "Updated labor rates."
        elif kind == "machine":
            hint = "Updated machine rates."
        self._apply_simple_rates(hint=hint)

    def _on_rate_mode_changed(self, *_: Any) -> None:
        if getattr(self, "_building_editor", False):
            return

        mode = str(self.rate_mode.get() or "").strip().lower()
        if isinstance(self.settings, dict):
            self.settings["rate_mode"] = mode
            self._save_settings()

        try:
            if self.vars_df is not None:
                self._populate_editor_tab(self.vars_df)
            else:
                self._populate_editor_tab(coerce_or_make_vars_df(None))
        except Exception:
            self._populate_editor_tab(coerce_or_make_vars_df(None))

    def _register_editor_field(self, label: str, var: tk.Variable, label_widget: ttk.Label | None) -> None:
        if not isinstance(label, str) or not isinstance(var, tk.Variable):
            return
        self.editor_vars[label] = var
        if label_widget is not None:
            self.editor_label_widgets[label] = label_widget
            self.editor_label_base[label] = label
        else:
            self.editor_label_base.setdefault(label, label)
        self.editor_value_sources.pop(label, None)
        self._mark_label_source(label, None)
        self._bind_editor_var(label, var)

    def _bind_editor_var(self, label: str, var: tk.Variable) -> None:
        def _on_write(*_):
            if self._building_editor or self._editor_set_depth > 0:
                return
            self._update_editor_override_from_label(label, var.get())
            self._mark_label_source(label, "User")
            self.reprice(hint=f"Updated {label}.")

        var.trace_add("write", _on_write)

    def _mark_quote_dirty(self, hint: str | None = None) -> None:
        self._quote_dirty = True
        message = "Quote editor updated."
        if isinstance(hint, str):
            cleaned = hint.strip()
            if cleaned:
                message = cleaned.splitlines()[0]
        try:
            self.status_var.set(f"{message} Click Generate Quote to refresh totals.")
        except Exception:
            pass

    def _clear_quote_dirty(self) -> None:
        self._quote_dirty = False

    def _mark_label_source(self, label: str, src: str | None) -> None:
        widget = self.editor_label_widgets.get(label)
        base = self.editor_label_base.get(label, label)
        if widget is not None:
            if src == "LLM":
                widget.configure(text=f"{base}  (LLM)", foreground="#1463FF")
            elif src == "User":
                widget.configure(text=f"{base}  (User)", foreground="#22863a")
            else:
                widget.configure(text=base, foreground="")
        if src:
            self.editor_value_sources[label] = src
        else:
            self.editor_value_sources.pop(label, None)

    def _set_editor(self, label: str, value: Any, source_tag: str = "LLM") -> None:
        if self.editor_value_sources.get(label) == "User":
            return
        var = self.editor_vars.get(label)
        if var is None:
            return
        if isinstance(value, float):
            txt = f"{value:.3f}"
        else:
            txt = str(value)
        self._editor_set_depth += 1
        try:
            var.set(txt)
        finally:
            self._editor_set_depth -= 1
        self._mark_label_source(label, source_tag)

    def _fill_editor_if_blank(self, label: str, value: Any, source: str = "GEO") -> None:
        if value is None:
            return
        if self.editor_value_sources.get(label) == "User":
            return
        var = self.editor_vars.get(label)
        if var is None:
            return
        raw = var.get()
        current = str(raw).strip() if raw is not None else ""
        fill = False
        if not current:
            fill = True
        else:
            try:
                fill = float(current) == 0.0
            except Exception:
                fill = False
        if not fill:
            return
        if isinstance(value, (int, float)):
            txt = f"{float(value):.3f}"
        else:
            txt = str(value)
        self._editor_set_depth += 1
        try:
            var.set(txt)
        finally:
            self._editor_set_depth -= 1
        self._mark_label_source(label, source)

    def _apply_geo_defaults(self, geo_data: dict[str, Any] | None) -> None:
        defaults = infer_geo_override_defaults(geo_data)
        if not defaults:
            return

        for label, value in defaults.items():
            if value is None:
                continue
            if isinstance(value, str):
                var = self.editor_vars.get(label)
                if var is None:
                    continue
                current = str(var.get() or "").strip()
                if label == "Material":
                    if current and current != self.default_material_display:
                        continue
                elif current:
                    continue
                self._set_editor(label, value, "GEO")
                continue

            self._fill_editor_if_blank(label, value, source="GEO")
    def _update_editor_override_from_label(self, label: str, raw_value: str) -> None:
        key = EDITOR_TO_SUGG.get(label)
        if key is None:
            return
        converter = EDITOR_FROM_UI.get(label)
        value: Any | None = None
        if converter is not None:
            try:
                value = converter(raw_value)
            except Exception:
                value = None
        path = key if isinstance(key, tuple) else (key,)
        if value is None:
            self._set_user_override_value(path, None)
        else:
            self._set_user_override_value(path, value)

    def apply_llm_to_editor(self, sugg: dict, baseline_ctx: dict) -> None:
        if not isinstance(sugg, dict) or not isinstance(baseline_ctx, dict):
            return
        for key, spec in SUGG_TO_EDITOR.items():
            label, to_ui, _ = spec
            if isinstance(key, tuple):
                root, sub = key
                val = ((sugg.get(root) or {}).get(sub))
            else:
                val = sugg.get(key)
            if val is None:
                continue
            try:
                ui_val = to_ui(val)
            except Exception:
                continue
            self._set_editor(label, ui_val, "LLM")

        mults = sugg.get("process_hour_multipliers") or {}
        eff_hours: dict[str, float] = {}
        base_hours = baseline_ctx.get("process_hours")
        if isinstance(base_hours, dict):
            for proc, hours in base_hours.items():
                val = _coerce_float_or_none(hours)
                if val is not None:
                    eff_hours[str(proc)] = float(val)
        for proc, mult in mults.items():
            if proc in eff_hours:
                try:
                    base_val = float(eff_hours[proc])
                except Exception:
                    continue
                clamped_mult = clamp(mult, LLM_MULTIPLIER_MIN, LLM_MULTIPLIER_MAX, 1.0)
                try:
                    eff_hours[proc] = base_val * float(clamped_mult)
                except Exception:
                    eff_hours[proc] = base_val
        for proc, (label, scale) in PROC_MULT_TARGETS.items():
            if proc in eff_hours:
                try:
                    derived = eff_hours[proc] * float(scale)
                except Exception:
                    continue
                self._set_editor(label, derived, "LLM")

        self.effective_process_hours = eff_hours
        self.effective_scrap = float(sugg.get("scrap_pct", baseline_ctx.get("scrap_pct", 0.0)) or 0.0)
        self.effective_setups = int(sugg.get("setups", baseline_ctx.get("setups", 1)) or 1)
        self.effective_fixture = str(sugg.get("fixture", baseline_ctx.get("fixture", "standard")) or "standard")

        if not self._reprice_in_progress:
            self.reprice(hint="LLM adjustments applied.")

    def _set_user_override_value(self, path: Tuple[str, ...], value: Any):
        cur = self.quote_state.user_overrides
        if not isinstance(cur, dict):
            self.quote_state.user_overrides = {}
            cur = self.quote_state.user_overrides
        node = cur
        stack: list[tuple[dict, str]] = []
        for key in path[:-1]:
            stack.append((node, key))
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                if value is None:
                    return
                nxt = {}
                node[key] = nxt
            node = nxt
        leaf_key = path[-1]
        if value is None:
            node.pop(leaf_key, None)
            while stack:
                parent, pkey = stack.pop()
                child = parent.get(pkey)
                if isinstance(child, dict) and not child:
                    parent.pop(pkey, None)
                else:
                    break
        else:
            node[leaf_key] = value

    # ----- Full flow: CAD ? GEO ? LLM ? Quote -----
    def action_full_flow(self):
        # ---------- choose file ----------
        self.status_var.set("Opening CAD/Drawingï¿½")
        path = filedialog.askopenfilename(
            title="Select CAD/Drawing",
            filetypes=[
                ("CAD/Drawing", "*.step *.stp *.iges *.igs *.brep *.stl *.dwg *.dxf *.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            self.status_var.set("Ready")
            return

        ext = Path(path).suffix.lower()
        self.status_var.set(f"Processing {os.path.basename(path)}ï¿½")

        # ---------- 2D branch: PDF / DWG / DXF ----------
        if ext in (".pdf", ".dwg", ".dxf"):
            try:
                structured_pdf = None
                if ext == ".pdf":
                    structured_pdf = self.geometry_loader.extract_pdf_all(path)
                    g2d = self.geometry_loader.extract_2d_features_from_pdf_vector(path)   # PyMuPDF vector-only MVP
                else:
                    g2d = self.geometry_loader.extract_2d_features_from_dxf_or_dwg(path)   # ezdxf / ODA

                if not isinstance(g2d, dict):
                    g2d = {}


                if self.vars_df is None:
                    vp = find_variables_near(path)
                    if not vp:
                        dialog_kwargs: dict[str, Any] = {"title": "Select variables CSV/XLSX"}
                        dialog_kwargs.update(self._variables_dialog_defaults())
                        vp = filedialog.askopenfilename(**dialog_kwargs)
                    if vp:
                        try:
                            core_df, full_df = read_variables_file(vp, return_full=True)
                            core_df_t = typing.cast(pd.DataFrame, core_df)
                            full_df_t = typing.cast(pd.DataFrame, full_df)
                            self._refresh_variables_cache(core_df_t, full_df_t)
                            self._set_last_variables_path(vp)
                        except Exception as read_err:
                            messagebox.showerror("Variables", f"Failed to read variables file:\n{read_err}\n\nContinuing with defaults.")
                            self.vars_df = None
                            self.vars_df_full = None
                    else:
                        messagebox.showinfo("Variables", "No variables file provided; using defaults.")
                        self.vars_df = None
                self.vars_df = coerce_or_make_vars_df(self.vars_df)

                if structured_pdf:
                    try:
                        estimate = infer_pdf_estimate(structured_pdf)
                    except Exception:
                        estimate = {}
                    if estimate:
                        self.vars_df = merge_estimate_into_vars(self.vars_df, estimate)

                self.vars_df = apply_2d_features_to_variables(self.vars_df, g2d, params=self.params, rates=self.rates)
                self.geo = g2d
                geo_dict = g2d if isinstance(g2d, dict) else {}
                self.geo_context = dict(geo_dict)
                inner_geo_raw = geo_dict.get("geo")
                inner_geo = inner_geo_raw if isinstance(inner_geo_raw, dict) else {}
                read_more_geo_raw = geo_dict.get("geo_read_more")
                read_more_geo = read_more_geo_raw if isinstance(read_more_geo_raw, dict) else {}
                hole_count_val = (
                    _coerce_float_or_none(read_more_geo.get("hole_count_geom"))
                    or _coerce_float_or_none(inner_geo.get("hole_count"))
                    or _coerce_float_or_none(geo_dict.get("hole_count"))
                    or 0
                )
                edge_len_in = (
                    _coerce_float_or_none(read_more_geo.get("edge_len_in"))
                    or _coerce_float_or_none(inner_geo.get("edge_len_in"))
                    or 0.0
                )
                tap_qty_val = int(
                    _coerce_float_or_none(read_more_geo.get("tap_qty"))
                    or _coerce_float_or_none(inner_geo.get("tap_qty"))
                    or _coerce_float_or_none(geo_dict.get("tap_qty"))
                    or 0
                )
                cbore_qty_val = int(
                    _coerce_float_or_none(read_more_geo.get("cbore_qty"))
                    or _coerce_float_or_none(inner_geo.get("cbore_qty"))
                    or _coerce_float_or_none(geo_dict.get("cbore_qty"))
                    or 0
                )
                self.geo_context.update(
                    {
                        "hole_count": int(hole_count_val or 0),
                        "edge_len_in": float(edge_len_in or 0.0),
                        "tap_qty": tap_qty_val,
                        "cbore_qty": cbore_qty_val,
                    }
                )
                self._log_geo(g2d)

                self._populate_editor_tab(self.vars_df)
                self.nb.select(self.tab_editor)
                self.status_var.set(f"{ext.upper()} variables loaded. Review the Quote Editor and generate the quote.")
                return
            except Exception as e:
                messagebox.showerror(
                    "2D Import Error",
                    f"Failed to process {ext.upper()} file:\n{e}\n\n"
                    "Tip (DWG): set ODA_CONVERTER_EXE or DWG2DXF_EXE to a converter that accepts <input.dwg> <output.dxf>."
                )
                self.status_var.set("Ready")
                return

        # ---------- 3D branch: STEP / IGES / BREP / STL ----------
        geo = None
        try:
            # Fast path (your OCC feature extractor for 3D)
            geo = self.geometry_service.extract_occ_features(path)  # handles STEP/IGES/BREP

        except Exception:
            geo = None

        if geo is None:
            if ext == ".stl":
                stl_geo = None
                try:
                    stl_geo = self.geometry_service.enrich_stl(path)  # trimesh-based

                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    messagebox.showerror(
                        "2D Import Error",
                        f"Failed to process STL file:\n{e}\n\n{tb}"
                    )
                    self.status_var.set("Ready")
                    return
                if stl_geo is None:
                    messagebox.showerror(
                        "2D Import Error",
                        "STL import produced no 2D geometry."
                    )
                    self.status_var.set("Ready")
                    return
                geo = _map_geo_to_double_underscore(stl_geo)
            else:
                try:
                    if ext in (".step", ".stp"):
                        shape = self.geometry_service.read_step(path)
                    else:
                        shape = self.geometry_service.read_model(path)            # IGES/BREP and others
                    _ = safe_bbox(shape)
                    g = self.geometry_service.enrich_occ(shape)             # OCC-based geometry features

                    geo = _map_geo_to_double_underscore(g)
                except Exception as e:
                    messagebox.showerror(
                        "CAD Import Error",
                        f"Failed to read CAD file:\n{e}\n\n"
                        "Tip (DWG): upload DXF/STEP/SAT, or set ODA_CONVERTER_EXE to your dwg2dxf wrapper."
                    )
                    self.status_var.set("Ready")
                    return
        # ---------- variables + LLM hours + quote ----------
        if self.vars_df is None:
            vp = find_variables_near(path)
            if not vp:
                dialog_kwargs: dict[str, Any] = {"title": "Select variables CSV/XLSX"}
                dialog_kwargs.update(self._variables_dialog_defaults())
                vp = filedialog.askopenfilename(**dialog_kwargs)
            if not vp:
                messagebox.showinfo("Variables", "No variables file provided.")
                self.status_var.set("Ready")
                return
            try:
                core_df, full_df = read_variables_file(vp, return_full=True)
                core_df_t = typing.cast(pd.DataFrame, core_df)
                full_df_t = typing.cast(pd.DataFrame, full_df)
                self._refresh_variables_cache(core_df_t, full_df_t)
                self._set_last_variables_path(vp)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                messagebox.showerror("Variables", f"Failed to read variables file:\n{e}\n\n{tb}")
                self.status_var.set("Ready")
                return

        self.vars_df = coerce_or_make_vars_df(self.vars_df)

        # Merge GEO rows
        try:
            for k, v in geo.items():
                self.vars_df = upsert_var_row(self.vars_df, k, v, dtype="number")
        except Exception as e:
            messagebox.showerror("Variables", f"Failed to update variables with GEO rows:\n{e}")
            self.status_var.set("Ready")
            return

        # LLM hour estimation
        self.status_var.set("Estimating hours with LLMï¿½")
        self._reset_llm_logs()
        decision_log = {}
        client = None
        if self.llm_enabled.get():
            client = self.get_llm_client(self.llm_model_path.get().strip() or None)
        est_raw = infer_hours_and_overrides_from_geo(geo, params=self.params, rates=self.rates, client=client)
        est = clamp_llm_hours(est_raw, geo, params=self.params)
        self.vars_df = apply_llm_hours_to_variables(self.vars_df, est, allow_overwrite_nonzero=True, log=decision_log)
        self.geo = geo
        self.geo_context = dict(geo or {})
        self._log_geo(geo)

        self._populate_editor_tab(self.vars_df)
        self.nb.select(self.tab_editor)
        self.status_var.set("Variables loaded. Review the Quote Editor and click Generate Quote.")
        return

    # Back-compat: keep the old button working
    def open_flow(self):
        # Delegate to the new full-flow handler
        return self.action_full_flow()

    def apply_overrides(self, notify: bool = False) -> None:
        raw_param_values = {k: var.get() for k, var in self.param_vars.items()}

        updates = {}
        for key, raw_value in raw_param_values.items():
            if key.endswith("Path"):
                updates[key] = raw_value
                continue
            try:
                updates[key] = float(str(raw_value).strip())
            except Exception:
                updates[key] = self.params.get(key, 0.0)

        quantity_val = None
        if "Quantity" in updates:
            qty_raw = raw_param_values.get("Quantity", "")
            try:
                qty_value = float(str(qty_raw).strip())
            except Exception:
                qty_value = float(self.params.get("Quantity", 1) or 1)
            if math.isnan(qty_value):
                qty_value = float(self.params.get("Quantity", 1) or 1)
            quantity_val = max(1, int(round(qty_value)))
            updates["Quantity"] = quantity_val

        for key, val in list(updates.items()):
            if key.endswith("Pct") and not isinstance(val, str):
                updates[key] = parse_pct(val)

        self.params.update(updates)

        if quantity_val is not None:
            if "Quantity" in self.param_vars:
                self.param_vars["Quantity"].set(str(quantity_val))

        if self.vars_df is not None and raw_param_values:
            normalized_items = self.vars_df["Item"].astype(str).apply(_normalize_item_text)
            param_to_items = {
                "OverheadPct": ["overhead %", "overhead"],
                "GA_Pct": ["g&a %", "ga %"],
                "ContingencyPct": ["contingency %"],
                "MarginPct": ["profit margin %", "margin %"],
                "ExpeditePct": ["expedite %"],
                "InsurancePct": ["insurance %"],
                "VendorMarkupPct": ["vendor markup %"],
                "MinLotCharge": ["min lot charge"],
                "Quantity": ["quantity", "qty", "lot size"],
            }
            for key, labels in param_to_items.items():
                if key not in raw_param_values:
                    continue
                value = raw_param_values.get(key, "")
                if key == "Quantity" and quantity_val is not None:
                    value = str(quantity_val)
                elif isinstance(value, str) and not value.strip():
                    value = str(updates.get(key, self.params.get(key, "")))
                for label in labels:
                    mask = normalized_items == label
                    if mask.any():
                        self.vars_df.loc[mask, "Example Values / Options"] = value
                        break

        if self._simple_rate_mode_active():
            self._apply_simple_rates(trigger_reprice=False)
            self._sync_simple_rate_fields()
        else:
            rate_raw_values = {k: var.get() for k, var in self.rate_vars.items()}
            rate_updates = {}
            for key, raw_value in rate_raw_values.items():
                try:
                    rate_updates[key] = float(str(raw_value).strip())
                except Exception:
                    fallback = _coerce_float_or_none(self.rates.get(key))
                    rate_updates[key] = fallback if fallback is not None else 0.0
            self.rates.update(rate_updates)
            for key, value in rate_updates.items():
                var = self.rate_vars.get(key)
                if var is not None:
                    var.set(self._format_rate_value(value))

        if notify:
            messagebox.showinfo("Overrides", "Overrides applied.")
        self.status_var.set("Overrides applied.")

    def save_overrides(self):

        self.apply_overrides()
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")], initialfile="overrides.json")
        if not path: return
        data = {"params": self.params, "rates": self.rates}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Overrides", "Saved to:\n{path}")
            self.status_var.set(f"Saved overrides to {path}")
        except Exception:
            messagebox.showerror("Overrides", f"Save failed:\n{{e}}")
            self.status_var.set("Failed to save overrides.")

    def _path_key(self, path: Tuple[str, ...]) -> str:
        return ".".join(path)

    def load_overrides(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "params" in data: self.params.update(data["params"])
            if "rates" in data:
                try:
                    self.rates.update({k: float(v) for k, v in data["rates"].items()})
                except Exception:
                    self.rates.update(data["rates"])
            for k,v in self.param_vars.items():
                v.set(str(self.params.get(k, "")))
            if self._simple_rate_mode_active():
                self._sync_simple_rate_fields()
            else:
                for k,v in self.rate_vars.items():
                    v.set(self._format_rate_value(self.rates.get(k, "")))
            messagebox.showinfo("Overrides", "Overrides loaded.")
            self.status_var.set(f"Loaded overrides from {path}")
        except Exception:
            messagebox.showerror("Overrides", f"Load failed:\n{{e}}")
            self.status_var.set("Failed to load overrides.")

    def _exportable_vars_records(self) -> list[dict[str, Any]]:
        if self.vars_df is None:
            return []
        df_snapshot = self.vars_df.copy(deep=True)
        try:
            for item_name, string_var in self.quote_vars.items():
                mask = df_snapshot["Item"] == item_name
                if mask.any():
                    df_snapshot.loc[mask, "Example Values / Options"] = string_var.get()
        except Exception:
            pass

        records: list[dict[str, Any]] = []
        for _, row in df_snapshot.iterrows():
            record: dict[str, Any] = {}
            for column, value in row.items():
                column_name = str(column)
                if pd.isna(value):
                    record[column_name] = None
                elif hasattr(value, "item"):
                    try:
                        record[column_name] = value.item()
                    except Exception:
                        record[column_name] = value
                else:
                    record[column_name] = value
            records.append(record)
        return records

    def export_quote_session(self) -> None:
        if self.vars_df is None:
            messagebox.showinfo("Export Quote Session", "Load a quote before exporting the session.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Quote Session", "*.json"), ("JSON", "*.json"), ("All", "*.*")],
            initialfile="quote_session.json",
        )
        if not path:
            return

        previous_status = self.status_var.get()
        try:
            self.apply_overrides()
        finally:
            self.status_var.set(previous_status)

        ui_vars = {label: var.get() for label, var in self.quote_vars.items()}
        if ui_vars:
            self.quote_state.ui_vars = dict(ui_vars)
        self.quote_state.rates = dict(self.rates)
        if self.geo:
            self.quote_state.geo = dict(self.geo)

        session_payload = {
            "version": 1,
            "exported_at": time.time(),
            "params": dict(self.params),
            "rates": dict(self.rates),
            "geo": dict(self.geo or {}),
            "geo_context": dict(self.geo_context or {}),
            "vars_df": self._exportable_vars_records(),
            "quote_state": self.quote_state.to_dict(),
            "llm": {
                "enabled": bool(self.llm_enabled.get()),
                "apply_adjustments": bool(self.apply_llm_adj.get()),
                "model_path": self.llm_model_path.get().strip(),
                "thread_limit": self.llm_thread_limit.get().strip(),
            },
            "status": self.status_var.get(),
        }

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(session_payload, handle, indent=2)
            messagebox.showinfo("Export Quote Session", f"Session saved to:\n{path}")
            self.status_var.set(f"Quote session exported to {path}")
        except Exception as exc:
            messagebox.showerror("Export Quote Session", f"Failed to export session:\n{exc}")
            self.status_var.set("Failed to export quote session.")

    def import_quote_session(self) -> None:
        path = filedialog.askopenfilename(
            title="Import Quote Session",
            filetypes=[("Quote Session", "*.json"), ("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            messagebox.showerror("Import Quote Session", f"Failed to read session:\n{exc}")
            self.status_var.set("Failed to import quote session.")
            return

        if not isinstance(payload, dict):
            messagebox.showerror("Import Quote Session", "Session file is not a valid JSON object.")
            self.status_var.set("Invalid quote session file.")
            return

        params_payload = payload.get("params")
        self.params = PARAMS_DEFAULT.copy()
        if isinstance(params_payload, dict):
            self.params.update(params_payload)

        rates_payload = payload.get("rates")
        self.rates = RATES_DEFAULT.copy()
        if isinstance(rates_payload, dict):
            self.rates.update(rates_payload)

        llm_payload = payload.get("llm") or {}
        if isinstance(llm_payload, dict):
            self.llm_enabled.set(bool(llm_payload.get("enabled", True)))
            self.apply_llm_adj.set(bool(llm_payload.get("apply_adjustments", True)))
            model_path = llm_payload.get("model_path")
            if isinstance(model_path, str):
                self.llm_model_path.set(model_path)
            thread_limit = llm_payload.get("thread_limit")
            if isinstance(thread_limit, str):
                self.llm_thread_limit.set(thread_limit)
            elif isinstance(thread_limit, (int, float)):
                try:
                    self.llm_thread_limit.set(str(int(thread_limit)))
                except Exception:
                    self.llm_thread_limit.set("")
            self._apply_llm_thread_limit_env(persist=True)

        geo_payload = payload.get("geo")
        self.geo = dict(geo_payload) if isinstance(geo_payload, dict) else {}
        geo_context_payload = payload.get("geo_context")
        if isinstance(geo_context_payload, dict):
            self.geo_context = dict(geo_context_payload)
        else:
            self.geo_context = dict(self.geo)

        vars_payload = payload.get("vars_df")
        has_records = isinstance(vars_payload, list) and len(vars_payload) > 0
        if isinstance(vars_payload, list):
            try:
                self.vars_df = pd.DataFrame.from_records(vars_payload)
            except Exception:
                self.vars_df = None
        else:
            self.vars_df = None

        quote_state_payload = payload.get("quote_state")
        try:
            self.quote_state = QuoteState.from_dict(quote_state_payload)
        except Exception as exc:
            messagebox.showerror("Import Quote Session", f"Quote state invalid:\n{exc}")
            self.status_var.set("Failed to import quote session.")
            return

        ensure_accept_flags(self.quote_state)

        self.quote_state.rates = dict(self.rates)
        if not self.quote_state.geo and self.geo:
            self.quote_state.geo = dict(self.geo)

        if self.geo:
            try:
                self._log_geo(self.geo)
            except Exception:
                pass

        if self.vars_df is not None:
            self._populate_editor_tab(self.vars_df)
        else:
            self._populate_editor_tab(coerce_or_make_vars_df(None))

        for key, var in self.param_vars.items():
            var.set(str(self.params.get(key, "")))
        if self._simple_rate_mode_active():
            self._sync_simple_rate_fields()
        else:
            for key, var in self.rate_vars.items():
                var.set(self._format_rate_value(self.rates.get(key, "")))

        self.status_var.set(f"Quote session imported from {path}")

        if has_records and self.vars_df is not None and not self.vars_df.empty:
            self.gen_quote(reuse_suggestions=True)

    # ----- LLM tab -----
    def _build_llm(self, parent):
        row=0
        ttk.Checkbutton(parent, text="Enable LLM (Qwen via llama-cpp, offline)", variable=self.llm_enabled).grid(row=row, column=0, sticky="w", pady=(6,2)); row+=1
        ttk.Label(parent, text="Qwen GGUF model path").grid(row=row, column=0, sticky="e", padx=5, pady=3)
        ttk.Entry(parent, textvariable=self.llm_model_path, width=80).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(parent, text="Browse...", command=self._pick_model).grid(row=row, column=2, padx=5); row+=1
        validate_cmd = self.register(self._validate_thread_limit)
        ttk.Label(parent, text="Max CPU threads (blank = auto)").grid(row=row, column=0, sticky="e", padx=5, pady=3)
        ttk.Entry(
            parent,
            textvariable=self.llm_thread_limit,
            width=12,
            validate="key",
            validatecommand=(validate_cmd, "%P"),
        ).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        ttk.Label(
            parent,
            text="Lower this if llama.cpp overwhelms your machine.",
        ).grid(row=row, column=2, sticky="w", padx=5, pady=3)
        row += 1
        ttk.Checkbutton(parent, text="Apply LLM adjustments to params", variable=self.apply_llm_adj).grid(row=row, column=0, sticky="w", pady=(0,6)); row+=1
        ttk.Button(parent, text="Run LLM on current GEO", command=self.run_llm).grid(row=row, column=0, sticky="w", padx=5, pady=6); row+=1
        self.llm_txt = tk.Text(parent, wrap="word", height=24); self.llm_txt.grid(row=row, column=0, columnspan=3, sticky="nsew")
        parent.grid_columnconfigure(1, weight=1); parent.grid_rowconfigure(row, weight=1)

    def _pick_model(self):
        p=filedialog.askopenfilename(title="Choose Qwen *.gguf", filetypes=[("GGUF","*.gguf"),("All","*.*")])
        if p:
            self.llm_model_path.set(p)
            os.environ["QWEN_GGUF_PATH"] = p

    def run_llm(self):
        self.llm_txt.delete("1.0","end")
        if not self.llm_enabled.get():
            self.llm_txt.insert("end","LLM disabled (toggle ON to use it).\n"); return
        if not self.geo:
            messagebox.showinfo("LLM","Load a CAD first so we have GEO context."); return
        mp=self.llm_model_path.get().strip() or os.environ.get("QWEN_GGUF_PATH","")
        if not (mp and Path(mp).is_file() and mp.lower().endswith(".gguf")):
            self.llm_txt.insert("end","No GGUF model found. Put one in D:\\CAD_Quoting_Tool\\models or set QWEN_GGUF_PATH.\n")
            return
        os.environ["QWEN_GGUF_PATH"]=mp
        try:
            out = infer_shop_overrides_from_geo(self.geo, params=self.params, rates=self.rates)
        except Exception as e:
            logger.exception("LLM override inference failed")
            self.llm_txt.insert("end", f"LLM error: {e}\n"); return
        self.llm_txt.insert("end", json.dumps(out, indent=2))
        if self.apply_llm_adj.get() and isinstance(out, dict):
            adj = out.get("LLM_Adjustments", {})
            try:
                self.params["OverheadPct"] += float(adj.get("OverheadPct_add", 0.0) or 0.0)
                self.params["MarginPct"] += float(adj.get("MarginPct_add", 0.0) or 0.0)
                self.params["ConsumablesFlat"] += float(adj.get("ConsumablesFlat_add", 0.0) or 0.0)
                for k,v in self.param_vars.items(): v.set(str(self.params.get(k, "")))
                messagebox.showinfo("LLM", "Applied LLM adjustments to parameters.")
            except Exception:
                pass

    def open_llm_inspector(self):
        import json
        import tkinter as tk
        from pathlib import Path
        from tkinter import messagebox, scrolledtext

        debug_dir = Path(__file__).with_name("llm_debug")
        files = sorted(debug_dir.glob("llm_snapshot_*.json"))
        if not files:
            messagebox.showinfo("LLM Inspector", "No snapshots yet.")
            return

        latest = files[-1]
        try:
            raw = latest.read_text(encoding="utf-8")
            try:
                data = json.loads(raw)
                shown = json.dumps(data, indent=2)
            except Exception:
                shown = raw
        except Exception as e:
            messagebox.showerror("LLM Inspector", f"Failed to read: {e}")
            return

        win = tk.Toplevel(self)
        win.title(f"LLM Inspector — {latest.name}")
        win.geometry("900x700")

        txt = scrolledtext.ScrolledText(win, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", shown)
        txt.configure(state="disabled")

    # ----- Flow + Output -----
    def _log_geo(self, d):
        self.geo_txt.delete("1.0","end")
        self.geo_txt.insert("end", json.dumps(d, indent=2))

    def _log_out(self, d):
        widget = self.output_text_widgets.get("simplified") if hasattr(self, "output_text_widgets") else None
        if widget is None:
            return
        widget.insert("end", d + "\n")
        widget.see("end")
        try:
            self.output_nb.select(self.output_tab_simplified)
        except Exception:
            pass

    def reprice(self, hint: str | None = None) -> None:
        if self.auto_reprice_enabled:
            if self._reprice_in_progress:
                return
            self.gen_quote(reuse_suggestions=True)
            return

        self._mark_quote_dirty(hint)

    def gen_quote(self, reuse_suggestions: bool = False) -> None:
        already_repricing = self._reprice_in_progress
        if not already_repricing:
            self._reprice_in_progress = True
        succeeded = False
        try:
            try:
                self.status_var.set("Generating quote…")
                self.update_idletasks()
            except Exception:
                pass
            if self.vars_df is None:
                self.vars_df = coerce_or_make_vars_df(None)
            for item_name, string_var in self.quote_vars.items():
                mask = self.vars_df["Item"] == item_name
                if mask.any():
                    self.vars_df.loc[mask, "Example Values / Options"] = string_var.get()

            self.apply_overrides(notify=False)

            try:
                ui_vars = {
                    str(row["Item"]): row["Example Values / Options"]
                    for _, row in self.vars_df.iterrows()
                }
            except Exception:
                ui_vars = {}

            llm_suggest = self.LLM_SUGGEST
            if self.llm_enabled.get() and llm_suggest is None:
                llm_suggest = self._ensure_llm_loaded()

            try:
                self._reset_llm_logs()
                client = None
                if self.llm_enabled.get():
                    client = self.get_llm_client(self.llm_model_path.get().strip() or None)
                res = compute_quote_from_df(
                    self.vars_df,
                    params=self.params,
                    rates=self.rates,
                    default_params=self.default_params_template,
                    default_rates=self.default_rates_template,
                    default_material_display=self.default_material_display,
                    material_vendor_csv=self.settings.get("material_vendor_csv", "") if isinstance(self.settings, dict) else "",
                    llm_enabled=self.llm_enabled.get(),
                    llm_model_path=self.llm_model_path.get().strip() or None,
                    llm_client=client,
                    geo=self.geo,
                    ui_vars=ui_vars,
                    quote_state=self.quote_state,
                    reuse_suggestions=reuse_suggestions,
                    llm_suggest=llm_suggest,

                )
            except ValueError as err:
                # Log full traceback for debugging ambiguous DataFrame truthiness, etc.
                import datetime
                import traceback
                try:
                    with open("debug.log", "a", encoding="utf-8") as f:
                        f.write(f"\n[{datetime.datetime.now().isoformat()}] Quote blocked (ValueError):\n")
                        f.write(traceback.format_exc())
                        f.write("\n")
                except Exception:
                    pass
                messagebox.showerror("Quote blocked", str(err))
                self.status_var.set("Quote blocked.")
                return
            except Exception as err:
                # Catch-all so failures surface in the UI and logs
                import datetime
                import traceback
                tb = traceback.format_exc()
                try:
                    with open("debug.log", "a", encoding="utf-8") as f:
                        f.write(f"\n[{datetime.datetime.now().isoformat()}] Quote error:\n{tb}\n")
                except Exception:
                    pass
                messagebox.showerror("Quote error", f"Unexpected error during pricing. Check debug.log.\n\n{err}")
                self.status_var.set("Quote failed.")
                return

            baseline_ctx = self.quote_state.baseline or {}
            suggestions_ctx = self.quote_state.suggestions or {}
            if baseline_ctx and suggestions_ctx:
                self.apply_llm_to_editor(suggestions_ctx, baseline_ctx)

            model_path = self.llm_model_path.get().strip()
            llm_explanation = get_llm_quote_explanation(res, model_path)
            if not isinstance(res, dict):
                res = {}
            simplified_report = render_quote(
                res,
                currency="$",
                show_zeros=False,
                llm_explanation=llm_explanation,
            )
            full_report = render_quote(
                res,
                currency="$",
                show_zeros=True,
                llm_explanation=llm_explanation,
            )

            for name, report_text in (
                ("simplified", simplified_report),
                ("full", full_report),
            ):
                widget = self.output_text_widgets.get(name)
                if widget is None:
                    continue
                widget.delete("1.0", "end")
                widget.insert("end", report_text, "rcol")

            try:
                self.output_nb.select(self.output_tab_simplified)
            except Exception:
                pass

            self.nb.select(self.tab_out)
            self.status_var.set(f"Quote Generated! Final Price: ${res.get('price', 0):,.2f}")
            succeeded = True
        finally:
            if succeeded:
                self._clear_quote_dirty()
            if not already_repricing:
                self._reprice_in_progress = False

# Helper: map our existing enrich_geo_* output to GEO__ keys
def _map_geo_to_double_underscore(g: dict) -> dict:
    g = dict(g or {})
    out = {}
    def getf(k):
        try:
            return float(g[k])
        except Exception:
            return None
    L = getf("GEO-01_Length_mm")
    W = getf("GEO-02_Width_mm")
    H = getf("GEO-03_Height_mm")
    if L is not None:
        out["GEO__BBox_X_mm"] = L
    if W is not None:
        out["GEO__BBox_Y_mm"] = W
    if H is not None:
        out["GEO__BBox_Z_mm"] = H
    dims = [d for d in [L, W, H] if d is not None]
    if dims:
        out["GEO__MaxDim_mm"] = max(dims)
        out["GEO__MinDim_mm"] = min(dims)
        out["GEO__Stock_Thickness_mm"] = min(dims)
    v = getf("GEO-Volume_mm3") or getf("GEO-Volume_mm3")  # keep both spellings if present
    if v is not None:
        out["GEO__Volume_mm3"] = v
    a = getf("GEO-SurfaceArea_mm2")
    if a is not None:
        out["GEO__SurfaceArea_mm2"] = a
    fc = getf("Feature_Face_Count") or getf("GEO_Face_Count")
    if fc is not None:
        out["GEO__Face_Count"] = fc
    wedm = getf("GEO_WEDM_PathLen_mm")
    if wedm is not None:
        out["GEO__WEDM_PathLen_mm"] = wedm
    # derived area/volume ratio if possible
    if a is not None and v:
        try:
            out["GEO__Area_to_Volume"] = a / v if v else 0.0
        except Exception:
            pass
    return out

def _collect_geo_features_from_df(df):
    geo = {}
    if df is None:
        return geo
    items = df["Item"].astype(str)
    m = items.str.startswith("GEO__", na=False)
    for _, row in df.loc[m].iterrows():
        k = str(row["Item"]).strip()
        try:
            value = row["Example Values / Options"]
            geo[k] = float(value) if value is not None else 0.0
        except Exception:
            continue
    return geo

def update_variables_df_with_geo(df, geo: dict):
    cols = ["Item", "Example Values / Options", "Data Type / Input Method"]
    for col in cols:
        if col not in df.columns:
            raise ValueError("Variables sheet missing column: " + col)
    items = df["Item"].astype(str)
    for k, v in geo.items():
        m = items.str.fullmatch(k, case=False)
        if m.any():
            df.loc[m, "Example Values / Options"] = v
            df.loc[m, "Data Type / Input Method"] = "number"
        else:
            df.loc[len(df)] = [k, v, "number"]
    return df

def _rule_based_overrides(geo: dict, params: dict, rates: dict):
    rp, rr = {}, {}
    thk = float(geo.get("GEO__Stock_Thickness_mm", 0.0) or 0.0)
    wedm = float(geo.get("GEO__WEDM_PathLen_mm", 0.0) or 0.0)
    if thk > 50:
        rp["OverheadPct"] = params.get("OverheadPct", 0.15) + 0.02
    if wedm > 500:
        rr["WireEDMRate"] = rates.get("WireEDMRate", 140.0) * 1.05
    return {"params": rp, "rates": rr}

def _run_llm_json_stub(prompt: str, model_path: str):
    return {"params": {}, "rates": {}}

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD Quoting Tool UI")
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print a JSON dump of relevant environment configuration and exit.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Initialise subsystems but do not launch the Tkinter GUI.",
    )
    return parser


def _main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.print_env:
        logger.info("Runtime environment:\n%s", json.dumps(describe_runtime_environment(), indent=2))
        return 0

    if args.no_gui:
        return 0

    pricing_registry = create_default_registry()
    pricing_engine = PricingEngine(pricing_registry)
    App(pricing_engine).mainloop()

    return 0


if __name__ == "__main__":
    exit_code = _main()
    if exit_code == 0:
        try:
            from cad_quoter.pricing import (
                PricingEngine,
                create_default_registry,
                ensure_material_backup_csv,
            )

            engine = PricingEngine(create_default_registry())
            csv_path = ensure_material_backup_csv()
            quote = engine.get_usd_per_kg(
                "aluminum", "usd_per_kg", vendor_csv=csv_path, providers=("vendor_csv",)
            )
            print(f"[smoke] aluminum ${quote.usd_per_kg:.2f}/kg via {quote.source}")
        except Exception as exc:  # pragma: no cover - smoke guard
            print(f"[smoke] pricing run failed: {exc}", file=sys.stderr)
            exit_code = 1
    sys.exit(exit_code)
