
# -*- coding: utf-8 -*-
# app_gui_occ_flow_v8_single_autollm.py
"""
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
import json, math, os, time, gc
from collections import Counter
from fractions import Fraction
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class AppEnvironment:
    """Runtime configuration extracted from environment variables.

    Centralising this logic makes it trivial for reviewers to audit which
    environment variables are consulted and how defaults are derived.  The
    configuration can also be serialised for diagnostics without touching
    global module state.
    """

    llm_debug_enabled: bool = False
    llm_debug_dir: Path = field(default_factory=Path)

    @classmethod
    def from_env(cls) -> "AppEnvironment":
        debug_enabled = bool(int(os.getenv("LLM_DEBUG", "1")))
        debug_dir_raw = os.getenv("LLM_DEBUG_DIR")
        debug_dir = Path(debug_dir_raw) if debug_dir_raw else Path(__file__).with_name("llm_debug")
        debug_dir.mkdir(exist_ok=True)
        return cls(llm_debug_enabled=debug_enabled, llm_debug_dir=debug_dir)


APP_ENV = AppEnvironment.from_env()


def describe_runtime_environment() -> dict[str, str]:
    """Return a redacted snapshot of runtime configuration for auditors."""

    info = {"llm_debug_enabled": str(APP_ENV.llm_debug_enabled)}
    info["llm_debug_dir"] = str(APP_ENV.llm_debug_dir)
    for key in ("QWEN_GGUF_PATH", "ODA_CONVERTER_EXE", "DWG2DXF_EXE", "METALS_API_KEY"):
        value = os.getenv(key)
        if not value:
            info[key.lower()] = ""
            continue
        if key.endswith("_KEY"):
            info[key.lower()] = "<redacted>"
        else:
            info[key.lower()] = value
    return info

import copy
import re
import sys
import textwrap
import tkinter as tk
import tkinter.font as tkfont
import urllib.request
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from OCP.TopAbs   import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp   import TopExp, TopExp_Explorer
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS   import TopoDS, TopoDS_Face, TopoDS_Shape
from OCP.BRep     import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface as _BAS
import pandas as pd

from llama_cpp import Llama  # type: ignore

try:
    from geo_read_more import build_geo_from_dxf as build_geo_from_dxf_path
except Exception:
    build_geo_from_dxf_path = None  # type: ignore[assignment]


def _normalize_lookup_key(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-z]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _ensure_scrap_pct(val) -> float:
    """
    Coerce UI/LLM scrap into a sane fraction in [0, 0.25].
    Accepts 15 (percent) or 0.15 (fraction).
    """

    try:
        x = float(val)
    except Exception:
        return 0.0
    if x > 1.0:  # looks like %
        x = x / 100.0
    if not (x >= 0.0 and math.isfinite(x)):
        return 0.0
    return min(0.25, max(0.0, x))


def _match_items_contains(items: pd.Series, pattern: str) -> pd.Series:
    """
    Case-insensitive regex match over Items.
    Convert capturing groups to non-capturing to avoid pandas warning.
    Fall back to literal if regex fails.
    """

    pat = _to_noncapturing(pattern) if "_to_noncapturing" in globals() else pattern
    try:
        return items.str.contains(pat, case=False, regex=True, na=False)
    except Exception:
        return items.str.contains(re.escape(pattern), case=False, regex=True, na=False)


def parse_llm_json(text: str):
    """
    Accepts raw model text. Strips ``` fences, grabs the first {...} block,
    returns dict or {}.
    """
    if not isinstance(text, str):
        return {}
    text2 = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text2, flags=re.DOTALL)
    if not m:
        return {}
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
        try:
            return json.loads(frag2)
        except Exception:
            return {}


SYSTEM_SUGGEST = """You are a manufacturing estimator.
Given GEO + baseline + bounds, propose bounded adjustments that improve realism.
ALWAYS return a complete JSON object with THESE KEYS, even if values are unchanged:
{
  "process_hour_multipliers": {"drilling": <float>, "milling": <float>},
  "process_hour_adders": {"inspection": <float>},
  "scrap_pct": <float>,           // fraction 0.00–0.25
  "setups": <int>,                // 1–4
  "fixture": "<string>",          // short phrase
  "notes": ["<short reason>"],
  "no_change_reason": "<why you kept near baseline, if so>"
}
All outputs MUST respect bounds in `bounds`.
Prefer small, explainable changes (±10–50%). Never leave fields out.
You may use payload["seed"] heuristics to bias adjustments when helpful."""


# Field mapping: LLM suggestion key -> (Editor label, to_editor, from_editor)
SUGG_TO_EDITOR = {
    "scrap_pct": (
        "Scrap Percent (%)",
        lambda f: f * 100.0,
        lambda s: float(s) / 100.0,
    ),
    "setups": (
        "Number of Milling Setups",
        int,
        int,
    ),
    ("process_hour_adders", "inspection"): (
        "In-Process Inspection Hours",
        float,
        float,
    ),
    "fixture": (
        "Fixture plan",
        str,
        str,
    ),
    ("add_pass_through", "Consumables /Hr"): (
        "Consumables /Hr Cost",
        float,
        float,
    ),
    ("add_pass_through", "Utilities"): (
        "Utilities Cost",
        float,
        float,
    ),
    ("add_pass_through", "Shipping"): (
        "Shipping Cost",
        float,
        float,
    ),
    "contingency_pct": (
        "ContingencyPct",
        float,
        float,
    ),
    "fixture_material_cost_delta": (
        "Fixture Material Cost",
        float,
        float,
    ),
}

# Multipliers are applied to computed process hours, then we push the new hours to editor fields.
PROC_MULT_TARGETS = {
    "drilling": ("CMM Run Time min", 60.0),
    "milling": ("Roughing Cycle Time", 1.0),
}

EDITOR_TO_SUGG = {spec[0]: key for key, spec in SUGG_TO_EDITOR.items()}
EDITOR_FROM_UI = {spec[0]: spec[2] for _, spec in SUGG_TO_EDITOR.items()}


VL_MODEL = r"D:\CAD_Quoting_Tool\models\qwen2.5-vl-7b-instruct-q4_k_m.gguf"
MM_PROJ = r"D:\CAD_Quoting_Tool\models\mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"


def load_qwen_vl(
    n_ctx: int = 8192,
    n_gpu_layers: int = 20,
    n_threads: int | None = None,
):
    """Load Qwen2.5-VL with vision projector configured for llama.cpp."""

    if n_threads is None:
        cpu_count = os.cpu_count() or 8
        n_threads = max(4, cpu_count // 2)

    try:
        llm = Llama(
            model_path=VL_MODEL,
            mmproj_path=MM_PROJ,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format="qwen2_vl",
            verbose=False,
        )
        _ = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Return JSON {\"ok\":true}."},
                {"role": "user", "content": "ping"},
            ],
            max_tokens=16,
            temperature=0,
        )
        return llm
    except Exception:
        if n_ctx > 4096:
            gc.collect()
            return load_qwen_vl(n_ctx=4096, n_gpu_layers=0, n_threads=n_threads)
        raise


def _auto_accept_suggestions(suggestions: dict[str, Any] | None) -> dict[str, Any]:
    accept: dict[str, Any] = {}
    if not isinstance(suggestions, dict):
        return accept
    for bucket in ("process_hour_multipliers", "process_hour_adders", "add_pass_through"):
        data = suggestions.get(bucket)
        if isinstance(data, dict) and data:
            accept[bucket] = {str(k): True for k in data.keys()}
    for scalar_key in ("scrap_pct", "contingency_pct", "setups", "fixture"):
        if scalar_key in suggestions:
            accept[scalar_key] = True
    if suggestions.get("notes"):
        accept["notes"] = True
    return accept



def bin_diams_mm(diams: Iterable[float | int | None], step: float = 0.1) -> Dict[float, int]:
    """Return a histogram of diameters rounded to ``step`` millimetres.

    ``None`` values and non-numeric entries are ignored.  The resulting
    dictionary is sorted by quantity descending (and diameter ascending when
    counts match) so callers can easily present the most common tool families
    first in downstream UI tables or LLM payloads.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    counter: Counter[float] = Counter()
    for value in diams:
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        counter[round(num / step) * step] += 1

    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


try:  # Optional dependency – ezdxf may be missing in some environments.
    import ezdxf  # type: ignore
except Exception as _ezdxf_exc:  # pragma: no cover - import guard
    ezdxf = None  # type: ignore
    _EZDXF_IMPORT_ERROR: Exception | None = _ezdxf_exc
else:  # pragma: no cover - exercised when ezdxf is installed
    _EZDXF_IMPORT_ERROR = None


_INSUNITS_TO_MM = {
    0: 1.0,
    1: 25.4,
    2: 304.8,
    3: 914.4,
    4: 1.0,
    5: 10.0,
    6: 1000.0,
    7: 1.0,
    8: 0.0254,
    9: 0.0000254,
    10: 1_000_000.0,
}


def _require_ezdxf() -> None:
    if ezdxf is None:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "ezdxf is required for DXF parsing but is not installed"
        ) from _EZDXF_IMPORT_ERROR


def _unit_scale(doc) -> float:
    insunits = int((doc.header.get("$INSUNITS", 4)) if doc else 4)
    return float(_INSUNITS_TO_MM.get(insunits, 1.0))


def read_dxf(path: str):
    """Load ``path`` with :mod:`ezdxf` and return the document object."""

    _require_ezdxf()
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return ezdxf.readfile(path)


def extract_entities(doc) -> Dict[str, Any]:
    """Return a deterministic feature summary for ``doc``."""

    scale = _unit_scale(doc)
    bbox_info = contours_from_polylines(doc, _scale=scale)
    holes = holes_from_circles(doc, _scale=scale)
    lines = text_harvest(doc)

    provenance: List[Dict[str, Any]] = []
    if bbox_info.get("bbox_mm"):
        provenance.append({"field": "bbox_mm", "source": "dxf_poly"})

    return {
        "bbox_mm": bbox_info.get("bbox_mm"),
        "perimeter_mm": bbox_info.get("perimeter_mm"),
        "hole_diams_mm": holes,
        "lines": lines,
        "provenance": provenance,
    }


def text_harvest(doc) -> List[str]:
    """Collect text strings from TEXT/MTEXT entities (including blocks)."""

    if doc is None:
        return []

    lines: List[str] = []
    msp = doc.modelspace()

    def _append_text(entity) -> None:
        if entity is None:
            return
        try:
            if entity.dxftype() == "MTEXT":
                text = entity.plain_text()
            else:
                text = entity.dxf.text
        except Exception:
            text = None
        if text:
            for frag in str(text).splitlines():
                frag = frag.strip()
                if frag:
                    lines.append(frag)

    for entity in msp:
        etype = entity.dxftype()
        if etype in {"TEXT", "MTEXT"}:
            _append_text(entity)
        elif etype == "INSERT":
            try:
                for sub in entity.virtual_entities():
                    if sub.dxftype() in {"TEXT", "MTEXT"}:
                        _append_text(sub)
            except Exception:
                continue

    return lines


def contours_from_polylines(
    doc,
    *,
    _scale: float | None = None,
) -> Dict[str, Any]:
    """Compute bounding box and perimeter from closed polylines."""

    if doc is None:
        return {"bbox_mm": None, "perimeter_mm": None}

    scale = float(_scale or _unit_scale(doc))
    msp = doc.modelspace()

    def _iter_points(entity) -> Sequence[Tuple[float, float]]:
        if entity.dxftype() == "LWPOLYLINE":
            return [
                (float(x) * scale, float(y) * scale)
                for x, y, *_ in entity.get_points("xy")
            ]
        if entity.dxftype() == "POLYLINE":
            return [
                (float(v.dxf.location.x) * scale, float(v.dxf.location.y) * scale)
                for v in entity.vertices()
            ]
        return []

    def _polygon_area(pts: Sequence[Tuple[float, float]]) -> float:
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def _perimeter(pts: Sequence[Tuple[float, float]]) -> float:
        if len(pts) < 2:
            return 0.0
        per = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            per += math.hypot(x2 - x1, y2 - y1)
        per += math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
        return per

    best_pts: Sequence[Tuple[float, float]] = []
    best_area = 0.0
    total_perimeter = 0.0

    for entity in msp:
        if entity.dxftype() not in {"LWPOLYLINE", "POLYLINE"}:
            continue
        closed = bool(getattr(entity, "closed", False))
        if entity.dxftype() == "POLYLINE":
            closed = bool(entity.is_closed)
        if not closed:
            continue
        pts = list(_iter_points(entity))
        if len(pts) < 3:
            continue
        area = abs(_polygon_area(pts))
        total_perimeter += _perimeter(pts)
        if area > best_area:
            best_pts = pts
            best_area = area

    if not best_pts:
        return {"bbox_mm": None, "perimeter_mm": None}

    xs = [p[0] for p in best_pts]
    ys = [p[1] for p in best_pts]
    bbox_mm = [max(xs) - min(xs), max(ys) - min(ys)]

    return {"bbox_mm": bbox_mm, "perimeter_mm": total_perimeter}


def holes_from_circles(doc, *, _scale: float | None = None) -> List[float]:
    """Return diameters (mm) for CIRCLE entities and full arcs."""

    if doc is None:
        return []

    scale = float(_scale or _unit_scale(doc))
    msp = doc.modelspace()
    holes: List[float] = []

    def _maybe_add(radius: float) -> None:
        if radius <= 0:
            return
        holes.append(2.0 * float(radius) * scale)

    for entity in msp:
        etype = entity.dxftype()
        if etype == "CIRCLE":
            _maybe_add(float(entity.dxf.radius))
        elif etype == "ARC":
            start = float(entity.dxf.start_angle)
            end = float(entity.dxf.end_angle)
            sweep = (end - start) % 360.0
            if math.isclose(sweep, 360.0, abs_tol=1e-3):
                _maybe_add(float(entity.dxf.radius))

    return holes


def read_step(path: str) -> "TopoDS_Shape":
    """Wrapper around :func:`read_step_shape` with extra validation."""

    shape = read_step_shape(str(path))
    if shape is None or shape.IsNull():
        raise RuntimeError("STEP produced an empty shape")
    return shape


def compute_mass_props(shape) -> Dict[str, float | None]:
    """Return volume/area mass properties for ``shape``."""

    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        raise ValueError("shape must be a valid TopoDS_Shape")

    surf_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf_props)
    area = float(surf_props.Mass())

    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol_props)
    volume = float(vol_props.Mass())

    return {"volume_mm3": volume, "area_mm2": area, "mass_kg": None}


def find_cylindrical_faces(shape) -> List[Dict[str, Any]]:
    """Return cylindrical faces with basic geometric descriptors."""

    results: List[Dict[str, Any]] = []
    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        return results

    try:
        from OCP.GeomAbs import GeomAbs_Cylinder  # type: ignore
    except Exception:
        from OCC.Core.GeomAbs import GeomAbs_Cylinder  # type: ignore

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        try:
            face = ensure_face(explorer.Current())
        except Exception:
            explorer.Next()
            continue

        try:
            surf_adaptor = _BAS(face)
            st = surf_adaptor.GetType()
        except Exception:
            explorer.Next()
            continue

        if st != GeomAbs_Cylinder:
            explorer.Next()
            continue

        try:
            cylinder = surf_adaptor.Cylinder()
            axis = cylinder.Axis()
            loc = axis.Location()
            direction = axis.Direction()
            results.append(
                {
                    "face": face,
                    "r_mm": float(cylinder.Radius()),
                    "axis": [loc.X(), loc.Y(), loc.Z()],
                    "dir": [direction.X(), direction.Y(), direction.Z()],
                }
            )
        except Exception:
            pass

        explorer.Next()

    return results


def detect_through_holes(shape, cyl_faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-effort through-hole classification from cylindrical faces."""

    holes: List[Dict[str, Any]] = []
    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        return holes

    for idx, entry in enumerate(cyl_faces):
        radius = float(entry.get("r_mm", 0.0))
        axis = entry.get("axis")
        direction = entry.get("dir")
        holes.append(
            {
                "dia_mm": radius * 2.0 if radius else None,
                "depth_mm": None,
                "thru": None,
                "axis": {"origin": axis, "dir": direction},
                "face_id": idx,
                "source": "step_cyl",
            }
        )

    return holes


def find_planar_pockets(shape) -> List[Dict[str, Any]]:
    """Return planar faces as pocket placeholders with area estimates."""

    pockets: List[Dict[str, Any]] = []
    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        return pockets

    try:
        from OCP.GeomAbs import GeomAbs_Plane  # type: ignore
    except Exception:
        from OCC.Core.GeomAbs import GeomAbs_Plane  # type: ignore

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        try:
            face = ensure_face(explorer.Current())
        except Exception:
            explorer.Next()
            continue

        try:
            surf_adaptor = _BAS(face)
            st = surf_adaptor.GetType()
        except Exception:
            explorer.Next()
            continue

        if st != GeomAbs_Plane:
            explorer.Next()
            continue

        try:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            pockets.append(
                {
                    "depth_mm": None,
                    "floor_area_mm2": float(props.Mass()),
                    "source": "step_planar",
                }
            )
        except Exception:
            pass

        explorer.Next()

    return pockets


def render_step_thumbs(shape, out_dir: str) -> Dict[str, str]:
    raise RuntimeError("STEP thumbnail rendering is not supported in this environment")


@dataclass
class QuoteState:
    geo: dict = field(default_factory=dict)
    ui_vars: dict = field(default_factory=dict)
    rates: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    llm_raw: dict = field(default_factory=dict)
    suggestions: dict = field(default_factory=dict)
    user_overrides: dict = field(default_factory=dict)
    effective: dict = field(default_factory=dict)
    effective_sources: dict = field(default_factory=dict)
    accept_llm: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)
    material_source: str | None = None
    guard_context: dict = field(default_factory=dict)


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



def build_suggest_payload(geo, baseline, rates, bounds) -> dict:
    geo = geo or {}
    baseline = baseline or {}
    rates = rates or {}
    bounds = bounds or {}

    derived = geo.get("derived") or {}
    hole_bins = derived.get("hole_bins") or {}
    hole_bins_top = {}
    if isinstance(hole_bins, dict):
        hole_bins_top = dict(sorted(hole_bins.items(), key=lambda kv: -kv[1])[:8])

    raw_thickness = geo.get("thickness_mm")
    thickness_mm = None
    if isinstance(raw_thickness, dict):
        for key in ("value", "mm", "thickness_mm"):
            if raw_thickness.get(key) is not None:
                try:
                    thickness_mm = float(raw_thickness.get(key))
                    break
                except Exception:
                    continue
    else:
        try:
            thickness_mm = float(raw_thickness)
        except Exception:
            thickness_mm = None
    if thickness_mm is None:
        try:
            thickness_mm = float(geo.get("thickness"))
        except Exception:
            thickness_mm = None

    material_val = geo.get("material")
    if isinstance(material_val, dict):
        material_name = (
            material_val.get("name")
            or material_val.get("display")
            or material_val.get("material")
        )
    else:
        material_name = material_val
    if not material_name:
        material_name = "Steel"

    hole_count_val = _as_float_or_none(geo.get("hole_count"))
    if hole_count_val is None:
        hole_count_val = _as_float_or_none(derived.get("hole_count"))
    hole_count = int(hole_count_val or 0)

    tap_qty = derived.get("tap_qty")
    try:
        tap_qty = int(tap_qty)
    except Exception:
        tap_qty = 0

    cbore_qty = derived.get("cbore_qty")
    try:
        cbore_qty = int(cbore_qty)
    except Exception:
        cbore_qty = 0

    csk_qty = derived.get("csk_qty")
    try:
        csk_qty = int(csk_qty)
    except Exception:
        csk_qty = 0

    tap_minutes_hint = _as_float_or_none(derived.get("tap_minutes_hint"))
    cbore_minutes_hint = _as_float_or_none(derived.get("cbore_minutes_hint"))
    csk_minutes_hint = _as_float_or_none(derived.get("csk_minutes_hint"))
    tap_class_counts = derived.get("tap_class_counts") if isinstance(derived.get("tap_class_counts"), dict) else {}
    tap_details = derived.get("tap_details") if isinstance(derived.get("tap_details"), list) else []
    npt_qty = 0
    try:
        npt_qty = int(derived.get("npt_qty") or 0)
    except Exception:
        npt_qty = 0
    inference_knobs = derived.get("inference_knobs") if isinstance(derived.get("inference_knobs"), dict) else {}
    has_ldr_notes = bool(derived.get("has_ldr_notes"))
    max_hole_depth_in = _as_float_or_none(derived.get("max_hole_depth_in"))
    plate_area_in2 = _as_float_or_none(derived.get("plate_area_in2"))
    finish_flags_raw = derived.get("finish_flags")
    if isinstance(finish_flags_raw, (list, tuple, set)):
        finish_flags = [str(flag).strip() for flag in finish_flags_raw if isinstance(flag, str) and flag.strip()]
    elif isinstance(finish_flags_raw, str) and finish_flags_raw.strip():
        finish_flags = [finish_flags_raw.strip()]
    else:
        finish_flags = []
    has_tight_tol = bool(derived.get("has_tight_tol"))
    stock_guess_val = derived.get("stock_guess")
    stock_guess = str(stock_guess_val).strip() if isinstance(stock_guess_val, str) and stock_guess_val.strip() else None

    seed = {
        "suggest_drilling_if_many_holes": hole_count >= 50,
        "suggest_setups_if_from_back_ops": bool(derived.get("needs_back_face")),
        "nudge_drilling_for_thickness": bool(thickness_mm and thickness_mm > 12.0),
        "add_inspection_if_many_taps": tap_qty >= 8,
        "add_milling_if_cbore_present": cbore_qty >= 2,
        "plate_with_back_ops": bool((geo.get("meta") or {}).get("is_2d_plate") and derived.get("needs_back_face")),
    }
    if has_ldr_notes:
        seed["has_leader_notes"] = True
    if max_hole_depth_in is not None:
        seed["max_hole_depth_in"] = max_hole_depth_in
    if plate_area_in2 is not None:
        seed["plate_area_in2"] = plate_area_in2
    if has_tight_tol:
        seed["has_tight_tol"] = True
    if finish_flags:
        seed["finish_flags"] = finish_flags[:6]
    if stock_guess:
        seed["stock_guess"] = stock_guess
    if tap_minutes_hint:
        seed["tapping_minutes_hint"] = tap_minutes_hint
    if cbore_minutes_hint:
        seed["counterbore_minutes_hint"] = cbore_minutes_hint
    if csk_minutes_hint:
        seed["countersink_minutes_hint"] = csk_minutes_hint
    if tap_class_counts:
        seed["tap_class_counts"] = tap_class_counts
    if tap_details:
        seed["tap_details"] = tap_details[:10]
    if npt_qty:
        seed["npt_qty"] = npt_qty
    if inference_knobs:
        seed["inference_knobs"] = inference_knobs

    return {
        "purpose": "quote_suggestions",
        "geo": {
            "is_2d_plate": bool((geo.get("meta") or {}).get("is_2d_plate", True)),
            "hole_count": hole_count,
            "tap_qty": tap_qty,
            "cbore_qty": cbore_qty,
            "csk_qty": csk_qty,
            "hole_bins_top": hole_bins_top,
            "thickness_mm": thickness_mm,
            "material": material_name,
            "bbox_mm": geo.get("bbox_mm"),
            "tap_minutes_hint": tap_minutes_hint,
            "cbore_minutes_hint": cbore_minutes_hint,
            "csk_minutes_hint": csk_minutes_hint,
            "tap_class_counts": tap_class_counts,
            "tap_details": tap_details,
            "npt_qty": npt_qty,
            "inference_knobs": inference_knobs,
            "has_leader_notes": has_ldr_notes,
            "max_hole_depth_in": max_hole_depth_in,
            "plate_area_in2": plate_area_in2,
            "has_tight_tol": has_tight_tol,
            "finish_flags": finish_flags,
            "stock_guess": stock_guess,
        },
        "baseline": {
            "process_hours": baseline.get("process_hours"),
            "scrap_pct": baseline.get("scrap_pct", 0.0),
            "pass_through": baseline.get("pass_through", {}),
        },
        "rates": rates,
        "bounds": bounds,
        "seed": seed,
    }


def run_llm_suggestions(LLM, payload: dict) -> tuple[dict, str, dict]:
    parsed, raw, usage = LLM.ask_json(
        system_prompt=SYSTEM_SUGGEST,
        user_prompt=json.dumps(payload, indent=2),
        temperature=0.3,
        max_tokens=512,
        context=payload,
        params={"top_p": 0.9},
    )
    if not parsed:
        parsed = parse_llm_json(raw)
    if not parsed:
        baseline = payload.get("baseline") or {}
        parsed = {
            "process_hour_multipliers": {"drilling": 1.0, "milling": 1.0},
            "process_hour_adders": {"inspection": 0.0},
            "scrap_pct": baseline.get("scrap_pct", 0.0),
            "setups": int(baseline.get("setups", 1) or 1),
            "fixture": baseline.get("fixture", "standard") or "standard",
            "notes": ["no parse; using baseline"],
            "no_change_reason": "fallback",
        }
    return parsed, raw, usage or {}


def sanitize_suggestions(s: dict, bounds: dict) -> dict:
    bounds = bounds or {}

    def clamp(value, lo, hi):
        try:
            return max(lo, min(hi, float(value)))
        except Exception:
            return lo

    mults: dict[str, float] = {}
    for proc, val in (s.get("process_hour_multipliers") or {}).items():
        try:
            mults[str(proc)] = clamp(val, bounds.get("mult_min", 0.5), bounds.get("mult_max", 3.0))
        except Exception:
            continue

    adders: dict[str, float] = {}
    for proc, val in (s.get("process_hour_adders") or {}).items():
        try:
            adders[str(proc)] = clamp(val, 0.0, bounds.get("adder_max_hr", 5.0))
        except Exception:
            continue

    scrap = clamp(
        s.get("scrap_pct", 0.0),
        bounds.get("scrap_min", 0.0),
        bounds.get("scrap_max", 0.25),
    )

    try:
        setups = int(s.get("setups", 1))
    except Exception:
        setups = 1
    setups = max(1, min(4, setups))

    fixture = s.get("fixture", "standard")
    fixture = str(fixture)[:80] if fixture is not None else "standard"

    notes_raw = s.get("notes") or []
    notes: list[str] = []
    for note in notes_raw:
        if not isinstance(note, str):
            continue
        cleaned = note.strip()
        if cleaned:
            notes.append(cleaned[:120])

    no_change = s.get("no_change_reason") or ""
    no_change = str(no_change)

    return {
        "process_hour_multipliers": mults or {"drilling": 1.0, "milling": 1.0},
        "process_hour_adders": adders or {"inspection": 0.0},
        "scrap_pct": scrap,
        "setups": setups,
        "fixture": fixture,
        "notes": notes,
        "no_change_reason": no_change,
    }


def apply_suggestions(baseline: dict, s: dict) -> dict:
    eff = copy.deepcopy(baseline or {})
    base_hours = eff.get("process_hours") if isinstance(eff.get("process_hours"), dict) else {}
    ph = {k: float(_as_float_or_none(v) or 0.0) for k, v in base_hours.items()}

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

def _nested_get(data: dict | None, path: Tuple[str, ...], default: Any = None) -> Any:
    cur = data or {}
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _nested_set(data: dict, path: Tuple[str, ...], value: Any) -> None:
    cur = data
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _ensure_nested_bool(data: dict, path: Tuple[str, ...], default: bool = False) -> bool:
    cur = data
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    leaf = cur.get(path[-1])
    if not isinstance(leaf, bool):
        cur[path[-1]] = bool(default)
    return bool(cur[path[-1]])


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


def overrides_to_suggestions(overrides: dict | None) -> dict:
    overrides = overrides or {}
    suggestions: dict[str, Any] = {}
    if isinstance(overrides.get("process_hour_multipliers"), dict):
        suggestions["process_hour_multipliers"] = dict(overrides["process_hour_multipliers"])
    if isinstance(overrides.get("process_hour_adders"), dict):
        suggestions["process_hour_adders"] = dict(overrides["process_hour_adders"])
    if isinstance(overrides.get("add_pass_through"), dict):
        suggestions["add_pass_through"] = dict(overrides["add_pass_through"])
    if overrides.get("scrap_pct_override") is not None:
        suggestions["scrap_pct"] = overrides.get("scrap_pct_override")
    if overrides.get("fixture_material_cost_delta") is not None:
        suggestions["fixture_material_cost_delta"] = overrides.get("fixture_material_cost_delta")
    if overrides.get("contingency_pct_override") is not None:
        suggestions["contingency_pct"] = overrides.get("contingency_pct_override")
    setup_plan = overrides.get("setup_recommendation")
    if isinstance(setup_plan, dict):
        if setup_plan.get("setups") is not None:
            suggestions["setups"] = setup_plan.get("setups")
        if setup_plan.get("fixture") is not None:
            suggestions["fixture"] = setup_plan.get("fixture")
    elif overrides.get("setups") is not None:
        suggestions["setups"] = overrides.get("setups")
    if overrides.get("fixture") is not None:
        suggestions["fixture"] = overrides.get("fixture")
    if isinstance(overrides.get("notes"), list):
        suggestions["notes"] = list(overrides["notes"])
    return suggestions


def suggestions_to_overrides(suggestions: dict | None) -> dict:
    suggestions = suggestions or {}
    out: dict[str, Any] = {}
    phm = suggestions.get("process_hour_multipliers")
    if isinstance(phm, dict):
        out["process_hour_multipliers"] = dict(phm)
    pha = suggestions.get("process_hour_adders")
    if isinstance(pha, dict):
        out["process_hour_adders"] = dict(pha)
    apt = suggestions.get("add_pass_through")
    if isinstance(apt, dict):
        out["add_pass_through"] = dict(apt)
    if suggestions.get("scrap_pct") is not None:
        out["scrap_pct_override"] = suggestions.get("scrap_pct")
    if suggestions.get("fixture_material_cost_delta") is not None:
        out["fixture_material_cost_delta"] = suggestions.get("fixture_material_cost_delta")
    if suggestions.get("contingency_pct") is not None:
        out["contingency_pct_override"] = suggestions.get("contingency_pct")
    setups = suggestions.get("setups")
    fixture = suggestions.get("fixture")
    if setups is not None or fixture is not None:
        out["setup_recommendation"] = {}
        if setups is not None:
            out["setup_recommendation"]["setups"] = setups
        if fixture is not None:
            out["setup_recommendation"]["fixture"] = fixture
    if isinstance(suggestions.get("notes"), list):
        out["notes"] = list(suggestions["notes"])
    return out


def _collect_process_keys(*dicts: Iterable[dict]) -> set[str]:
    keys: set[str] = set()
    for d in dicts:
        if isinstance(d, dict):
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
    suggestions = suggestions or {}
    overrides = overrides or {}
    guard_ctx = guard_ctx or {}

    bounds = baseline.get("_bounds") if isinstance(baseline, dict) else None
    if isinstance(bounds, dict):
        bounds = {str(k): v for k, v in bounds.items()}
    else:
        bounds = {}

    def _clamp(value: float, kind: str, label: str, source: str) -> tuple[float, bool]:
        clamped = value
        changed = False
        if kind == "multiplier":
            mult_min = _as_float_or_none(bounds.get("mult_min")) or 0.5
            mult_max = _as_float_or_none(bounds.get("mult_max")) or 3.0
            clamped = max(mult_min, min(mult_max, float(value)))
        elif kind == "adder":
            adder_max = _as_float_or_none(bounds.get("adder_max_hr")) or 5.0
            clamped = max(0.0, min(adder_max, float(value)))
        elif kind == "scrap":
            scrap_min = max(0.0, _as_float_or_none(bounds.get("scrap_min")) or 0.0)
            scrap_max = _as_float_or_none(bounds.get("scrap_max")) or 0.25
            clamped = max(scrap_min, min(scrap_max, float(value)))
        elif kind == "setups":
            clamped = int(max(1, min(4, round(float(value)))))
        if not math.isclose(float(clamped), float(value), rel_tol=1e-6, abs_tol=1e-6):
            note = f"{label} {float(value):.3f} → {float(clamped):.3f} ({source})"
            clamp_notes.append(note)
            changed = True
        return clamped, changed

    eff = copy.deepcopy(baseline)
    eff.pop("_bounds", None)
    clamp_notes: list[str] = []
    source_tags: dict[str, Any] = {}

    baseline_hours_raw = baseline.get("process_hours") if isinstance(baseline.get("process_hours"), dict) else {}
    baseline_hours: dict[str, float] = {}
    for proc, hours in (baseline_hours_raw or {}).items():
        val = _as_float_or_none(hours)
        if val is not None:
            baseline_hours[str(proc)] = float(val)

    sugg_mult = suggestions.get("process_hour_multipliers") if isinstance(suggestions.get("process_hour_multipliers"), dict) else {}
    over_mult = overrides.get("process_hour_multipliers") if isinstance(overrides.get("process_hour_multipliers"), dict) else {}
    mult_keys = sorted(_collect_process_keys(baseline_hours, sugg_mult, over_mult))
    final_hours: dict[str, float] = dict(baseline_hours)
    final_mults: dict[str, float] = {}
    mult_sources: dict[str, str] = {}
    for proc in mult_keys:
        base_hr = baseline_hours.get(proc, 0.0)
        source = "baseline"
        val = 1.0
        if proc in over_mult and over_mult[proc] is not None:
            cand = _as_float_or_none(over_mult.get(proc))
            if cand is not None:
                val = float(cand)
                val, _ = _clamp(val, "multiplier", f"multiplier[{proc}]", "user override")
                source = "user"
        elif proc in sugg_mult and sugg_mult[proc] is not None:
            cand = _as_float_or_none(sugg_mult.get(proc))
            if cand is not None:
                val = float(cand)
                val, _ = _clamp(val, "multiplier", f"multiplier[{proc}]", "LLM")
                source = "llm"
        final_mults[proc] = float(val)
        mult_sources[proc] = source
        final_hours[proc] = float(base_hr) * float(val)

    sugg_add = suggestions.get("process_hour_adders") if isinstance(suggestions.get("process_hour_adders"), dict) else {}
    over_add = overrides.get("process_hour_adders") if isinstance(overrides.get("process_hour_adders"), dict) else {}
    add_keys = sorted(_collect_process_keys(sugg_add, over_add))
    final_adders: dict[str, float] = {}
    add_sources: dict[str, str] = {}
    for proc in add_keys:
        source = "baseline"
        add_val = 0.0
        if proc in over_add and over_add[proc] is not None:
            cand = _as_float_or_none(over_add.get(proc))
            if cand is not None:
                add_val = float(cand)
                add_val, _ = _clamp(add_val, "adder", f"adder[{proc}]", "user override")
                source = "user"
        elif proc in sugg_add and sugg_add[proc] is not None:
            cand = _as_float_or_none(sugg_add.get(proc))
            if cand is not None:
                add_val = float(cand)
                add_val, _ = _clamp(add_val, "adder", f"adder[{proc}]", "LLM")
                source = "llm"
        if not math.isclose(add_val, 0.0, abs_tol=1e-9):
            final_adders[proc] = add_val
            final_hours[proc] = final_hours.get(proc, 0.0) + add_val
        add_sources[proc] = source

    sugg_pass = suggestions.get("add_pass_through") if isinstance(suggestions.get("add_pass_through"), dict) else {}
    over_pass = overrides.get("add_pass_through") if isinstance(overrides.get("add_pass_through"), dict) else {}
    pass_keys = sorted(_collect_process_keys(sugg_pass, over_pass))
    final_pass: dict[str, float] = {}
    pass_sources: dict[str, str] = {}
    for key in pass_keys:
        source = "baseline"
        val = 0.0
        if key in over_pass and over_pass[key] is not None:
            cand = _as_float_or_none(over_pass.get(key))
            if cand is not None:
                val = float(cand)
                source = "user"
        elif key in sugg_pass and sugg_pass[key] is not None:
            cand = _as_float_or_none(sugg_pass.get(key))
            if cand is not None:
                val = float(cand)
                source = "llm"
        if not math.isclose(val, 0.0, abs_tol=1e-9):
            final_pass[key] = val
        pass_sources[key] = source
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
        cand = _as_float_or_none(scrap_user)
        if cand is not None:
            scrap_val = float(cand)
            scrap_val, _ = _clamp(scrap_val, "scrap", "scrap_pct", "user override")
            scrap_source = "user"
    elif scrap_sugg is not None:
        cand = _as_float_or_none(scrap_sugg)
        if cand is not None:
            scrap_val = float(cand)
            scrap_val, _ = _clamp(scrap_val, "scrap", "scrap_pct", "LLM")
            scrap_source = "llm"
    eff["scrap_pct"] = float(scrap_val)
    source_tags["scrap_pct"] = scrap_source

    fixture_delta_user = overrides.get("fixture_material_cost_delta")
    fixture_delta_sugg = suggestions.get("fixture_material_cost_delta")
    fixture_delta_source = "baseline"
    fixture_delta_val = None
    if fixture_delta_user is not None:
        cand = _as_float_or_none(fixture_delta_user)
        if cand is not None:
            fixture_delta_val = float(cand)
            fixture_delta_source = "user"
    elif fixture_delta_sugg is not None:
        cand = _as_float_or_none(fixture_delta_sugg)
        if cand is not None:
            fixture_delta_val = float(cand)
            fixture_delta_source = "llm"
    if fixture_delta_val is not None:
        eff["fixture_material_cost_delta"] = fixture_delta_val
    source_tags["fixture_material_cost_delta"] = fixture_delta_source

    contingency_base = baseline.get("contingency_pct")
    contingency_user = overrides.get("contingency_pct") or overrides.get("contingency_pct_override")
    contingency_sugg = suggestions.get("contingency_pct")
    contingency_source = "baseline"
    contingency_val = contingency_base if contingency_base is not None else None
    if contingency_user is not None:
        cand = _as_float_or_none(contingency_user)
        if cand is not None:
            contingency_val = float(cand)
            contingency_val, _ = _clamp(contingency_val, "scrap", "contingency_pct", "user override")
            contingency_source = "user"
    elif contingency_sugg is not None:
        cand = _as_float_or_none(contingency_sugg)
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
        cand = _as_float_or_none(setups_user)
        if cand is not None:
            setups_val, _ = _clamp(cand, "setups", "setups", "user override")
            setups_source = "user"
    elif setups_sugg is not None:
        cand = _as_float_or_none(setups_sugg)
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

    final_hours_dict = eff.get("process_hours") if isinstance(eff.get("process_hours"), dict) else {}
    hole_count_guard = _coerce_float_or_none(guard_ctx.get("hole_count"))
    try:
        hole_count_guard_int = int(round(float(hole_count_guard))) if hole_count_guard is not None else 0
    except Exception:
        hole_count_guard_int = 0
    min_sec_per_hole = _as_float_or_none(guard_ctx.get("min_sec_per_hole"))
    min_sec_per_hole = float(min_sec_per_hole) if min_sec_per_hole is not None else 9.0
    if hole_count_guard_int > 0 and isinstance(final_hours_dict, dict) and "drilling" in final_hours_dict:
        current_drill = _as_float_or_none(final_hours_dict.get("drilling"))
        if current_drill is not None:
            drill_floor_hr = (hole_count_guard_int * min_sec_per_hole) / 3600.0
            if current_drill < drill_floor_hr - 1e-6:
                clamp_notes.append(
                    f"process_hours[drilling] {current_drill:.3f} → {drill_floor_hr:.3f} (guardrail)"
                )
                final_hours_dict["drilling"] = drill_floor_hr

    tap_qty_guard = _coerce_float_or_none(guard_ctx.get("tap_qty"))
    try:
        tap_qty_guard_int = int(round(float(tap_qty_guard))) if tap_qty_guard is not None else 0
    except Exception:
        tap_qty_guard_int = 0
    min_min_per_tap = _as_float_or_none(guard_ctx.get("min_min_per_tap"))
    min_min_per_tap = float(min_min_per_tap) if min_min_per_tap is not None else 0.2
    if tap_qty_guard_int > 0 and isinstance(final_hours_dict, dict) and "tapping" in final_hours_dict:
        current_tap = _as_float_or_none(final_hours_dict.get("tapping"))
        if current_tap is not None:
            tap_floor_hr = (tap_qty_guard_int * min_min_per_tap) / 60.0
            if current_tap < tap_floor_hr - 1e-6:
                clamp_notes.append(
                    f"process_hours[tapping] {current_tap:.3f} → {tap_floor_hr:.3f} (guardrail)"
                )
                final_hours_dict["tapping"] = tap_floor_hr

    if guard_ctx.get("needs_back_face"):
        current_setups = eff.get("setups")
        setups_val = _as_float_or_none(current_setups)
        try:
            setups_int = int(round(float(setups_val))) if setups_val is not None else int(current_setups)
        except Exception:
            try:
                setups_int = int(current_setups)
            except Exception:
                setups_int = 0
        if setups_int < 2:
            eff["setups"] = 2
            clamp_notes.append(f"setups {setups_int} → 2 (back-side guardrail)")
            source_tags["setups"] = "guardrail"

    finish_flags_ctx = guard_ctx.get("finish_flags")
    baseline_pass_ctx = guard_ctx.get("baseline_pass_through") if isinstance(guard_ctx.get("baseline_pass_through"), dict) else {}
    finish_pass_key = str(guard_ctx.get("finish_pass_key") or "Outsourced Vendors")
    finish_floor = _as_float_or_none(guard_ctx.get("finish_cost_floor"))
    finish_floor = float(finish_floor) if finish_floor is not None else 50.0
    if finish_flags_ctx and finish_floor > 0:
        combined_pass: dict[str, float] = {}
        for key, value in baseline_pass_ctx.items():
            val = _as_float_or_none(value)
            if val is not None:
                combined_pass[str(key)] = float(val)
        for key, value in (final_pass.items() if isinstance(final_pass, dict) else []):
            val = _as_float_or_none(value)
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

    for scalar_key in ("scrap_pct", "contingency_pct", "setups", "fixture"):
        if scalar_key in suggestions and accept.get(scalar_key):
            applied[scalar_key] = suggestions.get(scalar_key)

    if "notes" in suggestions:
        applied["notes"] = suggestions.get("notes")

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
    inner_geo_ctx = geo_ctx.get("geo") if isinstance(geo_ctx.get("geo"), dict) else {}
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
            current = _as_float_or_none(eff_hours.get("drilling")) or 0.0
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
def build_narrative(state: QuoteState) -> str:
    g = state.geo or {}
    b = state.baseline or {}
    e = state.effective or {}

    def _as_int(val):
        try:
            return int(float(val))
        except Exception:
            return 0

    holes = _as_int(g.get("hole_count"))
    if holes <= 0:
        holes = _as_int(g.get("hole_count_override"))
    if holes <= 0:
        diam_list = g.get("hole_diams_mm")
        if isinstance(diam_list, (list, tuple)):
            holes = len(diam_list)

    thick_mm = _as_float_or_none(g.get("thickness_mm"))
    if thick_mm is None:
        thick_mm = _as_float_or_none(state.ui_vars.get("Thickness (in)"))
        if thick_mm is not None:
            thick_mm *= 25.4
    mat = g.get("material") or state.ui_vars.get("Material") or "material"
    mat = str(mat).title()

    setups = int(_as_float_or_none(e.get("setups")) or _as_float_or_none(b.get("setups")) or 1)
    scrap_pct = round(100.0 * float(_as_float_or_none(e.get("scrap_pct")) or 0.0), 1)

    proc_hours = e.get("process_hours") if isinstance(e.get("process_hours"), dict) else {}
    top = []
    for name, hours in sorted(proc_hours.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        try:
            top.append(f"{name} {float(hours):.2f} h")
        except Exception:
            continue
    top_text = ", ".join(top) if top else "Baseline machining"

    deltas: list[str] = []
    base_hours = b.get("process_hours") if isinstance(b.get("process_hours"), dict) else {}
    for proc, base_hr in base_hours.items():
        try:
            base_val = float(base_hr)
            new_val = float(proc_hours.get(proc, 0.0))
        except Exception:
            continue
        diff = new_val - base_val
        if abs(diff) >= 0.01:
            sign = "↑" if diff > 0 else "↓"
            deltas.append(f"{proc} {sign}{abs(diff):.2f} h")
    delta_text = ", ".join(deltas)

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
        parts.append(f"Adjustments vs baseline: {delta_text}.")

    material_source = state.material_source or "shop defaults"
    parts.append(f"Material priced via {material_source}; scrap {scrap_pct}% applied.")

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
    passes = effective.get("add_pass_through") if isinstance(effective.get("add_pass_through"), dict) else {}
    if passes:
        cleaned_pass = {k: float(v) for k, v in passes.items() if v is not None and not math.isclose(float(v), 0.0, abs_tol=1e-6)}
        if cleaned_pass:
            out["add_pass_through"] = cleaned_pass
    scrap_eff = effective.get("scrap_pct")
    scrap_base = baseline.get("scrap_pct")
    if scrap_eff is not None and (scrap_base is None or not math.isclose(float(scrap_eff), float(scrap_base or 0.0), abs_tol=1e-6)):
        out["scrap_pct_override"] = float(scrap_eff)
    fixture_delta = effective.get("fixture_material_cost_delta")
    if fixture_delta is not None and not math.isclose(float(fixture_delta), 0.0, abs_tol=1e-6):
        out["fixture_material_cost_delta"] = float(fixture_delta)
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

    for key in ("scrap_pct", "fixture_material_cost_delta", "contingency_pct", "setups", "fixture"):
        if key in suggestions and not isinstance(accept.get(key), bool):
            accept[key] = False
        if key not in suggestions and key in accept and not isinstance(accept.get(key), dict):
            # keep user toggles if overrides exist even without suggestions
            continue


def iter_suggestion_rows(state: QuoteState) -> list[dict]:
    rows: list[dict] = []
    baseline = state.baseline or {}
    suggestions = state.suggestions or {}
    overrides = state.user_overrides or {}
    effective = state.effective or {}
    sources = state.effective_sources or {}
    accept = state.accept_llm or {}

    baseline_hours_raw = baseline.get("process_hours") if isinstance(baseline.get("process_hours"), dict) else {}
    baseline_hours = {}
    for key, value in (baseline_hours_raw or {}).items():
        try:
            if abs(float(value)) > 1e-6:
                baseline_hours[key] = float(value)
        except Exception:
            continue
    sugg_mult = suggestions.get("process_hour_multipliers") if isinstance(suggestions.get("process_hour_multipliers"), dict) else {}
    over_mult = overrides.get("process_hour_multipliers") if isinstance(overrides.get("process_hour_multipliers"), dict) else {}
    eff_mult = effective.get("process_hour_multipliers") if isinstance(effective.get("process_hour_multipliers"), dict) else {}
    src_mult = sources.get("process_hour_multipliers") if isinstance(sources.get("process_hour_multipliers"), dict) else {}
    keys_mult = sorted(_collect_process_keys(baseline_hours, sugg_mult, over_mult))
    for key in keys_mult:
        rows.append({
            "path": ("process_hour_multipliers", key),
            "label": f"Process × {key}",
            "kind": "multiplier",
            "baseline": 1.0,
            "llm": sugg_mult.get(key),
            "user": over_mult.get(key),
            "accept": bool((accept.get("process_hour_multipliers") or {}).get(key)),
            "effective": eff_mult.get(key, 1.0),
            "source": src_mult.get(key, "baseline"),
        })

    sugg_add = suggestions.get("process_hour_adders") if isinstance(suggestions.get("process_hour_adders"), dict) else {}
    over_add = overrides.get("process_hour_adders") if isinstance(overrides.get("process_hour_adders"), dict) else {}
    eff_add = effective.get("process_hour_adders") if isinstance(effective.get("process_hour_adders"), dict) else {}
    src_add = sources.get("process_hour_adders") if isinstance(sources.get("process_hour_adders"), dict) else {}
    keys_add = sorted(_collect_process_keys(baseline_hours, sugg_add, over_add))
    for key in keys_add:
        rows.append({
            "path": ("process_hour_adders", key),
            "label": f"Process +hr {key}",
            "kind": "hours",
            "baseline": 0.0,
            "llm": sugg_add.get(key),
            "user": over_add.get(key),
            "accept": bool((accept.get("process_hour_adders") or {}).get(key)),
            "effective": eff_add.get(key, 0.0),
            "source": src_add.get(key, "baseline"),
        })

    sugg_pass = suggestions.get("add_pass_through") if isinstance(suggestions.get("add_pass_through"), dict) else {}
    over_pass = overrides.get("add_pass_through") if isinstance(overrides.get("add_pass_through"), dict) else {}
    base_pass = baseline.get("pass_through") if isinstance(baseline.get("pass_through"), dict) else {}
    eff_pass = effective.get("add_pass_through") if isinstance(effective.get("add_pass_through"), dict) else {}
    src_pass = sources.get("add_pass_through") if isinstance(sources.get("add_pass_through"), dict) else {}
    keys_pass = sorted(_collect_process_keys(base_pass, sugg_pass, over_pass))
    for key in keys_pass:
        base_amount = base_pass.get(key)
        label = f"Pass-through Δ {key}"
        if base_amount not in (None, ""):
            try:
                label = f"{label} (base {_format_value(base_amount, 'currency')})"
            except Exception:
                pass
        rows.append({
            "path": ("add_pass_through", key),
            "label": label,
            "kind": "currency",
            "baseline": 0.0,
            "llm": sugg_pass.get(key),
            "user": over_pass.get(key),
            "accept": bool((accept.get("add_pass_through") or {}).get(key)),
            "effective": eff_pass.get(key, 0.0),
            "source": src_pass.get(key, "baseline"),
        })

    scrap_base = baseline.get("scrap_pct")
    scrap_llm = suggestions.get("scrap_pct")
    scrap_user = overrides.get("scrap_pct")
    scrap_eff = effective.get("scrap_pct")
    scrap_src = sources.get("scrap_pct", "baseline")
    if any(v is not None for v in (scrap_base, scrap_llm, scrap_user)):
        rows.append({
            "path": ("scrap_pct",),
            "label": "Scrap %",
            "kind": "percent",
            "baseline": scrap_base,
            "llm": scrap_llm,
            "user": scrap_user,
            "accept": bool(accept.get("scrap_pct")),
            "effective": scrap_eff,
            "source": scrap_src,
        })

    cont_base = baseline.get("contingency_pct")
    cont_llm = suggestions.get("contingency_pct")
    cont_user = overrides.get("contingency_pct")
    cont_eff = effective.get("contingency_pct")
    cont_src = sources.get("contingency_pct", "baseline")
    if any(v is not None for v in (cont_base, cont_llm, cont_user)):
        rows.append({
            "path": ("contingency_pct",),
            "label": "Contingency %",
            "kind": "percent",
            "baseline": cont_base,
            "llm": cont_llm,
            "user": cont_user,
            "accept": bool(accept.get("contingency_pct")),
            "effective": cont_eff,
            "source": cont_src,
        })

    fixture_delta_llm = suggestions.get("fixture_material_cost_delta")
    fixture_delta_user = overrides.get("fixture_material_cost_delta")
    fixture_delta_eff = effective.get("fixture_material_cost_delta")
    fixture_delta_src = sources.get("fixture_material_cost_delta", "baseline")
    if any(v is not None for v in (fixture_delta_llm, fixture_delta_user, fixture_delta_eff)):
        rows.append({
            "path": ("fixture_material_cost_delta",),
            "label": "Fixture material Δ",
            "kind": "currency",
            "baseline": 0.0,
            "llm": fixture_delta_llm,
            "user": fixture_delta_user,
            "accept": bool(accept.get("fixture_material_cost_delta")),
            "effective": fixture_delta_eff or 0.0,
            "source": fixture_delta_src,
        })

    setups_base = baseline.get("setups")
    setups_llm = suggestions.get("setups")
    setups_user = overrides.get("setups")
    setups_eff = effective.get("setups")
    setups_src = sources.get("setups", "baseline")
    if any(v is not None for v in (setups_base, setups_llm, setups_user)):
        rows.append({
            "path": ("setups",),
            "label": "Setups",
            "kind": "int",
            "baseline": setups_base,
            "llm": setups_llm,
            "user": setups_user,
            "accept": bool(accept.get("setups")),
            "effective": setups_eff,
            "source": setups_src,
        })

    fixture_base = baseline.get("fixture")
    fixture_llm = suggestions.get("fixture")
    fixture_user = overrides.get("fixture")
    fixture_eff = effective.get("fixture")
    fixture_src = sources.get("fixture", "baseline")
    if any(v is not None for v in (fixture_base, fixture_llm, fixture_user, fixture_eff)):
        rows.append({
            "path": ("fixture",),
            "label": "Fixture plan",
            "kind": "text",
            "baseline": fixture_base,
            "llm": fixture_llm,
            "user": fixture_user,
            "accept": bool(accept.get("fixture")),
            "effective": fixture_eff,
            "source": fixture_src,
        })

    return rows


MATERIAL_DROPDOWN_OPTIONS = [
    "Aluminum",
    "Berylium Copper",
    "Bismuth",
    "Brass",
    "Carbide",
    "Ceramic",
    "Cobalt",
    "Copper",
    "Gold",
    "Inconel",
    "Indium",
    "Lead",
    "Nickel",
    "Nickel Silver",
    "Palladium",
    "Phosphor Bronze",
    "Phosphorus",
    "Silver",
    "Stainless Steel",
    "Steel",
    "Tin",
    "Titanium",
    "Tool Steel",
    "Tungsten",
    "Other (enter custom price)",
]

# Default material for the dropdown / pricing logic
DEFAULT_MATERIAL_DISPLAY = "Steel"
DEFAULT_MATERIAL_KEY = _normalize_lookup_key(DEFAULT_MATERIAL_DISPLAY)

_MATERIAL_ADDITIONAL_KEYWORDS = {
    "Aluminum": {"aluminium"},
    "Berylium Copper": {"beryllium copper", "c172", "cube"},
    "Brass": {"c360", "c260"},
    "Copper": {"cu"},
    "Inconel": {"in718", "in625"},
    "Nickel": {"ni"},
    "Nickel Silver": {"german silver"},
    "Phosphor Bronze": {"phosphorbronze"},
    "Stainless Steel": {"stainless"},
    "Steel": {"carbon steel"},
    "Titanium": {"ti", "ti6al4v"},
    "Tool Steel": {"h13", "o1", "a2", "d2"},
}

MATERIAL_KEYWORDS: Dict[str, set[str]] = {}
for display in MATERIAL_DROPDOWN_OPTIONS:
    key = _normalize_lookup_key(display)
    if not key:
        continue
    extras = {_normalize_lookup_key(term) for term in _MATERIAL_ADDITIONAL_KEYWORDS.get(display, set())}
    extras.discard("")
    MATERIAL_KEYWORDS[key] = {key} | extras

MATERIAL_DISPLAY_BY_KEY: Dict[str, str] = {}
for display in MATERIAL_DROPDOWN_OPTIONS:
    key = _normalize_lookup_key(display)
    if key:
        MATERIAL_DISPLAY_BY_KEY[key] = display

MATERIAL_OTHER_KEY = _normalize_lookup_key(MATERIAL_DROPDOWN_OPTIONS[-1])


MATERIAL_MAP: dict[str, dict[str, float | str]] = {
    # Aluminum alloys → Aluminum cash index
    "6061": {"symbol": "XAL", "basis": "index_usd_per_tonne", "loss_factor": 0.0, "wieland_key": "6061"},
    "6061-T6": {"symbol": "XAL", "basis": "index_usd_per_tonne", "wieland_key": "6061-T6"},
    "7075": {"symbol": "XAL", "basis": "index_usd_per_tonne", "wieland_key": "7075"},

    # Stainless → approximate with Nickel + premium or use vendor CSV override
    "304": {"symbol": "XNI", "basis": "index_usd_per_tonne", "premium_usd_per_kg": 1.20, "wieland_key": "304"},
    "316": {"symbol": "XNI", "basis": "index_usd_per_tonne", "premium_usd_per_kg": 1.80, "wieland_key": "316"},

    # Copper alloys → Copper
    "C110": {"symbol": "XCU", "basis": "index_usd_per_tonne", "wieland_key": "C110"},

    # Precious (if you ever quote)
    "AU-9999": {"symbol": "XAU", "basis": "usd_per_troy_oz"},
}


def _coerce_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except Exception:
            m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    return None
    return None


def _price_value_to_per_gram(value: float, label: str) -> float | None:
    try:
        base = float(value)
    except Exception:
        return None
    label_lower = str(label or "").lower()
    label_compact = label_lower.replace(" ", "")

    def _has_any(*patterns: str) -> bool:
        return any(p in label_lower or p in label_compact for p in patterns)

    if _has_any("perg", "/g", "$/g", "per gram"):
        return base
    if _has_any("perkg", "/kg", "$/kg", "per kilogram"):
        return base / 1000.0
    if _has_any("perlb", "/lb", "$/lb", "per pound", "/lbs", "$/lbs"):
        return base / 453.59237
    if _has_any("peroz", "/oz", "$/oz", "per ounce"):
        return base / 28.349523125
    return None


# ---------- Material price backup (CSV) ----------
LB_PER_KG = 2.2046226218
BACKUP_CSV_NAME = "material_price_backup.csv"


def _usdkg_to_usdlb(x: float) -> float:
    return float(x) / LB_PER_KG if x is not None else x


def ensure_material_backup_csv(path: str | None = None) -> str:
    """
    Creates a small CSV with dummy prices if it doesn't exist, next to this script.
    Columns: material_key,usd_per_kg,usd_per_lb,notes
    """
    from pathlib import Path

    path = Path(path) if path else Path(__file__).with_name(BACKUP_CSV_NAME)
    if path.exists():
        return str(path)

    rows = [
        # material_key,   usd/kg,  usd/lb (blank -> derived),      notes
        ("steel",           2.20,   "",  "dummy base"),
        ("stainless steel", 4.00,   "",  "dummy base"),
        ("aluminum",        2.80,   "",  "dummy base"),
        ("copper",          9.50,   "",  "dummy base"),
        ("brass",           7.80,   "",  "dummy base"),
        ("titanium",       17.00,   "",  "dummy base"),
    ]

    lines = ["material_key,usd_per_kg,usd_per_lb,notes\n"]
    for m, kg, lb, note in rows:
        if lb in ("", None):
            lb = kg / LB_PER_KG
        lines.append(f"{m},{float(kg):.6f},{float(lb):.6f},{note}\n")

    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def load_backup_prices_csv(path: str | None = None) -> dict:
    """
    Returns: {normalized_key: {"usd_per_kg":float,"usd_per_lb":float,"notes":str}, ...}
    """
    from pathlib import Path
    import pandas as pd

    path = Path(path) if path else Path(ensure_material_backup_csv())
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        key = _normalize_lookup_key(r["material_key"])
        usdkg = _coerce_float_or_none(r.get("usd_per_kg"))
        usdlb = _coerce_float_or_none(r.get("usd_per_lb"))
        if usdkg and not usdlb:
            usdlb = _usdkg_to_usdlb(usdkg)
        if usdlb and not usdkg:
            usdkg = float(usdlb) * LB_PER_KG
        if usdkg and usdlb:
            out[key] = {"usd_per_kg": float(usdkg), "usd_per_lb": float(usdlb), "notes": str(r.get("notes", ""))}
    return out


def resolve_material_unit_price(display_name: str, unit: str = "kg") -> tuple[float, str]:
    """
    Try Wieland first; if missing/error, fall back to local CSV; finally hardcode steel.
    unit in {"kg","lb"}; returns (price, source_tag)
    """
    key = _normalize_lookup_key(display_name)
    # 1) Wieland (if installed)
    try:
        from wieland_scraper import get_live_material_price

        price, src = get_live_material_price(display_name, unit=unit, fallback_usd_per_kg=float("nan"))
        if price is not None and math.isfinite(float(price)):
            return float(price), src
    except Exception as e:
        src = f"wieland_error:{type(e).__name__}"
    else:
        src = ""

    # 2) CSV
    try:
        table = load_backup_prices_csv()
        rec = table.get(key)
        if not rec:
            if "stainless" in key:
                rec = table.get("stainless steel")
            elif "steel" in key:
                rec = table.get("steel")
            elif "alum" in key:
                rec = table.get("aluminum")
            elif "copper" in key or "cu" in key:
                rec = table.get("copper")
            elif "brass" in key:
                rec = table.get("brass")
        if rec:
            price = rec["usd_per_kg"] if unit == "kg" else rec["usd_per_lb"]
            return float(price), f"backup_csv:{BACKUP_CSV_NAME}"
    except Exception:
        pass

    # 3) Last-resort hardcoded steel
    hard = 2.20 if unit == "kg" else (2.20 / LB_PER_KG)
    return hard, "hardcoded_default"


def FACE_OF(s: TopoDS_Shape) -> TopoDS_Face:
    # already a Face?
    if isinstance(s, TopoDS_Face) or type(s).__name__ == "TopoDS_Face":
        return s
    # cast Shape→Face using the static caster present in your wheel
    if hasattr(TopoDS, "Face_s"):
        f = TopoDS.Face_s(s)
        if hasattr(f, "IsNull") and f.IsNull():
            raise TypeError("TopoDS.Face_s returned null")
        return f
    raise TypeError("No Face_s caster available in this OCP wheel.")


SHOW_LLM_HOURS_DEBUG = False # set True only when debugging
if sys.platform == 'win32':
    occ_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(occ_bin):
        os.add_dll_directory(occ_bin)
from tkinter import ttk, filedialog, messagebox
import subprocess, tempfile, shutil


# --------------------- helpers: model discovery ---------------------
PREFERRED_MODEL_DIRS = [
    r"D:\CAD_Quoting_Tool\models",
]

def _pick_best_gguf(paths):
    # Prefer files with 'qwen' and 'instruct'; otherwise choose the largest.
    if not paths:
        return ""
    paths = [Path(p) for p in paths if Path(p).is_file() and p.lower().endswith(".gguf")]
    if not paths:
        return ""
    preferred = [p for p in paths if ("qwen" in p.name.lower() and "instr" in p.name.lower())]
    pool = preferred or paths
    # Largest by size
    try:
        best = max(pool, key=lambda p: p.stat().st_size)
    except Exception:
        best = pool[0]
    return str(best)

def find_default_qwen_model():
    # 1) Env
    envp = os.environ.get("QWEN_GGUF_PATH", "")
    if envp and Path(envp).is_file():
        return envp
    # 2) D:\CAD_Quoting_Tool\models
    for d in PREFERRED_MODEL_DIRS:
        dpath = Path(d)
        if dpath.is_dir():
            ggufs = list(dpath.glob("*.gguf"))
            choice = _pick_best_gguf([str(p) for p in ggufs])
            if choice: return choice
    # 3) script_dir\models
    try:
        sdir = Path(__file__).resolve().parent
        ggufs = list((sdir / "models").glob("*.gguf"))
        choice = _pick_best_gguf([str(p) for p in ggufs])
        if choice: return choice
    except Exception:
        pass
    # 4) cwd\models
    ggufs = list((Path.cwd() / "models").glob("*.gguf"))
    choice = _pick_best_gguf([str(p) for p in ggufs])
    if choice: return choice
    return ""

# Optional trimesh for STL
try:
    import trimesh  # type: ignore
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

_HAS_EZDXF  = False
_HAS_ODAFC  = False      # ezdxf.addons.odafc (uses ODA File Converter)
_EZDXF_VER  = "unknown"

try:
    import ezdxf
    _EZDXF_VER = getattr(ezdxf, "__version__", "unknown")
    _HAS_EZDXF = True
except Exception:
    ezdxf = None  # keep name defined

# odafc is optional; don't fail if it's not present
try:
    if _HAS_EZDXF:
        from ezdxf.addons import odafc  # type: ignore
        _HAS_ODAFC = True
except Exception:
    _HAS_ODAFC = False

# Hole chart helpers (optional)
try:
    from hole_table_parser import parse_hole_table_lines
except Exception:
    parse_hole_table_lines = None

try:
    from dxf_text_extract import extract_text_lines_from_dxf
except Exception:
    extract_text_lines_from_dxf = None

# numpy is optional for a few small calcs; degrade gracefully if missing
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore
# ---------- OCC / OCP compatibility ----------
STACK = None

def _make_bnd_add_ocp():
    # Try every known symbol name in OCP builds
    candidates = []
    try:
        # module-level function: Add(...)
        from OCP.BRepBndLib import Add as _add1
        candidates.append(_add1)
    except Exception:
        pass
    try:
        # module-level function: Add_s(...)
        from OCP.BRepBndLib import Add_s as _add1s
        candidates.append(_add1s)
    except Exception:
        pass
    try:
        # module-level function: BRepBndLib_Add(...)
        from OCP.BRepBndLib import BRepBndLib_Add as _add2
        candidates.append(_add2)
    except Exception:
        pass
    try:
        # module-level function: brepbndlib_Add(...)
        from OCP.BRepBndLib import brepbndlib_Add as _add3
        candidates.append(_add3)
    except Exception:
        pass
    try:
        from OCP.BRepBndLib import BRepBndLib as _klass
        for attr in ("Add", "Add_s", "AddClose_s", "AddOptimal_s", "AddOBB_s"):
            fn = getattr(_klass, attr, None)
            if fn:
                candidates.append(fn)
    except Exception:
        pass

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

try:
    # Prefer OCP (CadQuery/ocp bindings)
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCP.BRep import BRep_Builder
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Bnd import Bnd_Box

    try:
        bnd_add = _make_bnd_add_ocp()
        STACK = "ocp"
    except Exception:
        # Fallback: try dynamic dispatch at call time for OCP builds missing Add
        def bnd_add(shape, box, use_triangulation=True):
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
                try:
                    module = __import__(mod_name, fromlist=[attr])
                    fn = getattr(module, attr)
                    return fn(shape, box, use_triangulation)
                except Exception:
                    continue
            raise RuntimeError("No BRepBndLib.Add available in this build")
except Exception:
    # Fallback to pythonocc-core
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add as _brep_add

    def bnd_add(shape, box, use_triangulation=True):
        _brep_add(shape, box, use_triangulation)

    STACK = "pythonocc"
# ---------- end shim ----------
# ----- one-backend imports -----
try:
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    BACKEND = "ocp"
except Exception:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods_Edge, topods_Shell, topods_Solid
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    BACKEND = "pythonocc"

def _typename(o):  # small helper
    return getattr(o, "__class__", type(o)).__name__

def as_face(obj):
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

def iter_faces(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield ensure_face(exp.Current())
        exp.Next()

def face_surface(face_like):
    f = ensure_face(face_like)
    if hasattr(BRep_Tool, "Surface_s"):
        s = BRep_Tool.Surface_s(f)
    else:
        s = BRep_Tool.Surface(f)
    loc = (BRep_Tool.Location(f) if hasattr(BRep_Tool, "Location")
           else (f.Location() if hasattr(f, "Location") else None))
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
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Solid, TopoDS_Shell

    def _TO_EDGE(s):
        if type(s).__name__ in ("TopoDS_Edge", "Edge"):
            return s
        if hasattr(TopoDS, "Edge_s"):
            return TopoDS.Edge_s(s)
        try:
            from OCP.TopoDS import topods as _topods
            return _topods.Edge(s)
        except Exception as e:
            raise TypeError(f"Cannot cast to Edge from {type(s).__name__}") from e

    def _TO_SOLID(s):
        if type(s).__name__ in ("TopoDS_Solid", "Solid"):
            return s
        if hasattr(TopoDS, "Solid_s"):
            return TopoDS.Solid_s(s)
        from OCP.TopoDS import topods as _topods
        return _topods.Solid(s)

    def _TO_SHELL(s):
        if type(s).__name__ in ("TopoDS_Shell", "Shell"):
            return s
        if hasattr(TopoDS, "Shell_s"):
            return TopoDS.Shell_s(s)
        from OCP.TopoDS import topods as _topods
        return _topods.Shell(s)
else:
    # Resolve within OCC.Core only
    try:
        from OCC.Core.TopoDS import topods_Edge as _TO_EDGE
    except Exception:
        try:
            from OCC.Core.TopoDS import Edge as _TO_EDGE
        except Exception:
            from OCC.Core.TopoDS import topods as _occ_topods
            _TO_EDGE = getattr(_occ_topods, "Edge")
    try:
        from OCC.Core.TopoDS import topods_Solid as _TO_SOLID
    except Exception:
        from OCC.Core.TopoDS import topods as _occ_topods
        _TO_SOLID = getattr(_occ_topods, "Solid")
    try:
        from OCC.Core.TopoDS import topods_Shell as _TO_SHELL
    except Exception:
        from OCC.Core.TopoDS import topods as _occ_topods
        _TO_SHELL = getattr(_occ_topods, "Shell")

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
    if hasattr(obj, "IsNull") and obj.IsNull():
        raise TypeError("Expected non-null TopoDS_Shape")
    return obj

# Safe casters: no-ops if already cast; unwrap list nodes; check kind
def to_edge_safe(obj):
    obj = _unwrap_value(obj)
    if _is_named(obj, ("TopoDS_Edge", "Edge")):
        return obj
    obj = ensure_shape(obj)
    return _TO_EDGE(obj)

# Choose stack
try:
    # OCP / CadQuery bindings
    from OCP.BRepGProp import BRepGProp as _BRepGProp_mod
    STACK_GPROP = "ocp"
except Exception:
    # pythonocc-core
    try:
        from OCC.Core.BRepGProp import BRepGProp as _BRepGProp_mod
    except Exception:
        from types import SimpleNamespace
        from OCC.Core.BRepGProp import (
            brepgprop_LinearProperties as _lp,
            brepgprop_SurfaceProperties as _sp,
            brepgprop_VolumeProperties as _vp,
        )
        _BRepGProp_mod = SimpleNamespace(
            LinearProperties=_lp,
            SurfaceProperties=_sp,
            VolumeProperties=_vp,
        )
    def _to_edge_occ(s):
        try:
            from OCC.Core.TopoDS import topods_Edge as _fn
        except Exception:
            from OCC.Core.TopoDS import Edge as _fn
        return _fn(s)

    _TO_EDGE = _to_edge_occ
    STACK_GPROP = "pythonocc"

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
        except Exception as e:
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

    amap = TopTools_IndexedDataMapOfShapeListOfShape()
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

def ensure_face(obj):
    if obj is None:
        raise TypeError("Expected a face, got None")
    if isinstance(obj, TopoDS_Face) or type(obj).__name__ == "TopoDS_Face":
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
    """Raise a clear error if ezdxf is missing."""
    if not _HAS_EZDXF:
        raise RuntimeError("ezdxf not installed. Install with pip/conda (package name: 'ezdxf').")
    return ezdxf

def get_dwg_converter_path() -> str:
    """Resolve a DWG?DXF converter path (.bat/.cmd/.exe)."""
    exe = os.environ.get("ODA_CONVERTER_EXE") or os.environ.get("DWG2DXF_EXE")
    # Fall back to a local wrapper next to this script if it exists.
    local = str(Path(__file__).with_name("dwg2dxf_wrapper.bat"))
    if not exe and Path(local).exists():
        exe = local
    return exe or ""

def have_dwg_support() -> bool:
    """True if we can open DWG (either odafc or an external converter is available)."""
    return _HAS_ODAFC or bool(get_dwg_converter_path())
def get_import_diagnostics_text() -> str:
    import sys, shutil, os
    lines = []
    lines.append(f"Python: {sys.executable}")
    try:
        import fitz  # PyMuPDF
        lines.append("PyMuPDF: OK")
    except Exception as e:
        lines.append(f"PyMuPDF: MISSING ({e})")

    try:
        import ezdxf
        lines.append(f"ezdxf: {getattr(ezdxf, '__version__', 'unknown')}")
        try:
            from ezdxf.addons import odafc
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
    _HAS_PYMUPDF = False
def upsert_var_row(df, item, value, dtype="number"):
    """
    Upsert one row by Item name (case-insensitive).
    - Forces `item` and `value` to scalars.
    - Works on the sanitized 3-column df (Item, Example..., Data Type...).
    """
    import numpy as np
    import pandas as pd

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
    row = {c: "" for c in cols}
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
    require_ezdxf()
    if path.suffix.lower() == ".dwg":
        # Prefer explicit converter/wrapper if configured (works even if ODA isnï¿½t on PATH)
        exe = get_dwg_converter_path()
        if exe:
            dxf_path = convert_dwg_to_dxf(str(path))
            return ezdxf.readfile(dxf_path)
        # Fallback: odafc (requires ODAFileConverter on PATH)
        if _HAS_ODAFC:
            return odafc.readfile(str(path))
        raise RuntimeError(
            "DWG import needs ODA File Converter. Set ODA_CONVERTER_EXE to the exe "
            "or place dwg2dxf_wrapper.bat next to the script."
        )
    return ezdxf.readfile(str(path))  # DXF directly


def dxf_to_structured(doc: Drawing) -> dict:
    msp = doc.modelspace()
    header = doc.header
    insunits = header.get("$INSUNITS", 0)  # 1 = inches, 6 = meters, etc.
    # Normalize a simple unit string for quoting
    UNITS = {0:"unitless",1:"in",2:"ft",3:"mi",4:"mm",5:"cm",6:"m"}
    units = UNITS.get(insunits, "unitless")

    texts = []
    for e in msp.query("TEXT"):
        texts.append({"type":"TEXT","text":e.dxf.text, "xy":tuple(e.dxf.insert[:2])})
    for e in msp.query("MTEXT"):
        # .text is formatted; .plain_text() strips inline codes
        texts.append({"type":"MTEXT","text":e.plain_text(), "xy":tuple(e.dxf.insert[:2])})
    for e in msp.query("MLEADER"):
        if e.context and e.context.mtext:
            texts.append({"type":"MLEADER","text":e.context.mtext.plain_text()})

    dims = []
    for d in msp.query("DIMENSION"):
        # d.dxf.text often contains the displayed text; fall back to raw if needed
        label = (d.dxf.text or "").strip()
        # Try to find diameter / radius / WxH patterns in the label
        m = DIM_RE.search(label.replace(",", "."))
        dims.append({"raw": label, "pattern": m.groups() if m else None})

    # Quick & dirty material/thickness/title block scrapes from text blobs
    material = next((t["text"].split(":")[-1].strip()
                     for t in texts if re.search(r"\\bMATERIAL\\b", t["text"], re.I)), None)
    thickness = next((re.search(r"(?:(?:THICK|THK)[\\s:=]*)([0-9.]+)\\s*(mm|in|inch|inches)?", t["text"], re.I)
                      for t in texts), None)
    if thickness:
        thickness = {"value": float(thickness.group(1)), "units": (thickness.group(2) or units)}

    return {
        "source_units": units,
        "texts": texts,
        "dimensions": dims,
        "material": material,
        "thickness": thickness,
    }

def pdf_to_structured(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    # 1) TEXT blocks (vector text if present)
    words = page.get_text("words")  # [(x0,y0,x1,y1,"word", block_no, line_no, word_no), ...]
    text_blocks = page.get_text("blocks")  # paragraph-level
    raw_text = page.get_text("text")       # simple text stream

    # 2) Tables (handy for title blocks / BOMs)
    tables = []
    try:
        tabs = page.find_tables()  # since PyMuPDF 1.23+
        for t in tabs.tables:
            tables.append(t.extract())
    except Exception:
        pass

    # 3) Vector line-art (for figure geometry / bounding clusters)
    drawings = page.get_drawings()  # vector paths, strokes, fills
    # Optional: cluster into groups that form a single figure (1.24+)
    # clusters = page.cluster_drawings()

    # 4) Page raster for Qwen
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_path = pdf_path.with_suffix(f".p{page_index}.png")
    pix.save(png_path)

    return {
        "raw_text": raw_text,
        "blocks": text_blocks,
        "tables": tables,
        "drawings_count": len(drawings),
        "image_path": str(png_path),
    }

QWEN_SYSTEM = """You are a quoting assistant. Only confirm or fill missing parameters.
Prefer vector-extracted numbers over reading pixels. If units conflict, explain and ask for clarification.
Return strict JSON with fields:
{overall_size_mm, thickness_mm, material, hole_counts_by_dia_mm, slots, threads, tolerance_notes, finish, qty, uncertainties[]}"""

def build_llm_payload(structured: dict, page_image_path: str | None):
    messages = [
        {"role":"system","content": QWEN_SYSTEM},
        {"role":"user","content":[
            {"type":"text","text":"Vector-extracted data (trusted):\n" + json.dumps(structured, ensure_ascii=False)},
        ]}
    ]
    if page_image_path:
        messages[1]["content"].append({"type":"image_url","image_url":{"url":"file://"+page_image_path}})
    return messages



# ==== OpenCascade compat (works with OCP OR OCC.Core) ====
import tempfile, subprocess
from pathlib import Path

try:
    # ---- OCP branch ----
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader
    from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRep import BRep_Tool        # OCP version
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid,
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    
    # ADD THESE TWO IMPORTS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopoDS import (
        TopoDS_Shape, TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound
    )
    from OCP.TopExp import TopExp_Explorer, TopExp
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone,
        GeomAbs_BSplineSurface, GeomAbs_BezierSurface
    )
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface
    from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Pln
    BACKEND_OCC = "OCP"

    def BRepTools_UVBounds(face):
        return BRepTools.UVBounds(face)


    def _brep_read(path: str) -> TopoDS_Shape:
        s = TopoDS_Shape()
        builder = BRep_Builder()
        if hasattr(BRepTools, "Read_s"):
            ok = BRepTools.Read_s(s, str(path), builder)
        else:
            ok = BRepTools.Read(s, str(path), builder)
        if ok is False:
            raise RuntimeError("BREP read failed")
        return s



except Exception:
    # ---- OCC.Core branch ----
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRep import BRep_Tool          # ? OCC version
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid,
    )
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    # ADD TopTools import and TopoDS_Face for the fix below
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.TopoDS import (
        TopoDS_Shape, TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound,
        TopoDS_Face
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
    from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone,
        GeomAbs_BSplineSurface, GeomAbs_BezierSurface
    )
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Pln
    BACKEND_OCC = "OCC.Core"

    # BRepGProp shim (pythonocc uses free functions)
    from OCC.Core.BRepGProp import (
        brepgprop_SurfaceProperties,
        brepgprop_LinearProperties,
        brepgprop_VolumeProperties,
    )
    class _BRepGPropShim:
        @staticmethod
        def SurfaceProperties_s(shape_or_face, gprops): brepgprop_SurfaceProperties(shape_or_face, gprops)
        @staticmethod
        def LinearProperties_s(edge, gprops):          brepgprop_LinearProperties(edge, gprops)
        @staticmethod
        def VolumeProperties_s(shape, gprops):         brepgprop_VolumeProperties(shape, gprops)
    BRepGProp = _BRepGPropShim



    # UV bounds and brep read are free functions
    from OCC.Core.BRepTools import BRepTools, breptools_Read as _occ_breptools_read
    def BRepTools_UVBounds(face):
        fn = getattr(BRepTools, "UVBounds", None)
        if fn is None:
            from OCC.Core.BRepTools import breptools_UVBounds as _legacy
            return _legacy(face)
        return fn(face)
    def _brep_read(path: str) -> TopoDS_Shape:
        s = TopoDS_Shape()
        ok = _occ_breptools_read(s, str(path), BRep_Builder())
        if ok is False:
            raise RuntimeError("BREP read failed")
        return s

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

    fixer = ShapeFix_Shape(shape)
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

def read_step_shape(path: str) -> TopoDS_Shape:
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
    print("[DBG] shape type:", type(shape).__name__, "IsNull:", getattr(shape, "IsNull", lambda: True)())
    amap = map_shapes_and_ancestors(shape, TopAbs_EDGE, TopAbs_FACE)
    print("[DBG] map size:", amap.Size())
    # DEBUG: sanity probe for STEP faces
    if os.environ.get("STEP_PROBE", "0") == "1":
        cnt = 0
        try:
            for f in iter_faces(shape):
                _surf, _loc = face_surface(f)
                cnt += 1
        except Exception as _e:
            # Keep debug non-fatal; report and continue
            print(f"[STEP_PROBE] error during face probe: {_e}")
        else:
            print(f"[STEP_PROBE] faces={cnt}")

    fx = ShapeFix_Shape(shape)
    fx.Perform()
    return fx.Shape()

from OCP.TopoDS import TopoDS_Shape  # or OCC.Core.TopoDS on pythonocc

def safe_bbox(shape: TopoDS_Shape):
    if shape is None or shape.IsNull():
        raise ValueError("Cannot compute bounding box of a null shape.")
    box = Bnd_Box()
    bnd_add(shape, box, True)  # <- uses whichever binding is available
    return box
def read_step_or_iges_or_brep(path: str) -> TopoDS_Shape:
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
    try:
        if exe_lower.endswith(".bat") or exe_lower.endswith(".cmd"):
            # ? run batch via cmd.exe so it actually executes
            cmd = ["cmd", "/c", exe, str(dwg), str(out_dxf)]
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elif "odafileconverter" in exe_lower:
            # ? official ODAFileConverter CLI
            cmd = [exe, str(dwg.parent), str(out_dir), out_ver, "DXF", "0", "0", dwg.name]
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            # ? generic exe that accepts <in> <out>
            cmd = [exe, str(dwg), str(out_dxf)]
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield to_solid(exp.Current())
        exp.Next()

def explode_compound(shape: TopoDS_Shape):
    """If the file is a big COMPOUND, break it into shapes (parts/bodies)."""
    exp = TopExp_Explorer(shape, TopAbs_COMPOUND)
    if exp.More():
        # Itï¿½s a compound ï¿½ return its shells/solids/faces as needed
        solids = list(iter_solids(shape))
        if solids:
            return solids
        # fallback to shells
        sh = TopExp_Explorer(shape, TopAbs_SHELL)
        shells = []
        while sh.More():
            shells.append(to_shell(sh.Current()))
            sh.Next()
        return shells
    return [shape]



def safe_bounding_box(shape: TopoDS_Shape) -> Bnd_Box:
    """Public wrapper maintained for compatibility."""
    return safe_bbox(shape)

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
            from OCP.BRepLProp import BRepLProp_SLProps
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
    for z in z_values:
        plane = gp_Pln(gp_Pnt(0,0,z), gp_Dir(0,0,1))
        sec = BRepAlgoAPI_Section(shape, plane, False); sec.Build()
        if not sec.IsDone(): continue
        w = sec.Shape()
        it = TopExp_Explorer(w, TopAbs_EDGE)
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

def _hole_groups_from_cylinders(shape, bbox=None):
    if bbox is None: bbox = _bbox(shape)
    from collections import defaultdict
    groups = defaultdict(lambda: {"dia_mm":0.0,"depth_mm":0.0,"through":False,"count":0})
    for f in iter_faces(shape):
        if _face_type(f) == "cylindrical":
            ga = GeomAdaptor_Surface(face_surface(f)[0])
            try:
                cyl = ga.Cylinder(); r = abs(cyl.Radius()); ax = cyl.Axis().Direction()
                fb = _bbox(f)
                def proj(x,y,z): return x*ax.X() + y*ax.Y() + z*ax.Z()
                span = abs(proj(fb[3],fb[4],fb[5]) - proj(fb[0],fb[1],fb[2]))
                dia = 2.0*r
                bmin = proj(*bbox[:3]); bmax = proj(*bbox[3:]); bspan = abs(bmax-bmin)
                through = span > 0.9*bspan
                key = (round(dia,2), round(span,2), through)
                groups[key]["dia_mm"] = round(dia,2)
                groups[key]["depth_mm"] = round(span,2)
                groups[key]["through"] = through
                groups[key]["count"] += 1
            except Exception:
                pass
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
    print(f"[{time.time() - start_time:.2f}s] Starting enrich_geo_stl for {path}")
    if not _HAS_TRIMESH:
        raise RuntimeError("trimesh not available to process STL")
    
    print(f"[{time.time() - start_time:.2f}s] Loading mesh...")
    mesh = trimesh.load(path, force='mesh')
    print(f"[{time.time() - start_time:.2f}s] Mesh loaded. Faces: {len(mesh.faces)}")
    
    if mesh.is_empty:
        raise RuntimeError("Empty STL mesh")
    
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    L, W, H = float(xmax-xmin), float(ymax-ymin), float(zmax-zmin)
    area_total = float(mesh.area)
    volume = float(mesh.volume) if mesh.is_volume else 0.0
    faces = int(len(mesh.faces))
    
    print(f"[{time.time() - start_time:.2f}s] Calculating WEDM length...")
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
    print(f"[{time.time() - start_time:.2f}s] WEDM length calculated.")
    
    print(f"[{time.time() - start_time:.2f}s] Calculating deburr length...")
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
    print(f"[{time.time() - start_time:.2f}s] Deburr length calculated.")
    
    bbox_vol = max(L*W*H, 1e-9)
    complexity = (faces / max(volume, bbox_vol)) * 100.0
    
    print(f"[{time.time() - start_time:.2f}s] Calculating 3-axis accessibility...")
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
    print(f"[{time.time() - start_time:.2f}s] 3-axis accessibility calculated.")
    
    try:
        center = list(map(float, mesh.center_mass))
    except Exception:
        center = [0.0,0.0,0.0]
    
    print(f"[{time.time() - start_time:.2f}s] Finished enrich_geo_stl.")
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


def load_cad_any(path: str) -> TopoDS_Shape:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in ("step", "stp", "iges", "igs", "brep"):
        return read_step_or_iges_or_brep(path)
    if ext == "dwg":
        dxf_path = convert_dwg_to_dxf(path)
        return read_dxf_as_occ_shape(dxf_path)
    if ext == "dxf":
        return read_dxf_as_occ_shape(path)
    
    raise RuntimeError(f"Unsupported file type for shape loading: {ext}")
def read_cad_any(path: str):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.IGESControl import IGESControl_Reader
    from OCP.TopoDS import TopoDS_Shape

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
        s = TopoDS_Shape()
        if not BRepTools.Read(s, path, None):
            raise RuntimeError("BREP read failed")
        return s
    if ext == ".dxf":
        return read_dxf_as_occ_shape(path)
    if ext == ".dwg":
        conv = os.environ.get("ODA_CONVERTER_EXE") or os.environ.get("DWG2DXF_EXE")
        print(f"INFO: Using DWG converter: {conv}")
        dxf_path = convert_dwg_to_dxf(path)
        return read_dxf_as_occ_shape(dxf_path)
    raise RuntimeError(f"Unsupported CAD format: {ext}")

# ----------------- Offline Qwen via llama-cpp -----------------
def _llama_missing_msg():
    return ("llama-cpp-python is not installed.\n" 
            "Install in your env:\n" 
            "  python -m pip install -U llama-cpp-python\n")

class _LocalLLM:
    """
    Minimal, tolerant wrapper around llama-cpp-python.
    - Chat if available; otherwise use completion.
    - Always returns best-effort JSON dict.
    """
    def __init__(self, model_path: str,
                 n_ctx: int = int(os.getenv("QWEN_N_CTX", 32768)),
                 n_gpu_layers: int = int(os.getenv("QWEN_N_GPU_LAYERS", 0)),
                 n_threads: int | None = None):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(_llama_missing_msg()) from e
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model GGUF not found: {model_path}")
        self._Llama = Llama
        self._llm = None
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads

    def _ensure(self):
        if self._llm is None:
            llama_kwargs = dict(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                n_batch=int(os.getenv("QWEN_N_BATCH", 1024)),
                logits_all=False,
                verbose=False,
            )
            rope_scale = os.getenv("ROPE_FREQ_SCALE")
            if rope_scale:
                try:
                    llama_kwargs["rope_freq_scale"] = float(rope_scale)
                except Exception:
                    pass
            self._llm = self._Llama(**llama_kwargs)

    def ask_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        context: dict | None = None,
        params: dict | None = None,
    ):
        self._ensure()
        p = {
            "temperature": float(os.getenv("QWEN_TEMP", temperature)),
            "top_p": float(os.getenv("QWEN_TOP_P", 0.90)),
            "repeat_penalty": float(os.getenv("QWEN_REPEAT_PENALTY", 1.05)),
            "max_tokens": int(os.getenv("QWEN_MAX_TOKENS", max_tokens)),
        }
        if params:
            p.update(params)

        req = {
            "model": getattr(self, "model_path", "local"),
            "n_ctx": getattr(self, "n_ctx", None),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "params": p,
            "context_payload": context or {},
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        out = self._llm.create_chat_completion(
            messages=req["messages"],
            temperature=p["temperature"],
            top_p=p["top_p"],
            repeat_penalty=p["repeat_penalty"],
            max_tokens=p["max_tokens"],
        )
        text = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})

        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            parsed = parse_llm_json(text)

        if APP_ENV.llm_debug_enabled:
            snap = {
                "request": req,
                "raw_response_text": text,
                "parsed_response": parsed,
                "usage": usage,
            }
            fn = APP_ENV.llm_debug_dir / f"llm_snapshot_{int(time.time())}.json"
            fn.write_text(json.dumps(snap, indent=2), encoding="utf-8")

        return parsed, text, usage

    def close(self):
        """Best-effort cleanup to avoid llama-cpp destructor warnings."""
        try:
            llm = getattr(self, "_llm", None)
            self._llm = None
            if llm is None:
                return
            # Ensure attribute exists to satisfy some versions' __del__ checks
            try:
                getattr(llm, "sampler")
            except Exception:
                try:
                    setattr(llm, "sampler", None)
                except Exception:
                    pass
            # Prefer explicit close if available
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                pass
            # Drop strong refs
            try:
                del llm
            except Exception:
                pass
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

# ---- LLM hours inference ----
def infer_hours_and_overrides_from_geo(geo: dict, params: dict | None = None, rates: dict | None = None) -> dict:
    """
    Ask the local LLM to estimate HOURS for each process + a few knobs.
    Returns a dict with 'hours', 'setups', 'inspection', 'notes'.
    """
    params = params or {}
    rates = rates or {}

    system = (
        "You are a senior manufacturing estimator in a tool & die/CNC job shop. "
        "Given CAD GEO features (mm/mm2/mm3), estimate realistic HOURS for a small lot of machined parts. "
        "Prefer simple, conservative estimates. Output ONLY JSON."
    )

    # keep schema explicit and small; minutes only where noted
    schema = {
        "hours": {
            "Programming_Hours": 0.0,
            "CAM_Programming_Hours": 0.0,
            "Engineering_Hours": 0.0,
            "Fixture_Build_Hours": 0.0,

            "Roughing_Cycle_Time_hr": 0.0,
            "Semi_Finish_Cycle_Time_hr": 0.0,
            "Finishing_Cycle_Time_hr": 0.0,

            "OD_Turning_Hours": 0.0,
            "ID_Bore_Drill_Hours": 0.0,
            "Threading_Hours": 0.0,
            "Cutoff_Hours": 0.0,

            "WireEDM_Hours": 0.0,
            "SinkerEDM_Hours": 0.0,

            "Grinding_Surface_Hours": 0.0,
            "Grinding_ODID_Hours": 0.0,
            "Jig_Grind_Hours": 0.0,

            "Lapping_Hours": 0.0,
            "Deburr_Hours": 0.0,
            "Tumble_Hours": 0.0,
            "Blast_Hours": 0.0,
            "Laser_Mark_Hours": 0.0,
            "Masking_Hours": 0.0,

            "InProcess_Inspection_Hours": 0.0,
            "Final_Inspection_Hours": 0.0,
            "CMM_Programming_Hours": 0.0,
            "CMM_RunTime_min": 0.0,  # minutes

            "Saw_Waterjet_Hours": 0.0,
            "Assembly_Hours": 0.0,
            "Packaging_Labor_Hours": 0.0,

            "EHS_Hours": 0.0
        },
        "setups": {
            "Milling_Setups": 1,
            "Setup_Hours_per_Setup": 0.3
        },
        "inspection": {
            "FAIR_Required": False,
            "Source_Inspection_Required": False
        },
        "notes": []
    }

    prompt = f'''
GEO (mm / mm2 / mm3):
{json.dumps(geo, indent=2)}

Rules of thumb:
- Small rectangular blocks with few features: Programming 0.2ï¿½1.0 hr, CAM 0.2ï¿½1.0 hr, Engineering 0 hr.
- Use setups 1ï¿½2 unless 3-axis accessibility is low (<0.6) or faces with unique normals > 4.
- Deburr 0.1ï¿½0.4 hr unless thin walls/freeform edges (then up to 0.8 hr).
- Final inspection ~0.2ï¿½0.6 hr; CMM only if tolerances < 0.02 mm or GD&T heavy.
- Grinding/EDM/Turning hours should be 0 unless features clearly require them.
- Never return huge numbers for tiny parts (<80 mm max dim).

Return JSON with this structure (numbers only, minutes only for CMM_RunTime_min):
{json.dumps(schema, indent=2)}
'''

    # try local GGUF if available
    try:
        mp = os.environ.get("QWEN_GGUF_PATH")
        if mp and Path(mp).is_file():
            q = _LocalLLM(mp)
            parsed, raw_text, _usage = q.ask_json(
                system,
                prompt,
                temperature=0.1,
                max_tokens=1024,
            )
            if not isinstance(parsed, dict):
                parsed = parse_llm_json(raw_text)
            if isinstance(parsed, dict) and "hours" in parsed:
                return parsed
    except Exception:
        pass

    # Fallback heuristics if LLM unavailable
    faces = float(geo.get("GEO__Face_Count", 0) or 0)
    max_dim = float(geo.get("GEO__MaxDim_mm", 0) or 0)
    small_simple = (max_dim and max_dim <= 80.0 and faces < 300)

    hours = {k: 0.0 for k in schema["hours"]}
    if small_simple:
        hours["Programming_Hours"] = 0.5
        hours["CAM_Programming_Hours"] = 0.5
        hours["Engineering_Hours"] = 0.0
        hours["Roughing_Cycle_Time_hr"] = 0.5
        hours["Finishing_Cycle_Time_hr"] = 0.2
        hours["Deburr_Hours"] = 0.2
        hours["Final_Inspection_Hours"] = 0.2
        setups = {"Milling_Setups": 1, "Setup_Hours_per_Setup": 0.3}
    else:
        setups = {"Milling_Setups": 2, "Setup_Hours_per_Setup": 0.4}

    return {"hours": hours, "setups": setups, "inspection": {"FAIR_Required": False, "Source_Inspection_Required": False}, "notes": ["heuristic fallback"]}

def clamp_llm_hours(est: dict, geo: dict, params: dict | None = None) -> dict:
    params = params or {}
    h = est.get("hours", {}).copy()

    max_dim = float(geo.get("GEO__MaxDim_mm", 0) or 0)
    setups  = int(round(est.get("setups", {}).get("Milling_Setups", 1)))
    simple  = (max_dim and max_dim <= 80.0 and setups <= 2)

    # default caps (you can expose these in Overrides UI)
    caps_simple = {
        "Programming_Hours": 1.0, "CAM_Programming_Hours": 1.0, "Engineering_Hours": 1.0,
        "Roughing_Cycle_Time_hr": 1.0, "Finishing_Cycle_Time_hr": 0.6, "Semi_Finish_Cycle_Time_hr": 0.4,
        "Deburr_Hours": 0.6, "Final_Inspection_Hours": 0.6,
        "Assembly_Hours": 0.5
    }
    caps_general = {
        "Programming_Hours": 4.0, "CAM_Programming_Hours": 4.0, "Engineering_Hours": 8.0,
        "Deburr_Hours": 2.0, "Final_Inspection_Hours": 2.0,
    }

    for k, v in list(h.items()):
        v = max(0.0, float(v or 0))
        cap = (caps_simple if simple else caps_general).get(k, None)
        if cap is not None:
            v = min(v, cap)
        h[k] = v

    # Minutes sanity
    if "CMM_RunTime_min" in h:
        h["CMM_RunTime_min"] = max(0.0, min(float(h["CMM_RunTime_min"] or 0), 120.0))

    est["hours"] = h
    return est

def apply_llm_hours_to_variables(df, est: dict, allow_overwrite_nonzero=False, log: dict | None = None):
    """
    Writes LLM hour estimates into the Variables df.
    Only touches time rows; if allow_overwrite_nonzero=False, it will skip rows that already have a non-zero value.
    """
    import pandas as pd
    mapping = {
        "Programming_Hours":              "Programming Hours",
        "CAM_Programming_Hours":          "CAM Programming Hours",
        "Engineering_Hours":              "Engineering (Docs/Fixture Design) Hours",
        "Fixture_Build_Hours":            "Fixture Build Hours",

        "Roughing_Cycle_Time_hr":         "Roughing Cycle Time",
        "Semi_Finish_Cycle_Time_hr":      "Semi-Finish Cycle Time",
        "Finishing_Cycle_Time_hr":        "Finishing Cycle Time",

        "OD_Turning_Hours":               "OD Turning Hours",
        "ID_Bore_Drill_Hours":            "ID Boring/Drilling Hours",
        "Threading_Hours":                "Threading Hours",
        "Cutoff_Hours":                   "Cut-Off / Parting Hours",

        "WireEDM_Hours":                  "WEDM Hours",
        "SinkerEDM_Hours":                "Sinker EDM Hours",

        "Grinding_Surface_Hours":         "Grinding (Surface) Hours",
        "Grinding_ODID_Hours":            "Grinding (OD/ID) Hours",
        "Jig_Grind_Hours":                "Jig Grind Hours",

        "Lapping_Hours":                  "Lapping Hours",
        "Deburr_Hours":                   "Deburr Hours",
        "Tumble_Hours":                   "Tumbling Hours",
        "Blast_Hours":                    "Bead Blasting Hours",
        "Laser_Mark_Hours":               "Laser Mark Hours",
        "Masking_Hours":                  "Masking Hours",

        "InProcess_Inspection_Hours":     "In-Process Inspection Hours",
        "Final_Inspection_Hours":         "Final Inspection Hours",
        "CMM_Programming_Hours":          "CMM Programming Hours",
        "CMM_RunTime_min":                "CMM Run Time min",

        "Saw_Waterjet_Hours":             "Sawing Hours",
        "Assembly_Hours":                 "Assembly Hours",
        "Packaging_Labor_Hours":          "Packaging Labor Hours",
        "EHS_Hours":                      "EHS Training/Compliance Hours"
    }

    def upsert(name, value, dtype="number"):
        nonlocal df
        # ensure core columns exist
        for c in ("Item", "Example Values / Options", "Data Type / Input Method"):
            if c not in df.columns: df[c] = ""
        m = df["Item"].astype(str).str.fullmatch(str(name), case=False, na=False)
        if m.any():
            cur = pd.to_numeric(df.loc[m, "Example Values / Options"], errors="coerce").fillna(0.0).iloc[0]
            if (not allow_overwrite_nonzero) and (float(cur) != 0.0):
                return False
            df.loc[m, "Example Values / Options"] = value
            df.loc[m, "Data Type / Input Method"] = dtype
        else:
            row = {c: None for c in df.columns}
            row["Item"] = name
            row["Example Values / Options"] = value
            row["Data Type / Input Method"] = dtype
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return True

    applied = {}
    for k, v in (est.get("hours") or {}).items():
        item = mapping.get(k)
        if not item: continue
        ok = upsert(item, round(float(v or 0), 3), "number")
        if ok: applied[item] = v

    # Setups
    setups = est.get("setups") or {}
    if "Milling_Setups" in setups:
        upsert("Number of Milling Setups", int(setups["Milling_Setups"]), "number")
    if "Setup_Hours_per_Setup" in setups:
        upsert("Setup Hours / Setup", round(float(setups["Setup_Hours_per_Setup"]), 3), "number")

    # Inspection flags ? set to 0/1 as numbers
    insp = est.get("inspection", {}) or {}
    if "FAIR_Required" in insp:
        upsert("FAIR Required", 1 if insp.get("FAIR_Required") else 0, "number")
    if "Source_Inspection_Required" in insp:
        upsert("Source Inspection Requirement", 1 if insp.get("Source_Inspection_Required") else 0, "number")

    if log is not None:
        log["llm_hours_applied"] = applied
    return df

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
        r"\b(Material\s*MOQ)\b", r"\b(Material\s*Surcharge|Volatility)\b",
        r"\b(Material\s*Cost|Raw\s*Material\s*Cost)\b",

        r"(Programming|CAM\s*Programming|2D\s*CAM|3D\s*CAM|Simulation|Verification|DFM\s*Review|Tool\s*Library|Setup\s*Sheets)",
        r"\b(CAM\s*Programming|CAM\s*Sim|Post\s*Processing)\b",
        r"(Fixture\s*Design|Process\s*Sheet|Traveler|Documentation|Complex\s*Assembly\s*Doc)",

        r"(Fixture\s*Build|Custom\s*Fixture\s*Build)", r"(Fixture\s*Material\s*Cost|Fixture\s*Hardware)",

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

def _items_used_by_estimator(df):
    pats = _estimator_patterns()
    items = df["Item"].astype(str)
    used = []
    import re as _re
    for it in items:
        if any(p.search(it) for p in pats):
            # Skip explicit Rate rows to prevent LLM from editing machine rates
            if _re.search(r"\brate\b", it, _re.I):
                continue
            used.append(it)
    return used

# --- Build prompt + run model -----------------------------------------------
def _parse_pct_like(x):
    try:
        v = float(x)
        return v/100.0 if v > 1.0 else v
    except Exception:
        return None

def build_llm_sheet_prompt(geo: dict, allowed_items: list[str], params_snapshot: dict) -> tuple[str, str]:
    sys = (
        "You are a manufacturing estimator for precision machining (milling/turning/EDM/grinding). "
        "You will receive:\n" 
        "1) GEO_* features extracted from CAD (units in mm/mm^2/mm^3), and\n" 
        "2) a list of allowed Variables sheet row names you may edit.\n" 
        "Return STRICT JSON only.\n" 
        "Rules:\n" 
        "- Only edit rows that are in the provided allowed_items list.\n" 
        "- Use numbers only (no text). Time is in HOURS unless the row name includes 'min'.\n" 
        "- Pass counts/integers for fields like '... Passes' or '... Count'.\n" 
        "- Percents must be decimals (e.g., 0.20 for 20%).\n" 
        "- Be conservative; if you are unsure, skip the edit.\n" 
        "- You may also suggest param nudges in 'params' for: OEE_EfficiencyPct, FiveAxisMultiplier, TightToleranceMultiplier,\n" 
        "  MillingConsumablesPerHr, TurningConsumablesPerHr, EDMConsumablesPerHr, GrindingConsumablesPerHr, InspectionConsumablesPerHr,\n" 
        "  UtilitiesPerSpindleHr, ConsumablesFlat. Do not include other keys.\n"
    )
    import json, textwrap
    u = {
        "geo": geo,
        "allowed_items": allowed_items,
        "params_snapshot": {k: params_snapshot.get(k) for k in [
            "OEE_EfficiencyPct","FiveAxisMultiplier","TightToleranceMultiplier",
            "MillingConsumablesPerHr","TurningConsumablesPerHr","EDMConsumablesPerHr",
            "GrindingConsumablesPerHr","InspectionConsumablesPerHr",
            "UtilitiesPerSpindleHr","ConsumablesFlat"
        ]},
        "output_shape": {
            "sheet_edits": [{"item": "<exact string from allowed_items>", "value": 0.0, "why": "<short reason, optional>"}],
            "params": {
                "OEE_EfficiencyPct": 1.0,
                "FiveAxisMultiplier": 1.0,
                "TightToleranceMultiplier": 1.0,
                "MillingConsumablesPerHr": 0.0,
                "TurningConsumablesPerHr": 0.0,
                "EDMConsumablesPerHr": 0.0,
                "GrindingConsumablesPerHr": 0.0,
                "InspectionConsumablesPerHr": 0.0,
                "UtilitiesPerSpindleHr": 0.0,
                "ConsumablesFlat": 0.0
                # you may also include optional reasons per param as:
                # "_why": {"OEE_EfficiencyPct":"...", "FiveAxisMultiplier":"...", ...}
            }
        }
    }
    user = textwrap.dedent(
        "Given the following, produce JSON ONLY in the shape shown above. "
        "Prefer these typical relationships:\n" 
        "- WEDM: more path length/thickness -> higher 'EDM Passes' (int) or 'WEDM Hours'; adjust 'EDM Cut Rate_mm^2/min' if too aggressive.\n" 
        "- Grinding: large grind volume -> adjust 'Grind MRR_mm^3/min', 'Grinding Passes', and 'Dress Frequency passes' / 'Dress Time min'.\n" 
        "- Milling setups: high face count or multiple normals -> increase 'Number of Milling Setups' and 'Setup Hours / Setup'.\n" 
        "- Small min wall or high area/volume -> consider 'Thin Wall Factor' and 'Tolerance Multiplier'.\n" 
        "- Inspection: very small parts or tight tolerance -> bump 'CMM Run Time min' or 'Final Inspection'.\n" 
        "- Only include edits you are confident about.\n\n"
        f"INPUT:\n{json.dumps(u, ensure_ascii=False)}"
    )
    return sys, user

def llm_sheet_and_param_overrides(geo: dict, df, params: dict, model_path: str) -> dict:
    allowed = _items_used_by_estimator(df)
    if not allowed:
        return {"sheet_edits": [], "params": {}, "meta": {}}

    sys, usr = build_llm_sheet_prompt(geo, allowed, params)
    prompt_sha = hashlib.sha256((sys + "\n" + usr).encode("utf-8")).hexdigest()

    error_text = ""
    raw_text = ""
    usage = {}
    try:
        llm = _LocalLLM(model_path)
        parsed, raw_text, usage = llm.ask_json(sys, usr, temperature=0.15, max_tokens=900)
        if not isinstance(parsed, dict):
            parsed = parse_llm_json(raw_text)
        js = parsed if isinstance(parsed, dict) else {}
        model_name = Path(model_path).name
    except Exception as e:
        js, model_name = {}, "LLM-unavailable"
        try:
            error_text = f"{type(e).__name__}: {e}"
        except Exception:
            error_text = "LLM error"

    sheet_edits = []
    for e in js.get("sheet_edits", []):
        item = str(e.get("item","")); item = item.strip()
        if item and item in allowed:
            val = e.get("value", None)
            why = e.get("why", "").strip() if isinstance(e.get("why",""), str) else ""
            if isinstance(val, (int, float, str)):
                try:
                    v = float(val)
                except Exception:
                    v = _parse_pct_like(val)
                    if v is None:
                        continue
                sheet_edits.append({"item": item, "value": v, "why": why})

    param_allow = {
        "OEE_EfficiencyPct","FiveAxisMultiplier","TightToleranceMultiplier",
        "MillingConsumablesPerHr","TurningConsumablesPerHr","EDMConsumablesPerHr",
        "GrindingConsumablesPerHr","InspectionConsumablesPerHr",
        "UtilitiesPerSpindleHr","ConsumablesFlat"
    }
    pmap = js.get("params", {}) if isinstance(js.get("params", {}), dict) else {}
    param_whys = pmap.get("_why", {}) if isinstance(pmap.get("_why", {}), dict) else {}
    param_edits = {}
    for k, v in pmap.items():
        if k == "_why":
            continue
        if k in param_allow:
            try:
                param_edits[k] = float(v)
            except Exception:
                pv = _parse_pct_like(v)
                if pv is not None:
                    param_edits[k] = pv

    meta = {
        "model": model_name,
        "prompt_sha256": prompt_sha,
        "allowed_items_count": len(allowed),
        "error": error_text,
        "raw_text": raw_text,
        "usage": usage,
    }
    return {"sheet_edits": sheet_edits, "params": param_edits, "param_whys": param_whys, "allowed_items": allowed, "meta": meta}

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

def apply_param_edits_to_overrides_ui(self, param_edits: dict):
    if not param_edits:
        return
    for k, v in param_edits.items():
        if hasattr(self, "param_vars") and k in self.param_vars:
            self.param_vars[k].set(str(v))
        else:
            self.params[k] = v
    if hasattr(self, "apply_overrides"):
        try:
            self.apply_overrides()
            return
        except Exception:
            pass
    # Fallback normalization if apply_overrides is unavailable
    def fnum(s, d=0.0):
        try: return float(str(s).strip())
        except Exception: return d
    p = {k:fnum(self.param_vars[k].get(), self.params.get(k,0.0)) for k in getattr(self, 'param_vars', {}).keys()}
    for kk in list(p.keys()):
        if kk.endswith("Pct"):
            params[kk] = parse_pct(params[kk])
    self.params.update(p)

# ================== LLM DECISION LOG / AUDIT ==================
import hashlib, json as _json_audit, time as _time_audit
LOGS_DIR = Path(r"D:\\CAD_Quoting_Tool\\Logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _now_iso():
    return _time_audit.strftime("%Y-%m-%dT%H:%M:%S")

def _coerce_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return x

def _used_item_values(df, used_items):
    vals = {}
    items = df["Item"].astype(str)
    for name in used_items:
        m = items.str.fullmatch(name, case=False, na=False)
        if m.any():
            v = df.loc[m, "Example Values / Options"].iloc[0]
            vals[name] = _coerce_num(v)
    return vals

def _diff_map(before: dict, after: dict):
    changes = []
    keys = set(before.keys()) | set(after.keys())
    for k in sorted(keys):
        bv = before.get(k, None)
        av = after.get(k, None)
        if bv != av:
            changes.append({"key": k, "before": bv, "after": av})
    return changes

def render_llm_log_text(log: dict) -> str:
    m  = lambda v: f"${{float(v):,.2f}}" if isinstance(v, (int, float)) else str(v)
    lines = []
    lines.append(f"LLM DECISION LOG ï¿½ {log['timestamp']}")
    lines.append(f"Model: {log.get('model','?')}  |  Prompt SHA256: {log.get('prompt_sha256','')[:12]}ï¿½")
    if log.get('llm_error'):
        lines.append(f"LLM error: {log['llm_error']}")
    lines.append(f"Allowed items: {log.get('allowed_items_count',0)} | Sheet edits suggested/applied: {log.get('sheet_edits_suggested',0)}/{log.get('sheet_edits_applied',0)}")
    lines.append("")
    lines.append(f"Price before: {m(log['price_before'])}   ?   after: {m(log['price_after'])}   " 
                 f"?: {m(log['price_after']-log['price_before'])} " 
                 f"({(((log['price_after']-log['price_before'])/log['price_before']*100) if log['price_before'] else 0):.2f}%)")
    lines.append("")
    if log.get("sheet_changes"):
        lines.append("Sheet changes:")
        for ch in log["sheet_changes"]:
            why = f"  ï¿½ why: {ch['why']}" if ch.get("why") else ""
            lines.append(f"  ï¿½ {ch['key']}: {ch['before']} ? {ch['after']}{why}")
        lines.append("")
    if log.get("param_changes"):
        lines.append("Param nudges:")
        for ch in log["param_changes"]:
            why = f"  ï¿½ why: {ch['why']}" if ch.get("why") else ""
            lines.append(f"  ï¿½ {ch['key']}: {ch['before']} ? {ch['after']}{why}")
        lines.append("")
    if log.get("geo_summary"):
        lines.append("GEO summary:")
        for k, v in log["geo_summary"]:
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines)
def save_llm_log_json(log: dict) -> Path:
    fname = f"llm_log_{_time_audit.strftime('%Y%m%d_%H%M%S')}.json"
    path = LOGS_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        _json_audit.dump(log, f, indent=2)
    return path
# =============================================================

# ----------------- Variables & quote -----------------
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

CORE_COLS = ["Item", "Example Values / Options", "Data Type / Input Method"]

def _coerce_core_types(df_core: "pd.DataFrame") -> "pd.DataFrame":
    """Light normalization for estimator expectations."""
    core = df_core.copy()
    core["Item"] = core["Item"].astype(str)
    core["Data Type / Input Method"] = core["Data Type / Input Method"].astype(str).str.lower()
    # Leave "Example Values / Options" as-is (can be text or number); estimator coerces later.
    return core

def sanitize_vars_df(df_full: "pd.DataFrame") -> "pd.DataFrame":
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

def read_variables_file(path: str, return_full: bool = False):
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
        df_full = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError("Variables must be .xlsx or .csv")

    core = sanitize_vars_df(df_full)

    return (core, df_full) if return_full else core

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
    nre_detail   = breakdown.get("nre_detail", {}) or {}
    nre          = breakdown.get("nre", {}) or {}
    material     = breakdown.get("material", {}) or {}
    process_costs= breakdown.get("process_costs", {}) or {}
    pass_through = breakdown.get("pass_through", {}) or {}
    applied_pcts = breakdown.get("applied_pcts", {}) or {}
    process_meta = {str(k).lower(): (v or {}) for k, v in (breakdown.get("process_meta", {}) or {}).items()}
    rates        = breakdown.get("rates", {}) or {}
    params       = breakdown.get("params", {}) or {}
    qty          = int(breakdown.get("qty", 1) or 1)
    price        = float(result.get("price", totals.get("price", 0.0)))

    # Optional: LLM decision bullets can be placed either on result or breakdown
    llm_notes = (result.get("llm_notes") or breakdown.get("llm_notes") or [])[:8]

    # ---- helpers -------------------------------------------------------------
    divider = "-" * int(page_width)

    def _m(x) -> str:
        return f"{currency}{float(x):,.2f}"

    def _h(x) -> str:
        return f"{float(x):.2f} hr"

    def _pct(x) -> str:
        return f"{float(x or 0.0) * 100:.1f}%"

    def _fmt_dim(val) -> str:
        if isinstance(val, (int, float)):
            return f"{float(val):.3f}".rstrip("0").rstrip(".")
        if val is None:
            return "—"
        text = str(val).strip()
        return text if text else "—"

    def write_line(s: str, indent: str = ""):
        lines.append(f"{indent}{s}")

    def row(label: str, val: float, indent: str = ""):
        # left-label, right-amount aligned to page_width
        left = f"{indent}{label}"
        right = _m(val)
        pad = max(1, page_width - len(left) - len(right))
        lines.append(f"{left}{' ' * pad}{right}")

    def add_process_notes(key: str, indent: str = "    "):
        k = str(key).lower()
        meta = process_meta.get(k) or {}
        # show hours/rate if available
        hr  = meta.get("hr")
        rate= meta.get("rate") or rates.get(k.title() + "Rate")
        if hr:
            write_line(f"{_h(hr)} @ {_m(rate or 0)}/hr", indent)

    def add_pass_basis(key: str, indent: str = "    "):
        basis_map = breakdown.get("pass_basis", {}) or {}
        info = basis_map.get(key) or {}
        txt = info.get("basis") or info.get("note")
        if txt:
            write_line(str(txt), indent)

    # ---- header --------------------------------------------------------------
    lines: list[str] = []
    ui_vars = result.get("ui_vars") or {}
    if not isinstance(ui_vars, dict):
        ui_vars = {}
    lines.append(f"QUOTE SUMMARY - Qty {qty}")
    lines.append(divider)
    row("Final Price per Part:", price)
    row("Total Labor Cost:", float(totals.get("labor_cost", 0.0)))
    row("Total Direct Costs:", float(totals.get("direct_costs", 0.0)))
    lines.append("")

    narrative = result.get("narrative") or breakdown.get("narrative")
    if narrative:
        lines.append("Why this price")
        lines.append(divider)
        if isinstance(narrative, str):
            parts = [seg.strip() for seg in re.split(r"(?<=\.)\s+", narrative) if seg.strip()]
            if not parts:
                parts = [narrative.strip()]
        else:
            parts = [str(line).strip() for line in narrative if str(line).strip()]
        for line in parts:
            write_line(line, "  ")
        lines.append("")

    # ---- material & stock (compact; shown only if we actually have data) -----
    mat_lines = []
    if material:
        mass_g = material.get("mass_g")
        upg    = material.get("unit_price_per_g")
        minchg = material.get("supplier_min_charge")
        surpct = material.get("surcharge_pct")
        matcost= material.get("material_cost")
        scrap  = material.get("scrap_pct", None)  # will show only if present in breakdown
        unit_price_kg = material.get("unit_price_usd_per_kg")
        unit_price_lb = material.get("unit_price_usd_per_lb")
        price_source  = material.get("unit_price_source") or material.get("source")
        price_asof    = material.get("unit_price_asof")

        have_any = any(v for v in [mass_g, upg, minchg, surpct, matcost, scrap])
        if have_any:
            mat_lines.append("Material & Stock")
            mat_lines.append(divider)
            if matcost or show_zeros: row("Material Cost (computed):", float(matcost or 0.0))
            if mass_g or show_zeros:  write_line(f"Mass: {float(mass_g or 0.0):,.1f} g", "  ")
            if upg or show_zeros:
                per_kg = f"{_m(unit_price_kg)} / kg" if unit_price_kg else ""
                per_lb = f"{_m(unit_price_lb)} / lb" if unit_price_lb else ""
                per_g = f"{currency}{float(upg or 0):,.4f} / g"
                extras: list[str] = []
                display_line = per_g
                if per_kg:
                    display_line = per_kg
                    if per_lb:
                        extras.append(per_lb)
                    extras.append(per_g)
                elif per_lb:
                    display_line = per_lb
                    extras.append(per_g)
                if price_asof:
                    extras.append(f"as of {price_asof}")
                extra = f" ({', '.join(extras)})" if extras else ""
                write_line(f"Unit Price: {display_line}{extra}", "  ")
            if price_source:
                write_line(f"Source: {price_source}", "  ")
            if minchg or show_zeros:  write_line(f"Supplier Min Charge: {_m(minchg or 0)}", "  ")
            if surpct or show_zeros:  write_line(f"Surcharge: {_pct(surpct or 0)}", "  ")
            if scrap is not None:     write_line(f"Scrap %: {_pct(scrap)}", "  ")
            stock_L = _fmt_dim(ui_vars.get("Plate Length (in)"))
            stock_W = _fmt_dim(ui_vars.get("Plate Width (in)"))
            th_in = ui_vars.get("Thickness (in)")
            if th_in in (None, ""):
                th_in = g.get("thickness_in_guess")
            if not th_in:
                th_in = 1.0
            stock_T = _fmt_dim(th_in)
            mat_lines.append(f"  Stock used: {stock_L} × {stock_W} × {stock_T} in")
            mat_lines.append("")

    lines.extend(mat_lines)

    # ---- NRE / Setup costs ---------------------------------------------------
    lines.append("NRE / Setup Costs (per lot)")
    lines.append(divider)
    prog = nre_detail.get("programming") or {}
    fix  = nre_detail.get("fixture") or {}

    # Programming & Eng (auto-hide if zero unless show_zeros)
    if (prog.get("per_lot", 0.0) > 0) or show_zeros or any(prog.get(k) for k in ("prog_hr", "cam_hr", "eng_hr")):
        row("Programming & Eng:", float(prog.get("per_lot", 0.0)))
        if prog.get("prog_hr"): write_line(f"- Programmer: {_h(prog['prog_hr'])} @ {_m(prog.get('prog_rate', 0))}/hr", "    ")
        if prog.get("cam_hr"):  write_line(f"- CAM: {_h(prog['cam_hr'])} @ {_m(prog.get('cam_rate', 0))}/hr", "    ")
        if prog.get("eng_hr"):  write_line(f"- Engineering: {_h(prog['eng_hr'])} @ {_m(prog.get('eng_rate', 0))}/hr", "    ")

    # Fixturing (with renamed subline)
    if (fix.get("per_lot", 0.0) > 0) or show_zeros or any(fix.get(k) for k in ("build_hr", "mat_cost")):
        row("Fixturing:", float(fix.get("per_lot", 0.0)))
        if fix.get("build_hr"):
            write_line(f"- Build Labor: {_h(fix['build_hr'])} @ {_m(fix.get('build_rate', 0))}/hr", "    ")
        write_line(f"- Fixture Material Cost: {_m(fix.get('mat_cost', 0.0))}", "    ")

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
    lines.append("Process & Labor Costs")
    lines.append(divider)
    proc_total = 0.0
    for key, value in sorted((process_costs or {}).items(), key=lambda kv: kv[1], reverse=True):
        if (value > 0) or show_zeros:
            label = key.replace("_", " ").title()
            row(label, float(value), indent="  ")
            add_process_notes(key, indent="    ")
            proc_total += float(value or 0.0)
    row("Total", proc_total, indent="  ")
    lines.append("")

    # ---- Pass-Through & Direct (auto include non-zeros; sorted desc) --------
    lines.append("Pass-Through & Direct Costs")
    lines.append(divider)
    pass_total = 0.0
    for key, value in sorted((pass_through or {}).items(), key=lambda kv: kv[1], reverse=True):
        if (value > 0) or show_zeros:
            # cosmetic: "consumables_hr_cost" → "Consumables /Hr Cost"
            label = key.replace("_", " ").replace("hr", "/hr").title()
            row(label, float(value), indent="  ")
            add_pass_basis(key, indent="    ")
            pass_total += float(value or 0.0)
    row("Total", pass_total, indent="  ")
    lines.append("")

    # ---- Pricing ladder ------------------------------------------------------
    lines.append("Pricing Ladder")
    lines.append(divider)
    subtotal         = float(totals.get("subtotal",         proc_total + pass_total))
    with_overhead    = float(totals.get("with_overhead",    subtotal))
    with_ga          = float(totals.get("with_ga",          with_overhead))
    with_contingency = float(totals.get("with_contingency", with_ga))
    with_expedite    = float(totals.get("with_expedite",    with_contingency))

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

    # ---- Why This Price (your existing free-form LLM explanation) -----------
    if llm_explanation:
        lines.append("Why This Price")
        lines.append(divider)
        import textwrap
        for snippet in textwrap.wrap(llm_explanation.strip(), width=page_width):
            lines.append(snippet)

    return "\n".join(lines)
# ===== QUOTE CONFIG (edit-friendly) ==========================================
RATES_DEFAULT = {
    "ProgrammingRate": 120.0,
    "MillingRate": 120.0,
    "DrillingRate": 120.0,
    "TurningRate": 115.0,
    "WireEDMRate": 140.0,
    "SinkerEDMRate": 150.0,
    "SurfaceGrindRate": 120.0,
    "ODIDGrindRate": 125.0,
    "JigGrindRate": 150.0,
    "LappingRate": 130.0,
    "InspectionRate": 110.0,
    "FinishingRate": 100.0,
    "SawWaterjetRate": 100.0,
    "FixtureBuildRate": 120.0,
    "AssemblyRate": 110.0,
    "EngineerRate": 140.0,
    "CAMRate": 125.0,
}

PARAMS_DEFAULT = {
    "OverheadPct": 0.15,
    "GA_Pct": 0.08,
    "ContingencyPct": 0.00,
    "ExpeditePct": 0.00,
    "MarginPct": 0.35,
    "InsurancePct": 0.00,
    "VendorMarkupPct": 0.00,
    "MinLotCharge": 0.00,
    "Quantity": 1,
    # Machine capability (shared with LLM for drilling/setup planning)
    "MachineMaxRPM": 8000,
    "MachineMaxTorqueNm": 70.0,
    "MachineMaxZTravel_mm": 500.0,
    # Light stock catalog so the LLM can pick reasonable blanks
    "StockCatalog": [
        {
            "sku": "plate_steel_0.25_12x12",
            "material": "steel",
            "form": "plate",
            "length_mm": 304.8,
            "width_mm": 304.8,
            "thickness_mm": 6.35,
            "max_weight_kg": 7.0,
        },
        {
            "sku": "plate_steel_0.5_12x12",
            "material": "steel",
            "form": "plate",
            "length_mm": 304.8,
            "width_mm": 304.8,
            "thickness_mm": 12.7,
            "max_weight_kg": 14.0,
        },
        {
            "sku": "plate_aluminum_0.25_12x24",
            "material": "aluminum",
            "form": "plate",
            "length_mm": 609.6,
            "width_mm": 304.8,
            "thickness_mm": 6.35,
            "max_weight_kg": 12.0,
        },
        {
            "sku": "plate_aluminum_0.5_12x24",
            "material": "aluminum",
            "form": "plate",
            "length_mm": 609.6,
            "width_mm": 304.8,
            "thickness_mm": 12.7,
            "max_weight_kg": 24.0,
        },
    ],
    # Programming heuristics
    "ProgSimpleDim_mm": 80.0,
    "ProgCapHr": 1.0,
    "ProgMaxToMillingRatio": 1.5,
    # Efficiency & multipliers
    "OEE_EfficiencyPct": 1.0,
    "FiveAxisMultiplier": 1.0,
    "TightToleranceMultiplier": 1.0,
    # Adders
    "MillingConsumablesPerHr": 4.0,
    "TurningConsumablesPerHr": 3.0,
    "EDMConsumablesPerHr": 6.0,
    "GrindingConsumablesPerHr": 3.0,
    "InspectionConsumablesPerHr": 1.0,
    "UtilitiesPerSpindleHr": 2.5,
    "ConsumablesFlat": 35.0,
    # Misc
    "MaterialOther": 50.0,
    "MaterialVendorCSVPath": "",
    "LLMModelPath": "",
}

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


@dataclass
class PricingRegistry:
    """Provide access to default shop rates for pricing calculations."""

    default_rates: Dict[str, float] = field(default_factory=lambda: copy.deepcopy(RATES_DEFAULT))

    def copy_default_rates(self) -> Dict[str, float]:
        return copy.deepcopy(self.default_rates)


@dataclass
class GeometryLoader:
    """Expose helpers used when enriching DXF geometry with extra context."""

    build_geo_from_dxf: Optional[Callable[[str], Dict[str, Any]]] = None


@dataclass
class LLMClient:
    """Provide callables for constructing optional LLM helpers."""

    primary_loader: Optional[Callable[[], Any]] = None
    fallback_loader: Optional[Callable[[], Any]] = None

    def load_primary(self) -> Any:
        if self.primary_loader is None:
            raise RuntimeError("No primary LLM loader configured")
        return self.primary_loader()

    def load_fallback(self) -> Any:
        if self.fallback_loader is None:
            raise RuntimeError("No fallback LLM loader configured")
        return self.fallback_loader()

# Common regex pieces (kept non-capturing to avoid pandas warnings)
TIME_RE = r"\b(?:hours?|hrs?|hr|time|min(?:ute)?s?)\b"
MONEY_RE = r"(?:rate|/hr|per\s*hour|per\s*hr|price|cost|$)"

# ===== QUOTE HELPERS ========================================================

def alt(*terms: str) -> str:
    """Build a non-capturing (?:a|b|c) regex; terms are already escaped/regex-ready."""
    return r"(?:%s)" % "|".join(terms)

def pct(value: Any, default: float = 0.0) -> float:
    """Accept 0-1 or 0-100 and return 0-1."""
    try:
        v = float(value)
        return v / 100.0 if v > 1.0 else v
    except Exception:
        return default


PRICE_CACHE: dict[str, tuple[float, float, str]] = {}
CACHE_TTL_S = 60 * 30  # 30 minutes


def _usd_per_kg_from_quote(symbol: str, quote: float, basis: str) -> float:
    if basis == "index_usd_per_tonne":
        return float(quote) / 1000.0
    if basis == "usd_per_troy_oz":
        return float(quote) / 31.1034768 * 1000.0
    if basis == "usd_per_lb":
        return float(quote) / 2.2046226218
    return float(quote)


class PriceProvider:
    name = "base"

    def get(self, symbol: str) -> tuple[float, str]:
        raise NotImplementedError


class MetalsAPI(PriceProvider):
    """Metals-API / Commodities-API compatible provider."""

    name = "metals_api"
    base_url = "https://api.metals-api.com/v1/latest"

    def get(self, symbol: str) -> tuple[float, str]:
        api_key = os.getenv("METALS_API_KEY")
        if not api_key:
            raise RuntimeError("METALS_API_KEY not set")
        url = f"{self.base_url}?access_key={api_key}&base=USD&symbols={symbol}"
        with urllib.request.urlopen(url, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not data.get("success", True) and "rates" not in data:
            raise RuntimeError(str(data)[:200])
        rate = float(data["rates"][symbol])
        ts = data.get("timestamp")
        asof = time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)) if ts else "now"
        return rate, asof


class VendorCSV(PriceProvider):
    name = "vendor_csv"

    def __init__(self, path: str):
        self.path = path
        self._rows: dict[str, float] | None = None

    def _load(self) -> None:
        if self._rows is not None:
            return
        import csv

        self._rows = {}
        if os.path.isfile(self.path):
            with open(self.path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if not row:
                        continue
                    sym, usd_per_kg, *_ = row
                    self._rows[sym.strip().upper()] = float(usd_per_kg)

    def get(self, symbol: str) -> tuple[float, str]:
        self._load()
        assert self._rows is not None
        price = self._rows.get(symbol.upper())
        if price is None:
            raise KeyError(symbol)
        return float(price), "vendor_csv"


PROVIDERS: list[PriceProvider] = []
CURRENT_VENDOR_CSV_PATH: str | None = None


def init_providers(vendor_csv_path: str | None = None) -> None:
    global PROVIDERS, CURRENT_VENDOR_CSV_PATH
    PROVIDERS = []
    CURRENT_VENDOR_CSV_PATH = vendor_csv_path or ""
    if vendor_csv_path:
        PROVIDERS.append(VendorCSV(vendor_csv_path))
    PROVIDERS.append(MetalsAPI())


def _try_wieland_price(candidates: list[str]) -> tuple[float | None, str]:
    """Attempt to use the Wieland scraper for USD/kg pricing."""
    try:
        from wieland_scraper import get_live_material_price_usd_per_kg
    except Exception:
        return None, ""

    for candidate in candidates:
        label = str(candidate or "").strip()
        if not label:
            continue
        try:
            price, source = get_live_material_price_usd_per_kg(label, fallback_usd_per_kg=-1.0)
        except Exception:
            continue
        if not isinstance(price, (int, float)):
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        if str(source or "").lower().startswith("house_rate"):
            continue
        return float(price), str(source)

    return None, ""


def get_usd_per_kg(symbol: str, basis: str) -> tuple[float, str]:
    cached = PRICE_CACHE.get(symbol)
    now = time.time()
    if cached and (now - cached[0] < CACHE_TTL_S):
        _, usd_per_kg, source = cached
        return usd_per_kg, source

    last_err: Exception | None = None
    for provider in PROVIDERS:
        try:
            quote, asof = provider.get(symbol)
            usd_per_kg = _usd_per_kg_from_quote(symbol, quote, basis)
            PRICE_CACHE[symbol] = (now, usd_per_kg, f"{provider.name}@{asof}")
            return usd_per_kg, f"{provider.name}@{asof}"
        except Exception as err:  # pragma: no cover - provider failures fall back below
            last_err = err

    raise RuntimeError(f"All price providers failed for {symbol}: {last_err}")


def compute_material_cost(material_name: str,
                          mass_kg: float,
                          scrap_frac: float,
                          overrides: dict[str, Any] | None,
                          vendor_csv: str | None,
                          *,
                          default_material_display: str = DEFAULT_MATERIAL_DISPLAY) -> tuple[float, dict[str, Any]]:
    global CURRENT_VENDOR_CSV_PATH
    vendor_csv = vendor_csv or ""
    if (not PROVIDERS) or (vendor_csv != (CURRENT_VENDOR_CSV_PATH or "")):
        init_providers(vendor_csv)

    overrides = overrides or {}
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

    usd_per_kg: float | None = None
    source = ""
    basis_used = basis

    # Highest priority: explicit vendor CSV overrides
    if vendor_csv:
        for provider in PROVIDERS:
            if not isinstance(provider, VendorCSV):
                continue
            try:
                vendor_price, vendor_asof = provider.get(symbol)
            except Exception:
                continue
            usd_per_kg = float(vendor_price)
            source = f"{provider.name}@{vendor_asof}" if vendor_asof else provider.name
            basis_used = "usd_per_kg"
            break

    # Next try Wieland scraped data
    if usd_per_kg is None:
        price_candidates = []
        wieland_key = meta.get("wieland_key")
        if wieland_key:
            price_candidates.append(str(wieland_key))
        if material_name:
            price_candidates.append(str(material_name))
        if key:
            price_candidates.append(str(key))
        if symbol:
            price_candidates.append(str(symbol))

        usd_wieland, source_wieland = _try_wieland_price(price_candidates)
        if usd_wieland is not None:
            usd_per_kg = usd_wieland
            source = source_wieland or "wieland"
            basis_used = "usd_per_kg"
        else:
            price_candidates = []

    else:
        price_candidates = []

    # Fallback to live/index providers
    if usd_per_kg is None:
        usd_per_kg, source = get_usd_per_kg(symbol, basis)
        basis_used = basis

    # Final fallback to local resolver (Wieland → CSV → default steel)
    if (usd_per_kg is None) or (not math.isfinite(float(usd_per_kg))) or (usd_per_kg <= 0):
        resolver_name = material_name or ""
        if not resolver_name:
            fallback_key = _normalize_lookup_key(default_material_display)
            resolver_name = MATERIAL_DISPLAY_BY_KEY.get(fallback_key, default_material_display)
        try:
            resolved_price, resolver_source = resolve_material_unit_price(resolver_name, unit="kg")
        except Exception:
            resolved_price, resolver_source = None, ""
        if resolved_price and math.isfinite(float(resolved_price)):
            usd_per_kg = float(resolved_price)
            source = resolver_source or source or "resolver"
            basis_used = "usd_per_kg"

    if usd_per_kg is None:
        usd_per_kg = 0.0
        source = source or "price_unavailable"

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

    detail = {
        "material_name": material_name,
        "symbol": symbol,
        "basis": basis_used,
        "source": source,
        "mass_g_net": float(mass_kg) * 1000.0,
        "mass_g": effective_kg * 1000.0,
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
    return cost, detail


def _material_family(material: str) -> str:
    name = (material or "").strip().lower()
    if not name:
        return "steel"
    if any(tag in name for tag in ("alum", "6061", "7075", "2024", "5052", "5083")):
        return "alum"
    if any(tag in name for tag in ("stainless", "17-4", "17 4", "316", "304", "ss")):
        return "stainless"
    return "steel"


def require_plate_inputs(geo: dict, ui_vars: dict[str, Any] | None) -> None:
    ui_vars = ui_vars or {}
    thickness_val = ui_vars.get("Thickness (in)")
    thickness_in = _coerce_float_or_none(thickness_val)
    if thickness_in is None:
        try:
            thickness_in = float(thickness_val)
        except Exception:
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


def net_mass_kg(plate_L_in, plate_W_in, t_in, hole_d_mm, density_g_cc: float = 7.85):
    if not (plate_L_in and plate_W_in and t_in):
        return None
    try:
        L = float(plate_L_in)
        W = float(plate_W_in)
        T = float(t_in)
    except Exception:
        return None
    if L <= 0 or W <= 0 or T <= 0:
        return None
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
    return mass_g / 1000.0 if mass_g else 0.0


def estimate_drilling_hours(hole_diams_mm: list[float], thickness_mm: float, mat_key: str) -> float:
    """
    Conservative plate-drilling model with floors so 100+ holes don't collapse to minutes.
    """
    if not hole_diams_mm or thickness_mm <= 0:
        return 0.0

    mat = (mat_key or "").lower()

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
    tfac = max(0.7, min(2.0, thickness_mm / 6.35))

    from collections import Counter

    groups = Counter(round(float(d), 2) for d in hole_diams_mm if d and math.isfinite(d))
    toolchange_s = 15.0

    total_sec = 0.0
    for d, qty in groups.items():
        if qty <= 0:
            continue
        per = sec_per_hole(float(d)) * mfac * tfac
        total_sec += qty * per
        total_sec += toolchange_s

    return total_sec / 3600.0


def estimate_tapping_hours(tap_qty: int, thickness_in: float, mat_key: str) -> float:
    """Simple heuristic for tapping: average 0.20 min/hole.
    If you later classify sizes, split into small/large families and adjust.
    """
    try:
        qty = int(tap_qty or 0)
    except Exception:
        qty = 0
    if qty <= 0:
        return 0.0
    avg_tap_min = 0.20  # minutes per hole
    return (qty * avg_tap_min) / 60.0


def _drilling_floor_hours(hole_count: int) -> float:
    min_sec_per_hole = 9.0
    return max((float(hole_count or 0) * min_sec_per_hole) / 3600.0, 0.0)


def validate_drilling_reasonableness(hole_count: int, drill_hr_after_overrides: float) -> tuple[bool, str]:
    floor_hr = _drilling_floor_hours(hole_count)
    ok = drill_hr_after_overrides >= floor_hr
    msg = f"drilling hours ({drill_hr_after_overrides:.2f} h) below floor ({floor_hr:.2f} h) for {hole_count} holes"
    return ok, msg


def validate_quote_before_pricing(
    geo: dict,
    process_costs: dict[str, float],
    pass_through: dict[str, Any],
    process_hours: dict[str, float] | None = None,
) -> None:
    issues: list[str] = []
    hole_cost = sum(float(process_costs.get(k, 0.0)) for k in ("drilling", "milling"))
    if geo.get("hole_diams_mm") and hole_cost < 50:
        issues.append("Unusually low machining time for number of holes.")
    material_cost = float(pass_through.get("Material", 0.0) or 0.0)
    if material_cost < 5.0:
        issues.append("Material cost is near zero; check material & thickness.")
    try:
        hole_count_val = int(float(geo.get("hole_count", 0)))
    except Exception:
        hole_count_val = len(geo.get("hole_diams_mm") or [])
    if hole_count_val <= 0:
        hole_count_val = len(geo.get("hole_diams_mm") or [])
    if issues:
        raise ValueError("Quote blocked:\n- " + "\n- ".join(issues))


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
                          geo: dict[str, Any] | None = None,
                          ui_vars: dict[str, Any] | None = None,
                          quote_state: QuoteState | None = None,
                          reuse_suggestions: bool = False,
                          llm_suggest: Any | None = None) -> Dict[str, Any]:
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

    params_defaults = default_params if default_params is not None else QuoteConfiguration().default_params
    rates_defaults = default_rates if default_rates is not None else PricingRegistry().default_rates
    params = {**params_defaults, **(params or {})}
    rates = {**rates_defaults, **(rates or {})}
    rates.setdefault("DrillingRate", rates.get("MillingRate", 0.0))
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
    quote_state.ui_vars = dict(ui_vars)
    quote_state.rates = dict(rates)
    geo_context = dict(geo or {})
    inner_geo_raw = geo_context.get("geo")
    inner_geo = dict(inner_geo_raw) if isinstance(inner_geo_raw, dict) else {}
    geo_context["geo"] = inner_geo

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
        m  = contains(pattern)
        if not m.any():
            return float(default)

        looks_time   = items.str.contains(TIME_RE,   case=False, regex=True, na=False)
        looks_money  = items.str.contains(MONEY_RE,  case=False, regex=True, na=False)
        typed_money  = dtt.str.contains(r"(?:rate|currency|price|cost)", case=False, na=False)
        excl = looks_money | typed_money
        if exclude_pattern:
            excl = excl | contains(exclude_pattern)

        mm = m & ~excl & looks_time
        if not mm.any():
            return float(default)

        mins_mask = items.str.contains(r"\bmin(?:ute)?s?\b", case=False, regex=True, na=False) & mm
        hrs_mask  = mm & ~mins_mask
        hrs_sum  = pd.to_numeric(vals[hrs_mask],  errors="coerce").fillna(0.0).sum()
        mins_sum = pd.to_numeric(vals[mins_mask], errors="coerce").fillna(0.0).sum()
        return float(hrs_sum) + float(mins_sum) / 60.0

    # --- OPTIONAL: let sheet override params/rates if those rows exist ---
    def sheet_num(pat, default=None):
        v = first_num(pat, float('nan'))
        return v if v == v else default  # NaN check

    def sheet_pct(pat, default=None):
        from math import isnan
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
    rate_from_sheet(r"Rate\s*-\s*CAM",           "CAMRate")
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

    # ---- knobs & qty ---------------------------------------------------------
    OverheadPct    = num_pct(r"\b" + alt('Overhead','Shop Overhead') + r"\b", params["OverheadPct"])
    MarginPct      = num_pct(r"\b" + alt('Margin','Profit Margin') + r"\b", params["MarginPct"])
    GA_Pct         = num_pct(r"\b" + alt('G&A','General\s*&\s*Admin') + r"\b", params["GA_Pct"])
    ContingencyPct = num_pct(r"\b" + alt('Contingency','Risk\s*Adder') + r"\b",  params["ContingencyPct"])
    ExpeditePct    = num_pct(r"\b" + alt('Expedite','Rush\s*Fee') + r"\b",      params["ExpeditePct"])

    priority = strv(alt('PM-01_Quote_Priority','Quote\s*Priority'), "").strip().lower()
    if priority not in ("expedite", "critical"):
        ExpeditePct = 0.0

    Qty = max(1, int(first_num(r"\b" + alt('Qty','Lot Size','Quantity') + r"\b", 1)))
    amortize_programming = True  # single switch if you want to expose later

    # ---- geometry (optional) -------------------------------------------------
    GEO_wedm_len_mm = first_num(r"\bGEO__?(?:WEDM_PathLen_mm|EDM_Length_mm)\b", 0.0)
    GEO_vol_mm3     = first_num(r"\bGEO__?(?:Volume|Volume_mm3|Net_Volume_mm3)\b", 0.0)

    # ---- material ------------------------------------------------------------
    vol_cm3       = first_num(r"\b(?:Net\s*Volume|Volume_net|Volume\s*\(cm\^?3\))\b", GEO_vol_mm3 / 1000.0)
    density_g_cc  = first_num(r"\b(?:Density|Material\s*Density)\b", 14.5)
    scrap_pct_raw = num_pct(r"\b(?:Scrap\s*%|Expected\s*Scrap)\b", 0.0)
    scrap_pct = _ensure_scrap_pct(scrap_pct_raw)
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

    # --- SCRAP BASELINE (always defined) ---
    try:
        ui_scrap_val = ui_vars.get("Scrap Percent (%)") if "ui_vars" in locals() else None
    except Exception:
        ui_scrap_val = None

    if ui_scrap_val is None:
        try:
            mask = df["Item"].astype(str).str.fullmatch(r"Scrap Percent \(%\)", case=False, na=False)
            ui_scrap_val = float(df.loc[mask, "Example Values / Options"].iloc[0])
        except Exception:
            ui_scrap_val = 0.0

    scrap_pct_baseline = _ensure_scrap_pct(ui_scrap_val)

    if is_plate_2d:
        length_in_val = _coerce_float_or_none(ui_vars.get("Plate Length (in)"))
        width_in_val = _coerce_float_or_none(ui_vars.get("Plate Width (in)"))
        thickness_mm_val = _coerce_float_or_none(geo_context.get("thickness_mm"))
        if thickness_mm_val is None:
            t_in = _coerce_float_or_none(ui_vars.get("Thickness (in)"))
            if t_in is not None:
                thickness_mm_val = float(t_in) * 25.4
        thickness_mm_val = float(thickness_mm_val or 0.0)
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
        density_lookup = {"steel": 7.85, "stainless": 7.9, "alum": 2.70}
        density_g_cc = density_lookup.get(
            _material_family(geo_context.get("material") or material_name),
            density_lookup["steel"],
        )
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
        net_mass_calc = net_mass_kg(length_in_val, width_in_val, thickness_in_for_mass, holes_for_mass, density_g_cc)
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

    net_mass_g = max(0.0, vol_cm3 * density_g_cc)
    effective_mass_g = net_mass_g * (1.0 + scrap_pct)
    mass_g = effective_mass_g
    mass_kg = net_mass_g / 1000.0

    unit_price_per_g  = first_num(r"\b(?:Material\s*Price.*(?:per\s*g|/g)|Unit\s*Price\s*/\s*g)\b", 0.0)
    supplier_min_charge = first_num(r"\b(?:Supplier\s*Min\s*Charge|min\s*charge)\b", 0.0)
    surcharge_pct = num_pct(r"\b(?:Material\s*Surcharge|Volatility)\b", 0.0)
    explicit_mat  = num(r"\b(?:Material\s*Cost|Raw\s*Material\s*Cost)\b", 0.0)

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
            default_material_display=default_material_display or DEFAULT_MATERIAL_DISPLAY,
        )
    except Exception as err:
        material_detail = {
            "material_name": material_name,
            "mass_g": effective_mass_g,
            "mass_g_net": net_mass_g,
            "scrap_pct": scrap_pct,
            "error": str(err),
        }
        provider_cost = None

    manual_baseline = unit_price_per_g * effective_mass_g
    base_cost = provider_cost if provider_cost is not None else manual_baseline
    base_cost = float(base_cost or 0.0)

    cost_with_min = max(base_cost, float(supplier_min_charge or 0.0))
    cost_with_surcharge = cost_with_min * (1.0 + max(0.0, float(surcharge_pct or 0.0)))
    material_cost = max(cost_with_surcharge, float(explicit_mat or 0.0))

    if material_detail:
        material_detail.update({
            "supplier_min_charge": float(supplier_min_charge or 0.0),
            "surcharge_pct": float(surcharge_pct or 0.0),
            "explicit_cost_override": float(explicit_mat or 0.0),
            "material_cost_before_overrides": base_cost,
            "material_cost": float(material_cost),
        })
        unit_price_from_detail = material_detail.get("unit_price_usd_per_kg")
        if unit_price_from_detail is not None:
            try:
                unit_price_per_g = float(unit_price_from_detail) / 1000.0
            except Exception:
                pass

    material_direct_cost = float(material_cost)

    material_detail_for_breakdown = dict(material_detail)
    material_detail_for_breakdown.setdefault("material_name", material_name)
    material_detail_for_breakdown.setdefault("mass_g", mass_g)
    material_detail_for_breakdown.setdefault("mass_g_net", net_mass_g)
    material_detail_for_breakdown.setdefault("scrap_pct", scrap_pct)
    material_detail_for_breakdown["unit_price_per_g"] = float(unit_price_per_g or 0.0)
    if "unit_price_usd_per_kg" not in material_detail_for_breakdown and unit_price_per_g:
        material_detail_for_breakdown["unit_price_usd_per_kg"] = float(unit_price_per_g) * 1000.0
    material_detail_for_breakdown["material_direct_cost"] = material_direct_cost
    if provider_cost is not None:
        material_detail_for_breakdown["provider_cost"] = float(provider_cost)

    if is_plate_2d:
        mat_for_price = geo_context.get("material") or material_name
        try:
            mat_usd_per_kg, mat_src = resolve_material_unit_price(mat_for_price, unit="kg")
        except Exception:
            mat_usd_per_kg, mat_src = 0.0, ""
        mat_usd_per_kg = float(mat_usd_per_kg or 0.0)
        effective_scrap_source = scrap_pct if scrap_pct else scrap_pct_baseline
        effective_scrap = _ensure_scrap_pct(effective_scrap_source)
        material_cost = mass_kg * (1.0 + effective_scrap) * mat_usd_per_kg
        material_direct_cost = round(material_cost, 2)
        material_detail_for_breakdown.update({
            "material_name": mat_for_price,
            "mass_g_net": mass_kg * 1000.0,
            "mass_g": mass_kg * 1000.0 * (1.0 + effective_scrap),
            "scrap_pct": effective_scrap,
            "unit_price_usd_per_kg": mat_usd_per_kg,
            "unit_price_per_g": mat_usd_per_kg / 1000.0 if mat_usd_per_kg else material_detail_for_breakdown.get("unit_price_per_g", 0.0),
            "unit_price_usd_per_lb": (mat_usd_per_kg / LB_PER_KG) if mat_usd_per_kg else material_detail_for_breakdown.get("unit_price_usd_per_lb"),
            "source": mat_src or material_detail_for_breakdown.get("source", ""),
        })
        material_detail_for_breakdown["material_cost"] = material_direct_cost
        material_detail_for_breakdown["material_direct_cost"] = material_direct_cost
        scrap_pct = effective_scrap

    # ---- programming / cam / dfm --------------------------------------------
    prog_hr = sum_time(r"(?:Programming|2D\s*CAM|3D\s*CAM|Simulation|Verification|DFM|Setup\s*Sheets)", exclude_pattern=r"\bCMM\b")
    cam_hr  = sum_time(r"(?:CAM\s*Programming|CAM\s*Sim|Post\s*Processing)")
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
    if is_simple:
        prog_hr = min(prog_hr, params["ProgCapHr"])
    if milling_hr > 0:
        prog_hr = min(prog_hr, params["ProgMaxToMillingRatio"] * milling_hr)

    programming_cost   = prog_hr * rates["ProgrammingRate"] + cam_hr * rates["CAMRate"] + eng_hr * rates["EngineerRate"]
    prog_per_lot       = programming_cost
    programming_per_part = (prog_per_lot / Qty) if (amortize_programming and Qty > 1) else prog_per_lot

    # ---- fixture -------------------------------------------------------------
    fixture_build_hr = sum_time(r"(?:Fixture\s*Build|Custom\s*Fixture\s*Build)")
    fixture_mat_cost = num(r"(?:Fixture\s*Material\s*Cost|Fixture\s*Hardware)")
    # Explicit fields for clarity downstream
    fixture_material_cost = float(fixture_mat_cost)
    fixture_labor_cost    = float(fixture_build_hr) * float(rates["FixtureBuildRate"])
    fixture_cost          = fixture_labor_cost + fixture_material_cost
    fixture_labor_per_part = (fixture_labor_cost / Qty) if Qty > 1 else fixture_labor_cost
    fixture_per_part       = (fixture_cost / Qty) if Qty > 1 else fixture_cost

    nre_detail = {
        "programming": {
            "prog_hr": float(prog_hr), "prog_rate": rates["ProgrammingRate"],
            "cam_hr": float(cam_hr),   "cam_rate": rates["CAMRate"],
            "eng_hr": float(eng_hr),   "eng_rate": rates["EngineerRate"],
            "per_lot": prog_per_lot, "per_part": programming_per_part,
            "amortized": bool(amortize_programming and Qty > 1)
        },
        "fixture": {
            "build_hr": float(fixture_build_hr), "build_rate": rates["FixtureBuildRate"],
            "labor_cost": float(fixture_labor_cost),
            "mat_cost": float(fixture_material_cost),
            "per_lot": fixture_cost, "per_part": fixture_per_part
        }
    }

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
    inproc_hr   = sum_time(r"(?:In[- ]?Process\s*Inspection)")
    final_hr    = sum_time(r"(?:Final\s*Inspection|Manual\s*Inspection)")
    cmm_prog_hr = sum_time(r"(?:CMM\s*Programming)")
    cmm_run_hr  = sum_time(r"(?:CMM\s*Run\s*Time)\b") + sum_time(r"(?:CMM\s*Run\s*Time\s*min)")
    fair_hr     = sum_time(r"(?:FAIR|ISIR|PPAP)")
    srcinsp_hr  = sum_time(r"(?:Source\s*Inspection)")

    inspection_cost = (
        (inproc_hr + final_hr) * rates["InspectionRate"] +
        cmm_prog_hr * rates["InspectionRate"] +
        cmm_run_hr  * rates["InspectionRate"] +
        fair_hr     * rates["InspectionRate"] +
        srcinsp_hr  * rates["InspectionRate"]
    )

    # Consumables & utilities
    spindle_hr = (eff(milling_hr) + turning_hr + wedm_hr + sinker_hr + surf_grind_hr + jig_grind_hr + odid_grind_hr)
    consumables_hr_cost = (
        params["MillingConsumablesPerHr"]    * eff(milling_hr) +
        params["TurningConsumablesPerHr"]    * turning_hr +
        params["EDMConsumablesPerHr"]        * (wedm_hr + sinker_hr) +
        params["GrindingConsumablesPerHr"]   * (surf_grind_hr + jig_grind_hr + odid_grind_hr) +
        params["InspectionConsumablesPerHr"] * (inproc_hr + final_hr + cmm_prog_hr + cmm_run_hr + fair_hr + srcinsp_hr)
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

    # Outsourced vendors
    heat_treat_cost  = num(r"(?:Outsourced\s*Heat\s*Treat|Heat\s*Treat\s*Cost)")
    coating_cost     = num(r"(?:Plating|Coating|Anodize|Black\s*Oxide|DLC|PVD|CVD)")
    passivation_cost = num(r"(?:Passivation|Cleaning\s*Vendor)")
    outsourced_costs = heat_treat_cost + coating_cost + passivation_cost

    # Packaging / logistics
    packaging_hr      = sum_time(r"(?:Packaging|Boxing|Crating\s*Labor)")
    crate_nre_cost    = num(r"(?:Custom\s*Crate\s*NRE)")
    packaging_mat     = num(r"(?:Packaging\s*Materials|Foam|Trays)")
    shipping_cost     = num(r"(?:Freight|Shipping\s*Cost)")
    insurance_pct     = num_pct(r"(?:Insurance|Liability\s*Adder)", params["InsurancePct"])
    packaging_cost    = packaging_hr * rates["AssemblyRate"] + crate_nre_cost + packaging_mat

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

    drill_material_key = geo_context.get("material") or material_name
    drill_hr = estimate_drilling_hours(hole_diams_list, float(thickness_for_drill or 0.0), drill_material_key or "")
    drill_rate = float(rates.get("DrillingRate") or rates.get("MillingRate", 0.0) or 0.0)
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
    deburr_total_hr = round(deburr_manual_hr_float + deburr_auto_hr_raw, 3)
    deburr_auto_hr = round(deburr_auto_hr_raw, 3)
    deburr_edge_hr = round(deburr_edge_hr_raw, 3)
    deburr_holes_hr = round(deburr_holes_hr_raw, 3)
    deburr_manual_hr_rounded = round(deburr_manual_hr_float, 3)

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
    process_costs["drilling"] = baseline_drill_hr * drill_rate
    process_costs["tapping"] = tapping_hr_rounded * drill_rate
    process_costs["counterbore"] = cbore_hr_rounded * drill_rate
    process_costs["countersink"] = csk_hr_rounded * drill_rate
    deburr_cost = deburr_total_hr * finishing_rate
    process_costs["deburr"] = deburr_cost

    process_costs_baseline = {k: float(v) for k, v in process_costs.items()}

    process_meta = {
        "milling":          {"hr": eff(milling_hr), "rate": rates.get("MillingRate", 0.0)},
        "turning":          {"hr": eff(turning_hr), "rate": rates.get("TurningRate", 0.0)},
        "wire_edm":         {"hr": eff(wedm_hr),    "rate": rates.get("WireEDMRate", 0.0)},
        "sinker_edm":       {"hr": eff(sinker_hr),  "rate": rates.get("SinkerEDMRate", 0.0)},
        "grinding":         {"hr": eff(grinding_hr),"rate": rates.get("SurfaceGrindRate", 0.0)},
        "lapping_honing":   {"hr": lap_hr,          "rate": rates.get("LappingRate", 0.0)},
        "finishing_deburr": {"hr": finishing_misc_hr, "rate": finishing_rate},
        "inspection":       {"hr": inspection_hr_total, "rate": rates.get("InspectionRate", 0.0)},
        "saw_waterjet":     {"hr": sawing_hr,       "rate": rates.get("SawWaterjetRate", 0.0)},
        "assembly":         {"hr": assembly_hr,     "rate": rates.get("AssemblyRate", 0.0)},
        "packaging":        {"hr": packaging_hr,    "rate": rates.get("PackagingRate", rates.get("AssemblyRate", 0.0))},
        "ehs_compliance":   {"hr": ehs_hr,          "rate": rates.get("InspectionRate", 0.0)},
    }
    process_meta["drilling"] = {"hr": baseline_drill_hr, "rate": drill_rate}
    process_meta["tapping"] = {"hr": tapping_hr_rounded, "rate": drill_rate}
    process_meta["counterbore"] = {"hr": cbore_hr_rounded, "rate": drill_rate}
    process_meta["countersink"] = {"hr": csk_hr_rounded, "rate": drill_rate}
    process_meta["deburr"] = {"hr": deburr_total_hr, "rate": finishing_rate}
    if deburr_auto_hr:
        process_meta["deburr"]["auto_calc_hr"] = deburr_auto_hr
    if deburr_manual_hr_rounded:
        process_meta["deburr"]["manual_hr"] = deburr_manual_hr_rounded
    if deburr_edge_hr:
        process_meta["deburr"]["edge_hr"] = deburr_edge_hr
    if deburr_holes_hr:
        process_meta["deburr"]["hole_touch_hr"] = deburr_holes_hr
    for key, meta in process_meta.items():
        rate = float(meta.get("rate", 0.0))
        hr = float(meta.get("hr", 0.0))
        meta["base_extra"] = process_costs.get(key, 0.0) - hr * rate

    process_hours_baseline = {k: float(meta.get("hr", 0.0)) for k, meta in process_meta.items()}


    pass_meta = {
        "Material": {"basis": "Stock / raw material"},
        "Fixture Material": {"basis": "Fixture raw stock"},
        "Hardware / BOM": {"basis": "Pass-through hardware / BOM"},
        "Outsourced Vendors": {"basis": "Outside processing vendors"},
        "Shipping": {"basis": "Freight & logistics"},
        "Consumables /Hr": {"basis": "Machine & inspection hours $/hr"},
        "Utilities": {"basis": "Spindle/inspection hours $/hr"},
        "Consumables Flat": {"basis": "Fixed shop supplies"},
    }

    mat_source = material_detail_for_breakdown.get("source")
    if mat_source:
        mat_symbol = material_detail_for_breakdown.get("symbol")
        if mat_symbol:
            pass_meta["Material"]["basis"] = f"{mat_symbol} via {mat_source}"
        else:
            pass_meta["Material"]["basis"] = f"Source: {mat_source}"
    quote_state.material_source = pass_meta.get("Material", {}).get("basis") or mat_source or "shop defaults"

    pass_through = {
        "Material": material_direct_cost,
        "Fixture Material": fixture_material_cost,
        "Hardware / BOM": hardware_cost,
        "Outsourced Vendors": outsourced_costs,
        "Shipping": shipping_cost,
        "Consumables /Hr": consumables_hr_cost,
        "Utilities": utilities_cost,
        "Consumables Flat": consumables_flat,
    }
    pass_through_baseline = {k: float(v) for k, v in pass_through.items()}

    fix_detail = nre_detail.get("fixture", {})
    fixture_plan_desc = None
    try:
        fb = float(fixture_build_hr)
    except Exception:
        fb = 0.0
    try:
        fm = float(fixture_material_cost)
    except Exception:
        fm = 0.0
    if fb or fm:
        pieces: list[str] = []
        if fb:
            pieces.append(f"{fb:.2f} hr build")
        if fm:
            pieces.append(f"${fm:,.2f} material")
        fixture_plan_desc = ", ".join(pieces)
    strategy = fix_detail.get("strategy") if isinstance(fix_detail, dict) else None
    if isinstance(strategy, str) and strategy.strip():
        if fixture_plan_desc:
            fixture_plan_desc = f"{fixture_plan_desc} ({strategy.strip()})"
        else:
            fixture_plan_desc = strategy.strip()

    baseline_data = {
        "process_hours": process_hours_baseline,
        "pass_through": pass_through_baseline,
        "scrap_pct": scrap_pct_baseline,
        "setups": int(setups),
        "contingency_pct": ContingencyPct,
    }
    if fixture_plan_desc:
        baseline_data["fixture"] = fixture_plan_desc
    quote_state.baseline = baseline_data

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
    if isinstance(hole_groups_geo, list):
        hole_groups_clean: list[dict[str, Any]] = []
        for grp in hole_groups_geo:
            if not isinstance(grp, dict):
                continue
            cleaned = {
                "dia_mm": _coerce_float_or_none(grp.get("dia_mm")),
                "depth_mm": _coerce_float_or_none(grp.get("depth_mm")),
                "through": bool(grp.get("through")),
                "count": int(_coerce_float_or_none(grp.get("count")) or 0),
            }
            hole_groups_clean.append(cleaned)
        hole_groups_geo = hole_groups_clean

    dfm_geo = {
        "min_wall_mm": _coerce_float_or_none(geo_context.get("GEO_MinWall_mm")),
        "thin_wall": bool(geo_context.get("GEO_ThinWall_Present")),
        "largest_plane_mm2": _coerce_float_or_none(geo_context.get("GEO_LargestPlane_Area_mm2")),
        "unique_normals": int(_coerce_float_or_none(geo_context.get("GEO_Setup_UniqueNormals")) or 0),
        "face_count": int(_coerce_float_or_none(geo_context.get("Feature_Face_Count")) or 0),
        "deburr_edge_len_mm": _coerce_float_or_none(geo_context.get("GEO_Deburr_EdgeLen_mm")),
    }

    tolerance_inputs: dict[str, Any] = {}
    for key, value in (ui_vars or {}).items():
        label = str(key)
        if re.search(r"(tolerance|finish|surface)", label, re.IGNORECASE):
            tolerance_inputs[label] = value

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
        "material_cost_baseline": material_cost,
        "bbox_mm": bbox_info,
        "machine_limits": machine_limits,
        "stock_catalog": stock_catalog,
        "part_mass_g_est": part_mass_g_est,
        "dfm_geo": dfm_geo,
    }
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
        "material_key": geo_context.get("material") or material_name,
        "drilling_hr_baseline": baseline_drill_hr,
    })
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


    baseline_setups = int(_as_float_or_none(setups) or _as_float_or_none(baseline_data.get("setups")) or 1)
    baseline_fixture = baseline_data.get("fixture")

    llm_baseline_payload = {
        "process_hours": process_hours_baseline,
        "pass_through": pass_through_baseline,
        "scrap_pct": scrap_pct_baseline,
        "setups": baseline_setups,
        "fixture": baseline_fixture,
    }
    llm_bounds = {
        "mult_min": 0.5,
        "mult_max": 3.0,
        "adder_max_hr": 5.0,
        "scrap_min": 0.0,
        "scrap_max": 0.25,
    }

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
    if max_hole_depth_in is not None:
        derived_block["max_hole_depth_in"] = max_hole_depth_in
    if plate_area_in2 is not None:
        derived_block["plate_area_in2"] = plate_area_in2
    if finish_flags_list:
        derived_block["finish_flags"] = finish_flags_list
    if stock_guess:
        derived_block["stock_guess"] = stock_guess

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
                    chat_out = llm_suggest.create_chat_completion(
                        messages=[
                            {"role": "system", "content": SYSTEM_SUGGEST},
                            {"role": "user", "content": json.dumps(payload, indent=2)},
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
                    llm_obj = _LocalLLM(llm_model_path)
                    s_raw, s_text, s_usage = run_llm_suggestions(llm_obj, payload)
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
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                "params": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 512},
                "context_payload": payload,
                "raw_response_text": s_text,
                "parsed_response": s_raw,
                "sanitized": sanitized_struct,
                "usage": s_usage,
            }
            snap_path = APP_ENV.llm_debug_dir / f"llm_snapshot_{int(time.time())}.json"
            snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")

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
    scrap_pct = _ensure_scrap_pct(effective_scrap_val)

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

    applied_process: dict[str, dict[str, Any]] = {}
    applied_pass: dict[str, dict[str, Any]] = {}

    def _normalize_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    process_key_map = {_normalize_key(k): k for k in process_costs.keys()}
    pass_key_map = {_normalize_key(k): k for k in pass_through.keys()}

    def _friendly_process(name: str) -> str:
        return name.replace("_", " ").title()

    def _cost_of(proc_key: str, hours: float) -> float:
        parts = proc_key.replace("_", " ").split()
        rate_key = "".join(part.title() for part in parts) + "Rate"
        return float(hours) * float(rates.get(rate_key, rates.get("MillingRate", 120.0)))

    material_direct_cost_base = material_direct_cost
    fixture_material_cost_base = fixture_material_cost

    old_scrap = _ensure_scrap_pct(features.get("scrap_pct", scrap_pct))
    new_scrap = overrides.get("scrap_pct_override") if overrides else None
    if new_scrap is not None:
        new_scrap = _ensure_scrap_pct(new_scrap)
        if not math.isclose(new_scrap, old_scrap, abs_tol=1e-6):
            baseline = float(features.get("material_cost_baseline", material_direct_cost_base))
            scaled = baseline * ((1.0 + new_scrap) / max(1e-6, (1.0 + old_scrap)))
            scaled = round(scaled, 2)
            pass_through["Material"] = scaled
            entry = applied_pass.setdefault("Material", {"old_value": float(material_direct_cost_base), "notes": []})
            entry["notes"].append(f"scrap to {new_scrap * 100:.1f}%")
            entry["new_value"] = scaled
            features["scrap_pct"] = new_scrap
            scrap_pct = new_scrap
            src_tag = (quote_state.effective_sources.get("scrap_pct") if quote_state and isinstance(quote_state.effective_sources, dict) else None)
            suffix = ""
            if src_tag == "user":
                suffix = " (user override)"
            elif src_tag == "llm":
                suffix = " (LLM)"
            llm_notes.append(f"Scrap {old_scrap * 100:.1f}% → {new_scrap * 100:.1f}%{suffix}")

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
        mult = clamp(mult, 0.5, 3.0, 1.0)
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
        add_hr = clamp(add_hr, 0.0, 5.0, 0.0)
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

    src_pass_map = {}
    if isinstance(quote_state.effective_sources, dict):
        src_pass_map = quote_state.effective_sources.get("add_pass_through") or {}
    add_pass = overrides.get("add_pass_through") if overrides else {}
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
        src_tag = src_pass_map.get(label)
        suffix = ""
        if src_tag == "user":
            suffix = " (user override)"
        elif src_tag == "llm":
            suffix = " (LLM)"
        llm_notes.append(f"{actual_label}: +${float(add_val):,.0f}{suffix}")

    delta_fix_mat = overrides.get("fixture_material_cost_delta") if overrides else None
    if isinstance(delta_fix_mat, (int, float)) and delta_fix_mat:
        delta_fix_mat = clamp(delta_fix_mat, -200.0, 200.0, 0.0)
        if delta_fix_mat:
            old_val = float(pass_through.get("Fixture Material", fixture_material_cost_base))
            new_val = round(old_val + float(delta_fix_mat), 2)
            pass_through["Fixture Material"] = new_val
            pass_key_map[_normalize_key("Fixture Material")] = "Fixture Material"
            entry = applied_pass.setdefault("Fixture Material", {"old_value": old_val, "notes": []})
            entry["notes"].append(f"Δ${float(delta_fix_mat):+.2f}")
            entry["new_value"] = new_val
            fix_detail["mat_cost"] = round(float(fix_detail.get("mat_cost", 0.0)) + float(delta_fix_mat), 2)
            src_tag = None
            if isinstance(quote_state.effective_sources, dict):
                src_tag = quote_state.effective_sources.get("fixture_material_cost_delta")
            suffix = ""
            if src_tag == "user":
                suffix = " (user override)"
            elif src_tag == "llm":
                suffix = " (LLM)"
            llm_notes.append(f"Fixture material ${float(delta_fix_mat):+.0f}{suffix}")

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

    process_hours_final = {k: float(process_meta.get(k, {}).get("hr", 0.0)) for k in process_meta}

    hole_count_for_guard = 0
    try:
        hole_count_for_guard = int(float(geo_context.get("hole_count", 0) or 0))
    except Exception:
        pass
    if hole_count_for_guard <= 0:
        hole_count_for_guard = len(geo_context.get("hole_diams_mm", []) or [])

    if hole_count_for_guard > 0:
        drill_hr_after_overrides = float(process_hours_final.get("drilling", 0.0))
        ok, why = validate_drilling_reasonableness(hole_count_for_guard, drill_hr_after_overrides)
        if not ok:
            floor_hr = _drilling_floor_hours(hole_count_for_guard)
            new_hr = max(drill_hr_after_overrides, floor_hr)
            drill_meta = process_meta.get("drilling")
            if drill_meta:
                rate = float(drill_meta.get("rate", 0.0))
                base_extra = float(drill_meta.get("base_extra", 0.0))
                old_cost = float(process_costs.get("drilling", 0.0))
                entry = applied_process.setdefault(
                    "drilling",
                    {
                        "old_hr": float(drill_meta.get("hr", 0.0)),
                        "old_cost": old_cost,
                        "notes": [],
                    },
                )
                entry.setdefault("notes", []).append(f"Raised to floor {floor_hr:.2f} h")
                drill_meta["hr"] = new_hr
                process_costs["drilling"] = round(new_hr * rate + base_extra, 2)
            process_hours_final["drilling"] = new_hr
            llm_notes.append(f"Raised drilling to floor: {why}")
            process_hours_final = {k: float(process_meta.get(k, {}).get("hr", 0.0)) for k in process_meta}

    baseline_total_hours = sum(float(process_hours_baseline.get(k, 0.0)) for k in process_hours_baseline)
    final_total_hours = sum(float(process_hours_final.get(k, 0.0)) for k in process_hours_final)
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

    validate_quote_before_pricing(geo_context, process_costs, pass_through, process_hours_final)

    for key, entry in applied_process.items():
        entry["new_hr"] = float(process_meta.get(key, {}).get("hr", entry.get("new_hr", entry["old_hr"])))
        entry["new_cost"] = float(process_costs.get(key, entry.get("new_cost", entry["old_cost"])))
        entry["delta_hr"] = entry["new_hr"] - entry["old_hr"]
        entry["delta_cost"] = entry["new_cost"] - entry["old_cost"]

    for label, entry in applied_pass.items():
        entry["new_value"] = float(pass_through.get(label, entry.get("new_value", entry["old_value"])))
        entry["delta_value"] = entry["new_value"] - entry["old_value"]

    material_direct_cost = float(pass_through.get("Material", material_direct_cost_base))
    fixture_material_cost = float(pass_through.get("Fixture Material", fixture_material_cost_base))
    hardware_cost = float(pass_through.get("Hardware / BOM", hardware_cost))
    outsourced_costs = float(pass_through.get("Outsourced Vendors", outsourced_costs))
    shipping_cost = float(pass_through.get("Shipping", shipping_cost))
    consumables_hr_cost = float(pass_through.get("Consumables /Hr", consumables_hr_cost))
    utilities_cost = float(pass_through.get("Utilities", utilities_cost))
    consumables_flat = float(pass_through.get("Consumables Flat", consumables_flat))

    labor_cost = programming_per_part + fixture_labor_per_part + sum(process_costs.values())

    base_direct_costs = sum(pass_through.values())
    vendor_markup = params["VendorMarkupPct"]
    insurance_cost = insurance_pct * (labor_cost + base_direct_costs)
    vendor_marked_add = vendor_markup * (outsourced_costs + shipping_cost)

    pass_meta.setdefault("Insurance", {"basis": "Applied at insurance pct"})
    pass_meta.setdefault("Vendor Markup Added", {"basis": "Vendor + freight markup"})
    pass_through["Insurance"] = round(insurance_cost, 2)
    pass_through["Vendor Markup Added"] = round(vendor_marked_add, 2)

    total_direct_costs = base_direct_costs + insurance_cost + vendor_marked_add
    direct_costs = total_direct_costs

    subtotal = labor_cost + direct_costs

    with_overhead = subtotal * (1.0 + OverheadPct)
    with_ga = with_overhead * (1.0 + GA_Pct)
    with_cont = with_ga * (1.0 + ContingencyPct)
    with_expedite = with_cont * (1.0 + ExpeditePct)

    price_before_margin = with_expedite
    price = price_before_margin * (1.0 + MarginPct)

    min_lot = float(params["MinLotCharge"] or 0.0)
    if price < min_lot:
        price = min_lot

    labor_costs_display: dict[str, float] = {}
    labor_cost_details: dict[str, str] = {}
    for key, value in sorted(process_costs.items(), key=lambda kv: kv[1], reverse=True):
        label = key.replace('_', ' ').title()
        labor_costs_display[label] = value
        meta = process_meta.get(key, {})
        hr = float(meta.get("hr", 0.0))
        rate = float(meta.get("rate", 0.0))
        extra = float(meta.get("base_extra", 0.0))
        detail_bits: list[str] = []
        if hr > 0:
            detail_bits.append(f"{hr:.2f} hr @ ${rate:,.2f}/hr")
        if abs(extra) > 1e-6:
            detail_bits.append(f"includes ${extra:,.2f} extras")
        proc_notes = applied_process.get(key, {}).get("notes")
        if proc_notes:
            detail_bits.append("LLM: " + ", ".join(proc_notes))
        if detail_bits:
            labor_cost_details[label] = "; ".join(detail_bits)

    direct_costs_display: dict[str, float] = {label: float(value) for label, value in pass_through.items()}
    direct_cost_details: dict[str, str] = {}
    for label, value in direct_costs_display.items():
        detail_bits: list[str] = []
        basis = pass_meta.get(label, {}).get("basis")
        if basis:
            detail_bits.append(f"Basis: {basis}")
        if label == "Insurance":
            detail_bits.append(f"Applied {insurance_pct:.1%} of labor + directs")
        elif label == "Vendor Markup Added":
            detail_bits.append(f"Markup {vendor_markup:.1%} on vendors + shipping")
        elif label == "Material":
            source = material_detail_for_breakdown.get("source")
            symbol = material_detail_for_breakdown.get("symbol")
            unit_price_kg = material_detail_for_breakdown.get("unit_price_usd_per_kg")
            unit_price_lb = material_detail_for_breakdown.get("unit_price_usd_per_lb")
            premium = material_detail_for_breakdown.get("vendor_premium_usd_per_kg")
            asof = material_detail_for_breakdown.get("unit_price_asof")
            if source:
                detail_bits.append(f"Source: {source}")
            if symbol:
                detail_bits.append(f"Symbol: {symbol}")
            if unit_price_kg:
                detail_bits.append(f"Unit ${unit_price_kg:,.2f}/kg")
            if unit_price_lb:
                detail_bits.append(f"Unit ${unit_price_lb:,.2f}/lb")
            if premium:
                detail_bits.append(f"Premium ${premium:,.2f}/kg")
            if asof:
                detail_bits.append(f"As of {asof}")
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
        if prog_detail.get("cam_hr"):
            details.append(f"CAM {prog_detail['cam_hr']:.2f} hr @ ${prog_detail.get('cam_rate',0):,.2f}/hr")
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
        if fixture_material_cost:
            details.append(f"Material ${fixture_material_cost:,.2f}")
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
    if applied_multipliers_log:
        llm_cost_log["applied_multipliers"] = applied_multipliers_log
    if applied_adders_log:
        llm_cost_log["applied_adders"] = applied_adders_log
    if isinstance(quote_state.effective_sources, dict) and quote_state.effective_sources:
        llm_cost_log["effective_sources"] = quote_state.effective_sources
    if isinstance(quote_state.user_overrides, dict) and quote_state.user_overrides:
        llm_cost_log["user_overrides"] = quote_state.user_overrides

    breakdown = {
        "qty": Qty,
        "scrap_pct": scrap_pct,
        "fixture_material_cost": fixture_material_cost,
        "material_direct_cost": material_direct_cost,
        "total_direct_costs": round(total_direct_costs, 2),
        "material": material_detail_for_breakdown,
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

    decision_state = {
        "baseline": copy.deepcopy(quote_state.baseline),
        "suggestions": copy.deepcopy(quote_state.suggestions),
        "user_overrides": copy.deepcopy(quote_state.user_overrides),
        "effective": copy.deepcopy(quote_state.effective),
        "effective_sources": copy.deepcopy(quote_state.effective_sources),
    }
    breakdown["decision_state"] = decision_state

    narrative_text = build_narrative(quote_state)
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
                latest.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        except Exception:
            pass

    material_source_final = (
        quote_state.material_source
        or material_detail_for_breakdown.get("source")
        or (pass_meta.get("Material", {}) or {}).get("basis")
        or "shop defaults"
    )

    return {
        "price": price,
        "labor": labor_cost,
        "with_overhead": with_overhead,
        "breakdown": breakdown,
        "narrative": narrative_text,
        "decision_state": decision_state,
        "geo": copy.deepcopy(quote_state.geo),
        "ui_vars": copy.deepcopy(quote_state.ui_vars),
        "material_source": material_source_final,
    }


# ---------- Tracing ----------
def trace_hours_from_df(df):
    import pandas as pd
    out={}
    def grab(regex,label):
        items=df["Item"].astype(str)
        m=items.str.contains(regex, flags=re.I, regex=True, na=False)
        looks_time=items.str.contains(r"(hours?|hrs?|hr|time|min(ute)?s?)\b", flags=re.I, regex=True, na=False)
        mm=m & looks_time
        if mm.any():
            sub=df.loc[mm, ["Item","Example Values / Options"]].copy()
            sub["Example Values / Options"]=pd.to_numeric(sub["Example Values / Options"], errors="coerce").fillna(0.0)
            out[label]=[(str(rates["Item"]), float(rates["Example Values / Options"])) for _,r in sub.iterrows()]
    grab(r"(Fixture\s*Design|Process\s*Sheet|Traveler|Documentation)","Engineering")
    grab(r"(CAM\s*Programming|CAM\s*Sim|Post\s*Processing)","CAM")
    grab(r"(Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)","Assembly")
    return out

# ---------- Explanation ----------
def explain_quote(breakdown: dict[str,Any], hour_trace=None) -> str:
    tot=breakdown.get("totals",{}); pro=breakdown.get("process_costs",{}); nre=breakdown.get("nre",{}); pt=breakdown.get("pass_through",{}); ap=breakdown.get("applied_pcts",{})
    lines=[f"Includes Overhead {ap.get('OverheadPct',0):.0%}, Margin {ap.get('MarginPct',0):.0%}."]
    if nre.get("programming_per_part",0)>0: lines.append(f"NRE spread across lot: ${nre['programming_per_part']:.2f}")
    major={k:v for k,v in pro.items() if v and v>0}
    if major:
        top=sorted(major.items(), key=lambda kv: kv[1], reverse=True)[:4]
        lines.append("Major processes: " + ", ".join(f"{k.replace('_',' ')} ${v:,.0f}" for k,v in top))
    if pt.get("Hardware / BOM",0) or pt.get("Outsourced Vendors",0):
        lines.append(
            f"Pass-throughs: hardware ${pt.get('Hardware / BOM',0):.0f}, "
            f"vendors ${pt.get('Outsourced Vendors',0):.0f}, ship ${pt.get('Shipping',0):.0f}"
        )
    if hour_trace:
        for bucket, rows in hour_trace.items():
            total=sum(v for _,v in rows)
            if total>0:
                items="; ".join(f"{name}={val}" for name,val in rows[:3])
                lines.append(f"{bucket} time from: {items} (total ~{total:g} hr)")
    return "\n".join(lines)

# ---------- Redaction ----------
def redact_text(s: str) -> str:
    out=s
    for pat,repl in _REDACT_PATTERNS: out=pat.sub(repl,out)
    return out
def redact_df(df):
    out=df.copy()
    for c in out.columns:
        if out[c].dtype==object:
            out[c]=out[c].astype(str).map(redact_text)
    return out

# ---------- Process router ----------
def route_process_family(geo: dict[str,Any]) -> str:
    max_dim=float(geo.get("GEO__MaxDim_mm",0) or 0); wedm=float(geo.get("GEO_WEDM_PathLen_mm",0) or 0); free=float(geo.get("GEO_Area_Freeform_mm2",0) or 0); area=float(geo.get("GEO__SurfaceArea_mm2",1) or 1); fr=(free/area) if area>0 else 0.0
    if wedm>200 and max_dim<300: return "wire_edm"
    if fr>0.5: return "die"
    if max_dim>600: return "plate"
    return "cnc"

# ---------- Simple Tk editor ----------
def edit_variables_tk(df):
    import tkinter as tk
    from tkinter import ttk, messagebox
    cols=["Item","Example Values / Options","Data Type / Input Method"]
    for c in cols:
        if c not in df.columns: df[c]=""
    win=tk.Toplevel(); win.title("Edit Variables")
    tree=ttk.Treeview(win, columns=cols, show="headings")
    for c in cols: tree.heading(c,text=c); tree.column(c,width=220,anchor="w")
    for i,row in df[cols].iterrows(): tree.insert("","end",iid=str(i),values=[row[c] for c in cols])
    tree.pack(fill="both",expand=True)
    entry=ttk.Entry(win); entry.pack(fill="x")
    cur={"col":0}
    def on_click(e):
        item=tree.focus()
        if not item: return
        idx=int(tree.identify_column(e.x).replace("#",""))-1
        curates["col"]=idx; vals=list(tree.item(item,"values")); entry.delete(0,"end"); entry.insert(0,vals[idx])
    def on_return(e=None):
        item=tree.focus()
        if not item: return
        idx=curates["col"]; vals=list(tree.item(item,"values")); vals[idx]=entry.get(); tree.item(item,values=vals)
    def on_save():
        for iid in tree.get_children():
            vals=tree.item(iid,"values"); i=int(iid)
            for k,v in zip(cols, vals): df.at[i,k]=v
        messagebox.showinfo("Saved","Variables updated in memory; Save in app to persist.")
    tree.bind("<Button-1>", on_click); entry.bind("<Return>", on_return); ttk.Button(win,text="Save",command=on_save).pack()
    win.grab_set(); win.wait_window(); return df
def read_dxf_as_occ_shape(dxf_path: str):
    # minimal DXF?OCC triangulated shape (3DFACE/MESH/POLYFACE), fallback: extrude closed polyline
    import ezdxf, numpy as np
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid)
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE

    def tri_face(p0,p1,p2):
        poly = BRepBuilderAPI_MakePolygon()
        for p in (p0,p1,p2,p0):
            poly.Add(gp_Pnt(*p))
        return BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()

    def sew(faces):
        sew = BRepBuilderAPI_Sewing(1.0e-6, True, True, True, True)
        for f in faces: sew.Add(f)
        sew.Perform()
        return sew.SewedShape()

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    INSUNITS = doc.header.get("$INSUNITS", 1)  # 1=in, 4=mm, 2=ft, 6=m
    u2mm = {1:25.4, 4:1.0, 2:304.8, 6:1000.0}.get(INSUNITS, 1.0)

    tris = []
    # 3DFACE
    for e in msp.query("3DFACE"):
        verts = [e.dxf.vtx0, e.dxf.vtx1, e.dxf.vtx2, e.dxf.vtx3]
        pts = [(vx * u2mm, vy * u2mm, vz * u2mm) for (vx, vy, vz) in verts]
        if len(pts) < 3:
            continue
        tris.append((pts[0], pts[1], pts[2]))
        if len(pts) == 4:
            v3 = np.array(pts[2])
            v4 = np.array(pts[3])
            if np.linalg.norm(v3 - v4) > 1e-12:
                tris.append((pts[0], pts[2], pts[3]))
    # POLYFACE
    for e in msp.query("POLYFACE"):
        for f in e.faces():
            pts=[(v.dxf.location.x*u2mm, v.dxf.location.y*u2mm, v.dxf.location.z*u2mm) for v in f.vertices()]
            if len(pts)>=3:
                tris.append((pts[0], pts[1], pts[2]))
                for k in range(3,len(pts)):
                    tris.append((pts[0], pts[k-1], pts[k]))
    # MESH
    for e in msp.query("MESH"):
        for f in e.faces_as_vertices():
            pts=[(v.x*u2mm, v.y*u2mm, v.z*u2mm) for v in f]
            if len(pts)>=3:
                tris.append((pts[0], pts[1], pts[2]))
                for k in range(3,len(pts)):
                    tris.append((pts[0], pts[k-1], pts[k]))

    faces = [tri_face(*t) for t in tris]
    shape = None
    if faces:
        sewed = sew(faces)
        try:
            solid = BRepBuilderAPI_MakeSolid()
            exp = TopExp_Explorer(sewed, TopAbs_FACE)
            while exp.More():
                solid.Add(exp.Current()); exp.Next()
            fix = ShapeFix_Solid(solid.Solid()); fix.Perform()
            shape = fix.Solid()
        except Exception:
            shape = sewed

    # Fallback: 2D closed polyline ? extrude small thickness
    if (shape is None) or shape.IsNull():
        for pl in msp.query("LWPOLYLINE"):
            if pl.closed:
                pts2d = [(x*u2mm, y*u2mm, 0.0) for x,y,_ in pl.get_points("xyb")]
                poly = BRepBuilderAPI_MakePolygon()
                for q in pts2d: poly.Add(gp_Pnt(*q))
                poly.Close()
                face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
                thk_mm = float(os.environ.get("DXF_EXTRUDE_THK_MM", "5.0"))
                shape = BRepPrimAPI_MakePrism(face, gp_Vec(0,0,thk_mm)).Shape()
                break

    if (shape is None) or shape.IsNull():
        raise RuntimeError("DXF contained no 3D geometry I can use. Prefer STEP/SAT if possible.")
    return shape

# ---- 2D: PDF (PyMuPDF) -------------------------------------------------------
try:
    import fitz  # old import name
    _HAS_PYMUPDF = True
except Exception:
    try:
        import pymupdf as fitz  # new import name
        _HAS_PYMUPDF = True
    except Exception:
        fitz = None  # allow the rest of the app to import

def extract_2d_features_from_pdf_vector(pdf_path: str) -> dict:
    if not _HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) not installed. pip install pymupdf")

    import math
    page = fitz.open(pdf_path)[0]
    text = page.get_text("text").lower()
    drawings = page.get_drawings()

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
        ("In-Process Inspection Hours", 0.0, "number"),
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
        ("FAIR Required", 0, "number"),
        ("Source Inspection Requirement", 0, "number"),
        ("Quantity", 1, "number"),
        ("Material", "", "text"),
        ("Thickness (in)", 0.0, "number"),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLS)

def coerce_or_make_vars_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not set(REQUIRED_COLS).issubset(df.columns):
        return default_variables_template().copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
    return df

def extract_pdf_all(pdf_path: Path, dpi: int = 300) -> dict:
    if not _HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) not installed. pip install pymupdf")
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for idx, page in enumerate(doc):
        text = page.get_text("text") or ""
        blocks = page.get_text("blocks")
        tables = []
        try:
            found = page.find_tables()
            for table in getattr(found, "tables", []):
                tables.append(table.extract())
        except Exception:
            pass
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
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
    best = max(pages, key=lambda p: (params["table_count"] > 0, params["char_count"])) if pages else {"text": "", "image_path": ""}
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
    schema = json.dumps(JSON_SCHEMA, indent=2)
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
    for sp in _spaces(doc) or []:
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
            pts = [(float(x) * to_in, float(y) * to_in) for x, y, *_ in pl.get_points()]
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
    for sp in _spaces(doc) or []:
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


def filtered_circles(doc, to_in: float, plate_bbox: tuple[float, float, float, float] | None = None) -> list[tuple[float, float, float, int]]:
    circles: list[tuple[float, float, float]] = []
    for sp in _spaces(doc) or []:
        try:
            entities = sp.query("CIRCLE")
        except Exception:
            entities = []
        for c in entities:
            layer = (getattr(getattr(c, "dxf", object()), "layer", "") or "").upper()
            if layer in SKIP_LAYERS:
                continue
            try:
                d = round(2.0 * float(c.dxf.radius) * float(to_in), 4)
            except Exception:
                continue
            if not (0.06 <= d <= 2.5):
                continue
            try:
                x, y = c.dxf.center[:2]
            except Exception:
                continue
            if plate_bbox:
                xmin, ymin, xmax, ymax = plate_bbox
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    continue
            circles.append((x, y, d))

    clustered: list[tuple[float, float, float, int]] = []
    tol = 0.01
    for x, y, d in circles:
        for idx, (cx, cy, cd, count) in enumerate(clustered):
            if abs(cx - x) < tol and abs(cy - y) < tol and abs(cd - d) < 1e-3:
                clustered[idx] = (cx, cy, cd, count + 1)
                break
        else:
            clustered.append((x, y, d, 1))
    return clustered


def classify_concentric(doc, to_in: float, tol_center: float = 0.005, min_gap_in: float = 0.03) -> dict[str, Any]:
    pts: list[tuple[float, float, float]] = []
    for sp in _spaces(doc) or []:
        try:
            entities = sp.query("CIRCLE")
        except Exception:
            entities = []
        for c in entities:
            layer = (getattr(getattr(c, "dxf", object()), "layer", "") or "").upper()
            if layer in SKIP_LAYERS:
                continue
            try:
                r = float(c.dxf.radius) * float(to_in)
            except Exception:
                continue
            dia = 2.0 * r
            if not (0.04 <= dia <= 6.0):
                continue
            try:
                x, y = c.dxf.center[:2]
            except Exception:
                continue
            pts.append((x, y, r))
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
    for sp in _spaces(doc) or []:
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


def _spaces(doc):
    if doc is None:
        return
    seen: set[int] = set()
    try:
        msp = doc.modelspace()
    except Exception:
        msp = None
    if msp is not None and id(msp) not in seen:
        seen.add(id(msp))
        yield msp
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
        yield es


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

    for sp in _spaces(doc) or []:
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
            if side == "BACK" or (entry.get("raw") and "BACK" in str(entry.get("raw")).upper()):
                back_ops = True
            elif entry.get("double_sided") and (entry.get("cbore") or entry.get("csk")):
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
                "add_pass_through.Hardware / BOM",
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
    for sp in _spaces(doc) or []:
        try:
            polylines = list(sp.query("LWPOLYLINE"))
        except Exception:
            polylines = []
        for pl in polylines:
            try:
                pts = pl.get_points("xy")
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
        holes.extend(leader_entries)

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


def extract_2d_features_from_dxf_or_dwg(
    path: str,
    *,
    build_geo_from_dxf: Optional[Callable[[str], dict]] = None,
) -> dict:
    if not _HAS_EZDXF:
        raise RuntimeError("ezdxf not installed. pip/conda install ezdxf")

    # --- load doc ---
    dxf_text_path: str | None = None
    doc = None
    lower_path = path.lower()
    if lower_path.endswith(".dwg"):
        if _HAS_ODAFC:
            # uses ODAFileConverter through ezdxf, no env var needed
            doc = odafc.readfile(path)
        else:
            dxf_path = convert_dwg_to_dxf(path, out_ver="ACAD2018")  # needs ODA_CONVERTER_EXE or DWG2DXF_EXE
            dxf_text_path = dxf_path
            doc = ezdxf.readfile(dxf_path)
    else:
        doc = ezdxf.readfile(path)
        dxf_text_path = path

    sp = doc.modelspace()
    units = detect_units_scale(doc)
    to_in = float(units.get("to_in", 1.0) or 1.0)
    u2mm = to_in * 25.4

    geo = _build_geo_from_ezdxf_doc(doc)

    geo_read_more: dict[str, Any] | None = None
    extra_geo_loader = build_geo_from_dxf or build_geo_from_dxf_path
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
                prov = geo.get("provenance") if isinstance(geo.get("provenance"), dict) else {}
                merged = dict(prov)
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
        pts = e.get_points("xyb")
        for i in range(len(pts)):
            x1,y1,_ = pts[i]; x2,y2,_ = pts[(i+1) % len(pts)]
            per += math.hypot(x2-x1, y2-y1)
    for e in sp.query("POLYLINE"):
        vs = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        for i in range(len(vs)-1):
            x1,y1 = vs[i]; x2,y2 = vs[i+1]
            per += math.hypot(x2-x1, y2-y1)
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

    if extract_text_lines_from_dxf and dxf_text_path:
        try:
            chart_lines = extract_text_lines_from_dxf(dxf_text_path)
        except Exception:
            chart_lines = []
    if not chart_lines:
        chart_lines = _extract_text_lines_from_ezdxf_doc(doc)

    if chart_lines:
        chart_summary = summarize_hole_chart_lines(chart_lines)
    if chart_lines and parse_hole_table_lines:
        try:
            hole_rows = parse_hole_table_lines(chart_lines)
        except Exception:
            hole_rows = []
        if hole_rows:
            chart_ops = hole_rows_to_ops(hole_rows)
            chart_source = "dxf_text_regex"
            chart_reconcile = reconcile_holes(entity_holes_mm, chart_ops)
    if chart_summary:
        geo.setdefault("chart_summary", chart_summary)
        if chart_summary.get("tap_qty"):
            geo["tap_qty"] = max(int(geo.get("tap_qty", 0) or 0), int(chart_summary.get("tap_qty") or 0))
        if chart_summary.get("cbore_qty"):
            geo["cbore_qty"] = max(int(geo.get("cbore_qty", 0) or 0), int(chart_summary.get("cbore_qty") or 0))
        if chart_summary.get("csk_qty"):
            geo["csk_qty"] = max(int(geo.get("csk_qty", 0) or 0), int(chart_summary.get("csk_qty") or 0))
        deepest_chart = _coerce_float_or_none(chart_summary.get("deepest_hole_in"))
        if deepest_chart and (not geo.get("thickness_in_guess") or deepest_chart > float(geo.get("thickness_in_guess") or 0.0)):
            geo["thickness_in_guess"] = deepest_chart
            prov = geo.get("provenance") or {}
            if isinstance(prov, dict):
                prov["thickness"] = "HOLE TABLE depth max"
                geo["provenance"] = prov
        if chart_summary.get("from_back"):
            geo["from_back"] = True
            geo["needs_back_face"] = True
            notes = geo.get("notes")
            if isinstance(notes, list) and not any("back" in str(n).lower() for n in notes):
                notes.append("Hole chart references BACK operations.")

    # scrape text for thickness/material
    txt = " ".join([t.dxf.text for t in sp.query("TEXT")] +
                   [m.plain_text() for m in sp.query("MTEXT")]).lower()
    import re
    thickness_mm = None
    m = re.search(r"(thk|thickness)\s*[:=]?\s*([0-9.]+)\s*(mm|in|in\.|\")", txt)
    if m:
        thickness_mm = float(m.group(2)) * (25.4 if (m.group(3).startswith("in") or m.group(3)=='"') else 1.0)
    material = None
    mm = re.search(r"(matl|material)\s*[:=]?\s*([a-z0-9 \-\+]+)", txt)
    if mm:
        material = mm.group(2).strip()

    if not thickness_mm and geo.get("thickness_in_guess"):
        thickness_mm = float(geo["thickness_in_guess"]) * 25.4
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
            try:
                edge_len_in = float(geo_read_more.get("edge_len_in"))
            except Exception:
                edge_len_in = None
            if edge_len_in and edge_len_in > 0:
                result["edge_length_in"] = edge_len_in
                result["edge_len_in"] = edge_len_in
                result["profile_length_mm"] = round(edge_len_in * 25.4, 2)
            try:
                outline_area_in2 = float(geo_read_more.get("outline_area_in2"))
            except Exception:
                outline_area_in2 = None
            if outline_area_in2 and outline_area_in2 > 0:
                result["outline_area_in2"] = outline_area_in2
                result["outline_area_mm2"] = round(outline_area_in2 * (25.4 ** 2), 2)
            if geo_read_more.get("hole_diam_families_in"):
                result["hole_diam_families_in"] = dict(geo_read_more.get("hole_diam_families_in") or {})
            if geo_read_more.get("hole_table_families_in"):
                result["hole_table_families_in"] = dict(geo_read_more.get("hole_table_families_in") or {})
            try:
                hole_count_geom = int(float(geo_read_more.get("hole_count_geom")))
            except Exception:
                hole_count_geom = None
            if hole_count_geom and hole_count_geom > int(result.get("hole_count", 0) or 0):
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
                feature_counts = result.get("feature_counts") if isinstance(result.get("feature_counts"), dict) else {}
                feature_counts = dict(feature_counts)
                if geo_read_more.get("tap_qty"):
                    feature_counts["tap_qty"] = max(int(feature_counts.get("tap_qty", 0) or 0), int(geo_read_more.get("tap_qty") or 0))
                if geo_read_more.get("cbore_qty"):
                    feature_counts["cbore_qty"] = max(int(feature_counts.get("cbore_qty", 0) or 0), int(geo_read_more.get("cbore_qty") or 0))
                if geo_read_more.get("csk_qty"):
                    feature_counts["csk_qty"] = max(int(feature_counts.get("csk_qty", 0) or 0), int(geo_read_more.get("csk_qty") or 0))
                if feature_counts:
                    result["feature_counts"] = feature_counts
            for qty_key in ("tap_qty", "cbore_qty", "csk_qty"):
                try:
                    candidate = int(float(geo_read_more.get(qty_key, 0) or 0))
                except Exception:
                    candidate = 0
                if candidate:
                    current = int(float(result.get(qty_key, 0) or 0)) if result.get(qty_key) else 0
                    result[qty_key] = max(current, candidate)
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
            try:
                text = entity.plain_text() if hasattr(entity, "plain_text") else entity.dxf.text
            except Exception:
                text = ""
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
                        try:
                            text = sub.plain_text() if hasattr(sub, "plain_text") else sub.dxf.text
                        except Exception:
                            text = ""
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
            try:
                dia_val = float(op.get("dia_mm"))
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


def _to_noncapturing(expr: str) -> str:
    """
    Convert every capturing '(' to non-capturing '(?:', preserving
    escaped parens and existing '(?...)' constructs.
    """
    out: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        prev = expr[i - 1] if i > 0 else ''
        nxt = expr[i + 1] if i + 1 < len(expr) else ''
        if ch == '(' and prev != '\\' and nxt != '?':
            out.append('(?:')
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)

def apply_2d_features_to_variables(df, g2d: dict, *, params: dict, rates: dict):
    """Write a few cycle-time rows based on 2D perimeter/holes so compute_quote_from_df() can price it."""



    geo = g2d.get("geo") if isinstance(g2d, dict) else None
    thickness_mm = _coerce_float_or_none(g2d.get("thickness_mm")) if isinstance(g2d, dict) else None
    thickness_in = (float(thickness_mm) / 25.4) if thickness_mm else None
    thickness_from_deepest = False
    if (not thickness_in) and isinstance(geo, dict):
        guess = _coerce_float_or_none(geo.get("thickness_in_guess"))
        if guess:
            try:
                bounded = min(3.0, max(0.125, float(guess)))
                thickness_in = bounded
                thickness_from_deepest = True
            except Exception:
                thickness_in = None

    material_note = ""
    if isinstance(geo, dict) and geo.get("material_note"):
        material_note = str(geo.get("material_note") or "").strip()
    material_value = material_note or (g2d.get("material") if isinstance(g2d, dict) else "") or "Steel"
    df = upsert_var_row(df, "Material", material_value, dtype="text")
    if thickness_in:
        df = upsert_var_row(
            df,
            "Thickness (in)",
            round(float(thickness_in), 4),
            dtype="number",
        )
    elif thickness_from_deepest:
        df = upsert_var_row(df, "Thickness (in)", 0.125, dtype="number")

    plate_len = None
    plate_wid = None
    if isinstance(geo, dict):
        plate_len = geo.get("plate_len_in")
        plate_wid = geo.get("plate_wid_in")
    df = upsert_var_row(
        df,
        "Plate Length (in)",
        round(float(plate_len), 3) if plate_len else 12.0,
        dtype="number",
    )
    df = upsert_var_row(
        df,
        "Plate Width (in)",
        round(float(plate_wid), 3) if plate_wid else 14.0,
        dtype="number",
    )
    df = upsert_var_row(df, "Scrap Percent (%)", 15.0, dtype="number")

    def _update_if_blank(label: str, value: Any, dtype: str = "number", zero_is_blank: bool = True) -> None:
        if value is None:
            return
        mask = df["Item"].astype(str).str.fullmatch(label, case=False, na=False)
        if mask.any():
            existing_raw = str(df.loc[mask, "Example Values / Options"].iloc[0]).strip()
            existing_val = _coerce_float_or_none(existing_raw)
            is_blank = (existing_raw == "") or (zero_is_blank and (existing_val is None or abs(existing_val) < 1e-9))
            if not is_blank:
                return
            df.loc[mask, "Example Values / Options"] = value
            df.loc[mask, "Data Type / Input Method"] = dtype
        else:
            df.loc[len(df)] = [label, value, dtype]

    tap_qty_geo = None
    from_back_geo = False
    if isinstance(geo, dict):
        tap_qty_geo = _coerce_float_or_none(geo.get("tap_qty"))
        from_back_geo = bool(geo.get("needs_back_face") or geo.get("from_back"))
    if tap_qty_geo and tap_qty_geo > 0:
        _update_if_blank("Tap Qty (LLM/GEO)", int(round(tap_qty_geo)))
    if from_back_geo:
        mask_setups = df["Item"].astype(str).str.fullmatch("Number of Milling Setups", case=False, na=False)
        if mask_setups.any():
            current = _coerce_float_or_none(df.loc[mask_setups, "Example Values / Options"].iloc[0])
            if current is None or current < 2:
                df.loc[mask_setups, "Example Values / Options"] = 2
                df.loc[mask_setups, "Data Type / Input Method"] = "number"
        else:
            df.loc[len(df)] = ["Number of Milling Setups", 2, "number"]

    def set_row(pattern: str, value: float):
        # Use the module-level helper (fixes NameError in DWG import path).
        regex = _to_noncapturing(pattern)
        mask = df["Item"].astype(str).str.contains(regex, case=False, regex=True, na=False)
        if mask.any():
            df.loc[mask, "Example Values / Options"] = value
        else:
            df.loc[len(df)] = [pattern, value, "number"]
    L = float(g2d.get("profile_length_mm", 0.0))
    t = float(g2d.get("thickness_mm") or 6.0)
    holes = g2d.get("hole_diams_mm", [])
    # crude process pick
    use_jet = (t <= 12.0 and (L >= 300.0 or len(holes) >= 2))

    if use_jet:
        cut_min = L / (300.0 if t <= 10 else 120.0)   # mm/min ? minutes
        deburr_min = (L/1000.0) * 2.0
        set_row(r"(Sawing|Waterjet|Blank\s*Prep)", round(cut_min/60.0, 3))
        set_row(r"(Deburr|Edge\s*Break)", round(deburr_min/60.0, 3))
        set_row(r"(Programming|2D\s*CAM)", 0.5)
        set_row(r"(Setup\s*Time\s*per\s*Setup)", 0.25)
        set_row(r"(Milling\s*Setups)", 1)
    else:
        mill_min = L / 800.0
        drill_min = max(0.2, (t/50.0)) * max(1, len(holes))
        deburr_min = (L/1000.0) * 3.0
        set_row(r"(Finishing\s*Cycle\s*Time)", round(mill_min/60.0, 3))
        set_row(r"(ID\s*Boring|Drilling|Reaming)", round(drill_min/60.0, 3))
        set_row(r"(Deburr|Edge\s*Break)", round(deburr_min/60.0, 3))
        set_row(r"(Programming|2D\s*CAM)", 0.75)
        set_row(r"(Setup\s*Time\s*per\s*Setup)", 0.5)
        set_row(r"(Milling\s*Setups)", 1)

    set_row(r"(Final\s*Inspection|Manual\s*Inspection)", 0.2)
    set_row(r"(Packaging|Boxing|Crating\s*Labor)", 0.1)
    return df.reset_index(drop=True)

def get_llm_quote_explanation(result: dict, model_path: str) -> str:
    """
    Returns a single-paragraph, friendly explanation of the main cost drivers.
    Works with the local _LocalLLM.ask_json(), and has a safe fallback if no model.
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
    effective_scrap = _ensure_scrap_pct(effective_scrap)
    scrap_pct_percent = round(100.0 * effective_scrap, 1)

    ctx = {
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
        drivers: list[dict[str, float]] = []
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

        top_processes_raw = data.get("top_processes") if isinstance(data.get("top_processes"), list) else []
        top_processes: list[dict[str, float]] = []
        for entry in top_processes_raw:
            if not isinstance(entry, dict):
                continue
            name_val = str(entry.get("name") or "").strip()
            usd_val = _to_float(entry.get("usd"))
            if name_val and usd_val is not None:
                top_processes.append({"name": name_val, "usd": usd_val})
        if not top_processes:
            top_processes = []
            for proc in ctx["rollup"].get("top_processes", []):
                name_val = str(proc.get("name") or "").strip()
                usd_val = _to_float(proc.get("usd"))
                if name_val and usd_val is not None:
                    top_processes.append({"name": name_val, "usd": usd_val})

        material_section = data.get("material") if isinstance(data.get("material"), dict) else {}
        scrap_val = _to_float(material_section.get("scrap_pct"))
        if scrap_val is None:
            scrap_val = float(ctx["scrap_pct"])
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

            scrap_display = material_struct.get("scrap_pct", ctx["scrap_pct"])
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
                f"Labor ${labor_cost:.2f} ({drivers[0]['pct_of_subtotal']:.1f}%) and directs "
                f"${direct_costs:.2f} ({drivers[1]['pct_of_subtotal']:.1f}%) drive cost. "
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

    user_prompt = "```json\n" + json.dumps(ctx, indent=2, sort_keys=True) + "\n```"

    try:
        llm = _LocalLLM(model_path)
        parsed, raw_text, _usage = llm.ask_json(
            system_prompt,
            user_prompt,
            temperature=0.4,
            max_tokens=256,
            context=ctx,
        )
        return _render_explanation(parsed, raw_text)
    except Exception:
        return _render_explanation()


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
        "fixture_material_cost_delta": 0.0,       # dollars (+/-)
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
        llm = _LocalLLM(model_path)
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
            "mult_min": 0.5,
            "mult_max": 3.0,
            "add_hr_min": 0.0,
            "add_hr_max": 5.0,
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
                prompt_body = json.dumps(payload, indent=2)
            except TypeError:
                prompt_body = json.dumps(_jsonify(payload), indent=2)
            prompt = "```json\n" + prompt_body + "\n```"
            parsed, raw_text, usage = llm.ask_json(
                system_prompt,
                prompt,
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
            " Respect multipliers in [0.5,3.0] and adders in [0,5] hours."
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
            " \"notes\":[""...""]}. Keep scrap_pct within [0,0.25] and hour adders within [0,5]."
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
        " or {} if no issues. Keep multipliers within [0.5,3.0] and adders within [0,5]."
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
            " \"notes\":[...]}}. Keep hour adders within [0,5] and return {} if no change is needed."
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
        clamped = clamp(val, 0.5, 3.0, 1.0)
        container = _ensure_mults_dict()
        norm = str(name).lower()
        prev = container.get(norm)
        if prev is None:
            container[norm] = clamped
            return
        new_val = clamp(prev * clamped, 0.5, 3.0, 1.0)
        if not math.isclose(prev * clamped, new_val, abs_tol=1e-6):
            clamp_notes.append(f"{source} multiplier clipped for {norm}")
        container[norm] = new_val

    def _merge_adder(name: str, value, source: str) -> None:
        val = _as_float(value)
        if val is None:
            return
        clamped = clamp(val, 0.0, 5.0, 0.0)
        if clamped <= 0:
            return
        container = _ensure_adders_dict()
        norm = str(name).lower()
        prev = float(container.get(norm, 0.0))
        new_val = clamp(prev + clamped, 0.0, 5.0, 0.0)
        if not math.isclose(prev + clamped, new_val, abs_tol=1e-6):
            clamp_notes.append(f"{source} {prev + clamped:.2f} hr clipped to 5.0 for {norm}")
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
            clamped_val = clamp(v, 0.5, 3.0, 1.0)
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
            clamped_val = clamp(v, 0.0, 5.0, 0.0)
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
            clean_pass[str(k)] = clamped_val
            if not math.isclose(orig, clamped_val, abs_tol=1e-6):
                clamp_notes.append(
                    f"add_pass_through[{k}] {orig:.2f} → {clamped_val:.2f}"
                )
        else:
            clamp_notes.append(f"add_pass_through[{k}] non-numeric")
    if clean_pass:
        out["add_pass_through"] = clean_pass

    fmd = parsed.get("fixture_material_cost_delta", None)
    if isinstance(fmd, (int, float)):
        orig = float(fmd)
        clamped_val = clamp(fmd, -200.0, 200.0, 0.0)
        out["fixture_material_cost_delta"] = clamped_val
        if not math.isclose(orig, clamped_val, abs_tol=1e-6):
            clamp_notes.append(
                f"fixture_material_cost_delta {orig:.2f} → {clamped_val:.2f}"
            )

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
                saw_hr_clamped = clamp(saw_hr, 0.0, 5.0, 0.0)
                clean_stock_plan["sawing_hr"] = saw_hr_clamped
                _merge_adder("saw_waterjet", saw_hr_clamped, "stock_plan.sawing_hr")
            handling_hr = _as_float(stock_plan_raw.get("handling_hr"))
            if handling_hr and handling_hr > 0:
                handling_hr_clamped = clamp(handling_hr, 0.0, 5.0, 0.0)
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
            clean_setup["setup_adders_hr"] = clamp(setup_hr, 0.0, 5.0, 0.0)
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
            inproc_clamped = clamp(inproc_hr, 0.0, 5.0, 0.0)
            clean_tol["in_process_inspection_hr"] = inproc_clamped
            _merge_adder("inspection", inproc_clamped, "tolerance_in_process_hr")
        final_hr = _as_float(tol_raw.get("final_inspection_hr") or tol_raw.get("final_hr"))
        if final_hr and final_hr > 0:
            final_clamped = clamp(final_hr, 0.0, 5.0, 0.0)
            clean_tol["final_inspection_hr"] = final_clamped
            _merge_adder("inspection", final_clamped, "tolerance_final_hr")
        finish_hr = _as_float(tol_raw.get("finishing_hr") or tol_raw.get("finish_hr"))
        if finish_hr and finish_hr > 0:
            finish_clamped = clamp(finish_hr, 0.0, 5.0, 0.0)
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
    return out, meta
# ----------------- GUI -----------------
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
        configuration: QuoteConfiguration | None = None,
        geometry_loader: GeometryLoader | None = None,
        pricing_registry: PricingRegistry | None = None,
        llm_client: LLMClient | None = None,
    ):
        super().__init__()
        self.configuration = configuration or QuoteConfiguration()
        self.geometry_loader = geometry_loader or GeometryLoader(build_geo_from_dxf=build_geo_from_dxf_path)
        self.pricing_registry = pricing_registry or PricingRegistry()
        self.llm_client = llm_client or LLMClient(
            primary_loader=lambda: load_qwen_vl(n_ctx=8192, n_gpu_layers=20),
            fallback_loader=lambda: load_qwen_vl(n_ctx=4096, n_gpu_layers=0),
        )
        self.title("Compos-AI")
        self.geometry("1260x900")

        self.vars_df = None
        self.vars_df_full = None
        self.geo = None
        self.geo_context: dict[str, Any] = {}
        self.params = self.configuration.copy_default_params()
        self.rates = self.pricing_registry.copy_default_rates()
        self.quote_state = QuoteState()
        self.settings_path = Path(__file__).with_name("app_settings.json")
        self.settings = self._load_settings()
        if isinstance(self.settings, dict):
            vendor_csv = str(self.settings.get("material_vendor_csv", "") or "")
            if vendor_csv:
                self.params["MaterialVendorCSVPath"] = vendor_csv

        # LLM defaults: ON + auto model discovery
        default_model = find_default_qwen_model()
        if default_model:
            os.environ["QWEN_GGUF_PATH"] = default_model
            self.params["LLMModelPath"] = default_model

        self.llm_enabled = tk.BooleanVar(value=True)
        self.apply_llm_adj = tk.BooleanVar(value=True)
        self.llm_model_path = tk.StringVar(value=default_model)

        # Create a Menu Bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Overrides...", command=self.load_overrides)
        file_menu.add_command(label="Save Overrides...", command=self.save_overrides)
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
        self.editor_vars: dict[str, tk.StringVar] = {}
        self.editor_label_widgets: dict[str, ttk.Label] = {}
        self.editor_label_base: dict[str, str] = {}
        self.editor_value_sources: dict[str, str] = {}
        self._editor_set_depth = 0
        self._building_editor = False
        self._reprice_in_progress = False
        self.effective_process_hours: dict[str, float] = {}
        self.effective_scrap: float = 0.0
        self.effective_setups: int = 1
        self.effective_fixture: str = "standard"

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5)
        status_bar.pack(side="bottom", fill="x")

        # GEO (single pane; CAD open handled by top bar)
        self.geo_txt = tk.Text(self.tab_geo, wrap="word"); self.geo_txt.pack(fill="both", expand=True)

        # LLM (hidden frame is still built to keep functionality without a visible tab)
        if hasattr(self, "_build_llm"):
            try:
                self._build_llm(self.tab_llm)
            except Exception:
                pass

        # Output
        self.out_txt = tk.Text(self.tab_out, wrap="word")
        self.out_txt.pack(fill="both", expand=True)
        try:
            self.out_txt.tag_configure("rcol", tabs=("4.8i right",), tabstyle="tabular")
        except tk.TclError:
            self.out_txt.tag_configure("rcol", tabs=("4.8i right",))
        #self.out_txt = tk.Text(self.tab_out, wrap="word", font=("Consolas", 10)); self.out_txt.pack(fill="both", expand=True)

        self.LLM_SUGGEST = None
        if self.llm_client.primary_loader is not None:
            try:
                self.LLM_SUGGEST = self.llm_client.load_primary()
            except Exception as exc:
                self.status_var.set(f"Vision LLM failed ({exc}); retrying CPU mode.")
                try:
                    self.LLM_SUGGEST = self.llm_client.load_fallback()
                except Exception as exc2:
                    self.status_var.set(f"Vision LLM unavailable: {exc2}")
                    self.LLM_SUGGEST = None

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
        self.status_var.set(f"Material vendor CSV set to {path}")

    def clear_material_vendor_csv(self) -> None:
        if not isinstance(self.settings, dict):
            self.settings = {}
        self.settings["material_vendor_csv"] = ""
        self.params["MaterialVendorCSVPath"] = ""
        self._save_settings()
        self.status_var.set("Material vendor CSV cleared.")

    def _populate_editor_tab(self, df: pd.DataFrame) -> None:
        df = coerce_or_make_vars_df(df)
        """Rebuild the Quote Editor tab using the latest variables dataframe."""
        def _ensure_row(dataframe: pd.DataFrame, item: str, value: Any, dtype: str = "number") -> pd.DataFrame:
            mask = dataframe["Item"].astype(str).str.fullmatch(item, case=False)
            if mask.any():
                return dataframe
            return upsert_var_row(dataframe, item, value, dtype=dtype)

        df = _ensure_row(df, "Scrap Percent (%)", 15.0, dtype="number")
        df = _ensure_row(df, "Plate Length (in)", 12.0, dtype="number")
        df = _ensure_row(df, "Plate Width (in)", 14.0, dtype="number")
        df = _ensure_row(df, "Thickness (in)", 0.25, dtype="number")
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

        def normalize_item(value: str) -> str:
            cleaned = re.sub(r"[^0-9a-z&$]+", " ", str(value).strip().lower())
            return re.sub(r"\s+", " ", cleaned).strip()

        items_series = df["Item"].astype(str)
        normalized_items = items_series.apply(normalize_item)
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
        skip_items = {normalize_item(item) for item in raw_skip_items}


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
            if material_choice_var is None or material_price_var is None:
                return
            choice = material_choice_var.get().strip()
            if not choice:
                return
            norm_choice = _normalize_lookup_key(choice)
            if norm_choice == MATERIAL_OTHER_KEY:
                return
            price = material_lookup.get(norm_choice)
            if price is None:
                try:
                    price_per_kg, _src = resolve_material_unit_price(choice, unit="kg")
                except Exception:
                    return
                if not price_per_kg:
                    return
                price = float(price_per_kg) / 1000.0
            current_val = _coerce_float_or_none(material_price_var.get())
            if current_val is not None and abs(current_val - price) < 1e-6:
                return
            material_price_var.set(f"{price:.4f}")

        for _, row_data in df.iterrows():
            item_name = str(row_data["Item"])
            normalized_name = normalize_item(item_name)
            if normalized_name in skip_items:
                continue
            label_widget = ttk.Label(quote_frame, text=item_name, wraplength=400)
            label_widget.grid(row=row_index, column=0, sticky="w", padx=5, pady=2)
            initial_raw = row_data["Example Values / Options"]
            initial_value = str(initial_raw) if initial_raw is not None else ""
            if normalized_name in {"material"}:
                var = tk.StringVar(value=self.configuration.default_material_display)
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
                    quote_frame,
                    textvariable=var,
                    values=MATERIAL_DROPDOWN_OPTIONS,
                    width=32,
                )
                combo.grid(row=row_index, column=1, sticky="w", padx=5, pady=2)
                combo.bind("<<ComboboxSelected>>", update_material_price)
                var.trace_add("write", update_material_price)
                material_choice_var = var
                self.var_material = var
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            elif re.search(r"(Material\s*Price.*(per\s*gram|per\s*g|/g)|Unit\s*Price\s*/\s*g)", item_name, flags=re.IGNORECASE):
                var = tk.StringVar(value=initial_value)
                ttk.Entry(quote_frame, textvariable=var, width=30).grid(row=row_index, column=1, sticky="w", padx=5, pady=2)
                material_price_var = var
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            else:
                var = tk.StringVar(value=initial_value)
                ttk.Entry(quote_frame, textvariable=var, width=30).grid(row=row_index, column=1, sticky="w", padx=5, pady=2)
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            row_index += 1

        if material_choice_var is not None and material_price_var is not None:
            existing = _coerce_float_or_none(material_price_var.get())
            if existing is None or abs(existing) < 1e-9:
                update_material_price()

        def create_global_entries(parent_frame: ttk.Labelframe, keys, data_source, var_dict, columns: int = 2) -> None:
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
        create_global_entries(rates_frame, sorted(self.rates.keys()), self.rates, self.rate_vars, columns=3)

        self._building_editor = False
        try:
            self._apply_geo_defaults(self.geo)
        except Exception:
            pass

    def _register_editor_field(self, label: str, var: tk.StringVar, label_widget: ttk.Label | None) -> None:
        if not isinstance(label, str) or not isinstance(var, tk.StringVar):
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

    def _bind_editor_var(self, label: str, var: tk.StringVar) -> None:
        def _on_write(*_):
            if self._building_editor or self._editor_set_depth > 0:
                return
            self._update_editor_override_from_label(label, var.get())
            self._mark_label_source(label, "User")
            self.reprice()

        var.trace_add("write", _on_write)

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
        if not isinstance(geo_data, dict):
            return

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
            return

        def find_value(*keys: str) -> Any:
            for key in keys:
                for src in sources:
                    if key in src:
                        val = src.get(key)
                        if val not in (None, ""):
                            return val
            return None

        plate_len_in = find_value("plate_len_in", "plate_length_in")
        plate_wid_in = find_value("plate_wid_in", "plate_width_in")

        thickness_in = find_value("thickness_in_guess", "thickness_in", "deepest_hole_in")
        if thickness_in is None:
            thickness_mm = find_value("thickness_mm")
            if thickness_mm is not None:
                try:
                    thickness_in = float(thickness_mm) / 25.4
                except Exception:
                    thickness_in = None

        self._fill_editor_if_blank("Plate Length (in)", plate_len_in)
        self._fill_editor_if_blank("Plate Width (in)", plate_wid_in)
        self._fill_editor_if_blank("Thickness (in)", thickness_in)

        from_back = False
        for key in ("holes_from_back", "needs_back_face", "from_back"):
            if any(bool(src.get(key)) for src in sources if isinstance(src, dict)):
                from_back = True
                break
        if from_back:
            self._fill_editor_if_blank("Number of Milling Setups", 2, source="GEO")

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
                    eff_hours[proc] = float(eff_hours[proc]) * float(mult)
                except Exception:
                    continue
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
            self.reprice()

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
                    structured_pdf = extract_pdf_all(Path(path))
                    g2d = extract_2d_features_from_pdf_vector(path)   # PyMuPDF vector-only MVP
                else:
                    g2d = extract_2d_features_from_dxf_or_dwg(
                        path,
                        build_geo_from_dxf=self.geometry_loader.build_geo_from_dxf,
                    )   # ezdxf / ODA

                if self.vars_df is None:
                    vp = find_variables_near(path) or filedialog.askopenfilename(title="Select variables CSV/XLSX")
                    if vp:
                        try:
                            core_df, full_df = read_variables_file(vp, return_full=True)
                            self.vars_df = core_df
                            self.vars_df_full = full_df
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
                self.geo_context = dict(g2d or {})
                inner_geo = g2d.get("geo") if isinstance(g2d.get("geo"), dict) else {}
                read_more_geo = g2d.get("geo_read_more") if isinstance(g2d.get("geo_read_more"), dict) else {}
                hole_count_val = (
                    _coerce_float_or_none(read_more_geo.get("hole_count_geom"))
                    or _coerce_float_or_none(inner_geo.get("hole_count"))
                    or _coerce_float_or_none(g2d.get("hole_count"))
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
                    or _coerce_float_or_none(g2d.get("tap_qty"))
                    or 0
                )
                cbore_qty_val = int(
                    _coerce_float_or_none(read_more_geo.get("cbore_qty"))
                    or _coerce_float_or_none(inner_geo.get("cbore_qty"))
                    or _coerce_float_or_none(g2d.get("cbore_qty"))
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
            geo = extract_features_with_occ(path)  # handles STEP/IGES/BREP
        except Exception:
            geo = None

        if geo is None:
            if ext == ".stl":
                stl_geo = None
                try:
                    stl_geo = enrich_geo_stl(path)  # trimesh-based
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
                        shape = read_step_shape(path)
                    else:
                        shape = read_cad_any(path)            # IGES/BREP and others
                    _ = safe_bbox(shape)
                    g = enrich_geo_occ(shape)             # OCC-based geometry features
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
                vp = filedialog.askopenfilename(title="Select variables CSV/XLSX")
            if not vp:
                messagebox.showinfo("Variables", "No variables file provided.")
                self.status_var.set("Ready")
                return
            try:
                core_df, full_df = read_variables_file(vp, return_full=True)
                self.vars_df = core_df
                self.vars_df_full = full_df
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
        decision_log = {}
        est_raw = infer_hours_and_overrides_from_geo(geo, params=self.params, rates=self.rates)
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
            def normalize_item(value: str) -> str:
                cleaned = re.sub(r"[^0-9a-z&$]+", " ", str(value).strip().lower())
                return re.sub(r"\s+", " ", cleaned).strip()
            normalized_items = self.vars_df["Item"].astype(str).apply(normalize_item)
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

        rate_raw_values = {k: var.get() for k, var in self.rate_vars.items()}
        rate_updates = {}
        for key, raw_value in rate_raw_values.items():
            try:
                rate_updates[key] = float(str(raw_value).strip())
            except Exception:
                rate_updates[key] = self.rates.get(key, 0.0)
        self.rates.update(rate_updates)

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
            messagebox.showinfo("Overrides", f"Saved to:\n{{path}}")
            self.status_var.set(f"Saved overrides to {path}")
        except Exception as e:
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
            if "rates" in data: self.rates.update(data["rates"])
            for k,v in self.param_vars.items(): v.set(str(self.params.get(k, "")))
            for k,v in self.rate_vars.items():  v.set(str(self.rates.get(k, "")))
            messagebox.showinfo("Overrides", "Overrides loaded.")
            self.status_var.set(f"Loaded overrides from {path}")
        except Exception as e:
            messagebox.showerror("Overrides", f"Load failed:\n{{e}}")
            self.status_var.set("Failed to load overrides.")

    # ----- LLM tab ----- 
    def _build_llm(self, parent):
        row=0
        ttk.Checkbutton(parent, text="Enable LLM (Qwen via llama-cpp, offline)", variable=self.llm_enabled).grid(row=row, column=0, sticky="w", pady=(6,2)); row+=1
        ttk.Label(parent, text="Qwen GGUF model path").grid(row=row, column=0, sticky="e", padx=5, pady=3)
        ttk.Entry(parent, textvariable=self.llm_model_path, width=80).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(parent, text="Browse...", command=self._pick_model).grid(row=row, column=2, padx=5); row+=1
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
            out = infer_shop_overrides_from_geo(self.geo)
        except Exception as e:
            self.llm_txt.insert("end", f"LLM error: {{e}}\n"); return
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
        from tkinter import scrolledtext, messagebox
        from pathlib import Path

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
        self.out_txt.insert("end", d+"\n")
        self.out_txt.see("end")

    def reprice(self) -> None:
        if self._reprice_in_progress:
            return
        self.gen_quote(reuse_suggestions=True)

    def gen_quote(self, reuse_suggestions: bool = False) -> None:
        already_repricing = self._reprice_in_progress
        if not already_repricing:
            self._reprice_in_progress = True
        try:
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

            try:
                res = compute_quote_from_df(
                    self.vars_df,
                    params=self.params,
                    rates=self.rates,
                    default_params=self.configuration.default_params,
                    default_rates=self.pricing_registry.default_rates,
                    default_material_display=self.configuration.default_material_display,
                    material_vendor_csv=self.settings.get("material_vendor_csv", "") if isinstance(self.settings, dict) else "",
                    llm_enabled=self.llm_enabled.get(),
                    llm_model_path=self.llm_model_path.get().strip() or None,
                    geo=self.geo,
                    ui_vars=ui_vars,
                    quote_state=self.quote_state,
                    reuse_suggestions=reuse_suggestions,
                    llm_suggest=self.LLM_SUGGEST,
                )
            except ValueError as err:
                messagebox.showerror("Quote blocked", str(err))
                self.status_var.set("Quote blocked.")
                return

            baseline_ctx = self.quote_state.baseline or {}
            suggestions_ctx = self.quote_state.suggestions or {}
            if baseline_ctx and suggestions_ctx:
                self.apply_llm_to_editor(suggestions_ctx, baseline_ctx)

            model_path = self.llm_model_path.get().strip()
            llm_explanation = get_llm_quote_explanation(res, model_path)
            report = render_quote(res, currency="$", show_zeros=False, llm_explanation=llm_explanation)
            self.out_txt.delete("1.0", "end")
            self.out_txt.insert("end", report, "rcol")

            self.nb.select(self.tab_out)
            self.status_var.set(f"Quote Generated! Final Price: ${res.get('price', 0):,.2f}")
        finally:
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
    if L is not None: out["GEO__BBox_X_mm"] = L
    if W is not None: out["GEO__BBox_Y_mm"] = W
    if H is not None: out["GEO__BBox_Z_mm"] = H
    dims = [d for d in [L,W,H] if d is not None]
    if dims:
        out["GEO__MaxDim_mm"] = max(dims)
        out["GEO__MinDim_mm"] = min(dims)
        out["GEO__Stock_Thickness_mm"] = min(dims)
    v = getf("GEO-Volume_mm3") or getf("GEO-Volume_mm3")  # keep both spellings if present
    if v is not None: out["GEO__Volume_mm3"] = v
    a = getf("GEO-SurfaceArea_mm2")
    if a is not None: out["GEO__SurfaceArea_mm2"] = a
    fc = getf("Feature_Face_Count") or getf("GEO_Face_Count")
    if fc is not None: out["GEO__Face_Count"] = fc
    wedm = getf("GEO_WEDM_PathLen_mm")
    if wedm is not None: out["GEO__WEDM_PathLen_mm"] = wedm
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
    vals  = df["Example Values / Options"]
    m = items.str.startswith("GEO__", na=False)
    for _, row in df.loc[m].iterrows():
        k = str(row["Item"]).strip()
        try:
            geo[k] = float(row["Example Values / Options"]) if row["Example Values / Options"] is not None else 0.0
        except Exception:
            continue
    return geo

def update_variables_df_with_geo(df, geo: dict):
    cols = ["Item","Example Values / Options","Data Type / Input Method"]
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
    if thk > 50: rparams["OverheadPct"] = params.get("OverheadPct", 0.15) + 0.02
    if wedm > 500: rrates["WireEDMRate"] = rates.get("WireEDMRate", 140.0) * 1.05
    return {"params": rp, "rates": rr}

def _run_llm_json_stub(prompt: str, model_path: str):
    return {"params": {}, "rates": {}}

def suggest_overrides_from_cad(df_vars, params, rates, model_path: str):
    geo = _collect_geo_features_from_df(df_vars)
    base = _rule_based_overrides(geo, params, rates)
    llm_json = _run_llm_json_stub("", model_path)
    merged_p = {**params, **base.get("params", {}), **llm_json.get("params", {})}
    merged_r = {**rates,  **base.get("rates", {}),  **llm_json.get("rates", {})}
    return {"params": merged_p, "rates": merged_r, "geo": geo}

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
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.print_env:
        print(json.dumps(describe_runtime_environment(), indent=2))
        return 0

    if args.no_gui:
        return 0

    configuration = QuoteConfiguration(
        default_params=copy.deepcopy(PARAMS_DEFAULT),
        default_material_display=DEFAULT_MATERIAL_DISPLAY,
    )
    pricing_registry = PricingRegistry(default_rates=copy.deepcopy(RATES_DEFAULT))
    geometry_loader = GeometryLoader(build_geo_from_dxf=build_geo_from_dxf_path)
    llm_client = LLMClient(
        primary_loader=lambda: load_qwen_vl(n_ctx=8192, n_gpu_layers=20),
        fallback_loader=lambda: load_qwen_vl(n_ctx=4096, n_gpu_layers=0),
    )

    App(
        configuration=configuration,
        geometry_loader=geometry_loader,
        pricing_registry=pricing_registry,
        llm_client=llm_client,
    ).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
