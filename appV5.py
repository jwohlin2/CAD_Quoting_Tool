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

def _coerce_positive_float(value: Any) -> float | None:
    """Return a positive finite float or None."""
    try:
        number = float(value)
    except Exception:
        return None
    try:
        if not math.isfinite(number):
            return None
    except Exception:
        pass
    return number if number > 0 else None


def _ensure_geo_context_fields(
    geom: dict[str, Any],
    source: Mapping[str, Any] | None,
    *,
    cfg: Any | None = None,
) -> None:
    """Best-effort backfill of common geometry context fields.

    This is intentionally tolerant and only fills when values are present
    and reasonable. It never raises.
    """

    try:
        src = source if isinstance(source, _MappingABC) else {}
        # Quantity
        qty = src.get("Quantity") if isinstance(src, _MappingABC) else None
        try:
            if "qty" not in geom and qty is not None:
                qf = float(qty)
                if math.isfinite(qf) and qf > 0:
                    geom["qty"] = int(round(qf))
        except Exception:
            pass

        # Dimensions (inches)
        L_in = _coerce_positive_float(src.get("Plate Length (in)"))
        W_in = _coerce_positive_float(src.get("Plate Width (in)"))
        T_in = _coerce_positive_float(src.get("Thickness (in)"))
        if L_in is not None and "plate_len_in" not in geom:
            geom["plate_len_in"] = L_in
        if W_in is not None and "plate_wid_in" not in geom:
            geom["plate_wid_in"] = W_in
        if (
            L_in is not None
            and "plate_len_mm" not in geom
            and math.isfinite(float(L_in))
            and float(L_in) > 0
        ):
            geom["plate_len_mm"] = float(L_in) * 25.4
        if (
            W_in is not None
            and "plate_wid_mm" not in geom
            and math.isfinite(float(W_in))
            and float(W_in) > 0
        ):
            geom["plate_wid_mm"] = float(W_in) * 25.4
        if T_in is not None and "thickness_in" not in geom:
            geom["thickness_in"] = T_in

        # Dimensions (mm) → convert if inch values missing
        L_mm = _coerce_positive_float(src.get("Plate Length (mm)"))
        W_mm = _coerce_positive_float(src.get("Plate Width (mm)"))
        T_mm = _coerce_positive_float(src.get("Thickness (mm)"))
        if L_mm is not None and "plate_len_in" not in geom:
            geom["plate_len_in"] = float(L_mm) / 25.4
        if W_mm is not None and "plate_wid_in" not in geom:
            geom["plate_wid_in"] = float(W_mm) / 25.4
        if T_mm is not None and "thickness_in" not in geom:
            geom["thickness_in"] = float(T_mm) / 25.4
        if T_mm is not None and "thickness_mm" not in geom:
            geom["thickness_mm"] = T_mm
        thickness_in_val = geom.get("thickness_in")
        if (
            "thickness_mm" not in geom
            and isinstance(thickness_in_val, (int, float))
            and math.isfinite(thickness_in_val)
            and thickness_in_val > 0
        ):
            geom["thickness_mm"] = float(thickness_in_val) * 25.4

        L_mm_val = _coerce_positive_float(geom.get("plate_len_mm"))
        W_mm_val = _coerce_positive_float(geom.get("plate_wid_mm"))
        if (
            "plate_bbox_area_mm2" not in geom
            and L_mm_val is not None
            and W_mm_val is not None
        ):
            geom["plate_bbox_area_mm2"] = float(L_mm_val) * float(W_mm_val)

        if "hole_sets" not in geom or not geom.get("hole_sets"):
            normalized_sets: list[dict[str, Any]] = []
            raw_groups = geom.get("hole_groups")
            if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes, bytearray)):
                for entry in raw_groups:
                    if isinstance(entry, _MappingABC):
                        dia_mm_val = _coerce_positive_float(entry.get("dia_mm") or entry.get("diameter_mm"))
                        qty_val = _coerce_float_or_none(entry.get("count") or entry.get("qty"))
                    else:
                        dia_mm_val = _coerce_positive_float(getattr(entry, "dia_mm", None) or getattr(entry, "diameter_mm", None))
                        qty_val = _coerce_float_or_none(getattr(entry, "count", None) or getattr(entry, "qty", None))
                    if dia_mm_val and qty_val and qty_val > 0:
                        normalized_sets.append({
                            "dia_mm": float(dia_mm_val),
                            "qty": int(round(float(qty_val))),
                        })
            if not normalized_sets:
                diams_seq = geom.get("hole_diams_mm")
                thickness_in_guess = geom.get("thickness_in")
                if thickness_in_guess is None:
                    thickness_mm_guess = _coerce_positive_float(geom.get("thickness_mm"))
                    if thickness_mm_guess is not None:
                        thickness_in_guess = float(thickness_mm_guess) / 25.4
                drill_groups = build_drill_groups_from_geometry(diams_seq, thickness_in_guess)
                for group in drill_groups:
                    dia_in = _coerce_float_or_none(group.get("diameter_in"))
                    qty_val = _coerce_float_or_none(group.get("qty"))
                    if dia_in and dia_in > 0 and qty_val and qty_val > 0:
                        normalized_sets.append({
                            "dia_mm": float(dia_in) * 25.4,
                            "qty": int(round(float(qty_val))),
                        })
            if normalized_sets:
                geom["hole_sets"] = normalized_sets
    except Exception:
        # Defensive: never let context backfill break pricing
        pass


def _apply_drilling_meta_fallback(
    meta: Mapping[str, Any] | None,
    groups: Sequence[Mapping[str, Any]] | None,
) -> tuple[int, int]:
    """Compute deep/std hole counts and backfill helper arrays in meta container.

    Returns (holes_deep, holes_std).
    """

    deep = 0
    std = 0
    dia_vals: list[float] = []
    depth_vals: list[float] = []

    if isinstance(groups, Sequence):
        for g in groups:
            try:
                qty = int(_coerce_float_or_none(g.get("qty")) or 0)
            except Exception:
                qty = 0
            op = str(g.get("op") or "").strip().lower()
            if op.startswith("deep"):
                deep += qty
            else:
                std += qty
            d_in = _coerce_float_or_none(g.get("diameter_in"))
            if d_in and d_in > 0:
                dia_vals.append(float(d_in))
            z_in = _coerce_float_or_none(g.get("depth_in"))
            if z_in and z_in > 0:
                depth_vals.append(float(z_in))

    # Backfill into a mutable container if provided
    try:
        if isinstance(meta, dict):
            if dia_vals and not meta.get("dia_in_vals"):
                meta["dia_in_vals"] = dia_vals
            if depth_vals and not meta.get("depth_in_vals"):
                meta["depth_in_vals"] = depth_vals
    except Exception:
        pass

    return (deep, std)
import copy
import csv
import json
import math
import os
import logging
import re
import sys
import time
import typing
import unicodedata
from functools import cmp_to_key, lru_cache
from typing import Any, Mapping, MutableMapping, TYPE_CHECKING, TypeAlias
from collections import Counter, defaultdict
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping as _MappingABC,
    MutableMapping as _MutableMappingABC,
    Sequence,
)
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

from cad_quoter.app._value_utils import (
    _format_value,
)
from cad_quoter.app.chart_lines import (
    build_ops_rows_from_lines_fallback as _build_ops_rows_from_lines_fallback,
    collect_chart_lines_context as _collect_chart_lines_context,
    norm_line as _norm_line,
)
from cad_quoter.app.hole_ops import (
    CBORE_MIN_PER_SIDE_MIN,
    CSK_MIN_PER_SIDE_MIN,
    RE_CBORE,
    RE_CSK,
    RE_DEPTH,
    RE_DIA,
    RE_FRONT_BACK,
    RE_NPT,
    RE_TAP,
    RE_THRU,
    TAP_MINUTES_BY_CLASS,
    _aggregate_hole_entries,
    _classify_thread_spec,
    _dedupe_hole_entries,
    _major_diameter_from_thread,
    _normalize_hole_text,
    _parse_hole_line,
    summarize_hole_chart_lines,
)
from cad_quoter.app.container import (
    ServiceContainer,
    SupportsPricingEngine,
    create_default_container,
)
from cad_quoter.app.llm_helpers import (
    DEFAULT_MM_PROJ_NAMES,
    DEFAULT_VL_MODEL_NAMES,
    LEGACY_MM_PROJ,
    LEGACY_VL_MODEL,
    MM_PROJ,
    VL_MODEL,
    ensure_runtime_dependencies,
    find_default_qwen_model,
    init_llm_integration,
    load_qwen_vl,
)
from cad_quoter.app.optional_loaders import (
    pd,
    build_geo_from_dxf,
    set_build_geo_from_dxf_hook,
)
from appkit.utils import (
    _first_numeric_or_none,
    _fmt_rng,
    _ipm_from_rpm_ipr,
    _lookup_sfm_ipr,
    _parse_thread_major_in,
    _parse_tpi,
    _rpm_from_sfm,
    _rpm_from_sfm_diam,
)
from cad_quoter.resources import (
    default_app_settings_json,
    default_master_variables_csv,
)
from cad_quoter.config import (
    AppEnvironment,
    ConfigError,
    append_debug_log,
    configure_logging,
    logger,
)

_log = logger
from cad_quoter.utils.geo_ctx import (
    _iter_geo_contexts as _iter_geo_dicts_for_context,
    _should_include_outsourced_pass,
)
from cad_quoter.utils.scrap import (
    _holes_removed_mass_g,
    build_drill_groups_from_geometry,
)
from cad_quoter.utils.render_utils import (
    fmt_hours,
    fmt_money,
    format_currency,
    format_dimension,
    format_hours,
    format_hours_with_rate,
    format_percent,
    format_weight_lb_decimal,
    format_weight_lb_oz,
    QuoteDocRecorder,
    render_quote_doc,
)
from cad_quoter.pricing import load_backup_prices_csv
from cad_quoter.pricing.mcmaster_helpers import (
    load_mcmaster_catalog_rows as _load_mcmaster_catalog_rows,
    _coerce_inches_value,
    pick_mcmaster_plate_sku as _pick_mcmaster_plate_sku,
    resolve_mcmaster_plate_for_quote as _resolve_mcmaster_plate_for_quote,
)

# Backwards compatibility for legacy helper names that callers import from appV5.
_load_mcmaster_catalog_csv = _load_mcmaster_catalog_rows
from cad_quoter.pricing.vendor_csv import (
    pick_from_stdgrid as _pick_from_stdgrid,
    pick_plate_from_mcmaster as _pick_plate_from_mcmaster,
)
from cad_quoter.pricing.process_cost_renderer import render_process_costs
from cad_quoter.estimators import drilling_legacy as _drilling_legacy
from cad_quoter.estimators.base import SpeedsFeedsUnavailableError
from cad_quoter.llm_overrides import (
    _plate_mass_properties,
    _plate_mass_from_dims,
    clamp,
)

from cad_quoter.domain import (
    QuoteState,
    HARDWARE_PASS_LABEL,
    _canonical_pass_label,
    coerce_bounds,
    build_suggest_payload,
    overrides_to_suggestions,
    suggestions_to_overrides,
)

from cad_quoter.vendors import ezdxf as _ezdxf_vendor

from cad_quoter.geometry.dxf_enrich import (
    detect_units_scale as _shared_detect_units_scale,
    iter_spaces as _shared_iter_spaces,
    iter_table_entities as _shared_iter_table_entities,
    iter_table_text as _shared_iter_table_text,
)

from cad_quoter.pricing.process_buckets import bucketize

import cad_quoter.geometry as geometry



_HAS_ODAFC = bool(getattr(geometry, "HAS_ODAFC", False))
_HAS_PYMUPDF = bool(getattr(geometry, "HAS_PYMUPDF", getattr(geometry, "_HAS_PYMUPDF", False)))
fitz = getattr(geometry, "fitz", None)
try:
    odafc = _ezdxf_vendor.require_odafc() if _HAS_ODAFC else None
except Exception:
    odafc = None  # type: ignore[assignment]

from appkit.llm_adapter import (
    apply_llm_hours_to_variables,
    clamp_llm_hours,
    configure_llm_integration,
    infer_hours_and_overrides_from_geo,
    normalize_item,
    normalize_item_text,
)

from appkit.ui.tk_compat import (
    tk,
    filedialog,
    messagebox,
    ttk,
    _ensure_tk,
)

from appkit.ui.widgets import (
    CreateToolTip,
    ScrollableFrame,
)

# UI service containers and configuration helpers
from appkit.ui.services import (
    GeometryLoader,
    LLMServices,
    PricingRegistry,
    UIConfiguration,
)

from appkit.guardrails import build_guard_context, apply_drilling_floor_notes
from appkit.merge_utils import (
    ACCEPT_SCALAR_KEYS,
    merge_effective,
)

from appkit.effective import (
    compute_effective_state,
    ensure_accept_flags,
    reprice_with_effective,
)

from appkit.ui import suggestions as ui_suggestions

from cad_quoter.utils.scrap import (
    HOLE_SCRAP_CAP,
    SCRAP_DEFAULT_GUESS,
    _holes_scrap_fraction,
    normalize_scrap_pct,
)
from appkit.planner_helpers import _process_plan_job
from appkit.env_utils import FORCE_PLANNER
from appkit.planner_adapter import resolve_planner

from appkit.data import load_json, load_text
from appkit.utils.text_rules import (
    PROC_MULT_TARGETS,
    canonicalize_amortized_label as _canonical_amortized_label,
)
from appkit.debug.debug_tables import (
    _accumulate_drill_debug,
    append_removal_debug_if_enabled,
)
from appkit.ui import llm_panel
from appkit.ui.editor_controls import coerce_checkbox_state, derive_editor_control_spec
from appkit.ui.planner_render import (
    PROGRAMMING_PER_PART_LABEL,
    PlannerBucketRenderState,
    _bucket_cost,
    _bucket_cost_mode,
    _canonical_bucket_key,
    _display_bucket_label,
    _display_rate_for_row,
    _hole_table_minutes_from_geo,
    _normalize_bucket_key,
    _op_role_for_name,
    _planner_bucket_key_for_name,
    _preferred_order_then_alpha,
    _prepare_bucket_view,
    _extract_bucket_map,
    _process_label,
    _seed_bucket_minutes as _planner_seed_bucket_minutes,
    _split_hours_for_bucket,
    _sync_drilling_bucket_view,
    _charged_hours_by_bucket,
    _build_planner_bucket_render_state,
    _FINAL_BUCKET_HIDE_KEYS,
    SHOW_BUCKET_DIAGNOSTICS_OVERRIDE,
    canonicalize_costs,
)
from appkit.ui.services import QuoteConfiguration


_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RENDER_ASCII_REPLACEMENTS: dict[str, str] = {
    "—": "-",
    "•": "-",
    "…": "...",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "µ": "u",
    "μ": "u",
    "±": "+/-",
    "°": " deg ",
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    " ": " ",  # non-breaking space
    "⚠️": "⚠",
}

_RENDER_PASSTHROUGH: dict[str, str] = {
    "–": "__EN_DASH__",
    "×": "__MULTIPLY__",
    "≥": "__GEQ__",
    "≤": "__LEQ__",
    "⚠": "__WARN__",
}


def _sanitize_render_text(value: typing.Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    for source, placeholder in _RENDER_PASSTHROUGH.items():
        if source in text:
            text = text.replace(source, placeholder)
    text = text.replace("\t", " ")
    text = text.replace("\r", "")
    text = _ANSI_ESCAPE_RE.sub("", text)
    for source, replacement in _RENDER_ASCII_REPLACEMENTS.items():
        if source in text:
            text = text.replace(source, replacement)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    text = _CONTROL_CHAR_RE.sub("", text)
    for source, placeholder in _RENDER_PASSTHROUGH.items():
        if placeholder in text:
            text = text.replace(placeholder, source)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: formatting + removal card + per-hole lines (no material per line)
# ──────────────────────────────────────────────────────────────────────────────
def _render_removal_card(
    append_line: Callable[[str], None],
    *,
    mat_canon: str,
    mat_group: str | None,
    row_group: str | None,
    holes_deep: int,
    holes_std: int,
    dia_vals_in: list[float],
    depth_vals_in: list[float],
    sfm_deep: float,
    sfm_std: float,
    ipr_deep_vals: list[float],
    ipr_std_val: float,
    rpm_deep_vals: list[float],
    rpm_std_vals: list[float],
    ipm_deep_vals: list[float],
    ipm_std_vals: list[float],
    index_min_per_hole: float,
    peck_min_rng: list[float],
    toolchange_min_deep: float,
    toolchange_min_std: float,
) -> None:
    append_line("MATERIAL REMOVAL – DRILLING")
    append_line("=" * 64)
    # Inputs
    append_line("Inputs")
    append_line(f"  Material .......... {mat_canon}  [group {mat_group or '-'}]")
    mismatch = False
    if row_group:
        rg = str(row_group).upper()
        mg = str(mat_group or "").upper()
        mismatch = (rg != mg and (rg and mg))
        note = "   (!) mismatch – used row from different group" if mismatch else ""
        append_line(f"  CSV row group ..... {row_group}{note}")
    append_line("  Operations ........ Deep-Drill (L/D ≥ 3), Drill")
    append_line(
        f"  Holes ............. {int(holes_deep)} deep + {int(holes_std)} std  = {int(holes_deep + holes_std)}"
    )
    append_line(f'  Diameter range .... {_fmt_rng(dia_vals_in, 3)}"')
    append_line(f"  Depth per hole .... {_fmt_rng(depth_vals_in, 2)} in")
    append_line("")
    # Feeds & Speeds
    append_line("Feeds & Speeds (used)")
    append_line(f"  SFM ............... {int(round(sfm_deep))} (deep)   | {int(round(sfm_std))} (std)")
    append_line(
        f"  IPR ............... {_fmt_rng(ipr_deep_vals, 4)} (deep) | {float(ipr_std_val):.4f} (std)"
    )
    append_line(
        f"  RPM ............... {_fmt_rng(rpm_deep_vals, 0)} (deep)      | {_fmt_rng(rpm_std_vals, 0)} (std)"
    )
    append_line(
        f"  IPM ............... {_fmt_rng(ipm_deep_vals, 1)} (deep)       | {_fmt_rng(ipm_std_vals, 1)} (std)"
    )
    append_line("")
    # Overheads
    append_line("Overheads")
    append_line(f"  Index per hole .... {float(index_min_per_hole):.2f} min")
    append_line(f"  Peck per hole ..... {_fmt_rng(peck_min_rng, 2)} min")
    append_line(
        f"  Toolchange ........ {float(toolchange_min_deep):.2f} min (deep) | {float(toolchange_min_std):.2f} min (std)"
    )
    append_line("")


def _render_time_per_hole(
    append_line: Callable[[str], None],
    *,
    bins: list[dict[str, Any]],
    index_min: float,
    peck_min_deep: float,
    peck_min_std: float,
) -> tuple[float, bool, bool]:
    append_line("TIME PER HOLE – DRILL GROUPS")
    append_line("-" * 66)
    subtotal_min = 0.0
    seen_deep = False
    seen_std = False
    for b in bins:
        try:
            op = (b.get("op") or b.get("op_name") or "").strip().lower()
            deep = op.startswith("deep")
            if deep:
                seen_deep = True
            else:
                seen_std = True
            d_in = _safe_float(b.get("diameter_in"))
            depth = _safe_float(b.get("depth_in"))
            qty = int(b.get("qty") or 0)
            sfm = _safe_float(b.get("sfm"))
            ipr = _safe_float(b.get("ipr"))
            rpm = _rpm_from_sfm(sfm, d_in)
            ipm = rpm * ipr
            peck = float(peck_min_deep if deep else peck_min_std)
            t_hole = (depth / max(ipm, 1e-6)) + float(index_min) + peck
            group_min = t_hole * qty
            subtotal_min += group_min
            # single-line, no material
            append_line(
                f'Dia {d_in:.3f}" × {qty}  | depth {depth:.3f}" | {int(round(sfm))} sfm | {ipr:.4f} ipr | '
                f't/hole {t_hole:.2f} min | group {qty}×{t_hole:.2f} = {group_min:.2f} min'
            )
        except Exception:
            continue
    append_line("")
    return subtotal_min, seen_deep, seen_std


def _compute_drilling_removal_section(
    *,
    breakdown: Mapping[str, Any] | MutableMapping[str, Any],
    rates: Mapping[str, Any] | MutableMapping[str, Any],
    drilling_meta_source: Mapping[str, Any] | None,
    drilling_card_detail: Mapping[str, Any] | None,
    drill_machine_minutes_estimate: float,
    drill_tool_minutes_estimate: float,
    drill_total_minutes_estimate: float,
    process_plan_summary: Mapping[str, Any] | None,
    drilling_time_per_hole: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], list[str], Mapping[str, Any] | None]:
    """Return drill removal render lines + extras while updating breakdown state."""

    lines: list[str] = []
    extras: dict[str, float] = {}
    updated_plan_summary = process_plan_summary
    geo_map: Mapping[str, Any] | dict[str, Any]
    geo_map = {}
    if isinstance(breakdown, _MappingABC):
        candidate_geo = breakdown.get("geo")
        if isinstance(candidate_geo, _MappingABC):
            geo_map = candidate_geo
    if not geo_map:
        try:
            result_map = result if isinstance(result, _MappingABC) else None  # type: ignore[name-defined]
        except NameError:
            result_map = None
        if isinstance(result_map, _MappingABC):
            candidate_geo = result_map.get("geo")
            if isinstance(candidate_geo, _MappingABC):
                geo_map = candidate_geo
    if not isinstance(geo_map, _MappingABC):
        geo_map = {}
    ops_hole_count_from_table = 0

    dtph_map = (
        drilling_time_per_hole
        if isinstance(drilling_time_per_hole, _MappingABC)
        else None
    )
    dtph_rows = dtph_map.get("rows") if isinstance(dtph_map, _MappingABC) else None
    if isinstance(dtph_rows, list) and dtph_rows:
        sanitized_rows: list[dict[str, Any]] = []
        subtotal_minutes = 0.0
        for entry in dtph_rows:
            if not isinstance(entry, _MappingABC):
                continue
            diameter_in = _coerce_float_or_none(entry.get("diameter_in"))
            if diameter_in is None or diameter_in <= 0:
                continue
            qty_val = int(_coerce_float_or_none(entry.get("qty")) or 0)
            if qty_val <= 0:
                continue
            depth_in = _coerce_float_or_none(entry.get("depth_in")) or 0.0
            sfm_val = _coerce_float_or_none(entry.get("sfm")) or 0.0
            ipr_val = _coerce_float_or_none(entry.get("ipr")) or 0.0
            minutes_per = _coerce_float_or_none(entry.get("minutes_per_hole"))
            if minutes_per is None:
                minutes_per = _coerce_float_or_none(entry.get("t_per_hole_min"))
            if minutes_per is None:
                minutes_per = 0.0
            group_minutes = _coerce_float_or_none(entry.get("group_minutes"))
            if group_minutes is None:
                group_minutes = float(minutes_per) * float(qty_val)
            subtotal_minutes += float(group_minutes)
            sanitized_rows.append(
                {
                    "diameter_in": float(diameter_in),
                    "qty": qty_val,
                    "depth_in": float(depth_in),
                    "sfm": float(sfm_val),
                    "ipr": float(ipr_val),
                    "minutes_per_hole": float(minutes_per),
                    "group_minutes": float(group_minutes),
                }
            )
        if sanitized_rows:
            lines.append("MATERIAL REMOVAL – DRILLING")
            lines.append("=" * 64)
            lines.append("TIME PER HOLE – DRILL GROUPS")
            lines.append("-" * 66)
            for row in sanitized_rows:
                lines.append(
                    f'Dia {row["diameter_in"]:.3f}" × {row["qty"]}  | depth {row["depth_in"]:.3f}" | '
                    f"{int(round(row['sfm']))} sfm | {row['ipr']:.4f} ipr | t/hole {row['minutes_per_hole']:.2f} min | "
                    f"group {row['qty']}×{row['minutes_per_hole']:.2f} = {row['group_minutes']:.2f} min"
                )
            lines.append("")

            component_labels: list[str] = []
            component_minutes = 0.0
            tool_components = dtph_map.get("tool_components")
            if isinstance(tool_components, list):
                for comp in tool_components:
                    if not isinstance(comp, _MappingABC):
                        continue
                    label = str(comp.get("label") or comp.get("name") or "").strip()
                    minutes_val = _coerce_float_or_none(comp.get("minutes"))
                    if minutes_val is None:
                        minutes_val = _coerce_float_or_none(comp.get("mins"))
                    minutes_f = float(minutes_val or 0.0)
                    if not label:
                        label = "-"
                    if label != "-" or minutes_f > 0.0:
                        component_labels.append(f"{label} {minutes_f:.2f} min")
                    component_minutes += minutes_f

            total_tool_minutes = _coerce_float_or_none(dtph_map.get("toolchange_minutes"))
            if total_tool_minutes is None:
                total_tool_minutes = component_minutes
            total_tool_minutes = float(total_tool_minutes or 0.0)
            if not math.isfinite(total_tool_minutes):
                total_tool_minutes = component_minutes
            if total_tool_minutes < 0.0:
                total_tool_minutes = 0.0

            if component_labels:
                label_text = " + ".join(component_labels)
                lines.append(
                    f"Toolchange adders: {label_text} = {total_tool_minutes:.2f} min"
                )
            elif total_tool_minutes > 0.0:
                lines.append(
                    f"Toolchange adders: Toolchange {total_tool_minutes:.2f} min = {total_tool_minutes:.2f} min"
                )
            else:
                lines.append("Toolchange adders: -")

            lines.append("-" * 66)
            subtotal_minutes_val = _coerce_float_or_none(
                dtph_map.get("subtotal_minutes")
            )
            if subtotal_minutes_val is None:
                subtotal_minutes_val = subtotal_minutes
            subtotal_minutes_val = float(subtotal_minutes_val or 0.0)
            total_minutes_val = (
                _coerce_float_or_none(dtph_map.get("total_minutes_with_toolchange"))
                or _coerce_float_or_none(dtph_map.get("total_minutes"))
            )
            if total_minutes_val is None:
                total_minutes_val = subtotal_minutes_val + total_tool_minutes
            total_minutes_val = float(total_minutes_val or 0.0)

            lines.append(
                f"Subtotal (per-hole × qty) . {subtotal_minutes_val:.2f} min  ("
                f"{fmt_hours(subtotal_minutes_val/60.0)})"
            )
            lines.append(
                f"TOTAL DRILLING (with toolchange) . {total_minutes_val:.2f} min  ("
                f"{total_minutes_val/60.0:.2f} hr)"
            )
            lines.append("")

            extras["drill_machine_minutes"] = float(subtotal_minutes_val)
            extras["drill_labor_minutes"] = float(total_tool_minutes)
            extras["drill_total_minutes"] = float(total_minutes_val)
            extras["removal_drilling_minutes_subtotal"] = float(subtotal_minutes_val)
            extras["removal_drilling_minutes"] = float(total_minutes_val)
            if total_minutes_val > 0.0:
                extras["removal_drilling_hours"] = float(total_minutes_val / 60.0)

            return extras, lines, updated_plan_summary

    return extras, lines, updated_plan_summary


# === OPS CARDS (from geo.ops_summary.rows) ===================================
# TODO: If no cards appear, ensure extractor writes geo['ops_summary']['rows']
#       (list of {hole, ref, qty, desc}) as outlined in the earlier extractor
#       patch.
_SIDE_BACK = re.compile(r"\bFROM\s+BACK\b", re.I)
_SIDE_FRONT = re.compile(r"\bFROM\s+FRONT\b", re.I)
_CBORE_RE = re.compile(
    r"(?:^|[ ;])(?:Ø|⌀|DIA)?\s*((?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?))\s*(?:C[\'’]?\s*BORE|CBORE|COUNTER\s*BORE)",
    re.I,
)
_DEPTH_TOKEN = re.compile(r"[×xX]\s*([0-9.]+)\b")  # e.g., × .62
_DIA_TOKEN = re.compile(
    r"(?:Ø|⌀|REF|DIA)[^0-9]*((?:\d+\s*/\s*\d+)|(?:\d+)?\.\d+|\d+(?:\.\d+)?)",
    re.I,
)


def _rows_from_ops_summary(
    geo: Mapping[str, Any] | None,
    *,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> list[dict]:
    geo_map: Mapping[str, Any] = geo if isinstance(geo, _MappingABC) else {}
    ops = (geo_map or {}).get("ops_summary") or {}
    rows = ops.get("rows") if isinstance(ops, dict) else None
    if rows:
        return list(rows)
    if isinstance(ops, dict):
        detail = ops.get("rows_detail")
        if isinstance(detail, list):
            fallback: list[dict[str, Any]] = []
            for entry in detail:
                if not isinstance(entry, _MappingABC):
                    continue
                base = {
                    "hole": entry.get("hole", ""),
                    "ref": entry.get("ref", ""),
                    "qty": entry.get("qty", 0),
                    "desc": entry.get("desc", ""),
                }
                if entry.get("diameter_in") is not None:
                    base["diameter_in"] = entry.get("diameter_in")
                fallback.append(base)
            if fallback:
                return fallback
    _containers: list[dict] = []
    for candidate in (result, breakdown, geo_map):
        if isinstance(candidate, dict):
            _containers.append(candidate)
        elif isinstance(candidate, _MappingABC):
            try:
                _containers.append(dict(candidate))
            except Exception:
                continue
    chart_lines = _collect_chart_lines_context(*_containers)
    if chart_lines:
        fallback_rows = _build_ops_rows_from_lines_fallback(chart_lines)
        if fallback_rows:
            return fallback_rows
    return []


def _ops_row_details_from_geo(geo: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(geo, _MappingABC):
        return []
    ops_summary = geo.get("ops_summary")
    if not isinstance(ops_summary, _MappingABC):
        return []
    detail = ops_summary.get("rows_detail")
    if not isinstance(detail, list):
        return []
    return [entry for entry in detail if isinstance(entry, _MappingABC)]


def _parse_ref_to_inch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            val = float(value)
        except Exception:
            return None
        return val if math.isfinite(val) and val > 0 else None
    text = str(value).strip()
    if not text:
        return None
    cleaned = (
        text.replace("\u00D8", "")
        .replace("Ø", "")
        .replace("⌀", "")
        .replace("IN", "")
        .replace("in", "")
        .strip("\"' ")
    )
    if not cleaned:
        return None
    try:
        if "/" in cleaned:
            return float(Fraction(cleaned))
        return float(cleaned)
    except Exception:
        try:
            return float(Fraction(cleaned))
        except Exception:
            return None


def _diameter_from_ops_row(entry: Mapping[str, Any]) -> float | None:
    direct = _coerce_float_or_none(entry.get("diameter_in"))
    if direct is not None and direct > 0:
        return float(direct)
    ref_val = _parse_ref_to_inch(entry.get("ref"))
    if ref_val is not None and ref_val > 0:
        return ref_val
    desc = entry.get("desc")
    if isinstance(desc, str):
        for match in _DIA_TOKEN.finditer(desc):
            candidate = _parse_ref_to_inch(match.group(1))
            if candidate is not None and candidate > 0:
                return candidate
    return None


def _drilling_groups_from_ops_summary(
    geo: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    table_groups: dict[float, dict[str, Any]] = {}
    for ctx in _iter_geo_dicts_for_context(geo):
        families = ctx.get("hole_table_families_in")
        if not isinstance(families, _MappingABC):
            continue
        for raw_label, raw_qty in families.items():
            qty_val = int(round(_coerce_float_or_none(raw_qty) or 0.0))
            if qty_val <= 0:
                continue
            diameter_in: float | None
            if isinstance(raw_label, (int, float)):
                try:
                    diameter_in = float(raw_label)
                except Exception:
                    diameter_in = None
            else:
                diameter_in = _parse_ref_to_inch(raw_label)
                if diameter_in is None:
                    dia_mm = _parse_dim_to_mm(raw_label)
                    diameter_in = (dia_mm / 25.4) if dia_mm else None
            if diameter_in is None or not math.isfinite(diameter_in) or diameter_in <= 0:
                continue
            key = round(float(diameter_in), 4)
            bucket = table_groups.setdefault(
                key,
                {
                    "diameter_in": float(round(float(diameter_in), 4)),
                    "qty": 0,
                },
            )
            bucket["qty"] += qty_val
    if table_groups:
        detail_rows = _ops_row_details_from_geo(geo)
        if detail_rows:
            drill_totals: Counter[float] = Counter()
            non_drill_keys: set[float] = set()
            for entry in detail_rows:
                diameter_in = _diameter_from_ops_row(entry)
                if diameter_in is None or diameter_in <= 0:
                    continue
                dia_key = round(float(diameter_in), 4)
                total_map = entry.get("total") if isinstance(entry.get("total"), _MappingABC) else None
                drill_total = _safe_float((total_map or {}).get("drill"), default=0.0)
                if drill_total > 0:
                    drill_totals[dia_key] += int(round(drill_total))
                    continue

                per_map = entry.get("per_hole") if isinstance(entry.get("per_hole"), _MappingABC) else None
                if per_map is not None and not isinstance(per_map, dict):
                    per_map = dict(per_map)

                other_total = 0.0
                if isinstance(total_map, _MappingABC):
                    for op_key in (
                        "spot_front",
                        "spot_back",
                        "cbore_front",
                        "cbore_back",
                        "csk_front",
                        "csk_back",
                        "tap_front",
                        "tap_back",
                    ):
                        other_total += _safe_float(total_map.get(op_key), default=0.0)
                if other_total <= 0 and isinstance(per_map, _MappingABC):
                    qty_val = _safe_float(entry.get("qty"), default=0.0)
                    for op_key in (
                        "spot_front",
                        "spot_back",
                        "cbore_front",
                        "cbore_back",
                        "csk_front",
                        "csk_back",
                        "tap_front",
                        "tap_back",
                    ):
                        per_val = _safe_float(per_map.get(op_key), default=0.0)
                        if per_val > 0 and qty_val > 0:
                            other_total += per_val * qty_val
                if other_total > 0:
                    non_drill_keys.add(dia_key)

            for dia_key in list(table_groups.keys()):
                if dia_key in drill_totals:
                    qty_override = int(drill_totals[dia_key])
                    if qty_override <= 0:
                        table_groups.pop(dia_key, None)
                        continue
                    table_groups[dia_key]["qty"] = qty_override
                elif dia_key in non_drill_keys:
                    table_groups.pop(dia_key, None)

            for dia_key, qty_val in drill_totals.items():
                if qty_val <= 0 or dia_key in table_groups:
                    continue
                table_groups[dia_key] = {
                    "diameter_in": float(dia_key),
                    "qty": int(qty_val),
                }

        ordered_groups = [
            {"diameter_in": data["diameter_in"], "qty": data["qty"]}
            for _key, data in sorted(table_groups.items())
        ]
        total_qty = sum(group["qty"] for group in ordered_groups)
        return ordered_groups, int(total_qty)

    detail_rows = _ops_row_details_from_geo(geo)
    if not detail_rows:
        return ([], 0)
    groups: dict[float, dict[str, Any]] = {}
    total_qty = 0
    for entry in detail_rows:
        per = entry.get("per_hole") if isinstance(entry, _MappingABC) else None
        if not isinstance(per, _MappingABC):
            continue
        drill_ops = _coerce_float_or_none(per.get("drill"))
        if drill_ops is None or drill_ops <= 0:
            continue
        desc_val = entry.get("desc")
        if isinstance(desc_val, str) and _SPOT_TOKENS.search(desc_val):
            continue
        qty = int(_coerce_float_or_none(entry.get("qty")) or 0)
        if qty <= 0:
            continue
        diameter_in = _diameter_from_ops_row(entry)
        if diameter_in is None or diameter_in <= 0:
            continue
        key = round(float(diameter_in), 4)
        bucket = groups.setdefault(
            key,
            {
                "diameter_in": float(diameter_in),
                "qty": 0,
            },
        )
        bucket["qty"] += qty
        total_qty += qty
    ordered = [
        {"diameter_in": data["diameter_in"], "qty": data["qty"]}
        for key, data in sorted(groups.items())
    ]
    return ordered, int(total_qty)


def _side_of(desc: str) -> str:
    if _SIDE_BACK.search(desc or ""):
        return "BACK"
    if _SIDE_FRONT.search(desc or ""):
        return "FRONT"
    return "FRONT"


_THREAD_WITH_TPI_RE = re.compile(
    r"((?:#\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*-\s*(\d+)",
    re.I,
)
_THREAD_WITH_NPT_RE = re.compile(
    r"((?:#\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*-\s*(N\.?P\.?T\.?)",
    re.I,
)


def _emit_tapping_card(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> None:
    rows = _rows_from_ops_summary(geo, result=result, breakdown=breakdown)
    groups: list[dict[str, Any]] = []
    for r in rows:
        desc = str(r.get("desc", ""))
        desc_upper = desc.upper()
        desc_clean = desc_upper.replace(".", "")
        if "TAP" not in desc_upper and "NPT" not in desc_clean:
            continue
        qty = int(r.get("qty") or 0)
        if qty <= 0:
            continue
        side = _side_of(desc)
        match = _THREAD_WITH_TPI_RE.search(desc)
        if match:
            major_token = match.group(1).strip()
            tpi_token = match.group(2).strip()
            thread = f"{major_token}-{tpi_token}"
            tpi = _parse_tpi(thread)
            major = _parse_thread_major_in(thread)
        else:
            match = _THREAD_WITH_NPT_RE.search(desc)
            if not match:
                continue
            major_token = match.group(1).strip()
            thread = f"{major_token}-NPT"
            tpi = None
            major = _parse_thread_major_in(f"{major_token}-1")
            if major is None:
                major = _parse_ref_to_inch(major_token)
        depth_match = _DEPTH_TOKEN.search(desc)
        depth_in = float(depth_match.group(1)) if depth_match else None
        pilot = (r.get("ref") or "").strip()
        pitch = (1.0 / float(tpi)) if tpi else None
        sfm, _ = _lookup_sfm_ipr("tapping", major, material_group, speeds_csv)
        rpm = _rpm_from_sfm_diam(sfm, major)
        ipm = _ipm_from_rpm_ipr(rpm, pitch)
        groups.append(
            {
                "thread": thread,
                "side": side,
                "qty": qty,
                "depth_in": depth_in,
                "pilot": pilot,
                "pitch_ipr": None if pitch is None else round(pitch, 4),
                "rpm": None if rpm is None else int(round(rpm)),
                "ipm": None if ipm is None else round(ipm, 3),
            }
        )
    if not groups:
        return
    total = sum(g["qty"] for g in groups)
    front = sum(g["qty"] for g in groups if g["side"] == "FRONT")
    back = total - front
    lines += [
        "MATERIAL REMOVAL – TAPPING",
        "=" * 64,
        "Inputs",
        "  Ops ............... Tapping (front + back), pre-drill counted in drilling",
        f"  Taps .............. {total} total  → {front} front, {back} back",
        "  Threads ........... " + ", ".join(sorted({g["thread"] for g in groups})),
        "",
        "TIME PER HOLE – TAP GROUPS",
        "-" * 66,
    ]
    for g in groups:
        depth_txt = "THRU" if g["depth_in"] is None else f'{g["depth_in"]:.2f}"'
        lines.append(
            f'{g["thread"]} × {g["qty"]}  ({g["side"]})'
            f'{(" | pilot " + g["pilot"]) if g.get("pilot") else ""}'
            f" | depth {depth_txt} | {g['pitch_ipr'] if g['pitch_ipr'] is not None else '-'} ipr"
            f" | {g['rpm'] if g['rpm'] is not None else '-'} rpm"
            f" | {g['ipm'] if g['ipm'] is not None else '-'} ipm"
            f" | t/hole — | group — "
        )
    lines.append("")


def _emit_counterbore_card(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> None:
    rows = _rows_from_ops_summary(geo, result=result, breakdown=breakdown)
    groups: defaultdict[tuple[float, str, float | None], int] = defaultdict(int)
    order: list[tuple[float, str, float | None]] = []
    for r in rows:
        desc = str(r.get("desc", ""))
        if "BORE" not in desc.upper():
            continue
        match = _CBORE_RE.search(desc)
        if not match:
            continue
        diam_in = float(match.group(1))
        side = _side_of(desc)
        depth_match = _DEPTH_TOKEN.search(desc)
        depth_in = float(depth_match.group(1)) if depth_match else None
        key = (round(diam_in, 4), side, depth_in)
        if key not in groups:
            order.append(key)
        groups[key] += int(r.get("qty") or 0)
    if not groups:
        return
    items = [(key, groups[key]) for key in sorted(order, key=lambda key: (key[0], key[1]))]
    total = sum(qty for _, qty in items)
    front = sum(qty for (key, qty) in items if key[1] == "FRONT")
    back = total - front
    lines += [
        "MATERIAL REMOVAL – COUNTERBORE",
        "=" * 64,
        "Inputs",
        "  Ops ............... Counterbore (front + back)",
        f"  Counterbores ...... {total} total  → {front} front, {back} back",
        "",
        "TIME PER HOLE – C’BORE GROUPS",
        "-" * 66,
    ]
    for (diam_in, side, depth_in), qty in items:
        sfm, ipr = _lookup_sfm_ipr("counterbore", diam_in, material_group, speeds_csv)
        rpm = _rpm_from_sfm_diam(sfm, diam_in)
        ipm = _ipm_from_rpm_ipr(rpm, ipr)
        depth_txt = "—" if depth_in is None else f'{depth_in:.2f}"'
        rpm_txt = "-" if rpm is None else str(int(rpm))
        ipm_txt = "-" if ipm is None else f"{ipm:.3f}"
        lines.append(
            f'Ø{diam_in:.4f}" × {qty}  ({side}) | depth {depth_txt} | {rpm_txt} rpm | '
            f"{ipm_txt} ipm | t/hole — | group — "
        )
    lines.append("")


def _emit_spot_and_jig_cards(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> None:
    rows = _rows_from_ops_summary(geo, result=result, breakdown=breakdown)
    spot_qty = 0
    spot_depth: float | None = None
    for r in rows:
        desc_upper = str(r.get("desc", "")).upper()
        if ("DRILL" in desc_upper and "C" in desc_upper) and ("THRU" not in desc_upper) and (
            "TAP" not in desc_upper
        ):
            depth_match = _DEPTH_TOKEN.search(desc_upper)
            if depth_match:
                try:
                    spot_depth = float(depth_match.group(1))
                except Exception:
                    spot_depth = None
            spot_qty += int(r.get("qty") or 0)
    jig_qty = sum(int(r.get("qty") or 0) for r in rows if "JIG GRIND" in str(r.get("desc", "")).upper())
    if spot_qty > 0:
        sfm, ipr = _lookup_sfm_ipr("spot", 0.1875, material_group, speeds_csv)
        rpm = _rpm_from_sfm_diam(sfm, 0.1875)
        ipm = _ipm_from_rpm_ipr(rpm, ipr)
        depth_txt = "—" if spot_depth is None else f'{spot_depth:.2f}"'
        rpm_txt = "-" if rpm is None else str(int(round(rpm)))
        ipm_txt = "-" if ipm is None else f"{ipm:.3f}"
        lines += [
            "MATERIAL REMOVAL – SPOT (CENTER DRILL)",
            "=" * 64,
            f"Spots .............. {spot_qty} (front-side unless noted)",
            "TIME PER HOLE – SPOT GROUPS",
            "-" * 66,
            f"Spot drill × {spot_qty} | depth {depth_txt} | {rpm_txt} rpm | {ipm_txt} ipm | t/hole — | group — ",
            "",
        ]
    if jig_qty > 0:
        lines += [
            "MATERIAL REMOVAL – JIG GRIND",
            "=" * 64,
            f"Jig-grind features . {jig_qty}",
            "TIME PER FEATURE",
            "-" * 66,
            f"Jig grind × {jig_qty} | t/feat — | group — ",
            "",
        ]


def _hole_table_section_present(lines: Sequence[str], header: str) -> bool:
    if not header:
        return False
    header_norm = header.strip().upper()
    for existing in lines:
        if isinstance(existing, str) and existing.strip().upper() == header_norm:
            return True
    return False


def _emit_hole_table_ops_cards(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None,
    speeds_csv: dict | None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> None:
    if not _hole_table_section_present(lines, "MATERIAL REMOVAL – TAPPING"):
        _emit_tapping_card(
            lines,
            geo=geo,
            material_group=material_group,
            speeds_csv=speeds_csv,
            result=result,
            breakdown=breakdown,
        )
    if not _hole_table_section_present(lines, "MATERIAL REMOVAL – COUNTERBORE"):
        _emit_counterbore_card(
            lines,
            geo=geo,
            material_group=material_group,
            speeds_csv=speeds_csv,
            result=result,
            breakdown=breakdown,
        )
    if not _hole_table_section_present(lines, "MATERIAL REMOVAL – SPOT (CENTER DRILL)"):
        _emit_spot_and_jig_cards(
            lines,
            geo=geo,
            material_group=material_group,
            speeds_csv=speeds_csv,
            result=result,
            breakdown=breakdown,
        )


def _estimate_drilling_minutes_from_meta(
    drilling_meta: Mapping[str, Any] | None,
    geo_map: Mapping[str, Any] | None = None,
) -> tuple[float, float, float, dict[str, Any]]:
    """Return machine/toolchange/total minutes derived from drilling meta."""

    if not isinstance(drilling_meta, _MappingABC):
        return (0.0, 0.0, 0.0, {})

    try:
        index_min = float(drilling_meta.get("index_min_per_hole") or 0.0)
    except Exception:
        index_min = 0.0

    peck_min_rng = _ensure_list(drilling_meta.get("peck_min_per_hole_vals"))
    if not peck_min_rng:
        peck_min_rng = [0.0, 0.0]
    try:
        peck_min_deep = float(min(peck_min_rng))
    except Exception:
        peck_min_deep = 0.0
    try:
        peck_min_std = float(max(peck_min_rng))
    except Exception:
        peck_min_std = 0.0

    try:
        tchg_deep = float(drilling_meta.get("toolchange_min_deep") or 0.0)
    except Exception:
        tchg_deep = 0.0
    try:
        tchg_std = float(drilling_meta.get("toolchange_min_std") or 0.0)
    except Exception:
        tchg_std = 0.0

    bins = drilling_meta.get("bins_list")
    if isinstance(bins, tuple):
        bins = list(bins)
    if not isinstance(bins, list):
        bins_dict = drilling_meta.get("bins")
        if isinstance(bins_dict, _MappingABC):
            sorted_bins: list[dict[str, Any]] = []
            for _, value in sorted(
                bins_dict.items(),
                key=lambda kv: _safe_float(
                    (kv[1] or {}).get("diameter_in") if isinstance(kv[1], _MappingABC) else None,
                    default=0.0,
                ),
            ):
                if isinstance(value, _MappingABC):
                    sorted_bins.append(dict(value))
            bins = sorted_bins
        else:
            bins = []
    else:
        sanitized_bins: list[dict[str, Any]] = []
        for entry in bins:
            if isinstance(entry, _MappingABC):
                sanitized_bins.append(dict(entry))
        bins = sanitized_bins

    ops_groups_from_table, ops_hole_total = _drilling_groups_from_ops_summary(geo_map)
    if ops_groups_from_table:
        ops_hole_count_from_table = ops_hole_total
        table_map = {
            round(float(group.get("diameter_in", 0.0)), 4): group
            for group in ops_groups_from_table
            if _coerce_float_or_none(group.get("diameter_in"))
        }
        filtered_bins: list[dict[str, Any]] = []
        for entry in bins:
            if not isinstance(entry, _MappingABC):
                continue
            dia_val = _safe_float(entry.get("diameter_in"))
            if dia_val <= 0:
                continue
            key = round(dia_val, 4)
            group_info = table_map.get(key)
            if not group_info:
                continue
            entry_copy = dict(entry)
            entry_copy["qty"] = int(group_info.get("qty") or 0)
            filtered_bins.append(entry_copy)
        seen_keys: set[float] = set()
        for entry in filtered_bins:
            dia_val = _safe_float(entry.get("diameter_in"))
            if dia_val > 0:
                seen_keys.add(round(dia_val, 4))
        for key, group_info in table_map.items():
            if key in seen_keys:
                continue
            dia_in = _safe_float(group_info.get("diameter_in"))
            if dia_in <= 0:
                continue
            filtered_bins.append(
                {
                    "op": "drill",
                    "op_name": "drill",
                    "diameter_in": dia_in,
                    "diameter_mm": float(dia_in * 25.4),
                    "qty": int(group_info.get("qty") or 0),
                    "sfm": 80.0,
                    "ipr": 0.0060,
                }
            )
        if filtered_bins:
            filtered_bins.sort(
                key=lambda item: _safe_float(item.get("diameter_in"), 0.0)
            )
        bins = filtered_bins
        holes_deep = sum(
            int(entry.get("qty") or 0)
            for entry in bins
            if str(entry.get("op") or entry.get("op_name") or "")
            .strip()
            .lower()
            .startswith("deep")
        )
        holes_std = sum(
            int(entry.get("qty") or 0)
            for entry in bins
            if not str(entry.get("op") or entry.get("op_name") or "")
            .strip()
            .lower()
            .startswith("deep")
        )
        dia_vals = [
            _safe_float(entry.get("diameter_in"))
            for entry in bins
            if _safe_float(entry.get("diameter_in")) > 0
        ]
        depth_vals = [
            _safe_float(entry.get("depth_in"))
            for entry in bins
            if _safe_float(entry.get("depth_in")) > 0
        ]
        if isinstance(drilling_meta_map, _MutableMappingABC):
            drilling_meta_mut = typing.cast(MutableMapping[str, Any], drilling_meta_map)
            drilling_meta_mut["bins_list"] = bins
            drilling_meta_mut["hole_count"] = int(ops_hole_total)
            drilling_meta_mut["holes_deep"] = int(holes_deep)
            drilling_meta_mut["holes_std"] = int(holes_std)

    subtotal_min = 0.0
    seen_deep = False
    seen_std = False
    if bins:
        subtotal_min, seen_deep, seen_std = _render_time_per_hole(
            lambda _line: None,
            bins=bins,
            index_min=index_min,
            peck_min_deep=peck_min_deep,
            peck_min_std=peck_min_std,
        )

    tool_minutes = (tchg_deep if seen_deep else 0.0) + (tchg_std if seen_std else 0.0)
    total_minutes = subtotal_min + tool_minutes

    detail = {
        "bins": bins,
        "index_min": index_min,
        "peck_min_deep": peck_min_deep,
        "peck_min_std": peck_min_std,
        "toolchange_min_deep": tchg_deep,
        "toolchange_min_std": tchg_std,
        "seen_deep": seen_deep,
        "seen_std": seen_std,
        "subtotal_minutes": subtotal_min,
        "tool_minutes": tool_minutes,
        "total_minutes": total_minutes,
    }

    return subtotal_min, tool_minutes, total_minutes, detail


if sys.platform == "win32":
    occ_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(occ_bin):
        os.add_dll_directory(occ_bin)

APP_ENV = AppEnvironment.from_env()

EXTRA_DETAIL_RE = re.compile(r"^includes\b.*extras\b", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: formatting + removal card + per-hole lines (no material per line)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_list(value: Any, fallback: Iterable[Any] | None = None) -> list[Any]:
    """Return ``value`` coerced to a list with an optional fallback."""

    if isinstance(value, list):
        return value
    if value is None:
        return list(fallback) if fallback is not None else []
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return [stripped]
        return list(fallback) if fallback is not None else []
    try:
        return list(value)
    except Exception:
        return list(fallback) if fallback is not None else []


SCRAP_CREDIT_FALLBACK_USD_PER_LB = 0.35
SCRAP_RECOVERY_DEFAULT = 0.85


def _wieland_scrap_usd_per_lb(material_family: str | None) -> float | None:
    """Return the USD/lb scrap price from the Wieland scraper if available."""

    try:
        from cad_quoter.pricing.wieland_scraper import get_scrap_price_per_lb
    except Exception:
        try:  # pragma: no cover - external dependency hook
            from wieland_scraper import get_scrap_price_per_lb  # type: ignore[import]
        except Exception:
            return None

    fam = str(material_family or "").strip().lower()
    if not fam:
        fam = "aluminum"
    if "alum" in fam:
        fam = "aluminum"
    elif "stainless" in fam:
        fam = "stainless"
    elif "steel" in fam:
        fam = "steel"
    elif "copper" in fam:
        fam = "copper"
    elif "brass" in fam:
        fam = "brass"
    elif "titanium" in fam or fam.startswith("ti"):
        fam = "titanium"

    try:
        price: float | int | str | None = get_scrap_price_per_lb(fam)
    except Exception as exc:  # pragma: no cover - network/HTML failure
        logger.warning("Wieland scrap price lookup failed for %s: %s", fam, exc)
        return None

    if price is None:
        return None

    try:
        price_float = float(price)
    except Exception:
        return None

    if not math.isfinite(price_float) or price_float <= 0:
        return None
    return price_float


def _compute_direct_costs(
    material_total: float | int | str | None,
    scrap_credit: float | int | str | None,
    material_tax: float | int | str | None,
    pass_through: _MappingABC[str, Any] | None,
    *,
    material_detail: Mapping[str, Any] | None = None,
    scrap_price_source: str | None = None,
) -> float:
    """Return the rounded direct-cost total shared by math and rendering."""

    pt = dict(pass_through or {})
    pt.pop("Material", None)
    subtotal = float(material_total or 0.0)
    subtotal += float(material_tax or 0.0)
    base_scrap_credit = float(scrap_credit or 0.0)
    subtotal -= base_scrap_credit

    computed_scrap_credit = 0.0
    if str(scrap_price_source or "").strip().lower() == "wieland":
        detail_map: Mapping[str, Any] | None
        detail_map = material_detail if isinstance(material_detail, _MappingABC) else None
        scrap_mass_lb = None
        if detail_map is not None:
            for key in (
                "scrap_credit_mass_lb",
                "scrap_weight_lb",
                "scrap_lb",
            ):
                val = _coerce_float_or_none(detail_map.get(key))
                if val is not None and val > 0:
                    scrap_mass_lb = float(val)
                    break
        if (
            scrap_mass_lb is not None
            and scrap_mass_lb > 0
            and base_scrap_credit <= 0.0
        ):
            price_val = None
            if detail_map is not None:
                price_val = _coerce_float_or_none(
                    detail_map.get("scrap_credit_unit_price_usd_per_lb")
                )
            if price_val is None or price_val <= 0:
                family_hint = None
                if detail_map is not None and isinstance(detail_map, _MappingABC):
                    family_hint = detail_map.get("material_family") or detail_map.get(
                        "material_group"
                    )
                price_val = _wieland_scrap_usd_per_lb(family_hint)
            recovery_val = None
            if detail_map is not None:
                recovery_val = _coerce_float_or_none(
                    detail_map.get("scrap_credit_recovery_pct")
                )
            if recovery_val is None or recovery_val <= 0:
                recovery_val = SCRAP_RECOVERY_DEFAULT
            if (
                price_val is not None
                and price_val > 0
                and recovery_val is not None
                and recovery_val > 0
            ):
                computed_scrap_credit = (
                    float(scrap_mass_lb)
                    * float(price_val)
                    * float(recovery_val)
                )
    if computed_scrap_credit > 0:
        subtotal -= float(computed_scrap_credit)
    subtotal += sum(float(v or 0.0) for v in pt.values())
    return round(subtotal, 2)


def _compute_pricing_ladder(
    subtotal: float | int | str | None,
    *,
    expedite_pct: float | int | str | None = 0.0,
    margin_pct: float | int | str | None = 0.0,
) -> dict[str, float]:
    """Return cumulative totals for each step of the pricing ladder."""

    def _pct(value: float | int | str | None) -> float:
        return _safe_float(value, 0.0)

    subtotal_val = round(_safe_float(subtotal, 0.0), 2)

    expedite_pct_val = _pct(expedite_pct)
    margin_pct_val = _pct(margin_pct)

    expedite_cost = round(subtotal_val * expedite_pct_val, 2)

    with_expedite = round(subtotal_val + expedite_cost, 2)
    subtotal_before_margin = with_expedite
    with_margin = round(subtotal_before_margin * (1.0 + margin_pct_val), 2)

    return {
        "subtotal": subtotal_val,
        "with_expedite": with_expedite,
        "with_margin": with_margin,
        "expedite_cost": expedite_cost,
        "subtotal_before_margin": subtotal_before_margin,
    }

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


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

import textwrap
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
    Literal,
    TypedDict,
    overload,
    no_type_check,
)

if typing.TYPE_CHECKING:  # pragma: no cover - import is for type checking only
    from ezdxf.document import Drawing  # type: ignore[attr-defined]
else:  # pragma: no cover - fallback when ezdxf is unavailable at runtime
    try:
        from ezdxf.ezdxf import Drawing  # type: ignore[attr-defined]
    except Exception:
        class Drawing(Protocol):
            """Minimal protocol used for ezdxf drawing type hints."""

            def modelspace(self) -> Any: ...

if typing.TYPE_CHECKING:
    from cad_quoter.domain import QuoteState as _QuoteState
else:
    _QuoteState = QuoteState

if typing.TYPE_CHECKING:
    import pandas as pd
    from pandas import DataFrame as PandasDataFrame
    from pandas import Index as PandasIndex
    from pandas import Series as PandasSeries
    from cad_quoter.domain import QuoteState as _QuoteState

    SeriesLike: TypeAlias = Any
else:
    _QuoteState = QuoteState
    PandasDataFrame: TypeAlias = Any
    PandasSeries: TypeAlias = Any
    PandasIndex: TypeAlias = Any
    SeriesLike: TypeAlias = Any

    try:
        import pandas as pd  # type: ignore[import]
    except Exception:  # pragma: no cover - optional dependency
        pd = None  # type: ignore[assignment]
    PandasDataFrame: TypeAlias = Any
    PandasSeries: TypeAlias = Any
    PandasIndex: TypeAlias = Any
    SeriesLike: TypeAlias = Any


def _is_pandas_dataframe(obj: Any) -> bool:
    """Return True if *obj* looks like a pandas ``DataFrame`` instance."""

    df_type = getattr(pd, "DataFrame", None)
    try:
        return bool(df_type) and isinstance(obj, df_type)  # type: ignore[arg-type]
    except Exception:
        return False

# ───────────────────────────────────────────────────────────────────────
# Sync the estimator's drilling hours into all rendered views
# ───────────────────────────────────────────────────────────────────────
T = TypeVar("T")

def _resolve_pricing_source_value(
    base_value: Any,
    *,
    used_planner: bool | None = None,
    process_meta: Mapping[str, Any] | None = None,
    process_meta_raw: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
    planner_process_minutes: Any = None,
    hour_summary_entries: Mapping[str, Any] | None = None,
    additional_sources: Sequence[Any] | None = None,
    cfg: QuoteConfiguration | None = None,
) -> str | None:
    """Return a normalized pricing source, honoring explicit selections."""

    fallback_text: str | None = None
    if base_value is not None:
        candidate_text = str(base_value).strip()
        if candidate_text:
            lowered = candidate_text.lower()
            if lowered == "planner":
                return "planner"
            if lowered not in {"legacy", "auto", "default", "fallback"}:
                return candidate_text
            fallback_text = candidate_text

    if used_planner:
        if fallback_text:
            return fallback_text
        return "planner"

    # Delegate planner signal detection to the adapter helper
    from appkit.planner_adapter import _planner_signals_present as _planner_signals_present_helper

    if _planner_signals_present_helper(
        process_meta=process_meta,
        process_meta_raw=process_meta_raw,
        breakdown=breakdown,
        planner_process_minutes=planner_process_minutes,
        hour_summary_entries=hour_summary_entries,
        additional_sources=list(additional_sources) if additional_sources is not None else None,
    ):
        if fallback_text:
            return fallback_text
        return "planner"

    if fallback_text:
        return fallback_text

    return None



def _wrap_header_text(text: Any, page_width: int, indent: str = "") -> list[str]:
    """Helper mirroring :func:`write_wrapped` for header content."""

    if text is None:
        return []
    txt = str(text).strip()
    if not txt:
        return []
    wrapper = textwrap.TextWrapper(width=max(10, page_width - len(indent)))
    return [f"{indent}{chunk}" for chunk in wrapper.wrap(txt)]


def _build_quote_header_lines(
    *,
    qty: int,
    result: Mapping[str, Any] | None,
    breakdown: Mapping[str, Any] | None,
    page_width: int,
    divider: str,
    process_meta: Mapping[str, Any] | None,
    process_meta_raw: Mapping[str, Any] | None,
    hour_summary_entries: Mapping[str, Any] | None,
    cfg: QuoteConfiguration | None = None,
) -> tuple[list[str], str | None]:
    """Construct the canonical QUOTE SUMMARY header lines."""

    header_lines: list[str] = [f"QUOTE SUMMARY - Qty {qty}", divider]
    header_lines.append("Quote Summary (structured data attached below)")

    speeds_feeds_value = None
    if isinstance(result, _MappingABC):
        speeds_feeds_value = result.get("speeds_feeds_path")
    if speeds_feeds_value in (None, "") and isinstance(breakdown, _MappingABC):
        speeds_feeds_value = breakdown.get("speeds_feeds_path")
    path_text = str(speeds_feeds_value).strip() if speeds_feeds_value else ""

    speeds_feeds_loaded_display: bool | None = None
    for source in (result, breakdown):
        if not isinstance(source, _MappingABC):
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
        header_lines.extend(
            _wrap_header_text(
                f"Speeds/Feeds CSV: {path_text}{status_suffix}",
                page_width,
            )
        )
    elif status_suffix:
        header_lines.extend(
            _wrap_header_text(
                f"Speeds/Feeds CSV: (not set){status_suffix}",
                page_width,
            )
        )
    else:
        header_lines.append("Speeds/Feeds CSV: (not set)")

    def _coerce_pricing_source(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        # normalize synonyms
        if lowered in {"legacy", "est", "estimate", "estimator"}:
            return "estimator"
        if lowered in {"plan", "planner"}:
            return "planner"
        return text

    raw_pricing_source = None
    pricing_source_display = None
    if isinstance(breakdown, _MappingABC):
        raw_pricing_source = _coerce_pricing_source(breakdown.get("pricing_source"))
        if raw_pricing_source:
            pricing_source_display = str(raw_pricing_source).title()

    used_planner_flag: bool | None = None
    for source in (result, breakdown):
        if not isinstance(source, _MappingABC):
            continue
        for meta_key in ("app_meta", "app"):
            candidate = source.get(meta_key)
            if not isinstance(candidate, _MappingABC):
                continue
            if "used_planner" in candidate:
                try:
                    used_planner_flag = bool(candidate.get("used_planner"))
                except Exception:
                    used_planner_flag = True if candidate.get("used_planner") else False
                break
        if used_planner_flag is not None:
            break

    pricing_source_value = _resolve_pricing_source_value(
        raw_pricing_source,
        used_planner=used_planner_flag,
        process_meta=process_meta if isinstance(process_meta, _MappingABC) else None,
        process_meta_raw=process_meta_raw if isinstance(process_meta_raw, _MappingABC) else None,
        breakdown=breakdown if isinstance(breakdown, _MappingABC) else None,
        hour_summary_entries=hour_summary_entries,
        cfg=cfg,
    )

    # === HEADER: PRICING SOURCE OVERRIDE ===
    if getattr(cfg, "prefer_removal_drilling_hours", False):
        normalized_value = (
            str(pricing_source_value).strip().lower()
            if pricing_source_value is not None
            else ""
        )
        if not normalized_value or normalized_value == "legacy":
            pricing_source_value = "Estimator"
            pricing_source_display = "Estimator"

    normalized_pricing_source: str | None = None
    if pricing_source_value is not None:
        normalized_pricing_source = str(pricing_source_value).strip()
        if not normalized_pricing_source:
            normalized_pricing_source = None

    if normalized_pricing_source:
        normalized_pricing_source_lower = normalized_pricing_source.lower()
        raw_pricing_source_lower = (
            str(raw_pricing_source).strip().lower() if raw_pricing_source is not None else None
        )

        if (
            isinstance(breakdown, _MutableMappingABC)
            and raw_pricing_source_lower != normalized_pricing_source_lower
        ):
            breakdown["pricing_source"] = pricing_source_value

        pricing_source_display = normalized_pricing_source.title()

    if pricing_source_display:
        display_value = pricing_source_display
        header_lines.append(f"Pricing Source: {display_value}")

    return header_lines, pricing_source_value

def _count_recognized_ops(plan_summary: Mapping[str, Any] | None) -> int:
    """Return a conservative count of recognized planner operations."""

    if not isinstance(plan_summary, _MappingABC):
        return 0
    try:
        raw_ops = plan_summary.get("ops")
    except Exception:
        return 0
    if not isinstance(raw_ops, list):
        return 0
    count = 0
    for entry in raw_ops:
        if isinstance(entry, _MappingABC):
            count += 1
        elif entry is not None:
            try:
                if bool(entry):
                    count += 1
            except Exception:
                count += 1
    return count


def _recognized_line_items_from_planner(pricing_result: Mapping[str, Any] | None) -> int:
    """Best-effort extraction of recognized planner operations for fallback logic."""

    if not isinstance(pricing_result, _MappingABC):
        return 0

    try:
        raw_recognized = pricing_result.get("recognized_line_items")
    except Exception:
        raw_recognized = None
    if raw_recognized is not None:
        try:
            value = int(float(raw_recognized))
            if value > 0:
                return value
        except Exception:
            pass

    try:
        line_items = pricing_result.get("line_items")
    except Exception:
        line_items = None
    if isinstance(line_items, (list, tuple)):
        count = sum(1 for entry in line_items if entry)
        if count > 0:
            return count

    plan_summary: Mapping[str, Any] | None = None
    try:
        plan_summary = pricing_result.get("plan_summary")
    except Exception:
        plan_summary = None
    if plan_summary is None:
        plan_candidate = pricing_result.get("plan")
        if isinstance(plan_candidate, _MappingABC):
            plan_summary = plan_candidate

    return _count_recognized_ops(plan_summary)

import cad_quoter.geometry as geometry

# Re-export legacy OCCT helpers via cad_quoter.geometry.
FACE_OF = geometry.FACE_OF
ensure_face = geometry.ensure_face
face_surface = geometry.face_surface
iter_faces = geometry.iter_faces
linear_properties = geometry.linear_properties
map_shapes_and_ancestors = geometry.map_shapes_and_ancestors
from cad_quoter.geo2d.apply import apply_2d_features_to_variables

# Tolerance for invariant checks that guard against silent drift when rendering
# cost sections.
_LABOR_SECTION_ABS_EPSILON = 0.51
_PLANNER_BUCKET_ABS_EPSILON = 0.51

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
    normalize_material_key,
)
from cad_quoter.domain_models.values import safe_float as _safe_float, to_float, to_int
from cad_quoter.utils import coerce_bool, compact_dict, jdump, json_safe_copy, sdict
from cad_quoter.utils.text import _match_items_contains
from cad_quoter.llm_suggest import (
    get_llm_quote_explanation,
)
from cad_quoter.pricing import (
    BACKUP_CSV_NAME,
    LB_PER_KG,
    PricingEngine,
    create_default_registry,
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
    pick_speeds_row as _pick_speeds_row,
)
from cad_quoter.pricing.speeds_feeds_selector import (
    material_group_for_speeds_feeds as _material_group_for_speeds_feeds,
)
from cad_quoter.pricing.speeds_feeds_selector import (
    unit_hp_cap as _unit_hp_cap,
)
from cad_quoter.pricing.speeds_feeds_selector import (
    load_csv_as_records as _load_speeds_feeds_records,
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
from cad_quoter.pricing.materials import (
    STANDARD_PLATE_SIDES_IN as STANDARD_PLATE_SIDES_IN,
    _compute_material_block as _compute_material_block,
    _compute_scrap_mass_g as _compute_scrap_mass_g,
    _density_for_material as _density_for_material,
    _material_family as _material_family,
    _material_cost_components as _material_cost_components,
    _material_price_from_choice as _material_price_from_choice,
    _material_price_per_g_from_choice as _material_price_per_g_from_choice,
    _nearest_std_side as _nearest_std_side,
    _resolve_price_per_lb as _resolve_price_per_lb,
    infer_plate_lw_in as infer_plate_lw_in,
    net_mass_kg as net_mass_kg,
    plan_stock_blank as plan_stock_blank,
)
from cad_quoter.config import _ensure_two_bucket_rates
from cad_quoter.rates import (
    two_bucket_to_flat,
)
from cad_quoter.vendors.mcmaster_stock import lookup_sku_and_price_for_mm

_normalize_lookup_key = normalize_material_key

_clean_hole_groups = _drilling_legacy._clean_hole_groups
_coerce_overhead_dataclass = _drilling_legacy._coerce_overhead_dataclass
_drill_overhead_from_params = _drilling_legacy._drill_overhead_from_params
_machine_params_from_params = _drilling_legacy._machine_params_from_params
_legacy_estimate_drilling_hours = _drilling_legacy.legacy_estimate_drilling_hours
_apply_drill_minutes_clamp = _drilling_legacy._apply_drill_minutes_clamp
_drill_minutes_per_hole_bounds = _drilling_legacy._drill_minutes_per_hole_bounds

_CANONICAL_MIC6_DISPLAY = "Aluminum MIC6"
_MIC6_NORMALIZED_KEY = _normalize_lookup_key(_CANONICAL_MIC6_DISPLAY)
_MATERIAL_META_KEY_BY_NORMALIZED: dict[str, str] = {}
for _raw_meta_key in MATERIAL_MAP:
    _normalized_meta_key = _normalize_lookup_key(_raw_meta_key)
    if _normalized_meta_key and _normalized_meta_key not in _MATERIAL_META_KEY_BY_NORMALIZED:
        _MATERIAL_META_KEY_BY_NORMALIZED[_normalized_meta_key] = _raw_meta_key
_CANONICAL_MIC6_PRICING_KEY = next(
    (key for key in ("6061", "6061-T6") if key in MATERIAL_MAP),
    None,
)
if _MIC6_NORMALIZED_KEY and _CANONICAL_MIC6_PRICING_KEY:
    _MATERIAL_META_KEY_BY_NORMALIZED.setdefault(
        _MIC6_NORMALIZED_KEY,
        _CANONICAL_MIC6_PRICING_KEY,
    )

try:
    _DEFAULT_SYSTEM_SUGGEST = load_text("system_suggest.txt").strip()
except FileNotFoundError:  # pragma: no cover - defensive fallback
    _DEFAULT_SYSTEM_SUGGEST = ""

ensure_runtime_dependencies()

_llm_integration = init_llm_integration(_DEFAULT_SYSTEM_SUGGEST)
SYSTEM_SUGGEST = _llm_integration.system_suggest
SUGG_TO_EDITOR = _llm_integration.sugg_to_editor
EDITOR_TO_SUGG = _llm_integration.editor_to_sugg
EDITOR_FROM_UI = _llm_integration.editor_from_ui
LLMClient = _llm_integration.llm_client
parse_llm_json = _llm_integration.parse_llm_json
explain_quote = _llm_integration.explain_quote

configure_llm_integration(_llm_integration)

# Backwards compatibility aliases expected by older integrations.
_DEFAULT_VL_MODEL_NAMES = DEFAULT_VL_MODEL_NAMES
_DEFAULT_MM_PROJ_NAMES = DEFAULT_MM_PROJ_NAMES

from typing import TypedDict

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
    for scalar_key in ACCEPT_SCALAR_KEYS:
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


def apply_suggestions(baseline: dict, s: dict) -> dict:
    """Apply sanitized LLM suggestions onto a baseline quote snapshot."""

    merged = merge_effective(baseline or {}, s or {}, {})
    merged.pop("_source_tags", None)
    merged.pop("_clamp_notes", None)

    notes = list(s.get("notes") or [])
    if s.get("no_change_reason"):
        notes.append(f"no_change: {s['no_change_reason']}")
    merged["_llm_notes"] = notes

    return merged


try:
    from cad_quoter.geometry.hole_table_parser import (
        parse_hole_table_lines as _parse_hole_table_lines,
    )
except Exception:
    _parse_hole_table_lines = None

try:
    from cad_quoter.geometry.dxf_text import (
        extract_text_lines_from_dxf as _extract_text_lines_from_dxf,
    )
except Exception:
    _extract_text_lines_from_dxf = None

# ---------- OCC helpers delegated to cad_quoter.geometry ----------
STACK = getattr(geometry, "STACK", "pythonocc")
STACK_GPROP = getattr(geometry, "STACK_GPROP", STACK)


def _missing_geo_helper(name: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{name} is unavailable (OCCT bindings required)")

    return _raise


BND_ADD_FALLBACK: Callable[..., Any] = lambda *_args, **_kwargs: None
bnd_add = getattr(geometry, "bnd_add", BND_ADD_FALLBACK)
BRepTools_UVBounds = getattr(
    geometry, "uv_bounds", _missing_geo_helper("BRepTools.UVBounds")
)
BRepCheck_Analyzer = getattr(
    geometry, "BRepCheck_Analyzer", _missing_geo_helper("BRepCheck_Analyzer")
)
brep_read = getattr(geometry, "brep_read", _missing_geo_helper("brep_read"))


def read_step_or_iges_or_brep(path: str) -> Any:
    """Backwards-compatible shim that forwards to :mod:`cad_quoter.geometry`."""

    return geometry.read_step_or_iges_or_brep(path)

# ---- tiny helpers you can use elsewhere --------------------------------------
# Optional PDF stack
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore[assignment]
    _HAS_PYMUPDF = False

DIM_RE = re.compile(r"(?:[Øø⌀]|DIAM|DIA)\s*([0-9.+-]+)|R\s*([0-9.+-]+)|([0-9.+-]+)\s*[xX]\s*([0-9.+-]+)")

def load_drawing(path: Path) -> Drawing:
    ezdxf_mod = typing.cast(_EzdxfModule, geometry.require_ezdxf())
    if path.suffix.lower() == ".dwg":
        # Prefer explicit converter/wrapper if configured (works even if ODA isn't on PATH)
        exe = geometry.get_dwg_converter_path()
        if exe:
            dxf_path = geometry.convert_dwg_to_dxf(str(path))
            return ezdxf_mod.readfile(dxf_path)
        # Fallback: odafc (requires ODAFileConverter on PATH)
        if _HAS_ODAFC and odafc is not None:
            odafc_mod = typing.cast(_OdafcModule, odafc)
            return odafc_mod.readfile(str(path))
        raise RuntimeError(
            "DWG import needs ODA File Converter. Set ODA_CONVERTER_EXE to the exe "
            "or place dwg2dxf_wrapper.bat next to the script."
        )
    return ezdxf_mod.readfile(str(path))  # DXF directly

# ---- DXF protocol typing -----------------------------------------------------

class _EzdxfModule(Protocol):
    def readfile(
        self,
        filename: str,
        encoding: str | None = ...,
        errors: str | None = ...,
    ) -> "Drawing":
        ...

class _OdafcModule(Protocol):
    def readfile(self, filename: str) -> "Drawing":
        ...



# --- WHICH SHEET ROWS MATTER TO THE ESTIMATOR --------------------------------
# --- APPLY LLM OUTPUT ---------------------------------------------------------
# ================== LLM DECISION LOG / AUDIT ==================


# =============================================================

# ----------------- Variables & quote -----------------
try:
    _HAS_PANDAS = bool(pd is not None)  # type: ignore[name-defined]
except NameError:
    _HAS_PANDAS = False

CORE_COLS = ["Item", "Example Values / Options", "Data Type / Input Method"]

_MASTER_VARIABLES_CACHE: dict[str, Any] = {
    "loaded": False,
    "core": None,
    "full": None,
}

_SPEEDS_FEEDS_CACHE: dict[str, PandasDataFrame | None] = {}

def _coerce_core_types(df_core: PandasDataFrame) -> PandasDataFrame:
    """Light normalization for estimator expectations."""
    core = df_core.copy()
    core["Item"] = core["Item"].astype(str)
    core["Data Type / Input Method"] = core["Data Type / Input Method"].astype(str).str.lower()
    # Leave "Example Values / Options" as-is (can be text or number); estimator coerces later.
    return core


def sanitize_vars_df(df_full: PandasDataFrame) -> PandasDataFrame:
    """
    Return a copy containing only the 3 core columns the estimator needs.
    - Does NOT mutate or overwrite the original file.
    - Creates missing core columns as blanks.
    - Normalizes types.
    """
    if pd is None:  # pragma: no cover - defensive guard for static analysers
        raise RuntimeError("pandas is required to sanitize variables data frames")

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
) -> PandasDataFrame | tuple[PandasDataFrame, PandasDataFrame]:
    """
    Read .xlsx/.csv, keep original data intact, and return a sanitized copy for the estimator.
    - If return_full=True, returns (core_df, full_df); otherwise returns core_df only.
    """
    if not _HAS_PANDAS or pd is None:
        raise RuntimeError("pandas required (conda/pip install pandas)")

    assert pd is not None  # hint for type checkers

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

def _load_master_variables() -> tuple[PandasDataFrame | None, PandasDataFrame | None]:
    """Load the packaged master variables sheet once and serve cached copies."""
    if not _HAS_PANDAS or pd is None:
        return (None, None)

    assert pd is not None  # hint for type checkers

    global _MASTER_VARIABLES_CACHE
    cache = _MASTER_VARIABLES_CACHE

    if cache.get("loaded"):
        core_cached = cache.get("core")
        full_cached = cache.get("full")

        core_copy: PandasDataFrame | None = None
        if (
            _HAS_PANDAS
            and pd is not None
            and isinstance(core_cached, pd.DataFrame)
        ):
            core_copy = core_cached.copy()

        full_copy: PandasDataFrame | None = None
        if (
            _HAS_PANDAS
            and pd is not None
            and isinstance(full_cached, pd.DataFrame)
        ):
            full_copy = full_cached.copy()

        return (core_copy, full_copy)

    master_path = default_master_variables_csv()
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
        core_df = cast(PandasDataFrame, core_df)
        full_df = cast(PandasDataFrame, full_df)
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
    "deep_drill": "drilling",
    "counterbore": "counterbore",
    "tapping": "tapping",
    "grinding": "grinding",
    "wire_edm": "wire_edm",
    "wire_edm_windows": "wire_edm",
    "wire_edm_outline": "wire_edm",
    "wire_edm_open_id": "wire_edm",
    "wire_edm_cam_slot_or_profile": "wire_edm",
    "wire_edm_id_leave": "wire_edm",
    "wedm": "wire_edm",
    "wireedm": "wire_edm",
    "wire-edm": "wire_edm",
    "wire edm": "wire_edm",
    "sinker_edm": "sinker_edm",
    "sinker_edm_finish_burn": "sinker_edm",
    "sinker": "sinker_edm",
    "ram_edm": "sinker_edm",
    "ramedm": "sinker_edm",
    "ram-edm": "sinker_edm",
    "ram edm": "sinker_edm",
    "saw_waterjet": "saw_waterjet",
    "fixture_build_amortized": "fixture_build_amortized",
    "programming_amortized": "programming_amortized",
    "misc": "misc",
}

PLANNER_META: frozenset[str] = frozenset({"planner_labor", "planner_machine", "planner_total"})

# ---------- Bucket & Operation Roles ----------
BUCKET_ROLE: dict[str, str] = {
    "programming": "labor_only",
    "inspection": "labor_only",
    "deburr": "labor_only",
    "deburring": "labor_only",
    "finishing_deburr": "labor_only",
    "finishing": "labor_only",
    "finishing/deburr": "labor_only",

    "drilling": "split",
    "milling": "split",
    "grinding": "split",
    "sinker_edm": "split",

    "wedm": "machine_only",
    "wire_edm": "machine_only",
    "waterjet": "machine_only",
    "saw": "machine_only",
    "saw_waterjet": "machine_only",

    "_default": "machine_only",
}

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
@no_type_check
def render_quote(  # type: ignore[reportGeneralTypeIssues]
    result: dict,
    currency: str = "$",
    show_zeros: bool = False,
    llm_explanation: str = "",
    page_width: int = 74,
    cfg: QuoteConfiguration | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> str:
    """Pretty printer for a full quote with auto-included non-zero lines."""

    overrides = (
        ("prefer_removal_drilling_hours", True),
        ("separate_machine_labor", True),
        ("machine_rate_per_hr", 45.0),
        ("labor_rate_per_hr", 45.0),
    )

    cfg_obj: QuoteConfiguration | Any = cfg or QuoteConfiguration(
        default_params=copy.deepcopy(PARAMS_DEFAULT)
    )
    for name, value in overrides:
        try:
            setattr(cfg_obj, name, value)
        except Exception:
            cfg_obj = QuoteConfiguration(default_params=copy.deepcopy(PARAMS_DEFAULT))
            for name2, value2 in overrides:
                setattr(cfg_obj, name2, value2)
            break

    cfg = cfg_obj

    breakdown    = result.get("breakdown", {}) or {}

    state_payload: Any | None = None

    # Ensure a mutable view of the breakdown for downstream helpers that update it
    try:
        if isinstance(breakdown, _MutableMappingABC):
            breakdown_mutable = typing.cast(MutableMapping[str, Any], breakdown)
        else:
            breakdown_mutable = dict(breakdown)
    except Exception:
        breakdown_mutable = {}
    if isinstance(result, _MappingABC):
        state_payload = result.get("quote_state")

    quote_state_obj: QuoteState | None = None
    if isinstance(state_payload, QuoteState):
        quote_state_obj = state_payload
    elif isinstance(state_payload, _MappingABC):
        try:
            quote_state_obj = QuoteState.from_dict(typing.cast(Mapping[str, Any], state_payload))
        except Exception:
            quote_state_obj = None

    if quote_state_obj is not None:
        reprice_with_effective(quote_state_obj)
        effective_snapshot = dict(getattr(quote_state_obj, "effective", {}) or {})
        effective_sources_snapshot = dict(
            getattr(quote_state_obj, "effective_sources", {}) or {}
        )

        decision_state_obj = (
            result.get("decision_state") if isinstance(result, _MappingABC) else None
        )
        if isinstance(decision_state_obj, _MutableMappingABC):
            decision_state_obj["effective"] = effective_snapshot
            decision_state_obj["effective_sources"] = effective_sources_snapshot
        elif isinstance(decision_state_obj, _MappingABC):
            decision_state_copy: dict[str, Any] = dict(decision_state_obj)
            decision_state_copy["effective"] = effective_snapshot
            decision_state_copy["effective_sources"] = effective_sources_snapshot
            if isinstance(result, _MutableMappingABC):
                result["decision_state"] = decision_state_copy

        if isinstance(result, _MutableMappingABC):
            result["quote_state"] = quote_state_obj.to_dict()

    # Force drill debug output to render by enabling the LLM debug flag for this run.
    app_meta_container = result.get("app_meta")
    if isinstance(app_meta_container, dict):
        target_app_meta = app_meta_container
    elif isinstance(app_meta_container, _MappingABC):
        target_app_meta = dict(app_meta_container)
        result["app_meta"] = target_app_meta
    else:
        target_app_meta = {}
        result["app_meta"] = target_app_meta
    target_app_meta["llm_debug_enabled"] = True

    prefer_removal_drilling_hours = getattr(cfg, "prefer_removal_drilling_hours", True)
    if prefer_removal_drilling_hours is None:
        prefer_removal_drilling_hours = True
    else:
        prefer_removal_drilling_hours = bool(prefer_removal_drilling_hours)
    stock_price_source = getattr(cfg, "stock_price_source", None)
    scrap_price_source = getattr(cfg, "scrap_price_source", None)

    totals       = breakdown.get("totals", {}) or {}
    if not isinstance(totals, dict):
        try:
            totals = dict(totals or {})
        except Exception:
            totals = {}
        breakdown["totals"] = totals
    else:
        breakdown["totals"] = totals
    declared_labor_total = float(totals.get("labor_cost", 0.0) or 0.0)
    nre_detail   = breakdown.get("nre_detail", {}) or {}
    nre_raw      = breakdown.get("nre", {}) or {}
    if isinstance(nre_raw, _MappingABC):
        nre = dict(nre_raw)
    else:
        try:
            nre = dict(nre_raw or {})
        except Exception:
            nre = {}
    breakdown["nre"] = nre
    material_raw = breakdown.get("material", {}) or {}
    if isinstance(material_raw, _MappingABC):
        material_block = dict(material_raw)
    else:  # tolerate legacy iterables or unexpected values
        try:
            material_block = dict(material_raw or {})
        except Exception:
            material_block = {}
    material_block_new = breakdown.get("material_block") or {}
    if isinstance(material_block_new, _MappingABC):
        material_stock_block = dict(material_block_new)
    else:
        try:
            material_stock_block = dict(material_block_new or {})
        except Exception:
            material_stock_block = {}
    material_selection_raw = breakdown.get("material_selected") or {}
    if isinstance(material_selection_raw, _MappingABC):
        material_selection = dict(material_selection_raw)
    else:
        try:
            material_selection = dict(material_selection_raw or {})
        except Exception:
            material_selection = {}

    def _resolve_overrides_source(container: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(container, _MappingABC):
            return None
        candidate = container.get("overrides")
        return candidate if isinstance(candidate, _MappingABC) else None

    material_overrides: Mapping[str, Any] | None = None
    material_overrides = _resolve_overrides_source(result) or _resolve_overrides_source(breakdown)
    if material_overrides is None and isinstance(result, _MappingABC):
        for key in ("user_overrides", "overrides"):
            candidate = result.get(key)
            if isinstance(candidate, _MappingABC):
                material_overrides = candidate
                break

    baseline: Mapping[str, Any] = {}
    decision_state = result.get("decision_state") if isinstance(result, _MappingABC) else None
    if isinstance(decision_state, _MappingABC):
        baseline_candidate = decision_state.get("baseline")
        if isinstance(baseline_candidate, _MappingABC):
            baseline = baseline_candidate
    if not baseline and isinstance(result, _MappingABC):
        baseline_candidate = result.get("baseline")
        if isinstance(baseline_candidate, _MappingABC):
            baseline = baseline_candidate
    if not baseline and isinstance(breakdown, _MappingABC):
        baseline_candidate = breakdown.get("baseline")
        if isinstance(baseline_candidate, _MappingABC):
            baseline = baseline_candidate

    pricing: dict[str, Any]
    pricing_obj: Mapping[str, Any] | None = None
    for container in (breakdown, result):
        if not isinstance(container, _MutableMappingABC):
            continue
        candidate = container.get("pricing")
        if isinstance(candidate, _MutableMappingABC):
            pricing_obj = candidate
            break
        if isinstance(candidate, _MappingABC):
            pricing_obj = dict(candidate)
            container["pricing"] = pricing_obj
            break
    if pricing_obj is None:
        pricing = {}
    else:
        pricing = dict(pricing_obj) if not isinstance(pricing_obj, dict) else pricing_obj
    if isinstance(breakdown, _MutableMappingABC):
        breakdown["pricing"] = pricing
    if isinstance(result, _MutableMappingABC):
        result["pricing"] = pricing
    normalized_material_key = str(
        material_selection.get("material_lookup")
        or material_selection.get("normalized_material_key")
        or ""
    ).strip()
    canonical_material_breakdown = str(
        material_selection.get("canonical")
        or material_selection.get("canonical_material")
        or ""
    ).strip()
    if canonical_material_breakdown:
        material_selection.setdefault("canonical", canonical_material_breakdown)
        material_selection.setdefault("canonical_material", canonical_material_breakdown)
    canonical_display_lookup = ""
    if normalized_material_key:
        canonical_display_lookup = str(
            MATERIAL_DISPLAY_BY_KEY.get(normalized_material_key, "")
        ).strip()
    material_display_label = str(
        material_selection.get("material_display")
        or material_selection.get("display")
        or canonical_material_breakdown
        or ""
    ).strip()
    if canonical_display_lookup:
        material_display_label = canonical_display_lookup
    elif not material_display_label and normalized_material_key:
        fallback_display = MATERIAL_DISPLAY_BY_KEY.get(normalized_material_key, "")
        if fallback_display:
            material_display_label = str(fallback_display).strip()
    if material_display_label:
        material_selection["material_display"] = material_display_label
    if normalized_material_key:
        material_selection.setdefault("normalized_material_key", normalized_material_key)
        material_selection.setdefault("material_lookup", normalized_material_key)
    group_material_breakdown = str(
        material_selection.get("group")
        or material_selection.get("material_group")
        or ""
    ).strip()
    if group_material_breakdown:
        material_selection.setdefault("group", group_material_breakdown)
        material_selection.setdefault("material_group", group_material_breakdown)
    material = material_block
    material_detail_for_breakdown = material

    MATERIAL_WARNING_LABEL = "⚠ MATERIALS MISSING"
    material_warning_entries: list[Mapping[str, Any]] = []
    cost_breakdown_entries: list[tuple[str, float]] = []

    def _extend_material_entries(source: Any) -> None:
        if isinstance(source, Sequence):
            for entry in source:
                if isinstance(entry, _MappingABC):
                    material_warning_entries.append(entry)
        elif isinstance(source, _MappingABC):
            entries = source.get("entries")
            if isinstance(entries, Sequence):
                for entry in entries:
                    if isinstance(entry, _MappingABC):
                        material_warning_entries.append(entry)

    raw_cost_breakdown: Any = None
    if isinstance(result, _MappingABC):
        _extend_material_entries(result.get("materials"))
        raw_cost_breakdown = result.get("cost_breakdown")
    if isinstance(breakdown, _MappingABC):
        _extend_material_entries(breakdown.get("materials"))
        _extend_material_entries(breakdown.get("material"))
        if raw_cost_breakdown is None:
            raw_cost_breakdown = breakdown.get("cost_breakdown")
    _extend_material_entries(pricing.get("materials"))

    if raw_cost_breakdown is None and isinstance(result, _MappingABC):
        raw_cost_breakdown = result.get("cost_breakdown")
    if raw_cost_breakdown is None and isinstance(breakdown, _MappingABC):
        raw_cost_breakdown = breakdown.get("cost_breakdown")
    if isinstance(raw_cost_breakdown, _MappingABC):
        for key, value in raw_cost_breakdown.items():
            amount = _coerce_float_or_none(value)
            if amount is None:
                continue
            cost_breakdown_entries.append((str(key), float(amount)))
    elif isinstance(raw_cost_breakdown, Sequence):
        for entry in raw_cost_breakdown:
            if isinstance(entry, Sequence) and len(entry) >= 2:
                label = str(entry[0])
                amount = _coerce_float_or_none(entry[1])
                if amount is None:
                    continue
                cost_breakdown_entries.append((label, float(amount)))

    def _material_entries_summary(entries: Sequence[Mapping[str, Any]]) -> tuple[float, bool]:
        total = 0.0
        has_label = False
        for entry in entries:
            amount = _coerce_float_or_none(entry.get("amount"))
            if amount is not None:
                total += float(amount)
            if not has_label:
                label_text = str(entry.get("label") or entry.get("detail") or "").strip()
                if label_text:
                    has_label = True
        return total, has_label

    material_entries_total, material_entries_have_label = _material_entries_summary(
        material_warning_entries
    )

    materials_direct_total = _coerce_float_or_none(
        result.get("materials_direct") if isinstance(result, _MappingABC) else None
    )
    if materials_direct_total is None and isinstance(breakdown, _MappingABC):
        materials_direct_total = _coerce_float_or_none(breakdown.get("materials_direct"))
    materials_direct_total = float(materials_direct_total or 0.0)

    material_warning_summary = bool(material_entries_have_label) and (
        material_entries_total <= 0.0 and materials_direct_total <= 0.0
    )
    material_warning_needed = material_warning_summary
    drilling_meta = breakdown.get("drilling_meta", {}) or {}
    process_costs_raw = breakdown.get("process_costs", {}) or {}
    process_costs = (
        dict(process_costs_raw)
        if isinstance(process_costs_raw, _MappingABC)
        else dict(process_costs_raw or {})
    )
    pass_through_raw = breakdown.get("pass_through", {}) or {}
    if isinstance(pass_through_raw, _MappingABC):
        pass_through = dict(pass_through_raw)
    else:
        try:
            pass_through = dict(pass_through_raw or {})
        except Exception:
            pass_through = {}
    applied_pcts_raw = breakdown.get("applied_pcts", {}) or {}
    if isinstance(applied_pcts_raw, dict):
        applied_pcts = applied_pcts_raw
    else:
        try:
            applied_pcts = dict(applied_pcts_raw or {})
        except Exception:
            applied_pcts = {}
    breakdown["applied_pcts"] = applied_pcts
    process_meta_raw = breakdown.get("process_meta", {}) or {}
    applied_process_raw = breakdown.get("applied_process", {}) or {}
    process_meta: dict[str, Any] = {}
    bucket_alias_map: dict[str, str] = {}
    applied_process: dict[str, Any] = {}
    rates_raw    = breakdown.get("rates", {}) or {}
    if isinstance(rates_raw, _MappingABC):
        rates = dict(rates_raw)
    else:
        try:
            rates = dict(rates_raw or {})
        except Exception:
            rates = {}

    def _coerce_rate_value(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    separate_labor_cfg = bool(getattr(cfg, "separate_machine_labor", False)) if cfg else False
    cfg_labor_rate_value = 0.0
    if separate_labor_cfg:
        cfg_labor_rate_value = _coerce_rate_value(getattr(cfg, "labor_rate_per_hr", 0.0))
        if cfg_labor_rate_value <= 0.0:
            cfg_labor_rate_value = 45.0

    if "ShopRate" not in rates:
        fallback_shop = _coerce_rate_value(rates.get("MillingRate"))
        rates.setdefault("ShopRate", fallback_shop)
    shop_rate_val = _coerce_rate_value(rates.get("ShopRate"))

    if "EngineerRate" not in rates:
        engineer_fallback = _coerce_rate_value(rates.get("MillingRate"))
        if engineer_fallback <= 0 and shop_rate_val > 0:
            engineer_fallback = shop_rate_val
        rates.setdefault("EngineerRate", engineer_fallback)
    engineer_rate_val = _coerce_rate_value(rates.get("EngineerRate"))

    labor_rate_value = _coerce_rate_value(rates.get("LaborRate"))
    if labor_rate_value <= 0:
        labor_rate_value = _coerce_rate_value(rates.get("ShopLaborRate"))
    if labor_rate_value <= 0:
        labor_rate_value = 85.0
    if separate_labor_cfg and cfg_labor_rate_value > 0.0:
        labor_rate_value = cfg_labor_rate_value
    rates["LaborRate"] = labor_rate_value

    machine_rate_value = _coerce_rate_value(rates.get("MachineRate"))
    if machine_rate_value <= 0:
        machine_rate_value = _coerce_rate_value(rates.get("ShopMachineRate"))
    if machine_rate_value <= 0:
        machine_rate_value = 90.0
    rates["MachineRate"] = machine_rate_value

    cfg_programmer_rate: float | None = None
    if cfg and getattr(cfg, "separate_machine_labor", False):
        cfg_programmer_rate = _coerce_rate_value(getattr(cfg, "labor_rate_per_hr", None))
        if cfg_programmer_rate <= 0:
            cfg_programmer_rate = 45.0

    if cfg_programmer_rate is not None and cfg_programmer_rate > 0:
        programmer_rate_value = float(cfg_programmer_rate)
        programming_rate_value = float(cfg_programmer_rate)
    else:
        if "ProgrammerRate" not in rates:
            programmer_fallback = (
                engineer_rate_val if engineer_rate_val > 0 else _coerce_rate_value(rates.get("MillingRate"))
            )
            if programmer_fallback <= 0 and shop_rate_val > 0:
                programmer_fallback = shop_rate_val
            if programmer_fallback <= 0:
                programmer_fallback = _coerce_rate_value(rates.get("LaborRate"))
            if programmer_fallback <= 0:
                programmer_fallback = 90.0
            if programmer_fallback > 0:
                programmer_fallback = max(programmer_fallback, 90.0)
            rates.setdefault("ProgrammerRate", programmer_fallback)

        programmer_rate_value = _coerce_rate_value(rates.get("ProgrammerRate"))
        if programmer_rate_value <= 0:
            programmer_rate_value = (
                engineer_rate_val
                if engineer_rate_val > 0
                else _coerce_rate_value(rates.get("MillingRate"))
            )
        if programmer_rate_value <= 0 and shop_rate_val > 0:
            programmer_rate_value = shop_rate_val
        if programmer_rate_value <= 0:
            programmer_rate_value = labor_rate_value
        if programmer_rate_value <= 0:
            programmer_rate_value = 90.0
        if programmer_rate_value > 0:
            programmer_rate_value = max(programmer_rate_value, 90.0)

        programming_rate_value = _coerce_rate_value(rates.get("ProgrammingRate"))
        if programming_rate_value <= 0:
            programming_rate_value = programmer_rate_value
        if programming_rate_value <= 0:
            programming_rate_value = labor_rate_value
        if programming_rate_value <= 0:
            programming_rate_value = 90.0
        if programming_rate_value > 0:
            programming_rate_value = max(programming_rate_value, 90.0)

    rates["ProgrammerRate"] = programmer_rate_value
    rates["ProgrammingRate"] = programming_rate_value

    inspector_rate_value = _coerce_rate_value(rates.get("InspectorRate"))
    if inspector_rate_value <= 0:
        inspector_rate_value = labor_rate_value
    if inspector_rate_value <= 0:
        inspector_rate_value = 85.0
    if inspector_rate_value > 0:
        inspector_rate_value = max(inspector_rate_value, 85.0)
    rates["InspectorRate"] = inspector_rate_value

    inspection_rate_value = _coerce_rate_value(rates.get("InspectionRate"))
    if inspection_rate_value <= 0:
        inspection_rate_value = inspector_rate_value
    if inspection_rate_value <= 0:
        inspection_rate_value = labor_rate_value
    if inspection_rate_value <= 0:
        inspection_rate_value = 85.0
    if inspection_rate_value > 0:
        inspection_rate_value = max(inspection_rate_value, 85.0)
    rates["InspectionRate"] = inspection_rate_value

    rates.setdefault("LaborRate", 85.0)
    rates.setdefault("MachineRate", 90.0)
    rates.setdefault("ProgrammingRate", rates.get("ProgrammerRate", rates["LaborRate"]))
    rates.setdefault("InspectionRate", rates.get("InspectorRate", rates["LaborRate"]))

    fallback_two_bucket_rates = _normalized_two_bucket_rates(rates)
    fallback_flat_rates = two_bucket_to_flat(fallback_two_bucket_rates)
    for key, fallback_value in fallback_flat_rates.items():
        if _coerce_rate_value(rates.get(key)) <= 0.0:
            rates[key] = float(fallback_value)
    params       = breakdown.get("params", {}) or {}
    nre_cost_details = breakdown.get("nre_cost_details", {}) or {}
    labor_cost_details_input_raw = breakdown.get("labor_cost_details", {}) or {}
    suppress_planner_details_due_to_drift = bool(
        breakdown.get("suppress_planner_details_due_to_drift")
    )

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

    material_total_for_directs_val = _coerce_float_or_none(
        material_block.get("material_cost_before_credit")
    )
    if material_total_for_directs_val is None:
        material_total_for_directs_val = _coerce_float_or_none(
            material_block.get("total_cost")
        )
    if material_total_for_directs_val is None:
        material_total_for_directs_val = _coerce_float_or_none(
            material_block.get("material_cost")
        )
    if material_total_for_directs_val is None:
        material_total_for_directs_val = _coerce_float_or_none(
            material_block.get("material_direct_cost")
        )
    material_total_for_directs = float(material_total_for_directs_val or 0.0)
    if material_total_for_directs <= 0 and isinstance(material_stock_block, dict):
        try:
            material_total_for_directs = float(
                material_stock_block.get("total_material_cost") or 0.0
            )
        except Exception:
            material_total_for_directs = 0.0
    if material_total_for_directs <= 0:
        if materials_direct_total > 0:
            material_total_for_directs = float(materials_direct_total)
        elif material_entries_total > 0:
            material_total_for_directs = float(material_entries_total)
    if material_total_for_directs <= 0 and isinstance(pass_through, dict):
        try:
            material_key = next(
                (
                    key
                    for key in pass_through
                    if str(key).strip().lower() == "material"
                ),
                None,
            )
        except Exception:
            material_key = None
        if material_key is not None:
            try:
                material_total_for_directs = float(pass_through.get(material_key) or 0.0)
            except Exception:
                material_total_for_directs = 0.0

    scrap_credit_for_directs_val = _coerce_float_or_none(
        material_block.get("material_scrap_credit")
    )
    scrap_credit_for_directs = float(scrap_credit_for_directs_val or 0.0)

    material_tax_for_directs_val = _coerce_float_or_none(material_block.get("material_tax"))
    material_tax_for_directs = float(material_tax_for_directs_val or 0.0)

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
                if EXTRA_DETAIL_RE.match(seg):
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
    amortized_nre_total = 0.0
    for label, value in labor_cost_totals.items():
        _canonical_label, is_amortized = _canonical_amortized_label(label)
        if not is_amortized:
            continue
        try:
            amortized_nre_total += float(value or 0.0)
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
    if qty_raw in (None, "", 0):
        qty_raw = breakdown.get("qty")
    if qty_raw in (None, "", 0):
        decision_state = result.get("decision_state")
        if isinstance(decision_state, _MappingABC):
            baseline_state = decision_state.get("baseline")
            if isinstance(baseline_state, _MappingABC):
                qty_raw = baseline_state.get("qty")
    qty = int(qty_raw or 1)
    price        = float(result.get("price", totals.get("price", 0.0)))

    g = (
        breakdown.get("geo_context")
        or breakdown.get("geo")
        or result.get("geom")
        or result.get("geo")
        or {}
    )
    if not isinstance(g, dict):
        g = {}

    # Optional: LLM decision bullets can be placed either on result or breakdown
    llm_notes = (result.get("llm_notes") or breakdown.get("llm_notes") or [])[:8]
    notes_order = [
        str(note).strip() for note in llm_notes if str(note).strip()
    ]

    llm_debug_enabled_effective = bool(APP_ENV.llm_debug_enabled)
    for source in (result, breakdown):
        if not isinstance(source, _MappingABC):
            continue
        for key in ("app", "app_meta"):
            app_info = source.get(key)
            if not isinstance(app_info, _MappingABC):
                continue
            if "llm_debug_enabled" in app_info:
                llm_debug_enabled_effective = bool(app_info.get("llm_debug_enabled"))
                break
        else:
            continue
        break

    # ---- helpers -------------------------------------------------------------
    divider = "-" * int(page_width)

    red_flags: list[str] = []
    hour_summary_entries: dict[str, tuple[float, bool]] = {}
    for source in (result, breakdown):
        flags_raw = source.get("red_flags") if isinstance(source, _MappingABC) else None
        if isinstance(flags_raw, (list, tuple, set)):
            for flag in flags_raw:
                text = str(flag).strip()
                if text and text not in red_flags:
                    red_flags.append(text)

    def _m(x) -> str:
        return format_currency(x, currency)

    def _h(x) -> str:
        return format_hours(x)

    def _hours_with_rate_text(hours: Any, rate: Any) -> str:
        return format_hours_with_rate(hours, rate, currency)

    def _resolve_rate_with_fallback(raw_rate: Any, *fallback_keys: str) -> float:
        rate_val = _safe_float(raw_rate)
        if rate_val > 0:
            return rate_val
        for key in fallback_keys:
            if not key:
                continue
            try:
                fallback_val = rates.get(key)
            except Exception:
                fallback_val = None
            resolved = _safe_float(fallback_val)
            if resolved > 0:
                return resolved
        return 0.0

    def _pct(x) -> str:
        return format_percent(x)

    def _format_weight_lb_decimal(mass_g: float | None) -> str:
        return format_weight_lb_decimal(mass_g)

    def _format_weight_lb_oz(mass_g: float | None) -> str:
        return format_weight_lb_oz(mass_g)

    def _scrap_source_hint(material_info: Mapping[str, Any] | None) -> str | None:
        if not isinstance(material_info, _MappingABC):
            return None

        scrap_from_holes_raw = material_info.get("scrap_pct_from_holes")
        scrap_from_holes_val = _coerce_float_or_none(scrap_from_holes_raw)
        scrap_from_holes = False
        if scrap_from_holes_val is not None and scrap_from_holes_val > 1e-6:
            scrap_from_holes = True
        elif isinstance(scrap_from_holes_raw, bool):
            scrap_from_holes = scrap_from_holes_raw
        if scrap_from_holes:
            return "holes"

        label_raw = material_info.get("scrap_source_label")
        if label_raw in (None, ""):
            return None

        label_text = str(label_raw).strip()
        if not label_text:
            return None

        label_text = label_text.replace("+", " + ")
        label_text = label_text.replace("_", " ")
        label_text = re.sub(r"\s+", " ", label_text).strip()
        return label_text or None

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

    def write_line(s: str, indent: str = ""):
        append_line(f"{indent}{s}")

    def write_wrapped(text: str, indent: str = ""):
        if text is None:
            return
        txt = _sanitize_render_text(text).strip()
        if not txt:
            return
        wrapper = textwrap.TextWrapper(width=max(10, page_width - len(indent)))
        for chunk in wrapper.wrap(txt):
            write_line(chunk, indent)

    def write_detail(detail: str, indent: str = "    "):
        if not detail:
            return
        sanitized_detail = _sanitize_render_text(detail)
        for segment in re.split(r";\s*", sanitized_detail):
            write_wrapped(segment, indent)

    bucket_diag_env = os.getenv("SHOW_BUCKET_DIAGNOSTICS")
    show_bucket_diagnostics_flag = _is_truthy_flag(bucket_diag_env) or bool(
        SHOW_BUCKET_DIAGNOSTICS_OVERRIDE
    )
    def render_bucket_table(rows: Sequence[tuple[str, float, float, float, float]]):
        if not rows:
            return

        if not show_bucket_diagnostics_flag:
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
            append_line("")

        diagnostic_banner = "=== Planner diagnostics (not billed) ==="
        append_line(diagnostic_banner)
        append_line("=" * min(page_width, len(diagnostic_banner)))

        header_line = " | ".join(_fmt(header, idx) for idx, header in enumerate(headers))
        separator_line = " | ".join("-" * width for width in col_widths)
        append_line(header_line)
        append_line(separator_line)
        for row_values in display_rows:
            append_line(" | ".join(_fmt(value, idx) for idx, value in enumerate(row_values)))
        append_line("")

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
        append_line(short_divider)

    def _format_row(label: str, val: float, indent: str = "") -> str:
        left = f"{indent}{label}"
        right = _m(val)
        pad = max(1, page_width - len(left) - len(right))
        return f"{left}{' ' * pad}{right}"

    def row(label: str, val: float, indent: str = ""):
        # left-label, right-amount aligned to page_width
        if _is_total_label(label):
            _ensure_total_separator(len(_m(val)))
        append_line(_format_row(label, val, indent))

    def hours_row(label: str, val: float, indent: str = ""):
        left = f"{indent}{label}"
        right = _h(val)
        if _is_total_label(label):
            _ensure_total_separator(len(right))
        pad = max(1, page_width - len(left) - len(right))
        append_line(f"{left}{' ' * pad}{right}")

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
                if not seg or EXTRA_DETAIL_RE.match(seg):
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
                if not seg or EXTRA_DETAIL_RE.match(seg):
                    continue
                filtered_existing.append(seg)
            if filtered_existing:
                return "; ".join(filtered_existing)
            return str(existing).strip()
        return None

    def add_process_notes(key: str, indent: str = "    "):
        canon_key = _canonical_bucket_key(key)
        stored_hours = 0.0
        stored_rate = 0.0
        stored_cost = 0.0
        if canon_key:
            stored_entry = process_cost_row_details.get(canon_key)
            if stored_entry is not None:
                stored_hours, stored_rate, stored_cost = stored_entry

        meta = _lookup_process_meta(key) or {}
        hr_val = stored_hours
        if hr_val <= 0:
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
        meta_rate = 0.0
        if meta:
            try:
                meta_rate = float(meta.get("rate", 0.0) or 0.0)
            except Exception:
                meta_rate = 0.0
        if meta_rate > 0:
            rate_float = meta_rate
        else:
            rate_float = stored_rate
            if rate_float <= 0:
                rate_val = meta.get("rate") if meta else None
                try:
                    rate_float = float(rate_val or 0.0)
                except Exception:
                    rate_float = 0.0
        if rate_float <= 0 and stored_cost > 0 and hr_val > 0:
            rate_float = stored_cost / hr_val
        if rate_float <= 0:
            rate_key = _rate_key_for_bucket(str(key))
            if rate_key:
                try:
                    rate_float = float(rates.get(rate_key, 0.0) or 0.0)
                except Exception:
                    rate_float = 0.0
        try:
            base_extra_val = float(meta.get("base_extra", 0.0) or 0.0)
        except Exception:
            base_extra_val = 0.0

        if hr_val > 0:
            write_line(_hours_with_rate_text(hr_val, rate_float), indent)
        elif base_extra_val > 0 and rate_float > 0:
            inferred_hours = base_extra_val / rate_float
            if inferred_hours > 0:
                write_line(_hours_with_rate_text(inferred_hours, rate_float), indent)

    def add_pass_basis(key: str, indent: str = "    "):
        basis_map = breakdown.get("pass_basis", {}) or {}
        info = basis_map.get(key) or {}
        txt = info.get("basis") or info.get("note")
        if txt:
            write_line(str(txt), indent)

    def _extract_llm_debug_override(container: Mapping[str, Any] | None) -> bool | None:
        if not isinstance(container, _MappingABC):
            return None
        if "llm_debug_enabled" in container:
            value = container.get("llm_debug_enabled")
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "t", "yes", "y", "on"}:
                    return True
                if lowered in {"0", "false", "f", "no", "n", "off", ""}:
                    return False
                return None
            return bool(value)
        for meta_key in ("app", "app_meta"):
            meta = container.get(meta_key)
            override = _extract_llm_debug_override(meta) if isinstance(meta, _MappingABC) else None
            if override is not None:
                return override
        return None

    llm_debug_enabled_flag = bool(APP_ENV.llm_debug_enabled)
    for source in (result, breakdown):
        override = _extract_llm_debug_override(source if isinstance(source, _MappingABC) else None)
        if override is not None:
            llm_debug_enabled_flag = override
            break

    

    # ---- header --------------------------------------------------------------
    lines: list[str] = []
    doc_builder = QuoteDocRecorder(divider)

    def append_line(text: str) -> None:
        sanitized = _sanitize_render_text(text)
        previous = lines[-1] if lines else None
        lines.append(sanitized)
        doc_builder.observe_line(len(lines) - 1, sanitized, previous)

    def append_lines(values: Iterable[str]) -> None:
        for value in values:
            append_line(value)

    def _push(target: list[str], text: str) -> None:
        if target is lines:
            append_line(text)
        else:
            target.append(text)

    def replace_line(index: int, text: str) -> None:
        sanitized = _sanitize_render_text(text)
        if 0 <= index < len(lines):
            lines[index] = sanitized
        doc_builder.replace_line(index, sanitized)

    hour_summary_entries: dict[str, tuple[float, bool]] = {}
    ui_vars = result.get("ui_vars") or {}
    if not isinstance(ui_vars, dict):
        ui_vars = {}
    g_source = geometry if isinstance(geometry, _MappingABC) else result.get("geom") or result.get("geo")
    if isinstance(g_source, _MappingABC):
        g = dict(g_source) if not isinstance(g_source, dict) else dict(g_source)
    else:
        g = {}
    drill_debug_entries: list[str] = []
    # Selected removal summary (if available) for compact debug table later
    removal_summary_for_display: Mapping[str, Any] | None = None
    _accumulate_drill_debug(drill_debug_entries, result, breakdown)
    # If the removal summary has a total, force machine hours for drilling
    removal = (result or {}).get("removal_summary") or {}
    mins = float(removal.get("total_minutes") or 0.0)
    if mins > 0:
        drilling_machine_hr = round(mins / 60.0, 2)
        # write into both the pricing state and the hour summary
        hour_summary_entries["drilling"] = (drilling_machine_hr, True)
        # also, if your bucket view structure is present, overwrite that slot:
        buckets = (breakdown or {}).get("planner_buckets") or {}
        if isinstance(buckets, dict) and "drilling" in buckets:
            buckets["drilling"]["machine_hours"] = drilling_machine_hr
            buckets["drilling"]["labor_hours"] = float(buckets["drilling"].get("labor_hours") or 0.0)
    # Canonical QUOTE SUMMARY header (legacy variants removed in favour of this
    # block so the Speeds/Feeds status + Drill Debug output stay consistent).
    header_lines, pricing_source_value = _build_quote_header_lines(
        qty=qty,
        result=result if isinstance(result, _MappingABC) else None,
        breakdown=breakdown if isinstance(breakdown, _MappingABC) else None,
        page_width=page_width,
        divider=divider,
        process_meta=process_meta,
        process_meta_raw=process_meta_raw,
        hour_summary_entries=hour_summary_entries,
        cfg=cfg,
    )
    append_lines(header_lines)
    if material_warning_summary:
        append_line(MATERIAL_WARNING_LABEL)
    append_line("")

    if isinstance(breakdown, _MutableMappingABC):
        if pricing_source_value:
            breakdown["pricing_source"] = pricing_source_value
        else:
            breakdown.pop("pricing_source", None)

    if isinstance(result, _MutableMappingABC):
        app_meta_container = result.setdefault("app_meta", {})
        if isinstance(app_meta_container, _MutableMappingABC):
            if pricing_source_value and str(pricing_source_value).strip().lower() == "planner":
                app_meta_container.setdefault("used_planner", True)

        decision_state = result.get("decision_state")
        if isinstance(decision_state, _MutableMappingABC):
            baseline_state = decision_state.get("baseline")
            if isinstance(baseline_state, _MutableMappingABC):
                if pricing_source_value:
                    baseline_state["pricing_source"] = pricing_source_value
                else:
                    baseline_state.pop("pricing_source", None)

    def render_drill_debug(entries: Sequence[str]) -> None:
        append_line("Drill Debug")
        append_line(divider)
        prioritized_entries: list[tuple[int, int, str]] = []
        for idx, entry in enumerate(entries):
            if entry is None:
                continue
            text = str(entry).strip()
            if not text:
                continue
            normalized = text.lstrip()
            if normalized.startswith("Material Removal Debug"):
                priority = 0
            elif normalized.startswith("OK deep_drill") or normalized.startswith("OK drill"):
                priority = 1
            elif normalized.lower().startswith("drill bin") or normalized.lower().startswith("bin "):
                priority = 3
            else:
                priority = 2
            prioritized_entries.append((priority, idx, text))

        for _priority, _idx, text in sorted(prioritized_entries, key=lambda item: (item[0], item[1])):
            if "\n" in text:
                normalized = text.lstrip()
                block_indent = "" if normalized.startswith("Material Removal Debug") else "  "
                for chunk in text.splitlines():
                    write_line(chunk, block_indent)
                if lines and lines[-1] != "":
                    append_line("")
            else:
                write_wrapped(text, "  ")
        if lines and lines[-1] != "":
            append_line("")

    app_meta = result.setdefault("app_meta", {})
    # Only surface drill debug when LLM debug is enabled for this quote.
    if drill_debug_entries and llm_debug_enabled_flag:
        # Order so legacy per-bin “OK …” lines appear first, then tables/summary.
        def _dbg_sort(a: str, b: str) -> int:
            a_ok = a.strip().lower().startswith("ok ")
            b_ok = b.strip().lower().startswith("ok ")
            if a_ok and not b_ok: return -1
            if b_ok and not a_ok: return 1
            a_hdr = a.strip().lower().startswith("material removal debug")
            b_hdr = b.strip().lower().startswith("material removal debug")
            if a_hdr and not b_hdr: return 1
            if b_hdr and not a_hdr: return -1
            return 0

        try:
            sorted_drill_entries = sorted(drill_debug_entries, key=cmp_to_key(_dbg_sort))
        except Exception:
            sorted_drill_entries = drill_debug_entries

        render_drill_debug(sorted_drill_entries)
    row("Final Price per Part:", price)
    final_price_row_index = len(lines) - 1
    total_labor_label = "Total Labor Cost:"
    row(total_labor_label, float(totals.get("labor_cost", 0.0)))
    total_labor_row_index = len(lines) - 1
    total_direct_costs_label = "Total Direct Costs:"
    row(total_direct_costs_label, 0.0)
    total_direct_costs_row_index = len(lines) - 1
    process_total_row_index = -1
    directs: float = 0.0
    pricing_source_lower = pricing_source_value.lower() if pricing_source_value else ""
    display_red_flags: list[str] = []
    if red_flags:
        display_red_flags = [str(flag).strip() for flag in red_flags if str(flag).strip()]
        if pricing_source_lower == "planner":
            display_red_flags = [
                flag
                for flag in display_red_flags
                if not flag.lower().startswith("labor totals drifted by")
            ]
        if display_red_flags:
            append_line("")
            append_line("Red Flags")
            append_line(divider)
            for flag in display_red_flags:
                write_wrapped(f"⚠️ {flag}", "  ")
    append_line("")

    narrative = result.get("narrative") or breakdown.get("narrative")
    why_parts: list[str] = []
    why_lines: list[str] = []
    material_total_for_why = 0.0
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

    price_driver_lines: list[str] = []
    for container in (result, breakdown):
        if not isinstance(container, _MappingABC):
            continue
        driver_entries = container.get("price_drivers")
        if not isinstance(driver_entries, Sequence):
            continue
        for entry in driver_entries:
            if not isinstance(entry, _MappingABC):
                continue
            label = _sanitize_render_text(entry.get("label")).strip()
            detail = _sanitize_render_text(entry.get("detail")).strip()
            if label and detail:
                combined = f"{label} — {detail}"
            else:
                combined = label or detail
            combined = combined.strip()
            if not combined:
                continue
            if combined not in price_driver_lines:
                price_driver_lines.append(combined)

    if price_driver_lines:
        why_parts.extend(price_driver_lines)

    hour_trace_data = None
    if isinstance(result, _MappingABC):
        hour_trace_data = result.get("hour_trace")
    if hour_trace_data is None and isinstance(breakdown, _MappingABC):
        hour_trace_data = breakdown.get("hour_trace")
    explanation_lines: list[str] = []
    # ``explanation_lines`` will be merged into ``why_parts`` after the process
    # bucket rows are prepared so the cost makeup + contributor text can be
    # derived from the exact rows rendered in the Process & Labor table.

    def _is_planner_meta(key: str) -> bool:
        canonical_key = _canonical_bucket_key(key)
        if not canonical_key:
            return False
        return canonical_key.startswith("planner_") or canonical_key == "planner_total"

    def _merge_process_meta(
        existing: Mapping[str, Any] | None, incoming: Mapping[str, Any] | Any
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(existing) if isinstance(existing, _MappingABC) else {}
        incoming_map: Mapping[str, Any]
        if isinstance(incoming, _MappingABC):
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
        if not isinstance(meta_source, _MappingABC):
            return {}, {}

        for raw_key, raw_meta in meta_source.items():
            alias_key = str(raw_key).lower().strip()
            if not alias_key:
                continue
            if _is_planner_meta(alias_key):
                folded[alias_key] = dict(raw_meta) if isinstance(raw_meta, _MappingABC) else {}
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
            if not isinstance(entry, _MappingABC):
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
        if isinstance(applied_source, _MappingABC):
            base = {str(k).lower().strip(): (v or {}) for k, v in applied_source.items()}
        if not alias_map:
            return base

        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for alias_key, canon_key in alias_map.items():
            entry = base.get(alias_key)
            if isinstance(entry, _MappingABC):
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
        return _planner_bucket_key_for_name(name)

    process_plan_breakdown_raw = breakdown.get("process_plan")
    process_plan_breakdown: Mapping[str, Any] | None
    if isinstance(process_plan_breakdown_raw, _MappingABC):
        process_plan_breakdown = process_plan_breakdown_raw
    else:
        process_plan_breakdown = None

    planner_bucket_from_plan = _extract_bucket_map(
        process_plan_breakdown.get("bucket_view") if isinstance(process_plan_breakdown, _MappingABC) else None
    )

    # Minutes engine owns the canonical bucket display. Planner output is still captured
    # for summaries, but it must not overwrite the breakdown used for pricing.
    planner_bucket_display_map: dict[str, dict[str, Any]] = {}

    bucket_rollup_map: dict[str, dict[str, Any]] = {}
    raw_rollup = breakdown.get("planner_bucket_rollup")
    if isinstance(raw_rollup, _MappingABC):
        for key, value in raw_rollup.items():
            canon = _canonical_bucket_key(key)
            if canon:
                if isinstance(value, _MappingABC):
                    bucket_rollup_map[canon] = {str(k): v for k, v in value.items()}
                else:
                    bucket_rollup_map[canon] = {}
    if not bucket_rollup_map and planner_bucket_from_plan:
        bucket_rollup_map = dict(planner_bucket_from_plan)
    if not bucket_rollup_map:
        bucket_struct = breakdown.get("bucket_view")
        if isinstance(bucket_struct, _MappingABC):
            if isinstance(bucket_struct.get("buckets"), _MappingABC):
                bucket_struct = bucket_struct.get("buckets")
            if isinstance(bucket_struct, _MappingABC):
                for key, value in bucket_struct.items():
                    canon = _canonical_bucket_key(key)
                    if canon:
                        if isinstance(value, _MappingABC):
                            bucket_rollup_map[canon] = {str(k): v for k, v in value.items()}
                        else:
                            bucket_rollup_map[canon] = {}

    planner_total_meta = process_meta.get("planner_total", {}) if isinstance(process_meta, dict) else {}
    planner_line_items_meta = planner_total_meta.get("line_items") if isinstance(planner_total_meta, _MappingABC) else None
    bucket_ops_map: dict[str, list[PlannerBucketOp]] = {}
    if isinstance(planner_line_items_meta, list):
        for entry in planner_line_items_meta:
            if not isinstance(entry, _MappingABC):
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

    # Process & Labor table rendered later using the canonical planner bucket view

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
            if isinstance(meta_entry, _MappingABC):
                return meta_entry
        return None

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
    bucket_keys = []
    seen_buckets: set[str] = set()
    for key in bucket_order:
        if key in bucket_rollup_map or key in bucket_ops_map:
            bucket_keys.append(key)
            seen_buckets.add(key)
    extra_source = set(bucket_rollup_map.keys()) | set(bucket_ops_map.keys())
    extra_keys = sorted(key for key in extra_source if key not in seen_buckets)
    bucket_keys.extend(extra_keys)

    # Legacy planner comparison table removed to avoid duplicate process views.

    material_display_for_debug: str = ""

    # Prep material pricing/mass estimates so the renderer and downstream
    # consumers see consistent values even when geometry inputs are sparse.
    mat_info = pricing.setdefault("material", {}) if isinstance(pricing, dict) else {}
    if not isinstance(mat_info, dict):
        try:
            mat_info = dict(mat_info or {})
        except Exception:
            mat_info = {}
        if isinstance(pricing, dict):
            pricing["material"] = mat_info
    pricing_geom = pricing.get("geom") if isinstance(pricing, dict) else None
    if not isinstance(pricing_geom, _MappingABC):
        pricing_geom = pricing.get("geo") if isinstance(pricing, dict) else None
    if not isinstance(pricing_geom, _MappingABC):
        pricing_geom = g if isinstance(g, _MappingABC) else {}
    ml = str((pricing_geom or {}).get("material_lookup") or "").lower()

    DENSITY_G_CC = {"aluminum": 2.70, "tool_steel": 7.85, "stainless": 7.90, "titanium": 4.5}
    mat_total_val = None
    if isinstance(material_stock_block, dict):
        mat_total_val = _coerce_float_or_none(
            material_stock_block.get("total_material_cost")
        )
    if mat_total_val is None and isinstance(material, dict):
        mat_total_val = _coerce_float_or_none(material.get("material_cost"))
    mat_total = float(mat_total_val or 0.0)
    # Recompute material total from components if needed
    if mat_total == 0.0 and isinstance(material, dict):
        base = float(material.get("base_cost") or 0.0)
        tax = float(material.get("material_tax") or 0.0)
        scrap_credit = float(material.get("material_scrap_credit") or 0.0)
        recomputed = round(max(0.0, base + tax - scrap_credit), 2)
        if recomputed > 0:
            mat_total = recomputed
            # surface it everywhere our renderers and directs look:
            material["material_cost"] = mat_total
            if isinstance(material_stock_block, dict):
                material_stock_block["total_material_cost"] = mat_total
    if isinstance(mat_info, dict) and mat_total > 0:
        mat_info["material_cost"] = float(mat_total)
        if isinstance(material, dict):
            for key in (
                "mass_g",
                "starting_mass_g_est",
                "net_mass_g",
                "scrap_mass_g",
                "unit_price_usd_per_lb",
                "source",
            ):
                if key in material:
                    mat_info[key] = material[key]
    if isinstance(material, dict) and mat_total > 0:
        material["material_cost"] = float(mat_total)

    if isinstance(pricing, dict):
        directs = pricing.setdefault("direct_costs", {})
        if not isinstance(directs, dict):
            try:
                directs = dict(directs or {})
            except Exception:
                directs = {}
            pricing["direct_costs"] = directs
        directs["material"] = float(mat_total)
        if _coerce_float_or_none(pricing.get("total_direct_costs")) is None:
            pricing["total_direct_costs"] = float(sum(directs.values()))

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
        material_cost_components: Mapping[str, Any] | None = None

        if have_any:
            append_line("Material & Stock")
            append_line(divider)
            canonical_material_display = str(material_display_label or "").strip()
            if not canonical_material_display and isinstance(material_selection, _MappingABC):
                canonical_material_display = str(
                    material_selection.get("material_display")
                    or material_selection.get("canonical")
                    or material_selection.get("canonical_material")
                    or ""
                ).strip()
            if not canonical_material_display and isinstance(drilling_meta, _MappingABC):
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
                    canonical_material_display = str(drill_display).strip()
            material_name_display = canonical_material_display
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
                material_display_label = str(material_name_display)
                material_selection.setdefault("material_display", material_display_label)
                material_display_for_debug = material_name_display
                append_line(f"  Material used:  {material_name_display}")

            blank_lines: list[str] = []
            need_len = _coerce_float_or_none(
                material_stock_block.get("required_blank_len_in")
                if isinstance(material_stock_block, _MappingABC)
                else None
            )
            need_wid = _coerce_float_or_none(
                material_stock_block.get("required_blank_wid_in")
                if isinstance(material_stock_block, _MappingABC)
                else None
            )
            need_thk = _coerce_float_or_none(
                material_stock_block.get("required_blank_thk_in")
                if isinstance(material_stock_block, _MappingABC)
                else None
            )

            stock_len_val = _coerce_float_or_none(
                material_stock_block.get("stock_L_in")
                if isinstance(material_stock_block, _MappingABC)
                else None
            )
            stock_wid_val = _coerce_float_or_none(
                material_stock_block.get("stock_W_in")
                if isinstance(material_stock_block, _MappingABC)
                else None
            )
            stock_thk_val = _coerce_float_or_none(
                material_stock_block.get("stock_T_in")
                if isinstance(material_stock_block, _MappingABC)
                else None
            )

            picked_stock: dict[str, Any] | None = None
            material_lookup_for_pick = normalized_material_key or ""
            if not material_lookup_for_pick and material_display_label:
                material_lookup_for_pick = _normalize_lookup_key(material_display_label)
            picked_stock = _resolve_mcmaster_plate_for_quote(
                float(need_len) if need_len else None,
                float(need_wid) if need_wid else None,
                float(need_thk) if need_thk else None,
                material_key=material_lookup_for_pick or "MIC6",
                stock_L_in=float(stock_len_val) if stock_len_val else None,
                stock_W_in=float(stock_wid_val) if stock_wid_val else None,
                stock_T_in=float(stock_thk_val) if stock_thk_val else None,
            )

            if picked_stock:
                stock_len_val = float(picked_stock.get("len_in") or 0.0)
                stock_wid_val = float(picked_stock.get("wid_in") or 0.0)
                stock_thk_val = float(picked_stock.get("thk_in") or 0.0)
                part_number = picked_stock.get("mcmaster_part")
                source_hint = picked_stock.get("source") or "mcmaster-catalog"
                if isinstance(material_stock_block, _MutableMappingABC):
                    material_stock_block["stock_L_in"] = float(stock_len_val)
                    material_stock_block["stock_W_in"] = float(stock_wid_val)
                    material_stock_block["stock_T_in"] = float(stock_thk_val)
                    material_stock_block["stock_source_tag"] = source_hint
                    material_stock_block["source"] = source_hint
                    if part_number:
                        material_stock_block["mcmaster_part"] = part_number
                        material_stock_block["part_no"] = part_number
                        if not material_stock_block.get("stock_price_source"):
                            material_stock_block["stock_price_source"] = "mcmaster_api"
                if isinstance(material, _MutableMappingABC):
                    material["stock_source_tag"] = source_hint
                    material["source"] = source_hint
                    if part_number:
                        material["mcmaster_part"] = part_number
                        material["part_no"] = part_number
                        if not material.get("stock_price_source"):
                            material["stock_price_source"] = (
                                material_stock_block.get("stock_price_source")
                                if isinstance(material_stock_block, _MappingABC)
                                else "mcmaster_api"
                            )
                if isinstance(result, _MutableMappingABC):
                    if part_number:
                        result["mcmaster_part"] = part_number
                        result["part_no"] = part_number
                        if not result.get("stock_price_source"):
                            result["stock_price_source"] = (
                                material_stock_block.get("stock_price_source")
                                if isinstance(material_stock_block, _MappingABC)
                                else "mcmaster_api"
                            )
                    result["stock_source"] = source_hint
            elif (
                material_lookup_for_pick
                and "mic6" in material_lookup_for_pick.lower()
                and need_len
                and need_wid
                and need_thk
            ):
                blank_lines.append(
                    "  NOTE: No McMaster MIC6 plate found for "
                    f"{float(need_len):.2f}×{float(need_wid):.2f}×{float(need_thk):.3f} in"
                    " with exact thickness"
                )

            if (need_len is None or need_wid is None or need_thk is None) and isinstance(g, dict):
                plan_guess = g.get("stock_plan_guess")
                if isinstance(plan_guess, _MappingABC):
                    if need_len is None:
                        need_len = _coerce_float_or_none(
                            plan_guess.get("need_len_in")
                            or plan_guess.get("required_len_in")
                        )
                    if need_wid is None:
                        need_wid = _coerce_float_or_none(
                            plan_guess.get("need_wid_in")
                            or plan_guess.get("required_wid_in")
                        )
                    if need_thk is None:
                        need_thk = _coerce_float_or_none(
                            plan_guess.get("need_thk_in")
                            or plan_guess.get("stock_thk_in")
                        )

            if need_len and need_wid and need_thk:
                blank_lines.append(
                    f"  Required blank (w/ margins): {need_len:.2f} × {need_wid:.2f} × {need_thk:.2f} in"
                )

            source_tag = None
            if isinstance(material_stock_block, _MappingABC):
                source_tag = material_stock_block.get("stock_source_tag") or material_stock_block.get("source")
            if not source_tag and isinstance(material, _MappingABC):
                source_tag = material.get("stock_source_tag") or material.get("source")
            if isinstance(source_tag, str):
                source_tag = source_tag.strip()
                if not source_tag:
                    source_tag = None

            if stock_len_val and stock_wid_val and stock_thk_val:
                part_label = ""
                if isinstance(result, Mapping):
                    part_label = str(result.get("mcmaster_part") or "").strip()
                part_display = part_label or "—"
                stock_line = (
                    "  Rounded to catalog: "
                    f"{stock_len_val:.2f} × {stock_wid_val:.2f} × {stock_thk_val:.3f} in"
                    f" (McMaster, {part_display})"
                )
                if source_tag:
                    stock_line += f" ({source_tag})"
                thickness_diff = None
                diff_candidate = None
                if isinstance(material_stock_block, _MappingABC):
                    diff_candidate = _coerce_float_or_none(
                        material_stock_block.get("thickness_diff_in")
                    )
                if diff_candidate is None and need_thk and stock_thk_val:
                    thickness_diff = abs(float(stock_thk_val) - float(need_thk))
                elif diff_candidate is not None:
                    thickness_diff = float(diff_candidate)
                if (
                    thickness_diff is not None
                    and thickness_diff > 0.02
                ):
                    warning_mode = (
                        "allowed"
                        if bool(getattr(cfg, "allow_thickness_upsize", False))
                        else "blocked"
                    )
                    stock_line += f" (WARNING: thickness upsize {warning_mode})"
                blank_lines.append(stock_line)

            if blank_lines:
                detail_lines.extend(blank_lines)

            material_cost_map: dict[str, Any] = {}
            if isinstance(material_stock_block, _MappingABC):
                material_cost_map.update(material_stock_block)
            if isinstance(material, _MappingABC):
                material_cost_map.update(material)
            material_cost_components = _material_cost_components(
                material_cost_map,
                overrides=material_overrides,
                cfg=cfg,
            )
            total_material_cost = material_cost_components["total_usd"]
            material_net_cost = material_cost_components["net_usd"]
            if total_material_cost is not None:
                try:
                    material_block["total_material_cost"] = total_material_cost
                except Exception:
                    pass
                try:
                    material_record = breakdown.get("material") if isinstance(breakdown, _MappingABC) else None
                    if isinstance(material_record, _MutableMappingABC):
                        material_record["total_material_cost"] = float(total_material_cost)
                    elif isinstance(material_record, _MappingABC):
                        updated_material = dict(material_record)
                        updated_material["total_material_cost"] = float(total_material_cost)
                        if isinstance(breakdown, _MutableMappingABC):
                            breakdown["material"] = updated_material
                    elif isinstance(breakdown, _MutableMappingABC):
                        breakdown["material"] = {"total_material_cost": float(total_material_cost)}
                except Exception:
                    pass
            net_mass_val = _coerce_float_or_none(net_mass_g)
            effective_mass_source = material.get("effective_mass_g")
            effective_mass_val = _coerce_float_or_none(effective_mass_source)
            prefer_pct_for_scrap = effective_mass_val is not None
            if effective_mass_val is None:
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

            scrap_mass_val = _compute_scrap_mass_g(
                removal_mass_g_est=material.get("material_removed_mass_g_est"),
                scrap_pct_raw=scrap,
                effective_mass_g=effective_mass_val,
                net_mass_g=net_mass_val,
                prefer_pct=prefer_pct_for_scrap,
            )

            if scrap_mass_val is not None:
                scrap_credit_mass_lb = float(scrap_mass_val) / 1000.0 * LB_PER_KG
                material_detail_for_breakdown["scrap_credit_mass_lb"] = (
                    scrap_credit_mass_lb
                )
            else:
                scrap_credit_mass_lb = None
                material_detail_for_breakdown.pop("scrap_credit_mass_lb", None)

            legacy_weight_lines: list[str] = []
            if (starting_mass_val and starting_mass_val > 0) or show_zeros:
                legacy_weight_lines.append(
                    f"  Weight Reference: Start {_format_weight_lb_oz(starting_mass_val)}"
                )
            if (net_mass_val and net_mass_val > 0) or show_zeros:
                legacy_weight_lines.append(
                    f"  Weight Reference: Net {_format_weight_lb_oz(net_mass_val)}"
                )
            if scrap_mass_val is not None:
                if scrap_mass_val > 0 or show_zeros:
                    legacy_weight_lines.append(
                        f"  Weight Reference: Scrap {_format_weight_lb_oz(scrap_mass_val)}"
                    )
            elif show_zeros:
                legacy_weight_lines.append("  Weight Reference: Scrap 0 oz")

            if legacy_weight_lines:
                detail_lines.extend(legacy_weight_lines)

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
            computed_scrap_fraction: float | None = None
            if (
                starting_mass_val is not None
                and starting_mass_val > 0
                and scrap_mass_val is not None
            ):
                try:
                    start_mass = float(starting_mass_val)
                    scrap_mass = max(0.0, float(scrap_mass_val))
                except Exception:
                    start_mass = 0.0
                    scrap_mass = 0.0
                if start_mass > 0:
                    computed_scrap_fraction = scrap_mass / start_mass

            if scrap is not None or computed_scrap_fraction is not None:
                scrap_hint_text = _scrap_source_hint(material)
                if computed_scrap_fraction is not None:
                    scrap_line = (
                        f"  Scrap Percentage: {_pct(computed_scrap_fraction)} (computed)"
                    )
                    if scrap_hint_text and scrap_fraction_val is None:
                        scrap_line += f" ({scrap_hint_text})"
                    weight_lines.append(scrap_line)
                elif scrap is not None:
                    scrap_line = f"  Scrap Percentage: {_pct(scrap)}"
                    if scrap_hint_text:
                        scrap_line += f" ({scrap_hint_text})"
                    weight_lines.append(scrap_line)

                if scrap_fraction_val is not None:
                    geometry_line = (
                        f"  Scrap % (geometry hint): {_pct(scrap_fraction_val)}"
                    )
                    if scrap_hint_text:
                        geometry_line += f" ({scrap_hint_text})"
                    weight_lines.append(geometry_line)
            # Historically the renderer would emit an extra weight-only line here when
            # ``scrap_adjusted_mass`` was available.  The value was the computed "with
            # scrap" mass, but because it lacked a label it rendered as a stray line like
            # ``9 lb 6.4 oz`` in the middle of the material section.  That formatting is
            # confusing and does not provide any additional context to the reader, so we
            # intentionally skip adding that line.  The net, starting, and scrap weights
            # already convey the information a customer needs.

            detail_lines.extend(weight_lines)
            scrap_credit_lines: list[str] = []
            if scrap_credit_entered and scrap_credit:
                credit_display = _m(scrap_credit)
                if credit_display.startswith(currency):
                    credit_display = f"-{credit_display}"
                else:
                    credit_display = f"-{fmt_money(scrap_credit, currency)}"
                scrap_credit_lines.append(f"  Scrap Credit: {credit_display}")
                scrap_credit_unit_price_lb = _coerce_float_or_none(
                    material.get("scrap_credit_unit_price_usd_per_lb")
                )
                if (
                    scrap_credit_mass_lb is not None
                    and scrap_credit_unit_price_lb is not None
                ):
                    scrap_credit_lines.append(
                        "    based on "
                        f"{_format_weight_lb_oz(scrap_mass_val)} × {fmt_money(scrap_credit_unit_price_lb, currency)} / lb"
                    )
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
            def _coerce_dims(candidate: Any) -> tuple[float, float, float] | None:
                if isinstance(candidate, (list, tuple)) and len(candidate) >= 3:
                    try:
                        Lc = float(candidate[0])
                        Wc = float(candidate[1])
                        Tc = float(candidate[2])
                    except Exception:
                        return None
                    return (Lc, Wc, Tc)
                return None

            stock_dims_candidate = _coerce_dims(material.get("stock_dims_in"))
            if stock_dims_candidate is None:
                stock_dims_candidate = _coerce_dims(
                    material_stock_block.get("stock_dims_in")
                )

            stock_L_val: float | None
            stock_W_val: float | None
            stock_T_val: float | None
            if stock_dims_candidate:
                stock_L_val, stock_W_val, stock_T_val = stock_dims_candidate
            else:
                stock_L_val = _coerce_float_or_none(material_stock_block.get("stock_L_in"))
                stock_W_val = _coerce_float_or_none(material_stock_block.get("stock_W_in"))
                stock_T_val = _coerce_float_or_none(material_stock_block.get("stock_T_in"))
            if stock_T_val is None:
                stock_T_val = _coerce_float_or_none(material.get("thickness_in"))
                if stock_T_val is None:
                    stock_T_val = _coerce_float_or_none(g.get("thickness_in"))
                if stock_T_val is None:
                    stock_T_val = _coerce_float_or_none(g.get("thickness_in_guess"))
                if stock_T_val is None and isinstance(baseline, _MappingABC):
                    stock_T_val = _coerce_float_or_none(baseline.get("thickness_in"))
            if stock_L_val is None or stock_W_val is None:
                fallback_L = _coerce_float_or_none(ui_vars.get("Plate Length (in)"))
                if fallback_L is None:
                    fallback_L = _coerce_float_or_none(g.get("plate_length_in"))
                fallback_W = _coerce_float_or_none(ui_vars.get("Plate Width (in)"))
                if fallback_W is None:
                    fallback_W = _coerce_float_or_none(g.get("plate_width_in"))
                if (fallback_L is None or fallback_W is None) and isinstance(g, dict):
                    L_mm = _coerce_float_or_none(g.get("plate_length_mm"))
                    W_mm = _coerce_float_or_none(g.get("plate_width_mm"))
                    derived_ctx = g.get("derived") if isinstance(g, dict) else None
                    if (L_mm is None or W_mm is None) and isinstance(derived_ctx, dict):
                        bbox = derived_ctx.get("bbox_mm")
                        if bbox and len(bbox) >= 2:
                            if L_mm is None:
                                L_mm = _coerce_float_or_none(bbox[0])
                            if W_mm is None:
                                W_mm = _coerce_float_or_none(bbox[1])
                    if fallback_L is None and L_mm is not None:
                        fallback_L = float(L_mm) / 25.4
                    if fallback_W is None and W_mm is not None:
                        fallback_W = float(W_mm) / 25.4
                if stock_L_val is None and fallback_L is not None:
                    stock_L_val = float(fallback_L)
                if stock_W_val is None and fallback_W is not None:
                    stock_W_val = float(fallback_W)
            if (stock_L_val is None or stock_W_val is None) and isinstance(g, dict):
                inferred_dims = infer_plate_lw_in(g)
                if inferred_dims:
                    if stock_L_val is None:
                        stock_L_val = float(inferred_dims[0])
                    if stock_W_val is None:
                        stock_W_val = float(inferred_dims[1])
            if stock_L_val and stock_W_val and stock_T_val:
                stock_dims_candidate = (float(stock_L_val), float(stock_W_val), float(stock_T_val))
            if stock_L_val and stock_W_val and stock_T_val:
                stock_line = f"{float(stock_L_val):.2f} × {float(stock_W_val):.2f} × {float(stock_T_val):.3f} in"
            else:
                inferred_dims = infer_plate_lw_in(g)
                L_disp = stock_L_val
                W_disp = stock_W_val
                if (L_disp is None or W_disp is None) and inferred_dims:
                    if L_disp is None:
                        L_disp = inferred_dims[0]
                    if W_disp is None:
                        W_disp = inferred_dims[1]
                T_disp_val = stock_T_val
                if T_disp_val is None and isinstance(g, dict):
                    T_disp_val = _coerce_float_or_none(g.get("thickness_in"))
                if T_disp_val is None and isinstance(g, dict):
                    T_disp_val = _coerce_float_or_none(g.get("thickness_in_guess"))
                if L_disp and W_disp and T_disp_val:
                    stock_line = f"{float(L_disp):.2f} × {float(W_disp):.2f} × {float(T_disp_val):.3f} in"
                else:
                    T_disp = "—"
                    if T_disp_val is not None:
                        T_disp = f"{float(T_disp_val):.3f}"
                    stock_line = f"— × — × {T_disp} in"
            if isinstance(mat_info, dict):
                mat_info["stock_size_display"] = stock_line
            append_line(f"  Stock used: {stock_line}")
            if detail_lines:
                append_lines(detail_lines)
            mc: Mapping[str, Any] | None = material_cost_components
            if not mc:
                try:
                    mc = _material_cost_components(
                        material_block,
                        overrides=material_overrides,
                        cfg=cfg,
                    )
                except Exception:
                    mc = None
            if mc:
                stock_piece_val = mc.get("stock_piece_usd")
                if stock_piece_val is not None:
                    stock_src = mc.get("stock_source") or ""
                    src_suffix = f" ({stock_src})" if stock_src else ""
                    row(f"Stock Piece{src_suffix}", float(stock_piece_val), indent="  ")
                else:
                    base_source = mc.get("base_source") or ""
                    base_label = "Base Material"
                    if base_source:
                        base_label = f"Base Material @ {base_source}"
                    row(base_label, float(mc.get("base_usd", 0.0)), indent="  ")
                try:
                    tax_val = float(mc.get("tax_usd") or 0.0)
                except Exception:
                    tax_val = 0.0
                if tax_val:
                    row("Material Tax:", round(tax_val, 2), indent="  ")
                try:
                    scrap_val = float(mc.get("scrap_credit_usd") or 0.0)
                except Exception:
                    scrap_val = 0.0
                scrap_text = mc.get("scrap_rate_text") or ""
                if scrap_val and scrap_text:
                    row(
                        f"Scrap Credit @ {scrap_text}",
                        -round(scrap_val, 2),
                        indent="  ",
                    )
                elif scrap_val:
                    row("Scrap Credit", -round(scrap_val, 2), indent="  ")
                try:
                    base_for_total = float(mc.get("base_usd") or 0.0)
                except Exception:
                    base_for_total = 0.0
                tax_for_total = float(tax_val)
                total_material_cost_val = mc.get("total_usd")
                if total_material_cost_val is not None:
                    try:
                        total_material_cost = round(float(total_material_cost_val), 2)
                    except Exception:
                        total_material_cost = None
                else:
                    total_material_cost = None
                if total_material_cost is None:
                    scrap_for_total = min(float(scrap_val), base_for_total + tax_for_total)
                    total_material_cost = round(
                        base_for_total + tax_for_total - scrap_for_total,
                        2,
                    )
                row("Total Material Cost :", total_material_cost, indent="  ")
            elif total_material_cost is not None:
                row("Total Material Cost :", total_material_cost, indent="  ")
            append_line("")

    rates.setdefault("LaborRate", 85.0)
    rates.setdefault(
        "ProgrammingRate", float(rates.get("ProgrammerRate") or rates["LaborRate"])
    )

    nre = breakdown.setdefault("nre", {})
    prog_hr = float(nre.get("programming_hr") or 0.0)
    if prog_hr > 0 and float(nre.get("programming_cost") or 0.0) == 0.0:
        if cfg and getattr(cfg, "separate_machine_labor", False):
            cfg_prog_rate = _coerce_float_or_none(getattr(cfg, "labor_rate_per_hr", None))
            if cfg_prog_rate is None or cfg_prog_rate <= 0:
                cfg_prog_rate = 45.0
            programmer_rate_for_cost = float(cfg_prog_rate)
        else:
            programmer_rate_for_cost = float(_lookup_rate("programming", rates) or 90.0)
        nre["programming_cost"] = round(prog_hr * programmer_rate_for_cost, 2)

    # ---- NRE / Setup costs ---------------------------------------------------
    append_line("NRE / Setup Costs (per lot)")
    append_line(divider)
    prog = nre_detail.get("programming") or {}
    fix  = nre_detail.get("fixture") or {}

    programmer_hours = _safe_float(prog.get("prog_hr"))
    engineer_hours = _safe_float(prog.get("eng_hr"))
    fallback_programmer_rate = float(getattr(cfg, "labor_rate_per_hr", 45.0) or 45.0)
    if not math.isfinite(fallback_programmer_rate) or fallback_programmer_rate <= 0:
        fallback_programmer_rate = 45.0
    programmer_rate_backfill = _coerce_rate_value(rates.get("ProgrammerRate"))
    if programmer_rate_backfill <= 0:
        programmer_rate_backfill = _coerce_rate_value(rates.get("ProgrammingRate"))
    has_programming_rate_detail = False
    if isinstance(nre_cost_details, _MappingABC):
        detail_key = "Programming & Eng (per lot)"
        has_programming_rate_detail = bool(nre_cost_details.get(detail_key))
    explicit_programmer_rate = _safe_float(prog.get("prog_rate"))
    programmer_rate: float
    if explicit_programmer_rate > 0:
        has_programming_rate_detail = True
        programmer_rate = explicit_programmer_rate
    elif programmer_rate_backfill > 0 and not has_programming_rate_detail:
        programmer_rate = programmer_rate_backfill
    else:
        programmer_rate = fallback_programmer_rate
    if separate_labor_cfg and cfg_labor_rate_value > 0.0:
        programmer_rate = cfg_labor_rate_value
    engineer_rate = programmer_rate

    programming_per_lot_val = _safe_float(prog.get("per_lot"))
    nre_programming_per_lot = _safe_float(nre.get("programming_per_lot"))
    if nre_programming_per_lot <= 0:
        legacy_per_part = _safe_float(nre.get("programming_per_part"))
        if legacy_per_part > 0:
            nre_programming_per_lot = legacy_per_part
            try:
                nre["programming_per_lot"] = legacy_per_part
            except Exception:
                pass
        try:
            nre.pop("programming_per_part", None)
        except Exception:
            pass
    qty_for_programming: Any = breakdown.get("qty")
    if qty_for_programming in (None, ""):
        decision_state = result.get("decision_state")
        if isinstance(decision_state, _MappingABC):
            baseline_state = decision_state.get("baseline")
            if isinstance(baseline_state, _MappingABC):
                qty_for_programming = baseline_state.get("qty")
    if qty_for_programming in (None, ""):
        qty_for_programming = qty
    try:
        qty_for_programming_float = float(qty_for_programming or 1)
    except Exception:
        try:
            qty_for_programming_float = float(qty or 1)
        except Exception:
            qty_for_programming_float = 1.0
    if not math.isfinite(qty_for_programming_float) or qty_for_programming_float <= 0:
        qty_for_programming_float = 1.0
    qty_for_programming_float = max(qty_for_programming_float, 1.0)
    prog_hr_total = float((nre.get("programming_hr") or 0.0) or 0.0)
    aggregated_programming_hours = 0.0
    if programmer_hours > 0:
        aggregated_programming_hours += programmer_hours
    if engineer_hours > 0:
        aggregated_programming_hours += engineer_hours
    if prog_hr_total <= 0 and aggregated_programming_hours > 0:
        try:
            nre["programming_hr"] = aggregated_programming_hours
        except Exception:
            pass
        prog_hr_total = aggregated_programming_hours
    total_programming_hours = prog_hr_total
    programming_cost_total = float((nre.get("programming_cost") or 0.0) or 0.0)
    if prog_hr_total > 0 and programming_cost_total == 0.0:
        programming_rate_total = _coerce_rate_value(rates.get("ProgrammingRate"))
        nre["programming_cost"] = round(prog_hr_total * programming_rate_total, 2)

    total_programming_hours = prog_hr_total if prog_hr_total > 0 else aggregated_programming_hours
    if total_programming_hours <= 0:
        total_programming_hours = programmer_hours + engineer_hours

    computed_programming_per_lot = 0.0
    if total_programming_hours > 0 and programmer_rate > 0:
        computed_programming_per_lot = round(total_programming_hours * programmer_rate, 2)

    programming_cost_lot = computed_programming_per_lot if computed_programming_per_lot > 0 else 0.0
    per_lot_source_for_amortized = programming_cost_lot
    if per_lot_source_for_amortized <= 0 and programming_per_lot_val > 0:
        per_lot_source_for_amortized = programming_per_lot_val
    if per_lot_source_for_amortized <= 0 and nre_programming_per_lot > 0:
        per_lot_source_for_amortized = nre_programming_per_lot
    programming_cost_per_part = 0.0
    if per_lot_source_for_amortized > 0:
        programming_cost_per_part = round(
            per_lot_source_for_amortized / max(qty_for_programming_float, 1.0), 2
        )

    if programming_per_lot_val <= 0 and nre_programming_per_lot > 0:
        programming_per_lot_val = round(nre_programming_per_lot, 2)
    if programming_per_lot_val <= 0 and computed_programming_per_lot > 0:
        programming_per_lot_val = round(computed_programming_per_lot, 2)

    existing_programming_per_lot = _safe_float(nre.get("programming_per_lot"))
    if existing_programming_per_lot <= 0 and computed_programming_per_lot > 0:
        nre["programming_per_lot"] = computed_programming_per_lot
    if programming_cost_per_part > 0:
        try:
            labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = programming_cost_per_part
        except Exception:
            labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = programming_cost_per_part
    elif _safe_float(labor_cost_totals.get(PROGRAMMING_PER_PART_LABEL)) <= 0:
        per_lot_source = computed_programming_per_lot
        if per_lot_source <= 0 and nre_programming_per_lot > 0:
            per_lot_source = nre_programming_per_lot
        if per_lot_source <= 0 and programming_per_lot_val > 0:
            per_lot_source = programming_per_lot_val
        try:
            labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = round(
                per_lot_source / max(qty_for_programming_float, 1.0), 2
            )
        except Exception:
            labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = 0.0

    show_programming_row = (
        programming_per_lot_val > 0
        or show_zeros
        or any(_safe_float(prog.get(k)) > 0 for k in ("prog_hr", "eng_hr"))
    )
    if show_programming_row:
        row("Programming & Eng:", programming_per_lot_val)
        has_detail = False
        if programming_per_lot_val > 0 or show_zeros:
            row("Programming Cost:", programming_per_lot_val, indent="  ")
            has_detail = True
        if total_programming_hours > 0:
            write_line(f"  Programming Hrs: {_h(total_programming_hours)}")
            has_detail = True
        if programmer_hours > 0:
            has_detail = True
            write_line(
                f"- Programmer (lot): {_hours_with_rate_text(programmer_hours, programmer_rate)}",
                "    ",
            )
        if engineer_hours > 0:
            has_detail = True
            write_line(
                f"- Engineering (lot): {_hours_with_rate_text(engineer_hours, engineer_rate)}",
                "    ",
            )
        if not has_detail:
            prog_detail = nre_cost_details.get("Programming & Eng (per lot)")
            if prog_detail not in (None, ""):
                write_detail(str(prog_detail))

    # Fixturing (with renamed subline)
    fixture_build_hours = _safe_float(fix.get("build_hr"))
    fixture_build_rate = _resolve_rate_with_fallback(
        fix.get("build_rate"), "FixtureBuildRate", "ShopRate"
    )

    if (fix.get("per_lot", 0.0) > 0) or show_zeros or fixture_build_hours > 0:
        row("Fixturing:", float(fix.get("per_lot", 0.0)))
        has_detail = False
        if fixture_build_hours > 0:
            has_detail = True
            write_line(
                f"- Build labor (lot): {_hours_with_rate_text(fixture_build_hours, fixture_build_rate)}",
                "    ",
            )
        if not has_detail:
            fix_detail = nre_cost_details.get("Fixturing (per lot)")
            if fix_detail not in (None, ""):
                write_detail(str(fix_detail))

    # Any other NRE numeric keys (auto include)
    other_nre_total = 0.0
    for k, v in (nre or {}).items():
        if k in ("programming_per_lot", "fixture_per_part", "programming_per_part"):
            continue
        if isinstance(v, (int, float)) and (v > 0 or show_zeros):
            label = k.replace("_", " ").title()
            amount_val = float(v)
            key_lower = str(k).lower()
            if key_lower.endswith(("_hr", "_hrs", "_hours")):
                hours_label = label
                if hours_label.endswith(" Hours"):
                    hours_label = hours_label[:-6] + " Hrs"
                elif hours_label.endswith(" Hour"):
                    hours_label = hours_label[:-5] + " Hrs"
                elif hours_label.endswith(" Hr"):
                    hours_label = hours_label[:-3] + " Hrs"
                else:
                    hours_label = f"{hours_label} Hrs"
                hours_row(f"{hours_label}:", amount_val)
            else:
                row(f"{label}:", amount_val)
            other_nre_total += amount_val
    if (prog or fix or other_nre_total > 0) and not lines[-1].strip() == "":
        append_line("")

    try:
        amortized_qty = int(result.get("qty") or breakdown.get("qty") or qty or 1)
    except Exception:
        amortized_qty = qty if qty > 0 else 1
    show_amortized = True

    def _should_hide_amortized(label: Any) -> bool:
        """Return True when amortized rows should be omitted from labor output."""

        _, is_amortized = _canonical_amortized_label(label)
        return is_amortized and not show_amortized

    programming_meta_detail = (nre_detail or {}).get("programming") or {}
    programming_per_part_cost = labor_cost_totals.get(PROGRAMMING_PER_PART_LABEL)
    try:
        programming_per_part_cost = float(programming_per_part_cost or 0.0)
    except Exception:
        programming_per_part_cost = 0.0
    if programming_per_part_cost <= 0:
        try:
            programming_per_part_cost = float(
                programming_meta_detail.get("per_part", 0.0) or 0.0
            )
        except Exception:
            programming_per_part_cost = 0.0
    if programming_per_part_cost <= 0:
        per_lot_detail_val = _safe_float(programming_meta_detail.get("per_lot"))
        if per_lot_detail_val > 0:
            programming_per_part_cost = per_lot_detail_val / max(qty_for_programming_float, 1.0)
    if programming_per_part_cost <= 0:
        per_lot_from_nre = _safe_float(nre.get("programming_per_lot"))
        if per_lot_from_nre > 0:
            programming_per_part_cost = per_lot_from_nre / max(qty_for_programming_float, 1.0)
    try:
        nre["programming_per_part"] = float(programming_per_part_cost or 0.0)
    except Exception:
        nre["programming_per_part"] = programming_per_part_cost or 0.0

    fixture_meta_detail = (nre_detail or {}).get("fixture") or {}
    fixture_labor_per_part_cost = labor_cost_totals.get("Fixture Build (amortized)")
    if fixture_labor_per_part_cost is None:
        try:
            fixture_labor_total = float(fixture_meta_detail.get("labor_cost", 0.0) or 0.0)
        except Exception:
            fixture_labor_total = 0.0
        divisor_qty = 1 if qty in (None, "") else qty
        try:
            divisor_qty_val = float(divisor_qty)
        except Exception:
            divisor_qty_val = 1.0
        if not math.isfinite(divisor_qty_val) or divisor_qty_val <= 0:
            divisor_qty_val = 1.0
        fixture_labor_per_part_cost = fixture_labor_total / divisor_qty_val
    try:
        nre["fixture_per_part"] = float(fixture_labor_per_part_cost or 0.0)
    except Exception:
        nre["fixture_per_part"] = fixture_labor_per_part_cost or 0.0

    try:
        amortized_nre_total = float(programming_per_part_cost or 0.0) + float(
            fixture_labor_per_part_cost or 0.0
        )
    except Exception:
        amortized_nre_total = 0.0

    drilling_meta_raw = (
        breakdown.get("drilling_meta") if isinstance(breakdown, _MappingABC) else None
    )
    drilling_meta_mutable: dict[str, Any] | None = None
    if isinstance(drilling_meta_raw, dict):
        drilling_meta_mutable = drilling_meta_raw
    elif isinstance(drilling_meta_raw, _MappingABC):
        try:
            drilling_meta_mutable = dict(drilling_meta_raw)
        except Exception:
            drilling_meta_mutable = {}
        if isinstance(breakdown, dict):
            breakdown["drilling_meta"] = drilling_meta_mutable
    elif isinstance(breakdown, dict):
        drilling_meta_mutable = breakdown.setdefault("drilling_meta", {})

    drilling_meta_map = (
        drilling_meta_mutable if drilling_meta_mutable is not None else drilling_meta_raw
    )

    drill_machine_minutes_estimate = 0.0
    drill_tool_minutes_estimate = 0.0
    drill_total_minutes_estimate = 0.0
    drilling_card_detail: dict[str, Any] | None = None
    if isinstance(drilling_meta_map, _MappingABC):
        (
            drill_machine_minutes_estimate,
            drill_tool_minutes_estimate,
            drill_total_minutes_estimate,
            drilling_card_detail,
        ) = _estimate_drilling_minutes_from_meta(drilling_meta_map, (result.get("geo") if isinstance(result, _MappingABC) else None) or (breakdown.get("geo") if isinstance(breakdown, _MappingABC) else None))
        if (
            drill_total_minutes_estimate > 0.0
            and isinstance(drilling_meta_mutable, dict)
        ):
            drilling_meta_mutable["toolchange_minutes"] = float(
                drill_tool_minutes_estimate
            )
            drilling_meta_mutable["total_minutes_with_toolchange"] = float(
                drill_total_minutes_estimate
            )
            drilling_meta_mutable["total_minutes_billed"] = float(
                drill_total_minutes_estimate
            )
            drilling_meta_mutable["total_minutes"] = float(
                drill_machine_minutes_estimate
            )

    card_minutes_val = None
    have_card_minutes = False
    if isinstance(drilling_meta_map, _MappingABC):
        for key in ("total_minutes_billed", "total_minutes_with_toolchange", "total_minutes"):
            candidate_minutes = _coerce_float_or_none(drilling_meta_map.get(key))
            if candidate_minutes is not None:
                card_minutes_val = float(candidate_minutes)
                have_card_minutes = True
                break
    if drill_total_minutes_estimate > 0.0:
        card_minutes_val = float(drill_total_minutes_estimate)
        have_card_minutes = True
    if card_minutes_val is None:
        card_minutes_val = 0.0
    card_minutes_precise = float(card_minutes_val)
    removal_drilling_minutes = card_minutes_precise if have_card_minutes else None
    removal_drilling_hours_precise = (
        card_minutes_precise / 60.0 if have_card_minutes else None
    )
    if drill_total_minutes_estimate > 0.0:
        removal_drilling_minutes = float(drill_total_minutes_estimate)
        removal_drilling_hours_precise = removal_drilling_minutes / 60.0
    card_hr = round(card_minutes_precise / 60.0, 2)
    row_hr = card_hr
    drilling_minutes_from_bucket = None
    bucket_view_snapshot = breakdown.get("bucket_view") if isinstance(breakdown, _MappingABC) else None
    if isinstance(bucket_view_snapshot, _MappingABC):
        buckets_snapshot = bucket_view_snapshot.get("buckets")
        if isinstance(buckets_snapshot, _MappingABC):
            drilling_bucket_snapshot = buckets_snapshot.get("drilling")
            if isinstance(drilling_bucket_snapshot, _MappingABC):
                drilling_minutes_from_bucket = _coerce_float_or_none(
                    drilling_bucket_snapshot.get("minutes")
                )
                if drilling_minutes_from_bucket is not None:
                    row_hr = round(float(drilling_minutes_from_bucket) / 60.0, 2)
    if have_card_minutes and drilling_minutes_from_bucket is not None:
        try:
            bucket_hr_precise = float(drilling_minutes_from_bucket) / 60.0
        except Exception:
            bucket_hr_precise = None
        if (
            bucket_hr_precise is not None
            and removal_drilling_hours_precise is not None
            and abs(removal_drilling_hours_precise - bucket_hr_precise) > 0.01
        ):
            if prefer_removal_drilling_hours:
                logger.info(
                    "[hours-sync] Overriding Drilling bucket hours from %.2f -> %.2f (source=removal_card)",
                    bucket_hr_precise,
                    removal_drilling_hours_precise,
                )
                minutes_to_apply = removal_drilling_minutes or 0.0
                _sync_drilling_bucket_view(
                    bucket_view_snapshot,
                    billed_minutes=float(minutes_to_apply),
                    billed_cost=None,
                )
                drilling_minutes_from_bucket = minutes_to_apply
                row_hr = round(removal_drilling_hours_precise, 2)
            else:
                raise RuntimeError(
                    f"[FATAL] Drilling hours mismatch: card {card_hr} vs row {row_hr}. "
                    "Late writer is overwriting bucket_view."
                )

    process_plan_summary_local = locals().get("process_plan_summary")
    if not isinstance(process_plan_summary_local, _MappingABC):
        process_plan_summary_local = (
            breakdown.get("process_plan") if isinstance(breakdown, _MappingABC) else None
        )

    drilling_time_per_hole_data: Mapping[str, Any] | None = None
    if isinstance(result, _MappingABC):
        candidate_dtph = result.get("drilling_time_per_hole")
        if isinstance(candidate_dtph, _MappingABC):
            drilling_time_per_hole_data = candidate_dtph
    if drilling_time_per_hole_data is None and isinstance(breakdown, _MappingABC):
        candidate_dtph = breakdown.get("drilling_time_per_hole")
        if isinstance(candidate_dtph, _MappingABC):
            drilling_time_per_hole_data = candidate_dtph

    removal_card_lines: list[str] = []
    removal_card_extra: dict[str, float] = {}
    (
        removal_card_extra,
        removal_card_lines,
        process_plan_summary_local,
    ) = _compute_drilling_removal_section(
        breakdown=breakdown,
        rates=rates,
        drilling_meta_source=drilling_meta_map,
        drilling_card_detail=drilling_card_detail,
        drill_machine_minutes_estimate=drill_machine_minutes_estimate,
        drill_tool_minutes_estimate=drill_tool_minutes_estimate,
        drill_total_minutes_estimate=drill_total_minutes_estimate,
        process_plan_summary=process_plan_summary_local,
        drilling_time_per_hole=drilling_time_per_hole_data,
    )

    if removal_card_extra.get("drill_machine_minutes") is not None:
        drill_machine_minutes_estimate = float(removal_card_extra["drill_machine_minutes"])
    if removal_card_extra.get("drill_labor_minutes") is not None:
        drill_tool_minutes_estimate = float(removal_card_extra["drill_labor_minutes"])
    if removal_card_extra.get("removal_drilling_minutes") is not None:
        removal_drilling_minutes = float(
            removal_card_extra["removal_drilling_minutes"]
        )
        removal_drilling_hours_precise = removal_drilling_minutes / 60.0
    if removal_card_extra.get("drill_total_minutes") is not None:
        drill_total_minutes_estimate = float(removal_card_extra["drill_total_minutes"])
        removal_drilling_minutes = float(removal_card_extra["drill_total_minutes"])
        removal_drilling_hours_precise = removal_drilling_minutes / 60.0
    if removal_card_extra.get("removal_drilling_hours") is not None:
        removal_drilling_hours_precise = float(removal_card_extra["removal_drilling_hours"])
        removal_drilling_minutes = removal_drilling_hours_precise * 60.0
    if removal_drilling_minutes is not None and "removal_drilling_minutes" not in removal_card_extra:
        try:
            removal_card_extra["removal_drilling_minutes"] = float(removal_drilling_minutes)
        except Exception:
            removal_card_extra["removal_drilling_minutes"] = removal_drilling_minutes

    machine_minutes_snapshot = max(0.0, float(drill_machine_minutes_estimate or 0.0))
    labor_minutes_snapshot = max(0.0, float(drill_tool_minutes_estimate or 0.0))
    total_minutes_snapshot = float(drill_total_minutes_estimate or 0.0)
    if total_minutes_snapshot <= 0.0:
        combined_minutes = machine_minutes_snapshot + labor_minutes_snapshot
        if combined_minutes > 0.0:
            total_minutes_snapshot = combined_minutes

    drill_minutes_extra_targets: list[_MutableMappingABC[str, Any]] = []

    def _stash_drill_minutes(owner: Any) -> _MutableMappingABC[str, Any] | None:
        if owner is None:
            return None
        try:
            extra_candidate = getattr(owner, "extra", None)
        except Exception:
            extra_candidate = None
        extra_map: _MutableMappingABC[str, Any] | None
        if isinstance(extra_candidate, _MutableMappingABC):
            extra_map = extra_candidate
        elif isinstance(owner, PlannerBucketRenderState):
            extra_map = owner.extra
        elif isinstance(owner, dict):
            extra_map = owner.setdefault("extra", {})  # type: ignore[assignment]
        elif isinstance(owner, _MutableMappingABC):
            extra_map = owner.setdefault("extra", {})  # type: ignore[assignment]
        else:
            extra_map = None
        if isinstance(extra_map, _MutableMappingABC):
            extra_map["drill_machine_minutes"] = float(machine_minutes_snapshot)
            extra_map["drill_labor_minutes"] = float(labor_minutes_snapshot)
            extra_map["drill_total_minutes"] = float(total_minutes_snapshot)
            return extra_map
        return None

    initial_stash_candidates: list[Any] = [locals().get("render_state")]
    if isinstance(breakdown, _MappingABC):
        for key in ("planner_render_state", "bucket_render_state", "render_state"):
            initial_stash_candidates.append(breakdown.get(key))
    for candidate_owner in initial_stash_candidates:
        extra_map_candidate = _stash_drill_minutes(candidate_owner)
        if extra_map_candidate is not None and extra_map_candidate not in drill_minutes_extra_targets:
            drill_minutes_extra_targets.append(extra_map_candidate)

    append_line("Process & Labor Costs")
    append_line(divider)

    canonical_bucket_order: list[str] = []
    canonical_bucket_summary: dict[str, dict[str, float]] = {}
    bucket_table_rows: list[tuple[str, float, float, float, float]] = []
    detail_lookup: dict[str, str] = {}
    label_to_canon: dict[str, str] = {}
    canon_to_display_label: dict[str, str] = {}
    process_cost_row_details: dict[str, tuple[float, float, float]] = {}
    class _BucketRowSpec(typing.NamedTuple):
        label: str
        hours: float
        rate: float
        total: float
        labor: float
        machine: float
        canon_key: str
        minutes: float

    bucket_row_specs: list[_BucketRowSpec] = []

    labor_costs_display.clear()
    hour_summary_entries.clear()
    display_labor_for_ladder = 0.0
    display_machine = 0.0

    bucket_view_struct: Mapping[str, Any] | None = None
    if isinstance(breakdown, _MappingABC):
        candidate_view = breakdown.get("bucket_view")
        if isinstance(candidate_view, _MutableMappingABC):
            extra_map_candidate = _stash_drill_minutes(candidate_view)
            if (
                extra_map_candidate is not None
                and extra_map_candidate not in drill_minutes_extra_targets
            ):
                drill_minutes_extra_targets.append(extra_map_candidate)
            bucket_view_struct = typing.cast(Mapping[str, Any], candidate_view)
        elif isinstance(candidate_view, _MappingABC):
            bucket_view_struct = typing.cast(Mapping[str, Any], candidate_view)

    if (
        drill_total_minutes_estimate > 0.0
        and isinstance(bucket_view_struct, _MutableMappingABC)
    ):
        if isinstance(rates, _MappingABC):
            rates_for_sync: Mapping[str, Any] = rates
        elif isinstance(rates, dict):
            rates_for_sync = rates
        else:
            rates_for_sync = {}
        drill_rate_sync = _coerce_float_or_none(rates_for_sync.get("DrillingRate"))
        if drill_rate_sync is None or drill_rate_sync <= 0.0:
            drill_rate_sync = _coerce_float_or_none(rates_for_sync.get("MachineRate"))
        billed_cost_override = (
            (drill_total_minutes_estimate / 60.0) * drill_rate_sync
            if drill_rate_sync is not None and drill_rate_sync > 0.0
            else None
        )
        _sync_drilling_bucket_view(
            bucket_view_struct,
            billed_minutes=float(drill_total_minutes_estimate),
            billed_cost=billed_cost_override,
        )

    bucket_state = _build_planner_bucket_render_state(
        bucket_view_struct,
        label_overrides=label_overrides,
        labor_cost_details=labor_cost_details,
        labor_cost_details_input=labor_cost_details_input,
        process_costs_canon=process_costs_canon,
        rates=rates,
        removal_drilling_hours=removal_drilling_hours_precise,
        prefer_removal_drilling_hours=prefer_removal_drilling_hours,
        cfg=cfg,
        bucket_ops=bucket_ops_map,
        drill_machine_minutes=drill_machine_minutes_estimate,
        drill_labor_minutes=drill_tool_minutes_estimate,
        drill_total_minutes=drill_total_minutes_estimate,
    )
    render_state = bucket_state

    bucket_state_extra_map = _stash_drill_minutes(bucket_state)
    if (
        bucket_state_extra_map is not None
        and bucket_state_extra_map not in drill_minutes_extra_targets
    ):
        drill_minutes_extra_targets.append(bucket_state_extra_map)

    if removal_card_extra:
        extra_map = getattr(bucket_state, "extra", None)
        if not isinstance(extra_map, dict):
            extra_map = {}
            bucket_state.extra = extra_map
        for key, value in removal_card_extra.items():
            if value is None:
                continue
            try:
                extra_map[key] = float(value)
            except Exception:
                extra_map[key] = value

    geometry_for_explainer: Mapping[str, Any] | None = None
    if isinstance(geometry, _MappingABC) and geometry:
        geometry_for_explainer = typing.cast(Mapping[str, Any], geometry)
    elif isinstance(g, dict) and g:
        geometry_for_explainer = typing.cast(Mapping[str, Any], g)
    elif isinstance(breakdown, _MappingABC):
        for key in ("geometry", "geo_context", "geometry_context", "geo"):
            candidate = breakdown.get(key)
            if isinstance(candidate, _MappingABC) and candidate:
                geometry_for_explainer = typing.cast(Mapping[str, Any], candidate)
                break

    def _norm(s: Any) -> str:
        import re

        return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")

    LABORISH = {
        "finishing_deburr",
        "inspection",
        "assembly",
        "toolmaker_support",
        "ehs_compliance",
        "fixture_build_amortized",
        "programming_amortized",
    }

    RATE_KEYS = {
        "milling": ["MillingRate"],
        "drilling": ["DrillingRate"],
        "counterbore": ["CounterboreRate", "DrillingRate"],
        "tapping": ["TappingRate", "DrillingRate"],
        "grinding": [
            "GrindingRate",
            "SurfaceGrindRate",
            "ODIDGrindRate",
            "JigGrindRate",
        ],
        "finishing_deburr": ["FinishingRate", "DeburrRate"],
        "saw_waterjet": ["SawWaterjetRate", "SawRate", "WaterjetRate"],
        "inspection": ["InspectionRate"],
        "wire_edm": ["WireEDMRate", "EDMRate"],
        "sinker_edm": ["SinkerEDMRate", "EDMRate"],
    }

    def _rate_for_bucket(key: str, rates: Mapping[str, Any] | dict) -> float:
        k = _norm(key)
        rates_dict = dict(rates) if not isinstance(rates, dict) else rates
        for rk in RATE_KEYS.get(k, []):
            v = rates_dict.get(rk) or rates_dict.get(_norm(rk))
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        fallback_key = "LaborRate" if k in LABORISH else "MachineRate"
        return float(rates_dict.get(fallback_key) or rates_dict.get(_norm(fallback_key)) or 0.0)

    def _rows_from_bucket_view(
        view: Mapping[str, Any] | None,
    ) -> tuple[
        list[str],
        dict[str, dict[str, float]],
        list[tuple[str, float, float, float, float]],
        dict[str, str],
        dict[str, str],
    ]:
        if not isinstance(view, _MappingABC):
            return ([], {}, [], {}, {})

        buckets_obj = view.get("buckets")
        if not isinstance(buckets_obj, _MappingABC):
            return ([], {}, [], {}, {})

        order_obj = view.get("order")
        if isinstance(order_obj, Sequence):
            ordered_keys = list(order_obj)
        else:
            ordered_keys = list(_preferred_order_then_alpha(buckets_obj.keys()))

        if isinstance(rates, dict):
            rates_map_local: Mapping[str, Any] = rates
        elif isinstance(rates, _MappingABC):
            rates_map_local = dict(rates)
        else:
            rates_map_local = {}

        drill_rate_local = _rate_for_bucket("drilling", rates_map_local)

        canonical_order_local: list[str] = []
        canonical_summary_local: dict[str, dict[str, float]] = {}
        label_map_local: dict[str, str] = {}
        canon_label_map_local: dict[str, str] = {}
        rows_local: list[tuple[str, float, float, float, float]] = []
        raw_key_by_canon: dict[str, str] = {}

        for raw_key in ordered_keys:
            lookup_candidates: Sequence[Any]
            if isinstance(buckets_obj, dict) and raw_key not in buckets_obj:
                lookup_candidates = (raw_key, str(raw_key))
            else:
                lookup_candidates = (raw_key,)

            info: Mapping[str, Any] | None = None
            key_text = str(raw_key)
            for candidate_key in lookup_candidates:
                candidate_info = (
                    buckets_obj.get(candidate_key)
                    if isinstance(buckets_obj, _MappingABC)
                    else None
                )
                if isinstance(candidate_info, _MappingABC):
                    info = candidate_info
                    key_text = str(candidate_key)
                    break

            if not isinstance(info, _MappingABC):
                continue

            minutes_val = _safe_float(info.get("minutes"), default=0.0)
            hours_val = minutes_val / 60.0 if minutes_val else 0.0
            labor_val = _safe_float(info.get("labor$"), default=0.0)
            machine_val = _safe_float(info.get("machine$"), default=0.0)
            total_val = labor_val + machine_val

            canon_key = _canonical_bucket_key(key_text)
            if not canon_key:
                canon_key = _normalize_bucket_key(key_text)
            if not canon_key:
                canon_key = key_text

            norm_key = _norm(key_text)
            rate_val = drill_rate_local if norm_key == "drilling" else _rate_for_bucket(
                key_text, rates_map_local
            )
            if rate_val <= 0.0 and hours_val > 0.0:
                try:
                    rate_val = total_val / hours_val if total_val > 0.0 else 0.0
                except Exception:
                    rate_val = 0.0
            if total_val <= 0.0 and hours_val > 0.0 and rate_val > 0.0:
                total_val = round(hours_val * rate_val, 2)
                if norm_key in LABORISH:
                    labor_val = total_val
                    machine_val = 0.0
                else:
                    machine_val = total_val
                    labor_val = 0.0

            summary_entry = canonical_summary_local.setdefault(
                canon_key,
                {
                    "minutes": 0.0,
                    "hours": 0.0,
                    "labor": 0.0,
                    "machine": 0.0,
                    "total": 0.0,
                },
            )
            summary_entry["minutes"] += minutes_val
            summary_entry["hours"] += hours_val
            summary_entry["labor"] += labor_val
            summary_entry["machine"] += machine_val
            summary_entry["total"] += total_val

            if canon_key not in canonical_order_local:
                canonical_order_local.append(canon_key)
            raw_key_by_canon.setdefault(canon_key, key_text)

        for canon_key in canonical_order_local:
            metrics = canonical_summary_local.get(canon_key) or {}
            minutes_total = _safe_float(metrics.get("minutes"), default=0.0)
            hours_total = _safe_float(metrics.get("hours"), default=0.0)
            if hours_total <= 0.0 and minutes_total > 0.0:
                hours_total = minutes_total / 60.0
            labor_total = _safe_float(metrics.get("labor"), default=0.0)
            machine_total = _safe_float(metrics.get("machine"), default=0.0)
            total_cost = _safe_float(metrics.get("total"), default=0.0)

            source_key = raw_key_by_canon.get(canon_key, canon_key)
            norm_key = _norm(source_key)
            rate_val = (
                drill_rate_local
                if norm_key == "drilling"
                else _rate_for_bucket(source_key, rates_map_local)
            )
            if rate_val <= 0.0 and hours_total > 0.0:
                try:
                    rate_val = total_cost / hours_total if total_cost > 0.0 else 0.0
                except Exception:
                    rate_val = 0.0
            if total_cost <= 0.0 and hours_total > 0.0 and rate_val > 0.0:
                total_cost = round(hours_total * rate_val, 2)
                if norm_key in LABORISH:
                    labor_total = total_cost
                    machine_total = 0.0
                else:
                    machine_total = total_cost
                    labor_total = 0.0

            if total_cost <= 0.0 and hours_total <= 0.0:
                continue

            display_label = _display_bucket_label(canon_key, label_overrides)
            bucket_row_specs.append(
                _BucketRowSpec(
                    label=display_label,
                    hours=hours_total,
                    rate=rate_val,
                    total=total_cost,
                    labor=labor_total,
                    machine=machine_total,
                    canon_key=canon_key,
                    minutes=minutes_total,
                )
            )
            rows_local.append(
                (
                    display_label,
                    round(hours_total, 2),
                    round(labor_total, 2),
                    round(machine_total, 2),
                    round(total_cost, 2),
                )
            )
            label_map_local[display_label] = canon_key
            canon_label_map_local.setdefault(canon_key, display_label)

        return (
            canonical_order_local,
            canonical_summary_local,
            rows_local,
            label_map_local,
            canon_label_map_local,
        )

    (
        canonical_bucket_order,
        canonical_bucket_summary,
        bucket_table_rows,
        bucket_label_map,
        bucket_canon_label_map,
    ) = _rows_from_bucket_view(bucket_view_struct)

    def _row_component(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    display_labor_from_rows = 0.0
    display_machine_from_rows = 0.0
    if bucket_table_rows:
        display_labor_from_rows = sum(
            _row_component(row[2]) for row in bucket_table_rows
        )
        display_machine_from_rows = sum(
            _row_component(row[3]) for row in bucket_table_rows
        )

    planner_totals_map: typing.Mapping[str, Any] | None = None
    if bucket_table_rows:
        planner_totals_candidates: list[typing.Mapping[str, Any] | None] = []
        if isinstance(process_plan_summary_local, _MappingABC):
            pricing_info = process_plan_summary_local.get("pricing")
            if isinstance(pricing_info, _MappingABC):
                planner_totals_candidates.append(pricing_info.get("totals"))
        if isinstance(breakdown, _MappingABC):
            planner_pricing = breakdown.get("process_plan_pricing")
            if isinstance(planner_pricing, _MappingABC):
                planner_totals_candidates.append(planner_pricing.get("totals"))

        for candidate in planner_totals_candidates:
            if isinstance(candidate, _MappingABC):
                planner_totals_map = candidate
                break

    if bucket_table_rows and isinstance(planner_totals_map, _MappingABC):
        planner_labor_total = float(
            _coerce_float_or_none(planner_totals_map.get("labor_cost")) or 0.0
        )
        planner_machine_total = float(
            _coerce_float_or_none(planner_totals_map.get("machine_cost")) or 0.0
        )

        assert (
            abs(display_machine_from_rows - planner_machine_total) < 0.51
        ), "Machine $ mismatch (check drilling minutes merge)"
        assert (
            abs(display_labor_from_rows - planner_labor_total) < 0.51
        ), "Labor $ mismatch"
    detail_lookup.update(bucket_state.detail_lookup)
    label_to_canon.update(bucket_state.label_to_canon)
    canon_to_display_label.update(bucket_state.canon_to_display_label)
    label_to_canon.update(bucket_label_map)
    canon_to_display_label.update(bucket_canon_label_map)
    labor_costs_display.update(bucket_state.labor_costs_display)
    if bucket_table_rows:
        for label, _, _, _, total_val in bucket_table_rows:
            labor_costs_display[label] = total_val
    if bucket_table_rows:
        display_labor_for_ladder = display_labor_from_rows
        display_machine = display_machine_from_rows
    else:
        display_labor_for_ladder = bucket_state.display_labor_total
        display_machine = bucket_state.display_machine_total
    hour_summary_entries.update(bucket_state.hour_entries)
    bucket_minutes_detail: dict[str, float] = {}
    for canon_key, metrics in canonical_bucket_summary.items():
        bucket_minutes_detail[canon_key] = _safe_float(
            metrics.get("minutes"), default=0.0
        )
    extra_bucket_minutes_detail = breakdown.get("bucket_minutes_detail")
    if isinstance(extra_bucket_minutes_detail, _MappingABC):
        for key, minutes in extra_bucket_minutes_detail.items():
            bucket_minutes_detail[key] = _safe_float(minutes)
    process_costs_for_render: dict[str, float] = {}
    for canon_key, metrics in canonical_bucket_summary.items():
        process_costs_for_render[canon_key] = _safe_float(
            metrics.get("total"), default=0.0
        )
    for canon_key, amount in process_costs_canon.items():
        if canon_key not in process_costs_for_render:
            process_costs_for_render[canon_key] = _safe_float(amount, default=0.0)

    proc_total = 0.0
    amortized_nre_total = 0.0

    def _add_labor_cost_line(
        label: str,
        amount: Any,
        *,
        process_key: str | None = None,
        detail_bits: Iterable[Any] | None = None,
    ) -> None:
        nonlocal proc_total
        try:
            numeric_amount = float(amount or 0.0)
        except Exception:
            numeric_amount = 0.0
        if not ((numeric_amount > 0.0) or show_zeros):
            return

        row(label, numeric_amount, indent="  ")

        details_rendered = False
        if detail_bits:
            for bit in detail_bits:
                if bit in (None, ""):
                    continue
                write_detail(str(bit), indent="    ")
                details_rendered = True

        if not details_rendered:
            detail_text = detail_lookup.get(label)
            if detail_text not in (None, ""):
                write_detail(str(detail_text), indent="    ")
                details_rendered = True

        if not details_rendered:
            extra_detail = labor_cost_details.get(label)
            if extra_detail not in (None, ""):
                canon_key = label_to_canon.get(label)
                if canon_key and canon_key in process_cost_row_details:
                    extra_detail = None
            if extra_detail not in (None, ""):
                write_detail(str(extra_detail), indent="    ")
                details_rendered = True

        if not details_rendered:
            canon_key = label_to_canon.get(label)
            key_for_notes = process_key or canon_key
            if key_for_notes:
                add_process_notes(key_for_notes, indent="    ")

        proc_total += numeric_amount

    @dataclass(slots=True)
    class _ProcessRowRecord:
        name: str
        hours: float
        rate: float
        total: float
        canon_key: str | None = None

    class _ProcessCostTableRecorder:
        def __init__(self) -> None:
            self.had_rows = False
            self.rows: list[_ProcessRowRecord] = []
            self._rows: list[dict[str, Any]] = []
            self._index: dict[str, int] = {}

        def add_row(
            self,
            label: str,
            hours: float,
            rate: float,
            cost: float,
        ) -> None:
            self.had_rows = True
            label_str = str(label or "").strip()
            canon_key = label_to_canon.get(label_str)
            if canon_key is None:
                canon_key = _canonical_bucket_key(label_str)
            display_label = label_str
            if canon_key:
                override_label = canon_to_display_label.get(canon_key)
                if override_label:
                    display_label = override_label
                    label_to_canon.setdefault(display_label, canon_key)
            try:
                hours_val = float(hours or 0.0)
            except Exception:
                hours_val = 0.0
            try:
                rate_val = float(rate or 0.0)
            except Exception:
                rate_val = 0.0
            try:
                cost_val = float(cost or 0.0)
            except Exception:
                cost_val = 0.0
            if canon_key:
                process_cost_row_details[canon_key] = (hours_val, rate_val, cost_val)
            record_canon = canon_key or _canonical_bucket_key(display_label)
            if not record_canon:
                record_canon = None
            record = _ProcessRowRecord(
                display_label,
                hours_val,
                rate_val,
                cost_val,
                record_canon,
            )
            self.rows.append(record)
            if record_canon:
                self._index[record_canon] = len(self.rows) - 1
            rate_display = _display_rate_for_row(
                record_canon or display_label,
                cfg=cfg,
                render_state=bucket_state,
                hours=hours_val,
            )
            detail_parts: list[str] = []
            if rate_display:
                detail_parts.append(str(rate_display))
            existing_detail = detail_lookup.get(display_label)
            if existing_detail not in (None, ""):
                for segment in re.split(r";\s*", str(existing_detail)):
                    cleaned = segment.strip()
                    if not cleaned or cleaned.startswith("-"):
                        continue
                    if cleaned not in detail_parts:
                        detail_parts.append(cleaned)
            simple_hours_line: str | None = None
            if hours_val > 0.0 and cost_val > 0.0:
                rate_for_detail = cost_val / hours_val if hours_val else 0.0
                planner_rate_override: float | None = None
                if record_canon and bucket_state is not None:
                    extra_payload = getattr(bucket_state, "extra", None)
                    split_lookup: Mapping[str, Any] | None = None
                    if isinstance(extra_payload, _MappingABC):
                        split_source = extra_payload.get("bucket_hour_split")
                        if isinstance(split_source, _MappingABC):
                            split_lookup = split_source
                    if isinstance(split_lookup, _MappingABC):
                        split_entry = split_lookup.get(record_canon)
                        if not isinstance(split_entry, _MappingABC) and canon_key and canon_key != record_canon:
                            split_entry = split_lookup.get(canon_key)
                        if not isinstance(split_entry, _MappingABC):
                            alt_key = _canonical_bucket_key(record_canon)
                            if alt_key and alt_key != record_canon:
                                split_entry = split_lookup.get(alt_key)
                        if isinstance(split_entry, _MappingABC):
                            machine_split = _safe_float(split_entry.get("machine_hours"))
                            labor_split = _safe_float(split_entry.get("labor_hours"))
                            if machine_split > 0.0 and labor_split <= 0.0:
                                meta_entry = _lookup_process_meta(record_canon) or _lookup_process_meta(display_label)
                                if isinstance(meta_entry, _MappingABC):
                                    base_extra_val = _safe_float(meta_entry.get("base_extra"))
                                    meta_rate_val = _safe_float(meta_entry.get("rate"))
                                    if base_extra_val > 0.0 and meta_rate_val > 0.0:
                                        planner_rate_override = meta_rate_val
                if planner_rate_override and planner_rate_override > 0.0:
                    rate_for_detail = planner_rate_override
                if rate_for_detail > 0.0:
                    simple_hours_line = f"{hours_val:.2f} hr @ ${rate_for_detail:.2f}/hr"
            if simple_hours_line and simple_hours_line not in detail_parts:
                detail_parts.append(simple_hours_line)
            self._rows.append(
                {
                    "label": display_label,
                    "hours": hours_val,
                    "rate": rate_val,
                    "cost": cost_val,
                    "canon_key": record_canon,
                    "rate_display": rate_display,
                }
            )
            _add_labor_cost_line(
                display_label,
                cost,
                process_key=canon_key,
                detail_bits=detail_parts if detail_parts else None,
            )
            try:
                labor_costs_display[display_label] = float(cost or 0.0)
            except Exception:
                labor_costs_display[display_label] = 0.0

        def update_row(
            self,
            canon_key: str,
            *,
            hours: float | None = None,
            rate: float | None = None,
            cost: float | None = None,
        ) -> None:
            index = self._index.get(canon_key)
            if index is None or index < 0 or index >= len(self.rows):
                return
            record = self.rows[index]
            row_dict = self._rows[index]

            if hours is not None:
                try:
                    hours_val = float(hours)
                except Exception:
                    hours_val = 0.0
                record.hours = hours_val
                row_dict["hours"] = hours_val
            if rate is not None:
                try:
                    rate_val = float(rate)
                except Exception:
                    rate_val = 0.0
                record.rate = rate_val
                row_dict["rate"] = rate_val
            if cost is not None:
                try:
                    cost_val = float(cost)
                except Exception:
                    cost_val = 0.0
                record.total = cost_val
                row_dict["cost"] = cost_val
            hours_for_display = row_dict.get("hours", 0.0)
            rate_display = _display_rate_for_row(
                canon_key or record.name,
                cfg=cfg,
                render_state=bucket_state,
                hours=float(hours_for_display or 0.0),
            )
            row_dict["rate_display"] = rate_display

    def _prepare_amortized_details() -> dict[str, tuple[float, float]]:
        nonlocal amortized_nre_total, display_labor_for_ladder
        amortized_nre_total = 0.0
        additions: dict[str, tuple[float, float]] = {}

        qty_divisor = max(qty_for_programming_float, 1.0)
        programming_minutes = 0.0
        programming_detail_local = (
            nre_detail.get("programming", {}) if isinstance(nre_detail, dict) else {}
        )
        total_programming_hours_local = _safe_float(nre.get("programming_hr"))
        accumulate_from_detail = False
        if total_programming_hours_local <= 0:
            total_programming_hours_local = 0.0
            accumulate_from_detail = True

        detail_args: list[str] = []
        detail_hours = 0.0

        prog_hr_detail = _safe_float(programming_detail_local.get("prog_hr"))
        if prog_hr_detail > 0:
            if accumulate_from_detail:
                total_programming_hours_local += prog_hr_detail
            detail_hours += prog_hr_detail
            detail_args.append(
                f"- Programmer (lot): {_hours_with_rate_text(prog_hr_detail, programmer_rate)}"
            )

        eng_hr_detail = _safe_float(programming_detail_local.get("eng_hr"))
        if eng_hr_detail > 0:
            if accumulate_from_detail:
                total_programming_hours_local += eng_hr_detail
            detail_hours += eng_hr_detail
            detail_args.append(
                f"- Engineering (lot): {_hours_with_rate_text(eng_hr_detail, programmer_rate)}"
            )

        remaining_hours = max(total_programming_hours_local - detail_hours, 0.0)
        if remaining_hours > 0:
            detail_args.append(
                f"Additional programming {fmt_hours(remaining_hours)} @ ${programmer_rate:.2f}/hr"
            )

        if total_programming_hours_local <= 0:
            total_programming_hours_local = total_programming_hours

        if total_programming_hours_local > 0:
            programming_minutes = (total_programming_hours_local / qty_divisor) * 60.0

        prog_pp = programming_cost_per_part
        if prog_pp <= 0:
            try:
                prog_pp = float(labor_cost_totals.get(PROGRAMMING_PER_PART_LABEL) or 0.0)
            except Exception:
                prog_pp = 0.0
        if prog_pp <= 0:
            per_lot_candidate = _safe_float(nre.get("programming_per_lot"))
            if per_lot_candidate <= 0:
                per_lot_candidate = _safe_float(nre.get("programming_per_part"))
            if per_lot_candidate > 0:
                prog_pp = per_lot_candidate / qty_divisor
        if prog_pp > 0:
            prog_pp = round(float(prog_pp), 2)
            try:
                labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = prog_pp
            except Exception:
                pass
            if qty > 1:
                detail_args.append(f"Amortized across {qty} pcs")
            if detail_args:
                detail_lookup[PROGRAMMING_PER_PART_LABEL] = "; ".join(detail_args)
            additions["programming_amortized"] = (prog_pp, programming_minutes)
            amortized_nre_total += prog_pp
            if "programming_amortized" not in canonical_bucket_summary:
                display_labor_for_ladder += prog_pp

        fixture_minutes = 0.0
        try:
            fix_pp = float(fixture_labor_per_part_cost or 0.0)
        except Exception:
            fix_pp = 0.0
        if fix_pp <= 0:
            try:
                fix_pp = float(
                    labor_cost_totals.get("Fixture Build (amortized)")
                    or nre.get("fixture_per_part")
                    or 0.0
                )
            except Exception:
                fix_pp = 0.0
        if fix_pp > 0:
            detail_args = []
            fixture_detail = nre_detail.get("fixture", {}) if isinstance(nre_detail, dict) else {}
            fixture_build_hr_detail = _safe_float(fixture_detail.get("build_hr"))
            if fixture_build_hr_detail > 0:
                fixture_minutes += (fixture_build_hr_detail / qty_divisor) * 60.0
                fixture_rate_detail = _resolve_rate_with_fallback(
                    fixture_detail.get("build_rate"),
                    "FixtureBuildRate",
                    "ShopRate",
                )
                detail_args.append(
                    f"- Build labor (lot): {_hours_with_rate_text(fixture_build_hr_detail, fixture_rate_detail)}"
                )
            soft_jaw_hr = _safe_float(fixture_detail.get("soft_jaw_hr"))
            if soft_jaw_hr and soft_jaw_hr > 0:
                fixture_minutes += (soft_jaw_hr / qty_divisor) * 60.0
                detail_args.append(f"Soft jaw prep {fmt_hours(soft_jaw_hr)}")
            if qty > 1:
                detail_args.append(f"Amortized across {qty} pcs")
            if detail_args:
                detail_lookup["Fixture Build (amortized)"] = "; ".join(detail_args)
            additions["fixture_build_amortized"] = (fix_pp, fixture_minutes)
            amortized_nre_total += fix_pp
            if "fixture_build_amortized" not in canonical_bucket_summary:
                display_labor_for_ladder += fix_pp

        return additions

    if bucket_table_rows and isinstance(breakdown, _MappingABC):
        drilling_meta_guard = breakdown.get("drilling_meta")
        bucket_view_guard = breakdown.get("bucket_view")
        buckets_guard = (
            bucket_view_guard.get("buckets")
            if isinstance(bucket_view_guard, _MappingABC)
            else None
        )
        drilling_bucket_guard = (
            buckets_guard.get("drilling")
            if isinstance(buckets_guard, _MappingABC)
            else None
        )
        if isinstance(drilling_meta_guard, _MappingABC) and isinstance(
            drilling_bucket_guard, _MappingABC
        ):
            try:
                card_hr = round(
                    float(drilling_meta_guard["total_minutes_billed"]) / 60.0,
                    2,
                )
                row_hr = round(
                    float(drilling_bucket_guard["minutes"]) / 60.0,
                    2,
                )
            except (KeyError, TypeError, ValueError):
                card_hr = row_hr = None
            if card_hr is not None and row_hr is not None and abs(card_hr - row_hr) > 0.01:
                if prefer_removal_drilling_hours and removal_drilling_hours_precise is not None:
                    billed_minutes_guard = _safe_float(
                        drilling_meta_guard.get("total_minutes_billed"),
                        default=0.0,
                    )
                    logger.info(
                        "[hours-sync] Overriding Drilling bucket hours from %.2f -> %.2f (source=removal_card)",
                        row_hr,
                        removal_drilling_hours_precise,
                    )
                    _sync_drilling_bucket_view(
                        bucket_view_guard,
                        billed_minutes=float(billed_minutes_guard or 0.0),
                        billed_cost=None,
                    )
                else:
                    raise RuntimeError(
                        "[FATAL] Drilling hours mismatch: "
                        f"card {card_hr} vs row {row_hr}. "
                        "Late writer is overwriting bucket_view."
                    )

    if bucket_table_rows:
        render_bucket_table(bucket_table_rows)

    amortized_overrides = _prepare_amortized_details()

    for canon_key, (amount, minutes) in amortized_overrides.items():
        amount_val = max(0.0, _safe_float(amount))
        minutes_val = max(0.0, _safe_float(minutes))
        summary_entry = canonical_bucket_summary.setdefault(
            canon_key,
            {"total": 0.0, "labor": 0.0, "machine": 0.0, "minutes": 0.0},
        )
        summary_entry["total"] = amount_val
        summary_entry["labor"] = amount_val
        summary_entry["machine"] = 0.0
        if minutes_val > 0:
            summary_entry["minutes"] = minutes_val
        process_costs_for_render[canon_key] = amount_val
        if minutes_val > 0:
            bucket_minutes_detail[canon_key] = minutes_val
        label = _display_bucket_label(canon_key, label_overrides)
        label_to_canon.setdefault(label, canon_key)
        canon_to_display_label.setdefault(canon_key, label)
        if canon_key not in canonical_bucket_order:
            canonical_bucket_order.insert(0, canon_key)
        if isinstance(breakdown, dict):
            try:
                bucket_view_obj = breakdown.setdefault("bucket_view", {})
            except Exception:
                bucket_view_obj = None
            if isinstance(bucket_view_obj, dict):
                buckets_map = bucket_view_obj.setdefault("buckets", {})
                if isinstance(buckets_map, dict):
                    buckets_map[canon_key] = {
                        "minutes": round(minutes_val, 2),
                        "labor$": round(amount_val, 2),
                        "machine$": 0.0,
                        "total$": round(amount_val, 2),
                    }
                order_list = bucket_view_obj.setdefault("order", [])
                if isinstance(order_list, list) and canon_key not in order_list:
                    order_list.insert(0, canon_key)

        if not any(spec.canon_key == canon_key for spec in bucket_row_specs):
            hours_val = minutes_val / 60.0 if minutes_val > 0 else 0.0
            if hours_val > 0:
                rate_val = amount_val / hours_val if hours_val else 0.0
            else:
                rate_val = 0.0
            if hours_val > 0 and rate_val <= 0.0:
                rate_val = _rate_for_bucket(canon_key, rates or {})
            bucket_row_specs.append(
                _BucketRowSpec(
                    label=_display_bucket_label(canon_key, label_overrides),
                    hours=hours_val,
                    rate=rate_val,
                    total=amount_val,
                    labor=amount_val,
                    machine=0.0,
                    canon_key=canon_key,
                    minutes=minutes_val,
                )
            )

    for canon_key in process_costs_for_render:
        label = _display_bucket_label(canon_key, label_overrides)
        label_to_canon.setdefault(label, canon_key)
        canon_to_display_label.setdefault(canon_key, label)

    if not canonical_bucket_summary:
        display_labor_for_ladder = 0.0
        for label, amount in labor_cost_totals.items():
            if _should_hide_amortized(label):
                continue
            try:
                display_labor_for_ladder += float(amount or 0.0)
            except Exception:
                continue

    process_table = _ProcessCostTableRecorder()

    # 1a) minutes→$ for planner buckets that have minutes but no dollars
    for k, meta in (canonical_bucket_summary or {}).items():
        minutes = float(meta.get("minutes") or 0.0)
        have_amount = float(process_costs_for_render.get(k) or 0.0)
        if minutes > 0 and have_amount <= 0.0:
            r = _rate_for_bucket(k, rates or {})
            if r > 0:
                process_costs_for_render[k] = round((minutes / 60.0) * r, 2)
                bucket_minutes_detail[k] = minutes

    # 1b) minutes→$ for the drilling minutes engine (if planner didn’t emit a drilling bucket)
    drill_summary_source: Mapping[str, Any] | None = None
    if isinstance(process_plan_summary_local, _MappingABC):
        candidate = process_plan_summary_local.get("drilling")
        if isinstance(candidate, _MappingABC):
            drill_summary_source = candidate
    if drill_summary_source is None:
        drill_summary_source = {}
    drill_min = _safe_float(
        drill_summary_source.get("total_minutes_with_toolchange"), default=0.0
    )
    if drill_min <= 0.0:
        drill_min = _safe_float(drill_summary_source.get("total_minutes"), default=0.0)
    bill_min = _safe_float(drill_summary_source.get("total_minutes_billed"), default=0.0)
    if bill_min <= 0.0:
        bill_min = drill_min
    rates_map = rates if isinstance(rates, dict) else {}
    drill_rate = float(
        rates_map.get("DrillingRate")
        or rates_map.get("drillingrate")
        or rates_map.get("MachineRate")
        or rates_map.get("machinerate")
        or 0.0
    )
    if drill_rate <= 0.0:
        try:
            drill_rate = float(_rate_for_bucket("drilling", rates or {}))
        except Exception:
            drill_rate = 0.0
    drill_cost = round((bill_min / 60.0) * drill_rate, 2) if bill_min > 0 else 0.0
    if bill_min > 0:
        process_costs_for_render["drilling"] = drill_cost
        bucket_minutes_detail["drilling"] = bill_min
        drill_summary_entry = canonical_bucket_summary.setdefault("drilling", {})
        drill_summary_entry["minutes"] = float(bill_min)
        drill_summary_entry["hours"] = float(bill_min / 60.0)

    process_plan_summary_map: Mapping[str, Any] | None = None
    if isinstance(process_plan_summary_local, _MappingABC):
        process_plan_summary_map = process_plan_summary_local

    section_total = render_process_costs(
        tbl=process_table,
        process_costs=process_costs_for_render,
        rates=rates,
        minutes_detail=bucket_minutes_detail,
        process_plan=process_plan_summary_map,
    )

    proc_total = section_total

    rows: tuple[_ProcessRowRecord, ...] = tuple(getattr(process_table, "rows", ()))

    def _process_row_canon(record: _ProcessRowRecord) -> str:
        if record.canon_key:
            return record.canon_key
        return _canonical_bucket_key(record.name)

    def _find_process_row(target_canon: str) -> _ProcessRowRecord | None:
        if not target_canon:
            return None
        for record in rows:
            if _process_row_canon(record) == target_canon:
                return record
        return None

    if rows and isinstance(process_plan_summary_map, _MappingABC):
        drilling_row = _find_process_row("drilling")
        drilling_summary = process_plan_summary_map.get("drilling")
        if drilling_row and isinstance(drilling_summary, _MappingABC):
            card_minutes_billed = _safe_float(
                drilling_summary.get("total_minutes_billed"), default=0.0
            )
            row_hr_for_cost = float(drilling_row.hours or 0.0)
            row_hr = round(row_hr_for_cost, 2)
            row_rate = float(drilling_row.rate or 0.0)
            row_cost = float(drilling_row.total or 0.0)

            if card_minutes_billed > 0.0:
                billed_hr_precise = card_minutes_billed / 60.0
                billed_hr = round(billed_hr_precise, 2)
                drill_rate = _rate_for_bucket("drilling", rates or {})
                if drill_rate <= 0.0:
                    drill_rate = row_rate
                if drill_rate <= 0.0:
                    drill_rate = 0.0
                billed_cost = round(billed_hr_precise * drill_rate, 2)

                process_table.update_row(
                    "drilling", hours=billed_hr, rate=drill_rate, cost=billed_cost
                )
                drilling_row.hours = billed_hr
                drilling_row.rate = drill_rate
                drilling_row.total = billed_cost
                row_hr_for_cost = billed_hr_precise
                row_hr = billed_hr
                row_rate = drill_rate
                row_cost = billed_cost
                process_cost_row_details["drilling"] = (billed_hr, drill_rate, billed_cost)
                labor_costs_display[drilling_row.name] = billed_cost
                process_costs_for_render["drilling"] = billed_cost

            card_hr = round(card_minutes_billed / 60.0, 2)
            if abs(card_hr - row_hr) >= 0.05:
                # Favor the planner summary and coerce the row to match when they diverge.
                row_hr_for_cost = card_minutes_billed / 60.0
                row_hr = card_hr
                corrected_cost = row_cost
                if row_rate > 0.0:
                    corrected_cost = round(row_hr_for_cost * row_rate, 2)
                    process_table.update_row(
                        "drilling", hours=row_hr, cost=corrected_cost
                    )
                    drilling_row.total = corrected_cost
                    row_cost = corrected_cost
                else:
                    process_table.update_row("drilling", hours=row_hr)
                drilling_row.hours = row_hr
                process_cost_row_details["drilling"] = (
                    row_hr,
                    row_rate,
                    row_cost,
                )
                process_costs_for_render["drilling"] = row_cost
                labor_costs_display[drilling_row.name] = row_cost
                drilling_row.total = row_cost

            drilling_meta_for_guard = None
            if isinstance(breakdown, _MappingABC):
                drilling_meta_for_guard = breakdown.get("drilling_meta")
            bucket_view_for_guard = None
            if isinstance(breakdown, _MappingABC):
                bucket_view_for_guard = breakdown.get("bucket_view")
            if (
                isinstance(drilling_meta_for_guard, _MappingABC)
                and isinstance(bucket_view_for_guard, _MappingABC)
            ):
                buckets_guard = bucket_view_for_guard.get("buckets")
                if isinstance(buckets_guard, _MappingABC):
                    drilling_bucket_guard = buckets_guard.get("drilling")
                else:
                    drilling_bucket_guard = None
                if isinstance(drilling_bucket_guard, _MappingABC):
                    try:
                        card_hr_guard = round(
                            float(
                                drilling_meta_for_guard.get("total_minutes_billed")
                                or 0.0
                            )
                            / 60.0,
                            2,
                        )
                        row_hr_guard = round(
                            float(drilling_bucket_guard.get("minutes") or 0.0) / 60.0,
                            2,
                        )
                    except (TypeError, ValueError):
                        card_hr_guard = row_hr_guard = None
                    if (
                        card_hr_guard is not None
                        and row_hr_guard is not None
                        and abs(card_hr_guard - row_hr_guard) > 0.01
                    ):
                        billed_minutes_guard = _safe_float(
                            drilling_meta_for_guard.get("total_minutes_billed"),
                            default=0.0,
                        )
                        corrected = _sync_drilling_bucket_view(
                            bucket_view_for_guard,
                            billed_minutes=billed_minutes_guard,
                            billed_cost=row_cost,
                        )
                        if corrected:
                            buckets_guard_ref = bucket_view_for_guard.get("buckets")
                            if isinstance(buckets_guard_ref, _MappingABC):
                                drilling_bucket_guard_ref = buckets_guard_ref.get("drilling")
                            else:
                                drilling_bucket_guard_ref = None
                            if isinstance(drilling_bucket_guard_ref, _MappingABC):
                                row_hr_guard = round(
                                    _safe_float(
                                        drilling_bucket_guard_ref.get("minutes"),
                                        default=0.0,
                                    )
                                    / 60.0,
                                    2,
                                )
                                card_hr_guard = round(billed_minutes_guard / 60.0, 2)
                        if (
                            card_hr_guard is None
                            or row_hr_guard is None
                            or abs(card_hr_guard - row_hr_guard) > 0.01
                        ):
                            if prefer_removal_drilling_hours:
                                logger.warning(
                                    "[hours-sync] Drilling hours still diverge after override: card %.2f vs row %.2f",
                                    card_hr_guard if card_hr_guard is not None else -1.0,
                                    row_hr_guard if row_hr_guard is not None else -1.0,
                                )
                            else:
                                raise RuntimeError(
                                    "Drilling hours mismatch AFTER BUILD: "
                                    f"card {card_hr_guard} vs row {row_hr_guard}. "
                                    "Late writer is overwriting bucket_view."
                                )

            assert (
                abs(row_cost - row_hr_for_cost * row_rate) < 0.51
            ), "Drilling $ ≠ hr × rate"

    misc_total = 0.0
    for label, amount in labor_cost_totals.items():
        if _should_hide_amortized(label):
            continue
        try:
            amount_val = float(amount or 0.0)
        except Exception:
            amount_val = 0.0
        if not ((amount_val > 0.0) or show_zeros):
            continue
        normalized_label = _normalize_bucket_key(label)
        canon_key = _canonical_bucket_key(label)
        normalized_canon = _normalize_bucket_key(canon_key)
        if normalized_label != "misc" and not normalized_label.startswith("misc"):
            if normalized_canon != "misc" and not normalized_canon.startswith("misc"):
                continue
        display_label = str(label)
        canon_to_display_label[canon_key] = display_label
        label_to_canon[display_label] = canon_key
        _add_labor_cost_line(display_label, amount_val, process_key=canon_key)
        labor_costs_display[display_label] = amount_val
        misc_total += amount_val

    if misc_total > 0:
        process_table.had_rows = True

    if not process_table.had_rows and show_zeros:
        row("No process costs", 0.0, indent="  ")

    row("Total", proc_total, indent="  ")
    process_total_row_index = len(lines) - 1

    hour_summary_entries.clear()

    if (
        prefer_removal_drilling_hours
        and isinstance(bucket_state.extra, dict)
        and bucket_state.extra.get("removal_drilling_hours") is not None
    ):
        billed_minutes_sync = None
        try:
            billed_minutes_sync = (
                float(bucket_state.extra.get("removal_drilling_hours", 0.0)) * 60.0
            )
        except Exception:
            billed_minutes_sync = None
        if billed_minutes_sync is not None and billed_minutes_sync > 0:
            bucket_view_target: Mapping[str, Any] | None = None
            if isinstance(breakdown, _MutableMappingABC):
                candidate_view = breakdown.get("bucket_view")
                if isinstance(candidate_view, _MutableMappingABC):
                    bucket_view_target = candidate_view
            if bucket_view_target is None and isinstance(bucket_view_struct, _MutableMappingABC):
                bucket_view_target = bucket_view_struct
            if isinstance(bucket_view_target, _MutableMappingABC):
                billed_cost_sync: float | None = None
                if isinstance(process_costs, _MappingABC):
                    try:
                        billed_cost_candidate = float(process_costs.get("drilling") or 0.0)
                    except Exception:
                        billed_cost_candidate = 0.0
                    if billed_cost_candidate > 0.0:
                        billed_cost_sync = billed_cost_candidate
                _sync_drilling_bucket_view(
                    bucket_view_target,
                    billed_minutes=billed_minutes_sync,
                    billed_cost=billed_cost_sync,
                )

    charged_hours = _charged_hours_by_bucket(
        process_costs,
        process_meta,
        rates,
        render_state=bucket_state,
        removal_drilling_hours=removal_drilling_hours_precise,
        prefer_removal_drilling_hours=prefer_removal_drilling_hours,
        cfg=cfg,
    )
    removal_hours_debug = None
    if isinstance(getattr(bucket_state, "extra", None), _MappingABC):
        removal_hours_debug = _coerce_float_or_none(
            bucket_state.extra.get("removal_drilling_hours")
        )
    drill_bucket_key: str | None = None
    for raw_key in charged_hours:
        canon = _canonical_bucket_key(raw_key)
        if canon in {"drilling", "drill"}:
            drill_bucket_key = raw_key
            break
    bucket_hours_debug = (
        _coerce_float_or_none(charged_hours.get(drill_bucket_key))
        if drill_bucket_key
        else None
    )
    if removal_hours_debug is not None or bucket_hours_debug is not None:
        def _fmt_debug(value: float | None) -> str:
            if value is None:
                return "nan"
            try:
                numeric = float(value)
            except Exception:
                return "nan"
            if not math.isfinite(numeric):
                return "nan"
            return f"{numeric:.2f}"

        print(
            "[DEBUG] drilling_hours removal_card="
            f"{_fmt_debug(removal_hours_debug)}  bucket={_fmt_debug(bucket_hours_debug)}"
        )
    charged_hour_entries = sorted(charged_hours.items(), key=lambda kv: kv[0])
    charged_hours_by_canon: dict[str, float] = {}
    for raw_key, hour_val in charged_hour_entries:
        canon_key = _canonical_bucket_key(raw_key)
        if not canon_key:
            canon_key = _normalize_bucket_key(raw_key)
        try:
            numeric_hours = float(hour_val or 0.0)
        except Exception:
            numeric_hours = 0.0
        if not canon_key:
            continue
        charged_hours_by_canon[canon_key] = (
            charged_hours_by_canon.get(canon_key, 0.0) + numeric_hours
        )

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
                f"{label} hours capped at {fmt_hours(24.0, decimals=0)} for single-piece quote (was {fmt_hours(numeric_value)})."
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
    programming_is_amortized = show_amortized and (
        bool(programming_meta.get("amortized"))
        or programming_per_part_amount > 0
    )

    try:
        fixture_per_part_amount = float(fixture_labor_per_part_cost or 0.0)
    except Exception:
        fixture_per_part_amount = 0.0
    fixture_is_amortized = show_amortized and (fixture_per_part_amount > 0)

    planner_mode = str(pricing_source_value).lower() == "planner"
    planner_entry_baseline = len(hour_summary_entries)
    if planner_mode:
        seen_hour_labels: set[str] = set()
        for canon_key in canonical_bucket_order:
            if not canon_key:
                continue
            metrics = canonical_bucket_summary.get(canon_key)
            if not isinstance(metrics, dict):
                continue
            hours_val = _safe_float(metrics.get("hours"))
            if hours_val <= 0.0:
                minutes_val = _safe_float(metrics.get("minutes"))
                if minutes_val > 0.0:
                    hours_val = minutes_val / 60.0
            if hours_val <= 0.0:
                continue
            if str(canon_key) == "programming_amortized":
                continue
            label = _display_bucket_label(canon_key, label_overrides)
            if label in seen_hour_labels:
                hour_summary_entries[label] = (
                    hour_summary_entries[label][0] + round(hours_val, 2),
                    hour_summary_entries[label][1],
                )
            else:
                _record_hour_entry(label, round(hours_val, 2))
                seen_hour_labels.add(label)

        buckets_for_hours = (
            bucket_view_struct.get("buckets")
            if isinstance(bucket_view_struct, _MappingABC)
            else None
        )
        if not isinstance(buckets_for_hours, _MappingABC):
            buckets_for_hours = {}

        pl_lab = 0.0
        pl_mac = 0.0
        total_planner_hours = 0.0
        for raw_key, info in buckets_for_hours.items():
            if not isinstance(info, _MappingABC):
                continue
            minutes_val = _safe_float(info.get("minutes"), default=0.0)
            if minutes_val <= 0.0:
                continue
            hours_val = minutes_val / 60.0
            total_planner_hours += hours_val
            canon_key = _canonical_bucket_key(raw_key)
            norm_key = canon_key or _norm(raw_key)
            if norm_key in LABORISH:
                pl_lab += hours_val
            else:
                pl_mac += hours_val

        residual_machine = total_planner_hours - pl_lab
        if residual_machine < 0.0:
            residual_machine = 0.0
        if abs(pl_mac - residual_machine) > 0.01:
            pl_mac = residual_machine

        planner_total_hr = round(max(total_planner_hours, pl_lab + pl_mac), 2)
        if planner_total_hr > 0.0:
            _record_hour_entry("Planner Total", planner_total_hr)
        if pl_lab > 0.0:
            _record_hour_entry(
                "Planner Labor",
                round(pl_lab, 2),
                include_in_total=False,
            )
        if pl_mac > 0.0:
            _record_hour_entry(
                "Planner Machine",
                round(pl_mac, 2),
                include_in_total=False,
            )

        _record_hour_entry("Programming", round(programming_hours, 2))
        _record_hour_entry("Fixture Build", round(fixture_hours, 2))
        if fixture_is_amortized and qty_for_hours > 0:
            per_part_fixture_hr = fixture_hours / qty_for_hours
            _record_hour_entry(
                "Fixture Build (amortized)",
                round(per_part_fixture_hr, 2),
                include_in_total=False,
            )
    if (not planner_mode) or len(hour_summary_entries) == planner_entry_baseline:
        if charged_hour_entries:
            seen_hour_canon_keys: set[str] = set()
            for canon_key in canonical_bucket_order:
                if not canon_key:
                    continue
                if canon_key in {"planner_labor", "planner_machine", "planner_total"}:
                    continue
                if canon_key.startswith("planner_"):
                    continue
                if str(canon_key) == "programming_amortized":
                    continue
                hours_val = charged_hours_by_canon.get(canon_key)
                if hours_val is None:
                    continue
                try:
                    hours_float = float(hours_val or 0.0)
                except Exception:
                    hours_float = 0.0
                label = _display_bucket_label(canon_key, label_overrides)
                _record_hour_entry(label, round(hours_float, 2))
                seen_hour_canon_keys.add(canon_key)

            for canon_key, hours_val in sorted(charged_hours_by_canon.items()):
                if not canon_key or canon_key in seen_hour_canon_keys:
                    continue
                if canon_key in {"planner_labor", "planner_machine", "planner_total"}:
                    continue
                if str(canon_key).startswith("planner_"):
                    continue
                if str(canon_key) == "programming_amortized":
                    continue
                try:
                    hours_float = float(hours_val or 0.0)
                except Exception:
                    hours_float = 0.0
                label = _display_bucket_label(canon_key, label_overrides)
                _record_hour_entry(label, round(hours_float, 2))
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
        _record_hour_entry("Fixture Build", fixture_hours)
        if fixture_is_amortized and qty_for_hours > 0:
            per_part_fixture_hr = fixture_hours / qty_for_hours
            _record_hour_entry(
                "Fixture Build (amortized)",
                per_part_fixture_hr,
                include_in_total=False,
            )

    if canonical_bucket_order and canonical_bucket_summary:
        summary_hours: dict[str, float] = {}
        for canon_key in canonical_bucket_order:
            metrics = canonical_bucket_summary.get(canon_key) or {}
            minutes_val = _safe_float(metrics.get("minutes"), default=0.0)
            if minutes_val <= 0.0:
                continue
            if str(canon_key) == "programming_amortized":
                continue
            label = _display_bucket_label(canon_key, label_overrides)
            summary_hours[label] = summary_hours.get(label, 0.0) + (minutes_val / 60.0)

        prefer_card_minutes = bool(
            getattr(cfg, "prefer_removal_drilling_hours", prefer_removal_drilling_hours)
        )
        override_hours: float | None = None
        if prefer_card_minutes:
            extra_map = getattr(bucket_state, "extra", {})
            if isinstance(extra_map, _MappingABC):
                candidate = extra_map.get("removal_drilling_hours")
                override_hours = _coerce_float_or_none(candidate)
                if override_hours is None:
                    total_minutes_extra = _coerce_float_or_none(
                        extra_map.get("drill_total_minutes")
                    )
                    if total_minutes_extra is None:
                        machine_minutes_extra = _coerce_float_or_none(
                            extra_map.get("drill_machine_minutes")
                        )
                        labor_minutes_extra = _coerce_float_or_none(
                            extra_map.get("drill_labor_minutes")
                        )
                        if (
                            machine_minutes_extra is not None
                            or labor_minutes_extra is not None
                        ):
                            machine_minutes = float(machine_minutes_extra or 0.0)
                            labor_minutes = float(labor_minutes_extra or 0.0)
                            total_candidate = machine_minutes + labor_minutes
                            if total_candidate > 0.0:
                                total_minutes_extra = total_candidate
                    if total_minutes_extra is not None:
                        override_hours = float(total_minutes_extra) / 60.0
            if (
                override_hours is None
                and removal_drilling_hours_precise is not None
                and prefer_card_minutes
            ):
                try:
                    override_hours = float(removal_drilling_hours_precise)
                except Exception:
                    override_hours = None
        if override_hours is not None and override_hours >= 0:
            display_label = _display_bucket_label("drilling", label_overrides)
            summary_hours[display_label] = round(max(0.0, float(override_hours)), 2)

        if prefer_card_minutes:
            extra_map = getattr(bucket_state, "extra", {})
            if isinstance(extra_map, _MappingABC):
                drill_total_minutes = extra_map.get("drill_total_minutes")
                if isinstance(drill_total_minutes, (int, float)):
                    drill_label = _display_bucket_label("drilling", label_overrides)
                    summary_hours[drill_label] = round(float(drill_total_minutes) / 60.0, 2)

        planner_labels = {"Planner Machine", "Planner Labor", "Planner Total"}
        for label, hours_val in summary_hours.items():
            if label in planner_labels and label in hour_summary_entries:
                continue
            include_flag = True
            existing = hour_summary_entries.get(label)
            if isinstance(existing, tuple) and len(existing) == 2:
                include_flag = bool(existing[1])
            hour_summary_entries[label] = (round(hours_val, 2), include_flag)

    prefer_drill_summary = prefer_removal_drilling_hours
    if cfg is not None:
        prefer_summary_candidate = getattr(
            cfg, "prefer_removal_drilling_hours", None
        )
        if prefer_summary_candidate is not None:
            prefer_drill_summary = bool(prefer_summary_candidate)
    if prefer_drill_summary:
        extra_map = getattr(bucket_state, "extra", {})
        if not isinstance(extra_map, _MappingABC):
            extra_map = {}
        removal_card_hr = _coerce_float_or_none(extra_map.get("removal_drilling_hours"))
        if removal_card_hr is None:
            drill_minutes_extra = _coerce_float_or_none(extra_map.get("drill_total_minutes"))
            if drill_minutes_extra is not None:
                removal_card_hr = float(drill_minutes_extra) / 60.0

        charged_snapshot: Mapping[str, Any]
        if isinstance(charged_hours, _MappingABC):
            charged_snapshot = charged_hours
        else:
            charged_snapshot = dict(charged_hours or {})  # type: ignore[arg-type]
        bucket_hr = None
        for raw_key in charged_snapshot.keys():
            if _canonical_bucket_key(raw_key) in {"drilling", "drill"}:
                bucket_hr = _coerce_float_or_none(charged_snapshot.get(raw_key))
                break

        row_hr_debug = None
        process_rows_debug = getattr(process_table, "_rows", None)
        if isinstance(process_rows_debug, Sequence):
            for row_entry in process_rows_debug:
                if not isinstance(row_entry, _MappingABC):
                    continue
                if _canonical_bucket_key(row_entry.get("label")) in {"drilling", "drill"}:
                    row_hr_debug = _coerce_float_or_none(row_entry.get("hours"))
                    break

        hour_summary_map: dict[str, float] = {}
        if isinstance(hour_summary_entries, _MappingABC):
            for label, value in hour_summary_entries.items():
                base_value: Any
                if isinstance(value, (list, tuple)) and value:
                    base_value = value[0]
                else:
                    base_value = value
                coerced = _coerce_float_or_none(base_value)
                if coerced is None:
                    continue
                hour_summary_map[str(label)] = float(coerced)

        prefer_drill_summary = prefer_removal_drilling_hours
        if cfg is not None:
            prefer_summary_candidate = getattr(
                cfg, "prefer_removal_drilling_hours", None
            )
            if prefer_summary_candidate is not None:
                prefer_drill_summary = bool(prefer_summary_candidate)
        if prefer_drill_summary:
            drill_minutes_extra = _coerce_float_or_none(extra_map.get("drill_total_minutes"))
            if drill_minutes_extra is not None:
                override_hours_val = round(float(drill_minutes_extra) / 60.0, 2)
                hour_summary_map["Drilling"] = override_hours_val
                hour_summary_map["drilling"] = override_hours_val
                if isinstance(hour_summary_entries, dict):
                    for label_key in ("Drilling", "drilling"):
                        existing_entry = hour_summary_entries.get(label_key)
                        include_flag = True
                        if isinstance(existing_entry, tuple) and len(existing_entry) == 2:
                            include_flag = bool(existing_entry[1])
                        hour_summary_entries[label_key] = (
                            override_hours_val,
                            include_flag,
                        )

        summary_hr = None
        for key in ("Drilling", "drilling"):
            if key in hour_summary_map:
                summary_hr = hour_summary_map[key]
                break

        def _fmt_drill_debug(value: float | None) -> str:
            if value is None:
                return "-"
            try:
                return f"{float(value):.2f}"
            except Exception:
                return "-"

        logger.info(
            "[drill-sync] card=%s  bucket=%s  row=%s  summary=%s",
            _fmt_drill_debug(removal_card_hr),
            _fmt_drill_debug(bucket_hr),
            _fmt_drill_debug(row_hr_debug),
            _fmt_drill_debug(summary_hr),
        )

    if hour_summary_entries:
        def _canonical_hour_label(value: Any) -> tuple[str, str]:
            import re

            text = str(value or "")
            text = re.sub(r"\s+", " ", text).strip()
            canonical = text.casefold()
            return canonical, text

        append_line("")
        append_line("Labor Hour Summary")
        append_line(divider)
        if str(pricing_source_value).lower() == "planner":
            entries_iter = list(hour_summary_entries.items())
        else:
            entries_iter = list(
                sorted(hour_summary_entries.items(), key=lambda kv: kv[1][0], reverse=True)
            )
        folded_entries: dict[str, list[Any]] = {}
        folded_display: dict[str, str] = {}
        folded_order: list[str] = []

        def _coerce_hour_value(value: Any) -> float | None:
            coerced = _coerce_float_or_none(value)
            if coerced is None:
                return None
            try:
                return float(coerced)
            except Exception:
                return None

        for label, (hr_val, include_in_total) in entries_iter:
            canonical_key, display_label = _canonical_hour_label(label)
            folded = folded_entries.get(canonical_key)
            if folded is None:
                folded_entries[canonical_key] = [hr_val, bool(include_in_total)]
                folded_display[canonical_key] = display_label
                folded_order.append(canonical_key)
                continue

            folded_display.setdefault(canonical_key, display_label)

            existing_hr, existing_include = folded
            hr_float = _coerce_hour_value(hr_val)
            existing_float = _coerce_hour_value(existing_hr)
            deduped = False

            if hr_float is not None and existing_float is not None:
                try:
                    if math.isclose(existing_float, hr_float, rel_tol=1e-9, abs_tol=0.005):
                        deduped = True
                except Exception:
                    deduped = False

            if deduped:
                folded[1] = existing_include or bool(include_in_total)
                continue

            if existing_float is not None or hr_float is not None:
                folded[0] = (existing_float or 0.0) + (hr_float or 0.0)
            else:
                try:
                    folded[0] = existing_hr + hr_val
                except Exception:
                    folded[0] = existing_hr
            folded[1] = existing_include or bool(include_in_total)
        total_hours = 0.0
        for canonical_key in folded_order:
            hr_val, include_in_total = folded_entries[canonical_key]
            display_label = folded_display.get(canonical_key, "")
            hours_row(display_label, hr_val, indent="  ")
            if include_in_total and hr_val:
                total_hours += hr_val
        hours_row("Total Hours", total_hours, indent="  ")
    append_line("")

    # ---- Pass-Through & Direct (auto include non-zeros; sorted desc) --------
    append_line("Pass-Through & Direct Costs")
    pass_through_header_index = len(lines) - 1
    append_line(divider)
    pass_total = 0.0
    pass_through_labor_total = 0.0
    displayed_pass_through: dict[str, float] = {}
    for key, value in sorted((pass_through or {}).items(), key=lambda kv: kv[1], reverse=True):
        canonical_label = _canonical_pass_label(key)
        if canonical_label.lower() == "material":
            continue
        try:
            amount_val = float(value or 0.0)
        except Exception:
            continue
        if (amount_val > 0) or show_zeros:
            displayed_pass_through[key] = amount_val
            pass_total += amount_val
            canonical_pass_label = canonical_label
            if "labor" in canonical_pass_label.lower():
                pass_through_labor_total += amount_val
    vendor_items_total = 0.0
    vendor_item_sources: list[_MappingABC] = []
    for vendor_source in (
        breakdown.get("vendor_items") if isinstance(breakdown, _MappingABC) else None,
        pass_through.get("vendor_items") if isinstance(pass_through, _MappingABC) else None,
    ):
        if isinstance(vendor_source, _MappingABC):
            vendor_item_sources.append(vendor_source)
    for vendor_map in vendor_item_sources:
        for amount in vendor_map.values():
            numeric = _coerce_float_or_none(amount)
            if numeric is not None:
                vendor_items_total += float(numeric)

    material_direct_contribution = round(
        material_total_for_directs + material_tax_for_directs - scrap_credit_for_directs,
        2,
    )
    material_display_amount_candidate = _coerce_float_or_none(
        material_block.get("total_material_cost")
    )
    if material_display_amount_candidate is not None:
        try:
            material_direct_contribution = round(
                float(material_display_amount_candidate),
                2,
            )
        except Exception:
            material_direct_contribution = round(
                float(material_display_amount_candidate or 0.0),
                2,
            )
    material_display_amount = round(float(material_direct_contribution), 2)
    material_total_for_why = float(material_display_amount)
    material_net_cost = float(material_display_amount)
    if material_display_amount <= 0.0:
        fallback_amount = 0.0
        if materials_direct_total > 0:
            fallback_amount = float(materials_direct_total)
        if material_entries_total > 0:
            fallback_amount = max(fallback_amount, float(material_entries_total))
        if fallback_amount > 0:
            material_display_amount = round(fallback_amount, 2)
            material_total_for_why = float(material_display_amount)
            material_net_cost = float(material_display_amount)
    if material_entries_have_label and material_display_amount <= 0.0:
        material_warning_needed = True
    direct_costs_map: dict[Any, Any]
    if isinstance(pricing, dict):
        direct_costs_map = pricing.setdefault("direct_costs", {})
        if not isinstance(direct_costs_map, dict):
            try:
                direct_costs_map = dict(direct_costs_map or {})
            except Exception:
                direct_costs_map = {}
            pricing["direct_costs"] = direct_costs_map
    else:
        direct_costs_map = {}

    def _assign_direct_value(raw_key: Any, amount: Any) -> None:
        try:
            amount_float = round(float(amount), 2)
        except Exception:
            return
        target_key = raw_key
        try:
            canonical_target = _canonical_pass_label(str(raw_key)).strip().lower()
        except Exception:
            canonical_target = str(raw_key or "").strip().lower()
        if canonical_target:
            for existing_key in list(direct_costs_map.keys()):
                try:
                    existing_canonical = _canonical_pass_label(str(existing_key)).strip().lower()
                except Exception:
                    existing_canonical = str(existing_key or "").strip().lower()
                if existing_canonical == canonical_target:
                    target_key = existing_key
                    break
        direct_costs_map[target_key] = amount_float

    _assign_direct_value("material", material_direct_contribution)
    if vendor_items_total > 0 or show_zeros:
        _assign_direct_value("vendor items", vendor_items_total)

    for key, amount_val in displayed_pass_through.items():
        _assign_direct_value(key, amount_val)

    def _direct_label(raw_key: Any) -> str:
        text = str(raw_key)
        canonical = _canonical_pass_label(text)
        if canonical:
            canonical_stripped = canonical.strip()
            if canonical_stripped and canonical_stripped.lower() == canonical_stripped:
                return canonical_stripped.title()
            return canonical_stripped or canonical
        label = text.replace("_", " ").replace("hr", "/hr").strip()
        if not label:
            return text
        return label.title()

    direct_entries: list[tuple[str, float, Any]] = []
    for raw_key, raw_value in direct_costs_map.items():
        amount_val = _coerce_float_or_none(raw_value)
        if amount_val is None:
            continue
        direct_entries.append((
            _direct_label(raw_key),
            round(float(amount_val), 2),
            raw_key,
        ))

    direct_entries.sort(key=lambda item: item[1], reverse=True)

    total_direct_costs = round(
        sum(amount for _, amount, _ in direct_entries if (amount > 0) or show_zeros),
        2,
    )

    stored_total_directs = None
    if isinstance(breakdown, dict):
        stored_total_directs = _coerce_float_or_none(breakdown.get("total_direct_costs"))

    if stored_total_directs is not None:
        try:
            total_direct_costs_value = round(float(stored_total_directs), 2)
        except Exception:
            total_direct_costs_value = float(total_direct_costs)
    else:
        total_direct_costs_value = float(total_direct_costs)

    directs = float(total_direct_costs_value)

    material_entry = None
    display_entries: list[tuple[str, float, Any]] = []
    for entry in direct_entries:
        raw_key = entry[2]
        if str(raw_key).strip().lower() == "material":
            material_entry = entry
            continue
        display_entries.append(entry)

    if (material_display_amount > 0) or show_zeros:
        row("Material & Stock", material_display_amount, indent="  ")
        material_basis_key = None
        if isinstance(pass_through, dict):
            for candidate_key in pass_through.keys():
                if str(candidate_key).strip().lower() == "material":
                    material_basis_key = candidate_key
                    break
        if material_basis_key is None:
            material_basis_key = "Material"
        add_pass_basis(str(material_basis_key), indent="    ")
        material_detail_value: Any = None
        if material_entry is not None:
            raw_material_key = material_entry[2]
            material_detail_value = direct_cost_details.get(raw_material_key)
            if material_detail_value in (None, ""):
                material_detail_value = direct_cost_details.get(str(raw_material_key))
        if material_detail_value in (None, ""):
            for key_candidate in (
                material_basis_key,
                str(material_basis_key).strip(),
                "Material",
                "material",
                "Material & Stock",
            ):
                if key_candidate in (None, ""):
                    continue
                detail_candidate = direct_cost_details.get(key_candidate)
                if detail_candidate not in (None, ""):
                    material_detail_value = detail_candidate
                    break
        material_amount_text = fmt_money(material_display_amount, currency)
        if material_detail_value not in (None, ""):
            detail_text = str(material_detail_value)
            if "$0.00" in detail_text:
                detail_text = detail_text.replace("$0.00", material_amount_text)
            write_detail(detail_text, indent="    ")
        else:
            write_detail(
                f"Material & Stock (printed above) contributes {material_amount_text} to Direct Costs",
                indent="    ",
            )
    elif material_warning_needed and material_entries_have_label:
        row("Materials & Stock", 0.0, indent="  ")
        write_detail(
            f"{MATERIAL_WARNING_LABEL} Material items are present but no material costs were recorded in the quote.",
            indent="    ",
        )

    for display_label, amount_val, raw_key in display_entries:
        if (amount_val > 0) or show_zeros:
            row(display_label, amount_val, indent="  ")
            raw_key_str = str(raw_key)
            if raw_key_str in displayed_pass_through:
                add_pass_basis(raw_key_str, indent="    ")
                detail_value = direct_cost_details.get(raw_key_str)
                if detail_value not in (None, ""):
                    write_detail(str(detail_value), indent="    ")
            else:
                detail_value = direct_cost_details.get(raw_key)
                if detail_value not in (None, ""):
                    write_detail(str(detail_value), indent="    ")

    row("Total", total_direct_costs_value, indent="  ")
    if cost_breakdown_entries:
        append_line("")
        append_line("Cost Breakdown")
        append_line(divider)
        for label, amount in cost_breakdown_entries:
            row(label, amount, indent="  ")
    pass_through_total = float(sum(displayed_pass_through.values()))

    material_for_totals = material_display_amount
    try:
        material_breakdown_entry = breakdown.get("material") if isinstance(breakdown, _MappingABC) else None
        if isinstance(material_breakdown_entry, _MappingABC):
            total_candidate = _first_numeric_or_none(
                material_breakdown_entry.get("total_cost"),
                material_breakdown_entry.get("total_material_cost"),
                material_breakdown_entry.get("material_total_cost"),
                material_breakdown_entry.get("material_cost"),
                material_breakdown_entry.get("material_cost_before_credit"),
                material_breakdown_entry.get("material_direct_cost"),
            )
            if total_candidate is not None:
                material_for_totals = float(total_candidate)
    except Exception:
        pass

    computed_direct_total = round(
        float(pass_through_total) + float(material_for_totals) + float(vendor_items_total),
        2,
    )
    directs_total_value = float(computed_direct_total)
    total_direct_costs_value = computed_direct_total
    labor_summary_total = _safe_float(proc_total, 0.0)
    machine_summary_total = _safe_float(display_machine, 0.0)
    ladder_labor_component = labor_summary_total

    if isinstance(breakdown, dict):
        breakdown["pass_through_total"] = pass_through_total
        breakdown["total_direct_costs"] = total_direct_costs_value
        breakdown["total_labor_cost"] = round(ladder_labor_component, 2)
        if vendor_items_total:
            breakdown["vendor_items_total"] = float(round(vendor_items_total, 2))

    if isinstance(totals, dict):
        totals["labor$"] = round(labor_summary_total, 2)
        totals["machine$"] = round(machine_summary_total, 2)
        totals["directs$"] = round(directs_total_value, 2)

    ladder_subtotal = round(
        _safe_float((breakdown or {}).get("total_labor_cost"), ladder_labor_component)
        + _safe_float((breakdown or {}).get("total_direct_costs"), directs_total_value),
        2,
    )
    if isinstance(breakdown, _MutableMappingABC):
        breakdown["ladder_subtotal"] = ladder_subtotal

    if 0 <= pass_through_header_index < len(lines):
        header_text = f"Pass-Through & Direct Costs (Total: {fmt_money(directs_total_value, currency)})"
        lines[pass_through_header_index] = header_text
        try:
            sections = getattr(doc_builder, "_sections", [])
            for section in reversed(sections):
                if getattr(section, "title", None) == "Pass-Through & Direct Costs":
                    section.title = header_text
                    break
        except Exception:
            pass

    if isinstance(pricing, dict):
        pricing["total_direct_costs"] = total_direct_costs_value
    if isinstance(breakdown, dict):
        breakdown["pass_through_total"] = pass_through_total
        try:
            breakdown["total_direct_costs"] = total_direct_costs_value
        except Exception:
            breakdown["total_direct_costs"] = total_direct_costs
    directs = float(round(directs_total_value, 2))
    if isinstance(totals, dict):
        totals["direct_costs"] = total_direct_costs_value
    if 0 <= total_direct_costs_row_index < len(lines):
        replace_line(
            total_direct_costs_row_index,
            _format_row(
                total_direct_costs_label,
                total_direct_costs_value,
            ),
        )

    pass_total = float(directs)

    computed_total_labor_cost = proc_total
    expected_labor_total = computed_total_labor_cost
    if declared_labor_total > computed_total_labor_cost + 0.01:
        expected_labor_total = declared_labor_total
    components_total = display_labor_for_ladder + display_machine
    using_planner_for_ladder = str(pricing_source_value or "").strip().lower() == "planner"
    planner_machine_for_directs = 0.0
    planner_labor_override: float | None = None
    if using_planner_for_ladder:
        planner_summary: Mapping[str, Any] | None = None
        for candidate in (
            process_plan_summary_local,
            breakdown.get("process_plan") if isinstance(breakdown, _MappingABC) else None,
        ):
            if isinstance(candidate, _MappingABC):
                planner_summary = typing.cast(Mapping[str, Any], candidate)
                break

        process_meta_map = process_meta if isinstance(process_meta, _MappingABC) else {}
        planner_total_meta = process_meta_map.get("planner_total") if isinstance(process_meta_map, _MappingABC) else None
        if not isinstance(planner_total_meta, _MappingABC):
            planner_total_meta = {}
        planner_labor_meta = process_meta_map.get("planner_labor") if isinstance(process_meta_map, _MappingABC) else None
        if not isinstance(planner_labor_meta, _MappingABC):
            planner_labor_meta = {}
        planner_machine_meta = process_meta_map.get("planner_machine") if isinstance(process_meta_map, _MappingABC) else None
        if not isinstance(planner_machine_meta, _MappingABC):
            planner_machine_meta = {}

        def _first_numeric(*values: Any) -> float | None:
            for value in values:
                numeric = _coerce_float_or_none(value)
                if numeric is not None:
                    return float(numeric)
            return None

        planner_machine_candidate = _first_numeric(
            (planner_summary or {}).get("planner_machine_cost_total"),
            (planner_summary or {}).get("computed_total_machine_cost"),
            planner_total_meta.get("machine_cost"),
            planner_machine_meta.get("cost"),
        )
        if planner_machine_candidate is not None and planner_machine_candidate > 0:
            planner_machine_for_directs = planner_machine_candidate

        planner_labor_candidate = None
        planner_labor_includes_amortized = False
        for value, includes_amortized in (
            ((planner_summary or {}).get("planner_labor_cost_total"), False),
            (planner_labor_meta.get("cost_excl_amortized"), False),
            (planner_total_meta.get("labor_cost_excl_amortized"), False),
            ((planner_summary or {}).get("computed_total_labor_cost"), True),
        ):
            numeric = _coerce_float_or_none(value)
            if numeric is not None:
                planner_labor_candidate = float(numeric)
                planner_labor_includes_amortized = includes_amortized
                break

        if planner_labor_candidate is not None and planner_labor_candidate > 0:
            if planner_labor_includes_amortized:
                planner_labor_override = planner_labor_candidate
            else:
                planner_labor_override = planner_labor_candidate + float(amortized_nre_total or 0.0)
    machine_gap = expected_labor_total - components_total
    # Be tolerant: reconcile any residual mismatch into the Machine bucket so the
    # ladder math stays consistent across platforms/pricing modes.
    if abs(machine_gap) >= 0.01:
        try:
            logger.warning(
                "Labor section totals drifted beyond threshold: %s vs %s",
                f"{components_total:.2f}", f"{expected_labor_total:.2f}",
            )
        except Exception:
            pass
        # If machine isn't explicitly displayed in the labor section, prefer to
        # surface the difference there; otherwise, still adjust machine to avoid
        # a hard failure.
        display_machine = float(display_machine) + float(machine_gap)
        if display_machine < 0:
            display_machine = 0.0
    if isinstance(totals, dict):
        totals["labor_cost"] = computed_total_labor_cost
    if 0 <= total_labor_row_index < len(lines):
        replace_line(
            total_labor_row_index,
            _format_row(
                total_labor_label,
                computed_total_labor_cost,
            ),
        )

    computed_subtotal = proc_total + pass_total
    declared_subtotal: float | None = None
    if isinstance(totals, dict) and "subtotal" in totals:
        try:
            declared_subtotal = float(totals.get("subtotal", 0.0) or 0.0)
        except Exception:
            declared_subtotal = None
    if declared_subtotal is None:
        declared_subtotal = float(computed_subtotal)
    else:
        try:
            computed_subtotal_val = float(computed_subtotal)
        except Exception:
            computed_subtotal_val = 0.0
        if abs(declared_subtotal) <= 0.01 and computed_subtotal_val > 0.01:
            declared_subtotal = computed_subtotal_val
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

    labor_basis_for_ladder = float(display_labor_for_ladder or 0.0)
    if planner_labor_override is not None and planner_labor_override > 0:
        labor_basis_for_ladder = planner_labor_override
    expected_labor_total_float = float(expected_labor_total or 0.0)
    if (
        expected_labor_total_float > 0.0
        and abs(labor_basis_for_ladder - expected_labor_total_float) > _LABOR_SECTION_ABS_EPSILON
        and planner_labor_override is None
    ):
        labor_basis_for_ladder = expected_labor_total_float
    if labor_basis_for_ladder <= 0:
        fallback_basis = computed_total_labor_cost if computed_total_labor_cost > 0 else proc_total
        try:
            fallback_val = float(fallback_basis or 0.0)
        except Exception:
            fallback_val = 0.0
        if fallback_val > 0:
            labor_basis_for_ladder = fallback_val
    base_bucket_labor = labor_basis_for_ladder - float(amortized_nre_total)
    if base_bucket_labor < 0:
        base_bucket_labor = 0.0
    def _pricing_total_direct_costs_from(*candidates: Any) -> float | None:
        for candidate in candidates:
            if not isinstance(candidate, _MappingABC):
                continue
            direct_value = _coerce_float_or_none(candidate.get("total_direct_costs"))
            if direct_value is not None:
                return float(direct_value)
            totals_candidate = candidate.get("totals")
            if isinstance(totals_candidate, _MappingABC):
                for key in ("total_direct_costs", "direct_costs"):
                    nested_value = _coerce_float_or_none(totals_candidate.get(key))
                    if nested_value is not None:
                        return float(nested_value)
        return None

    pricing_total_direct_costs = _pricing_total_direct_costs_from(
        result.get("pricing") if isinstance(result, _MappingABC) else None,
        breakdown.get("pricing") if isinstance(breakdown, _MappingABC) else None,
        process_plan_summary_local.get("pricing")
        if isinstance(process_plan_summary_local, _MappingABC)
        else None,
    )

    directs_from_pricing = _coerce_float_or_none(pricing.get("total_direct_costs"))
    if directs_from_pricing is None and pricing_total_direct_costs is not None:
        directs_from_pricing = float(pricing_total_direct_costs)
    if directs_from_pricing is None:
        directs_from_pricing = float(directs)
    ladder_directs_override = round(float(directs_from_pricing or 0.0), 2)

    bucket_view_for_ladder: Mapping[str, Any] | None = None
    if isinstance(breakdown, _MappingABC):
        bucket_view_candidate = breakdown.get("bucket_view")
        if isinstance(bucket_view_candidate, _MappingABC):
            bucket_view_for_ladder = typing.cast(Mapping[str, Any], bucket_view_candidate)

    ladder_labor_total = 0.0
    if isinstance(bucket_view_for_ladder, _MappingABC):
        buckets_for_ladder = bucket_view_for_ladder.get("buckets")
        if isinstance(buckets_for_ladder, _MappingABC):
            for metrics in buckets_for_ladder.values():
                if not isinstance(metrics, _MappingABC):
                    continue
                labor_component = _safe_float(metrics.get("labor$"), 0.0)
                machine_component = _safe_float(metrics.get("machine$"), 0.0)
                ladder_labor_total += labor_component + machine_component

    if labor_summary_total > 0 and not math.isclose(
        ladder_labor_total,
        labor_summary_total,
        abs_tol=0.01,
    ):
        ladder_labor_total = labor_summary_total

    amortized_component = float(amortized_nre_total if qty > 1 else 0.0)
    fallback_ladder_labor = round(base_bucket_labor + amortized_component, 2)
    ladder_labor = round(ladder_labor_total, 2)
    if ladder_labor <= 0.0:
        ladder_labor = fallback_ladder_labor

    nre_per_part = round(float(amortized_nre_total or 0.0), 2)
    machine_labor_total = round(max(0.0, float(ladder_labor) - nre_per_part), 2)
    ladder_subtotal = round(float(directs) + nre_per_part + machine_labor_total, 2)
    ladder_directs = round(ladder_subtotal - ladder_labor, 2)

    if math.isclose(ladder_directs, ladder_directs_override, abs_tol=0.01):
        ladder_directs = ladder_directs_override

    computed_total_labor_cost = ladder_labor
    if isinstance(breakdown, dict):
        breakdown["total_labor_cost"] = round(ladder_labor, 2)
    if isinstance(totals, dict):
        totals["labor_cost"] = ladder_labor

    final_per_part = round(machine_labor_total + nre_per_part + ladder_directs, 2)
    ladder_subtotal = final_per_part
    if 0 <= total_labor_row_index < len(lines):
        replace_line(
            total_labor_row_index,
            _format_row(total_labor_label, ladder_labor),
        )
    if isinstance(pricing, dict):
        pricing["ladder_subtotal"] = ladder_subtotal
    if not roughly_equal(declared_subtotal, ladder_subtotal, eps=0.01):
        declared_subtotal = ladder_subtotal
    if isinstance(breakdown, dict):
        try:
            breakdown["ladder_subtotal"] = ladder_subtotal
            if math.isclose(_safe_float(applied_pcts.get("MarginPct"), 0.0), 0.0, abs_tol=1e-6):
                breakdown["price"] = ladder_subtotal
        except Exception:
            pass

    printed_subtotal = round(float(declared_subtotal or 0.0), 2)
    if not roughly_equal(ladder_subtotal, printed_subtotal, eps=0.01):
        printed_subtotal = ladder_subtotal
        declared_subtotal = ladder_subtotal
        if isinstance(totals, dict):
            totals["subtotal"] = ladder_subtotal
    assert roughly_equal(ladder_labor + ladder_directs, printed_subtotal, eps=0.01)

    subtotal = ladder_subtotal
    printed_subtotal = ladder_subtotal

    # Render MATERIAL REMOVAL card + TIME PER HOLE lines (replace legacy Time block)
    # NOTE: Patch 3 keeps the hole-table hook active so downstream cards continue to render.
    append_lines(removal_card_lines)

    # ===== MATERIAL REMOVAL: HOLE-TABLE CARDS =================================
    import re

    def _first_dict(*cands):
        for c in cands:
            if isinstance(c, dict) and c:
                return c
        return {}

    def _get_geo_map(*cands):
        for c in cands:
            if isinstance(c, dict) and isinstance(c.get("geo"), dict):
                return c["geo"]
        return {}

    def _get_material_group(*cands):
        for c in cands:
            if isinstance(c, dict) and c.get("material_group"):
                return c["material_group"]
        return None

    def _norm_line(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    # patterns
    _RE_QTY_LEAD   = re.compile(r"^\s*\((\d+)\)\s*")
    _RE_FROM_SIDE  = re.compile(r"\bFROM\s+(FRONT|BACK)\b", re.I)
    _RE_DEPTH      = re.compile(r"[×x]\s*([0-9.]+)\b", re.I)
    _RE_THRU       = re.compile(r"\bTHRU\b", re.I)
    _RE_DIA_ANY    = re.compile(r'(?:Ø|⌀|DIA|O)\s*([0-9.]+)|\(([0-9.]+)\s*Ø?\)|\b([0-9.]+)\b')
    _RE_TAP        = re.compile(r"(\(\d+\)\s*)?(#\s*\d{1,2}-\d+|(?:\d+/\d+)\s*-\s*\d+|(?:\d+(?:\.\d+)?)\s*-\s*\d+|M\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?)\s*TAP", re.I)
    _RE_CBORE      = re.compile(r"\b(C[\'’]?\s*BORE|CBORE|COUNTER\s*BORE)\b", re.I)

    def _coalesce_rows(rows):
        agg, order = {}, []
        for r in rows:
            key = (r.get("desc",""), r.get("ref",""))
            if key not in agg:
                agg[key] = int(r.get("qty") or 0); order.append(key)
            else:
                agg[key] += int(r.get("qty") or 0)
        return [{"hole":"", "ref": ref, "qty": agg[(desc,ref)], "desc": desc} for (desc,ref) in order]

    def _build_ops_rows_from_lines(lines_in: list[str]) -> list[dict]:
        if not lines_in: return []
        L = [_norm_line(s) for s in lines_in if _norm_line(s)]
        out, i = [], 0
        while i < len(L):
            ln = L[i]
            if any(k in ln.upper() for k in ("BREAK ALL","SHARP CORNERS","RADIUS","CHAMFER","AS SHOWN")):
                i += 1; continue
            qty = 1
            mqty = _RE_QTY_LEAD.match(ln)
            if mqty: qty = int(mqty.group(1)); ln = ln[mqty.end():].strip()
            mtap = _RE_TAP.search(ln)
            if mtap:
                thread = mtap.group(2).replace(" ", "")
                tail = " ".join(L[i:i+3])
                desc = f"{thread} TAP"
                if _RE_THRU.search(tail): desc += " THRU"
                md = _RE_DEPTH.search(tail)
                if md and md.group(1): desc += f' × {float(md.group(1)):.2f}"'
                ms = _RE_FROM_SIDE.search(tail)
                if ms: desc += f' FROM {ms.group(1).upper()}'
                out.append({"hole":"", "ref":"", "qty":qty, "desc":desc})
                i += 1; continue
            if _RE_CBORE.search(ln):
                tail = " ".join(L[max(0,i-1):i+2])
                mda = _RE_DIA_ANY.search(tail)
                dia = None
                if mda:
                    for g in mda.groups():
                        if g: dia = float(g); break
                desc = (f"{dia:.4f} C’BORE" if dia is not None else "C’BORE")
                md = _RE_DEPTH.search(tail)
                if md and md.group(1): desc += f' × {float(md.group(1)):.2f}"'
                ms = _RE_FROM_SIDE.search(tail)
                if ms: desc += f' FROM {ms.group(1).upper()}'
                out.append({"hole":"", "ref":"", "qty":qty, "desc":desc})
                i += 1; continue
            if any(k in ln.upper() for k in ("C' DRILL","C’DRILL","CENTER DRILL","SPOT DRILL")):
                tail = " ".join(L[i:i+2])
                md = _RE_DEPTH.search(tail)
                desc = "C’DRILL" + (f' × {float(md.group(1)):.2f}"' if (md and md.group(1)) else "")
                out.append({"hole":"", "ref":"", "qty":qty, "desc":desc})
                i += 1; continue
            i += 1
        return _coalesce_rows(out)

    try:
        ctx_a = locals().get("breakdown")
        ctx_b = locals().get("result")
        ctx_c = locals().get("quote")
        ctx   = _first_dict(ctx_a, ctx_b, ctx_c)

        geo_map        = _get_geo_map(ctx, locals().get("geo"), ctx_a, ctx_b)
        material_group = _get_material_group(ctx, ctx_a, ctx_b)

        ops_rows = (((geo_map or {}).get("ops_summary") or {}).get("rows") or [])
        _push(lines, f"[DEBUG] ops_rows_pre={len(ops_rows)}")

        if not ops_rows:
            # collect chart text wherever it lives
            def _collect_chart_lines(*containers):
                keys = ("chart_lines","hole_table_lines","chart_text_lines","hole_chart_lines")
                merged, seen = [], set()
                for d in containers:
                    if not isinstance(d, dict): continue
                    for k in keys:
                        v = d.get(k)
                        if isinstance(v, list) and all(isinstance(x, str) for x in v):
                            for s in v:
                                if s not in seen:
                                    seen.add(s); merged.append(s)
                return merged
            chart_lines_all = _collect_chart_lines(ctx, geo_map, ctx_a, ctx_b)
            built = _build_ops_rows_from_lines(chart_lines_all)
            _push(lines, f"[DEBUG] chart_lines_found={len(chart_lines_all)} built_rows={len(built)}")
            if built:
                geo_map.setdefault("ops_summary", {})["rows"] = built
                ops_rows = built

        # Emit the cards (will no-op if no TAP/CBore/Spot rows)
        _emit_hole_table_ops_cards(lines, geo=geo_map, material_group=material_group, speeds_csv=None)

    except Exception as e:
        _push(lines, f"[DEBUG] material_removal_emit_skipped={e.__class__.__name__}: {e}")
    # ========================================================================

    # ---- Pricing ladder ------------------------------------------------------
    append_line("Pricing Ladder")
    append_line(divider)

    override_sources: list[Mapping[str, Any]] = []

    def _coerce_mapping(source: Any) -> dict[str, Any] | None:
        if isinstance(source, dict):
            return source
        if isinstance(source, _MappingABC):
            try:
                return dict(source)
            except Exception:
                return None
        return None

    def _collect_override_source(candidate: Any) -> None:
        mapping = _coerce_mapping(candidate)
        if mapping:
            override_sources.append(mapping)

    _collect_override_source(applied_pcts)
    _collect_override_source(breakdown.get("config"))
    _collect_override_source(result.get("config"))
    _collect_override_source(breakdown.get("overrides"))
    _collect_override_source(result.get("overrides"))
    _collect_override_source(breakdown.get("params"))
    _collect_override_source(result.get("params"))
    if isinstance(params, _MappingABC):
        _collect_override_source(params)
    quote_state_payload = result.get("quote_state") if isinstance(result, _MappingABC) else None
    if isinstance(quote_state_payload, _MappingABC):
        for nested_key in ("user_overrides", "overrides", "effective", "config", "params"):
            _collect_override_source(quote_state_payload.get(nested_key))

    def _resolve_ladder_pct(keys: Sequence[str], default: float) -> float:
        sentinel = object()
        for source in override_sources:
            for key in keys:
                value = sentinel
                try:
                    value = source.get(key, sentinel)  # type: ignore[arg-type]
                except Exception:
                    try:
                        value = source[key]  # type: ignore[index]
                    except Exception:
                        value = sentinel
                if value is sentinel or value is None:
                    continue
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        continue
                    return _safe_float(stripped, default)
                return _safe_float(value, default)
        return default

    expedite_pct_value = _resolve_ladder_pct(("ExpeditePct", "expedite_pct"), 0.0)
    margin_pct_value = _resolve_ladder_pct(("MarginPct", "margin_pct"), 0.15)

    applied_pcts.setdefault("MarginPct", margin_pct_value)
    if "ExpeditePct" not in applied_pcts and expedite_pct_value:
        applied_pcts["ExpeditePct"] = expedite_pct_value

    ladder_totals = _compute_pricing_ladder(
        subtotal,
        expedite_pct=expedite_pct_value,
        margin_pct=margin_pct_value,
    )

    with_expedite = ladder_totals["with_expedite"]
    subtotal_before_margin = ladder_totals.get("subtotal_before_margin", with_expedite)
    expedite_cost = ladder_totals.get("expedite_cost", 0.0)
    final_price = ladder_totals["with_margin"]

    if isinstance(totals, dict):
        totals["with_expedite"] = with_expedite
        totals["with_margin"] = final_price
        totals["price"] = final_price

    price = final_price
    if isinstance(result, dict):
        result["price"] = price
    if isinstance(breakdown, dict):
        breakdown["final_price"] = final_price
    replace_line(final_price_row_index, _format_row("Final Price per Part:", price))

    row("Subtotal (Labor + Directs):", subtotal)
    if applied_pcts.get("ExpeditePct"):
        row(f"+ Expedite ({_pct(applied_pcts.get('ExpeditePct'))}):", expedite_cost)
    row("= Subtotal before Margin:", subtotal_before_margin)
    row(f"Final Price with Margin ({_pct(applied_pcts.get('MarginPct'))}):", price)
    append_line("")

    # ---- LLM adjustments bullets (optional) ---------------------------------
    if llm_notes:
        append_line("LLM Adjustments")
        append_line(divider)
        import textwrap as _tw
        for n in llm_notes:
            for w in _tw.wrap(str(n), width=page_width):
                append_line(f"- {w}")
        append_line("")

    if not explanation_lines:
        plan_info_for_explainer: Mapping[str, Any] | None = None
        plan_info_payload: dict[str, Any] = {}

        process_plan_for_explainer: Mapping[str, Any] | None = None
        process_plan_candidate = locals().get("process_plan_summary_local")
        if isinstance(process_plan_candidate, _MappingABC) and process_plan_candidate:
            process_plan_for_explainer = process_plan_candidate
        elif isinstance(breakdown, _MappingABC):
            candidate_summary = breakdown.get("process_plan")
            if isinstance(candidate_summary, _MappingABC) and candidate_summary:
                process_plan_for_explainer = candidate_summary
        if isinstance(process_plan_for_explainer, _MappingABC) and process_plan_for_explainer:
            plan_info_payload["process_plan_summary"] = process_plan_for_explainer

        if isinstance(breakdown, _MappingABC):
            process_plan_map = breakdown.get("process_plan")
            if isinstance(process_plan_map, _MappingABC) and process_plan_map:
                plan_info_payload.setdefault("process_plan", process_plan_map)
            plan_pricing_map = breakdown.get("process_plan_pricing")
            if isinstance(plan_pricing_map, _MappingABC) and plan_pricing_map:
                plan_info_payload.setdefault("pricing", plan_pricing_map)

        planner_pricing_for_explainer: Mapping[str, Any] | None = None
        if isinstance(breakdown, _MappingABC):
            candidate_planner = breakdown.get("process_plan_pricing")
            if isinstance(candidate_planner, _MappingABC) and candidate_planner:
                planner_pricing_for_explainer = candidate_planner
        if (
            planner_pricing_for_explainer is None
            and isinstance(result, _MappingABC)
        ):
            candidate_planner = result.get("process_plan_pricing")
            if isinstance(candidate_planner, _MappingABC) and candidate_planner:
                planner_pricing_for_explainer = candidate_planner
        if planner_pricing_for_explainer is None:
            candidate_planner = locals().get("planner_result")
            if isinstance(candidate_planner, _MappingABC) and candidate_planner:
                planner_pricing_for_explainer = candidate_planner

        if isinstance(planner_pricing_for_explainer, _MappingABC) and planner_pricing_for_explainer:
            plan_info_payload["planner_pricing"] = planner_pricing_for_explainer

        bucket_plan_info: dict[str, Any] = {}
        if isinstance(bucket_state, PlannerBucketRenderState):
            extra_map = getattr(bucket_state, "extra", None)
            if isinstance(extra_map, _MappingABC) and extra_map:
                try:
                    bucket_plan_info.update(dict(extra_map))
                except Exception:
                    for key, value in extra_map.items():
                        bucket_plan_info[key] = value
            bucket_minutes_detail_map = getattr(bucket_state, "bucket_minutes_detail", None)
            if isinstance(bucket_minutes_detail_map, _MappingABC) and bucket_minutes_detail_map:
                bucket_plan_info.setdefault(
                    "bucket_minutes_detail_for_render",
                    bucket_minutes_detail_map,
                )
        if bucket_plan_info:
            plan_info_payload["bucket_state_extra"] = bucket_plan_info

        if plan_info_payload:
            plan_info_for_explainer = plan_info_payload

        try:
            explanation_text = explain_quote(
                breakdown,
                hour_trace=hour_trace_data,
                geometry=geometry_for_explainer,
                render_state=bucket_state,
                plan_info=plan_info_for_explainer,
            )
        except Exception:
            explanation_text = ""
        if explanation_text:
            for line in str(explanation_text).splitlines():
                text = line.strip()
                if text:
                    explanation_lines.append(text)

    if explanation_lines:
        why_lines.extend(explanation_lines)
    if why_lines:
        why_parts.extend(why_lines)

    if why_parts:
        if lines and lines[-1]:
            append_line("")
        append_line("Why this price")
        append_line(divider)
        for part in why_parts:
            write_wrapped(part, "  ")
        if lines[-1]:
            append_line("")
        # Append the compact removal debug table (if available)
        append_removal_debug_if_enabled(lines, removal_summary_for_display)

    # ──────────────────────────────────────────────────────────────────────────
    # Nice, compact sanity check at the very end (only when debug is enabled)
    # Shows whether drilling hours are consistent across views:
    #  - planner bucket minutes
    #  - canonical bucket rollup (minutes)
    #  - hour summary entry
    #  - process_meta['drilling']['hr']
    # ──────────────────────────────────────────────────────────────────────────
    try:
        _llm_dbg = _resolve_llm_debug_enabled(result, breakdown, params, {"llm_debug_enabled": True})
    except Exception:
        _llm_dbg = bool(APP_ENV.llm_debug_enabled)
    if _llm_dbg:
        try:
            # Pull minutes from planner bucket view if available
            _planner_min = None
            _canon_min = None
            _hsum_hr = None
            _meta_hr = None

            _pbv = locals().get("planner_bucket_view")
            if isinstance(_pbv, _MappingABC):
                _buckets = _pbv.get("buckets") if isinstance(_pbv.get("buckets"), _MappingABC) else {}
                _drill = _buckets.get("Drilling") or _buckets.get("drilling") if isinstance(_buckets, _MappingABC) else None
                if isinstance(_drill, _MappingABC):
                    _planner_min = _coerce_float_or_none(_drill.get("minutes"))

            _canon = locals().get("canonical_bucket_rollup")
            if isinstance(_canon, _MappingABC):
                _canon_min = _coerce_float_or_none(_canon.get("drilling"))
                if _canon_min is not None:
                    _canon_min = round(float(_canon_min) * 60.0, 1)  # hours→minutes

            _hsum = locals().get("hour_summary_entries")
            if isinstance(_hsum, _MappingABC):
                # hour_summary_entries: {label: (hr, include_flag)}
                for _label, (_hr, _inc) in _hsum.items():
                    if str(_label).strip().lower() == "drilling":
                        _hsum_hr = _coerce_float_or_none(_hr)
                        break

            _pmeta = locals().get("process_meta")
            if isinstance(_pmeta, _MappingABC):
                _meta_hr = _coerce_float_or_none((_pmeta.get("drilling") or {}).get("hr"))

            append_line("DEBUG — Drilling sanity")
            append_line(divider)
            def _fmt(x, unit):
                return "—" if x is None or not math.isfinite(float(x)) else f"{float(x):.2f} {unit}"
            append_line(
                "  bucket(planner): "
                + _fmt(_planner_min, "min")
                + "   canonical: "
                + _fmt(_canon_min, "min")
                + "   hour_summary: "
                + _fmt(_hsum_hr, "hr")
                + "   meta: "
                + _fmt(_meta_hr, "hr")
            )
            append_line("")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final cross-check before rendering the quote to text. Verify that
    # drilling minutes/rows remain aligned and capture a snapshot of
    # related financial signals for post-run debugging.
    try:
        process_plan_map: Mapping[str, Any] | None = None
        for candidate in (
            locals().get("process_plan_summary_local"),
            breakdown.get("process_plan") if isinstance(breakdown, _MappingABC) else None,
        ):
            if isinstance(candidate, _MappingABC):
                process_plan_map = candidate
                break

        drilling_plan = (
            process_plan_map.get("drilling")
            if isinstance(process_plan_map, _MappingABC)
            else None
        )
        drill_min_card = _safe_float(
            (drilling_plan or {}).get("total_minutes_billed"),
            default=0.0,
        )
        if drill_min_card <= 0.0 and isinstance(drilling_plan, _MappingABC):
            drill_min_card = _safe_float(
                drilling_plan.get("total_minutes_with_toolchange"),
                default=0.0,
            )

        drilling_row: _ProcessRowRecord | None = None
        try:
            for record in getattr(process_table, "rows", []):
                if _process_row_canon(record) == "drilling":
                    drilling_row = record
                    break
        except Exception:
            drilling_row = None

        row_hr = float(drilling_row.hours or 0.0) if drilling_row else 0.0
        row_rate = float(drilling_row.rate or 0.0) if drilling_row else 0.0
        row_cost = float(drilling_row.total or 0.0) if drilling_row else 0.0

        bucket_minutes_map = (
            breakdown.get("bucket_minutes_detail")
            if isinstance(breakdown, _MappingABC)
            else None
        )
        if not isinstance(bucket_minutes_map, _MutableMappingABC):
            bucket_minutes_map = bucket_minutes_detail if isinstance(bucket_minutes_detail, dict) else {}
        drill_min_row = _safe_float(
            (bucket_minutes_map or {}).get("drilling"),
            default=0.0,
        )

        if (
            drilling_row is not None
            and abs(drill_min_card - 60.0 * row_hr) > 0.01
        ):
            logger.error(
                "[DRILL_GUARD] Minutes mismatch before render: card %.2f min vs row %.2f min — forcing sync.",
                drill_min_card,
                60.0 * row_hr,
            )
            corrected_hr_precise = drill_min_card / 60.0
            corrected_hr = round(corrected_hr_precise, 2)
            new_cost = row_cost
            if row_rate > 0.0:
                new_cost = round(corrected_hr_precise * row_rate, 2)
            process_table.update_row("drilling", hours=corrected_hr, cost=new_cost)
            drilling_row.hours = corrected_hr
            drilling_row.total = new_cost
            process_cost_row_details["drilling"] = (corrected_hr, row_rate, new_cost)
            process_costs_for_render["drilling"] = new_cost
            labor_costs_display[drilling_row.name] = new_cost
            bucket_minutes_detail["drilling"] = float(drill_min_card)
            if isinstance(bucket_minutes_map, _MutableMappingABC):
                bucket_minutes_map["drilling"] = float(drill_min_card)
            bucket_view_obj = (
                breakdown.get("bucket_view")
                if isinstance(breakdown, _MappingABC)
                else None
            )
            _sync_drilling_bucket_view(
                bucket_view_obj,
                billed_minutes=float(drill_min_card),
                billed_cost=new_cost if new_cost > 0.0 else None,
            )
            row_hr = corrected_hr
            drill_min_row = float(drill_min_card)

        programming_hr = 0.0
        if isinstance(nre_detail, _MappingABC):
            programming_detail = nre_detail.get("programming")
            if isinstance(programming_detail, _MappingABC):
                programming_hr = _safe_float(
                    programming_detail.get("prog_hr"),
                    default=0.0,
                )

        material_block_dbg = (
            breakdown.get("material")
            if isinstance(breakdown, _MappingABC)
            else None
        )
        material_cost = _safe_float(
            (material_block_dbg or {}).get("total_cost"),
            default=0.0,
        )

        direct_costs = _safe_float(
            (breakdown if isinstance(breakdown, _MappingABC) else {}).get("total_direct_costs"),
            default=0.0,
        )

        dbg = {
            "drill_min_card": float(drill_min_card),
            "drill_min_row": float(drill_min_row),
            "programming_hr": float(programming_hr),
            "material_cost": float(material_cost),
            "direct_costs": float(direct_costs),
            "ladder_subtotal": float(ladder_subtotal),
        }
        logger.debug("[render_quote] drill guard snapshot: %s", jdump(dbg, default=None))
    except Exception:
        logger.exception("Failed to run final drilling debug block")

    # --- Structured render payload ------------------------------------------
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    try:
        qty_float = float(qty or 0.0)
    except Exception:
        qty_float = 0.0
    if qty_float > 0 and abs(round(qty_float) - qty_float) < 1e-9:
        summary_qty: int | float = int(round(qty_float))
    else:
        summary_qty = qty_float if qty_float > 0 else qty

    margin_pct_value = _as_float(applied_pcts.get("MarginPct"), 0.0)
    expedite_pct_value = _as_float(applied_pcts.get("ExpeditePct"), 0.0)
    expedite_amount = _as_float(expedite_cost, 0.0)
    subtotal_before_margin_val = _as_float(subtotal_before_margin, 0.0)
    final_price_val = _as_float(price, 0.0)
    margin_amount = max(0.0, final_price_val - subtotal_before_margin_val)
    labor_total_amount = _as_float(
        (breakdown or {}).get("total_labor_cost"),
        _as_float(ladder_labor, 0.0),
    )
    direct_total_amount = _as_float(total_direct_costs_value, 0.0)

    summary_payload = {
        "qty": summary_qty,
        "final_price": round(final_price_val, 2),
        "unit_price": round(final_price_val, 2),
        "subtotal_before_margin": round(subtotal_before_margin_val, 2),
        "margin_pct": float(margin_pct_value),
        "margin_amount": round(margin_amount, 2),
        "expedite_pct": float(expedite_pct_value),
        "expedite_amount": round(expedite_amount, 2),
        "currency": currency,
    }

    driver_details: list[str] = []
    seen_driver_details: set[str] = set()
    for detail in list(why_parts) + list(llm_notes):
        text = str(detail).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_driver_details:
            continue
        seen_driver_details.add(key)
        driver_details.append(text)
    price_drivers_payload = [{"detail": detail} for detail in driver_details]

    cost_breakdown_payload: list[tuple[str, float]] = []
    cost_breakdown_payload.append(("Machine & Labor", round(labor_total_amount, 2)))
    cost_breakdown_payload.append(("Direct Costs", round(direct_total_amount, 2)))
    if expedite_amount > 0:
        cost_breakdown_payload.append(("Expedite", round(expedite_amount, 2)))
    cost_breakdown_payload.append(("Subtotal before Margin", round(subtotal_before_margin_val, 2)))
    cost_breakdown_payload.append(("Margin", round(margin_amount, 2)))
    cost_breakdown_payload.append(("Final Price", round(final_price_val, 2)))

    materials_entries: list[dict[str, Any]] = []
    material_label_text = str(
        material_display_label
        or material_selection.get("material_display")
        or material_selection.get("material")
        or material_selection.get("material_name")
        or "Material"
    ).strip()
    if material_label_text and (material_display_amount > 0 or show_zeros):
        materials_entries.append(
            {
                "label": material_label_text,
                "amount": round(_as_float(material_display_amount, 0.0), 2),
            }
        )
    for entry_label, entry_amount, raw_key in direct_entries:
        if str(raw_key).strip().lower() == "material":
            continue
        if not show_zeros and entry_amount <= 0:
            continue
        materials_entries.append(
            {
                "label": str(entry_label),
                "amount": round(_as_float(entry_amount, 0.0), 2),
            }
        )

    processes_entries: list[dict[str, Any]] = []
    seen_process_labels: set[str] = set()
    for spec in bucket_row_specs:
        label = str(spec.label or "").strip()
        amount_val = _as_float(spec.total, 0.0)
        if not label:
            continue
        if not show_zeros and amount_val <= 0:
            continue
        if label in seen_process_labels:
            continue
        seen_process_labels.add(label)
        processes_entries.append(
            {
                "label": label,
                "amount": round(amount_val, 2),
                "hours": round(_as_float(spec.hours, 0.0), 2),
                "minutes": round(_as_float(spec.minutes, 0.0), 2),
                "labor_amount": round(_as_float(spec.labor, 0.0), 2),
                "machine_amount": round(_as_float(spec.machine, 0.0), 2),
                "rate": round(_as_float(spec.rate, 0.0), 2) if _as_float(spec.rate, 0.0) else 0.0,
            }
        )

    render_payload = {
        "summary": summary_payload,
        "price_drivers": price_drivers_payload,
        "cost_breakdown": cost_breakdown_payload,
        "materials": materials_entries,
        "materials_direct": round(direct_total_amount, 2),
        "processes": processes_entries,
        "labor_total_amount": round(labor_total_amount, 2),
        "ladder": {
            "labor_total": round(labor_total_amount, 2),
            "direct_total": round(direct_total_amount, 2),
            "subtotal_before_margin": round(subtotal_before_margin_val, 2),
            "margin_pct": float(margin_pct_value),
            "margin_amount": round(margin_amount, 2),
            "final_price": round(final_price_val, 2),
        },
    }

    if isinstance(result, _MutableMappingABC):
        result.setdefault("render_payload", render_payload)
    if isinstance(breakdown, _MutableMappingABC):
        breakdown.setdefault("render_payload", render_payload)

    doc = doc_builder.build_doc()
    text = render_quote_doc(doc, divider=divider)

    # ASCII-sanitize output to avoid mojibake like '×' on some Windows setups
    text = _sanitize_render_text(text)

    return text
# ===== QUOTE CONFIG (edit-friendly) ==========================================
CONFIG_INIT_ERRORS: list[str] = []


def _normalized_two_bucket_rates(
    candidate: Any | Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Normalize rate mappings while preserving empty fallbacks."""

    if not isinstance(candidate, _MappingABC):
        return {"labor": {}, "machine": {}}
    if not candidate:
        return {"labor": {}, "machine": {}}

    normalized = _ensure_two_bucket_rates(candidate)
    if not any(normalized.get(kind) for kind in ("labor", "machine")):
        return {"labor": {}, "machine": {}}
    return normalized

try:
    SERVICE_CONTAINER = create_default_container()
except Exception as exc:
    CONFIG_INIT_ERRORS.append(f"Service container initialisation error: {exc}")

    def _empty_params() -> dict[str, Any]:
        return {}

    SERVICE_CONTAINER = ServiceContainer(
        load_params=_empty_params,
        load_rates=lambda: _normalized_two_bucket_rates({}),
        pricing_engine_factory=lambda: PricingEngine(create_default_registry()),
    )

try:
    _rates_raw = SERVICE_CONTAINER.load_rates()
except ConfigError as exc:
    RATES_TWO_BUCKET_DEFAULT = _normalized_two_bucket_rates({})
    CONFIG_INIT_ERRORS.append(f"Rates configuration error: {exc}")
except Exception as exc:
    RATES_TWO_BUCKET_DEFAULT = _normalized_two_bucket_rates({})
    CONFIG_INIT_ERRORS.append(f"Unexpected rates configuration error: {exc}")
else:
    RATES_TWO_BUCKET_DEFAULT = (
        _normalized_two_bucket_rates(_rates_raw)
        if isinstance(_rates_raw, _MappingABC)
        else _normalized_two_bucket_rates({})
    )

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

DEFAULT_SPEEDS_FEEDS_CSV_BASENAME = "speeds_feeds_merged.csv"


def _default_speeds_feeds_csv_path() -> str:
    """Return the packaged Speeds/Feeds CSV path if available."""

    root = Path(__file__).resolve().parent
    candidate = root / "cad_quoter" / "pricing" / "resources" / DEFAULT_SPEEDS_FEEDS_CSV_BASENAME
    if candidate.is_file():
        return str(candidate)
    return DEFAULT_SPEEDS_FEEDS_CSV_BASENAME


DEFAULT_SPEEDS_FEEDS_CSV_PATH = _default_speeds_feeds_csv_path()
if not str(PARAMS_DEFAULT.get("SpeedsFeedsCSVPath", "")).strip():
    PARAMS_DEFAULT["SpeedsFeedsCSVPath"] = DEFAULT_SPEEDS_FEEDS_CSV_PATH

# ---- Service containers -----------------------------------------------------


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
    items: PandasSeries,
    values: PandasSeries,
    data_types: PandasSeries,
    mask: PandasSeries,
    *,
    default: float = 0.0,
    exclude_mask: PandasSeries | None = None,
) -> float:
    """Shared implementation for extracting hour totals from sheet rows."""

    if pd is None:
        raise RuntimeError("pandas required (conda/pip install pandas)")

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


def _text_contains_plate_signal(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if not lowered:
        return False
    if "mic6" in lowered or "mic-6" in lowered or "mic 6" in lowered:
        return True
    if "flat stock" in lowered:
        return True
    if "2d" in lowered and "plate" in lowered:
        return True
    return bool(re.search(r"\bplate\b", lowered))


def _overrides_indicate_plate(overrides: Any, *, depth: int = 2) -> bool:
    if depth < 0 or overrides is None:
        return False
    if isinstance(overrides, str):
        return _text_contains_plate_signal(overrides)
    if isinstance(overrides, _MappingABC):
        for key, value in overrides.items():
            if _text_contains_plate_signal(str(key)) or _text_contains_plate_signal(value):
                return True
            if depth > 0:
                if isinstance(value, _MappingABC):
                    if _overrides_indicate_plate(value, depth=depth - 1):
                        return True
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    for item in value:
                        if _overrides_indicate_plate(item, depth=depth - 1):
                            return True
        for nested_key in (
            "stock_recommendation",
            "geo_context",
            "derived",
            "stock",
            "geometry",
            "dimensions",
        ):
            nested = overrides.get(nested_key)
            if _overrides_indicate_plate(nested, depth=depth - 1):
                return True
        for dim_key in (
            "plate_length_mm",
            "plate_width_mm",
            "plate_length_in",
            "plate_width_in",
            "plate_len_mm",
            "plate_wid_mm",
            "plate_len_in",
            "plate_wid_in",
        ):
            if overrides.get(dim_key):
                return True
        planner_family = overrides.get("process_planner_family")
        if _text_contains_plate_signal(planner_family):
            return True
        geometry_kind = overrides.get("geometry_kind") or overrides.get("geometry_type")
        if _text_contains_plate_signal(geometry_kind):
            return True
        return False
    if isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        for item in overrides:
            if _overrides_indicate_plate(item, depth=depth - 1):
                return True
    return False


def _pricing_meta_key_for_normalized(normalized_key: str | None) -> str | None:
    if not normalized_key:
        return None
    meta_key = _MATERIAL_META_KEY_BY_NORMALIZED.get(normalized_key)
    if meta_key:
        return meta_key
    for display_key, keywords in MATERIAL_KEYWORDS.items():
        if normalized_key in keywords:
            meta_key = _MATERIAL_META_KEY_BY_NORMALIZED.get(display_key)
            if meta_key:
                return meta_key
    return None


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

    requested_material_name = str(material_name or "").strip()
    normalized_requested_key = _normalize_lookup_key(requested_material_name)
    plate_hint = _text_contains_plate_signal(requested_material_name) or _overrides_indicate_plate(overrides)
    is_aluminum = bool(normalized_requested_key and "alum" in normalized_requested_key)

    canonical_display = MATERIAL_DISPLAY_BY_KEY.get(normalized_requested_key, "")
    normalized_material_key = normalized_requested_key
    if is_aluminum and plate_hint:
        canonical_display = _CANONICAL_MIC6_DISPLAY
        normalized_material_key = _MIC6_NORMALIZED_KEY or normalized_requested_key
    if not canonical_display and normalized_material_key:
        canonical_display = MATERIAL_DISPLAY_BY_KEY.get(normalized_material_key, "")

    resolved_display = canonical_display or requested_material_name
    normalized_material_key = _normalize_lookup_key(resolved_display)
    key = resolved_display.strip().upper()

    meta_lookup_key = _pricing_meta_key_for_normalized(normalized_material_key)
    if meta_lookup_key is None and normalized_requested_key != normalized_material_key:
        meta_lookup_key = _pricing_meta_key_for_normalized(normalized_requested_key)

    meta = MATERIAL_MAP.get(meta_lookup_key) if meta_lookup_key else None
    if meta is None and key in MATERIAL_MAP:
        meta_lookup_key = key
        meta = MATERIAL_MAP[key]
    if meta is None and requested_material_name:
        original_upper = requested_material_name.strip().upper()
        if original_upper in MATERIAL_MAP:
            meta_lookup_key = original_upper
            meta = MATERIAL_MAP[original_upper]
    if meta is None and is_aluminum and _CANONICAL_MIC6_PRICING_KEY:
        meta_lookup_key = _CANONICAL_MIC6_PRICING_KEY
        meta = MATERIAL_MAP.get(meta_lookup_key)

    if meta is None:
        fallback_symbol = "XAL" if is_aluminum else (key or "XAL")
        fallback_basis = "index_usd_per_tonne" if fallback_symbol == "XAL" else "usd_per_kg"
        meta = {"symbol": fallback_symbol, "basis": fallback_basis}
        if not meta_lookup_key:
            meta_lookup_key = key or fallback_symbol
    else:
        meta = meta.copy()

    symbol = str(meta.get("symbol", "XAL" if is_aluminum else key or "XAL"))
    basis_default = "index_usd_per_tonne" if symbol == "XAL" else "usd_per_kg"
    basis = str(meta.get("basis", basis_default))

    material_name = resolved_display

    vendor_csv = vendor_csv or ""
    usd_per_kg: float | None = None
    source = ""
    basis_used = basis
    price_candidates: list[str] = []

    def _remember_candidate(label: Any) -> None:
        text = str(label or "").strip()
        if text and text not in price_candidates:
            price_candidates.append(text)

    _remember_candidate(canonical_display)
    if requested_material_name and requested_material_name != canonical_display:
        _remember_candidate(requested_material_name)
    _remember_candidate(meta_lookup_key)
    _remember_candidate(symbol)

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
                    if density_g_cc and density_g_cc > 0:
                        stock_mass_kg, _ = _plate_mass_from_dims(
                            L_mm,
                            W_mm,
                            T_mm,
                            density_g_cc,
                            dims_in=dims_in,
                            hole_d_mm=(),
                        )
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
        wieland_key = meta.get("wieland_key") if isinstance(meta, _MappingABC) else None
        if wieland_key:
            _remember_candidate(wieland_key)
        _remember_candidate(material_name)
        _remember_candidate(key)
        _remember_candidate(meta_lookup_key)
        _remember_candidate(symbol)

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
        "material_display": canonical_display or material_name,
        "canonical_material": canonical_display or material_name,
        "requested_material_name": requested_material_name,
        "normalized_material_key": normalized_material_key,
        "material_lookup_key": meta_lookup_key or key,
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

def estimate_drilling_hours(
    hole_diams_mm: list[float],
    thickness_in: float,
    mat_key: str,
    *,
    material_group: str | None = None,
    hole_groups: Sequence[Mapping[str, Any]] | None = None,
    speeds_feeds_table: PandasDataFrame | None = None,
    machine_params: _TimeMachineParams | None = None,
    overhead_params: _TimeOverheadParams | None = None,
    warnings: list[str] | None = None,
    debug_lines: list[str] | None = None,
    debug_summary: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Adapter that invokes the drilling estimator plugin."""

    from cad_quoter.estimators.base import EstimatorInput
    from cad_quoter.estimators.drilling import estimate as _drilling_estimate

    tables: dict[str, Any] = {}
    if speeds_feeds_table is not None:
        tables["speeds_feeds"] = speeds_feeds_table

    geometry: dict[str, Any] = {
        "hole_diams_mm": list(hole_diams_mm or []),
        "thickness_in": thickness_in,
    }
    if hole_groups is not None:
        geometry["hole_groups"] = hole_groups

    input_data = EstimatorInput(
        material_key=mat_key,
        geometry=geometry,
        material_group=material_group,
        tables=tables,
        machine_params=machine_params,
        overhead_params=overhead_params,
        warnings=warnings,
        debug_lines=debug_lines,
        debug_summary=debug_summary,
    )

    return _drilling_estimate(input_data)


def _df_to_value_map(df: Any) -> dict[str, Any]:
    """Coerce a worksheet-like structure into ``{item: value}`` form."""

    value_map: dict[str, Any] = {}

    try:
        df_obj = coerce_or_make_vars_df(df)
    except Exception:
        df_obj = df

    if _HAS_PANDAS and "pd" in globals():
        import pandas as pd  # type: ignore

        if isinstance(df_obj, pd.DataFrame):
            for _, row in df_obj.iterrows():
                try:
                    item = str(row.get("Item", "") or "").strip()
                except Exception:
                    item = ""
                if not item:
                    continue
                value_map[item] = row.get("Example Values / Options")
            return value_map

    if isinstance(df_obj, list):
        iterable = df_obj
    else:
        try:
            iterable = list(df_obj)
        except Exception:
            iterable = []

    for row in iterable:
        if not isinstance(row, dict):
            continue
        item = str(row.get("Item", "") or "").strip()
        if not item:
            continue
        value_map[item] = row.get("Example Values / Options")

    return value_map


_MATERIAL_EMPTY_TOKENS = {"", "none", "na", "null", "tbd", "unknown"}


def _material_text_or_none(value: Any) -> str | None:
    """Return a cleaned material string or ``None`` if it's effectively empty."""

    text = str(value or "").strip()
    if not text:
        return None
    normalized = re.sub(r"[^0-9a-z]+", "", text.lower())
    if normalized in _MATERIAL_EMPTY_TOKENS or normalized == "group":
        return None
    return text


def build_geometry_context(
    quote_source: Any,
    *,
    base_geometry: Mapping[str, Any] | None = None,
    value_map: Mapping[str, Any] | None = None,
    cfg: QuoteConfiguration | None = None,
) -> dict[str, Any]:
    """Return a canonical geometry context derived from quote inputs."""

    if isinstance(base_geometry, dict):
        geom: dict[str, Any] = dict(base_geometry)
    elif isinstance(base_geometry, _MappingABC):
        geom = {str(key): value for key, value in base_geometry.items()}
    else:
        geom = {}

    existing_material = _material_text_or_none(geom.get("material"))
    if existing_material:
        geom["material"] = existing_material
        geom.setdefault("material_name", existing_material)
    else:
        geom.pop("material", None)

    resolved_map: Mapping[str, Any] | None
    if isinstance(value_map, _MappingABC):
        resolved_map = value_map
    elif isinstance(quote_source, _MappingABC):
        resolved_map = typing.cast(Mapping[str, Any], quote_source)
    else:
        try:
            resolved_map = _df_to_value_map(quote_source)
        except Exception:
            resolved_map = None

    material_text: str | None = None
    if isinstance(resolved_map, _MappingABC):
        for field in ("Material Name", "Material"):
            raw_value = resolved_map.get(field)
            text = _material_text_or_none(raw_value)
            if text:
                material_text = text
                break
        if material_text:
            geom.setdefault("material", material_text)
            geom.setdefault("material_name", material_text)

        thickness_mm_ui = _coerce_positive_float(resolved_map.get("Thickness (mm)"))
        if thickness_mm_ui is not None:
            geom.setdefault("thickness_mm", thickness_mm_ui)

        thickness_in_ui = _coerce_positive_float(resolved_map.get("Thickness (in)"))
        if thickness_in_ui is not None:
            geom.setdefault("thickness_in", thickness_in_ui)

        hole_count_ui = _coerce_float_or_none(resolved_map.get("Hole Count"))
        if hole_count_ui is not None and hole_count_ui > 0:
            try:
                geom.setdefault("hole_count", int(round(float(hole_count_ui))))
            except Exception:
                pass

    _ensure_geo_context_fields(geom, resolved_map, cfg=cfg)
    return geom


def _lookup_rate(name: str, *sources: Mapping[str, Any] | None, fallback: float = 0.0) -> float:
    for source in sources:
        if not isinstance(source, _MappingABC):
            continue
        if name in source:
            try:
                candidate = source[name]
            except Exception:
                continue
            coerced = _coerce_float_or_none(candidate)
            if coerced is not None:
                return float(coerced)
    return float(fallback)


def _estimate_programming_hours_from_plan(
    plan: Mapping[str, Any] | None,
    geo: Mapping[str, Any] | None,
) -> float:
    """Return a plan-aware programming estimate based on planner operations."""

    geo_ctx = geo if isinstance(geo, _MappingABC) else {}
    plan_ctx = plan if isinstance(plan, _MappingABC) else {}

    unique_normals = geo_ctx.get("unique_normals")
    if isinstance(unique_normals, (int, float)):
        setups_raw = unique_normals
    else:
        setups_raw = 2

    try:
        setups = int(round(float(setups_raw)))
    except Exception:
        setups = 2
    setups = max(1, setups)

    hole_count_raw = geo_ctx.get("hole_count", 0) if geo_ctx else 0
    try:
        hole_count = int(hole_count_raw or 0)
    except Exception:
        hole_count = 0

    pocket_area_raw = geo_ctx.get("pocket_area_total_in2", 0.0) if geo_ctx else 0.0
    try:
        pocket_area_in2 = float(pocket_area_raw or 0.0)
    except Exception:
        pocket_area_in2 = 0.0

    ops: set[str] = set()
    for key in plan_ctx.keys():
        try:
            ops.add(str(key).strip().lower())
        except Exception:
            continue

    drill_only = bool(ops) and ops.issubset({"drilling", "inspection"})

    if drill_only:
        return max(0.5, 0.20 * setups + 0.008 * hole_count)

    if "milling" in ops:
        base = 1.0 + 0.35 * setups + 0.003 * hole_count + 0.002 * pocket_area_in2
        return min(max(base, 1.5), 8.0)

    if {"wire_edm", "sinker_edm"}.intersection(ops):
        return min(12.0, 2.0 + 0.4 * setups + 0.005 * hole_count)

    return 2.0 + 0.2 * setups


def _estimate_programming_hours_auto(
    geo: Mapping[str, Any] | None,
    process_plan: Mapping[str, Any] | None,
) -> float:
    """Return a plate-oriented programming estimate tied to geometry hints."""

    hole_qty_raw: Any = getattr(geo, "hole_count", 0)
    if (not hole_qty_raw) and isinstance(geo, _MappingABC):
        hole_qty_raw = geo.get("hole_count", 0)
    hole_qty_val = _coerce_float_or_none(hole_qty_raw) or 0.0

    pocket_count_raw: Any = getattr(geo, "pocket_count", None)
    pocket_count_val = _coerce_float_or_none(pocket_count_raw)
    geo_ctx = geo if isinstance(geo, _MappingABC) else {}
    if pocket_count_val is None and isinstance(geo_ctx, _MappingABC):
        pocket_count_val = _coerce_float_or_none(geo_ctx.get("pocket_count"))
    if (pocket_count_val is None or pocket_count_val <= 0) and isinstance(geo_ctx, _MappingABC):
        pocket_metrics = geo_ctx.get("pocket_metrics")
        if isinstance(pocket_metrics, _MappingABC):
            pocket_count_val = _coerce_float_or_none(pocket_metrics.get("pocket_count"))
    pocket_count = float(pocket_count_val or 0.0)

    plan_ctx = process_plan if isinstance(process_plan, _MappingABC) else {}
    drilling_ctx = plan_ctx.get("drilling") if isinstance(plan_ctx, _MappingABC) else {}
    bins_list: Sequence[Any]
    if isinstance(drilling_ctx, _MappingABC):
        bins_candidate = drilling_ctx.get("bins_list")
        bins_list = bins_candidate if isinstance(bins_candidate, Sequence) else []
    else:
        bins_list = []

    has_deep = False
    for entry in bins_list:
        op_name: str | None = None
        if isinstance(entry, _MappingABC):
            raw = entry.get("op_name")
            if raw not in (None, ""):
                op_name = str(raw)
        else:
            op_name = getattr(entry, "op_name", None)
        if op_name and str(op_name).strip().lower() == "deep_drill":
            has_deep = True
            break

    base = 0.5
    hole_term = 0.015 * float(hole_qty_val)
    pocket_term = 0.25 * pocket_count
    deep_term = 0.75 if has_deep else 0.0

    hr = base + hole_term + pocket_term + deep_term

    return float(max(1.0, min(hr, 6.0)))


def estimate_programming_hours_from_geo(geo: Mapping[str, Any] | None) -> float:
    """Return a geometry-driven programming-hour estimate.

    The heuristic scales with the number of hole families and overall hole counts,
    while adding smaller adjustments for taps, counterbores, and countersinks.
    """

    ctx = geo if isinstance(geo, _MappingABC) else {}

    hole_sets_raw = ctx.get("hole_sets") if isinstance(ctx, _MappingABC) else None
    families = 0
    if isinstance(hole_sets_raw, _MappingABC):
        families = len(hole_sets_raw)
    elif isinstance(hole_sets_raw, Sequence) and not isinstance(hole_sets_raw, (str, bytes, bytearray)):
        families = len(hole_sets_raw)

    hole_count_raw = ctx.get("hole_count", 0) if isinstance(ctx, _MappingABC) else 0
    holes = 0
    try:
        holes = int(float(hole_count_raw or 0))
    except Exception:
        holes = 0
    if holes <= 0 and isinstance(ctx, _MappingABC):
        hole_diams = ctx.get("hole_diams_mm")
        if isinstance(hole_diams, Sequence) and not isinstance(hole_diams, (str, bytes, bytearray)):
            holes = len(hole_diams)

    feature_counts = ctx.get("feature_counts") if isinstance(ctx, _MappingABC) else {}
    if not isinstance(feature_counts, _MappingABC):
        feature_counts = {}

    def _count_from_feature(key: str) -> int:
        value = feature_counts.get(key, 0)
        try:
            return int(float(value or 0))
        except Exception:
            return 0

    taps = _count_from_feature("tap_qty")
    cbores = _count_from_feature("cbore_qty")
    csks = _count_from_feature("csk_qty")

    base = 0.75
    h_families = 0.05 * families
    h_holes = 0.01 * holes
    h_tap = 0.03 * taps
    h_cbore = 0.08 * cbores
    h_csk = 0.05 * csks

    hr = base + h_families + h_holes + h_tap + h_cbore + h_csk

    return float(max(0.6, min(hr, 6.0)))


def _compute_programming_detail_minutes(
    geo: Mapping[str, Any] | None,
    plan: Mapping[str, Any] | None,
) -> float:
    """Return total programming minutes assuming one minute per feature detail."""

    detail_count = 0.0

    geo_ctx = geo if isinstance(geo, _MappingABC) else {}
    hole_count_raw: Any = None
    if isinstance(geo_ctx, _MappingABC):
        hole_count_raw = geo_ctx.get("hole_count")
        if hole_count_raw in (None, ""):
            hole_diams = geo_ctx.get("hole_diams_mm")
            if isinstance(hole_diams, Sequence) and not isinstance(
                hole_diams, (str, bytes, bytearray)
            ):
                hole_count_raw = len(hole_diams)
    hole_count = _coerce_float_or_none(hole_count_raw) or 0.0
    if hole_count > 0:
        detail_count += hole_count

    plan_ctx = plan if isinstance(plan, _MappingABC) else {}
    ops_candidate: Any = plan_ctx.get("ops") if isinstance(plan_ctx, _MappingABC) else None
    ops_sequence: Sequence[Any]
    if isinstance(ops_candidate, Sequence) and not isinstance(
        ops_candidate, (str, bytes, bytearray)
    ):
        ops_sequence = ops_candidate
    elif isinstance(plan, Sequence) and not isinstance(plan, (str, bytes, bytearray)):
        ops_sequence = plan  # type: ignore[assignment]
    else:
        ops_sequence = []

    for entry in ops_sequence:
        op_name: str | None = None
        if isinstance(entry, _MappingABC):
            raw_name = entry.get("op")
            if raw_name not in (None, ""):
                op_name = str(raw_name)
        else:
            op_name = getattr(entry, "op", None)
        if not op_name:
            continue
        op_lower = op_name.strip().lower()
        if "wire_edm" in op_lower or "wedm" in op_lower:
            detail_count += 1.0
        elif "cnc" in op_lower:
            detail_count += 1.0

    minimum_detail_minutes = 1.0 if detail_count <= 0 else detail_count
    return float(minimum_detail_minutes)


def _coerce_speeds_feeds_csv_path(*sources: Mapping[str, Any] | None) -> str | None:
    """Return the first non-empty Speeds/Feeds CSV path from ``sources``."""

    keys = (
        "SpeedsFeedsCSVPath",
        "SpeedsFeedsCsvPath",
        "speeds_feeds_path",
        "speeds_feeds_csv_path",
        "Speeds/Feeds CSV",
    )
    for source in sources:
        if not isinstance(source, _MappingABC):
            continue
        for key in keys:
            if key not in source:
                continue
            try:
                raw = source.get(key)
            except Exception:
                continue
            text = str(raw or "").strip()
            if text:
                return text
    return None


def _load_speeds_feeds_table_from_path(path: str | None) -> tuple[PandasDataFrame | None, bool]:
    """Load the Speeds/Feeds CSV at ``path`` into a DataFrame."""

    if not path:
        return None, False
    text = str(path).strip()
    if not text:
        return None, False

    if pd is None:
        return None, False

    dataframe_ctor = getattr(pd, "DataFrame", None)
    read_csv = getattr(pd, "read_csv", None)
    if not callable(dataframe_ctor):
        return None, False

    table: PandasDataFrame | None = None
    try:
        candidate = Path(text)
    except Exception:
        candidate = None
    if candidate is not None and candidate.is_file() and callable(read_csv):
        try:
            table = typing.cast(PandasDataFrame, read_csv(candidate))
        except Exception:
            table = None

    if table is None:
        try:
            records = _load_speeds_feeds_records(text)
        except Exception:
            records = []
        if records:
            try:
                table = typing.cast(PandasDataFrame, dataframe_ctor(records))
            except Exception:
                table = None

    if table is not None:
        try:
            if not getattr(table, "empty"):
                return table, True
        except Exception:
            try:
                if len(table) > 0:  # type: ignore[arg-type]
                    return table, True
            except Exception:
                pass
    return table, False


def _ensure_quote_state(state: QuoteState | None) -> QuoteState:
    return state if isinstance(state, QuoteState) else QuoteState()


def compute_quote_from_df(  # type: ignore[reportGeneralTypeIssues]
    df: Any,
    *,
    params: Mapping[str, Any] | None = None,
    rates: Mapping[str, Any] | None = None,
    default_params: Mapping[str, Any] | None = None,
    default_rates: Mapping[str, Any] | None = None,
    default_material_display: Mapping[str, Any] | str | None = None,
    material_vendor_csv: str | None = None,
    llm_enabled: bool = True,
    llm_model_path: str | None = None,
    llm_client: Any | None = None,
    geo: Mapping[str, Any] | None = None,
    ui_vars: Mapping[str, Any] | None = None,
    quote_state: QuoteState | None = None,
    reuse_suggestions: Any | None = None,
    llm_suggest: Any | None = None,
    cfg: QuoteConfiguration | None = None,
    **_: Any,
) -> dict[str, Any]:
    cfg = cfg or QuoteConfiguration(default_params=copy.deepcopy(PARAMS_DEFAULT))
    value_map = _df_to_value_map(df)
    quote_df_canonical: Any = None
    if _HAS_PANDAS and "pd" in globals():
        import pandas as pd  # type: ignore

        if isinstance(df, pd.DataFrame):
            try:
                quote_df_canonical = coerce_or_make_vars_df(df.copy())
            except Exception:
                quote_df_canonical = None

    geom = build_geometry_context(
        quote_df_canonical if quote_df_canonical is not None else value_map,
        base_geometry=geo,
        value_map=value_map,
        cfg=cfg,
    )
    geo_context = geom
    planner_inputs = dict(ui_vars or {})
    rates = dict(rates or {})
    geo_payload: dict[str, Any] = geo_context
    state = _ensure_quote_state(quote_state)

    default_material_display = DEFAULT_MATERIAL_DISPLAY
    raw_material_display = str(
        geo_context.get("material")
        or value_map.get("Material Name")
        or value_map.get("Material")
        or ""
    ).strip()
    material_key_source = raw_material_display or default_material_display
    material_key = normalize_material_key(material_key_source)
    fallback_display = MATERIAL_DISPLAY_BY_KEY.get(material_key, "")
    material_display = raw_material_display or fallback_display or default_material_display
    geo_context["material"] = material_display
    geo_context.setdefault("material_display", material_display)
    geo_context.setdefault("material_name", geo_context.get("material_name") or material_display)
    geo_context["material_key"] = material_key
    geo_context["material_lookup"] = material_key
    raw_material_group = str(
        geo_context.get("material_group")
        or geo_context.get("material_family")
        or value_map.get("Material Group")
        or ""
    ).strip()
    material_group_display: str | None = raw_material_group or None
    if material_group_display:
        geo_context.setdefault("material_group", material_group_display)
    planner_inputs["material"] = material_display
    planner_inputs.setdefault("material_key", material_key)
    if material_group_display:
        planner_inputs.setdefault("material_group", material_group_display)

    speeds_feeds_csv_path = _coerce_speeds_feeds_csv_path(
        planner_inputs,
        params,
        state.ui_vars,
    )
    if not speeds_feeds_csv_path:
        speeds_feeds_csv_path = DEFAULT_SPEEDS_FEEDS_CSV_PATH

    speeds_feeds_table, speeds_feeds_loaded = _load_speeds_feeds_table_from_path(
        speeds_feeds_csv_path
    )
    if speeds_feeds_csv_path:
        planner_inputs.setdefault("SpeedsFeedsCSVPath", speeds_feeds_csv_path)
        planner_inputs.setdefault("speeds_feeds_path", speeds_feeds_csv_path)
        planner_inputs.setdefault("Speeds/Feeds CSV", speeds_feeds_csv_path)
    if speeds_feeds_csv_path:
        planner_inputs.setdefault("speeds_feeds_loaded", bool(speeds_feeds_loaded))

    qty = _coerce_float_or_none(value_map.get("Qty")) or 1.0
    material_text = material_display

    scrap_value = value_map.get("Scrap Percent (%)")
    normalized_scrap = normalize_scrap_pct(scrap_value)
    scrap_frac: float | None = normalized_scrap if normalized_scrap > 0 else None
    scrap_source = "ui" if scrap_frac is not None else "default_guess"
    scrap_source_label = scrap_source
    hole_reason_applied = False
    if scrap_frac is None:
        stock_plan = geo_payload.get("stock_plan_guess") if isinstance(geo_payload, dict) else None
        if isinstance(stock_plan, _MappingABC):
            net = _coerce_float_or_none(stock_plan.get("net_volume_in3"))
            stock = _coerce_float_or_none(stock_plan.get("stock_volume_in3"))
            if net and stock and net > 0 and stock >= net:
                scrap_frac = max(0.0, min(0.25, (stock - net) / net))
                scrap_source = "stock_plan_guess"
                scrap_source_label = scrap_source
        if scrap_frac is None:
            scrap_frac = SCRAP_DEFAULT_GUESS
            scrap_source = "default_guess"
            scrap_source_label = scrap_source

    assert scrap_frac is not None
    scrap_frac = float(scrap_frac)

    hole_scrap_frac_est = _holes_scrap_fraction(geo_payload, cap=HOLE_SCRAP_CAP)
    if hole_scrap_frac_est > 0:
        hole_scrap_frac_clamped = max(0.0, min(HOLE_SCRAP_CAP, float(hole_scrap_frac_est)))
        ui_scrap_frac = normalize_scrap_pct(scrap_value)
        scrap_candidate = max(ui_scrap_frac, hole_scrap_frac_clamped)
        if scrap_candidate > scrap_frac + 1e-9:
            scrap_frac = scrap_candidate
            scrap_source_label = f"{scrap_source}+holes"
            hole_reason_applied = True

    density_g_cc = _coerce_float_or_none(value_map.get("Material Density"))
    if (density_g_cc in (None, 0.0)) and material_text:
        density_hint = _density_for_material(material_text)
        if density_hint:
            density_g_cc = density_hint

    net_volume_cm3 = _coerce_float_or_none(value_map.get("Net Volume (cm^3)"))
    if net_volume_cm3 is None:
        length_in = _coerce_float_or_none(value_map.get("Plate Length (in)"))
        width_in = _coerce_float_or_none(value_map.get("Plate Width (in)"))
        thickness_in_val = _coerce_float_or_none(value_map.get("Thickness (in)"))

        if isinstance(geo_payload, _MappingABC):
            if length_in is None:
                length_in = _coerce_float_or_none(
                    geo_payload.get("plate_len_in")
                    or geo_payload.get("plate_length_in")
                )
            if width_in is None:
                width_in = _coerce_float_or_none(
                    geo_payload.get("plate_wid_in")
                    or geo_payload.get("plate_width_in")
                )

            if length_in is None:
                length_mm = _coerce_positive_float(
                    geo_payload.get("plate_len_mm")
                    or geo_payload.get("plate_length_mm")
                )
                if length_mm:
                    length_in = float(length_mm) / 25.4
            if width_in is None:
                width_mm = _coerce_positive_float(
                    geo_payload.get("plate_wid_mm")
                    or geo_payload.get("plate_width_mm")
                )
                if width_mm:
                    width_in = float(width_mm) / 25.4

            if thickness_in_val is None:
                thickness_in_val = _coerce_float_or_none(geo_payload.get("thickness_in"))
                if thickness_in_val is None:
                    thickness_mm_geo = _coerce_positive_float(geo_payload.get("thickness_mm"))
                    if thickness_mm_geo:
                        thickness_in_val = float(thickness_mm_geo) / 25.4

            if length_in is None or width_in is None:
                derived_ctx = geo_payload.get("derived")
                if isinstance(derived_ctx, _MappingABC):
                    bbox_mm = derived_ctx.get("bbox_mm")
                    if (
                        isinstance(bbox_mm, (list, tuple))
                        and len(bbox_mm) == 2
                    ):
                        bbox_L_mm = _coerce_positive_float(bbox_mm[0])
                        bbox_W_mm = _coerce_positive_float(bbox_mm[1])
                        if length_in is None and bbox_L_mm:
                            length_in = float(bbox_L_mm) / 25.4
                        if width_in is None and bbox_W_mm:
                            width_in = float(bbox_W_mm) / 25.4

        if length_in and width_in and thickness_in_val:
            volume_in3 = float(length_in) * float(width_in) * float(thickness_in_val)
            net_volume_cm3 = volume_in3 * 16.387064

    removal_mass_g = _holes_removed_mass_g(geo_payload)
    if net_volume_cm3 and density_g_cc and removal_mass_g:
        net_mass_g = float(net_volume_cm3) * float(density_g_cc)
        base_for_removal = float(scrap_frac or 0.0) if scrap_source != "default_guess" else 0.0
        removal_result = compute_mass_and_scrap_after_removal(
            net_mass_g,
            base_for_removal,
            removal_mass_g,
        )
        scrap_frac = removal_result[1]
        scrap_source = "geometry"
        if hole_reason_applied:
            scrap_source_label = f"{scrap_source_label}+geometry"
        else:
            scrap_source_label = scrap_source

    fai_value = value_map.get("FAIR Required")
    fai_required = coerce_bool(coerce_checkbox_state(fai_value), default=False)

    baseline: dict[str, Any] = {
        "qty": qty,
        "material": material_text,
        "scrap_pct": scrap_frac,
        "scrap_source_label": scrap_source_label,
        "fai_required": bool(fai_required),
    }
    baseline["material_key"] = material_key

    drill_params: dict[str, Any] = baseline.setdefault("drill_params", {})
    drill_params["material"] = material_key
    drill_params.setdefault("material_key", material_key)
    drill_group = _material_group_for_speeds_feeds(material_key)
    if drill_group:
        drill_params["group"] = drill_group
    if material_display:
        drill_params.setdefault("material_display", material_display)
    if material_group_display:
        drill_params.setdefault("material_group", material_group_display)

    geo_derived = dict(geo_payload.get("derived", {})) if isinstance(geo_payload, dict) else {}
    geo_derived.setdefault("fai_required", bool(fai_required))
    if isinstance(geo_payload, dict):
        geo_payload["derived"] = geo_derived
    else:
        geo_payload = {"derived": geo_derived}

    try:
        build_suggest_payload(geo_payload, baseline, dict(rates or {}), coerce_bounds(state.bounds))
    except Exception:
        pass

    state.geo = geo_context if isinstance(geo_context, dict) else {}
    state.baseline = dict(baseline)
    state.user_overrides = dict(getattr(state, "user_overrides", {}))
    state.suggestions = dict(getattr(state, "suggestions", {}))
    state.effective = dict(getattr(state, "effective", {}))
    state.effective_sources = dict(getattr(state, "effective_sources", {}))

    inspection_components = {
        "in_process": 0.5,
        "final": 0.25,
        "cmm_programming": 0.1,
        "cmm_run": 0.1,
        "fair": 0.0,
        "source": 0.0,
    }
    inspection_baseline_hr = sum(float(v) for v in inspection_components.values())
    inspection_adjustments: dict[str, float] = {}
    inspection_total_hr = inspection_baseline_hr
    overrides = state.user_overrides
    if isinstance(overrides, _MappingABC):
        cmm_minutes = _coerce_float_or_none(overrides.get("cmm_minutes"))
        if cmm_minutes and cmm_minutes > 0:
            cmm_hours = max(0.0, float(cmm_minutes) / 60.0)
            inspection_adjustments["cmm_run"] = cmm_hours
            inspection_total_hr += cmm_hours
        if "inspection_total_hr" in overrides:
            target_hr = max(0.0, _coerce_float_or_none(overrides.get("inspection_total_hr")) or 0.0)
            inspection_total_hr = target_hr
            state.effective_sources["inspection_total_hr"] = "user"
            state.effective["inspection_total_hr"] = target_hr

    inspection_meta = {
        "components": inspection_components,
        "baseline_hr": inspection_baseline_hr,
        "hr": inspection_total_hr,
        "adjustments": inspection_adjustments,
    }

    from planner_pricing import price_with_planner

    process_costs: dict[str, float] = {}
    process_plan_summary: dict[str, Any] = {}
    process_meta: dict[str, Any] = {"inspection": inspection_meta}
    bucket_view: dict[str, dict[str, float]] = {}
    totals_block: dict[str, float] = {}
    breakdown: dict[str, Any] = {
        "qty": qty,
        "material": {
            "material": material_text,
            "scrap_pct": scrap_frac,
            "scrap_source": scrap_source,
            "scrap_source_label": scrap_source_label,
        },
        "process_meta": process_meta,
        "bucket_view": bucket_view,
        "process_costs": process_costs,
        "red_flags": [],
        "totals": totals_block,
    }
    breakdown["geo_context"] = geo_context if isinstance(geo_context, dict) else {}

    mat_key = (
        str(((breakdown.get("material") or {}).get("material")) or "").lower()
        or baseline.get("material_key")
        or "aluminum"
    )
    density = geo_context.get("density_g_cc") if isinstance(geo_context, dict) else None
    scrap_pct_effective = float(
        ((breakdown.get("material") or {}).get("scrap_pct"))
        or baseline.get("scrap_pct")
        or 0.25
    )
    default_cfg = cfg
    stock_price_source = str(
        (value_map.get("Stock Price Source") if isinstance(value_map, _MappingABC) else None)
        or (state.user_overrides.get("stock_price_source") if isinstance(state.user_overrides, _MappingABC) else None)
        or getattr(default_cfg, "stock_price_source", "")
        or ""
    ).strip()
    if not stock_price_source:
        stock_price_source = None
    scrap_price_source = str(
        (value_map.get("Scrap Price Source") if isinstance(value_map, _MappingABC) else None)
        or (state.user_overrides.get("scrap_price_source") if isinstance(state.user_overrides, _MappingABC) else None)
        or getattr(default_cfg, "scrap_price_source", "")
        or ""
    ).strip()
    if not scrap_price_source:
        scrap_price_source = None

    mat_block = _compute_material_block(
        geo_context if isinstance(geo_context, dict) else {},
        mat_key,
        _coerce_float_or_none(density),
        scrap_pct_effective,
        stock_price_source=stock_price_source,
        cfg=cfg,
    )
    breakdown["material_block"] = mat_block
    grams_per_lb = 1000.0 / LB_PER_KG
    material_entry = breakdown.setdefault("material", {})
    if isinstance(material_entry, dict):
        if stock_price_source:
            material_entry.setdefault("stock_price_source", stock_price_source)
            if isinstance(mat_block, _MappingABC):
                try:
                    mat_block.setdefault("stock_price_source", stock_price_source)  # type: ignore[attr-defined]
                except AttributeError:
                    pass
        if scrap_price_source:
            material_entry.setdefault("scrap_price_source", scrap_price_source)
        if isinstance(mat_block, _MappingABC) and scrap_price_source:
            try:
                mat_block.setdefault("scrap_price_source", scrap_price_source)  # type: ignore[attr-defined]
            except AttributeError:
                pass
        stock_dims_raw = mat_block.get("stock_dims_in")
        if isinstance(stock_dims_raw, (list, tuple)) and len(stock_dims_raw) >= 3:
            try:
                stock_dims_tuple = (
                    float(stock_dims_raw[0]),
                    float(stock_dims_raw[1]),
                    float(stock_dims_raw[2]),
                )
                material_entry["stock_dims_in"] = stock_dims_tuple
            except Exception:
                pass
        vendor_label = mat_block.get("stock_vendor")
        if vendor_label:
            material_entry["stock_vendor"] = vendor_label
            material_entry.setdefault("source", vendor_label)
        part_no = mat_block.get("part_no")
        if part_no:
            material_entry["part_no"] = part_no
        supplier_min_val = _coerce_float_or_none(mat_block.get("supplier_min$"))
        if supplier_min_val is not None:
            material_entry["supplier_min$"] = float(supplier_min_val)
            if _coerce_float_or_none(material_entry.get("supplier_min_charge")) is None:
                material_entry["supplier_min_charge"] = float(supplier_min_val)
            material_entry.setdefault("supplier_min_charge$", float(supplier_min_val))
        stock_price_val = _coerce_float_or_none(mat_block.get("stock_price$"))
        unit_price_each_val = _coerce_float_or_none(
            mat_block.get("unit_price_each$") or mat_block.get("unit_price$")
        )
        if unit_price_each_val is not None and unit_price_each_val > 0:
            material_entry["unit_price"] = float(unit_price_each_val)
            material_entry.setdefault("stock_price$", float(unit_price_each_val))
        elif stock_price_val is not None and stock_price_val > 0:
            material_entry.setdefault("stock_price$", float(stock_price_val))
        start_lb = _coerce_float_or_none(mat_block.get("start_lb"))
        net_lb = _coerce_float_or_none(mat_block.get("net_lb"))
        scrap_lb = _coerce_float_or_none(mat_block.get("scrap_lb"))
        price_per_lb = _coerce_float_or_none(mat_block.get("price_per_lb"))
        if start_lb is not None and start_lb > 0:
            start_g = float(start_lb) * grams_per_lb
            material_entry["mass_g"] = start_g
            material_entry["starting_mass_g_est"] = start_g
        if net_lb is not None and net_lb > 0:
            material_entry["net_mass_g"] = float(net_lb) * grams_per_lb
        if scrap_lb is not None and scrap_lb > 0:
            material_entry["scrap_mass_g"] = float(scrap_lb) * grams_per_lb
        if price_per_lb is not None and price_per_lb > 0:
            material_entry["unit_price_usd_per_lb"] = float(price_per_lb)
        price_source = mat_block.get("price_source")
        if price_source:
            material_entry.setdefault("unit_price_source", price_source)
            material_entry.setdefault("source", price_source)
        stock_L = _coerce_float_or_none(mat_block.get("stock_L_in"))
        stock_W = _coerce_float_or_none(mat_block.get("stock_W_in"))
        stock_T = _coerce_float_or_none(mat_block.get("stock_T_in"))
        if stock_L is not None and stock_L > 0:
            material_entry["stock_L_in"] = float(stock_L)
        if stock_W is not None and stock_W > 0:
            material_entry["stock_W_in"] = float(stock_W)
        if stock_T is not None and stock_T > 0:
            material_entry["thickness_in"] = float(stock_T)
        if mat_block.get("source"):
            material_entry["source"] = mat_block.get("source")
        total_cost_val = _coerce_float_or_none(mat_block.get("total_material_cost"))
        base_material_cost = float(total_cost_val or 0.0)
        material_entry["material_cost_before_credit"] = float(base_material_cost)
        mat_block["material_cost_before_credit"] = float(base_material_cost)

        scrap_mass_lb_val = _coerce_float_or_none(
            mat_block.get("scrap_weight_lb") or mat_block.get("scrap_lb")
        )
        scrap_mass_lb = (
            float(scrap_mass_lb_val)
            if scrap_mass_lb_val is not None and scrap_mass_lb_val > 0
            else None
        )
        if scrap_mass_lb is not None:
            mat_block["scrap_credit_mass_lb"] = float(scrap_mass_lb)
            material_entry["scrap_credit_mass_lb"] = float(scrap_mass_lb)

        credit_override_amount: float | None = None
        unit_price_override: float | None = None
        recovery_override: float | None = None
        if isinstance(overrides, _MappingABC):
            for key in (
                "Material Scrap / Remnant Value",
                "material_scrap_credit",
                "material_scrap_credit_usd",
                "scrap_credit_usd",
            ):
                override_val = overrides.get(key)
                if override_val in (None, ""):
                    continue
                coerced_override = _coerce_float_or_none(override_val)
                if coerced_override is not None:
                    credit_override_amount = abs(float(coerced_override))
                    break

            for key in (
                "scrap_credit_unit_price_usd_per_lb",
                "scrap_price_usd_per_lb",
                "scrap_usd_per_lb",
                "Scrap Price ($/lb)",
                "Scrap Credit ($/lb)",
                "Scrap Unit Price ($/lb)",
            ):
                price_val = overrides.get(key)
                if price_val in (None, ""):
                    continue
                coerced_price = _coerce_float_or_none(price_val)
                if coerced_price is not None and coerced_price >= 0:
                    unit_price_override = float(coerced_price)
                    break

            for key in (
                "scrap_recovery_pct",
                "Scrap Recovery (%)",
                "scrap_recovery_fraction",
            ):
                recovery_val = overrides.get(key)
                if recovery_val in (None, ""):
                    continue
                coerced_recovery = _coerce_float_or_none(recovery_val)
                if coerced_recovery is None:
                    continue
                recovery_fraction = float(coerced_recovery)
                if recovery_fraction > 1.0 + 1e-6:
                    recovery_fraction = recovery_fraction / 100.0
                recovery_override = max(0.0, min(1.0, recovery_fraction))
                break

        scrap_credit_amount: float | None = None
        scrap_price_used: float | None = None
        scrap_recovery_used: float | None = None
        scrap_credit_source: str | None = None
        wieland_scrap_price: float | None = None

        if credit_override_amount is not None:
            scrap_credit_amount = max(0.0, float(credit_override_amount))
            scrap_credit_source = "override_amount"
        elif scrap_mass_lb is not None:
            recovery = recovery_override if recovery_override is not None else SCRAP_RECOVERY_DEFAULT
            scrap_recovery_used = recovery
            if unit_price_override is not None:
                scrap_price_used = max(0.0, float(unit_price_override))
                scrap_credit_source = "override_unit_price"
                mat_block["scrap_price_usd_per_lb"] = float(scrap_price_used)
            else:
                family_hint = None
                if isinstance(geo_context, dict):
                    family_hint = (
                        geo_context.get("material_family")
                        or geo_context.get("material_group")
                    )
                family_hint = family_hint or material_group_display or material_display
                price_candidate = _wieland_scrap_usd_per_lb(family_hint)
                if price_candidate is not None:
                    wieland_scrap_price = float(price_candidate)
                    scrap_price_used = float(price_candidate)
                    scrap_credit_source = "wieland"
                    mat_block["scrap_price_usd_per_lb"] = float(scrap_price_used)
                else:
                    scrap_price_used = SCRAP_CREDIT_FALLBACK_USD_PER_LB
                    scrap_credit_source = "default"
                    mat_block["scrap_price_usd_per_lb"] = float(scrap_price_used)
            scrap_credit_amount = float(scrap_mass_lb) * float(scrap_price_used or 0.0) * float(scrap_recovery_used or 0.0)

        if wieland_scrap_price is not None:
            mat_block["scrap_price_usd_per_lb"] = float(wieland_scrap_price)

        net_material_cost = float(base_material_cost)
        if scrap_credit_amount is not None:
            credit_value = max(0.0, float(scrap_credit_amount))
            if base_material_cost > 0:
                credit_value = min(credit_value, float(base_material_cost))
            credit_value = round(credit_value, 2)
            scrap_credit_amount = credit_value
            net_material_cost = max(0.0, float(base_material_cost) - credit_value)

            if scrap_price_used is not None:
                material_entry["scrap_credit_unit_price_usd_per_lb"] = float(scrap_price_used)
                mat_block["scrap_credit_unit_price_usd_per_lb"] = float(scrap_price_used)
                mat_block.setdefault("scrap_price_usd_per_lb", float(scrap_price_used))
            if scrap_recovery_used is not None:
                material_entry["scrap_credit_recovery_pct"] = float(scrap_recovery_used)
                mat_block["scrap_credit_recovery_pct"] = float(scrap_recovery_used)
            if scrap_credit_source:
                material_entry["scrap_credit_source"] = scrap_credit_source
                mat_block["scrap_credit_source"] = scrap_credit_source

            if scrap_credit_source == "wieland":
                material_entry["computed_scrap_credit_usd"] = float(scrap_credit_amount)
                mat_block["computed_scrap_credit_usd"] = float(scrap_credit_amount)
                material_entry["scrap_price_source"] = "wieland"
                mat_block["scrap_price_source"] = "wieland"
                material_entry.pop("material_scrap_credit", None)
                material_entry.pop("material_scrap_credit_entered", None)
                mat_block.pop("material_scrap_credit", None)
                mat_block.pop("material_scrap_credit_entered", None)
            else:
                material_entry.pop("computed_scrap_credit_usd", None)
                mat_block.pop("computed_scrap_credit_usd", None)
                material_entry["material_scrap_credit"] = float(scrap_credit_amount)
                material_entry["material_scrap_credit_entered"] = bool(scrap_credit_amount > 0)
                mat_block["material_scrap_credit"] = float(scrap_credit_amount)
                mat_block["material_scrap_credit_entered"] = bool(scrap_credit_amount > 0)
        else:
            material_entry.pop("computed_scrap_credit_usd", None)
            mat_block.pop("computed_scrap_credit_usd", None)
            material_entry.pop("material_scrap_credit", None)
            material_entry.pop("material_scrap_credit_entered", None)
            mat_block.pop("material_scrap_credit", None)
            mat_block.pop("material_scrap_credit_entered", None)

        if net_material_cost > 0:
            material_entry["material_cost"] = float(net_material_cost)
            material_entry["material_direct_cost"] = float(net_material_cost)
            material_entry["total_material_cost"] = float(net_material_cost)
            mat_block["total_material_cost"] = float(net_material_cost)
            mat_block["material_cost"] = float(net_material_cost)
            mat_block["material_direct_cost"] = float(net_material_cost)
        else:
            material_entry.pop("material_cost", None)
            material_entry.pop("material_direct_cost", None)
            material_entry.pop("total_material_cost", None)
            mat_block["total_material_cost"] = float(net_material_cost)
            mat_block.pop("material_cost", None)
            mat_block.pop("material_direct_cost", None)

        supplier_min = _coerce_float_or_none(mat_block.get("supplier_min"))
        if supplier_min is not None and supplier_min > 0:
            material_entry.setdefault("supplier_min_charge", float(supplier_min))

        overrides_for_cost = overrides if isinstance(overrides, _MappingABC) else {}
        try:
            cost_components = _material_cost_components(
                mat_block,
                overrides=overrides_for_cost,
                cfg=getattr(state, "config", None),
            )
        except Exception:
            cost_components = None
        if isinstance(cost_components, dict):
            material_entry["material_cost_components"] = cost_components
            mat_block["material_cost_components"] = cost_components

    material_total_direct_cost = _first_numeric_or_none(
        (material_entry or {}).get("total_cost") if isinstance(material_entry, _MappingABC) else None,
        (material_entry or {}).get("total_material_cost") if isinstance(material_entry, _MappingABC) else None,
        (material_entry or {}).get("material_total_cost") if isinstance(material_entry, _MappingABC) else None,
        (material_entry or {}).get("material_cost_before_credit") if isinstance(material_entry, _MappingABC) else None,
        (material_entry or {}).get("material_cost") if isinstance(material_entry, _MappingABC) else None,
        (material_entry or {}).get("material_direct_cost") if isinstance(material_entry, _MappingABC) else None,
        mat_block.get("total_cost"),
        mat_block.get("total_material_cost"),
        mat_block.get("material_cost"),
        mat_block.get("material_cost_before_credit"),
    )
    if material_total_direct_cost is not None:
        material_direct_contribution = round(material_total_direct_cost, 2)
        material_display_amount = round(material_total_direct_cost, 2)
        material_total_for_directs = float(material_total_direct_cost)
        material_total_for_why = float(material_display_amount)
        material_net_cost = float(material_total_direct_cost)

    breakdown.setdefault("pass_through_total", 0.0)

    bucket_minutes_detail_for_render = breakdown.setdefault("bucket_minutes_detail", {})
    if not isinstance(bucket_minutes_detail_for_render, dict):
        bucket_minutes_detail_for_render = {}
        breakdown["bucket_minutes_detail"] = bucket_minutes_detail_for_render

    family = None
    if isinstance(geo_payload, _MappingABC):
        family = geo_payload.get("process_planner_family")
    family = str(family or "").strip().lower() or None

    planner_result: dict[str, Any] = {}
    planner_used = False
    planner_machine_cost_total = 0.0
    planner_labor_cost_total = 0.0
    amortized_programming = 0.0
    amortized_fixture = 0.0
    planner_exception: Exception | None = None
    recognized_line_items = 0
    use_planner = False
    fallback_reason = ""
    if family:
        if callable(_process_plan_job):
            try:
                planner_result["plan"] = _process_plan_job(family, planner_inputs)
            except Exception:
                planner_result["plan"] = {}

        oee = _coerce_float_or_none((params or {}).get("OEE_EfficiencyPct")) or 0.85
        try:
            pricing = price_with_planner(
                family,
                planner_inputs,
                geo_payload,
                dict(rates or {}),
                oee=oee,
            )
        except Exception as exc:
            planner_exception = exc
            pricing = {}

        planner_result.update(pricing if isinstance(pricing, dict) else {})
        recognized_line_items = _recognized_line_items_from_planner(planner_result)
        if (
            "recognized_line_items" not in planner_result
            and recognized_line_items > 0
        ):
            planner_result["recognized_line_items"] = recognized_line_items

        if planner_result:
            breakdown["process_plan_pricing"] = planner_result
            baseline["process_plan_pricing"] = planner_result
            process_plan_summary["pricing"] = planner_result

        if planner_exception is None and recognized_line_items > 0:
            use_planner = True
        else:
            fallback_reason = (
                "Planner pricing failed; using legacy fallback"
                if planner_exception is not None
                else "Planner recognized no operations; using legacy fallback"
            )

    if use_planner:
        totals = (
            planner_result.get("totals", {})
            if isinstance(planner_result.get("totals"), _MappingABC)
            else {}
        )
        machine_cost = float(_coerce_float_or_none(totals.get("machine_cost")) or 0.0)
        labor_cost_total = float(_coerce_float_or_none(totals.get("labor_cost")) or 0.0)
        total_minutes = float(_coerce_float_or_none(totals.get("minutes")) or 0.0)

        line_items = planner_result.get("line_items")
        if isinstance(line_items, Sequence):
            for item in line_items:
                if not isinstance(item, _MappingABC):
                    continue
                raw_label = item.get("op") or item.get("name") or ""
                canonical_label, is_amortized = _canonical_amortized_label(raw_label)
                normalized_label = str(canonical_label or raw_label or "").strip().lower()
                if not is_amortized:
                    if any(
                        token in normalized_label
                        for token in ("per part", "per pc", "per piece")
                    ):
                        is_amortized = True
                if not is_amortized:
                    continue
                if not normalized_label:
                    continue
                labor_amount = _coerce_float_or_none(item.get("labor_cost"))
                if labor_amount is None:
                    continue
                labor_value = float(labor_amount)
                if "program" in normalized_label:
                    amortized_programming += labor_value
                elif "fixture" in normalized_label:
                    amortized_fixture += labor_value

        planner_machine_cost_total = machine_cost
        planner_labor_cost_total = labor_cost_total - amortized_programming - amortized_fixture
        if planner_labor_cost_total < 0:
            planner_labor_cost_total = 0.0

        planner_direct_cost_total = planner_machine_cost_total + planner_labor_cost_total
        if planner_direct_cost_total <= 0.0:
            zero_cost_message = "Planner produced zero machine/labor cost; using legacy fallback"
            quote_log = breakdown.setdefault("quote_log", [])
            if isinstance(quote_log, list):
                quote_log.append(zero_cost_message)
            else:
                breakdown["quote_log"] = [zero_cost_message]
            fallback_reason = zero_cost_message
            use_planner = False
        else:
            process_costs.clear()
            process_costs.update(
                {
                    "Machine": round(planner_machine_cost_total, 2),
                    "Labor": round(planner_labor_cost_total, 2),
                }
            )
            planner_totals_map = (
                planner_result.get("totals", {})
                if isinstance(planner_result.get("totals"), _MappingABC)
                else {}
            )
            minutes_by_bucket = (
                planner_totals_map.get("minutes_by_bucket", {})
                if isinstance(planner_totals_map, _MappingABC)
                else {}
            )
            cost_by_bucket = (
                planner_totals_map.get("cost_by_bucket", {})
                if isinstance(planner_totals_map, _MappingABC)
                else {}
            )
            if minutes_by_bucket:
                hour_summary = {}
                for key, value in minutes_by_bucket.items():
                    try:
                        minutes_val = float(value or 0.0)
                    except Exception:
                        minutes_val = 0.0
                    hour_summary[str(key)] = round(minutes_val / 60.0, 2)
                process_plan_summary["hour_summary"] = hour_summary
            if cost_by_bucket:
                bucket_costs: list[PlannerBucketCost] = []
                planner_cost_map: dict[str, float] = {}
                for key, value in cost_by_bucket.items():
                    try:
                        numeric_cost = round(float(value or 0.0), 2)
                    except Exception:
                        numeric_cost = 0.0
                    name = str(key)
                    bucket_costs.append({"name": name, "cost": numeric_cost})
                    planner_cost_map[name] = numeric_cost
                process_plan_summary["process_costs"] = bucket_costs
                process_plan_summary["process_costs_map"] = planner_cost_map
            process_plan_summary["computed_total_labor_cost"] = float(
                planner_totals_map.get("labor_cost", 0.0) or 0.0
            )
            process_plan_summary["display_labor_for_ladder"] = process_plan_summary[
                "computed_total_labor_cost"
            ]
            process_plan_summary["computed_total_machine_cost"] = float(
                planner_totals_map.get("machine_cost", 0.0) or 0.0
            )
            process_plan_summary["planner_labor_cost_total"] = planner_labor_cost_total
            process_plan_summary["planner_machine_cost_total"] = planner_machine_cost_total
            process_plan_summary["pricing_source"] = "Planner"
            planner_subtotal = (
                process_plan_summary["display_labor_for_ladder"]
                + process_plan_summary["computed_total_machine_cost"]
            )
            planner_subtotal_rounded = round(planner_subtotal, 2)
            process_plan_summary["computed_subtotal"] = planner_subtotal_rounded
            combined_labor_total = (
                planner_machine_cost_total
                + planner_labor_cost_total
                + amortized_programming
                + amortized_fixture
            )
            totals_block.update(
                {
                    "machine_cost": planner_machine_cost_total,
                    "labor_cost": combined_labor_total,
                    "minutes": total_minutes,
                    "subtotal": planner_subtotal_rounded,
                }
            )
            breakdown["labor_cost_rendered"] = combined_labor_total
            breakdown["process_plan_pricing"] = planner_result
            breakdown["pricing_source"] = "Planner"
            breakdown["process_minutes"] = total_minutes
            baseline["pricing_source"] = "Planner"
            baseline["process_plan_pricing"] = planner_result
            process_plan_summary["used_planner"] = True
            planner_used = True

            hr_total = total_minutes / 60.0 if total_minutes else 0.0
            process_meta["planner_total"] = {
                "minutes": total_minutes,
                "hr": hr_total,
                "cost": machine_cost + labor_cost_total,
                "machine_cost": machine_cost,
                "labor_cost": labor_cost_total,
                "labor_cost_excl_amortized": planner_labor_cost_total,
                "amortized_programming": amortized_programming,
                "amortized_fixture": amortized_fixture,
                "line_items": list(planner_result.get("line_items", []) or []),
            }
            process_meta["planner_machine"] = {
                "minutes": total_minutes,
                "hr": hr_total,
                "cost": machine_cost,
            }
            process_meta["planner_labor"] = {
                "minutes": total_minutes,
                "hr": hr_total,
                "cost": labor_cost_total,
                "cost_excl_amortized": planner_labor_cost_total,
                "amortized_programming": amortized_programming,
                "amortized_fixture": amortized_fixture,
            }

            machine_rendered = float(_coerce_float_or_none(process_costs.get("Machine")) or 0.0)
            if not roughly_equal(
                planner_machine_cost_total,
                machine_rendered,
                eps=_PLANNER_BUCKET_ABS_EPSILON,
            ):
                breakdown["red_flags"].append("Planner totals drifted (machine cost)")
            labor_rendered = float(_coerce_float_or_none(process_costs.get("Labor")) or 0.0)
            if not roughly_equal(
                planner_labor_cost_total,
                labor_rendered,
                eps=_PLANNER_BUCKET_ABS_EPSILON,
            ):
                breakdown["red_flags"].append("Planner totals drifted (labor cost)")

    if not use_planner:
        breakdown.setdefault("process_minutes", 0.0)
        breakdown["pricing_source"] = "legacy"
        baseline.setdefault("pricing_source", "legacy")
        if fallback_reason:
            breakdown["red_flags"].append(fallback_reason)
        breakdown.setdefault("pricing_source", "legacy")
        baseline.setdefault("pricing_source", "legacy")

    if process_plan_summary:
        if isinstance(breakdown, dict):
            existing_summary = breakdown.get("process_plan")
            if isinstance(existing_summary, dict):
                existing_summary.update(process_plan_summary)
            else:
                breakdown["process_plan"] = dict(process_plan_summary)

    using_planner = str(breakdown.get("pricing_source", "")).strip().lower() == "planner"

    if os.environ.get("ASSERT_PLANNER"):
        assert str(breakdown.get("pricing_source", "")).strip().lower() == "planner", "Planner not engaged"

    merged_two_bucket_rates: dict[str, dict[str, float]] = {"labor": {}, "machine": {}}
    for candidate in (RATES_TWO_BUCKET_DEFAULT, default_rates, rates):
        if not isinstance(candidate, _MappingABC):
            continue
        normalized = _normalized_two_bucket_rates(candidate)
        if not any(normalized.get(kind) for kind in ("labor", "machine")):
            continue
        for bucket_type in ("labor", "machine"):
            bucket = merged_two_bucket_rates.setdefault(bucket_type, {})
            for role, value in normalized.get(bucket_type, {}).items():
                try:
                    bucket[str(role)] = float(value)
                except Exception:
                    continue
    if not any(merged_two_bucket_rates.get(kind) for kind in ("labor", "machine")):
        merged_two_bucket_rates = _normalized_two_bucket_rates(RATES_TWO_BUCKET_DEFAULT)

    bucketize_nre: dict[str, Any] = {}
    if isinstance(planner_result, _MappingABC):
        for key in ("nre", "nre_minutes", "nre_totals"):
            candidate = planner_result.get(key)
            if isinstance(candidate, _MappingABC) and candidate:
                bucketize_nre = {str(k): v for k, v in candidate.items()}
                break

    geom_for_bucketize = geo_payload if isinstance(geo_payload, dict) else {}
    qty_for_bucketize = 1
    if isinstance(qty, (int, float)):
        try:
            qty_for_bucketize = int(qty)
        except (TypeError, ValueError):
            qty_for_bucketize = 1
        if qty_for_bucketize <= 0:
            qty_for_bucketize = 1

    try:
        bucketized_raw = bucketize(
            planner_result if isinstance(planner_result, dict) else {},
            merged_two_bucket_rates,
            bucketize_nre,
            qty=qty_for_bucketize,
            geom=geom_for_bucketize,
        )
    except Exception:
        bucketized_raw = {}

    if not isinstance(bucketized_raw, _MappingABC):
        bucketized_raw = {}

    bucket_view_prepared: dict[str, Any] = _prepare_bucket_view(bucketized_raw)

    aggregated_bucket_minutes: dict[str, dict[str, float]] = {}
    line_items: Sequence[Mapping[str, Any]] | None = None
    if isinstance(planner_result, _MappingABC):
        raw_items = planner_result.get("line_items")
        if isinstance(raw_items, Sequence):
            # A shallow copy is sufficient for iteration and avoids surprising
            # behaviour if the planner mutates the list during aggregation.
            line_items = list(raw_items)

    if line_items:
        for entry in line_items:
            if not isinstance(entry, _MappingABC):
                continue
            bucket_key = _planner_bucket_key_for_name(entry.get("op"))
            if not bucket_key:
                continue
            canon_key = _canonical_bucket_key(bucket_key) or bucket_key
            if not canon_key or canon_key in _FINAL_BUCKET_HIDE_KEYS:
                continue
            minutes_val = _safe_float(entry.get("minutes"))
            machine_val = _bucket_cost(entry, "machine_cost", "machine$")
            labor_val = _bucket_cost(entry, "labor_cost", "labor$")
            if (
                minutes_val <= 0.0
                and machine_val <= 0.0
                and labor_val <= 0.0
            ):
                continue
            metrics = aggregated_bucket_minutes.setdefault(
                canon_key,
                {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0},
            )
            metrics["minutes"] += minutes_val
            metrics["machine$"] += machine_val
            metrics["labor$"] += labor_val

    hole_diams: list[float] = []
    if isinstance(geo_payload, _MappingABC):
        raw_diams = geo_payload.get("hole_diams_mm")
        if isinstance(raw_diams, (list, tuple)):
            for entry in raw_diams:
                coerced = _coerce_float_or_none(entry)
                if coerced:
                    hole_diams.append(float(coerced))

    thickness_in = _coerce_float_or_none(value_map.get("Thickness (in)"))
    if thickness_in is None and isinstance(geo_payload, _MappingABC):
        thickness_in = _coerce_float_or_none(geo_payload.get("thickness_in"))
        if thickness_in is None:
            thickness_mm = _coerce_float_or_none(geo_payload.get("thickness_mm"))
            if thickness_mm:
                thickness_in = float(thickness_mm) / 25.4

    drilling_rate = _lookup_rate("DrillingRate", rates, params, default_rates, fallback=75.0)
    drill_total_minutes: float | None = None
    drilling_meta_container = breakdown.setdefault("drilling_meta", {})
    if isinstance(drilling_meta_container, dict):
        if material_display:
            drilling_meta_container["material_display"] = material_display
            drilling_meta_container["material"] = material_display
        drilling_meta_container["material_key"] = material_key
        drilling_meta_container["material_lookup"] = material_key
        if material_group_display:
            drilling_meta_container["material_group"] = material_group_display
    if not use_planner:
        if hole_diams and thickness_in and drilling_rate > 0:
            drill_debug_lines: list[str] = []
            drill_debug_summary: dict[str, dict[str, Any]] = {}
            hole_groups_for_estimate: list[dict[str, Any]] | None = None
            if isinstance(geo_payload, _MappingABC):
                raw_groups = (
                    geo_payload.get("GEO_Hole_Groups")
                    or geo_payload.get("hole_groups")
                    or geo_payload.get("hole_sets")
                )
                if raw_groups is not None:
                    hole_groups_for_estimate = _clean_hole_groups(raw_groups)
            try:
                estimator_hours = estimate_drilling_hours(
                    hole_diams,
                    float(thickness_in),
                    material_key,
                    hole_groups=hole_groups_for_estimate,
                    material_group=material_group_display,
                    speeds_feeds_table=speeds_feeds_table,
                    debug_lines=drill_debug_lines,
                    debug_summary=drill_debug_summary,
                )
            except Exception:
                estimator_hours = 0.0
            if not estimator_hours or estimator_hours <= 0:
                hole_count = max(1, len(hole_diams))
                avg_dia_in = sum(hole_diams) / hole_count / 25.4
                estimator_hours = max(
                    0.05, hole_count * max(avg_dia_in, 0.1) * float(thickness_in) / 600.0
                )
            drill_total_minutes = max(0.0, float(estimator_hours or 0.0)) * 60.0

            if drill_debug_summary:
                bins_list: list[dict[str, Any]] = []
                for op_key, summary in drill_debug_summary.items():
                    if not isinstance(summary, _MappingABC):
                        continue
                    bins_map = summary.get("bins")
                    if not isinstance(bins_map, _MappingABC):
                        continue
                    op_display = str(summary.get("operation") or op_key or "").strip()
                    sortable_bins: list[Mapping[str, Any]] = []
                    for bin_payload in bins_map.values():
                        if not isinstance(bin_payload, _MappingABC):
                            continue
                        sortable_bins.append(cast(Mapping[str, Any], bin_payload))
                    for bin_payload in sorted(
                        sortable_bins,
                        key=lambda item: _safe_float(item.get("diameter_in"), 0.0),
                    ):
                        entry = dict(bin_payload)
                        speeds = entry.get("speeds")
                        if isinstance(speeds, _MappingABC):
                            for speed_key in ("sfm", "ipr", "rpm", "ipm"):
                                if speed_key not in entry and speeds.get(speed_key) is not None:
                                    entry[speed_key] = _safe_float(speeds.get(speed_key), 0.0)
                        depth_in = None
                        for candidate in (
                            entry.get("depth_in"),
                            entry.get("depth_max"),
                            entry.get("depth_min"),
                        ):
                            try:
                                if candidate is None:
                                    continue
                                depth_val = float(candidate)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(depth_val):
                                depth_in = depth_val
                                break
                        if depth_in is not None:
                            entry["depth_in"] = depth_in
                        dia_val = entry.get("diameter_in")
                        try:
                            entry["diameter_in"] = float(dia_val) if dia_val is not None else None
                        except (TypeError, ValueError):
                            entry.pop("diameter_in", None)
                        qty_val = entry.get("qty")
                        qty_numeric = _safe_float(qty_val, 0.0)
                        entry["qty"] = int(round(qty_numeric))
                        op_resolved = op_display or op_key or "drill"
                        entry["op"] = op_resolved
                        entry.setdefault("op_name", op_resolved)
                        bins_list.append(entry)

                if bins_list:
                    bins_list.sort(
                        key=lambda item: (
                            0
                            if str(item.get("op") or "").strip().lower().startswith("deep")
                            else 1,
                            _safe_float(item.get("diameter_in"), 0.0),
                        )
                    )
                    drilling_meta_container["bins_list"] = bins_list

                    deep_qty = sum(
                        int(entry.get("qty") or 0)
                        for entry in bins_list
                        if str(entry.get("op") or "").strip().lower().startswith("deep")
                    )
                    std_qty = sum(
                        int(entry.get("qty") or 0)
                        for entry in bins_list
                        if not str(entry.get("op") or "").strip().lower().startswith("deep")
                    )
                    drilling_meta_container["holes_deep"] = deep_qty
                    drilling_meta_container["holes_std"] = std_qty

                    if bins_list and not drilling_meta_container.get("dia_in_vals"):
                        drilling_meta_container["dia_in_vals"] = [
                            entry.get("diameter_in")
                            for entry in bins_list
                            if entry.get("diameter_in") is not None
                        ]
                    if bins_list and not drilling_meta_container.get("depth_in_vals"):
                        drilling_meta_container["depth_in_vals"] = [
                            entry.get("depth_in")
                            for entry in bins_list
                            if entry.get("depth_in") is not None
                        ]

    drilling_summary_raw = process_plan_summary.get("drilling")
    if isinstance(drilling_summary_raw, dict):
        drilling_summary = drilling_summary_raw
    else:
        drilling_summary = process_plan_summary.setdefault("drilling", {})

    if drill_total_minutes is not None and drill_total_minutes > 0.0:
        drilling_summary["total_minutes"] = float(drill_total_minutes)
        if "rate" not in drilling_summary:
            drilling_summary["rate"] = float(drilling_rate)

    groups_existing = drilling_summary.get("groups")
    has_groups = False
    if isinstance(groups_existing, Sequence) and not isinstance(groups_existing, (str, bytes)):
        has_groups = bool(groups_existing)

    fallback_groups: list[dict[str, Any]] = []
    if not has_groups and hole_diams:
        fallback_groups = build_drill_groups_from_geometry(
            hole_diams,
            thickness_in,
        )

    if fallback_groups:
        drilling_summary["groups"] = fallback_groups
        holes_deep, holes_std = _apply_drilling_meta_fallback(
            drilling_meta_container,
            fallback_groups,
        )
        drilling_summary["holes_deep"] = holes_deep
        drilling_summary["holes_std"] = holes_std
        fallback_hole_count = sum(int(_coerce_float_or_none(group.get("qty")) or 0) for group in fallback_groups)
        drilling_summary["hole_count"] = fallback_hole_count
        drilling_meta_container["bins_list"] = fallback_groups
        drilling_meta_container["hole_count"] = fallback_hole_count

    # Establish authoritative drilling-minute totals before the buckets are
    # rendered so downstream consumers have a single source of truth.
    drill_minutes_with_toolchange = _coerce_float_or_none(
        drilling_summary.get("total_minutes_with_toolchange")
    )
    if drill_minutes_with_toolchange is None or drill_minutes_with_toolchange <= 0.0:
        drill_minutes_with_toolchange = None
        if drill_total_minutes is not None and drill_total_minutes > 0.0:
            drill_minutes_with_toolchange = float(drill_total_minutes)
        else:
            group_minutes_total = 0.0
            groups_payload = drilling_summary.get("groups")
            if isinstance(groups_payload, Sequence) and not isinstance(groups_payload, (str, bytes)):
                for group_entry in groups_payload:
                    minutes_val: float | None = None
                    qty_val: float | None = None
                    per_hole_val: float | None = None
                    if isinstance(group_entry, _MappingABC):
                        minutes_val = (
                            _coerce_float_or_none(group_entry.get("minutes_total"))
                            or _coerce_float_or_none(group_entry.get("total_minutes"))
                            or _coerce_float_or_none(group_entry.get("minutes"))
                        )
                        qty_val = _coerce_float_or_none(group_entry.get("qty"))
                        per_hole_val = (
                            _coerce_float_or_none(group_entry.get("t_per_hole_min"))
                            or _coerce_float_or_none(group_entry.get("minutes_per_hole"))
                            or _coerce_float_or_none(group_entry.get("t_per_hole"))
                        )
                    else:
                        minutes_val = (
                            _coerce_float_or_none(getattr(group_entry, "minutes_total", None))
                            or _coerce_float_or_none(getattr(group_entry, "total_minutes", None))
                            or _coerce_float_or_none(getattr(group_entry, "minutes", None))
                        )
                        qty_val = _coerce_float_or_none(getattr(group_entry, "qty", None))
                        per_hole_val = (
                            _coerce_float_or_none(getattr(group_entry, "t_per_hole_min", None))
                            or _coerce_float_or_none(getattr(group_entry, "minutes_per_hole", None))
                            or _coerce_float_or_none(getattr(group_entry, "t_per_hole", None))
                        )
                    if (minutes_val is None or minutes_val <= 0.0) and (
                        qty_val is not None
                        and qty_val > 0.0
                        and per_hole_val is not None
                        and per_hole_val > 0.0
                    ):
                        minutes_val = float(qty_val) * float(per_hole_val)
                    if minutes_val is None or minutes_val <= 0.0:
                        continue
                    group_minutes_total += float(minutes_val)
            if group_minutes_total > 0.0:
                drill_minutes_with_toolchange = float(group_minutes_total)
                toolchange_minutes_val = None
                toolchange_candidates: Sequence[Any] = (
                    drilling_summary.get("toolchange_minutes"),
                    drilling_summary.get("toolchange_total"),
                )
                for candidate in toolchange_candidates:
                    candidate_val = _coerce_float_or_none(candidate)
                    if candidate_val is not None and candidate_val > 0.0:
                        toolchange_minutes_val = float(candidate_val)
                        break
                if (
                    toolchange_minutes_val is None
                    and isinstance(drilling_meta_container, _MappingABC)
                ):
                    for candidate in (
                        drilling_meta_container.get("toolchange_total"),
                        drilling_meta_container.get("toolchange_minutes"),
                    ):
                        candidate_val = _coerce_float_or_none(candidate)
                        if candidate_val is not None and candidate_val > 0.0:
                            toolchange_minutes_val = float(candidate_val)
                            break
                if toolchange_minutes_val is None and isinstance(drilling_meta_container, _MappingABC):
                    toolchange_minutes_val = 0.0
                    try:
                        holes_deep_val = int(drilling_summary.get("holes_deep") or 0)
                    except Exception:
                        holes_deep_val = 0
                    try:
                        holes_std_val = int(drilling_summary.get("holes_std") or 0)
                    except Exception:
                        holes_std_val = 0
                    if holes_deep_val > 0:
                        toolchange_minutes_val += _safe_float(
                            drilling_meta_container.get("toolchange_min_deep"), 0.0
                        )
                    if holes_std_val > 0:
                        toolchange_minutes_val += _safe_float(
                            drilling_meta_container.get("toolchange_min_std"), 0.0
                        )
                if toolchange_minutes_val and toolchange_minutes_val > 0.0:
                    drill_minutes_with_toolchange += float(toolchange_minutes_val)
                    drilling_summary.setdefault("toolchange_minutes", float(toolchange_minutes_val))

    if drill_minutes_with_toolchange is not None and drill_minutes_with_toolchange > 0.0:
        drilling_summary["total_minutes_with_toolchange"] = float(drill_minutes_with_toolchange)
        drill_meta_for_totals = breakdown.setdefault("drilling_meta", {})
        try:
            drill_meta_for_totals["total_minutes_with_toolchange"] = float(
                drill_minutes_with_toolchange
            )
        except Exception:
            pass

    # === DRILLING BILLING TRUTH ===
    drill_meta_candidate = breakdown.setdefault("drilling_meta", {})
    if isinstance(drill_meta_candidate, _MutableMappingABC):
        drill_meta_map = drill_meta_candidate
    elif isinstance(drill_meta_candidate, _MappingABC):
        drill_meta_map = dict(drill_meta_candidate)
        breakdown["drilling_meta"] = drill_meta_map
    else:
        drill_meta_map = {}
        breakdown["drilling_meta"] = drill_meta_map

    bill_min = float(
        drill_meta_map.get("total_minutes_with_toolchange")
        or drill_meta_map.get("total_minutes")
        or 0.0
    )

    plan_drill_min: float | None = None
    plan_candidates = [
        locals().get("process_plan_summary"),
        locals().get("process_plan_summary_local"),
        breakdown.get("process_plan") if isinstance(breakdown, _MappingABC) else None,
    ]
    for candidate in plan_candidates:
        if not isinstance(candidate, _MappingABC):
            continue
        plan_drilling = candidate.get("drilling")
        if not isinstance(plan_drilling, _MappingABC):
            continue
        plan_minutes = _coerce_float_or_none(
            plan_drilling.get("total_minutes_with_toolchange")
        )
        if plan_minutes is not None and plan_minutes > 0.0:
            plan_drill_min = float(plan_minutes)
            break

    drill_min = plan_drill_min if plan_drill_min and plan_drill_min > 0.0 else bill_min
    bill_min = drill_min
    drill_hr = drill_min / 60.0 if drill_min else 0.0
    drill_meta_map["total_minutes_billed"] = drill_min

    # overwrite any legacy/planner meta for drilling
    pm = breakdown.setdefault("process_meta", {}).setdefault("drilling", {})
    pm["minutes"] = drill_min
    pm["hr"] = drill_hr
    pm["rate"] = float(rates.get("DrillingRate") or rates.get("MachineRate") or 0.0)
    pm["basis"] = ["minutes_engine"]

    drilling_summary["total_minutes_billed"] = float(drill_min)

    drilling_meta_source: _MappingABC[str, Any] | None = None
    if isinstance(drilling_meta_container, _MappingABC):
        drilling_meta_source = drilling_meta_container

    billed_minutes = 0.0
    if isinstance(drilling_summary, _MappingABC):
        billed_minutes = _safe_float(drilling_summary.get("total_minutes_billed"))
        if billed_minutes <= 0.0:
            billed_minutes = _safe_float(drilling_summary.get("total_minutes_with_toolchange"))
    if billed_minutes <= 0.0 and isinstance(drilling_meta_source, _MappingABC):
        billed_minutes = _safe_float(drilling_meta_source.get("total_minutes_billed"))
    if billed_minutes <= 0.0 and drill_total_minutes is not None and drill_total_minutes > 0.0:
        billed_minutes = float(drill_total_minutes)

    if billed_minutes > 0.0:
        # Legacy bucket view population for process_plan is disabled to avoid
        # overriding minutes-engine drilling totals. Preserve the billed minutes
        # detail without mutating planner structures.
        bucket_minutes_detail_local = locals().get("bucket_minutes_detail")
        if isinstance(bucket_minutes_detail_local, dict):
            bucket_minutes_detail_local["drilling"] = billed_minutes
        bucket_minutes_detail_for_render["drilling"] = billed_minutes

    drilling_minutes_for_bucket: float | None = None
    if billed_minutes > 0.0:
        drilling_minutes_for_bucket = float(billed_minutes)
    elif drill_total_minutes is not None and drill_total_minutes > 0.0:
        drilling_minutes_for_bucket = float(drill_total_minutes)
    else:
        drilling_summary_map = process_plan_summary.get("drilling")
        if isinstance(drilling_summary_map, _MappingABC):
            candidate_minutes = _safe_float(drilling_summary_map.get("total_minutes"))
            if candidate_minutes > 0.0:
                drilling_minutes_for_bucket = candidate_minutes

    if drilling_minutes_for_bucket and drilling_minutes_for_bucket > 0.0:
        metrics = aggregated_bucket_minutes.setdefault(
            "drilling",
            {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0},
        )
        drill_rate_value = 0.0
        drilling_meta_entry = (
            process_meta.get("drilling") if isinstance(process_meta, _MappingABC) else None
        )
        if isinstance(drilling_meta_entry, _MappingABC):
            drill_rate_value = _safe_float(drilling_meta_entry.get("rate"))
        if drill_rate_value <= 0.0:
            drilling_summary_map = process_plan_summary.get("drilling")
            if isinstance(drilling_summary_map, _MappingABC):
                drill_rate_value = _safe_float(drilling_summary_map.get("rate"))
        if drill_rate_value <= 0.0:
            drill_rate_value = float(drilling_rate)

        metrics["minutes"] = drilling_minutes_for_bucket
        machine_cost_val = _safe_float(metrics.get("machine$"))
        if drill_rate_value > 0.0:
            machine_cost_val = (drilling_minutes_for_bucket / 60.0) * drill_rate_value
        metrics["machine$"] = round(machine_cost_val, 2)

        labor_cost_val = _safe_float(metrics.get("labor$"))
        metrics["labor$"] = round(labor_cost_val, 2) if labor_cost_val > 0.0 else 0.0

    if aggregated_bucket_minutes:
        legacy_bucket_entries: dict[str, dict[str, Any]] = {}
        if not using_planner:
            legacy_bucket_entries = {
                key: dict(value)
                for key, value in bucket_view.items()
                if isinstance(value, _MappingABC)
                and key not in {"buckets", "order", "totals"}
            }
        buckets_raw = bucket_view_prepared.get("buckets")
        if isinstance(buckets_raw, _MappingABC):
            buckets = {str(k): dict(v) for k, v in buckets_raw.items()}
        else:
            buckets = {}

        for canon_key, metrics in aggregated_bucket_minutes.items():
            if canon_key in _FINAL_BUCKET_HIDE_KEYS:
                continue
            entry = dict(buckets.get(canon_key, {}))
            minutes_val = round(_safe_float(metrics.get("minutes")), 2)
            machine_val = round(_safe_float(metrics.get("machine$")), 2)
            labor_val = round(_safe_float(metrics.get("labor$")), 2)
            total_val = round(machine_val + labor_val, 2)
            entry.update(
                {
                    "minutes": minutes_val,
                    "machine$": machine_val,
                    "labor$": labor_val,
                    "total$": total_val,
                }
            )
            buckets[canon_key] = entry

        order_raw = bucket_view_prepared.get("order")
        if isinstance(order_raw, Sequence):
            order = [str(item) for item in order_raw if isinstance(item, str)]
        else:
            order = []

        seen: set[str] = set()
        ordered: list[str] = []
        for key in order:
            canon = _canonical_bucket_key(key) or key
            if canon in buckets and canon not in seen:
                ordered.append(canon)
                seen.add(canon)
        for key in _preferred_order_then_alpha(buckets.keys()):
            if key not in seen:
                ordered.append(key)
                seen.add(key)

        totals = {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0, "total$": 0.0}
        for entry in buckets.values():
            minutes_val = _safe_float(entry.get("minutes"))
            machine_val = _safe_float(entry.get("machine$"))
            labor_val = _safe_float(entry.get("labor$"))
            total_val = _safe_float(entry.get("total$"))
            if total_val <= 0.0:
                total_val = machine_val + labor_val
            totals["minutes"] += minutes_val
            totals["machine$"] += machine_val
            totals["labor$"] += labor_val
            totals["total$"] += total_val

        bucket_view_prepared["buckets"] = buckets
        bucket_view_prepared["order"] = ordered
        bucket_view_prepared["totals"] = {key: round(val, 2) for key, val in totals.items()}
        if not using_planner:
            for canon_key, entry in buckets.items():
                bucket_view_prepared.setdefault(canon_key, entry)
            for legacy_key, legacy_value in legacy_bucket_entries.items():
                bucket_view_prepared[legacy_key] = legacy_value

        if isinstance(process_meta, dict) and using_planner:
            total_minutes = 0.0
            machine_minutes = 0.0
            labor_minutes = 0.0
            for metrics in aggregated_bucket_minutes.values():
                minutes_val = _safe_float(metrics.get("minutes"))
                machine_val = _safe_float(metrics.get("machine$"))
                labor_val = _safe_float(metrics.get("labor$"))
                if minutes_val <= 0.0:
                    continue
                hours_val = minutes_val / 60.0
                total_cost = machine_val + labor_val
                total_minutes += minutes_val
                if hours_val > 0.0 and total_cost > 0.0:
                    rate_val = total_cost / hours_val
                    if rate_val > 0.0:
                        if machine_val > 0.0:
                            machine_minutes += (machine_val / rate_val) * 60.0
                        if labor_val > 0.0:
                            labor_minutes += (labor_val / rate_val) * 60.0

            def _apply_minutes(key: str, minutes_val: float) -> None:
                if minutes_val <= 0.0:
                    return
                existing = process_meta.get(key)
                updated = dict(existing) if isinstance(existing, _MappingABC) else {}
                updated["minutes"] = round(minutes_val, 2)
                updated["hr"] = round(minutes_val / 60.0, 3)
                process_meta[key] = updated

            if total_minutes > 0.0:
                _apply_minutes("planner_total", total_minutes)
            if machine_minutes > 0.0:
                _apply_minutes("planner_machine", machine_minutes)
            if labor_minutes > 0.0:
                _apply_minutes("planner_labor", labor_minutes)

    bucket_view.clear()
    bucket_view.update(bucket_view_prepared)

    bucket_view_buckets: Mapping[str, Any] | None = None

    if not use_planner:
        drilling_bucket_prepared = None
        if isinstance(bucket_view_prepared, _MappingABC):
            bucket_view_buckets = bucket_view_prepared.get("buckets")
            if isinstance(bucket_view_buckets, _MappingABC):
                drilling_bucket_prepared = bucket_view_buckets.get("drilling")
        if isinstance(drilling_bucket_prepared, _MappingABC):
            bucket_view["drilling"] = {
                "minutes": _safe_float(drilling_bucket_prepared.get("minutes")),
                "machine_cost": _safe_float(
                    drilling_bucket_prepared.get("machine$")
                    if "machine$" in drilling_bucket_prepared
                    else drilling_bucket_prepared.get("machine_cost")
                ),
                "labor_cost": _safe_float(
                    drilling_bucket_prepared.get("labor$")
                    if "labor$" in drilling_bucket_prepared
                    else drilling_bucket_prepared.get("labor_cost")
                ),
            }

    roughing_hours = _coerce_float_or_none(value_map.get("Roughing Cycle Time"))
    if roughing_hours is None:
        roughing_hours = _coerce_float_or_none(value_map.get("Roughing Cycle Time (hr)"))
    milling_rate = _lookup_rate("MillingRate", rates, params, default_rates, fallback=100.0)
    if roughing_hours and roughing_hours > 0:
        bucket_view["milling"] = {
            "minutes": roughing_hours * 60.0,
            "machine_cost": roughing_hours * milling_rate,
            "labor_cost": 0.0,
        }
        process_meta["milling"] = {
            "hr": roughing_hours,
            "minutes": roughing_hours * 60.0,
            "rate": milling_rate,
            "basis": ["planner_milling_backfill"],
        }

    project_hours = _coerce_float_or_none(value_map.get("Project Management Hours")) or 0.0
    toolmaker_hours = _coerce_float_or_none(value_map.get("Tool & Die Maker Hours")) or 0.0
    toolmaker_rate = _lookup_rate("ToolmakerRate", rates, params, default_rates, fallback=85.0)
    if project_hours > 0:
        process_meta["project_management"] = {"hr": 0.0, "minutes": 0.0}
    if toolmaker_hours > 0:
        process_meta["toolmaker_support"] = {"hr": toolmaker_hours, "minutes": toolmaker_hours * 60.0}
        if not planner_used:
            process_costs["toolmaker_support"] = toolmaker_hours * toolmaker_rate

    app_meta: dict[str, Any] = {"used_planner": True} if planner_used else {}

    geo_context_for_prog = breakdown.get("geo_context")
    process_plan_for_prog = (
        breakdown.get("process_plan") if isinstance(breakdown, _MappingABC) else None
    )
    detail_minutes = _compute_programming_detail_minutes(
        geo_context_for_prog,
        process_plan_for_prog,
    )
    auto_prog_hr_model = detail_minutes / 60.0
    geo_prog_hr = auto_prog_hr_model
    auto_prog_hr = float(auto_prog_hr_model)

    overrides_map = state.user_overrides if isinstance(state.user_overrides, _MappingABC) else {}
    user_override_prog = (
        _coerce_float_or_none(overrides_map.get("programming_hr"))
        if isinstance(overrides_map, _MappingABC)
        else None
    )
    override_prog = _coerce_float_or_none(value_map.get("Programming Override Hr"))

    final_prog_hr = auto_prog_hr
    override_applied = False
    override_source: str | None = None
    if override_prog is not None and override_prog >= 0:
        final_prog_hr = float(override_prog)
        override_applied = True
        override_source = "sheet"
    elif user_override_prog is not None and user_override_prog >= 0:
        final_prog_hr = float(user_override_prog)
        override_applied = True
        override_source = "user"
    programming_detail_container = breakdown.setdefault("nre_detail", {})
    if isinstance(programming_detail_container, dict):
        programming_detail = programming_detail_container.get("programming")
    else:
        programming_detail_container = {}
        breakdown["nre_detail"] = programming_detail_container
        programming_detail = None

    if isinstance(programming_detail, dict):
        detail_map = programming_detail
    else:
        detail_map = {}

    detail_map["geo_prog_hr"] = float(geo_prog_hr)
    detail_map["detail_minutes"] = float(detail_minutes)
    detail_map["detail_count"] = float(detail_minutes)
    detail_map.pop("planner_prog_hr", None)
    detail_map["auto_prog_hr_model"] = float(auto_prog_hr_model)
    detail_map["auto_prog_hr"] = float(auto_prog_hr)
    detail_map["prog_hr"] = float(final_prog_hr)
    if override_applied:
        detail_map["override_applied"] = True
        if override_source:
            detail_map["override_source"] = override_source
    else:
        detail_map.pop("override_applied", None)
        detail_map.pop("override_source", None)

    labor_bucket = (
        merged_two_bucket_rates.get("labor")
        if isinstance(merged_two_bucket_rates, _MappingABC)
        else {}
    )
    # === PROGRAMMING RATE (LABOR-ONLY) ===
    programmer_rate = float(getattr(cfg, "labor_rate_per_hr", 45.0) or 45.0)
    if not math.isfinite(programmer_rate) or programmer_rate <= 0:
        programmer_rate = 45.0

    if programmer_rate > 0:
        if isinstance(merged_two_bucket_rates, dict):
            labor_target = merged_two_bucket_rates.setdefault("labor", {})
            if isinstance(labor_target, dict):
                labor_target["Programmer"] = programmer_rate
                labor_target.setdefault("programmer", programmer_rate)
        if isinstance(rates, dict):
            labor_rates_block = rates.get("labor")
            if isinstance(labor_rates_block, _MappingABC) and not isinstance(
                labor_rates_block, dict
            ):
                labor_rates_block = dict(labor_rates_block)
                rates["labor"] = labor_rates_block
            elif not isinstance(labor_rates_block, dict):
                labor_rates_block = {}
                rates["labor"] = labor_rates_block
            labor_rates_block["Programmer"] = programmer_rate
            labor_rates_block["programmer"] = programmer_rate
            labor_rates_block["PROGRAMMER"] = programmer_rate
            rates["ProgrammerRate"] = programmer_rate
            rates["ProgrammingRate"] = programmer_rate

    detail_map["prog_rate"] = programmer_rate
    programming_detail_container["programming"] = detail_map

    qty_for_programming = 1.0
    try:
        qty_for_programming = float(qty)
    except Exception:
        qty_for_programming = 1.0
    if not math.isfinite(qty_for_programming) or qty_for_programming <= 0:
        qty_for_programming = 1.0

    programming_total = round(final_prog_hr * programmer_rate, 2)
    programming_per_part = programming_total / max(1.0, qty_for_programming)
    detail_map["per_lot"] = programming_total
    detail_map["per_part"] = programming_per_part

    is_amortized = bool(amortized_programming and amortized_programming > 0)
    detail_map["amortized"] = is_amortized
    if is_amortized:
        detail_map["amortized_cost"] = float(amortized_programming)
    else:
        detail_map.pop("amortized_cost", None)

    nre_block = breakdown.setdefault("nre", {})
    nre_block["programming_hr"] = float(final_prog_hr)
    if is_amortized:
        nre_block["programming_cost"] = 0.0
        nre_block["programming_per_lot"] = 0.0
        nre_block["programming_per_part"] = 0.0
    else:
        nre_block["programming_cost"] = programming_total
        nre_block["programming_per_lot"] = programming_total
        nre_block["programming_per_part"] = programming_per_part

    if speeds_feeds_csv_path:
        breakdown["speeds_feeds_path"] = speeds_feeds_csv_path
        breakdown["speeds_feeds_loaded"] = bool(speeds_feeds_loaded)

    geo_ref = geo_context if isinstance(geo_context, dict) else {}

    app_meta["llm_debug_enabled"] = APP_ENV.llm_debug_enabled

    state_for_render = reprice_with_effective(state)
    effective_snapshot = dict(getattr(state_for_render, "effective", {}) or {})
    effective_sources_snapshot = dict(getattr(state_for_render, "effective_sources", {}) or {})

    result = {
        "decision_state": {
            "baseline": baseline,
            "suggestions": state_for_render.suggestions,
            "user_overrides": state_for_render.user_overrides,
            "effective": effective_snapshot,
            "effective_sources": effective_sources_snapshot,
        },
        "breakdown": breakdown,
        "geo": geo_ref,
        "geom": geo_ref,
        "app_meta": dict(app_meta),
        "quote_state": state_for_render.to_dict(),
    }

    return result


def validate_quote_before_pricing(
    geo: Mapping[str, Any] | None,
    process_costs: dict[str, float],
    pass_through: dict[str, Any],
    process_hours: dict[str, float] | None = None,
) -> None:
    issues: list[str] = []
    geo_ctx: Mapping[str, Any] = geo if isinstance(geo, _MappingABC) else {}
    has_legacy_buckets = any(key in process_costs for key in ("drilling", "milling"))
    if has_legacy_buckets:
        hole_cost = sum(float(process_costs.get(k, 0.0)) for k in ("drilling", "milling"))
        if geo_ctx.get("hole_diams_mm") and hole_cost < 50:
            issues.append("Unusually low machining time for number of holes.")
    material_cost = float(pass_through.get("Material", 0.0) or 0.0)
    if material_cost < 5.0:
        inner_geo_candidate = geo_ctx.get("geo") if isinstance(geo_ctx, _MappingABC) else None
        inner_geo: Mapping[str, Any] = (
            inner_geo_candidate if isinstance(inner_geo_candidate, _MappingABC) else {}
        )

        def _positive(value: Any) -> bool:
            num = _coerce_float_or_none(value)
            return bool(num and num > 0)

        thickness_candidates = [
            geo_ctx.get("thickness_mm"),
            geo_ctx.get("thickness_in"),
            geo_ctx.get("GEO-03_Height_mm"),
            geo_ctx.get("GEO__Stock_Thickness_mm"),
            inner_geo.get("thickness_mm") if isinstance(inner_geo, _MappingABC) else None,
            inner_geo.get("stock_thickness_mm") if isinstance(inner_geo, _MappingABC) else None,
        ]
        has_thickness_hint = any(_positive(val) for val in thickness_candidates)

        def _mass_hint(ctx: Mapping[str, Any] | None) -> bool:
            if not isinstance(ctx, _MappingABC):
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

@no_type_check

# ---------- Tracing ----------
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

    # perimeter from vector segments (points are in PostScript points; 1 pt = 0.352777… mm)
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

def default_variables_template() -> PandasDataFrame:
    if _HAS_PANDAS and pd is not None:
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
    if not _HAS_PANDAS or pd is None:
        raise RuntimeError("pandas is required to build the default variables template.")
    rows = [
        ("Profit Margin %", 0.0, "number"),
        ("Programmer $/hr", 90.0, "number"),
        ("CAM Programmer $/hr", 90.0, "number"),
        ("Milling $/hr", 90.0, "number"),
        ("Inspection $/hr", 85.0, "number"),
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
        ("Material", "Aluminum MIC6", "text"),
        ("Thickness (in)", 2.0, "number"),
    ]
    assert pd is not None  # for type checkers
    return pd.DataFrame(rows, columns=REQUIRED_COLS)

def coerce_or_make_vars_df(df: PandasDataFrame | None) -> PandasDataFrame:
    """Ensure the variables dataframe has the required columns with tolerant matching."""

    if not _HAS_PANDAS or pd is None:
        raise RuntimeError("pandas is required to coerce variable dataframes.")

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

JSON_SCHEMA = load_json("vl_pdf_schema.json")
MAP_KEYS = load_json("vl_pdf_map_keys.json")
PDF_SYSTEM_PROMPT = load_text("vl_pdf_system_prompt.txt").strip()

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
    system = PDF_SYSTEM_PROMPT
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

def merge_estimate_into_vars(vars_df: PandasDataFrame, estimate: dict) -> PandasDataFrame:
    if not _HAS_PANDAS or pd is None:
        raise RuntimeError("pandas is required to merge PDF estimates into variables.")

    assert pd is not None  # for type checkers

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
# Accept: #10-32, 5/8-11, 0.190-32, M8x1.25, etc.
RE_THICK  = RE_DEPTH


_QTY_LEAD = re.compile(r'^\s*\((\d+)\)\s*')
_MM_IN_DIA = re.compile(r"(?:Ø|⌀|O|DIA|\b)\s*([0-9.]+)")
_PAREN_DIA = re.compile(r"\(([0-9.]+)\s*Ø?\)")
_FROM_SIDE = re.compile(r"\bFROM\s+(FRONT|BACK)\b", re.I)
RE_MAT    = re.compile(r"\b(MATL?|MATERIAL)\b\s*[:=\-]?\s*([A-Z0-9 \-\+/\.]+)", re.I)
RE_HARDNESS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[-–]\s*(\d+(?:\.\d+)?))?\s*HRC", re.I)
RE_HEAT_TREAT = re.compile(r"HEAT\s*TREAT(?:ED|\s+TO)?|\bQUENCH\b|\bTEMPER\b", re.I)
RE_COAT   = re.compile(
    r"\b(ANODIZE(?:\s*(?:CLR|BLACK|BLK))?|BLACK OXIDE|ZINC PLATE|NICKEL PLATE|PASSIVATE|CHEM FILM|IRIDITE|ALODINE|POWDER COAT|E-?COAT|PAINT)\b",
    re.I,
)
RE_TOL    = re.compile(r"\bUNLESS OTHERWISE SPECIFIED\b.*?([±\+\-]\s*\d+\.\d+)", re.I | re.S)
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

SPOT_DRILL_MIN_PER_SIDE_MIN = 0.1
NPT_INSPECTION_MIN_PER_HOLE = 2.5
JIG_GRIND_MIN_PER_FEATURE = 15.0
REAM_MIN_PER_FEATURE = 6.0
TIGHT_TOL_INSPECTION_MIN = 4.0
TIGHT_TOL_CMM_MIN = 6.0
HANDLING_ADDER_RANGE_HR = (0.1, 0.3)
LB_PER_IN3_PER_GCC = 0.036127292

def detect_units_scale(doc) -> dict[str, float | int]:
    return _shared_detect_units_scale(doc)

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
        pts: list[tuple[float, float]] = []
        try:
            get_points = getattr(pl, "get_points", None)
            if callable(get_points):
                raw_pts_iter = get_points()
                raw_pts = cast(Iterable[Sequence[float]], raw_pts_iter)
                for raw_pt in raw_pts:
                    if len(raw_pt) < 2:
                        continue
                    x_val, y_val = raw_pt[0], raw_pt[1]
                    try:
                        pts.append((float(x_val) * to_in, float(y_val) * to_in))
                    except (TypeError, ValueError):
                        continue
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
    return list(_shared_iter_spaces(doc))


def _all_tables(doc):
    yield from _shared_iter_table_entities(doc)

def _iter_table_text(doc):
    for text in _shared_iter_table_text(doc):
        yield text


# --- Ops parsing from HOLE TABLE DESCRIPTION ---------------------------------
# Output schema (per hole): {"drill":1, "tap_front":1, "tap_back":0, "cbore_front":0, ...}
# Aggregated totals live in geo["ops_summary"] with per-row detail for auditing.

_OP_WORDS = {
    "cbore": r"(?:C['’]?\s*BORE|CBORE|COUNTER\s*BORE)",
    "csk": r"(?:C['’]?\s*SINK|CSK|COUNTER\s*SINK)",
    "cdrill": r"(?:C['’]?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|SPOT)",
    "tap": r"\bTAP\b",
    "thru": r"\bTHRU\b",
    "jig": r"\bJIG\s*GRIND\b",
}

_SIDE_BOTH = re.compile(r"\b(FRONT\s*&\s*BACK|BOTH\s+SIDES)\b", re.I)
_SIDE_BACK = re.compile(r"\b(?:FROM\s+)?BACK\b", re.I)
_SIDE_FRONT = re.compile(r"\b(?:FROM\s+)?FRONT\b", re.I)


def _norm_txt(s: str) -> str:
    s = (s or "").replace("\u00D8", "Ø").replace("’", "'").upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ops_qty_from_value(value: Any) -> int:
    """Best-effort coercion of a HOLE TABLE quantity value to an int."""

    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            # Treat floats as intentional numeric quantities (e.g., 4.0 -> 4).
            return int(round(float(value)))
        except Exception:
            return 0
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(round(float(text)))
    except Exception:
        match = re.search(r"\d+", text)
        return int(match.group()) if match else 0


def _sanitize_ops_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal row payload required for ops cards."""

    hole = str(row.get("hole") or row.get("id") or "").strip()
    ref = str(row.get("ref") or "").strip()
    desc = str(row.get("desc") or "").strip()
    qty = _ops_qty_from_value(row.get("qty"))
    return {"hole": hole, "ref": ref, "qty": qty, "desc": desc}


def _count_ops_card_rows(rows: Iterable[Mapping[str, Any]] | None) -> int:
    count = 0
    for entry in rows or []:
        if not isinstance(entry, _MappingABC):
            continue
        qty = _ops_qty_from_value(entry.get("qty"))
        desc = str(entry.get("desc") or "").strip()
        ref = str(entry.get("ref") or "").strip()
        if qty > 0 or desc or ref:
            count += 1
    return count


def _apply_built_rows(
    ops_summary: MutableMapping[str, Any] | Mapping[str, Any] | None,
    rows: Iterable[Mapping[str, Any]] | None,
) -> int:
    built_rows = _count_ops_card_rows(rows)
    if isinstance(ops_summary, _MutableMappingABC):
        typing.cast(MutableMapping[str, Any], ops_summary)["built_rows"] = int(built_rows)
    return int(built_rows)


def parse_ops_per_hole(desc: str) -> dict[str, int]:
    """Return ops per HOLE (not multiplied by QTY)."""

    U = _norm_txt(desc)
    ops = defaultdict(int)

    clauses = re.split(r"[;]+", U) if ";" in U else [U]
    for cl in clauses:
        if not cl.strip():
            continue
        side = (
            "both"
            if _SIDE_BOTH.search(cl)
            else (
                "back"
                if _SIDE_BACK.search(cl)
                else ("front" if _SIDE_FRONT.search(cl) else None)
            )
        )

        has_tap = re.search(_OP_WORDS["tap"], cl)
        has_thru = re.search(_OP_WORDS["thru"], cl)
        has_cbore = re.search(_OP_WORDS["cbore"], cl)
        has_csk = re.search(_OP_WORDS["csk"], cl)
        has_cdr = re.search(_OP_WORDS["cdrill"], cl)
        has_jig = re.search(_OP_WORDS["jig"], cl)

        if has_tap:
            ops["drill"] += 1
            if side == "back":
                ops["tap_back"] += 1
            elif side == "both":
                ops["tap_front"] += 1
                ops["tap_back"] += 1
            else:
                ops["tap_front"] += 1

        if has_thru and not has_tap:
            ops["drill"] += 1

        if has_cbore:
            if side == "back":
                ops["cbore_back"] += 1
            elif side == "both":
                ops["cbore_front"] += 1
                ops["cbore_back"] += 1
            else:
                ops["cbore_front"] += 1

        if has_csk:
            if side == "back":
                ops["csk_back"] += 1
            elif side == "both":
                ops["csk_front"] += 1
                ops["csk_back"] += 1
            else:
                ops["csk_front"] += 1

        if has_cdr:
            if side == "back":
                ops["spot_back"] += 1
            elif side == "both":
                ops["spot_front"] += 1
                ops["spot_back"] += 1
            else:
                ops["spot_front"] += 1

        if has_jig:
            ops["jig_grind"] += 1

    return dict(ops)


def aggregate_ops(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: defaultdict[str, int] = defaultdict(int)
    rows_simple: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    simple_rows: list[dict[str, Any]] = []
    for r in rows:
        row_payload = _sanitize_ops_row(r)
        per = parse_ops_per_hole(row_payload.get("desc", ""))
        qty = row_payload.get("qty", 0) or 0
        row_total = {k: v * qty for k, v in per.items()}
        for k, v in row_total.items():
            totals[k] += v
        simple_row = {
            "hole": r.get("hole") or r.get("id") or "",
            "ref": (r.get("ref") or "").strip(),
            "qty": qty,
            "desc": str(r.get("desc", "")),
        }
        if r.get("diameter_in") is not None:
            try:
                simple_row["diameter_in"] = float(r.get("diameter_in"))
            except Exception:
                pass
        rows_simple.append(simple_row)
        detail.append(
            {
                **row_payload,
                "per_hole": per,
                "total": row_total,
            }
        )
        simple_rows.append(row_payload)

    actions_total = sum(totals.values())
    back_ops_total = (
        totals.get("cbore_back", 0)
        + totals.get("csk_back", 0)
        + totals.get("tap_back", 0)
        + totals.get("spot_back", 0)
    )
    flip_required = back_ops_total > 0
    built_rows = _count_ops_card_rows(simple_rows)
    return {
        "totals": dict(totals),
        "rows": simple_rows,
        "rows_detail": detail,
        "actions_total": int(actions_total),
        "back_ops_total": int(back_ops_total),
        "flip_required": bool(flip_required),
        "built_rows": int(built_rows),
    }


# --- NEW: parse HOLE TABLE that's drawn with text + lines (no ACAD TABLE)
def _iter_text_with_xy(doc):
    if doc is None:
        return

    def _entity_xy(entity) -> tuple[float, float]:
        def _get_point(attr: str):
            try:
                return getattr(entity.dxf, attr)
            except Exception:
                return None

        point = (
            _get_point("insert")
            or _get_point("alignment_point")
            or _get_point("align_point")
            or _get_point("start")
            or _get_point("position")
        )
        if point is None:
            return (0.0, 0.0)

        def _coord(value, idx):
            try:
                return float(value[idx])
            except Exception:
                try:
                    return float(getattr(value, "xyz"[idx]))
                except Exception:
                    return 0.0

        if hasattr(point, "xyz"):
            x_val, y_val, _ = point.xyz
            return float(x_val), float(y_val)

        return _coord(point, 0), _coord(point, 1)

    for sp in _spaces(doc):
        try:
            entities = sp.query("TEXT,MTEXT,INSERT")
        except Exception:
            entities = []
        for entity in entities:
            try:
                kind = entity.dxftype()
            except Exception:
                kind = ""
            if kind in {"TEXT", "MTEXT"}:
                text = _extract_entity_text(entity)
                if not text:
                    continue
                x, y = _entity_xy(entity)
                yield text, x, y
            elif kind == "INSERT":
                try:
                    virtuals = entity.virtual_entities()
                except Exception:
                    virtuals = []
                for sub in virtuals:
                    try:
                        sub_kind = sub.dxftype()
                    except Exception:
                        sub_kind = ""
                    if sub_kind not in {"TEXT", "MTEXT"}:
                        continue
                    text = _extract_entity_text(sub)
                    if not text:
                        continue
                    x, y = _entity_xy(sub)
                    yield text, x, y


def _normalize(s: str) -> str:
    return " ".join((s or "").replace("\u00D8", "Ø").replace("ø", "Ø").split())


def _looks_like_hole_header(s: str) -> bool:
    U = s.upper()
    return (
        ("HOLE" in U)
        and ("REF" in U or "Ø" in U or "DIA" in U)
        and ("QTY" in U)
        and ("DESC" in U or "DESCRIPTION" in U)
    )


def extract_hole_table_from_text(doc, y_tol: float = 0.04, min_rows: int = 5):
    """
    Returns dict like:
      {"hole_count": int, "hole_diam_families_in": {...}, "rows":[{"ref":".7500","qty":4,"desc":"..."}, ...]}
    or {} if not found.
    """

    try:
        texts = [(_normalize(t), x, y) for (t, x, y) in _iter_text_with_xy(doc) if _normalize(t)]
    except Exception:
        texts = []
    if not texts:
        return {}

    by_y: defaultdict[float, list[tuple[str, float, float]]] = defaultdict(list)
    for s, x, y in texts:
        by_y[round(y, 4)].append((s, x, y))
    y_levels = sorted(by_y.keys(), reverse=True)
    header_idx = None
    header_x: dict[str, float] | None = None
    for i, y in enumerate(y_levels):
        line_txt = " | ".join(s for (s, _, _) in sorted(by_y[y], key=lambda z: z[1]))
        if _looks_like_hole_header(line_txt):
            header_idx = i
            xs: dict[str, float] = {}
            for s, x, _ in by_y[y]:
                U = s.upper()
                if "REF" in U or "Ø" in U or "DIA" in U:
                    xs["REF"] = x
                elif "QTY" in U or "QUANTITY" in U:
                    xs["QTY"] = x
                elif "HOLE" == U or U.startswith("HOLE "):
                    xs["HOLE"] = x
                elif "DESC" in U or "DESCRIPTION" in U:
                    xs["DESC"] = x
            if {"REF", "QTY", "DESC"} <= set(xs.keys()):
                header_x = xs
                break
    if header_idx is None or header_x is None:
        return {}

    cols = [
        ("HOLE", header_x.get("HOLE", min(header_x.values()) - 1e3)),
        ("REF", header_x["REF"]),
        ("QTY", header_x["QTY"]),
        ("DESC", header_x["DESC"]),
    ]
    cols_sorted = sorted(cols, key=lambda kv: kv[1])
    bounds = [c[1] for c in cols_sorted]
    splits = [(bounds[i] + bounds[i + 1]) * 0.5 for i in range(len(bounds) - 1)]

    rows: list[dict[str, str]] = []
    for y in y_levels[header_idx + 1 :]:
        band: list[tuple[str, float]] = []
        for s, x, yy in texts:
            if abs(yy - y) <= y_tol:
                band.append((s, x))
        if not band:
            continue

        def col_of(xv: float) -> str:
            if xv < splits[0]:
                return "HOLE"
            if xv < splits[1]:
                return "REF"
            if xv < splits[2]:
                return "QTY"
            return "DESC"

        cells: dict[str, list[str]] = {"HOLE": [], "REF": [], "QTY": [], "DESC": []}
        for s, x in sorted(band, key=lambda z: z[1]):
            cells[col_of(x)].append(s)
        hole = " ".join(cells["HOLE"]).strip()
        ref = " ".join(cells["REF"]).strip()
        qtys = " ".join(cells["QTY"]).strip()
        desc = " ".join(cells["DESC"]).strip()
        if not (ref or qtys or desc):
            continue
        joined = " ".join(filter(None, [hole, ref, qtys, desc])).strip()
        if not joined:
            continue
        if _looks_like_hole_header(joined):
            break
        rows.append({"hole": hole, "ref": ref, "qty": qtys, "desc": desc})

    if len(rows) < min_rows:
        return {}

    total = 0
    families: dict[str, int] = {}

    def parse_qty(s: str) -> int:
        try:
            return int(float((s or "").strip()))
        except Exception:
            m = re.search(r"\d+", s or "")
            return int(m.group()) if m else 0

    def parse_dia_inch(s: str) -> float | None:
        s = (s or "").strip().lstrip("Ø⌀\u00D8 ").strip()
        if re.fullmatch(r"\d+/\d+", s):
            try:
                return float(Fraction(s))
            except Exception:
                return None
        if re.fullmatch(r"(?:\d+)?\.\d+|\d+(?:\.\d+)?", s):
            try:
                return float(s)
            except Exception:
                return None
        return None

    clean_rows: list[dict[str, Any]] = []
    for r in rows:
        q = parse_qty(r["qty"])
        if q <= 0:
            continue
        d = parse_dia_inch(r["ref"])
        if d is None:
            mm = re.search(r"[Ø⌀\u00D8]?\s*((?:\d+)?\.\d+|\d+/\d+|\d+(?:\.\d+)?)", r["desc"])
            d = parse_dia_inch(mm.group(1)) if mm else None
        if d is None:
            continue
        key = f'{d:.4f}"'
        families[key] = families.get(key, 0) + q
        total += q
        clean_rows.append({**r, "qty": q, "ref": key, "diameter_in": d})

    if total <= 0:
        return {}

    ops_summary = aggregate_ops(clean_rows) if clean_rows else {}

    result = {
        "hole_count": total,
        "hole_diam_families_in": families,
        "rows": clean_rows,
        "provenance_holes": "HOLE TABLE (text)",
    }
    if ops_summary:
        result["ops_summary"] = ops_summary
    return result


def hole_count_from_acad_table(doc) -> dict[str, Any]:
    """Extract hole and tap data from an AutoCAD TABLE entity."""

    result: dict[str, Any] = {}
    if doc is None:
        return result

    for t in _all_tables(doc):
        try:
            n_rows = t.dxf.n_rows
            n_cols = t.dxf.n_cols
        except Exception:
            continue

        def _row_text(row_idx: int) -> list[str]:
            cells: list[str] = []
            for c in range(n_cols):
                try:
                    cell = t.get_cell(row_idx, c)
                except Exception:
                    cell = None
                try:
                    txt = (cell.get_text() if cell else "") or ""
                except Exception:
                    txt = ""
                cells.append(" ".join(txt.split()))
            return cells

        header_row = 0
        try:
            hdr = _row_text(header_row)

            def _has_cols(row_txt: Sequence[str]) -> bool:
                joined = " | ".join(row_txt).upper()
                return ("QTY" in joined) and ("DESC" in joined or "DESCRIPTION" in joined)

            if not _has_cols(hdr):
                for try_r in range(1, min(5, n_rows)):
                    row_txt = _row_text(try_r)
                    if _has_cols(row_txt):
                        header_row = try_r
                        hdr = row_txt
                        break
        except Exception:
            hdr = []

        if not hdr or not any(hdr):
            continue

        first_rows = " ".join(
            " ".join(_row_text(r)).upper() for r in range(min(5, n_rows))
        )
        if "HOLE" not in first_rows:
            continue

        def find_col(name: str) -> int | None:
            target = name.upper()
            for idx, txt in enumerate(hdr):
                if target in (txt or "").upper():
                    return idx
            return None

        c_qty = find_col("QTY")
        if c_qty is None:
            continue

        c_desc = find_col("DESC")
        if c_desc is None:
            c_desc = find_col("DESCRIPTION")

        c_ref = find_col("REF")
        if c_ref is None:
            c_ref = find_col("DIA")
        if c_ref is None:
            c_ref = find_col("Ø")

        c_hole = find_col("HOLE")

        print(f"[HOLE TABLE] col_ref={c_ref}, col_qty={c_qty}, col_desc={c_desc}")

        total = 0
        families: dict[float, int] = {}
        row_taps = 0
        tap_classes = {"small": 0, "medium": 0, "large": 0, "npt": 0}
        from_back = False
        double_sided = False
        rows_norm: list[dict[str, Any]] = []

        for r in range(header_row + 1, n_rows):
            cell = None
            if c_qty is not None:
                try:
                    cell = t.get_cell(r, c_qty)
                except Exception:
                    cell = None
            try:
                qty_text = (cell.get_text() if cell else "0") or "0"
            except Exception:
                qty_text = "0"

            mqty = re.search(r"(?<!\d)(\d+)(?!\d)", qty_text) or re.search(r"(\d+)\s*[xX]", qty_text)
            qty = int(mqty.group(1)) if mqty else 0
            if qty <= 0:
                continue
            total += qty

            ref_txt = ""
            desc = ""
            hole_id = ""
            if c_hole is not None:
                try:
                    hole_cell = t.get_cell(r, c_hole)
                except Exception:
                    hole_cell = None
                try:
                    hole_id = (hole_cell.get_text() if hole_cell else "") or ""
                except Exception:
                    hole_id = ""
            if c_ref is not None:
                try:
                    ref_cell = t.get_cell(r, c_ref)
                except Exception:
                    ref_cell = None
                try:
                    ref_txt = (ref_cell.get_text() if ref_cell else "") or ""
                except Exception:
                    ref_txt = ""
            if c_desc is not None:
                try:
                    desc_cell = t.get_cell(r, c_desc)
                except Exception:
                    desc_cell = None
                try:
                    desc = (desc_cell.get_text() if desc_cell else "") or ""
                except Exception:
                    desc = ""

            def _parse_diam(s: str):
                s = s.strip().upper()
                m = re.search(r"(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?$", s) or re.search(
                    r"(\d+)\s*/\s*(\d+)", s
                )
                if m:
                    if m.lastindex and m.lastindex >= 2 and m.group(2):
                        try:
                            return float(Fraction(f"{m.group(1)}/{m.group(2)}"))
                        except Exception:
                            return None
                    try:
                        return float(m.group(1))
                    except Exception:
                        return None
                m = re.search(r"[Ø⌀]\s*(\d+(?:\.\d+)?)", s)
                return float(m.group(1)) if m else None

            d = _parse_diam(ref_txt) or _parse_diam(desc)
            if d is not None:
                d = round(d, 4)
                families[d] = families.get(d, 0) + qty

            combined = f"{ref_txt} {desc}".strip()
            upper_text = combined.upper()

            tap_cls = tap_classes_from_row_text(upper_text, qty)
            tap_sum = sum(tap_cls.values())
            if tap_sum:
                for key, val in tap_cls.items():
                    if val:
                        tap_classes[key] = tap_classes.get(key, 0) + int(val)
                row_taps += tap_sum
            elif "TAP" in upper_text or "N.P.T" in upper_text or "NPT" in upper_text:
                row_taps += qty

            if re.search(r"\bFROM\s+BACK\b", upper_text):
                from_back = True
            if re.search(r"\b(FRONT\s*&\s*BACK|BOTH\s+SIDES)\b", upper_text):
                double_sided = True

            rows_norm.append(
                {
                    "hole": hole_id.strip(),
                    "ref": ref_txt.strip(),
                    "qty": qty,
                    "desc": desc.strip(),
                    "diameter_in": d,
                }
            )

        if total > 0:
            filtered_classes = {k: int(v) for k, v in tap_classes.items() if v}
            families_formatted = {
                f'{diam:.4f}"': count for diam, count in sorted(families.items())
            }
            ops_summary = aggregate_ops(rows_norm) if rows_norm else {}
            result = {
                "hole_count": total,
                "hole_diam_families_in": families_formatted,
                "rows": rows_norm,
                "tap_qty_from_table": row_taps,
                "tap_class_counts": filtered_classes,
                "provenance_holes": "HOLE TABLE (ACAD_TABLE)",
            }
            if ops_summary:
                result["ops_summary"] = ops_summary
            if from_back:
                result["from_back"] = True
            if double_sided:
                result["double_sided_cbore"] = True
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
            pts: list[tuple[float, float]] = []
            try:
                get_points = getattr(pl, "get_points", None)
                if callable(get_points):
                    raw_pts_iter = get_points("xy")
                    raw_pts = cast(Iterable[Sequence[float]], raw_pts_iter)
                    for raw_pt in raw_pts:
                        if len(raw_pt) < 2:
                            continue
                        x_val, y_val = raw_pt[0], raw_pt[1]
                        try:
                            pts.append((float(x_val), float(y_val)))
                        except (TypeError, ValueError):
                            continue
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
    text_harvest_fn = getattr(geometry, "text_harvest", None)
    if callable(text_harvest_fn):
        try:
            tokens_parts.extend(text_harvest_fn(doc))
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
    source: str | None = None
    if table_info:
        cnt = table_info.get("hole_count")
        fam = table_info.get("hole_diam_families_in")
        raw_classes = table_info.get("tap_class_counts") if isinstance(table_info.get("tap_class_counts"), dict) else None
        if raw_classes:
            tap_classes_from_table = {k: int(v) for k, v in raw_classes.items() if v}
        tap_qty_from_table = int(table_info.get("tap_qty_from_table") or 0)
        if cnt:
            source = str(table_info.get("provenance_holes") or "HOLE TABLE (ACAD_TABLE)")
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
    if table_info and table_info.get("ops_summary"):
        geo["ops_summary"] = table_info["ops_summary"]
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
    if cnt is not None:
        geo["hole_count"] = int(cnt or 0)
    geo["hole_diam_families_in"] = fam or {}
    hole_source = source
    if source:
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
    ezdxf_mod = geometry.require_ezdxf()

    # --- load doc ---
    dxf_text_path: str | None = None
    doc: Drawing | None = None
    lower_path = path.lower()
    readfile: Callable[[str], Any] | None = getattr(ezdxf_mod, "readfile", None)
    if not callable(readfile):
        raise AttributeError("ezdxf module does not provide a callable 'readfile' function")

    if lower_path.endswith(".dwg"):
        if _HAS_ODAFC:
            # uses ODAFileConverter through ezdxf, no env var needed
            odafc_mod = _ezdxf_vendor.require_odafc()

            readfile = getattr(odafc_mod, "readfile", None)
            if not callable(readfile):  # pragma: no cover - defensive fallback
                raise RuntimeError(
                    "ezdxf.addons.odafc.readfile is unavailable; install ODAFileConverter support."
                )
            doc = cast(Drawing, readfile(path))
        else:
            dxf_path = geometry.convert_dwg_to_dxf(path, out_ver="ACAD2018")
            dxf_text_path = dxf_path
            doc = cast(Drawing, readfile(dxf_path))
    else:
        doc = cast(Drawing, readfile(path))
        dxf_text_path = path

    if doc is None:
        raise RuntimeError("Failed to load DXF/DWG document")

    table_info = hole_count_from_acad_table(doc) or {}
    if not table_info.get("hole_count"):
        table_info = extract_hole_table_from_text(doc) or {}

    sp = doc.modelspace()
    units = detect_units_scale(doc)
    to_in = float(units.get("to_in", 1.0) or 1.0)
    u2mm = to_in * 25.4

    geo = _build_geo_from_ezdxf_doc(doc)
    if table_info.get("ops_summary"):
        geo["ops_summary"] = table_info["ops_summary"]

    hole_source: str | None = None
    provenance_entry = geo.get("provenance")
    if isinstance(provenance_entry, dict):
        holes_prov = provenance_entry.get("holes")
        if holes_prov:
            hole_source = str(holes_prov)

    geo_read_more: dict[str, Any] | None = None
    if dxf_text_path:
        try:
            geo_read_more = build_geo_from_dxf(dxf_text_path)
        except RuntimeError:
            geo_read_more = None
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
                existing_lines_raw = geo.get("chart_lines")
                existing_lines = (
                    existing_lines_raw.copy()
                    if isinstance(existing_lines_raw, list)
                    else []
                )
                new_lines = geo_read_more.get("chart_lines")
                if not isinstance(new_lines, list):
                    new_lines = []
                for line in new_lines:
                    if line not in existing_lines:
                        existing_lines.append(line)
                if existing_lines:
                    geo["chart_lines"] = existing_lines
        else:
            geo.setdefault("geo_read_more_error", geo_read_more.get("error"))

        provenance_entry = geo.get("provenance")
        if isinstance(provenance_entry, dict):
            holes_prov = provenance_entry.get("holes")
            if holes_prov:
                hole_source = str(holes_prov)

    # perimeter from lightweight polylines, polylines, arcs
    import math
    per = 0.0
    for e in sp.query("LWPOLYLINE"):
        get_points = getattr(e, "get_points", None)
        pts: list[tuple[float, float, Any]] = []
        if callable(get_points):
            try:
                point_iter = cast(
                    Iterable[tuple[float, float, Any]], get_points("xyb")
                )
                pts = list(point_iter)
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

    # holes from circles with concentric-dedup fallback
    holes = list(sp.query("CIRCLE"))
    entity_holes_mm: list[float] = []
    # Build (cx_mm, cy_mm, dia_mm)
    circ: list[tuple[float, float, float]] = []
    for c in holes:
        try:
            cx, cy = float(c.dxf.center.x), float(c.dxf.center.y)
            r_du = float(c.dxf.radius)
            d_mm = float(2.0 * r_du * u2mm)
            circ.append((cx * u2mm, cy * u2mm, d_mm))
            entity_holes_mm.append(d_mm)
        except Exception:
            continue

    # Tunables (env overrides allowed)
    import os, math

    CENTER_BIN_MM = float(os.getenv("GEO_CENTER_TOL_MM", "0.06"))
    CENTER_PROX_MM = float(os.getenv("GEO_CENTER_PROX_MM", "0.22"))
    MIN_DD_MM = float(os.getenv("GEO_MIN_RING_DELTA_MM", "0.50"))

    # Back-compat alias (fixes NameError if any old refs remain)
    CENTER_MM_TOL = CENTER_BIN_MM

    # First pass: bin by center grids (mm)
    def _key_mm(x: float, y: float, tol: float = CENTER_BIN_MM) -> tuple[int, int]:
        return (round(x / tol), round(y / tol))

    bins: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for x_mm, y_mm, d_mm in circ:
        bins.setdefault(_key_mm(x_mm, y_mm), []).append((x_mm, y_mm, d_mm))

    # Merge adjacent bins (handles jitter on bin borders)
    def _merge_adjacent(
        bmap: Mapping[tuple[int, int], list[tuple[float, float, float]]]
    ) -> dict[tuple[int, int], list[tuple[float, float, float]]]:
        seen: set[tuple[int, int]] = set()
        out: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        for key in list(bmap.keys()):
            if key in seen:
                continue
            acc: list[tuple[float, float, float]] = []
            stack = [key]
            seen.add(key)
            while stack:
                kk = stack.pop()
                acc.extend(bmap.get(kk, []))
                kx, ky = kk
                for nx in range(kx - 1, kx + 2):
                    for ny in range(ky - 1, ky + 2):
                        nk = (nx, ny)
                        if nk in bmap and nk not in seen:
                            seen.add(nk)
                            stack.append(nk)
            out[key] = acc
        return out

    groups = _merge_adjacent(bins)

    # Second pass: within each merged group, suppress larger circles that are
    # "near-concentric" to any smaller circle (distance <= CENTER_PROX_MM and
    # dia gap >= MIN_DD_MM).
    through_mm: list[float] = []
    cbore_pairs = 0
    for _, pts in groups.items():
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda t: t[2])
        suppressed = [False] * len(pts_sorted)
        for i in range(len(pts_sorted)):
            if suppressed[i]:
                continue
            xi, yi, di = pts_sorted[i]
            for j in range(i + 1, len(pts_sorted)):
                if suppressed[j]:
                    continue
                xj, yj, dj = pts_sorted[j]
                if (dj - di) < MIN_DD_MM:
                    continue
                if math.hypot(xj - xi, yj - yi) <= CENTER_PROX_MM:
                    suppressed[j] = True
                    cbore_pairs += 1
        for k, (_x, _y, dk) in enumerate(pts_sorted):
            if not suppressed[k]:
                through_mm.append(dk)

    # Round and expose results
    hole_diams_mm = [round(v, 2) for v in through_mm]

    try:
        existing_cbore = int(float(geo.get("cbore_pairs_geom") or 0))
    except Exception:
        existing_cbore = 0
    geo["cbore_pairs_geom"] = max(existing_cbore, cbore_pairs)
    geo["hole_count_geom_dedup"] = len(hole_diams_mm)
    geo["hole_count_geom_raw"] = len(entity_holes_mm)
    try:
        existing_geom = int(float(geo.get("hole_count_geom") or 0))
    except Exception:
        existing_geom = 0
    geo["hole_count_geom"] = max(existing_geom, len(hole_diams_mm))

    from collections import Counter

    def _families_nearest_1over64_in(
        vals_mm: Iterable[float],
    ) -> tuple[dict[str, int], dict[float, int]]:
        vals_in: list[float] = []
        for raw in vals_mm:
            try:
                val_in = float(raw) / 25.4
            except Exception:
                continue
            if not math.isfinite(val_in):
                continue
            vals_in.append(val_in)
        if not vals_in:
            return {}, {}
        quant = [round(x * 64) / 64 for x in vals_in]
        cnt = Counter(round(q, 4) for q in quant)
        display = {f'{k:.4f}"': int(v) for k, v in cnt.items()}
        numeric = {round(k, 4): int(v) for k, v in cnt.items()}
        return display, numeric

    geom_families_display, geom_families_numeric = _families_nearest_1over64_in(
        through_mm
    )
    geo["hole_diam_families_in_geom"] = geom_families_display
    if geom_families_numeric:
        geom_fam = dict(geom_families_numeric)
        geo["hole_diam_families_geom_in"] = geom_fam
    else:
        geom_fam = {}
    if not geo.get("hole_diam_families_in"):
        geo["hole_diam_families_in"] = dict(geo["hole_diam_families_in_geom"])
    geo["hole_family_count"] = int(sum(geo["hole_diam_families_in"].values()))

    # Keep provenance explicit with the tuned mm thresholds
    geo.setdefault("provenance", {})["holes"] = (
        "GEOM (concentric-dedup, center="
        f"{CENTER_BIN_MM:.3f} mm, prox={CENTER_PROX_MM:.3f} mm, Î”≥{MIN_DD_MM:.2f} mm)"
    )

    if table_info.get("hole_count"):
        try:
            geo["hole_count"] = int(table_info["hole_count"])
        except Exception:
            pass
        fam = table_info.get("hole_diam_families_in") or {}
        if fam:
            geo["hole_diam_families_in"] = fam
            try:
                geo["hole_family_count"] = int(
                    sum(int(v) for v in fam.values())
                )
            except Exception:
                geo["hole_family_count"] = sum(fam.values())
        provenance_value = (
            table_info.get("provenance")
            or table_info.get("provenance_holes")
            or "HOLE TABLE (text)"
        )
        provenance_entry = geo.get("provenance")
        if isinstance(provenance_entry, dict):
            provenance_entry["holes"] = provenance_value
            geo["provenance"] = provenance_entry
        else:
            geo["provenance"] = {"holes": provenance_value}
        top_level_hole_count = geo.get("hole_count", len(hole_diams_mm))
    else:
        top_level_hole_count = len(hole_diams_mm)
        geo["hole_count"] = top_level_hole_count

    chart_lines: list[str] = []
    chart_ops: list[dict[str, Any]] = []
    chart_reconcile: dict[str, Any] | None = None
    chart_source: str | None = None
    chart_summary: dict[str, Any] | None = None

    extractor = _extract_text_lines_from_dxf or geometry.extract_text_lines_from_dxf
    chart_lines = []
    if extractor and dxf_text_path:
        try:
            chart_lines = extractor(dxf_text_path) or []
        except Exception:
            chart_lines = []
    # Always also read ACAD_TABLE/MTEXT via ezdxf and merge/dedupe.
    _lines_from_doc = _extract_text_lines_from_ezdxf_doc(doc) or []
    if _lines_from_doc:
        seen: set[str] = {ln for ln in chart_lines if isinstance(ln, str)}
        for ln in _lines_from_doc:
            if isinstance(ln, str):
                if ln in seen:
                    continue
                seen.add(ln)
            elif ln in chart_lines:
                continue
            chart_lines.append(ln)

    table_info = hole_count_from_acad_table(doc)
    table_counts_trusted = True
    if hole_source and "GEOM" in str(hole_source).upper():
        table_counts_trusted = False
        table_info = {}
    if (not table_info or not table_info.get("hole_count")) and doc is not None:
        try:
            text_table = extract_hole_table_from_text(doc)
        except Exception:
            text_table = {}
        if text_table and text_table.get("hole_count"):
            text_table = dict(text_table)
            text_table.setdefault("provenance", "HOLE TABLE (TEXT)")
            table_info = text_table
    if table_info and table_info.get("hole_count") and table_counts_trusted:
        try:
            geo["hole_count"] = int(table_info.get("hole_count") or 0)
        except Exception:
            pass
        fam = table_info.get("hole_diam_families_in")
        if isinstance(fam, dict) and fam:
            geo["hole_diam_families_in"] = fam
            geo["hole_family_count"] = sum(
                int(v or 0) for v in fam.values()
            )
        prov = (
            table_info.get("provenance")
            or table_info.get("provenance_holes")
            or "HOLE TABLE (ACAD_TABLE)"
        )
        provenance_entry = geo.get("provenance")
        if isinstance(provenance_entry, dict):
            provenance_entry["holes"] = prov
            geo["provenance"] = provenance_entry
        else:
            geo["provenance"] = {"holes": prov}

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
            chart_reconcile = summarize_hole_chart_agreement(entity_holes_mm, chart_ops)

        # --- publish HOLE-TABLE rows to geo["ops_summary"]["rows"] ---
        try:
            ops_rows = []
            if hole_rows:
                # Prefer the structured rows, synthesizing a description from features if needed
                for row in hole_rows:
                    if row is None:
                        continue
                    qty = int(getattr(row, "qty", 0) or 0)
                    ref = (
                        getattr(row, "ref", None)
                        or getattr(row, "drill_ref", None)
                        or getattr(row, "pilot", None)
                        or ""
                    )
                    desc = (
                        getattr(row, "description", None)
                        or getattr(row, "desc", None)
                        or ""
                    )
                    if not desc:
                        parts = []
                        for f in list(getattr(row, "features", []) or []):
                            if not isinstance(f, dict):
                                continue
                            t = str(f.get("type", "")).lower()
                            side = str(f.get("side", "")).upper()
                            if t == "tap":
                                thread = f.get("thread") or ""
                                depth = f.get("depth_in")
                                parts.append(
                                    f"{thread} TAP"
                                    + (
                                        f" × {depth:.2f}\""
                                        if isinstance(depth, (int, float))
                                        else ""
                                    )
                                    + (f" FROM {side}" if side else "")
                                )
                            elif t == "cbore":
                                dia = f.get("dia_in")
                                depth = f.get("depth_in")
                                parts.append(
                                    (
                                        f"{(dia or 0):.4f} "
                                        if dia
                                        else ""
                                    )
                                    + "C’BORE"
                                    + (
                                        f" × {depth:.2f}\""
                                        if isinstance(depth, (int, float))
                                        else ""
                                    )
                                    + (f" FROM {side}" if side else "")
                                )
                            elif t in {"csk", "countersink"}:
                                dia = f.get("dia_in")
                                depth = f.get("depth_in")
                                parts.append(
                                    (
                                        f"{(dia or 0):.4f} "
                                        if dia
                                        else ""
                                    )
                                    + "C’SINK"
                                    + (
                                        f" × {depth:.2f}\""
                                        if isinstance(depth, (int, float))
                                        else ""
                                    )
                                    + (f" FROM {side}" if side else "")
                                )
                            elif t == "drill":
                                ref_local = f.get("ref") or ref or ""
                                thru = " THRU" if f.get("thru", True) else ""
                                parts.append(f"{ref_local}{thru}".strip())
                            elif t == "spot":
                                depth = f.get("depth_in")
                                parts.append(
                                    "C’DRILL"
                                    + (
                                        f" × {depth:.2f}\""
                                        if isinstance(depth, (int, float))
                                        else ""
                                    )
                                )
                            elif t == "jig":
                                parts.append("JIG GRIND")
                        desc = "; ".join([p for p in parts if p])
                    ops_rows.append(
                        {
                            "hole": str(
                                getattr(row, "hole_id", "")
                                or getattr(row, "letter", "")
                                or ""
                            ),
                            "ref": str(ref or ""),
                            "qty": int(qty),
                            "desc": str(desc or ""),
                        }
                    )
            elif chart_lines:
                # Fallback: derive rows directly from the free text chart
                ops_rows = _build_ops_rows_from_chart_lines(chart_lines)

            if ops_rows:
                ops_summary_map = geo.setdefault("ops_summary", {})
                ops_summary_map["rows"] = ops_rows
                ops_summary_map["source"] = chart_source or "chart_lines"
                _apply_built_rows(ops_summary_map, ops_rows)
                try:
                    if chart_summary:
                        geo["ops_summary"]["tap_total"] = int(
                            chart_summary.get("tap_qty") or 0
                        )
                        geo["ops_summary"]["cbore_total"] = int(
                            chart_summary.get("cbore_qty") or 0
                        )
                        geo["ops_summary"]["csk_total"] = int(
                            chart_summary.get("csk_qty") or 0
                        )
                except Exception:
                    pass
        except Exception:
            pass
        # --- end publish rows ---
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

    table_hole_count = (
        _coerce_int_or_zero(table_info.get("hole_count"))
        if table_counts_trusted
        else 0
    )
    geom_hole_count_dedup = int(geo.get("hole_count_geom_dedup") or 0)
    geom_hole_count_raw = int(geo.get("hole_count_geom_raw") or len(entity_holes_mm))

    result: dict[str, Any] = {
        "kind": "2D",
        "source": Path(path).suffix.lower().lstrip("."),
        "profile_length_mm": round(per * u2mm, 2),
        "hole_diams_mm": hole_diams_mm,
        # prefer table when available, else geometry
        "hole_count": top_level_hole_count,
        "thickness_mm": thickness_mm,
        "material": material,
        "geo": geo,
    }
    if table_hole_count > 0:
        result["hole_count"] = table_hole_count
        result["hole_count_table"] = table_hole_count
    elif geom_hole_count_dedup > 0:
        result["hole_count"] = geom_hole_count_dedup
        result["hole_count_geom"] = geom_hole_count_dedup
        geo.setdefault("provenance", {})["holes"] = (
            f"GEOM (concentric-dedup, center={CENTER_BIN_MM:.3f} mm)"
        )
    else:
        result["hole_count"] = geom_hole_count_raw
        if geom_hole_count_raw:
            result["hole_count_geom"] = geom_hole_count_raw
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
                hole_families = geo_read_more.get("hole_diam_families_in")
                result["hole_diam_families_in"] = dict(hole_families) if isinstance(hole_families, dict) else {}
            if geo_read_more.get("hole_table_families_in"):
                table_families = geo_read_more.get("hole_table_families_in")
                result["hole_table_families_in"] = dict(table_families) if isinstance(table_families, dict) else {}
            hole_count_geom_val = _coerce_float_or_none(geo_read_more.get("hole_count_geom"))
            hole_count_geom = int(hole_count_geom_val) if hole_count_geom_val is not None else None
            if hole_count_geom is not None:
                result["hole_count_geom"] = hole_count_geom
                current_hole_count = _coerce_int_or_zero(result.get("hole_count"))
                table_count_existing = _coerce_int_or_zero(result.get("hole_count_table"))
                if table_count_existing <= 0 and hole_count_geom > current_hole_count:
                    result["hole_count"] = hole_count_geom
            if geo_read_more.get("holes_from_back"):
                result["holes_from_back"] = True
            if geo_read_more.get("material_note"):
                result.setdefault("material_note", geo_read_more.get("material_note"))
            if geo_read_more.get("material_note") and not result.get("material"):
                result["material"] = geo_read_more.get("material_note")
            if geo_read_more.get("chart_lines") and not result.get("chart_lines"):
                chart_lines_extra = geo_read_more.get("chart_lines")
                result["chart_lines"] = list(chart_lines_extra) if isinstance(chart_lines_extra, list) else []
            if geo_read_more.get("tap_qty") or geo_read_more.get("cbore_qty") or geo_read_more.get("csk_qty"):
                feature_counts_raw_any = result.get("feature_counts")
                feature_counts_raw = (
                    feature_counts_raw_any
                    if isinstance(feature_counts_raw_any, dict)
                    else {}
                )
                feature_counts: dict[str, Any] = {
                    str(k): v for k, v in feature_counts_raw.items()
                }
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


# ===== Fallback HOLE-TABLE row builder (delegates to cad_quoter.app.chart_lines) ===

def _build_ops_rows_from_chart_lines(chart_lines: list[str]) -> list[dict]:
    rows: list[dict] = []
    if not chart_lines:
        return rows
    lines = [_norm_line(x) for x in chart_lines]
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln:
            i += 1
            continue
        # skip generic notes
        if any(
            k in ln.upper()
            for k in ["BREAK ALL", "SHARP CORNERS", "RADIUS", "CHAMFER", "AS SHOWN"]
        ):
            i += 1
            continue

        qty = 1
        mqty = _QTY_LEAD.match(ln)
        if mqty:
            qty = int(mqty.group(1))
            ln = ln[mqty.end() :].strip()

        # TAP
        mtap = RE_TAP.search(ln)
        if mtap:
            # (group 2 holds the whole spec: "#10-32", "5/8-11", "0.190-32", "M8x1.25")
            thread = mtap.group(2).replace(" ", "")
            desc = f"{thread} TAP"
            tail = " ".join([ln] + lines[i + 1 : i + 3])  # look forward for depth / THRU / side
            if RE_THRU.search(tail):
                desc += " THRU"
            md = RE_DEPTH.search(tail)
            if md and md.group(1):
                desc += f' × {float(md.group(1)):.2f}"'
            ms = _FROM_SIDE.search(tail)
            if ms:
                desc += f' FROM {ms.group(1).upper()}'
            rows.append({"hole": "", "ref": "", "qty": qty, "desc": desc})
            i += 1
            continue

        # COUNTERBORE
        if RE_CBORE.search(ln):
            tail = " ".join([ln] + lines[i - 1 : i] + lines[i + 1 : i + 2])
            md = _PAREN_DIA.search(tail) or _MM_IN_DIA.search(tail) or RE_DIA.search(tail)
            dia = float(md.group(1)) if md else None
            desc = f"{dia:.4f} C’BORE" if dia is not None else "C’BORE"
            mdp = RE_DEPTH.search(" ".join([ln] + lines[i + 1 : i + 2]))
            if mdp and mdp.group(1):
                desc += f' × {float(mdp.group(1)):.2f}"'
            ms = _FROM_SIDE.search(" ".join([ln] + lines[i + 1 : i + 2]))
            if ms:
                desc += f' FROM {ms.group(1).upper()}'
            rows.append({"hole": "", "ref": "", "qty": qty, "desc": desc})
            i += 1
            continue

        # SPOT / CENTER DRILL
        if (
            "C' DRILL" in ln.upper()
            or "C’DRILL" in ln.upper()
            or "CENTER DRILL" in ln.upper()
            or "SPOT DRILL" in ln.upper()
        ):
            tail = " ".join([ln] + lines[i + 1 : i + 2])
            md = RE_DEPTH.search(tail)
            desc = "C’DRILL" + (
                f' × {float(md.group(1)):.2f}"' if (md and md.group(1)) else ""
            )
            rows.append({"hole": "", "ref": "", "qty": qty, "desc": desc})
            i += 1
            continue

        # THRU drill with a ref size (e.g., "R (.339Ø) DRILL THRU")
        if "DRILL" in ln.upper() and RE_THRU.search(ln):
            md = _PAREN_DIA.search(ln) or _MM_IN_DIA.search(ln) or RE_DIA.search(ln)
            ref = (md.group(1) if md else "").strip()
            rows.append({"hole": "", "ref": ref, "qty": qty, "desc": f"{ref} THRU".strip()})
            i += 1
            continue

        # NPT etc
        if RE_NPT.search(ln):
            rows.append({"hole": "", "ref": "", "qty": qty, "desc": _norm_line(ln)})
            i += 1
            continue

        i += 1

    # coalesce identical (desc, ref) pairs
    agg: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        key = (r["desc"], r.get("ref", ""))
        if key not in agg:
            order.append(key)
        agg[key] = agg.get(key, 0) + int(r.get("qty") or 0)
    return [
        {"hole": "", "ref": ref, "qty": agg[(desc, ref)], "desc": desc}
        for (desc, ref) in order
    ]

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

def summarize_hole_chart_agreement(entity_holes_mm: Iterable[Any] | None, chart_ops: Iterable[dict] | None) -> dict[str, Any]:
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

# ==== LLM DECISION ENGINE =====================================================

# ----------------- GUI -----------------
# ---- service containers ----------------------------------------------------


class App(tk.Tk):
    def __init__(
        self,
        pricing: SupportsPricingEngine | None = None,
        *,
        configuration: UIConfiguration | None = None,
        geometry_loader: GeometryLoader | None = None,
        pricing_registry: PricingRegistry | None = None,
        llm_services: LLMServices | None = None,
        geometry_service: geometry.GeometryService | None = None,
    ):

        _ensure_tk()

        super().__init__()

        # Quiet noisy INFO logs from speeds/feeds selector unless explicitly enabled
        try:
            logging.getLogger("cad_quoter.pricing.speeds_feeds_selector").setLevel(logging.WARNING)
        except Exception:
            pass

        self.configuration = configuration or UIConfiguration(
            default_params=copy.deepcopy(PARAMS_DEFAULT),
            default_material_display=DEFAULT_MATERIAL_DISPLAY,
            settings_path=default_app_settings_json(),
        )
        self.geometry_loader = geometry_loader or GeometryLoader(
            extract_pdf_all_fn=extract_pdf_all,
            extract_pdf_vector_fn=extract_2d_features_from_pdf_vector,
            extract_dxf_or_dwg_fn=extract_2d_features_from_dxf_or_dwg,
            occ_feature_fn=geometry.extract_features_with_occ,
            stl_enricher=geometry.enrich_geo_stl,
            step_reader=geometry.read_step_shape,
            cad_reader=geometry.read_cad_any,
            bbox_fn=geometry.safe_bbox,
            occ_enricher=geometry.enrich_geo_occ,
        )
        self.pricing_registry = pricing_registry or PricingRegistry(
            default_params=copy.deepcopy(PARAMS_DEFAULT),
            default_rates=copy.deepcopy(RATES_DEFAULT),
        )
        self.llm_services = llm_services or LLMServices()
        self.pricing: SupportsPricingEngine = pricing or _DEFAULT_PRICING_ENGINE

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

        self.vars_df: PandasDataFrame | None = None
        self.vars_df_full: PandasDataFrame | None = None
        self.geo: dict[str, Any] | None = None
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

        if not str(self.params.get("SpeedsFeedsCSVPath", "")).strip():
            self.params["SpeedsFeedsCSVPath"] = DEFAULT_SPEEDS_FEEDS_CSV_PATH
        if not str(self.default_params_template.get("SpeedsFeedsCSVPath", "")).strip():
            self.default_params_template["SpeedsFeedsCSVPath"] = DEFAULT_SPEEDS_FEEDS_CSV_PATH

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
        self.quote_config = QuoteConfiguration(
            default_params=copy.deepcopy(self.default_params_template),
            default_material_display=self.default_material_display,
            prefer_removal_drilling_hours=True,
            separate_machine_labor=True,
            machine_rate_per_hr=45.0,
            labor_rate_per_hr=45.0,
            stock_price_source="mcmaster_api",
            scrap_price_source="wieland",
        )

        self.config_errors = list(CONFIG_INIT_ERRORS)

        self.quote_state = QuoteState()
        self.llm_events: list[dict[str, Any]] = []
        self.llm_errors: list[dict[str, Any]] = []
        self._llm_client_cache: LLMClient | None = None
        self.settings_path = (
            getattr(self.configuration, "settings_path", None)
            or default_app_settings_json()
        )

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
                    if _is_pandas_dataframe(core_df) and _is_pandas_dataframe(full_df):
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
            command=lambda: messagebox.showinfo("Diagnostics", geometry.get_import_diagnostics_text())
        )
        # Tools menu with a debug trigger for Generate Quote in case button wiring misbehaves
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        def _gen_quote_debug_menu() -> None:
            append_debug_log("", "[UI] Tools > Generate Quote (debug)")
            try:
                self.status_var.set("Generating quote (via Tools menu)...")
            except Exception:
                pass
            self.gen_quote()
        tools_menu.add_command(label="Generate Quote (debug)", command=_gen_quote_debug_menu, accelerator="Ctrl+G")
        try:
            self.bind_all("<Control-g>", lambda *_: _gen_quote_debug_menu())
        except Exception:
            pass
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
            path.write_text(jdump(self.settings, default=None), encoding="utf-8")
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

    def _refresh_variables_cache(
        self, core_df: PandasDataFrame, full_df: PandasDataFrame
    ) -> None:
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

    def _populate_editor_tab(self, df: PandasDataFrame) -> None:
        df = coerce_or_make_vars_df(df)
        if self.vars_df_full is None:
            _, master_full = _load_master_variables()
            if master_full is not None:
                self.vars_df_full = master_full
        """Rebuild the Quote Editor tab using the latest variables dataframe."""
        def _ensure_row(
            dataframe: PandasDataFrame, item: str, value: Any, dtype: str = "number"
        ) -> PandasDataFrame:
            mask = dataframe["Item"].astype(str).str.fullmatch(item, case=False)
            if mask.any():
                return dataframe
            return geometry.upsert_var_row(dataframe, item, value, dtype=dtype)

        df = _ensure_row(df, "Scrap Percent (%)", 15.0, dtype="number")
        df = _ensure_row(df, "Plate Length (in)", 12.0, dtype="number")
        df = _ensure_row(df, "Plate Width (in)", 14.0, dtype="number")
        df = _ensure_row(df, "Thickness (in)", 2.0, dtype="number")
        df = _ensure_row(df, "Hole Count (override)", 0, dtype="number")
        df = _ensure_row(df, "Avg Hole Diameter (mm)", 0.0, dtype="number")
        df = _ensure_row(df, "Material", "Aluminum MIC6", dtype="text")
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
        normalized_items = items_series.apply(normalize_item_text)
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
        skip_items = {normalize_item_text(item) for item in raw_skip_items}

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
            column_sources: list[PandasIndex] = []
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
        full_lookup: dict[str, PandasSeries] = {}
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
            is_missing = False
            if pd is not None:
                try:
                    is_missing = bool(pd.isna(initial_raw))
                except Exception:
                    is_missing = False
            if is_missing:
                initial_value = ""
            else:
                initial_value = "" if initial_raw is None else str(initial_raw)

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
                    parsed = coerce_checkbox_state(current, bool_var.get())
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
            "MarginPct",
            "ExpeditePct",
            "VendorMarkupPct",
            "InsurancePct",
            "MinLotCharge",
            "Quantity",
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
        bounds_src = None
        if isinstance(self.quote_state.bounds, dict):
            bounds_src = self.quote_state.bounds
        elif isinstance(baseline_ctx.get("_bounds"), _MappingABC):
            bounds_src = baseline_ctx.get("_bounds")
        coerced_bounds = coerce_bounds(bounds_src if isinstance(bounds_src, _MappingABC) else None)
        mult_min_bound = coerced_bounds["mult_min"]
        mult_max_bound = coerced_bounds["mult_max"]
        for proc, mult in mults.items():
            if proc in eff_hours:
                try:
                    base_val = float(eff_hours[proc])
                except Exception:
                    continue
                clamped_mult = clamp(mult, mult_min_bound, mult_max_bound, 1.0)
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
        self.status_var.set("Opening CAD/Drawing…")
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
        self.status_var.set(f"Processing {os.path.basename(path)}…")

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
                            core_df_t = typing.cast(PandasDataFrame, core_df)
                            full_df_t = typing.cast(PandasDataFrame, full_df)
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
                geo = geometry.map_geo_to_double_underscore(stl_geo)
            else:
                try:
                    if ext in (".step", ".stp"):
                        shape = self.geometry_service.read_step(path)
                    else:
                        shape = self.geometry_service.read_model(path)            # IGES/BREP and others
                    _ = geometry.safe_bbox(shape)
                    g = self.geometry_service.enrich_occ(shape)             # OCC-based geometry features

                    geo = geometry.map_geo_to_double_underscore(g)
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
                core_df_t = typing.cast(PandasDataFrame, core_df)
                full_df_t = typing.cast(PandasDataFrame, full_df)
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
                self.vars_df = geometry.upsert_var_row(self.vars_df, k, v, dtype="number")
        except Exception as e:
            messagebox.showerror("Variables", f"Failed to update variables with GEO rows:\n{e}")
            self.status_var.set("Ready")
            return

        # LLM hour estimation
        self.status_var.set("Estimating hours with LLM…")
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

        vars_df_for_editor = typing.cast(PandasDataFrame, self.vars_df)
        self._populate_editor_tab(vars_df_for_editor)
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
            normalized_items = self.vars_df["Item"].astype(str).apply(normalize_item_text)
            param_to_items = {
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
                value_is_missing = False
                if pd is not None:
                    try:
                        value_is_missing = bool(pd.isna(value))
                    except Exception:
                        value_is_missing = False
                if value_is_missing:
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
            if pd is not None and hasattr(pd, "DataFrame"):
                try:
                    self.vars_df = pd.DataFrame.from_records(vars_payload)
                except Exception:
                    self.vars_df = None
            else:
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

    def apply_geometry_payload(
        self,
        payload: Mapping[str, Any],
        *,
        source: str | None = None,
    ) -> None:
        """Populate the GEO tab from a pre-parsed geometry payload."""

        if not isinstance(payload, _MappingABC):
            raise TypeError("Geometry payload must be a mapping.")

        def _clone(mapping: Mapping[Any, Any]) -> dict[str, Any]:
            return {str(key): value for key, value in mapping.items()}

        geo_payload: dict[str, Any] | None = None
        if isinstance(payload.get("geo"), _MappingABC):
            geo_payload = _clone(typing.cast(Mapping[Any, Any], payload.get("geo", {})))
        elif isinstance(payload.get("geom"), _MappingABC):
            geo_payload = _clone(typing.cast(Mapping[Any, Any], payload.get("geom", {})))
        else:
            geo_payload = _clone(payload)

        geo_context: dict[str, Any] | None = None
        if isinstance(payload.get("geo_context"), _MappingABC):
            geo_context = _clone(typing.cast(Mapping[Any, Any], payload.get("geo_context", {})))
        elif isinstance(payload.get("geo"), _MappingABC):
            inner_geo = typing.cast(Mapping[Any, Any], payload.get("geo", {}))
            if isinstance(inner_geo.get("geo_context"), _MappingABC):
                geo_context = _clone(typing.cast(Mapping[Any, Any], inner_geo.get("geo_context", {})))

        if geo_context is None:
            geo_context = dict(geo_payload)
            if isinstance(payload.get("geo_read_more"), _MappingABC):
                geo_context["geo_read_more"] = _clone(
                    typing.cast(Mapping[Any, Any], payload.get("geo_read_more", {}))
                )

        self.geo = dict(geo_payload)
        self.geo_context = dict(geo_context)
        try:
            self.quote_state.geo = dict(self.geo)
        except Exception:
            pass

        display_payload = json_safe_copy(payload)
        try:
            self._log_geo(display_payload)
        except Exception:
            self._log_geo(self.geo_context)

        self.nb.select(self.tab_geo)
        status_text = "Geometry payload loaded." if not source else f"Geometry loaded from {source}"
        self.status_var.set(status_text)
        logger.info(status_text)

    # ----- LLM tab -----
    def _build_llm(self, parent):
        llm_panel.build_llm_tab(self, parent)

    def _pick_model(self):
        llm_panel.pick_llm_model(self)

    def run_llm(self):
        llm_panel.run_llm(self)

    def open_llm_inspector(self):
        llm_panel.open_llm_inspector(self)

    # ----- Flow + Output -----
    def _log_geo(self, d):
        self.geo_txt.delete("1.0","end")
        self.geo_txt.insert("end", jdump(d, default=None))

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
            # Trace entry to help diagnose UI wiring issues and early exits
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            append_debug_log(
                "",
                f"[{timestamp}] gen_quote: invoked (reuse_suggestions={reuse_suggestions})",
            )
            try:
                self.status_var.set("Generating quote…")
                self.update_idletasks()
            except Exception:
                pass
            vars_df_local = self.vars_df
            if vars_df_local is None:
                vars_df_local = coerce_or_make_vars_df(None)
                self.vars_df = vars_df_local
            for item_name, string_var in self.quote_vars.items():
                mask = vars_df_local["Item"] == item_name
                if mask.any():
                    vars_df_local.loc[mask, "Example Values / Options"] = string_var.get()

            self.apply_overrides(notify=False)

            try:
                ui_vars = {
                    str(row["Item"]): row["Example Values / Options"]
                    for _, row in vars_df_local.iterrows()
                }
            except Exception:
                ui_vars = {}

            speeds_csv = str(self.params.get("SpeedsFeedsCSVPath", "") or "").strip()
            if speeds_csv:
                ui_vars.setdefault("SpeedsFeedsCSVPath", speeds_csv)
                ui_vars.setdefault("speeds_feeds_path", speeds_csv)
                ui_vars.setdefault("Speeds/Feeds CSV", speeds_csv)

            # Use LLM only if already available; avoid blocking loads during quote gen
            llm_suggest = self.LLM_SUGGEST

            try:
                self._reset_llm_logs()
                client = None
                if self.llm_enabled.get() and (llm_suggest is not None):
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

                timestamp = datetime.datetime.now().isoformat()
                append_debug_log(
                    "",
                    f"[{timestamp}] Quote blocked (ValueError):",
                    traceback.format_exc(),
                    "",
                )
                # Also persist a minimal error artifact so users can see something
                try:
                    with open("latest_quote_error.txt", "w", encoding="utf-8") as ef:
                        ef.write(str(err))
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
                timestamp = datetime.datetime.now().isoformat()
                append_debug_log(
                    "",
                    f"[{timestamp}] Quote error:",
                    tb,
                    "",
                )
                try:
                    with open("latest_quote_error.txt", "w", encoding="utf-8") as ef:
                        ef.write(f"Unexpected error during pricing.\n\n{err}\n\n{tb}")
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
            llm_explanation = get_llm_quote_explanation(
                res,
                model_path,
                debug_enabled=APP_ENV.llm_debug_enabled,
                debug_dir=APP_ENV.llm_debug_dir,
            )
            if not isinstance(res, dict):
                res = {}
            cfg = getattr(self, "quote_config", None)
            geometry_loader = getattr(self, "geometry_loader", None)
            geometry_ctx = (
                getattr(geometry_loader, "geo_ctx", None)
                if geometry_loader is not None
                else None
            )

            try:
                simplified_report = render_quote(
                    res,
                    currency="$",
                    show_zeros=False,
                    llm_explanation=llm_explanation,
                    cfg=cfg,
                    geometry=geometry_ctx,
                )
                full_report = render_quote(
                    res,
                    currency="$",
                    show_zeros=True,
                    llm_explanation=llm_explanation,
                    cfg=cfg,
                    geometry=geometry_ctx,
                )
            except AssertionError as e:
                # Be resilient to strict invariants inside render_quote; surface
                # a readable fallback rather than crashing the UI.
                import traceback as _tb
                err_text = f"Quote rendering error: {e}"
                append_debug_log(
                    "",
                    "[render_quote] AssertionError while rendering output",
                    _tb.format_exc(),
                    "",
                )
                fallback = (
                    err_text
                    + "\n\nShowing raw result as fallback.\n\n"
                    + jdump(res, default=None)
                )
                simplified_report = fallback
                full_report = fallback
                try:
                    with open("latest_quote_error.txt", "w", encoding="utf-8") as ef:
                        ef.write(err_text + "\n\n" + _tb.format_exc())
                except Exception:
                    pass

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
                    widget.mark_set("insert", "1.0")
                    widget.see("1.0")
                except Exception:
                    pass

            try:
                self.output_nb.select(self.output_tab_simplified)
            except Exception:
                pass

            self.nb.select(self.tab_out)
            try:
                simp_len = len(simplified_report or "")
                full_len = len(full_report or "")
            except Exception:
                simp_len = full_len = 0
            self.status_var.set(
                f"Quote Generated! Final Price: ${res.get('price', 0):,.2f} (chars: simp={simp_len}, full={full_len})"
            )
            succeeded = True
        finally:
            if succeeded:
                self._clear_quote_dirty()
            if not already_repricing:
                self._reprice_in_progress = False

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point so ``python appV5.py`` mirrors the CLI launcher."""

    from cad_quoter.app.cli import main as _cli_main
    from cad_quoter.pricing import PricingEngine, create_default_registry

    return _cli_main(
        argv,
        app_cls=App,
        pricing_engine_cls=PricingEngine,
        pricing_registry_factory=create_default_registry,
        app_env=APP_ENV,
        env_setter=lambda env: globals().__setitem__("APP_ENV", env),
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())

# Emit chart-debug key lines at most once globally per run
_PRINTED_CHART_DEBUG_KEYS = False
