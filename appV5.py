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

import os, sys, logging

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="[%(levelname)s] %(message)s",
)


def dbg(lines, msg: str):
    """Log to terminal, and also mirror into the quote if `lines` is provided."""

    logging.debug(msg)
    try:
        if lines is not None:
            lines.append(f"[DEBUG] {msg}")
    except Exception:
        pass


import typing
from io import TextIOWrapper
from pathlib import Path

_stdout = sys.stdout
if isinstance(_stdout, TextIOWrapper):
    try:
        _stdout.reconfigure(encoding="utf-8")  # py3.7+
    except Exception:
        pass

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_SRC = _SCRIPT_DIR / "cad_quoter_pkg" / "src"
if _PKG_SRC.is_dir():
    _pkg_src_str = str(_PKG_SRC)
    if _pkg_src_str not in sys.path:
        sys.path.insert(0, _pkg_src_str)

from cad_quoter.app.quote_doc import (
    build_quote_header_lines,
    _sanitize_render_text,
)



import copy
import csv
import json
import math
import re
import time
from functools import cmp_to_key, lru_cache
from typing import Any, Mapping, MutableMapping, Sequence, TYPE_CHECKING, Protocol, TypeGuard
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

from cad_quoter.app._value_utils import (
    _format_value,
)
from cad_quoter.app.op_parser import (
    LETTER_DRILLS,
    _CB_DIA_RE,
    _DRILL_THRU,
    _JIG_RE_TXT,
    _LETTER_RE,
    _SIZE_INCH_RE,
    _SPOT_RE_TXT,
    _TAP_RE,
    _X_DEPTH_RE,
    _parse_ops_and_claims as _shared_parse_ops_and_claims,
    _parse_qty as _shared_parse_qty,
    _side as _shared_side,
)
from typing import Any as _AnyForCoerce

try:
    from cad_quoter.geometry.dxf_text import (
        HOLE_TOKENS as _DXF_HOLE_TOKENS,
        harvest_text_lines as _harvest_dxf_text_lines,
    )
except Exception:
    _DXF_HOLE_TOKENS = re.compile(
        r"(?:\bTAP\b|C[’']?\s*BORE|CBORE|COUNTER\s*BORE|"
        r"C[’']?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|"
        r"\bJIG\s*GRIND\b|\bDRILL\s+THRU\b|\bN\.?P\.?T\.?\b)",
        re.IGNORECASE,
    )
    _harvest_dxf_text_lines = None


def _coerce_positive_float(value: _AnyForCoerce) -> float | None:
    """Best-effort positive finite float coercion without importing utils early.

    Avoids a circular import between utils.numeric and domain_models.values at
    app startup by providing a local fallback.
    """

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


_MM_DIM_TOKEN = re.compile(
    r"(?:Ø|⌀|DIA|REF)?\s*((?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?))\s*(?:MM|MILLIM(?:E|E)T(?:E|)RS?)",
    re.IGNORECASE,
)

_COUNTERDRILL_RE = re.compile(
    r"\b(?:C[’']\s*DRILL|C\s*DRILL|COUNTER\s*DRILL|COUNTERDRILL)\b",
    re.IGNORECASE,
)
_CENTER_OR_SPOT_RE = re.compile(
    r"\b(CENTER\s*DRILL|SPOT\s*DRILL|SPOT)\b",
    re.IGNORECASE,
)
_JIG_RE = re.compile(r"\bJIG\s*GRIND\b", re.IGNORECASE)

_DRILL_REMOVAL_MINUTES_MIN = 0.0
_DRILL_REMOVAL_MINUTES_MAX = 600.0


def _seed_drill_bins_from_geo(geo: dict) -> dict[float, int]:
    """
    Robustly build {diam_in: qty} from GEO. Handles multiple shapes/keys and
    gracefully falls back to raw hole lists.
    """

    if not isinstance(geo, dict):
        return {}

    # Preferred “families” maps people keep around in different names:
    candidates = [
        "hole_diam_families_geom_in",
        "hole_diam_families_in",
        "hole_diam_families_geom",
        "hole_diam_families",
    ]
    out: dict[float, int] = {}

    # 1) Direct family maps
    for key in candidates:
        fam = geo.get(key)
        if isinstance(fam, dict) and fam:
            for k, v in fam.items():
                try:
                    d = float(str(k).replace('"', '').strip())
                    q = int(v) if v is not None else 0
                    if q > 0:
                        d = round(d, 4)
                        out[d] = out.get(d, 0) + q
                except Exception:
                    continue
            if out:
                return out  # done

    # 2) Rebuild from raw hole lists (in or mm)
    holes_in = geo.get("hole_diams_in") or geo.get("hole_diams_geom_in")
    holes_mm = geo.get("hole_diams_mm") or geo.get("hole_diams_geom_mm")

    def _acc_from_list(seq, mm=False):
        nonlocal out
        if not isinstance(seq, (list, tuple)):
            return
        for x in seq:
            try:
                d = float(x)
                if mm:
                    d /= 25.4
                # snap to 0.001” bins to avoid float scatter
                d = round(d, 3)
                out[d] = out.get(d, 0) + 1
            except Exception:
                continue

    if holes_in:
        _acc_from_list(holes_in, mm=False)
    if not out and holes_mm:
        _acc_from_list(holes_mm, mm=True)

    return out


def _parse_dim_to_mm(value: Any) -> float | None:
    """Parse a dimension string containing millimeter units to a float value."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            mm_val = float(value)
        except Exception:
            return None
        return mm_val if math.isfinite(mm_val) and mm_val > 0 else None

    text = str(value).strip()
    if not text:
        return None

    match = _MM_DIM_TOKEN.search(text)
    if not match:
        return None

    token = match.group(1).replace(" ", "")
    try:
        if "/" in token:
            mm_val = float(Fraction(token))
        else:
            mm_val = float(token)
    except Exception:
        return None

    return mm_val if math.isfinite(mm_val) and mm_val > 0 else None


def _sanitize_drill_removal_minutes(minutes_value: Any) -> float:
    """Clamp drill removal minutes to sane bounds before storing in extras."""

    try:
        minutes = float(minutes_value)
    except Exception:
        logging.error(
            f"[unit] removal DRILL minutes insane; dropping. raw={minutes_value}"
        )
        return 0.0

    if not math.isfinite(minutes) or not (
        _DRILL_REMOVAL_MINUTES_MIN <= minutes <= _DRILL_REMOVAL_MINUTES_MAX
    ):
        logging.error(
            f"[unit] removal DRILL minutes insane; dropping. raw={minutes}"
        )
        return 0.0

    return minutes


def _sum_count_values(candidate: Any) -> int:
    """Best-effort sum of numeric-ish values in mappings or sequences."""

    if isinstance(candidate, (_MappingABC, dict)):
        values = candidate.values()  # type: ignore[assignment]
    elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        values = candidate
    else:
        return 0

    total = 0
    for value in values:
        try:
            total += int(round(float(value)))
        except Exception:
            try:
                total += int(value)  # type: ignore[arg-type]
            except Exception:
                continue
    return total


from cad_quoter.app.chart_lines import (
    collect_chart_lines_context as _collect_chart_lines_context,
)
from ops_audit import audit_operations
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
    build_ops_rows_from_lines_fallback as _build_ops_rows_from_lines_fallback,
    summarize_hole_chart_agreement,
    update_geo_ops_summary_from_hole_rows,
    _aggregate_hole_entries,
    _classify_thread_spec,
    _dedupe_hole_entries,
    _DIA_TOKEN,
    _major_diameter_from_thread,
    _normalize_hole_text,
    _parse_hole_line,
    _parse_ref_to_inch,
    _SPOT_TOKENS,
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
from cad_quoter.app.variables import (
    CORE_COLS,
    _coerce_core_types,
    _load_master_variables,
    find_variables_near,
    read_variables_file,
    sanitize_vars_df,
)
from cad_quoter.material_density import LB_PER_IN3_PER_GCC as _LB_PER_IN3_PER_GCC
from cad_quoter.utils.machining import (
    _first_numeric_or_none,
    _fmt_rng,
    _ipm_from_rpm_ipr,
    _lookup_sfm_ipr,
    _parse_thread_major_in,
    _parse_tpi,
    _rpm_from_sfm_diam,
)
if TYPE_CHECKING:
    from cad_quoter_pkg.src.cad_quoter.resources import default_app_settings_json
else:
    from cad_quoter.resources import (
        default_app_settings_json,
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
    _apply_drilling_meta_fallback,
    _ensure_geo_context_fields,
    _iter_geo_contexts as _iter_geo_dicts_for_context,
    _should_include_outsourced_pass,
)
from cad_quoter.utils.scrap import (
    _holes_removed_mass_g,
    build_drill_groups_from_geometry,
)
if TYPE_CHECKING:
    from cad_quoter_pkg.src.cad_quoter.utils.render_utils import (
        QuoteDocRecorder as _QuoteDocRecorder,
        fmt_hours as _fmt_hours,
        fmt_money as _fmt_money,
        format_currency as _format_currency,
        format_dimension as _format_dimension,
        format_hours as _format_hours,
        format_hours_with_rate as _format_hours_with_rate,
        format_percent as _format_percent,
        format_weight_lb_decimal as _format_weight_lb_decimal,
        format_weight_lb_oz as _format_weight_lb_oz,
        render_quote_doc as _render_quote_doc,
    )
    from cad_quoter_pkg.src.cad_quoter.pricing import load_backup_prices_csv
else:
    from cad_quoter.utils.render_utils import (
        fmt_hours as _fmt_hours,
        fmt_money as _fmt_money,
        format_currency as _format_currency,
        format_dimension as _format_dimension,
        format_hours as _format_hours,
        format_hours_with_rate as _format_hours_with_rate,
        format_percent as _format_percent,
        format_weight_lb_decimal as _format_weight_lb_decimal,
        format_weight_lb_oz as _format_weight_lb_oz,
        QuoteDocRecorder as _QuoteDocRecorder,
        render_quote_doc as _render_quote_doc,
    )
    from cad_quoter.pricing import load_backup_prices_csv

fmt_hours = _fmt_hours
fmt_money = _fmt_money
format_currency = _format_currency
format_dimension = _format_dimension
format_hours = _format_hours
format_hours_with_rate = _format_hours_with_rate
format_percent = _format_percent
format_weight_lb_decimal = _format_weight_lb_decimal
format_weight_lb_oz = _format_weight_lb_oz
QuoteDocRecorder = _QuoteDocRecorder
render_quote_doc = _render_quote_doc
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
if typing.TYPE_CHECKING:
    from cad_quoter.pricing.process_view import (
        _ProcessCostTableRecorder as _ProcessCostTableRecorderType,
        _ProcessRowRecord as _ProcessRowRecordType,
        _merge_process_meta as _merge_process_meta_fn,
        _fold_process_meta as _fold_process_meta_fn,
        _fold_applied_process as _fold_applied_process_fn,
        _lookup_process_meta as _lookup_process_meta_fn,
    )
else:
    _ProcessCostTableRecorderType = typing.Any  # type: ignore[assignment]
    _ProcessRowRecordType = typing.Any  # type: ignore[assignment]
    _merge_process_meta_fn = typing.Callable[..., typing.Any]
    _fold_process_meta_fn = typing.Callable[..., typing.Any]
    _fold_applied_process_fn = typing.Callable[..., typing.Any]
    _lookup_process_meta_fn = typing.Callable[..., typing.Any]


@lru_cache(maxsize=1)
def _load_process_view_module():
    import cad_quoter.pricing.process_view as _process_view_module

    return _process_view_module


def _ProcessCostTableRecorder(*args, **kwargs):
    return _load_process_view_module()._ProcessCostTableRecorder(*args, **kwargs)


def _ProcessRowRecord(*args, **kwargs):
    return _load_process_view_module()._ProcessRowRecord(*args, **kwargs)


def _merge_process_meta(*args, **kwargs):
    return _load_process_view_module()._merge_process_meta(*args, **kwargs)


def _fold_process_meta(*args, **kwargs):
    return _load_process_view_module()._fold_process_meta(*args, **kwargs)


def _fold_applied_process(*args, **kwargs):
    return _load_process_view_module()._fold_applied_process(*args, **kwargs)


def _lookup_process_meta(*args, **kwargs):
    return _load_process_view_module()._lookup_process_meta(*args, **kwargs)


# ==== BUCKET SEEDING (single source of truth) ===========================
def _minutes_to_hours(m: Any) -> float:
    return _as_float(m, 0.0) / 60.0


def minutes_to_hours(m: Any) -> float:
    return _minutes_to_hours(m)


def _set_bucket_minutes_cost(
    bvo: MutableMapping[str, Any] | Mapping[str, Any] | None,
    key: str,
    minutes: float,
    machine_rate: float,
    labor_rate: float,
) -> None:
    minutes_val = _as_float(minutes, 0.0)
    if not (0.0 <= minutes_val <= 10_000.0):
        logging.warning(f"[bucket] ignoring {key} minutes out of range: {minutes}")
        minutes_val = 0.0

    machine_rate_val = _as_float(machine_rate, 0.0)
    labor_rate_val = _as_float(labor_rate, 0.0)

    buckets_obj: MutableMapping[str, Any] | None = None
    if isinstance(bvo, dict):
        buckets_obj = bvo.setdefault("buckets", {})
    elif isinstance(bvo, _MutableMappingABC):
        buckets_obj = typing.cast(MutableMapping[str, Any], bvo.setdefault("buckets", {}))
    else:
        return

    if buckets_obj is None:
        return

    machine_cost = (minutes_val / 60.0) * machine_rate_val
    labor_cost = (minutes_val / 60.0) * labor_rate_val

    buckets_obj[key] = {
        "minutes": minutes_val,
        "machine$": round(machine_cost, 2),
        "labor$": round(labor_cost, 2),
        "total$": round(machine_cost + labor_cost, 2),
    }


def _normalize_buckets(bucket_view_obj: MutableMapping[str, Any] | Mapping[str, Any] | None) -> None:
    if not isinstance(bucket_view_obj, (_MutableMappingABC, dict)):
        return

    alias = {
        "programming_amortized": "programming",
        "spotdrill": "spot_drill",
        "spot-drill": "spot_drill",
        "jiggrind": "jig_grind",
        "jig-grind": "jig_grind",
    }

    try:
        buckets_obj = bucket_view_obj.get("buckets")
    except Exception:
        buckets_obj = None

    if isinstance(buckets_obj, dict):
        source_items = buckets_obj.items()
    elif isinstance(buckets_obj, _MappingABC):
        source_items = buckets_obj.items()
    else:
        source_items = ()

    norm: dict[str, dict[str, float]] = {}
    for raw_key, entry in source_items:
        try:
            key = str(raw_key or "")
        except Exception:
            key = ""
        if not key or not isinstance(entry, _MappingABC):
            continue
        nk = alias.get(key, key)
        dst = norm.setdefault(
            nk,
            {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0, "total$": 0.0},
        )
        dst["minutes"] += _as_float(entry.get("minutes"), 0.0)
        dst["machine$"] += _as_float(entry.get("machine$"), 0.0)
        dst["labor$"] += _as_float(entry.get("labor$"), 0.0)
        dst["total$"] = round(dst["machine$"] + dst["labor$"], 2)

    bucket_view_obj["buckets"] = norm

    try:
        buckets_snapshot = (
            bucket_view_obj.get("buckets")
            if isinstance(bucket_view_obj, (_MappingABC, dict))
            else None
        )
    except Exception:
        buckets_snapshot = None
    logging.debug(f"[buckets-final] {buckets_snapshot or {}}")


def _get_chart_lines_for_ops(
    breakdown: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    *,
    ctx: Mapping[str, Any] | None = None,
    ctx_a: Mapping[str, Any] | None = None,
    ctx_b: Mapping[str, Any] | None = None,
) -> list[str]:
    def _geo(m):
        if not isinstance(m, _MappingABC):
            return {}
        return m.get("geo_context") or m.get("geo") or m.get("geom") or {}

    sources: tuple[Mapping[str, Any] | None, ...] = (
        ctx,
        ctx_a,
        ctx_b,
        breakdown,
        result,
    )

    for src in sources:
        g = _geo(src)
        if isinstance(g, _MappingABC):
            cl = g.get("chart_lines")
            if isinstance(cl, list) and cl:
                return [str(x) for x in cl if isinstance(x, (str, bytes))]
        if isinstance(src, _MappingABC):
            cl = src.get("chart_lines")
            if isinstance(cl, list) and cl:
                return [str(x) for x in cl if isinstance(x, (str, bytes))]
    return []


def _bucket_add_minutes(
    breakdown_mutable: MutableMapping[str, Any],
    key: str,
    minutes: float,
    rates: Mapping[str, Any] | None,
) -> None:
    if minutes <= 0:
        return
    # mirror drilling's bucket insertion
    bucket_view_obj = breakdown_mutable.setdefault("bucket_view", {})
    buckets_obj = bucket_view_obj.setdefault("buckets", {})
    mode = _bucket_cost_mode(key)
    rate = _lookup_bucket_rate(key, rates) or _lookup_bucket_rate("machine", rates) or 0.0
    labor_rate = _lookup_bucket_rate("labor", rates) or 0.0
    if mode == "labor":
        labor_cost = round((minutes / 60.0) * (labor_rate if labor_rate > 0 else rate), 2)
        machine_cost = 0.0
    else:
        machine_cost = round((minutes / 60.0) * rate, 2)
        labor_cost = 0.0
    total_cost = round(machine_cost + labor_cost, 2)
    entry = buckets_obj.setdefault(
        key, {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0}
    )
    entry["minutes"] = round(float(minutes), 2)
    entry["labor$"] = labor_cost
    entry["machine$"] = machine_cost
    entry["total$"] = total_cost
    order = bucket_view_obj.setdefault("order", [])
    if key not in order:
        order.append(key)


def _build_ops_cards_from_chart_lines(
    *,
    breakdown: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    rates: Mapping[str, Any] | None,
    breakdown_mutable: MutableMapping[str, Any] | None,
    ctx=None,
    ctx_a=None,
    ctx_b=None,
) -> list[str]:
    """Return extra MATERIAL REMOVAL cards for Counterbore / Spot / Jig."""

    lines: list[str] = []
    # Pull chart lines exactly like your tapping fallback does
    chart_lines: list[str] = []
    geo_map = ((result or {}).get("geo") if isinstance(result, _MappingABC) else None) \
              or ((breakdown or {}).get("geo") if isinstance(breakdown, _MappingABC) else None) \
              or {}
    try:
        chart_lines = _collect_chart_lines_context(ctx, geo_map, ctx_a, ctx_b) or []
    except Exception:
        chart_lines = []
    # If the collector returns empty, fall back to any chart_lines on result/breakdown
    if not chart_lines:
        chart_lines = _get_chart_lines_for_ops(breakdown, result) or []

    # Also expose rows if we’ve already built them (ops_summary.rows)
    ops_rows = []
    try:
        ops_rows = (((geo_map or {}).get("ops_summary") or {}).get("rows") or [])
    except Exception:
        ops_rows = []
    if not isinstance(ops_rows, list):
        ops_rows = []

    cleaned_chart_lines: list[str] = []
    for raw in chart_lines or []:
        cleaned = _clean_mtext(str(raw or ""))
        if cleaned:
            cleaned_chart_lines.append(cleaned)
    joined_chart = _join_wrapped_chart_lines(cleaned_chart_lines)
    # Early debug
    lines.append(
        f"[DEBUG] ops_cards_inputs: chart_lines={len(joined_chart)} rows={len(ops_rows)}"
    )
    if not joined_chart and not ops_rows:
        return lines
    chart_claims = _parse_ops_and_claims(joined_chart)

    cb_groups: dict[tuple[float | None, str, float | None], int] = dict(chart_claims.get("cb_groups") or {})
    spot_qty = int(chart_claims.get("spot") or 0)
    jig_qty = int(chart_claims.get("jig") or 0)

    row_lines: list[str] = []
    for entry in ops_rows:
        if not isinstance(entry, _MappingABC):
            continue
        desc = str(entry.get("desc") or "")
        if not desc.strip():
            continue
        try:
            qty_val = int(entry.get("qty") or 0)
        except Exception:
            qty_val = 0
        prefix = f"({qty_val}) " if qty_val > 0 else ""
        row_lines.append(prefix + desc)

    if row_lines:
        row_claims = _parse_ops_and_claims(row_lines)
        if not cb_groups and row_claims.get("cb_groups"):
            cb_groups = dict(row_claims["cb_groups"])

        spot_from_rows = int(row_claims.get("spot") or 0)
        jig_from_rows = int(row_claims.get("jig") or 0)

        if spot_qty <= 0:
            spot_qty = spot_from_rows
        else:
            spot_qty = max(spot_qty, spot_from_rows)

        if jig_qty <= 0:
            jig_qty = jig_from_rows
        else:
            jig_qty = max(jig_qty, jig_from_rows)

    # --- Emit COUNTERBORE card ----------------------------------------------
    if cb_groups:
        total_cb = sum(cb_groups.values())
        front_cb = sum(q for (d, side, dep), q in cb_groups.items() if side == "FRONT")
        back_cb = sum(q for (d, side, dep), q in cb_groups.items() if side == "BACK")
        try:
            ebo = breakdown_mutable.setdefault("extra_bucket_ops", {})
            total_cb = sum(cb_groups.values())
            if total_cb > 0:
                if front_cb > 0:
                    ebo.setdefault("counterbore", []).append(
                        {"name": "Counterbore", "qty": int(front_cb), "side": "front"}
                    )
                if back_cb > 0:
                    ebo.setdefault("counterbore", []).append(
                        {"name": "Counterbore", "qty": int(back_cb), "side": "back"}
                    )
            if spot_qty > 0:
                ebo.setdefault("spot", []).append(
                    {"name": "Spot drill", "qty": int(spot_qty), "side": "front"}
                )
            if jig_qty > 0:
                ebo.setdefault("jig-grind", []).append(
                    {"name": "Jig-grind", "qty": int(jig_qty), "side": None}
                )
        except Exception:
            pass

        lines += [
            "MATERIAL REMOVAL – COUNTERBORE",
            "=" * 64,
            "Inputs",
            f"  Ops ............... Counterbore (front + back)",
            f"  Counterbores ...... {total_cb} total  → {front_cb} front, {back_cb} back",
            "",
            "TIME PER HOLE – C’BORE GROUPS",
            "-" * 66,
        ]
        cb_minutes = 0.0
        per = float(CBORE_MIN_PER_SIDE_MIN or 0.15)
        for (dia, side, depth), qty in sorted(
            cb_groups.items(),
            key=lambda k: ((k[0][0] or 0.0), k[0][1], (k[0][2] or 0.0)),
        ):
            dia_txt = "—" if dia is None else f'Ø{dia:.4f}"'
            dep_txt = "—" if depth is None else f'{depth:.2f}"'
            t_group = qty * per
            cb_minutes += t_group
            lines.append(
                f"{dia_txt} × {qty}  ({side}) | depth {dep_txt} | t/hole {per:.2f} min | group {qty}×{per:.2f} = {t_group:.2f} min"
            )
        lines.append("")
        if isinstance(breakdown_mutable, _MutableMappingABC):
            _bucket_add_minutes(breakdown_mutable, "counterbore", cb_minutes, rates)

    # --- Emit SPOT & JIG cards ----------------------------------------------
    if spot_qty > 0:
        per_spot = 0.05
        t_group = spot_qty * per_spot
        lines += [
            "MATERIAL REMOVAL – SPOT (CENTER DRILL)",
            "=" * 64,
            "TIME PER HOLE – SPOT GROUPS",
            "-" * 66,
            f"Spot drill × {spot_qty} | t/hole {per_spot:.2f} min | group {spot_qty}×{per_spot:.2f} = {t_group:.2f} min",
            "",
        ]
        if isinstance(breakdown_mutable, _MutableMappingABC):
            _bucket_add_minutes(breakdown_mutable, "drilling", t_group, rates)

    if jig_qty > 0:
        per_jig = float(globals().get("JIG_GRIND_MIN_PER_FEATURE") or 0.75)  # minutes/feature
        t_group = jig_qty * per_jig
        lines += [
            "MATERIAL REMOVAL – JIG GRIND",
            "=" * 64,
            "TIME PER FEATURE",
            "-" * 66,
            f"Jig grind × {jig_qty} | t/feat {per_jig:.2f} min | group {jig_qty}×{per_jig:.2f} = {t_group:.2f} min",
            "",
        ]
        if isinstance(breakdown_mutable, _MutableMappingABC):
            _bucket_add_minutes(breakdown_mutable, "grinding", t_group, rates)

    return lines



# --- Inline ops-cards builder from chart_lines (fallback to rows) ------------
import re, math
_JOIN_QTY_RE = re.compile(r"^\s*\((\d+)\)\s*")   # e.g. "(4) ..."

# --- MTEXT cleaning ---------------------------------------------------------
_MT_ALIGN_RE = re.compile(r"\\A\d;")
_MT_BREAK_RE = re.compile(r"\\P", re.I)
_MT_SYMS = {"%%C": "Ø", "%%c": "Ø", "%%D": "°", "%%d": "°", "%%P": "±", "%%p": "±"}


def _clean_mtext(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for k, v in _MT_SYMS.items():
        s = s.replace(k, v)
    s = _MT_ALIGN_RE.sub("", s)
    s = _MT_BREAK_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- Row start tokens (incl Ø and %%C) -------------------------------------
_JOIN_START_TOKENS = re.compile(
    r"(?:^\s*\(\d+\)\s*)"                   # "(n) ..."
    r"|(?:\bTAP\b|N\.?P\.?T\.?)"            # TAP / NPT
    r"|(?:C[’']?\s*BORE|CBORE|COUNTER\s*BORE)"
    r"|(?:[Ø⌀\u00D8]|%%[Cc])"                  # Ø/%%C
    r"|(?:C[’']?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL\b)",
    re.I,
)


def _join_wrapped_chart_lines(chart_lines: list[str]) -> list[str]:
    if not chart_lines:
        return []
    out: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            out.append(re.sub(r"\s+", " ", buf).strip())
        buf = ""

    for raw in chart_lines:
        s = _clean_mtext(str(raw or ""))
        if not s:
            continue
        if _JOIN_START_TOKENS.search(s):
            # ✨ glue rule: if this line is the NPT continuation and the buffer ends with DRILL THRU,
            # append instead of starting a new row. This keeps the pilot drill call-out together
            # with the NPT note so downstream logic sees them as a single feature description.
            if re.search(r"\bN\.?P\.?T\.?\b", s, re.I) and re.search(r"\bDRILL\s+THRU\b", buf or "", re.I):
                buf += " " + s
                continue
            flush()
            buf = s
        else:
            buf += " " + s
    flush()
    return out
_parse_qty = _shared_parse_qty
_side = _shared_side


_TAP_PILOT_IN: dict[str, float] = {
    "#10-32": 0.1590,  # #21
    "5/8-11": 0.5312,  # 17/32
    "5/16-24": 0.2720,  # I
    "5/16-18": 0.2610,  # G
    "3/8-24": 0.3320,  # Q
}

_NPT_PILOT_IN: dict[str, float] = {
    "1/8": 0.3390,  # R
}


def _collect_pilot_claims_from_rows(geo: Mapping[str, Any] | None) -> list[float]:
    vals: list[float] = []
    if not isinstance(geo, (_MappingABC, dict)):
        return vals

    ops_summary = geo.get("ops_summary") if isinstance(geo, (_MappingABC, dict)) else None
    rows: Sequence[Mapping[str, Any]] | None = None
    if isinstance(ops_summary, (_MappingABC, dict)):
        candidate = ops_summary.get("rows")
        if isinstance(candidate, Sequence):
            rows = typing.cast(Sequence[Mapping[str, Any]], candidate)

    if not rows:
        return vals

    for row in rows:
        if not isinstance(row, (_MappingABC, dict)):
            continue
        desc_raw = row.get("desc") or row.get("name") or ""
        desc = str(desc_raw or "").upper()
        if not desc:
            continue

        qty_raw = row.get("qty", 1)
        qty = 1
        if isinstance(qty_raw, (int, float)):
            qty = int(qty_raw)
        elif isinstance(qty_raw, str):
            match = re.search(r"(\d+)", qty_raw)
            if match:
                try:
                    qty = int(match.group(1))
                except Exception:
                    qty = 1
        if qty <= 0:
            qty = 1

        desc_compact = desc.replace(" ", "")
        # UN taps
        for spec, dia in _TAP_PILOT_IN.items():
            if spec.replace(" ", "") in desc_compact:
                vals.extend([float(dia)] * qty)

        # NPT taps
        if "NPT" in desc:
            match = re.search(r"((?:\d+/\d+)|(?:\d+\.\d+)|(?:\d+))\s*-\s*27\s*NPT|\b1/8\b", desc)
            if match:
                vals.extend([_NPT_PILOT_IN["1/8"]] * qty)

        # explicit decimals, e.g. "Ø0.201 DRILL THRU" or ".339 THRU"
        match_decimal = re.search(r"(\d*\.\d+)\s*(?:DRILL\s*)?THRU", desc)
        if match_decimal:
            try:
                vals.extend([float(match_decimal.group(1))] * qty)
            except Exception:
                pass

    return vals


def _record_drill_claims(
    breakdown_mutable: MutableMapping[str, Any] | None,
    claimed: Iterable[float],
) -> None:
    if not claimed or breakdown_mutable is None:
        return
    if not isinstance(breakdown_mutable, (_MutableMappingABC, dict)):
        return
    try:
        drilling_meta = breakdown_mutable.setdefault("drilling_meta", {})
    except Exception:
        return
    if not isinstance(drilling_meta, (_MutableMappingABC, dict)):
        try:
            drilling_meta = dict(drilling_meta)  # type: ignore[arg-type]
        except Exception:
            return
        try:
            breakdown_mutable["drilling_meta"] = drilling_meta  # type: ignore[index]
        except Exception:
            return
    if drilling_meta.get("claimed_pilot_diams"):
        return
    cleaned: list[float] = []
    for value in claimed:
        try:
            num = float(value)
        except Exception:
            continue
        if not math.isfinite(num):
            continue
        cleaned.append(num)
    if not cleaned:
        return
    try:
        drilling_meta["claimed_pilot_diams"] = list(cleaned)
    except Exception:
        return
    try:
        counts = Counter(round(val, 4) for val in cleaned)
        drilling_meta["claimed_pilot_counts"] = {
            f"{key:.4f}": int(count)
            for key, count in counts.items()
        }
    except Exception:
        pass


def _count_counterdrill(lines_joined: Sequence[str] | None) -> int:
    total = 0
    if not lines_joined:
        return 0
    for raw in lines_joined:
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        upper = s.upper()
        if _CENTER_OR_SPOT_RE.search(upper):
            continue
        if _COUNTERDRILL_RE.search(upper):
            prefix = re.match(r"\s*\((\d+)\)", raw)
            if prefix:
                try:
                    total += int(prefix.group(1))
                    continue
                except Exception:
                    pass
            total += 1
    return total


def _count_jig(lines_joined: Sequence[str] | None) -> int:
    total = 0
    if not lines_joined:
        return 0
    for raw in lines_joined:
        if raw is None:
            continue
        text = str(raw)
        if not text.strip():
            continue
        if _JIG_RE.search(text):
            prefix = re.match(r"\s*\((\d+)\)", text)
            if prefix:
                try:
                    total += int(prefix.group(1))
                    continue
                except Exception:
                    pass
            total += 1
    return total


def _parse_ops_and_claims(joined_lines: Sequence[str] | None) -> dict[str, Any]:
    return _shared_parse_ops_and_claims(joined_lines, cleaner=_clean_mtext)



def _adjust_drill_counts(
    counts_by_diam_raw: dict[float, int],
    ops_claims: dict,
    _logger=None,
) -> dict[float, int]:
    """Return adjusted drill groups: subtract pilots, counterbores, and ignore large bores."""

    if not counts_by_diam_raw:
        return {}

    counts = {
        round(float(d), 4): max(0, int(q))
        for d, q in counts_by_diam_raw.items()
        if int(q) > 0
    }

    if not counts:
        return {}

    bins = sorted(counts.keys())

    def _nearest(val: float) -> float | None:
        return min(bins, key=lambda b: abs(b - val)) if bins else None

    # (a) subtract pilot drills claimed by TAP/NPT/explicit DRILL THRU, mapped to nearest geom bin
    raw_claims = list(ops_claims.get("claimed_pilot_diams") or [])
    claim_ctr: Counter[float] = Counter()
    for value in raw_claims:
        try:
            num = float(value)
        except Exception:
            continue
        if not 0.05 <= num <= 3.0:
            continue
        claim_ctr[round(num, 4)] += 1

    for val, qty in claim_ctr.items():
        tgt = _nearest(val)
        if tgt is not None and abs(tgt - val) <= 0.015:
            counts[tgt] = max(0, counts[tgt] - max(0, int(qty)))

    # (b) subtract counterbore face diameters (don’t double-count big faces as drills)
    cb_face_ctr: Counter[float] = Counter()
    for (diam, _side, _depth), qty in (ops_claims.get("cb_groups") or {}).items():
        if diam is None:
            continue
        try:
            num = float(diam)
        except Exception:
            continue
        if not 0.05 <= num <= 3.0:
            continue
        cb_face_ctr[round(num, 4)] += max(0, int(qty))

    for face_dia, qty in cb_face_ctr.items():
        tgt = _nearest(face_dia)
        if tgt is not None and abs(tgt - face_dia) <= 0.02:
            counts[tgt] = max(0, counts[tgt] - max(0, int(qty)))

    # (c) treat very large diameters as bores/pockets (not drills)
    for dia in list(counts.keys()):
        if dia >= 1.0:
            counts[dia] = 0

    return counts


def _apply_ops_audit_counts(
    ops_counts: MutableMapping[str, int],
    *,
    drill_actions: int,
    ops_claims: Mapping[str, Any] | None,
) -> MutableMapping[str, int]:
    """Merge chart-derived action counts into ``ops_counts`` for audit display."""

    def _coerce_int(value: Any) -> int:
        try:
            number = float(value)
        except Exception:
            return 0
        if not math.isfinite(number):
            return 0
        try:
            return int(round(number))
        except Exception:
            return 0

    drill_total = max(0, _coerce_int(drill_actions))
    if drill_total > _coerce_int(ops_counts.get("drills")):
        ops_counts["drills"] = drill_total

    claims = ops_claims if isinstance(ops_claims, Mapping) else None

    tap_total = 0
    cb_total = 0
    cb_front = 0
    cb_back = 0
    spot_total = 0
    counterdrill_total = 0
    jig_total = 0

    if claims:
        tap_total = max(0, _coerce_int(claims.get("tap")))

        cb_groups = claims.get("cb_groups")
        if isinstance(cb_groups, Mapping):
            for key, qty in cb_groups.items():
                qty_int = max(0, _coerce_int(qty))
                cb_total += qty_int
                side = ""
                if isinstance(key, tuple) and len(key) >= 2:
                    try:
                        side = str(key[1] or "")
                    except Exception:
                        side = ""
                side_norm = side.upper()
                if side_norm == "FRONT":
                    cb_front += qty_int
                elif side_norm == "BACK":
                    cb_back += qty_int

        spot_total = max(0, _coerce_int(claims.get("spot")))
        counterdrill_total = max(0, _coerce_int(claims.get("counterdrill")))
        jig_total = max(0, _coerce_int(claims.get("jig")))

    if tap_total > _coerce_int(ops_counts.get("taps_total")):
        ops_counts["taps_total"] = tap_total

    if cb_total > _coerce_int(ops_counts.get("counterbores_total")):
        ops_counts["counterbores_total"] = cb_total
        if cb_front > _coerce_int(ops_counts.get("counterbores_front")):
            ops_counts["counterbores_front"] = cb_front
        if cb_back > _coerce_int(ops_counts.get("counterbores_back")):
            ops_counts["counterbores_back"] = cb_back

    if spot_total > _coerce_int(ops_counts.get("spot")):
        ops_counts["spot"] = spot_total

    if counterdrill_total > _coerce_int(ops_counts.get("counterdrill")):
        ops_counts["counterdrill"] = counterdrill_total

    if jig_total > _coerce_int(ops_counts.get("jig_grind")):
        ops_counts["jig_grind"] = jig_total

    ops_counts["actions_total"] = (
        max(0, _coerce_int(ops_counts.get("drills")))
        + max(0, _coerce_int(ops_counts.get("taps_total")))
        + max(0, _coerce_int(ops_counts.get("counterbores_total")))
        + max(0, _coerce_int(ops_counts.get("counterdrill")))
        + max(0, _coerce_int(ops_counts.get("spot")))
        + max(0, _coerce_int(ops_counts.get("jig_grind")))
    )

    ops_counts["_audit_claims"] = {
        "drill": drill_total,
        "tap": tap_total,
        "cbore": cb_total,
        "spot": spot_total,
        "counterdrill": counterdrill_total,
        "jig": jig_total,
    }

    return ops_counts


def _append_counterbore_spot_jig_cards(
    *,
    lines_out: list[str],
    chart_lines: list[str] | None,
    rows: list[dict] | None,
    breakdown_mutable,
    rates,
) -> int:
    cb_groups: dict[tuple[float | None, str, float | None], int] = {}  # (dia, side, depth) -> qty
    spot_qty = 0
    jig_qty = 0

    def _extract_counterbore_dia(text: str) -> float | None:
        """Return a numeric counterbore diameter from ``text`` if present."""

        mcb = _CB_DIA_RE.search(text)
        if not mcb:
            mcb = re.search(
                r"(?:Ø|%%[Cc])?\s*(\d+(?:\.\d+)?|\.\d+)\s*C[’']?\s*BORE",
                text,
                re.IGNORECASE,
            )
            if not mcb:
                return None
            raw = mcb.group(1)
        else:
            raw = (mcb.group("numA") or mcb.group("numB") or "").replace(" ", "")
        if not raw:
            return None
        try:
            return float(Fraction(raw)) if "/" in raw else float(raw)
        except Exception:
            return None

    # ---------- PASS A: parse CHART LINES (what you already have: 10) ----------
    if isinstance(chart_lines, list):
        # helpful debug
        for i, ln in enumerate(chart_lines[:6]):
            lines_out.append(f"[DEBUG] chart[{i}]: {ln}")
        for raw in chart_lines:
            s = str(raw or "")
            if not s.strip(): continue
            U = s.upper()
            qty = _parse_qty(s)
            side = _side(U)
            dia = _extract_counterbore_dia(s)
            if dia is not None:
                mdepth = _X_DEPTH_RE.search(s)
                depth = float(mdepth.group(1)) if mdepth else None
                if side == "BOTH":
                    for sd in ("FRONT","BACK"):
                        cb_groups[(dia, sd, depth)] = cb_groups.get((dia, sd, depth), 0) + qty
                else:
                    cb_groups[(dia, side, depth)] = cb_groups.get((dia, side, depth), 0) + qty
                continue
            if (
                _SPOT_RE_TXT.search(s)
                and not _DRILL_THRU.search(s)
                and not ("TAP" in U or _TAP_RE.search(s))
            ):
                spot_qty += qty
                continue
            if _JIG_RE_TXT.search(s):
                jig_qty += qty
                continue

    # ---------- PASS B: fallback to ROWS (your built_rows=3) ----------
    if not cb_groups and isinstance(rows, list):
        for r in rows:
            qty = int((r or {}).get("qty") or 0)
            if qty <= 0: continue
            s = str((r or {}).get("desc") or "")
            if not s.strip(): continue
            U = s.upper()
            side = _side(U)
            dia = _extract_counterbore_dia(s)
            if dia is not None:
                mdepth = _X_DEPTH_RE.search(s)
                depth = float(mdepth.group(1)) if mdepth else None
                if side == "BOTH":
                    for sd in ("FRONT","BACK"):
                        cb_groups[(dia, sd, depth)] = cb_groups.get((dia, sd, depth), 0) + qty
                else:
                    cb_groups[(dia, side, depth)] = cb_groups.get((dia, side, depth), 0) + qty
            else:
                if (
                    _SPOT_RE_TXT.search(s)
                    and not _DRILL_THRU.search(s)
                    and not ("TAP" in U or _TAP_RE.search(s))
                ):
                    spot_qty += qty
                if _JIG_RE_TXT.search(s):
                    jig_qty += qty

    appended = 0

    # ---------- Emit COUNTERBORE card ----------
    if cb_groups:
        total_cb = sum(cb_groups.values())
        front_cb = sum(q for (d,s,dep), q in cb_groups.items() if s=="FRONT")
        back_cb  = sum(q for (d,s,dep), q in cb_groups.items() if s=="BACK")
        try:
            ebo = breakdown_mutable.setdefault("extra_bucket_ops", {})
            total_cb = sum(cb_groups.values())
            if total_cb > 0:
                if front_cb > 0:
                    ebo.setdefault("counterbore", []).append(
                        {"name": "Counterbore", "qty": int(front_cb), "side": "front"}
                    )
                if back_cb > 0:
                    ebo.setdefault("counterbore", []).append(
                        {"name": "Counterbore", "qty": int(back_cb), "side": "back"}
                    )
            if spot_qty > 0:
                ebo.setdefault("spot", []).append(
                    {"name": "Spot drill", "qty": int(spot_qty), "side": "front"}
                )
            if jig_qty > 0:
                ebo.setdefault("jig-grind", []).append(
                    {"name": "Jig-grind", "qty": int(jig_qty), "side": None}
                )
        except Exception:
            pass

        lines_out.extend([
            "MATERIAL REMOVAL – COUNTERBORE",
            "="*64,
            "Inputs",
            "  Ops ............... Counterbore (front + back)",
            f"  Counterbores ...... {total_cb} total  → {front_cb} front, {back_cb} back",
            "",
            "TIME PER HOLE – C’BORE GROUPS",
            "-"*66,
        ])
        per = float(globals().get("CBORE_MIN_PER_SIDE_MIN") or 0.15)  # minutes/side
        cb_minutes = 0.0
        for (dia, side, depth), qty in sorted(cb_groups.items(), key=lambda k:(k[0][0] or 0.0, k[0][1], k[0][2] or 0.0)):
            dia_txt = "—" if dia is None else f'Ø{dia:.4f}"'
            dep_txt = "—" if depth is None else f'{depth:.2f}"'
            t_group = qty * per
            cb_minutes += t_group
            lines_out.append(f"{dia_txt} × {qty}  ({side}) | depth {dep_txt} | t/hole {per:.2f} min | group {qty}×{per:.2f} = {t_group:.2f} min")
        lines_out.append("")
        try:
            _set_bucket_minutes_cost(
                breakdown_mutable.setdefault("bucket_view", {}),
                "counterbore", cb_minutes,
                _lookup_bucket_rate("counterbore", rates) or _lookup_bucket_rate("machine", rates) or 53.76,
                _lookup_bucket_rate("labor", rates) or 25.46,
            )
        except Exception:
            pass
        appended += 1

    # ---------- Emit SPOT card ----------
    if spot_qty > 0:
        per_spot = 0.05
        t_group = spot_qty * per_spot
        lines_out.extend([
            "MATERIAL REMOVAL – SPOT (CENTER DRILL)",
            "="*64,
            "TIME PER HOLE – SPOT GROUPS",
            "-"*66,
            f"Spot drill × {spot_qty} | t/hole {per_spot:.2f} min | group {spot_qty}×{per_spot:.2f} = {t_group:.2f} min",
            "",
        ])
        try:
            _set_bucket_minutes_cost(
                breakdown_mutable.setdefault("bucket_view", {}),
                "drilling", t_group,
                _lookup_bucket_rate("machine", rates) or 53.76,
                _lookup_bucket_rate("labor", rates) or 25.46,
            )
        except Exception:
            pass
        appended += 1

    # ---------- Emit JIG GRIND card ----------
    if jig_qty > 0:
        per_jig = float(globals().get("JIG_GRIND_MIN_PER_FEATURE") or 0.75)  # minutes/feature
        t_group = jig_qty * per_jig
        lines_out.extend([
            "MATERIAL REMOVAL – JIG GRIND",
            "="*64,
            "TIME PER FEATURE",
            "-"*66,
            f"Jig grind × {jig_qty} | t/feat {per_jig:.2f} min | group {jig_qty}×{per_jig:.2f} = {t_group:.2f} min",
            "",
        ])
        try:
            _set_bucket_minutes_cost(
                breakdown_mutable.setdefault("bucket_view", {}),
                "grinding", t_group,
                _lookup_bucket_rate("grinding", rates) or _lookup_bucket_rate("machine", rates) or 53.76,
                _lookup_bucket_rate("labor", rates) or 25.46,
            )
        except Exception:
            pass
        appended += 1

    return appended



def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _safe_rpm_from_sfm_diam(sfm: float, dia_in: float) -> float:
    """Return a clamped RPM derived from surface speed and diameter."""

    d = max(_as_float(dia_in, 0.0), 0.001)
    rpm = (sfm * 12.0) / (math.pi * d)
    if not math.isfinite(rpm) or rpm <= 0.0:
        rpm = 300.0
    return max(50.0, min(rpm, 10000.0))


def _safe_ipm(ipr: float, rpm: float) -> float:
    """Return a clamped inches-per-minute feed rate."""

    ipr_val = max(_as_float(ipr, 0.0), 0.0001)
    rpm_val = max(_as_float(rpm, 0.0), 1.0)
    ipm = ipr_val * rpm_val
    return max(0.05, min(ipm, 200.0))


def rpm_from_sfm(sfm: float, tool_diam_in: float) -> float:
    """Wrapper that exposes :func:`_safe_rpm_from_sfm_diam` under a friendlier name."""

    return _safe_rpm_from_sfm_diam(_as_float(sfm, 0.0), _as_float(tool_diam_in, 0.0))


def ipm_from_rpm_ipt(rpm: float, flutes: float, ipt: float) -> float:
    """Compute inches-per-minute from RPM, flute count and chip load (IPT)."""

    rpm_val = max(_as_float(rpm, 0.0), 1.0)
    flutes_val = max(_as_float(flutes, 0.0), 1.0)
    ipt_val = max(_as_float(ipt, 0.0), 1e-5)
    return rpm_val * flutes_val * ipt_val


def _speeds_feeds_records_from_table(sf_df: Any) -> list[Mapping[str, Any]]:
    """Normalize a Speeds/Feeds table-like object into a list of records."""

    if sf_df is None:
        default_path = globals().get("DEFAULT_SPEEDS_FEEDS_CSV_PATH")
        if default_path:
            try:
                records = _load_speeds_feeds_records(str(default_path))
            except Exception:
                records = []
            return [entry for entry in records if isinstance(entry, _MappingABC)]
        return []

    if _is_pandas_dataframe(sf_df):
        try:
            records = sf_df.to_dict(orient="records")  # type: ignore[call-arg, attr-defined]
        except Exception:
            records = []
        return [entry for entry in records if isinstance(entry, _MappingABC)]

    if isinstance(sf_df, list):
        return [entry for entry in sf_df if isinstance(entry, _MappingABC)]

    if isinstance(sf_df, tuple):
        return [entry for entry in sf_df if isinstance(entry, _MappingABC)]

    if isinstance(sf_df, _MappingABC):
        try:
            iterable = list(sf_df.values())
        except Exception:
            iterable = []
        return [entry for entry in iterable if isinstance(entry, _MappingABC)]

    records: list[Mapping[str, Any]] = []
    try:
        iterator = iter(sf_df)  # type: ignore[arg-type]
    except TypeError:
        return records

    for entry in iterator:
        if isinstance(entry, _MappingABC):
            records.append(entry)
    return records


_MILL_OP_ALIASES: dict[str, tuple[str, ...]] = {
    "face": ("FaceMill", "Face_Mill", "Facing", "Endmill_Profile"),
    "contour": ("Endmill_Profile", "Endmill_Slot", "Profiling"),
}


_MILL_DEFAULTS_BY_OP: dict[str, dict[str, float]] = {
    "face": {
        "sfm": 600.0,
        "ipt": 0.0035,
        "flutes": 6.0,
        "stepover_pct": 0.55,
        "max_ap_in": 0.050,
        "index_min_per_pass": 0.04,
        "toolchange_min": 0.5,
    },
    "contour": {
        "sfm": 450.0,
        "ipt": 0.0025,
        "flutes": 4.0,
        "stepover_pct": 0.40,
        "max_ap_in": 0.200,
        "index_min_per_pass": 0.02,
        "toolchange_min": 0.5,
    },
}


def lookup_mill_params(
    sf_df: Any,
    material_group: str | None,
    *,
    op: str,
    tool_diam_in: float,
) -> dict[str, float]:
    """Pick milling parameters from the Speeds/Feeds table for the requested op."""

    op_key = str(op or "").strip().lower()
    defaults = dict(_MILL_DEFAULTS_BY_OP.get(op_key, _MILL_DEFAULTS_BY_OP["contour"]))
    tool_diam = max(_as_float(tool_diam_in, 0.0), 0.0)

    records = _speeds_feeds_records_from_table(sf_df)
    if not records:
        return defaults

    aliases = _MILL_OP_ALIASES.get(op_key, (op,))
    aliases_lc = tuple(str(alias or "").strip().lower() for alias in aliases)
    mg_norm = str(material_group or "").strip().lower()

    def _row_matches(row: Mapping[str, Any]) -> bool:
        op_val = str(row.get("operation") or "").strip().lower()
        return op_val in aliases_lc if aliases_lc else op_val == op_key

    filtered = [row for row in records if _row_matches(row)]
    if not filtered:
        filtered = records

    chosen: Mapping[str, Any] | None = None
    if mg_norm:
        for row in filtered:
            row_group = str(row.get("material_group") or "").strip().lower()
            if row_group == mg_norm:
                chosen = row
                break
    if chosen is None and filtered:
        chosen = filtered[0]

    if not isinstance(chosen, _MappingABC):
        return defaults

    sfm_val = _coerce_float_or_none(chosen.get("sfm_start"))
    if sfm_val and sfm_val > 0.0:
        defaults["sfm"] = float(sfm_val)

    doc_val = _coerce_float_or_none(chosen.get("doc_axial_in"))
    if doc_val and doc_val > 0.0:
        defaults["max_ap_in"] = float(doc_val)

    woc_pct = _coerce_float_or_none(chosen.get("woc_radial_pct"))
    if woc_pct and woc_pct > 0.0:
        defaults["stepover_pct"] = max(0.05, min(float(woc_pct) / 100.0, 1.0))

    chip_load_candidates: list[tuple[float, float]] = []
    for diam, col in (
        (0.125, "fz_ipr_0_125in"),
        (0.25, "fz_ipr_0_25in"),
        (0.5, "fz_ipr_0_5in"),
    ):
        candidate = _coerce_float_or_none(chosen.get(col))
        if candidate and candidate > 0.0:
            chip_load_candidates.append((diam, float(candidate)))

    chip_load_candidates.sort(key=lambda item: item[0])
    ipt_val = None
    for diam, chip in chip_load_candidates:
        if tool_diam >= diam - 1e-6:
            ipt_val = chip
    if ipt_val is None and chip_load_candidates:
        ipt_val = chip_load_candidates[0][1]
    if ipt_val and ipt_val > 0.0:
        defaults["ipt"] = float(ipt_val)

    if tool_diam >= 1.0:
        defaults["flutes"] = max(defaults.get("flutes", 4.0), 5.0)
    elif tool_diam <= 0.25:
        defaults["flutes"] = max(3.0, defaults.get("flutes", 3.0))

    return defaults


def _geom_dims_from_payload(geom: Mapping[str, Any] | None) -> tuple[float, float, float]:
    """Extract length/width/thickness in inches from a geometry mapping."""

    if not isinstance(geom, _MappingABC):
        return 0.0, 0.0, 0.0

    length = 0.0
    width = 0.0
    thickness = 0.0

    bbox = geom.get("bbox")
    if isinstance(bbox, _MappingABC):
        length = _coerce_float_or_none(bbox.get("x")) or length
        width = _coerce_float_or_none(bbox.get("y")) or width
        thickness = _coerce_float_or_none(bbox.get("z")) or thickness

    if length <= 0.0 or width <= 0.0:
        plate = geom.get("plate")
        if isinstance(plate, _MappingABC):
            length = _coerce_float_or_none(plate.get("length_in")) or length
            width = _coerce_float_or_none(plate.get("width_in")) or width
            thickness = _coerce_float_or_none(plate.get("thickness_in")) or thickness

    if (length <= 0.0 or width <= 0.0) and isinstance(geom.get("bbox_mm"), _MappingABC):
        bbox_mm = typing.cast(Mapping[str, Any], geom.get("bbox_mm"))
        length = (_coerce_float_or_none(bbox_mm.get("x")) or 0.0) / 25.4 or length
        width = (_coerce_float_or_none(bbox_mm.get("y")) or 0.0) / 25.4 or width
        thickness = (_coerce_float_or_none(bbox_mm.get("z")) or 0.0) / 25.4 or thickness

    if thickness <= 0.0:
        thickness_mm = _coerce_float_or_none(geom.get("thickness_mm"))
        if thickness_mm and thickness_mm > 0.0:
            thickness = float(thickness_mm) / 25.4

    if thickness <= 0.0:
        thickness = _coerce_float_or_none(geom.get("thickness_in")) or thickness

    return max(length, 0.0), max(width, 0.0), max(thickness, 0.0)


def estimate_milling_minutes_from_geometry(
    geom: Mapping[str, Any] | None,
    sf_df: Any,
    material_group: str | None,
    rates: Mapping[str, Any] | None,
    *,
    emit_bottom_face: bool = False,
) -> dict[str, Any]:
    """Estimate milling minutes (face + perimeter) from geometric inputs."""

    length_in, width_in, thickness_in = _geom_dims_from_payload(geom)
    if length_in <= 0.0 or width_in <= 0.0:
        return {
            "minutes": 0.0,
            "machine$": 0.0,
            "labor$": 0.0,
            "total$": 0.0,
            "detail": {
                "face_top_min": 0.0,
                "face_bot_min": 0.0,
                "perim_rough_min": 0.0,
                "perim_finish_min": 0.0,
                "toolchanges_min": 0.0,
                "rpm_face": 0.0,
                "ipm_face": 0.0,
                "rpm_contour": 0.0,
                "ipm_contour": 0.0,
                "passes_face": 0,
                "passes_axial": 0,
            },
        }

    perimeter_len = _coerce_float_or_none((geom or {}).get("perimeter_len_in"))
    if not perimeter_len or perimeter_len <= 0.0:
        perimeter_len = 2.0 * (length_in + width_in)

    face_tool_diam = 2.0
    endmill_diam = 0.5

    face_params = lookup_mill_params(sf_df, material_group, op="face", tool_diam_in=face_tool_diam)
    contour_params = lookup_mill_params(
        sf_df,
        material_group,
        op="contour",
        tool_diam_in=endmill_diam,
    )

    rpm_face = rpm_from_sfm(face_params.get("sfm", 0.0), face_tool_diam)
    ipm_face = ipm_from_rpm_ipt(
        rpm_face,
        face_params.get("flutes", 4.0),
        face_params.get("ipt", 0.002),
    )

    stepover_pct = max(face_params.get("stepover_pct", 0.5), 0.05)
    stepover = stepover_pct * face_tool_diam
    passes_face = max(1, math.ceil(width_in / max(stepover, 1e-6)))
    path_len_top = passes_face * length_in * 1.10
    index_time = passes_face * max(face_params.get("index_min_per_pass", 0.03), 0.0)
    t_top_min = (path_len_top / max(ipm_face, 1e-6)) + index_time

    t_bot_min = t_top_min if emit_bottom_face else 0.0

    rpm_contour = rpm_from_sfm(contour_params.get("sfm", 0.0), endmill_diam)
    ipm_contour = ipm_from_rpm_ipt(
        rpm_contour,
        contour_params.get("flutes", 4.0),
        contour_params.get("ipt", 0.002),
    )

    ap = max(contour_params.get("max_ap_in", 0.050), 0.050)
    passes_axial = max(1, math.ceil(thickness_in / max(ap, 1e-6)))
    t_rough_min = (passes_axial * perimeter_len) / max(ipm_contour, 1e-6)

    finish_factor = 1.15
    t_finish_min = perimeter_len / max(ipm_contour / finish_factor, 1e-6)

    milling_min = t_top_min + t_bot_min + t_rough_min + t_finish_min

    toolchanges_min = 0.0
    if t_top_min + t_bot_min > 0.0:
        toolchanges_min += max(face_params.get("toolchange_min", 0.0), 0.0)
    if t_rough_min + t_finish_min > 0.0:
        toolchanges_min += max(contour_params.get("toolchange_min", 0.0), 0.0)

    total_min = milling_min + toolchanges_min

    def _rate_from_mapping(keys: Sequence[str], default: float) -> float:
        if isinstance(rates, _MappingABC):
            for key in keys:
                val = _coerce_float_or_none(rates.get(key))
                if val and val > 0.0:
                    return float(val)
        return default

    mach_rate = float(
        _rate_from_mapping(("machine_per_hour", "machine_rate", "milling_rate", "milling"), 90.0)
    )
    labor_rate = float(
        _rate_from_mapping(("labor_per_hour", "labor_rate", "milling_labor_rate", "labor"), 45.0)
    )

    milling_minutes = float(total_min)
    milling_attended_minutes = max(toolchanges_min, 0.0)

    machine_cost = (milling_minutes / 60.0) * mach_rate
    labor_cost = (milling_attended_minutes / 60.0) * labor_rate
    total_cost = machine_cost + labor_cost

    print(
        f"[CHECK/mill-rate] min={milling_minutes:.2f} hr={milling_minutes / 60.0:.2f} "
        f"mach_rate={mach_rate:.2f}/hr => machine$={machine_cost:.2f}"
    )

    logging.info(
        "[INFO] [milling] face_top=%.2fmin face_bot=%.2fmin rough_perim=%.2fmin "
        "finish_perim=%.2fmin toolchange=%.2fmin total=%.2fmin rpm(face)=%.0f "
        "ipm(face)=%.1f rpm(cnt)=%.0f ipm(cnt)=%.1f",
        t_top_min,
        t_bot_min,
        t_rough_min,
        t_finish_min,
        toolchanges_min,
        total_min,
        rpm_face,
        ipm_face,
        rpm_contour,
        ipm_contour,
    )

    return {
        "minutes": round(total_min, 2),
        "machine$": round(machine_cost, 2),
        "labor$": round(labor_cost, 2),
        "total$": round(total_cost, 2),
        "detail": {
            "face_top_min": round(t_top_min, 2),
            "face_bot_min": round(t_bot_min, 2),
            "perim_rough_min": round(t_rough_min, 2),
            "perim_finish_min": round(t_finish_min, 2),
            "toolchanges_min": round(toolchanges_min, 2),
            "rpm_face": round(rpm_face, 0),
            "ipm_face": round(ipm_face, 1),
            "rpm_contour": round(rpm_contour, 0),
            "ipm_contour": round(ipm_contour, 1),
            "passes_face": int(passes_face),
            "passes_axial": int(passes_axial),
        },
    }


def _clamp_minutes(v: Any, lo: float = 0.0, hi: float = 10000.0) -> float:
    minutes_val = _as_float(v, 0.0)
    if not (lo <= minutes_val <= hi):
        return 0.0
    return minutes_val


def sane_minutes_or_zero(x: Any, cap: float = 24 * 60 * 8) -> float:
    """Return a float minutes value or 0.0 when outside sane bounds."""

    try:
        minutes = float(x)
    except Exception:
        return 0.0

    if not math.isfinite(minutes):
        return 0.0

    if minutes < 0 or minutes > cap:
        print(f"[WARNING] [unit/clamp] minutes out-of-range; dropping. raw={minutes}")
        return 0.0

    return minutes


def _pick_drill_minutes(
    process_plan_summary: Mapping[str, Any] | None,
    extras: Mapping[str, Any] | None,
    lines: list[str] | None = None,
) -> float:
    meta_min = _as_float(
        (((process_plan_summary or {}).get("drilling") or {}).get("total_minutes_billed")),
        0.0,
    )
    removal_min_raw = _as_float((extras or {}).get("drill_total_minutes"), 0.0)
    removal_min = sane_minutes_or_zero(removal_min_raw)

    if removal_min > 0:
        chosen = removal_min
        src = "removal_card"
    else:
        chosen = sane_minutes_or_zero(meta_min)
        src = "planner_meta"

    chosen_clamped = _clamp_minutes(chosen)
    try:
        logger.info(
            "[drill-pick] meta_min=%.2f removal_min=%.2f -> %.2f (%s%s)",
            float(meta_min),
            float(removal_min),
            float(chosen_clamped),
            src,
            " CLAMPED" if chosen_clamped != chosen else "",
        )
    except Exception:
        pass
    if lines is not None:
        try:
            lines.append(
                "[drill-pick] meta_min="
                f"{meta_min:.2f} removal_min={removal_min:.2f} -> {chosen_clamped:.2f} "
                f"({src}{' CLAMPED' if chosen_clamped != chosen else ''})"
            )
        except Exception:
            pass
    return chosen_clamped


# --- HOLE TABLE helpers for CBORE / SPOT / JIG --------------------------------
def _row_txt(x):
    return str((x or {}).get("desc") or "").upper()


def _row_qty(x):
    try:
        return int(round(float((x or {}).get("qty") or 0)))
    except Exception:
        return 0


def _row_side(txt: str) -> str:
    u = (txt or "").upper()
    if re.search(RE_FRONT_BACK, u):
        # RE_FRONT_BACK already matches FRONT/BACK; prefer specificity if present
        if "BACK" in u and "FRONT" not in u:
            return "BACK"
        if "FRONT" in u and "BACK" not in u:
            return "FRONT"
    if "FROM BACK" in u:
        return "BACK"
    if "FROM FRONT" in u:
        return "FRONT"
    return "FRONT"


def _float_or_none(x):
    try:
        v = float(x)
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None


def _depth_from_text(txt: str) -> float | None:
    m = RE_DEPTH.search(txt or "")
    return _float_or_none(m.group(1)) if m else None


def _cbore_dia_from_text(txt: str) -> float | None:
    m = RE_DIA.search(txt or "")
    return _float_or_none(m.group(1)) if m else None


def _build_cbore_groups(rows: list[dict]) -> list[dict]:
    groups = {}
    for r in rows:
        txt = _row_txt(r)
        if not RE_CBORE.search(txt):  # uses imported RE_CBORE
            continue
        qty = _row_qty(r)
        if qty <= 0:
            continue
        side = _row_side(txt)
        depth = _depth_from_text(txt)
        dia = _cbore_dia_from_text(txt)
        key = (dia or -1.0, side, depth or -1.0)
        groups[key] = groups.get(key, 0) + qty
    out = []
    for (dia, side, depth), qty in groups.items():
        out.append(
            {
                "diam_in": None if dia < 0 else float(dia),
                "side": side,
                "depth_in": None if depth < 0 else float(depth),
                "qty": int(qty),
            }
        )
    return sorted(
        out,
        key=lambda g: ((g["diam_in"] or 0), g["side"], (g["depth_in"] or 0)),
    )


def _count_spot_and_jig(rows: list[dict]) -> tuple[int, int]:
    spot = jig = 0
    for r in rows:
        txt = _row_txt(r)
        qty = _row_qty(r)
        if qty <= 0:
            continue
        spot_hit = False
        try:
            if hasattr(_SPOT_TOKENS, "search"):
                spot_hit = bool(_SPOT_TOKENS.search(txt))
            else:
                spot_hit = any(tok in txt for tok in _SPOT_TOKENS)
        except Exception:
            spot_hit = False
        if spot_hit:
            spot += qty
        if re.search(r"\bJIG\s*GRIND\b", txt):
            jig += qty
    return spot, jig


def _build_ops_cards_from_chart_lines(
    *,
    breakdown: Mapping[str, Any] | MutableMapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    rates: Mapping[str, Any] | None,
    breakdown_mutable: MutableMapping[str, Any] | None = None,
    ctx: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
    ctx_a: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
    ctx_b: Mapping[str, Any] | MutableMapping[str, Any] | None = None,  # NEW: let us reuse your collector
) -> list[str]:
    """Return MATERIAL REMOVAL card lines derived from raw hole table text."""

    try:
        geo_map_obj: Mapping[str, Any] | MutableMapping[str, Any] | None = None
        if isinstance(result, _MappingABC):
            geo_map_obj = typing.cast(Mapping[str, Any] | MutableMapping[str, Any] | None, result.get("geo"))
        if not isinstance(geo_map_obj, _MappingABC) and isinstance(breakdown, _MappingABC):
            geo_map_obj = typing.cast(Mapping[str, Any] | MutableMapping[str, Any] | None, breakdown.get("geo"))
        if not isinstance(geo_map_obj, _MappingABC) and isinstance(ctx, _MappingABC):
            try:
                geo_map_obj = typing.cast(Mapping[str, Any] | MutableMapping[str, Any] | None, ctx.get("geo"))
            except Exception:
                geo_map_obj = None
        geo_map: Mapping[str, Any] | MutableMapping[str, Any]
        if isinstance(geo_map_obj, _MappingABC):
            geo_map = typing.cast(Mapping[str, Any] | MutableMapping[str, Any], geo_map_obj)
        else:
            geo_map = {}

        chart_lines: Sequence[str] | None
        try:
            chart_lines = _collect_chart_lines_context(ctx, geo_map, ctx_a, ctx_b)
        except Exception:
            chart_lines = _get_chart_lines_for_ops(
                breakdown,
                result,
                ctx=ctx,
                ctx_a=ctx_a,
                ctx_b=ctx_b,
            )

        if not chart_lines:
            chart_lines = _get_chart_lines_for_ops(
                breakdown,
                result,
                ctx=ctx,
                ctx_a=ctx_a,
                ctx_b=ctx_b,
            )
        if not chart_lines:
            return []

        if isinstance(chart_lines, list):
            chart_lines_list = chart_lines
        else:
            try:
                chart_lines_list = list(chart_lines)
            except Exception:
                return []
        if not chart_lines_list:
            return []

        cleaned_chart_lines: list[str] = []
        for raw_line in chart_lines_list:
            cleaned_line = _clean_mtext(str(raw_line or ""))
            if cleaned_line:
                cleaned_chart_lines.append(cleaned_line)

        joined_chart = _join_wrapped_chart_lines(cleaned_chart_lines)
        if not joined_chart:
            return []

        built_rows = _build_ops_rows_from_lines_fallback(joined_chart)
        if not built_rows:
            return []

        target_breakdown: MutableMapping[str, Any] | None = breakdown_mutable
        if target_breakdown is None:
            if isinstance(breakdown, dict):
                target_breakdown = breakdown
            elif isinstance(breakdown, _MutableMappingABC):
                target_breakdown = typing.cast(MutableMapping[str, Any], breakdown)

        bucket_view_obj: MutableMapping[str, Any] | Mapping[str, Any] | None = None
        try:
            if isinstance(target_breakdown, dict):
                bucket_view_obj = target_breakdown.setdefault("bucket_view", {})
            elif isinstance(target_breakdown, _MutableMappingABC):
                bucket_view_obj = typing.cast(
                    MutableMapping[str, Any],
                    target_breakdown.setdefault("bucket_view", {}),
                )
        except Exception:
            bucket_view_obj = None

        out_lines: list[str] = []

        cbores = _build_cbore_groups(built_rows)
        if cbores:
            out_lines.append("MATERIAL REMOVAL – COUNTERBORE")
            out_lines.append("=" * 64)
            total_cb = sum(g["qty"] for g in cbores)
            front_cb = sum(g["qty"] for g in cbores if g["side"] == "FRONT")
            back_cb = sum(g["qty"] for g in cbores if g["side"] == "BACK")
            out_lines.append("Inputs")
            out_lines.append("  Ops ............... Counterbore (front + back)")
            out_lines.append(
                f"  Counterbores ...... {total_cb} total  → {front_cb} front, {back_cb} back"
            )
            out_lines.append("")
            out_lines.append("TIME PER HOLE – C’BORE GROUPS")
            out_lines.append("-" * 66)

            cb_minutes = 0.0
            for group in cbores:
                depth_txt = "—" if group["depth_in"] is None else f'{group["depth_in"]:.2f}"'
                dia_txt = "—" if group["diam_in"] is None else f'Ø{group["diam_in"]:.4f}"'
                per = max(float(CBORE_MIN_PER_SIDE_MIN or 0.07), 0.01)
                t_group = group["qty"] * per
                cb_minutes += t_group
                out_lines.append(
                    f'{dia_txt} × {group["qty"]}  ({group["side"]}) | '
                    f"depth {depth_txt} | t/hole {per:.2f} min | "
                    f"group {group['qty']}×{per:.2f} = {t_group:.2f} min"
                )
            out_lines.append("")

            cb_mrate = (
                _lookup_bucket_rate("counterbore", rates)
                or _lookup_bucket_rate("machine", rates)
                or 53.76
            )
            cb_lrate = _lookup_bucket_rate("labor", rates) or 25.46
            _set_bucket_minutes_cost(bucket_view_obj, "counterbore", cb_minutes, cb_mrate, cb_lrate)
            try:
                bv = breakdown_mutable.setdefault("bucket_view", {})
                buckets = bv.setdefault("buckets", {})
                order = bv.setdefault("order", [])
                if "counterbore" in buckets and "counterbore" not in order:
                    if "drilling" in order:
                        order.insert(order.index("drilling") + 1, "counterbore")
                    else:
                        order.append("counterbore")
            except Exception:
                pass

        spot_qty, jig_qty = _count_spot_and_jig(built_rows)
        if spot_qty > 0:
            out_lines.append("MATERIAL REMOVAL – SPOT (CENTER DRILL)")
            out_lines.append("=" * 64)
            per = 0.05
            t_group = spot_qty * per
            out_lines.append("TIME PER HOLE – SPOT GROUPS")
            out_lines.append("-" * 66)
            out_lines.append(
                f"Spot drill × {spot_qty} | t/hole {per:.2f} min | "
                f"group {spot_qty}×{per:.2f} = {t_group:.2f} min"
            )
            out_lines.append("")
            _set_bucket_minutes_cost(
                bucket_view_obj,
                "drilling",
                t_group,
                _lookup_bucket_rate("machine", rates) or 53.76,
                _lookup_bucket_rate("labor", rates) or 25.46,
            )

        if jig_qty > 0:
            out_lines.append("MATERIAL REMOVAL – JIG GRIND")
            out_lines.append("=" * 64)
            per = 0.75
            t_group = jig_qty * per
            out_lines.append("TIME PER FEATURE")
            out_lines.append("-" * 66)
            out_lines.append(
                f"Jig grind × {jig_qty} | t/feat {per:.2f} min | "
                f"group {jig_qty}×{per:.2f} = {t_group:.2f} min"
            )
            out_lines.append("")
            _set_bucket_minutes_cost(
                bucket_view_obj,
                "grinding",
                t_group,
                _lookup_bucket_rate("grinding", rates) or _lookup_bucket_rate("machine", rates) or 53.76,
                _lookup_bucket_rate("labor", rates) or 25.46,
            )

        if out_lines:
            try:
                _normalize_buckets(breakdown.get("bucket_view"))
            except Exception:
                pass

        return out_lines
    except Exception:
        return []


def _emit_hole_table_ops_cards(
    lines: list[str],
    *,
    geo: Mapping[str, Any] | None,
    material_group: str | None = None,
    speeds_csv: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | MutableMapping[str, Any] | None = None,
    rates: Mapping[str, Any] | None = None,
) -> None:
    """Render hole-table derived operation cards (tapping first pass)."""

    try:
        ops_summary = ((geo or {}).get("ops_summary") or {}) if isinstance(geo, _MappingABC) else {}
        rows_obj: Any = ops_summary.get("rows") if isinstance(ops_summary, _MappingABC) else None
        if not rows_obj:
            return

        if not isinstance(rows_obj, list):
            try:
                rows_list = list(rows_obj)
            except Exception:
                return
            if isinstance(ops_summary, (_MutableMappingABC, dict)):
                typing.cast(MutableMapping[str, Any], ops_summary)["rows"] = rows_list
            rows = rows_list
        else:
            rows = rows_obj

        if not rows:
            return

        thickness_in = _resolve_part_thickness_in(
            geo,
            (breakdown.get("geo") if isinstance(breakdown, _MappingABC) else None),
            (result.get("geo") if isinstance(result, _MappingABC) else None),
        )

        bucket_view_obj: MutableMapping[str, Any] | Mapping[str, Any] | None = None
        try:
            if isinstance(breakdown, dict):
                bucket_view_obj = breakdown.setdefault("bucket_view", {})
            elif isinstance(breakdown, _MutableMappingABC):
                bucket_view_obj = typing.cast(
                    MutableMapping[str, Any],
                    breakdown.setdefault("bucket_view", {}),
                )
            elif isinstance(breakdown, _MappingABC):
                bucket_view_obj = typing.cast(Mapping[str, Any], breakdown.get("bucket_view"))
        except Exception:
            bucket_view_obj = None

        tap_rows = _finalize_tapping_rows(rows, thickness_in=thickness_in)

        def _extract_ops_claims_map(
            *sources: Mapping[str, Any] | MutableMapping[str, Any] | None,
        ) -> Mapping[str, Any] | None:
            for candidate in sources:
                if not isinstance(candidate, (_MappingABC, dict)):
                    continue
                try:
                    claims_candidate = candidate.get("ops_claims")  # type: ignore[index]
                except Exception:
                    claims_candidate = None
                if isinstance(claims_candidate, (_MappingABC, dict)):
                    return typing.cast(Mapping[str, Any], claims_candidate)
            return None

        ops_claims_map = _extract_ops_claims_map(geo, breakdown, result)
        if ops_claims_map is None and isinstance(ops_summary, _MappingABC):
            try:
                claims_candidate = ops_summary.get("claims")
            except Exception:
                claims_candidate = None
            if isinstance(claims_candidate, (_MappingABC, dict)):
                ops_claims_map = typing.cast(Mapping[str, Any], claims_candidate)

        def _int_or_zero(value: Any) -> int:
            try:
                num = float(value)
            except Exception:
                return 0
            if not math.isfinite(num):
                return 0
            return int(round(num))

        npt_qty = _int_or_zero(ops_claims_map.get("npt")) if ops_claims_map else 0
        if npt_qty <= 0:
            npt_qty = 0
            for entry in rows:
                if not isinstance(entry, _MappingABC):
                    continue
                try:
                    qty_val = int(entry.get("qty") or 0)
                except Exception:
                    qty_val = 0
                if qty_val <= 0:
                    continue
                desc_text = str(entry.get("desc") or "").upper()
                if "NPT" in desc_text and "TAP" not in desc_text:
                    npt_qty += qty_val

        if npt_qty > 0:
            existing_npt = any(
                "NPT" in str(row.get("thread") or row.get("desc") or "").upper()
                for row in tap_rows
            )
            if not existing_npt:
                depth_val = float(thickness_in or 0.0)
                depth_display = f"{depth_val:.2f}\"" if depth_val > 0 else "THRU"
                per_hole = 0.20
                tap_rows.append(
                    {
                        "label": "1/8- NPT TAP",
                        "desc": "1/8- NPT TAP",
                        "thread": "1/8- NPT",
                        "qty": int(npt_qty),
                        "side": "FRONT",
                        "t_per_hole_min": per_hole,
                        "feed_fmt": "- ipr | - rpm | - ipm",
                        "depth_in": depth_val,
                        "depth_in_display": depth_display,
                        "ipr": 0.0,
                        "rpm": 0,
                        "ipm": 0.0,
                    }
                )
        tap_total_min = 0.0
        if tap_rows:
            tap_total_min = _render_ops_card(
                lambda text: _push(lines, text),
                title="Material Removal – Tapping",
                rows=tap_rows,
            )

            tap_total_min = float(tap_total_min or 0.0)

            try:
                if isinstance(ops_summary, (_MutableMappingABC, dict)):
                    typing.cast(MutableMapping[str, Any], ops_summary)["tap_minutes_total"] = tap_total_min
            except Exception:
                pass

            tap_mrate = (
                _lookup_bucket_rate("tapping", rates)
                or _lookup_bucket_rate("machine", rates)
                or 45.0
            )
            tap_labor_explicit: float | None = None
            if isinstance(rates, _MappingABC):
                for raw_key, raw_value in rates.items():
                    key_text = str(raw_key).strip().lower()
                    if key_text in {"tapping_labor", "tappinglaborrate", "tapping_labor_rate"}:
                        candidate = _as_float(raw_value, 0.0)
                        if candidate > 0:
                            tap_labor_explicit = candidate
                        break
            if tap_labor_explicit is not None and tap_labor_explicit > 0:
                tap_lrate = tap_labor_explicit
            else:
                tap_lrate = (
                    _lookup_bucket_rate("labor", rates)
                    or 45.0
                )

            _set_bucket_minutes_cost(
                bucket_view_obj,
                "tapping",
                tap_total_min,
                float(tap_mrate or 0.0),
                float(tap_lrate or 0.0),
            )

            dbg_entry: Mapping[str, Any] | None = None
            try:
                if isinstance(bucket_view_obj, _MappingABC):
                    dbg_entry = (bucket_view_obj.get("buckets") or {}).get("tapping")  # type: ignore[assignment]
            except Exception:
                dbg_entry = None
            dbg_payload = dbg_entry if dbg_entry is not None else {}
            _push(lines, "[DEBUG] tapping_bucket=" + repr(dbg_payload))

        # --- COUNTERBORE CARD --------------------------------------------------
        cbores = _build_cbore_groups(rows)
        if cbores:
            _push(lines, "MATERIAL REMOVAL – COUNTERBORE")
            _push(lines, "=" * 64)
            total_cb = sum(g["qty"] for g in cbores)
            front_cb = sum(g["qty"] for g in cbores if g["side"] == "FRONT")
            back_cb = sum(g["qty"] for g in cbores if g["side"] == "BACK")
            _push(lines, f"Inputs")
            _push(lines, f"  Ops ............... Counterbore (front + back)")
            _push(lines, f"  Counterbores ...... {total_cb} total  → {front_cb} front, {back_cb} back")
            _push(lines, "")
            _push(lines, "TIME PER HOLE – C’BORE GROUPS")
            _push(lines, "-" * 66)

            cb_minutes = 0.0
            for g in cbores:
                depth_txt = "—" if g["depth_in"] is None else f'{g["depth_in"]:.2f}"'
                dia_txt = "—" if g["diam_in"] is None else f'Ø{g["diam_in"]:.4f}"'
                # Use your calibrated per-side minute (imported from hole_ops)
                per = max(float(CBORE_MIN_PER_SIDE_MIN or 0.07), 0.01)
                t_group = g["qty"] * per
                cb_minutes += t_group
                _push(
                    lines,
                    f'{dia_txt} × {g["qty"]}  ({g["side"]}) | depth {depth_txt} | t/hole {per:.2f} min | group {g["qty"]}×{per:.2f} = {t_group:.2f} min',
                )
            _push(lines, "")

            # push to buckets
            bucket_view_obj = None
            try:
                if isinstance(breakdown, dict):
                    bucket_view_obj = breakdown.setdefault("bucket_view", {})
                elif isinstance(breakdown, _MutableMappingABC):
                    bucket_view_obj = typing.cast(
                        MutableMapping[str, Any],
                        breakdown.setdefault("bucket_view", {}),
                    )
            except Exception:
                bucket_view_obj = None

            cb_mrate = _lookup_bucket_rate("counterbore", rates) or _lookup_bucket_rate("machine", rates) or 53.76
            cb_lrate = _lookup_bucket_rate("labor", rates) or 25.46
            _set_bucket_minutes_cost(bucket_view_obj, "counterbore", cb_minutes, cb_mrate, cb_lrate)
            try:
                bv: MutableMapping[str, Any] | None = None
                if isinstance(breakdown, dict):
                    bv = typing.cast(MutableMapping[str, Any], breakdown.setdefault("bucket_view", {}))
                elif isinstance(breakdown, _MutableMappingABC):
                    bv = typing.cast(MutableMapping[str, Any], breakdown).setdefault("bucket_view", {})
                elif isinstance(bucket_view_obj, _MutableMappingABC):
                    bv = typing.cast(MutableMapping[str, Any], bucket_view_obj)
                if bv is not None:
                    buckets = typing.cast(MutableMapping[str, Any], bv).setdefault("buckets", {})
                    order = typing.cast(list[str], bv.setdefault("order", []))
                    if "counterbore" in buckets and "counterbore" not in order:
                        if "drilling" in order:
                            order.insert(order.index("drilling") + 1, "counterbore")
                        else:
                            order.append("counterbore")
            except Exception:
                pass

        # --- SPOT & JIG-GRIND CARDS -------------------------------------------
        spot_qty, jig_qty = _count_spot_and_jig(rows)
        if spot_qty > 0:
            _push(lines, "MATERIAL REMOVAL – SPOT (CENTER DRILL)")
            _push(lines, "=" * 64)
            per = 0.05  # light index & peck allowance
            t_group = spot_qty * per
            _push(lines, "TIME PER HOLE – SPOT GROUPS")
            _push(lines, "-" * 66)
            _push(lines, f"Spot drill × {spot_qty} | t/hole {per:.2f} min | group {spot_qty}×{per:.2f} = {t_group:.2f} min")
            _push(lines, "")
            # put spots into drilling bucket (alias normalized)
            try:
                bucket_view_obj = bucket_view_obj or breakdown.setdefault("bucket_view", {})  # type: ignore[assignment]
            except Exception:
                bucket_view_obj = bucket_view_obj or None
            _set_bucket_minutes_cost(
                bucket_view_obj,
                "drilling",
                t_group,
                _lookup_bucket_rate("machine", rates) or 53.76,
                _lookup_bucket_rate("labor", rates) or 25.46,
            )

        if jig_qty > 0:
            _push(lines, "MATERIAL REMOVAL – JIG GRIND")
            _push(lines, "=" * 64)
            per = 0.75
            t_group = jig_qty * per
            _push(lines, "TIME PER FEATURE")
            _push(lines, "-" * 66)
            _push(lines, f"Jig grind × {jig_qty} | t/feat {per:.2f} min | group {jig_qty}×{per:.2f} = {t_group:.2f} min")
            _push(lines, "")
            try:
                bucket_view_obj = bucket_view_obj or breakdown.setdefault("bucket_view", {})  # type: ignore[assignment]
            except Exception:
                bucket_view_obj = bucket_view_obj or None
            _set_bucket_minutes_cost(
                bucket_view_obj,
                "grinding",
                t_group,
                _lookup_bucket_rate("grinding", rates) or _lookup_bucket_rate("machine", rates) or 53.76,
                _lookup_bucket_rate("labor", rates) or 25.46,
            )

        # normalize once at the end
        try:
            _normalize_buckets(breakdown.get("bucket_view"))  # merges aliases like spotdrill→spot_drill
        except Exception:
            pass
    except Exception as exc:
        _push(lines, f"[DEBUG] tapping_emit_skipped={exc.__class__.__name__}: {exc}")
        return
if TYPE_CHECKING:
    from cad_quoter_pkg.src.cad_quoter.estimators import drilling_legacy as _drilling_legacy
else:
    from cad_quoter.estimators import drilling_legacy as _drilling_legacy
from cad_quoter.estimators.base import SpeedsFeedsUnavailableError
from cad_quoter.llm_overrides import (
    _plate_mass_properties,
    _plate_mass_from_dims,
    clamp,
)

from cad_quoter.domain import (
    HARDWARE_PASS_LABEL,
    _canonical_pass_label,
    coerce_bounds,
    build_suggest_payload,
    overrides_to_suggestions,
    suggestions_to_overrides,
)

if typing.TYPE_CHECKING:  # pragma: no cover - aid static analysers in monorepo layout
    from cad_quoter_pkg.src.cad_quoter.domain_models.state import QuoteState
else:  # pragma: no cover - runtime shim retains the existing fallback behaviour
    try:
        from cad_quoter.domain import QuoteState
    except ImportError:
        from cad_quoter_pkg.src.cad_quoter.domain_models.state import QuoteState

if typing.TYPE_CHECKING:  # pragma: no cover - make vendor shim visible to Pylance
    from cad_quoter_pkg.src.cad_quoter.vendors import ezdxf as _ezdxf_vendor
else:
    from cad_quoter.vendors import ezdxf as _ezdxf_vendor

from cad_quoter.geometry.dxf_enrich import (
    detect_units_scale as _shared_detect_units_scale,
    iter_spaces as _shared_iter_spaces,
    iter_table_entities as _shared_iter_table_entities,
    iter_table_text as _shared_iter_table_text,
)

from cad_quoter.pricing.process_buckets import BUCKET_ROLE, PROCESS_BUCKETS, bucketize

if typing.TYPE_CHECKING:  # pragma: no cover - expose rich geometry types to analysers
    from cad_quoter_pkg.src.cad_quoter import geometry as geometry
    from cad_quoter_pkg.src.cad_quoter.geometry import (
        upsert_var_row as geometry_upsert_var_row,
    )
else:
    import cad_quoter.geometry as geometry  # type: ignore[import-not-found]

    try:
        from cad_quoter.geometry import upsert_var_row as geometry_upsert_var_row
    except ImportError:  # pragma: no cover - development fallback
        from cad_quoter_pkg.src.cad_quoter.geometry import (
            upsert_var_row as geometry_upsert_var_row,
        )

    geometry = typing.cast(typing.Any, geometry)



_RE_SPLIT = re.split
_RE_SUB = re.sub


def _infer_rect_from_holes(geo_candidate: Mapping[str, Any] | None) -> tuple[float, float]:
    dims = infer_plate_lw_in(geo_candidate)
    if not dims:
        return (0.0, 0.0)
    width, height = dims
    return float(width), float(height)

# Safe append to the current quote buffer
def _push(_lines, text):
    try:
        if isinstance(_lines, list):
            _lines.append(str(text))
        else:
            print(str(text))
    except Exception:
        pass


_HAS_ODAFC = bool(getattr(geometry, "HAS_ODAFC", False))
_HAS_PYMUPDF = bool(getattr(geometry, "HAS_PYMUPDF", getattr(geometry, "_HAS_PYMUPDF", False)))
fitz = getattr(geometry, "fitz", None)
try:
    odafc = _ezdxf_vendor.require_odafc() if _HAS_ODAFC else None
except Exception:
    odafc = None  # type: ignore[assignment]

from cad_quoter.app.llm_adapter import (
    apply_llm_hours_to_variables,
    clamp_llm_hours,
    configure_llm_integration,
    infer_hours_and_overrides_from_geo,
    normalize_item,
    normalize_item_text,
)

from cad_quoter.ui.tk_compat import (
    tk,
    filedialog,
    messagebox,
    ttk,
    _ensure_tk,
)

from cad_quoter.ui.widgets import (
    CreateToolTip,
    ScrollableFrame,
)

# UI service containers and configuration helpers
from cad_quoter.ui.services import (
    GeometryLoader,
    LLMServices,
    PricingRegistry,
    UIConfiguration,
    infer_geo_override_defaults,
)

from cad_quoter.app.guardrails import build_guard_context, apply_drilling_floor_notes
from cad_quoter.app.merge_utils import (
    ACCEPT_SCALAR_KEYS,
    merge_effective,
)

from cad_quoter.app.effective import (
    compute_effective_state,
    ensure_accept_flags,
    reprice_with_effective,
)

from cad_quoter.ui import suggestions as ui_suggestions

from cad_quoter.utils.scrap import (
    HOLE_SCRAP_CAP,
    SCRAP_DEFAULT_GUESS,
    _holes_scrap_fraction,
    normalize_scrap_pct,
)
from cad_quoter.utils.render_utils.tables import ascii_table, draw_kv_table
from cad_quoter.app.planner_helpers import _process_plan_job
from cad_quoter.app.env_flags import FORCE_PLANNER
from cad_quoter.app.planner_adapter import resolve_planner, resolve_pricing_source_value

from cad_quoter.resources.loading import load_json, load_text

# Mapping of PDF estimate keys to Quote Editor variables.
MAP_KEYS = load_json("vl_pdf_map_keys.json")
from cad_quoter.utils.text_rules import (
    PROC_MULT_TARGETS,
    canonicalize_amortized_label as _canonical_amortized_label,
)
from cad_quoter.utils.debug_tables import (
    _accumulate_drill_debug,
    append_removal_debug_if_enabled,
)
from cad_quoter.ui import llm_panel
from cad_quoter.ui import session_io
from cad_quoter.ui.editor_controls import coerce_checkbox_state, derive_editor_control_spec
from cad_quoter.ui.planner_render import (
    PROGRAMMING_AMORTIZED_LABEL,
    PROGRAMMING_PER_PART_LABEL,
    PlannerBucketRenderState,
    _bucket_cost,
    _bucket_cost_mode,
    _canonical_bucket_key,
    _display_bucket_label,
    _display_rate_for_row,
    _hole_table_minutes_from_geo,
    _rate_key_for_bucket,
    _lookup_bucket_rate,
    _normalize_bucket_key,
    _op_role_for_name,
    _planner_bucket_key_for_name,
    _preferred_order_then_alpha,
    _prepare_bucket_view,
    _extract_bucket_map,
    _process_label,
    _seed_bucket_minutes as _planner_seed_bucket_minutes,
    _normalize_buckets,
    _split_hours_for_bucket,
    _purge_legacy_drill_sync,
    _build_planner_bucket_render_state,
    _FINAL_BUCKET_HIDE_KEYS,
    SHOW_BUCKET_DIAGNOSTICS_OVERRIDE,
    canonicalize_costs,
)
from cad_quoter.ui.services import QuoteConfiguration
from cad_quoter.pricing.validation import validate_quote_before_pricing
from cad_quoter.utils.debug_tables import (
    _jsonify_debug_summary as _debug_jsonify_summary,
    _jsonify_debug_value as _debug_jsonify_value,
)




# ──────────────────────────────────────────────────────────────────────────────
# Helpers: formatting + removal card + per-hole lines (no material per line)
# ──────────────────────────────────────────────────────────────────────────────
def _render_removal_card(
    lines: list[str],
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
    _push(lines, "MATERIAL REMOVAL – DRILLING")
    _push(lines, "=" * 64)
    # Inputs
    _push(lines, "Inputs")
    _push(lines, f"  Material .......... {mat_canon}  [group {mat_group or '-'}]")
    mismatch = False
    if row_group:
        rg = str(row_group).upper()
        mg = str(mat_group or "").upper()
        mismatch = (rg != mg and (rg and mg))
        note = "   (!) mismatch – used row from different group" if mismatch else ""
        _push(lines, f"  CSV row group ..... {row_group}{note}")
    _push(lines, "  Operations ........ Deep-Drill (L/D ≥ 3), Drill")
    _push(
        lines,
        f"  Holes ............. {int(holes_deep)} deep + {int(holes_std)} std  = {int(holes_deep + holes_std)}"
    )
    _push(lines, f'  Diameter range .... {_fmt_rng(dia_vals_in, 3)}"')
    _push(lines, f"  Depth per hole .... {_fmt_rng(depth_vals_in, 2)} in")
    _push(lines, "")
    # Feeds & Speeds
    _push(lines, "Feeds & Speeds (used)")
    _push(lines, f"  SFM ............... {int(round(sfm_deep))} (deep)   | {int(round(sfm_std))} (std)")
    _push(
        lines,
        f"  IPR ............... {_fmt_rng(ipr_deep_vals, 4)} (deep) | {float(ipr_std_val):.4f} (std)"
    )
    _push(
        lines,
        f"  RPM ............... {_fmt_rng(rpm_deep_vals, 0)} (deep)      | {_fmt_rng(rpm_std_vals, 0)} (std)"
    )
    _push(
        lines,
        f"  IPM ............... {_fmt_rng(ipm_deep_vals, 1)} (deep)       | {_fmt_rng(ipm_std_vals, 1)} (std)"
    )
    _push(lines, "")
    # Overheads
    _push(lines, "Overheads")
    _push(lines, f"  Index per hole .... {float(index_min_per_hole):.2f} min")
    _push(lines, f"  Peck per hole ..... {_fmt_rng(peck_min_rng, 2)} min")
    _push(
        lines,
        f"  Toolchange ........ {float(toolchange_min_deep):.2f} min (deep) | {float(toolchange_min_std):.2f} min (std)"
    )
    _push(lines, "")


def _render_time_per_hole(
    lines: list[str],
    *,
    bins: list[dict[str, Any]],
    index_min: float,
    peck_min_deep: float,
    peck_min_std: float,
    extra_map: MutableMapping[str, Any] | None = None,
) -> tuple[float, bool, bool, list[dict[str, Any]]]:
    _push(lines, "TIME PER HOLE – DRILL GROUPS")
    _push(lines, "-" * 66)
    subtotal_minutes = 0.0
    drill_groups: list[dict[str, Any]] = []
    groups_processed = 0
    seen_deep = False
    seen_std = False
    for b in bins:
        try:
            op = (b.get("op") or b.get("op_name") or "").strip().lower()
            is_deep = op.startswith("deep")
            if is_deep:
                seen_deep = True
            else:
                seen_std = True
            dia_in = _safe_float(b.get("diameter_in"))
            depth_in = max(_safe_float(b.get("depth_in")), 0.0)
            qty = int(_as_float(b.get("qty"), 0.0) or 0)
            if dia_in <= 0.0 or depth_in <= 0.0 or qty <= 0:
                continue

            sfm = 39.0 if is_deep else 80.0
            ipr = 0.0020 if is_deep else 0.0060
            rpm = _safe_rpm_from_sfm_diam(sfm, dia_in)
            ipm = _safe_ipm(ipr, rpm)

            t_cut_min = depth_in / ipm
            peck_min = float(peck_min_deep if is_deep else peck_min_std)

            peck_step_in = _as_float(b.get("peck_step_in"), 0.0)
            if peck_step_in <= 0.0:
                peck_step_in = _as_float(b.get("peck_step"), 0.0)
            if peck_step_in <= 0.0:
                peck_step_in = _as_float(b.get("peck_depth_in"), 0.0)

            n_pecks = 0
            if peck_step_in > 0.0:
                n_pecks = max(0, math.ceil(depth_in / max(peck_step_in, 0.001)) - 1)
            else:
                n_pecks_val = (
                    _safe_float(b.get("peck_count"), 0.0)
                    or _safe_float(b.get("pecks"), 0.0)
                    or _safe_float(b.get("n_pecks"), 0.0)
                )
                if n_pecks_val and math.isfinite(n_pecks_val):
                    n_pecks = max(0, int(n_pecks_val))

            t_hole_min = t_cut_min + float(index_min) + n_pecks * peck_min
            t_group = qty * t_hole_min
            subtotal_minutes += t_group
            logging.debug(
                f"[removal/drill-line] dia={dia_in:.4f} depth_in={depth_in:.4f} "
                f"ipr={ipr:.4f} rpm={rpm:.1f} ipm={ipm:.3f} "
                f"t_cut_min={t_cut_min:.4f} index={float(index_min):.3f} "
                f"peck={peck_min:.3f}×{n_pecks} t_hole_min={t_hole_min:.4f} "
                f"qty={qty} t_group={t_group:.4f}"
            )
            groups_processed += 1
            drill_groups.append(
                {
                    "diameter_in": float(dia_in),
                    "qty": int(qty),
                    "depth_in": float(depth_in),
                    "sfm": float(sfm),
                    "ipr": float(ipr),
                    "t_hole_min": float(t_hole_min),
                    "t_group_min": float(t_group),
                }
            )
            _push(
                lines,
                f'Dia {dia_in:.3f}" × {qty}  | depth {depth_in:.3f}" | {int(round(sfm))} sfm | '
                f'{ipr:.4f} ipr | t/hole {t_hole_min:.2f} min | '
                f'group {qty}×{t_hole_min:.2f} = {t_group:.2f} min'
            )
        except Exception:
            continue
    _push(lines, "")
    logging.info(
        f"[removal/drill-sum] groups={groups_processed} subtotal_min={subtotal_minutes:.2f}"
    )
    if not (0.0 <= subtotal_minutes <= 600.0):
        logging.error(
            f"[unit] removal DRILL minutes insane; dropping. raw={subtotal_minutes}"
        )
        subtotal_minutes = 0.0
    if isinstance(extra_map, _MutableMappingABC):
        extra_map["drill_total_minutes"] = round(subtotal_minutes, 2)
        if drill_groups:
            try:
                extra_map["drill_groups"] = [dict(group) for group in drill_groups]
            except Exception:
                extra_map["drill_groups"] = drill_groups
        logging.info(
            f"[removal] drill_total_minutes={extra_map['drill_total_minutes']}"
        )
    return subtotal_minutes, seen_deep, seen_std, drill_groups


_thread_tpi = re.compile(r"#?\s*\d{1,2}\s*-\s*(\d+)", re.I)
_thread_frac = re.compile(r"(\d+)\s*/\s*(\d+)\s*-\s*(\d+)", re.I)
_thread_metric = re.compile(r"M\s*([\d.]+)\s*x\s*([\d.]+)", re.I)


def _thread_ipr(thread_str: str) -> float:
    """Return inches per revolution from thread designator."""

    s = (thread_str or "").strip().upper()
    m = _thread_tpi.search(s)
    if m:
        tpi = float(m.group(1))
        return 1.0 / tpi if tpi > 0 else 0.0
    m = _thread_frac.search(s)
    if m:
        tpi = float(m.group(3))
        return 1.0 / tpi if tpi > 0 else 0.0
    m = _thread_metric.search(s)
    if m:
        pitch_mm = float(m.group(2))
        return (pitch_mm / 25.4) if pitch_mm > 0 else 0.0
    return 0.0


def _tap_rpm_for_dia(dia_in: float, sfm: float = 60.0, rpm_cap: float = 1500.0) -> float:
    """Return an RPM estimate for a tap diameter."""

    if not dia_in or dia_in <= 0:
        return 400.0
    rpm = (sfm * 3.82) / float(dia_in)
    return max(100.0, min(rpm, rpm_cap))


def _derive_major_dia_in(thread_str: str) -> float:
    """Rough major diameter for taps; enough to get a reasonable RPM."""

    s = (thread_str or "").strip().upper()
    m = _thread_metric.search(s)
    if m:
        return float(m.group(1)) / 25.4
    m = _thread_frac.search(s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    if s.startswith("#"):
        num_map = {"#4": 0.112, "#6": 0.138, "#8": 0.164, "#10": 0.190, "#12": 0.216}
        for key, value in num_map.items():
            if s.startswith(key):
                return value
    return 0.25


def _resolve_part_thickness_in(
    *contexts: Mapping[str, Any] | MutableMapping[str, Any] | None,
    default: float = 2.0,
) -> float:
    """Best-effort thickness lookup from nested geo/breakdown contexts."""

    def _maybe_t(container: Mapping[str, Any] | MutableMapping[str, Any] | None) -> float | None:
        if not isinstance(container, _MappingABC):
            return None
        t_val = _coerce_positive_float(container.get("t"))
        return float(t_val) if t_val else None

    for ctx in contexts:
        if not isinstance(ctx, _MappingABC):
            continue
        for nested_key in ("required_blank_in", "bbox_in"):
            nested = ctx.get(nested_key)
            t_guess = _maybe_t(nested if isinstance(nested, _MappingABC) else None)
            if t_guess:
                return t_guess
        for key in ("thickness_in", "thk_in", "thickness", "t"):
            t_guess = _coerce_positive_float(ctx.get(key))
            if t_guess:
                return float(t_guess)
    return float(default)


def _finalize_tap_row(row: MutableMapping[str, Any], thickness_in: float) -> None:
    thread = row.get("thread") or row.get("desc") or ""
    ipr = _thread_ipr(thread)
    dia = _derive_major_dia_in(thread)
    rpm = _tap_rpm_for_dia(dia)
    ipm = rpm * ipr if ipr > 0 else 0.0

    depth_in = row.get("depth_in")
    depth_token = depth_in
    if isinstance(depth_token, str):
        if depth_token.strip().upper() in {"", "-", "THRU"}:
            depth_in = float(thickness_in or 0.0)
        else:
            try:
                depth_in = float(depth_token)
            except Exception:
                depth_in = float(thickness_in or 0.0)
    elif depth_in in (None, ""):
        depth_in = float(thickness_in or 0.0)
    else:
        try:
            depth_in = float(depth_in)
        except Exception:
            depth_in = float(thickness_in or 0.0)

    index_min = 0.08
    retract_fac = 2.0
    motion_min = (float(depth_in) / max(ipm, 1e-6)) * retract_fac if depth_in else 0.0
    t_per = motion_min + index_min

    row["ipr"] = round(ipr, 4)
    row["rpm"] = int(round(rpm))
    row["ipm"] = round(ipm, 3)
    row["t_per_hole_min"] = round(t_per, 3)
    row["feed_fmt"] = f'{row["ipr"]:.4f} ipr | {row["rpm"]} rpm | {row["ipm"]:.3f} ipm'
    row["depth_in"] = float(depth_in)
    row["depth_in_display"] = f'{float(depth_in):.2f}"'


def _finalize_tapping_rows(
    rows: Sequence[Any],
    *,
    thickness_in: float,
) -> list[MutableMapping[str, Any]]:
    """Normalize and finalize tapping rows, returning the mutable set."""

    finalized: list[MutableMapping[str, Any]] = []
    if not rows:
        return finalized

    for idx, entry in enumerate(rows):
        row_map: MutableMapping[str, Any] | None
        if isinstance(entry, dict):
            row_map = typing.cast(MutableMapping[str, Any], entry)
        elif isinstance(entry, _MutableMappingABC):
            row_map = typing.cast(MutableMapping[str, Any], entry)
        elif isinstance(entry, _MappingABC):
            try:
                row_dict = dict(entry)
            except Exception:
                continue
            row_map = typing.cast(MutableMapping[str, Any], row_dict)
            try:
                rows[idx] = row_dict  # type: ignore[index]
            except Exception:
                pass
        else:
            continue

        desc_text = str(row_map.get("desc", "") or "").upper()
        if "TAP" not in desc_text:
            continue
        _finalize_tap_row(row_map, thickness_in)
        finalized.append(row_map)

    return finalized


_COUNTERBORE_INDEX_MIN = 0.06
_COUNTERBORE_RETRACT_FACTOR = 1.3
_COUNTERBORE_EXTRA_TRAVEL_IN = 0.05
_SPOT_INDEX_MIN = 0.05
_SPOT_DEFAULT_DEPTH_IN = 0.1
_JIG_GRIND_RATE_IPM = 0.02
_JIG_GRIND_INDEX_MIN = 0.25
_MIN_IPM_DENOM = 0.1


def _format_ops_feed(ipr: float | None, rpm: float | None, ipm: float | None) -> str:
    ipr_txt = "-" if ipr is None else f"{ipr:.4f}"
    rpm_txt = "-" if rpm is None else f"{int(round(rpm))}"
    ipm_txt = "-" if ipm is None else f"{ipm:.3f}"
    return f"{ipr_txt} ipr | {rpm_txt} rpm | {ipm_txt} ipm"


def _normalize_speeds_csv_map(
    speeds_csv: Mapping[str, Any] | MutableMapping[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(speeds_csv, dict):
        return speeds_csv
    if isinstance(speeds_csv, (_MutableMappingABC, _MappingABC)):
        try:
            return dict(speeds_csv)
        except Exception:
            return None
    return None


def _runtime_counterbore(
    diameter_in: float | None,
    depth_in: float | None,
    *,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> tuple[float | None, str]:
    sfm, ipr = _lookup_sfm_ipr("counterbore", diameter_in, material_group, speeds_csv)
    rpm = _rpm_from_sfm_diam(sfm, diameter_in) if diameter_in else None
    ipm = _ipm_from_rpm_ipr(rpm, ipr)
    minutes: float | None = None
    if depth_in is not None and ipm is not None:
        travel = max(0.0, float(depth_in))
        cycle = (travel * _COUNTERBORE_RETRACT_FACTOR) + _COUNTERBORE_EXTRA_TRAVEL_IN
        minutes = (cycle / max(_MIN_IPM_DENOM, ipm)) + _COUNTERBORE_INDEX_MIN
    feed_fmt = _format_ops_feed(ipr, rpm, ipm)
    return minutes, feed_fmt


def _runtime_spot(
    depth_in: float | None,
    *,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> tuple[float | None, str]:
    tool_dia = 0.1875
    sfm, ipr = _lookup_sfm_ipr("spot", tool_dia, material_group, speeds_csv)
    rpm = _rpm_from_sfm_diam(sfm, tool_dia)
    ipm = _ipm_from_rpm_ipr(rpm, ipr)
    travel = depth_in if depth_in is not None else _SPOT_DEFAULT_DEPTH_IN
    minutes: float | None = None
    if ipm is not None:
        cycle = max(0.0, float(travel))
        minutes = (cycle / max(_MIN_IPM_DENOM, ipm)) + _SPOT_INDEX_MIN
    feed_fmt = _format_ops_feed(ipr, rpm, ipm)
    return minutes, feed_fmt


def _runtime_jig(depth_in: float | None) -> tuple[float | None, str]:
    travel = depth_in if depth_in is not None else 0.25
    ipm = _JIG_GRIND_RATE_IPM
    minutes = (max(0.0, float(travel)) / max(_MIN_IPM_DENOM, ipm)) + _JIG_GRIND_INDEX_MIN
    feed_fmt = _format_ops_feed(None, None, ipm)
    return minutes, feed_fmt


def _label_for_ops_kind(
    kind: str,
    *,
    diameter_in: float | None,
    ref_label: Any,
    ref_text: Any,
) -> str:
    base = str(ref_label or ref_text or "").strip()
    base_upper = base.upper()
    if kind == "cbore":
        if diameter_in is not None and diameter_in > 0:
            return f"{diameter_in:.4f} C’BORE"
        if any(token in base_upper for token in ("C’BORE", "CBORE", "COUNTERBORE")):
            return base
        return f"{base} C’BORE".strip()
    if kind == "spot":
        if any(token in base_upper for token in ("C’DRILL", "SPOT")):
            return base
        return (base + " C’DRILL").strip() or "C’DRILL"
    if kind in {"jig", "jig_grind"}:
        if "JIG" in base_upper:
            return base
        return (base + " JIG GRIND").strip() or "JIG GRIND"
    return base or kind


def _depth_from_desc(text: str) -> float | None:
    if not text:
        return None
    match = _DEPTH_TOKEN.search(text)
    if match:
        return _first_numeric_or_none(match.group(1))
    match = RE_DEPTH.search(text)
    if match:
        return _first_numeric_or_none(match.group(1))
    return None


def _extract_sides_from_desc(desc_upper: str) -> list[str]:
    if _SIDE_BOTH.search(desc_upper):
        return ["FRONT", "BACK"]
    if _SIDE_BACK.search(desc_upper):
        return ["BACK"]
    if _SIDE_FRONT.search(desc_upper):
        return ["FRONT"]
    return ["FRONT"]


def _runtime_for_kind(
    kind: str,
    *,
    diameter_in: float | None,
    depth_in: float | None,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> tuple[float | None, str]:
    if kind == "cbore":
        return _runtime_counterbore(diameter_in, depth_in, material_group=material_group, speeds_csv=speeds_csv)
    if kind == "spot":
        return _runtime_spot(depth_in, material_group=material_group, speeds_csv=speeds_csv)
    if kind in {"jig", "jig_grind"}:
        return _runtime_jig(depth_in)
    return None, _format_ops_feed(None, None, None)


def _build_ops_card_row(
    *,
    kind: str,
    label: str,
    qty: int,
    side: str,
    diameter_in: float | None,
    depth_in: float | None,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> dict[str, Any]:
    side_norm = str(side or "").upper()
    if side_norm not in {"FRONT", "BACK"}:
        side_norm = "FRONT"
    minutes, feed_fmt = _runtime_for_kind(
        kind,
        diameter_in=diameter_in,
        depth_in=depth_in,
        material_group=material_group,
        speeds_csv=speeds_csv,
    )
    minutes_val = float(minutes) if minutes is not None and math.isfinite(minutes) else 0.0
    return {
        "label": label or kind,
        "qty": int(qty),
        "side": side_norm,
        "depth_in": float(depth_in) if depth_in is not None and math.isfinite(float(depth_in)) else None,
        "depth_in_display": f'{float(depth_in):.2f}"' if depth_in is not None else "-",
        "feed_fmt": feed_fmt,
        "t_per_hole_min": round(minutes_val, 3),
    }


def _ops_card_rows_from_group_totals(
    ops_summary: Mapping[str, Any] | MutableMapping[str, Any] | None,
    *,
    kind: str,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(ops_summary, _MappingABC):
        return []
    group_totals = ops_summary.get("group_totals")
    if not isinstance(group_totals, _MappingABC):
        return []
    type_map = group_totals.get(kind)
    if not isinstance(type_map, _MappingABC):
        return []
    rows: list[dict[str, Any]] = []
    for side_map in type_map.values():
        if not isinstance(side_map, _MappingABC):
            continue
        for side_key, payload in side_map.items():
            if not isinstance(payload, _MappingABC):
                continue
            qty = _coerce_int_or_zero(payload.get("qty"))
            if qty <= 0:
                continue
            side_norm = str(side_key or "").upper()
            depth_in = _first_numeric_or_none(
                payload.get("depth_in_avg"),
                payload.get("depth_in"),
                payload.get("depth_in_max"),
                payload.get("depth_in_min"),
            )
            diameter_in = _first_numeric_or_none(
                payload.get("diameter_in"),
                payload.get("ref_dia_in"),
                _parse_ref_to_inch(payload.get("ref_label")),
                _parse_ref_to_inch(payload.get("ref")),
            )
            label = _label_for_ops_kind(
                kind,
                diameter_in=diameter_in,
                ref_label=payload.get("ref_label"),
                ref_text=payload.get("ref"),
            )
            rows.append(
                _build_ops_card_row(
                    kind=kind,
                    label=label,
                    qty=qty,
                    side=side_norm,
                    diameter_in=diameter_in,
                    depth_in=depth_in,
                    material_group=material_group,
                    speeds_csv=speeds_csv,
                )
            )
    return rows


def _ops_card_rows_from_detail(
    ops_summary: Mapping[str, Any] | MutableMapping[str, Any] | None,
    *,
    kind: str,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(ops_summary, _MappingABC):
        return []
    detail_obj = ops_summary.get("rows_detail")
    if not isinstance(detail_obj, list):
        return []
    aggregated: dict[tuple[str, str, float | None, float | None], dict[str, Any]] = {}
    for entry in detail_obj:
        if not isinstance(entry, _MappingABC):
            continue
        entry_type = str(entry.get("type") or "").lower()
        if entry_type != kind:
            continue
        qty = _coerce_int_or_zero(entry.get("qty"))
        if qty <= 0:
            continue
        sides = list(entry.get("sides") or [])
        if not sides:
            side_single = entry.get("side")
            if side_single:
                sides = [side_single]
        if not sides:
            sides = ["FRONT"]
        diameter_in = _first_numeric_or_none(
            entry.get("ref_dia_in"),
            entry.get("diameter_in"),
            _parse_ref_to_inch(entry.get("ref_label")),
            _parse_ref_to_inch(entry.get("ref")),
        )
        depth_in = _first_numeric_or_none(entry.get("depth_in"))
        label = _label_for_ops_kind(
            kind,
            diameter_in=diameter_in,
            ref_label=entry.get("ref_label"),
            ref_text=entry.get("ref"),
        )
        for side in sides:
            side_norm = str(side or "").upper() or "FRONT"
            key = (
                label,
                side_norm,
                round(diameter_in, 5) if diameter_in is not None else None,
                round(depth_in, 5) if depth_in is not None else None,
            )
            bucket = aggregated.setdefault(
                key,
                {
                    "label": label,
                    "qty": 0,
                    "side": side_norm,
                    "diameter_in": diameter_in,
                    "_depths": [],
                },
            )
            bucket["qty"] += qty
            if depth_in is not None:
                bucket.setdefault("_depths", []).append(float(depth_in))
            if bucket.get("diameter_in") is None and diameter_in is not None:
                bucket["diameter_in"] = diameter_in
    rows: list[dict[str, Any]] = []
    for bucket in aggregated.values():
        depth_values = [float(val) for val in bucket.pop("_depths", []) if isinstance(val, (int, float))]
        depth_in = sum(depth_values) / len(depth_values) if depth_values else None
        rows.append(
            _build_ops_card_row(
                kind=kind,
                label=bucket.get("label", kind),
                qty=bucket.get("qty", 0),
                side=bucket.get("side", "FRONT"),
                diameter_in=bucket.get("diameter_in"),
                depth_in=depth_in,
                material_group=material_group,
                speeds_csv=speeds_csv,
            )
        )
    return rows


def _ops_card_rows_from_simple(
    rows: Sequence[Any] | None,
    *,
    kind: str,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    if not rows:
        return extracted
    for entry in rows:
        if not isinstance(entry, _MappingABC):
            if isinstance(entry, dict):
                entry_map = entry
            else:
                continue
        else:
            entry_map = entry
        qty = _coerce_int_or_zero(entry_map.get("qty"))
        if qty <= 0:
            continue
        desc = str(entry_map.get("desc") or "")
        if not desc:
            continue
        desc_upper = desc.upper()
        if kind == "cbore" and not any(token in desc_upper for token in ("C’BORE", "CBORE", "COUNTERBORE")):
            continue
        if kind == "spot" and not any(token in desc_upper for token in ("SPOT", "C’DRILL", "CENTER DRILL")):
            continue
        if kind in {"jig", "jig_grind"} and "JIG" not in desc_upper:
            continue
        sides = _extract_sides_from_desc(desc_upper)
        diameter_in = _first_numeric_or_none(
            entry_map.get("diameter_in"),
            _parse_ref_to_inch(entry_map.get("ref")),
            _parse_ref_to_inch(desc),
        )
        depth_in = _first_numeric_or_none(entry_map.get("depth_in"), _depth_from_desc(desc))
        label = _label_for_ops_kind(
            kind,
            diameter_in=diameter_in,
            ref_label=entry_map.get("label"),
            ref_text=desc,
        )
        feed_fmt = str(entry_map.get("feed_fmt") or "").strip()
        minutes_raw = entry_map.get("t_per_hole_min")
        try:
            minutes_val = float(minutes_raw)
            if not math.isfinite(minutes_val) or minutes_val < 0:
                minutes_val = None
        except Exception:
            minutes_val = None
        if minutes_val is None:
            minutes_calc, feed_calc = _runtime_for_kind(
                kind,
                diameter_in=diameter_in,
                depth_in=depth_in,
                material_group=material_group,
                speeds_csv=speeds_csv,
            )
            if minutes_calc is not None:
                minutes_val = float(minutes_calc)
            if not feed_fmt:
                feed_fmt = feed_calc
        if not feed_fmt:
            _, feed_calc = _runtime_for_kind(
                kind,
                diameter_in=diameter_in,
                depth_in=depth_in,
                material_group=material_group,
                speeds_csv=speeds_csv,
            )
            feed_fmt = feed_calc
        minutes_clean = minutes_val if minutes_val is not None and math.isfinite(minutes_val) else 0.0
        for side in sides:
            row = _build_ops_card_row(
                kind=kind,
                label=label,
                qty=qty,
                side=side,
                diameter_in=diameter_in,
                depth_in=depth_in,
                material_group=material_group,
                speeds_csv=speeds_csv,
            )
            row["feed_fmt"] = feed_fmt or row.get("feed_fmt")
            row["t_per_hole_min"] = round(minutes_clean, 3)
            extracted.append(row)
    return extracted


def _collect_ops_rows_for_kind(
    *,
    ops_summary: Mapping[str, Any] | MutableMapping[str, Any] | None,
    rows: Sequence[Any] | None,
    kind: str,
    material_group: str | None,
    speeds_csv: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows_from_totals = _ops_card_rows_from_group_totals(
        ops_summary,
        kind=kind,
        material_group=material_group,
        speeds_csv=speeds_csv,
    )
    if rows_from_totals:
        return rows_from_totals
    rows_from_detail = _ops_card_rows_from_detail(
        ops_summary,
        kind=kind,
        material_group=material_group,
        speeds_csv=speeds_csv,
    )
    if rows_from_detail:
        return rows_from_detail
    return _ops_card_rows_from_simple(
        rows,
        kind=kind,
        material_group=material_group,
        speeds_csv=speeds_csv,
    )


def _render_ops_card(
    append_line: Callable[[str], None],
    *,
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> float:
    if not rows:
        return 0.0
    append_line(title.upper())
    append_line("-" * 66)
    total_min = 0.0
    for r in rows:
        qty = int(r.get("qty") or 0)
        t_ph = float(r.get("t_per_hole_min") or 0.0)
        grp = qty * t_ph
        total_min += grp
        depth_display = r.get("depth_in_display")
        if not depth_display:
            try:
                depth_val = float(r.get("depth_in", 0.0))
                depth_display = f"{depth_val:.3f}\""
            except Exception:
                depth_display = "-"
        feed_fmt = r.get("feed_fmt", "-")
        append_line(
            f'{r.get("label", r.get("desc", "?"))} × {qty}  '
            f'({(r.get("side", "") or "").upper() or "FRONT"}) | '
            f'depth {depth_display} | '
            f'{feed_fmt} | '
            f't/hole {t_ph:.2f} min | group {qty}×{t_ph:.2f} = {grp:.2f} min'
        )
    append_line("")
    return round(total_min, 2)


def _side_from(txt: str) -> str:
    text = str(txt or "").lower()
    if "(front" in text:
        return "front"
    if "(back" in text:
        return "back"
    return "unspecified"


def summarize_actions(
    removal_lines: list[str],
    planner_ops: list[dict],
    extra_bucket_ops: Mapping[str, Any] | None = None,
) -> None:
    """Log aggregated removal + planner operation counts for diagnostics."""

    from collections import defaultdict
    import re

    # Counters (totals + per side)
    total = defaultdict(int)
    by_side = defaultdict(lambda: defaultdict(int))

    # --- From MATERIAL REMOVAL lines (Drilling/Tapping sections) ---
    # Example lines:
    #  'Dia 0.281" × 20  (FRONT) | ...'
    #  '#10-32 TAP THRU × 2  (FRONT) | ...'
    drill_re = re.compile(r'^Dia\s+[\d\.]+"[ ]*[x×]\s*(\d+).*(\(.*?\))?', re.IGNORECASE)
    tap_re = re.compile(r'^\s*#?\d.*\bTAP\b.*[x×]\s+(\d+).*(\(.*?\))?', re.IGNORECASE)
    cbore_re = re.compile(
        r'^\s*(?:MATERIAL REMOVAL – COUNTERBORE|Ø[0-9.]+".*\sC’BORE|\bCBORE\b).*',
        re.IGNORECASE,
    )
    spotln_re = re.compile(r'^\s*MATERIAL REMOVAL – SPOT|Spot drill ×\s*(\d+)', re.IGNORECASE)
    jigln_re = re.compile(r'^\s*MATERIAL REMOVAL – JIG GRIND|Jig grind ×\s*(\d+)', re.IGNORECASE)
    qty_re = re.compile(r'[×x]\s*(\d+)')
    cbo_hdr_re = re.compile(r'^\s*MATERIAL\s+REMOVAL\s*[–-]\s*COUNTERBORE', re.IGNORECASE)
    cbo_line_re = re.compile(
        r'^\s*(?:Ø|%%[Cc])?\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+|\d+/\d+).*?[×xX]\s*(\d+).*?\((FRONT|BACK)\)',
        re.IGNORECASE,
    )
    cbo_line_noside_re = re.compile(
        r'^\s*(?:Ø|%%[Cc])?\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+|\d+/\d+).*?[×xX]\s*(\d+)',
        re.IGNORECASE,
    )
    cbo_simple_re = re.compile(
        r'^\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+)"?\s*[×xX]\s*(\d+)\s*\((FRONT|BACK)\)',
        re.IGNORECASE,
    )
    cbo_card_row_re = re.compile(
        r'^\s*Ø\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+)"?.*?[×xX]\s*(\d+)\s*\((FRONT|BACK)\)',
        re.IGNORECASE,
    )
    spot_card_row_re = re.compile(
        r'^\s*Spot\s+drill.*?[×xX]\s*(\d+)\s*\((FRONT|BACK)\)',
        re.IGNORECASE,
    )
    jig_card_row_re = re.compile(
        r'^\s*Jig\s+grind.*?[×xX]\s*(\d+)\s*(?:\((FRONT|BACK)\))?',
        re.IGNORECASE,
    )

    card_counts = {"counterbore": False, "spot": False, "jig_grind": False}
    active_card: str | None = None
    in_cbo = False

    for ln in removal_lines or []:
        if not isinstance(ln, str):
            continue

        u = ln.upper()
        header = u.strip()
        stripped = ln.strip()

        if cbo_hdr_re.search(header):
            in_cbo = True
            continue

        if in_cbo:
            simple_cbo = cbo_card_row_re.search(ln)
            if simple_cbo:
                qty = int(simple_cbo.group(2))
                side = simple_cbo.group(3).upper()
                total["counterbore"] += qty
                by_side["counterbore"][side.lower()] += qty
                card_counts["counterbore"] = True
                continue
            m = cbo_line_re.search(ln)
            if m:
                qty = int(m.group(2))
                side = m.group(3).upper()
                total["counterbore"] += qty
                by_side["counterbore"][side.lower()] += qty
                card_counts["counterbore"] = True
                continue
            m2 = cbo_line_noside_re.search(ln)
            if m2:
                qty = int(m2.group(2))
                total["counterbore"] += qty
                by_side["counterbore"]["front"] += qty
                card_counts["counterbore"] = True
                continue
            m3 = cbo_simple_re.search(ln)
            if m3:
                qty = int(m3.group(2))
                side = m3.group(3).upper()
                total["counterbore"] += qty
                by_side["counterbore"][side.lower()] += qty
                card_counts["counterbore"] = True
                continue

        if in_cbo and header.startswith("MATERIAL REMOVAL"):
            in_cbo = False

        if header.startswith("MATERIAL REMOVAL –"):
            if "COUNTERBORE" in header:
                active_card = "counterbore"
            elif "SPOT" in header:
                active_card = "spot"
            elif "JIG" in header:
                active_card = "jig_grind"
            elif "TAPPING" in header:
                active_card = "tap"
            else:
                active_card = None
            continue

        if not stripped:
            active_card = None
            in_cbo = False
            continue

        if stripped.startswith("-") or stripped.startswith("="):
            continue

        if not in_cbo:
            cb_simple_line = cbo_card_row_re.search(ln)
            if cb_simple_line:
                qty = int(cb_simple_line.group(2))
                side = cb_simple_line.group(3).upper()
                total["counterbore"] += qty
                by_side["counterbore"][side.lower()] += qty
                card_counts["counterbore"] = True
                continue

        if active_card != "spot":
            spot_simple_line = spot_card_row_re.search(ln)
            if spot_simple_line:
                qty = int(spot_simple_line.group(1))
                side = spot_simple_line.group(2).upper()
                total["spot"] += qty
                by_side["spot"][side.lower()] += qty
                card_counts["spot"] = True
                continue

        if active_card != "jig_grind":
            jig_simple_line = jig_card_row_re.search(ln)
            if jig_simple_line:
                qty = int(jig_simple_line.group(1))
                side_match = jig_simple_line.group(2) if jig_simple_line.lastindex and jig_simple_line.group(2) else None
                side = (side_match or _side_from(ln)).lower()
                total["jig_grind"] += qty
                by_side["jig_grind"][side] += qty
                card_counts["jig_grind"] = True
                continue

        if active_card in {"spot", "jig_grind"}:
            if active_card == "spot":
                spot_simple = spot_card_row_re.search(ln)
                if spot_simple:
                    qty = int(spot_simple.group(1))
                    side = _side_from(ln)
                    total["spot"] += qty
                    by_side["spot"][side] += qty
                    card_counts["spot"] = True
                    continue
            if active_card == "jig_grind":
                jig_simple = jig_card_row_re.search(ln)
                if jig_simple:
                    qty = int(jig_simple.group(1))
                    side = _side_from(ln)
                    total["jig_grind"] += qty
                    by_side["jig_grind"][side] += qty
                    card_counts["jig_grind"] = True
                    continue
            qty_match = qty_re.search(ln)
            if qty_match:
                qty = int(qty_match.group(1))
                side = _side_from(ln)
                total[active_card] += qty
                by_side[active_card][side] += qty
                card_counts[active_card] = True
            continue

        m = drill_re.search(ln)
        if m:
            qty = int(m.group(1))
            side = _side_from(ln)
            total["drill"] += qty
            by_side["drill"][side] += qty
            continue
        m = tap_re.search(ln)
        if m:
            qty = int(m.group(1))
            side = _side_from(ln)
            total["tap"] += qty
            by_side["tap"][side] += qty
            continue

        if not card_counts["counterbore"] and (cbore_re.search(ln) or "C’BORE GROUPS" in ln or "CBORE" in ln):
            m_qty = re.search(r'[×x]\s*(\d+)', ln)
            if m_qty:
                qty = int(m_qty.group(1))
                side = _side_from(ln)
                total["counterbore"] += qty
                by_side["counterbore"][side] += qty
                continue

        m = spotln_re.search(ln)
        if not card_counts["spot"] and m:
            qty = int(m.group(1)) if m.lastindex else 0
            if qty:
                total["spot"] += qty
                by_side["spot"]["front"] += qty
                continue

        m = jigln_re.search(ln)
        if not card_counts["jig_grind"] and m:
            qty = int(m.group(1)) if m.lastindex else 0
            if qty:
                total["jig_grind"] += qty
                by_side["jig_grind"]["unspecified"] += qty
                continue

    # --- From planner ops (Counterbore / Spot / Jig-grind) ---
    # Expect planner_ops items like {'name': 'counterbore', 'qty': 3, 'side': 'FRONT'}
    for op in planner_ops or []:
        if not isinstance(op, dict):
            continue
        name = (op.get("name", "") or "").lower()
        qty_raw = op.get("qty", 0)
        try:
            qty = int(float(qty_raw))
        except Exception:
            qty = 0
        if qty <= 0:
            continue
        side = (op.get("side") or "unspecified").lower()

        if "counterbore" in name or "c-bore" in name or "cbore" in name:
            if card_counts["counterbore"]:
                continue
            total["counterbore"] += qty
            by_side["counterbore"][side] += qty
        elif "spot" in name and "drill" in name:
            if card_counts["spot"]:
                continue
            total["spot"] += qty
            by_side["spot"][side] += qty
        elif "jig" in name and "grind" in name:
            if card_counts["jig_grind"]:
                continue
            total["jig_grind"] += qty
            by_side["jig_grind"][side] += qty

    # --- From extra bucket ops (fallback data published during pricing) ---
    def _bucket_key(name: str | None) -> str:
        if not isinstance(name, str):
            return ""
        bucket = name.strip().lower()
        if not bucket:
            return ""
        if bucket in {"drill", "drills", "drilling"}:
            return "drill"
        if bucket in {"tap", "taps", "tapping"}:
            return "tap"
        if bucket in {"counterbore", "c'bore", "cbore", "counter bore"}:
            return "counterbore"
        if bucket in {"spot", "spot drill", "spot-drill", "spot_drill"}:
            return "spot"
        if bucket in {"jig-grind", "jig_grind", "jig grind"}:
            return "jig_grind"
        if bucket in {"counterdrill", "counter-drill", "c'drill"}:
            return "counterdrill"
        return ""

    def _normalize_side_value(val: Any) -> str:
        if val is None:
            return "unspecified"
        if isinstance(val, str):
            side_txt = val.strip().lower()
        else:
            side_txt = str(val).strip().lower()
        if side_txt in {"front", "f"}:
            return "front"
        if side_txt in {"back", "b"}:
            return "back"
        if side_txt in {"both", "front/back", "front & back", "both sides", "both-sides"}:
            return "both"
        return "unspecified"

    def _iter_extra_entries(payload: Any) -> Iterable[Any]:
        if payload is None:
            return []
        if isinstance(payload, _MappingABC):
            rows_candidate = payload.get("rows")
            if isinstance(rows_candidate, Sequence) and not isinstance(rows_candidate, (str, bytes)):
                return rows_candidate
            return [payload]
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            return payload
        return []

    extra_totals = defaultdict(int)
    extra_by_side = defaultdict(lambda: defaultdict(int))

    if isinstance(extra_bucket_ops, _MappingABC):
        for bucket_name, entries in extra_bucket_ops.items():
            key = _bucket_key(bucket_name)
            if not key:
                continue
            for entry in _iter_extra_entries(entries):
                if isinstance(entry, _MappingABC):
                    qty_val = entry.get("qty")
                    side_val = entry.get("side")
                else:
                    qty_val = getattr(entry, "qty", None)
                    side_val = getattr(entry, "side", None)
                try:
                    qty = int(round(float(qty_val)))
                except Exception:
                    continue
                if qty <= 0:
                    continue
                side_norm = _normalize_side_value(side_val)
                if side_norm == "both":
                    extra_by_side[key]["front"] += qty
                    extra_by_side[key]["back"] += qty
                else:
                    extra_by_side[key][side_norm] += qty
                extra_totals[key] += qty

    for key, qty in extra_totals.items():
        if qty <= 0:
            continue
        if total[key] <= 0:
            total[key] += qty
            for side_label, side_qty in extra_by_side[key].items():
                if side_qty > 0:
                    by_side[key][side_label] += side_qty

    actions = {
        "Drills": int(total.get("drill", 0)),
        "Taps": int(total.get("tap", 0)),
        "Counterbores": int(total.get("counterbore", 0)),
        "Spot": int(total.get("spot", 0)),
        "Jig-grind": int(total.get("jig_grind", 0)),
    }
    actions_total = sum(actions.values())

    print(f"[ACTIONS] totals={dict(total)} total={actions_total}")
    for key, sides in by_side.items():
        print(f"[ACTIONS/{key}] by_side={dict(sides)}")

    print("Operation Counts")
    print("--------------------------------------------------------------------------")
    for label, count in actions.items():
        print(f"  {label:<14} {count}")
    print(f"  {'Actions total':<14} {actions_total}")


def _extract_milling_bucket(
    bucket_view: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(bucket_view, _MappingABC):
        return None
    try:
        buckets_obj = bucket_view.get("buckets")
    except Exception:
        buckets_obj = None
    if isinstance(buckets_obj, _MappingABC):
        candidate = buckets_obj.get("milling")
        if isinstance(candidate, _MappingABC):
            return typing.cast(Mapping[str, Any], candidate)
        if isinstance(candidate, dict):
            return candidate
    elif isinstance(buckets_obj, dict):
        candidate = buckets_obj.get("milling")
        if isinstance(candidate, _MappingABC):
            return typing.cast(Mapping[str, Any], candidate)
        if isinstance(candidate, dict):
            return candidate
    return None


def _render_milling_removal_card(
    append_line: Callable[[str], None],
    lines: Sequence[Any] | None,
    milling_bucket: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(milling_bucket, _MappingABC):
        return False

    existing = False
    if isinstance(lines, Sequence):
        for entry in lines:
            if not isinstance(entry, str):
                continue
            if entry.strip().upper().startswith("MATERIAL REMOVAL – MILLING"):
                existing = True
                break
    if existing:
        return False

    detail_obj = milling_bucket.get("detail") if isinstance(milling_bucket, _MappingABC) else None
    if not isinstance(detail_obj, _MappingABC):
        return False

    def _detail_int(key: str, default: int = 0) -> int:
        value = detail_obj.get(key)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(round(float(value)))
        if isinstance(value, str):
            try:
                parsed = float(value.strip())
            except Exception:
                return default
            if math.isfinite(parsed):
                return int(round(parsed))
        return default

    def _detail_float(key: str, default: float = 0.0) -> float:
        val = _coerce_float_or_none(detail_obj.get(key))
        if val is None or not math.isfinite(float(val)):
            return float(default)
        return float(val)

    passes_face = _detail_int("passes_face")
    passes_axial = _detail_int("passes_axial")
    rpm_face = _detail_int("rpm_face")
    rpm_contour = _detail_int("rpm_contour")
    ipm_face = _detail_float("ipm_face")
    ipm_contour = _detail_float("ipm_contour")
    face_top_min = _detail_float("face_top_min")
    face_bot_min = _detail_float("face_bot_min")
    perim_rough_min = _detail_float("perim_rough_min")
    perim_finish_min = _detail_float("perim_finish_min")
    toolchanges_min = _detail_float("toolchanges_min")

    if (
        passes_face <= 0
        and passes_axial <= 0
        and face_top_min <= 0
        and face_bot_min <= 0
        and perim_rough_min <= 0
        and perim_finish_min <= 0
        and toolchanges_min <= 0
    ):
        return False

    total_minutes_val = _coerce_float_or_none(milling_bucket.get("minutes"))
    if total_minutes_val is None or not math.isfinite(total_minutes_val):
        total_minutes_val = (
            face_top_min
            + max(0.0, face_bot_min)
            + perim_rough_min
            + perim_finish_min
            + max(0.0, toolchanges_min)
        )

    append_line("MATERIAL REMOVAL – MILLING")
    append_line("-" * 66)
    append_line(
        "FACE TOP | passes {passes} | {rpm} rpm | {ipm:.1f} ipm | t  {minutes:.2f} min".format(
            passes=passes_face,
            rpm=rpm_face,
            ipm=ipm_face,
            minutes=face_top_min,
        )
    )
    if face_bot_min > 0:
        append_line(
            "FACE BOT | passes {passes} | {rpm} rpm | {ipm:.1f} ipm | t  {minutes:.2f} min".format(
                passes=passes_face,
                rpm=rpm_face,
                ipm=ipm_face,
                minutes=face_bot_min,
            )
        )
    append_line(
        "PERIM ROUGH | axial passes {passes} | {rpm} rpm | {ipm:.1f} ipm | t  {minutes:.2f} min".format(
            passes=passes_axial,
            rpm=rpm_contour,
            ipm=ipm_contour,
            minutes=perim_rough_min,
        )
    )
    append_line(
        "PERIM FINISH | {rpm} rpm | {ipm:.1f} ipm | t  {minutes:.2f} min".format(
            rpm=rpm_contour,
            ipm=ipm_contour,
            minutes=perim_finish_min,
        )
    )
    if toolchanges_min > 0:
        append_line(f"Toolchange adders: {toolchanges_min:.2f} min")
    append_line("-" * 66)
    append_line(
        "TOTAL MILLING . {minutes:.2f} min  ({hours:.2f} hr)".format(
            minutes=total_minutes_val,
            hours=minutes_to_hours(total_minutes_val),
        )
    )
    append_line("")
    return True


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
    speeds_feeds_table: Any | None = None,
    material_group: str | None = None,
    drilling_time_per_hole: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], list[str], Mapping[str, Any] | None]:
    """Return drill removal render lines + extras while updating breakdown state."""

    lines: list[str] = []

    breakdown_mutable: MutableMapping[str, Any] | None
    if isinstance(breakdown, _MutableMappingABC):
        breakdown_mutable = typing.cast(MutableMapping[str, Any], breakdown)
    elif isinstance(breakdown, dict):
        breakdown_mutable = breakdown
    else:
        breakdown_mutable = None

    def _push(target: list[str], text: Any) -> None:
        try:
            target.append(str(text))
        except Exception:
            pass

    def _sum_count_values(candidate: Any) -> int:
        """Best-effort sum of numeric values from mappings or sequences."""

        if isinstance(candidate, (_MappingABC, dict)):
            values = candidate.values()  # type: ignore[assignment]
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            values = candidate
        else:
            return 0

        total = 0
        for value in values:
            try:
                total += int(round(float(value)))
            except Exception:
                try:
                    total += int(value)  # type: ignore[arg-type]
                except Exception:
                    continue
        return total

    drill_bins_raw_total = 0
    drill_bins_adj_total = 0

    pricing_buckets: MutableMapping[str, Any] | dict[str, Any] = {}
    bucket_view_obj: MutableMapping[str, Any] | Mapping[str, Any] | None = None
    try:
        if isinstance(breakdown, dict):
            bucket_view_obj = breakdown.setdefault("bucket_view", {})
        elif isinstance(breakdown, _MutableMappingABC):
            bucket_view_obj = typing.cast(
                MutableMapping[str, Any],
                breakdown.setdefault("bucket_view", {}),
            )
        elif isinstance(breakdown, _MappingABC):
            bucket_view_obj = typing.cast(Mapping[str, Any], breakdown.get("bucket_view"))
    except Exception:
        bucket_view_obj = None

    buckets_obj: MutableMapping[str, Any] | Mapping[str, Any] | None = None
    if isinstance(bucket_view_obj, _MutableMappingABC):
        buckets_obj = bucket_view_obj.setdefault("buckets", {})
    elif isinstance(bucket_view_obj, dict):
        buckets_obj = bucket_view_obj.setdefault("buckets", {})
    elif isinstance(bucket_view_obj, _MappingABC):
        buckets_obj = bucket_view_obj.get("buckets")

    if isinstance(buckets_obj, _MutableMappingABC):
        pricing_buckets = buckets_obj
    elif isinstance(buckets_obj, dict):
        pricing_buckets = buckets_obj
        if isinstance(bucket_view_obj, _MutableMappingABC):
            bucket_view_obj["buckets"] = pricing_buckets
        elif isinstance(bucket_view_obj, dict):
            bucket_view_obj["buckets"] = pricing_buckets
    elif isinstance(buckets_obj, _MappingABC):
        try:
            pricing_buckets = dict(buckets_obj)
            if isinstance(bucket_view_obj, _MutableMappingABC):
                bucket_view_obj["buckets"] = pricing_buckets
            elif isinstance(bucket_view_obj, dict):
                bucket_view_obj["buckets"] = pricing_buckets
        except Exception:
            pricing_buckets = {}

    def _seed_bucket_minutes(bucket_key: str, minutes: float) -> None:
        try:
            minutes_val = float(minutes)
        except Exception:
            return
        if not isinstance(pricing_buckets, (_MutableMappingABC, dict)):
            return
        try:
            entry = pricing_buckets.get(bucket_key) if isinstance(pricing_buckets, Mapping) else None
            if not isinstance(entry, _MutableMappingABC):
                if isinstance(entry, _MappingABC):
                    entry = dict(entry)
                elif isinstance(entry, dict):
                    entry = entry
                else:
                    entry = {}
            entry["minutes"] = minutes_val
            if isinstance(pricing_buckets, _MutableMappingABC):
                pricing_buckets[bucket_key] = entry  # type: ignore[index]
            else:
                pricing_buckets[bucket_key] = entry
        except Exception:
            pass
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
    try:
        geo_map = (
            ((breakdown or {}).get("geo") if isinstance(breakdown, dict) else {})
            or ((result or {}).get("geo") if isinstance(result, dict) else {})
            or {}
        )
    except NameError:
        geo_map = (
            ((breakdown or {}).get("geo") if isinstance(breakdown, dict) else {})
            or {}
        )

    if isinstance(geo_map, _MappingABC) and not isinstance(geo_map, dict):
        try:
            geo_map = dict(geo_map)
        except Exception:
            geo_map = {}
    if not isinstance(geo_map, dict):
        geo_map = {}

    fam = (
        geo_map.get("hole_diam_families_geom_in")
        or geo_map.get("hole_diam_families_in")
        or {}
    )
    counts_by_diam_raw: dict[float, int] = {}
    for k, v in (fam or {}).items():
        try:
            counts_by_diam_raw[float(k)] = int(v)
        except Exception:
            continue

    if not counts_by_diam_raw or sum(int(v) for v in counts_by_diam_raw.values()) == 0:
        counts_by_diam_raw = _seed_drill_bins_from_geo(geo_map)

    _push(lines, f"[DEBUG] drill_families_from_geo={sum(counts_by_diam_raw.values())}")

    (
        tap_minutes_inferred,
        cbore_minutes_inferred,
        spot_minutes_inferred,
        jig_minutes_inferred,
    ) = _hole_table_minutes_from_geo(geo_map)

    inferred_minutes = {
        "tapping": tap_minutes_inferred,
        "counterbore": cbore_minutes_inferred,
        "drilling": spot_minutes_inferred,
        "grinding": jig_minutes_inferred,
    }

    if any(minutes > 0.0 for minutes in inferred_minutes.values()):
        seeded_via_planner = False
        if isinstance(breakdown, (_MutableMappingABC, dict)):
            try:
                _planner_seed_bucket_minutes(
                    typing.cast(MutableMapping[str, Any], breakdown),
                    tapping_min=tap_minutes_inferred,
                    cbore_min=cbore_minutes_inferred,
                    spot_min=spot_minutes_inferred,
                    jig_min=jig_minutes_inferred,
                )
                seeded_via_planner = True
            except Exception:
                seeded_via_planner = False

        if not seeded_via_planner:
            for bucket_key, minutes in inferred_minutes.items():
                if minutes <= 0.0:
                    continue
                existing_minutes = 0.0
                if isinstance(pricing_buckets, Mapping):
                    current_entry = pricing_buckets.get(bucket_key)
                    if isinstance(current_entry, _MappingABC):
                        existing_minutes = float(
                            _coerce_float_or_none(current_entry.get("minutes")) or 0.0
                        )
                    elif isinstance(current_entry, dict):
                        try:
                            existing_minutes = float(current_entry.get("minutes") or 0.0)
                        except Exception:
                            existing_minutes = 0.0
                _seed_bucket_minutes(bucket_key, existing_minutes + float(minutes))

    ops_hole_count_from_table = 0
    drill_actions_from_groups = 0
    ops_claims: dict[str, Any] = {}

    dtph_map_candidate = (
        drilling_time_per_hole
        if isinstance(drilling_time_per_hole, _MappingABC)
        else None
    )
    dtph_map: Mapping[str, Any]
    if isinstance(dtph_map_candidate, _MappingABC):
        dtph_map = typing.cast(Mapping[str, Any], dtph_map_candidate)
    else:
        dtph_map = {}

    dtph_rows = dtph_map.get("rows")
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
            geo_map_for_drill: Mapping[str, Any] | dict[str, Any]
            if isinstance(geo_map, (_MappingABC, dict)) and geo_map:
                geo_map_for_drill = geo_map
            else:
                geo_map_for_drill = {}
                if isinstance(breakdown, (_MappingABC, dict)):
                    try:
                        geo_candidate = breakdown.get("geo")  # type: ignore[index]
                    except Exception:
                        geo_candidate = None
                    if isinstance(geo_candidate, (_MappingABC, dict)):
                        geo_map_for_drill = geo_candidate
                if not geo_map_for_drill:
                    try:
                        result_map = (
                            result
                            if isinstance(result, (_MappingABC, dict))
                            else None
                        )
                    except NameError:
                        result_map = None
                    if isinstance(result_map, (_MappingABC, dict)):
                        geo_candidate = result_map.get("geo")
                        if isinstance(geo_candidate, (_MappingABC, dict)):
                            geo_map_for_drill = geo_candidate
            if not isinstance(geo_map_for_drill, (_MappingABC, dict)):
                geo_map_for_drill = {}

            chart_lines_all = list((geo_map_for_drill.get("chart_lines") or []))
            joined_lines = _join_wrapped_chart_lines(
                [_clean_mtext(x) for x in chart_lines_all]
            )
            ops_claims_raw = _parse_ops_and_claims(joined_lines)
            if isinstance(ops_claims_raw, (_MappingABC, dict)):
                ops_claims = dict(ops_claims_raw)
            else:
                ops_claims = {}

            pilot_from_rows = _collect_pilot_claims_from_rows(geo_map_for_drill)
            pilot_from_chart = list(ops_claims.get("claimed_pilot_diams") or [])
            ops_claims["claimed_pilot_diams"] = pilot_from_rows + pilot_from_chart

            ops_hint: dict[str, Any] = {}
            try:
                hint_payload = geo_map_for_drill.get("ops_totals_hint")
            except Exception:
                hint_payload = None
            if isinstance(hint_payload, (_MappingABC, dict)):
                ops_hint = dict(hint_payload)

            if counts_by_diam_raw:
                counts_source = counts_by_diam_raw
            else:
                fallback_counts: dict[float, int] = {}
                for row in sanitized_rows:
                    key = round(float(row.get("diameter_in", 0.0) or 0.0), 4)
                    qty_val = int(row.get("qty", 0))
                    if qty_val <= 0:
                        continue
                    fallback_counts[key] = fallback_counts.get(key, 0) + qty_val
                counts_by_diam_raw = fallback_counts
                counts_source = counts_by_diam_raw

            geo_map: dict[str, Any] = {}
            if isinstance(breakdown, dict):
                geo_candidate = breakdown.get("geo")
                if isinstance(geo_candidate, dict):
                    geo_map = geo_candidate
                elif isinstance(geo_candidate, _MappingABC):
                    try:
                        geo_map = dict(geo_candidate)
                    except Exception:
                        geo_map = {}
            if not geo_map:
                try:
                    result_map = result if isinstance(result, dict) else {}
                except NameError:
                    result_map = {}
                geo_candidate = (
                    result_map.get("geo") if isinstance(result_map, dict) else {}
                )
                if isinstance(geo_candidate, dict):
                    geo_map = geo_candidate
                elif isinstance(geo_candidate, _MappingABC):
                    try:
                        geo_map = dict(geo_candidate)
                    except Exception:
                        geo_map = {}

            if not counts_by_diam_raw or sum(int(v) for v in counts_by_diam_raw.values()) == 0:
                counts_by_diam_raw = _seed_drill_bins_from_geo(geo_map)

            _push(lines, f"[DEBUG] DRILL bins raw={sum(counts_by_diam_raw.values())}")

            counts_by_diam = _adjust_drill_counts(
                counts_source,
                ops_claims,
            )
            if isinstance(breakdown_mutable, (_MutableMappingABC, dict)):
                try:
                    extra_bucket_ops = typing.cast(
                        MutableMapping[str, Any],
                        breakdown_mutable,
                    ).setdefault("extra_bucket_ops", {})
                    drill_entries = extra_bucket_ops.setdefault("drill", [])
                    drill_entries.append(
                        {
                            "name": "Drill",
                            "qty": int(sum(max(0, int(v)) for v in counts_by_diam.values())),
                            "side": None,
                        }
                    )
                except Exception:
                    pass
            drill_actions_from_groups = int(
                sum(max(0, int(v)) for v in counts_by_diam.values())
            )
            extras["drill_actions_from_groups"] = drill_actions_from_groups
            _push(lines, f"[DEBUG] DRILL bins adj={drill_actions_from_groups}")
            ops_hole_count_from_table = drill_actions_from_groups

            remaining_counts = dict(counts_by_diam)
            adjusted_rows: list[dict[str, Any]] = []
            for row in sanitized_rows:
                key = round(float(row.get("diameter_in", 0.0) or 0.0), 4)
                available = int(remaining_counts.get(key, 0))
                if available <= 0:
                    continue
                orig_qty = int(row.get("qty", 0))
                use_qty = min(orig_qty, available)
                if use_qty <= 0:
                    continue
                new_row = dict(row)
                new_row["qty"] = use_qty
                minutes_per = float(new_row.get("minutes_per_hole", 0.0) or 0.0)
                new_row["group_minutes"] = minutes_per * float(use_qty)
                remaining_counts[key] = max(0, available - use_qty)
                adjusted_rows.append(new_row)

            sanitized_rows = [row for row in adjusted_rows if int(row.get("qty", 0)) > 0]
            printed_sum = sum(int(row.get("qty", 0)) for row in sanitized_rows)
            _push(
                lines,
                f"[DEBUG] DRILL printed_sum={printed_sum} audit_drill={drill_actions_from_groups}",
            )
            subtotal_minutes = sum(
                float(row.get("group_minutes", 0.0) or 0.0) for row in sanitized_rows
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
            drill_minutes_subtotal_raw = _sanitize_drill_removal_minutes(
                subtotal_minutes_val or 0.0
            )
            drill_minutes_subtotal = round(drill_minutes_subtotal_raw, 2)
            total_minutes_val = (
                _coerce_float_or_none(dtph_map.get("total_minutes_with_toolchange"))
                or _coerce_float_or_none(dtph_map.get("total_minutes"))
            )
            if total_minutes_val is None:
                total_minutes_val = drill_minutes_subtotal_raw + total_tool_minutes
            total_minutes_val = sane_minutes_or_zero(total_minutes_val)

            drill_minutes_total = round(float(total_minutes_val or 0.0), 2)
            _push(lines, f"[DEBUG] drilling_minutes_total={drill_minutes_total:.2f} min")
            _push(
                lines,
                f"Subtotal (per-hole × qty) . {drill_minutes_subtotal:.2f} min  ("
                f"{fmt_hours(minutes_to_hours(drill_minutes_subtotal))})",
            )
            _push(
                lines,
                "TOTAL DRILLING (with toolchange) . "
                f"{drill_minutes_total:.2f} min  ("
                f"{fmt_hours(minutes_to_hours(drill_minutes_total))})",
            )
            lines.append("")

            extras["drill_machine_minutes"] = float(drill_minutes_subtotal)
            extras["drill_labor_minutes"] = float(total_tool_minutes)
            extras["drill_total_minutes"] = drill_minutes_subtotal
            logging.info(
                f"[removal] drill_total_minutes={extras['drill_total_minutes']}"
            )
            extras["removal_drilling_minutes_subtotal"] = float(drill_minutes_subtotal)
            extras["removal_drilling_minutes"] = float(drill_minutes_total)
            if drill_minutes_total > 0.0:
                extras["removal_drilling_hours"] = minutes_to_hours(drill_minutes_total)

            extra_map = extras
            drill_minutes_total = _pick_drill_minutes(
                process_plan_summary,
                extra_map,
            )  # minutes
            drill_mrate = (
                _lookup_bucket_rate("drilling", rates)
                or _lookup_bucket_rate("machine", rates)
                or 45.0
            )
            drill_lrate = (
                _lookup_bucket_rate("drilling_labor", rates)
                or _lookup_bucket_rate("labor", rates)
                or 45.0
            )

            _purge_legacy_drill_sync(bucket_view_obj)
            _set_bucket_minutes_cost(
                bucket_view_obj,
                "drilling",
                drill_minutes_total,
                drill_mrate,
                drill_lrate,
            )

            summary_target: MutableMapping[str, Any]
            if isinstance(updated_plan_summary, _MutableMappingABC):
                summary_target = typing.cast(MutableMapping[str, Any], updated_plan_summary)
            elif isinstance(updated_plan_summary, dict):
                summary_target = updated_plan_summary
            elif isinstance(updated_plan_summary, _MappingABC):
                summary_target = dict(updated_plan_summary)
            elif isinstance(process_plan_summary, _MutableMappingABC):
                summary_target = typing.cast(MutableMapping[str, Any], process_plan_summary)
            elif isinstance(process_plan_summary, dict):
                summary_target = process_plan_summary
            elif isinstance(process_plan_summary, _MappingABC):
                summary_target = dict(process_plan_summary)
            else:
                summary_target = {}
            updated_plan_summary = summary_target

            drilling_summary_map = summary_target.setdefault("drilling", {})
            if isinstance(drilling_summary_map, _MutableMappingABC):
                drilling_summary = typing.cast(MutableMapping[str, Any], drilling_summary_map)
            elif isinstance(drilling_summary_map, dict):
                drilling_summary = drilling_summary_map
                summary_target["drilling"] = drilling_summary
            else:
                drilling_summary = {}
                summary_target["drilling"] = drilling_summary

            drilling_summary["total_minutes_billed"] = float(drill_minutes_total)
            drilling_summary["source"] = "removal_card"

            buckets_for_cleanup: Mapping[str, Any] | None = None
            if isinstance(bucket_view_obj, (_MappingABC, dict)):
                try:
                    buckets_for_cleanup = bucket_view_obj.get("buckets")
                except Exception:
                    buckets_for_cleanup = None

            drilling_bucket_entry: MutableMapping[str, Any] | dict[str, Any] | None = None
            milling_bucket_entry: MutableMapping[str, Any] | dict[str, Any] | None = None
            if isinstance(buckets_for_cleanup, _MutableMappingABC):
                candidate = buckets_for_cleanup.get("drilling")
                if isinstance(candidate, _MutableMappingABC):
                    drilling_bucket_entry = typing.cast(MutableMapping[str, Any], candidate)
                elif isinstance(candidate, dict):
                    drilling_bucket_entry = candidate
                milling_candidate = buckets_for_cleanup.get("milling")
                if isinstance(milling_candidate, _MutableMappingABC):
                    milling_bucket_entry = typing.cast(MutableMapping[str, Any], milling_candidate)
                elif isinstance(milling_candidate, dict):
                    milling_bucket_entry = milling_candidate
            elif isinstance(buckets_for_cleanup, dict):
                candidate = buckets_for_cleanup.get("drilling")
                if isinstance(candidate, dict):
                    drilling_bucket_entry = candidate
                elif isinstance(candidate, _MutableMappingABC):
                    drilling_bucket_entry = typing.cast(MutableMapping[str, Any], candidate)
                milling_candidate = buckets_for_cleanup.get("milling")
                if isinstance(milling_candidate, dict):
                    milling_bucket_entry = milling_candidate
                elif isinstance(milling_candidate, _MutableMappingABC):
                    milling_bucket_entry = typing.cast(MutableMapping[str, Any], milling_candidate)

            if isinstance(drilling_bucket_entry, (_MutableMappingABC, dict)):
                drilling_bucket_entry.pop("synced_minutes", None)
                drilling_bucket_entry.pop("synced_hours", None)

            if isinstance(milling_bucket_entry, (_MutableMappingABC, dict)):
                milling_bucket_entry.pop("synced_minutes", None)
                milling_bucket_entry.pop("synced_hours", None)

            drilling_bucket = None
            if isinstance(bucket_view_obj, (_MappingABC, dict)):
                drilling_bucket = (bucket_view_obj.get("buckets") or {}).get("drilling")
            logging.info(
                f"[bucket] drilling_minutes={drill_minutes_total} drilling_bucket={drilling_bucket}"
            )

            milling_hours_candidates: list[Any] = []
            if isinstance(drilling_card_detail, _MappingABC):
                for key in (
                    "milling_hours",
                    "roughing_hours",
                    "milling_total_hours",
                    "roughing_total_hours",
                ):
                    milling_hours_candidates.append(drilling_card_detail.get(key))
            if isinstance(updated_plan_summary, _MappingABC):
                milling_summary = updated_plan_summary.get("milling")
                if isinstance(milling_summary, _MappingABC):
                    for key in (
                        "hours",
                        "hr",
                        "machine_hours",
                        "total_hours",
                        "roughing_hours",
                    ):
                        milling_hours_candidates.append(milling_summary.get(key))

            milling_hours: float = 0.0
            for candidate in milling_hours_candidates:
                minutes_val = _coerce_float_or_none(candidate)
                if minutes_val is not None and math.isfinite(minutes_val) and minutes_val > 0.0:
                    milling_hours = float(minutes_val)
                    break

            milling_hours = float(milling_hours or 0.0)
            milling_minutes_total = 60.0 * milling_hours
            mill_mrate = (
                _lookup_bucket_rate("milling", rates)
                or _lookup_bucket_rate("machine", rates)
                or 45.0
            )
            mill_lrate = (
                _lookup_bucket_rate("milling_labor", rates)
                or _lookup_bucket_rate("labor", rates)
                or 45.0
            )

            milling_estimate_result: Mapping[str, Any] | None = None
            if milling_minutes_total <= 0.0 and geo_map:
                emit_bottom_face_hint = False
                pricing_hints = geo_map.get("pricing_hints") if isinstance(geo_map, _MappingABC) else None
                if isinstance(pricing_hints, _MappingABC):
                    emit_bottom_face_hint = bool(pricing_hints.get("face_both_sides"))
                try:
                    milling_estimate_result = estimate_milling_minutes_from_geometry(
                        geo_map,
                        speeds_feeds_table,
                        material_group,
                        {
                            "machine_per_hour": mill_mrate,
                            "labor_per_hour": mill_lrate,
                        },
                        emit_bottom_face=emit_bottom_face_hint,
                    )
                except Exception as exc:
                    logging.debug("[milling-estimate] failed: %s", exc, exc_info=False)
                    milling_estimate_result = None
                minutes_est = (
                    _coerce_float_or_none((milling_estimate_result or {}).get("minutes"))
                    if milling_estimate_result
                    else None
                )
                if minutes_est and minutes_est > 0.0:
                    milling_minutes_total = float(minutes_est)
                    extras["milling_minutes_estimated"] = float(minutes_est)
                    dbg(lines, f"milling_minutes_estimated={minutes_est:.2f}")
                elif milling_estimate_result:
                    dbg(lines, f"milling_estimate_failed={milling_estimate_result}")

            _set_bucket_minutes_cost(
                bucket_view_obj,
                "milling",
                milling_minutes_total,
                mill_mrate,
                mill_lrate,
            )

            if milling_estimate_result and isinstance(bucket_view_obj, (_MutableMappingABC, dict)):
                try:
                    buckets_map = bucket_view_obj.setdefault("buckets", {})
                    if isinstance(buckets_map, dict):
                        entry = buckets_map.get("milling")
                        if isinstance(entry, dict):
                            detail_payload = milling_estimate_result.get("detail")
                            if isinstance(detail_payload, _MappingABC):
                                entry.setdefault("detail", dict(detail_payload))
                        buckets_map["milling"] = entry  # type: ignore[index]
                    elif isinstance(buckets_map, _MutableMappingABC):
                        entry = buckets_map.get("milling")
                        if isinstance(entry, _MutableMappingABC):
                            detail_payload = milling_estimate_result.get("detail")
                            if isinstance(detail_payload, _MappingABC):
                                entry.setdefault("detail", dict(detail_payload))
                except Exception:
                    pass

            if milling_estimate_result:
                dbg(lines, f"milling_estimate_detail={milling_estimate_result}")

            _normalize_buckets(bucket_view_obj)

            buckets_final: Mapping[str, Any] | None = None
            if isinstance(bucket_view_obj, (_MappingABC, dict)):
                buckets_final = bucket_view_obj.get("buckets")
            dbg(lines, f"buckets_final={buckets_final or {}}")

            drilling_dbg_entry: Mapping[str, Any] | None = None
            milling_dbg_entry: Mapping[str, Any] | None = None
            if isinstance(bucket_view_obj, _MappingABC):
                try:
                    buckets_snapshot = bucket_view_obj.get("buckets")
                except Exception:
                    buckets_snapshot = None
                if isinstance(buckets_snapshot, _MappingABC):
                    buckets_snapshot_map = typing.cast(Mapping[str, Any], buckets_snapshot)
                    drilling_dbg_entry = buckets_snapshot_map.get("drilling")
                    milling_dbg_entry = buckets_snapshot_map.get("milling")
            if not isinstance(drilling_dbg_entry, _MappingABC):
                drilling_dbg_entry = {}
            if not isinstance(milling_dbg_entry, _MappingABC):
                milling_dbg_entry = {}
            dbg(lines, f"drilling_bucket={drilling_dbg_entry}")
            dbg(lines, f"milling_bucket={milling_dbg_entry}")

            return extras, lines, updated_plan_summary

    _normalize_buckets(bucket_view_obj)

    return extras, lines, updated_plan_summary


def _adjusted_drill_groups_for_display(
    breakdown: Mapping[str, Any] | MutableMapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    drilling_meta_source: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return drill groups with geometry-based adjustments applied."""

    geo_contexts: list[Mapping[str, Any]] = []

    def _collect_geo(container: Mapping[str, Any] | MutableMapping[str, Any] | None) -> None:
        if not isinstance(container, _MappingABC):
            return
        for key in ("geo", "geom"):
            candidate = container.get(key)
            if isinstance(candidate, _MappingABC):
                geo_contexts.append(typing.cast(Mapping[str, Any], candidate))
        if isinstance(container, Mapping):
            direct = container.get("hole_table_geo")
            if isinstance(direct, _MappingABC):
                geo_contexts.append(typing.cast(Mapping[str, Any], direct))

    _collect_geo(breakdown)
    _collect_geo(result)
    _collect_geo(drilling_meta_source)

    expanded_geo: list[Mapping[str, Any]] = []
    for geo_map in geo_contexts:
        expanded_geo.append(geo_map)
        for nested in _iter_geo_dicts_for_context(geo_map):
            if isinstance(nested, _MappingABC):
                expanded_geo.append(typing.cast(Mapping[str, Any], nested))

    hole_diams_mm: list[float] = []
    thickness_in: float | None = None
    ops_claims_map: Mapping[str, Any] | None = None
    primary_geo: Mapping[str, Any] | None = expanded_geo[0] if expanded_geo else None

    def _extend_hole_diams(source: Mapping[str, Any] | None) -> None:
        if not isinstance(source, _MappingABC):
            return
        for key in ("hole_diams_mm_precise", "hole_diams_mm", "hole_diams"):
            seq = source.get(key)
            if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes, bytearray)):
                for entry in seq:
                    value = _coerce_float_or_none(entry)
                    if value is not None and value > 0:
                        hole_diams_mm.append(float(value))

    def _maybe_thickness(source: Mapping[str, Any] | None) -> None:
        nonlocal thickness_in
        if not isinstance(source, _MappingABC) or thickness_in is not None:
            return
        thickness_candidates = (
            source.get("thickness_in"),
            source.get("plate_thickness_in"),
            source.get("stock_thickness_in"),
        )
        for candidate in thickness_candidates:
            t_val = _coerce_float_or_none(candidate)
            if t_val is not None and t_val > 0:
                thickness_in = float(t_val)
                return
        thickness_mm = _coerce_float_or_none(source.get("thickness_mm"))
        if thickness_mm is not None and thickness_mm > 0:
            thickness_in = float(thickness_mm) / 25.4

    def _maybe_ops_claims(source: Mapping[str, Any] | None) -> None:
        nonlocal ops_claims_map
        if not isinstance(source, _MappingABC) or ops_claims_map is not None:
            return
        candidate = source.get("ops_claims")
        if isinstance(candidate, _MappingABC):
            ops_claims_map = typing.cast(Mapping[str, Any], candidate)
            return
        summary = source.get("ops_summary")
        if isinstance(summary, _MappingABC):
            claims = summary.get("claims")
            if isinstance(claims, _MappingABC):
                ops_claims_map = typing.cast(Mapping[str, Any], claims)

    for ctx in expanded_geo:
        _extend_hole_diams(ctx)
        _maybe_thickness(ctx)
        _maybe_ops_claims(ctx)

    if not hole_diams_mm:
        for container in (breakdown, result, drilling_meta_source):
            if isinstance(container, _MappingABC):
                _extend_hole_diams(container)
                _maybe_thickness(container)
                _maybe_ops_claims(container)

    if not hole_diams_mm:
        return []

    try:
        return build_drill_groups_from_geometry(
            hole_diams_mm,
            thickness_in,
            ops_claims_map,
            primary_geo,
            drop_large_holes=False,
        )
    except Exception:
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
    # Prefer grouped totals from ops_summary when available (parser_rules_v2 path).
    for ctx in _iter_geo_dicts_for_context(geo):
        ops_summary = ctx.get("ops_summary") if isinstance(ctx, _MappingABC) else None
        if not isinstance(ops_summary, _MappingABC):
            continue
        grouped = ops_summary.get("group_totals")
        if not isinstance(grouped, _MappingABC):
            continue
        drill_map = grouped.get("drill") or grouped.get("drilling")
        if not isinstance(drill_map, _MappingABC):
            continue
        groups: dict[float, dict[str, Any]] = {}
        total_qty = 0
        for ref_key, side_map in drill_map.items():
            if not isinstance(side_map, _MappingABC):
                continue
            qty_sum = 0
            dia_val: float | None = None
            depth_candidates: list[float] = []
            for side_info in side_map.values():
                if not isinstance(side_info, _MappingABC):
                    continue
                qty_val = _coerce_int_or_zero(side_info.get("qty"))
                if qty_val <= 0:
                    continue
                qty_sum += qty_val
                if dia_val is None:
                    dia_candidate = _coerce_float_or_none(side_info.get("diameter_in"))
                    if dia_candidate is None:
                        dia_candidate = _coerce_float_or_none(side_info.get("ref_dia_in"))
                    if dia_candidate is None and isinstance(ref_key, str):
                        dia_candidate = _coerce_float_or_none(_parse_ref_to_inch(ref_key))
                    if dia_candidate is not None and math.isfinite(dia_candidate):
                        dia_val = float(dia_candidate)
                depth_val = (
                    _coerce_float_or_none(side_info.get("depth_in_max"))
                    or _coerce_float_or_none(side_info.get("depth_in_avg"))
                    or _coerce_float_or_none(side_info.get("depth_in_min"))
                )
                if depth_val is not None and math.isfinite(depth_val):
                    depth_candidates.append(float(depth_val))
            if qty_sum <= 0 or dia_val is None:
                continue
            key = round(float(dia_val), 4)
            entry = groups.setdefault(
                key,
                {
                    "diameter_in": float(round(float(dia_val), 4)),
                    "qty": 0,
                },
            )
            entry["qty"] += int(qty_sum)
            if depth_candidates:
                entry["depth_in"] = max(depth_candidates)
            total_qty += qty_sum
        if groups:
            ordered = [
                {"diameter_in": data["diameter_in"], "qty": data["qty"], **(
                    {"depth_in": data["depth_in"]}
                    if "depth_in" in data
                    else {}
                )}
                for _, data in sorted(groups.items())
            ]
            return ordered, int(total_qty)

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


def _estimate_drilling_minutes_from_meta(
    drilling_meta: Mapping[str, Any] | None,
    geo_map: Mapping[str, Any] | None = None,
) -> tuple[float, float, float, dict[str, Any]]:
    """Return machine/toolchange/total minutes derived from drilling meta."""

    if not isinstance(drilling_meta, _MappingABC):
        return (0.0, 0.0, 0.0, {})

    drilling_meta_map: Mapping[str, Any] | MutableMapping[str, Any] = typing.cast(
        Mapping[str, Any] | MutableMapping[str, Any], drilling_meta
    )
    drilling_meta_mut: MutableMapping[str, Any] | None = None

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
            depth_hint = (
                _coerce_float_or_none(group_info.get("depth_in"))
                or _coerce_float_or_none(group_info.get("depth_in_max"))
            )
            if depth_hint is not None and depth_hint > 0:
                entry_copy["depth_in"] = float(depth_hint)
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
            depth_hint = (
                _coerce_float_or_none(group_info.get("depth_in"))
                or _coerce_float_or_none(group_info.get("depth_in_max"))
            )
            filtered_bins.append(
                {
                    "op": "drill",
                    "op_name": "drill",
                    "diameter_in": dia_in,
                    "diameter_mm": float(dia_in * 25.4),
                    "qty": int(group_info.get("qty") or 0),
                    "depth_in": float(depth_hint)
                    if depth_hint is not None and depth_hint > 0
                    else _safe_float(group_info.get("depth_in")),
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
    drill_groups_for_detail: list[dict[str, Any]] = []
    if bins:
        _local_lines: list[str] = []
        subtotal_min, seen_deep, seen_std, drill_groups = _render_time_per_hole(
            _local_lines,
            bins=bins,
            index_min=index_min,
            peck_min_deep=peck_min_deep,
            peck_min_std=peck_min_std,
            extra_map=drilling_meta_mut,
        )
        if drill_groups and isinstance(drilling_meta_mut, _MutableMappingABC):
            try:
                drilling_meta_mut["drill_groups"] = [dict(group) for group in drill_groups]
            except Exception:
                drilling_meta_mut["drill_groups"] = drill_groups
        drill_groups_for_detail = drill_groups

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
        "drill_groups": drill_groups_for_detail,
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

    block: dict[str, Any] = {}
    if isinstance(material_detail, _MappingABC):
        block.update(material_detail)

    def _assign_if_missing(key: str, value: Any) -> None:
        if key in block:
            return
        if value in (None, ""):
            return
        coerced = _coerce_float_or_none(value)
        if coerced is None:
            return
        block[key] = float(coerced)

    _assign_if_missing("material_cost_before_credit", material_total)
    _assign_if_missing("material_cost", material_total)
    _assign_if_missing("material_direct_cost", material_total)
    _assign_if_missing("material_cost_pre_credit", material_total)
    _assign_if_missing("material_base_cost", material_total)
    _assign_if_missing("material_scrap_credit", scrap_credit)
    _assign_if_missing("scrap_credit_usd", scrap_credit)
    _assign_if_missing("material_tax", material_tax)
    _assign_if_missing("material_tax_usd", material_tax)

    if scrap_price_source:
        try:
            source_text = str(scrap_price_source).strip()
        except Exception:
            source_text = ""
        if source_text:
            block.setdefault("scrap_price_source", source_text)
            block.setdefault("scrap_credit_source", source_text)

    try:
        components = _material_cost_components(block, overrides=None, cfg=None)
    except Exception:
        components = None

    if isinstance(components, _MappingABC):
        material_total_usd = _coerce_float_or_none(components.get("total_usd")) or 0.0
    else:
        base_val = _coerce_float_or_none(material_total) or 0.0
        tax_val = _coerce_float_or_none(material_tax) or 0.0
        scrap_val = _coerce_float_or_none(scrap_credit) or 0.0
        material_total_usd = float(base_val) + float(tax_val) - float(scrap_val)

    pass_through_total = 0.0
    for key, raw_value in (pass_through or {}).items():
        try:
            if str(key).strip().lower() == "material":
                continue
        except Exception:
            pass
        amount = _coerce_float_or_none(raw_value)
        if amount is not None:
            pass_through_total += float(amount)

    total = round(float(material_total_usd) + float(pass_through_total), 2)
    if total < 0:
        return 0.0
    return total


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
    import pandas as pd  # type: ignore[import-not-found]
    from pandas import DataFrame as PandasDataFrame  # type: ignore[import-not-found]
    from pandas import Index as PandasIndex  # type: ignore[import-not-found]
    from pandas import Series as PandasSeries  # type: ignore[import-not-found]
    from cad_quoter_pkg.src.cad_quoter.geometry import GeometryService as GeometryServiceType
else:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - optional dependency
        pd = typing.cast("Any", None)

    PandasDataFrame = typing.Any
    PandasSeries = typing.Any
    PandasIndex = typing.Any
    GeometryServiceType = typing.Any

SeriesLike = typing.Any


def _is_pandas_dataframe(obj: Any) -> TypeGuard[PandasDataFrame]:
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
FACE_OF = typing.cast(Any, getattr(geometry, "FACE_OF"))
ensure_face = typing.cast(Any, getattr(geometry, "ensure_face"))
face_surface = typing.cast(Any, getattr(geometry, "face_surface"))
iter_faces = typing.cast(Any, getattr(geometry, "iter_faces"))
linear_properties = typing.cast(Any, getattr(geometry, "linear_properties"))
map_shapes_and_ancestors = typing.cast(Any, getattr(geometry, "map_shapes_and_ancestors"))
from cad_quoter.geo2d.apply import apply_2d_features_to_variables

# Tolerance for invariant checks that guard against silent drift when rendering
# cost sections.
_LABOR_SECTION_ABS_EPSILON = 0.51
_PLANNER_BUCKET_ABS_EPSILON = 0.51

if typing.TYPE_CHECKING:  # pragma: no cover - guide static analysis to the concrete modules
    from cad_quoter_pkg.src.cad_quoter.domain_models.materials import (
        DEFAULT_MATERIAL_DISPLAY,
        DEFAULT_MATERIAL_KEY,
        MATERIAL_DENSITY_G_CC_BY_KEY,
        MATERIAL_DENSITY_G_CC_BY_KEYWORD,
        MATERIAL_DISPLAY_BY_KEY,
        MATERIAL_DROPDOWN_OPTIONS,
        MATERIAL_KEYWORDS,
        MATERIAL_MAP,
        MATERIAL_OTHER_KEY,
        normalize_material_key,
    )
    from cad_quoter_pkg.src.cad_quoter.domain_models import (
        coerce_float_or_none as _coerce_float_or_none,
    )
else:  # pragma: no cover - retain runtime namespace package fallback
    try:
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
            coerce_float_or_none as _coerce_float_or_none,
            normalize_material_key,
        )
    except ImportError:
        from cad_quoter_pkg.src.cad_quoter.domain_models.materials import (
            DEFAULT_MATERIAL_DISPLAY,
            DEFAULT_MATERIAL_KEY,
            MATERIAL_DENSITY_G_CC_BY_KEY,
            MATERIAL_DENSITY_G_CC_BY_KEYWORD,
            MATERIAL_DISPLAY_BY_KEY,
            MATERIAL_DROPDOWN_OPTIONS,
            MATERIAL_KEYWORDS,
            MATERIAL_MAP,
            MATERIAL_OTHER_KEY,
            normalize_material_key,
        )
        from cad_quoter_pkg.src.cad_quoter.domain_models import (
            coerce_float_or_none as _coerce_float_or_none,
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
from cad_quoter.pricing.milling_estimator import estimate_milling_minutes_from_geometry
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
import cad_quoter.pricing.materials as _materials
from cad_quoter.pricing.materials import (
    STANDARD_PLATE_SIDES_IN as STANDARD_PLATE_SIDES_IN,
    _compute_material_block as _compute_material_block,
    _compute_scrap_mass_g as _compute_scrap_mass_g,
    _hole_margin_inches as _hole_margin_inches,
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
from cad_quoter.rates import LABOR_RATE_KEYS, MACHINE_RATE_KEYS, two_bucket_to_flat
from cad_quoter.vendors.mcmaster_stock import lookup_sku_and_price_for_mm

SCRAP_RECOVERY_DEFAULT = _materials.SCRAP_RECOVERY_DEFAULT


def _fail_live_price(*_args: Any, **_kwargs: Any) -> None:
    """Sentinel used by tests to simulate Wieland API failures."""

    raise RuntimeError("live material pricing is unavailable")


def _jsonify_debug_value(value: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """Proxy to :func:`cad_quoter.utils.debug_tables._jsonify_debug_value`."""

    return _debug_jsonify_value(value, depth=depth, max_depth=max_depth)


def _jsonify_debug_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Proxy to :func:`cad_quoter.utils.debug_tables._jsonify_debug_summary`."""

    return _debug_jsonify_summary(summary)


try:
    import builtins as _builtins

    _builtins_any = typing.cast(Any, _builtins)
    if getattr(_builtins_any, "_fail_live_price", None) is None:  # pragma: no cover - test shim
        setattr(_builtins_any, "_fail_live_price", _fail_live_price)
except Exception:  # pragma: no cover - defensive
    pass

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

if TYPE_CHECKING:
    from cad_quoter.llm import LLMClient as LLMClientType
else:  # pragma: no cover - typing fallback
    LLMClientType = typing.Any

LLMClient = _llm_integration.llm_client
parse_llm_json = _llm_integration.parse_llm_json
explain_quote = _llm_integration.explain_quote

configure_llm_integration(_llm_integration)

# Backwards compatibility aliases expected by older integrations.
_DEFAULT_VL_MODEL_NAMES = DEFAULT_VL_MODEL_NAMES
_DEFAULT_MM_PROJ_NAMES = DEFAULT_MM_PROJ_NAMES

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

_read_step_or_iges_or_brep_impl = typing.cast(
    Callable[[str | Path], Any],
    getattr(
        geometry,
        "read_step_or_iges_or_brep",
        _missing_geo_helper("read_step_or_iges_or_brep"),
    ),
)
_require_ezdxf = typing.cast(
    Callable[[], Any],
    getattr(geometry, "require_ezdxf", _missing_geo_helper("require_ezdxf")),
)
_convert_dwg_to_dxf = typing.cast(
    Callable[[str], str],
    getattr(geometry, "convert_dwg_to_dxf", _missing_geo_helper("convert_dwg_to_dxf")),
)
_get_dwg_converter_path = typing.cast(
    Callable[[], str | None],
    getattr(geometry, "get_dwg_converter_path", lambda: None),
)


def read_step_or_iges_or_brep(path: str) -> Any:
    """Backwards-compatible shim that forwards to :mod:`cad_quoter.geometry`."""

    return _read_step_or_iges_or_brep_impl(path)

# ---- tiny helpers you can use elsewhere --------------------------------------
# Optional PDF stack
try:
    import fitz  # type: ignore[import-not-found]  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore[assignment]
    _HAS_PYMUPDF = False

DIM_RE = re.compile(r"(?:[Øø⌀]|DIAM|DIA)\s*([0-9.+-]+)|R\s*([0-9.+-]+)|([0-9.+-]+)\s*[x×]\s*([0-9.+-]+)")

def load_drawing(path: Path) -> Drawing:
    ezdxf_mod = typing.cast(_EzdxfModule, _require_ezdxf())
    if path.suffix.lower() == ".dwg":
        # Prefer explicit converter/wrapper if configured (works even if ODA isn't on PATH)
        exe = _get_dwg_converter_path()
        if exe:
            dxf_path = _convert_dwg_to_dxf(str(path))
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

_SPEEDS_FEEDS_CACHE: dict[str, PandasDataFrame | None] = {}

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

PREFERRED_PROCESS_BUCKET_ORDER: tuple[str, ...] = PROCESS_BUCKETS.order

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


class PlannerBucketCost(TypedDict):
    name: str
    cost: float

# ---------- Bucket & Operation Roles ----------
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
        ("machine_rate_per_hr", 90.0),
        ("labor_rate_per_hr", 45.0),
        ("milling_attended_fraction", 1.0),
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

    def _material_cost_components_from(
        *containers: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        for container in containers:
            if not isinstance(container, _MappingABC):
                continue
            candidate = container.get("material_cost_components")
            if isinstance(candidate, _MappingABC):
                return candidate
        return None

    material_cost_components = _material_cost_components_from(
        material_block,
        material_stock_block,
        material_selection,
        breakdown.get("material"),
        breakdown.get("material_block"),
    )

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
    default_flat_rates: Mapping[str, Any] = RATES_DEFAULT if isinstance(RATES_DEFAULT, Mapping) else {}

    def _default_rate(*keys: str) -> float:
        for key in keys:
            if not key:
                continue
            try:
                value = default_flat_rates.get(key)
            except Exception:
                value = None
            numeric = _coerce_rate_value(value)
            if numeric > 0.0:
                return numeric
        return 0.0

    default_machine_rate_value = _default_rate("MachineRate", "MillingRate", "CNC_Mill")
    default_labor_rate_value = _default_rate("LaborRate", "Machinist", "DefaultLaborRate")
    default_programmer_rate_value = _default_rate("ProgrammingRate", "Programmer")
    default_inspector_rate_value = _default_rate("InspectionRate", "Inspector")

    cfg_labor_rate_value = 0.0
    if separate_labor_cfg:
        cfg_labor_rate_value = _coerce_rate_value(getattr(cfg, "labor_rate_per_hr", 0.0))
        if cfg_labor_rate_value <= 0.0:
            cfg_labor_rate_value = default_labor_rate_value or 45.0

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
        labor_rate_value = default_labor_rate_value or 45.0
    if separate_labor_cfg and cfg_labor_rate_value > 0.0:
        labor_rate_value = cfg_labor_rate_value
    rates["LaborRate"] = labor_rate_value

    machine_rate_value = _coerce_rate_value(rates.get("MachineRate"))
    if machine_rate_value <= 0:
        machine_rate_value = _coerce_rate_value(rates.get("ShopMachineRate"))
    if machine_rate_value <= 0:
        machine_rate_value = default_machine_rate_value or labor_rate_value
    rates["MachineRate"] = machine_rate_value

    cfg_programmer_rate: float | None = None
    if cfg and getattr(cfg, "separate_machine_labor", False):
        cfg_programmer_rate = _coerce_rate_value(getattr(cfg, "labor_rate_per_hr", None))
        if cfg_programmer_rate <= 0:
            cfg_programmer_rate = default_programmer_rate_value or default_labor_rate_value or 45.0

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
                programmer_fallback = default_programmer_rate_value or default_labor_rate_value
            if programmer_fallback > 0 and default_programmer_rate_value > 0:
                programmer_fallback = max(programmer_fallback, default_programmer_rate_value)
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
            programmer_rate_value = default_programmer_rate_value or default_labor_rate_value
        if programmer_rate_value > 0 and default_programmer_rate_value > 0:
            programmer_rate_value = max(programmer_rate_value, default_programmer_rate_value)

        programming_rate_value = _coerce_rate_value(rates.get("ProgrammingRate"))
        if programming_rate_value <= 0:
            programming_rate_value = programmer_rate_value
        if programming_rate_value <= 0:
            programming_rate_value = labor_rate_value
        if programming_rate_value <= 0:
            programming_rate_value = default_programmer_rate_value or programmer_rate_value
        if programming_rate_value > 0 and default_programmer_rate_value > 0:
            programming_rate_value = max(programming_rate_value, default_programmer_rate_value)

    rates["ProgrammerRate"] = programmer_rate_value
    rates["ProgrammingRate"] = programming_rate_value

    inspector_rate_value = _coerce_rate_value(rates.get("InspectorRate"))
    if inspector_rate_value <= 0:
        inspector_rate_value = labor_rate_value
    if inspector_rate_value <= 0:
        inspector_rate_value = default_inspector_rate_value or default_labor_rate_value
    if inspector_rate_value > 0 and default_inspector_rate_value > 0:
        inspector_rate_value = max(inspector_rate_value, default_inspector_rate_value)
    rates["InspectorRate"] = inspector_rate_value

    inspection_rate_value = _coerce_rate_value(rates.get("InspectionRate"))
    if inspection_rate_value <= 0:
        inspection_rate_value = inspector_rate_value
    if inspection_rate_value <= 0:
        inspection_rate_value = labor_rate_value
    if inspection_rate_value <= 0:
        inspection_rate_value = default_inspector_rate_value or default_labor_rate_value
    if inspection_rate_value > 0 and default_inspector_rate_value > 0:
        inspection_rate_value = max(inspection_rate_value, default_inspector_rate_value)
    rates["InspectionRate"] = inspection_rate_value

    if default_labor_rate_value > 0:
        rates.setdefault("LaborRate", default_labor_rate_value)
    else:
        rates.setdefault("LaborRate", 45.0)
    if default_machine_rate_value > 0:
        rates.setdefault("MachineRate", default_machine_rate_value)
    else:
        rates.setdefault("MachineRate", labor_rate_value or 45.0)
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

    material_component_total: float | None = None
    material_component_net: float | None = None
    if isinstance(material_cost_components, _MappingABC):
        base_component = _coerce_float_or_none(material_cost_components.get("base_usd"))
        if base_component is not None:
            material_total_for_directs = float(base_component)
        scrap_component = _coerce_float_or_none(
            material_cost_components.get("scrap_credit_usd")
        )
        if scrap_component is not None:
            scrap_credit_for_directs = float(scrap_component)
        tax_component = _coerce_float_or_none(material_cost_components.get("tax_usd"))
        if tax_component is not None:
            material_tax_for_directs = float(tax_component)
        total_component = _coerce_float_or_none(material_cost_components.get("total_usd"))
        if total_component is not None:
            material_component_total = float(total_component)
        net_component = _coerce_float_or_none(material_cost_components.get("net_usd"))
        if net_component is not None:
            material_component_net = float(net_component)

    def _merge_detail_text(existing: str | None, new_value: Any) -> str:
        segments: list[str] = []
        seen: set[str] = set()
        for candidate in (existing, new_value):
            if candidate is None:
                continue
            # Split on semicolons and trim whitespace
            for segment in str(candidate).split(";"):
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
    bucket_entries_for_totals_map: dict[str, Mapping[str, Any]] = {}
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

    geo_context: Mapping[str, Any] | None = None
    if isinstance(breakdown, _MappingABC):
        candidate = breakdown.get("geo_context")
        if isinstance(candidate, _MappingABC):
            geo_context = candidate
    if geo_context is None and isinstance(result, _MappingABC):
        candidate = result.get("geo_context")
        if isinstance(candidate, _MappingABC):
            geo_context = candidate
    if geo_context is None and isinstance(geometry, _MappingABC):
        geo_context = geometry

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

    def _two_bucket_rate(kind: str, *bucket_keys: str | None) -> float:
        """Look up a machine or labor rate from the two-bucket defaults."""

        try:
            mapping_candidate = fallback_two_bucket_rates.get(kind, {})
        except Exception:
            mapping_candidate = {}

        if isinstance(mapping_candidate, _MappingABC):
            mapping: dict[str, Any] = {str(k): v for k, v in mapping_candidate.items()}
        elif isinstance(mapping_candidate, dict):
            mapping = mapping_candidate
        else:
            mapping = {}

        seen: set[str] = set()
        for raw_key in bucket_keys:
            if raw_key in (None, ""):
                continue
            raw_text = str(raw_key).strip()
            if not raw_text:
                continue

            candidates = [raw_text]
            canon = _canonical_bucket_key(raw_text)
            if canon:
                candidates.append(str(canon))
            normalized = _normalize_bucket_key(raw_text)
            if normalized:
                candidates.append(str(normalized))
            lowered = raw_text.lower()
            if lowered:
                candidates.append(lowered)

            for candidate in candidates:
                candidate_clean = candidate.strip()
                if not candidate_clean or candidate_clean in seen:
                    continue
                seen.add(candidate_clean)
                value = mapping.get(candidate_clean)
                if value in (None, ""):
                    continue
                try:
                    numeric = float(value)
                except Exception:
                    continue
                if numeric > 0.0:
                    return numeric

        return 0.0

    def _pct(x) -> str:
        return format_percent(x)

    def _format_weight_lb_decimal(mass_g: float | None) -> str:
        return format_weight_lb_decimal(mass_g)

    def _format_weight_lb_oz(mass_g: float | None) -> str:
        return format_weight_lb_oz(mass_g)

    def _format_row(label: Any, amount: Any) -> str:
        """Format a left label and a right-aligned currency amount on one line."""

        left = str(label or "").strip()
        try:
            amt = float(amount or 0.0)
        except Exception:
            amt = 0.0
        right = _m(amt)
        # leave at least 2 spaces between label and amount
        total = max(10, int(page_width))
        pad = max(2, total - len(left) - len(right))
        return f"{left}{' ' * pad}{right}"

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
        label_text = " ".join(label_text.split())
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

    # Define before first use to avoid closure-order issues
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
    doc_builder = QuoteDocRecorder(divider)

    class _QuoteLines(list[str]):
        def append(self, text: str) -> None:  # type: ignore[override]
            sanitized = _sanitize_render_text(text)
            previous = self[-1] if self else None
            super().append(sanitized)
            doc_builder.observe_line(len(self) - 1, sanitized, previous)

    lines: list[str] = _QuoteLines()

    def append_line(value: Any) -> None:
        _push(lines, value)

    def append_lines(values: Iterable[str]) -> None:
        for value in values:
            append_line(value)

    def replace_line(index: int, text: str) -> None:
        sanitized = _sanitize_render_text(text)
        if 0 <= index < len(lines):
            lines[index] = sanitized
        doc_builder.replace_line(index, sanitized)

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
        for segment in (s.strip() for s in sanitized_detail.split(";")):
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

        headers = ("Bucket", "Minutes", "Labor $", "Machine $", "Total $")

        display_rows: list[tuple[str, str, str, str, str]] = []
        for bucket, minutes, labor_val, machine_val, total_val in rows:
            display_rows.append(
                (
                    str(bucket),
                    f"{float(minutes):.2f}",
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

        if lines and lines[-1] != "":
            append_line("")

        diagnostic_banner = "=== Planner diagnostics (not billed) ==="
        append_line(diagnostic_banner)
        append_line("=" * min(page_width, len(diagnostic_banner)))

        table_text = ascii_table(
            headers,
            display_rows,
            col_widths=col_widths,
            col_aligns=("L", "R", "R", "R", "R"),
            header_aligns=("L", "R", "R", "R", "R"),
        )
        table_lines = table_text.splitlines()

        if len(table_lines) >= 4:
            header_cells = table_lines[1].strip("|").split("|")
            separator_line = " | ".join("-" * width for width in col_widths)
            append_line(" | ".join(header_cells))
            append_line(separator_line)
            for body_line in table_lines[3:-1]:
                if not body_line.startswith("|"):
                    continue
                body_cells = body_line.strip("|").split("|")
                append_line(" | ".join(body_cells))

        append_line("")

    def _is_total_label(label: str) -> bool:
        clean = str(label or "").strip()
        if not clean:
            return False
        clean = clean.rstrip(":")
        clean = clean.lstrip("= ")
        return clean.lower().startswith("total")

    def _maybe_insert_total_separator(width: int) -> None:
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

    def _render_kv_line(label: str, value: str, indent: str = "") -> str:
        left = f"{indent}{label}"
        right = value
        right_width = max(len(right), 1)
        pad = max(1, page_width - len(left) - len(right))
        left_width = len(left) + pad
        table_text = draw_kv_table(
            [(left, right)],
            left_width=left_width,
            right_width=right_width,
            left_align="L",
            right_align="R",
        )
        for line in table_text.splitlines():
            if line.startswith("|") and line.endswith("|"):
                body = line[1:-1]
                try:
                    left_segment, right_segment = body.split("|", 1)
                    return f"{left_segment}{right_segment}"
                except ValueError:
                    break
        return f"{left}{' ' * pad}{right}"

    def row(label: str, val: float, indent: str = ""):
        right = _m(val)
        if _is_total_label(label):
            _maybe_insert_total_separator(len(right))
        append_line(_render_kv_line(label, right, indent))

    def hours_row(label: str, val: float, indent: str = ""):
        right = _h(val)
        if _is_total_label(label):
            _maybe_insert_total_separator(len(right))
        append_line(_render_kv_line(label, right, indent))

    def _render_process_and_hours_from_buckets(
        lines: list[str], bucket_view_obj: Mapping[str, Any] | None
    ) -> tuple[float, float, list[tuple[str, float, float, float, float]]]:
        try:
            buckets_candidate = (
                bucket_view_obj.get("buckets") if bucket_view_obj else None
            )
        except Exception:
            buckets_candidate = None
        if isinstance(buckets_candidate, dict):
            buckets = buckets_candidate
        elif isinstance(buckets_candidate, _MappingABC):
            try:
                buckets = dict(buckets_candidate)
            except Exception:
                buckets = {}
        else:
            buckets = {}

        order = [
            "programming",
            "programming_amortized",
            "milling",
            "turning",
            "drilling",
            "tapping",
            "counterbore",
            "countersink",
            "spot_drill",
            "grinding",
            "jig_grind",
            "finishing_deburr",
            "saw_waterjet",
            "wire_edm",
            "sinker_edm",
            "assembly",
            "inspection",
        ]

        canonical_entries: dict[str, dict[str, float]] = {}
        if isinstance(buckets, _MappingABC):
            for raw_key, raw_entry in buckets.items():
                if not isinstance(raw_entry, _MappingABC):
                    continue
                key_str = str(raw_key)
                canon_key = (
                    _canonical_bucket_key(key_str)
                    or _normalize_bucket_key(key_str)
                    or key_str
                )
                minutes_val = max(0.0, _as_float(raw_entry.get("minutes"), 0.0))
                machine_val = max(0.0, _as_float(raw_entry.get("machine$"), 0.0))
                labor_val = max(0.0, _as_float(raw_entry.get("labor$"), 0.0))
                total_val = max(0.0, _as_float(raw_entry.get("total$"), 0.0))
                if total_val <= 0.0:
                    total_val = round(machine_val + labor_val, 2)
                canonical_entries[canon_key] = {
                    "minutes": minutes_val,
                    "machine$": machine_val,
                    "labor$": labor_val,
                    "total$": total_val,
                }

        milling_entry = canonical_entries.get("milling")
        if milling_entry:
            milling_meta = _lookup_process_meta(process_meta, "milling") or {}

            def _maybe_float(value: Any) -> float | None:
                try:
                    number = float(value)
                except Exception:
                    return None
                if not math.isfinite(number):
                    return None
                return number

            milling_minutes = _safe_float(milling_entry.get("minutes"), default=0.0)
            meta_minutes = _safe_float(milling_meta.get("minutes"), default=0.0)
            meta_hours = _safe_float(milling_meta.get("hr"), default=0.0)
            if meta_minutes > 0.0:
                milling_minutes = meta_minutes
            elif meta_hours > 0.0:
                milling_minutes = meta_hours * 60.0

            if milling_minutes > 0.0:
                milling_hours = milling_minutes / 60.0

                def _rate_from_candidates(
                    mapping: Mapping[str, Any] | None,
                    keys: Sequence[str],
                    default: float,
                ) -> float:
                    if not isinstance(mapping, _MappingABC):
                        mapping = {}
                    for key in keys:
                        if not key:
                            continue
                        try:
                            raw = mapping.get(key)  # type: ignore[index]
                        except Exception:
                            raw = None
                        rate_val = _maybe_float(raw)
                        if rate_val is not None and rate_val > 0.0:
                            return rate_val
                    return default

                machine_rate = _rate_from_candidates(
                    rates,
                    (
                        "machine_per_hour",
                        "machine_rate",
                        "milling_rate",
                        "MachineRate",
                        "MillingRate",
                        "ShopMachineRate",
                        "ShopRate",
                    ),
                    90.0,
                )
                labor_rate = _rate_from_candidates(
                    rates,
                    (
                        "labor_per_hour",
                        "labor_rate",
                        "milling_labor_rate",
                        "LaborRate",
                        "ShopLaborRate",
                    ),
                    45.0,
                )

                if cfg is not None:
                    cfg_machine = _maybe_float(getattr(cfg, "machine_rate_per_hr", None))
                    if cfg_machine is not None and cfg_machine > 0.0:
                        machine_rate = cfg_machine
                    cfg_labor = _maybe_float(getattr(cfg, "labor_rate_per_hr", None))
                    if cfg_labor is not None and cfg_labor > 0.0:
                        labor_rate = cfg_labor

                config_sources: list[Mapping[str, Any]] = []

                def _add_config_source(candidate: Any) -> None:
                    if isinstance(candidate, dict):
                        config_sources.append(candidate)
                    elif isinstance(candidate, _MappingABC):
                        config_sources.append(dict(candidate))

                for container in (breakdown, result):
                    if not isinstance(container, _MappingABC):
                        continue
                    _add_config_source(container.get("config"))
                    _add_config_source(container.get("params"))

                if isinstance(result, _MappingABC):
                    quote_state_payload = result.get("quote_state")
                    if isinstance(quote_state_payload, _MappingABC):
                        _add_config_source(quote_state_payload.get("config"))
                        _add_config_source(quote_state_payload.get("params"))

                attended_frac: float | None = None
                if cfg is not None:
                    cfg_frac = _maybe_float(getattr(cfg, "milling_attended_fraction", None))
                    if cfg_frac is not None:
                        attended_frac = cfg_frac

                for source in config_sources:
                    try:
                        candidate = source.get("milling_attended_fraction")
                    except Exception:
                        candidate = None
                    frac_val = _maybe_float(candidate)
                    if frac_val is not None:
                        attended_frac = frac_val
                        break

                if attended_frac is None:
                    attended_frac = 1.0
                attended_frac = max(0.0, min(attended_frac, 1.0))
                milling_labor_hours = milling_hours * attended_frac

                machine_cost = milling_hours * machine_rate
                labor_cost = milling_labor_hours * labor_rate
                total_cost = machine_cost + labor_cost

                milling_entry["minutes"] = round(milling_minutes, 2)
                milling_entry["machine$"] = round(machine_cost, 2)
                milling_entry["labor$"] = round(labor_cost, 2)
                milling_entry["total$"] = round(total_cost, 2)
                canonical_entries["milling"] = milling_entry

                if isinstance(buckets, dict):
                    milling_bucket = buckets.get("milling")
                    if isinstance(milling_bucket, dict):
                        milling_bucket.update(
                            {
                                "minutes": milling_entry["minutes"],
                                "machine$": milling_entry["machine$"],
                                "labor$": milling_entry["labor$"],
                                "total$": milling_entry["total$"],
                            }
                        )

                aggregated_metrics_container = locals().get("aggregated_bucket_minutes")
                if isinstance(aggregated_metrics_container, dict):
                    milling_metrics = aggregated_metrics_container.get("milling")
                    if isinstance(milling_metrics, dict):
                        milling_metrics.update(
                            {
                                "minutes": milling_entry["minutes"],
                                "machine$": milling_entry["machine$"],
                                "labor$": milling_entry["labor$"],
                            }
                        )

                print(
                    f"[CHECK/milling] min={milling_minutes:.2f} hr={milling_hours:.2f} "
                    f"mach_rate={machine_rate:.2f}/hr labor_rate={labor_rate:.2f}/hr "
                    f"machine$={milling_entry['machine$']:.2f} "
                    f"labor$={milling_entry['labor$']:.2f} total$={milling_entry['total$']:.2f}"
                )

        def _append_process_row(
            rows: list[tuple[str, float, float, float, float]],
            label: str,
            minutes_val: float,
            machine_val: float,
            labor_val: float,
            total_val: float,
        ) -> None:
            minutes_clean = max(0.0, _as_float(minutes_val, 0.0))
            machine_clean = max(0.0, _as_float(machine_val, 0.0))
            labor_clean = max(0.0, _as_float(labor_val, 0.0))
            total_clean = max(0.0, _as_float(total_val, 0.0))
            if total_clean <= 0.0:
                total_clean = round(machine_clean + labor_clean, 2)
            if (
                total_clean <= 0.0
                and machine_clean <= 0.0
                and labor_clean <= 0.0
                and minutes_clean <= 0.0
            ):
                return
            rows.append(
                (
                    str(label),
                    minutes_clean,
                    machine_clean,
                    labor_clean,
                    total_clean,
                )
            )

        def _label_for_bucket(canon_key: str) -> str:
            display_label = _display_bucket_label(canon_key, label_overrides)
            if display_label:
                return display_label
            return canon_key or ""

        lines.append("Process & Labor Costs")
        lines.append("-" * 74)
        rows: list[tuple[str, float, float, float, float]] = []

        programming_entry: dict[str, float] | None = None
        programming_entry_label = PROGRAMMING_PER_PART_LABEL
        for candidate in ("programming_amortized", "programming"):
            entry = canonical_entries.pop(candidate, None)
            if entry is not None:
                programming_entry = entry
                if candidate == "programming_amortized":
                    programming_entry_label = PROGRAMMING_AMORTIZED_LABEL
                else:
                    programming_entry_label = PROGRAMMING_PER_PART_LABEL
                break

        prog_minutes = 0.0
        prog_total = 0.0
        if programming_entry is not None:
            prog_minutes = programming_entry.get("minutes", 0.0)
            prog_total = programming_entry.get("total$", 0.0)
            if prog_total <= 0.0:
                prog_total = programming_entry.get("labor$", 0.0)
        if prog_total <= 0.0:
            try:
                prog_total = max(
                    0.0,
                    _as_float(labor_cost_totals.get(PROGRAMMING_PER_PART_LABEL), 0.0),
                )
            except Exception:
                prog_total = 0.0
        if prog_minutes <= 0.0:
            try:
                prog_minutes = max(0.0, _as_float(programming_minutes, 0.0))
            except NameError:
                prog_minutes = 0.0
        if prog_total > 0.0 or prog_minutes > 0.0:
            _append_process_row(
                rows,
                programming_entry_label,
                prog_minutes,
                0.0,
                prog_total,
                prog_total,
            )

        def _consume_entry(canon_key: str) -> None:
            entry = canonical_entries.pop(canon_key, None)
            if not entry:
                return
            minutes_val = entry.get("minutes", 0.0)
            machine_val = entry.get("machine$", 0.0)
            labor_val = entry.get("labor$", 0.0)
            total_val = entry.get("total$", 0.0)
            _append_process_row(
                rows,
                _label_for_bucket(canon_key),
                minutes_val,
                machine_val,
                labor_val,
                total_val,
            )

        for bucket_key in order:
            _consume_entry(bucket_key)

        if canonical_entries:
            for canon_key, entry in sorted(
                canonical_entries.items(),
                key=lambda item: _label_for_bucket(item[0]).lower(),
            ):
                minutes_val = entry.get("minutes", 0.0)
                machine_val = entry.get("machine$", 0.0)
                labor_val = entry.get("labor$", 0.0)
                total_val = entry.get("total$", 0.0)
                _append_process_row(
                    rows,
                    _label_for_bucket(canon_key),
                    minutes_val,
                    machine_val,
                    labor_val,
                    total_val,
                )

        row_canon_keys = {
            _canonical_bucket_key(row_label) or _normalize_bucket_key(row_label)
            for row_label, *_ in rows
        }
        if "tapping" not in row_canon_keys:
            tapping_bucket: Mapping[str, Any] | None = None
            for candidate_key in ("tapping", "Tapping"):
                entry = buckets.get(candidate_key)
                if isinstance(entry, _MappingABC):
                    tapping_bucket = entry
                    break
            if tapping_bucket is None:
                tapping_bucket = {}
            tapping_minutes = _as_float(tapping_bucket.get("minutes", 0.0), 0.0)
            tapping_machine = _as_float(tapping_bucket.get("machine$", 0.0), 0.0)
            tapping_labor = _as_float(tapping_bucket.get("labor$", 0.0), 0.0)
            tapping_total = _as_float(tapping_bucket.get("total$", 0.0), 0.0)
            if (
                tapping_minutes > 0.0
                or tapping_machine > 0.0
                or tapping_labor > 0.0
                or tapping_total > 0.0
            ):
                _append_process_row(
                    rows,
                    _label_for_bucket("tapping"),
                    tapping_minutes,
                    tapping_machine,
                    tapping_labor,
                    tapping_total,
                )

        total_cost = sum(row[4] for row in rows)
        total_minutes = sum(row[1] for row in rows)

        if rows:
            headers = ("Process", "Minutes", "Machine $", "Labor $", "Total $")
            display_rows: list[tuple[str, str, str, str, str]] = []
            for name, minutes_val, machine_val, labor_val, total_val in rows:
                display_rows.append(
                    (
                        str(name),
                        f"{minutes_val:,.2f}",
                        _m(machine_val),
                        _m(labor_val),
                        _m(total_val),
                    )
                )

            total_row = ("Total", "", "", "", _m(total_cost))
            width_candidates = display_rows + [total_row]
            col_widths = [len(header) for header in headers]
            for row_values in width_candidates:
                for idx, value in enumerate(row_values):
                    col_widths[idx] = max(col_widths[idx], len(value))

            def _format_row(values: Sequence[str]) -> str:
                pieces: list[str] = []
                for idx, value in enumerate(values):
                    align = "L" if idx == 0 else "R"
                    width = col_widths[idx]
                    if align == "L":
                        pieces.append(value.ljust(width))
                    else:
                        pieces.append(value.rjust(width))
                return "  " + "  ".join(pieces)

            header_line = _format_row(headers)
            separator_line = "  " + "  ".join("-" * width for width in col_widths)
            lines.append(header_line)
            lines.append(separator_line)
            for row in display_rows:
                lines.append(_format_row(row))
            lines.append(separator_line)
            lines.append(_format_row(total_row))
            for display_label, *_ in rows:
                add_process_notes(display_label)
            lines.append("")
            return total_cost, total_minutes, rows

        lines.append("  (no bucket data)")
        lines.append("")
        return 0.0, 0.0, []

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
            for segment in _RE_SPLIT(r";\s*", str(existing)):
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
            for segment in _RE_SPLIT(r";\s*", str(existing)):
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

        bucket_minutes_val = 0.0
        bucket_entry: Mapping[str, Any] | None = None
        spec_for_bucket: Any = None
        lookup_candidates: list[str] = []
        if canon_key:
            lookup_candidates.append(canon_key)
        normalized_key = _normalize_bucket_key(key)
        if normalized_key:
            lookup_candidates.append(normalized_key)
        key_str = str(key or "").strip()
        if key_str:
            lookup_candidates.append(key_str)
        display_label = (
            _display_bucket_label(canon_key, label_overrides)
            if canon_key
            else key_str
        )
        if display_label:
            lookup_candidates.append(display_label)

        skip_detail_labels = {
            PROGRAMMING_PER_PART_LABEL,
            PROGRAMMING_AMORTIZED_LABEL,
            "Fixture Build (amortized)",
        }

        if display_label not in skip_detail_labels:
            for candidate in lookup_candidates:
                detail_candidate = detail_lookup.get(str(candidate or ""))
                if detail_candidate not in (None, ""):
                    write_detail(str(detail_candidate), indent)
                    return

        for candidate in lookup_candidates:
            if bucket_minutes_val <= 0.0 and candidate in bucket_minutes_detail:
                bucket_minutes_val = _safe_float(
                    bucket_minutes_detail.get(candidate), default=0.0
                )
            if bucket_entry is None and candidate in bucket_entries_for_totals_map:
                bucket_entry = typing.cast(
                    Mapping[str, Any], bucket_entries_for_totals_map[candidate]
                )
            if spec_for_bucket is None:
                spec_candidate = None
                try:
                    spec_candidate = bucket_specs_by_canon.get(candidate)  # type: ignore[name-defined]
                except Exception:
                    spec_candidate = None
                if not spec_candidate:
                    try:
                        spec_candidate = bucket_specs_for_render.get(candidate)  # type: ignore[name-defined]
                    except Exception:
                        spec_candidate = None
                if spec_candidate is not None:
                    spec_for_bucket = spec_candidate

        meta = _lookup_process_meta(process_meta, key) or {}
        canon_for_notes = str(
            _canonical_bucket_key(key)
            or _normalize_bucket_key(key)
            or (key or "")
        ).strip().lower()
        footer_hours = 0.0
        has_bucket_minutes = False
        if bucket_minutes_val > 0.0:
            footer_hours = bucket_minutes_val / 60.0
            has_bucket_minutes = footer_hours > 0.0
        elif isinstance(bucket_entry, _MappingABC):
            entry_minutes = _safe_float(bucket_entry.get("minutes"), default=0.0)
            if entry_minutes > 0.0:
                footer_hours = entry_minutes / 60.0
                has_bucket_minutes = footer_hours > 0.0
        if not has_bucket_minutes:
            return

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
            if rate_float <= 0 and meta:
                rate_val = meta.get("rate")
                try:
                    rate_float = float(rate_val or 0.0)
                except Exception:
                    rate_float = 0.0
        if rate_float <= 0 and stored_cost > 0:
            hours_for_rate = footer_hours if footer_hours > 0 else stored_hours
            if hours_for_rate > 0:
                rate_float = stored_cost / hours_for_rate
        if rate_float <= 0:
            rate_key = _rate_key_for_bucket(str(key))
            if rate_key:
                try:
                    rate_float = float(rates.get(rate_key, 0.0) or 0.0)
                except Exception:
                    rate_float = 0.0

        total_from_bucket = 0.0
        if isinstance(bucket_entry, _MappingABC):
            machine_component = _safe_float(bucket_entry.get("machine$"), default=0.0)
            labor_component = _safe_float(bucket_entry.get("labor$"), default=0.0)
            total_from_bucket = _safe_float(bucket_entry.get("total$"), default=0.0)
            if total_from_bucket <= 0.0:
                total_from_bucket = machine_component + labor_component
        if total_from_bucket <= 0.0 and spec_for_bucket is not None:
            try:
                total_from_bucket = float(getattr(spec_for_bucket, "total", 0.0) or 0.0)
            except Exception:
                total_from_bucket = 0.0
        if total_from_bucket > 0.0:
            stored_cost = total_from_bucket
            if footer_hours > 0.0 and rate_float <= 0.0:
                rate_float = total_from_bucket / footer_hours

        # Milling/Drilling/Inspection: prefer canonical rates instead of
        # reverse-computing them from bucket totals, which can drift when the
        # user overrides costs.
        canonical_minutes = bucket_minutes_val
        if canonical_minutes <= 0.0 and isinstance(bucket_entry, _MappingABC):
            canonical_minutes = _safe_float(bucket_entry.get("minutes"), default=0.0)
        if canonical_minutes <= 0.0 and footer_hours > 0.0:
            canonical_minutes = footer_hours * 60.0

        def _cfg_rate_fallback(attr: str) -> float:
            try:
                return float(getattr(cfg, attr, 0.0) or 0.0)
            except Exception:
                return 0.0

        if canonical_minutes > 0.0 and canon_for_notes in {"milling", "drilling", "inspection"}:
            hours_val = canonical_minutes / 60.0
            machine_component_val = machine_component if "machine_component" in locals() else 0.0
            labor_component_val = labor_component if "labor_component" in locals() else 0.0

            def _bucket_rate_value(
                bucket_key: str,
                *,
                mode: str,
                component: float,
                rate_key: str | None,
                fallback_keys: tuple[str, ...],
            ) -> float:
                if hours_val > 0.0 and component > 0.0:
                    rate_val = component / hours_val
                    if rate_val > 0.0:
                        return rate_val
                source_map: Mapping[str, Any] | None = None
                try:
                    source_map = merged_two_bucket_rates.get(mode, {})
                except Exception:
                    source_map = None
                if isinstance(source_map, Mapping):
                    for candidate in (
                        bucket_key,
                        _normalize_bucket_key(bucket_key),
                        _display_bucket_label(bucket_key, None),
                    ):
                        if not candidate:
                            continue
                        resolved = _safe_float(source_map.get(candidate), default=0.0)
                        if resolved > 0.0:
                            return resolved
                raw = rates.get(rate_key) if rate_key else None
                return _resolve_rate_with_fallback(raw, *fallback_keys)

            if canon_for_notes == "milling":
                machine_rate = _bucket_rate_value(
                    "milling",
                    mode="machine",
                    component=machine_component_val,
                    rate_key="MillingRate",
                    fallback_keys=("MachineRate", "machine_rate", "machine"),
                )
                if machine_rate <= 0.0:
                    cfg_machine = _cfg_rate_fallback("machine_rate_per_hr")
                    if cfg_machine > 0.0:
                        machine_rate = cfg_machine
                labor_rate = _bucket_rate_value(
                    "milling",
                    mode="labor",
                    component=labor_component_val,
                    rate_key="MillingLaborRate",
                    fallback_keys=("LaborRate", "labor_rate", "labor"),
                )
                if labor_rate <= 0.0:
                    cfg_labor = _cfg_rate_fallback("labor_rate_per_hr")
                    if cfg_labor > 0.0:
                        labor_rate = cfg_labor
                line = f"Milling: {hours_val:.2f} hr"
                if machine_rate > 0.0:
                    line += f" @ ${machine_rate:.2f}/hr (machine)"
                else:
                    line += " (machine)"
                if labor_component_val > 0.0 and labor_rate > 0.0:
                    line += f" + ${labor_rate:.2f}/hr (labor)"
                write_line(line, indent)
                return

            if canon_for_notes == "drilling":
                machine_rate = _bucket_rate_value(
                    "drilling",
                    mode="machine",
                    component=machine_component_val,
                    rate_key="DrillingRate",
                    fallback_keys=("MachineRate", "machine_rate", "machine"),
                )
                if machine_rate <= 0.0:
                    cfg_machine = _cfg_rate_fallback("machine_rate_per_hr")
                    if cfg_machine > 0.0:
                        machine_rate = cfg_machine
                labor_rate = _bucket_rate_value(
                    "drilling",
                    mode="labor",
                    component=labor_component_val,
                    rate_key="DrillingLaborRate",
                    fallback_keys=("LaborRate", "labor_rate", "labor"),
                )
                if labor_rate <= 0.0:
                    cfg_labor = _cfg_rate_fallback("labor_rate_per_hr")
                    if cfg_labor > 0.0:
                        labor_rate = cfg_labor
                line = f"Drilling: {hours_val:.2f} hr"
                if machine_rate > 0.0:
                    line += f" @ ${machine_rate:.2f}/hr (machine)"
                else:
                    line += " (machine)"
                if labor_component_val > 0.0 and labor_rate > 0.0:
                    line += f" + ${labor_rate:.2f}/hr (labor)"
                write_line(line, indent)
                return

            if canon_for_notes == "inspection":
                labor_rate = _bucket_rate_value(
                    "inspection",
                    mode="labor",
                    component=labor_component_val,
                    rate_key="InspectionRate",
                    fallback_keys=("LaborRate", "labor_rate", "labor"),
                )
                if labor_rate <= 0.0:
                    cfg_labor = _cfg_rate_fallback("labor_rate_per_hr")
                    if cfg_labor > 0.0:
                        labor_rate = cfg_labor
                line = f"Inspection: {hours_val:.2f} hr"
                if labor_rate > 0.0:
                    line += f" @ ${labor_rate:.2f}/hr (labor)"
                else:
                    line += " (labor)"
                write_line(line, indent)
                return

        write_line(_hours_with_rate_text(footer_hours, rate_float), indent)

    def add_pass_basis(key: str, indent: str = "    "):
        basis_map = breakdown.get("pass_basis", {}) or {}
        info = basis_map.get(key) or {}
        txt = info.get("basis") or info.get("note")
        if txt:
            write_line(str(txt), indent)

    hour_summary_entries: dict[str, tuple[float, bool]] = {}
    ui_vars = result.get("ui_vars") or {}
    if not isinstance(ui_vars, dict):
        ui_vars = {}
    g_source = geometry if isinstance(geometry, _MappingABC) else result.get("geom") or result.get("geo")
    if isinstance(g_source, _MappingABC):
        g = dict(g_source) if not isinstance(g_source, dict) else dict(g_source)
    else:
        g = {}
    # Ensure a consistent alias used throughout this renderer
    geo_context = g
    drill_debug_entries: list[str] = []
    # Selected removal summary (if available) for compact debug table later
    removal_summary_for_display: Mapping[str, Any] | None = None
    _accumulate_drill_debug(drill_debug_entries, result, breakdown)
    # If the removal summary has a total, force machine hours for drilling
    removal = (result or {}).get("removal_summary") or {}
    mins = float(removal.get("total_minutes") or 0.0)

    def _has_planner_drilling_bucket(candidate: Any) -> bool:
        if isinstance(candidate, _MappingABC):
            items_iter = candidate.items()
        elif isinstance(candidate, dict):
            items_iter = candidate.items()
        else:
            return False
        for raw_key, raw_value in items_iter:
            key_text = str(raw_key or "").strip().lower()
            if key_text == "drilling":
                return True
            if key_text == "buckets" and raw_value is not candidate:
                if _has_planner_drilling_bucket(raw_value):
                    return True
        return False

    buckets = (breakdown or {}).get("planner_buckets") or {}
    bucket_view_candidate: Any = None
    if isinstance(breakdown, _MappingABC):
        bucket_view_candidate = breakdown.get("bucket_view")
    elif isinstance(breakdown, dict):
        bucket_view_candidate = breakdown.get("bucket_view")

    planner_has_drilling_bucket = False
    if not planner_has_drilling_bucket:
        planner_has_drilling_bucket = _has_planner_drilling_bucket(bucket_view_candidate)
    if not planner_has_drilling_bucket:
        planner_has_drilling_bucket = _has_planner_drilling_bucket(buckets)

    # Canonical QUOTE SUMMARY header (legacy variants removed in favour of this
    # block so the Speeds/Feeds status + Drill Debug output stay consistent).
    header_lines, pricing_source_value = build_quote_header_lines(
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
        _push(lines, MATERIAL_WARNING_LABEL)
    _push(lines, "")

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
        _push(lines, "Drill Debug")
        _push(lines, divider)
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
                    _push(lines, "")
            else:
                write_wrapped(text, "  ")
        if lines and lines[-1] != "":
            _push(lines, "")

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
    total_process_cost_label = "Total Process Cost:"
    row(total_process_cost_label, float(totals.get("labor_cost", 0.0)))
    total_process_cost_row_index = len(lines) - 1
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
            _push(lines, "")
            _push(lines, "Red Flags")
            _push(lines, divider)
            for flag in display_red_flags:
                write_wrapped(f"⚠️ {flag}", "  ")
    _push(lines, "")

    narrative = result.get("narrative") or breakdown.get("narrative")
    why_parts: list[str] = []
    why_lines: list[str] = []
    bucket_why_summary_line: str | None = None
    material_total_for_why = 0.0
    if narrative:
        if isinstance(narrative, str):
            parts = [seg.strip() for seg in _RE_SPLIT(r"(?<=\.)\s+", narrative) if seg.strip()]
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
    planner_ops_summary: list[dict[str, Any]] = []
    extra_bucket_ops: MutableMapping[str, Any] | dict[str, Any]
    extra_bucket_ops = {}
    if isinstance(breakdown, _MappingABC):
        try:
            extra_bucket_ops = dict(breakdown.get("extra_bucket_ops") or {})
        except Exception:
            extra_bucket_ops = {}
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

            name_text = str(op_name or "").strip()
            qty_candidate = entry.get("qty")
            try:
                qty_val = int(float(qty_candidate))
            except Exception:
                qty_val = 0
            side_val = (
                entry.get("side")
                or entry.get("face")
                or entry.get("orientation")
                or entry.get("side_label")
            )
            if name_text:
                planner_ops_summary.append(
                    {"name": name_text, "qty": qty_val, "side": side_val}
                )

    if isinstance(extra_bucket_ops, _MappingABC) and extra_bucket_ops:
        seen_summary_keys = {
            (
                str(item.get("name") or "").strip(),
                (str(item.get("side") or "").strip().lower() or None),
            )
            for item in planner_ops_summary
            if isinstance(item, _MappingABC)
        }
        for raw_bucket, raw_entries in extra_bucket_ops.items():
            if not isinstance(raw_entries, Sequence):
                continue
            bucket_key = _canonical_bucket_key(raw_bucket) or _planner_bucket_for_op(
                str(raw_bucket)
            )
            for entry in raw_entries:
                if not isinstance(entry, _MappingABC):
                    continue
                name_text = str(entry.get("name") or entry.get("op") or "").strip()
                if not name_text:
                    continue
                qty_raw = entry.get("qty")
                try:
                    qty_val = int(float(qty_raw))
                except Exception:
                    qty_val = 0
                side_val = entry.get("side")
                side_key = (str(side_val or "").strip().lower() or None)
                summary_key = (name_text, side_key)
                if summary_key not in seen_summary_keys:
                    planner_ops_summary.append(
                        {"name": name_text, "qty": qty_val, "side": side_val}
                    )
                    seen_summary_keys.add(summary_key)
                if not bucket_key:
                    continue
                minutes_val_raw = entry.get("minutes")
                if minutes_val_raw in (None, ""):
                    minutes_val_raw = entry.get("mins")
                try:
                    minutes_val = float(minutes_val_raw)
                except Exception:
                    minutes_val = 0.0
                machine_raw = entry.get("machine") or entry.get("machine_cost")
                try:
                    machine_val = float(machine_raw)
                except Exception:
                    machine_val = 0.0
                labor_raw = entry.get("labor") or entry.get("labor_cost")
                try:
                    labor_val = float(labor_raw)
                except Exception:
                    labor_val = 0.0
                total_raw = entry.get("total") or entry.get("total_cost")
                try:
                    total_val = float(total_raw)
                except Exception:
                    total_val = machine_val + labor_val
                if not any(value > 0 for value in (minutes_val, machine_val, labor_val, total_val)):
                    continue
                bucket_ops = bucket_ops_map.setdefault(bucket_key, [])
                bucket_ops.append(
                    {
                        "op": name_text,
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
        "programming": PROGRAMMING_PER_PART_LABEL,
        "programming_amortized": PROGRAMMING_PER_PART_LABEL,
    }

    # Process & Labor table rendered later using the canonical planner bucket view

    bucket_order = list(PROCESS_BUCKETS.order)
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

    def _lookup_blank(container: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(container, _MappingABC):
            return None
        for key in ("required_blank_in", "bbox_in"):
            entry = container.get(key)
            if isinstance(entry, _MappingABC):
                return entry
        return None

    blank_sources: list[Mapping[str, Any] | None] = []
    for parent in (breakdown, result):
        if isinstance(parent, _MappingABC):
            blank_sources.append(_lookup_blank(parent.get("geo")))
            blank_sources.append(_lookup_blank(parent.get("geo_context")))
            blank_sources.append(_lookup_blank(parent.get("geom")))
    blank_sources.append(_lookup_blank(geo_context if isinstance(geo_context, _MappingABC) else None))
    blank_sources.append(_lookup_blank(g if isinstance(g, _MappingABC) else None))
    blank_sources.append(_lookup_blank(pricing_geom if isinstance(pricing_geom, _MappingABC) else None))

    req_map: Mapping[str, Any] | None = next((entry for entry in blank_sources if entry), None)
    w_val = _coerce_positive_float(req_map.get("w")) if req_map else None
    h_val = _coerce_positive_float(req_map.get("h")) if req_map else None
    t_val = _coerce_positive_float(req_map.get("t")) if req_map else None
    w = float(w_val) if w_val else 0.0
    h = float(h_val) if h_val else 0.0
    t = float(t_val) if t_val else 0.0
    if t <= 0:
        t = 2.0

    if w <= 0 or h <= 0:
        geo_candidates: list[Mapping[str, Any]] = []
        for parent in (breakdown, result):
            if isinstance(parent, _MappingABC):
                for key in ("geo", "geo_context", "geom"):
                    candidate = parent.get(key)
                    if isinstance(candidate, _MappingABC):
                        geo_candidates.append(candidate)
        if isinstance(geo_context, _MappingABC):
            geo_candidates.append(geo_context)
        if isinstance(g, _MappingABC):
            geo_candidates.append(g)
        inferred_w = inferred_h = 0.0
        for geo_candidate in geo_candidates:
            inferred_w, inferred_h = _infer_rect_from_holes(geo_candidate)
            if inferred_w > 0 and inferred_h > 0:
                break
        if inferred_w > 0 and w <= 0:
            w = float(inferred_w)
        if inferred_h > 0 and h <= 0:
            h = float(inferred_h)

    required_blank = (float(w), float(h), float(t))
    blank_len = max(required_blank[0], required_blank[1]) if required_blank[0] > 0 and required_blank[1] > 0 else 0.0
    blank_wid = min(required_blank[0], required_blank[1]) if required_blank[0] > 0 and required_blank[1] > 0 else 0.0

    def _blank_has_dims(candidate: Mapping[str, Any] | None) -> bool:
        if not isinstance(candidate, _MappingABC):
            return False
        return bool(
            _coerce_positive_float(candidate.get("w"))
            and _coerce_positive_float(candidate.get("h"))
        )

    def _apply_blank_hint(container: Mapping[str, Any] | None) -> None:
        if not isinstance(container, _MutableMappingABC):
            return
        if required_blank[0] <= 0 or required_blank[1] <= 0:
            return
        hint_payload: dict[str, Any] = {"w": float(required_blank[0]), "h": float(required_blank[1])}
        if required_blank[2] > 0:
            hint_payload["t"] = float(required_blank[2])
        existing_blank = container.get("required_blank_in")
        if not _blank_has_dims(existing_blank):
            container["required_blank_in"] = dict(hint_payload)
        existing_bbox = container.get("bbox_in")
        if not _blank_has_dims(existing_bbox):
            container.setdefault("bbox_in", dict(hint_payload))

    for context in (
        geo_context if isinstance(geo_context, _MutableMappingABC) else None,
        g if isinstance(g, _MutableMappingABC) else None,
        pricing_geom if isinstance(pricing_geom, _MutableMappingABC) else None,
    ):
        _apply_blank_hint(context)
    for parent in (breakdown, result):
        if isinstance(parent, _MutableMappingABC):
            for key in ("geo", "geo_context", "geom"):
                _apply_blank_hint(parent.get(key))

    for container in (
        material if isinstance(material, _MutableMappingABC) else None,
        material_stock_block if isinstance(material_stock_block, _MutableMappingABC) else None,
    ):
        _apply_blank_hint(container)

    if blank_len > 0 and isinstance(material_stock_block, _MutableMappingABC):
        if _coerce_positive_float(material_stock_block.get("required_blank_len_in")) is None:
            material_stock_block["required_blank_len_in"] = float(blank_len)
        if _coerce_positive_float(material_stock_block.get("required_blank_wid_in")) is None:
            material_stock_block["required_blank_wid_in"] = float(blank_wid)
        if required_blank[2] > 0 and _coerce_positive_float(
            material_stock_block.get("required_blank_thk_in")
        ) is None:
            material_stock_block["required_blank_thk_in"] = float(required_blank[2])

    required_blank_len = float(blank_len) if blank_len > 0 else 0.0
    required_blank_wid = float(blank_wid) if blank_wid > 0 else 0.0

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
            _push(lines, "Material & Stock")
            _push(lines, divider)
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
                _push(lines, f"  Material used:  {material_name_display}")

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

            if (need_len is None or need_len <= 0) and required_blank_len > 0:
                need_len = float(required_blank_len)
            if (need_wid is None or need_wid <= 0) and required_blank_wid > 0:
                need_wid = float(required_blank_wid)
            if (need_thk is None or need_thk <= 0) and required_blank[2] > 0:
                need_thk = float(required_blank[2])

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
            _push(lines, "")

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
    _push(lines, "NRE / Setup Costs (per lot)")
    _push(lines, divider)
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
        _push(lines, "")

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
    removal_drilling_hours_precise: float | None = None
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
    removal_summary_lines: list[str] = []
    removal_summary_extra_lines: list[str] = []
    removal_summary_lines: list[str] = []
    removal_card_extra: dict[str, float] = {}
    speeds_feeds_table = None
    if isinstance(result, _MappingABC):
        candidate_sf = result.get("speeds_feeds_table")
        if candidate_sf is not None:
            speeds_feeds_table = candidate_sf
    if speeds_feeds_table is None and isinstance(breakdown, _MappingABC):
        candidate_sf = breakdown.get("speeds_feeds_table")
        if candidate_sf is not None:
            speeds_feeds_table = candidate_sf

    material_group_display: str | None = None
    if isinstance(drilling_meta_map, _MappingABC):
        for key in ("material_group", "group"):
            candidate_group = drilling_meta_map.get(key)
            if isinstance(candidate_group, str) and candidate_group.strip():
                material_group_display = candidate_group.strip()
                break
    if material_group_display is None and isinstance(result, _MappingABC):
        candidate_group = result.get("material_group")
        if isinstance(candidate_group, str) and candidate_group.strip():
            material_group_display = candidate_group.strip()
    if material_group_display is None and isinstance(breakdown, _MappingABC):
        candidate_group = breakdown.get("material_group")
        if isinstance(candidate_group, str) and candidate_group.strip():
            material_group_display = candidate_group.strip()

    ctx_a: Mapping[str, Any] | None = (
        typing.cast(Mapping[str, Any], breakdown)
        if isinstance(breakdown, _MappingABC)
        else None
    )
    ctx_b: Mapping[str, Any] | None = (
        typing.cast(Mapping[str, Any], result)
        if isinstance(result, _MappingABC)
        else None
    )
    ctx: Mapping[str, Any] | None = None
    quote_candidate = locals().get("quote")
    if isinstance(quote_candidate, dict) and quote_candidate:
        ctx = typing.cast(Mapping[str, Any], quote_candidate)
    elif isinstance(ctx_a, dict) and ctx_a:
        ctx = ctx_a
    elif isinstance(ctx_b, dict) and ctx_b:
        ctx = ctx_b
    elif isinstance(ctx_a, _MappingABC):
        ctx = ctx_a
    elif isinstance(ctx_b, _MappingABC):
        ctx = ctx_b

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
        speeds_feeds_table=speeds_feeds_table,
        material_group=material_group_display,
        drilling_time_per_hole=drilling_time_per_hole_data,
    )

    drill_actions_from_groups = 0
    if isinstance(removal_card_extra, (_MappingABC, dict)):
        try:
            drill_actions_from_groups = int(
                round(
                    float(
                        typing.cast(Mapping[str, Any], removal_card_extra).get(
                            "drill_actions_from_groups", 0
                        )
                        or 0
                    )
                )
            )
        except Exception:
            drill_actions_from_groups = 0

    adjusted_drill_groups = _adjusted_drill_groups_for_display(
        breakdown,
        typing.cast(Mapping[str, Any], result) if isinstance(result, _MappingABC) else None,
        typing.cast(Mapping[str, Any], drilling_meta_map)
        if isinstance(drilling_meta_map, _MappingABC)
        else None,
    )
    if adjusted_drill_groups and isinstance(removal_card_lines, list):
        counts_by_diam: dict[float, int] = {}
        depth_by_diam: dict[float, float] = {}
        for group in adjusted_drill_groups:
            if not isinstance(group, _MappingABC):
                continue
            dia_val = _coerce_float_or_none(group.get("diameter_in"))
            qty_val = _coerce_float_or_none(group.get("qty"))
            if dia_val is None or qty_val is None:
                continue
            qty_int = int(round(float(qty_val)))
            if qty_int <= 0:
                continue
            key = round(float(dia_val), 4)
            counts_by_diam[key] = qty_int
            depth_val = _coerce_float_or_none(group.get("depth_in"))
            if depth_val is not None and math.isfinite(depth_val):
                depth_by_diam.setdefault(key, float(depth_val))

        if counts_by_diam:
            info_by_dia: dict[float, dict[str, float]] = {}

            def _merge_group_info(source: Sequence[Any] | None) -> None:
                if not isinstance(source, Sequence):
                    return
                for entry in source:
                    if not isinstance(entry, _MappingABC):
                        continue
                    dia_candidate = _coerce_float_or_none(entry.get("diameter_in"))
                    if dia_candidate is None:
                        continue
                    dia_key = round(float(dia_candidate), 4)
                    target = info_by_dia.setdefault(dia_key, {})
                    depth_candidate = _coerce_float_or_none(entry.get("depth_in"))
                    if depth_candidate is not None and math.isfinite(depth_candidate):
                        target["depth"] = float(depth_candidate)
                    sfm_candidate = _coerce_float_or_none(entry.get("sfm"))
                    if sfm_candidate is not None and math.isfinite(sfm_candidate):
                        target["sfm"] = float(sfm_candidate)
                    ipr_candidate = _coerce_float_or_none(entry.get("ipr"))
                    if ipr_candidate is not None and math.isfinite(ipr_candidate):
                        target["ipr"] = float(ipr_candidate)
                    per_candidate = (
                        _coerce_float_or_none(entry.get("minutes_per_hole"))
                        or _coerce_float_or_none(entry.get("t_per_hole_min"))
                    )
                    if per_candidate is not None and math.isfinite(per_candidate):
                        target["t_per"] = float(per_candidate)
                    qty_candidate = _coerce_float_or_none(entry.get("qty"))
                    if qty_candidate is not None and math.isfinite(qty_candidate):
                        target["qty"] = float(qty_candidate)

            dtph_rows_source = None
            if isinstance(drilling_time_per_hole_data, _MappingABC):
                dtph_rows_source = drilling_time_per_hole_data.get("rows")
            _merge_group_info(dtph_rows_source if isinstance(dtph_rows_source, Sequence) else None)

            if isinstance(drilling_card_detail, _MappingABC):
                _merge_group_info(drilling_card_detail.get("drill_groups"))

            if isinstance(drilling_meta_map, _MappingABC):
                _merge_group_info(drilling_meta_map.get("bins_list"))

            hdr = "MATERIAL REMOVAL – DRILLING"
            try:
                start_idx = next(
                    idx
                    for idx, item in enumerate(removal_card_lines)
                    if isinstance(item, str) and item.strip().upper().startswith(hdr)
                )
            except StopIteration:
                start_idx = None

            if start_idx is not None:
                end_idx = start_idx + 1
                while end_idx < len(removal_card_lines):
                    text = str(removal_card_lines[end_idx] or "")
                    if (
                        text.strip().startswith("MATERIAL REMOVAL –")
                        and end_idx > start_idx
                    ):
                        break
                    end_idx += 1

                group_start = start_idx + 1
                while group_start < len(removal_card_lines):
                    text = str(removal_card_lines[group_start] or "")
                    if text.strip().upper().startswith("DIA "):
                        break
                    group_start += 1

                tail_start = group_start
                while tail_start < len(removal_card_lines):
                    text = str(removal_card_lines[tail_start] or "")
                    if text.strip().startswith("Toolchange adders"):
                        break
                    tail_start += 1

                if (
                    start_idx < len(removal_card_lines)
                    and tail_start < len(removal_card_lines)
                    and tail_start < end_idx
                ):
                    existing_info: dict[float, dict[str, float]] = {}
                    drill_line_re = re.compile(
                        r"Dia\s+([0-9.]+)\"\s*[x×]\s*(\d+).*?depth\s*([0-9.]+)\"\s*\|\s*([0-9.]+)\s*sfm\s*\|\s*([0-9.]+)\s*ipr\s*\|\s*t/hole\s*([0-9.]+)",
                        re.IGNORECASE,
                    )
                    for line in removal_card_lines[group_start:tail_start]:
                        if not isinstance(line, str):
                            continue
                        match = drill_line_re.search(line)
                        if not match:
                            continue
                        dia_key = round(float(match.group(1)), 4)
                        existing_info[dia_key] = {
                            "qty": float(match.group(2)),
                            "depth": float(match.group(3)),
                            "sfm": float(match.group(4)),
                            "ipr": float(match.group(5)),
                            "t_per": float(match.group(6)),
                        }
                    for dia_key, payload in existing_info.items():
                        info_by_dia.setdefault(dia_key, {}).update(payload)

                    existing_counts = {
                        dia: int(round(info.get("qty", 0.0)))
                        for dia, info in existing_info.items()
                        if info.get("qty", 0.0) > 0
                    }
                    if (
                        {dia for dia, qty in existing_counts.items() if qty > 0} !=
                        {dia for dia, qty in counts_by_diam.items() if qty > 0}
                        or any(existing_counts.get(dia) != qty for dia, qty in counts_by_diam.items())
                    ):
                        header_lines = [
                            str(item)
                            for item in removal_card_lines[start_idx:group_start]
                        ]
                        tail_lines = [
                            str(item)
                            for item in removal_card_lines[tail_start:end_idx]
                        ]
                        new_group_lines: list[str] = []
                        for dia_key in sorted(counts_by_diam.keys()):
                            qty = counts_by_diam[dia_key]
                            info = info_by_dia.get(dia_key, {})
                            depth_val = depth_by_diam.get(dia_key) or info.get("depth") or 0.0
                            sfm_val = info.get("sfm", 0.0)
                            ipr_val = info.get("ipr", 0.0)
                            t_per_val = info.get("t_per", 0.0)
                            group_total = qty * t_per_val
                            new_group_lines.append(
                                f'Dia {dia_key:.3f}" × {qty}  | depth {depth_val:.3f}" | '
                                f"{int(round(sfm_val))} sfm | {ipr_val:.4f} ipr | "
                                f"t/hole {t_per_val:.2f} min | group {qty}×{t_per_val:.2f} = {group_total:.2f} min"
                            )
                        replacement = header_lines + new_group_lines + [""] + tail_lines
                        removal_card_lines[start_idx:end_idx] = replacement

    # Extra MATERIAL REMOVAL cards from HOLE TABLE text (Counterbore / Spot / Jig)
    extra_ops_lines = _build_ops_cards_from_chart_lines(
        breakdown=breakdown,
        result=result,
        rates=rates,
        breakdown_mutable=breakdown_mutable,  # so buckets get minutes
        ctx=ctx,
        ctx_a=ctx_a,
        ctx_b=ctx_b,
    )
    if extra_ops_lines:
        removal_card_lines.extend(extra_ops_lines)
        lines.extend(extra_ops_lines)
        try:
            _normalize_buckets(breakdown.get("bucket_view"))
        except Exception:
            pass
    else:
        try:
            _push(
                lines,
                "[DEBUG] extra_ops_lines=0 (no chart_lines/rows visible at callsite)",
            )
        except Exception:
            pass

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
    if (
        removal_drilling_hours_precise is not None
        and math.isfinite(removal_drilling_hours_precise)
    ):
        hours_snapshot = float(removal_drilling_hours_precise)
        if total_minutes_snapshot <= 0.0 and hours_snapshot > 0.0:
            total_minutes_snapshot = hours_snapshot * 60.0
        elif math.isclose(total_minutes_snapshot, hours_snapshot, rel_tol=1e-9, abs_tol=1e-6):
            total_minutes_snapshot = hours_snapshot * 60.0
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
            total_minutes_sanitized = _sanitize_drill_removal_minutes(
                total_minutes_snapshot or 0.0
            )
            minutes_value = round(total_minutes_sanitized, 2)
            extra_map["drill_total_minutes"] = minutes_value
            try:
                logger.info("[removal] drill_total_minutes=%s", minutes_value)
            except Exception:
                pass
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
        return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")

    laborish_aliases: set[str] = set()
    for bucket_key, role in BUCKET_ROLE.items():
        if bucket_key == "_default" or role != "labor_only":
            continue
        for alias in PROCESS_BUCKETS.aliases(bucket_key):
            laborish_aliases.add(alias)
    LABORISH = {_norm(alias) for alias in laborish_aliases if alias}

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
            if minutes_val < 0.0:
                minutes_val = 0.0
            hours_val = _minutes_to_hours(minutes_val) if minutes_val else 0.0
            labor_val = _safe_float(info.get("labor$"), default=0.0)
            machine_val = _safe_float(info.get("machine$"), default=0.0)
            total_val = _safe_float(info.get("total$"), default=0.0)
            if total_val <= 0.0:
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
            if rate_val <= 0.0:
                rate_val = 0.0

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
                hours_total = _minutes_to_hours(minutes_total)
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
            if rate_val <= 0.0:
                rate_val = 0.0

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

    # Planner totals reconciliation previously compared bucket rows against legacy
    # planner aggregates. Those comparisons produced noisy warnings and are no
    # longer useful now that rendering is sourced directly from bucket data.
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
    bucket_minutes_detail: dict[str, float] = {}
    bucket_minutes_source = getattr(bucket_state, "bucket_minutes_detail", None)
    if isinstance(bucket_minutes_source, _MappingABC):
        for key, minutes in bucket_minutes_source.items():
            minutes_val = _safe_float(minutes, default=0.0)
            key_str = str(key)
            bucket_minutes_detail[key_str] = minutes_val
            canon = _canonical_bucket_key(key_str) or _normalize_bucket_key(key_str)
            if canon:
                bucket_minutes_detail[canon] = minutes_val
                label = _display_bucket_label(canon, label_overrides)
                if label:
                    bucket_minutes_detail[label] = minutes_val
    else:
        for canon_key, metrics in canonical_bucket_summary.items():
            minutes_val = _safe_float(metrics.get("minutes"), default=0.0)
            bucket_minutes_detail[canon_key] = minutes_val
            label = _display_bucket_label(canon_key, label_overrides)
            if label:
                bucket_minutes_detail[label] = minutes_val
    extra_bucket_minutes_detail = breakdown.get("bucket_minutes_detail")
    if isinstance(extra_bucket_minutes_detail, _MappingABC):
        for key, minutes in extra_bucket_minutes_detail.items():
            minutes_val = _safe_float(minutes, default=0.0)
            key_str = str(key)
            bucket_minutes_detail[key_str] = minutes_val
            canon = _canonical_bucket_key(key_str) or _normalize_bucket_key(key_str)
            if canon:
                bucket_minutes_detail[canon] = minutes_val
                label = _display_bucket_label(canon, label_overrides)
                if label:
                    bucket_minutes_detail[label] = minutes_val

    bucket_hour_summary: dict[str, tuple[float, bool]] = {}
    for raw_key, minutes in bucket_minutes_detail.items():
        minutes_val = _safe_float(minutes, default=0.0)
        if minutes_val <= 0.0:
            continue
        canon_key = _canonical_bucket_key(raw_key) or _normalize_bucket_key(raw_key) or str(raw_key)
        label = _display_bucket_label(canon_key, label_overrides)
        bucket_hour_summary[label] = (round(minutes_val / 60.0, 2), True)
    hour_summary_entries.clear()
    hour_summary_entries.update(bucket_hour_summary)
    process_costs_for_render: dict[str, float] = {}
    bucket_costs_source = getattr(bucket_state, "process_costs_for_render", None)
    if isinstance(bucket_costs_source, _MappingABC):
        for key, amount in bucket_costs_source.items():
            process_costs_for_render[str(key)] = _safe_float(amount, default=0.0)
    else:
        for canon_key, metrics in canonical_bucket_summary.items():
            process_costs_for_render[canon_key] = _safe_float(
                metrics.get("total"), default=0.0
            )

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

        proc_total += numeric_amount

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
                float(drilling_meta_guard["total_minutes_billed"])
                float(drilling_bucket_guard["minutes"])
            except (KeyError, TypeError, ValueError):
                pass

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
            if label:
                bucket_minutes_detail[label] = minutes_val
        label = _display_bucket_label(canon_key, label_overrides)
        label_to_canon.setdefault(label, canon_key)
        canon_to_display_label.setdefault(canon_key, label)
        if canon_key not in canonical_bucket_order:
            canonical_bucket_order.insert(0, canon_key)
        hours_val = minutes_val / 60.0 if minutes_val > 0 else 0.0
        rate_val = 0.0
        if hours_val > 0.0 and amount_val > 0.0:
            rate_val = amount_val / hours_val
        elif hours_val > 0.0:
            rate_val = _rate_for_bucket(canon_key, rates or {})
            if rate_val < 0.0:
                rate_val = 0.0
        new_spec = _BucketRowSpec(
            label=label,
            hours=hours_val,
            rate=rate_val,
            total=amount_val,
            labor=amount_val,
            machine=0.0,
            canon_key=canon_key,
            minutes=minutes_val,
        )
        existing_index = None
        for _idx, _spec in enumerate(bucket_row_specs):
            if _spec.canon_key == canon_key:
                existing_index = _idx
                break
        if existing_index is not None:
            bucket_row_specs.pop(existing_index)
        try:
            insert_at = canonical_bucket_order.index(canon_key)
        except ValueError:
            insert_at = 0
        bucket_row_specs.insert(max(insert_at, 0), new_spec)
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

    process_table = _ProcessCostTableRecorder(
        cfg=cfg,
        bucket_state=bucket_state,
        detail_lookup=detail_lookup,
        label_to_canon=label_to_canon,
        canon_to_display_label=canon_to_display_label,
        process_cost_row_details=process_cost_row_details,
        labor_costs_display=labor_costs_display,
        add_labor_cost_line=_add_labor_cost_line,
        process_meta=process_meta,
    )

    # 1a) minutes→$ for planner buckets that have minutes but no dollars
    for k, meta in (canonical_bucket_summary or {}).items():
        minutes = float(meta.get("minutes") or 0.0)
        have_amount = float(process_costs_for_render.get(k) or 0.0)
        if minutes > 0 and have_amount <= 0.0:
            r = _rate_for_bucket(k, rates or {})
            if r > 0:
                process_costs_for_render[k] = round((minutes / 60.0) * r, 2)
                bucket_minutes_detail[k] = minutes
                label = _display_bucket_label(k, label_overrides)
                if label:
                    bucket_minutes_detail[label] = minutes

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
        drill_label = _display_bucket_label("drilling", label_overrides)
        if drill_label:
            bucket_minutes_detail[drill_label] = bill_min
        drill_summary_entry = canonical_bucket_summary.setdefault("drilling", {})
        drill_summary_entry["minutes"] = float(bill_min)
        drill_summary_entry["hours"] = float(bill_min / 60.0)

    process_plan_summary_map: Mapping[str, Any] | None = None
    if isinstance(process_plan_summary_local, _MappingABC):
        process_plan_summary_map = process_plan_summary_local

    ordered_specs: list[_BucketRowSpec] = []
    if bucket_row_specs:
        seen_canon_specs: set[str] = set()
        for canon_key in canonical_bucket_order:
            if not canon_key:
                continue
            for spec in bucket_row_specs:
                if spec.canon_key == canon_key and canon_key not in seen_canon_specs:
                    ordered_specs.append(spec)
                    seen_canon_specs.add(canon_key)
                    break
        for spec in bucket_row_specs:
            canon_key = spec.canon_key
            if canon_key and canon_key in seen_canon_specs:
                continue
            ordered_specs.append(spec)
            if canon_key:
                seen_canon_specs.add(canon_key)
    else:
        table_rows_snapshot = getattr(bucket_state, "table_rows", None)
        if isinstance(table_rows_snapshot, Sequence):
            for entry in table_rows_snapshot:
                if not isinstance(entry, Sequence) or len(entry) < 5:
                    continue
                label_val, hours_val, labor_val, machine_val, total_val = entry[:5]
                try:
                    hours_numeric = float(hours_val or 0.0)
                except Exception:
                    hours_numeric = 0.0
                try:
                    total_numeric = float(total_val or 0.0)
                except Exception:
                    total_numeric = 0.0
                try:
                    labor_numeric = float(labor_val or 0.0)
                except Exception:
                    labor_numeric = 0.0
                try:
                    machine_numeric = float(machine_val or 0.0)
                except Exception:
                    machine_numeric = 0.0
                minutes_numeric = hours_numeric * 60.0
                rate_numeric = 0.0
                canon_key = label_to_canon.get(str(label_val)) or _canonical_bucket_key(
                    label_val
                )
                ordered_specs.append(
                    _BucketRowSpec(
                        label=str(label_val),
                        hours=hours_numeric,
                        rate=rate_numeric,
                        total=total_numeric,
                        labor=labor_numeric,
                        machine=machine_numeric,
                        canon_key=canon_key or str(label_val),
                        minutes=minutes_numeric,
                    )
                )

    rows: tuple[_ProcessRowRecordType, ...] = tuple(getattr(process_table, "rows", ()))

    def _process_row_canon(record: _ProcessRowRecordType) -> str:
        if record.canon_key:
            return record.canon_key
        return _canonical_bucket_key(record.name)

    def _find_process_row(target_canon: str) -> _ProcessRowRecordType | None:
        if not target_canon:
            return None
        for record in rows:
            if _process_row_canon(record) == target_canon:
                return record
        return None

    for spec in ordered_specs:
        canon_key = spec.canon_key
        if canon_key:
            process_costs_for_render[canon_key] = _safe_float(spec.total, default=0.0)
            minutes_val = _safe_float(spec.minutes, default=0.0)
            bucket_minutes_detail[canon_key] = minutes_val
            label = _display_bucket_label(canon_key, label_overrides)
            if label:
                bucket_minutes_detail[label] = minutes_val
        process_table.add_row(spec.label, spec.hours, spec.rate, spec.total)

    bucket_specs_for_render: dict[str, _BucketRowSpec] = {
        spec.label: spec for spec in ordered_specs
    }
    bucket_specs_by_canon: dict[str, _BucketRowSpec] = {}
    for spec in ordered_specs:
        if spec.canon_key and spec.canon_key not in bucket_specs_by_canon:
            bucket_specs_by_canon[spec.canon_key] = spec

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

    bucket_render_order = [
        "programming",
        "milling",
        "drilling",
        "tapping",
        "counterbore",
        "spot_drill",
        "jig_grind",
        "inspection",
    ]
    printed_bucket_labels: set[str] = set()
    if bucket_specs_for_render:
        process_table.had_rows = True
        for canon_key in bucket_render_order:
            spec = bucket_specs_by_canon.get(canon_key)
            if spec is None:
                continue
            if spec.label in printed_bucket_labels:
                continue
            _add_labor_cost_line(spec.label, spec.total, process_key=canon_key)
            printed_bucket_labels.add(spec.label)
        for spec in ordered_specs:
            if spec.label in printed_bucket_labels:
                continue
            _add_labor_cost_line(spec.label, spec.total, process_key=spec.canon_key)
            printed_bucket_labels.add(spec.label)
        for label in printed_bucket_labels:
            labor_cost_totals.pop(label, None)

    emit_flags: dict[str, bool] = {}
    if printed_bucket_labels:
        emitted_canon_keys: set[str] = set()
        for label in printed_bucket_labels:
            canon = label_to_canon.get(label)
            if not canon:
                canon = _canonical_bucket_key(label)
            if canon:
                normalized = _canonical_bucket_key(canon) or _normalize_bucket_key(canon) or str(canon)
                emitted_canon_keys.add(str(normalized))

        def _bucket_emitted(key: str) -> bool:
            canon_key = _canonical_bucket_key(key) or _normalize_bucket_key(key)
            if canon_key:
                return str(canon_key) in emitted_canon_keys
            return str(key) in emitted_canon_keys

        emit_flags = {bucket: _bucket_emitted(bucket) for bucket in ("milling", "drilling", "tapping", "inspection")}

    if emit_flags and isinstance(process_plan_summary_local, (_MappingABC, dict)):
        _zero_planner_if_row_suppressed(process_plan_summary_local, emit_flags)

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

    hour_summary_entries.clear()
    if canonical_bucket_order and canonical_bucket_summary:
        bucket_hours: dict[str, float] = {}
        for canon_key in canonical_bucket_order:
            if not canon_key:
                continue
            metrics = canonical_bucket_summary.get(canon_key) or {}
            minutes_val = _safe_float(metrics.get("minutes"), default=0.0)
            if minutes_val <= 0.0:
                continue
            label = _display_bucket_label(canon_key, label_overrides)
            bucket_hours[label] = bucket_hours.get(label, 0.0) + (
                minutes_val / 60.0
            )

        for label, hours_val in sorted(bucket_hours.items()):
            if not ((hours_val > 0.0) or show_zeros):
                continue
            hour_summary_entries[label] = (round(hours_val, 2), True)

    if misc_total > 0:
        process_table.had_rows = True

    bucket_view_obj = locals().get("bucket_view_obj") if "bucket_view_obj" in locals() else None
    bucket_view_for_render: Mapping[str, Any] | None
    if isinstance(bucket_view_obj, (_MappingABC, dict)):
        bucket_view_for_render = typing.cast(Mapping[str, Any], bucket_view_obj)
    elif isinstance(bucket_view_struct, (_MappingABC, dict)):
        bucket_view_for_render = typing.cast(Mapping[str, Any], bucket_view_struct)
    else:
        bucket_view_for_render = None

    bucket_seed_target: MutableMapping[str, Any] | Mapping[str, Any] | None = None
    for candidate_name in ("bucket_view_obj", "bucket_view_struct"):
        candidate = locals().get(candidate_name)
        if isinstance(candidate, (_MutableMappingABC, dict)):
            bucket_seed_target = typing.cast(MutableMapping[str, Any], candidate)
            break

    process_plan_summary_candidate: Mapping[str, Any] | None = None
    for candidate_name in ("process_plan_summary", "process_plan_summary_local"):
        candidate = locals().get(candidate_name)
        if isinstance(candidate, _MappingABC):
            process_plan_summary_candidate = typing.cast(Mapping[str, Any], candidate)
            break

    extra_map_candidate = locals().get("extra_map") if "extra_map" in locals() else None
    if not isinstance(extra_map_candidate, _MappingABC):
        extra_map_candidate = None
    else:
        extra_map_candidate = typing.cast(Mapping[str, Any], extra_map_candidate)

    rates_candidate = locals().get("rates") if "rates" in locals() else None

    drill_minutes_total = _pick_drill_minutes(
        process_plan_summary_candidate,
        typing.cast(Mapping[str, Any] | None, extra_map_candidate),
    )
    drill_mrate = (
        _lookup_bucket_rate("drilling", rates_candidate)
        or _lookup_bucket_rate("machine", rates_candidate)
        or 45.0
    )
    drill_lrate = (
        _lookup_bucket_rate("drilling_labor", rates_candidate)
        or _lookup_bucket_rate("labor", rates_candidate)
        or 45.0
    )

    if bucket_seed_target is not None:
        _purge_legacy_drill_sync(bucket_seed_target)
        _set_bucket_minutes_cost(
            bucket_seed_target,
            "drilling",
            drill_minutes_total,
            drill_mrate,
            drill_lrate,
        )
        try:
            drilling_bucket_snapshot = (
                (bucket_seed_target.get("buckets") or {}).get("drilling")
                if isinstance(bucket_seed_target, (_MappingABC, dict))
                else None
            )
        except Exception:
            drilling_bucket_snapshot = None
        logger.info(
            "[bucket] drilling_minutes=%s drilling_bucket=%s",
            drill_minutes_total,
            drilling_bucket_snapshot,
        )

    if isinstance(bucket_view_obj, (_MutableMappingABC, dict)):
        _normalize_buckets(typing.cast(MutableMapping[str, Any], bucket_view_obj))
    elif isinstance(bucket_view_struct, (_MutableMappingABC, dict)):
        _normalize_buckets(typing.cast(MutableMapping[str, Any], bucket_view_struct))

    process_section_start = len(lines)

    bucket_entries_for_totals: Mapping[str, Any] | None = None
    if isinstance(bucket_view_for_render, (_MappingABC, dict)):
        try:
            bucket_entries_candidate = bucket_view_for_render.get("buckets")
        except Exception:
            bucket_entries_candidate = None
        if isinstance(bucket_entries_candidate, dict):
            bucket_entries_for_totals = bucket_entries_candidate
        elif isinstance(bucket_entries_candidate, _MappingABC):
            try:
                bucket_entries_for_totals = dict(bucket_entries_candidate)
            except Exception:
                bucket_entries_for_totals = {}
    if bucket_entries_for_totals is None:
        bucket_entries_for_totals = {}

    bucket_entries_for_totals_map = {}

    preferred_bucket_order = [
        "programming",
        "milling",
        "drilling",
        "tapping",
        "counterbore",
        "spot_drill",
        "jig_grind",
        "inspection",
    ]

    proc_total_rendered = 0.0
    hrs_total_rendered = 0.0
    seen_bucket_keys: set[str] = set()
    for bucket_key in list(preferred_bucket_order) + [
        key for key in bucket_entries_for_totals if key not in preferred_bucket_order
    ]:
        entry = bucket_entries_for_totals.get(bucket_key)
        if not isinstance(entry, _MappingABC) or bucket_key in seen_bucket_keys:
            continue
        seen_bucket_keys.add(bucket_key)

        canon_key = (
            _canonical_bucket_key(bucket_key)
            or _normalize_bucket_key(bucket_key)
            or str(bucket_key)
        )
        normalized_key = _normalize_bucket_key(bucket_key)
        display_label = _display_bucket_label(canon_key, label_overrides)

        lookup_keys = {
            str(bucket_key),
            canon_key,
            normalized_key,
            display_label,
        }
        for lookup_key in lookup_keys:
            if lookup_key:
                bucket_entries_for_totals_map[str(lookup_key)] = entry

    process_rows_total, process_rows_minutes, process_rows_rendered = _render_process_and_hours_from_buckets(
        lines,
        bucket_view_for_render,
    )
    proc_total_rendered = process_rows_total
    hrs_total_rendered = process_rows_minutes / 60.0 if process_rows_minutes > 0 else 0.0
    proc_machine = sum(row[2] for row in process_rows_rendered)
    proc_labor = sum(row[3] for row in process_rows_rendered)
    machine_sum = proc_machine
    labor_sum = proc_labor
    if process_rows_rendered:
        top_rows = sorted(
            process_rows_rendered,
            key=lambda r: r[4],
            reverse=True,
        )[:3]
        top_lines = [
            f"{name} ${total:,.2f}" for (name, _, _, _, total) in top_rows
        ]
        for line in top_lines:
            if line not in why_lines:
                why_lines.append(line)
        summary_bits: list[str] = [
            f"Machine {_m(machine_sum)}",
            f"Labor {_m(labor_sum)}",
        ]
        top_summary = [
            f"{name} {_m(total)}" for (name, _, _, _, total) in top_rows if total > 0
        ]
        if top_summary:
            summary_bits.append("largest bucket(s): " + ", ".join(top_summary))
        bucket_why_summary_line = "Process buckets — " + "; ".join(summary_bits)
    if proc_total_rendered or hrs_total_rendered:
        for offset, text in enumerate(lines[process_section_start:]):
            stripped = str(text or "").strip()
            if stripped.lower().startswith("total") and "$" in stripped:
                process_total_row_index = process_section_start + offset
                break
    proc_total = proc_total_rendered

    # ---- Pass-Through & Direct (auto include non-zeros; sorted desc) --------
    _push(lines, "Pass-Through & Direct Costs")
    pass_through_header_index = len(lines) - 1
    _push(lines, divider)
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
    material_display_amount_candidate: float | None
    if material_component_total is not None:
        material_display_amount_candidate = float(material_component_total)
        material_direct_contribution = round(float(material_component_total), 2)
    else:
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
    if material_component_net is not None:
        material_net_cost = float(material_component_net)
    else:
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
        _push(lines, "")
        _push(lines, "Cost Breakdown")
        _push(lines, divider)
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

    total_process_cost_value = round(float(proc_total or 0.0), 2)
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
    if 0 <= total_process_cost_row_index < len(lines):
        replace_line(
            total_process_cost_row_index,
            _format_row(
                total_process_cost_label,
                total_process_cost_value,
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
    if 0 <= total_process_cost_row_index < len(lines):
        replace_line(
            total_process_cost_row_index,
            _format_row(total_process_cost_label, total_process_cost_value),
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
    # -- Ensure extra ops cards are appended to the SAME list that gets printed --
    geo_map_candidate = ((result or {}).get("geo") if isinstance(result, _MappingABC) else None) \
                        or ((breakdown or {}).get("geo") if isinstance(breakdown, _MappingABC) else None) \
                        or {}
    chart_lines_source: Any
    if isinstance(geo_map_candidate, (_MappingABC, dict)):
        geo_map = geo_map_candidate
        try:
            chart_lines_source = geo_map.get("chart_lines")  # type: ignore[index]
        except Exception:
            chart_lines_source = None
    else:
        geo_map = {}
        if isinstance(geo_map_candidate, Iterable) and not isinstance(
            geo_map_candidate, (str, bytes, bytearray)
        ):
            chart_lines_source = geo_map_candidate
        else:
            chart_lines_source = None
    chart_lines_all = list(chart_lines_source or [])
    if not chart_lines_all:
        try:
            chart_lines_all = _collect_chart_lines_context(ctx, geo_map, ctx_a, ctx_b) or []
        except Exception:
            chart_lines_all = []

    # CLEAN then JOIN
    cleaned = [_clean_mtext(x) for x in chart_lines_all]
    joined_lines = _join_wrapped_chart_lines(cleaned)
    counterdrill_qty = _count_counterdrill(joined_lines)

    try:
        ops_claims_preview = _parse_ops_and_claims(joined_lines)
    except Exception:
        ops_claims_preview = {}
    if not isinstance(ops_claims_preview, dict):
        ops_claims_preview = {}
    try:
        preview_counterdrill = int(round(float(ops_claims_preview.get("counterdrill", 0))))
    except Exception:
        preview_counterdrill = 0
    if counterdrill_qty > preview_counterdrill:
        ops_claims_preview["counterdrill"] = counterdrill_qty
    else:
        ops_claims_preview["counterdrill"] = preview_counterdrill
    jig_qty_fallback = _count_jig(joined_lines)
    if jig_qty_fallback and int(ops_claims_preview.get("jig", 0) or 0) <= 0:
        ops_claims_preview["jig"] = jig_qty_fallback
    _push(
        lines,
        "[DEBUG] at_print_ops cb={cb} tap={tap} npt={npt} spot={spot} counterdrill={counterdrill} jig={jig}".format(
            cb=int(ops_claims_preview.get("cb_total", 0)),
            tap=int(ops_claims_preview.get("tap", 0)),
            npt=int(ops_claims_preview.get("npt", 0)),
            spot=int(ops_claims_preview.get("spot", 0)),
            counterdrill=int(ops_claims_preview.get("counterdrill", 0)),
            jig=int(ops_claims_preview.get("jig", 0)),
        ),
    )

    # Use any rows already built earlier (if present)
    try:
        ops_rows_now = (((geo_map or {}).get("ops_summary") or {}).get("rows") or [])
    except Exception:
        ops_rows_now = []
    if not isinstance(ops_rows_now, list):
        ops_rows_now = []

    ops_claims: dict[str, int] = {}
    try:
        if any(
            int(ops_claims_preview.get(key, 0)) > 0
            for key in ("cb_total", "tap", "npt", "spot", "counterdrill", "jig")
        ):
            ops_claims = {
                "cb_total": int(ops_claims_preview.get("cb_total", 0)),
                "cb_front": int(ops_claims_preview.get("cb_front", 0)),
                "cb_back": int(ops_claims_preview.get("cb_back", 0)),
                "tap": int(ops_claims_preview.get("tap", 0)),
                "npt": int(ops_claims_preview.get("npt", 0)),
                "spot": int(ops_claims_preview.get("spot", 0)),
                "counterdrill": int(ops_claims_preview.get("counterdrill", 0)),
                "jig": int(ops_claims_preview.get("jig", 0)),
            }
    except Exception:
        ops_claims = {}

    def _normalize_ops_claims_map(candidate: Any) -> dict[str, int]:
        if not isinstance(candidate, (_MappingABC, dict)):
            return {}
        normalized: dict[str, int] = {}
        for key, value in candidate.items():
            try:
                normalized[str(key)] = int(round(float(value)))
            except Exception:
                continue
        return normalized

    def _stash_ops_claims(claims: dict[str, int]) -> None:
        if not claims:
            return
        if isinstance(breakdown_mutable, _MutableMappingABC):
            try:
                breakdown_mutable["_ops_claims"] = dict(claims)
            except Exception:
                pass

    try:
        stashed_claims = (
            breakdown_mutable.get("_ops_claims")
            if isinstance(breakdown_mutable, _MappingABC)
            else None
        )
    except Exception:
        stashed_claims = None
    if stashed_claims:
        ops_claims = _normalize_ops_claims_map(stashed_claims)

    def _extract_ops_claims(source: Any) -> dict[str, int]:
        if not isinstance(source, (_MappingABC, dict)):
            return {}
        try:
            claims_candidate = source.get("ops_claims")  # type: ignore[index]
            if not isinstance(claims_candidate, (_MappingABC, dict)):
                claims_candidate = source.get("_ops_claims")  # type: ignore[index]
        except Exception:
            return {}
        if not isinstance(claims_candidate, (_MappingABC, dict)):
            return {}
        extracted: dict[str, int] = {}
        for key, value in claims_candidate.items():
            try:
                extracted[str(key)] = int(round(float(value)))
            except Exception:
                continue
        return extracted

    for candidate_source in (geo_map, breakdown, result):
        extracted_claims = _extract_ops_claims(candidate_source)
        if extracted_claims:
            ops_claims = extracted_claims
            _stash_ops_claims(dict(ops_claims))
            break

    if not ops_claims and (joined_lines or ops_rows_now):
        try:
            computed_claims = _preseed_ops_from_chart_lines(
                chart_lines=joined_lines,
                rows=ops_rows_now,
                breakdown_mutable={},
                rates=rates,
            )
        except Exception:
            computed_claims = {}
        if isinstance(computed_claims, dict):
            extracted: dict[str, int] = {}
            for key, value in computed_claims.items():
                try:
                    extracted[str(key)] = int(round(float(value)))
                except Exception:
                    continue
            if extracted:
                ops_claims = extracted
                _stash_ops_claims(dict(ops_claims))

    current_counterdrill = 0
    try:
        current_counterdrill = int(round(float(ops_claims.get("counterdrill", 0))))
    except Exception:
        current_counterdrill = 0
    counterdrill_claim = max(counterdrill_qty, current_counterdrill)
    if counterdrill_claim > 0:
        ops_claims["counterdrill"] = counterdrill_claim
    elif "counterdrill" in ops_claims:
        ops_claims["counterdrill"] = 0

    for key in ("cb_total", "cb_front", "cb_back", "tap", "npt", "spot", "counterdrill", "jig"):
        ops_claims.setdefault(key, 0)

    def _extract_ops_hint(source: Any) -> dict[str, int]:
        if not isinstance(source, (_MappingABC, dict)):
            return {}
        hint_candidate: Any
        try:
            hint_candidate = source.get("ops_hint")  # type: ignore[index]
        except Exception:
            hint_candidate = None
        if not isinstance(hint_candidate, (_MappingABC, dict)):
            return {}
        extracted_hint: dict[str, int] = {}
        for key, value in hint_candidate.items():
            try:
                extracted_hint[str(key)] = int(round(float(value)))
            except Exception:
                continue
        return extracted_hint

    ops_hint: dict[str, int] = {}
    for candidate_source in (geo_map, breakdown_mutable, breakdown, result):
        extracted_hint = _extract_ops_hint(candidate_source)
        if extracted_hint:
            ops_hint = extracted_hint
            break

    # Append extra MATERIAL REMOVAL cards (Counterbore / Spot / Jig) from JOINED lines
    _appended_at_print = _append_counterbore_spot_jig_cards(
        lines_out=removal_card_lines,     # <— append directly to the printed list
        chart_lines=joined_lines,   # <- use joined lines
        rows=ops_rows_now,
        breakdown_mutable=breakdown_mutable,
        rates=rates,
    )
    _push(lines, f"[DEBUG] extra_ops_appended_at_print={_appended_at_print}")

    def _card_heading_exists(heading: str) -> bool:
        heading_norm = heading.strip().upper()
        for entry in removal_card_lines:
            if isinstance(entry, str) and entry.strip().upper() == heading_norm:
                return True
        return False

    actions_summary_ready = False

    def _collect_removal_summary_lines() -> list[str]:
        nonlocal actions_summary_ready
        try:
            ebo = breakdown_mutable.setdefault("extra_bucket_ops", {})
            if ops_claims.get("cb_front", 0) > 0:
                cb_front_qty = int(ops_claims.get("cb_front", 0) or 0)
                ebo.setdefault("counterbore", []).append(
                    {
                        "name": "Counterbore",
                        "qty": cb_front_qty,
                        "side": "front",
                    }
                )
            if ops_claims.get("cb_back", 0) > 0:
                cb_back_qty = int(ops_claims.get("cb_back", 0) or 0)
                ebo.setdefault("counterbore", []).append(
                    {
                        "name": "Counterbore",
                        "qty": cb_back_qty,
                        "side": "back",
                    }
                )
            if ops_claims.get("tap", 0) > 0:
                tap_qty = int(ops_claims.get("tap", 0) or 0)
                ebo.setdefault("tap", []).append(
                    {
                        "name": "Tap",
                        "qty": tap_qty,
                        "side": "front",
                    }
                )
            if ops_claims.get("npt", 0) > 0:
                npt_qty = int(ops_claims.get("npt", 0) or 0)
                ebo.setdefault("tap", []).append(
                    {
                        "name": "NPT tap",
                        "qty": npt_qty,
                        "side": "front",
                    }
                )
            if ops_claims.get("spot", 0) > 0:
                spot_qty = int(ops_claims.get("spot", 0) or 0)
                ebo.setdefault("spot", []).append(
                    {
                        "name": "Spot drill",
                        "qty": spot_qty,
                        "side": "front",
                    }
                )
            if ops_claims.get("counterdrill", 0) > 0:
                counterdrill_qty_local = int(ops_claims.get("counterdrill", 0) or 0)
                ebo.setdefault("counterdrill", []).append(
                    {
                        "name": "Counterdrill",
                        "qty": counterdrill_qty_local,
                        "side": "front",
                    }
                )
            if ops_claims.get("jig", 0) > 0:
                jig_qty = int(ops_claims.get("jig", 0) or 0)
                ebo.setdefault("jig-grind", []).append(
                    {
                        "name": "Jig-grind",
                        "qty": jig_qty,
                        "side": None,
                    }
                )
        except Exception:
            pass

        try:
            if "counterbore" in ops_hint:
                cb_need = int(ops_hint.get("counterbore", 0) or 0)
                cb_have = int(ops_claims.get("cb_total", 0) or 0)
                if cb_need > cb_have:
                    ops_claims["cb_total"] = cb_need
            if "spot" in ops_hint:
                ops_claims["spot"] = int(ops_hint.get("spot", 0) or 0)
            if "counterdrill" in ops_hint:
                hint_counterdrill = int(ops_hint.get("counterdrill", 0) or 0)
                existing_counterdrill = int(ops_claims.get("counterdrill", 0) or 0)
                ops_claims["counterdrill"] = max(existing_counterdrill, hint_counterdrill)
            if "jig" in ops_hint:
                ops_claims["jig"] = int(ops_hint.get("jig", 0) or 0)
            if "tap" in ops_hint:
                ops_claims["tap"] = int(ops_claims.get("tap", 0) or 0) + int(
                    ops_hint.get("tap", 0) or 0
                )
        except Exception:
            pass

        if (ops_claims.get("spot") or 0) > 0:
            spot_qty = int(ops_claims.get("spot", 0) or 0)
            spot_heading = "MATERIAL REMOVAL – SPOT (CENTER DRILL)"
            if not _card_heading_exists(spot_heading):
                removal_card_lines.extend([
                    spot_heading,
                    "=" * 64,
                    "TIME PER HOLE – SPOT GROUPS",
                    "-" * 66,
                    f"Spot drill × {spot_qty} | t/hole 0.05 min | group {spot_qty}×0.05 = {spot_qty * 0.05:.2f} min",
                    "",
                ])

        if (ops_claims.get("counterdrill") or 0) > 0:
            counterdrill_qty_local = int(ops_claims.get("counterdrill", 0) or 0)
            counterdrill_heading = "MATERIAL REMOVAL – COUNTERDRILL"
            if not _card_heading_exists(counterdrill_heading):
                removal_card_lines.extend(
                    [
                        counterdrill_heading,
                        "=" * 64,
                        "TIME PER HOLE – COUNTERDRILL GROUPS",
                        "-" * 66,
                        f"Counterdrill × {counterdrill_qty_local} | group {counterdrill_qty_local}",
                        "",
                    ]
                )

        if (ops_claims.get("jig") or 0) > 0:
            jig_qty = int(ops_claims.get("jig", 0) or 0)
            jig_heading = "MATERIAL REMOVAL – JIG GRIND"
            if not _card_heading_exists(jig_heading):
                per = float(globals().get("JIG_GRIND_MIN_PER_FEATURE") or 0.75)
                removal_card_lines.extend([
                    jig_heading,
                    "=" * 64,
                    "TIME PER FEATURE",
                    "-" * 66,
                    f"Jig grind × {jig_qty} | t/feat {per:.2f} min | group {jig_qty}×{per:.2f} = {jig_qty * per:.2f} min",
                    "",
                ])

        summary_lines = [
            str(line) for line in removal_card_lines if isinstance(line, str)
        ]
        if removal_summary_extra_lines:
            for entry in removal_summary_extra_lines:
                entry_text = str(entry)
                if entry_text not in summary_lines:
                    summary_lines.append(entry_text)

        append_lines(removal_card_lines)

        try:
            _normalize_buckets(breakdown.get("bucket_view"))
        except Exception:
            pass

        removal_drill_heading = "MATERIAL REMOVAL – DRILLING"
        removal_card_has_drill = False
        for line_text in removal_card_lines:
            if isinstance(line_text, str) and line_text.strip().upper().startswith(
                removal_drill_heading
            ):
                removal_card_has_drill = True
                break

        if removal_card_has_drill:
            milling_bucket_obj = None
            bucket_view_snapshot = (
                breakdown.get("bucket_view") if isinstance(breakdown, _MappingABC) else None
            )
            if isinstance(bucket_view_snapshot, (_MappingABC, dict)):
                milling_bucket_obj = _extract_milling_bucket(bucket_view_snapshot)
            _render_milling_removal_card(append_line, lines, milling_bucket_obj)

        if not removal_card_has_drill:
            drill_groups_render: list[dict[str, float]] = []

            def _extract_groups(rows: Sequence[Any] | None) -> bool:
                nonlocal drill_groups_render
                if not isinstance(rows, Sequence) or not rows:
                    return False
                extracted: list[dict[str, float]] = []
                for entry in rows:
                    if not isinstance(entry, _MappingABC):
                        continue
                    qty_val = int(_coerce_float_or_none(entry.get("qty")) or 0)
                    if qty_val <= 0:
                        continue
                    dia_val = _coerce_float_or_none(entry.get("diameter_in"))
                    if dia_val is None or dia_val <= 0:
                        continue
                    depth_val = _coerce_float_or_none(entry.get("depth_in")) or 0.0
                    sfm_val = _coerce_float_or_none(entry.get("sfm")) or 0.0
                    ipr_val = _coerce_float_or_none(entry.get("ipr")) or 0.0
                    per_hole = (
                        _coerce_float_or_none(entry.get("t_hole_min"))
                        or _coerce_float_or_none(entry.get("t_per_hole_min"))
                        or _coerce_float_or_none(entry.get("minutes_per_hole"))
                    )
                    group_total = (
                        _coerce_float_or_none(entry.get("t_group_min"))
                        or _coerce_float_or_none(entry.get("group_minutes"))
                    )
                    if per_hole is None and group_total is not None and qty_val > 0:
                        per_hole = float(group_total) / float(qty_val)
                    if per_hole is None:
                        per_hole = 0.0
                    if group_total is None:
                        group_total = float(qty_val) * float(per_hole)
                    extracted.append(
                        {
                            "diameter_in": float(dia_val),
                            "qty": float(qty_val),
                            "depth_in": float(depth_val),
                            "sfm": float(sfm_val),
                            "ipr": float(ipr_val),
                            "t_hole_min": float(per_hole),
                            "t_group_min": float(group_total),
                        }
                    )
                if extracted:
                    drill_groups_render = extracted
                    return True
                return False

            dtph_rows_source = None
            if isinstance(drilling_time_per_hole_data, _MappingABC):
                dtph_rows_source = drilling_time_per_hole_data.get("rows")
            if not _extract_groups(dtph_rows_source):
                detail_groups = None
                if isinstance(drilling_card_detail, _MappingABC):
                    detail_groups = drilling_card_detail.get("drill_groups")
                if not _extract_groups(detail_groups):
                    meta_groups = None
                    if isinstance(drilling_meta_map, _MappingABC):
                        meta_groups = drilling_meta_map.get("drill_groups")
                        if not meta_groups:
                            meta_groups = drilling_meta_map.get("bins_list")
                    _extract_groups(meta_groups)

            if drill_groups_render:
                append_line(removal_drill_heading)
                append_line("-" * 66)
                for group in drill_groups_render:
                    dia = float(group.get("diameter_in", 0.0))
                    qty = int(round(float(group.get("qty", 0.0))))
                    depth = float(group.get("depth_in", 0.0))
                    sfm = float(group.get("sfm", 0.0))
                    ipr = float(group.get("ipr", 0.0))
                    t_hole = float(group.get("t_hole_min", 0.0))
                    t_group = float(group.get("t_group_min", qty * t_hole))
                    append_line(
                        f'Dia {dia:.3f}" × {qty}  | depth {depth:.3f}" | '
                        f"{int(round(sfm))} sfm | {ipr:.4f} ipr | "
                        f"t/hole {t_hole:.2f} min | group {qty}×{t_hole:.2f} = {t_group:.2f} min"
                    )
                subtotal_minutes_raw = 0.0
                for group in drill_groups_render:
                    try:
                        subtotal_minutes_raw += float(group.get("t_group_min", 0.0))
                    except Exception:
                        continue

                subtotal_minutes_val = None
                tool_components_source: Sequence[Any] | None = None
                toolchange_total_min: float | None = None
                total_minutes_val: float | None = None

                def _resolve_mapping(candidate: Any) -> Mapping[str, Any] | None:
                    if isinstance(candidate, _MappingABC):
                        return typing.cast(Mapping[str, Any], candidate)
                    if isinstance(candidate, dict):
                        return candidate
                    return None

                for source_candidate in (
                    drilling_time_per_hole_data,
                    drilling_card_detail,
                    drilling_meta_map,
                ):
                    source_map = _resolve_mapping(source_candidate)
                    if not source_map:
                        continue
                    if subtotal_minutes_val is None:
                        subtotal_minutes_val = _coerce_float_or_none(
                            source_map.get("subtotal_minutes")
                        )
                    if tool_components_source is None:
                        components = source_map.get("tool_components")
                        if isinstance(components, Sequence) and components:
                            tool_components_source = components
                    if toolchange_total_min is None:
                        for key in ("toolchange_minutes", "toolchange_total"):
                            candidate_val = _coerce_float_or_none(source_map.get(key))
                            if candidate_val is not None:
                                toolchange_total_min = float(candidate_val)
                                break
                    if total_minutes_val is None:
                        total_candidate = (
                            _coerce_float_or_none(
                                source_map.get("total_minutes_with_toolchange")
                            )
                            or _coerce_float_or_none(source_map.get("total_minutes"))
                        )
                        if total_candidate is not None:
                            total_minutes_val = float(total_candidate)

                component_labels: list[str] = []
                component_minutes = 0.0
                if isinstance(tool_components_source, Sequence):
                    for comp in tool_components_source:
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

                if subtotal_minutes_val is None:
                    subtotal_minutes_val = subtotal_minutes_raw
                drill_minutes_subtotal_raw = _sanitize_drill_removal_minutes(
                    subtotal_minutes_val or 0.0
                )
                drill_minutes_subtotal = round(drill_minutes_subtotal_raw, 2)

                if toolchange_total_min is None:
                    toolchange_total_min = component_minutes
                try:
                    toolchange_total_min = float(toolchange_total_min or 0.0)
                except Exception:
                    toolchange_total_min = 0.0
                if not math.isfinite(toolchange_total_min):
                    toolchange_total_min = component_minutes
                if toolchange_total_min < 0.0:
                    toolchange_total_min = 0.0

                if total_minutes_val is None:
                    total_minutes_val = drill_minutes_subtotal_raw + toolchange_total_min
                total_minutes_val = _sanitize_drill_removal_minutes(total_minutes_val or 0.0)
                drill_minutes_total = round(total_minutes_val, 2)

                if component_labels:
                    label_text = " + ".join(component_labels)
                    append_line(
                        f"Toolchange adders: {label_text} = {toolchange_total_min:.2f} min"
                    )
                elif toolchange_total_min > 0.0:
                    append_line(
                        f"Toolchange adders: Toolchange {toolchange_total_min:.2f} min = {toolchange_total_min:.2f} min"
                    )
                else:
                    append_line("Toolchange adders: -")

                append_line("-" * 66)
                append_line(
                    f"Subtotal (per-hole × qty) . {drill_minutes_subtotal:.2f} min  ("
                    f"{fmt_hours(minutes_to_hours(drill_minutes_subtotal))})"
                )
                append_line(
                    "TOTAL DRILLING (with toolchange) . "
                    f"{drill_minutes_total:.2f} min  ("
                    f"{fmt_hours(minutes_to_hours(drill_minutes_total))})"
                )
                append_line("")

                milling_bucket_obj = None
                bucket_view_snapshot = (
                    breakdown.get("bucket_view") if isinstance(breakdown, _MappingABC) else None
                )
                if isinstance(bucket_view_snapshot, (_MappingABC, dict)):
                    milling_bucket_obj = _extract_milling_bucket(bucket_view_snapshot)
                _render_milling_removal_card(append_line, lines, milling_bucket_obj)

        # ===== MATERIAL REMOVAL: HOLE-TABLE CARDS =================================
        # use module-level 're'

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

    ctx_a = locals().get("breakdown")
    ctx_b = locals().get("result")
    ctx_c = locals().get("quote")
    ctx = _first_dict(ctx_a, ctx_b, ctx_c)

    geo_map = _get_geo_map(ctx, locals().get("geo"), ctx_a, ctx_b)
    material_group = _get_material_group(ctx, ctx_a, ctx_b)

    ops_summary_map = None
    ops_rows: list[Any] = []

    try:
        ops_summary_payload = (
            geo_map.get("ops_summary") if isinstance(geo_map, _MappingABC) else None
        )
        ops_summary_map = (
            ops_summary_payload
            if isinstance(ops_summary_payload, (_MutableMappingABC, dict))
            else None
        )
        ops_rows = (
            ((ops_summary_map or {}).get("rows") or [])
            if isinstance(ops_summary_map, _MappingABC)
            else []
        )
        _push(lines, f"[DEBUG] ops_rows_pre={len(ops_rows)}")

    except Exception as e:
        _push(lines, f"[DEBUG] material_removal_emit_skipped={e.__class__.__name__}: {e}")
    else:
        try:
            if not ops_rows:
                chart_lines_all = _collect_chart_lines_context(ctx, geo_map, ctx_a, ctx_b)
                cleaned_chart_lines: list[str] = []
                for raw_line in chart_lines_all or []:
                    cleaned_line = _clean_mtext(str(raw_line or ""))
                    if cleaned_line:
                        cleaned_chart_lines.append(cleaned_line)

                joined_early = _join_wrapped_chart_lines(cleaned_chart_lines)
                built = _build_ops_rows_from_lines_fallback(joined_early)
                _push(
                    lines,
                    f"[DEBUG] chart_lines_found={len(joined_early)} built_rows={len(built)}",
                )
                if built:
                    plate_thickness = _resolve_part_thickness_in(
                        geo_map,
                        ctx.get("geo") if isinstance(ctx, _MappingABC) else None,
                        ctx_a.get("geo") if isinstance(ctx_a, _MappingABC) else None,
                    )
                    _finalize_tapping_rows(built, thickness_in=plate_thickness)

                    # Persist ops-summary rows for other consumers
                    ops_summary_map = geo_map.setdefault("ops_summary", {})
                    ops_summary_map["rows"] = built
                    ops_rows = built

                    # Persist chart lines for other consumers
                    geo_map.setdefault("chart_lines", list(chart_lines_all))

                    ops_claims = _parse_ops_and_claims(joined_early)
                    breakdown_mutable["_ops_claims"] = ops_claims
                    _push(
                        lines,
                        f"[DEBUG] preseed_ops cb={ops_claims['cb_total']} tap={ops_claims['tap']} "
                        f"npt={ops_claims['npt']} spot={ops_claims['spot']} jig={ops_claims['jig']}",
                    )

                    # Publish structured ops for planner_ops_summary
                    try:
                        ebo = breakdown_mutable.setdefault("extra_bucket_ops", {})
                        if ops_claims["cb_front"] > 0:
                            ebo.setdefault("counterbore", []).append(
                                {"name": "Counterbore", "qty": int(ops_claims["cb_front"]), "side": "front"}
                            )
                        if ops_claims["cb_back"] > 0:
                            ebo.setdefault("counterbore", []).append(
                                {"name": "Counterbore", "qty": int(ops_claims["cb_back"]), "side": "back"}
                            )
                        if ops_claims["tap"] > 0:
                            ebo.setdefault("tap", []).append(
                                {"name": "Tap", "qty": int(ops_claims["tap"]), "side": "front"}
                            )
                        if ops_claims["npt"] > 0:
                            ebo.setdefault("tap", []).append(
                                {"name": "NPT tap", "qty": int(ops_claims["npt"]), "side": "front"}
                            )
                        if ops_claims["spot"] > 0:
                            ebo.setdefault("spot", []).append(
                                {"name": "Spot drill", "qty": int(ops_claims["spot"]), "side": "front"}
                            )
                        if ops_claims["jig"] > 0:
                            ebo.setdefault("jig-grind", []).append(
                                {"name": "Jig-grind", "qty": int(ops_claims["jig"]), "side": None}
                            )
                    except Exception:
                        pass

                counts_by_diam_raw_obj = locals().get("counts_by_diam_raw")
                counts_by_diam_obj = locals().get("counts_by_diam")
                drilling_meta_snapshot = locals().get("drilling_meta_container")
                if not isinstance(drilling_meta_snapshot, (_MappingABC, dict)):
                    drilling_meta_snapshot = None
                    if isinstance(breakdown_mutable, (_MappingABC, dict)):
                        candidate_meta = breakdown_mutable.get("drilling_meta")
                        if isinstance(candidate_meta, (_MappingABC, dict)):
                            drilling_meta_snapshot = candidate_meta
                if not isinstance(counts_by_diam_raw_obj, (_MappingABC, dict, Sequence)):
                    counts_by_diam_raw_obj = None
                if not isinstance(counts_by_diam_obj, (_MappingABC, dict, Sequence)):
                    counts_by_diam_obj = None
                if counts_by_diam_raw_obj is None and isinstance(drilling_meta_snapshot, _MappingABC):
                    candidate = drilling_meta_snapshot.get("counts_by_diam_raw")
                    if isinstance(candidate, (_MappingABC, dict, Sequence)):
                        counts_by_diam_raw_obj = candidate
                if counts_by_diam_obj is None and isinstance(drilling_meta_snapshot, _MappingABC):
                    candidate = drilling_meta_snapshot.get("counts_by_diam")
                    if isinstance(candidate, (_MappingABC, dict, Sequence)):
                        counts_by_diam_obj = candidate

                drill_bins_raw_total = _sum_count_values(counts_by_diam_raw_obj)
                drill_bins_adj_total = _sum_count_values(counts_by_diam_obj)
                _push(lines, f"[DEBUG] chart_lines_found={len(chart_lines_all)}")
                _push(
                    lines,
                    f"[DEBUG] at_print_ops cb={ops_claims.get('cb_total', 0)} tap={ops_claims.get('tap', 0)} "
                    f"npt={ops_claims.get('npt', 0)} spot={ops_claims.get('spot', 0)} "
                    f"counterdrill={ops_claims.get('counterdrill', 0)} jig={ops_claims.get('jig', 0)}",
                )
                _push(
                    lines,
                    f"[DEBUG] DRILL bins raw={drill_bins_raw_total} adj={drill_bins_adj_total}",
                )

                # Seed minutes so Process table shows rows
                try:
                    bv = breakdown_mutable.setdefault("bucket_view", {})
                    order = bv.setdefault("order", [])
                    if "counterbore" in bv.get("buckets", {}) and "counterbore" not in order:
                        if "drilling" in order:
                            order.insert(order.index("drilling") + 1, "counterbore")
                        else:
                            order.append("counterbore")
                    if "grinding" in bv.get("buckets", {}) and "grinding" not in order:
                        order.append("grinding")
                    _normalize_buckets(bv)
                except Exception:
                    pass

                try:
                    _normalize_buckets(breakdown_mutable.get("bucket_view"))
                except Exception:
                    pass

                # Append extra MATERIAL REMOVAL cards (Counterbore / Spot / Jig)
                _appended = _append_counterbore_spot_jig_cards(
                    lines_out=removal_card_lines,
                    chart_lines=chart_lines_all,
                    rows=built,
                    breakdown_mutable=breakdown_mutable,
                    rates=rates,
                )
                _push(lines, f"[DEBUG] extra_ops_appended={_appended}")

            # Extra MATERIAL REMOVAL cards from HOLE TABLE text (Counterbore / Spot / Jig)
            extra_ops_lines = _build_ops_cards_from_chart_lines(
                breakdown=breakdown,
                result=result,
                rates=rates,
                breakdown_mutable=breakdown_mutable,  # so buckets get minutes
                ctx=ctx,
                ctx_a=ctx_a,
                ctx_b=ctx_b,
            )
            if extra_ops_lines:
                removal_card_lines.extend(extra_ops_lines)
                lines.extend(extra_ops_lines)
                _push(lines, f"[DEBUG] extra_ops_lines={len(extra_ops_lines)}")
                for entry in extra_ops_lines:
                    if isinstance(entry, str):
                        removal_summary_extra_lines.append(entry)
                    else:
                        removal_summary_extra_lines.append(str(entry))

            # Emit the cards (will no-op if no TAP/CBore/Spot rows)
            pre_ops_len = len(lines)

            _emit_hole_table_ops_cards(
                lines,
                geo=geo_map,
                material_group=material_group,
                speeds_csv=None,
                result=result,
                breakdown=breakdown,
                rates=rates,
            )

            pre_ops_start = locals().get("pre_ops_len", len(lines))

            new_ops_lines = [
                entry
                for entry in lines[pre_ops_start:]
                if isinstance(entry, str) and not entry.startswith("[DEBUG]")
            ]
            removal_summary_extra_lines.extend(new_ops_lines)

            if not new_ops_lines:
                breakdown_mutable: MutableMapping[str, Any] | None
                if isinstance(breakdown, dict):
                    breakdown_mutable = breakdown
                elif isinstance(breakdown, _MutableMappingABC):
                    breakdown_mutable = typing.cast(MutableMapping[str, Any], breakdown)
                else:
                    breakdown_mutable = None

                fallback_lines = _build_ops_cards_from_chart_lines(
                    breakdown=breakdown,
                    result=result,
                    rates=rates,
                    breakdown_mutable=breakdown_mutable,
                    ctx=ctx,
                    ctx_a=ctx_a,
                    ctx_b=ctx_b,
                )
                if fallback_lines:
                    lines.extend(fallback_lines)
                    for entry in fallback_lines:
                        if isinstance(entry, str) and not entry.startswith("[DEBUG]"):
                            removal_summary_extra_lines.append(entry)
        except Exception as e:
            _push(
                lines,
                f"[DEBUG] material_removal_emit_skipped={e.__class__.__name__}: {e}",
            )

        removal_summary_lines = [
            str(line) for line in removal_card_lines if isinstance(line, str)
        ]
        if removal_summary_extra_lines:
            removal_summary_lines.extend(removal_summary_extra_lines)

            printed_sections = {
                "tapping": any(
                    isinstance(s, str)
                    and s.strip().upper().startswith("MATERIAL REMOVAL – TAPPING")
                    for s in removal_card_lines
                ),
                "counterbore": any(
                    isinstance(s, str)
                    and s.strip().upper().startswith("MATERIAL REMOVAL – COUNTERBORE")
                    for s in removal_card_lines
                ),
                "spot": any(
                    isinstance(s, str)
                    and s.strip().upper().startswith("MATERIAL REMOVAL – SPOT")
                    for s in removal_card_lines
                ),
                "jig": any(
                    isinstance(s, str)
                    and s.strip().upper().startswith("MATERIAL REMOVAL – JIG")
                    for s in removal_card_lines
                ),
            }

            skip_names = set()
            if printed_sections["tapping"]:
                skip_names.update({"tap", "npt tap"})
            if printed_sections["counterbore"]:
                skip_names.add("counterbore")
            if printed_sections["spot"]:
                skip_names.add("spot drill")
            if printed_sections["jig"]:
                skip_names.add("jig-grind")

            actions_summary_ready = True
            try:
                extra_bucket_ops_for_summary: dict[str, Any] = {}
                if isinstance(extra_bucket_ops, _MappingABC):
                    extra_bucket_ops_for_summary.update(extra_bucket_ops)
                elif isinstance(extra_bucket_ops, dict):
                    extra_bucket_ops_for_summary.update(extra_bucket_ops)
                extra_map_candidate = getattr(bucket_state, "extra", None)
                if isinstance(extra_map_candidate, _MappingABC):
                    extra_bucket_ops_candidate = extra_map_candidate.get("bucket_ops")
                    if isinstance(extra_bucket_ops_candidate, _MappingABC):
                        extra_bucket_ops_for_summary.update(extra_bucket_ops_candidate)
                if isinstance(extra_bucket_ops_for_summary, _MappingABC):
                    for _, entries in extra_bucket_ops_for_summary.items():
                        if not isinstance(entries, Sequence):
                            continue
                        for entry in entries:
                            if not isinstance(entry, _MappingABC):
                                continue
                            name_text = str(entry.get("name") or entry.get("op") or "").strip()
                            name_lower = name_text.lower()
                            qty_candidate = entry.get("qty")
                            try:
                                qty_val = int(float(qty_candidate))
                            except Exception:
                                qty_val = 0
                            side_val = entry.get("side")
                            if name_text and name_lower not in skip_names:
                                planner_ops_summary.append(
                                    {"name": name_text, "qty": qty_val, "side": side_val}
                                )
            except Exception as exc:
                logging.debug(
                    "[actions-summary] skipped due to %s: %s",
                    exc.__class__.__name__,
                    exc,
                    exc_info=False,
                )

        milling_bucket_obj = None
        bucket_view_snapshot = (
            breakdown.get("bucket_view") if isinstance(breakdown, _MappingABC) else None
        )
        if isinstance(bucket_view_snapshot, (_MappingABC, dict)):
            milling_bucket_obj = _extract_milling_bucket(bucket_view_snapshot)
        _render_milling_removal_card(append_line, lines, milling_bucket_obj)

    removal_summary_lines = _collect_removal_summary_lines()
    # ========================================================================

    planner_ops_rows_for_audit: Any
    if isinstance(ops_summary_map, _MappingABC):
        planner_ops_rows_for_audit = ops_summary_map
    else:
        planner_ops_rows_for_audit = ops_rows

    removal_summary_lines_for_audit = locals().get("removal_summary_lines") or []
    removal_sections_text = "\n".join(
        str(line)
        for line in removal_summary_lines_for_audit
        if isinstance(line, str)
    )

    ops_counts = audit_operations(
        planner_ops_rows_for_audit,
        removal_sections_text,
    )

    ops_counts = _apply_ops_audit_counts(
        typing.cast(MutableMapping[str, int], ops_counts),
        drill_actions=drill_actions_from_groups,
        ops_claims=ops_claims,
    )

    def _resolve_drilling_summary_candidate(*containers: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        for container in containers:
            if not isinstance(container, _MappingABC):
                continue
            for key in ("drilling", "drilling_summary"):
                candidate = container.get(key)
                if isinstance(candidate, _MappingABC):
                    return typing.cast(Mapping[str, Any], candidate)
        return None

    drilling_summary: Mapping[str, Any] | None = None
    drilling_summary = _resolve_drilling_summary_candidate(
        typing.cast(Mapping[str, Any] | None, locals().get("process_plan_summary")),
        typing.cast(Mapping[str, Any] | None, breakdown.get("process_plan"))
        if isinstance(breakdown, _MappingABC)
        else None,
        typing.cast(Mapping[str, Any] | None, result.get("process_plan"))
        if isinstance(result, _MappingABC)
        else None,
        breakdown if isinstance(breakdown, _MappingABC) else None,
        result if isinstance(result, _MappingABC) else None,
    )
    if drilling_summary is None:
        drilling_summary = {}

    drill_actions = int(ops_counts.get("drills", 0))

    try:
        extra_bucket_ops = breakdown_mutable.setdefault("extra_bucket_ops", {})
        if drill_actions > 0:
            extra_bucket_ops.setdefault("drill", []).append(
                {"name": "Drill", "qty": drill_actions, "side": None}
            )
        if (ops_claims.get("tap") or 0) > 0:
            extra_bucket_ops.setdefault("tap", []).append(
                {"name": "Tap", "qty": int(ops_claims["tap"]), "side": "front"}
            )
        if (ops_claims.get("npt") or 0) > 0:
            extra_bucket_ops.setdefault("tap", []).append(
                {"name": "NPT tap", "qty": int(ops_claims["npt"]), "side": "front"}
            )
        if (ops_claims.get("cb_front") or 0) > 0:
            extra_bucket_ops.setdefault("counterbore", []).append(
                {"name": "Counterbore", "qty": int(ops_claims["cb_front"]), "side": "front"}
            )
        if (ops_claims.get("cb_back") or 0) > 0:
            extra_bucket_ops.setdefault("counterbore", []).append(
                {"name": "Counterbore", "qty": int(ops_claims["cb_back"]), "side": "back"}
            )
        if (ops_claims.get("spot") or 0) > 0:
            extra_bucket_ops.setdefault("spot", []).append(
                {"name": "Spot drill", "qty": int(ops_claims["spot"]), "side": "front"}
            )
        if (ops_claims.get("counterdrill") or 0) > 0:
            extra_bucket_ops.setdefault("counterdrill", []).append(
                {
                    "name": "Counterdrill",
                    "qty": int(ops_claims["counterdrill"]),
                    "side": "front",
                }
            )
        if (ops_claims.get("jig") or 0) > 0:
            extra_bucket_ops.setdefault("jig-grind", []).append(
                {"name": "Jig-grind", "qty": int(ops_claims["jig"]), "side": None}
            )
    except Exception:
        pass

    drilling_summary_candidate = locals().get("drilling_summary")
    if isinstance(drilling_summary_candidate, (_MappingABC, dict)):
        drilling_summary = typing.cast(Mapping[str, Any], drilling_summary_candidate)
    else:
        drilling_summary = {}
    def _coerce_adjusted_total(value: Any) -> int | None:
        if value is None:
            return None
        try:
            numeric_val = float(value)
        except Exception:
            return None
        if not math.isfinite(numeric_val):
            return None
        try:
            return int(round(numeric_val))
        except Exception:
            return None

    def _extract_adjusted_total(source: Any) -> int | None:
        if isinstance(source, _MappingABC):
            for key in (
                "drill_actions_adjusted_total",
                "actions_adjusted_total",
                "adjusted_actions_total",
                "adjusted_total",
                "hole_count",
            ):
                try:
                    candidate_value = source.get(key)
                except Exception:
                    candidate_value = None
                candidate_int = _coerce_adjusted_total(candidate_value)
                if candidate_int is not None:
                    return candidate_int
        return _coerce_adjusted_total(source)

    drill_actions_adjusted_total: int | None = None
    for candidate_source in (
        drilling_summary,
        result.get("drilling_summary") if isinstance(result, _MappingABC) else None,
        breakdown.get("drilling_summary") if isinstance(breakdown, _MappingABC) else None,
        result.get("drill_actions_adjusted_total") if isinstance(result, _MappingABC) else None,
        breakdown.get("drill_actions_adjusted_total") if isinstance(breakdown, _MappingABC) else None,
    ):
        adjusted_candidate = _extract_adjusted_total(candidate_source)
        if adjusted_candidate is not None:
            drill_actions_adjusted_total = adjusted_candidate
            break

    drill_actions_adjusted_display = drill_actions_adjusted_total
    if drill_actions_adjusted_display is None:
        groups_payload = drilling_summary.get("groups") if isinstance(drilling_summary, _MappingABC) else None
        if isinstance(groups_payload, Sequence) and not isinstance(groups_payload, (str, bytes)):
            total_qty = 0
            seen_entry = False
            for entry in groups_payload:
                qty_candidate: Any
                if isinstance(entry, _MappingABC):
                    qty_candidate = entry.get("qty")
                else:
                    qty_candidate = getattr(entry, "qty", None)
                if qty_candidate is not None:
                    seen_entry = True
                try:
                    qty_int = int(round(float(qty_candidate))) if qty_candidate is not None else 0
                except Exception:
                    qty_int = 0
                if qty_int > 0:
                    total_qty += qty_int
            if seen_entry:
                drill_actions_adjusted_display = total_qty
                if drill_actions_adjusted_total is None:
                    drill_actions_adjusted_total = total_qty
    if drill_actions_adjusted_display is not None:
        _push(lines, f"[DEBUG] drill_actions_adjusted={int(drill_actions_adjusted_display)}")
    for key in ("tap", "npt", "cb_total", "spot", "counterdrill", "jig"):
        if key not in ops_claims:
            ops_claims[key] = 0
    counts_by_diam_current = locals().get("counts_by_diam")
    drill_bins_adj_logged = _sum_count_values(counts_by_diam_current)
    if drill_bins_adj_logged <= 0:
        try:
            drill_bins_adj_logged = int(drill_bins_adj_total)
        except Exception:
            drill_bins_adj_logged = 0
    _push(
        lines,
        f"[DEBUG] OPS TALLY drill={drill_bins_adj_logged} tap={ops_claims.get('tap', 0) + ops_claims.get('npt', 0)} "
        f"cbore={ops_claims.get('cb_total', 0)} spot={ops_claims.get('spot', 0)} "
        f"counterdrill={ops_claims.get('counterdrill', 0)} jig={ops_claims.get('jig', 0)}",
    )
    _push(
        lines,
        f"[DEBUG] OPS TALLY  drill={drill_actions} tap={ops_claims['tap']} "
        f"npt={ops_claims['npt']} cbore={ops_claims['cb_total']} "
        f"spot={ops_claims['spot']} counterdrill={ops_claims['counterdrill']} "
        f"jig={ops_claims['jig']}",
    )

    print(
        f"[ops-audit] drills={ops_counts.get('drills', 0)} "
        f"taps_total={ops_counts.get('taps_total', 0)} "
        f"(F={ops_counts.get('taps_front', 0)}, B={ops_counts.get('taps_back', 0)}) "
        f"cbore_total={ops_counts.get('counterbores_total', 0)} "
        f"(F={ops_counts.get('counterbores_front', 0)}, B={ops_counts.get('counterbores_back', 0)}) "
        f"spot={ops_counts.get('spot', 0)} "
        f"counterdrill={ops_counts.get('counterdrill', 0)} "
        f"jig_grind={ops_counts.get('jig_grind', 0)} "
        f"actions={ops_counts.get('actions_total', 0)}"
    )

    lines.append("OPERATION AUDIT – Action counts")
    lines.append("-" * 66)
    lines.append(f" Drills:        {ops_counts.get('drills', 0)}")
    lines.append(
        " Taps:          "
        f"{ops_counts.get('taps_total', 0)}  (Front {ops_counts.get('taps_front', 0)} / Back {ops_counts.get('taps_back', 0)})"
    )
    lines.append(
        " Counterbores:  "
        f"{ops_counts.get('counterbores_total', 0)}  (Front {ops_counts.get('counterbores_front', 0)} / Back {ops_counts.get('counterbores_back', 0)})"
    )
    lines.append(f" Spot:          {ops_counts.get('spot', 0)}")
    lines.append(f" Counterdrill:  {ops_counts.get('counterdrill', 0)}")
    lines.append(f" Jig-grind:     {ops_counts.get('jig_grind', 0)}")
    lines.append(f" Actions total: {ops_counts.get('actions_total', 0)}")
    lines.append("")

    # ---- Pricing ladder ------------------------------------------------------
    _push(lines, "Pricing Ladder")
    _push(lines, divider)

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

    subtotal_before_margin_val = _safe_float(subtotal_before_margin, 0.0)
    final_price_val = _safe_float(price, 0.0)
    expedite_amount_val = _safe_float(expedite_cost, 0.0)
    ladder_subtotal_val = _safe_float(ladder_totals.get("subtotal"), subtotal_before_margin_val - expedite_amount_val)

    quick_what_if_entries: list[dict[str, Any]] = []
    margin_slider_payload: dict[str, Any] | None = None
    slider_sample_points: list[dict[str, Any]] = []

    def _pct_label(value: float) -> str:
        try:
            pct_value = float(value)
        except Exception:
            pct_value = 0.0
        if not math.isfinite(pct_value):
            pct_value = 0.0
        pct_value = max(0.0, pct_value)
        text = f"{pct_value * 100:.1f}".rstrip("0").rstrip(".")
        if not text:
            text = "0"
        return f"{text}%"

    def _normalize_quick_entries(source: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, float | None]] = set()

        def _try_add(candidate: Mapping[str, Any]) -> None:
            label = str(
                candidate.get("label")
                or candidate.get("name")
                or candidate.get("title")
                or candidate.get("scenario")
                or ""
            ).strip()
            detail = str(
                candidate.get("detail")
                or candidate.get("description")
                or candidate.get("notes")
                or ""
            ).strip()

            unit_price_val: float | None = None
            for price_key in (
                "unit_price",
                "unitPrice",
                "price",
                "unit_price_usd",
                "unitPriceUsd",
                "value",
            ):
                if price_key in candidate:
                    try:
                        unit_price_val = float(candidate[price_key])
                    except Exception:
                        continue
                    else:
                        break

            delta_val: float | None = None
            for delta_key in ("delta", "delta_price", "delta_amount", "change", "difference"):
                if delta_key in candidate:
                    try:
                        delta_val = float(candidate[delta_key])
                    except Exception:
                        continue
                    else:
                        break

            margin_val: float | None = None
            for margin_key in ("margin_pct", "margin", "margin_percent", "marginPercent"):
                if margin_key in candidate:
                    try:
                        margin_val = float(candidate[margin_key])
                    except Exception:
                        continue
                    else:
                        break

            if (
                not label
                and not detail
                and unit_price_val is None
                and delta_val is None
                and margin_val is None
            ):
                return

            entry: dict[str, Any] = {}
            if label:
                entry["label"] = label
            if unit_price_val is not None and math.isfinite(unit_price_val):
                entry["unit_price"] = round(unit_price_val, 2)
            if delta_val is not None and math.isfinite(delta_val):
                entry["delta"] = round(delta_val, 2)
            if detail:
                entry["detail"] = detail
            if margin_val is not None and math.isfinite(margin_val):
                entry["margin_pct"] = float(margin_val)
            entry["currency"] = currency

            key = (entry.get("label", ""), entry.get("unit_price"))
            if key in seen_keys:
                return
            seen_keys.add(key)
            normalized.append(entry)

        def _walk(obj: Any) -> None:
            if len(normalized) >= 10:
                return
            if isinstance(obj, _MappingABC):
                _try_add(obj)
                for child in obj.values():
                    if isinstance(child, (dict, _MappingABC, list, tuple, set)):
                        _walk(child)
            elif isinstance(obj, (list, tuple, set)):
                for child in obj:
                    if isinstance(child, (dict, _MappingABC, list, tuple, set)):
                        _walk(child)

        _walk(source)
        return normalized

    quick_source_value: Any | None = None
    quick_key_candidates = (
        "quick_what_ifs",
        "quickWhatIfs",
        "quick_what_if",
        "quick_whatifs",
        "what_if_options",
        "what_if_scenarios",
    )

    for container in (result, breakdown):
        if not isinstance(container, _MappingABC):
            continue
        for key in quick_key_candidates:
            if key in container:
                quick_source_value = container.get(key)
                break
        if quick_source_value is not None:
            break
        for key, value in container.items():
            try:
                key_text = str(key).strip().lower()
            except Exception:
                continue
            if "quick" in key_text and "what" in key_text:
                quick_source_value = value
                break
        if quick_source_value is not None:
            break

    if quick_source_value is None and isinstance(decision_state, _MappingABC):
        for key in quick_key_candidates:
            if key in decision_state:
                quick_source_value = decision_state.get(key)
                break
        if quick_source_value is None:
            for key, value in decision_state.items():
                try:
                    key_text = str(key).strip().lower()
                except Exception:
                    continue
                if "quick" in key_text and "what" in key_text:
                    quick_source_value = value
                    break

    if quick_source_value is not None:
        try:
            quick_what_if_entries = _normalize_quick_entries(quick_source_value)
        except Exception:
            quick_what_if_entries = []

    if not quick_what_if_entries:
        generated: list[dict[str, Any]] = []

        if subtotal_before_margin_val > 0.0:
            margin_step = 0.05
            margin_down = round(max(0.0, margin_pct_value - margin_step), 4)
            if margin_pct_value - margin_down >= 0.005:
                price_down = round(subtotal_before_margin_val * (1.0 + margin_down), 2)
                delta_down = round(price_down - final_price_val, 2)
                generated.append(
                    {
                        "label": f"Margin {_pct_label(margin_down)}",
                        "unit_price": price_down,
                        "delta": delta_down,
                        "detail": f"Adjust margin to {_pct_label(margin_down)}.",
                        "margin_pct": float(margin_down),
                        "currency": currency,
                    }
                )

            margin_up = round(min(1.0, margin_pct_value + margin_step), 4)
            if margin_up - margin_pct_value >= 0.005:
                price_up = round(subtotal_before_margin_val * (1.0 + margin_up), 2)
                delta_up = round(price_up - final_price_val, 2)
                generated.append(
                    {
                        "label": f"Margin {_pct_label(margin_up)}",
                        "unit_price": price_up,
                        "delta": delta_up,
                        "detail": f"Adjust margin to {_pct_label(margin_up)}.",
                        "margin_pct": float(margin_up),
                        "currency": currency,
                    }
                )

        if expedite_pct_value > 0.0 and ladder_subtotal_val > 0.0:
            price_without_expedite = round(ladder_subtotal_val * (1.0 + margin_pct_value), 2)
            delta_expedite = round(price_without_expedite - final_price_val, 2)
            generated.append(
                {
                    "label": "Remove expedite",
                    "unit_price": price_without_expedite,
                    "delta": delta_expedite,
                    "detail": f"Removes expedite surcharge ({_pct_label(expedite_pct_value)}).",
                    "margin_pct": float(margin_pct_value),
                    "currency": currency,
                }
            )

        quick_what_if_entries = generated

    if quick_what_if_entries:
        deduped: list[dict[str, Any]] = []
        seen_labels: set[tuple[str, float | None]] = set()
        for entry in quick_what_if_entries:
            label_text = str(entry.get("label") or "").strip()
            if not label_text:
                label_text = f"Scenario {len(deduped) + 1}"
                entry["label"] = label_text
            try:
                price_val = float(entry.get("unit_price", 0.0))
            except Exception:
                price_val = 0.0
            key = (label_text.lower(), round(price_val, 2))
            if key in seen_labels:
                continue
            seen_labels.add(key)
            try:
                entry["unit_price"] = round(float(entry.get("unit_price", 0.0)), 2)
            except Exception:
                entry.pop("unit_price", None)
            if "delta" in entry:
                try:
                    entry["delta"] = round(float(entry["delta"]), 2)
                except Exception:
                    entry.pop("delta", None)
            entry.setdefault("currency", currency)
            deduped.append(entry)
        quick_what_if_entries = deduped

    if subtotal_before_margin_val > 0.0:
        slider_min_pct = 0.0
        slider_step_pct = 0.01
        slider_max_pct = margin_pct_value + 0.1
        slider_max_pct = max(slider_max_pct, 0.3)
        slider_max_pct = min(max(slider_max_pct, margin_pct_value), 1.0)

        slider_ticks: set[float] = set()
        slider_ticks.add(round(slider_min_pct, 4))
        slider_ticks.add(round(max(slider_min_pct, min(slider_max_pct, margin_pct_value)), 4))
        slider_ticks.add(round(slider_max_pct, 4))

        display_step = 0.05
        if slider_max_pct > slider_min_pct and display_step > 0:
            steps = int(math.floor((slider_max_pct - slider_min_pct) / display_step + 1e-6))
            for idx in range(steps + 1):
                pct_val = slider_min_pct + idx * display_step
                if pct_val < slider_min_pct - 1e-9 or pct_val > slider_max_pct + 1e-9:
                    continue
                slider_ticks.add(round(max(slider_min_pct, min(slider_max_pct, pct_val)), 4))

        slider_points: list[dict[str, Any]] = []
        for pct_val in sorted(slider_ticks):
            price_point = round(subtotal_before_margin_val * (1.0 + pct_val), 2)
            slider_points.append(
                {
                    "margin_pct": float(pct_val),
                    "label": _pct_label(pct_val),
                    "unit_price": price_point,
                    "currency": currency,
                }
            )

        if slider_points:
            margin_slider_payload = {
                "base_unit_price": round(subtotal_before_margin_val, 2),
                "current_pct": float(round(margin_pct_value, 6)),
                "current_price": round(final_price_val, 2),
                "min_pct": float(round(slider_min_pct, 6)),
                "max_pct": float(round(slider_max_pct, 6)),
                "step_pct": float(round(slider_step_pct, 6)),
                "points": slider_points,
                "currency": currency,
            }

            sample_points: list[dict[str, Any]] = []
            min_point = slider_points[0]
            max_point = slider_points[-1]
            current_point = next(
                (
                    point
                    for point in slider_points
                    if math.isclose(point["margin_pct"], margin_pct_value, rel_tol=0.0, abs_tol=1e-6)
                ),
                None,
            )
            sample_points.append(min_point)
            if current_point and current_point not in sample_points:
                sample_points.append(current_point)
            if len(slider_points) > 2:
                mid_point = slider_points[len(slider_points) // 2]
                if mid_point not in sample_points:
                    sample_points.append(mid_point)
            if max_point not in sample_points:
                sample_points.append(max_point)

            seen_points: set[float] = set()
            for point in sample_points:
                pct_val = float(point.get("margin_pct", 0.0))
                rounded_key = round(pct_val, 4)
                if rounded_key in seen_points:
                    continue
                seen_points.add(rounded_key)
                slider_sample_points.append(
                    {
                        "margin_pct": pct_val,
                        "label": str(point.get("label") or _pct_label(pct_val)),
                        "unit_price": float(point.get("unit_price", 0.0)),
                        "currency": str(point.get("currency") or currency),
                        "is_current": bool(
                            math.isclose(pct_val, margin_pct_value, rel_tol=0.0, abs_tol=1e-6)
                        ),
                    }
                )

    row("Subtotal (Labor + Directs):", subtotal)
    if applied_pcts.get("ExpeditePct"):
        row(f"+ Expedite ({_pct(applied_pcts.get('ExpeditePct'))}):", expedite_cost)
    row("= Subtotal before Margin:", subtotal_before_margin)
    row(f"Final Price with Margin ({_pct(applied_pcts.get('MarginPct'))}):", price)
    _push(lines, "")

    def _ensure_blank_line() -> None:
        if lines and lines[-1] != "":
            _push(lines, "")

    def _format_dotted_line(label: str, value_text: str, *, indent: str = "  ") -> str:
        base = f"{indent}{label}"
        try:
            total_width = int(page_width)
        except Exception:
            total_width = 74
        total_width = max(32, min(120, total_width))
        spacing = total_width - len(base) - len(value_text) - 1
        if spacing < 2:
            return f"{base} {value_text}"
        return f"{base}{'.' * spacing} {value_text}"

    section_counter = 0
    quick_section_lines: list[str] = []

    def _append_section_heading(title: str) -> None:
        nonlocal section_counter
        if section_counter > 0 and (not quick_section_lines or quick_section_lines[-1] != ""):
            quick_section_lines.append("")
        heading = f"{chr(ord('A') + section_counter)}) {title}"
        section_counter += 1
        quick_section_lines.append(heading)

    if slider_sample_points:
        qty_display: str
        if isinstance(qty, int) and qty > 0:
            qty_display = str(qty)
        else:
            qty_display = str(max(1, int(round(_safe_float(qty, 1.0)))))
        _append_section_heading(f"Margin Slider (Qty = {qty_display})")
        for point in sorted(slider_sample_points, key=lambda p: p.get("margin_pct", 0.0)):
            label_text = f"{str(point.get('label') or '')} margin".strip()
            if not label_text:
                label_text = "Margin"
            if point.get("is_current"):
                label_text = f"{label_text} (current)"
            amount_text = fmt_money(_safe_float(point.get("unit_price"), 0.0), point.get("currency", currency))
            quick_section_lines.append(_format_dotted_line(label_text, amount_text))

    current_qty = qty if isinstance(qty, int) and qty > 0 else max(1, int(round(_safe_float(qty, 1.0))))
    direct_per_part = max(0.0, _safe_float(directs, 0.0))
    labor_machine_per_part = max(0.0, _safe_float(machine_labor_total, 0.0))
    amortized_per_part = max(0.0, _safe_float(nre_per_part, 0.0))
    amortized_per_lot = amortized_per_part * max(1, current_qty)
    pass_through_lot = max(0.0, _safe_float(pass_through_total, 0.0))
    vendor_items_lot = max(0.0, _safe_float(vendor_items_total, 0.0))
    direct_fixed_lot = pass_through_lot + vendor_items_lot
    divisor = max(1, current_qty)
    direct_variable_per_part = max(0.0, direct_per_part - (direct_fixed_lot / divisor))

    qty_candidates_raw = [1, 2, 5, 10]
    if current_qty not in qty_candidates_raw:
        qty_candidates_raw.append(current_qty)
    qty_candidates = sorted({q for q in qty_candidates_raw if isinstance(q, int) and q > 0})

    qty_break_rows: list[tuple[int, float, float, float, float]] = []
    for candidate_qty in qty_candidates:
        labor_part = labor_machine_per_part + (amortized_per_lot / candidate_qty if candidate_qty > 0 else 0.0)
        direct_part = direct_variable_per_part + (direct_fixed_lot / candidate_qty if candidate_qty > 0 else 0.0)
        base_subtotal_candidate = labor_part + direct_part
        subtotal_with_expedite = base_subtotal_candidate * (1.0 + max(0.0, expedite_pct_value))
        final_candidate = subtotal_with_expedite * (1.0 + max(0.0, margin_pct_value))
        qty_break_rows.append(
            (
                candidate_qty,
                round(labor_part, 2),
                round(direct_part, 2),
                round(subtotal_with_expedite, 2),
                round(final_candidate, 2),
            )
        )

    if qty_break_rows:
        heading_text = f"Qty break (assumes same ops; programming amortized; {_pct_label(margin_pct_value)} margin"
        if expedite_pct_value > 0:
            heading_text += f"; expedite {_pct_label(expedite_pct_value)}"
        heading_text += ")"
        _append_section_heading(heading_text)
        quick_section_lines.append("  Qty, Labor $/part, Directs $/part, Subtotal, Final")
        for row_qty, labor_val, direct_val, subtotal_val, final_val in qty_break_rows:
            qty_field = str(row_qty).rjust(3)
            labor_field = fmt_money(labor_val, currency).rjust(12)
            direct_field = fmt_money(direct_val, currency).rjust(12)
            subtotal_field = fmt_money(subtotal_val, currency).rjust(12)
            final_field = fmt_money(final_val, currency).rjust(12)
            quick_section_lines.append(
                f"  {qty_field},   {labor_field}, {direct_field}, {subtotal_field}, {final_field}"
            )

    other_quick_entries: list[dict[str, Any]] = []
    if quick_what_if_entries:
        for entry in quick_what_if_entries:
            margin_present = entry.get("margin_pct") is not None
            label_lower = str(entry.get("label") or "").strip().lower()
            if slider_sample_points and margin_present and "margin" in label_lower:
                continue
            other_quick_entries.append(entry)

    if other_quick_entries:
        _append_section_heading("Other quick toggles")
        for entry in other_quick_entries:
            label_text = str(entry.get("label") or "").strip() or "Scenario"
            amount_val = _safe_float(entry.get("unit_price"), 0.0)
            amount_text = fmt_money(amount_val, entry.get("currency", currency))
            delta_val = entry.get("delta")
            if delta_val is not None:
                delta_float = _safe_float(delta_val, 0.0)
                if delta_float < -0.01:
                    delta_prefix = "-"
                elif abs(delta_float) <= 0.01:
                    delta_prefix = "±"
                else:
                    delta_prefix = "+"
                delta_text = fmt_money(abs(delta_float), entry.get("currency", currency))
                base_line = f"  {label_text}: {amount_text} ({delta_prefix}{delta_text})"
            else:
                base_line = f"  {label_text}: {amount_text}"
            detail_text = str(entry.get("detail") or "").strip()
            if detail_text:
                base_line = f"{base_line} — {detail_text}"
            quick_section_lines.append(base_line)

    while quick_section_lines and quick_section_lines[-1] == "":
        quick_section_lines.pop()

    if quick_section_lines:
        _ensure_blank_line()
        _push(lines, "QUICK WHAT-IFS (INTERNAL KNOBS)")
        _push(lines, divider)
        _push(lines, "Quick What-Ifs")
        for text_line in quick_section_lines:
            _push(lines, text_line)
        _ensure_blank_line()

    # ---- LLM adjustments bullets (optional) ---------------------------------
    if llm_notes:
        _push(lines, "LLM Adjustments")
        _push(lines, divider)
        import textwrap as _tw
        for n in llm_notes:
            for w in _tw.wrap(str(n), width=page_width):
                _push(lines, f"- {w}")
        _push(lines, "")

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

        if process_rows_rendered:
            plan_info_payload.setdefault(
                "process_rows_rendered",
                [
                    (
                        name,
                        minutes,
                        machine,
                        labor,
                        total,
                    )
                    for (name, minutes, machine, labor, total) in process_rows_rendered
                ],
            )

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

    if bucket_why_summary_line:
        summary_text = bucket_why_summary_line.strip()
        if summary_text and summary_text not in why_parts:
            why_parts.append(summary_text)

    if why_parts:
        if lines and lines[-1]:
            _push(lines, "")
        _push(lines, "Why this price")
        _push(lines, divider)
        for part in why_parts:
            write_wrapped(part, "  ")
        if lines[-1]:
            _push(lines, "")
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

            _push(lines, "DEBUG — Drilling sanity")
            _push(lines, divider)
            def _fmt(x, unit):
                return "—" if x is None or not math.isfinite(float(x)) else f"{float(x):.2f} {unit}"
            _push(lines, 
                "  bucket(planner): "
                + _fmt(_planner_min, "min")
                + "   canonical: "
                + _fmt(_canon_min, "min")
                + "   hour_summary: "
                + _fmt(_hsum_hr, "hr")
                + "   meta: "
                + _fmt(_meta_hr, "hr")
            )
            _push(lines, "")
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
    def _render_as_float(value: Any, default: float = 0.0) -> float:
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

    margin_pct_value = _render_as_float(applied_pcts.get("MarginPct"), 0.0)
    expedite_pct_value = _render_as_float(applied_pcts.get("ExpeditePct"), 0.0)
    expedite_amount = _render_as_float(expedite_cost, 0.0)
    subtotal_before_margin_val = _render_as_float(subtotal_before_margin, 0.0)
    final_price_val = _render_as_float(price, 0.0)
    margin_amount = max(0.0, final_price_val - subtotal_before_margin_val)
    labor_total_amount = _render_as_float(
        (breakdown or {}).get("total_labor_cost"),
        _render_as_float(ladder_labor, 0.0),
    )
    direct_total_amount = _render_as_float(total_direct_costs_value, 0.0)

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
                "amount": round(_render_as_float(material_display_amount, 0.0), 2),
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
                "amount": round(_render_as_float(entry_amount, 0.0), 2),
            }
        )

    processes_entries: list[dict[str, Any]] = []
    seen_process_labels: set[str] = set()
    for spec in bucket_row_specs:
        label = str(spec.label or "").strip()
        amount_val = _render_as_float(spec.total, 0.0)
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
                "hours": round(_render_as_float(spec.hours, 0.0), 2),
                "minutes": round(_render_as_float(spec.minutes, 0.0), 2),
                "labor_amount": round(_render_as_float(spec.labor, 0.0), 2),
                "machine_amount": round(_render_as_float(spec.machine, 0.0), 2),
                "rate": round(_render_as_float(spec.rate, 0.0), 2) if _render_as_float(spec.rate, 0.0) else 0.0,
            }
        )

    if qty <= 1:
        for entry in processes_entries:
            if str(entry.get("label")) == PROGRAMMING_PER_PART_LABEL:
                entry["label"] = PROGRAMMING_AMORTIZED_LABEL

    def _process_table_rows_from_view(
        view: Mapping[str, Any] | None,
    ) -> list[list[str]]:
        if not isinstance(view, _MappingABC):
            return []
        buckets_obj = view.get("buckets")
        if not isinstance(buckets_obj, _MappingABC):
            return []

        def _as_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        normalized: dict[str, Mapping[str, Any]] = {}
        for raw_key, raw_entry in buckets_obj.items():
            if not isinstance(raw_entry, _MappingABC):
                continue
            key_text = str(raw_key or "").strip()
            if not key_text:
                continue
            key_lower = key_text.lower()
            normalized.setdefault(key_lower, raw_entry)
            normalized.setdefault(key_lower.replace(" ", "_"), raw_entry)
            canon = _canonical_bucket_key(key_text)
            if canon:
                normalized.setdefault(str(canon).lower(), raw_entry)

        pretty_labels = {
            "programming_amortized": PROGRAMMING_AMORTIZED_LABEL,
            "programming": PROGRAMMING_PER_PART_LABEL,
            "milling": "Milling",
            "drilling": "Drilling",
            "tapping": "Tapping",
            "inspection": "Inspection",
        }
        order = (
            "programming_amortized",
            "milling",
            "drilling",
            "tapping",
            "inspection",
        )

        def _lookup_entry(key: str) -> Mapping[str, Any] | None:
            candidate = normalized.get(key)
            if candidate is not None:
                return candidate
            alt = key.replace("_", " ")
            return normalized.get(alt)

        rows: list[list[str]] = []
        for canon_key in order:
            entry = _lookup_entry(canon_key)
            label_key = canon_key
            if entry is None and canon_key == "programming_amortized":
                entry = _lookup_entry("programming")
                label_key = "programming"
            if not isinstance(entry, _MappingABC):
                continue
            minutes_val = _as_float(entry.get("minutes"))
            if minutes_val <= 0.0:
                continue
            machine_val = _as_float(entry.get("machine$"))
            labor_val = _as_float(entry.get("labor$"))
            total_val = entry.get("total$")
            total_float = _as_float(total_val)
            if total_float <= 0.0:
                total_float = machine_val + labor_val
            label = pretty_labels.get(
                label_key,
                str(label_key).replace("_", " ").title(),
            )
            rows.append(
                [
                    label,
                    f"{minutes_val:.2f}",
                    f"${machine_val:.2f}",
                    f"${labor_val:.2f}",
                    f"${total_float:.2f}",
                ]
            )
        return rows

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

    process_table_rows_payload = _process_table_rows_from_view(bucket_view_struct)
    if not process_table_rows_payload:
        process_table_rows_payload = _process_table_rows_from_view(bucket_view_obj)
    if process_table_rows_payload:
        render_payload["process_table"] = process_table_rows_payload

    if quick_what_if_entries:
        render_payload["quick_what_ifs"] = quick_what_if_entries
    if margin_slider_payload is not None:
        render_payload["margin_slider"] = margin_slider_payload

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
if "parser_rules_v2" not in PARAMS_DEFAULT:
    PARAMS_DEFAULT["parser_rules_v2"] = False

def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

_PARSER_RULES_ENV = os.getenv("CADQ_PARSER_RULES_V2")
PARSER_RULES_V2_ENABLED = bool(PARAMS_DEFAULT.get("parser_rules_v2")) or _truthy_env(_PARSER_RULES_ENV)
if PARSER_RULES_V2_ENABLED:
    logging.info("[rules] parser_rules_v2=ON (env=%s, config=%s)", _PARSER_RULES_ENV, PARAMS_DEFAULT.get("parser_rules_v2"))
else:
    logging.info("[rules] parser_rules_v2=OFF (env=%s, config=%s)", _PARSER_RULES_ENV, PARAMS_DEFAULT.get("parser_rules_v2"))

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

MONEY_RE = r"(?:rate|/hr|per\s*hour|per\s*hr|price|cost|\$)"

# ===== QUOTE HELPERS ========================================================

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


def _zero_planner_if_row_suppressed(
    pps: Mapping[str, Any] | MutableMapping[str, Any] | None,
    emit_flags: Mapping[str, bool] | None,
) -> None:
    """Zero planner section minutes when the corresponding row is hidden."""

    if not isinstance(pps, (_MutableMappingABC, dict)):
        return

    emit = emit_flags or {}

    def _targets(container: Mapping[str, Any] | MutableMapping[str, Any] | None) -> list[MutableMapping[str, Any]]:
        if isinstance(container, dict):
            return [container]
        if isinstance(container, _MutableMappingABC):
            return [typing.cast(MutableMapping[str, Any], container)]
        return []

    sections: list[MutableMapping[str, Any]] = _targets(pps)
    try:
        planner_sections = pps.get("planner_sections")  # type: ignore[call-arg]
    except Exception:
        planner_sections = None
    sections.extend(_targets(planner_sections))

    if not sections:
        return

    for key in ("milling", "drilling", "tapping", "inspection"):
        if emit.get(key, False):
            continue
        for container in sections:
            try:
                entry = container.get(key)
            except Exception:
                entry = None
            if not isinstance(entry, (_MutableMappingABC, dict)):
                continue
            try:
                entry["total_minutes_billed"] = 0.0
            except Exception:
                pass
            try:
                if "total_minutes_with_toolchange" in entry:
                    entry["total_minutes_with_toolchange"] = 0.0
            except Exception:
                pass
            try:
                if "total_minutes" in entry:
                    entry["total_minutes"] = 0.0
            except Exception:
                pass
            try:
                entry["source"] = "suppressed"
            except Exception:
                pass


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
    if isinstance(geo_context, _MappingABC) and not isinstance(geo_context, dict):
        geo_context = dict(geo_context)
    elif not isinstance(geo_context, dict):
        geo_context = {}
    planner_inputs = dict(ui_vars or {})
    rates = dict(rates or {})
    geo_payload: dict[str, Any] = geo_context
    if isinstance(geo_payload, dict):
        existing_family = str(geo_payload.get("process_planner_family") or "").strip()
        hole_count_val = _coerce_float_or_none(geo_payload.get("hole_count")) or 0.0
        thickness_in_val = _coerce_float_or_none(geo_payload.get("thickness_in"))
        if not existing_family and hole_count_val > 0 and thickness_in_val:
            geo_payload["process_planner_family"] = "die_plate"
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

    from cad_quoter.pricing.planner import price_with_planner

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
    geo_for_breakdown = geo_context if isinstance(geo_context, dict) else {}
    breakdown["geo_context"] = geo_for_breakdown
    breakdown["geo"] = geo_for_breakdown

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

    mat_block_raw = _compute_material_block(
        geo_context if isinstance(geo_context, dict) else {},
        mat_key,
        _coerce_float_or_none(density),
        scrap_pct_effective,
        stock_price_source=stock_price_source,
        cfg=cfg,
    )
    if isinstance(mat_block_raw, dict):
        mat_block = mat_block_raw
    elif isinstance(mat_block_raw, _MappingABC):
        mat_block = dict(mat_block_raw)
    else:
        mat_block = {}
    breakdown["material_block"] = mat_block
    grams_per_lb = 1000.0 / LB_PER_KG
    material_entry_raw = breakdown.setdefault("material", {})
    if isinstance(material_entry_raw, dict):
        material_entry: dict[str, Any] = material_entry_raw
    elif isinstance(material_entry_raw, _MappingABC):
        material_entry = dict(material_entry_raw)
        breakdown["material"] = material_entry
    else:
        material_entry = {}
        breakdown["material"] = material_entry

    if stock_price_source:
        material_entry.setdefault("stock_price_source", stock_price_source)
        mat_block.setdefault("stock_price_source", stock_price_source)
    if scrap_price_source:
        material_entry.setdefault("scrap_price_source", scrap_price_source)
        mat_block.setdefault("scrap_price_source", scrap_price_source)

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
    base_material_cost = round(base_material_cost, 2)
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
        recovery = (
            recovery_override
            if recovery_override is not None
            else _materials.SCRAP_RECOVERY_DEFAULT
        )
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
            if family_hint is not None and not isinstance(family_hint, str):
                family_hint = str(family_hint)
            if isinstance(family_hint, str):
                family_hint = family_hint.strip() or None
            price_candidate = _wieland_scrap_usd_per_lb(family_hint)
            if price_candidate is not None:
                wieland_scrap_price = float(price_candidate)
                scrap_price_used = float(price_candidate)
                scrap_credit_source = "wieland"
                mat_block["scrap_price_usd_per_lb"] = float(scrap_price_used)
            else:
                scrap_price_used = _materials.SCRAP_PRICE_FALLBACK_USD_PER_LB
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

    bucket_minutes_detail_raw = breakdown.setdefault("bucket_minutes_detail", {})
    if isinstance(bucket_minutes_detail_raw, dict):
        bucket_minutes_detail_for_render = bucket_minutes_detail_raw
    elif isinstance(bucket_minutes_detail_raw, _MappingABC):
        bucket_minutes_detail_for_render = dict(bucket_minutes_detail_raw)
        breakdown["bucket_minutes_detail"] = bucket_minutes_detail_for_render
    else:
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
            if isinstance(breakdown, _MutableMappingABC):
                breakdown["pricing_source"] = "Planner"

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
            if (
                planner_machine_cost_total > 0.0
                and machine_rendered > 0.0
                and abs(planner_machine_cost_total - machine_rendered) > _PLANNER_BUCKET_ABS_EPSILON
            ):
                breakdown["red_flags"].append("Planner totals drifted (machine cost)")
            labor_rendered = float(_coerce_float_or_none(process_costs.get("Labor")) or 0.0)
            if (
                planner_labor_cost_total > 0.0
                and labor_rendered > 0.0
                and abs(planner_labor_cost_total - labor_rendered) > _PLANNER_BUCKET_ABS_EPSILON
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
            minutes_val = float(_safe_float(entry.get("minutes")))
            machine_val = float(_bucket_cost(entry, "machine_cost", "machine$"))
            labor_val = float(_bucket_cost(entry, "labor_cost", "labor$"))
            if canon_key == "milling" and labor_val > 0.0:
                machine_val += labor_val
                labor_val = 0.0
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
    drill_debug_lines: list[str] = []
    if not use_planner:
        if hole_diams and thickness_in and drilling_rate > 0:
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

    drill_actions_adjusted_total = None
    groups_existing = drilling_summary.get("groups")
    has_groups = False
    if isinstance(groups_existing, Sequence) and not isinstance(groups_existing, (str, bytes)):
        has_groups = bool(groups_existing)

    fallback_groups: list[dict[str, Any]] = []
    ops_claims_map: Mapping[str, Any] | None = None
    if isinstance(geo_payload, _MappingABC):
        claims_candidate = geo_payload.get("ops_claims")
        if isinstance(claims_candidate, _MappingABC):
            ops_claims_map = claims_candidate
        else:
            summary_candidate = geo_payload.get("ops_summary")
            if isinstance(summary_candidate, _MappingABC):
                claims_candidate = summary_candidate.get("claims")
                if isinstance(claims_candidate, _MappingABC):
                    ops_claims_map = claims_candidate
    if not has_groups and hole_diams:
        fallback_groups = build_drill_groups_from_geometry(
            hole_diams,
            thickness_in,
            ops_claims_map,
            geo_payload,
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
        drill_actions_adjusted_total = fallback_hole_count

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

        drilling_labor_rate = _lookup_rate(
            "DrillingLaborRate", rates, params, default_rates, fallback=0.0
        )
        if drilling_labor_rate <= 0.0:
            drilling_labor_rate = _lookup_rate(
                "LaborRate", rates, params, default_rates, fallback=45.0
            )

        attended_fraction: float | None = None
        if isinstance(drilling_meta_entry, _MappingABC):
            for key in (
                "attended_fraction",
                "attended_frac",
                "labor_fraction",
                "labor_attended_fraction",
            ):
                attended_candidate = drilling_meta_entry.get(key)
                if attended_candidate is None:
                    continue
                attended_value = _coerce_float_or_none(attended_candidate)
                if attended_value is not None:
                    attended_fraction = attended_value
                    break
        if attended_fraction is None:
            attended_fraction = 1.0
        attended_fraction = max(0.0, min(attended_fraction, 1.0))

        labor_cost_val = (drilling_minutes_for_bucket / 60.0) * drilling_labor_rate
        labor_cost_val *= attended_fraction
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

        geometry_for_milling: Mapping[str, Any] | None = None
        if isinstance(planner_result, _MappingABC):
            geom_candidate = planner_result.get("geometry")
            if isinstance(geom_candidate, _MappingABC):
                geometry_for_milling = geom_candidate
            else:
                geom_candidate = planner_result.get("geom")
                if isinstance(geom_candidate, _MappingABC):
                    geometry_for_milling = geom_candidate
        if geometry_for_milling is None and isinstance(geo_payload, _MappingABC):
            geometry_for_milling = geo_payload

        milling_backfill = None
        if geometry_for_milling is not None:
            try:
                milling_backfill = estimate_milling_minutes_from_geometry(
                    geom=geometry_for_milling,
                    sf_df=speeds_feeds_table,
                    material_group=material_group_display,
                    rates=rates,
                    emit_bottom_face=bool(geometry_for_milling.get("flip_required")),
                )
            except Exception:
                milling_backfill = None

        if milling_backfill:
            new_minutes = float(_safe_float(milling_backfill.get("minutes")))
            new_machine = float(_safe_float(milling_backfill.get("machine$")))
            new_labor = float(_safe_float(milling_backfill.get("labor$")))
            new_total = float(_safe_float(milling_backfill.get("total$")))
            if new_total <= 0.0:
                new_total = new_machine + new_labor

            if new_minutes > 0.0 and (new_machine > 0.0 or new_labor > 0.0):
                existing_entry = buckets.get("milling")
                existing_minutes = (
                    float(_safe_float(existing_entry.get("minutes")))
                    if isinstance(existing_entry, _MappingABC)
                    else 0.0
                )
                existing_machine = (
                    float(_safe_float(existing_entry.get("machine$")))
                    if isinstance(existing_entry, _MappingABC)
                    else 0.0
                )
                existing_labor = (
                    float(_safe_float(existing_entry.get("labor$")))
                    if isinstance(existing_entry, _MappingABC)
                    else 0.0
                )

                should_update_minutes = existing_minutes <= 0.01
                should_update_costs = abs(existing_machine) <= 0.01 and abs(existing_labor) <= 0.01

                if should_update_minutes or should_update_costs or existing_entry is None:
                    entry_minutes = new_minutes if should_update_minutes or existing_entry is None else existing_minutes
                    entry_machine = new_machine if should_update_costs or existing_entry is None else existing_machine
                    entry_labor = new_labor if should_update_costs or existing_entry is None else existing_labor
                    if entry_labor > 0.0:
                        entry_machine += entry_labor
                        entry_labor = 0.0
                    entry_total = entry_machine + entry_labor

                    buckets["milling"] = {
                        "minutes": entry_minutes,
                        "machine$": entry_machine,
                        "labor$": entry_labor,
                        "total$": entry_total,
                    }
                    aggregated_bucket_minutes["milling"] = {
                        "minutes": entry_minutes,
                        "machine$": entry_machine,
                        "labor$": entry_labor,
                    }
                    print(f"[INFO] [bucket/milling] {buckets['milling']}")

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
            minutes_val = float(_safe_float(entry.get("minutes")))
            machine_val = float(_safe_float(entry.get("machine$")))
            labor_val = float(_safe_float(entry.get("labor$")))
            total_val = float(_safe_float(entry.get("total$")))
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
                minutes_val = float(_safe_float(metrics.get("minutes")))
                machine_val = float(_safe_float(metrics.get("machine$")))
                labor_val = float(_safe_float(metrics.get("labor$")))
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

            extra_candidates = [
                locals().get("extra_map"),
                locals().get("removal_card_extra"),
                getattr(locals().get("bucket_state"), "extra", None),
                bucket_view.get("extra") if isinstance(bucket_view, (_MappingABC, dict)) else None,
            ]
            extra_map: Mapping[str, Any] | None = None
            for candidate in extra_candidates:
                if isinstance(candidate, _MappingABC):
                    extra_map = typing.cast(Mapping[str, Any], candidate)
                    break

            debug_lines = locals().get("lines")
            debug_lines_list: list[str] | None = (
                debug_lines if isinstance(debug_lines, list) else None
            )

            drill_minutes_total = _pick_drill_minutes(
                process_plan_summary,
                extra_map,
            )  # minutes
            drill_minutes_total = _safe_float(drill_minutes_total, 0.0)
            assert 0 <= drill_minutes_total <= 10000, (
                f"drilling minutes out of range: {drill_minutes_total}"
            )
            drill_mrate = (
                _lookup_bucket_rate("drilling", rates)
                or _lookup_bucket_rate("machine", rates)
                or 45.0
            )
            drill_lrate = (
                _lookup_bucket_rate("drilling_labor", rates)
                or _lookup_bucket_rate("labor", rates)
                or 45.0
            )

            _purge_legacy_drill_sync(bucket_view)
            _set_bucket_minutes_cost(
                bucket_view,
                "drilling",
                drill_minutes_total,
                drill_mrate,
                drill_lrate,
            )

            drilling_bucket = None
            if isinstance(bucket_view, (_MappingABC, dict)):
                drilling_bucket = (bucket_view.get("buckets") or {}).get("drilling")
            logging.info(
                f"[bucket] drilling_minutes={drill_minutes_total} drilling_bucket={drilling_bucket}"
            )

            _normalize_buckets(bucket_view)

            buckets_final: Mapping[str, Any] | None = None
            if isinstance(bucket_view, (_MappingABC, dict)):
                buckets_final = bucket_view.get("buckets")
            dbg(debug_lines_list, f"buckets_final={buckets_final or {}}")

            drilling_dbg_entry: Mapping[str, Any] | None = None
            try:
                buckets_dbg = bucket_view.get("buckets") if isinstance(bucket_view, dict) else None
                if buckets_dbg is None and isinstance(bucket_view, _MappingABC):
                    buckets_dbg = bucket_view.get("buckets")
                if isinstance(buckets_dbg, (_MappingABC, dict)):
                    drilling_dbg_entry = buckets_dbg.get("drilling")  # type: ignore[index]
            except Exception:
                drilling_dbg_entry = None

            debug_lines = typing.cast(
                list[str] | None,
                locals().get("lines") if "lines" in locals() else None,
            )
            if debug_lines is not None:
                dbg(debug_lines, f"drilling_bucket={drilling_dbg_entry or {}}")

    roughing_hours = _coerce_float_or_none(value_map.get("Roughing Cycle Time"))
    if roughing_hours is None:
        roughing_hours = _coerce_float_or_none(value_map.get("Roughing Cycle Time (hr)"))
    milling_rate = _lookup_rate("MillingRate", rates, params, default_rates, fallback=100.0)
    milling_hours = float(roughing_hours or 0.0)
    if milling_hours > 0:
        milling_minutes_total = milling_hours * 60.0
        bucket_view["milling"] = {
            "minutes": milling_minutes_total,
            "machine_cost": milling_hours * milling_rate,
            "labor_cost": 0.0,
        }
        process_meta["milling"] = {
            "hr": milling_hours,
            "minutes": milling_minutes_total,
            "rate": milling_rate,
            "basis": ["planner_milling_backfill"],
        }

        milling_minutes_seed = float(milling_minutes_total)
        milling_machine_rate_seed = float(milling_rate or 0.0)
        milling_labor_rate_seed = (
            _lookup_bucket_rate("milling_labor", rates)
            or _lookup_bucket_rate("labor", rates)
            or 0.0
        )

        _set_bucket_minutes_cost(
            bucket_view,
            "milling",
            milling_minutes_seed,
            milling_machine_rate_seed,
            float(milling_labor_rate_seed or 0.0),
        )

        _normalize_buckets(bucket_view)

        milling_dbg_entry: Mapping[str, Any] | None = None
        try:
            buckets_dbg = bucket_view.get("buckets") if isinstance(bucket_view, dict) else None
            if buckets_dbg is None and isinstance(bucket_view, _MappingABC):
                buckets_dbg = bucket_view.get("buckets")
            if isinstance(buckets_dbg, (_MappingABC, dict)):
                milling_dbg_entry = buckets_dbg.get("milling")  # type: ignore[index]
        except Exception:
            milling_dbg_entry = None

        debug_lines = typing.cast(
            list[str] | None,
            locals().get("lines") if "lines" in locals() else None,
        )
        if debug_lines is not None:
            _push(debug_lines, f"[DEBUG] milling_bucket={milling_dbg_entry or {}}")

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
    # use module-level 're'
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

    df = typing.cast(Any, df)

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
    columns_attr = getattr(df, "columns", None)
    if columns_attr is None:
        raise AttributeError("DataFrame-like object must expose a 'columns' attribute")

    for col in list(columns_attr):
        key = _norm_col(col)
        if key in canon_map:
            rename[col] = canon_map[key]

    if rename:
        rename_fn = getattr(df, "rename", None)
        if callable(rename_fn):
            renamed = rename_fn(columns=rename)
            if renamed is not None:
                df = typing.cast(Any, renamed)
            columns_attr = getattr(df, "columns", columns_attr)

    if df is None:
        raise TypeError("DataFrame normalization returned None")

    if not hasattr(df, "__setitem__"):
        raise TypeError("DataFrame-like object must support item assignment")

    for col in REQUIRED_COLS:
        if col not in columns_attr:
            df[col] = ""
            columns_attr = getattr(df, "columns", columns_attr)

    return df

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
LB_PER_IN3_PER_GCC = _LB_PER_IN3_PER_GCC

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


def _aggregate_ops_legacy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy aggregate implementation (regex-based) preserved for fallback."""

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
        diameter_val = _coerce_float_or_none(r.get("diameter_in"))
        if diameter_val is not None:
            simple_row["diameter_in"] = float(diameter_val)
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
        "rows": rows_simple,
        "rows_detail": detail,
        "actions_total": int(actions_total),
        "back_ops_total": int(back_ops_total),
        "flip_required": bool(flip_required),
        "built_rows": int(built_rows),
    }


def _parser_rules_v2_enabled() -> bool:
    """Return True when parser_rules_v2 feature flag is enabled."""

    env_val = os.getenv("PARSER_RULES_V2")
    if env_val is not None:
        normalized = str(env_val).strip().lower()
        if normalized in {"", "0", "false", "off", "no"}:
            return False
        return True
    try:
        params_obj = PARAMS_DEFAULT  # type: ignore[name-defined]
    except NameError:
        params_obj = {}
    try:
        if isinstance(params_obj, _MappingABC):
            for key in ("parser_rules_v2", "ParserRulesV2", "parserRulesV2"):
                if key in params_obj:
                    normalized = str(params_obj.get(key)).strip().lower()
                    if normalized in {"", "0", "false", "off", "no"}:
                        return False
                    return True
    except Exception:
        pass
    return False


def _normalize_ops_entries(
    ops_entries: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize chart-derived operation entries into a consistent schema."""

    normalized: list[dict[str, Any]] = []
    if not ops_entries:
        return normalized

    def _truthy_flag(value: Any) -> bool:
        try:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            text = str(value).strip().lower()
            if not text:
                return False
            return text not in {"0", "false", "off", "no", "n"}
        except Exception:
            return False

    for entry in ops_entries:
        if not isinstance(entry, _MappingABC):
            continue
        raw_type = str(entry.get("type") or entry.get("op") or "").strip().lower()
        derived_ops: list[tuple[str, Mapping[str, Any]]] = []

        ops_payload = entry.get("ops")
        if isinstance(ops_payload, Iterable):
            for op_payload in ops_payload:
                if not isinstance(op_payload, _MappingABC):
                    continue
                payload_type = str(
                    op_payload.get("type") or op_payload.get("op") or ""
                ).strip()
                if not payload_type:
                    continue
                derived_ops.append((payload_type, op_payload))

        if not derived_ops:
            if raw_type:
                derived_ops.append((raw_type, entry))
            else:
                if entry.get("tap"):
                    derived_ops.append(("tap", entry))
                if _truthy_flag(entry.get("cbore")):
                    derived_ops.append(("cbore", entry))
                if _truthy_flag(entry.get("csk")):
                    derived_ops.append(("csk", entry))
                if _truthy_flag(entry.get("thru")):
                    derived_ops.append(("drill", entry))
                if _truthy_flag(entry.get("jig_grind")):
                    derived_ops.append(("jig_grind", entry))

        for derived_type, payload in derived_ops:
            op = dict(entry)
            op.pop("ops", None)
            if isinstance(payload, _MappingABC) and payload is not entry:
                for key, value in payload.items():
                    op[key] = value
            op_type_raw = derived_type.strip().lower()
            if op_type_raw in {"counterbore", "c'bore"}:
                op_type = "cbore"
            elif op_type_raw in {"countersink", "csink", "c'sink", "csk"}:
                op_type = "csk"
            elif op_type_raw in {"spot", "spot_drill", "center_drill", "c'drill"}:
                op_type = "spot"
            elif op_type_raw in {"jig", "jig_grind", "jig-grind"}:
                op_type = "jig_grind"
            elif op_type_raw in {"tapping", "tap"}:
                op_type = "tap"
            elif op_type_raw in {"drill", "deep_drill"}:
                op_type = "drill"
            else:
                op_type = op_type_raw

            qty = _coerce_int_or_zero(op.get("qty"))
            if qty <= 0:
                continue

            side_raw = str(op.get("side") or op.get("face") or "").strip()
            from_face = str(entry.get("from_face") or "").strip()
            if not side_raw and from_face:
                side_raw = from_face
            side_upper = side_raw.upper()
            double_sided = _truthy_flag(op.get("double_sided")) or side_upper in {
                "FRONT & BACK",
                "FRONT AND BACK",
                "BOTH",
                "BOTH SIDES",
                "2 SIDES",
                "TWO SIDES",
            }
            if side_upper == "BACK" or _truthy_flag(op.get("from_back")):
                base_side = "BACK"
            elif side_upper == "FRONT":
                base_side = "FRONT"
            else:
                base_side = "FRONT"
            sides = ["FRONT", "BACK"] if double_sided else [base_side]

            ref = str(op.get("ref") or op.get("hole") or "").strip()
            thread = str(op.get("thread") or op.get("tap") or "").strip()

            depth_in = _coerce_float_or_none(op.get("depth_in"))
            if depth_in is None:
                depth_mm = _coerce_float_or_none(op.get("depth_mm"))
                if depth_mm is not None:
                    depth_in = float(depth_mm) / 25.4

            dia_in = _coerce_float_or_none(op.get("dia_in"))
            if dia_in is None:
                dia_in = _coerce_float_or_none(op.get("diameter_in"))
            if dia_in is None:
                dia_mm = _coerce_float_or_none(op.get("dia_mm"))
                if dia_mm is not None:
                    dia_in = float(dia_mm) / 25.4
            if dia_in is None:
                major_mm = _coerce_float_or_none(op.get("major_mm"))
                if major_mm is not None:
                    dia_in = float(major_mm) / 25.4
            if dia_in is None:
                dia_in = _coerce_float_or_none(op.get("ref_dia_in"))
            if dia_in is None and ref:
                ref_in = _parse_ref_to_inch(ref)
                if ref_in is not None:
                    dia_in = ref_in

            if thread and dia_in is None:
                dia_in = _coerce_float_or_none(entry.get("major_dia_in"))

            ref_label: str
            if thread:
                ref_label = thread
            elif dia_in is not None:
                ref_label = f"Ø{dia_in:.4f}"
            else:
                ref_label = ref

            normalized.append(
                {
                    "type": op_type,
                    "qty": int(qty),
                    "sides": sides,
                    "side": sides[0] if sides else "",
                    "double_sided": bool(double_sided),
                    "ref": ref,
                    "ref_label": ref_label,
                    "thread": thread,
                    "ref_dia_in": float(dia_in) if dia_in is not None else None,
                    "depth_in": float(depth_in) if depth_in is not None else None,
                    "thru": bool(_truthy_flag(entry.get("thru"))),
                    "source": entry.get("source"),
                    "claimed_by_tap": bool(op.get("claimed_by_tap")),
                    "pilot_for_thread": op.get("pilot_for_thread"),
                }
            )

    return normalized


def aggregate_ops(
    rows: list[dict[str, Any]],
    ops_entries: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    legacy_summary = _aggregate_ops_legacy(rows)
    normalized_ops = _normalize_ops_entries(ops_entries)
    if not normalized_ops:
        return legacy_summary

    totals: defaultdict[str, int] = defaultdict(int)
    group_totals: defaultdict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    detail: list[dict[str, Any]] = []
    drill_claim_bins: Counter[float] = Counter()
    tap_claim_bins: Counter[float] = Counter()
    drill_group_refs: dict[float, list[dict[str, Any]]] = defaultdict(list)
    drill_detail_refs: dict[float, list[dict[str, Any]]] = defaultdict(list)
    claimed_pilot_diams: list[float] = []

    rows_simple = list(legacy_summary.get("rows") or [])
    built_rows = int(legacy_summary.get("built_rows") or _count_ops_card_rows(rows_simple))

    def _group_entry(
        type_key: str,
        ref_key: str,
        side_key: str,
        *,
        ref_label: str,
        ref_text: str,
        diameter_in: float | None,
    ) -> dict[str, Any]:
        type_bucket = group_totals.setdefault(type_key, {})
        ref_bucket = type_bucket.setdefault(ref_key, {})
        entry = ref_bucket.get(side_key)
        if entry is None:
            entry = {
                "type": type_key,
                "ref": ref_text,
                "ref_label": ref_label,
                "diameter_in": float(diameter_in) if diameter_in is not None else None,
                "qty": 0,
                "sources": [],
                "_depths": [],
            }
            ref_bucket[side_key] = entry
        return entry

    for op in normalized_ops:
        op_type = op["type"]
        qty = int(op.get("qty") or 0)
        if qty <= 0:
            continue
        sides = list(op.get("sides") or []) or ["FRONT"]
        ref_dia = _coerce_float_or_none(op.get("ref_dia_in"))
        ref_label = str(op.get("ref_label") or op.get("ref") or "").strip()
        if ref_dia is not None:
            ref_key = f"{ref_dia:.4f}"
        elif op.get("thread"):
            ref_key = str(op.get("thread")).strip()
        elif ref_label:
            ref_key = ref_label
        else:
            ref_key = op_type

        detail_entry = {
            "type": op_type,
            "qty": qty,
            "sides": sides,
            "side": sides[0] if sides else "",
            "double_sided": bool(op.get("double_sided")),
            "ref": str(op.get("ref") or ""),
            "ref_label": ref_label,
            "ref_dia_in": float(ref_dia) if ref_dia is not None else None,
            "depth_in": _coerce_float_or_none(op.get("depth_in")),
            "thread": str(op.get("thread") or op.get("tap") or ""),
            "thru": bool(op.get("thru")),
            "source": op.get("source"),
        }
        pilot_flag = bool(op.get("claimed_by_tap") or op.get("pilot_for_thread"))
        dia_key: float | None = None
        if ref_dia is not None and math.isfinite(ref_dia):
            dia_key = round(float(ref_dia), 4)

        if op.get("claimed_by_tap"):
            detail_entry["claimed_by_tap"] = True
        if op.get("pilot_for_thread"):
            detail_entry["pilot_for_thread"] = op.get("pilot_for_thread")
        detail.append(detail_entry)

        for side_key in sides:
            side_norm = "BACK" if side_key.upper() == "BACK" else "FRONT"
            bucket = _group_entry(
                op_type,
                ref_key,
                side_norm,
                ref_label=ref_label,
                ref_text=str(op.get("ref") or ""),
                diameter_in=ref_dia,
            )
            bucket["qty"] = int(bucket.get("qty", 0)) + qty
            depth_val = _coerce_float_or_none(op.get("depth_in"))
            if depth_val is not None:
                bucket.setdefault("_depths", []).append(float(depth_val))
            source_val = op.get("source")
            if source_val:
                try:
                    source_text = str(source_val)
                except Exception:
                    source_text = None
                if source_text and source_text not in bucket["sources"]:
                    bucket["sources"].append(source_text)

            if op_type == "tap":
                totals[f"tap_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            elif op_type == "cbore":
                totals[f"cbore_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            elif op_type == "csk":
                totals[f"csk_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            elif op_type == "spot":
                totals[f"spot_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            if op_type == "drill" and pilot_flag and dia_key is not None:
                if bucket not in drill_group_refs.setdefault(dia_key, []):
                    drill_group_refs[dia_key].append(bucket)
        if op_type == "drill":
            totals["drill"] += qty
            if pilot_flag and dia_key is not None:
                drill_claim_bins[dia_key] += qty
                drill_detail_refs[dia_key].append(detail_entry)
        elif op_type == "jig_grind":
            totals["jig_grind"] += qty
        if op_type == "tap" and dia_key is not None and dia_key > 0:
            tap_claim_bins[dia_key] += qty

    totals["tap_total"] = totals.get("tap_front", 0) + totals.get("tap_back", 0)
    totals["cbore_total"] = totals.get("cbore_front", 0) + totals.get("cbore_back", 0)
    totals["csk_total"] = totals.get("csk_front", 0) + totals.get("csk_back", 0)
    totals["spot_total"] = totals.get("spot_front", 0) + totals.get("spot_back", 0)

    drill_subtracted_total = 0
    processed_dia_keys: set[float] = set()
    for dia_key, available in drill_claim_bins.items():
        tap_claim_qty = tap_claim_bins.get(dia_key, 0)
        subtract = min(available, tap_claim_qty) if tap_claim_qty else available
        if subtract <= 0:
            processed_dia_keys.add(dia_key)
            continue
        drill_subtracted_total += subtract
        claimed_pilot_diams.extend([float(dia_key)] * subtract)
        remaining = subtract
        for bucket in drill_group_refs.get(dia_key, []):
            qty_val = int(_coerce_float_or_none(bucket.get("qty")) or 0)
            if qty_val <= 0:
                continue
            take = min(qty_val, remaining)
            bucket["qty"] = qty_val - take
            remaining -= take
            if remaining <= 0:
                break
        remaining_detail = subtract
        for entry in drill_detail_refs.get(dia_key, []):
            entry_qty = int(_coerce_float_or_none(entry.get("qty")) or 0)
            if entry_qty <= 0:
                continue
            take = min(entry_qty, remaining_detail)
            entry["qty"] = entry_qty - take
            remaining_detail -= take
            if remaining_detail <= 0:
                break
        processed_dia_keys.add(dia_key)
    for dia_key, claim_qty in tap_claim_bins.items():
        if dia_key in processed_dia_keys:
            continue
        available = drill_claim_bins.get(dia_key, 0)
        subtract = min(available, claim_qty)
        if subtract <= 0:
            continue
        drill_subtracted_total += subtract
        claimed_pilot_diams.extend([float(dia_key)] * subtract)
        remaining = subtract
        for bucket in drill_group_refs.get(dia_key, []):
            qty_val = int(_coerce_float_or_none(bucket.get("qty")) or 0)
            if qty_val <= 0:
                continue
            take = min(qty_val, remaining)
            bucket["qty"] = qty_val - take
            remaining -= take
            if remaining <= 0:
                break
        remaining_detail = subtract
        for entry in drill_detail_refs.get(dia_key, []):
            entry_qty = int(_coerce_float_or_none(entry.get("qty")) or 0)
            if entry_qty <= 0:
                continue
            take = min(entry_qty, remaining_detail)
            entry["qty"] = entry_qty - take
            remaining_detail -= take
            if remaining_detail <= 0:
                break

    if drill_subtracted_total:
        totals["drill"] = max(0, totals.get("drill", 0) - drill_subtracted_total)
        detail = [
            entry
            for entry in detail
            if not (entry.get("type") == "drill" and int(_coerce_float_or_none(entry.get("qty")) or 0) <= 0)
        ]

    actions_total = (
        totals.get("drill", 0)
        + totals.get("tap_front", 0)
        + totals.get("tap_back", 0)
        + totals.get("cbore_front", 0)
        + totals.get("cbore_back", 0)
        + totals.get("csk_front", 0)
        + totals.get("csk_back", 0)
        + totals.get("spot_front", 0)
        + totals.get("spot_back", 0)
        + totals.get("jig_grind", 0)
    )

    back_ops_total = (
        totals.get("cbore_back", 0)
        + totals.get("csk_back", 0)
        + totals.get("tap_back", 0)
        + totals.get("spot_back", 0)
    )

    grouped_final: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for type_key, ref_map in group_totals.items():
        grouped_final[type_key] = {}
        for ref_key, side_map in ref_map.items():
            grouped_final[type_key][ref_key] = {}
            for side_key, payload in side_map.items():
                qty_val = int(_coerce_float_or_none(payload.get("qty")) or 0)
                if qty_val <= 0:
                    continue
                depths = payload.pop("_depths", [])
                depth_vals = [
                    _coerce_float_or_none(val)
                    for val in depths
                    if _coerce_float_or_none(val) is not None
                ]
                if depth_vals:
                    depth_clean = [float(val) for val in depth_vals if val is not None]
                    if depth_clean:
                        payload["depth_in_avg"] = sum(depth_clean) / len(depth_clean)
                        payload["depth_in_max"] = max(depth_clean)
                        payload["depth_in_min"] = min(depth_clean)
                if not payload.get("sources"):
                    payload.pop("sources", None)
                grouped_final[type_key][ref_key][side_key] = payload

    summary = {
        "totals": {key: int(totals.get(key, 0)) for key in (
            "drill",
            "tap_front",
            "tap_back",
            "tap_total",
            "cbore_front",
            "cbore_back",
            "cbore_total",
            "csk_front",
            "csk_back",
            "csk_total",
            "spot_front",
            "spot_back",
            "spot_total",
            "jig_grind",
        )},
        "rows": rows_simple,
        "rows_detail": detail,
        "actions_total": int(actions_total),
        "back_ops_total": int(back_ops_total),
        "flip_required": bool(back_ops_total > 0),
        "built_rows": int(built_rows),
        "group_totals": grouped_final,
    }

    if claimed_pilot_diams:
        summary["claims"] = {"claimed_pilot_diams": [float(val) for val in claimed_pilot_diams]}

    if _parser_rules_v2_enabled():
        legacy_totals = (legacy_summary or {}).get("totals") or {}
        try:
            logging.info(
                "[counts] legacy=%s new=%s",
                {k: int(legacy_totals.get(k, 0)) for k in sorted(legacy_totals.keys())},
                summary["totals"],
            )
        except Exception:
            pass

    return summary


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

    ops_entries: list[dict[str, Any]] = []
    for row in clean_rows:
        try:
            qty_val = int(row.get("qty") or 0)
        except Exception:
            qty_val = 0
        ref_text = str(row.get("ref") or "")
        desc_text = str(row.get("desc") or "")
        line_parts = []
        if qty_val > 0:
            line_parts.append(f"QTY {qty_val}")
        if ref_text:
            line_parts.append(ref_text)
        if desc_text:
            line_parts.append(desc_text)
        line_text = " ".join(line_parts).strip()
        entry = _parse_hole_line(line_text, 1.0, source="TEXT_TABLE") if line_text else None
        if entry:
            if qty_val > 0:
                entry["qty"] = qty_val
            if row.get("diameter_in") is not None:
                entry["ref_dia_in"] = row.get("diameter_in")
            entry.setdefault("ref", ref_text)
            ops_entries.append(entry)

    ops_summary = aggregate_ops(clean_rows, ops_entries=ops_entries) if clean_rows else {}

    result = {
        "hole_count": total,
        "hole_diam_families_in": families,
        "rows": clean_rows,
        "provenance_holes": "HOLE TABLE (text)",
    }
    if ops_summary:
        result["ops_summary"] = ops_summary
        claims_payload = ops_summary.get("claims") if isinstance(ops_summary, dict) else None
        if isinstance(claims_payload, dict) and claims_payload:
            result["ops_claims"] = dict(claims_payload)
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

            mqty = re.search(r"(?<!\d)(\d+)(?!\d)", qty_text) or re.search(r"(\d+)\s*[x×]", qty_text)
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
            ops_entries: list[dict[str, Any]] = []
            for row in rows_norm:
                try:
                    qty_val = int(row.get("qty") or 0)
                except Exception:
                    qty_val = 0
                ref_val = str(row.get("ref") or "")
                desc_val = str(row.get("desc") or "")
                line_parts = []
                if qty_val > 0:
                    line_parts.append(f"QTY {qty_val}")
                if ref_val:
                    line_parts.append(ref_val)
                if desc_val:
                    line_parts.append(desc_val)
                line_text = " ".join(line_parts).strip()
                entry = _parse_hole_line(line_text, 1.0, source="ACAD_TABLE") if line_text else None
                if entry:
                    if qty_val > 0:
                        entry["qty"] = qty_val
                    if row.get("diameter_in") is not None:
                        entry["ref_dia_in"] = row.get("diameter_in")
                    entry.setdefault("ref", ref_val)
                    ops_entries.append(entry)
            ops_summary = (
                aggregate_ops(rows_norm, ops_entries=ops_entries) if rows_norm else {}
            )
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
                claims_payload = ops_summary.get("claims") if isinstance(ops_summary, dict) else None
                if isinstance(claims_payload, dict) and claims_payload:
                    result["ops_claims"] = dict(claims_payload)
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
            harvested_tokens = text_harvest_fn(doc)
        except Exception:
            harvested_tokens = None
        else:
            if isinstance(harvested_tokens, (str, bytes)):
                tokens_parts.append(str(harvested_tokens))
            elif isinstance(harvested_tokens, Iterable):
                tokens_parts.extend(
                    str(part)
                    for part in harvested_tokens
                    if part not in (None, "")
                )
    for line in table_lines:
        tokens_parts.append(line)
    if isinstance(leaders, (str, bytes)):
        tokens_parts.append(str(leaders))
    elif isinstance(leaders, Iterable):
        tokens_parts.extend(str(item) for item in leaders if item)
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
    hole_cloud_points = filtered_circles(doc, float(to_in))
    hole_bbox_in: dict[str, Any] | None = None
    if hole_cloud_points:
        scale = float(to_in)
        xs_in = [float(x) * scale for x, _, _, _ in hole_cloud_points]
        ys_in = [float(y) * scale for _, y, _, _ in hole_cloud_points]
        if xs_in and ys_in:
            xmin = min(xs_in)
            xmax = max(xs_in)
            ymin = min(ys_in)
            ymax = max(ys_in)
            span_w = max(0.0, xmax - xmin)
            span_h = max(0.0, ymax - ymin)
            margin_each = _hole_margin_inches(span_w, span_h)
            width_with_margin = span_w + 2.0 * margin_each
            height_with_margin = span_h + 2.0 * margin_each
            hole_bbox_in = {
                "xmin": round(xmin, 4),
                "xmax": round(xmax, 4),
                "ymin": round(ymin, 4),
                "ymax": round(ymax, 4),
                "span_w": round(span_w, 4),
                "span_h": round(span_h, 4),
                "margin_each_in": round(margin_each, 3),
                "w": round(width_with_margin, 3),
                "h": round(height_with_margin, 3),
                "source": "hole_cloud",
            }
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
    derived_entries: dict[str, Any] = {}
    if hole_bbox_in:
        geo["hole_cloud_bbox_in"] = hole_bbox_in
        derived_entries["hole_cloud_bbox_in"] = hole_bbox_in

    thickness_hint = _coerce_positive_float(thickness_guess)
    if stock_plan:
        need_len_in = _coerce_positive_float(stock_plan.get("need_len_in"))
        need_wid_in = _coerce_positive_float(stock_plan.get("need_wid_in"))
        need_thk_in = _coerce_positive_float(
            stock_plan.get("need_thk_in") or stock_plan.get("stock_thk_in")
        )
        if need_thk_in:
            thickness_hint = thickness_hint or float(need_thk_in)
        if need_len_in and need_wid_in:
            dims_sorted = sorted(
                [float(need_len_in), float(need_wid_in)], reverse=True
            )
            blank_dict: dict[str, Any] = {
                "h": round(dims_sorted[0], 3),
                "w": round(dims_sorted[1], 3),
            }
            if thickness_hint:
                blank_dict["t"] = round(float(thickness_hint), 3)
            geo["required_blank_in"] = blank_dict
            derived_entries["required_blank_in"] = blank_dict

    if "required_blank_in" not in geo and hole_bbox_in:
        hole_w = _coerce_positive_float(hole_bbox_in.get("w"))
        hole_h = _coerce_positive_float(hole_bbox_in.get("h"))
        if hole_w and hole_h:
            dims_sorted = sorted([float(hole_h), float(hole_w)], reverse=True)
            blank_dict = {
                "h": round(dims_sorted[0], 3),
                "w": round(dims_sorted[1], 3),
            }
            if thickness_hint:
                blank_dict["t"] = round(float(thickness_hint), 3)
            geo["required_blank_in"] = blank_dict
            derived_entries.setdefault("required_blank_in", blank_dict)

    len_hint = _coerce_positive_float(geo.get("plate_len_in"))
    wid_hint = _coerce_positive_float(geo.get("plate_wid_in"))
    if (len_hint is None or wid_hint is None) and hole_bbox_in:
        span_h = _coerce_positive_float(hole_bbox_in.get("span_h"))
        span_w = _coerce_positive_float(hole_bbox_in.get("span_w"))
        if len_hint is None and span_h:
            len_hint = float(span_h)
        if wid_hint is None and span_w:
            wid_hint = float(span_w)
    if len_hint and wid_hint:
        dims_sorted = sorted([float(len_hint), float(wid_hint)], reverse=True)
        bbox_dict: dict[str, Any] = {
            "h": round(dims_sorted[0], 3),
            "w": round(dims_sorted[1], 3),
        }
        if thickness_hint:
            bbox_dict["t"] = round(float(thickness_hint), 3)
        geo.setdefault("bbox_in", bbox_dict)
        derived_entries.setdefault("bbox_in", bbox_dict)

    if derived_entries:
        existing_derived = geo.get("derived")
        if isinstance(existing_derived, _MappingABC):
            merged = dict(existing_derived)
            merged.update(derived_entries)
            geo["derived"] = merged
        else:
            geo["derived"] = derived_entries
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

def extract_2d_features_from_dxf_or_dwg(path: str | Path) -> dict[str, Any]:
    ezdxf_mod = geometry.require_ezdxf()

    # --- load doc ---
    dxf_text_path: str | None = None
    doc: Drawing | None = None
    path_str = str(path)
    lower_path = path_str.lower()
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
            doc = cast(Drawing, readfile(path_str))
        else:
            dxf_path = geometry.convert_dwg_to_dxf(path_str, out_ver="ACAD2018")
            dxf_text_path = dxf_path
            doc = cast(Drawing, readfile(dxf_path))
    else:
        doc = cast(Drawing, readfile(path_str))
        dxf_text_path = path_str

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
    _chart_all_lines: list[str] = []
    _chart_seen: set[str] = set()

    def _append_chart_line(value: Any) -> None:
        if not isinstance(value, str):
            return
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            return
        if normalized in _chart_seen:
            return
        _chart_seen.add(normalized)
        _chart_all_lines.append(normalized)

    if extractor and dxf_text_path:
        try:
            raw_lines = extractor(dxf_text_path, include_tables=True)
        except TypeError:
            try:
                raw_lines = extractor(dxf_text_path)
            except Exception:
                raw_lines = []
        except Exception:
            raw_lines = []
        if raw_lines:
            for ln in raw_lines:
                _append_chart_line(ln)

    # Always also read ACAD_TABLE/MTEXT via ezdxf and merge/dedupe.
    _lines_from_doc = _extract_text_lines_from_ezdxf_doc(doc, include_tables=True) or []
    for ln in _lines_from_doc:
        _append_chart_line(ln)

    if _DXF_HOLE_TOKENS is not None:
        chart_lines = [ln for ln in _chart_all_lines if _DXF_HOLE_TOKENS.search(ln)]
    else:
        chart_lines = list(_chart_all_lines)
    geo["debug_chart_counts"] = {
        "all_text": len(_chart_all_lines),
        "chart_lines": len(chart_lines),
    }

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
    parser_kwargs: dict[str, Any] = {}
    if PARSER_RULES_V2_ENABLED:
        parser_kwargs = {
            "rules_v2": True,
            "block_thickness_in": thickness_guess,
        }
    if chart_lines and parser:
        try:
            if parser_kwargs:
                hole_rows = parser(chart_lines, **parser_kwargs)
            else:
                hole_rows = parser(chart_lines)
        except TypeError:
            hole_rows = parser(chart_lines)
        except Exception:
            hole_rows = []
        if hole_rows:
            chart_ops = hole_rows_to_ops(hole_rows)
            chart_source = "dxf_text_regex"
            chart_reconcile = summarize_hole_chart_agreement(
                entity_holes_mm, chart_ops
            )

        try:
            ops_rows = update_geo_ops_summary_from_hole_rows(
                geo,
                hole_rows=hole_rows,
                chart_lines=chart_lines,
                chart_source=chart_source,
                chart_summary=chart_summary,
                apply_built_rows=_apply_built_rows,
            )
        except Exception:
            ops_rows = []
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
def _extract_text_lines_from_ezdxf_doc(
    doc: Any, *, include_tables: bool = True
) -> list[str]:
    """Harvest TEXT / MTEXT strings from an ezdxf Drawing."""

    if doc is None:
        return []

    if _harvest_dxf_text_lines is not None:
        try:
            return _harvest_dxf_text_lines(doc, include_tables=include_tables)
        except Exception:
            pass

    try:
        msp = doc.modelspace()
    except Exception:
        return []

    lines: list[str] = []

    def _append(text: Any) -> None:
        if not isinstance(text, str):
            text = str(text or "")
        cleaned = re.sub(r"\s+", " ", text).strip()
        if cleaned:
            lines.append(cleaned)

    def _harvest_insert(entity: Any) -> None:
        try:
            virtuals = entity.virtual_entities()
        except Exception:
            return
        for child in virtuals:
            try:
                ctype = child.dxftype()
            except Exception:
                continue
            if ctype in ("TEXT", "MTEXT"):
                _append(_extract_entity_text(child))
            elif ctype == "INSERT":
                _harvest_insert(child)

    def _harvest_entities(entities: Iterable[Any]) -> None:
        for entity in entities:
            try:
                kind = entity.dxftype()
            except Exception:
                continue
            if kind in ("TEXT", "MTEXT"):
                _append(_extract_entity_text(entity))
            elif kind == "INSERT":
                _harvest_insert(entity)

    _harvest_entities(msp)

    if include_tables:
        def _iter_table_strings(table: Any) -> Iterator[str]:
            try:
                cells = getattr(table, "cells")
            except Exception:
                cells = None

            if cells is not None:
                for cell in cells:  # type: ignore[assignment]
                    try:
                        text = getattr(cell, "text")
                    except Exception:
                        text = ""
                    if callable(text):
                        try:
                            text = text()
                        except Exception:
                            text = ""
                    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
                    if normalized:
                        yield normalized
                return

            try:
                nrows = int(getattr(table, "nrows", 0))
                ncols = int(getattr(table, "ncols", 0))
            except Exception:
                nrows = ncols = 0

            for row in range(max(nrows, 0)):
                for col in range(max(ncols, 0)):
                    try:
                        cell = table.cell(row, col)  # type: ignore[call-arg]
                    except Exception:
                        continue
                    try:
                        text = getattr(cell, "text")
                    except Exception:
                        text = ""
                    if callable(text):
                        try:
                            text = text()
                        except Exception:
                            text = ""
                    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
                    if normalized:
                        yield normalized

        for table in getattr(msp, "query", lambda *_: [])("TABLE"):  # type: ignore[misc]
            for text in _iter_table_strings(table):
                _append(text)

        try:
            layouts = list(doc.layouts.names())
        except Exception:
            layouts = []
        for name in layouts:
            if isinstance(name, str) and name.lower() == "model":
                continue
            try:
                layout = doc.layouts.get(name)
            except Exception:
                continue
            if layout is None:
                continue
            space = getattr(layout, "entity_space", layout)
            try:
                _harvest_entities(space)
            except Exception:
                pass
            for table in getattr(layout, "query", lambda *_: [])("TABLE"):  # type: ignore[misc]
                for text in _iter_table_strings(table):
                    _append(text)

    return lines


_MM_TO_IN = 1.0 / 25.4


def hole_rows_to_ops(rows: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Flatten parsed HoleRow objects into estimator-friendly operations."""

    ops: list[dict[str, Any]] = []
    if not rows:
        return ops

    def _mm_to_in(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value) * _MM_TO_IN
        except Exception:
            return None

    def _normalize_side(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if "back" in text:
            return "back"
        if "front" in text:
            return "front"
        return None

    def _feature_sides(feature: Mapping[str, Any]) -> list[str]:
        sides: list[str] = []
        sides_raw = feature.get("sides")
        if isinstance(sides_raw, (list, tuple, set)):
            for entry in sides_raw:
                norm = _normalize_side(entry)
                if norm and norm not in sides:
                    sides.append(norm)
        if feature.get("double_sided"):
            for candidate in ("front", "back"):
                if candidate not in sides:
                    sides.append(candidate)
        if feature.get("from_back") and "back" not in sides:
            sides.append("back")
        if not sides:
            side = feature.get("side") or feature.get("from_face")
            norm = _normalize_side(side)
            if norm:
                sides.append(norm)
        if not sides:
            sides.append("front")
        return sides

    def _resolve_depth(feature: Mapping[str, Any], thickness_in: float | None) -> float | None:
        depth_in = feature.get("depth_in")
        if isinstance(depth_in, (int, float)):
            try:
                return float(depth_in)
            except Exception:
                depth_in = None
        depth_mm = feature.get("depth_mm")
        depth = _mm_to_in(depth_mm) if depth_mm is not None else None
        if depth is not None:
            return depth
        raw_depth = feature.get("depth")
        if isinstance(raw_depth, (int, float)):
            return float(raw_depth)
        if feature.get("thru") and thickness_in is not None:
            try:
                return float(thickness_in) + 0.05
            except Exception:
                return None
        return None

    def _resolve_ref_dia(feature: Mapping[str, Any]) -> float | None:
        for key in ("ref_dia", "ref_dia_in", "dia_in", "major_in"):
            val = feature.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        for key in ("dia_mm", "major_mm"):
            val = feature.get(key)
            converted = _mm_to_in(val) if val is not None else None
            if converted is not None:
                return converted
        return None

    def _resolve_thread(feature: Mapping[str, Any]) -> str | None:
        thread = feature.get("thread")
        if not isinstance(thread, str):
            return None
        cleaned = thread.strip()
        return cleaned or None

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
        thickness_in = getattr(row, "block_thickness_in", None)
        if thickness_in is None:
            thickness_in = getattr(row, "thickness_in", None)
        for feature in features:
            if not isinstance(feature, dict):
                continue
            feature_type = str(feature.get("type") or "").lower()
            ref_dia = _resolve_ref_dia(feature)
            depth_in = _resolve_depth(feature, thickness_in)
            thread = _resolve_thread(feature)
            for side in _feature_sides(feature):
                qty = feature.get("qty")
                if not isinstance(qty, int):
                    qty = qty_val
                op = {
                    "type": feature_type,
                    "ref": ref_val,
                    "qty": qty,
                    "ref_dia": ref_dia,
                    "depth_in": depth_in,
                    "thread": thread,
                    "side": side,
                }
                ops.append(op)
                if PARSER_RULES_V2_ENABLED:
                    logging.info(
                        "[rules] op %s/%s side=%s qty=%s depth=%s thread=%s",
                        ref_val,
                        feature_type or "?",
                        side,
                        qty,
                        f"{depth_in:.3f}" if isinstance(depth_in, (int, float)) else "-",
                        thread or "-",
                    )
    return ops


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
        geometry_service: GeometryServiceType | None = None,
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
            extract_pdf_vector_fn=extract_2d_features_from_pdf_vector,
            extract_dxf_or_dwg_fn=extract_2d_features_from_dxf_or_dwg,
            occ_feature_fn=typing.cast(
                Callable[[str | Path], Any], geometry.extract_features_with_occ
            ),
            stl_enricher=geometry.enrich_geo_stl,
            step_reader=typing.cast(Callable[[str | Path], Any], geometry.read_step_shape),
            cad_reader=typing.cast(Callable[[str | Path], Any], geometry.read_cad_any),
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
        self._llm_client_cache: LLMClientType | None = None
        self.settings_path = (
            getattr(self.configuration, "settings_path", None)
            or default_app_settings_json()
        )

        self.settings = self.llm_services.load_settings(self.settings_path)
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
                        core_df_t = typing.cast(PandasDataFrame, core_df)
                        full_df_t = typing.cast(PandasDataFrame, full_df)
                        self._refresh_variables_cache(core_df_t, full_df_t)
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
        self._sync_llm_thread_limit(persist=False)

        # Create a Menu Bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Overrides...", command=self.load_overrides)
        file_menu.add_command(label="Save Overrides...", command=self.save_overrides)
        file_menu.add_separator()
        file_menu.add_command(
            label="Import Quote Session...",
            command=lambda: session_io.import_quote_session(self),
        )
        file_menu.add_command(
            label="Export Quote Session...",
            command=lambda: session_io.export_quote_session(self),
        )
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

    def get_llm_client(self, model_path: str | None = None) -> LLMClientType | None:
        path = (model_path or "").strip()
        if not path and hasattr(self, "llm_model_path"):
            path = (self.llm_model_path.get().strip() if self.llm_model_path.get() else "")
        if not path:
            path = os.environ.get("QWEN_GGUF_PATH", "")
        path = path.strip()
        if not path:
            return None
        self._sync_llm_thread_limit(persist=False)
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

    def _get_last_variables_path(self) -> str:
        if isinstance(self.settings, dict):
            return str(self.settings.get("last_variables_path", "") or "").strip()
        return ""

    def _set_last_variables_path(self, path: str | None) -> None:
        if not isinstance(self.settings, dict):
            self.settings = {}
        value = str(path) if path else ""
        self.settings["last_variables_path"] = value
        self.llm_services.save_settings(self.settings_path, self.settings)

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

    def _sync_llm_thread_limit(self, *, persist: bool) -> int | None:
        """Apply the current thread limit via ``LLMServices`` and manage cache state."""

        limit = self._current_llm_thread_limit()
        prior = getattr(self, "_llm_thread_limit_applied", None)

        updated_settings = self.llm_services.apply_thread_limit_env(
            limit,
            settings=self.settings,
            persist=persist,
            settings_path=self.settings_path if persist else None,
        )
        if isinstance(updated_settings, dict):
            self.settings = updated_settings

        if limit != prior:
            self._llm_thread_limit_applied = limit
            self._invalidate_llm_client_cache()

        return limit

    def _on_llm_thread_limit_changed(self, *_: object) -> None:
        self._sync_llm_thread_limit(persist=True)

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
        self.llm_services.save_settings(self.settings_path, self.settings)
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
        self.llm_services.save_settings(self.settings_path, self.settings)
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
            limit = self._sync_llm_thread_limit(persist=False)
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
                limit = self._sync_llm_thread_limit(persist=False)
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
            return geometry_upsert_var_row(dataframe, item, value, dtype=dtype)

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
            qty_column = typing.cast(
                PandasSeries,
                df.loc[qty_mask, "Example Values / Options"],
            )
            qty_raw = qty_column.iloc[0]
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
            self.llm_services.save_settings(self.settings_path, self.settings)

        try:
            if self.vars_df is not None:
                self._populate_editor_tab(typing.cast(PandasDataFrame, self.vars_df))
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

                vars_df_for_editor = typing.cast(PandasDataFrame, self.vars_df)
                self._populate_editor_tab(vars_df_for_editor)
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

        geo_dict: dict[str, Any] = geo if isinstance(geo, dict) else dict(geo or {})

        # Merge GEO rows
        try:
            for k, v in geo_dict.items():
                self.vars_df = geometry_upsert_var_row(self.vars_df, k, v, dtype="number")
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
        geo_for_llm = geo_dict
        est_raw = infer_hours_and_overrides_from_geo(
            geo_for_llm,
            params=self.params,
            rates=self.rates,
            client=client,
        )
        est = clamp_llm_hours(est_raw, geo_for_llm, params=self.params)
        vars_df_before_llm = self.vars_df
        vars_df_after_llm = apply_llm_hours_to_variables(
            vars_df_before_llm, est, allow_overwrite_nonzero=True, log=decision_log
        )
        if vars_df_after_llm is None:
            vars_df_after_llm = coerce_or_make_vars_df(vars_df_before_llm)
        self.vars_df = vars_df_after_llm

        self.geo = geo_for_llm
        self.geo_context = dict(geo_for_llm)
        self._log_geo(geo_for_llm)

        vars_df_for_editor = self.vars_df
        if vars_df_for_editor is None:  # pragma: no cover - defensive type check
            raise RuntimeError("Variables dataframe was not initialized")
        self._populate_editor_tab(typing.cast(PandasDataFrame, vars_df_for_editor))
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
                try:
                    import datetime as _dt
                    append_debug_log(
                        "",
                        f"[{_dt.datetime.now().isoformat()}] compute_quote_from_df returned",
                        f"top_keys={list(res.keys())[:20] if isinstance(res, dict) else type(res)}",
                    )
                except Exception:
                    pass
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
                # Persist reports for diagnosis even if UI widgets fail to update
                try:
                    with open("latest_quote_simplified.txt", "w", encoding="utf-8") as f:
                        f.write(simplified_report or "")
                    with open("latest_quote_full.txt", "w", encoding="utf-8") as f:
                        f.write(full_report or "")
                except Exception:
                    pass
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
                try:
                    with open("latest_quote_simplified.txt", "w", encoding="utf-8") as f:
                        f.write(simplified_report or "")
                    with open("latest_quote_full.txt", "w", encoding="utf-8") as f:
                        f.write(full_report or "")
                except Exception:
                    pass
            except Exception as e:
                # Catch any other errors from render_quote and still produce a fallback
                import traceback as _tb
                err_text = f"Quote rendering error (Exception): {e}"
                append_debug_log(
                    "",
                    "[render_quote] Exception while rendering output",
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
                try:
                    with open("latest_quote_simplified.txt", "w", encoding="utf-8") as f:
                        f.write(simplified_report or "")
                    with open("latest_quote_full.txt", "w", encoding="utf-8") as f:
                        f.write(full_report or "")
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

    if typing.TYPE_CHECKING:
        from cad_quoter_pkg.src.cad_quoter.pricing import (
            PricingEngine as _PricingEngine,
            create_default_registry as _create_default_registry,
        )
    else:  # pragma: no cover - executed at runtime
        from cad_quoter.pricing import (
            PricingEngine as _PricingEngine,
            create_default_registry as _create_default_registry,
        )

    return _cli_main(
        argv,
        app_cls=App,
        pricing_engine_cls=_PricingEngine,
        pricing_registry_factory=_create_default_registry,
        app_env=APP_ENV,
        env_setter=lambda env: globals().__setitem__("APP_ENV", env),
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())

# Emit chart-debug key lines at most once globally per run
_PRINTED_CHART_DEBUG_KEYS = False



