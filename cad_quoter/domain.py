"""Domain helpers shared across the CAD Quoter application."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    import pandas as pd

__all__ = [
    "QuoteState",
    "as_float_or_none",
    "build_suggest_payload",
    "ensure_scrap_pct",
    "match_items_contains",
    "normalize_lookup_key",
]


_CAPTURING_GROUP_RE = re.compile(r"\((?!\?[:P<!=])")


def normalize_lookup_key(value: str) -> str:
    """Normalise user input for dictionary lookups."""

    cleaned = re.sub(r"[^0-9a-z]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def ensure_scrap_pct(val) -> float:
    """Coerce UI/LLM scrap into a sane fraction in [0, 0.25]."""

    try:
        x = float(val)
    except Exception:
        return 0.0
    if x > 1.0:
        x = x / 100.0
    if not (x >= 0.0 and math.isfinite(x)):
        return 0.0
    return min(0.25, max(0.0, x))


def match_items_contains(items: "pd.Series", pattern: str) -> "pd.Series":
    """Case-insensitive regex match over items with graceful fallback."""

    def _to_noncapturing(expr: str) -> str:
        return _CAPTURING_GROUP_RE.sub("(?:", expr)

    pat = _to_noncapturing(pattern)
    try:
        return items.str.contains(pat, case=False, regex=True, na=False)
    except Exception:
        return items.str.contains(re.escape(pattern), case=False, regex=True, na=False)


def as_float_or_none(value: Any) -> float | None:
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

    hole_count_val = as_float_or_none(geo.get("hole_count"))
    if hole_count_val is None:
        hole_count_val = as_float_or_none(derived.get("hole_count"))
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

    tap_minutes_hint = as_float_or_none(derived.get("tap_minutes_hint"))
    cbore_minutes_hint = as_float_or_none(derived.get("cbore_minutes_hint"))
    csk_minutes_hint = as_float_or_none(derived.get("csk_minutes_hint"))
    tap_class_counts = derived.get("tap_class_counts") if isinstance(derived.get("tap_class_counts"), dict) else {}
    tap_details = derived.get("tap_details") if isinstance(derived.get("tap_details"), list) else []
    npt_qty = 0
    try:
        npt_qty = int(derived.get("npt_qty") or 0)
    except Exception:
        npt_qty = 0
    inference_knobs = derived.get("inference_knobs") if isinstance(derived.get("inference_knobs"), dict) else {}
    has_ldr_notes = bool(derived.get("has_ldr_notes"))
    max_hole_depth_in = as_float_or_none(derived.get("max_hole_depth_in"))
    plate_area_in2 = as_float_or_none(derived.get("plate_area_in2"))
    finish_flags_raw = derived.get("finish_flags")
    if isinstance(finish_flags_raw, (list, tuple, set)):
        finish_flags = [
            str(flag).strip() for flag in finish_flags_raw if isinstance(flag, str) and flag.strip()
        ]
    elif isinstance(finish_flags_raw, str) and finish_flags_raw.strip():
        finish_flags = [finish_flags_raw.strip()]
    else:
        finish_flags = []
    has_tight_tol = bool(derived.get("has_tight_tol"))
    stock_guess_val = derived.get("stock_guess")
    stock_guess = (
        str(stock_guess_val).strip()
        if isinstance(stock_guess_val, str) and stock_guess_val.strip()
        else None
    )

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


# Backwards compatibility aliases for legacy imports
_normalize_lookup_key = normalize_lookup_key
_ensure_scrap_pct = ensure_scrap_pct
_match_items_contains = match_items_contains
_as_float_or_none = as_float_or_none
