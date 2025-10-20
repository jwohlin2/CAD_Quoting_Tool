"""Shared helpers for deriving LLM-driven cost overrides."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping as _MappingABC, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.llm import LLMClient, parse_llm_json
from cad_quoter.pass_labels import (
    HARDWARE_PASS_LABEL,
    LEGACY_HARDWARE_PASS_LABEL,
    _HARDWARE_LABEL_ALIASES,
    _canonical_pass_label,
)
from cad_quoter.utils import jdump
from cad_quoter.utils.render_utils import fmt_hours

LLM_MULTIPLIER_MIN = 0.25
LLM_MULTIPLIER_MAX = 4.0
LLM_ADDER_MAX = 8.0


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
def _plate_mass_properties(
    plate_L_in: Any,
    plate_W_in: Any,
    t_in: Any,
    density_g_cc: Any,
    hole_d_mm: Any,
) -> tuple[float | None, float | None]:
    """Return net mass (kg) and removed mass (g) for a plate with optional holes."""

    length_in = _coerce_float_or_none(plate_L_in)
    width_in = _coerce_float_or_none(plate_W_in)
    thickness_in = _coerce_float_or_none(t_in)
    density = _coerce_float_or_none(density_g_cc)

    if (
        density is None
        or density <= 0
        or length_in is None
        or width_in is None
        or thickness_in is None
        or length_in <= 0
        or width_in <= 0
        or thickness_in <= 0
    ):
        return (None, None)

    volume_in3 = float(length_in) * float(width_in) * float(thickness_in)
    plate_volume_cm3 = volume_in3 * 16.387064

    thickness_mm = float(thickness_in) * 25.4
    removed_volume_mm3 = 0.0

    if thickness_mm > 0:
        if isinstance(hole_d_mm, _MappingABC):
            hole_iter = hole_d_mm.values()
        elif isinstance(hole_d_mm, Sequence) and not isinstance(hole_d_mm, (str, bytes)):
            hole_iter = hole_d_mm
        elif hole_d_mm is None:
            hole_iter = ()
        else:
            hole_iter = (hole_d_mm,)

        for raw_d in hole_iter:
            diameter_mm = _coerce_float_or_none(raw_d)
            if diameter_mm is None or diameter_mm <= 0:
                continue
            radius_mm = float(diameter_mm) / 2.0
            removed_volume_mm3 += math.pi * (radius_mm**2) * thickness_mm

    removed_volume_cm3 = removed_volume_mm3 / 1000.0
    removed_mass_g = removed_volume_cm3 * float(density)

    net_volume_cm3 = max(plate_volume_cm3 - removed_volume_cm3, 0.0)
    net_mass_g = net_volume_cm3 * float(density)

    return net_mass_g / 1000.0, removed_mass_g


def _plate_mass_from_dims(
    length_mm: Any,
    width_mm: Any,
    thickness_mm: Any,
    density_g_cc: Any,
    *,
    dims_in: Sequence[Any] | None = None,
    hole_d_mm: Any = (),
) -> tuple[float | None, float | None]:
    """Compute plate mass from dimensions in millimeters."""

    length_mm_val = _coerce_float_or_none(length_mm)
    width_mm_val = _coerce_float_or_none(width_mm)
    thickness_mm_val = _coerce_float_or_none(thickness_mm)

    dims_in_vals: list[float | None] = [None, None, None]
    if isinstance(dims_in, Sequence) and not isinstance(dims_in, (str, bytes)):
        dims_list = list(dims_in)
        for idx in range(min(3, len(dims_list))):
            dims_in_vals[idx] = _coerce_float_or_none(dims_list[idx])

    length_in = dims_in_vals[0]
    width_in = dims_in_vals[1]
    thickness_in = dims_in_vals[2]

    if length_in is None and length_mm_val:
        length_in = float(length_mm_val) / 25.4
    if width_in is None and width_mm_val:
        width_in = float(width_mm_val) / 25.4
    if thickness_in is None and thickness_mm_val:
        thickness_in = float(thickness_mm_val) / 25.4

    return _plate_mass_properties(
        length_in,
        width_in,
        thickness_in,
        density_g_cc,
        hole_d_mm,
    )


def coerce_bounds(bounds: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize LLM bounds into a canonical structure."""

    if bounds is None:
        bounds_map: Mapping[str, Any] = {}
    else:
        bounds_map = bounds

    mult_min = _as_float_or_none(bounds_map.get("mult_min"))
    if mult_min is None:
        mult_min = LLM_MULTIPLIER_MIN
    else:
        mult_min = max(LLM_MULTIPLIER_MIN, float(mult_min))

    mult_max = _as_float_or_none(bounds_map.get("mult_max"))
    if mult_max is None:
        mult_max = LLM_MULTIPLIER_MAX
    else:
        mult_max = min(LLM_MULTIPLIER_MAX, float(mult_max))
    mult_max = max(mult_max, mult_min)

    adder_min = _as_float_or_none(bounds_map.get("adder_min_hr"))
    if adder_min is None:
        adder_min = _as_float_or_none(bounds_map.get("add_hr_min"))
    adder_min = max(0.0, float(adder_min)) if adder_min is not None else 0.0

    adder_max = _as_float_or_none(bounds_map.get("adder_max_hr"))
    add_hr_cap = _as_float_or_none(bounds_map.get("add_hr_max"))
    if adder_max is None and add_hr_cap is not None:
        adder_max = float(add_hr_cap)
    elif adder_max is not None and add_hr_cap is not None:
        adder_max = min(float(adder_max), float(add_hr_cap))
    if adder_max is None:
        adder_max = LLM_ADDER_MAX
    adder_max = max(adder_min, min(LLM_ADDER_MAX, float(adder_max)))

    scrap_min = _as_float_or_none(bounds_map.get("scrap_min"))
    scrap_min = max(0.0, float(scrap_min)) if scrap_min is not None else 0.0

    scrap_max = _as_float_or_none(bounds_map.get("scrap_max"))
    scrap_max = float(scrap_max) if scrap_max is not None else 0.25
    scrap_max = max(scrap_max, scrap_min)

    bucket_caps_raw = bounds_map.get("adder_bucket_max") or bounds_map.get("add_hr_bucket_max")
    bucket_caps: dict[str, float] = {}
    if isinstance(bucket_caps_raw, _MappingABC):
        for key, raw in bucket_caps_raw.items():
            cap_val = _as_float_or_none(raw)
            if cap_val is None:
                continue
            bucket_caps[str(key).lower()] = max(adder_min, min(adder_max, float(cap_val)))

    return {
        "mult_min": mult_min,
        "mult_max": mult_max,
        "adder_min_hr": adder_min,
        "adder_max_hr": adder_max,
        "scrap_min": scrap_min,
        "scrap_max": scrap_max,
        "adder_bucket_max": bucket_caps,
    }


def _default_llm_bounds_dict() -> dict[str, Any]:
    """Return the sanitized default LLM guardrail bounds."""

    return coerce_bounds({})


def get_llm_bound_defaults() -> dict[str, Any]:
    """Return a mutable copy of the default LLM guardrail bounds."""

    return dict(coerce_bounds(LLM_BOUND_DEFAULTS))


LLM_BOUND_DEFAULTS: Mapping[str, Any] = MappingProxyType(_default_llm_bounds_dict())


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
    debug_enabled: bool = False,
    debug_dir: Path | str | None = None,
    llm_client_cls: type[LLMClient] = LLMClient,
) -> tuple[dict, dict]:
    """Ask the local LLM for cost overrides based on CAD features and base costs."""

    debug_path = Path(debug_dir) if debug_dir else None

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
        llm = llm_client_cls(
            model_path,
            debug_enabled=debug_enabled,
            debug_dir=debug_path,
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

    bbox_feature_raw = features.get("bbox_mm")
    bbox_feature = bbox_feature_raw if isinstance(bbox_feature_raw, dict) else {}
    part_dims: list[float] = []
    for key in ("length_mm", "width_mm", "height_mm"):
        val = _as_float(bbox_feature.get(key))
        if val and val > 0:
            part_dims.append(val)
    if thickness_feature and thickness_feature > 0:
        part_dims.append(thickness_feature)
    part_dims_sorted = sorted([d for d in part_dims if d > 0], reverse=True)

    stock_catalog_raw = features.get("stock_catalog")
    stock_catalog = (
        list(stock_catalog_raw)
        if isinstance(stock_catalog_raw, (list, tuple))
        else []
    )
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

    if part_dims_sorted:
        ctx.setdefault("stock_focus", {})
        ctx["stock_focus"]["part_dims_mm"] = part_dims_sorted
        ctx["stock_focus"]["fits_catalog"] = part_fits_catalog
    if not part_fits_catalog:
        ctx.setdefault("stock_focus", {})
        ctx["stock_focus"]["needs_attention"] = True

    stock_plan_context = ctx.setdefault("stock_plan", {})
    stock_plan_context.setdefault("part_mass_est_g", part_mass_est)
    stock_plan_context.setdefault("density_g_cc", density_for_stock)

    signals_ctx = ctx.setdefault("signals", {})
    signals_ctx.setdefault("stock_catalog", stock_catalog)
    signals_ctx.setdefault("part_mass_est_g", part_mass_est)
    signals_ctx.setdefault("density_g_cc", density_for_stock)
    if not part_fits_catalog:
        signals_ctx.setdefault("stock_focus", "no catalog fit")

    ctx["baseline"] = base_costs
    ctx["features"] = features
    ctx.setdefault("bounds", {})

    bounds_ctx = dict(ctx["bounds"])
    llm_bound_defaults = get_llm_bound_defaults()
    bounds_ctx.update(
        {
            "mult_min": llm_bound_defaults["mult_min"],
            "mult_max": llm_bound_defaults["mult_max"],
            "add_hr_min": 0.0,
            "add_hr_max": llm_bound_defaults["adder_max_hr"],
            "scrap_min": 0.0,
            "scrap_max": 0.25,
        }
    )
    ctx["bounds"] = bounds_ctx
    coerced_bounds = coerce_bounds(bounds_ctx)
    mult_min_bound = coerced_bounds["mult_min"]
    mult_max_bound = coerced_bounds["mult_max"]
    adder_min_bound = coerced_bounds["adder_min_hr"]
    adder_max_bound = coerced_bounds["adder_max_hr"]
    scrap_min_bound = coerced_bounds["scrap_min"]
    scrap_max_bound = coerced_bounds["scrap_max"]
    bucket_caps_bound = coerced_bounds.get("adder_bucket_max", {})

    def _adder_limit(name: str | None) -> float:
        if name is None:
            return adder_max_bound
        bucket_cap = bucket_caps_bound.get(str(name).lower())
        if bucket_cap is None:
            return adder_max_bound
        return max(adder_min_bound, float(bucket_cap))

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
                prompt_body = jdump(payload)
            except TypeError:
                prompt_body = jdump(_jsonify(payload), default=None)
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
            " \"final_inspection_hr\":0.1, \"finishing_hr\":0.1, \"suggested_surface_finish\":\"...\","\
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
        clamped = clamp(val, mult_min_bound, mult_max_bound, 1.0)
        container = _ensure_mults_dict()
        norm = str(name).lower()
        prev = container.get(norm)
        if prev is None:
            container[norm] = clamped
            return
        new_val = clamp(prev * clamped, mult_min_bound, mult_max_bound, 1.0)
        if not math.isclose(prev * clamped, new_val, abs_tol=1e-6):
            clamp_notes.append(f"{source} multiplier clipped for {norm}")
        container[norm] = new_val

    def _merge_adder(name: str, value, source: str) -> None:
        val = _as_float(value)
        if val is None:
            return
        norm = str(name).lower()
        limit = _adder_limit(norm)
        clamped = clamp(val, adder_min_bound, limit, adder_min_bound)
        if clamped <= 0:
            return
        container = _ensure_adders_dict()
        prev = float(container.get(norm, 0.0))
        new_val = clamp(prev + clamped, adder_min_bound, limit, adder_min_bound)
        if not math.isclose(prev + clamped, new_val, abs_tol=1e-6):
            clamp_notes.append(
                f"{source} {fmt_hours(prev + clamped)} clipped to {fmt_hours(limit, decimals=1)} for {norm}"
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
        clamped_scrap = clamp(scr, scrap_min_bound, scrap_max_bound, None)
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
            clamped_val = clamp(v, mult_min_bound, mult_max_bound, 1.0)
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
            limit = _adder_limit(k)
            clamped_val = clamp(v, adder_min_bound, limit, adder_min_bound)
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
                    cleaned_group["depth_mm"] = round(depth, 3)
                if isinstance(peck, str) and peck.strip():
                    cleaned_group["strategy"] = peck.strip()[:120]
                clean_notes = _clean_notes_list(notes)
                if clean_notes:
                    cleaned_group["notes"] = clean_notes
                drill_groups_clean.append(cleaned_group)
            if drill_groups_clean:
                out["drilling_groups"] = drill_groups_clean

    stock_plan_raw = parsed.get("stock_recommendation") or parsed.get("stock_plan")
    if isinstance(stock_plan_raw, dict):
        clean_stock_plan: dict[str, Any] = {}
        stock_dims = stock_plan_raw.get("stock_dims") or stock_plan_raw.get("dims_mm")
        if isinstance(stock_dims, (list, tuple)):
            clean_stock_plan["stock_dims_mm"] = [float(_as_float(val) or 0.0) for val in stock_dims[:3]]
        dims_map = stock_plan_raw.get("dims_mm") if isinstance(stock_plan_raw.get("dims_mm"), dict) else {}
        for key in ("length_mm", "width_mm", "height_mm", "thickness_mm"):
            val = _as_float(dims_map.get(key))
            if val and val > 0:
                clean_stock_plan[key] = float(val)

        if stock_plan_raw.get("catalog_match"):
            clean_stock_plan["catalog_match"] = True
            clean_stock_plan["catalog_entry"] = stock_plan_raw.get("catalog_entry")

        mass_entry = stock_plan_raw.get("mass_kg")
        if mass_entry is None:
            mass_entry = stock_plan_raw.get("stock_mass_kg")
        mass_kg = _as_float(mass_entry)
        if mass_kg is None or mass_kg <= 0:
            mass_kg, _ = _plate_mass_from_dims(
                stock_plan_raw.get("length_mm"),
                stock_plan_raw.get("width_mm"),
                stock_plan_raw.get("thickness_mm"),
                density_for_stock,
                dims_in=stock_plan_raw.get("dims_in"),
                hole_d_mm=stock_plan_raw.get("hole_d_mm"),
            )
        if mass_kg and mass_kg > 0:
            clean_stock_plan["mass_kg"] = mass_kg

        plan_scrap = stock_plan_raw.get("scrap_pct")
        if plan_scrap is not None:
            frac = _as_float(plan_scrap)
            scrap_frac = clamp(frac, scrap_min_bound, scrap_max_bound, None)
            if scrap_frac is not None:
                clean_stock_plan["scrap_pct"] = scrap_frac
                if "scrap_pct_override" not in out:
                    out["scrap_pct_override"] = scrap_frac

        plan_notes = _clean_notes_list(stock_plan_raw.get("notes"))
        if plan_notes:
            clean_stock_plan["notes"] = plan_notes

        saw_hr = _as_float(stock_plan_raw.get("sawing_hr") or stock_plan_raw.get("saw_hr"))
        if saw_hr and saw_hr > 0:
            saw_limit = _adder_limit("saw_waterjet")
            saw_hr_clamped = clamp(saw_hr, adder_min_bound, saw_limit, adder_min_bound)
            clean_stock_plan["sawing_hr"] = saw_hr_clamped
            _merge_adder("saw_waterjet", saw_hr_clamped, "stock_plan.sawing_hr")
        handling_hr = _as_float(stock_plan_raw.get("handling_hr"))
        if handling_hr and handling_hr > 0:
            handling_limit = _adder_limit("assembly")
            handling_hr_clamped = clamp(
                handling_hr, adder_min_bound, handling_limit, adder_min_bound
            )
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
            setup_limit = _adder_limit("setup_adders_hr")
            clean_setup["setup_adders_hr"] = clamp(
                setup_hr, adder_min_bound, setup_limit, adder_min_bound
            )
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
            inproc_limit = _adder_limit("inspection")
            inproc_clamped = clamp(
                inproc_hr, adder_min_bound, inproc_limit, adder_min_bound
            )
            clean_tol["in_process_inspection_hr"] = inproc_clamped
            _merge_adder("inspection", inproc_clamped, "tolerance_in_process_hr")
        final_hr = _as_float(tol_raw.get("final_inspection_hr") or tol_raw.get("final_hr"))
        if final_hr and final_hr > 0:
            final_limit = _adder_limit("inspection")
            final_clamped = clamp(final_hr, adder_min_bound, final_limit, adder_min_bound)
            clean_tol["final_inspection_hr"] = final_clamped
            _merge_adder("inspection", final_clamped, "tolerance_final_hr")
        finish_hr = _as_float(tol_raw.get("finishing_hr") or tol_raw.get("finish_hr"))
        if finish_hr and finish_hr > 0:
            finish_limit = _adder_limit("finishing_deburr")
            finish_clamped = clamp(
                finish_hr, adder_min_bound, finish_limit, adder_min_bound
            )
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

    if debug_enabled and debug_path is not None:
        try:
            debug_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        else:
            system_prompt = globals().get("SYSTEM_SUGGEST")
            snap = {
                "model": getattr(llm, "model_path", None) or model_path,
                "n_ctx": getattr(llm, "n_ctx", None),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": jdump(ctx, sort_keys=True)},
                ],
                "params": {"temperature": 0.2, "max_tokens": 256},
                "context_payload": ctx,
                "raw_response_text": raw_text_combined,
                "parsed_response": raw_by_task,
                "sanitized": out,
                "usage": combined_usage,
            }
            try:
                snap_path = debug_path / f"llm_snapshot_{int(os.path.getmtime(model_path))}.json"
            except Exception:
                snap_path = debug_path / f"llm_snapshot_{int(math.floor(math.modf(math.pi)[0] * 1e9))}.json"
            try:
                snap_path.write_text(jdump(snap, default=None), encoding="utf-8")
            except Exception:
                pass

    return out, meta


__all__ = [
    "HARDWARE_PASS_LABEL",
    "LEGACY_HARDWARE_PASS_LABEL",
    "LLM_MULTIPLIER_MIN",
    "LLM_MULTIPLIER_MAX",
    "LLM_ADDER_MAX",
    "_canonical_pass_label",
    "_plate_mass_properties",
    "_plate_mass_from_dims",
    "_as_float_or_none",
    "coerce_bounds",
    "get_llm_bound_defaults",
    "clamp",
    "_safe_get",
    "get_llm_overrides",
]
