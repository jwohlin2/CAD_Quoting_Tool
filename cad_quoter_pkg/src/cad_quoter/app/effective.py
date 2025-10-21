from __future__ import annotations

import copy
import math
from typing import Any

from cad_quoter.domain import (
    QuoteState,
    canonicalize_pass_through_map,
)
from cad_quoter.app.guardrails import apply_drilling_floor_notes, build_guard_context
from cad_quoter.app.merge_utils import (
    ACCEPT_SCALAR_KEYS,
    SUGGESTION_SCALAR_KEYS,
    merge_effective,
)


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

    for key in ACCEPT_SCALAR_KEYS:
        if key in suggestions and not isinstance(accept.get(key), bool):
            accept[key] = False
        if key not in suggestions and key in accept and not isinstance(accept.get(key), dict):
            # keep user toggles if overrides exist even without suggestions
            continue
    if isinstance(suggestions.get("operation_sequence"), list) and not isinstance(
        accept.get("operation_sequence"), bool
    ):
        accept["operation_sequence"] = False
    if isinstance(suggestions.get("drilling_strategy"), dict) and not isinstance(
        accept.get("drilling_strategy"), bool
    ):
        accept["drilling_strategy"] = False


def compute_effective_state(state: QuoteState) -> tuple[dict, dict]:
    existing_guard_ctx = getattr(state, "guard_context", None)
    if not isinstance(existing_guard_ctx, dict) or not existing_guard_ctx:
        try:
            state.guard_context = build_guard_context(state)
        except Exception:
            state.guard_context = dict(existing_guard_ctx or {})

    baseline = state.baseline or {}
    suggestions = state.suggestions or {}
    overrides = state.user_overrides or {}
    accept_raw = state.accept_llm
    accept = accept_raw if isinstance(accept_raw, dict) else {}

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

    for scalar_key in SUGGESTION_SCALAR_KEYS:
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

    guard_ctx = build_guard_context(state)
    state.guard_context = guard_ctx

    ensure_accept_flags(state)
    merged, sources = compute_effective_state(state)
    state.effective = merged
    state.effective_sources = sources

    apply_drilling_floor_notes(state, guard_ctx=guard_ctx)
    return state


def effective_to_overrides(effective: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or {}
    out: dict[str, Any] = {}
    mults = (
        effective.get("process_hour_multipliers")
        if isinstance(effective.get("process_hour_multipliers"), dict)
        else {}
    )
    if mults:
        cleaned = {
            k: float(v)
            for k, v in mults.items()
            if v is not None
            and not math.isclose(float(v), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        }
        if cleaned:
            out["process_hour_multipliers"] = cleaned
    adders = (
        effective.get("process_hour_adders")
        if isinstance(effective.get("process_hour_adders"), dict)
        else {}
    )
    if adders:
        cleaned_add = {
            k: float(v)
            for k, v in adders.items()
            if v is not None and not math.isclose(float(v), 0.0, abs_tol=1e-6)
        }
        if cleaned_add:
            out["process_hour_adders"] = cleaned_add
    passes = (
        effective.get("add_pass_through")
        if isinstance(effective.get("add_pass_through"), dict)
        else {}
    )
    if passes:
        canonical_passes = canonicalize_pass_through_map(passes)
        cleaned_pass = {
            k: float(v)
            for k, v in canonical_passes.items()
            if not math.isclose(float(v), 0.0, abs_tol=1e-6)
        }
        if cleaned_pass:
            out["add_pass_through"] = cleaned_pass
    scrap_eff = effective.get("scrap_pct")
    scrap_base = baseline.get("scrap_pct")
    if scrap_eff is not None and (
        scrap_base is None
        or not math.isclose(float(scrap_eff), float(scrap_base or 0.0), abs_tol=1e-6)
    ):
        out["scrap_pct_override"] = float(scrap_eff)
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
        if base_val is None or not math.isclose(
            float(eff_val), float(base_val or 0.0), rel_tol=1e-6, abs_tol=1e-6
        ):
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


__all__ = [
    "ensure_accept_flags",
    "compute_effective_state",
    "reprice_with_effective",
    "effective_to_overrides",
]
