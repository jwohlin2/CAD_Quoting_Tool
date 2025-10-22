from __future__ import annotations

import copy
import math
from collections.abc import Mapping as _MappingABC
from typing import Any

from cad_quoter.utils import coerce_bool
from cad_quoter.domain import (
    HARDWARE_PASS_LABEL,
    LEGACY_HARDWARE_PASS_LABEL,
    canonicalize_pass_through_map,
    coerce_bounds,
)
from cad_quoter.domain_models.values import to_float
from cad_quoter.app.guardrails import (
    enforce_finish_pass_guardrail,
    enforce_process_floor_guardrails,
    enforce_setups_guardrail,
)


ACCEPT_SCALAR_KEYS: tuple[str, ...] = (
    "scrap_pct",
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


SUGGESTION_SCALAR_KEYS: tuple[str, ...] = ACCEPT_SCALAR_KEYS


# ``merge_effective`` owns all merge behavior. These tables allow other helpers to
# share the same rules without duplicating the merge logic across the code base.
LOCKED_EFFECTIVE_FIELDS: frozenset[str] = frozenset(
    {
        "totals",
        "process_plan_pricing",
        "pricing_result",
        "process_hours",  # computed below from multipliers + adders
    }
)

SUGGESTIBLE_EFFECTIVE_FIELDS: frozenset[str] = frozenset(
    {
        "notes",
        "operation_sequence",
        "drilling_strategy",
        *SUGGESTION_SCALAR_KEYS,
    }
)

NUMERIC_EFFECTIVE_FIELDS: dict[str, tuple[float | None, float | None]] = {
    "fixture_build_hr": (0.0, 2.0),
    "soft_jaw_hr": (0.0, 1.0),
    "soft_jaw_material_cost": (0.0, 60.0),
    "handling_adder_hr": (0.0, 0.2),
    "cmm_minutes": (0.0, 60.0),
    "in_process_inspection_hr": (0.0, 0.5),
    "inspection_total_hr": (0.0, 12.0),
    "fai_prep_hr": (0.0, 1.0),
    "packaging_hours": (0.0, 0.5),
    "packaging_flat_cost": (0.0, 25.0),
    "shipping_cost": (0.0, None),
}

BOOL_EFFECTIVE_FIELDS: frozenset[str] = frozenset({"fai_required"})

TEXT_EFFECTIVE_FIELDS: dict[str, int] = {"shipping_hint": 80}

LIST_EFFECTIVE_FIELDS: frozenset[str] = frozenset({"operation_sequence"})

DICT_EFFECTIVE_FIELDS: frozenset[str] = frozenset({"drilling_strategy"})


def _collect_process_keys(*dicts: Mapping[str, Any] | None) -> set[str]:
    keys: set[str] = set()
    for d in dicts:
        if isinstance(d, _MappingABC):
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
    suggestions = {k: v for k, v in dict(suggestions or {}).items() if k not in LOCKED_EFFECTIVE_FIELDS}
    overrides = {k: v for k, v in dict(overrides or {}).items() if k not in LOCKED_EFFECTIVE_FIELDS}
    guard_ctx = dict(guard_ctx or {})

    bounds = baseline.get("_bounds") if isinstance(baseline, dict) else None
    if isinstance(bounds, dict):
        bounds = {str(k): v for k, v in bounds.items()}
    else:
        bounds = {}
    coerced_bounds = coerce_bounds(bounds)
    mult_min_bound = coerced_bounds["mult_min"]
    mult_max_bound = coerced_bounds["mult_max"]
    adder_min_bound = coerced_bounds["adder_min_hr"]
    adder_max_bound = coerced_bounds["adder_max_hr"]
    scrap_min_bound = coerced_bounds["scrap_min"]
    scrap_max_bound = coerced_bounds["scrap_max"]
    bucket_caps_bound = coerced_bounds.get("adder_bucket_max", {})

    def _clamp(value: float, kind: str, label: str, source: str) -> tuple[float, bool]:
        clamped = value
        changed = False
        source_norm = str(source).strip().lower()
        if kind == "multiplier":
            clamped = max(mult_min_bound, min(mult_max_bound, float(value)))
        elif kind == "adder":
            orig_val = float(value)
            raw_val = orig_val
            if source_norm == "llm" and raw_val > 240:
                raw_val = raw_val / 60.0
            bucket_name = None
            if "[" in label and "]" in label:
                bucket_name = label.split("[", 1)[-1].split("]", 1)[0].strip().lower()
            bucket_max = bucket_caps_bound.get(bucket_name) if bucket_name else None
            if bucket_max is None and bucket_name:
                bucket_max = bucket_caps_bound.get(bucket_name.lower())
            adder_cap = bucket_max if bucket_max is not None else adder_max_bound
            adder_cap = max(adder_min_bound, float(adder_cap))
            clamped = max(adder_min_bound, min(adder_cap, raw_val))
        elif kind == "scrap":
            clamped = max(scrap_min_bound, min(scrap_max_bound, float(value)))
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

    def _merge_bool_field(key: str) -> None:
        base_val = baseline.get(key) if isinstance(baseline.get(key), bool) else None
        value = base_val
        source = "baseline"
        if key in overrides:
            cand = coerce_bool(overrides.get(key))
            if cand is not None:
                value = cand
                source = "user"
        elif key in suggestions:
            cand = coerce_bool(suggestions.get(key))
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
    sugg_pass = canonicalize_pass_through_map(raw_sugg_pass)
    over_pass = canonicalize_pass_through_map(raw_over_pass)
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

    enforce_process_floor_guardrails(final_hours, guard_ctx, clamp_notes)

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

    if "notes" in SUGGESTIBLE_EFFECTIVE_FIELDS:
        notes_val: list[str] = []
        if isinstance(suggestions.get("notes"), list):
            notes_val.extend(
                [str(n).strip() for n in suggestions["notes"] if isinstance(n, str) and n.strip()]
            )
        if isinstance(overrides.get("notes"), list):
            notes_val.extend(
                [str(n).strip() for n in overrides["notes"] if isinstance(n, str) and n.strip()]
            )
        if notes_val:
            eff["notes"] = notes_val

    for key, (lo, hi) in NUMERIC_EFFECTIVE_FIELDS.items():
        _merge_numeric_field(key, lo, hi, key)

    for key in BOOL_EFFECTIVE_FIELDS:
        _merge_bool_field(key)

    for key, max_len in TEXT_EFFECTIVE_FIELDS.items():
        _merge_text_field(key, max_len=max_len)

    for key in LIST_EFFECTIVE_FIELDS:
        _merge_list_field(key)

    for key in DICT_EFFECTIVE_FIELDS:
        _merge_dict_field(key)

    enforce_setups_guardrail(eff, guard_ctx, clamp_notes, source_tags)
    final_pass = enforce_finish_pass_guardrail(
        eff,
        guard_ctx,
        final_pass,
        pass_sources,
        clamp_notes,
    )
    if isinstance(final_pass, dict) and final_pass:
        eff["add_pass_through"] = final_pass
    if pass_sources:
        source_tags["add_pass_through"] = pass_sources
    if clamp_notes:
        eff["_clamp_notes"] = clamp_notes
    eff["_source_tags"] = source_tags
    return eff
