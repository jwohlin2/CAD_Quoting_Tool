from __future__ import annotations

from collections.abc import Mapping as _MappingABC
from typing import Any, Mapping

from cad_quoter.domain import canonicalize_pass_through_map, coerce_bounds
from cad_quoter.llm.sanitizers import (
    as_float,
    as_int,
    clean_notes_list,
    clean_string,
    clean_string_list,
    clamp,
    coerce_bool_flag,
    sanitize_drilling_groups,
)


def overrides_to_suggestions(
    overrides: Mapping[str, Any] | None,
    *,
    bounds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert override inputs into the sanitized suggestion structure used by tests."""

    if not isinstance(overrides, _MappingABC):
        return {}

    guardrails = coerce_bounds(bounds or {})
    suggestions: dict[str, Any] = {}

    def _clamp_multiplier(value: Any) -> float | None:
        num = as_float(value)
        if num is None:
            return None
        return clamp(num, guardrails["mult_min"], guardrails["mult_max"])

    def _clamp_adder(key: str, value: Any) -> float | None:
        num = as_float(value)
        if num is None:
            return None
        lower = guardrails["adder_min_hr"]
        upper = guardrails["adder_max_hr"]
        bucket_caps = guardrails.get("adder_bucket_max") or {}
        if isinstance(bucket_caps, dict):
            bucket_cap = bucket_caps.get(str(key).lower())
            if isinstance(bucket_cap, (int, float)):
                upper = min(upper, float(bucket_cap))
        return clamp(num, lower, upper)

    mults_raw = overrides.get("process_hour_multipliers") or overrides.get("process_hour_mults")
    if isinstance(mults_raw, _MappingABC):
        mults: dict[str, float] = {}
        for key, value in mults_raw.items():
            cleaned = _clamp_multiplier(value)
            if cleaned is None:
                continue
            label = clean_string(key)
            if not label:
                continue
            mults[label] = cleaned
        if mults:
            suggestions["process_hour_multipliers"] = mults

    adders_raw = overrides.get("process_hour_adders") or overrides.get("process_hour_adds")
    if isinstance(adders_raw, _MappingABC):
        adders: dict[str, float] = {}
        for key, value in adders_raw.items():
            cleaned = _clamp_adder(str(key), value)
            if cleaned is None:
                continue
            label = clean_string(key)
            if not label:
                continue
            adders[label] = cleaned
        if adders:
            suggestions["process_hour_adders"] = adders

    add_pass_raw = overrides.get("add_pass_through") or overrides.get("pass_through")
    add_pass = canonicalize_pass_through_map(add_pass_raw)
    if add_pass:
        suggestions["add_pass_through"] = add_pass

    scrap_val = overrides.get("scrap_pct")
    if scrap_val is None:
        scrap_val = overrides.get("scrap_pct_override")
    scrap = as_float(scrap_val)
    if scrap is not None:
        scrap = clamp(float(scrap), guardrails["scrap_min"], guardrails["scrap_max"])
        suggestions["scrap_pct"] = scrap

    setups_val = as_int(overrides.get("setups"))
    if setups_val > 0:
        suggestions["setups"] = setups_val

    fixture = clean_string(overrides.get("fixture"))
    if fixture:
        suggestions["fixture"] = fixture

    notes = clean_notes_list(overrides.get("notes"))
    if notes:
        suggestions["notes"] = notes

    op_seq = clean_string_list(overrides.get("operation_sequence"))
    if op_seq:
        suggestions["operation_sequence"] = op_seq

    risks = clean_notes_list(overrides.get("dfm_risks"), limit=8)
    if risks:
        suggestions["dfm_risks"] = risks

    drilling_strategy = overrides.get("drilling_strategy")
    if isinstance(drilling_strategy, _MappingABC):
        cleaned: dict[str, Any] = {}
        multiplier = _clamp_multiplier(drilling_strategy.get("multiplier"))
        if multiplier is not None:
            cleaned["multiplier"] = multiplier
        per_hole = as_float(drilling_strategy.get("per_hole_floor_sec"))
        if per_hole is not None:
            cleaned["per_hole_floor_sec"] = max(0.0, float(per_hole))
        note = clean_string(drilling_strategy.get("note") or drilling_strategy.get("reason"))
        if note:
            cleaned["note"] = note
        if cleaned:
            suggestions["drilling_strategy"] = cleaned

    drilling_groups_raw = overrides.get("drilling_groups")
    if isinstance(drilling_groups_raw, (list, tuple)):
        cleaned_groups = sanitize_drilling_groups(drilling_groups_raw)
        if cleaned_groups:
            suggestions["drilling_groups"] = cleaned_groups

    stock_rec = overrides.get("stock_recommendation")
    if isinstance(stock_rec, _MappingABC):
        stock_clean: dict[str, Any] = {}
        stock_item = clean_string(stock_rec.get("stock_item"))
        if stock_item:
            stock_clean["stock_item"] = stock_item
        length = as_float(stock_rec.get("length_mm"))
        if length is not None:
            stock_clean["length_mm"] = float(length)
        if stock_clean:
            suggestions["stock_recommendation"] = stock_clean

    setup_rec = overrides.get("setup_recommendation")
    if isinstance(setup_rec, _MappingABC):
        setup_clean: dict[str, Any] = {}
        setup_count = as_int(setup_rec.get("setups"))
        if setup_count > 0:
            setup_clean["setups"] = setup_count
        if setup_clean:
            suggestions["setup_recommendation"] = setup_clean

    packaging = as_float(overrides.get("packaging_flat_cost"))
    if packaging is not None:
        suggestions["packaging_flat_cost"] = float(packaging)

    fai_required = coerce_bool_flag(overrides.get("fai_required"))
    if fai_required is not None:
        suggestions["fai_required"] = bool(fai_required)

    shipping_hint = clean_string(overrides.get("shipping_hint"))
    if shipping_hint:
        suggestions["shipping_hint"] = shipping_hint

    return suggestions


def suggestions_to_overrides(suggestions: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize LLM suggestions into a deterministic overrides payload."""

    if not isinstance(suggestions, _MappingABC):
        return {}

    overrides: dict[str, Any] = {}

    def _coerce_multiplier_map(raw: Any) -> dict[str, float]:
        result: dict[str, float] = {}
        if isinstance(raw, _MappingABC):
            for key, value in raw.items():
                label = clean_string(key)
                if not label:
                    continue
                num = as_float(value)
                if num is None:
                    continue
                result[label] = float(num)
        return result

    mults = _coerce_multiplier_map(suggestions.get("process_hour_multipliers"))
    if mults:
        overrides["process_hour_multipliers"] = mults

    adders = _coerce_multiplier_map(suggestions.get("process_hour_adders"))
    if adders:
        overrides["process_hour_adders"] = adders

    add_pass = canonicalize_pass_through_map(suggestions.get("add_pass_through"))
    if add_pass:
        overrides["add_pass_through"] = add_pass

    scrap = as_float(suggestions.get("scrap_pct"))
    if scrap is not None:
        overrides["scrap_pct"] = max(0.0, float(scrap))

    setups_val = as_int(suggestions.get("setups"))
    if setups_val > 0:
        overrides["setups"] = setups_val

    fixture = clean_string(suggestions.get("fixture"))
    if fixture:
        overrides["fixture"] = fixture

    notes = clean_notes_list(suggestions.get("notes"))
    if notes:
        overrides["notes"] = notes

    risks = clean_notes_list(suggestions.get("dfm_risks"), limit=8)
    if risks:
        overrides["dfm_risks"] = risks

    op_seq = clean_string_list(suggestions.get("operation_sequence"))
    if op_seq:
        overrides["operation_sequence"] = op_seq

    drilling_strategy = suggestions.get("drilling_strategy")
    if isinstance(drilling_strategy, _MappingABC):
        cleaned: dict[str, Any] = {}
        multiplier = as_float(drilling_strategy.get("multiplier"))
        if multiplier is not None:
            cleaned["multiplier"] = float(multiplier)
        per_hole = as_float(drilling_strategy.get("per_hole_floor_sec"))
        if per_hole is not None:
            cleaned["per_hole_floor_sec"] = max(0.0, float(per_hole))
        note = clean_string(drilling_strategy.get("note"))
        if note:
            cleaned["note"] = note
        if cleaned:
            overrides["drilling_strategy"] = cleaned

    packaging = as_float(suggestions.get("packaging_flat_cost"))
    if packaging is not None:
        overrides["packaging_flat_cost"] = float(packaging)

    fai = coerce_bool_flag(suggestions.get("fai_required"))
    if fai is not None:
        overrides["fai_required"] = bool(fai)

    shipping = clean_string(suggestions.get("shipping_hint"))
    if shipping:
        overrides["shipping_hint"] = shipping

    stock_rec = suggestions.get("stock_recommendation")
    if isinstance(stock_rec, _MappingABC):
        stock_clean: dict[str, Any] = {}
        stock_item = clean_string(stock_rec.get("stock_item"))
        if stock_item:
            stock_clean["stock_item"] = stock_item
        length = as_float(stock_rec.get("length_mm"))
        if length is not None:
            stock_clean["length_mm"] = float(length)
        if stock_clean:
            overrides["stock_recommendation"] = stock_clean

    setup_rec = suggestions.get("setup_recommendation")
    if isinstance(setup_rec, _MappingABC):
        setup_clean: dict[str, Any] = {}
        setup_count = as_int(setup_rec.get("setups"))
        if setup_count > 0:
            setup_clean["setups"] = setup_count
        if setup_clean:
            overrides["setup_recommendation"] = setup_clean

    return overrides
