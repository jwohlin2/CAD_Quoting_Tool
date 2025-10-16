from __future__ import annotations

from collections.abc import Mapping as _MappingABC
from typing import Any, Mapping

import math

from cad_quoter.domain import canonicalize_pass_through_map, coerce_bounds
from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.llm_overrides import clamp


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"y", "yes", "true", "1", "on"}:
            return True
        if text in {"n", "no", "false", "0", "off"}:
            return False
    return None


def _clean_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_str_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for entry in value:
            item = _clean_string(entry)
            if item:
                cleaned.append(item)
        return cleaned
    item = _clean_string(value)
    return [item] if item else []


def _coerce_int(value: Any) -> int | None:
    num = _coerce_float_or_none(value)
    if num is None or not math.isfinite(num):
        return None
    rounded = int(round(float(num)))
    return rounded if rounded > 0 else None


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
        num = _coerce_float_or_none(value)
        if num is None:
            return None
        return clamp(float(num), guardrails["mult_min"], guardrails["mult_max"])

    def _clamp_adder(key: str, value: Any) -> float | None:
        num = _coerce_float_or_none(value)
        if num is None:
            return None
        lower = guardrails["adder_min_hr"]
        upper = guardrails["adder_max_hr"]
        bucket_caps = guardrails.get("adder_bucket_max") or {}
        if isinstance(bucket_caps, dict):
            bucket_cap = bucket_caps.get(str(key).lower())
            if isinstance(bucket_cap, (int, float)):
                upper = min(upper, float(bucket_cap))
        return clamp(float(num), lower, upper)

    mults_raw = overrides.get("process_hour_multipliers") or overrides.get("process_hour_mults")
    if isinstance(mults_raw, _MappingABC):
        mults: dict[str, float] = {}
        for key, value in mults_raw.items():
            cleaned = _clamp_multiplier(value)
            if cleaned is None:
                continue
            label = _clean_string(key)
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
            label = _clean_string(key)
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
    scrap = _coerce_float_or_none(scrap_val)
    if scrap is not None:
        scrap = clamp(float(scrap), guardrails["scrap_min"], guardrails["scrap_max"])
        suggestions["scrap_pct"] = scrap

    setups_val = _coerce_int(overrides.get("setups"))
    if setups_val is not None:
        suggestions["setups"] = setups_val

    fixture = _clean_string(overrides.get("fixture"))
    if fixture:
        suggestions["fixture"] = fixture

    notes = _clean_str_list(overrides.get("notes"))
    if notes:
        suggestions["notes"] = notes

    op_seq = _clean_str_list(overrides.get("operation_sequence"))
    if op_seq:
        suggestions["operation_sequence"] = op_seq

    risks = _clean_str_list(overrides.get("dfm_risks"))
    if risks:
        suggestions["dfm_risks"] = risks

    drilling_strategy = overrides.get("drilling_strategy")
    if isinstance(drilling_strategy, _MappingABC):
        cleaned: dict[str, Any] = {}
        multiplier = _clamp_multiplier(drilling_strategy.get("multiplier"))
        if multiplier is not None:
            cleaned["multiplier"] = multiplier
        per_hole = _coerce_float_or_none(drilling_strategy.get("per_hole_floor_sec"))
        if per_hole is not None:
            cleaned["per_hole_floor_sec"] = max(0.0, float(per_hole))
        note = _clean_string(drilling_strategy.get("note") or drilling_strategy.get("reason"))
        if note:
            cleaned["note"] = note
        if cleaned:
            suggestions["drilling_strategy"] = cleaned

    drilling_groups_raw = overrides.get("drilling_groups")
    if isinstance(drilling_groups_raw, (list, tuple)):
        cleaned_groups: list[dict[str, Any]] = []
        for entry in drilling_groups_raw:
            if not isinstance(entry, _MappingABC):
                continue
            qty = _coerce_int(entry.get("qty") or entry.get("count"))
            dia = _coerce_float_or_none(entry.get("dia_mm") or entry.get("diameter_mm"))
            if qty is None or dia is None:
                continue
            cleaned_groups.append({"qty": qty, "dia_mm": float(dia)})
        if cleaned_groups:
            suggestions["drilling_groups"] = cleaned_groups

    stock_rec = overrides.get("stock_recommendation")
    if isinstance(stock_rec, _MappingABC):
        stock_clean: dict[str, Any] = {}
        stock_item = _clean_string(stock_rec.get("stock_item"))
        if stock_item:
            stock_clean["stock_item"] = stock_item
        length = _coerce_float_or_none(stock_rec.get("length_mm"))
        if length is not None:
            stock_clean["length_mm"] = float(length)
        if stock_clean:
            suggestions["stock_recommendation"] = stock_clean

    setup_rec = overrides.get("setup_recommendation")
    if isinstance(setup_rec, _MappingABC):
        setup_clean: dict[str, Any] = {}
        setup_count = _coerce_int(setup_rec.get("setups"))
        if setup_count is not None:
            setup_clean["setups"] = setup_count
        if setup_clean:
            suggestions["setup_recommendation"] = setup_clean

    packaging = _coerce_float_or_none(overrides.get("packaging_flat_cost"))
    if packaging is not None:
        suggestions["packaging_flat_cost"] = float(packaging)

    fai_required = _coerce_bool(overrides.get("fai_required"))
    if fai_required is not None:
        suggestions["fai_required"] = bool(fai_required)

    shipping_hint = _clean_string(overrides.get("shipping_hint"))
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
                label = _clean_string(key)
                if not label:
                    continue
                num = _coerce_float_or_none(value)
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

    scrap = _coerce_float_or_none(suggestions.get("scrap_pct"))
    if scrap is not None:
        overrides["scrap_pct"] = max(0.0, float(scrap))

    setups_val = _coerce_int(suggestions.get("setups"))
    if setups_val is not None:
        overrides["setups"] = setups_val

    fixture = _clean_string(suggestions.get("fixture"))
    if fixture:
        overrides["fixture"] = fixture

    notes = _clean_str_list(suggestions.get("notes"))
    if notes:
        overrides["notes"] = notes

    risks = _clean_str_list(suggestions.get("dfm_risks"))
    if risks:
        overrides["dfm_risks"] = risks

    op_seq = _clean_str_list(suggestions.get("operation_sequence"))
    if op_seq:
        overrides["operation_sequence"] = op_seq

    drilling_strategy = suggestions.get("drilling_strategy")
    if isinstance(drilling_strategy, _MappingABC):
        cleaned: dict[str, Any] = {}
        multiplier = _coerce_float_or_none(drilling_strategy.get("multiplier"))
        if multiplier is not None:
            cleaned["multiplier"] = float(multiplier)
        per_hole = _coerce_float_or_none(drilling_strategy.get("per_hole_floor_sec"))
        if per_hole is not None:
            cleaned["per_hole_floor_sec"] = max(0.0, float(per_hole))
        note = _clean_string(drilling_strategy.get("note"))
        if note:
            cleaned["note"] = note
        if cleaned:
            overrides["drilling_strategy"] = cleaned

    packaging = _coerce_float_or_none(suggestions.get("packaging_flat_cost"))
    if packaging is not None:
        overrides["packaging_flat_cost"] = float(packaging)

    fai = _coerce_bool(suggestions.get("fai_required"))
    if fai is not None:
        overrides["fai_required"] = bool(fai)

    shipping = _clean_string(suggestions.get("shipping_hint"))
    if shipping:
        overrides["shipping_hint"] = shipping

    stock_rec = suggestions.get("stock_recommendation")
    if isinstance(stock_rec, _MappingABC):
        stock_clean: dict[str, Any] = {}
        stock_item = _clean_string(stock_rec.get("stock_item"))
        if stock_item:
            stock_clean["stock_item"] = stock_item
        length = _coerce_float_or_none(stock_rec.get("length_mm"))
        if length is not None:
            stock_clean["length_mm"] = float(length)
        if stock_clean:
            overrides["stock_recommendation"] = stock_clean

    setup_rec = suggestions.get("setup_recommendation")
    if isinstance(setup_rec, _MappingABC):
        setup_clean: dict[str, Any] = {}
        setup_count = _coerce_int(setup_rec.get("setups"))
        if setup_count is not None:
            setup_clean["setups"] = setup_count
        if setup_clean:
            overrides["setup_recommendation"] = setup_clean

    return overrides
