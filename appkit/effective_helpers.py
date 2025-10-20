from __future__ import annotations

import copy
from typing import Any

from cad_quoter.domain import QuoteState

from appkit.guardrails import build_guard_context, apply_drilling_floor_notes
from appkit.merge_utils import (
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


__all__ = [
    "ensure_accept_flags",
    "compute_effective_state",
    "reprice_with_effective",
]
