from __future__ import annotations

from typing import Any, Mapping
from collections.abc import Mapping as _MappingABC

from appkit.env_utils import FORCE_ESTIMATOR, FORCE_PLANNER, _coerce_bool, _coerce_env_bool

DEFAULT_PLANNER_MODE = "planner"


def _resolve_planner_mode(
    params: Mapping[str, Any] | None,
    default_mode: str = DEFAULT_PLANNER_MODE,
) -> str:
    """Return the planner mode based on params and the FORCE flag."""

    if FORCE_ESTIMATOR:
        return "estimator"

    if FORCE_PLANNER:
        return DEFAULT_PLANNER_MODE

    if isinstance(params, _MappingABC):
        try:
            raw_mode = params.get("PlannerMode", default_mode)
        except Exception:
            raw_mode = default_mode
        try:
            planner_mode = str(raw_mode).strip().lower()
        except Exception:
            planner_mode = ""
        return planner_mode or default_mode

    return default_mode


def _resolve_planner_usage(
    planner_mode: str,
    signals: Mapping[str, Any] | None,
) -> bool:
    """Return True when planner pricing should be used."""

    if FORCE_ESTIMATOR:
        return False

    if FORCE_PLANNER:
        return True

    signals_map: Mapping[str, Any]
    if isinstance(signals, _MappingABC):
        signals_map = signals
    else:
        signals_map = {}

    has_line_items = bool(signals_map.get("line_items"))
    has_pricing = bool(signals_map.get("pricing_result"))
    has_totals = bool(signals_map.get("totals_present"))

    try:
        recognized_raw = signals_map.get("recognized_line_items", 0)
        recognized_count = int(recognized_raw)
    except Exception:
        recognized_count = 0
    has_recognized = recognized_count > 0

    base_signal = has_line_items or has_pricing or has_totals or has_recognized

    if planner_mode == "planner":
        used_planner = base_signal or has_totals
    elif planner_mode == "legacy":
        used_planner = has_line_items or has_recognized
    elif planner_mode == "estimator":
        used_planner = False
    else:  # auto / fallback
        used_planner = base_signal or has_recognized

    if has_line_items:
        used_planner = True

    return used_planner


def resolve_planner(
    params: Mapping[str, Any] | None,
    signals: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Determine whether planner pricing should be used and the mode."""

    planner_mode = _resolve_planner_mode(params)
    used_planner = _resolve_planner_usage(planner_mode, signals)

    return used_planner, planner_mode


def _planner_total_has_values(planner_meta_total: Mapping[str, Any]) -> bool:
    """Return True when planner total metadata includes tangible output."""

    for candidate_key in (
        "line_items",
        "total_cost",
        "cost",
        "minutes",
        "hr",
        "machine_cost",
        "labor_cost",
    ):
        try:
            value = planner_meta_total.get(candidate_key)  # type: ignore[arg-type]
        except Exception:
            value = None
        if isinstance(value, (list, tuple)) and value:
            return True
        try:
            if float(value or 0.0):  # type: ignore[arg-type]
                return True
        except Exception:
            continue
    return False


def _planner_meta_signals_present(meta_source: Mapping[str, Any]) -> bool:
    """Return True when planner metadata keys indicate planner output."""

    for raw_key, raw_meta in meta_source.items():
        key_lower = str(raw_key or "").strip().lower()
        if not key_lower:
            continue
        if key_lower.startswith("planner_"):
            if key_lower == "planner_total" and isinstance(raw_meta, _MappingABC):
                if _planner_total_has_values(raw_meta):
                    return True
                continue
            return True
    for alias in ("planner_total", "planner total"):
        planner_meta_total = meta_source.get(alias)
        if isinstance(planner_meta_total, _MappingABC) and _planner_total_has_values(planner_meta_total):
            return True
    return False


def _planner_signals_present(
    *,
    process_meta: Mapping[str, Any] | None = None,
    process_meta_raw: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
    planner_process_minutes: Any = None,
    hour_summary_entries: Mapping[str, Any] | None = None,
    additional_sources: list[Any] | None = None,
) -> bool:
    """Return True when planner output is detectable in the provided data."""

    meta_sources: list[Mapping[str, Any]] = []
    for candidate in (process_meta, process_meta_raw):
        if isinstance(candidate, _MappingABC) and candidate:
            meta_sources.append(candidate)
    for meta_source in meta_sources:
        if _planner_meta_signals_present(meta_source):
            return True

    if isinstance(hour_summary_entries, _MappingABC) and hour_summary_entries:
        for label, value in hour_summary_entries.items():
            key_lower = str(label or "").strip().lower()
            if not key_lower.startswith("planner"):
                continue
            base_value: Any
            if isinstance(value, (list, tuple)) and value:
                base_value = value[0]
            else:
                base_value = value
            if isinstance(base_value, (int, float)):
                if float(base_value) > 0.0:
                    return True
                continue
            if isinstance(base_value, str):
                text = base_value.strip()
                if not text:
                    continue
                try:
                    if float(text) > 0.0:
                        return True
                except Exception:
                    continue

    def _positive_minutes(value: Any) -> bool:
        if value is None:
            return False
        candidate: Any = value
        if isinstance(candidate, (list, tuple)):
            if not candidate:
                return False
            candidate = candidate[0]
        if isinstance(candidate, (list, tuple, dict, set)):
            return False
        if isinstance(candidate, str):
            candidate = candidate.strip()
            if not candidate:
                return False
        try:
            return float(candidate) > 0.0
        except (TypeError, ValueError):
            return False

    if isinstance(breakdown, _MappingABC):
        if _positive_minutes(breakdown.get("process_minutes")):
            return True

    if _positive_minutes(planner_process_minutes):
        return True

    breakdown_sources: list[Any] = []
    if isinstance(breakdown, _MappingABC):
        for key in (
            "process_plan",
            "planner_bucket_display_map",
            "planner_bucket_rollup",
            "process_plan_pricing",
            "planner_bucket_view",
            "process_plan_bucket_view",
            "bucket_view",
            "hour_summary",
            "process_plan_summary",
        ):
            breakdown_sources.append(breakdown.get(key))
    if additional_sources:
        breakdown_sources.extend(additional_sources)

    for candidate in breakdown_sources:
        if isinstance(candidate, _MappingABC) and candidate:
            return True
        if isinstance(candidate, (list, tuple)) and candidate:
            return True

    return False


__all__ = [
    "resolve_planner",
    "_planner_total_has_values",
    "_planner_meta_signals_present",
    "_planner_signals_present",
    "_coerce_bool",
    "_coerce_env_bool",
    "FORCE_PLANNER",
]

