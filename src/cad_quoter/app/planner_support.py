"""Planner-related resolution helpers and optional process planner wiring."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping as _MappingABC, MutableMapping as _MutableMappingABC
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TYPE_CHECKING, cast

from cad_quoter.app.env_flags import FORCE_ESTIMATOR, FORCE_PLANNER, _coerce_env_bool
from cad_quoter.utils import coerce_bool
from cad_quoter.domain_models.values import safe_float as _safe_float
from cad_quoter.utils.machining import _coerce_float_or_none
from cad_quoter.utils.text_rules import (
    canonicalize_amortized_label as _canonical_amortized_label,
)
from cad_quoter.pricing.process_buckets import bucketize
from cad_quoter.ui.planner_render import (
    _FINAL_BUCKET_HIDE_KEYS,
    _bucket_cost,
    _canonical_bucket_key,
    _planner_bucket_key_for_name,
    _prepare_bucket_view,
)

if TYPE_CHECKING:
    from cad_quoter.ui.services import QuoteConfiguration

DEFAULT_PLANNER_MODE = "planner"


# ---------------------------------------------------------------------------
# Planner usage resolution helpers
# ---------------------------------------------------------------------------


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


def resolve_pricing_source_value(
    base_value: Any,
    *,
    used_planner: bool | None = None,
    process_meta: Mapping[str, Any] | None = None,
    process_meta_raw: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
    planner_process_minutes: Any = None,
    hour_summary_entries: Mapping[str, Any] | None = None,
    additional_sources: Sequence[Any] | None = None,
    cfg: "QuoteConfiguration" | None = None,
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

    if _planner_signals_present(
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


# ---------------------------------------------------------------------------
# Optional process planner wiring
# ---------------------------------------------------------------------------

try:
    from cad_quoter.planning.process_planner import (
        PLANNERS as _PROCESS_PLANNERS,
    )
    from cad_quoter.planning.process_planner import (
        choose_skims as _planner_choose_skims,
    )
    from cad_quoter.planning.process_planner import (
        choose_wire_size as _planner_choose_wire_size,
    )
    from cad_quoter.planning.process_planner import (
        needs_wedm_for_windows as _planner_needs_wedm_for_windows,
    )
    from cad_quoter.planning.process_planner import (
        plan_job as _process_plan_job,
    )
except Exception:  # pragma: no cover - planner is optional at runtime
    _process_plan_job = None
    _PROCESS_PLANNERS = {}
    _planner_choose_wire_size = None
    _planner_choose_skims = None
    _planner_needs_wedm_for_windows = None
else:  # pragma: no cover - defensive guard for unexpected exports
    if not isinstance(_PROCESS_PLANNERS, dict):
        _PROCESS_PLANNERS = {}

_PROCESS_PLANNER_HELPERS: dict[str, Callable[..., Any]] = {}
if "_planner_choose_wire_size" in globals() and callable(_planner_choose_wire_size):
    _PROCESS_PLANNER_HELPERS["choose_wire_size"] = _planner_choose_wire_size  # type: ignore[index]
if "_planner_choose_skims" in globals() and callable(_planner_choose_skims):
    _PROCESS_PLANNER_HELPERS["choose_skims"] = _planner_choose_skims  # type: ignore[index]
if "_planner_needs_wedm_for_windows" in globals() and callable(
    _planner_needs_wedm_for_windows
):
    _PROCESS_PLANNER_HELPERS["needs_wedm_for_windows"] = _planner_needs_wedm_for_windows  # type: ignore[index]


__all__ = [
    "resolve_planner",
    "_planner_total_has_values",
    "_planner_meta_signals_present",
    "_planner_signals_present",
    "resolve_pricing_source_value",
    "apply_planner_result",
    "coerce_bool",
    "_coerce_env_bool",
    "FORCE_PLANNER",
    "_PROCESS_PLANNERS",
    "_PROCESS_PLANNER_HELPERS",
    "_process_plan_job",
]


# ---------------------------------------------------------------------------
# Planner application helpers
# ---------------------------------------------------------------------------


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


def _recognized_line_items_from_result(pricing_result: Mapping[str, Any] | None) -> int:
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


@dataclass(slots=True)
class PlannerApplicationResult:
    """Captured planner application side effects for downstream handling."""

    use_planner: bool
    planner_used: bool
    fallback_reason: str
    planner_machine_cost_total: float
    planner_labor_cost_total: float
    amortized_programming: float
    amortized_fixture: float
    bucket_view_prepared: dict[str, Any]
    aggregated_bucket_minutes: dict[str, dict[str, float]]


def apply_planner_result(
    planner_result: Mapping[str, Any] | None,
    *,
    breakdown: MutableMapping[str, Any],
    baseline: MutableMapping[str, Any],
    process_plan_summary: MutableMapping[str, Any],
    process_costs: MutableMapping[str, float],
    process_meta: MutableMapping[str, Any],
    totals_block: MutableMapping[str, float],
    bucketize_rates: Mapping[str, Mapping[str, float]] | None,
    bucketize_nre: Mapping[str, Any] | None,
    geom_for_bucketize: Mapping[str, Any] | None,
    qty_for_bucketize: int,
    planner_exception: Exception | None = None,
    fallback_reason: str = "",
    planner_bucket_abs_epsilon: float = 0.51,
) -> PlannerApplicationResult:
    """Apply planner pricing data to breakdown/process metadata and buckets."""

    red_flags = breakdown.setdefault("red_flags", [])
    if not isinstance(red_flags, list):
        red_flags = []
        breakdown["red_flags"] = red_flags

    if isinstance(planner_result, dict):
        planner_data: MutableMapping[str, Any] = planner_result
    elif isinstance(planner_result, _MutableMappingABC):
        planner_data = cast(MutableMapping[str, Any], planner_result)
    elif isinstance(planner_result, _MappingABC):
        planner_data = dict(planner_result)
    else:
        planner_data = {}

    recognized_line_items = _recognized_line_items_from_result(planner_data)
    if recognized_line_items > 0 and "recognized_line_items" not in planner_data:
        planner_data["recognized_line_items"] = recognized_line_items

    if planner_data:
        breakdown["process_plan_pricing"] = planner_data
        baseline["process_plan_pricing"] = planner_data
        process_plan_summary["pricing"] = planner_data
        breakdown["pricing_source"] = "Planner"

    use_planner = bool(planner_data) and planner_exception is None and recognized_line_items > 0
    fallback_reason_local = fallback_reason
    if planner_data and not use_planner:
        if planner_exception is not None:
            fallback_reason_local = "Planner pricing failed; using legacy fallback"
        elif recognized_line_items <= 0:
            fallback_reason_local = "Planner recognized no operations; using legacy fallback"

    aggregated_bucket_minutes: dict[str, dict[str, float]] = {}
    amortized_programming = 0.0
    amortized_fixture = 0.0
    planner_machine_cost_total = 0.0
    planner_labor_cost_total = 0.0
    planner_used = False

    line_items_list: Sequence[Mapping[str, Any]] = []
    if isinstance(planner_data.get("line_items"), Sequence):
        line_items_candidate = cast(Sequence[Any], planner_data.get("line_items"))
        line_items_list = [
            cast(Mapping[str, Any], entry)
            for entry in line_items_candidate
            if isinstance(entry, _MappingABC)
        ]

    if use_planner:
        totals_map = planner_data.get("totals") if isinstance(planner_data.get("totals"), _MappingABC) else {}
        totals_map = cast(Mapping[str, Any], totals_map)
        machine_cost = float(_coerce_float_or_none(totals_map.get("machine_cost")) or 0.0)
        labor_cost_total = float(_coerce_float_or_none(totals_map.get("labor_cost")) or 0.0)
        total_minutes = float(_coerce_float_or_none(totals_map.get("minutes")) or 0.0)

        for item in line_items_list:
            raw_label = item.get("op") or item.get("name") or ""
            canonical_label, is_amortized = _canonical_amortized_label(raw_label)
            normalized_label = str(canonical_label or raw_label or "").strip().lower()
            if not is_amortized and normalized_label:
                if any(token in normalized_label for token in ("per part", "per pc", "per piece")):
                    is_amortized = True
            if not is_amortized or not normalized_label:
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
            fallback_reason_local = zero_cost_message
            use_planner = False
        else:
            previous_machine = float(_coerce_float_or_none(process_costs.get("Machine")) or 0.0)
            previous_labor = float(_coerce_float_or_none(process_costs.get("Labor")) or 0.0)

            process_costs.clear()
            process_costs.update(
                {
                    "Machine": round(planner_machine_cost_total, 2),
                    "Labor": round(planner_labor_cost_total, 2),
                }
            )

            planner_totals_map = planner_data.get("totals") if isinstance(planner_data.get("totals"), _MappingABC) else {}
            planner_totals_map = cast(Mapping[str, Any], planner_totals_map)
            minutes_by_bucket = planner_totals_map.get("minutes_by_bucket") if isinstance(planner_totals_map, _MappingABC) else {}
            cost_by_bucket = planner_totals_map.get("cost_by_bucket") if isinstance(planner_totals_map, _MappingABC) else {}

            if isinstance(minutes_by_bucket, _MappingABC) and minutes_by_bucket:
                hour_summary: dict[str, float] = {}
                for key, value in minutes_by_bucket.items():
                    try:
                        minutes_val = float(value or 0.0)
                    except Exception:
                        minutes_val = 0.0
                    hour_summary[str(key)] = round(minutes_val / 60.0, 2)
                process_plan_summary["hour_summary"] = hour_summary

            if isinstance(cost_by_bucket, _MappingABC) and cost_by_bucket:
                bucket_costs: list[dict[str, float]] = []
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
            breakdown["pricing_source"] = "Planner"
            breakdown["process_minutes"] = total_minutes
            baseline["pricing_source"] = "Planner"
            process_plan_summary["used_planner"] = True
            planner_used = True

            hr_total = total_minutes / 60.0 if total_minutes else 0.0
            process_meta["planner_total"] = {
                "minutes": total_minutes,
                "hr": hr_total,
                "cost": machine_cost + labor_cost_total,
                "machine_cost": planner_machine_cost_total,
                "labor_cost": labor_cost_total,
                "labor_cost_excl_amortized": planner_labor_cost_total,
                "amortized_programming": amortized_programming,
                "amortized_fixture": amortized_fixture,
                "line_items": list(planner_data.get("line_items", []) or []),
            }
            process_meta["planner_machine"] = {
                "minutes": total_minutes,
                "hr": hr_total,
                "cost": planner_machine_cost_total,
            }
            process_meta["planner_labor"] = {
                "minutes": total_minutes,
                "hr": hr_total,
                "cost": labor_cost_total,
                "cost_excl_amortized": planner_labor_cost_total,
                "amortized_programming": amortized_programming,
                "amortized_fixture": amortized_fixture,
            }

            if (
                planner_machine_cost_total > 0.0
                and previous_machine > 0.0
                and abs(planner_machine_cost_total - previous_machine) > planner_bucket_abs_epsilon
            ):
                red_flags.append("Planner totals drifted (machine cost)")
            if (
                planner_labor_cost_total > 0.0
                and previous_labor > 0.0
                and abs(planner_labor_cost_total - previous_labor) > planner_bucket_abs_epsilon
            ):
                red_flags.append("Planner totals drifted (labor cost)")

    for entry in line_items_list:
        bucket_key = _planner_bucket_key_for_name(entry.get("op"))
        if not bucket_key:
            continue
        canon_key = _canonical_bucket_key(bucket_key) or bucket_key
        if not canon_key or canon_key in _FINAL_BUCKET_HIDE_KEYS:
            continue
        minutes_val = float(_safe_float(entry.get("minutes"), 0.0))
        machine_val = float(_bucket_cost(entry, "machine_cost", "machine$"))
        labor_val = float(_bucket_cost(entry, "labor_cost", "labor$"))
        if canon_key == "milling" and labor_val > 0.0:
            machine_val += labor_val
            labor_val = 0.0
        if minutes_val <= 0.0 and machine_val <= 0.0 and labor_val <= 0.0:
            continue
        metrics = aggregated_bucket_minutes.setdefault(
            canon_key,
            {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0},
        )
        metrics["minutes"] += minutes_val
        metrics["machine$"] += machine_val
        metrics["labor$"] += labor_val

    try:
        bucketized_raw = bucketize(
            planner_data if isinstance(planner_data, dict) else dict(planner_data),
            bucketize_rates or {},
            bucketize_nre or {},
            qty=qty_for_bucketize,
            geom=geom_for_bucketize or {},
        )
    except Exception:
        bucketized_raw = {}

    if not isinstance(bucketized_raw, _MappingABC):
        bucketized_raw = {}

    bucket_view_prepared = _prepare_bucket_view(bucketized_raw)

    return PlannerApplicationResult(
        use_planner=use_planner,
        planner_used=planner_used,
        fallback_reason=fallback_reason_local,
        planner_machine_cost_total=planner_machine_cost_total,
        planner_labor_cost_total=planner_labor_cost_total,
        amortized_programming=amortized_programming,
        amortized_fixture=amortized_fixture,
        bucket_view_prepared=bucket_view_prepared,
        aggregated_bucket_minutes=aggregated_bucket_minutes,
    )
