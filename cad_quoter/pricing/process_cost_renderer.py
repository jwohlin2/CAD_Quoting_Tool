from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import os
from typing import Any

from cad_quoter.pricing.rate_buckets import RATE_BUCKETS
from cad_quoter.pricing.process_buckets import (
    canonical_bucket_key,
    bucket_label,
    flatten_rates,
    lookup_rate,
)


ORDER: tuple[str, ...] = (
    "milling",
    "drilling",
    "counterbore",
    "tapping",
    "grinding",
    "wire_edm",
    "sinker_edm",
    "finishing_deburr",
    "saw_waterjet",
    "inspection",
    "toolmaker_support",
    "fixture_build_amortized",
    "programming_amortized",
    "misc",
)

HIDE_IN_COST: frozenset[str] = frozenset(
    {
        "planner_total",
        "planner_labor",
        "planner_machine",
        "misc",
    }
)

__all__ = ["ORDER", "canonicalize_costs", "render_process_costs"]


def _iter_items(data: Mapping[str, Any] | Iterable[Any] | None) -> Iterable[tuple[Any, Any]]:
    if isinstance(data, Mapping):
        return data.items()
    if isinstance(data, Iterable):
        items: list[tuple[Any, Any]] = []
        for entry in data:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                items.append((entry[0], entry[1]))
        return items
    return []


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def canonicalize_costs(process_costs: Mapping[str, Any] | Iterable[Any] | None) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, raw in _iter_items(process_costs):
        canon = canonical_bucket_key(key)
        if not canon:
            continue
        amount = _to_float(raw)
        if amount is None:
            continue
        totals[canon] = totals.get(canon, 0.0) + amount
    return totals


def _canonicalize_minutes(minutes_detail: Mapping[str, Any] | Iterable[Any] | None) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, raw in _iter_items(minutes_detail):
        canon = canonical_bucket_key(key)
        if not canon:
            continue
        minutes = _to_float(raw)
        if minutes is None:
            continue
        totals[canon] = totals.get(canon, 0.0) + minutes
    return totals


def _emit_cost_row(
    tbl: Any,
    *,
    label: str,
    hours: float,
    rate: float,
    cost: float,
) -> None:
    if tbl is None:
        return
    if hasattr(tbl, "add_row"):
        add_row = getattr(tbl, "add_row")
        try:
            add_row(label=label, hours=hours, rate=rate, cost=cost)
            return
        except TypeError:
            add_row(label, hours, rate, cost)  # type: ignore[call-arg]
            return
    if hasattr(tbl, "row"):
        row = getattr(tbl, "row")
        try:
            row(label=label, hours=hours, rate=rate, cost=cost)
            return
        except TypeError:
            row(label, hours, rate, cost)  # type: ignore[call-arg]
            return
    if hasattr(tbl, "append"):
        append = getattr(tbl, "append")
        append({"label": label, "hours": hours, "rate": rate, "cost": cost})
        return
    raise AttributeError("Table object must support add_row, row, or append")


def render_process_costs(
    tbl: Any,
    process_costs: Mapping[str, Any] | Iterable[Any] | None,
    rates: Mapping[str, Any] | None,
    minutes_detail: Mapping[str, Any] | Iterable[Any] | None,
    *,
    process_plan: Mapping[str, Any] | None = None,
) -> float:
    """Render process costs into *tbl* using a fixed bucket order."""

    costs = canonicalize_costs(process_costs)
    minutes = _canonicalize_minutes(minutes_detail)
    flat_rates, normalized_rates = flatten_rates(rates)

    debug_misc = os.environ.get("DEBUG_MISC") == "1"

    shown_total = 0.0
    hidden_total = 0.0
    planner_drilling_minutes: float | None = None
    if isinstance(process_plan, Mapping):
        drilling_plan = process_plan.get("drilling")
        if isinstance(drilling_plan, Mapping):
            billed = _to_float(drilling_plan.get("total_minutes_billed"))
            if billed is not None:
                planner_drilling_minutes = max(0.0, billed)

    for key in ORDER:
        raw_amount = float(costs.get(key, 0.0))
        hours = max(0.0, float(minutes.get(key, 0.0)) / 60.0)
        rate = lookup_rate(key, flat_rates, normalized_rates)
        inferred_amount: float | None = None
        if hours > 0 and rate > 0:
            inferred_amount = round(hours * rate, 2)
        amount = round(raw_amount, 2)
        if inferred_amount is not None:
            amount = inferred_amount
        costs[key] = amount
        if key in HIDE_IN_COST:
            if not math.isclose(amount, 0.0, abs_tol=1e-9):
                hidden_total += amount
            continue
        if key == "misc" and not debug_misc and raw_amount < 50.0:
            hidden_total += amount
            continue
        if math.isclose(amount, 0.0, abs_tol=1e-9):
            continue
        hours = max(0.0, float(minutes.get(key, 0.0)) / 60.0)
        if key == "drilling" and planner_drilling_minutes is not None:
            drill_hr_precise = max(0.0, planner_drilling_minutes / 60.0)
            card_hr = drill_hr_precise
            row_hr = hours if hours > 0.0 else drill_hr_precise
            if abs(row_hr - card_hr) > 1e-2:
                row_hr = card_hr
            hours = row_hr
        rate = lookup_rate(key, flat_rates, normalized_rates)
        if rate <= 0 and hours > 0:
            rate = amount / hours
        if key == "drilling" and planner_drilling_minutes is not None and rate > 0:
            amount = round(hours * rate, 2)
            costs[key] = amount
        _emit_cost_row(
            tbl,
            label=bucket_label(key),
            hours=hours,
            rate=rate,
            cost=amount,
        )
        shown_total += amount

    model_total = round(sum(float(value) for value in costs.values()) - hidden_total, 2)
    if not math.isclose(shown_total, model_total, abs_tol=0.01):
        raise AssertionError((shown_total, model_total))

    return shown_total
