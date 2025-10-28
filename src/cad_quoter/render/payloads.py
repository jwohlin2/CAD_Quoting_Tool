"""Helpers for building structured payloads used by quote rendering."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping


def _render_as_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` coerced to ``float`` with ``default`` as a fallback."""

    try:
        return float(value)
    except Exception:
        return default


def build_summary_payload(
    *,
    quote_qty: Any,
    subtotal_before_margin: Any,
    price: Any,
    applied_pcts: Mapping[str, Any] | MutableMapping[str, Any] | None,
    expedite_cost: Any,
    breakdown: Mapping[str, Any] | MutableMapping[str, Any] | None,
    ladder_labor: Any,
    total_direct_costs_value: Any,
    currency: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return the summary payload and accompanying monetary metrics.

    The helper mirrors the legacy inline logic inside ``render_quote`` and returns
    both the structured payload used by the renderer as well as the computed
    numeric values required by subsequent payload builders.
    """

    try:
        qty_float = float(quote_qty or 0.0)
    except Exception:
        qty_float = 0.0
    if qty_float > 0 and abs(round(qty_float) - qty_float) < 1e-9:
        summary_qty: int | float = int(round(qty_float))
    else:
        summary_qty = qty_float if qty_float > 0 else quote_qty

    applied = applied_pcts or {}
    margin_pct_value = _render_as_float(applied.get("MarginPct"), 0.0)
    expedite_pct_value = _render_as_float(applied.get("ExpeditePct"), 0.0)
    expedite_amount = _render_as_float(expedite_cost, 0.0)
    subtotal_before_margin_val = _render_as_float(subtotal_before_margin, 0.0)
    final_price_val = _render_as_float(price, 0.0)
    margin_amount = max(0.0, final_price_val - subtotal_before_margin_val)

    breakdown_map = breakdown or {}
    labor_total_amount = _render_as_float(
        (breakdown_map or {}).get("total_labor_cost"),
        _render_as_float(ladder_labor, 0.0),
    )
    direct_total_amount = _render_as_float(total_direct_costs_value, 0.0)

    summary_payload = {
        "qty": summary_qty,
        "final_price": round(final_price_val, 2),
        "unit_price": round(final_price_val, 2),
        "subtotal_before_margin": round(subtotal_before_margin_val, 2),
        "margin_pct": float(margin_pct_value),
        "margin_amount": round(margin_amount, 2),
        "expedite_pct": float(expedite_pct_value),
        "expedite_amount": round(expedite_amount, 2),
        "currency": currency,
    }

    metrics = {
        "subtotal_before_margin": subtotal_before_margin_val,
        "final_price": final_price_val,
        "margin_amount": margin_amount,
        "expedite_amount": expedite_amount,
        "labor_total_amount": labor_total_amount,
        "direct_total_amount": direct_total_amount,
    }
    return summary_payload, metrics


def build_price_drivers_payload(
    drivers: Iterable[Any],
    llm_notes: Iterable[Any] | None,
) -> list[dict[str, str]]:
    """Return the payload for price driver details.

    The function deduplicates entries (case-insensitive) while preserving order
    and mirrors the behaviour previously implemented inline.
    """

    detail_sources: list[Any] = list(drivers)
    if llm_notes is not None:
        detail_sources.extend(llm_notes)

    driver_details: list[str] = []
    seen_driver_details: set[str] = set()
    for detail in detail_sources:
        text = str(detail).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_driver_details:
            continue
        seen_driver_details.add(key)
        driver_details.append(text)
    return [{"detail": detail} for detail in driver_details]


def build_cost_breakdown_payload(
    *,
    labor_total_amount: Any,
    direct_total_amount: Any,
    expedite_amount: Any,
    subtotal_before_margin: Any,
    margin_amount: Any,
    final_price: Any,
) -> list[tuple[str, float]]:
    """Return the cost breakdown payload used by ``render_quote``."""

    labor_total_val = round(_render_as_float(labor_total_amount, 0.0), 2)
    direct_total_val = round(_render_as_float(direct_total_amount, 0.0), 2)
    expedite_raw = _render_as_float(expedite_amount, 0.0)
    expedite_val = round(expedite_raw, 2)
    subtotal_val = round(_render_as_float(subtotal_before_margin, 0.0), 2)
    margin_val = round(_render_as_float(margin_amount, 0.0), 2)
    final_price_val = round(_render_as_float(final_price, 0.0), 2)

    payload: list[tuple[str, float]] = []
    payload.append(("Machine & Labor", labor_total_val))
    payload.append(("Direct Costs", direct_total_val))
    if expedite_raw > 0:
        payload.append(("Expedite", expedite_val))
    payload.append(("Subtotal before Margin", subtotal_val))
    payload.append(("Margin", margin_val))
    payload.append(("Final Price", final_price_val))
    return payload


__all__ = [
    "build_summary_payload",
    "build_price_drivers_payload",
    "build_cost_breakdown_payload",
    "_render_as_float",
]
