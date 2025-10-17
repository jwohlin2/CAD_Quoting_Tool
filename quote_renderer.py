"""High-level renderer for the compact quote summary email.

The pricing engine produces a fairly rich dictionary containing final pricing,
driver information, and supplemental metadata.  This module translates that
plain data into a deterministic, email-safe ASCII presentation built on top of
``text_tables``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import math

from text_tables import (
    ColumnSpec,
    DEFAULT_WIDTH,
    draw_boxed_table,
    draw_kv_table,
    ellipsize,
    money,
    pct,
)


def _fmt_cell(value: object, align: str = "left", width: int = 12) -> str:
    text = f"{value}"
    if align == "right":
        return text.rjust(width)
    return text.ljust(width)


def _currency_from(data: Mapping[str, Any]) -> str:
    for key in ("currency", "unit_currency", "price_currency"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "$"


def _safe_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _to_pairs(source: Mapping[str, Any] | Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    if isinstance(source, Mapping):
        return [(str(k), source[k]) for k in source]
    return [(str(k), v) for k, v in source]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_section(title: str, body: str) -> str:
    return f"{title}\n{body}"


def _render_summary(data: Mapping[str, Any]) -> str:
    summary: Mapping[str, Any] = data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else {}
    currency = _currency_from(summary or data)
    qty = _coerce_float(_safe_get(summary, "qty", "quantity")) or _coerce_float(data.get("qty")) or 1
    subtotal = _coerce_float(_safe_get(summary, "subtotal", "subtotal_after_overhead"))
    subtotal_before_margin = _coerce_float(
        _safe_get(summary, "subtotal_before_margin", "subtotal_pre_margin")
    )
    if subtotal_before_margin is None:
        subtotal_before_margin = subtotal
    margin_pct = _coerce_float(_safe_get(summary, "margin_pct", "margin_percent", "margin"))
    final_price = _coerce_float(_safe_get(summary, "final_price", "price"))
    unit_price = _coerce_float(summary.get("unit_price"))
    if unit_price is None and final_price is not None and qty:
        unit_price = final_price / max(qty, 1.0)

    lead_time = summary.get("lead_time") or data.get("lead_time")

    pairs: list[tuple[str, str]] = []
    pairs.append(("Quantity", f"{int(qty)}"))
    if unit_price is not None:
        pairs.append(("Unit Price", money(unit_price, currency)))
    if subtotal_before_margin is not None:
        pairs.append(("Subtotal (pre-margin)", money(subtotal_before_margin, currency)))
    if subtotal is not None and subtotal_before_margin != subtotal:
        pairs.append(("Subtotal (with adjustments)", money(subtotal, currency)))
    if margin_pct is not None:
        pairs.append(("Margin", pct(margin_pct)))
    if final_price is not None:
        pairs.append(("Final Price", money(final_price, currency)))
    if lead_time:
        pairs.append(("Lead Time", str(lead_time)))

    width_left = 46
    width_right = DEFAULT_WIDTH - width_left - 3
    return draw_kv_table(pairs, width_left, width_right)


def _render_price_drivers(data: Mapping[str, Any]) -> str | None:
    drivers = data.get("price_drivers")
    if not drivers:
        return None
    rows: list[tuple[str, str]] = []
    for entry in drivers:
        if isinstance(entry, Mapping):
            raw_label = (
                entry.get("label")
                or entry.get("name")
                or entry.get("title")
                or ""
            )
            label = ellipsize(str(raw_label), 80)
            detail = (
                entry.get("detail")
                or entry.get("text")
                or entry.get("value")
                or entry.get("reason")
                or ""
            )
        else:
            label = ellipsize(str(entry[0]), 80)
            detail = entry[1] if len(entry) > 1 else ""
        rows.append((label, str(detail)))
    left = 68
    right = DEFAULT_WIDTH - left - 3
    return draw_kv_table(rows, left, right, left_align="L", right_align="L")


def _render_cost_breakdown(data: Mapping[str, Any]) -> str | None:
    breakdown_source = data.get("cost_breakdown")
    if not breakdown_source:
        return None
    if isinstance(breakdown_source, Mapping):
        items = list(breakdown_source.items())
    else:
        items = list(breakdown_source)

    subtotal_before_margin = _coerce_float(
        _safe_get(data.get("summary", {}), "subtotal_before_margin", "subtotal_pre_margin")
    )
    if subtotal_before_margin is None:
        subtotal_before_margin = _coerce_float(data.get("subtotal_before_margin"))
    if subtotal_before_margin is None:
        subtotal_before_margin = sum(
            float(v) for _k, v in items if isinstance(v, (int, float))
        )

    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)

    rows: list[list[str]] = []
    for raw_label, raw_amount in items:
        label = ellipsize(str(raw_label), 60)
        amount = _coerce_float(raw_amount) or 0.0
        percent = 0.0
        if subtotal_before_margin:
            percent = amount / subtotal_before_margin
        rows.append([label, money(amount, currency), pct(percent)])

    specs = (
        ColumnSpec(62, "L"),
        ColumnSpec(28, "R"),
        ColumnSpec(DEFAULT_WIDTH - 62 - 28 - 4, "R"),
    )
    headers = ("Cost Element", "Amount", "% of Subtotal")
    return draw_boxed_table(headers, rows, specs)


def _render_quick_what_ifs(data: Mapping[str, Any]) -> str | None:
    margins = data.get("what_if_margins") or data.get("margin_scenarios") or []
    quantities = data.get("what_if_quantities") or data.get("quantity_scenarios") or []
    scenarios: list[tuple[str, str]] = []
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)

    def _append(entries: Sequence[Any], prefix: str) -> None:
        for entry in entries:
            if isinstance(entry, Mapping):
                label = entry.get("label") or entry.get("name")
                if not label and prefix:
                    label = f"{prefix} {entry.get(prefix.lower())}" if entry.get(prefix.lower()) else prefix
                value = entry.get("price") or entry.get("value") or entry.get("amount")
            else:
                label = None
                value = None
                if isinstance(entry, Sequence) and entry:
                    label = entry[0]
                    if len(entry) > 1:
                        value = entry[1]
            if not label:
                continue
            if value is None:
                formatted_value = "—"
            else:
                numeric = _coerce_float(value)
                formatted_value = money(numeric, currency) if numeric is not None else str(value)
            scenarios.append((str(label), formatted_value))

    _append(margins, "Margin")
    _append(quantities, "Qty")

    if not scenarios:
        return None

    left = 50
    right = DEFAULT_WIDTH - left - 3
    return draw_kv_table(scenarios, left, right, left_align="L", right_align="R")


def _render_nre(data: Mapping[str, Any]) -> str | None:
    entries = data.get("nre") or data.get("nre_items")
    if not entries:
        return None
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)
    rows: list[list[str]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            label = entry.get("label") or entry.get("name") or ""
            detail = entry.get("detail") or entry.get("notes") or ""
            amount = _coerce_float(entry.get("amount"))
        else:
            label = entry[0] if entry else ""
            detail = entry[1] if len(entry) > 1 else ""
            amount = _coerce_float(entry[2]) if len(entry) > 2 else None
        rows.append(
            [
                ellipsize(str(label), 44),
                ellipsize(str(detail), 38) if detail else "",
                money(amount, currency),
            ]
        )
    specs = (
        ColumnSpec(44, "L"),
        ColumnSpec(40, "L"),
        ColumnSpec(DEFAULT_WIDTH - 44 - 40 - 5, "R"),
    )
    headers = ("Activity", "Detail", "Amount (per lot)")
    return draw_boxed_table(headers, rows, specs)


def _render_materials(data: Mapping[str, Any]) -> str | None:
    entries = data.get("materials") or data.get("material_summary")
    if not entries:
        return None
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)
    rows: list[list[str]] = []

    def _scrap_detail(entry: Mapping[str, Any], components: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for key in ("scrap_credit_mass_lb", "scrap_mass_lb", "scrap_weight_lb"):
            scrap_mass = _coerce_float(entry.get(key)) if isinstance(entry, Mapping) else None
            if scrap_mass is not None and scrap_mass > 0:
                parts.append(f"{scrap_mass:.2f} lb")
                break
        rate_text = components.get("scrap_rate_text") if isinstance(components, Mapping) else None
        if rate_text:
            parts.append(str(rate_text))
        return "; ".join(part for part in parts if part)

    for entry in entries:
        if isinstance(entry, Mapping):
            label = entry.get("label") or entry.get("name") or ""
            detail = entry.get("detail") or entry.get("spec") or entry.get("notes") or ""
            amount = _coerce_float(entry.get("amount"))
            components = (
                entry.get("material_cost_components")
                if isinstance(entry.get("material_cost_components"), Mapping)
                else None
            )
        else:
            label = entry[0] if entry else ""
            detail = entry[1] if len(entry) > 1 else ""
            amount = _coerce_float(entry[2]) if len(entry) > 2 else None
            components = None

        label_text = ellipsize(str(label), 40)
        detail_text = ellipsize(str(detail), 44) if detail else ""

        displayed_amount = amount
        if displayed_amount is None and isinstance(components, Mapping):
            for key in ("net_usd", "total_usd", "base_usd"):
                candidate = _coerce_float(components.get(key))
                if candidate is not None:
                    displayed_amount = candidate
                    break

        rows.append([
            label_text,
            detail_text,
            money(displayed_amount, currency),
        ])

        if not isinstance(components, Mapping):
            continue

        base_amount = None
        for key in ("stock_piece_usd", "base_usd"):
            candidate = _coerce_float(components.get(key))
            if candidate is not None:
                base_amount = candidate
                break
        if base_amount is not None and base_amount > 0:
            base_detail = components.get("stock_source") or components.get("base_source") or ""
            rows.append([
                ellipsize("  Stock piece", 40),
                ellipsize(str(base_detail), 44) if base_detail else "",
                money(base_amount, currency),
            ])

        tax_amount = _coerce_float(components.get("tax_usd"))
        if tax_amount is not None and abs(tax_amount) > 1e-9:
            rows.append([
                ellipsize("  Material tax", 40),
                "",
                money(tax_amount, currency),
            ])

        scrap_credit = _coerce_float(components.get("scrap_credit_usd"))
        if scrap_credit is not None and scrap_credit > 0:
            scrap_detail = _scrap_detail(entry if isinstance(entry, Mapping) else {}, components)
            rows.append([
                ellipsize("  Scrap credit", 40),
                ellipsize(scrap_detail, 44) if scrap_detail else "",
                money(-scrap_credit, currency),
            ])

        net_amount = _coerce_float(components.get("net_usd"))
        total_amount = _coerce_float(components.get("total_usd"))
        if net_amount is not None:
            rows.append([
                ellipsize("  Net material", 40),
                "",
                money(net_amount, currency),
            ])
        if (
            total_amount is not None
            and net_amount is not None
            and not math.isclose(total_amount, net_amount, rel_tol=1e-9, abs_tol=1e-9)
        ):
            rows.append([
                ellipsize("  Total (with tax)", 40),
                "",
                money(total_amount, currency),
            ])
    specs = (
        ColumnSpec(40, "L"),
        ColumnSpec(46, "L"),
        ColumnSpec(DEFAULT_WIDTH - 40 - 46 - 5, "R"),
    )
    headers = ("Material", "Detail", "Amount (per part)")
    return draw_boxed_table(headers, rows, specs)


def _render_processes(data: Mapping[str, Any]) -> str | None:
    entries = data.get("processes") or data.get("process_summary")
    if not entries:
        return None
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)
    rows: list[list[str]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            label = entry.get("label") or entry.get("name") or ""
            hours = _coerce_float(entry.get("hours"))
            rate = _coerce_float(entry.get("rate"))
            rate_display = entry.get("rate_display")
            amount = _coerce_float(entry.get("amount"))
        else:
            label = entry[0] if entry else ""
            hours = _coerce_float(entry[1]) if len(entry) > 1 else None
            rate = _coerce_float(entry[2]) if len(entry) > 2 else None
            amount = _coerce_float(entry[3]) if len(entry) > 3 else None
            rate_display = None
        label_text = _fmt_cell(ellipsize(str(label), 28), "left", 28)
        hours_text = _fmt_cell(f"{hours:.2f} hr" if hours is not None else "—", "right", 12)
        if rate_display:
            rate_text = _fmt_cell(str(rate_display), "right", 20)
        else:
            rate_text = _fmt_cell(
                money(rate, currency) + "/hr" if rate is not None else "—",
                "right",
                20,
            )
        amount_text = _fmt_cell(money(amount, currency), "right", 22)
        rows.append([label_text, hours_text, rate_text, amount_text])
    specs = (
        ColumnSpec(28, "L"),
        ColumnSpec(12, "R"),
        ColumnSpec(20, "R"),
        ColumnSpec(22, "R"),
    )
    headers = (
        _fmt_cell("Process", "left", 28),
        _fmt_cell("Hours", "right", 12),
        _fmt_cell("Rate", "right", 20),
        _fmt_cell("Amount (per part)", "right", 22),
    )
    return draw_boxed_table(headers, rows, specs)


def _render_cycle_reference(data: Mapping[str, Any]) -> str | None:
    entries = data.get("cycle_time_reference") or data.get("cycle_reference")
    if not entries:
        return None
    rows: list[list[str]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            label = entry.get("label") or entry.get("name") or ""
            time_val = _coerce_float(entry.get("time")) or _coerce_float(entry.get("hours"))
            notes = entry.get("notes") or entry.get("detail") or ""
        else:
            label = entry[0] if entry else ""
            time_val = _coerce_float(entry[1]) if len(entry) > 1 else None
            notes = entry[2] if len(entry) > 2 else ""
        rows.append(
            [
                ellipsize(str(label), 44),
                f"{time_val:.2f} hr" if time_val is not None else "—",
                ellipsize(str(notes), 40) if notes else "",
            ]
        )
    specs = (
        ColumnSpec(44, "L"),
        ColumnSpec(18, "R"),
        ColumnSpec(DEFAULT_WIDTH - 44 - 18 - 4, "L"),
    )
    headers = ("Activity", "Cycle Time", "Notes")
    return draw_boxed_table(headers, rows, specs)


def _render_top_cycle_contributors(data: Mapping[str, Any]) -> str | None:
    entries = data.get("top_cycle_time") or data.get("cycle_top") or data.get("cycle_contributors")
    if not entries:
        return None

    def _coerce_minutes(entry: Mapping[str, Any] | Sequence[Any]) -> float | None:
        if isinstance(entry, Mapping):
            for key in (
                "group_min",
                "minutes",
                "minutes_total",
                "total_minutes",
            ):
                minutes_val = _coerce_float(entry.get(key))
                if minutes_val is not None and minutes_val > 0:
                    return minutes_val
            hours_val = _coerce_float(entry.get("hours"))
            if hours_val is not None and hours_val > 0:
                return hours_val * 60.0
            qty = _coerce_float(entry.get("qty"))
            per_hole = _coerce_float(entry.get("t_per_hole_min")) or _coerce_float(
                entry.get("minutes_per_hole")
            )
            if qty and per_hole:
                return qty * per_hole
        else:
            if len(entry) > 2:
                minutes_val = _coerce_float(entry[2])
                if minutes_val is not None and minutes_val > 0:
                    return minutes_val
            if len(entry) > 1:
                hours_val = _coerce_float(entry[1])
                if hours_val is not None and hours_val > 0:
                    return hours_val * 60.0
        return None

    processed: list[tuple[str, float | None, str]] = []
    for raw in entries:
        if isinstance(raw, Mapping):
            label = (
                raw.get("group_label")
                or raw.get("label")
                or raw.get("group")
                or raw.get("op_name")
                or raw.get("name")
                or ""
            )
            qty = _coerce_float(raw.get("qty"))
            diameter = _coerce_float(raw.get("diameter_in"))
            depth = _coerce_float(raw.get("depth_in"))
            details: list[str] = []
            if qty is not None and qty > 0:
                qty_int = int(qty) if abs(qty - int(qty)) < 1e-6 else qty
                details.append(f"{qty_int} holes")
            if diameter is not None and diameter > 0:
                details.append(f"Ø {diameter:.3f}\"")
            if depth is not None and depth > 0:
                details.append(f"depth {depth:.2f}\"")
            detail_text = ", ".join(details)
        else:
            label = raw[0] if raw else ""
            detail_text = ""
        minutes_val = _coerce_minutes(raw)
        processed.append((str(label), minutes_val, detail_text))

    processed = [item for item in processed if item[0].strip()]
    if not processed:
        return None

    processed.sort(key=lambda item: item[1] or 0.0, reverse=True)
    rows: list[list[str]] = []
    for label, minutes_val, detail_text in processed[:5]:
        left_text = label
        if detail_text:
            left_text = f"{label} — {detail_text}" if label else detail_text
        time_display = "—"
        if minutes_val is not None and minutes_val > 0:
            if minutes_val >= 120:
                hours = minutes_val / 60.0
                time_display = f"{hours:.2f} hr"
            else:
                time_display = f"{minutes_val:.1f} min"
        rows.append([ellipsize(left_text, 76), time_display])

    left = 76
    right = DEFAULT_WIDTH - left - 3
    return draw_kv_table(rows, left, right, left_align="L", right_align="R")


def _render_traceability(data: Mapping[str, Any]) -> str | None:
    entries = data.get("traceability")
    if not entries:
        return None
    if isinstance(entries, Mapping):
        pairs = _to_pairs(entries)
    else:
        pairs = [(str(item[0]), item[1]) if isinstance(item, Sequence) and len(item) >= 2 else (str(item), "") for item in entries]
    formatted = [(ellipsize(str(k), 46), ellipsize(str(v), 60)) for k, v in pairs]
    return draw_kv_table(formatted, 46, DEFAULT_WIDTH - 46 - 3, left_align="L", right_align="L")


def render_quote(data: Mapping[str, Any]) -> str:
    """Render the quote information as a series of boxed ASCII tables."""

    sections: list[str] = []

    sections.append(_render_section("Quote Summary", _render_summary(data)))

    price_drivers = _render_price_drivers(data)
    if price_drivers:
        sections.append(_render_section("Price Drivers", price_drivers))

    cost_breakdown = _render_cost_breakdown(data)
    if cost_breakdown:
        sections.append(_render_section("Cost Breakdown (% of subtotal)", cost_breakdown))

    quick = _render_quick_what_ifs(data)
    if quick:
        sections.append(_render_section("Quick What-Ifs (margin + qty)", quick))

    nre = _render_nre(data)
    if nre:
        sections.append(_render_section("NRE (per lot)", nre))

    materials = _render_materials(data)
    if materials:
        sections.append(_render_section("Materials & Stock (rolled up)", materials))

    processes = _render_processes(data)
    if processes:
        sections.append(_render_section("Process & Labor (per part)", processes))

    cycle_reference = _render_cycle_reference(data)
    if cycle_reference:
        sections.append(_render_section("Cycle-Time Reference", cycle_reference))

    cycle_top = _render_top_cycle_contributors(data)
    if cycle_top:
        sections.append(_render_section("Top Cycle-Time Contributors (Top 5)", cycle_top))

    trace = _render_traceability(data)
    if trace:
        sections.append(_render_section("Traceability", trace))

    return "\n\n".join(sections)


__all__ = ["render_quote"]

