"""High-level renderer for the compact quote summary email.

The pricing engine produces a fairly rich dictionary containing final pricing,
driver information, and supplemental metadata.  This module translates that
plain data into a deterministic, email-safe ASCII presentation built on top of
``text_tables``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cad_quoter.utils.render_utils import format_weight_lb_decimal, format_weight_lb_oz
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
    bullets: list[str] = []
    for entry in drivers:
        if isinstance(entry, Mapping):
            label = str(entry.get("label") or entry.get("name") or "").strip()
            detail = str(entry.get("detail") or entry.get("value") or entry.get("reason") or "").strip()
        else:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                label = str(entry[0]).strip() if entry else ""
                detail = str(entry[1]).strip() if len(entry) > 1 else ""
            else:
                label = str(entry).strip()
                detail = ""
        bullet_body = detail or label
        if not bullet_body:
            continue
        if label and detail and detail != bullet_body:
            bullet_body = f"{label}: {detail}"
        bullets.append(f"- {ellipsize(bullet_body, DEFAULT_WIDTH - 4)}")
    return "\n".join(bullets) if bullets else None


def _render_cost_breakdown(data: Mapping[str, Any]) -> str | None:
    summary = data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else {}
    currency = _currency_from(summary or data)
    subtotal_before_margin = _coerce_float(
        _safe_get(summary, "subtotal_before_margin", "subtotal_pre_margin")
    )
    materials_entries = data.get("materials")
    direct_materials = 0.0
    if isinstance(materials_entries, Sequence):
        for entry in materials_entries:
            if not isinstance(entry, Mapping):
                continue
            amount = _coerce_float(entry.get("amount"))
            if amount is None:
                continue
            detail = str(entry.get("detail") or "").lower()
            if "pass-through" in detail:
                continue
            direct_materials += amount
    cost_breakdown_source = data.get("cost_breakdown")
    if direct_materials <= 0 and cost_breakdown_source:
        if isinstance(cost_breakdown_source, Mapping):
            direct_materials = _coerce_float(cost_breakdown_source.get("Direct Costs")) or 0.0
        else:
            for label, value in cost_breakdown_source:
                if str(label).strip().lower().startswith("direct"):
                    direct_materials = _coerce_float(value) or 0.0
                    break

    processes_entries = data.get("processes")
    direct_labor = 0.0
    if isinstance(processes_entries, Sequence):
        for entry in processes_entries:
            if not isinstance(entry, Mapping):
                continue
            amount = _coerce_float(entry.get("amount"))
            if amount is None:
                continue
            direct_labor += amount

    subtotal_basis = subtotal_before_margin
    if subtotal_basis is None or subtotal_basis <= 0:
        subtotal_basis = direct_materials + direct_labor

    margin_pct = _coerce_float(_safe_get(summary, "margin_pct", "margin_percent"))
    final_price = _coerce_float(_safe_get(summary, "final_price", "price"))
    if final_price is None and subtotal_before_margin is not None and margin_pct is not None:
        final_price = subtotal_before_margin * (1.0 + margin_pct)

    margin_amount = None
    if final_price is not None and subtotal_basis:
        margin_amount = final_price - subtotal_basis
    elif margin_pct is not None and subtotal_basis:
        margin_amount = subtotal_basis * margin_pct

    def _ratio(amount: float | None) -> float | None:
        if amount is None:
            return None
        if not subtotal_basis:
            return None
        return amount / subtotal_basis if subtotal_basis else None

    rows: list[list[str]] = []
    rows.append([
        "Direct Materials",
        money(direct_materials, currency),
        pct(_ratio(direct_materials)),
    ])
    rows.append([
        "Direct Labor (incl. programming)",
        money(direct_labor, currency),
        pct(_ratio(direct_labor)),
    ])
    margin_label = "Margin"
    if margin_pct is not None:
        margin_label = f"Margin ({pct(margin_pct)})"
    rows.append([
        margin_label,
        money(margin_amount, currency),
        pct(_ratio(margin_amount)),
    ])
    rows.append([
        "Final Price",
        money(final_price, currency),
        pct(_ratio(final_price)),
    ])

    specs = (
        ColumnSpec(62, "L"),
        ColumnSpec(28, "R"),
        ColumnSpec(DEFAULT_WIDTH - 62 - 28 - 4, "R"),
    )
    headers = ("Cost Element", "Amount", "% of Subtotal")
    return draw_boxed_table(headers, rows, specs)


def _render_quick_what_ifs(data: Mapping[str, Any]) -> str | None:
    summary = data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else {}
    currency = _currency_from(summary or data)
    qty = _coerce_float(_safe_get(summary, "qty", "quantity")) or 1.0
    subtotal_before_margin = _coerce_float(
        _safe_get(summary, "subtotal_before_margin", "subtotal_pre_margin")
    )
    unit_subtotal = None
    if subtotal_before_margin is not None and qty:
        unit_subtotal = subtotal_before_margin / qty

    sections: list[str] = []

    margin_rows: list[tuple[str, str]] = []
    for pct_value in (0.10, 0.15, 0.20, 0.25):
        if unit_subtotal is None:
            formatted_value = "—"
        else:
            price = unit_subtotal * (1.0 + pct_value)
            formatted_value = money(price, currency)
        margin_rows.append((f"Margin {pct(pct_value)}", formatted_value))
    margin_table = draw_kv_table(
        margin_rows,
        40,
        DEFAULT_WIDTH - 40 - 3,
        left_align="L",
        right_align="R",
    )
    sections.append("Margin Slider (Qty=1)\n" + margin_table)

    quantity_entries = data.get("what_if_quantities") or data.get("quantity_scenarios") or []
    qty_rows: list[list[str]] = []
    if isinstance(quantity_entries, Sequence):
        for entry in quantity_entries:
            qty_label: str | None = None
            price_value: float | str | None = None
            if isinstance(entry, Mapping):
                qty_label = entry.get("label") or entry.get("name")
                if not qty_label:
                    qty_value = entry.get("qty") or entry.get("quantity")
                    if qty_value is not None:
                        qty_label = f"Qty {qty_value}"
                price_value = (
                    entry.get("unit_price")
                    or entry.get("price")
                    or entry.get("amount")
                    or entry.get("value")
                )
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                if entry:
                    qty_label = str(entry[0])
                if len(entry) > 1:
                    price_value = entry[1]
            else:
                qty_label = str(entry)
            if not qty_label:
                continue
            if price_value is None:
                formatted_value = "—"
            else:
                numeric = _coerce_float(price_value)
                formatted_value = (
                    money(numeric, currency) if numeric is not None else str(price_value)
                )
            qty_rows.append([ellipsize(str(qty_label), 40), formatted_value])
    if qty_rows:
        specs = (
            ColumnSpec(42, "L"),
            ColumnSpec(DEFAULT_WIDTH - 42 - 3, "R"),
        )
        qty_table = draw_boxed_table(("Quantity", "Unit Price"), qty_rows, specs)
        sections.append("Qty Breaks\n" + qty_table)

    return "\n\n".join(sections) if sections else None


def _render_programming_nre(data: Mapping[str, Any]) -> str | None:
    mode = str(data.get("programming_mode") or "").strip().lower()
    if mode != "amortized":
        return None
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)
    per_lot_amount = _coerce_float(data.get("programming_per_lot"))
    detail_text = ""
    nre_entries = data.get("nre") or data.get("nre_items")
    if isinstance(nre_entries, Sequence):
        for entry in nre_entries:
            if isinstance(entry, Mapping):
                label = str(entry.get("label") or entry.get("name") or "").lower()
                if "program" in label:
                    detail_text = str(entry.get("detail") or entry.get("notes") or "").strip()
                    if per_lot_amount is None:
                        per_lot_amount = _coerce_float(entry.get("amount"))
                    break
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and entry:
                label = str(entry[0]).lower()
                if "program" in label:
                    if len(entry) > 1:
                        detail_text = str(entry[1]).strip()
                    if per_lot_amount is None and len(entry) > 2:
                        per_lot_amount = _coerce_float(entry[2])
                    break
    if per_lot_amount is None:
        return None
    rows = [
        (
            "Programming & Eng (per lot)",
            money(per_lot_amount, currency),
        )
    ]
    table = draw_kv_table(rows, 56, DEFAULT_WIDTH - 56 - 3, left_align="L", right_align="R")
    if detail_text:
        return table + f"\nDetail: {ellipsize(detail_text, DEFAULT_WIDTH - 8)}"
    return table


def _render_materials(data: Mapping[str, Any]) -> str | None:
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)
    cost_components = data.get("material_cost_components")
    weight_summary = data.get("material_weight")
    stock_meta = data.get("material_stock")
    if not any((cost_components, weight_summary, stock_meta)):
        return None

    rows: list[list[str]] = []
    if isinstance(cost_components, Mapping):
        base_amount = _coerce_float(cost_components.get("base_usd"))
        stock_piece_amount = _coerce_float(cost_components.get("stock_piece_usd"))
        tax_amount = _coerce_float(cost_components.get("tax_usd"))
        scrap_credit = _coerce_float(cost_components.get("scrap_credit_usd"))
        total_amount = _coerce_float(cost_components.get("total_usd"))
        base_source = str(cost_components.get("base_source") or cost_components.get("stock_source") or "").strip()

        if base_amount is not None:
            label = "Stock Piece" if stock_piece_amount is not None else "Base Material"
            if base_source:
                label = f"{label} ({base_source})"
            detail = ""
            if isinstance(stock_meta, Mapping):
                dims = []
                for key in ("stock_L_in", "stock_W_in", "stock_T_in"):
                    dims.append(_coerce_float(stock_meta.get(key)))
                dims_clean = [f"{float(val):.2f}" for val in dims if val is not None]
                if len(dims_clean) == 3:
                    detail = " × ".join(dims_clean[:2]) + f" × {dims_clean[2]} in"
            rows.append([
                ellipsize(label, 40),
                ellipsize(detail, 44),
                money(base_amount, currency),
            ])
        if tax_amount:
            rows.append([
                "Material Tax",
                "",
                money(tax_amount, currency),
            ])
        if scrap_credit and scrap_credit > 0:
            scrap_detail = str(cost_components.get("scrap_rate_text") or "").strip()
            amount_display = money(-scrap_credit, currency)
            rows.append([
                "Scrap Credit",
                ellipsize(scrap_detail, 44),
                amount_display,
            ])
        if total_amount is not None:
            rows.append([
                "Net Material Cost",
                "",
                money(total_amount, currency),
            ])

    if not rows:
        return None

    specs = (
        ColumnSpec(40, "L"),
        ColumnSpec(46, "L"),
        ColumnSpec(DEFAULT_WIDTH - 40 - 46 - 5, "R"),
    )
    headers = ("Material", "Detail", "Amount (per part)")
    table = draw_boxed_table(headers, rows, specs)

    weight_lines: list[str] = []
    if isinstance(weight_summary, Mapping):
        starting = _coerce_float(weight_summary.get("starting_mass_g"))
        net = _coerce_float(weight_summary.get("net_mass_g"))
        scrap_mass = _coerce_float(weight_summary.get("scrap_mass_g"))
        parts: list[str] = []
        if starting:
            parts.append(f"Start {format_weight_lb_oz(starting)}")
        if net:
            parts.append(f"Net {format_weight_lb_oz(net)}")
        if scrap_mass is not None:
            parts.append(f"Scrap {format_weight_lb_oz(scrap_mass)}")
        scrap_hint = str(weight_summary.get("scrap_hint") or "").strip()
        if parts:
            line = "Weight Reference: " + " | ".join(parts)
            if scrap_hint:
                line += f" ({scrap_hint})"
            weight_lines.append(line)
        geometry_pct = _coerce_float(weight_summary.get("scrap_pct_geometry"))
        computed_pct = _coerce_float(weight_summary.get("scrap_pct_computed"))
        entered_pct = _coerce_float(weight_summary.get("scrap_pct_entered"))
        if computed_pct is not None:
            extra = f"Computed Scrap: {pct(computed_pct)}"
            if scrap_hint:
                extra += f" ({scrap_hint})"
            weight_lines.append(extra)
        elif entered_pct is not None:
            extra = f"Scrap Percentage: {pct(entered_pct)}"
            if scrap_hint:
                extra += f" ({scrap_hint})"
            weight_lines.append(extra)
        if geometry_pct is not None:
            hint = f"Geometry Hint: {pct(geometry_pct)}"
            if scrap_hint:
                hint += f" ({scrap_hint})"
            weight_lines.append(hint)

    extra_lines: list[str] = []
    price_lines = data.get("material_price_lines")
    if isinstance(price_lines, Sequence):
        extra_lines.extend(str(line) for line in price_lines if line)
    if weight_lines:
        extra_lines.extend(weight_lines)
    if extra_lines:
        return table + "\n" + "\n".join(extra_lines)
    return table


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
        hours_text = _fmt_cell(f"{hours:.2f} hr" if hours is not None else "—", "right", 10)
        if rate_display:
            rate_text = _fmt_cell(str(rate_display), "right", 16)
        else:
            rate_text = _fmt_cell(
                money(rate, currency) + "/hr" if rate is not None else "—",
                "right",
                16,
            )
        amount_text = _fmt_cell(money(amount, currency), "right", 18)
        rows.append([label_text, hours_text, rate_text, amount_text])
    total_amount = sum(
        _coerce_float(entry.get("amount")) or 0.0
        for entry in entries
        if isinstance(entry, Mapping)
    )
    if total_amount:
        rows.append(
            [
                _fmt_cell("Total", "left", 28),
                _fmt_cell("", "right", 10),
                _fmt_cell("", "right", 16),
                _fmt_cell(money(total_amount, currency), "right", 18),
            ]
        )
    specs = (
        ColumnSpec(28, "L"),
        ColumnSpec(10, "R"),
        ColumnSpec(16, "R"),
        ColumnSpec(18, "R"),
    )
    headers = (
        _fmt_cell("Process", "left", 28),
        _fmt_cell("Hours", "right", 10),
        _fmt_cell("Rate", "right", 16),
        _fmt_cell("Amount (per part)", "right", 18),
    )
    return draw_boxed_table(headers, rows, specs)


def _render_cycle_reference(data: Mapping[str, Any]) -> str | None:
    entries = data.get("cycle_time_metrics")
    if not entries:
        return None
    rows: list[list[str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        label = ellipsize(str(entry.get("label") or entry.get("name") or ""), 44)
        planning_minutes = _coerce_float(entry.get("planning_minutes"))
        billed_minutes = _coerce_float(entry.get("billed_minutes"))
        billed_hours = _coerce_float(entry.get("billed_hours"))
        if billed_hours is None and billed_minutes is not None:
            billed_hours = billed_minutes / 60.0
        rows.append(
            [
                label,
                f"{planning_minutes:.1f} min" if planning_minutes is not None else "—",
                f"{billed_hours:.2f} hr" if billed_hours is not None else "—",
            ]
        )
    if not rows:
        return None
    specs = (
        ColumnSpec(44, "L"),
        ColumnSpec(24, "R"),
        ColumnSpec(DEFAULT_WIDTH - 44 - 24 - 4, "R"),
    )
    headers = ("Activity", "Planning Minutes", "Chargeable Hours")
    return draw_boxed_table(headers, rows, specs)


def _render_top_cycle_contributors(data: Mapping[str, Any]) -> str | None:
    entries = data.get("top_cycle_time") or data.get("cycle_top") or data.get("cycle_contributors")
    if not entries:
        return None
    rows: list[list[str]] = []
    for entry in entries[:5]:
        if isinstance(entry, Mapping):
            label = entry.get("label") or entry.get("name") or ""
            hours = _coerce_float(entry.get("hours"))
        else:
            label = entry[0] if entry else ""
            hours = _coerce_float(entry[1]) if len(entry) > 1 else None
        rows.append([ellipsize(str(label), 76), f"{hours:.2f} hr" if hours is not None else "—"])
    left = 76
    right = DEFAULT_WIDTH - left - 3
    return draw_kv_table(rows, left, right, left_align="L", right_align="R")


def _render_drill_groups(data: Mapping[str, Any]) -> str | None:
    groups = data.get("drill_groups")
    if not isinstance(groups, Sequence):
        return None
    rows: list[list[str]] = []
    for entry in groups:
        if not isinstance(entry, Mapping):
            continue
        label = ellipsize(str(entry.get("op") or entry.get("op_name") or ""), 36)
        qty = _coerce_float(entry.get("qty"))
        per_hole = _coerce_float(entry.get("t_per_hole_min"))
        total_minutes = _coerce_float(entry.get("minutes_total"))
        rows.append(
            [
                label,
                f"{qty:.0f}" if qty is not None else "—",
                f"{per_hole:.2f} min" if per_hole is not None else "—",
                f"{total_minutes:.2f} min" if total_minutes is not None else "—",
            ]
        )
    if not rows:
        return None
    specs = (
        ColumnSpec(38, "L"),
        ColumnSpec(12, "R"),
        ColumnSpec(20, "R"),
        ColumnSpec(DEFAULT_WIDTH - 38 - 12 - 20 - 5, "R"),
    )
    headers = ("Drill Group", "Qty", "Minutes / Hole", "Group Minutes")
    return draw_boxed_table(headers, rows, specs)


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
        sections.append(_render_section("Price Drivers & Assumptions", price_drivers))

    cost_breakdown = _render_cost_breakdown(data)
    if cost_breakdown:
        sections.append(_render_section("Cost Breakdown (% of subtotal)", cost_breakdown))

    quick = _render_quick_what_ifs(data)
    if quick:
        sections.append(_render_section("Quick What-Ifs", quick))

    programming = _render_programming_nre(data)
    if programming:
        sections.append(_render_section("NRE / Programming (per lot)", programming))

    materials = _render_materials(data)
    if materials:
        sections.append(_render_section("Materials & Stock (rolled up)", materials))

    processes = _render_processes(data)
    if processes:
        sections.append(_render_section("Process & Labor (per part — chargeable view)", processes))

    cycle_reference = _render_cycle_reference(data)
    if cycle_reference:
        sections.append(_render_section("Cycle-Time Reference", cycle_reference))

    drill_groups = _render_drill_groups(data)
    if drill_groups:
        sections.append(_render_section("Cycle-Time — Drill Groups", drill_groups))

    trace = _render_traceability(data)
    if trace:
        sections.append(_render_section("Traceability", trace))

    return "\n\n".join(sections)


__all__ = ["render_quote"]

