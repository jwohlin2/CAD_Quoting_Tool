"""High-level renderer for the compact quote summary email.

The pricing engine produces a fairly rich dictionary containing final pricing,
driver information, and supplemental metadata.  This module translates that
plain data into a deterministic, email-safe ASCII presentation built on top of
``text_tables``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from text_tables import (
    ascii_table,
    money,
    pct,
    ellipsize,
    draw_boxed_table,
    draw_kv_table,
    ColumnSpec,
    DEFAULT_WIDTH,
)

LB_PER_KG = 2.2046226218

# Used to surface a gentle hint when the pricing engine
# produced a zero materials figure that likely indicates
# missing inputs rather than truly free material.
MATERIALS_WARNING_LABEL = "Note: Materials omitted (engine reported zero)"


def _currency_from(data: Mapping[str, Any]) -> str:
    for key in ("currency", "unit_currency", "price_currency"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "$"


def _format_weight_lb_oz_text(mass_g: Any) -> str:
    grams = _coerce_float(mass_g)
    if grams is None or grams <= 0:
        return "0 oz"
    pounds_total = grams / 1000.0 * LB_PER_KG
    total_ounces = pounds_total * 16.0
    pounds = int(total_ounces // 16)
    ounces = total_ounces - pounds * 16
    precision = 1 if pounds > 0 or ounces >= 1.0 else 2
    ounces = round(ounces, precision)
    if ounces >= 16.0:
        pounds += 1
        ounces = 0.0
    parts: list[str] = []
    if pounds > 0:
        parts.append(f"{pounds} lb" if pounds != 1 else "1 lb")
    if ounces > 0 or pounds == 0:
        ounce_text = f"{ounces:.{precision}f}".rstrip("0").rstrip(".")
        if not ounce_text:
            ounce_text = "0"
        parts.append(f"{ounce_text} oz")
    return " ".join(parts) if parts else "0 oz"


def _format_percent_text(value: Any) -> str | None:
    pct_val = _coerce_float(value)
    if pct_val is None:
        return None
    if pct_val <= 1.0:
        pct_val *= 100.0
    return f"{pct_val:.1f}%"


def _normalize_vendor_label(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if "mcmaster" in cleaned.lower():
        return "McMaster"
    return cleaned


def _format_stock_line(stock_meta: Any) -> str | None:
    if not isinstance(stock_meta, Mapping):
        return None
    dims_raw = stock_meta.get("stock_dims_in")
    length = width = thickness = None
    if isinstance(dims_raw, Sequence) and len(dims_raw) >= 3:
        try:
            length = float(dims_raw[0])
            width = float(dims_raw[1])
            thickness = float(dims_raw[2])
        except Exception:
            length = width = thickness = None
    if length is None:
        length = _coerce_float(stock_meta.get("stock_L_in"))
    if width is None:
        width = _coerce_float(stock_meta.get("stock_W_in"))
    if thickness is None:
        thickness = _coerce_float(stock_meta.get("stock_T_in"))
    dims_text = None
    if length is not None and width is not None and thickness is not None:
        dims_text = f"{length:.2f} × {width:.2f} × {thickness:.3f} in"
    else:
        display = stock_meta.get("stock_size_display")
        if isinstance(display, str) and display.strip():
            dims_text = display.strip()
    vendor = ""
    for key in ("stock_source_tag", "source"):
        raw = stock_meta.get(key)
        if isinstance(raw, str) and raw.strip():
            vendor = _normalize_vendor_label(raw)
            if vendor:
                break
    part = stock_meta.get("mcmaster_part")
    part_text = part.strip() if isinstance(part, str) else ""
    extras = [text for text in (vendor, part_text) if text]
    if dims_text:
        if extras and "(" not in dims_text:
            dims_text = f"{dims_text} ({', '.join(extras)})"
        return f"Stock used: {dims_text}"
    if extras:
        return f"Stock used: {', '.join(extras)}"
    return None


def _format_weight_lines(weight_summary: Any) -> list[str]:
    if not isinstance(weight_summary, Mapping):
        return []
    lines: list[str] = []
    start_mass = _coerce_float(weight_summary.get("starting_mass_g"))
    if start_mass is not None and start_mass > 0:
        lines.append(f"Weight Reference: Start {_format_weight_lb_oz_text(start_mass)}")
    net_mass = _coerce_float(weight_summary.get("net_mass_g"))
    if net_mass is not None and net_mass > 0:
        lines.append(f"Net {_format_weight_lb_oz_text(net_mass)}")
    scrap_mass = _coerce_float(weight_summary.get("scrap_mass_g"))
    if scrap_mass is not None and scrap_mass > 0:
        lines.append(f"Scrap {_format_weight_lb_oz_text(scrap_mass)}")
    hint_raw = weight_summary.get("scrap_hint")
    hint_text = ""
    if isinstance(hint_raw, str) and hint_raw.strip():
        hint_text = f" ({hint_raw.strip()})"
    computed_pct_text = _format_percent_text(weight_summary.get("scrap_pct_computed"))
    if computed_pct_text:
        lines.append(f"Computed Scrap: {computed_pct_text}{hint_text}")
    geometry_pct_text = _format_percent_text(weight_summary.get("scrap_pct_geometry"))
    if geometry_pct_text:
        lines.append(f"Geometry Hint: {geometry_pct_text}{hint_text}")
    return lines


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

    headers = ["Quantity", f"{qty:,.0f}"]
    rows: list[list[str]] = []
    if unit_price is not None:
        rows.append(["Unit Price", money(unit_price, currency)])
    if subtotal_before_margin is not None:
        rows.append(["Subtotal (pre-margin)", money(subtotal_before_margin, currency)])
    if subtotal is not None and subtotal_before_margin != subtotal:
        rows.append(["Subtotal (with adjustments)", money(subtotal, currency)])
    if margin_pct is not None:
        rows.append(["Margin", pct(margin_pct)])
    if final_price is not None:
        rows.append(["Final Price", money(final_price, currency)])
    if lead_time:
        rows.append(["Lead Time", str(lead_time)])

    return ascii_table(
        headers=headers,
        rows=rows,
        col_widths=[46, 25],
        col_aligns=["left", "right"],
        header_aligns=["left", "right"],
    )


def _render_price_drivers(data: Mapping[str, Any]) -> str | None:
    drivers = data.get("price_drivers")
    if not drivers:
        return None
    rows: list[list[str]] = []
    for entry in drivers:
        if isinstance(entry, Mapping):
            label = str(entry.get("label") or entry.get("name") or "")
            detail = entry.get("detail") or entry.get("value") or entry.get("reason") or ""
        else:
            label = str(entry[0])
            detail = entry[1] if len(entry) > 1 else ""
        rows.append([label, str(detail)])

    return ascii_table(
        headers=["Driver", "Detail"],
        rows=rows,
        col_widths=[40, 51],
        col_aligns=["left", "left"],
        header_aligns=["left", "left"],
    )


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

    # Build source items for the breakdown table.
    # Prefer an explicit "cost_breakdown" payload if present;
    # otherwise, fall back to derived direct materials/labor
    # and include margin when available.
    cost_breakdown_source = data.get("cost_breakdown")
    items: list[tuple[str, Any]] = []
    if cost_breakdown_source:
        try:
            items = _to_pairs(cost_breakdown_source)  # type: ignore[arg-type]
        except Exception:
            items = []
    if not items:
        if direct_materials and direct_materials > 0:
            items.append(("Materials (direct)", direct_materials))
        if direct_labor and direct_labor > 0:
            items.append(("Labor (direct)", direct_labor))
        if margin_amount is not None and margin_amount > 0:
            items.append(("Margin", margin_amount))

    rows: list[list[str]] = []
    for raw_label, raw_amount in items:
        label = str(raw_label)
        amount = _coerce_float(raw_amount) or 0.0
        percent = 0.0
        if subtotal_before_margin:
            percent = amount / subtotal_before_margin
        rows.append([label, money(amount, currency), pct(percent)])
    warning_present = any(MATERIALS_WARNING_LABEL in row[0] for row in rows)
    materials_direct = None
    if isinstance(data, Mapping):
        materials_direct = _coerce_float(data.get("materials_direct"))
    materials_missing = materials_direct is not None and abs(materials_direct) <= 0.0005
    if materials_missing and not warning_present:
        rows.append(
            [
                ellipsize(MATERIALS_WARNING_LABEL, 60),
                money(0.0, currency),
                pct(0.0),
            ]
        )

    return ascii_table(
        headers=["Cost Element", "Amount", "% of Subtotal"],
        rows=rows,
        col_widths=[48, 18, 15],
        col_aligns=["left", "right", "right"],
        header_aligns=["left", "right", "right"],
    )


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

    # Join the sub-sections we built above. If nothing was
    # produced, return None so the caller can skip the section.
    return "\n\n".join(sections) if sections else None


def _sanitize_block(text: str) -> str:
    """Normalize unicode and strip ANSI/control chars for email-safe ASCII.

    - Converts common unicode punctuation to ASCII equivalents
    - Removes ANSI escape sequences
    - Replaces tabs with spaces
    - Normalizes to NFKC and strips non-ASCII (except newlines)
    """
    if text is None:
        return ""

    # Normalize newlines first
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Drop ANSI escape sequences (e.g., \x1b[31m ... \x1b[0m)
    s = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)

    # Replace tabs with a single space to avoid misalignment
    s = s.replace("\t", " ")

    # Replace a few common unicode punctuation characters
    replacements = {
        "\u2212": "-",  # minus sign
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote
        "\u2022": "-",  # bullet
        "\u00A0": " ",  # non-breaking space
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Unicode normalization, then strip remaining non-ASCII (keep newlines)
    s = unicodedata.normalize("NFKC", s)
    allowed_chars = {"×", "–", "—"}
    s = "".join(
        ch for ch in s if ch == "\n" or ch in allowed_chars or 32 <= ord(ch) <= 126
    )

    return s


def _render_programming_nre(data: Mapping[str, Any]) -> str | None:
    mode = str(data.get("programming_mode") or "").strip().lower()
    if mode != "amortized":
        return None
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)
    rows: list[list[str]] = []

    # Primary source: explicit NRE entries from payload
    entries = data.get("nre")
    if not isinstance(entries, Sequence):
        entries = []

    # If no explicit entries and a per-lot value is present, synthesize
    programming_per_lot = _coerce_float(data.get("programming_per_lot"))
    if (not entries) and (programming_per_lot is not None):
        entries = [
            {
                "label": "Programming & Eng (per lot)",
                "detail": "",
                "amount": programming_per_lot,
            }
        ]

    # Always show computed/specified programming-per-lot when available
    if programming_per_lot is not None and programming_per_lot > 0:
        rows.append([
            "Programming & Eng (per lot)",
            "",
            money(programming_per_lot, currency),
        ])

    for entry in entries:
        if isinstance(entry, Mapping):
            label = entry.get("label") or entry.get("name") or ""
            detail = entry.get("detail") or entry.get("notes") or ""
            amount = _coerce_float(entry.get("amount"))
        else:
            label = entry[0] if entry else ""
            detail = entry[1] if len(entry) > 1 else ""
            amount = _coerce_float(entry[2]) if len(entry) > 2 else None
        rows.append([
            str(label),
            str(detail) if detail else "",
            money(amount, currency),
        ])

    return ascii_table(
        headers=["Activity", "Detail", "Amount (per lot)"],
        rows=rows,
        col_widths=[38, 43, 16],
        col_aligns=["left", "left", "right"],
        header_aligns=["left", "left", "right"],
    )


def _render_materials(data: Mapping[str, Any]) -> str | None:
    currency = _currency_from(data.get("summary", {}) if isinstance(data.get("summary"), Mapping) else data)
    cost_components = data.get("material_cost_components")
    weight_summary = data.get("material_weight")
    stock_meta = data.get("material_stock")
    if not any((cost_components, weight_summary, stock_meta)):
        return None

    rows: list[list[str]] = []
    entries = data.get("materials")
    if not isinstance(entries, Sequence):
        entries = []

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
        rows.append([
            str(label),
            str(detail) if detail else "",
            money(amount, currency),
        ])

    table = ascii_table(
        headers=["Material", "Detail", "Amount (per part)"],
        rows=rows,
        col_widths=[36, 45, 16],
        col_aligns=["left", "left", "right"],
        header_aligns=["left", "left", "right"],
    )

    supplemental_lines: list[str] = []

    stock_line = _format_stock_line(stock_meta)
    if stock_line:
        supplemental_lines.append(stock_line)

    supplemental_lines.extend(_format_weight_lines(weight_summary))

    price_lines = data.get("material_price_lines")
    if isinstance(price_lines, Sequence):
        supplemental_lines.extend(
            str(line).strip() for line in price_lines if isinstance(line, str) and line.strip()
        )

    if supplemental_lines:
        return table + "\n" + "\n".join(supplemental_lines)

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
        hours_text = f"{hours:.2f} hr" if hours is not None else "—"
        if rate_display:
            rate_text = str(rate_display)
        else:
            rate_text = (
                money(rate, currency) + "/hr" if rate is not None else "—"
            )
        amount_text = money(amount, currency)
        rows.append([str(label), hours_text, rate_text, amount_text])

    return ascii_table(
        headers=["Process", "Hours", "Rate", "Amount (per part)"],
        rows=rows,
        col_widths=[28, 10, 22, 16],
        col_aligns=["left", "right", "right", "right"],
        header_aligns=["left", "right", "right", "right"],
    )


def _render_drilling_time_per_hole(data: Mapping[str, Any]) -> str | None:
    detail = data.get("drilling_time_per_hole")
    if not detail:
        return None
    if not isinstance(detail, Mapping):
        return None
    rows = detail.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        return None

    formatted_rows: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            source = row
        elif isinstance(row, Sequence) and row:
            # Accept tuple-like rows of the canonical order
            source = {
                "diameter_in": row[0] if len(row) > 0 else None,
                "qty": row[1] if len(row) > 1 else None,
                "depth_in": row[2] if len(row) > 2 else None,
                "sfm": row[3] if len(row) > 3 else None,
                "ipr": row[4] if len(row) > 4 else None,
                "minutes_per_hole": row[5] if len(row) > 5 else None,
                "group_minutes": row[6] if len(row) > 6 else None,
            }
        else:
            continue
        diameter = _coerce_float(source.get("diameter_in"))
        depth = _coerce_float(source.get("depth_in"))
        qty_raw = source.get("qty")
        qty_val = _coerce_float(qty_raw)
        if qty_val is None:
            qty_val = 0.0
        qty_display_value: float | int
        if float(qty_val).is_integer():
            qty_display_value = int(round(float(qty_val)))
        else:
            qty_display_value = float(qty_val)
        qty_text = (
            f"{qty_display_value}"
            if isinstance(qty_display_value, int)
            else f"{qty_display_value:.2f}".rstrip("0").rstrip(".")
        )
        sfm = _coerce_float(source.get("sfm"))
        ipr = _coerce_float(source.get("ipr"))
        minutes_per_hole = _coerce_float(source.get("minutes_per_hole"))
        if diameter is None or depth is None or minutes_per_hole is None:
            continue
        if sfm is None:
            sfm = 0.0
        if ipr is None:
            ipr = 0.0
        qty_for_calc = max(float(qty_val), 0.0)
        computed_group_minutes = minutes_per_hole * qty_for_calc
        formatted_rows.append(
            f'Dia {diameter:.3f}" × {qty_text}  | '
            f'depth {depth:.3f}" | {int(round(sfm))} sfm | '
            f'{ipr:.4f} ipr | t/hole {minutes_per_hole:.2f} min | '
            f'group {qty_text}×{minutes_per_hole:.2f} = {computed_group_minutes:.2f} min'
        )

    if not formatted_rows:
        return None

    tool_total = _coerce_float(detail.get("toolchange_minutes")) or 0.0
    tool_components = detail.get("tool_components") if isinstance(detail.get("tool_components"), Sequence) else []
    tool_parts: list[str] = []
    if isinstance(tool_components, Sequence):
        for component in tool_components:
            if not isinstance(component, Mapping):
                continue
            label = str(component.get("label") or "").strip()
            minutes = _coerce_float(component.get("minutes"))
            if not label or minutes is None or minutes <= 0:
                continue
            tool_parts.append(f"{label} {minutes:.2f} min")

    subtotal_minutes = _coerce_float(detail.get("subtotal_minutes"))
    total_minutes = _coerce_float(detail.get("total_minutes_with_toolchange"))

    divider = "-" * 66
    lines: list[str] = [divider]
    lines.extend(formatted_rows)
    if tool_total > 0.0:
        if tool_parts:
            tool_text = " + ".join(tool_parts)
            lines.append(f"Toolchange adders: {tool_text} = {tool_total:.2f} min")
        else:
            lines.append(f"Toolchange adders: {tool_total:.2f} min")
    else:
        lines.append("Toolchange adders: -")
    lines.append(divider)
    if subtotal_minutes is not None:
        lines.append(
            f"Subtotal (per-hole × qty) . {subtotal_minutes:.2f} min  ({(subtotal_minutes / 60.0):.2f} hr)"
        )
    if total_minutes is not None:
        lines.append(
            f"TOTAL DRILLING (with toolchange) . {total_minutes:.2f} min  ({(total_minutes / 60.0):.2f} hr)"
        )
    return "\n".join(lines)


def _render_cycle_reference(data: Mapping[str, Any]) -> str | None:
    entries = data.get("cycle_time_metrics")
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
        rows.append([
            str(label),
            f"{time_val:.2f} hr" if time_val is not None else "—",
            str(notes) if notes else "",
        ])

    return ascii_table(
        headers=["Activity", "Cycle Time", "Notes"],
        rows=rows,
        col_widths=[38, 12, 36],
        col_aligns=["left", "right", "left"],
        header_aligns=["left", "right", "left"],
    )


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
            label = entry[0] if entry else ""
            hours = _coerce_float(entry[1]) if len(entry) > 1 else None
        rows.append([str(label), f"{hours:.2f} hr" if hours is not None else "—"])

    return ascii_table(
        headers=["Activity", "Hours"],
        rows=rows,
        col_widths=[64, 12],
        col_aligns=["left", "right"],
        header_aligns=["left", "right"],
    )


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
    formatted = [[str(k), str(v)] for k, v in pairs]
    return ascii_table(
        headers=["Item", "Detail"],
        rows=formatted,
        col_widths=[38, 50],
        col_aligns=["left", "left"],
        header_aligns=["left", "left"],
    )


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

    drilling_time = _render_drilling_time_per_hole(data)
    if drilling_time:
        sections.append(_render_section("TIME PER HOLE – DRILL GROUPS", drilling_time))

    cycle_reference = _render_cycle_reference(data)
    if cycle_reference:
        sections.append(_render_section("Cycle-Time Reference", cycle_reference))

    drill_groups = _render_drill_groups(data)
    if drill_groups:
        sections.append(_render_section("Cycle-Time — Drill Groups", drill_groups))

    trace = _render_traceability(data)
    if trace:
        sections.append(_render_section("Traceability", trace))

    return _sanitize_block("\n\n".join(sections))


__all__ = ["render_quote"]
