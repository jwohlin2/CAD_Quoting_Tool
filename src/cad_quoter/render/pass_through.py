"""Pass-through section renderer extracted from the legacy quote output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cad_quoter.pass_labels import canonical_pass_label
from cad_quoter.utils.render_utils import format_currency

from .state import RenderState, _as_mapping


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _lookup_text(mapping: Mapping[Any, Any], key: str) -> Any:
    """Return the value in ``mapping`` that matches ``key`` case-insensitively."""

    target = str(key or "").strip().lower()
    if not target:
        return None
    for candidate, value in mapping.items():
        candidate_key = str(candidate or "").strip().lower()
        if candidate_key == target:
            return value
    return None


def _detail_lines(
    label: str,
    *,
    details_map: Mapping[Any, Any],
    basis_map: Mapping[Any, Any],
) -> list[str]:
    """Return detail text lines for ``label`` drawn from the legacy mappings."""

    lines: list[str] = []

    detail_value = _lookup_text(details_map, label)
    if detail_value not in (None, ""):
        lines.append(str(detail_value))

    basis_value = _lookup_text(basis_map, label)
    if isinstance(basis_value, Mapping):
        text_value = basis_value.get("basis") or basis_value.get("text")
    else:
        text_value = basis_value
    if text_value not in (None, ""):
        lines.append(str(text_value))

    return lines


def render_pass_through(state: RenderState) -> tuple[list[str], float, float]:
    """Return the pass-through section lines and summary totals."""

    breakdown = state.breakdown
    pass_through_map = _as_mapping(breakdown.get("pass_through"))
    details_map = _as_mapping(breakdown.get("direct_cost_details"))
    basis_map = _as_mapping(breakdown.get("pass_basis"))

    displayed_pass: dict[str, float] = {}
    pass_through_total = 0.0
    pass_through_labor_total = 0.0

    for raw_key, raw_value in pass_through_map.items():
        key = str(raw_key)
        lower_key = key.strip().lower()
        if lower_key == "vendor_items":
            continue
        canonical = canonical_pass_label(key)
        if canonical.lower() == "material":
            continue
        amount = _coerce_float(raw_value)
        if amount == 0.0 and not state.show_zeros:
            continue
        displayed_pass[key] = round(amount, 2)
        pass_through_total += amount
        if "labor" in canonical.lower():
            pass_through_labor_total += amount

    vendor_items_total = 0.0
    for container in (
        breakdown.get("vendor_items"),
        pass_through_map.get("vendor_items"),
    ):
        vendor_map = _as_mapping(container)
        for value in vendor_map.values():
            vendor_items_total += _coerce_float(value)

    material_block = _as_mapping(breakdown.get("material"))
    material_display_amount = _coerce_float(material_block.get("total_material_cost"))

    materials_direct_total = _coerce_float(breakdown.get("materials_direct"))
    if material_display_amount <= 0.0 and materials_direct_total > 0.0:
        material_display_amount = materials_direct_total

    materials_entries_total = 0.0
    material_entries_have_label = False
    materials_entries = breakdown.get("materials")
    if isinstance(materials_entries, Sequence):
        for entry in materials_entries:
            entry_map = _as_mapping(entry)
            if not entry_map:
                continue
            label = str(entry_map.get("label", "")).strip()
            if label:
                material_entries_have_label = True
            materials_entries_total += _coerce_float(entry_map.get("amount"))
    if material_display_amount <= 0.0 and materials_entries_total > 0.0:
        material_display_amount = materials_entries_total

    pass_through_total = round(pass_through_total, 2)
    vendor_items_total = round(vendor_items_total, 2)
    material_display_amount = round(material_display_amount, 2)

    direct_total_value = round(
        pass_through_total + vendor_items_total + material_display_amount,
        2,
    )

    pricing_map = state.pricing
    direct_costs_map = _as_mapping(pricing_map.get("direct_costs"))
    pricing_map["direct_costs"] = direct_costs_map

    def _assign_direct_value(label: str, amount: float) -> None:
        if amount == 0.0 and not state.show_zeros:
            return
        canonical_target = canonical_pass_label(label)
        target_key = None
        for existing_key in list(direct_costs_map.keys()):
            if canonical_pass_label(existing_key) == canonical_target:
                target_key = existing_key
                break
        if target_key is None:
            target_key = label
        try:
            current = float(direct_costs_map.get(target_key, 0.0) or 0.0)
        except Exception:
            current = 0.0
        direct_costs_map[target_key] = round(current + amount, 2)

    for label, amount in displayed_pass.items():
        _assign_direct_value(label, amount)

    if vendor_items_total > 0.0 or state.show_zeros:
        _assign_direct_value("vendor items", vendor_items_total)

    if material_display_amount > 0.0 or state.show_zeros:
        _assign_direct_value("Material & Stock", material_display_amount)

    pricing_map["total_direct_costs"] = direct_total_value

    if isinstance(state.totals, Mapping):
        state.totals["direct_costs"] = direct_total_value

    breakdown["pass_through_total"] = pass_through_total
    breakdown["total_direct_costs"] = direct_total_value
    if vendor_items_total > 0.0:
        breakdown["vendor_items_total"] = vendor_items_total

    material_warning_needed = material_entries_have_label and material_display_amount <= 0.0
    if material_warning_needed:
        breakdown["material_warning_needed"] = True
        state.warning_flags["material_warning"] = True
        state.material_warning_summary = True
    elif "material_warning_needed" not in breakdown:
        breakdown["material_warning_needed"] = False

    currency_text = format_currency(direct_total_value, state.currency)
    header_line = f"Pass-Through & Direct Costs (Total: {currency_text})"

    lines: list[str] = [header_line, state.divider or ""]

    if material_warning_needed:
        lines.append("⚠ MATERIALS MISSING – review material costs")

    details_map = {str(k): v for k, v in details_map.items()}
    basis_map = {str(k): _as_mapping(v) if isinstance(v, Mapping) else v for k, v in basis_map.items()}

    if material_display_amount > 0.0 or state.show_zeros or material_warning_needed:
        lines.append(state.format_row("Material & Stock", material_display_amount, indent="  "))
        material_details = [
            f"Material & Stock (printed above) contributes {format_currency(material_display_amount, state.currency)} to Direct Costs"
            if material_display_amount > 0.0
            else "Material & Stock contribution currently $0.00",
            *_detail_lines("Material", details_map=details_map, basis_map=basis_map),
        ]
        for detail in material_details:
            if detail in (None, ""):
                continue
            lines.append(f"    {detail}")

    for label, amount in sorted(
        displayed_pass.items(),
        key=lambda item: (-item[1], str(item[0]).lower()),
    ):
        lines.append(state.format_row(label, amount, indent="  "))
        for detail in _detail_lines(label, details_map=details_map, basis_map=basis_map):
            lines.append(f"    {detail}")

    if vendor_items_total > 0.0 or state.show_zeros:
        vendor_label = "vendor items"
        lines.append(state.format_row(vendor_label, vendor_items_total, indent="  "))
        for detail in _detail_lines(vendor_label, details_map=details_map, basis_map=basis_map):
            lines.append(f"    {detail}")

    lines.append(state.format_row("Total", direct_total_value, indent="  "))

    state.add_section(header_line, lines)
    state.update_placeholder("direct_cost_total", direct_total_value)

    return lines, direct_total_value, round(pass_through_labor_total, 2)
