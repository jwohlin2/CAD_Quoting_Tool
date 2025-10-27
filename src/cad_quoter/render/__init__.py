from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from cad_quoter.domain import _canonical_pass_label
from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.utils.render_utils import QuoteRow, QuoteSection
from cad_quoter.utils.rendering import fmt_money


@dataclass
class Placeholder:
    """Deferred update for previously rendered lines."""

    section_index: int
    offset: int
    formatter: Callable[[Any], str] | None = None
    pending_value: Any = None


@dataclass
class SectionRecord:
    """Captured section output and optional doc-builder view."""

    index: int
    title: str
    lines: list[str]
    header_index: int = 0
    doc_section: QuoteSection | None = None


class SectionBuilder:
    """Incrementally construct a quote section."""

    def __init__(
        self,
        state: "RenderState",
        title: str,
    ) -> None:
        self._state = state
        self._title = title
        self._lines: list[str] = [title, state.divider]

    def add_line(self, text: str) -> int:
        self._lines.append(str(text))
        return len(self._lines) - 1

    def add_blank_line(self) -> int:
        return self.add_line("")

    def add_detail(self, text: Any, *, indent: str = "    ") -> int | None:
        if text in (None, ""):
            return None
        detail = str(text).strip()
        if not detail:
            return None
        return self.add_line(f"{indent}{detail}")

    def add_row(self, label: str, amount: Any, *, indent: str = "  ") -> int:
        row_text = self._state.format_row(label, amount, indent=indent)
        self._lines.append(row_text)
        return len(self._lines) - 1

    def set_title(self, text: str) -> None:
        clean = str(text)
        self._title = clean
        self._lines[0] = clean

    def finalize(self) -> SectionRecord:
        record_index = len(self._state.sections)
        record = SectionRecord(
            index=record_index,
            title=self._title,
            lines=list(self._lines),
            header_index=0,
        )
        if self._state.track_doc_sections:
            rows: list[QuoteRow] = []
            for line in self._lines[2:]:
                rows.append(QuoteRow(index=self._state._next_row_index, text=line))
                self._state._next_row_index += 1
            record.doc_section = QuoteSection(title=self._title, rows=rows)
            self._state.doc_sections.append(record.doc_section)
        self._state._register_section(record)
        return record


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, MutableMapping):
        return dict(value)
    if isinstance(value, Mapping):
        try:
            return dict(value)
        except Exception:
            return {}
    return {}


def _extend_material_entries(dest: list[Mapping[str, Any]], source: Any) -> None:
    if isinstance(source, Mapping):
        entries = source.get("entries")
        if isinstance(entries, Sequence):
            for entry in entries:
                if isinstance(entry, Mapping):
                    dest.append(entry)
        elif any(key in source for key in ("amount", "label", "detail")):
            dest.append(source)
    elif isinstance(source, Sequence):
        for entry in source:
            if isinstance(entry, Mapping):
                dest.append(entry)


def _collect_cost_breakdown(source: Any) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    if isinstance(source, Mapping):
        for key, value in source.items():
            amount = _coerce_float_or_none(value)
            if amount is None:
                continue
            items.append((str(key), float(amount)))
    elif isinstance(source, Sequence):
        for entry in source:
            if isinstance(entry, Sequence) and len(entry) >= 2:
                label = str(entry[0])
                amount = _coerce_float_or_none(entry[1])
                if amount is None:
                    continue
                items.append((label, float(amount)))
    return items


def _first_numeric(*values: Any) -> float | None:
    for candidate in values:
        amount = _coerce_float_or_none(candidate)
        if amount is not None:
            return float(amount)
    return None


@dataclass
class RenderState:
    """Container tracking quote rendering context and derived values."""

    payload: MutableMapping[str, Any] | Mapping[str, Any]
    currency: str = "$"
    show_zeros: bool = False
    divider: str = "-" * 66
    page_width: int = 74
    track_doc_sections: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.payload, MutableMapping):
            self.payload = _normalize_mapping(self.payload)
        self.result = self.payload

        breakdown_candidate = self.result.get("breakdown") if isinstance(self.result, MutableMapping) else None
        if isinstance(breakdown_candidate, MutableMapping):
            self.breakdown = breakdown_candidate
        else:
            self.breakdown = {}
            if isinstance(self.result, MutableMapping):
                self.result["breakdown"] = self.breakdown

        totals_candidate = self.breakdown.get("totals") if isinstance(self.breakdown, MutableMapping) else None
        if isinstance(totals_candidate, MutableMapping):
            self.totals = totals_candidate
        else:
            self.totals = {}
            self.breakdown["totals"] = self.totals

        pricing_candidate = self.breakdown.get("pricing")
        if isinstance(pricing_candidate, MutableMapping):
            self.pricing = pricing_candidate
        else:
            self.pricing = {}
            self.breakdown["pricing"] = self.pricing

        pass_through_candidate = self.breakdown.get("pass_through")
        self.pass_through = _normalize_mapping(pass_through_candidate)
        self.breakdown["pass_through"] = self.pass_through

        direct_cost_details_candidate = self.breakdown.get("direct_cost_details")
        self.direct_cost_details = _normalize_mapping(direct_cost_details_candidate)
        self.breakdown["direct_cost_details"] = self.direct_cost_details

        pass_basis_candidate = self.breakdown.get("pass_basis")
        self.pass_basis = _normalize_mapping(pass_basis_candidate)
        if self.pass_basis:
            self.breakdown["pass_basis"] = self.pass_basis

        material_candidate = self.breakdown.get("material")
        if isinstance(material_candidate, MutableMapping):
            self.material_block = material_candidate
        else:
            self.material_block = _normalize_mapping(material_candidate)
            if self.material_block:
                self.breakdown["material"] = self.material_block

        material_components_candidate = self.material_block.get("material_cost_components")
        if not isinstance(material_components_candidate, Mapping):
            material_components_candidate = self.breakdown.get("material_cost_components")
        self.material_cost_components = _normalize_mapping(material_components_candidate)
        if self.material_cost_components:
            self.material_block.setdefault("material_cost_components", self.material_cost_components)

        self.material_total_for_directs = 0.0
        try:
            material_key = next(
                (key for key in self.pass_through if str(key).strip().lower() == "material"),
                None,
            )
        except Exception:
            material_key = None
        if material_key is not None:
            amount = _coerce_float_or_none(self.pass_through.get(material_key))
            if amount is not None:
                self.material_total_for_directs = float(amount)

        self.scrap_credit_for_directs = float(
            _coerce_float_or_none(self.material_block.get("material_scrap_credit")) or 0.0
        )
        self.material_tax_for_directs = float(
            _coerce_float_or_none(self.material_block.get("material_tax")) or 0.0
        )

        self.material_component_total: float | None = None
        self.material_component_net: float | None = None
        if self.material_cost_components:
            base_component = _coerce_float_or_none(self.material_cost_components.get("base_usd"))
            if base_component is not None:
                self.material_total_for_directs = float(base_component)
            scrap_component = _coerce_float_or_none(self.material_cost_components.get("scrap_credit_usd"))
            if scrap_component is not None:
                self.scrap_credit_for_directs = float(scrap_component)
            tax_component = _coerce_float_or_none(self.material_cost_components.get("tax_usd"))
            if tax_component is not None:
                self.material_tax_for_directs = float(tax_component)
            total_component = _coerce_float_or_none(self.material_cost_components.get("total_usd"))
            if total_component is not None:
                self.material_component_total = float(total_component)
            net_component = _coerce_float_or_none(self.material_cost_components.get("net_usd"))
            if net_component is not None:
                self.material_component_net = float(net_component)

        material_entries: list[Mapping[str, Any]] = []
        if isinstance(self.result, Mapping):
            _extend_material_entries(material_entries, self.result.get("materials"))
        _extend_material_entries(material_entries, self.breakdown.get("materials"))
        _extend_material_entries(material_entries, self.breakdown.get("material"))
        _extend_material_entries(material_entries, self.pricing.get("materials"))
        self.material_warning_entries = material_entries

        self.cost_breakdown_entries = _collect_cost_breakdown(
            self.breakdown.get("cost_breakdown")
            if isinstance(self.breakdown, Mapping)
            else None
        )
        if not self.cost_breakdown_entries and isinstance(self.result, Mapping):
            self.cost_breakdown_entries = _collect_cost_breakdown(self.result.get("cost_breakdown"))

        self.material_entries_total = 0.0
        self.material_entries_have_label = False
        for entry in material_entries:
            amount = _coerce_float_or_none(entry.get("amount"))
            if amount is not None:
                self.material_entries_total += float(amount)
            if not self.material_entries_have_label:
                text = str(entry.get("label") or entry.get("detail") or "").strip()
                if text:
                    self.material_entries_have_label = True

        materials_direct_total = _coerce_float_or_none(self.breakdown.get("materials_direct"))
        if materials_direct_total is None and isinstance(self.result, Mapping):
            materials_direct_total = _coerce_float_or_none(self.result.get("materials_direct"))
        self.materials_direct_total = float(materials_direct_total or 0.0)

        self.material_warning_summary = bool(self.material_entries_have_label) and (
            self.material_entries_total <= 0.0 and self.materials_direct_total <= 0.0
        )

        self.sections: list[SectionRecord] = []
        self.doc_sections: list[QuoteSection] = []
        self._next_row_index: int = 0
        self.placeholders: dict[str, Placeholder] = {}
        self.warning_flags: dict[str, bool] = {}
        self.pass_through_total: float = 0.0
        self.vendor_items_total: float = 0.0
        self.material_display_amount: float = 0.0
        self.material_warning_needed: bool = bool(self.material_warning_summary)

    # ------------------------------------------------------------------
    # Section management helpers
    # ------------------------------------------------------------------

    def new_section(self, title: str) -> SectionBuilder:
        return SectionBuilder(self, title)

    def _register_section(self, record: SectionRecord) -> None:
        record.index = len(self.sections)
        self.sections.append(record)
        for placeholder in self.placeholders.values():
            if placeholder.section_index == record.index and placeholder.pending_value is not None:
                self._apply_placeholder_value(placeholder, placeholder.pending_value)
                placeholder.pending_value = None

    def update_section_title(self, section_index: int, title: str) -> None:
        if not (0 <= section_index < len(self.sections)):
            return
        record = self.sections[section_index]
        record.title = title
        record.lines[record.header_index] = title
        if record.doc_section is not None:
            record.doc_section.title = title

    def register_placeholder(
        self,
        key: str,
        section_index: int,
        offset: int,
        formatter: Callable[[Any], str] | None = None,
    ) -> None:
        self.placeholders[key] = Placeholder(
            section_index=section_index,
            offset=offset,
            formatter=formatter,
        )

    def _apply_placeholder_value(self, placeholder: Placeholder, value: Any) -> None:
        new_text = placeholder.formatter(value) if placeholder.formatter else str(value)
        if not (0 <= placeholder.section_index < len(self.sections)):
            placeholder.pending_value = value
            return
        record = self.sections[placeholder.section_index]
        if not (0 <= placeholder.offset < len(record.lines)):
            return
        record.lines[placeholder.offset] = new_text
        if record.doc_section is not None:
            doc_offset = placeholder.offset - 2
            if 0 <= doc_offset < len(record.doc_section.rows):
                record.doc_section.rows[doc_offset].text = new_text

    def update_placeholder(self, key: str, value: Any) -> None:
        placeholder = self.placeholders.get(key)
        if placeholder is None:
            return
        if placeholder.section_index >= len(self.sections):
            placeholder.pending_value = value
            return
        self._apply_placeholder_value(placeholder, value)

    def final_lines(self) -> list[str]:
        combined: list[str] = []
        for record in self.sections:
            combined.extend(record.lines)
        return combined

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_money(self, amount: Any) -> str:
        return fmt_money(amount, self.currency)

    def format_row(self, label: str, amount: Any, *, indent: str = "") -> str:
        left = f"{indent}{str(label or '').strip()}"
        try:
            numeric = float(amount or 0.0)
        except Exception:
            numeric = 0.0
        right = self.format_money(numeric)
        total_width = max(10, int(self.page_width))
        pad = max(2, total_width - len(left) - len(right))
        return f"{left}{' ' * pad}{right}"


MATERIAL_WARNING_LABEL = "⚠ MATERIALS MISSING"


def _direct_label(raw_key: Any) -> str:
    text = str(raw_key)
    canonical = _canonical_pass_label(text)
    if canonical:
        canonical_stripped = canonical.strip()
        if canonical_stripped and canonical_stripped.lower() == canonical_stripped:
            return canonical_stripped.title()
        return canonical_stripped or canonical
    label = text.replace("_", " ").replace("hr", "/hr").strip()
    if not label:
        return text
    return label.title()


def render_pass_through(state: RenderState) -> list[str]:
    """Render the Pass-Through & Direct Costs section."""

    builder = state.new_section("Pass-Through & Direct Costs")

    displayed_pass_through: dict[str, float] = {}
    pass_total = 0.0
    pass_through = state.pass_through
    for key, value in sorted(pass_through.items(), key=lambda kv: _coerce_float_or_none(kv[1]) or 0.0, reverse=True):
        canonical_label = _canonical_pass_label(key)
        if canonical_label and canonical_label.lower() == "material":
            continue
        amount_val = _coerce_float_or_none(value)
        if amount_val is None:
            continue
        if amount_val > 0 or state.show_zeros:
            displayed_pass_through[str(key)] = float(amount_val)
            pass_total += float(amount_val)

    vendor_items_total = 0.0
    vendor_item_sources: list[Mapping[str, Any]] = []
    for candidate in (
        state.breakdown.get("vendor_items") if isinstance(state.breakdown, Mapping) else None,
        pass_through.get("vendor_items") if isinstance(pass_through, Mapping) else None,
    ):
        if isinstance(candidate, Mapping):
            vendor_item_sources.append(candidate)
    for vendor_map in vendor_item_sources:
        for amount in vendor_map.values():
            numeric = _coerce_float_or_none(amount)
            if numeric is not None:
                vendor_items_total += float(numeric)

    material_direct_contribution = round(
        float(state.material_total_for_directs)
        + float(state.material_tax_for_directs)
        - float(state.scrap_credit_for_directs),
        2,
    )

    material_display_amount = material_direct_contribution
    if state.material_component_total is not None:
        material_display_amount = round(float(state.material_component_total), 2)
    else:
        candidate = _coerce_float_or_none(state.material_block.get("total_material_cost"))
        if candidate is not None:
            material_display_amount = round(float(candidate), 2)

    material_total_for_why = float(material_display_amount)
    if state.material_component_net is not None:
        material_net_cost = float(state.material_component_net)
    else:
        material_net_cost = float(material_display_amount)

    if material_display_amount <= 0.0:
        fallback_amount = 0.0
        if state.materials_direct_total > 0:
            fallback_amount = float(state.materials_direct_total)
        if state.material_entries_total > 0:
            fallback_amount = max(fallback_amount, float(state.material_entries_total))
        if fallback_amount > 0:
            material_display_amount = round(fallback_amount, 2)
            material_total_for_why = float(material_display_amount)
            material_net_cost = float(material_display_amount)

    material_warning_needed = bool(state.material_warning_summary)
    if state.material_entries_have_label and material_display_amount <= 0.0:
        material_warning_needed = True

    direct_costs_map = state.pricing.setdefault("direct_costs", {})
    if not isinstance(direct_costs_map, MutableMapping):
        direct_costs_map = _normalize_mapping(direct_costs_map)
        state.pricing["direct_costs"] = direct_costs_map

    def _assign_direct_value(raw_key: Any, amount: Any) -> None:
        try:
            amount_float = round(float(amount), 2)
        except Exception:
            return
        target_key = raw_key
        canonical_target = str(_canonical_pass_label(str(raw_key)) or "").strip().lower()
        if canonical_target:
            for existing_key in list(direct_costs_map.keys()):
                existing_canonical = str(_canonical_pass_label(str(existing_key)) or "").strip().lower()
                if existing_canonical == canonical_target:
                    target_key = existing_key
                    break
        direct_costs_map[target_key] = amount_float

    _assign_direct_value("material", material_direct_contribution)
    if vendor_items_total > 0 or state.show_zeros:
        _assign_direct_value("vendor items", vendor_items_total)
    for key, amount_val in displayed_pass_through.items():
        _assign_direct_value(key, amount_val)

    direct_entries: list[tuple[str, float, Any]] = []
    for raw_key, raw_value in direct_costs_map.items():
        amount_val = _coerce_float_or_none(raw_value)
        if amount_val is None:
            continue
        direct_entries.append((_direct_label(raw_key), round(float(amount_val), 2), raw_key))
    direct_entries.sort(key=lambda item: item[1], reverse=True)

    material_entry = None
    display_entries: list[tuple[str, float, Any]] = []
    for entry in direct_entries:
        raw_key = entry[2]
        if str(raw_key).strip().lower() == "material":
            material_entry = entry
            continue
        display_entries.append(entry)

    pass_basis_map = state.pass_basis

    def add_pass_basis(key: str, *, indent: str = "    ") -> None:
        if not pass_basis_map:
            return
        info = pass_basis_map.get(key) if isinstance(pass_basis_map, Mapping) else None
        if not isinstance(info, Mapping):
            return
        text = info.get("basis") or info.get("note")
        if text:
            builder.add_detail(text, indent=indent)

    if (material_display_amount > 0) or state.show_zeros:
        builder.add_row("Material & Stock", material_display_amount, indent="  ")
        material_basis_key = None
        if isinstance(pass_through, Mapping):
            for candidate_key in pass_through.keys():
                if str(candidate_key).strip().lower() == "material":
                    material_basis_key = candidate_key
                    break
        if material_basis_key is None:
            material_basis_key = "Material"
        add_pass_basis(str(material_basis_key), indent="    ")

        material_detail_value: Any = None
        if material_entry is not None:
            raw_material_key = material_entry[2]
            material_detail_value = state.direct_cost_details.get(raw_material_key)
            if material_detail_value in (None, ""):
                material_detail_value = state.direct_cost_details.get(str(raw_material_key))
        if material_detail_value in (None, ""):
            for key_candidate in (
                material_basis_key,
                str(material_basis_key).strip(),
                "Material",
                "material",
                "Material & Stock",
            ):
                if key_candidate in (None, ""):
                    continue
                detail_candidate = state.direct_cost_details.get(key_candidate)
                if detail_candidate not in (None, ""):
                    material_detail_value = detail_candidate
                    break
        material_amount_text = state.format_money(material_display_amount)
        if material_detail_value not in (None, ""):
            detail_text = str(material_detail_value)
            if "$0.00" in detail_text:
                detail_text = detail_text.replace("$0.00", material_amount_text)
            builder.add_detail(detail_text, indent="    ")
        else:
            builder.add_detail(
                f"Material & Stock (printed above) contributes {material_amount_text} to Direct Costs",
                indent="    ",
            )
    elif material_warning_needed and state.material_entries_have_label:
        builder.add_row("Materials & Stock", 0.0, indent="  ")
        builder.add_detail(
            f"{MATERIAL_WARNING_LABEL} Material items are present but no material costs were recorded in the quote.",
            indent="    ",
        )

    for display_label, amount_val, raw_key in display_entries:
        if (amount_val > 0) or state.show_zeros:
            builder.add_row(display_label, amount_val, indent="  ")
            raw_key_str = str(raw_key)
            if raw_key_str in displayed_pass_through:
                add_pass_basis(raw_key_str, indent="    ")
                detail_value = state.direct_cost_details.get(raw_key_str)
                if detail_value not in (None, ""):
                    builder.add_detail(str(detail_value), indent="    ")
            else:
                detail_value = state.direct_cost_details.get(raw_key)
                if detail_value not in (None, ""):
                    builder.add_detail(str(detail_value), indent="    ")

    total_direct_costs_value = round(
        sum(amount for _, amount, _ in direct_entries if (amount > 0) or state.show_zeros),
        2,
    )
    builder.add_row("Total", total_direct_costs_value, indent="  ")

    if state.cost_breakdown_entries:
        builder.add_blank_line()
        builder.add_line("Cost Breakdown")
        builder.add_line(state.divider)
        for label, amount in state.cost_breakdown_entries:
            builder.add_row(label, amount, indent="  ")

    pass_through_total = float(sum(displayed_pass_through.values()))
    material_for_totals = material_display_amount
    material_breakdown_entry = (
        state.breakdown.get("material") if isinstance(state.breakdown, Mapping) else None
    )
    if isinstance(material_breakdown_entry, Mapping):
        total_candidate = _first_numeric(
            material_breakdown_entry.get("total_cost"),
            material_breakdown_entry.get("total_material_cost"),
            material_breakdown_entry.get("material_total_cost"),
            material_breakdown_entry.get("material_cost"),
            material_breakdown_entry.get("material_cost_before_credit"),
            material_breakdown_entry.get("material_direct_cost"),
        )
        if total_candidate is not None:
            material_for_totals = float(total_candidate)

    computed_direct_total = round(
        float(pass_through_total) + float(material_for_totals) + float(vendor_items_total),
        2,
    )

    header_text = (
        f"Pass-Through & Direct Costs (Total: {state.format_money(computed_direct_total)})"
    )
    builder.set_title(header_text)

    record = builder.finalize()

    state.pass_through_total = pass_through_total
    state.vendor_items_total = vendor_items_total
    state.material_display_amount = material_display_amount
    state.material_warning_needed = material_warning_needed
    state.warning_flags["material_warning"] = material_warning_needed

    state.breakdown["pass_through_total"] = pass_through_total
    state.breakdown["total_direct_costs"] = computed_direct_total
    if vendor_items_total:
        state.breakdown["vendor_items_total"] = float(round(vendor_items_total, 2))
    state.breakdown["material_warning_needed"] = material_warning_needed

    state.totals["direct_costs"] = computed_direct_total
    state.totals.setdefault("directs$", computed_direct_total)
    state.pricing["total_direct_costs"] = computed_direct_total

    state.update_placeholder("direct_cost_total", computed_direct_total)

    state.update_section_title(record.index, header_text)

    return record.lines


__all__ = [
    "RenderState",
    "render_pass_through",
    "SectionBuilder",
    "SectionRecord",
]
