"""Shared state helpers for quote rendering sections."""

from __future__ import annotations

from dataclasses import dataclass, field
import textwrap
from typing import Any, Callable, Iterable, Mapping, Sequence, TYPE_CHECKING

from cad_quoter.utils.render_utils import (
    QuoteDocRecorder,
    QuoteRow,
    QuoteSection,
    format_currency,
    format_hours,
    format_hours_with_rate,
)
from cad_quoter.utils.render_utils.tables import draw_kv_table
from cad_quoter.utils.text_rules import canonicalize_amortized_label as _canonical_amortized_label

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from cad_quoter.ui.services import QuoteConfiguration


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        try:
            return dict(value)
        except Exception:
            return {}
    if value in (None, ""):
        return {}
    try:
        return dict(value)  # type: ignore[arg-type]
    except Exception:
        return {}


def _coerce_rate_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


@dataclass(frozen=True)
class DisplayRow:
    """Structured representation of a single rendered line."""

    label: str
    value: float | None
    indent: str = ""
    kind: str = "money"


@dataclass
class SectionWriter:
    """Helper that accumulates rows and detail text for a render section."""

    state: "RenderState"
    rows: list[DisplayRow] = field(default_factory=list)
    detail_lines: list[str] = field(default_factory=list)
    events: list[tuple[str, int]] = field(default_factory=list)

    def row(self, label: str, value: Any, *, indent: str = "") -> None:
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        self.rows.append(DisplayRow(label=label, value=numeric, indent=indent, kind="money"))
        self.events.append(("row", len(self.rows) - 1))

    def hours_row(self, label: str, value: Any, *, indent: str = "") -> None:
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        self.rows.append(DisplayRow(label=label, value=numeric, indent=indent, kind="hours"))
        self.events.append(("row", len(self.rows) - 1))

    def write_line(self, text: Any, indent: str = "") -> None:
        if text in (None, ""):
            return
        clean = str(text)
        if not clean:
            return
        for chunk in self._wrap_text(clean, indent, preserve_leading=True):
            self.detail_lines.append(chunk)
            self.events.append(("detail", len(self.detail_lines) - 1))

    def write_detail(self, text: Any, indent: str = "    ") -> None:
        if text in (None, ""):
            return
        clean = str(text)
        segments = [segment.strip() for segment in clean.split(";")]
        for segment in segments:
            if not segment:
                continue
            for chunk in self._wrap_text(segment, indent):
                self.detail_lines.append(chunk)
                self.events.append(("detail", len(self.detail_lines) - 1))

    def _wrap_text(self, text: str, indent: str, preserve_leading: bool = False) -> Iterable[str]:
        working = text if preserve_leading else text.strip()
        if not working:
            return []
        width = max(10, self.state.page_width - len(indent))
        wrapper = textwrap.TextWrapper(
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            drop_whitespace=False,
        )
        return [f"{indent}{chunk}" for chunk in wrapper.wrap(working)]


@dataclass
class RenderSection:
    """Structured representation of a rendered section and its metadata."""

    index: int
    title: str
    lines: list[str]
    start_index: int
    doc_section: QuoteSection | None = None


@dataclass
class PlaceholderRecord:
    """Internal book-keeping for line placeholders."""

    name: str
    section_index: int
    line_index: int
    formatter: Callable[[Any], str] | None = None


class SectionBuilder:
    """Convenience builder for accumulating section output."""

    def __init__(self, state: "RenderState", title: str) -> None:
        self._state = state
        self.title = title
        self.lines: list[str] = []

    def add_line(self, text: Any) -> int:
        value = "" if text is None else str(text)
        self.lines.append(value)
        return len(self.lines) - 1

    def add_row(self, label: str, value: Any, *, indent: str = "", kind: str | None = None) -> int:
        text = self._state.format_row(label, value, indent=indent, kind=kind)
        self.lines.append(text)
        return len(self.lines) - 1

    def finalize(self) -> RenderSection:
        return self._state.add_section(self.title, self.lines)


@dataclass
class RenderState:
    """Mutable context shared across section renderers."""

    payload: Mapping[str, Any]
    currency: str = "$"
    show_zeros: bool = False
    page_width: int = 74
    separate_labor_cfg: bool = True
    default_labor_rate: float = 45.0
    divider: str | None = None
    process_meta: Mapping[str, Any] | None = None
    process_meta_raw: Mapping[str, Any] | None = None
    hour_summary_entries: Mapping[str, Any] | Sequence[Any] | None = None
    cfg: "QuoteConfiguration | None" = None
    llm_debug_enabled: bool = False
    drill_debug_entries: Sequence[Any] | None = None
    material_warning_summary: bool = False
    material_warning_label: str = ""
    pricing_source_value: str | None = None
    final_price_row_index: int = -1
    total_process_cost_row_index: int = -1
    total_direct_costs_row_index: int = -1
    process_total_row_index: int = -1
    lines: list[str] | None = None
    recorder: QuoteDocRecorder | None = None
    deferred_replacements: list[tuple[int, str]] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)
    sections: list[RenderSection] = field(default_factory=list)
    _placeholders: dict[str, PlaceholderRecord] = field(default_factory=dict)
    warning_flags: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        result_map = _as_mapping(self.payload)
        self.result: dict[str, Any] = result_map
        breakdown_map = _as_mapping(result_map.get("breakdown"))
        self.breakdown: dict[str, Any] = breakdown_map

        pricing_map = _as_mapping(result_map.get("pricing"))
        self.pricing: dict[str, Any] = pricing_map
        result_map["pricing"] = pricing_map

        if not self.divider:
            self.divider = "-" * max(self.page_width, 1)

        if self.lines is None:
            self.lines = []

        if self.drill_debug_entries is None:
            entries = result_map.get("drill_debug")
            if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                self.drill_debug_entries = list(entries)
            elif entries in (None, ""):
                self.drill_debug_entries = []
            else:
                self.drill_debug_entries = [entries]

        if self.hour_summary_entries is None:
            hour_entries = breakdown_map.get("hour_summary_entries")
            if isinstance(hour_entries, Mapping):
                self.hour_summary_entries = hour_entries
            else:
                self.hour_summary_entries = {}

        totals_map = _as_mapping(breakdown_map.get("totals"))
        self.totals: dict[str, Any] = totals_map
        breakdown_map["totals"] = totals_map

        if self.process_meta is None:
            meta = breakdown_map.get("process_meta")
            self.process_meta = meta if isinstance(meta, Mapping) else None

        if self.process_meta_raw is None:
            meta_raw = breakdown_map.get("process_meta_raw")
            self.process_meta_raw = meta_raw if isinstance(meta_raw, Mapping) else None

        if not self.material_warning_label:
            label = breakdown_map.get("material_warning_label")
            if isinstance(label, str):
                self.material_warning_label = label

        if not self.material_warning_summary:
            self.material_warning_summary = bool(
                breakdown_map.get("material_warning_needed")
            )

        self.rates: dict[str, float] = {}
        for container in (result_map.get("rates"), breakdown_map.get("rates")):
            mapping = _as_mapping(container)
            for key, value in mapping.items():
                try:
                    self.rates[str(key)] = float(value)
                except Exception:
                    continue

        self.cfg_labor_rate_value = (
            self.default_labor_rate if self.separate_labor_cfg else 0.0
        )

        self.nre_detail: dict[str, Any] = _as_mapping(breakdown_map.get("nre_detail"))
        self.nre: dict[str, Any] = _as_mapping(breakdown_map.get("nre"))
        breakdown_map["nre"] = self.nre
        self.nre_cost_details: dict[str, Any] = _as_mapping(
            breakdown_map.get("nre_cost_details")
        )

        self.programming_rate = _coerce_rate_value(self.nre.get("programming_rate"))
        if self.programming_rate <= 0:
            self.programming_rate = _coerce_rate_value(self.rates.get("ProgrammerRate"))
        if self.programming_rate <= 0:
            self.programming_rate = _coerce_rate_value(self.rates.get("ProgrammingRate"))
        self.programming_per_part = _coerce_rate_value(
            self.nre.get("programming_per_part")
        )
        self.fixture_per_part = _coerce_rate_value(self.nre.get("fixture_per_part"))

        labor_costs_raw = _as_mapping(breakdown_map.get("labor_costs"))
        self.labor_cost_totals: dict[str, float] = {}
        for label, value in labor_costs_raw.items():
            canonical_label, _ = _canonical_amortized_label(label)
            if not canonical_label:
                canonical_label = str(label)
            try:
                numeric = float(value)
            except Exception:
                continue
            self.labor_cost_totals[canonical_label] = (
                self.labor_cost_totals.get(canonical_label, 0.0) + numeric
            )
        breakdown_map["labor_costs"] = self.labor_cost_totals

        self.amortized_totals: dict[str, float] = {}
        self.amortized_nre_total: float = 0.0
        for label, value in self.labor_cost_totals.items():
            _, is_amortized = _canonical_amortized_label(label)
            if not is_amortized:
                continue
            try:
                self.amortized_nre_total += float(value or 0.0)
            except Exception:
                continue

        qty_candidate = result_map.get("qty")
        if qty_candidate in (None, "", 0):
            qty_candidate = breakdown_map.get("qty")
        if qty_candidate in (None, "", 0):
            decision_state = _as_mapping(result_map.get("decision_state"))
            baseline = _as_mapping(decision_state.get("baseline")) if decision_state else {}
            qty_candidate = baseline.get("qty")
        try:
            self.qty = int(qty_candidate or 1)
        except Exception:
            try:
                self.qty = int(float(qty_candidate or 1))
            except Exception:
                self.qty = 1
        if self.qty <= 0:
            self.qty = 1

        if self.material_warning_summary:
            self.warning_flags["material_warning"] = True

    def section(self) -> SectionWriter:
        return SectionWriter(self)

    def new_section(self, title: str) -> SectionBuilder:
        """Return a builder for a section titled ``title``."""

        return SectionBuilder(self, title)

    def add_section(self, title: str, lines: Sequence[Any]) -> RenderSection:
        """Register ``lines`` under ``title`` and return the section record."""

        normalized: list[str] = []
        for value in lines:
            if isinstance(value, DisplayRow):
                normalized.append(self.format_row(value))
            elif value in (None,):
                normalized.append("")
            else:
                normalized.append(str(value))

        start_index = len(self.lines or [])
        if self.lines is not None:
            self.lines.extend(normalized)

        doc_rows = [
            QuoteRow(index=start_index + offset, text=text)
            for offset, text in enumerate(normalized)
        ]
        doc_section = QuoteSection(title=title, rows=doc_rows)

        record = RenderSection(
            index=len(self.sections),
            title=title,
            lines=list(normalized),
            start_index=start_index,
            doc_section=doc_section,
        )
        self.sections.append(record)
        return record

    def defer_replacement(self, index: int, text: str) -> None:
        """Queue a line replacement to be applied after section rendering."""

        self.deferred_replacements.append((index, text))

    def apply_replacements(self, replace: Callable[[int, str], None]) -> None:
        """Apply all deferred replacements using *replace* and clear the queue."""

        while self.deferred_replacements:
            index, text = self.deferred_replacements.pop(0)
            replace(index, text)

    def _format_display_row(self, row: DisplayRow) -> str:
        label = f"{row.indent}{row.label}"
        if row.kind == "hours":
            right = format_hours(row.value)
        else:
            right = format_currency(row.value, self.currency)

        right = right or ""
        right_width = max(len(right), 1)
        pad = max(1, self.page_width - len(label) - len(right))
        left_width = len(label) + pad
        table_text = draw_kv_table(
            [(label, right)],
            left_width=left_width,
            right_width=right_width,
            left_align="L",
            right_align="R",
        )
        for line in table_text.splitlines():
            if line.startswith("|") and line.endswith("|"):
                body = line[1:-1]
                try:
                    left_segment, right_segment = body.split("|", 1)
                except ValueError:
                    continue
                return f"{left_segment}{right_segment}"
        return f"{label}{' ' * pad}{right}"

    def format_row(
        self,
        label_or_row: DisplayRow | str,
        value: Any | None = None,
        *,
        indent: str = "",
        kind: str | None = None,
    ) -> str:
        if isinstance(label_or_row, DisplayRow):
            row = label_or_row
        else:
            try:
                numeric = float(value) if value is not None else 0.0
            except Exception:
                numeric = 0.0
            row_kind = "hours" if (kind or "").lower() == "hours" else "money"
            row = DisplayRow(label=str(label_or_row), value=numeric, indent=indent, kind=row_kind)
        return self._format_display_row(row)

    def format_rows(self, rows: Sequence[DisplayRow]) -> list[str]:
        return [self._format_display_row(row) for row in rows]

    def register_placeholder(
        self,
        name: str,
        section_index: int,
        line_index: int,
        *,
        formatter: Callable[[Any], str] | None = None,
    ) -> None:
        """Register a placeholder for later update."""

        self._placeholders[name] = PlaceholderRecord(
            name=name,
            section_index=section_index,
            line_index=line_index,
            formatter=formatter,
        )

    def update_placeholder(self, name: str, value: Any) -> None:
        """Update the placeholder ``name`` with ``value`` if registered."""

        record = self._placeholders.get(name)
        if record is None:
            return

        if record.formatter is not None:
            try:
                text = record.formatter(value)
            except Exception:
                text = str(value)
        else:
            text = str(value)

        if not text and text != "":
            text = ""

        try:
            section = self.sections[record.section_index]
        except IndexError:
            return

        if 0 <= record.line_index < len(section.lines):
            section.lines[record.line_index] = text

        global_index = section.start_index + record.line_index
        if 0 <= global_index < len(self.lines or []):
            self.lines[global_index] = text

        if section.doc_section is not None and 0 <= record.line_index < len(section.doc_section.rows):
            section.doc_section.rows[record.line_index].text = text

    def hours_with_rate(self, hours: Any, rate: Any) -> str:
        return format_hours_with_rate(hours, rate, self.currency)

