"""Shared state helpers for quote rendering sections."""

from __future__ import annotations

from dataclasses import dataclass, field
import textwrap
from typing import Any, Callable, Iterable, Mapping, Sequence, TYPE_CHECKING

from cad_quoter.utils.render_utils import (
    QuoteDocRecorder,
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
    programming_rate: float = 0.0
    programming_per_part: float = 0.0
    fixture_per_part: float = 0.0

    def __post_init__(self) -> None:
        result_map = _as_mapping(self.payload)
        self.result: dict[str, Any] = result_map
        breakdown_map = _as_mapping(result_map.get("breakdown"))
        self.breakdown: dict[str, Any] = breakdown_map

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

    def section(self) -> SectionWriter:
        return SectionWriter(self)

    def defer_replacement(self, index: int, text: str) -> None:
        """Queue a line replacement to be applied after section rendering."""

        self.deferred_replacements.append((index, text))

    def apply_replacements(self, replace: Callable[[int, str], None]) -> None:
        """Apply all deferred replacements using *replace* and clear the queue."""

        while self.deferred_replacements:
            index, text = self.deferred_replacements.pop(0)
            replace(index, text)

    def format_row(self, row: DisplayRow) -> str:
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

    def format_rows(self, rows: Sequence[DisplayRow]) -> list[str]:
        return [self.format_row(row) for row in rows]

    def hours_with_rate(self, hours: Any, rate: Any) -> str:
        return format_hours_with_rate(hours, rate, self.currency)

