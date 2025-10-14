"""Utilities for formatting quote output and building renderable documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_currency(value: Any, currency: str) -> str:
    """Return *value* formatted as a currency string."""

    try:
        amount = float(value or 0.0)
    except Exception:
        amount = 0.0
    return f"{currency}{amount:,.2f}"


def format_hours(value: Any) -> str:
    """Format an hour value with a ``hr`` suffix."""

    try:
        hours = float(value or 0.0)
    except Exception:
        hours = 0.0
    return f"{hours:.2f} hr"


def format_hours_with_rate(hours: Any, rate: Any, currency: str) -> str:
    """Return a human readable ``hours × rate`` string."""

    try:
        hours_val = float(hours or 0.0)
    except Exception:
        hours_val = 0.0
    try:
        rate_val = float(rate or 0.0)
    except Exception:
        rate_val = 0.0
    hours_text = format_hours(hours_val)
    if rate_val <= 0:
        return f"{hours_text} @ —/hr"
    rate_text = format_currency(rate_val, currency)
    return f"{hours_text} @ {rate_text}/hr"


def format_percent(value: Any) -> str:
    """Return ``value`` as a percentage with a single decimal place."""

    try:
        pct = float(value or 0.0)
    except Exception:
        pct = 0.0
    return f"{pct * 100:.1f}%"


def format_dimension(value: Any) -> str:
    """Render numeric dimensions while keeping existing text untouched."""

    if isinstance(value, (int, float)):
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
        return text or "0"
    if value is None:
        return "—"
    text = str(value).strip()
    return text if text else "—"


def format_weight_lb_decimal(mass_g: float | None) -> str:
    """Convert grams into decimal pounds."""

    if mass_g is None:
        grams = 0.0
    else:
        try:
            grams = float(mass_g or 0.0)
        except Exception:
            grams = 0.0
    grams = max(0.0, grams)
    pounds = grams / 1000.0 * 2.2046226218487757
    if pounds <= 0:
        return "0.00 lb"
    text = f"{pounds:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text} lb"


def format_weight_lb_oz(mass_g: float | None) -> str:
    """Convert grams into a pounds/ounces breakdown."""

    if mass_g is None:
        grams = 0.0
    else:
        try:
            grams = float(mass_g or 0.0)
        except Exception:
            grams = 0.0
    grams = max(0.0, grams)
    if grams <= 0:
        return "0 oz"
    pounds_total = grams / 1000.0 * 2.2046226218487757
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


# ---------------------------------------------------------------------------
# Data schema for rendering
# ---------------------------------------------------------------------------


@dataclass
class QuoteRow:
    """Single rendered line of quote output."""

    index: int
    text: str


@dataclass
class QuoteSection:
    """Group of related rows under an optional title."""

    title: str | None
    rows: list[QuoteRow] = field(default_factory=list)


@dataclass
class QuoteDoc:
    """Structured representation of a rendered quote."""

    title: str
    sections: list[QuoteSection] = field(default_factory=list)


class QuoteDocRecorder:
    """Track lines emitted by the legacy renderer and build a schema."""

    def __init__(self, divider: str) -> None:
        self._divider = divider
        self._title: str | None = None
        self._sections: list[QuoteSection] = []
        self._current_section: QuoteSection | None = None
        self._line_map: dict[int, QuoteRow] = {}

    def observe_line(self, index: int, line: str, previous: str | None) -> None:
        """Record a raw line emitted by the renderer."""

        if self._title is None:
            self._title = line
            return
        if line == self._divider and previous is not None:
            # Promote the preceding line to a section title.
            if self._current_section and self._current_section.rows:
                last_row = self._current_section.rows[-1]
                if last_row.text == previous:
                    self._line_map.pop(last_row.index, None)
                    self._current_section.rows.pop()
            section = QuoteSection(title=previous, rows=[])
            self._sections.append(section)
            self._current_section = section
            return
        if self._current_section is None:
            # Lazily create a container for preamble content.
            self._current_section = QuoteSection(title=None, rows=[])
            self._sections.append(self._current_section)
        row = QuoteRow(index=index, text=line)
        self._current_section.rows.append(row)
        self._line_map[index] = row

    def replace_line(self, index: int, text: str) -> None:
        if index == 0:
            self._title = text
            return
        row = self._line_map.get(index)
        if row is not None:
            row.text = text

    def build_doc(self) -> QuoteDoc:
        title = self._title or ""
        sections = [section for section in self._sections if section.rows or section.title]
        return QuoteDoc(title=title, sections=sections)


def render_quote_doc(doc: QuoteDoc, *, divider: str) -> str:
    """Render a :class:`QuoteDoc` back into legacy text output."""

    output: list[str] = []
    if doc.title:
        output.append(doc.title)
    for section in doc.sections:
        if section.title:
            if not output or output[-1] != section.title:
                output.append(section.title)
            output.append(divider)
        output.extend(row.text for row in section.rows)
    return "\n".join(output)
