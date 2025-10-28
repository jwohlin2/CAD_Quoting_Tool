"""Utilities for emitting legacy quote text output."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TYPE_CHECKING

from cad_quoter.app.quote_doc import _sanitize_render_text
from cad_quoter.utils.render_utils import (
    QuoteDocRecorder,
    format_currency,
    format_hours,
    format_hours_with_rate,
)
from cad_quoter.utils.render_utils.tables import draw_kv_table

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from collections.abc import MutableSequence


@dataclass
class QuotePlaceholder:
    """Handle for replacing a line emitted by :class:`QuoteWriter`."""

    writer: "QuoteWriter"
    index: int

    def replace(self, text: Any) -> None:
        """Replace the placeholder with ``text`` using the parent writer."""

        self.writer.replace(self.index, text)

    def __int__(self) -> int:  # pragma: no cover - trivial proxy
        return self.index


class _ObservableLines(list[str]):
    """List-like container that delegates mutations back to the writer."""

    def __init__(self, writer: "QuoteWriter") -> None:
        super().__init__()
        self._writer = writer

    def append(self, text: str) -> None:  # type: ignore[override]
        self._writer._append(text)

    def extend(self, values: Iterable[str]) -> None:  # type: ignore[override]
        for value in values:
            self._writer._append(value)


class QuoteWriter:
    """Helper that wraps line emission and recorder bookkeeping."""

    def __init__(
        self,
        *,
        divider: str,
        page_width: int,
        currency: str,
        recorder: QuoteDocRecorder | None = None,
        lines: Sequence[str] | None = None,
    ) -> None:
        self.divider = divider
        self.page_width = max(10, int(page_width or 0))
        self.currency = currency
        self.recorder = recorder
        self._lines = _ObservableLines(self)
        if lines is not None:
            for line in lines:
                self._append(line)

    # ------------------------------------------------------------------
    # public helpers
    @property
    def lines(self) -> list[str]:
        """Return the mutable list of rendered lines."""

        return self._lines

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self._lines)

    # ------------------------------------------------------------------
    # low-level operations
    def _append(self, text: Any) -> int:
        sanitized = _sanitize_render_text(text)
        previous = self._lines[-1] if self._lines else None
        list.append(self._lines, sanitized)
        if self.recorder is not None:
            self.recorder.observe_line(len(self._lines) - 1, sanitized, previous)
        return len(self._lines) - 1

    def line(self, text: Any = "", *, indent: str = "") -> int:
        """Append ``text`` as a single line and return its index."""

        if indent:
            return self._append(f"{indent}{text}")
        return self._append(text)

    def blank(self) -> int:
        """Append an empty line and return its index."""

        return self.line("")

    def extend(self, values: Iterable[Any]) -> None:
        """Append each value from ``values`` as a separate line."""

        for value in values:
            self.line(value)

    def replace(self, index: int, text: Any) -> None:
        """Replace the line at ``index`` with ``text``."""

        sanitized = _sanitize_render_text(text)
        if 0 <= index < len(self._lines):
            self._lines[index] = sanitized
        if self.recorder is not None:
            self.recorder.replace_line(index, sanitized)

    # ------------------------------------------------------------------
    # higher-level helpers
    def wrap(self, text: Any, *, indent: str = "") -> None:
        """Emit ``text`` wrapped to the configured page width."""

        if text in (None, ""):
            return
        clean = _sanitize_render_text(text).strip()
        if not clean:
            return
        width = max(10, self.page_width - len(indent))
        wrapper = textwrap.TextWrapper(width=width)
        for chunk in wrapper.wrap(clean):
            self.line(chunk, indent=indent)

    def detail(self, text: Any, *, indent: str = "    ") -> None:
        """Emit a semicolon-delimited detail string."""

        if text in (None, ""):
            return
        clean = _sanitize_render_text(text)
        segments = [segment.strip() for segment in clean.split(";")]
        for segment in segments:
            if not segment:
                continue
            self.wrap(segment, indent=indent)

    def placeholder(self, text: Any = "") -> QuotePlaceholder:
        """Append ``text`` and return a placeholder handle for later update."""

        index = self.line(text)
        return QuotePlaceholder(self, index)

    # ------------------------------------------------------------------
    # row helpers
    def _is_total_label(self, label: str) -> bool:
        clean = str(label or "").strip()
        if not clean:
            return False
        clean = clean.rstrip(":")
        clean = clean.lstrip("= ")
        return clean.lower().startswith("total")

    def _maybe_insert_total_separator(self, width: int) -> None:
        if not self._lines:
            return
        width = max(0, int(width))
        if width <= 0:
            return
        if self._lines[-1] == self.divider:
            return
        pad = max(0, self.page_width - width)
        short_divider = " " * pad + "-" * width
        if self._lines[-1] == short_divider:
            return
        self.line(short_divider)

    def _render_kv_line(self, label: str, value: str, indent: str) -> str:
        left = f"{indent}{label}"
        right = value or ""
        right_width = max(len(right), 1)
        pad = max(1, self.page_width - len(left) - len(right))
        left_width = len(left) + pad
        table_text = draw_kv_table(
            [(left, right)],
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
        return f"{left}{' ' * pad}{right}"

    def row(self, label: str, value: Any, *, indent: str = "") -> int:
        """Emit a currency row."""

        formatted = format_currency(value, self.currency)
        if self._is_total_label(label):
            self._maybe_insert_total_separator(len(formatted))
        return self.line(self._render_kv_line(label, formatted, indent))

    def hours_row(self, label: str, value: Any, *, indent: str = "") -> int:
        """Emit an hours row."""

        formatted = format_hours(value)
        if self._is_total_label(label):
            self._maybe_insert_total_separator(len(formatted))
        return self.line(self._render_kv_line(label, formatted, indent))

    def hours_with_rate(self, hours: Any, rate: Any) -> str:
        """Return a formatted hours-with-rate string using the configured currency."""

        return format_hours_with_rate(hours, rate, self.currency)


__all__ = ["QuotePlaceholder", "QuoteWriter"]

