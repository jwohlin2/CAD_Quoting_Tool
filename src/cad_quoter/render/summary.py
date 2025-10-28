"""Summary section renderer built on :class:`~cad_quoter.render.writer.QuoteWriter`."""

from __future__ import annotations

from functools import cmp_to_key
import textwrap
from typing import TYPE_CHECKING, Iterable, Sequence

from cad_quoter.app.quote_doc import _sanitize_render_text

from .header import apply_pricing_source, render_header
from .writer import QuoteWriter

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from .state import RenderState


def _wrap_text(text: str, page_width: int, indent: str = "") -> list[str]:
    """Wrap ``text`` to the configured ``page_width`` honouring ``indent``."""

    clean = _sanitize_render_text(text).strip()
    if not clean:
        return []
    width = max(10, page_width - len(indent))
    wrapper = textwrap.TextWrapper(width=width)
    return [f"{indent}{segment}" for segment in wrapper.wrap(clean)]


def _render_drill_debug(
    entries: Sequence[str],
    *,
    page_width: int,
    divider: str,
) -> list[str]:
    """Return formatted drill debug lines for ``entries``."""

    if not entries:
        return []

    lines: list[str] = ["Drill Debug", divider]

    for entry in entries:
        text = _sanitize_render_text(entry).strip()
        if not text:
            continue
        if "\n" in text:
            normalized = text.lstrip()
            indent = "" if normalized.startswith("Material Removal Debug") else "  "
            for chunk in text.splitlines():
                chunk_text = _sanitize_render_text(chunk)
                if not chunk_text:
                    continue
                lines.append(f"{indent}{chunk_text}")
            if lines and lines[-1] != "":
                lines.append("")
        else:
            wrapped = _wrap_text(text, page_width, indent="  ")
            if wrapped:
                lines.extend(wrapped)

    if lines and lines[-1] != "":
        lines.append("")

    return lines


def _normalize_drill_debug_entries(entries: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for entry in entries:
        if entry is None:
            continue
        try:
            text = str(entry).strip()
        except Exception:
            continue
        if text:
            normalized.append(text)
    return normalized


def render_summary(state: "RenderState") -> tuple[list[str], list[str]]:
    """Emit the summary header and trailing sections for ``state``."""

    divider = state.divider or "-" * max(1, state.page_width)

    writer = QuoteWriter(
        divider=divider,
        page_width=state.page_width,
        currency=state.currency,
        recorder=state.recorder,
        lines=state.lines,
    )

    # Ensure downstream sections share the observable writer state
    state.lines = writer.lines

    header_start = len(writer.lines)
    header_lines = render_header(state)
    apply_pricing_source(state, state.breakdown)
    apply_pricing_source(state, state.result)
    writer.extend(header_lines)
    header_end = len(writer.lines)
    header_block = list(writer.lines[header_start:header_end])

    suffix_start = len(writer.lines)

    if state.material_warning_summary and state.material_warning_label:
        writer.line(state.material_warning_label)

    writer.blank()

    entries_raw: Sequence[object] | None = state.drill_debug_entries
    normalized_entries = _normalize_drill_debug_entries(entries_raw or ())

    if normalized_entries and state.llm_debug_enabled:

        def _dbg_sort(a: str, b: str) -> int:
            a_clean = a.strip().lower()
            b_clean = b.strip().lower()
            a_ok = a_clean.startswith("ok ")
            b_ok = b_clean.startswith("ok ")
            if a_ok and not b_ok:
                return -1
            if b_ok and not a_ok:
                return 1
            a_hdr = a_clean.startswith("material removal debug")
            b_hdr = b_clean.startswith("material removal debug")
            if a_hdr and not b_hdr:
                return 1
            if b_hdr and not a_hdr:
                return -1
            return 0

        try:
            ordered_entries = sorted(normalized_entries, key=cmp_to_key(_dbg_sort))
        except Exception:
            ordered_entries = normalized_entries

        writer.extend(
            _render_drill_debug(
                ordered_entries,
                page_width=state.page_width,
                divider=divider,
            )
        )

    suffix_end = len(writer.lines)
    suffix_block = list(writer.lines[suffix_start:suffix_end])

    state.summary_lines = list(header_block + suffix_block)

    return header_block, suffix_block
