"""Helpers for rendering structured quote sections."""

from __future__ import annotations

from .buckets import detect_planner_drilling, has_planner_drilling
from .header import render_header
from .material import render_material
from .nre import render_nre
from .pass_through import render_pass_through
from .state import RenderState
from .summary import render_summary
from .writer import QuoteWriter


def render_quote_sections(state: RenderState) -> list[list[str]]:
    """Return the quote sections emitted for ``state``."""

    summary_sections = render_summary(state)

    sections: list[list[str]] = []
    for block in summary_sections:
        if block:
            sections.append(block)

    writer = getattr(state, "writer", None)
    if not isinstance(writer, QuoteWriter):
        divider = state.divider or "-" * max(1, state.page_width)
        writer = QuoteWriter(
            divider=divider,
            page_width=state.page_width,
            currency=state.currency,
            recorder=state.recorder,
            lines=state.lines,
        )
        setattr(state, "writer", writer)
        state.lines = writer.lines

    material_section = render_material(state)
    if material_section:
        sections.append(material_section)

    nre_section = render_nre(state)
    if nre_section:
        sections.append(nre_section)

    if isinstance(writer, QuoteWriter):
        state.lines = writer.lines
        state.summary_lines = list(writer.lines)
    else:
        combined: list[str] = []
        for block in sections:
            combined.extend(block)
        state.lines = combined
        state.summary_lines = list(combined)

    return sections


__all__ = [
    "detect_planner_drilling",
    "has_planner_drilling",
    "render_header",
    "RenderState",
    "render_material",
    "render_nre",
    "render_pass_through",
    "render_summary",
    "render_quote_sections",
]
