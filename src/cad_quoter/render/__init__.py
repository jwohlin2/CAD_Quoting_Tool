"""Helpers for rendering structured quote sections."""

from __future__ import annotations

from .buckets import detect_planner_drilling, has_planner_drilling
from .header import render_header
from .appendix import render_appendix
from .material import render_material
from .nre import render_nre
from .pass_through import render_pass_through
from .process import render_process
from .state import RenderState
from .summary import render_summary
from .writer import QuoteWriter


def render_quote_sections(state: RenderState) -> list[list[str]]:
    """Return the quote sections emitted for ``state``."""

    summary_header, summary_suffix = render_summary(state)

    sections: list[list[str]] = []
    for block in (summary_header, summary_suffix):
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
    state.recorder = writer.recorder

    material_section = render_material(state)
    if material_section:
        sections.append(material_section)

    nre_section = render_nre(state)
    if nre_section:
        sections.append(nre_section)

    process_section = render_process(state)
    if process_section.lines:
        sections.append(process_section.lines)

    sections_before = len(state.sections)
    pass_through_section, pass_total, pass_labor_total = render_pass_through(state)
    setattr(state, "pass_through_total", pass_total)
    setattr(state, "pass_through_labor_total", pass_labor_total)
    if len(state.sections) > sections_before:
        del state.sections[sections_before:]
    if pass_through_section:
        sections.append(pass_through_section)

    appendix_section = render_appendix(state)
    if appendix_section:
        sections.append(appendix_section)

    final_writer = getattr(state, "writer", None)
    if isinstance(final_writer, QuoteWriter):
        state.lines = final_writer.lines
        state.summary_lines = list(final_writer.lines)
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
    "render_process",
    "render_appendix",
    "render_summary",
    "render_quote_sections",
]
