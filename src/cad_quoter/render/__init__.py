"""Helpers for rendering structured quote sections."""

from __future__ import annotations

from .buckets import detect_planner_drilling, has_planner_drilling
from .header import render_header
from .material import render_material
from .nre import render_nre
from .pass_through import render_pass_through
from .state import RenderState
from .summary import render_summary


def render_quote_sections(state: RenderState) -> list[list[str]]:
    """Return the quote sections emitted for ``state``."""

    summary_sections = render_summary(state)

    sections: list[list[str]] = []
    for block in summary_sections:
        if block:
            sections.append(block)

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
