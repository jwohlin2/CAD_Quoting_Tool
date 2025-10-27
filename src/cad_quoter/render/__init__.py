"""Rendering entrypoints for the modular quote document builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping

from cad_quoter.app.quote_doc import _sanitize_render_text
from cad_quoter.utils.rendering import QuoteDocRecorder

from .appendix import render_appendix
from .material import render_material
from .nre import render_nre
from .pass_through import render_pass_through
from .process import render_process
from .summary import render_summary

Sanitizer = Callable[[Any], str]
Callback = Callable[..., Any]


@dataclass(slots=True)
class RenderState:
    """Container shared by the quote rendering sections."""

    result: Mapping[str, Any] | None
    breakdown: Mapping[str, Any] | None
    totals: Mapping[str, Any] | None
    recorder: QuoteDocRecorder
    sanitize_text: Sanitizer = _sanitize_render_text
    callbacks: MutableMapping[str, Callback] | None = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def get_callback(self, name: str, default: Callback | None = None) -> Callback | None:
        """Return a registered callback for *name* if available."""

        if self.callbacks is None:
            return default
        return self.callbacks.get(name, default)


SECTION_RENDERERS: tuple[Callable[[RenderState], list[str]], ...] = (
    render_summary,
    render_material,
    render_nre,
    render_process,
    render_pass_through,
    render_appendix,
)


def render_quote_sections(state: RenderState) -> list[str]:
    """Render the quote by delegating to each section renderer in order."""

    lines: list[str] = []
    for render in SECTION_RENDERERS:
        section_lines = render(state)
        if section_lines:
            lines.extend(section_lines)
    return lines


__all__ = ["RenderState", "render_quote_sections", "SECTION_RENDERERS"]
