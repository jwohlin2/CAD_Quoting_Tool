from __future__ import annotations

import pytest

from cad_quoter.render import RenderState, render_quote_sections
from cad_quoter.utils.render_utils import QuoteDocRecorder


@pytest.fixture
def minimal_state() -> RenderState:
    payload: dict[str, object] = {"qty": 1, "breakdown": {}}
    state = RenderState(payload, page_width=74)
    state.lines = []
    state.recorder = QuoteDocRecorder(state.divider)
    return state


def test_render_quote_sections_emits_summary(minimal_state: RenderState) -> None:
    sections = render_quote_sections(minimal_state)

    assert sections, "at least one section should be rendered"
    header = sections[0]

    assert header[0] == "QUOTE SUMMARY - Qty 1"
    assert header[1] == minimal_state.divider
    assert header[2] == "Quote Summary (structured data attached below)"
    assert header[3] == "Speeds/Feeds CSV: (not set)"
    assert minimal_state.summary_lines[: len(header)] == header
    assert list(minimal_state.lines) == header + sections[1]
    assert minimal_state.deferred_replacements == []
