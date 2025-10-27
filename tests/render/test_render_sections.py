from __future__ import annotations

import pytest

from cad_quoter.render import RenderState, render_quote_sections
from cad_quoter.utils.rendering import QuoteDocRecorder


@pytest.fixture
def minimal_state() -> RenderState:
    return RenderState(
        result={},
        breakdown={},
        totals={},
        recorder=QuoteDocRecorder("---"),
        callbacks={},
    )


def _make_renderer(name: str):
    def _renderer(state: RenderState) -> list[str]:  # pragma: no cover - trivial
        return [name]

    return _renderer


def test_render_quote_sections_chains_outputs(monkeypatch: pytest.MonkeyPatch, minimal_state: RenderState) -> None:
    ordered_names = ["summary", "material", "nre", "process", "pass_through", "appendix"]
    monkeypatch.setattr(
        "cad_quoter.render.SECTION_RENDERERS",
        tuple(_make_renderer(name) for name in ordered_names),
        raising=False,
    )

    lines = render_quote_sections(minimal_state)

    assert lines == ordered_names
