from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cad_quoter.render import RenderState, render_quote_sections
from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import QuoteDocRecorder
from cad_quoter.app.quote_doc import build_quote_header_lines

from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload


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
    assert list(minimal_state.lines)[: len(header)] == header
    assert minimal_state.deferred_replacements == []
    assert isinstance(minimal_state.writer, QuoteWriter)
    assert minimal_state.summary_lines == list(minimal_state.lines)


def _read_snapshot_text() -> str:
    base_dir = Path(__file__).resolve().parents[1]
    snapshot_path = base_dir / "test_render_quote_snapshot_snapshots" / "quote_snapshot.txt"
    return snapshot_path.read_text(encoding="utf-8")


def _extract_snapshot_block(text: str, header: str) -> list[str]:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == header:
            start = index
            break
    if start is None:
        return []
    collected: list[str] = []
    for line in lines[start:]:
        collected.append(line)
        if not line.strip() and len(collected) > 1:
            break
    return collected


def test_render_quote_sections_material_and_nre_blocks() -> None:
    payload = _dummy_quote_payload()
    state = RenderState(
        copy.deepcopy(payload),
        page_width=74,
        drill_debug_entries=payload.get("drill_debug"),
    )
    state.lines = []
    state.recorder = QuoteDocRecorder(state.divider)

    sections = render_quote_sections(state)

    assert sections, "expected at least the summary block"
    flat_lines: list[str] = []
    for block in sections:
        flat_lines.extend(block)

    assert flat_lines == list(state.lines)
    assert state.summary_lines == list(state.lines)

    snapshot_text = _read_snapshot_text()
    expected_material = _extract_snapshot_block(snapshot_text, "Material & Stock")
    expected_nre = _extract_snapshot_block(snapshot_text, "NRE / Setup Costs (per lot)")

    breakdown = payload["breakdown"]
    expected_header, _ = build_quote_header_lines(
        qty=payload.get("qty", 1),
        result=payload,
        breakdown=breakdown,
        page_width=state.page_width,
        divider=state.divider,
        process_meta=breakdown.get("process_meta"),
        process_meta_raw=breakdown.get("process_meta_raw"),
        hour_summary_entries=breakdown.get("hour_summary", {}),
        cfg=None,
    )

    material_index = None
    nre_index = None
    for index, block in enumerate(sections):
        if index == 0:
            assert block == expected_header
        if block and block[0] == "Material & Stock":
            material_index = index
            assert block == expected_material
        if block and block[0] == "NRE / Setup Costs (per lot)":
            nre_index = index
            assert block == expected_nre

    assert material_index is not None and nre_index is not None
    assert 0 < material_index < nre_index

    assert state.deferred_replacements == []
    assert isinstance(state.writer, QuoteWriter)
    assert state.material_component_total == pytest.approx(420.0)
    assert state.material_net_cost == pytest.approx(420.0)
    assert state.sections == []
