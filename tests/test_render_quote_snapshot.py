"""Snapshot regression test for quote rendering output."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import appV5

from cad_quoter.render import RenderState, render_quote_sections
from cad_quoter.utils.render_utils import QuoteDocRecorder
from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload

SNAPSHOT_PATH = (
    Path(__file__).with_name("test_render_quote_snapshot_snapshots") / "quote_snapshot.txt"
)


def _normalize_render_text(text: str) -> str:
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.startswith("QUOTE SUMMARY"):
            start = index
            break
    relevant = [line.rstrip() for line in lines[start:] if not line.startswith("[")]
    return "\n".join(relevant).rstrip() + "\n"


def test_render_quote_matches_snapshot() -> None:
    payload = _dummy_quote_payload()

    state = RenderState(
        payload,
        page_width=74,
        drill_debug_entries=payload.get("drill_debug"),
    )
    state.lines = []
    state.recorder = QuoteDocRecorder(state.divider)
    render_quote_sections(state)

    logging.disable(logging.CRITICAL)
    try:
        rendered = appV5.render_quote(copy.deepcopy(payload), currency="$", show_zeros=False)
    finally:
        logging.disable(logging.NOTSET)

    normalized = _normalize_render_text(rendered)

    if not SNAPSHOT_PATH.exists():
        raise AssertionError(
            "Snapshot missing at "
            f"{SNAPSHOT_PATH}. Generate it from the current expected output."
        )

    expected = SNAPSHOT_PATH.read_text()
    assert normalized == expected
