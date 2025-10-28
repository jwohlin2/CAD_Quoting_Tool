"""Snapshot regression tests comparing the legacy and modular quote renderers."""

from __future__ import annotations

import copy
import json
import logging
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Iterable

import pytest


def _ensure_stub_modules(names: Iterable[str]) -> None:
    for name in names:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__spec__ = ModuleSpec(name, loader=None)
        sys.modules[name] = module


_ensure_stub_modules(("requests", "bs4", "lxml"))

import appV5  # noqa: E402  (import after stub setup)

from cad_quoter.render import detect_planner_drilling, has_planner_drilling
from cad_quoter.render.config import apply_render_overrides, ensure_mutable_breakdown
from cad_quoter.render.guards import render_drilling_guard
from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import QuoteDocRecorder, render_quote_doc


setattr(appV5, "apply_render_overrides", getattr(appV5, "apply_render_overrides", apply_render_overrides))
setattr(appV5, "ensure_mutable_breakdown", getattr(appV5, "ensure_mutable_breakdown", ensure_mutable_breakdown))
setattr(appV5, "QuoteWriter", getattr(appV5, "QuoteWriter", QuoteWriter))
setattr(appV5, "detect_planner_drilling", getattr(appV5, "detect_planner_drilling", detect_planner_drilling))
setattr(appV5, "render_state_has_planner_drilling", getattr(appV5, "render_state_has_planner_drilling", has_planner_drilling))
setattr(appV5, "render_drilling_guard", getattr(appV5, "render_drilling_guard", render_drilling_guard))
setattr(appV5, "QuoteDocRecorder", getattr(appV5, "QuoteDocRecorder", QuoteDocRecorder))

FIXTURE_DIR = Path(__file__).with_name("pricing") / "fixtures"
SNAPSHOT_DIR = Path(__file__).with_name("test_render_quote_snapshot_snapshots")

_FIXTURE_NAMES = sorted(path.stem for path in FIXTURE_DIR.glob("*.json"))


def _normalize_render_text(text: str) -> str:
    lines = text.splitlines()
    normalized = [line.rstrip(" ") for line in lines]
    while normalized and not normalized[-1]:
        normalized.pop()
    return "\n".join(normalized) + "\n"


def _load_fixture(name: str) -> dict:
    fixture_path = FIXTURE_DIR / f"{name}.json"
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class RecordingQuoteDocRecorder(QuoteDocRecorder):
    """Recorder that exposes raw legacy lines for regression comparisons."""

    latest: "RecordingQuoteDocRecorder | None" = None

    def __init__(self, divider: str) -> None:
        super().__init__(divider)
        self._raw_lines: dict[int, str] = {}
        RecordingQuoteDocRecorder.latest = self

    def observe_line(self, index: int, line: str, previous: str | None) -> None:  # type: ignore[override]
        self._raw_lines[index] = line
        super().observe_line(index, line, previous)

    def replace_line(self, index: int, text: str) -> None:  # type: ignore[override]
        self._raw_lines[index] = text
        super().replace_line(index, text)

    def legacy_text(self) -> str:
        if not self._raw_lines:
            return ""
        max_index = max(self._raw_lines)
        lines = [self._raw_lines.get(idx, "") for idx in range(max_index + 1)]
        return "\n".join(lines)


def _render_legacy_and_modern(payload: dict) -> tuple[str, str]:
    recorder_cls = RecordingQuoteDocRecorder
    recorder_cls.latest = None
    original_recorder = getattr(appV5, "QuoteDocRecorder", QuoteDocRecorder)
    payload_copy = copy.deepcopy(payload)

    setattr(appV5, "QuoteDocRecorder", recorder_cls)
    logging.disable(logging.CRITICAL)
    try:
        rendered = appV5.render_quote(payload_copy, currency="$", show_zeros=False)
    finally:
        logging.disable(logging.NOTSET)
        setattr(appV5, "QuoteDocRecorder", original_recorder)

    recorder = recorder_cls.latest
    if recorder is None:
        raise AssertionError("QuoteDocRecorder did not capture any output")

    doc = recorder.build_doc()
    modern_text = render_quote_doc(doc, divider=recorder._divider)

    assert _normalize_render_text(rendered) == _normalize_render_text(modern_text)
    return rendered, modern_text


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_render_quote_matches_snapshot(fixture_name: str) -> None:
    payload = _load_fixture(fixture_name)

    legacy_text, modern_text = _render_legacy_and_modern(payload)

    assert _normalize_render_text(modern_text) == _normalize_render_text(legacy_text)

    normalized = _normalize_render_text(legacy_text)
    snapshot_path = SNAPSHOT_DIR / f"{fixture_name}.txt"

    if not snapshot_path.exists():
        raise AssertionError(
            "Snapshot missing at "
            f"{snapshot_path}. Generate it from the current expected output."
        )

    expected = snapshot_path.read_text()
    assert normalized == expected
