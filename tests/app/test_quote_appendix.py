"""Regression tests for the appendix rendering helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import pytest

import appV5

from cad_quoter.app import quote_appendix
from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload

SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "test_render_quote_snapshot_snapshots"
    / "quote_snapshot.txt"
)


def _load_expected_appendix_lines() -> list[str]:
    """Return the appendix portion of the render snapshot."""

    text = SNAPSHOT_PATH.read_text()
    lines = text.splitlines()
    try:
        start = lines.index("Pricing Ladder")
    except ValueError as exc:  # pragma: no cover - guard for corrupted snapshot
        raise AssertionError("Snapshot is missing the appendix header") from exc
    return lines[start:]


def test_render_appendix_matches_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The appendix helper should render the historical snapshot verbatim."""

    captured: dict[str, object] = {}

    def _wrapper(state: quote_appendix.RenderState) -> quote_appendix.AppendixResult:
        result = quote_appendix.render_appendix(state)
        captured["result"] = result
        captured["final_price"] = state.get("final_price")
        captured["ladder_totals"] = state.get("ladder_totals")
        captured["appendix_lines"] = list(state.get("appendix_lines", ()))
        captured["all_lines"] = list(state.lines)
        return result

    monkeypatch.setattr(appV5, "render_quote_appendix", _wrapper)

    payload = _dummy_quote_payload()
    appV5.render_quote(copy.deepcopy(payload), currency="$", show_zeros=False)

    result = captured.get("result")
    assert isinstance(result, quote_appendix.AppendixResult)

    expected_lines = _load_expected_appendix_lines()
    actual_lines = list(result.lines)
    while actual_lines and actual_lines[-1] == "":
        actual_lines.pop()
    expected_trimmed = list(expected_lines)
    while expected_trimmed and expected_trimmed[-1] == "":
        expected_trimmed.pop()
    assert actual_lines == expected_trimmed
    appendix_lines = list(captured["appendix_lines"])  # type: ignore[index]
    while appendix_lines and appendix_lines[-1] == "":
        appendix_lines.pop()
    assert appendix_lines == expected_trimmed

    final_price = pytest.approx(result.final_price, rel=0, abs=1e-6)
    ladder_totals = captured["ladder_totals"]
    assert isinstance(ladder_totals, Mapping)
    assert ladder_totals["with_margin"] == final_price
    assert captured["final_price"] == final_price

    assert any("Final Price with Margin" in line for line in result.lines)
    assert result.quick_what_ifs, "Quick what-if scenarios should be populated"
    assert any("margin" in entry.get("label", "").lower() for entry in result.quick_what_ifs)
    assert result.margin_slider is not None
    assert pytest.approx(result.margin_slider.get("current_price", 0.0), rel=0, abs=1e-6) == final_price
    assert result.slider_sample_points
    assert any(point.get("is_current") for point in result.slider_sample_points)
    assert result.why_parts
    assert "Quote total" in " ".join(result.why_parts)
