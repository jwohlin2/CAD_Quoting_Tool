"""Helpers for rendering structured quote sections."""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any, Sequence

from cad_quoter.app.quote_doc import (
    build_quote_header_lines,
    _sanitize_render_text,
)

try:  # Python 3.11+: ``collections.abc`` already exports ``MutableMapping``
    from collections.abc import Mapping, MutableMapping
except ImportError:  # pragma: no cover - fallback for older versions
    from typing import Mapping, MutableMapping  # type: ignore

import textwrap

from .buckets import detect_planner_drilling, has_planner_drilling
from .material import render_material
from .nre import render_nre
from .pass_through import render_pass_through
from .state import RenderState


def _wrap_text(text: str, page_width: int, indent: str = "") -> list[str]:
    """Wrap *text* to ``page_width`` accounting for an optional indent."""

    clean = _sanitize_render_text(text).strip()
    if not clean:
        return []
    width = max(10, page_width - len(indent))
    wrapper = textwrap.TextWrapper(width=width)
    return [f"{indent}{segment}" for segment in wrapper.wrap(clean)]


def _render_drill_debug(
    entries: Sequence[str],
    *,
    page_width: int,
    divider: str,
) -> list[str]:
    """Return the formatted Drill Debug section for ``entries``."""

    if not entries:
        return []

    lines: list[str] = ["Drill Debug", divider]

    for entry in entries:
        text = _sanitize_render_text(entry).strip()
        if not text:
            continue
        if "\n" in text:
            normalized = text.lstrip()
            indent = "" if normalized.startswith("Material Removal Debug") else "  "
            for chunk in text.splitlines():
                chunk_text = _sanitize_render_text(chunk)
                if not chunk_text:
                    continue
                lines.append(f"{indent}{chunk_text}")
            if lines and lines[-1] != "":
                lines.append("")
        else:
            wrapped = _wrap_text(text, page_width, indent="  ")
            if wrapped:
                lines.extend(wrapped)

    if lines and lines[-1] != "":
        lines.append("")
    return lines


def render_summary(state: RenderState) -> tuple[list[str], list[str]]:
    """Return the QUOTE SUMMARY header and trailing separator lines."""

    header_lines, pricing_source_value = build_quote_header_lines(
        qty=state.qty,
        result=state.result,
        breakdown=state.breakdown,
        page_width=state.page_width,
        divider=state.divider,
        process_meta=state.process_meta,
        process_meta_raw=state.process_meta_raw,
        hour_summary_entries=state.hour_summary_entries,
        cfg=state.cfg,
    )
    state.pricing_source_value = pricing_source_value

    breakdown = state.breakdown if isinstance(state.breakdown, MutableMapping) else None
    result = state.result if isinstance(state.result, MutableMapping) else None

    if breakdown is not None:
        if pricing_source_value:
            breakdown["pricing_source"] = pricing_source_value
        else:
            breakdown.pop("pricing_source", None)

    if result is not None:
        app_meta_container = result.setdefault("app_meta", {})
        if isinstance(app_meta_container, MutableMapping):
            if pricing_source_value and str(pricing_source_value).strip().lower() == "planner":
                app_meta_container.setdefault("used_planner", True)

        decision_state = result.get("decision_state") if isinstance(result, Mapping) else None
        if isinstance(decision_state, MutableMapping):
            baseline_state = decision_state.get("baseline")
            if isinstance(baseline_state, MutableMapping):
                if pricing_source_value:
                    baseline_state["pricing_source"] = pricing_source_value
                else:
                    baseline_state.pop("pricing_source", None)

    suffix_lines: list[str] = []
    if state.material_warning_summary and state.material_warning_label:
        suffix_lines.append(state.material_warning_label)
    suffix_lines.append("")

    entries_raw = state.drill_debug_entries or []
    entries: list[str] = []
    for entry in entries_raw:
        if entry is None:
            continue
        text = str(entry).strip()
        if text:
            entries.append(text)

    if entries and state.llm_debug_enabled:
        def _dbg_sort(a: str, b: str) -> int:
            a_clean = a.strip().lower()
            b_clean = b.strip().lower()
            a_ok = a_clean.startswith("ok ")
            b_ok = b_clean.startswith("ok ")
            if a_ok and not b_ok:
                return -1
            if b_ok and not a_ok:
                return 1
            a_hdr = a_clean.startswith("material removal debug")
            b_hdr = b_clean.startswith("material removal debug")
            if a_hdr and not b_hdr:
                return 1
            if b_hdr and not a_hdr:
                return -1
            return 0

        try:
            ordered = sorted(entries, key=cmp_to_key(_dbg_sort))
        except Exception:
            ordered = entries

        suffix_lines.extend(
            _render_drill_debug(
                ordered,
                page_width=state.page_width,
                divider=state.divider,
            )
        )

    state.summary_lines = list(header_lines + suffix_lines)
    return header_lines, suffix_lines


def render_quote_sections(state: RenderState) -> list[list[str]]:
    """Return the quote sections emitted for ``state``.

    The current implementation focuses on the summary/header block but keeps
    a list interface so additional sections can be appended incrementally.
    """

    header_lines, suffix_lines = render_summary(state)

    sections: list[list[str]] = [header_lines]
    if suffix_lines:
        sections.append(suffix_lines)

    return sections


__all__ = [
    "RenderState",
    "detect_planner_drilling",
    "has_planner_drilling",
    "render_material",
    "render_nre",
    "render_pass_through",
    "render_summary",
    "render_quote_sections",
]
