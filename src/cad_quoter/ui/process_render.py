"""Helpers for rendering process & labor sections of quotes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from collections.abc import Mapping as _MappingABC
from collections.abc import MutableMapping as _MutableMappingABC

from cad_quoter.config import logger
from cad_quoter.utils.render_utils.tables import render_process_sections

from .planner_render import (
    PROGRAMMING_AMORTIZED_LABEL,
    PROGRAMMING_PER_PART_LABEL,
    _canonical_bucket_key,
    _display_bucket_label,
    _normalize_bucket_key,
    _normalize_buckets,
    _prepare_bucket_view,
    _purge_legacy_drill_sync,
    _set_bucket_minutes_cost,
)


@dataclass
class RenderState:
    """Container for state shared while rendering the process section."""

    lines: list[str]
    why_lines: list[str]
    page_width: int
    format_money: Callable[[Any], str]
    add_process_notes: Callable[[str], None]
    render_bucket_table: Callable[[Sequence[tuple[str, float, float, float, float]]], None]
    bucket_view_obj: Mapping[str, Any] | None = None
    bucket_view_struct: Mapping[str, Any] | None = None
    bucket_seed_target: MutableMapping[str, Any] | Mapping[str, Any] | None = None
    process_plan_summary: Mapping[str, Any] | None = None
    extra_map: Mapping[str, Any] | None = None
    rates: Mapping[str, Any] | None = None
    drill_minutes_total: float = 0.0
    drill_machine_rate: float = 0.0
    drill_labor_rate: float = 0.0
    process_meta: Mapping[str, Any] | None = None
    label_overrides: Mapping[str, Any] | None = None
    labor_cost_totals: Mapping[str, Any] | None = None
    programming_minutes: Any | None = None
    breakdown: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None
    cfg: Any | None = None
    canonical_bucket_key: Callable[[str | None], str | None] = _canonical_bucket_key
    normalize_bucket_key: Callable[[str | None], str] = _normalize_bucket_key
    display_bucket_label: Callable[[str, Mapping[str, Any] | None], str] = _display_bucket_label
    programming_per_part_label: str = PROGRAMMING_PER_PART_LABEL
    programming_amortized_label: str = PROGRAMMING_AMORTIZED_LABEL
    config_sources: Sequence[Mapping[str, Any]] | None = None
    bucket_entries_for_totals_map: MutableMapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    bucket_view_for_render: Mapping[str, Any] | None = None
    process_rows_rendered: list[tuple[str, float, float, float, float]] = field(
        default_factory=list
    )
    process_total_cost: float = 0.0
    process_total_minutes: float = 0.0
    process_total_row_index: int = -1
    bucket_why_summary_line: str | None = None


@dataclass(frozen=True)
class ProcessRenderResult:
    """Return value for :func:`render_process`."""

    lines: list[str]
    why_lines: list[str]
    bucket_summary: str | None


def _as_mapping(candidate: Any) -> Mapping[str, Any] | None:
    if isinstance(candidate, (_MappingABC, dict)):
        return candidate  # type: ignore[return-value]
    return None


def _ensure_dict(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    if isinstance(candidate, dict):
        return candidate
    try:
        return dict(candidate)
    except Exception:
        return {}


def render_process(state: RenderState) -> ProcessRenderResult:
    """Render the process table and return the generated lines and summary text."""

    bucket_view_for_render: Mapping[str, Any] | None = None
    if isinstance(state.bucket_view_obj, _MappingABC):
        bucket_view_for_render = state.bucket_view_obj
    elif isinstance(state.bucket_view_struct, _MappingABC):
        bucket_view_for_render = state.bucket_view_struct

    bucket_seed_target: MutableMapping[str, Any] | None = None
    for candidate in (
        state.bucket_seed_target,
        state.bucket_view_obj,
        state.bucket_view_struct,
    ):
        if isinstance(candidate, _MutableMappingABC):
            bucket_seed_target = candidate
            break

    if bucket_seed_target is not None:
        _purge_legacy_drill_sync(bucket_seed_target)
        _set_bucket_minutes_cost(
            bucket_seed_target,
            "drilling",
            state.drill_minutes_total,
            state.drill_machine_rate,
            state.drill_labor_rate,
        )
        try:
            drilling_bucket_snapshot = (
                (bucket_seed_target.get("buckets") or {}).get("drilling")
                if isinstance(bucket_seed_target, _MappingABC)
                else None
            )
        except Exception:
            drilling_bucket_snapshot = None
        logger.info(
            "[bucket] drilling_minutes=%s drilling_bucket=%s",
            state.drill_minutes_total,
            drilling_bucket_snapshot,
        )

    if isinstance(state.bucket_view_obj, _MutableMappingABC):
        _normalize_buckets(state.bucket_view_obj)
    elif isinstance(state.bucket_view_struct, _MutableMappingABC):
        _normalize_buckets(state.bucket_view_struct)

    if isinstance(bucket_view_for_render, _MappingABC) and not isinstance(
        bucket_view_for_render, _MutableMappingABC
    ):
        bucket_view_for_render = _prepare_bucket_view(bucket_view_for_render)

    state.bucket_view_for_render = bucket_view_for_render

    buckets_for_totals: Mapping[str, Any] | None = None
    if isinstance(bucket_view_for_render, _MappingABC):
        try:
            candidate = bucket_view_for_render.get("buckets")
        except Exception:
            candidate = None
        if isinstance(candidate, dict):
            buckets_for_totals = candidate
        elif isinstance(candidate, _MappingABC):
            try:
                buckets_for_totals = dict(candidate)
            except Exception:
                buckets_for_totals = {}

    if buckets_for_totals is None:
        buckets_for_totals = {}

    state.bucket_entries_for_totals_map.clear()

    preferred_bucket_order = [
        "programming",
        "milling",
        "drilling",
        "tapping",
        "counterbore",
        "spot_drill",
        "jig_grind",
        "inspection",
    ]

    seen_bucket_keys: set[str] = set()
    for bucket_key in list(preferred_bucket_order) + [
        key for key in buckets_for_totals if key not in preferred_bucket_order
    ]:
        entry = buckets_for_totals.get(bucket_key)
        if not isinstance(entry, _MappingABC) or bucket_key in seen_bucket_keys:
            continue
        seen_bucket_keys.add(bucket_key)

        canon_key = (
            state.canonical_bucket_key(bucket_key)
            or state.normalize_bucket_key(bucket_key)
            or str(bucket_key)
        )
        normalized_key = state.normalize_bucket_key(bucket_key)
        display_label = state.display_bucket_label(canon_key, state.label_overrides)

        lookup_keys = {
            str(bucket_key),
            canon_key,
            normalized_key,
            display_label,
        }
        for lookup_key in lookup_keys:
            if lookup_key:
                state.bucket_entries_for_totals_map[str(lookup_key)] = entry

    config_sources: list[Mapping[str, Any]] = []
    if state.config_sources:
        for source in state.config_sources:
            mapping = _as_mapping(source)
            if mapping is not None:
                config_sources.append(_ensure_dict(mapping))

    def _collect_config_sources(container: Mapping[str, Any] | None) -> None:
        if not isinstance(container, _MappingABC):
            return
        for key in ("config", "params"):
            try:
                candidate = container.get(key)
            except Exception:
                candidate = None
            mapping = _as_mapping(candidate)
            if mapping is not None:
                config_sources.append(_ensure_dict(mapping))

    _collect_config_sources(_as_mapping(state.breakdown))
    _collect_config_sources(_as_mapping(state.result))
    result_mapping = _as_mapping(state.result)
    if isinstance(result_mapping, _MappingABC):
        quote_state_payload = result_mapping.get("quote_state")
        _collect_config_sources(_as_mapping(quote_state_payload))

    process_section_lines, process_rows_total, process_rows_minutes, process_rows_rendered = (
        render_process_sections(
            bucket_view_for_render,
            process_meta=_as_mapping(state.process_meta),
            rates=_as_mapping(state.rates),
            cfg=state.cfg,
            label_overrides=_as_mapping(state.label_overrides),
            format_money=state.format_money,
            add_process_notes=state.add_process_notes,
            config_sources=config_sources,
            labor_cost_totals=_as_mapping(state.labor_cost_totals),
            programming_minutes=state.programming_minutes,
            page_width=state.page_width,
            canonical_bucket_key=state.canonical_bucket_key,
            normalize_bucket_key=state.normalize_bucket_key,
            display_bucket_label=state.display_bucket_label,
            programming_per_part_label=state.programming_per_part_label,
            programming_amortized_label=state.programming_amortized_label,
        )
    )

    state.process_rows_rendered = list(process_rows_rendered)
    state.process_total_cost = process_rows_total
    state.process_total_minutes = process_rows_minutes

    new_why_lines: list[str] = []
    bucket_summary_line: str | None = state.bucket_why_summary_line

    machine_sum = sum(row[2] for row in process_rows_rendered)
    labor_sum = sum(row[3] for row in process_rows_rendered)
    if process_rows_rendered:
        top_rows = sorted(
            process_rows_rendered,
            key=lambda r: r[4],
            reverse=True,
        )[:3]
        for name, _minutes, _machine, _labor, total in top_rows:
            new_why_lines.append(f"{name} ${total:,.2f}")

        summary_bits: list[str] = [
            f"Machine {state.format_money(machine_sum)}",
            f"Labor {state.format_money(labor_sum)}",
        ]
        top_summary = [
            f"{name} {state.format_money(total)}"
            for (name, _m, _mc, _lb, total) in top_rows
            if total > 0
        ]
        if top_summary:
            summary_bits.append("largest bucket(s): " + ", ".join(top_summary))
        bucket_summary_line = "Process buckets — " + "; ".join(summary_bits)

    process_section_start = len(state.lines)
    total_row_index = -1
    if process_rows_total or process_rows_minutes:
        for offset, text in enumerate(process_section_lines):
            stripped = str(text or "").strip()
            if stripped.lower().startswith("total") and "$" in stripped:
                total_row_index = process_section_start + offset
                break

    state.process_total_row_index = total_row_index
    state.bucket_why_summary_line = bucket_summary_line

    return ProcessRenderResult(
        lines=process_section_lines,
        why_lines=new_why_lines,
        bucket_summary=bucket_summary_line,
    )


__all__ = ["RenderState", "ProcessRenderResult", "render_process"]
