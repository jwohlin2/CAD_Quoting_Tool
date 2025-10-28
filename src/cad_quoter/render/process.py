"""Legacy Process & Labor section renderer helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TYPE_CHECKING

from cad_quoter.utils.render_utils import format_currency
from .writer import QuoteWriter

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from . import RenderState


@dataclass(frozen=True)
class ProcessRenderResult:
    """Structured return value for :func:`render_process`."""

    lines: list[str]
    why_lines: list[str]
    bucket_summary: str | None
    total_cost: float
    total_minutes: float
    machine_total: float
    labor_total: float
    rows: list[tuple[str, float, float, float, float]]


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    if value in (None, ""):
        return []
    return [value]


def _coerce_rows(rows: Iterable[Any]) -> list[tuple[str, float, float, float, float]]:
    coerced: list[tuple[str, float, float, float, float]] = []
    for entry in rows:
        if isinstance(entry, Sequence) and len(entry) >= 5:
            label = str(entry[0])
            try:
                minutes = float(entry[1])
            except Exception:
                minutes = 0.0
            try:
                machine = float(entry[2])
            except Exception:
                machine = 0.0
            try:
                labor = float(entry[3])
            except Exception:
                labor = 0.0
            try:
                total = float(entry[4])
            except Exception:
                total = machine + labor
            coerced.append((label, minutes, machine, labor, total))
    return coerced


def _writer_from_state(state: "RenderState") -> QuoteWriter | None:
    candidate = getattr(state, "writer", None)
    if isinstance(candidate, QuoteWriter):
        return candidate
    return None


def render_process(state: "RenderState") -> ProcessRenderResult:
    """Render the Process & Labor section for ``state``.

    The helper expects ``state`` to expose ``process_render_state`` and
    ``process_render_result`` attributes populated from the legacy planner
    renderer.  When a :class:`~cad_quoter.render.writer.QuoteWriter` instance is
    attached to ``state`` via ``state.writer`` the emitted lines are routed
    through the writer so downstream recorders observe the same mutations as the
    inline implementation in :mod:`appV5`.
    """

    process_state = getattr(state, "process_render_state", None)
    process_result = getattr(state, "process_render_result", None)

    if process_state is None or process_result is None:
        state.process_total_row_index = -1
        return ProcessRenderResult(
            lines=[],
            why_lines=[],
            bucket_summary=None,
            total_cost=0.0,
            total_minutes=0.0,
            machine_total=0.0,
            labor_total=0.0,
            rows=[],
        )

    raw_lines = [str(line) for line in _as_sequence(getattr(process_result, "lines", []))]
    writer = _writer_from_state(state)

    if writer is not None:
        start_index = len(writer.lines)
        for line in raw_lines:
            writer.line(line)
        emitted_lines = list(writer.lines[start_index:])
    else:
        if state.lines is None:
            state.lines = []
        start_index = len(state.lines)
        state.lines.extend(raw_lines)
        emitted_lines = list(raw_lines)

    rows = _coerce_rows(getattr(process_state, "process_rows_rendered", ()))
    total_cost = float(getattr(process_state, "process_total_cost", 0.0) or 0.0)
    total_minutes = float(getattr(process_state, "process_total_minutes", 0.0) or 0.0)

    machine_total = sum(row[2] for row in rows)
    labor_total = sum(row[3] for row in rows)

    top_rows = sorted(rows, key=lambda row: row[4], reverse=True)[:3]
    why_lines = [f"{name} ${total:,.2f}" for name, *_rest, total in top_rows]

    bucket_summary: str | None = getattr(process_result, "bucket_summary", None)
    if not bucket_summary and rows:
        summary_bits = [
            f"Machine {format_currency(machine_total, state.currency)}",
            f"Labor {format_currency(labor_total, state.currency)}",
        ]
        top_summary = [
            f"{name} {format_currency(total, state.currency)}"
            for name, *_rest, total in top_rows
            if total > 0
        ]
        if top_summary:
            summary_bits.append("largest bucket(s): " + ", ".join(top_summary))
        bucket_summary = "Process buckets — " + "; ".join(summary_bits)

    state.bucket_why_summary_line = bucket_summary
    state.process_total_row_index = -1
    if emitted_lines and (total_cost or total_minutes):
        for offset, text in enumerate(emitted_lines):
            stripped = str(text or "").strip()
            if stripped.lower().startswith("total") and "$" in stripped:
                state.process_total_row_index = start_index + offset
                break

    setattr(state, "process_rows_rendered", rows)
    setattr(state, "process_total_cost", total_cost)
    setattr(state, "process_total_minutes", total_minutes)

    return ProcessRenderResult(
        lines=emitted_lines,
        why_lines=why_lines,
        bucket_summary=bucket_summary,
        total_cost=total_cost,
        total_minutes=total_minutes,
        machine_total=machine_total,
        labor_total=labor_total,
        rows=rows,
    )


__all__ = ["ProcessRenderResult", "render_process"]
