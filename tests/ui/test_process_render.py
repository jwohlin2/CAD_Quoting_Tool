from __future__ import annotations

from typing import Any

from cad_quoter.ui.process_render import RenderState, render_process


def _fmt_money(value: Any) -> str:
    return f"${float(value):,.2f}"


def test_render_process_returns_lines_and_summary() -> None:
    lines: list[str] = []
    why_lines: list[str] = []
    notes: list[str] = []
    bucket_entries: dict[str, dict[str, float]] = {}

    state = RenderState(
        lines=lines,
        why_lines=why_lines,
        page_width=60,
        format_money=_fmt_money,
        add_process_notes=notes.append,
        render_bucket_table=lambda rows: None,
        bucket_view_obj={
            "buckets": {
                "milling": {
                    "minutes": 30,
                    "machine$": 60,
                    "labor$": 40,
                    "total$": 100,
                },
                "drilling": {
                    "minutes": 15,
                    "machine$": 20,
                    "labor$": 10,
                    "total$": 30,
                },
            }
        },
        bucket_entries_for_totals_map=bucket_entries,
        breakdown={},
        result={},
        labor_cost_totals={},
    )

    result = render_process(state)

    assert result.lines[0] == "Process & Labor Costs"
    assert any(line.startswith("  Process") for line in result.lines)
    assert state.process_total_cost == sum(row[4] for row in state.process_rows_rendered)
    assert state.process_total_minutes == sum(row[1] for row in state.process_rows_rendered)
    total_row_index = next(
        (
            idx
            for idx, text in enumerate(result.lines)
            if text.strip().lower().startswith("total")
        ),
        -1,
    )
    assert state.process_total_row_index == total_row_index
    assert state.bucket_why_summary_line is not None
    assert "milling" in {key.lower() for key in bucket_entries}
    noted_labels = {note.lower() for note in notes}
    assert "milling" in noted_labels
    # Summary lines bubble up the top contributors.
    assert any(line.startswith("Milling $") for line in result.why_lines)
    assert result.bucket_summary is not None


def test_render_process_handles_empty_view() -> None:
    lines: list[str] = []
    why_lines: list[str] = []
    bucket_entries: dict[str, dict[str, float]] = {}

    state = RenderState(
        lines=lines,
        why_lines=why_lines,
        page_width=60,
        format_money=_fmt_money,
        add_process_notes=lambda label: None,
        render_bucket_table=lambda rows: None,
        bucket_entries_for_totals_map=bucket_entries,
        breakdown={},
        result={},
        labor_cost_totals={},
    )

    result = render_process(state)

    assert "(no bucket data)" in result.lines[-2]
    assert state.process_total_cost == sum(row[4] for row in state.process_rows_rendered)
    assert state.process_total_minutes == sum(row[1] for row in state.process_rows_rendered)
    assert state.process_total_row_index == -1
    assert state.bucket_why_summary_line is None
