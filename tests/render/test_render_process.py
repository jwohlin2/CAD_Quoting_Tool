from __future__ import annotations

from types import SimpleNamespace

from cad_quoter.render import RenderState
from cad_quoter.render.process import render_process
from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import QuoteDocRecorder, format_currency


def _make_state() -> RenderState:
    payload: dict[str, object] = {"qty": 1, "breakdown": {}}
    state = RenderState(payload, page_width=72)
    recorder = QuoteDocRecorder(state.divider)
    writer = QuoteWriter(
        divider=state.divider,
        page_width=state.page_width,
        currency=state.currency,
        recorder=recorder,
    )
    state.recorder = recorder
    state.lines = writer.lines
    setattr(state, "writer", writer)
    return state


def test_render_process_appends_lines_and_totals() -> None:
    state = _make_state()

    process_rows = [
        ("Milling", 30.0, 100.0, 50.0, 150.0),
        ("Drilling", 15.0, 20.0, 30.0, 50.0),
    ]
    state.process_render_state = SimpleNamespace(
        process_rows_rendered=list(process_rows),
        process_total_cost=200.0,
        process_total_minutes=45.0,
    )
    state.process_render_result = SimpleNamespace(
        lines=["Process & Labor Costs", "  Total           $200.00"],
        why_lines=["legacy"],
        bucket_summary=None,
    )

    result = render_process(state)

    writer = getattr(state, "writer")
    assert writer is not None
    assert writer.lines == ["Process & Labor Costs", "  Total           $200.00"]

    assert result.lines == writer.lines
    assert result.total_cost == 200.0
    assert result.total_minutes == 45.0
    assert result.machine_total == 120.0
    assert result.labor_total == 80.0
    assert result.rows == process_rows
    assert state.process_total_row_index == 1

    expected_why = ["Milling $150.00", "Drilling $50.00"]
    assert result.why_lines == expected_why

    expected_summary = "Process buckets — " + "; ".join(
        [
            f"Machine {format_currency(120.0, state.currency)}",
            f"Labor {format_currency(80.0, state.currency)}",
            "largest bucket(s): "
            + ", ".join(
                [
                    f"Milling {format_currency(150.0, state.currency)}",
                    f"Drilling {format_currency(50.0, state.currency)}",
                ]
            ),
        ]
    )
    assert result.bucket_summary == expected_summary
    assert state.bucket_why_summary_line == expected_summary

