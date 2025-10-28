from __future__ import annotations

from cad_quoter.render import RenderState, render_summary
from cad_quoter.render.writer import QuoteWriter
from cad_quoter.app.quote_doc import build_quote_header_lines


def _divider(width: int) -> str:
    return "-" * width


def _sample_payload() -> dict:
    return {
        "qty": 12,
        "speeds_feeds_path": "/mnt/planner/feeds.csv",
        "app_meta": {},
        "decision_state": {
            "baseline": {
                "pricing_source": "planner",
            }
        },
        "drill_debug": [
            "OK drill group",
        ],
        "breakdown": {
            "qty": 12,
            "pricing_source": "planner",
            "drilling_meta": {
                "speeds_feeds_path": "/mnt/planner/feeds.csv",
                "speeds_feeds_loaded": True,
            },
        },
    }


def test_render_summary_matches_header_for_sample_payload() -> None:
    payload = _sample_payload()
    breakdown = payload["breakdown"]

    state = RenderState(
        payload,
        page_width=74,
        drill_debug_entries=payload.get("drill_debug"),
        hour_summary_entries={},
        material_warning_summary=False,
        material_warning_label="⚠ MATERIALS MISSING",
    )

    header_lines, suffix_lines = render_summary(state)

    expected_header, expected_source = build_quote_header_lines(
        qty=payload.get("qty", 1),
        result=payload,
        breakdown=breakdown,
        page_width=74,
        divider=_divider(74),
        process_meta=breakdown.get("process_meta"),
        process_meta_raw=breakdown.get("process_meta_raw"),
        hour_summary_entries={},
        cfg=None,
    )

    assert header_lines == expected_header
    assert suffix_lines == [""]
    assert state.pricing_source_value == expected_source
    assert breakdown.get("pricing_source") == expected_source
    assert state.summary_lines == header_lines + suffix_lines
    assert list(state.lines) == header_lines + suffix_lines
    assert isinstance(state.writer, QuoteWriter)


def test_render_summary_includes_drill_debug_section() -> None:
    entries: list[str] = [
        "Material Removal Debug – milling",
        "OK drill group",
    ]

    payload = {"qty": 1, "breakdown": {}, "drill_debug": entries}

    state = RenderState(
        payload,
        page_width=40,
        llm_debug_enabled=True,
        drill_debug_entries=entries,
        hour_summary_entries={},
        material_warning_summary=False,
        material_warning_label="⚠ MATERIALS MISSING",
    )

    header_lines, suffix_lines = render_summary(state)

    assert header_lines[0].startswith("QUOTE SUMMARY")
    assert suffix_lines[0] == ""
    assert suffix_lines[1] == "Drill Debug"
    assert suffix_lines[2] == _divider(40)
    assert suffix_lines[-1] == ""
    assert list(state.lines) == header_lines + suffix_lines
    assert isinstance(state.writer, QuoteWriter)


def test_render_summary_adds_material_warning_label() -> None:
    payload = {"qty": 1, "breakdown": {}}

    state = RenderState(
        payload,
        page_width=60,
        hour_summary_entries={},
        drill_debug_entries=[],
        material_warning_summary=True,
        material_warning_label="⚠ MATERIALS MISSING",
    )

    header_lines, suffix_lines = render_summary(state)

    assert header_lines[0].startswith("QUOTE SUMMARY")
    assert suffix_lines[0] == "⚠ MATERIALS MISSING"
    assert suffix_lines[1] == ""
    assert list(state.lines) == header_lines + suffix_lines
    assert isinstance(state.writer, QuoteWriter)
