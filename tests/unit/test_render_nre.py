from __future__ import annotations

import copy

import appV5

from cad_quoter.render import RenderState, render_nre

from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload


def _extract_legacy_nre(rendered: str) -> tuple[list[str], list[str]]:
    lines = rendered.splitlines()
    header = "NRE / Setup Costs (per lot)"
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start = idx
            break
    if start is None:
        return ([], [])
    body: list[str] = []
    for line in lines[start + 2 :]:
        stripped = line.strip()
        if not stripped and body:
            break
        if stripped.startswith("Pass-Through"):
            break
        if stripped:
            body.append(line.rstrip())
    rows: list[str] = []
    details: list[str] = []
    for line in body:
        if line.startswith("  Programming Hrs:") or line.startswith("    "):
            details.append(line)
        else:
            rows.append(line)
    return rows, details


def test_render_nre_matches_legacy_lines() -> None:
    payload = _dummy_quote_payload()

    legacy_text = appV5.render_quote(copy.deepcopy(payload), currency="$", show_zeros=False)
    legacy_rows, legacy_details = _extract_legacy_nre(legacy_text)

    state = RenderState(copy.deepcopy(payload), currency="$", show_zeros=False)
    rows, details = render_nre(state)

    formatted_rows = state.format_rows(rows)

    assert formatted_rows == legacy_rows[: len(formatted_rows)]
    assert details == legacy_details[: len(details)]
