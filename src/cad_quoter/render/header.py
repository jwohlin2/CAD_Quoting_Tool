"""Header rendering helpers for the quote renderer."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from cad_quoter.app.quote_doc import build_quote_header_lines

from .state import RenderState


def render_header(state: RenderState) -> list[str]:
    """Return the QUOTE SUMMARY header lines for ``state``.

    The helper mirrors the legacy behaviour from :func:`render_summary`, while
    ensuring the ``RenderState`` is updated with the resolved pricing source and
    associated metadata side effects.
    """

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

    apply_pricing_source(state, state.breakdown)
    apply_pricing_source(state, state.result)

    return header_lines


def apply_pricing_source(state: RenderState, payload: Mapping[str, Any] | None) -> None:
    """Propagate ``state.pricing_source_value`` into ``payload``.

    ``payload`` is expected to be a mutable mapping. The helper mirrors the
    mutations previously performed inline in :func:`render_summary`, keeping the
    breakdown metadata, ``app_meta`` planner hint, and any baseline state in
    sync with the resolved pricing source.
    """

    if not isinstance(payload, MutableMapping):
        return

    pricing_source_value = state.pricing_source_value
    normalized_value: str | None
    if pricing_source_value is None:
        normalized_value = None
    else:
        normalized_value = str(pricing_source_value).strip()
        if not normalized_value:
            normalized_value = None

    if payload is state.breakdown:
        if normalized_value is not None:
            payload["pricing_source"] = pricing_source_value
        else:
            payload.pop("pricing_source", None)

    if payload is state.result:
        app_meta_container = payload.setdefault("app_meta", {})
        if isinstance(app_meta_container, MutableMapping):
            normalized_lower = normalized_value.lower() if normalized_value else ""
            if normalized_lower == "planner":
                app_meta_container.setdefault("used_planner", True)

    baseline_container: MutableMapping[str, Any] | None = None

    if payload is state.result:
        decision_state = payload.get("decision_state")
        if isinstance(decision_state, MutableMapping):
            candidate = decision_state.get("baseline")
            if isinstance(candidate, MutableMapping):
                baseline_container = candidate

        if baseline_container is None:
            baseline_candidate = payload.get("baseline")
            if isinstance(baseline_candidate, MutableMapping):
                baseline_container = baseline_candidate
    elif payload is not state.breakdown:
        baseline_candidate = payload.get("baseline")
        if isinstance(baseline_candidate, MutableMapping):
            baseline_container = baseline_candidate

    if baseline_container is not None:
        if normalized_value is not None:
            baseline_container["pricing_source"] = pricing_source_value
        else:
            baseline_container.pop("pricing_source", None)

