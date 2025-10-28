from __future__ import annotations

import copy

from typing import Any, Mapping

import pytest

from cad_quoter.render import RenderState as ModernRenderState
from cad_quoter.utils.render_state import RenderState as LegacyRenderState

from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload


@pytest.fixture
def material_payload() -> dict:
    """Return a representative quote payload used across render tests."""

    return _dummy_quote_payload()


@pytest.fixture
def legacy_state_factory():
    """Factory that builds a legacy :mod:`appV5` render state."""

    def factory(
        payload: Mapping[str, Any],
        *,
        currency: str = "$",
        show_zeros: bool = False,
        page_width: int = 74,
        divider: str | None = None,
        geometry: Mapping[str, Any] | None = None,
    ) -> LegacyRenderState:
        result = copy.deepcopy(payload)
        if not isinstance(result, dict):
            result = dict(result)
        breakdown = result.setdefault("breakdown", {})
        if not isinstance(breakdown, dict):
            breakdown = dict(breakdown)
            result["breakdown"] = breakdown

        working_divider = divider or ("-" * page_width)
        lines: list[str] = []

        def append_line(value: Any) -> None:
            if value is None:
                lines.append("")
            else:
                lines.append(str(value))

        def append_lines(values: Any) -> None:
            if values is None:
                return
            for value in values:
                append_line(value)

        def write_wrapped(text: str, indent: str = "") -> None:
            if not text:
                return
            for line in str(text).splitlines():
                append_line(f"{indent}{line}".rstrip())

        state = LegacyRenderState(
            result=result,
            breakdown=breakdown,
            currency=currency,
            show_zeros=show_zeros,
            page_width=page_width,
            divider=working_divider,
            geometry=geometry or result.get("geom") or result.get("geo"),
            lines=lines,
            append_line=append_line,
            append_lines=append_lines,
            write_wrapped=write_wrapped,
        )
        return state

    return factory


@pytest.fixture
def modern_state_factory():
    """Factory that builds the modern render :class:`RenderState`."""

    def factory(
        payload: Mapping[str, Any],
        *,
        currency: str = "$",
        show_zeros: bool = False,
        page_width: int = 74,
        divider: str | None = None,
    ) -> ModernRenderState:
        working_divider = divider or ("-" * page_width)
        payload_copy = copy.deepcopy(payload)
        if not isinstance(payload_copy, dict):
            payload_copy = dict(payload_copy)

        return ModernRenderState(
            payload_copy,
            currency=currency,
            show_zeros=show_zeros,
            page_width=page_width,
            divider=working_divider,
        )

    return factory
