from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

import pytest

import appV5

from cad_quoter.render.material import render_material as modern_render_material

from tests.render.fixtures import legacy_state_factory, material_payload, modern_state_factory

Mutation = Callable[[dict[str, Any]], None]


def _clone(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        if isinstance(value, dict):
            return {key: _clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_clone(item) for item in value)
        return value


def _format_lines(state: Any, lines: list[Any]) -> list[str]:
    if not lines:
        return []
    formatter = getattr(state, "format_rows", None)
    if callable(formatter) and all(hasattr(row, "label") and hasattr(row, "value") for row in lines):
        try:
            return formatter(lines)
        except Exception:
            pass
    formatted: list[str] = []
    for line in lines:
        if line is None:
            formatted.append("")
        else:
            formatted.append(str(line))
    return formatted


def _collect_material_inputs(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
    captured: dict[str, Any] = {}
    original = appV5.render_material

    def capture(state: Any, **kwargs: Any) -> list[str]:
        if "kwargs" not in captured:
            captured["kwargs"] = _clone(kwargs)
            captured["config"] = {
                "currency": state.currency,
                "show_zeros": bool(getattr(state, "show_zeros", False)),
                "page_width": int(getattr(state, "page_width", 74)),
                "divider": getattr(state, "divider", "-" * int(getattr(state, "page_width", 74))),
            }
            captured["material_warning_needed"] = bool(getattr(state, "material_warning_needed", False))
        return original(state, **kwargs)

    try:
        if not hasattr(appV5, "apply_render_overrides"):
            from cad_quoter.render.config import apply_render_overrides as _apply_render_overrides

            appV5.apply_render_overrides = _apply_render_overrides  # type: ignore[attr-defined]
        if not hasattr(appV5, "ensure_mutable_breakdown"):
            from cad_quoter.render.config import ensure_mutable_breakdown as _ensure_mutable_breakdown

            appV5.ensure_mutable_breakdown = _ensure_mutable_breakdown  # type: ignore[attr-defined]
        if not hasattr(appV5, "QuoteWriter"):
            from cad_quoter.render.writer import QuoteWriter as _QuoteWriter

            appV5.QuoteWriter = _QuoteWriter  # type: ignore[attr-defined]
        if not hasattr(appV5, "QuoteDocRecorder"):
            from cad_quoter.utils.render_utils import QuoteDocRecorder as _QuoteDocRecorder

            appV5.QuoteDocRecorder = _QuoteDocRecorder  # type: ignore[attr-defined]
        if not hasattr(appV5, "detect_planner_drilling"):
            from cad_quoter.render import detect_planner_drilling as _detect_planner_drilling

            appV5.detect_planner_drilling = _detect_planner_drilling  # type: ignore[attr-defined]
        if not hasattr(appV5, "render_state_has_planner_drilling"):
            from cad_quoter.render import has_planner_drilling as _has_planner_drilling

            appV5.render_state_has_planner_drilling = _has_planner_drilling  # type: ignore[attr-defined]
        if not hasattr(appV5, "render_drilling_guard"):
            from cad_quoter.render.guards import render_drilling_guard as _render_drilling_guard

            appV5.render_drilling_guard = _render_drilling_guard  # type: ignore[attr-defined]
        appV5.render_material = capture
        appV5.render_quote(copy.deepcopy(payload), currency="$", show_zeros=False)
    finally:
        appV5.render_material = original

    kwargs_copy = captured.get("kwargs", {})
    config = captured.get(
        "config",
        {"currency": "$", "show_zeros": False, "page_width": 74, "divider": "-" * 74},
    )
    material_warning_needed = captured.get("material_warning_needed", False)
    return kwargs_copy, config, material_warning_needed


def _mutate_noop(payload: dict[str, Any]) -> None:
    return None


def _mutate_empty_material(payload: dict[str, Any]) -> None:
    breakdown = payload.setdefault("breakdown", {})
    if isinstance(breakdown, dict):
        breakdown["material"] = {}
        breakdown["material_block"] = {}
        breakdown["material_selected"] = {}
        breakdown.pop("materials", None)


def _mutate_remove_shipping(payload: dict[str, Any]) -> None:
    breakdown = payload.setdefault("breakdown", {})
    pass_through = breakdown.get("pass_through")
    if isinstance(pass_through, dict):
        pass_through.pop("Shipping", None)
    material_block = breakdown.get("material_block")
    if isinstance(material_block, dict):
        material_block.pop("shipping", None)


def _mutate_drop_cost_components(payload: dict[str, Any]) -> None:
    breakdown = payload.setdefault("breakdown", {})
    for key in ("material", "material_block", "material_selected"):
        container = breakdown.get(key)
        if isinstance(container, dict):
            container.pop("material_cost_components", None)


@pytest.mark.parametrize(
    "mutator",
    [
        pytest.param(_mutate_noop, id="baseline"),
        pytest.param(_mutate_empty_material, id="empty-material"),
        pytest.param(_mutate_remove_shipping, id="missing-shipping"),
        pytest.param(_mutate_drop_cost_components, id="no-cost-components"),
    ],
)
def test_render_material_matches_legacy(
    mutator: Mutation,
    material_payload: dict[str, Any],
    legacy_state_factory: Callable[..., Any],
    modern_state_factory: Callable[..., Any],
) -> None:
    base_payload = copy.deepcopy(material_payload)
    mutator(base_payload)

    kwargs, config, warning_flag = _collect_material_inputs(base_payload)

    legacy_payload = copy.deepcopy(base_payload)
    modern_payload = copy.deepcopy(base_payload)

    legacy_state = legacy_state_factory(
        legacy_payload,
        currency=config["currency"],
        show_zeros=config["show_zeros"],
        page_width=config["page_width"],
        divider=config["divider"],
        geometry=kwargs.get("g"),
    )
    modern_state = modern_state_factory(
        modern_payload,
        currency=config["currency"],
        show_zeros=config["show_zeros"],
        page_width=config["page_width"],
        divider=config["divider"],
    )

    legacy_state.material_warning_needed = warning_flag
    modern_state.material_warning_needed = warning_flag

    legacy_kwargs = copy.deepcopy(kwargs)
    modern_kwargs = copy.deepcopy(kwargs)

    legacy_lines = appV5.render_material(legacy_state, **legacy_kwargs)
    modern_lines = modern_render_material(modern_state, **modern_kwargs)

    expected_lines = _format_lines(legacy_state, legacy_lines)
    observed_lines = _format_lines(modern_state, modern_lines)

    assert observed_lines == expected_lines

    float_attrs = (
        "material_component_total",
        "material_component_net",
        "material_net_cost",
    )
    for attr in float_attrs:
        legacy_value = getattr(legacy_state, attr, None)
        modern_value = getattr(modern_state, attr, None)
        if legacy_value is None or modern_value is None:
            assert legacy_value == modern_value
        else:
            assert modern_value == pytest.approx(float(legacy_value))

    legacy_components = _clone(getattr(legacy_state, "material_cost_components", None))
    modern_components = _clone(getattr(modern_state, "material_cost_components", None))
    assert modern_components == legacy_components

    assert bool(getattr(modern_state, "material_warning_needed", False)) == bool(
        getattr(legacy_state, "material_warning_needed", False)
    )
