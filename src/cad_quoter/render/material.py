"""Render the Material & Stock section using legacy-compatible formatting."""

from __future__ import annotations

import math
from typing import Any, Mapping, TYPE_CHECKING

from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import (
    format_currency,
    format_percent,
    format_weight_lb_oz,
)

from .state import _as_mapping

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from . import RenderState


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` as ``float`` when possible, otherwise ``None``."""

    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _material_name(material_map: Mapping[str, Any], selection_map: Mapping[str, Any]) -> str | None:
    """Return the display name for the selected material."""

    for key in ("material_name", "material"):
        text = material_map.get(key)
        if not text:
            continue
        clean = str(text).strip()
        if clean:
            return clean
    canonical = selection_map.get("canonical")
    if canonical:
        clean = str(canonical).strip()
        if clean:
            return clean
    return None


def _format_short_divider(page_width: int, width: int = 7) -> str:
    pad = max(0, page_width - max(width, 0))
    return " " * pad + "-" * max(width, 0)


def render_material(state: "RenderState") -> list[str]:
    """Emit the Material & Stock section and update the shared writer state."""

    writer_candidate = getattr(state, "writer", None)
    if isinstance(writer_candidate, QuoteWriter):
        writer = writer_candidate
    else:
        writer = QuoteWriter(
            divider=state.divider or "-" * max(1, state.page_width),
            page_width=state.page_width,
            currency=state.currency,
            recorder=state.recorder,
            lines=state.lines,
        )
        setattr(state, "writer", writer)

    state.lines = writer.lines
    state.recorder = writer.recorder

    start_index = len(writer.lines)
    divider = state.divider or "-" * max(1, state.page_width)

    material_map = _as_mapping(state.breakdown.get("material"))
    selection_map = _as_mapping(state.breakdown.get("material_selected"))
    material_block = _as_mapping(state.breakdown.get("material_block"))

    material_name = _material_name(material_map, selection_map)

    mass_g = _coerce_float(material_block.get("mass_g"))
    if mass_g is None:
        mass_g = _coerce_float(material_map.get("mass_g"))

    net_mass_g = _coerce_float(material_block.get("net_mass_g"))
    if net_mass_g is None:
        net_mass_g = _coerce_float(material_block.get("mass_g_net"))
    if net_mass_g is None:
        net_mass_g = _coerce_float(material_map.get("net_mass_g"))

    scrap_pct_value = _coerce_float(material_block.get("scrap_pct"))
    if scrap_pct_value is None:
        scrap_pct_value = _coerce_float(material_map.get("scrap_pct"))

    scrap_mass_g = _coerce_float(material_block.get("scrap_mass_g"))
    if scrap_mass_g is None and mass_g is not None and scrap_pct_value is not None:
        scrap_mass_g = max(0.0, mass_g * scrap_pct_value)
    if scrap_mass_g is None and mass_g is not None and net_mass_g is not None:
        scrap_mass_g = max(0.0, mass_g - net_mass_g)

    unit_price = _coerce_float(material_map.get("unit_price_usd_per_lb"))
    if unit_price is None:
        unit_price = _coerce_float(material_block.get("unit_price_usd_per_lb"))
    if unit_price is None:
        unit_price = 0.0

    unit_price_text = format_currency(unit_price, state.currency)
    base_label = f"Base Material @ per-lb @ {unit_price_text}/lb"

    material_total = _coerce_float(material_map.get("material_cost"))
    if material_total is None:
        material_total = _coerce_float(material_block.get("total_material_cost"))
    if material_total is None:
        material_total = 0.0

    has_weight_info = any(
        [
            material_name,
            mass_g and mass_g > 0,
            net_mass_g and net_mass_g > 0,
            scrap_mass_g and scrap_mass_g > 0,
        ]
    )
    has_cost_info = material_total > 0 or state.show_zeros
    if not (material_name or has_weight_info or has_cost_info):
        return []

    writer.line("Material & Stock")
    writer.line(divider)

    if material_name:
        writer.line(f"  Material used:  {material_name}")

    if mass_g and mass_g > 0:
        writer.line(f"  Starting Weight: {format_weight_lb_oz(mass_g)}")
    elif state.show_zeros:
        writer.line("  Starting Weight: 0 oz")

    if net_mass_g and net_mass_g > 0:
        writer.line(f"  Net Weight: {format_weight_lb_oz(net_mass_g)}")
    elif state.show_zeros:
        writer.line("  Net Weight: 0 oz")

    if scrap_mass_g is not None and (scrap_mass_g > 0 or state.show_zeros):
        writer.line(f"  Scrap Weight: {format_weight_lb_oz(scrap_mass_g)}")

    scrap_fraction = None
    if mass_g and mass_g > 0 and scrap_mass_g is not None:
        try:
            scrap_fraction = max(0.0, float(scrap_mass_g)) / float(mass_g)
        except Exception:
            scrap_fraction = None

    if scrap_fraction is not None:
        writer.line(
            f"  Scrap Percentage: {format_percent(scrap_fraction)} (computed)"
        )
    elif scrap_pct_value is not None and scrap_pct_value > 0:
        writer.line(f"  Scrap Percentage: {format_percent(scrap_pct_value)}")

    scrap_hint_pct = _coerce_float(material_block.get("scrap_pct_hint"))
    if scrap_hint_pct is None:
        scrap_hint_pct = scrap_pct_value
    if scrap_hint_pct is not None and scrap_hint_pct > 0:
        writer.line(f"  Scrap % (geometry hint): {format_percent(scrap_hint_pct)}")

    if has_cost_info:
        writer.line(state.format_row(base_label, material_total, indent="  "))
        writer.line(_format_short_divider(state.page_width))
        writer.line(state.format_row("Total Material Cost :", material_total, indent="  "))
    writer.line("")

    state.material_component_total = material_total
    state.material_component_net = material_total if net_mass_g else None
    state.material_net_cost = material_total

    return list(writer.lines[start_index:])
