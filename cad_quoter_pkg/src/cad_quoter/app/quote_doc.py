"""Quote document rendering helpers shared across UI entrypoints."""

from __future__ import annotations

import re
import textwrap
import unicodedata
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.pricing.materials import LB_PER_KG, _compute_scrap_mass_g
from cad_quoter.utils.rendering import (
    fmt_money,
    format_currency,
    format_percent,
    format_weight_lb_decimal,
    format_weight_lb_oz,
)
from cad_quoter.utils.scrap import normalize_scrap_pct

from cad_quoter.ui.services import QuoteConfiguration

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RENDER_ASCII_REPLACEMENTS: dict[str, str] = {
    "—": "-",
    "•": "-",
    "…": "...",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "µ": "u",
    "μ": "u",
    "±": "+/-",
    "°": " deg ",
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    " ": " ",  # non-breaking space
    "⚠️": "⚠",
}

_RENDER_PASSTHROUGH: dict[str, str] = {
    "–": "__EN_DASH__",
    "×": "__MULTIPLY__",
    "≥": "__GEQ__",
    "≤": "__LEQ__",
    "⚠": "__WARN__",
}


def _sanitize_render_text(value: Any) -> str:
    """Return a sanitized ASCII-only string for quote document output."""

    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    for source, placeholder in _RENDER_PASSTHROUGH.items():
        if source in text:
            text = text.replace(source, placeholder)
    text = text.replace("\t", " ")
    text = text.replace("\r", "")
    text = _ANSI_ESCAPE_RE.sub("", text)
    for source, replacement in _RENDER_ASCII_REPLACEMENTS.items():
        if source in text:
            text = text.replace(source, replacement)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    text = _CONTROL_CHAR_RE.sub("", text)
    for source, placeholder in _RENDER_PASSTHROUGH.items():
        if placeholder in text:
            text = text.replace(placeholder, source)
    return text


def _wrap_header_text(text: Any, page_width: int, indent: str = "") -> list[str]:
    """Helper mirroring :func:`write_wrapped` for header content."""

    if text is None:
        return []
    txt = str(text).strip()
    if not txt:
        return []
    wrapper = textwrap.TextWrapper(width=max(10, page_width - len(indent)))
    return [f"{indent}{chunk}" for chunk in wrapper.wrap(txt)]


def wrap_header_text(text: Any, page_width: int, indent: str = "") -> list[str]:
    """Public wrapper for :func:`_wrap_header_text`."""

    return _wrap_header_text(text, page_width, indent)


def _resolve_pricing_source_value(
    base_value: Any,
    *,
    used_planner: bool | None = None,
    process_meta: Mapping[str, Any] | None = None,
    process_meta_raw: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
    planner_process_minutes: Any = None,
    hour_summary_entries: Mapping[str, Any] | None = None,
    additional_sources: Sequence[Any] | None = None,
    cfg: QuoteConfiguration | None = None,
) -> str | None:
    """Return a normalized pricing source, honoring explicit selections."""

    fallback_text: str | None = None
    if base_value is not None:
        candidate_text = str(base_value).strip()
        if candidate_text:
            lowered = candidate_text.lower()
            if lowered == "planner":
                return "planner"
            if lowered not in {"legacy", "auto", "default", "fallback"}:
                return candidate_text
            fallback_text = candidate_text

    if used_planner:
        if fallback_text:
            return fallback_text
        return "planner"

    # Delegate planner signal detection to the consolidated planner helpers
    from cad_quoter.app.planner_support import (
        _planner_signals_present as _planner_signals_present_helper,
    )

    if _planner_signals_present_helper(
        process_meta=process_meta,
        process_meta_raw=process_meta_raw,
        breakdown=breakdown,
        planner_process_minutes=planner_process_minutes,
        hour_summary_entries=hour_summary_entries,
        additional_sources=list(additional_sources) if additional_sources is not None else None,
    ):
        if fallback_text:
            return fallback_text
        return "planner"

    if fallback_text:
        return fallback_text

    return None


def _build_quote_header_lines(
    *,
    qty: int,
    result: Mapping[str, Any] | None,
    breakdown: Mapping[str, Any] | None,
    page_width: int,
    divider: str,
    process_meta: Mapping[str, Any] | None,
    process_meta_raw: Mapping[str, Any] | None,
    hour_summary_entries: Mapping[str, Any] | None,
    cfg: QuoteConfiguration | None = None,
) -> tuple[list[str], str | None]:
    """Construct the canonical QUOTE SUMMARY header lines."""

    header_lines: list[str] = [f"QUOTE SUMMARY - Qty {qty}", divider]
    header_lines.append("Quote Summary (structured data attached below)")

    speeds_feeds_value = None
    if isinstance(result, Mapping):
        speeds_feeds_value = result.get("speeds_feeds_path")
    if speeds_feeds_value in (None, "") and isinstance(breakdown, Mapping):
        speeds_feeds_value = breakdown.get("speeds_feeds_path")
    path_text = str(speeds_feeds_value).strip() if speeds_feeds_value else ""

    speeds_feeds_loaded_display: bool | None = None
    for source in (result, breakdown):
        if not isinstance(source, Mapping):
            continue
        if "speeds_feeds_loaded" in source:
            raw_flag = source.get("speeds_feeds_loaded")
            speeds_feeds_loaded_display = None if raw_flag is None else bool(raw_flag)
            break

    if speeds_feeds_loaded_display is True:
        status_suffix = " (loaded)"
    elif speeds_feeds_loaded_display is False:
        status_suffix = " (not loaded)"
    else:
        status_suffix = ""

    if path_text:
        header_lines.extend(
            _wrap_header_text(
                f"Speeds/Feeds CSV: {path_text}{status_suffix}",
                page_width,
            )
        )
    elif status_suffix:
        header_lines.extend(
            _wrap_header_text(
                f"Speeds/Feeds CSV: (not set){status_suffix}",
                page_width,
            )
        )
    else:
        header_lines.append("Speeds/Feeds CSV: (not set)")

    def _coerce_pricing_source(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        # normalize synonyms
        if lowered in {"legacy", "est", "estimate", "estimator"}:
            return "estimator"
        if lowered in {"plan", "planner"}:
            return "planner"
        return text

    raw_pricing_source = None
    pricing_source_display = None
    if isinstance(breakdown, Mapping):
        raw_pricing_source = _coerce_pricing_source(breakdown.get("pricing_source"))
        if raw_pricing_source:
            pricing_source_display = str(raw_pricing_source).title()

    used_planner_flag: bool | None = None
    for source in (result, breakdown):
        if not isinstance(source, Mapping):
            continue
        for meta_key in ("app_meta", "app"):
            candidate = source.get(meta_key)
            if not isinstance(candidate, Mapping):
                continue
            if "used_planner" in candidate:
                try:
                    used_planner_flag = bool(candidate.get("used_planner"))
                except Exception:
                    used_planner_flag = True if candidate.get("used_planner") else False
                break
        if used_planner_flag is not None:
            break

    pricing_source_value = _resolve_pricing_source_value(
        raw_pricing_source,
        used_planner=used_planner_flag,
        process_meta=process_meta if isinstance(process_meta, Mapping) else None,
        process_meta_raw=process_meta_raw if isinstance(process_meta_raw, Mapping) else None,
        breakdown=breakdown if isinstance(breakdown, Mapping) else None,
        hour_summary_entries=hour_summary_entries,
        cfg=cfg,
    )

    # === HEADER: PRICING SOURCE OVERRIDE ===
    if getattr(cfg, "prefer_removal_drilling_hours", False):
        normalized_value = (
            str(pricing_source_value).strip().lower()
            if pricing_source_value is not None
            else ""
        )
        if not normalized_value or normalized_value == "legacy":
            pricing_source_value = "Estimator"
            pricing_source_display = "Estimator"

    normalized_pricing_source: str | None = None
    if pricing_source_value is not None:
        normalized_pricing_source = str(pricing_source_value).strip()
        if not normalized_pricing_source:
            normalized_pricing_source = None

    if normalized_pricing_source:
        normalized_pricing_source_lower = normalized_pricing_source.lower()
        raw_pricing_source_lower = (
            str(raw_pricing_source).strip().lower() if raw_pricing_source is not None else None
        )

        if (
            isinstance(breakdown, MutableMapping)
            and raw_pricing_source_lower != normalized_pricing_source_lower
        ):
            breakdown["pricing_source"] = pricing_source_value

        pricing_source_display = normalized_pricing_source.title()

    if pricing_source_display:
        display_value = pricing_source_display
        header_lines.append(f"Pricing Source: {display_value}")

    return header_lines, pricing_source_value


def build_quote_header_lines(
    *,
    qty: int,
    result: Mapping[str, Any] | None,
    breakdown: Mapping[str, Any] | None,
    page_width: int,
    divider: str,
    process_meta: Mapping[str, Any] | None,
    process_meta_raw: Mapping[str, Any] | None,
    hour_summary_entries: Mapping[str, Any] | None,
    cfg: QuoteConfiguration | None = None,
) -> tuple[list[str], str | None]:
    """Public wrapper for :func:`_build_quote_header_lines`."""

    return _build_quote_header_lines(
        qty=qty,
        result=result,
        breakdown=breakdown,
        page_width=page_width,
        divider=divider,
        process_meta=process_meta,
        process_meta_raw=process_meta_raw,
        hour_summary_entries=hour_summary_entries,
        cfg=cfg,
    )


def _scrap_source_hint(
    material_info: Mapping[str, Any] | None,
    scrap_ctx: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a human-readable description of the scrap source."""

    if not isinstance(material_info, Mapping):
        material_info = {}
    if not isinstance(scrap_ctx, Mapping):
        scrap_ctx = {}

    scrap_from_holes_raw = material_info.get("scrap_pct_from_holes")
    scrap_from_holes_val = _coerce_float_or_none(scrap_from_holes_raw)
    scrap_from_holes = False
    if scrap_from_holes_val is not None and scrap_from_holes_val > 1e-6:
        scrap_from_holes = True
    elif isinstance(scrap_from_holes_raw, bool):
        scrap_from_holes = scrap_from_holes_raw
    if not scrap_from_holes:
        scrap_from_holes = bool(scrap_ctx.get("scrap_from_holes"))
    if scrap_from_holes:
        return "holes"

    label_raw = scrap_ctx.get("scrap_source_label")
    if label_raw in (None, ""):
        label_raw = material_info.get("scrap_source_label")
    if label_raw in (None, ""):
        return None

    label_text = str(label_raw).strip()
    if not label_text:
        return None

    label_text = label_text.replace("+", " + ")
    label_text = label_text.replace("_", " ")
    label_text = " ".join(label_text.split())
    return label_text or None


def _coerce_bool_flag(value: Any) -> bool:
    """Return True only for explicitly truthy values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"", "0", "false", "f", "no", "n", "off"}:
            return False
        return False
    return False


def build_material_detail_lines(
    material_block: Mapping[str, Any] | MutableMapping[str, Any] | None,
    *,
    scrap_context: Mapping[str, Any] | None,
    currency: str,
    show_zeros: bool,
    show_material_shipping: bool,
    shipping_total: float,
    shipping_source: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Return material detail lines and updates for the structured breakdown."""

    material_map = material_block if isinstance(material_block, Mapping) else {}
    scrap_ctx = scrap_context if isinstance(scrap_context, Mapping) else {}

    detail_lines: list[str] = []
    updates: dict[str, Any] = {}

    mass_g = material_map.get("mass_g")
    net_mass_g_raw = material_map.get("mass_g_net")
    if _coerce_float_or_none(net_mass_g_raw) is None:
        net_mass_g_raw = material_map.get("net_mass_g")

    effective_mass_val = _coerce_float_or_none(material_map.get("effective_mass_g"))
    if effective_mass_val is None:
        effective_mass_val = _coerce_float_or_none(mass_g)

    starting_mass_val = _coerce_float_or_none(mass_g)
    if starting_mass_val is None:
        starting_mass_val = _coerce_float_or_none(material_map.get("effective_mass_g"))
    if starting_mass_val is None:
        starting_mass_val = _coerce_float_or_none(scrap_ctx.get("starting_mass_g"))

    removal_mass_val = None
    for key in ("material_removed_mass_g", "material_removed_mass_g_est"):
        removal_mass_val = _coerce_float_or_none(scrap_ctx.get(key))
        if removal_mass_val is not None:
            break
        removal_mass_val = _coerce_float_or_none(material_map.get(key))
        if removal_mass_val is not None:
            break

    net_mass_val = _coerce_float_or_none(net_mass_g_raw)
    scrap_pct_raw = scrap_ctx.get("scrap_pct") if scrap_ctx else None
    if scrap_pct_raw is None:
        scrap_pct_raw = material_map.get("scrap_pct")

    prefer_pct = bool(scrap_ctx.get("prefer_pct_for_scrap")) if scrap_ctx else False
    if not prefer_pct:
        prefer_pct = effective_mass_val is not None

    if (
        net_mass_val is None
        and removal_mass_val is not None
        and removal_mass_val >= 0
    ):
        base_for_net = starting_mass_val
        if base_for_net is None:
            base_for_net = effective_mass_val
        if base_for_net is not None:
            net_mass_val = max(0.0, float(base_for_net) - float(removal_mass_val))
    if net_mass_val is None:
        net_mass_val = effective_mass_val

    scrap_mass_val = _compute_scrap_mass_g(
        removal_mass_g_est=removal_mass_val,
        scrap_pct_raw=scrap_pct_raw,
        effective_mass_g=effective_mass_val,
        net_mass_g=net_mass_val,
        prefer_pct=prefer_pct,
    )

    if scrap_mass_val is not None:
        updates["scrap_credit_mass_lb"] = float(scrap_mass_val) / 1000.0 * LB_PER_KG
    else:
        updates["scrap_credit_mass_lb"] = None

    weight_lines: list[str] = []
    if (starting_mass_val and starting_mass_val > 0) or show_zeros:
        weight_lines.append(
            f"  Starting Weight: {format_weight_lb_oz(starting_mass_val)}"
        )
    if (net_mass_val and net_mass_val > 0) or show_zeros:
        weight_lines.append(
            f"  Net Weight: {format_weight_lb_oz(net_mass_val)}"
        )
    if scrap_mass_val is not None:
        if scrap_mass_val > 0 or show_zeros:
            weight_lines.append(
                f"  Scrap Weight: {format_weight_lb_oz(scrap_mass_val)}"
            )
    elif show_zeros:
        weight_lines.append("  Scrap Weight: 0 oz")

    computed_scrap_fraction: float | None = None
    if (
        starting_mass_val is not None
        and starting_mass_val > 0
        and scrap_mass_val is not None
    ):
        try:
            start_mass = float(starting_mass_val)
            scrap_mass = max(0.0, float(scrap_mass_val))
        except Exception:
            start_mass = 0.0
            scrap_mass = 0.0
        if start_mass > 0:
            computed_scrap_fraction = scrap_mass / start_mass

    scrap_hint_text = scrap_ctx.get("scrap_source_hint") if scrap_ctx else None
    if scrap_hint_text is None:
        scrap_hint_text = _scrap_source_hint(material_map, scrap_ctx)

    scrap_fraction_hint = scrap_ctx.get("geometry_hint_fraction") if scrap_ctx else None
    if scrap_fraction_hint is None:
        scrap_fraction_hint = normalize_scrap_pct(scrap_pct_raw)
        if scrap_fraction_hint is not None and scrap_fraction_hint <= 0:
            scrap_fraction_hint = None

    if computed_scrap_fraction is not None:
        scrap_line = f"  Scrap Percentage: {format_percent(computed_scrap_fraction)} (computed)"
        if scrap_hint_text and scrap_fraction_hint is None:
            scrap_line += f" ({scrap_hint_text})"
        weight_lines.append(scrap_line)
    elif scrap_pct_raw is not None:
        scrap_line = f"  Scrap Percentage: {format_percent(scrap_pct_raw)}"
        if scrap_hint_text:
            scrap_line += f" ({scrap_hint_text})"
        weight_lines.append(scrap_line)

    if scrap_fraction_hint is not None:
        geometry_line = f"  Scrap % (geometry hint): {format_percent(scrap_fraction_hint)}"
        if scrap_hint_text:
            geometry_line += f" ({scrap_hint_text})"
        weight_lines.append(geometry_line)

    detail_lines.extend(weight_lines)

    scrap_credit_lines: list[str] = []
    scrap_credit_entered = _coerce_bool_flag(
        scrap_ctx.get("scrap_credit_entered") if scrap_ctx else None
    )
    if not scrap_credit_entered:
        scrap_credit_entered = _coerce_bool_flag(
            material_map.get("material_scrap_credit_entered")
        )
    scrap_credit_val = _coerce_float_or_none(
        scrap_ctx.get("scrap_credit") if scrap_ctx else None
    )
    if scrap_credit_val is None:
        scrap_credit_val = _coerce_float_or_none(material_map.get("material_scrap_credit"))
    scrap_credit = float(scrap_credit_val or 0.0)
    if scrap_credit_entered and scrap_credit:
        credit_display = format_currency(scrap_credit, currency)
        if credit_display.startswith(currency):
            credit_display = f"-{credit_display}"
        else:
            credit_display = f"-{fmt_money(scrap_credit, currency)}"
        scrap_credit_lines.append(f"  Scrap Credit: {credit_display}")
        scrap_credit_unit_price_lb = _coerce_float_or_none(
            scrap_ctx.get("scrap_credit_unit_price_usd_per_lb") if scrap_ctx else None
        )
        if scrap_credit_unit_price_lb is None:
            scrap_credit_unit_price_lb = _coerce_float_or_none(
                material_map.get("scrap_credit_unit_price_usd_per_lb")
            )
        if (
            scrap_mass_val is not None
            and scrap_credit_unit_price_lb is not None
        ):
            scrap_credit_lines.append(
                "    based on "
                f"{format_weight_lb_oz(scrap_mass_val)} × {fmt_money(scrap_credit_unit_price_lb, currency)} / lb"
            )
    if scrap_credit_lines:
        detail_lines.extend(scrap_credit_lines)

    base_cost_before_scrap = _coerce_float_or_none(
        material_map.get("material_cost_before_credit")
    )
    if base_cost_before_scrap is None:
        net_mass_for_base = _coerce_float_or_none(net_mass_val)
        if net_mass_for_base is None:
            net_mass_for_base = _coerce_float_or_none(material_map.get("net_mass_g"))
        if net_mass_for_base is not None and net_mass_for_base > 0:
            per_lb_value = _coerce_float_or_none(material_map.get("unit_price_usd_per_lb"))
            if per_lb_value is None:
                per_g_value = _coerce_float_or_none(material_map.get("unit_price_per_g"))
                if per_g_value is not None:
                    per_lb_value = per_g_value * (1000.0 / LB_PER_KG)
            if per_lb_value is not None:
                base_cost_before_scrap = (
                    float(net_mass_for_base) / 1000.0 * LB_PER_KG
                ) * float(per_lb_value)

    shipping_tax_lines: list[str] = []
    if base_cost_before_scrap is not None or show_zeros:
        base_val = float(base_cost_before_scrap or 0.0)
        if show_material_shipping and (shipping_total > 0 or show_zeros):
            if shipping_source:
                shipping_display = shipping_total
            else:
                shipping_display = base_val * 0.15
            shipping_tax_lines.append(
                f"  Shipping: {format_currency(shipping_display, currency)}"
            )
        tax_cost = base_val * 0.065
        if tax_cost > 0 or show_zeros:
            shipping_tax_lines.append(
                f"  Material Tax: {format_currency(tax_cost, currency)}"
            )

    if shipping_tax_lines:
        detail_lines.extend(shipping_tax_lines)

    unit_price_per_g = material_map.get("unit_price_per_g")
    unit_price_per_kg = material_map.get("unit_price_usd_per_kg")
    unit_price_per_lb = material_map.get("unit_price_usd_per_lb")
    price_asof = scrap_ctx.get("price_asof") if scrap_ctx else None
    if price_asof is None:
        price_asof = material_map.get("unit_price_asof")
    price_source = scrap_ctx.get("price_source") if scrap_ctx else None
    if price_source is None:
        price_source = material_map.get("unit_price_source") or material_map.get("source")
    supplier_min = scrap_ctx.get("supplier_min_charge") if scrap_ctx else None
    if supplier_min is None:
        supplier_min = material_map.get("supplier_min_charge")
    minchg = supplier_min

    price_lines: list[str] = []
    if unit_price_per_g or unit_price_per_kg or unit_price_per_lb or show_zeros:
        grams_per_lb = 1000.0 / LB_PER_KG
        per_lb_value = _coerce_float_or_none(unit_price_per_lb)
        if per_lb_value is None:
            per_kg_value = _coerce_float_or_none(unit_price_per_kg)
            if per_kg_value is not None:
                per_lb_value = per_kg_value / LB_PER_KG
        if per_lb_value is None:
            per_g_value = _coerce_float_or_none(unit_price_per_g)
            if per_g_value is not None:
                per_lb_value = per_g_value * grams_per_lb
        if per_lb_value is None and show_zeros:
            per_lb_value = 0.0

        if per_lb_value is not None:
            display_line = f"{fmt_money(per_lb_value, currency)} / lb"
            extras: list[str] = []
            if price_asof:
                extras.append(f"as of {price_asof}")
            extra = f" ({', '.join(extras)})" if extras else ""
            price_lines.append(f"  Material Price: {display_line}{extra}")

    if price_source:
        price_lines.append(f"  Source: {price_source}")
    if minchg or show_zeros:
        price_lines.append(
            f"  Supplier Min Charge: {format_currency(minchg or 0, currency)}"
        )

    if price_lines:
        last_line = detail_lines[-1] if detail_lines else ""
        if (
            detail_lines
            and last_line != ""
            and not last_line.lstrip().startswith("Scrap Credit:")
            and not last_line.lstrip().startswith("based on ")
        ):
            detail_lines.append("")
        detail_lines.extend(price_lines)

    return updates, detail_lines


__all__ = [
    "_sanitize_render_text",
    "_wrap_header_text",
    "wrap_header_text",
    "_resolve_pricing_source_value",
    "_build_quote_header_lines",
    "build_quote_header_lines",
    "build_material_detail_lines",
]

