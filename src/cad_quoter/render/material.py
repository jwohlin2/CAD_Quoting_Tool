"""Material & Stock section renderer mirroring the legacy implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, TYPE_CHECKING

from cad_quoter.app.quote_doc import build_material_detail_lines
from cad_quoter.domain_models import (
    MATERIAL_DISPLAY_BY_KEY,
    coerce_float_or_none as _coerce_float_or_none,
    normalize_material_key as _normalize_lookup_key,
)
from cad_quoter.pricing.mcmaster_helpers import (
    resolve_mcmaster_plate_for_quote as _resolve_mcmaster_plate_for_quote,
)
from cad_quoter.pricing.materials import (
    _material_cost_components,
    infer_plate_lw_in,
)
from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import format_currency
from cad_quoter.utils.numeric import coerce_positive_float

from .state import _as_mapping

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from . import RenderState


def _is_truthy_flag(value: Any) -> bool:
    """Return ``True`` for explicit truthy values mirroring :mod:`appV5`."""

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


def _ensure_mapping(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, MutableMapping):
        return candidate  # type: ignore[return-value]
    if isinstance(candidate, Mapping):
        try:
            return dict(candidate)
        except Exception:
            return {}
    if candidate in (None, ""):
        return {}
    try:
        return dict(candidate)  # type: ignore[arg-type]
    except Exception:
        return {}


def render_material(state: "RenderState") -> list[str]:
    """Return the legacy material section lines while updating ``state``."""

    divider = state.divider or "-" * max(10, state.page_width)

    writer_candidate = getattr(state, "writer", None)
    if isinstance(writer_candidate, QuoteWriter):
        writer = writer_candidate
        writer.divider = divider
        writer.page_width = max(10, int(state.page_width or 0))
        writer.currency = state.currency
        writer.recorder = state.recorder
    else:
        writer = QuoteWriter(
            divider=divider,
            page_width=state.page_width,
            currency=state.currency,
            recorder=state.recorder,
            lines=state.lines,
        )
        setattr(state, "writer", writer)

    state.lines = writer.lines
    state.recorder = writer.recorder

    start_index = len(writer.lines)
    block_lines: list[str] = []

    def _capture(prev_len: int) -> None:
        block_lines.extend(writer.lines[prev_len:])

    def push(text: Any) -> None:
        prev = len(writer.lines)
        writer.line(text)
        _capture(prev)

    def extend(values: Iterable[Any]) -> None:
        for value in values:
            push(value)

    def detail(text: Any, *, indent: str = "    ") -> None:
        prev = len(writer.lines)
        writer.detail(text, indent=indent)
        _capture(prev)

    def _is_total_label(label: str) -> bool:
        clean = str(label or "").strip()
        if not clean:
            return False
        clean = clean.rstrip(":")
        clean = clean.lstrip("= ")
        return clean.lower().startswith("total")

    def _maybe_insert_total_separator(width: int) -> None:
        if not block_lines:
            return
        width = max(0, int(width))
        if width <= 0:
            return
        if block_lines[-1] == divider:
            return
        pad = max(0, state.page_width - width)
        short_divider = " " * pad + "-" * width
        if block_lines and block_lines[-1] == short_divider:
            return
        push(short_divider)

    def _append_currency_row(label: str, value: float, indent: str = "") -> None:
        right = state.format_row(label, value, indent=indent)
        if _is_total_label(label):
            currency_text = format_currency(value, state.currency)
            _maybe_insert_total_separator(len(currency_text))
        push(right)

    result = state.result
    breakdown = state.breakdown
    cfg = state.cfg
    currency = state.currency
    show_zeros = state.show_zeros
    page_width = state.page_width

    material_warning_summary_flag = bool(breakdown.get("material_warning_needed"))
    state.material_warning_summary = (
        state.material_warning_summary or material_warning_summary_flag
    )

    if isinstance(result, MutableMapping):
        result_map: MutableMapping[str, Any] = result
    else:
        result_map = _ensure_mapping(result)

    if isinstance(breakdown, MutableMapping):
        breakdown_map: MutableMapping[str, Any] = breakdown
    else:
        breakdown_map = _ensure_mapping(breakdown)

    material_raw = breakdown_map.get("material", {})
    material_map = _ensure_mapping(material_raw)

    material_block_new = breakdown_map.get("material_block") or {}
    material_stock_block = _ensure_mapping(material_block_new)

    material_selection_raw = breakdown_map.get("material_selected") or {}
    material_selection = _ensure_mapping(material_selection_raw)

    def _material_cost_components_from(
        *containers: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        for container in containers:
            if not isinstance(container, Mapping):
                continue
            candidate = container.get("material_cost_components")
            if isinstance(candidate, Mapping):
                return candidate
        return None

    material_cost_components = _material_cost_components_from(
        material_map,
        material_stock_block,
        material_selection,
        breakdown_map.get("material"),
        breakdown_map.get("material_block"),
    )

    def _resolve_overrides_source(
        container: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if not isinstance(container, Mapping):
            return None
        candidate = container.get("overrides")
        return candidate if isinstance(candidate, Mapping) else None

    material_overrides: Mapping[str, Any] | None = _resolve_overrides_source(result_map)
    if material_overrides is None:
        material_overrides = _resolve_overrides_source(breakdown_map)
    if material_overrides is None and isinstance(result_map, Mapping):
        for key in ("user_overrides", "overrides"):
            candidate = result_map.get(key)
            if isinstance(candidate, Mapping):
                material_overrides = candidate
                break

    baseline: Mapping[str, Any] = {}
    decision_state = result_map.get("decision_state")
    if isinstance(decision_state, Mapping):
        baseline_candidate = decision_state.get("baseline")
        if isinstance(baseline_candidate, Mapping):
            baseline = baseline_candidate
    if not baseline and isinstance(result_map, Mapping):
        baseline_candidate = result_map.get("baseline")
        if isinstance(baseline_candidate, Mapping):
            baseline = baseline_candidate
    if not baseline and isinstance(breakdown_map, Mapping):
        baseline_candidate = breakdown_map.get("baseline")
        if isinstance(baseline_candidate, Mapping):
            baseline = baseline_candidate

    pricing_obj: Mapping[str, Any] | None = None
    for container in (breakdown_map, result_map):
        if not isinstance(container, Mapping):
            continue
        candidate = container.get("pricing")
        if isinstance(candidate, MutableMapping):
            pricing_obj = candidate
            break
        if isinstance(candidate, Mapping):
            pricing_obj = dict(candidate)
            if isinstance(container, MutableMapping):
                container["pricing"] = pricing_obj
            break
    if pricing_obj is None:
        pricing: dict[str, Any] = {}
    elif isinstance(pricing_obj, dict):
        pricing = pricing_obj
    else:
        pricing = dict(pricing_obj)
    if isinstance(breakdown_map, MutableMapping):
        breakdown_map["pricing"] = pricing
    if isinstance(result_map, MutableMapping):
        result_map["pricing"] = pricing

    normalized_material_key = str(
        material_selection.get("material_lookup")
        or material_selection.get("normalized_material_key")
        or ""
    ).strip()

    canonical_material_breakdown = str(
        material_selection.get("canonical")
        or material_selection.get("canonical_material")
        or ""
    ).strip()
    if canonical_material_breakdown:
        material_selection.setdefault("canonical", canonical_material_breakdown)
        material_selection.setdefault("canonical_material", canonical_material_breakdown)

    canonical_display_lookup = ""
    if normalized_material_key:
        canonical_display_lookup = str(
            MATERIAL_DISPLAY_BY_KEY.get(normalized_material_key, "")
        ).strip()

    material_display_label = str(
        material_selection.get("material_display")
        or material_selection.get("display")
        or canonical_material_breakdown
        or ""
    ).strip()
    if canonical_display_lookup:
        material_display_label = canonical_display_lookup
    elif not material_display_label and normalized_material_key:
        fallback_display = MATERIAL_DISPLAY_BY_KEY.get(normalized_material_key, "")
        if fallback_display:
            material_display_label = str(fallback_display).strip()
    if material_display_label:
        material_selection["material_display"] = material_display_label
    if normalized_material_key:
        material_selection.setdefault("normalized_material_key", normalized_material_key)
        material_selection.setdefault("material_lookup", normalized_material_key)

    material_block = material_map
    stock_map = material_stock_block

    material_warning_entries_raw = breakdown_map.get("materials")
    material_warning_entries: list[Mapping[str, Any]] = []
    if isinstance(material_warning_entries_raw, Iterable):
        for entry in material_warning_entries_raw:
            if isinstance(entry, Mapping):
                material_warning_entries.append(entry)

    def _material_entries_summary(entries: Iterable[Mapping[str, Any]]) -> tuple[float, bool]:
        total = 0.0
        has_label = False
        for entry in entries:
            amount = _coerce_float_or_none(entry.get("amount"))
            if amount is not None:
                total += float(amount)
            if not has_label:
                label_text = str(entry.get("label") or entry.get("detail") or "").strip()
                if label_text:
                    has_label = True
        return total, has_label

    material_entries_total, material_entries_have_label = _material_entries_summary(
        material_warning_entries
    )

    materials_direct_total = _coerce_float_or_none(result_map.get("materials_direct"))
    if materials_direct_total is None:
        materials_direct_total = _coerce_float_or_none(breakdown_map.get("materials_direct"))
    materials_direct_total = float(materials_direct_total or 0.0)

    material_warning_summary = bool(material_entries_have_label) and (
        material_entries_total <= 0.0 and materials_direct_total <= 0.0
    )
    state.material_warning_summary = state.material_warning_summary or material_warning_summary
    if material_warning_summary and not state.material_warning_label:
        state.material_warning_label = "⚠ MATERIALS MISSING – review material costs"

    drilling_meta = _ensure_mapping(breakdown_map.get("drilling_meta"))
    ui_vars = _ensure_mapping(result_map.get("ui_vars"))
    g = _ensure_mapping(breakdown_map.get("geom"))
    geo_context = _ensure_mapping(breakdown_map.get("geo_context"))
    pricing_geom = _ensure_mapping(pricing.get("geom")) if isinstance(pricing, Mapping) else {}
    material_detail_for_breakdown = _ensure_mapping(
        breakdown_map.get("material_detail")
    )

    state.material_warning_summary = (
        state.material_warning_summary or bool(material_warning_summary)
    )

    state.warning_flags.setdefault("material_warning", state.material_warning_summary)

    if isinstance(material_map, MutableMapping):
        working_material_map: MutableMapping[str, Any] = material_map
    else:
        working_material_map = dict(material_map)

    if isinstance(stock_map, MutableMapping):
        working_stock_map: MutableMapping[str, Any] = stock_map
    else:
        working_stock_map = dict(stock_map)

    block_lines.clear()

    # ------------------------------------------------------------------
    # Blank dimension inference (legacy parity)

    def _lookup_blank(container: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(container, Mapping):
            return None
        for key in ("required_blank_in", "bbox_in"):
            entry = container.get(key)
            if isinstance(entry, Mapping):
                return entry
        return None

    blank_sources: list[Mapping[str, Any] | None] = []
    for parent in (breakdown_map, result_map):
        if isinstance(parent, Mapping):
            blank_sources.append(_lookup_blank(_as_mapping(parent.get("geo"))))
            blank_sources.append(_lookup_blank(_as_mapping(parent.get("geo_context"))))
            blank_sources.append(_lookup_blank(_as_mapping(parent.get("geom"))))
    blank_sources.append(_lookup_blank(geo_context))
    blank_sources.append(_lookup_blank(g))
    blank_sources.append(_lookup_blank(pricing_geom))

    req_map: Mapping[str, Any] | None = next((entry for entry in blank_sources if entry), None)
    w_val = coerce_positive_float(req_map.get("w")) if req_map else None
    h_val = coerce_positive_float(req_map.get("h")) if req_map else None
    t_val = coerce_positive_float(req_map.get("t")) if req_map else None
    w = float(w_val) if w_val else 0.0
    h = float(h_val) if h_val else 0.0
    t = float(t_val) if t_val else 0.0
    if t <= 0:
        t = 2.0

    if w <= 0 or h <= 0:
        geo_candidates: list[Mapping[str, Any]] = []
        for parent in (breakdown_map, result_map):
            if isinstance(parent, Mapping):
                for key in ("geo", "geo_context", "geom"):
                    candidate = parent.get(key)
                    if isinstance(candidate, Mapping):
                        geo_candidates.append(candidate)
        if isinstance(geo_context, Mapping):
            geo_candidates.append(geo_context)
        if isinstance(g, Mapping):
            geo_candidates.append(g)
        if isinstance(pricing_geom, Mapping):
            geo_candidates.append(pricing_geom)
        for geo_candidate in geo_candidates:
            blank = _lookup_blank(geo_candidate)
            if not blank:
                continue
            if w <= 0:
                w_candidate = coerce_positive_float(blank.get("w"))
                if w_candidate:
                    w = float(w_candidate)
            if h <= 0:
                h_candidate = coerce_positive_float(blank.get("h"))
                if h_candidate:
                    h = float(h_candidate)
            if t <= 0:
                t_candidate = coerce_positive_float(blank.get("t"))
                if t_candidate:
                    t = float(t_candidate)
            if w > 0 and h > 0 and t > 0:
                break

    blank_len = max(w, h)
    blank_wid = min(w, h)
    required_blank = (blank_len, blank_wid, t)

    for context in (
        geo_context if isinstance(geo_context, MutableMapping) else None,
        g if isinstance(g, MutableMapping) else None,
        pricing_geom if isinstance(pricing_geom, MutableMapping) else None,
    ):
        if context is None:
            continue
        existing_blank = context.get("required_blank_in")
        if not isinstance(existing_blank, Mapping):
            context["required_blank_in"] = {
                "w": float(blank_wid),
                "h": float(blank_len),
                "t": float(required_blank[2]),
            }

    if blank_len > 0 and isinstance(working_stock_map, MutableMapping):
        if coerce_positive_float(working_stock_map.get("required_blank_len_in")) is None:
            working_stock_map["required_blank_len_in"] = float(blank_len)
        if coerce_positive_float(working_stock_map.get("required_blank_wid_in")) is None:
            working_stock_map["required_blank_wid_in"] = float(blank_wid)
        if required_blank[2] > 0 and coerce_positive_float(
            working_stock_map.get("required_blank_thk_in")
        ) is None:
            working_stock_map["required_blank_thk_in"] = float(required_blank[2])

    required_blank_len = float(blank_len) if blank_len > 0 else 0.0
    required_blank_wid = float(blank_wid) if blank_wid > 0 else 0.0

    if not working_material_map:
        return []

    mass_g = working_material_map.get("mass_g")
    net_mass_g = working_material_map.get("mass_g_net")
    if _coerce_float_or_none(net_mass_g) is None:
        fallback_net_mass = working_material_map.get("net_mass_g")
        if _coerce_float_or_none(fallback_net_mass) is not None:
            net_mass_g = fallback_net_mass
    upg = working_material_map.get("unit_price_per_g")
    minchg = working_material_map.get("supplier_min_charge")
    matcost = working_material_map.get("material_cost")
    scrap = working_material_map.get("scrap_pct", None)
    scrap_credit_entered = _is_truthy_flag(
        working_material_map.get("material_scrap_credit_entered")
    )
    scrap_credit = float(working_material_map.get("material_scrap_credit") or 0.0)
    unit_price_kg = working_material_map.get("unit_price_usd_per_kg")
    unit_price_lb = working_material_map.get("unit_price_usd_per_lb")
    price_source = working_material_map.get("unit_price_source") or working_material_map.get(
        "source"
    )
    price_asof = working_material_map.get("unit_price_asof")

    have_any = any(
        v
        for v in [
            mass_g,
            net_mass_g,
            upg,
            minchg,
            matcost,
            scrap,
            scrap_credit if scrap_credit_entered else 0.0,
        ]
    )

    detail_lines: list[str] = []
    total_material_cost: float | None = None
    material_net_cost: float | None = None

    if not have_any:
        return []

    push("Material & Stock")
    push(divider)

    canonical_material_display = str(material_display_label or "").strip()
    if not canonical_material_display and isinstance(material_selection, Mapping):
        canonical_material_display = str(
            material_selection.get("material_display")
            or material_selection.get("canonical")
            or material_selection.get("canonical_material")
            or ""
        ).strip()
    if not canonical_material_display and isinstance(drilling_meta, Mapping):
        drill_display = (
            drilling_meta.get("material") or drilling_meta.get("material_display")
        )
        if not drill_display:
            drill_display = (
                drilling_meta.get("material_key")
                or drilling_meta.get("material_lookup")
            )
            if drill_display:
                normalized_key = _normalize_lookup_key(drill_display)
                drill_display = MATERIAL_DISPLAY_BY_KEY.get(
                    normalized_key,
                    drill_display,
                )
        if drill_display:
            canonical_material_display = str(drill_display).strip()
    material_name_display = canonical_material_display
    if not material_name_display:
        material_name_display = (
            working_material_map.get("material_name")
            or working_material_map.get("material")
            or g.get("material")
            or ui_vars.get("Material")
            or ""
        )
    if isinstance(material_name_display, str):
        material_name_display = material_name_display.strip()
    else:
        material_name_display = str(material_name_display).strip()
    if material_name_display:
        material_display_label = str(material_name_display)
        material_selection.setdefault("material_display", material_display_label)
        push(f"  Material used:  {material_name_display}")

    blank_lines: list[str] = []
    need_len = _coerce_float_or_none(
        working_stock_map.get("required_blank_len_in")
        if isinstance(working_stock_map, Mapping)
        else None
    )
    need_wid = _coerce_float_or_none(
        working_stock_map.get("required_blank_wid_in")
        if isinstance(working_stock_map, Mapping)
        else None
    )
    need_thk = _coerce_float_or_none(
        working_stock_map.get("required_blank_thk_in")
        if isinstance(working_stock_map, Mapping)
        else None
    )

    if (need_len is None or need_len <= 0) and required_blank_len > 0:
        need_len = float(required_blank_len)
    if (need_wid is None or need_wid <= 0) and required_blank_wid > 0:
        need_wid = float(required_blank_wid)
    if (need_thk is None or need_thk <= 0) and required_blank[2] > 0:
        need_thk = float(required_blank[2])

    stock_len_val = _coerce_float_or_none(
        working_stock_map.get("stock_L_in")
        if isinstance(working_stock_map, Mapping)
        else None
    )
    stock_wid_val = _coerce_float_or_none(
        working_stock_map.get("stock_W_in")
        if isinstance(working_stock_map, Mapping)
        else None
    )
    stock_thk_val = _coerce_float_or_none(
        working_stock_map.get("stock_T_in")
        if isinstance(working_stock_map, Mapping)
        else None
    )

    material_lookup_for_pick = normalized_material_key or ""
    if not material_lookup_for_pick and material_display_label:
        material_lookup_for_pick = _normalize_lookup_key(material_display_label)
    picked_stock = _resolve_mcmaster_plate_for_quote(
        float(need_len) if need_len else None,
        float(need_wid) if need_wid else None,
        float(need_thk) if need_thk else None,
        material_key=material_lookup_for_pick or "MIC6",
        stock_L_in=float(stock_len_val) if stock_len_val else None,
        stock_W_in=float(stock_wid_val) if stock_wid_val else None,
        stock_T_in=float(stock_thk_val) if stock_thk_val else None,
    )

    if picked_stock:
        stock_len_val = float(picked_stock.get("len_in") or 0.0)
        stock_wid_val = float(picked_stock.get("wid_in") or 0.0)
        stock_thk_val = float(picked_stock.get("thk_in") or 0.0)
        part_number = picked_stock.get("mcmaster_part")
        source_hint = picked_stock.get("source") or "mcmaster-catalog"
        if isinstance(working_stock_map, MutableMapping):
            working_stock_map["stock_L_in"] = float(stock_len_val)
            working_stock_map["stock_W_in"] = float(stock_wid_val)
            working_stock_map["stock_T_in"] = float(stock_thk_val)
            working_stock_map["stock_source_tag"] = source_hint
            working_stock_map["source"] = source_hint
            if part_number:
                working_stock_map["mcmaster_part"] = part_number
                working_stock_map["part_no"] = part_number
                if not working_stock_map.get("stock_price_source"):
                    working_stock_map["stock_price_source"] = "mcmaster_api"
        if isinstance(working_material_map, MutableMapping):
            working_material_map["stock_source_tag"] = source_hint
            working_material_map["source"] = source_hint
            if part_number:
                working_material_map["mcmaster_part"] = part_number
                working_material_map["part_no"] = part_number
                if not working_material_map.get("stock_price_source"):
                    working_material_map["stock_price_source"] = (
                        working_stock_map.get("stock_price_source")
                        if isinstance(working_stock_map, Mapping)
                        else "mcmaster_api"
                    )
        if isinstance(result_map, MutableMapping):
            if part_number:
                result_map["mcmaster_part"] = part_number
                result_map["part_no"] = part_number
                if not result_map.get("stock_price_source"):
                    result_map["stock_price_source"] = (
                        working_stock_map.get("stock_price_source")
                        if isinstance(working_stock_map, Mapping)
                        else "mcmaster_api"
                    )
            result_map["stock_source"] = source_hint

    if (need_len is None or need_wid is None or need_thk is None) and isinstance(g, dict):
        plan_guess = g.get("stock_plan_guess")
        if isinstance(plan_guess, Mapping):
            if need_len is None:
                need_len = _coerce_float_or_none(
                    plan_guess.get("need_len_in") or plan_guess.get("required_len_in")
                )
            if need_wid is None:
                need_wid = _coerce_float_or_none(
                    plan_guess.get("need_wid_in") or plan_guess.get("required_wid_in")
                )
            if need_thk is None:
                need_thk = _coerce_float_or_none(
                    plan_guess.get("need_thk_in") or plan_guess.get("stock_thk_in")
                )

    if need_len and need_wid and need_thk:
        blank_lines.append(
            f"  Required blank (w/ margins): {need_len:.2f} × {need_wid:.2f} × {need_thk:.2f} in"
        )

    source_tag: str | None = None
    if isinstance(working_stock_map, Mapping):
        source_tag = (
            working_stock_map.get("stock_source_tag") or working_stock_map.get("source")
        )
    if not source_tag and isinstance(working_material_map, Mapping):
        source_tag = (
            working_material_map.get("stock_source_tag")
            or working_material_map.get("source")
        )
    if isinstance(source_tag, str):
        source_tag = source_tag.strip() or None

    if stock_len_val and stock_wid_val and stock_thk_val:
        part_label = ""
        if isinstance(result_map, Mapping):
            part_label = str(result_map.get("mcmaster_part") or "").strip()
        part_display = part_label or "—"
        stock_line = (
            "  Rounded to catalog: "
            f"{stock_len_val:.2f} × {stock_wid_val:.2f} × {stock_thk_val:.3f} in"
            f" (McMaster, {part_display})"
        )
        if source_tag:
            stock_line += f" ({source_tag})"
        thickness_diff = None
        diff_candidate = None
        if isinstance(working_stock_map, Mapping):
            diff_candidate = _coerce_float_or_none(working_stock_map.get("thickness_diff_in"))
        if diff_candidate is None and need_thk and stock_thk_val:
            thickness_diff = abs(float(stock_thk_val) - float(need_thk))
        elif diff_candidate is not None:
            thickness_diff = float(diff_candidate)
        if thickness_diff is not None and thickness_diff > 0.02:
            warning_mode = (
                "allowed" if bool(getattr(cfg, "allow_thickness_upsize", False)) else "blocked"
            )
            stock_line += f" (WARNING: thickness upsize {warning_mode})"
        blank_lines.append(stock_line)

    if blank_lines:
        detail_lines.extend(blank_lines)

    pass_through_map = _ensure_mapping(breakdown_map.get("pass_through"))
    shipping_pipeline = "pass_through"
    shipping_source: str | None = "pass_through"
    shipping_raw_value: Any = pass_through_map.get("Shipping")
    if not shipping_raw_value:
        shipping_raw_value = working_material_map.get("shipping")
        if shipping_raw_value:
            shipping_source = "material"
        else:
            shipping_source = None
    shipping_total = float(_coerce_float_or_none(shipping_raw_value) or 0.0)
    if shipping_pipeline == "pass_through":
        pass_through_map["Shipping"] = shipping_total
        if isinstance(working_material_map, MutableMapping):
            working_material_map.pop("shipping", None)
        show_material_shipping = False
    else:
        pass_through_map.pop("Shipping", None)
        if shipping_source and isinstance(working_material_map, MutableMapping):
            working_material_map["shipping"] = shipping_total
        show_material_shipping = (
            (shipping_total > 0)
            or (shipping_total == 0 and bool(shipping_source) and show_zeros)
        )
    if isinstance(breakdown_map, MutableMapping):
        breakdown_map["pass_through"] = pass_through_map

    material_cost_map: dict[str, Any] = {}
    if isinstance(working_stock_map, Mapping):
        material_cost_map.update(working_stock_map)
    if isinstance(working_material_map, Mapping):
        material_cost_map.update(working_material_map)
    computed_components = _material_cost_components(
        material_cost_map,
        overrides=material_overrides,
        cfg=cfg,
    )
    state.material_cost_components = computed_components
    total_material_cost = computed_components["total_usd"]
    material_net_cost = computed_components["net_usd"]
    if total_material_cost is not None:
        try:
            working_material_map["total_material_cost"] = total_material_cost
        except Exception:
            pass
        try:
            material_record = (
                breakdown_map.get("material") if isinstance(breakdown_map, Mapping) else None
            )
            if isinstance(material_record, MutableMapping):
                material_record["total_material_cost"] = float(total_material_cost)
            elif isinstance(material_record, Mapping):
                updated_material = dict(material_record)
                updated_material["total_material_cost"] = float(total_material_cost)
                if isinstance(breakdown_map, MutableMapping):
                    breakdown_map["material"] = updated_material
            elif isinstance(breakdown_map, MutableMapping):
                breakdown_map["material"] = {"total_material_cost": float(total_material_cost)}
        except Exception:
            pass

    scrap_context = {
        "scrap_pct": scrap,
        "scrap_credit_entered": scrap_credit_entered,
        "scrap_credit": scrap_credit,
        "scrap_credit_unit_price_usd_per_lb": working_material_map.get(
            "scrap_credit_unit_price_usd_per_lb"
        ),
        "price_source": price_source,
        "price_asof": price_asof,
        "supplier_min_charge": minchg,
    }
    material_updates, helper_detail_lines = build_material_detail_lines(
        working_material_map,
        scrap_context=scrap_context,
        currency=currency,
        show_zeros=show_zeros,
        show_material_shipping=show_material_shipping,
        shipping_total=shipping_total,
        shipping_source=shipping_source,
    )
    if isinstance(material_detail_for_breakdown, MutableMapping):
        for key, value in material_updates.items():
            if value is None:
                material_detail_for_breakdown.pop(key, None)
            else:
                material_detail_for_breakdown[key] = value
    detail_lines.extend(helper_detail_lines)

    def _coerce_dims(candidate: Any) -> tuple[float, float, float] | None:
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 3:
            try:
                Lc = float(candidate[0])
                Wc = float(candidate[1])
                Tc = float(candidate[2])
            except Exception:
                return None
            return (Lc, Wc, Tc)
        if isinstance(candidate, Mapping):
            try:
                Lc = float(candidate.get("len") or candidate.get("L") or candidate.get("length"))
                Wc = float(candidate.get("wid") or candidate.get("W") or candidate.get("width"))
                Tc = float(candidate.get("thk") or candidate.get("T") or candidate.get("thickness"))
            except Exception:
                return None
            return (Lc, Wc, Tc)
        return None

    stock_dims_candidate: tuple[float, float, float] | None = None
    for key in ("stock_dims", "stock_dimensions"):
        stock_dims_candidate = _coerce_dims(working_stock_map.get(key))
        if stock_dims_candidate:
            break

    stock_dims_sources: list[tuple[float, float, float]] = []
    if stock_dims_candidate:
        stock_dims_sources.append(stock_dims_candidate)
    dims_from_material = _coerce_dims(working_material_map.get("stock_dims"))
    if dims_from_material:
        stock_dims_sources.append(dims_from_material)
    dims_from_pricing = _coerce_dims(pricing.get("stock_dims")) if isinstance(pricing, Mapping) else None
    if dims_from_pricing:
        stock_dims_sources.append(dims_from_pricing)

    stock_L_val = stock_len_val
    stock_W_val = stock_wid_val
    stock_T_val = stock_thk_val

    for dims in stock_dims_sources:
        if dims[0] and not stock_L_val:
            stock_L_val = float(dims[0])
        if dims[1] and not stock_W_val:
            stock_W_val = float(dims[1])
        if dims[2] and not stock_T_val:
            stock_T_val = float(dims[2])

    if not stock_L_val or not stock_W_val:
        fallback_dims = infer_plate_lw_in(g)
        if fallback_dims:
            if not stock_L_val:
                stock_L_val = float(fallback_dims[0])
            if not stock_W_val:
                stock_W_val = float(fallback_dims[1])

    if stock_L_val and stock_W_val and stock_T_val:
        stock_line = f"{float(stock_L_val):.2f} × {float(stock_W_val):.2f} × {float(stock_T_val):.3f} in"
    else:
        inferred_dims = infer_plate_lw_in(g)
        L_disp = stock_L_val
        W_disp = stock_W_val
        if (L_disp is None or W_disp is None) and inferred_dims:
            if L_disp is None:
                L_disp = inferred_dims[0]
            if W_disp is None:
                W_disp = inferred_dims[1]
        T_disp_val = stock_T_val
        if T_disp_val is None and isinstance(g, Mapping):
            T_disp_val = _coerce_float_or_none(g.get("thickness_in"))
        if T_disp_val is None and isinstance(g, Mapping):
            T_disp_val = _coerce_float_or_none(g.get("thickness_in_guess"))
        if L_disp and W_disp and T_disp_val:
            stock_line = f"{float(L_disp):.2f} × {float(W_disp):.2f} × {float(T_disp_val):.3f} in"
        else:
            T_disp = "—"
            if T_disp_val is not None:
                T_disp = f"{float(T_disp_val):.3f}"
            stock_line = f"— × — × {T_disp} in"

    mat_info = breakdown_map.get("material_info") if isinstance(breakdown_map, Mapping) else None
    if isinstance(mat_info, dict):
        mat_info["stock_size_display"] = stock_line

    for line in detail_lines:
        if not line:
            continue
        detail(line, indent="  ")

    mc: Mapping[str, Any] | None = computed_components
    if not mc:
        try:
            mc = _material_cost_components(
                working_material_map,
                overrides=material_overrides,
                cfg=cfg,
            )
        except Exception:
            mc = None
    if mc:
        stock_piece_val = mc.get("stock_piece_usd")
        if stock_piece_val is not None:
            stock_src = mc.get("stock_source") or ""
            src_suffix = f" ({stock_src})" if stock_src else ""
            _append_currency_row(f"Stock Piece{src_suffix}", float(stock_piece_val), indent="  ")
        else:
            base_source = mc.get("base_source") or ""
            base_label = "Base Material"
            if base_source:
                base_label = f"Base Material @ {base_source}"
            _append_currency_row(base_label, float(mc.get("base_usd", 0.0)), indent="  ")
        try:
            tax_val = float(mc.get("tax_usd") or 0.0)
        except Exception:
            tax_val = 0.0
        if tax_val:
            _append_currency_row("Material Tax:", round(tax_val, 2), indent="  ")
        try:
            scrap_val = float(mc.get("scrap_credit_usd") or 0.0)
        except Exception:
            scrap_val = 0.0
        scrap_text = mc.get("scrap_rate_text") or ""
        if scrap_val and scrap_text:
            _append_currency_row(
                f"Scrap Credit @ {scrap_text}",
                -round(scrap_val, 2),
                indent="  ",
            )
        elif scrap_val:
            _append_currency_row("Scrap Credit", -round(scrap_val, 2), indent="  ")
        try:
            base_for_total = float(mc.get("base_usd") or 0.0)
        except Exception:
            base_for_total = 0.0
        tax_for_total = float(tax_val)
        total_material_cost_val = mc.get("total_usd")
        if total_material_cost_val is not None:
            try:
                total_material_cost = round(float(total_material_cost_val), 2)
            except Exception:
                total_material_cost = None
        else:
            total_material_cost = None
        if total_material_cost is None:
            scrap_for_total = min(float(scrap_val), base_for_total + tax_for_total)
            total_material_cost = round(
                base_for_total + tax_for_total - scrap_for_total,
                2,
            )
        _append_currency_row("Total Material Cost :", total_material_cost, indent="  ")
    elif total_material_cost is not None:
        _append_currency_row("Total Material Cost :", total_material_cost, indent="  ")
    push("")

    state.material_component_total = total_material_cost
    state.material_component_net = material_net_cost
    state.material_net_cost = material_net_cost

    return list(writer.lines[start_index:])
