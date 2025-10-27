"""Helpers for rendering the appendix section of quote output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import math
import textwrap

from collections.abc import Mapping as _MappingABC, MutableMapping as _MutableMappingABC

from cad_quoter.ui.planner_render import PlannerBucketRenderState


@dataclass
class RenderState:
    """Container describing the appendix rendering context."""

    lines: list[str]
    emit: Callable[[str], None]
    ensure_blank_line: Callable[[], None]
    divider: str
    page_width: int
    context: MutableMapping[str, Any]
    deferred_callbacks: list[Callable[["RenderState"], None]] = field(default_factory=list)

    def push(self, text: str) -> None:
        self.emit(text)

    def defer(self, callback: Callable[["RenderState"], None]) -> None:
        self.deferred_callbacks.append(callback)

    def run_deferred(self) -> None:
        while self.deferred_callbacks:
            callback = self.deferred_callbacks.pop(0)
            callback(self)

    def get(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.context[key] = value


@dataclass
class AppendixResult:
    """Structured output produced by :func:`render_appendix`."""

    lines: list[str]
    ladder_totals: Mapping[str, Any]
    expedite_pct: float
    margin_pct: float
    subtotal_before_margin: float
    expedite_cost: float
    final_price: float
    ladder_subtotal: float
    quick_what_ifs: list[dict[str, Any]]
    slider_sample_points: list[dict[str, Any]]
    margin_slider: dict[str, Any] | None
    why_parts: list[str]


def _pct_label(value: float) -> str:
    try:
        pct_value = float(value)
    except Exception:
        pct_value = 0.0
    if not math.isfinite(pct_value):
        pct_value = 0.0
    pct_value = max(0.0, pct_value)
    text = f"{pct_value * 100:.1f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return f"{text}%"


def render_appendix(state: RenderState) -> AppendixResult:
    """Emit appendix sections and return the rendered output metadata."""

    ctx = state.context
    lines = state.lines
    divider = state.divider
    page_width = state.page_width

    applied_pcts: MutableMapping[str, Any] = ctx["applied_pcts"]
    breakdown = ctx.get("breakdown")
    result = ctx.get("result")
    params = ctx.get("params")
    totals = ctx.get("totals")
    subtotal = ctx.get("subtotal")
    decision_state = ctx.get("decision_state")

    replace_line: Callable[[int, str], None] = ctx["replace_line"]
    format_row: Callable[[Any, Any], str] = ctx["format_row"]
    final_price_row_index: int = ctx["final_price_row_index"]

    safe_float: Callable[[Any, float], float] = ctx["safe_float"]
    compute_pricing_ladder: Callable[..., Mapping[str, Any]] = ctx["compute_pricing_ladder"]

    currency: str = ctx["currency"]
    fmt_money: Callable[[Any, str], str] = ctx["fmt_money"]
    pct_formatter: Callable[[Any], str] = ctx["pct_formatter"]
    row_fn: Callable[[str, Any], None] = ctx["row_fn"]

    qty = ctx.get("qty")
    directs = ctx.get("directs")
    machine_labor_total = ctx.get("machine_labor_total")
    nre_per_part = ctx.get("nre_per_part")
    pass_through_total = ctx.get("pass_through_total")
    vendor_items_total = ctx.get("vendor_items_total")

    ascii_table = ctx["ascii_table"]
    write_wrapped: Callable[[str, str], None] = ctx["write_wrapped"]
    llm_notes: Sequence[str] = ctx.get("llm_notes", ())
    explanation_lines: list[str] = ctx.get("explanation_lines", [])
    why_lines: list[str] = ctx.get("why_lines", [])
    why_parts: list[str] = ctx.get("why_parts", [])
    bucket_why_summary_line: str | None = ctx.get("bucket_why_summary_line")

    append_removal_debug_if_enabled: Callable[[list[str], Any], None] = ctx["append_removal_debug_if_enabled"]
    removal_summary_for_display = ctx.get("removal_summary_for_display")

    planner_result = ctx.get("planner_result")
    process_plan_summary_local = ctx.get("process_plan_summary_local")
    process_rows_rendered = ctx.get("process_rows_rendered")
    bucket_state = ctx.get("bucket_state")
    explain_quote: Callable[..., str] = ctx["explain_quote"]
    hour_trace_data = ctx.get("hour_trace_data")
    geometry_for_explainer = ctx.get("geometry_for_explainer")

    planner_bucket_view = ctx.get("planner_bucket_view")
    canonical_bucket_rollup = ctx.get("canonical_bucket_rollup")
    hour_summary_entries = ctx.get("hour_summary_entries")
    process_meta = ctx.get("process_meta")
    bucket_minutes_detail = ctx.get("bucket_minutes_detail")

    resolve_llm_debug_enabled: Callable[..., bool] = ctx["resolve_llm_debug_enabled"]
    app_env_llm_debug_enabled = ctx.get("app_env_llm_debug_enabled", False)

    coerce_float_or_none: Callable[[Any], float | None] = ctx["coerce_float_or_none"]

    quick_what_if_entries: list[dict[str, Any]] = []
    margin_slider_payload: dict[str, Any] | None = None
    slider_sample_points: list[dict[str, Any]] = []

    initial_line_count = len(lines)

    skip_pricing_ladder = bool(ctx.get("skip_pricing_ladder"))
    if not skip_pricing_ladder:
        # Pricing ladder banner
        state.push("Pricing Ladder")
        state.push(divider)

    override_sources: list[Mapping[str, Any]] = []

    def _coerce_mapping(source: Any) -> dict[str, Any] | None:
        if isinstance(source, dict):
            return source
        if isinstance(source, _MappingABC):
            try:
                return dict(source)
            except Exception:
                return None
        return None

    def _collect_override_source(candidate: Any) -> None:
        mapping = _coerce_mapping(candidate)
        if mapping:
            override_sources.append(mapping)

    _collect_override_source(applied_pcts)
    if isinstance(breakdown, _MappingABC):
        _collect_override_source(breakdown.get("config"))
        _collect_override_source(breakdown.get("overrides"))
        _collect_override_source(breakdown.get("params"))
    quote_state_payload: Mapping[str, Any] | None = None
    if isinstance(result, _MappingABC):
        _collect_override_source(result.get("config"))
        _collect_override_source(result.get("overrides"))
        _collect_override_source(result.get("params"))
        quote_state_payload = result.get("quote_state")
    if isinstance(params, _MappingABC):
        _collect_override_source(params)
    if isinstance(quote_state_payload, _MappingABC):
        for nested_key in ("user_overrides", "overrides", "effective", "config", "params"):
            _collect_override_source(quote_state_payload.get(nested_key))

    def _resolve_ladder_pct(keys: Sequence[str], default: float) -> float:
        sentinel = object()
        for source in override_sources:
            for key in keys:
                value = sentinel
                try:
                    value = source.get(key, sentinel)  # type: ignore[arg-type]
                except Exception:
                    try:
                        value = source[key]  # type: ignore[index]
                    except Exception:
                        value = sentinel
                if value is sentinel or value is None:
                    continue
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        continue
                    return safe_float(stripped, default)
                return safe_float(value, default)
        return default

    expedite_pct_value = _resolve_ladder_pct(("ExpeditePct", "expedite_pct"), 0.0)
    margin_pct_value = _resolve_ladder_pct(("MarginPct", "margin_pct"), 0.15)

    applied_pcts.setdefault("MarginPct", margin_pct_value)
    if "ExpeditePct" not in applied_pcts and expedite_pct_value:
        applied_pcts["ExpeditePct"] = expedite_pct_value

    ladder_totals = compute_pricing_ladder(
        subtotal,
        expedite_pct=expedite_pct_value,
        margin_pct=margin_pct_value,
    )

    with_expedite = ladder_totals["with_expedite"]
    subtotal_before_margin = ladder_totals.get("subtotal_before_margin", with_expedite)
    expedite_cost = ladder_totals.get("expedite_cost", 0.0)
    final_price = ladder_totals["with_margin"]

    if isinstance(totals, dict):
        totals["with_expedite"] = with_expedite
        totals["with_margin"] = final_price
        totals["price"] = final_price

    if isinstance(result, dict):
        result["price"] = final_price
    if isinstance(breakdown, dict):
        breakdown["final_price"] = final_price

    def _final_price_callback(state_obj: RenderState, price_value: Any = final_price) -> None:
        replace_line(final_price_row_index, format_row("Final Price per Part:", price_value))

    state.defer(_final_price_callback)

    subtotal_before_margin_val = safe_float(subtotal_before_margin, 0.0)
    final_price_val = safe_float(final_price, 0.0)
    expedite_amount_val = safe_float(expedite_cost, 0.0)
    ladder_subtotal_val = safe_float(
        ladder_totals.get("subtotal"),
        subtotal_before_margin_val - expedite_amount_val,
    )

    # Gather quick what-if entries
    def _normalize_quick_entries(source: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, float | None]] = set()

        def _try_add(candidate: Mapping[str, Any]) -> None:
            label = str(
                candidate.get("label")
                or candidate.get("name")
                or candidate.get("title")
                or candidate.get("scenario")
                or ""
            ).strip()
            detail = str(
                candidate.get("detail")
                or candidate.get("description")
                or candidate.get("notes")
                or ""
            ).strip()

            unit_price_val: float | None = None
            for price_key in (
                "unit_price",
                "unitPrice",
                "price",
                "unit_price_usd",
                "unitPriceUsd",
                "value",
            ):
                if price_key in candidate:
                    try:
                        unit_price_val = float(candidate[price_key])
                    except Exception:
                        continue
                    else:
                        break

            delta_val: float | None = None
            for delta_key in ("delta", "delta_price", "delta_amount", "change", "difference"):
                if delta_key in candidate:
                    try:
                        delta_val = float(candidate[delta_key])
                    except Exception:
                        continue
                    else:
                        break

            margin_val: float | None = None
            for margin_key in ("margin_pct", "margin", "margin_percent", "marginPercent"):
                if margin_key in candidate:
                    try:
                        margin_val = float(candidate[margin_key])
                    except Exception:
                        continue
                    else:
                        break

            if (
                not label
                and not detail
                and unit_price_val is None
                and delta_val is None
                and margin_val is None
            ):
                return

            entry: dict[str, Any] = {}
            if label:
                entry["label"] = label
            if unit_price_val is not None and math.isfinite(unit_price_val):
                entry["unit_price"] = round(unit_price_val, 2)
            if delta_val is not None and math.isfinite(delta_val):
                entry["delta"] = round(delta_val, 2)
            if detail:
                entry["detail"] = detail
            if margin_val is not None and math.isfinite(margin_val):
                entry["margin_pct"] = float(margin_val)
            entry["currency"] = currency

            key = (entry.get("label", ""), entry.get("unit_price"))
            if key in seen_keys:
                return
            seen_keys.add(key)
            normalized.append(entry)

        def _walk(obj: Any) -> None:
            if len(normalized) >= 10:
                return
            if isinstance(obj, _MappingABC):
                _try_add(obj)
                for child in obj.values():
                    if isinstance(child, (dict, _MappingABC, list, tuple, set)):
                        _walk(child)
            elif isinstance(obj, (list, tuple, set)):
                for child in obj:
                    if isinstance(child, (dict, _MappingABC, list, tuple, set)):
                        _walk(child)

        _walk(source)
        return normalized

    quick_source_value: Any | None = None
    quick_key_candidates = (
        "quick_what_ifs",
        "quickWhatIfs",
        "quick_what_if",
        "quick_whatifs",
        "what_if_options",
        "what_if_scenarios",
    )

    for container in (result, breakdown):
        if not isinstance(container, _MappingABC):
            continue
        for key in quick_key_candidates:
            if key in container:
                quick_source_value = container.get(key)
                break
        if quick_source_value is not None:
            break
        for key, value in container.items():
            try:
                key_text = str(key).strip().lower()
            except Exception:
                continue
            if "quick" in key_text and "what" in key_text:
                quick_source_value = value
                break
        if quick_source_value is not None:
            break

    if quick_source_value is None and isinstance(decision_state, _MappingABC):
        for key in quick_key_candidates:
            if key in decision_state:
                quick_source_value = decision_state.get(key)
                break
        if quick_source_value is None:
            for key, value in decision_state.items():
                try:
                    key_text = str(key).strip().lower()
                except Exception:
                    continue
                if "quick" in key_text and "what" in key_text:
                    quick_source_value = value
                    break

    if quick_source_value is not None:
        try:
            quick_what_if_entries = _normalize_quick_entries(quick_source_value)
        except Exception:
            quick_what_if_entries = []

    if not quick_what_if_entries:
        generated: list[dict[str, Any]] = []

        if subtotal_before_margin_val > 0.0:
            margin_step = 0.05
            margin_down = round(max(0.0, margin_pct_value - margin_step), 4)
            if margin_pct_value - margin_down >= 0.005:
                price_down = round(subtotal_before_margin_val * (1.0 + margin_down), 2)
                delta_down = round(price_down - final_price_val, 2)
                generated.append(
                    {
                        "label": f"Margin {_pct_label(margin_down)}",
                        "unit_price": price_down,
                        "delta": delta_down,
                        "detail": f"Adjust margin to {_pct_label(margin_down)}.",
                        "margin_pct": float(margin_down),
                        "currency": currency,
                    }
                )

            margin_up = round(min(1.0, margin_pct_value + margin_step), 4)
            if margin_up - margin_pct_value >= 0.005:
                price_up = round(subtotal_before_margin_val * (1.0 + margin_up), 2)
                delta_up = round(price_up - final_price_val, 2)
                generated.append(
                    {
                        "label": f"Margin {_pct_label(margin_up)}",
                        "unit_price": price_up,
                        "delta": delta_up,
                        "detail": f"Adjust margin to {_pct_label(margin_up)}.",
                        "margin_pct": float(margin_up),
                        "currency": currency,
                    }
                )

        if expedite_pct_value > 0.0 and ladder_subtotal_val > 0.0:
            price_without_expedite = round(ladder_subtotal_val * (1.0 + margin_pct_value), 2)
            delta_expedite = round(price_without_expedite - final_price_val, 2)
            generated.append(
                {
                    "label": "Remove expedite",
                    "unit_price": price_without_expedite,
                    "delta": delta_expedite,
                    "detail": f"Removes expedite surcharge ({_pct_label(expedite_pct_value)}).",
                    "margin_pct": float(margin_pct_value),
                    "currency": currency,
                }
            )

        quick_what_if_entries = generated

    if quick_what_if_entries:
        deduped: list[dict[str, Any]] = []
        seen_labels: set[tuple[str, float | None]] = set()
        for entry in quick_what_if_entries:
            label_text = str(entry.get("label") or "").strip()
            if not label_text:
                label_text = f"Scenario {len(deduped) + 1}"
                entry["label"] = label_text
            try:
                price_val = float(entry.get("unit_price", 0.0))
            except Exception:
                price_val = 0.0
            key = (label_text.lower(), round(price_val, 2))
            if key in seen_labels:
                continue
            seen_labels.add(key)
            try:
                entry["unit_price"] = round(float(entry.get("unit_price", 0.0)), 2)
            except Exception:
                entry.pop("unit_price", None)
            if "delta" in entry:
                try:
                    entry["delta"] = round(float(entry["delta"]), 2)
                except Exception:
                    entry.pop("delta", None)
            entry.setdefault("currency", currency)
            deduped.append(entry)
        quick_what_if_entries = deduped

    if subtotal_before_margin_val > 0.0:
        slider_min_pct = 0.0
        slider_step_pct = 0.01
        slider_max_pct = margin_pct_value + 0.1
        slider_max_pct = max(slider_max_pct, 0.3)
        slider_max_pct = min(max(slider_max_pct, margin_pct_value), 1.0)

        slider_ticks: set[float] = set()
        slider_ticks.add(round(slider_min_pct, 4))
        slider_ticks.add(round(max(slider_min_pct, min(slider_max_pct, margin_pct_value)), 4))
        slider_ticks.add(round(slider_max_pct, 4))

        display_step = 0.05
        if slider_max_pct > slider_min_pct and display_step > 0:
            steps = int(math.floor((slider_max_pct - slider_min_pct) / display_step + 1e-6))
            for idx in range(steps + 1):
                pct_val = slider_min_pct + idx * display_step
                if pct_val < slider_min_pct - 1e-9 or pct_val > slider_max_pct + 1e-9:
                    continue
                slider_ticks.add(round(max(slider_min_pct, min(slider_max_pct, pct_val)), 4))

        slider_points: list[dict[str, Any]] = []
        for pct_val in sorted(slider_ticks):
            price_point = round(subtotal_before_margin_val * (1.0 + pct_val), 2)
            slider_points.append(
                {
                    "margin_pct": float(pct_val),
                    "label": _pct_label(pct_val),
                    "unit_price": price_point,
                    "currency": currency,
                }
            )

        if slider_points:
            margin_slider_payload = {
                "base_unit_price": round(subtotal_before_margin_val, 2),
                "current_pct": float(round(margin_pct_value, 6)),
                "current_price": round(final_price_val, 2),
                "min_pct": float(round(slider_min_pct, 6)),
                "max_pct": float(round(slider_max_pct, 6)),
                "step_pct": float(round(slider_step_pct, 6)),
                "points": slider_points,
                "currency": currency,
            }

            sample_points: list[dict[str, Any]] = []
            min_point = slider_points[0]
            max_point = slider_points[-1]
            current_point = next(
                (
                    point
                    for point in slider_points
                    if math.isclose(point["margin_pct"], margin_pct_value, rel_tol=0.0, abs_tol=1e-6)
                ),
                None,
            )
            sample_points.append(min_point)
            if current_point and current_point not in sample_points:
                sample_points.append(current_point)
            if len(slider_points) > 2:
                mid_point = slider_points[len(slider_points) // 2]
                if mid_point not in sample_points:
                    sample_points.append(mid_point)
            if max_point not in sample_points:
                sample_points.append(max_point)

            seen_points: set[float] = set()
            for point in sample_points:
                pct_val = float(point.get("margin_pct", 0.0))
                rounded_key = round(pct_val, 4)
                if rounded_key in seen_points:
                    continue
                seen_points.add(rounded_key)
                slider_sample_points.append(
                    {
                        "margin_pct": pct_val,
                        "label": str(point.get("label") or _pct_label(pct_val)),
                        "unit_price": float(point.get("unit_price", 0.0)),
                        "currency": str(point.get("currency") or currency),
                        "is_current": bool(
                            math.isclose(pct_val, margin_pct_value, rel_tol=0.0, abs_tol=1e-6)
                        ),
                    }
                )

    if not skip_pricing_ladder:
        row_fn("Subtotal (Labor + Directs):", subtotal)
        if applied_pcts.get("ExpeditePct"):
            row_fn(
                f"+ Expedite ({pct_formatter(applied_pcts.get('ExpeditePct'))}):",
                expedite_cost,
            )
        row_fn("= Subtotal before Margin:", subtotal_before_margin)
        row_fn(
            f"Final Price with Margin ({pct_formatter(applied_pcts.get('MarginPct'))}):",
            final_price,
        )
        state.push("")

    def _format_dotted_line(label: str, value_text: str, *, indent: str = "  ") -> str:
        base = f"{indent}{label}"
        try:
            total_width = int(page_width)
        except Exception:
            total_width = 74
        total_width = max(32, min(120, total_width))
        spacing = total_width - len(base) - len(value_text) - 1
        if spacing < 2:
            return f"{base} {value_text}"
        return f"{base}{'.' * spacing} {value_text}"

    section_counter = 0
    quick_sections: list[list[str]] = []

    def _start_section(title: str) -> list[str]:
        nonlocal section_counter
        heading = f"{chr(ord('A') + section_counter)}) {title}"
        section_counter += 1
        block: list[str] = [heading]
        quick_sections.append(block)
        return block

    if slider_sample_points:
        if isinstance(qty, int) and qty > 0:
            qty_display = str(qty)
        else:
            qty_display = str(max(1, int(round(safe_float(qty, 1.0)))))
        section_lines = _start_section(f"Margin Slider (Qty = {qty_display})")
        for point in sorted(slider_sample_points, key=lambda p: p.get("margin_pct", 0.0)):
            label_text = f"{str(point.get('label') or '')} margin".strip() or "Margin"
            if point.get("is_current"):
                label_text = f"{label_text} (current)"
            amount_text = fmt_money(safe_float(point.get("unit_price"), 0.0), point.get("currency", currency))
            section_lines.append(_format_dotted_line(label_text, amount_text))

    current_qty = qty if isinstance(qty, int) and qty > 0 else max(1, int(round(safe_float(qty, 1.0))))
    direct_per_part = max(0.0, safe_float(directs, 0.0))
    labor_machine_per_part = max(0.0, safe_float(machine_labor_total, 0.0))
    amortized_per_part = max(0.0, safe_float(nre_per_part, 0.0))
    amortized_per_lot = amortized_per_part * max(1, current_qty)
    pass_through_lot = max(0.0, safe_float(pass_through_total, 0.0))
    vendor_items_lot = max(0.0, safe_float(vendor_items_total, 0.0))
    direct_fixed_lot = pass_through_lot + vendor_items_lot
    divisor = max(1, current_qty)
    direct_variable_per_part = max(0.0, direct_per_part - (direct_fixed_lot / divisor))

    qty_candidates_raw = [1, 2, 5, 10]
    if current_qty not in qty_candidates_raw:
        qty_candidates_raw.append(current_qty)
    qty_candidates = sorted({q for q in qty_candidates_raw if isinstance(q, int) and q > 0})

    qty_break_rows: list[tuple[int, float, float, float, float]] = []
    for candidate_qty in qty_candidates:
        labor_part = labor_machine_per_part + (amortized_per_lot / candidate_qty if candidate_qty > 0 else 0.0)
        direct_part = direct_variable_per_part + (direct_fixed_lot / candidate_qty if candidate_qty > 0 else 0.0)
        base_subtotal_candidate = labor_part + direct_part
        subtotal_with_expedite = base_subtotal_candidate * (1.0 + max(0.0, expedite_pct_value))
        final_candidate = subtotal_with_expedite * (1.0 + max(0.0, margin_pct_value))
        qty_break_rows.append(
            (
                candidate_qty,
                round(labor_part, 2),
                round(direct_part, 2),
                round(subtotal_with_expedite, 2),
                round(final_candidate, 2),
            )
        )

    if qty_break_rows:
        heading_text = f"Qty break (assumes same ops; programming amortized; {_pct_label(margin_pct_value)} margin"
        if expedite_pct_value > 0:
            heading_text += f"; expedite {_pct_label(expedite_pct_value)}"
        heading_text += ")"
        section_lines = _start_section(heading_text)
        qty_headers = [
            "Qty",
            "Labor $/part",
            "Directs $/part",
            "Subtotal",
            "Final",
        ]
        qty_rows_formatted = [
            [
                row_qty,
                fmt_money(labor_val, currency),
                fmt_money(direct_val, currency),
                fmt_money(subtotal_val, currency),
                fmt_money(final_val, currency),
            ]
            for row_qty, labor_val, direct_val, subtotal_val, final_val in qty_break_rows
        ]
        qty_table = ascii_table(
            qty_headers,
            qty_rows_formatted,
            col_widths=[5, 16, 16, 15, 15],
            col_aligns=["R", "R", "R", "R", "R"],
            header_aligns=["C", "C", "C", "C", "C"],
        )
        section_lines.extend(f"  {line}" for line in qty_table.splitlines())

    other_quick_entries: list[dict[str, Any]] = []
    if quick_what_if_entries:
        for entry in quick_what_if_entries:
            margin_present = entry.get("margin_pct") is not None
            label_lower = str(entry.get("label") or "").strip().lower()
            if slider_sample_points and margin_present and "margin" in label_lower:
                continue
            other_quick_entries.append(entry)

    if other_quick_entries:
        section_lines = _start_section("Other quick toggles")
        quick_toggle_rows: list[list[str]] = []
        for entry in other_quick_entries:
            label_text = str(entry.get("label") or "").strip() or "Scenario"
            amount_val = safe_float(entry.get("unit_price"), 0.0)
            amount_text = fmt_money(amount_val, entry.get("currency", currency))
            delta_val = entry.get("delta")
            if delta_val is not None:
                delta_float = safe_float(delta_val, 0.0)
                if delta_float < -0.01:
                    delta_prefix = "-"
                elif abs(delta_float) <= 0.01:
                    delta_prefix = "±"
                else:
                    delta_prefix = "+"
                delta_text = fmt_money(abs(delta_float), entry.get("currency", currency))
                delta_display = f"{delta_prefix}{delta_text}"
            else:
                delta_display = ""
            detail_text = str(entry.get("detail") or "").strip()
            quick_toggle_rows.append([
                f"{label_text}:",
                amount_text,
                delta_display,
                detail_text,
            ])
        quick_toggle_table = ascii_table(
            ["Scenario", "Unit price", "Δ", "Notes"],
            quick_toggle_rows,
            col_widths=[24, 12, 10, 20],
            col_aligns=["L", "R", "C", "L"],
            header_aligns=["L", "C", "C", "L"],
        )
        section_lines.extend(f"  {line}" for line in quick_toggle_table.splitlines())

    quick_section_lines: list[str] = []
    for block in quick_sections:
        if quick_section_lines and quick_section_lines[-1] != "":
            quick_section_lines.append("")
        quick_section_lines.extend(block)

    while quick_section_lines and quick_section_lines[-1] == "":
        quick_section_lines.pop()

    if quick_section_lines:
        state.ensure_blank_line()
        state.push("QUICK WHAT-IFS (INTERNAL KNOBS)")
        state.push(divider)
        state.push("Quick What-Ifs")
        for text_line in quick_section_lines:
            state.push(text_line)
        state.ensure_blank_line()

    if llm_notes:
        state.push("LLM Adjustments")
        state.push(divider)
        for note in llm_notes:
            for wrapped in textwrap.wrap(str(note), width=page_width):
                state.push(f"- {wrapped}")
        state.push("")

    if not explanation_lines:
        plan_info_for_explainer: Mapping[str, Any] | None = None
        plan_info_payload: dict[str, Any] = {}

        process_plan_for_explainer: Mapping[str, Any] | None = None
        if isinstance(process_plan_summary_local, _MappingABC) and process_plan_summary_local:
            process_plan_for_explainer = process_plan_summary_local
        elif isinstance(breakdown, _MappingABC):
            candidate_summary = breakdown.get("process_plan")
            if isinstance(candidate_summary, _MappingABC) and candidate_summary:
                process_plan_for_explainer = candidate_summary
        if isinstance(process_plan_for_explainer, _MappingABC) and process_plan_for_explainer:
            plan_info_payload["process_plan_summary"] = process_plan_for_explainer

        if isinstance(breakdown, _MappingABC):
            process_plan_map = breakdown.get("process_plan")
            if isinstance(process_plan_map, _MappingABC) and process_plan_map:
                plan_info_payload.setdefault("process_plan", process_plan_map)
            plan_pricing_map = breakdown.get("process_plan_pricing")
            if isinstance(plan_pricing_map, _MappingABC) and plan_pricing_map:
                plan_info_payload.setdefault("pricing", plan_pricing_map)

        planner_pricing_for_explainer: Mapping[str, Any] | None = None
        if isinstance(breakdown, _MappingABC):
            candidate_planner = breakdown.get("process_plan_pricing")
            if isinstance(candidate_planner, _MappingABC) and candidate_planner:
                planner_pricing_for_explainer = candidate_planner
        if planner_pricing_for_explainer is None and isinstance(result, _MappingABC):
            candidate_planner = result.get("process_plan_pricing")
            if isinstance(candidate_planner, _MappingABC) and candidate_planner:
                planner_pricing_for_explainer = candidate_planner
        if planner_pricing_for_explainer is None and isinstance(planner_result, _MappingABC):
            planner_pricing_for_explainer = planner_result

        if isinstance(planner_pricing_for_explainer, _MappingABC) and planner_pricing_for_explainer:
            plan_info_payload["planner_pricing"] = planner_pricing_for_explainer

        bucket_plan_info: dict[str, Any] = {}
        if isinstance(bucket_state, PlannerBucketRenderState):
            extra_map = getattr(bucket_state, "extra", None)
            if isinstance(extra_map, _MappingABC) and extra_map:
                try:
                    bucket_plan_info.update(dict(extra_map))
                except Exception:
                    for key, value in extra_map.items():
                        bucket_plan_info[key] = value
            bucket_minutes_detail_map = getattr(bucket_state, "bucket_minutes_detail", None)
            if isinstance(bucket_minutes_detail_map, _MappingABC) and bucket_minutes_detail_map:
                bucket_plan_info.setdefault(
                    "bucket_minutes_detail_for_render",
                    bucket_minutes_detail_map,
                )
        if bucket_plan_info:
            plan_info_payload["bucket_state_extra"] = bucket_plan_info

        if process_rows_rendered:
            plan_info_payload.setdefault(
                "process_rows_rendered",
                [
                    (
                        name,
                        minutes,
                        machine,
                        labor,
                        total,
                    )
                    for (name, minutes, machine, labor, total) in process_rows_rendered
                ],
            )

        if plan_info_payload:
            plan_info_for_explainer = plan_info_payload

        try:
            explanation_text = explain_quote(
                breakdown,
                hour_trace=hour_trace_data,
                geometry=geometry_for_explainer,
                render_state=bucket_state,
                plan_info=plan_info_for_explainer,
            )
        except Exception:
            explanation_text = ""
        if explanation_text:
            for line in str(explanation_text).splitlines():
                text = line.strip()
                if text:
                    explanation_lines.append(text)

    if explanation_lines:
        why_lines.extend(explanation_lines)
    if why_lines:
        why_parts.extend(why_lines)

    if bucket_why_summary_line:
        summary_text = bucket_why_summary_line.strip()
        if summary_text and summary_text not in why_parts:
            why_parts.append(summary_text)

    if why_parts:
        if lines and lines[-1]:
            state.push("")
        state.push("Why this price")
        state.push(divider)
        for part in why_parts:
            write_wrapped(part, "  ")
        if lines and lines[-1]:
            state.push("")
        append_removal_debug_if_enabled(lines, removal_summary_for_display)

    try:
        llm_debug_enabled = resolve_llm_debug_enabled(
            result,
            breakdown,
            params,
            {"llm_debug_enabled": bool(app_env_llm_debug_enabled)},
        )
    except Exception:
        llm_debug_enabled = bool(app_env_llm_debug_enabled)
    if llm_debug_enabled:
        try:
            planner_min = None
            canon_min = None
            hsum_hr = None
            meta_hr = None

            if isinstance(planner_bucket_view, _MappingABC):
                buckets_obj = (
                    planner_bucket_view.get("buckets")
                    if isinstance(planner_bucket_view.get("buckets"), _MappingABC)
                    else planner_bucket_view
                )
                if isinstance(buckets_obj, _MappingABC):
                    drill_entry = buckets_obj.get("Drilling") or buckets_obj.get("drilling")
                    if isinstance(drill_entry, _MappingABC):
                        planner_min = coerce_float_or_none(drill_entry.get("minutes"))

            if isinstance(canonical_bucket_rollup, _MappingABC):
                canon_min = coerce_float_or_none(canonical_bucket_rollup.get("drilling"))
                if canon_min is not None:
                    canon_min = round(float(canon_min) * 60.0, 1)

            if isinstance(hour_summary_entries, _MappingABC):
                for label, (hours, include_flag) in hour_summary_entries.items():
                    if str(label).strip().lower() == "drilling":
                        hsum_hr = coerce_float_or_none(hours)
                        break

            if isinstance(process_meta, _MappingABC):
                meta_hr = coerce_float_or_none((process_meta.get("drilling") or {}).get("hr"))

            state.push("DEBUG — Drilling sanity")
            state.push(divider)

            def _fmt(value: Any, unit: str) -> str:
                try:
                    numeric = float(value)
                except Exception:
                    return "—"
                if not math.isfinite(numeric):
                    return "—"
                return f"{numeric:.2f} {unit}"

            state.push(
                "  bucket(planner): "
                + _fmt(planner_min, "min")
                + "   canonical: "
                + _fmt(canon_min, "min")
                + "   hour_summary: "
                + _fmt(hsum_hr, "hr")
                + "   meta: "
                + _fmt(meta_hr, "hr")
            )
            state.push("")
        except Exception:
            pass

    state.run_deferred()

    context_start_index = ctx.get("start_index")
    if isinstance(context_start_index, int) and 0 <= context_start_index <= len(lines):
        appendix_start = context_start_index
    else:
        appendix_start = initial_line_count
    appended_lines = list(lines[appendix_start:])

    result_payload = AppendixResult(
        lines=appended_lines,
        ladder_totals=ladder_totals,
        expedite_pct=float(expedite_pct_value),
        margin_pct=float(margin_pct_value),
        subtotal_before_margin=subtotal_before_margin_val,
        expedite_cost=float(expedite_cost),
        final_price=float(final_price),
        ladder_subtotal=float(ladder_subtotal_val),
        quick_what_ifs=quick_what_if_entries,
        slider_sample_points=slider_sample_points,
        margin_slider=margin_slider_payload,
        why_parts=list(why_parts),
    )

    state.set("start_index", appendix_start)
    state.set("appendix_lines", appended_lines)
    state.set("ladder_totals", ladder_totals)
    state.set("why_parts", why_parts)
    state.set("quick_what_if_entries", quick_what_if_entries)
    state.set("slider_sample_points", slider_sample_points)
    state.set("margin_slider_payload", margin_slider_payload)
    state.set("expedite_pct_value", expedite_pct_value)
    state.set("margin_pct_value", margin_pct_value)
    state.set("subtotal_before_margin", subtotal_before_margin)
    state.set("expedite_cost", expedite_cost)
    state.set("final_price", final_price)
    state.set("ladder_subtotal", ladder_totals.get("subtotal"))

    return result_payload


__all__ = [
    "RenderState",
    "AppendixResult",
    "render_appendix",
]
