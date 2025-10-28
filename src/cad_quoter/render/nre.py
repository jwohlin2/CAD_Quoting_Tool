"""Render the legacy NRE section using the shared render state."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, TYPE_CHECKING

from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import format_hours, format_hours_with_rate
from cad_quoter.utils.text_rules import canonicalize_amortized_label as _canonical_amortized_label

from .state import _coerce_rate_value

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from . import RenderState


PROGRAMMING_PER_PART_LABEL = "Programming (per part)"
FIXTURE_AMORTIZED_LABEL = "Fixture Build (amortized)"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _ensure_mapping(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        return candidate
    return {}


def _resolve_rate_with_fallback(
    state: "RenderState", raw_rate: Any, *fallback_keys: str
) -> float:
    rate_val = _safe_float(raw_rate)
    if rate_val > 0:
        return rate_val
    for key in fallback_keys:
        if not key:
            continue
        resolved = _coerce_rate_value(state.rates.get(key))
        if resolved > 0:
            return resolved
    return 0.0


def render_nre(state: "RenderState") -> list[str]:
    """Return the legacy NRE section lines."""

    writer = QuoteWriter(
        divider=state.divider or "-" * max(10, state.page_width),
        page_width=state.page_width,
        currency=state.currency,
        recorder=state.recorder,
        lines=state.lines,
    )
    state.lines = writer.lines
    state.recorder = writer.recorder
    start_index = len(writer.lines)

    section = state.section()
    row = section.row
    hours_row = section.hours_row
    write_line = section.write_line
    write_detail = section.write_detail

    nre = state.nre
    nre_detail = state.nre_detail
    nre_cost_details = state.nre_cost_details
    labor_cost_totals = state.labor_cost_totals
    show_zeros = state.show_zeros

    writer.line("NRE / Setup Costs (per lot)")
    writer.line(state.divider or "")

    prog = dict(_ensure_mapping(nre_detail.get("programming")))
    fix = dict(_ensure_mapping(nre_detail.get("fixture")))

    programmer_hours = _safe_float(prog.get("prog_hr"))
    engineer_hours = _safe_float(prog.get("eng_hr"))

    fallback_programmer_rate = _safe_float(
        getattr(state.cfg, "labor_rate_per_hr", state.default_labor_rate)
        if state.cfg is not None
        else state.default_labor_rate,
        state.default_labor_rate,
    )
    if fallback_programmer_rate <= 0:
        fallback_programmer_rate = state.default_labor_rate

    programmer_rate_backfill = _coerce_rate_value(state.rates.get("ProgrammerRate"))
    if programmer_rate_backfill <= 0:
        programmer_rate_backfill = _coerce_rate_value(state.rates.get("ProgrammingRate"))

    has_programming_rate_detail = False
    if isinstance(nre_cost_details, Mapping):
        detail_key = "Programming & Eng (per lot)"
        has_programming_rate_detail = bool(nre_cost_details.get(detail_key))

    explicit_programmer_rate = _safe_float(prog.get("prog_rate"))
    if explicit_programmer_rate > 0:
        has_programming_rate_detail = True
        programmer_rate = explicit_programmer_rate
    elif programmer_rate_backfill > 0 and not has_programming_rate_detail:
        programmer_rate = programmer_rate_backfill
    else:
        programmer_rate = fallback_programmer_rate

    if state.separate_labor_cfg and state.cfg_labor_rate_value > 0:
        programmer_rate = state.cfg_labor_rate_value

    engineer_rate = programmer_rate
    state.programming_rate = programmer_rate

    programming_per_lot_val = _safe_float(prog.get("per_lot"))
    nre_programming_per_lot = _safe_float(nre.get("programming_per_lot"))
    if nre_programming_per_lot <= 0:
        legacy_per_part = _safe_float(nre.get("programming_per_part"))
        if legacy_per_part > 0:
            nre_programming_per_lot = legacy_per_part
            nre["programming_per_lot"] = legacy_per_part
        nre.pop("programming_per_part", None)

    qty_for_programming: Any = state.breakdown.get("qty")
    if qty_for_programming in (None, ""):
        decision_state = state.result.get("decision_state")
        if isinstance(decision_state, Mapping):
            baseline_state = decision_state.get("baseline")
            if isinstance(baseline_state, Mapping):
                qty_for_programming = baseline_state.get("qty")
    if qty_for_programming in (None, ""):
        qty_for_programming = state.qty

    try:
        qty_for_programming_float = float(qty_for_programming or 1)
    except Exception:
        qty_for_programming_float = float(state.qty or 1)
    if not math.isfinite(qty_for_programming_float) or qty_for_programming_float <= 0:
        qty_for_programming_float = 1.0

    prog_hr_total = _safe_float(nre.get("programming_hr"))
    aggregated_programming_hours = 0.0
    if programmer_hours > 0:
        aggregated_programming_hours += programmer_hours
    if engineer_hours > 0:
        aggregated_programming_hours += engineer_hours
    if prog_hr_total <= 0 and aggregated_programming_hours > 0:
        nre["programming_hr"] = aggregated_programming_hours
        prog_hr_total = aggregated_programming_hours

    total_programming_hours = prog_hr_total
    programming_cost_total = _safe_float(nre.get("programming_cost"))
    if prog_hr_total > 0 and programming_cost_total == 0:
        programming_rate_total = _coerce_rate_value(state.rates.get("ProgrammingRate"))
        nre["programming_cost"] = round(prog_hr_total * programming_rate_total, 2)

    if total_programming_hours <= 0:
        total_programming_hours = aggregated_programming_hours
    if total_programming_hours <= 0:
        total_programming_hours = programmer_hours + engineer_hours

    computed_programming_per_lot = 0.0
    if total_programming_hours > 0 and programmer_rate > 0:
        computed_programming_per_lot = round(total_programming_hours * programmer_rate, 2)

    programming_cost_lot = computed_programming_per_lot if computed_programming_per_lot > 0 else 0.0
    per_lot_source_for_amortized = programming_cost_lot
    if per_lot_source_for_amortized <= 0 and programming_per_lot_val > 0:
        per_lot_source_for_amortized = programming_per_lot_val
    if per_lot_source_for_amortized <= 0 and nre_programming_per_lot > 0:
        per_lot_source_for_amortized = nre_programming_per_lot

    programming_cost_per_part = 0.0
    if per_lot_source_for_amortized > 0:
        programming_cost_per_part = round(
            per_lot_source_for_amortized / max(qty_for_programming_float, 1.0), 2
        )

    if programming_per_lot_val <= 0 and nre_programming_per_lot > 0:
        programming_per_lot_val = round(nre_programming_per_lot, 2)
    if programming_per_lot_val <= 0 and computed_programming_per_lot > 0:
        programming_per_lot_val = round(computed_programming_per_lot, 2)

    if _safe_float(nre.get("programming_per_lot")) <= 0 and computed_programming_per_lot > 0:
        nre["programming_per_lot"] = computed_programming_per_lot

    if programming_cost_per_part > 0:
        labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = programming_cost_per_part
    elif _safe_float(labor_cost_totals.get(PROGRAMMING_PER_PART_LABEL)) <= 0:
        per_lot_source = computed_programming_per_lot
        if per_lot_source <= 0 and nre_programming_per_lot > 0:
            per_lot_source = nre_programming_per_lot
        if per_lot_source <= 0 and programming_per_lot_val > 0:
            per_lot_source = programming_per_lot_val
        try:
            labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = round(
                per_lot_source / max(qty_for_programming_float, 1.0), 2
            )
        except Exception:
            labor_cost_totals[PROGRAMMING_PER_PART_LABEL] = 0.0

    show_programming_row = (
        programming_per_lot_val > 0
        or show_zeros
        or any(_safe_float(prog.get(k)) > 0 for k in ("prog_hr", "eng_hr"))
    )
    if show_programming_row:
        row("Programming & Eng:", programming_per_lot_val)
        has_detail = False
        if programming_per_lot_val > 0 or show_zeros:
            row("Programming Cost:", programming_per_lot_val, indent="  ")
            has_detail = True
        if total_programming_hours > 0:
            write_line(f"  Programming Hrs: {format_hours(total_programming_hours)}")
            has_detail = True
        if programmer_hours > 0:
            has_detail = True
            write_line(
                f"- Programmer (lot): {format_hours_with_rate(programmer_hours, programmer_rate, state.currency)}",
                "    ",
            )
        if engineer_hours > 0:
            has_detail = True
            write_line(
                f"- Engineering (lot): {format_hours_with_rate(engineer_hours, engineer_rate, state.currency)}",
                "    ",
            )
        if not has_detail and isinstance(nre_cost_details, Mapping):
            prog_detail = nre_cost_details.get("Programming & Eng (per lot)")
            if prog_detail not in (None, ""):
                write_detail(str(prog_detail))

    fixture_build_hours = _safe_float(fix.get("build_hr"))
    fixture_build_rate = _resolve_rate_with_fallback(
        state, fix.get("build_rate"), "FixtureBuildRate", "ShopRate"
    )

    fixture_per_lot = _safe_float(fix.get("per_lot"))
    if fixture_per_lot > 0 or show_zeros or fixture_build_hours > 0:
        row("Fixturing:", fixture_per_lot)
        has_detail = False
        if fixture_build_hours > 0:
            has_detail = True
            write_line(
                f"- Build labor (lot): {format_hours_with_rate(fixture_build_hours, fixture_build_rate, state.currency)}",
                "    ",
            )
        if not has_detail and isinstance(nre_cost_details, Mapping):
            fix_detail = nre_cost_details.get("Fixturing (per lot)")
            if fix_detail not in (None, ""):
                write_detail(str(fix_detail))

    other_nre_total = 0.0
    for key, value in nre.items():
        if key in {"programming_per_lot", "fixture_per_part", "programming_per_part"}:
            continue
        if isinstance(value, (int, float)) and (value > 0 or show_zeros):
            label = str(key).replace("_", " ").title()
            amount_val = float(value)
            key_lower = str(key).lower()
            if key_lower.endswith(("_hr", "_hrs", "_hours")):
                hours_label = label
                if hours_label.endswith(" Hours"):
                    hours_label = f"{hours_label[:-6]} Hrs"
                elif hours_label.endswith(" Hour"):
                    hours_label = f"{hours_label[:-5]} Hrs"
                elif hours_label.endswith(" Hr"):
                    hours_label = f"{hours_label[:-3]} Hrs"
                else:
                    hours_label = f"{hours_label} Hrs"
                hours_row(f"{hours_label}:", amount_val)
            else:
                row(f"{label}:", amount_val)
            other_nre_total += amount_val

    programming_meta_detail = _ensure_mapping(nre_detail.get("programming"))
    programming_per_part_cost = _safe_float(
        labor_cost_totals.get(PROGRAMMING_PER_PART_LABEL)
    )
    if programming_per_part_cost <= 0:
        programming_per_part_cost = _safe_float(programming_meta_detail.get("per_part"))
    if programming_per_part_cost <= 0:
        per_lot_detail_val = _safe_float(programming_meta_detail.get("per_lot"))
        if per_lot_detail_val > 0:
            programming_per_part_cost = per_lot_detail_val / max(qty_for_programming_float, 1.0)
    if programming_per_part_cost <= 0:
        per_lot_from_nre = _safe_float(nre.get("programming_per_lot"))
        if per_lot_from_nre > 0:
            programming_per_part_cost = per_lot_from_nre / max(qty_for_programming_float, 1.0)
    nre["programming_per_part"] = programming_per_part_cost
    state.programming_per_part = programming_per_part_cost

    fixture_meta_detail = _ensure_mapping(nre_detail.get("fixture"))
    fixture_labor_per_part_cost = labor_cost_totals.get(FIXTURE_AMORTIZED_LABEL)
    if fixture_labor_per_part_cost is None:
        fixture_labor_per_part_cost = _safe_float(fixture_meta_detail.get("labor_cost"))
        divisor_qty = state.qty if state.qty else 1
        try:
            divisor_qty_val = float(divisor_qty)
        except Exception:
            divisor_qty_val = 1.0
        if not math.isfinite(divisor_qty_val) or divisor_qty_val <= 0:
            divisor_qty_val = 1.0
        fixture_labor_per_part_cost = (
            fixture_labor_per_part_cost / divisor_qty_val if divisor_qty_val else 0.0
        )
    fixture_labor_per_part_cost = _safe_float(fixture_labor_per_part_cost)
    nre["fixture_per_part"] = fixture_labor_per_part_cost
    state.fixture_per_part = fixture_labor_per_part_cost

    if fixture_labor_per_part_cost > 0:
        labor_cost_totals[FIXTURE_AMORTIZED_LABEL] = fixture_labor_per_part_cost
    else:
        labor_cost_totals.pop(FIXTURE_AMORTIZED_LABEL, None)

    amortized_pairs = [
        (PROGRAMMING_PER_PART_LABEL, programming_per_part_cost),
        (FIXTURE_AMORTIZED_LABEL, fixture_labor_per_part_cost),
    ]
    amortized_total = 0.0
    for label, value in amortized_pairs:
        numeric = _safe_float(value)
        canonical_label, is_amortized = _canonical_amortized_label(label)
        target_label = canonical_label or label
        if numeric <= 0 or not is_amortized:
            state.amortized_totals.pop(target_label, None)
            continue
        state.amortized_totals[target_label] = numeric
        amortized_total += numeric

    if amortized_total <= 0:
        try:
            amortized_total = float(
                _safe_float(programming_per_part_cost) + _safe_float(fixture_labor_per_part_cost)
            )
        except Exception:
            amortized_total = 0.0
    state.amortized_nre_total = amortized_total

    formatted_rows = state.format_rows(section.rows)
    for kind, index in section.events:
        if kind == "row":
            writer.line(formatted_rows[index])
        else:
            writer.line(section.detail_lines[index])

    if (
        (prog or fix or other_nre_total > 0)
        and section.events
        and writer.lines
        and writer.lines[-1].strip()
    ):
        writer.line("")

    return writer.lines[start_index:]
