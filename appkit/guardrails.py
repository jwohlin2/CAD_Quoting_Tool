"""Helpers for enforcing guardrails on effective quote data."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, TYPE_CHECKING

from cad_quoter.coerce import to_float, to_int
from cad_quoter.domain import (
    _as_float_or_none,
    _canonical_pass_label,
    canonicalize_pass_through_map,
)
from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from cad_quoter.domain_models.state import QuoteState


DEFAULT_MIN_SEC_PER_HOLE = 9.0
DEFAULT_MIN_MIN_PER_TAP = 0.2
DEFAULT_FINISH_FLOOR = 50.0


def build_guard_context(state: "QuoteState") -> dict[str, Any]:
    """Return a guardrail context derived from state geometry and baseline."""

    geo_ctx = state.geo or {}
    inner_geo_raw = geo_ctx.get("geo")
    inner_geo_ctx: dict[str, Any] = inner_geo_raw if isinstance(inner_geo_raw, dict) else {}

    hole_count_guard = _coerce_float_or_none(geo_ctx.get("hole_count"))
    if hole_count_guard is None:
        hole_count_guard = _coerce_float_or_none(inner_geo_ctx.get("hole_count"))
    if hole_count_guard is None:
        hole_list = geo_ctx.get("hole_diams_mm") or inner_geo_ctx.get("hole_diams_mm")
        if isinstance(hole_list, (list, tuple)):
            hole_count_guard = float(len(hole_list))

    tap_qty_guard = _coerce_float_or_none(geo_ctx.get("tap_qty"))
    if tap_qty_guard is None:
        tap_qty_guard = _coerce_float_or_none(inner_geo_ctx.get("tap_qty"))

    finish_flags_guard: set[str] = set()
    finishes_geo = geo_ctx.get("finishes") or inner_geo_ctx.get("finishes")
    if isinstance(finishes_geo, (list, tuple, set)):
        finish_flags_guard.update(
            str(flag).strip().upper()
            for flag in finishes_geo
            if isinstance(flag, str) and flag.strip()
        )
    explicit_finish_flags = geo_ctx.get("finish_flags") or inner_geo_ctx.get("finish_flags")
    if isinstance(explicit_finish_flags, (list, tuple, set)):
        finish_flags_guard.update(
            str(flag).strip().upper()
            for flag in explicit_finish_flags
            if isinstance(flag, str) and flag.strip()
        )

    guard_ctx: dict[str, Any] = {
        "hole_count": hole_count_guard,
        "tap_qty": tap_qty_guard,
        "min_sec_per_hole": DEFAULT_MIN_SEC_PER_HOLE,
        "min_min_per_tap": DEFAULT_MIN_MIN_PER_TAP,
        "needs_back_face": bool(
            geo_ctx.get("needs_back_face")
            or geo_ctx.get("from_back")
            or inner_geo_ctx.get("needs_back_face")
            or inner_geo_ctx.get("from_back")
        ),
        "baseline_pass_through": (
            state.baseline.get("pass_through")
            if isinstance(state.baseline.get("pass_through"), dict)
            else {}
        ),
    }
    if finish_flags_guard:
        guard_ctx["finish_flags"] = sorted(finish_flags_guard)
        guard_ctx.setdefault("finish_cost_floor", DEFAULT_FINISH_FLOOR)
    return guard_ctx


def enforce_process_floor_guardrails(
    final_hours: MutableMapping[str, Any],
    guard_ctx: Mapping[str, Any],
    clamp_notes: list[str],
) -> None:
    """Clamp drilling/tapping hours based on guard context floors."""

    hole_count_guard = _coerce_float_or_none(guard_ctx.get("hole_count"))
    try:
        hole_count_guard_int = int(round(float(hole_count_guard))) if hole_count_guard is not None else 0
    except Exception:
        hole_count_guard_int = 0
    min_sec_per_hole = to_float(guard_ctx.get("min_sec_per_hole"))
    min_sec_per_hole = float(min_sec_per_hole) if min_sec_per_hole is not None else DEFAULT_MIN_SEC_PER_HOLE
    if hole_count_guard_int > 0 and "drilling" in final_hours:
        current_drill = to_float(final_hours.get("drilling"))
        if current_drill is not None:
            drill_floor_hr = (hole_count_guard_int * min_sec_per_hole) / 3600.0
            if current_drill < drill_floor_hr - 1e-6:
                clamp_notes.append(
                    f"process_hours[drilling] {current_drill:.3f} → {drill_floor_hr:.3f} (guardrail)"
                )
                final_hours["drilling"] = drill_floor_hr

    tap_qty_guard = _coerce_float_or_none(guard_ctx.get("tap_qty"))
    try:
        tap_qty_guard_int = int(round(float(tap_qty_guard))) if tap_qty_guard is not None else 0
    except Exception:
        tap_qty_guard_int = 0
    min_min_per_tap = to_float(guard_ctx.get("min_min_per_tap"))
    min_min_per_tap = float(min_min_per_tap) if min_min_per_tap is not None else DEFAULT_MIN_MIN_PER_TAP
    if tap_qty_guard_int > 0 and "tapping" in final_hours:
        current_tap = to_float(final_hours.get("tapping"))
        if current_tap is not None:
            tap_floor_hr = (tap_qty_guard_int * min_min_per_tap) / 60.0
            if current_tap < tap_floor_hr - 1e-6:
                clamp_notes.append(
                    f"process_hours[tapping] {current_tap:.3f} → {tap_floor_hr:.3f} (guardrail)"
                )
                final_hours["tapping"] = tap_floor_hr


def enforce_setups_guardrail(
    effective: MutableMapping[str, Any],
    guard_ctx: Mapping[str, Any],
    clamp_notes: list[str],
    source_tags: MutableMapping[str, Any],
) -> None:
    """Ensure setups floor is honored when a back face is required."""

    if not guard_ctx.get("needs_back_face"):
        return

    current_setups = effective.get("setups")
    setups_int = to_int(current_setups) or 0
    if setups_int < 2:
        effective["setups"] = 2
        clamp_notes.append(f"setups {setups_int} → 2 (back-side guardrail)")
        source_tags["setups"] = "guardrail"


def enforce_finish_pass_guardrail(
    effective: MutableMapping[str, Any],
    guard_ctx: Mapping[str, Any],
    final_pass: MutableMapping[str, float] | None,
    pass_sources: MutableMapping[str, str],
    clamp_notes: list[str],
) -> MutableMapping[str, float] | None:
    """Ensure finishes meet minimum outsourced cost floors."""

    finish_flags_ctx = guard_ctx.get("finish_flags")
    finish_floor = _as_float_or_none(guard_ctx.get("finish_cost_floor"))
    finish_floor = float(finish_floor) if finish_floor is not None else DEFAULT_FINISH_FLOOR
    if not finish_flags_ctx or finish_floor <= 0:
        return final_pass

    baseline_pass_ctx_raw = (
        guard_ctx.get("baseline_pass_through")
        if isinstance(guard_ctx.get("baseline_pass_through"), dict)
        else {}
    )
    baseline_pass_ctx = canonicalize_pass_through_map(baseline_pass_ctx_raw)
    finish_pass_key = _canonical_pass_label(
        guard_ctx.get("finish_pass_key") or "Outsourced Vendors"
    )

    combined_pass: dict[str, float] = dict(baseline_pass_ctx)
    if isinstance(final_pass, Mapping):
        for key, value in final_pass.items():
            val = to_float(value)
            if val is not None:
                combined_pass[key] = combined_pass.get(key, 0.0) + float(val)

    current_finish_cost = combined_pass.get(finish_pass_key, 0.0)
    if current_finish_cost < finish_floor - 1e-6:
        needed = finish_floor - current_finish_cost
        if needed > 0:
            if not isinstance(final_pass, dict):
                final_pass = {}
            final_pass[finish_pass_key] = float(final_pass.get(finish_pass_key, 0.0) or 0.0) + needed
            clamp_notes.append(
                f"add_pass_through[{finish_pass_key}] {current_finish_cost:.2f} → {finish_floor:.2f} (finish guardrail)"
            )
            pass_sources[finish_pass_key] = "guardrail"
            effective["add_pass_through"] = final_pass
    return final_pass


def apply_drilling_floor_notes(
    state: "QuoteState",
    *,
    guard_ctx: Mapping[str, Any] | None = None,
) -> None:
    """Raise drilling hours to the guardrail floor and note the adjustment."""

    eff_hours_raw = state.effective.get("process_hours") if isinstance(state.effective, dict) else None
    if not isinstance(eff_hours_raw, dict) or not eff_hours_raw:
        return

    guard_ctx = dict(guard_ctx or getattr(state, "guard_context", {}) or {})
    min_sec_per_hole = to_float(guard_ctx.get("min_sec_per_hole"))
    min_sec_per_hole = float(min_sec_per_hole) if min_sec_per_hole is not None else DEFAULT_MIN_SEC_PER_HOLE

    try:
        hole_count_guard = int(float(guard_ctx.get("hole_count") or 0))
    except Exception:
        hole_count_guard = 0

    if hole_count_guard <= 0:
        geo_ctx = state.geo or {}
        try:
            hole_count_guard = int(float(geo_ctx.get("hole_count") or 0))
        except Exception:
            hole_count_guard = 0
        if hole_count_guard <= 0:
            holes = geo_ctx.get("hole_diams_mm")
            if isinstance(holes, (list, tuple)):
                hole_count_guard = len(holes)

    if hole_count_guard <= 0 or "drilling" not in eff_hours_raw:
        return

    current = to_float(eff_hours_raw.get("drilling")) or 0.0
    floor_hr = (hole_count_guard * min_sec_per_hole) / 3600.0
    if current >= floor_hr - 1e-6:
        return

    eff_hours_raw["drilling"] = floor_hr
    state.effective["process_hours"] = eff_hours_raw
    note = f"Raised drilling to floor for {hole_count_guard} holes"
    notes = state.effective.setdefault("notes", [])
    if isinstance(notes, list) and note not in notes:
        notes.append(note)

