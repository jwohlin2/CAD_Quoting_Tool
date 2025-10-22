"""Helpers for estimating milling effort from geometry payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from cad_quoter.pricing.planner import _geom as _normalize_geom, _material_factor
from cad_quoter.speeds_feeds import (
    coerce_table_to_records,
    normalize_material_group_code,
    normalize_operation,
)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _lookup_rate(
    rates: Mapping[str, Any] | None,
    *keys: str,
    default: float,
) -> float:
    if not isinstance(rates, Mapping):
        return default

    search_keys = {str(key or "").strip().lower() for key in keys if key}
    if not search_keys:
        search_keys = set()

    def _scan(mapping: Mapping[str, Any]) -> float | None:
        for raw_key, raw_value in mapping.items():
            key_text = str(raw_key or "").strip()
            if key_text and key_text.lower() in search_keys:
                value = _coerce_float(raw_value, default=-1.0)
                if value > 0.0:
                    return value
        for raw_value in mapping.values():
            if isinstance(raw_value, Mapping):
                found = _scan(raw_value)
                if found is not None:
                    return found
        return None

    found_value = _scan(rates)
    if found_value is None or found_value <= 0.0:
        return default
    return found_value


def _iter_records(table: Any | None) -> Sequence[Mapping[str, Any]]:
    records = coerce_table_to_records(table)
    if records:
        return records
    if isinstance(table, Mapping):
        return (table,)  # type: ignore[return-value]
    if isinstance(table, Sequence) and not isinstance(table, (str, bytes, bytearray)):
        return tuple(row for row in table if isinstance(row, Mapping))
    return tuple()


def _resolve_feed_ipm(
    table: Any | None,
    material_group: str | None,
    *,
    operations: Sequence[str],
) -> float:
    records = _iter_records(table)
    if not records:
        return 0.0

    ops_lookup = {normalize_operation(op) for op in operations}
    ops_lookup = {op for op in ops_lookup if op}

    if not ops_lookup:
        return 0.0

    group_text = str(material_group or "").strip().upper()
    simple_group = normalize_material_group_code(group_text) if group_text else ""

    best_rate = 0.0
    for row in records:
        op = normalize_operation(row.get("operation"))
        if op not in ops_lookup:
            continue

        row_group = str(row.get("material_group") or row.get("iso_group") or "").strip().upper()
        row_simple = normalize_material_group_code(row_group) if row_group else ""
        if group_text:
            if row_group != group_text and (not simple_group or row_simple != simple_group):
                continue

        rate = _coerce_float(row.get("linear_cut_rate_ipm"), 0.0)
        if rate <= 0.0:
            rate = _coerce_float(row.get("line_rate_ipm"), 0.0)
        if rate <= 0.0:
            rate = _coerce_float(row.get("feed_ipm"), 0.0)
        if rate > best_rate:
            best_rate = rate
    return best_rate


def _default_finish_ipm(material: str | None) -> float:
    text = (material or "").strip().lower()
    if "al" in text:
        return 120.0
    if "copper" in text or "brass" in text:
        return 90.0
    if "stainless" in text or "ss" in text:
        return 30.0
    if "tool" in text or "h13" in text or "s7" in text:
        return 25.0
    return 40.0


def _default_face_stepover(thickness_in: float) -> float:
    if thickness_in <= 0.0:
        return 0.5
    return max(0.3, min(0.8, thickness_in / 2.0))


def _material_label(geom: Mapping[str, Any] | None, material_group: str | None) -> str:
    if isinstance(geom, Mapping):
        for key in ("material", "material_text", "material_display"):
            candidate = geom.get(key)
            if candidate:
                return str(candidate)
    if material_group:
        return str(material_group)
    return ""


def estimate_milling_minutes_from_geometry(
    *,
    geom: Mapping[str, Any] | None,
    sf_df: Any | None,
    material_group: str | None,
    rates: Mapping[str, Any] | None,
    emit_bottom_face: bool = False,
) -> dict[str, float] | None:
    """Estimate milling bucket metrics from geometry and rate inputs."""

    geometry = _normalize_geom(dict(geom or {}))

    thickness_in = max(0.0, _coerce_float(geometry.get("thickness_in"), 0.0))
    pocket_area_in2 = max(0.0, _coerce_float(geometry.get("pocket_area_in2"), 0.0))
    plate_area_in2 = max(0.0, _coerce_float(geometry.get("plate_area_in2"), 0.0))
    edge_len_in = max(0.0, _coerce_float(geometry.get("edge_len_in"), 0.0))
    flip_required = bool(geometry.get("flip_required"))

    material_label = _material_label(geom, material_group)
    mrr_in3_min, _ = _material_factor(material_label)
    mrr_in3_min = max(0.3, mrr_in3_min)

    rough_minutes = 0.0
    if pocket_area_in2 > 0.0 and thickness_in > 0.0:
        removal_volume = pocket_area_in2 * thickness_in
        rough_minutes = (removal_volume / mrr_in3_min) * 60.0

    finish_ipm = _resolve_feed_ipm(
        sf_df,
        material_group,
        operations=("Endmill_Profile", "Finish_Mill", "Profile_Mill"),
    )
    if finish_ipm <= 0.0:
        finish_ipm = _default_finish_ipm(material_label)

    finish_minutes = 0.0
    if edge_len_in > 0.0 and finish_ipm > 0.0:
        finish_minutes = (edge_len_in / finish_ipm) * 60.0

    face_minutes = 0.0
    if plate_area_in2 > 0.0 and finish_ipm > 0.0:
        passes = 1 + int(bool(emit_bottom_face or flip_required))
        stepover_in = _default_face_stepover(thickness_in)
        effective_length = (plate_area_in2 / max(stepover_in, 1e-3)) * passes
        face_feed = finish_ipm * 0.75
        if face_feed <= 0.0:
            face_feed = finish_ipm
        face_minutes = (effective_length / max(face_feed, 1.0)) * 60.0

    total_minutes = rough_minutes + finish_minutes + face_minutes
    total_minutes = max(0.0, total_minutes)

    if total_minutes <= 0.0:
        return None

    mach_rate = float(_lookup_rate(rates, "MillingRate", "CNC_Mill", default=95.0))
    labor_rate = float(_lookup_rate(rates, "MillingLaborRate", "LaborRate", default=45.0))

    milling_minutes = float(total_minutes)
    milling_attended_minutes = 0.0

    machine_cost = (milling_minutes / 60.0) * mach_rate
    labor_cost = (milling_attended_minutes / 60.0) * labor_rate

    print(
        f"[CHECK/mill-rate] min={milling_minutes:.2f} hr={milling_minutes / 60.0:.2f} "
        f"mach_rate={mach_rate:.2f}/hr => machine$={machine_cost:.2f}"
    )

    return {
        "minutes": milling_minutes,
        "machine$": machine_cost,
        "labor$": labor_cost,
        "total$": machine_cost + labor_cost,
    }


__all__ = ["estimate_milling_minutes_from_geometry"]
