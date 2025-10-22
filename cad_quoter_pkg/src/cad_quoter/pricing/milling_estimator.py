"""Helpers for estimating milling effort from geometry payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any, Mapping as _MappingABC

from cad_quoter.pricing.planner import _geom as _normalize_geom, _material_factor
from cad_quoter.speeds_feeds import (
    coerce_table_to_records,
    normalize_material_group_code,
    normalize_operation,
    ipm_from_rpm_ipt,
    rpm_from_sfm,
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


_FZ_IPR_PATTERN = re.compile(r"fz_ipr_(\d+(?:_\d+)?)in\b")


def _parse_fz_diameter(key: str) -> float | None:
    match = _FZ_IPR_PATTERN.match(key.strip().lower())
    if not match:
        return None
    token = match.group(1).replace("_", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _row_feed_per_rev(row: Mapping[str, Any], tool_diam_in: float | None) -> float:
    candidates: list[tuple[float, float]] = []
    for raw_key, raw_value in row.items():
        if not isinstance(raw_key, str):
            continue
        diam = _parse_fz_diameter(raw_key)
        if diam is None:
            continue
        value = _coerce_float(raw_value, 0.0)
        if value <= 0.0:
            continue
        candidates.append((diam, value))

    if candidates:
        if tool_diam_in and tool_diam_in > 0.0:
            candidates.sort(key=lambda item: (abs(item[0] - tool_diam_in), -item[1]))
            return candidates[0][1]
        return max(candidates, key=lambda item: item[1])[1]

    for key in ("fz", "ipt", "ipr", "feed_per_tooth", "feed_per_rev"):
        value = _coerce_float(row.get(key), 0.0)
        if value > 0.0:
            return value
    return 0.0


def _derive_feed_ipm(
    row: Mapping[str, Any],
    *,
    tool_diam_in: float | None,
    default_flutes: int,
) -> float:
    feed = 0.0
    sfm = _coerce_float(row.get("sfm_start"), 0.0)
    if sfm <= 0.0:
        sfm = _coerce_float(row.get("sfm"), 0.0)

    rpm = 0.0
    if sfm > 0.0 and tool_diam_in and tool_diam_in > 0.0:
        rpm = rpm_from_sfm(sfm, tool_diam_in)

    feed_type = str(row.get("feed_type") or row.get("feed_unit") or "").strip().lower()
    per_rev = _row_feed_per_rev(row, tool_diam_in)

    if rpm > 0.0 and per_rev > 0.0:
        if feed_type == "fz":
            flutes = int(_coerce_float(row.get("flutes"), float(default_flutes)))
            flutes = max(flutes, 1)
            feed = ipm_from_rpm_ipt(rpm, flutes, per_rev)
        elif feed_type == "ipr":
            feed = rpm * per_rev

    if feed <= 0.0:
        feed = _coerce_float(row.get("feed_ipm"), 0.0)
    return feed


def _resolve_feed_ipm(
    table: Any | None,
    material_group: str | None,
    *,
    operations: Sequence[str],
    tool_diam_in: float | None,
    default_flutes: int,
) -> tuple[float, _MappingABC[str, Any] | None]:
    records = _iter_records(table)
    if not records:
        return 0.0, None

    ops_lookup = {normalize_operation(op) for op in operations}
    ops_lookup = {op for op in ops_lookup if op}

    if not ops_lookup:
        return 0.0, None

    group_text = str(material_group or "").strip().upper()
    simple_group = normalize_material_group_code(group_text) if group_text else ""

    best_rate = 0.0
    best_row: _MappingABC[str, Any] | None = None
    for row in records:
        op = normalize_operation(row.get("operation"))
        if op not in ops_lookup:
            continue

        row_group = str(row.get("material_group") or row.get("iso_group") or "").strip().upper()
        row_simple = normalize_material_group_code(row_group) if row_group else ""
        if group_text and row_group != group_text and (not simple_group or row_simple != simple_group):
            continue

        rate = _coerce_float(row.get("linear_cut_rate_ipm"), 0.0)
        if rate <= 0.0:
            rate = _coerce_float(row.get("line_rate_ipm"), 0.0)
        if rate <= 0.0:
            rate = _derive_feed_ipm(
                row,
                tool_diam_in=tool_diam_in,
                default_flutes=default_flutes,
            )
        if rate > best_rate:
            best_rate = rate
            best_row = row
    return best_rate, best_row


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
    raw_geom = geom if isinstance(geom, Mapping) else {}

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

    def _raw_value(key: str) -> Any:
        if not isinstance(raw_geom, Mapping):
            return None
        if key in raw_geom:
            return raw_geom[key]
        derived = raw_geom.get("derived")
        if isinstance(derived, Mapping):
            return derived.get(key)
        return None

    contour_tool_diam = 0.0
    for key in (
        "finish_tool_diam_in",
        "perimeter_tool_diam_in",
        "rough_tool_diam_in",
        "tool_diam_in",
    ):
        candidate = _coerce_float(geometry.get(key), 0.0)
        if candidate <= 0.0:
            candidate = _coerce_float(_raw_value(key), 0.0)
        if candidate > 0.0:
            contour_tool_diam = candidate
            break
    if contour_tool_diam <= 0.0:
        contour_tool_diam = max(
            0.25,
            min(0.75, math.sqrt(edge_len_in / math.pi) if edge_len_in > 0 else 0.5),
        )
        if not math.isfinite(contour_tool_diam) or contour_tool_diam <= 0.0:
            contour_tool_diam = 0.5

    finish_ipm, finish_row = _resolve_feed_ipm(
        sf_df,
        material_group,
        operations=("Endmill_Profile", "Finish_Mill", "Profile_Mill"),
        tool_diam_in=contour_tool_diam,
        default_flutes=4,
    )
    if finish_ipm <= 0.0:
        finish_ipm = _default_finish_ipm(material_label)
        finish_row = None

    finish_minutes = 0.0
    if edge_len_in > 0.0 and finish_ipm > 0.0:
        finish_minutes = (edge_len_in / finish_ipm) * 60.0

    face_minutes = 0.0
    if plate_area_in2 > 0.0 and finish_ipm > 0.0:
        passes = 1 + int(bool(emit_bottom_face or flip_required))
        face_tool_diam = 0.0
        for key in ("face_tool_diam_in", "rough_face_tool_diam_in", "face_cutter_diam_in"):
            candidate = _coerce_float(geometry.get(key), 0.0)
            if candidate <= 0.0:
                candidate = _coerce_float(_raw_value(key), 0.0)
            if candidate > 0.0:
                face_tool_diam = candidate
                break
        if face_tool_diam <= 0.0:
            face_tool_diam = 2.0

        face_ipm, face_row = _resolve_feed_ipm(
            sf_df,
            material_group,
            operations=("FaceMill", "Face_Mill", "Facing", "Endmill_Profile"),
            tool_diam_in=face_tool_diam,
            default_flutes=6,
        )

        stepover_pct = 0.0
        for candidate in (face_row, finish_row):
            if not isinstance(candidate, Mapping):
                continue
            stepover_pct = _coerce_float(candidate.get("stepover_pct"), 0.0)
            if stepover_pct <= 0.0:
                stepover_pct = _coerce_float(candidate.get("woc_radial_pct"), 0.0)
            if stepover_pct > 0.0:
                break

        if stepover_pct > 1.0:
            stepover_pct /= 100.0
        if stepover_pct > 0.0:
            stepover_pct = max(0.05, min(stepover_pct, 1.0))
            stepover_in = stepover_pct * face_tool_diam
        else:
            stepover_in = _default_face_stepover(thickness_in)

        effective_length = (plate_area_in2 / max(stepover_in, 1e-3)) * passes

        face_feed = face_ipm if face_ipm > 0.0 else finish_ipm
        face_feed = max(face_feed * 0.75, 1.0) if face_feed > 0.0 else 1.0
        face_minutes = (effective_length / face_feed) * 60.0

    total_minutes = rough_minutes + finish_minutes + face_minutes
    total_minutes = max(0.0, total_minutes)

    if total_minutes <= 0.0:
        return None

    machine_rate = _lookup_rate(rates, "MillingRate", "CNC_Mill", default=95.0)
    labor_rate = _lookup_rate(rates, "MillingLaborRate", "LaborRate", default=0.0)

    machine_cost = machine_rate * (total_minutes / 60.0)
    labor_cost = labor_rate * 0.0  # Milling bucket treated as machine-only here.

    return {
        "minutes": total_minutes,
        "machine$": machine_cost,
        "labor$": labor_cost,
        "total$": machine_cost + labor_cost,
    }


__all__ = ["estimate_milling_minutes_from_geometry"]
