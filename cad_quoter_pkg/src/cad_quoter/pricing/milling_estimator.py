"""Helpers for estimating milling effort from geometry payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import math
import re
from functools import lru_cache
from pathlib import Path
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


def _lookup_fraction(
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
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    return value
        for raw_value in mapping.values():
            if isinstance(raw_value, Mapping):
                found = _scan(raw_value)
                if found is not None:
                    return found
        return None

    found_value = _scan(rates)
    if found_value is None:
        return default
    if not math.isfinite(found_value):
        return default
    if found_value <= 0.0:
        return 0.0
    if found_value >= 1.0:
        return 1.0
    return float(found_value)


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


SPEEDS_CSV = Path(__file__).resolve().parent / "resources" / "speeds_feeds_merged.csv"


@lru_cache(maxsize=1)
def load_speeds_table() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with SPEEDS_CSV.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                if not raw:
                    continue

                def fnum(key: str, default: float | None = None) -> float | None:
                    value = (raw.get(key, "") or "").strip()
                    if not value:
                        return default
                    try:
                        return float(value)
                    except Exception:
                        return default

                rows.append(
                    {
                        "material": (raw.get("material", "") or "").strip(),
                        "material_group": (raw.get("material_group", "") or "").strip(),
                        "operation": (raw.get("operation", "") or "").strip().lower(),
                        "sfm_start": fnum("sfm_start", 0.0) or 0.0,
                        "feed_type": (raw.get("feed_type", "") or "").strip().lower(),
                        "fz_ipr_bins": [
                            (0.125, fnum("fz_ipr_0_125in")),
                            (0.25, fnum("fz_ipr_0_25in")),
                            (0.50, fnum("fz_ipr_0_5in")),
                        ],
                        "doc_axial_in": fnum("doc_axial_in", 0.0) or 0.0,
                        "woc_radial_pct": fnum("woc_radial_pct", 0.0) or 0.0,
                        "linear_cut_rate_ipm": fnum("linear_cut_rate_ipm"),
                        "ref_tap_ipm_per_in_dia": fnum("ref_tap_ipm_per_in_dia"),
                        "ref_cbore_ipm": fnum("ref_cbore_ipm"),
                        "ref_redrill_ipm": fnum("ref_redrill_ipm"),
                        "ref_jig_min_per_in_depth_per_in_dia": fnum(
                            "ref_jig_min_per_in_depth_per_in_dia"
                        ),
                    }
                )
    except FileNotFoundError:
        print(f"[INFO] [milling] speeds CSV not found: {SPEEDS_CSV}")
        return tuple()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARNING] [milling] failed to load speeds CSV: {exc}")
        return tuple()
    return tuple(rows)


def _operation_candidates(op: str | None) -> tuple[str, ...]:
    text = (op or "").strip().lower()
    if not text:
        return tuple()
    aliases: list[str] = [text]
    if text in {"milling", "mill"}:
        aliases.extend([
            "endmill_profile",
            "endmill_slot",
            "face_mill",
            "facemill",
        ])
    elif text in {"face", "face_mill", "facemill"}:
        aliases.extend(["face_mill", "facemill"])
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _normalize_material_group(material: str | None, explicit_group: str | None) -> str:
    group_text = (explicit_group or "").strip().upper()
    if group_text:
        return group_text
    material_text = (material or "").strip().lower()
    if material_text and len(material_text) <= 3 and material_text[0].isalpha():
        return material_text.upper()
    return ""


def _derive_group_from_material(material: str | None) -> str:
    material_text = (material or "").strip().lower()
    if not material_text:
        return ""
    table = load_speeds_table()
    for row in table:
        row_material = (row.get("material") or "").strip().lower()
        if material_text == row_material:
            return (row.get("material_group") or "").strip().upper()
    for row in table:
        row_material = (row.get("material") or "").strip().lower()
        if material_text and material_text in row_material:
            return (row.get("material_group") or "").strip().upper()
    return ""


def _find_row(
    material: str | None,
    op: str | None,
    *,
    material_group: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    op_candidates = _operation_candidates(op)
    if not op_candidates:
        return None, "none"

    table = load_speeds_table()
    if not table:
        return None, "none"

    material_text = (material or "").strip().lower()
    group_text = _normalize_material_group(material_text, material_group)
    if not group_text:
        group_text = _derive_group_from_material(material_text)

    if material_text:
        for op_name in op_candidates:
            for row in table:
                row_material = (row.get("material") or "").strip().lower()
                if row.get("operation") == op_name and material_text == row_material:
                    return row, "material"

    if group_text:
        for op_name in op_candidates:
            for row in table:
                row_group = (row.get("material_group") or "").strip().upper()
                if row.get("operation") == op_name and row_group == group_text:
                    return row, "group"

    for op_name in op_candidates:
        for row in table:
            if row.get("operation") == op_name:
                return row, "operation"
    return None, "none"


def _closest_bin_value(
    dia_in: float, bins: Sequence[tuple[float, float | None]]
) -> float | None:
    last: float | None = None
    for thresh, value in bins:
        if value is not None:
            last = value
        if dia_in <= thresh:
            return value if value is not None else last
    return last


def _rpm_from_sfm(sfm: float, dia_in: float) -> float:
    if dia_in <= 0.0 or sfm <= 0.0:
        return 0.0
    return (sfm * 12.0) / (math.pi * dia_in)


def _ipm_from_feed(
    row: Mapping[str, Any],
    rpm: float,
    dia_in: float,
    flutes: int,
) -> float:
    linear_override = row.get("linear_cut_rate_ipm")
    if linear_override:
        try:
            return max(0.0, float(linear_override))
        except Exception:
            return 0.0

    base = _closest_bin_value(dia_in, row.get("fz_ipr_bins", ()))
    if base is None or rpm <= 0.0:
        return 0.0

    feed_type = str(row.get("feed_type") or "").strip().lower()
    if feed_type == "ipr":
        return rpm * float(base)

    effective_flutes = max(int(flutes or 0), 1)
    return rpm * float(base) * effective_flutes


def _clamp_minutes(minutes: float, *, src: str = "milling") -> float:
    if minutes < 0.0:
        print(
            f"[WARNING] [{src}] negative minutes; clamped to 0.0 (raw={minutes:.2f})"
        )
        return 0.0
    if minutes > 24 * 60:
        print(
            f"[WARNING] [{src}] minutes out-of-range; clamped to 0.0 (raw={minutes:.2f})"
        )
        return 0.0
    return minutes


def _coerce_int(value: Any, default: int) -> int:
    try:
        result = int(float(value))
    except Exception:
        return default
    if result <= 0:
        return default
    return result


def _coerce_float_default(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if not math.isfinite(result):
        return default
    return result


def estimate_milling_minutes(
    material: str | None,
    paths: Sequence[Mapping[str, Any]] | None,
    *,
    material_group: str | None = None,
    default_flutes: int = 3,
    min_rpm: float = 300.0,
    max_rpm: float = 30000.0,
) -> tuple[float, list[dict[str, Any]]]:
    if not paths:
        return 0.0, []

    row, source = _find_row(material, "milling", material_group=material_group)
    if row is None:
        print("[INFO] [milling] no row for material; minutes=0")
        return 0.0, []

    details: list[dict[str, Any]] = []
    total_min = 0.0
    for idx, entry in enumerate(paths, 1):
        if not isinstance(entry, Mapping):
            continue

        dia = _coerce_float_default(entry.get("tool_dia_in"), 0.0)
        flutes = _coerce_int(entry.get("flutes"), default_flutes)
        length_in = _coerce_float_default(entry.get("length_in"), 0.0)
        entry_count = _coerce_int(entry.get("entry_count"), 0)
        overhead_sec = _coerce_float_default(entry.get("overhead_sec"), 0.0)

        sfm = _coerce_float_default(entry.get("sfm_override"), 0.0)
        if sfm <= 0.0:
            sfm = _coerce_float_default(row.get("sfm_start"), 0.0)

        rpm = _rpm_from_sfm(sfm, dia)
        rpm = max(min_rpm, min(max_rpm, rpm))

        override = entry.get("fz_or_ipr_override")
        if override is not None:
            override_val = _coerce_float_default(override, 0.0)
            feed_type = str(row.get("feed_type") or "").strip().lower()
            if feed_type == "ipr":
                ipm = rpm * override_val
            else:
                ipm = rpm * override_val * max(flutes, 1)
        else:
            ipm = _ipm_from_feed(row, rpm, dia, flutes)

        if ipm <= 0.0 or length_in <= 0.0:
            part_min = 0.0
        else:
            part_min = (length_in / ipm) * 60.0

        approach_retract_sec = 1.0
        oh_min = (entry_count * approach_retract_sec * 2.0 + overhead_sec) / 60.0

        minutes = _clamp_minutes(part_min + oh_min, src="milling")

        detail = {
            "idx": idx,
            "dia": dia,
            "flutes": flutes,
            "length_in": length_in,
            "sfm": sfm,
            "rpm": rpm,
            "ipm": ipm,
            "minutes": minutes,
            "overhead_min": oh_min,
            "doc_ax": entry.get("axial_doc_in") or row.get("doc_axial_in"),
            "woc_pct": entry.get("radial_woc_pct") or row.get("woc_radial_pct"),
            "source": source,
            "row_material": row.get("material"),
            "row_operation": row.get("operation"),
        }
        details.append(detail)
        total_min += minutes

        print(
            "[INFO] [mill-path]"
            f" idx={idx} dia={dia:.3f}\" flutes={flutes}"
            f" sfm={sfm:.0f} rpm={rpm:.0f} ipm={ipm:.2f}"
            f" len={length_in:.1f}in -> {minutes:.2f} min (overhead {oh_min:.2f}) [{source}]"
        )

    print(f"[INFO] [mill-sum] paths={len(details)} subtotal_min={total_min:.2f}")
    return total_min, details


MACHINE_RATE = {
    "milling": 90.00,
    "drilling": 95.00,
}
LABOR_RATE = 45.00


def build_milling_bucket(
    material: str | None,
    milling_paths: Sequence[Mapping[str, Any]] | None,
    *,
    material_group: str | None = None,
) -> dict[str, Any]:
    minutes, paths = estimate_milling_minutes(
        material,
        milling_paths,
        material_group=material_group,
    )

    mach_cost = (minutes / 60.0) * MACHINE_RATE["milling"]
    labor_cost = (minutes / 60.0) * LABOR_RATE
    bucket = {
        "minutes": round(minutes, 2),
        "machine$": round(mach_cost, 2),
        "labor$": round(labor_cost, 2),
        "total$": round(mach_cost + labor_cost, 2),
        "paths": paths,
    }
    print(
        f"[INFO] [mill-bucket] minutes={bucket['minutes']:.2f} "
        f"machine$={bucket['machine$']:.2f} labor$={bucket['labor$']:.2f} total$={bucket['total$']:.2f}"
    )
    return bucket


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
    material_label = _material_label(geom, material_group)

    def _extract_paths(source: Mapping[str, Any] | None) -> Sequence[Mapping[str, Any]] | None:
        if not isinstance(source, Mapping):
            return None
        direct = source.get("milling_paths")
        if isinstance(direct, Sequence):
            return direct  # type: ignore[return-value]
        derived = source.get("derived")
        if isinstance(derived, Mapping):
            derived_paths = derived.get("milling_paths")
            if isinstance(derived_paths, Sequence):
                return derived_paths  # type: ignore[return-value]
        return None

    milling_paths_raw = _extract_paths(raw_geom) or _extract_paths(geometry)
    milling_path_dicts: list[Mapping[str, Any]] = []
    if milling_paths_raw:
        for entry in milling_paths_raw:
            if isinstance(entry, Mapping):
                milling_path_dicts.append(dict(entry))

    if milling_path_dicts:
        bucket = build_milling_bucket(
            material_label or material_group,
            milling_path_dicts,
            material_group=material_group,
        )
        detail = {"paths": bucket.get("paths", [])}
        result = {
            "minutes": bucket.get("minutes", 0.0),
            "machine$": bucket.get("machine$", 0.0),
            "labor$": bucket.get("labor$", 0.0),
            "total$": bucket.get("total$", 0.0),
            "detail": detail,
        }
        if bucket.get("paths") is not None:
            result["paths"] = bucket.get("paths")
        return result

    thickness_in = max(0.0, _coerce_float(geometry.get("thickness_in"), 0.0))
    pocket_area_in2 = max(0.0, _coerce_float(geometry.get("pocket_area_in2"), 0.0))
    plate_area_in2 = max(0.0, _coerce_float(geometry.get("plate_area_in2"), 0.0))
    edge_len_in = max(0.0, _coerce_float(geometry.get("edge_len_in"), 0.0))
    flip_required = bool(geometry.get("flip_required"))

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

    mach_rate = float(_lookup_rate(rates, "MillingRate", "CNC_Mill", default=95.0))
    labor_rate = float(_lookup_rate(rates, "MillingLaborRate", "LaborRate", default=45.0))
    attend_ratio = _lookup_fraction(
        rates,
        "MillingAttendRatio",
        "MillingAttendFraction",
        "MillingAttendedFraction",
        "MillingAttendance",
        default=1.0,
    )

    milling_minutes = float(total_minutes)
    milling_attended_minutes = milling_minutes * max(0.0, min(attend_ratio, 1.0))

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


__all__ = [
    "estimate_milling_minutes_from_geometry",
    "load_speeds_table",
    "estimate_milling_minutes",
    "build_milling_bucket",
]
