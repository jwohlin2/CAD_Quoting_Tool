"""Helpers for milling speeds/feeds lookups."""

from __future__ import annotations

from math import pi
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["lookup_mill_params", "rpm_from_sfm", "ipm_from_rpm_ipt"]


def _nearest(rows: Iterable[Mapping[str, Any]], key: str, target: float) -> Mapping[str, Any] | None:
    """Return the row whose ``key`` value is closest to ``target``."""

    best_row: Mapping[str, Any] | None = None
    best_distance: float = float("inf")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_value = row.get(key, 0)
        try:
            value = float(raw_value)
        except Exception:
            continue
        distance = abs(value - target)
        if distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row


def _to_records(table: Any) -> list[dict[str, Any]]:
    """Return ``table`` as a list of mapping rows if possible."""

    if table is None:
        return []

    to_dict = getattr(table, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict("records")
            if isinstance(records, list):
                return [dict(row) for row in records if isinstance(row, Mapping)]
        except Exception:
            pass

    rows_attr = getattr(table, "_rows", None)
    if isinstance(rows_attr, list):
        return [dict(row) for row in rows_attr if isinstance(row, Mapping)]

    iterrows = getattr(table, "iterrows", None)
    if callable(iterrows):
        rows: list[dict[str, Any]] = []
        try:
            for _idx, row in iterrows():
                if isinstance(row, Mapping):
                    rows.append(dict(row))
                else:
                    try:
                        rows.append(dict(row))
                    except Exception:
                        continue
        except Exception:
            rows = []
        if rows:
            return rows

    if isinstance(table, Sequence) and not isinstance(table, (str, bytes, bytearray)):
        return [dict(row) for row in table if isinstance(row, Mapping)]

    return []


def lookup_mill_params(
    sf_df: Any,
    material_group: str,
    op: str,
    tool_diam_in: float,
) -> dict[str, float | int]:
    """Return milling parameters for ``material_group``/``op`` near ``tool_diam_in``."""

    records = _to_records(sf_df)
    target_material = str(material_group or "").strip().lower()
    target_operation = str(op or "").strip().lower()
    try:
        target_diameter = float(tool_diam_in)
    except Exception:
        target_diameter = 0.0

    def _normalize(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.strip().lower()

    material_exists = any("material_group" in row for row in records)
    if material_exists and target_material:
        material_rows = [row for row in records if _normalize(row.get("material_group")) == target_material]
        if material_rows:
            records = material_rows

    operation_exists = any("operation" in row for row in records)
    if operation_exists and target_operation:
        op_rows = [row for row in records if _normalize(row.get("operation")) == target_operation]
        if op_rows:
            records = op_rows

    row: Mapping[str, Any] | None = None
    if any("tool_diam_in" in row for row in records):
        row = _nearest(records, "tool_diam_in", target_diameter)
    if row is None and records:
        row = records[0]

    defaults: dict[str, float | int] = dict(
        sfm=800.0,
        ipt=0.0030,
        flutes=3,
        max_ap_in=0.100,
        max_ae_in=0.50 * target_diameter,
        stepover_pct=0.60,
        index_min_per_pass=0.02,
        toolchange_min=2.5,
    )

    def get(key: str) -> float | int:
        if row is None:
            return defaults[key]
        value = row.get(key)
        if value in (None, "", "nan"):
            return defaults[key]
        try:
            if key == "flutes":
                return int(value)
            return float(value)
        except Exception:
            return defaults[key]

    return {
        "sfm": float(get("sfm")),
        "ipt": float(get("ipt")),
        "flutes": int(get("flutes")),
        "max_ap_in": float(get("max_ap_in")),
        "max_ae_in": float(get("max_ae_in")),
        "stepover_pct": float(get("stepover_pct")),
        "index_min_per_pass": float(get("index_min_per_pass")),
        "toolchange_min": float(get("toolchange_min")),
    }


def rpm_from_sfm(sfm: float, tool_diam_in: float) -> float:
    """Return spindle RPM calculated from SFM and tool diameter."""

    return (float(sfm) * 12.0) / (pi * max(float(tool_diam_in), 1e-6))


def ipm_from_rpm_ipt(rpm: float, flutes: int, ipt: float) -> float:
    """Return feed rate (IPM) from RPM, flute count, and IPT."""

    return float(rpm) * max(int(flutes), 1) * float(ipt)
