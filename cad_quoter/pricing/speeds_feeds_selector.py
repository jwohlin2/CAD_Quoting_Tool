"""Lookup helpers for material-aware speeds and feeds selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import logging
import math
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = [
    "load_csv_as_records",
    "normalize_material",
    "pick_speeds_row",
    "unit_hp_cap",
]

log = logging.getLogger(__name__)

_RESOURCE_PACKAGES: Sequence[str] = (
    "cad_quoter.pricing.resources",
    "cad_quoter.resources",
)


def _iter_candidate_streams(path: str) -> Iterable[Any]:
    """Yield file-like objects for ``path`` from known locations."""

    candidate = Path(path)
    if candidate.is_file():
        yield candidate.open("r", encoding="utf-8", newline="")
        return

    if candidate.is_absolute():
        return

    for pkg in _RESOURCE_PACKAGES:
        try:
            root = resources.files(pkg)
        except (ModuleNotFoundError, FileNotFoundError):
            continue
        resource = root.joinpath(path)
        if not resource.is_file():
            continue
        try:
            yield resource.open("r", encoding="utf-8", newline="")
        except FileNotFoundError:
            continue

    here = Path(__file__).resolve().parent
    local_candidate = here / path
    if local_candidate.is_file():
        yield local_candidate.open("r", encoding="utf-8", newline="")


def load_csv_as_records(path: str) -> list[dict[str, Any]]:
    """Return the CSV at ``path`` as a list of dictionaries."""

    records: list[dict[str, Any]] = []
    for handle in _iter_candidate_streams(path):
        try:
            reader = csv.DictReader(handle)
            records = [dict(row) for row in reader]
        finally:
            handle.close()
        if records:
            break
    return records


def _normalise_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    normalised: list[dict[str, Any]] = []
    for row in rows:
        clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in dict(row).items()}
        op = str(clean.get("operation", "")).strip().lower().replace("-", "_")
        if not op:
            continue
        clean["_norm_operation"] = op
        clean["_norm_material"] = str(clean.get("material", "")).strip().lower()
        clean["_iso_group"] = str(clean.get("material_group", "")).strip().upper()
        normalised.append(clean)
    return tuple(normalised)


@lru_cache(maxsize=1)
def _load_material_map() -> tuple[dict[str, Any], ...]:
    records = load_csv_as_records("material_map.csv")
    normalised: list[dict[str, Any]] = []
    for row in records:
        clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        input_label = str(clean.get("input_label", "")).strip().lower()
        if not input_label:
            continue
        clean["_norm_input"] = input_label
        normalised.append(clean)
    return tuple(normalised)


@lru_cache(maxsize=1)
def _load_speeds_table() -> tuple[dict[str, Any], ...]:
    records = load_csv_as_records("speeds_feeds_merged.csv")
    return _normalise_rows(records)


def _coerce_records(table: Any | None) -> tuple[dict[str, Any], ...]:
    if table is None:
        return _load_speeds_table()
    if hasattr(table, "to_dict"):
        try:
            records = table.to_dict("records")  # type: ignore[attr-defined]
        except Exception:
            records = None
        if isinstance(records, list):
            return _normalise_rows(records)
    stub_rows = getattr(table, "_rows", None)
    if isinstance(stub_rows, list) and stub_rows:
        return _normalise_rows(stub_rows)
    if isinstance(table, Sequence):
        rows: list[Mapping[str, Any]] = [row for row in table if isinstance(row, Mapping)]  # type: ignore[arg-type]
        if rows:
            return _normalise_rows(rows)
    if isinstance(table, Mapping):
        return _normalise_rows([table])
    return _load_speeds_table()


def normalize_material(user_str: str | None) -> dict[str, Any] | None:
    """Return the best-fit material map record for ``user_str``."""

    lookup = (user_str or "").strip().lower()
    if not lookup:
        return None

    material_map = _load_material_map()
    for row in material_map:
        if lookup == row.get("_norm_input"):
            return row

    tokens = set(lookup.replace("-", " ").split())
    best_row: dict[str, Any] | None = None
    best_score = 0
    for row in material_map:
        label = row.get("_norm_input", "")
        row_tokens = set(str(label).replace("-", " ").split())
        score = len(tokens & row_tokens)
        if score > best_score:
            best_row = row
            best_score = score
    return best_row


def _norm_operation(operation: str | None) -> str:
    return str(operation or "").strip().lower().replace("-", "_")


def _select_op_rows(records: Sequence[dict[str, Any]], operation: str) -> list[dict[str, Any]]:
    target = _norm_operation(operation)
    return [row for row in records if row.get("_norm_operation") == target]


def _select_material_rows(
    records: Sequence[dict[str, Any]],
    operation: str,
    material: str | None,
) -> list[dict[str, Any]]:
    if not material:
        return []
    target = material.strip().lower()
    return [
        row
        for row in _select_op_rows(records, operation)
        if row.get("_norm_material") == target
    ]


def _select_group_rows(
    records: Sequence[dict[str, Any]],
    operation: str,
    group: str | None,
) -> list[dict[str, Any]]:
    if not group:
        return []
    target = group.strip().upper()
    return [
        row
        for row in _select_op_rows(records, operation)
        if row.get("_iso_group") == target
    ]


def pick_speeds_row(
    material_label: str | None,
    operation: str,
    tool_diameter_in: float | None = None,
    *,
    table: Any | None = None,
    tool_description: str | None = None,
) -> dict[str, Any] | None:
    """Select a speeds/feeds row based on material and operation."""

    records = _coerce_records(table)
    mm = normalize_material(material_label)
    canonical = (mm.get("canonical_material") if mm else None) or (material_label or "")
    iso = (mm.get("iso_group") if mm else "") or ""

    try:
        tool_dia = float(tool_diameter_in) if tool_diameter_in is not None else None
    except Exception:
        tool_dia = None
    if tool_dia is not None and (not math.isfinite(tool_dia) or tool_dia <= 0):
        tool_dia = None

    def _norm_key_map(row: Mapping[str, Any]) -> dict[str, str]:
        return {
            str(key).strip().lower().replace("-", "_").replace(" ", "_"): key
            for key in row.keys()
        }

    def _to_float(value: Any) -> float | None:
        if value in {None, "", "-"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _row_value(row: Mapping[str, Any], norm_map: dict[str, str], names: Sequence[str], *, scale: float = 1.0) -> float | None:
        for name in names:
            key = norm_map.get(name)
            if key is None:
                continue
            val = _to_float(row.get(key))
            if val is not None:
                return val * scale
        return None

    def _diameter_score(row: Mapping[str, Any]) -> float:
        if tool_dia is None:
            return 0.0
        norm_map = _norm_key_map(row)
        target = tool_dia
        score = 0.0

        exact_in = _row_value(
            row,
            norm_map,
            ("diameter_in", "drill_diameter_in", "tool_diameter_in", "dia_in", "diameter"),
        )
        if exact_in is None:
            exact_in = _row_value(
                row,
                norm_map,
                ("diameter_mm", "drill_diameter_mm", "tool_diameter_mm", "dia_mm"),
                scale=1.0 / 25.4,
            )
        if exact_in is not None:
            diff = abs(exact_in - target)
            tol = max(0.01, 0.05 * max(target, 1e-6))
            if diff <= 1e-6:
                score = max(score, 6.0)
            elif diff <= tol:
                score = max(score, 5.0 - (diff / tol))
            else:
                score = max(score, max(0.5, 3.0 - diff))

        lo_in = _row_value(
            row,
            norm_map,
            ("dia_min_in", "diameter_min_in", "tool_dia_min_in", "diameter_range_min_in"),
        )
        if lo_in is None:
            lo_in = _row_value(
                row,
                norm_map,
                ("dia_min_mm", "diameter_min_mm", "tool_dia_min_mm", "diameter_range_min_mm"),
                scale=1.0 / 25.4,
            )
        hi_in = _row_value(
            row,
            norm_map,
            ("dia_max_in", "diameter_max_in", "tool_dia_max_in", "diameter_range_max_in"),
        )
        if hi_in is None:
            hi_in = _row_value(
                row,
                norm_map,
                ("dia_max_mm", "diameter_max_mm", "tool_dia_max_mm", "diameter_range_max_mm"),
                scale=1.0 / 25.4,
            )
        if lo_in is not None or hi_in is not None:
            low = lo_in if lo_in is not None else -math.inf
            high = hi_in if hi_in is not None else math.inf
            if low <= target <= high:
                span = high - low if math.isfinite(low) and math.isfinite(high) else None
                range_score = 3.0
                if span is not None and span > 0:
                    range_score += max(0.0, 1.5 - min(span / max(target, 1e-6), 1.5))
                score = max(score, range_score)
            else:
                score = max(score, 0.1)

        return score

    def _order_by_diameter(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        if tool_dia is None:
            return list(rows)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            if isinstance(row, Mapping):
                scored.append((_diameter_score(row), idx, row))
            else:
                scored.append((0.0, idx, row))
        best_score = max(scored, key=lambda item: item[0])[0]
        if best_score <= 0:
            return [row for _, _, row in sorted(scored, key=lambda item: item[1])]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, _, row in scored]

    restricted_material = False
    canonical_lower = canonical.strip().lower()
    if canonical_lower:
        if any(term in canonical_lower for term in ("carbide", "ceramic")):
            if tool_description is None or "diamond" not in tool_description.lower():
                restricted_material = True

    selected: dict[str, Any] | None = None
    exact = _select_material_rows(records, operation, canonical)
    if exact:
        ordered = _order_by_diameter(exact)
        if ordered:
            selected = ordered[0]
    else:
        group_rows = _select_group_rows(records, operation, iso)
        if group_rows:
            ordered = _order_by_diameter(group_rows)
            if ordered:
                selected = ordered[0]
    if selected is None:
        op_rows = _select_op_rows(records, operation)
        if op_rows:
            ordered = _order_by_diameter(op_rows)
            selected = ordered[0]
            log.warning(
                "Using generic speeds/feeds row", extra={"operation": operation, "material": canonical}
            )

    if selected is not None and restricted_material and not _norm_operation(operation).startswith("wire_edm"):
        generic_rows = _select_op_rows(records, operation)
        if generic_rows:
            selected = generic_rows[0]
        log.warning(
            "Restricted material fallback for non-EDM operation",
            extra={"operation": operation, "material": canonical},
        )

    log_data = {
        "op": operation,
        "material_input": material_label,
        "canonical": canonical,
        "iso_group": iso,
        "row_material": selected.get("material") if selected else None,
        "row_group": selected.get("material_group") if selected else None,
        "sfm": selected.get("sfm_start") if selected else None,
        "feed_type": selected.get("feed_type") if selected else None,
        "fz_ipr_0.25in": selected.get("fz_ipr_0_25in") if selected else None,
    }
    log.info(log_data)
    return selected


def unit_hp_cap(material_label: str | None) -> float | None:
    mm = normalize_material(material_label)
    if not mm:
        return None
    value = mm.get("unit_hp_in3_per_hp")
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None

