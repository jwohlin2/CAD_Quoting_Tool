"""Lookup helpers for material-aware speeds and feeds selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import logging
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
    *,
    table: Any | None = None,
    tool_description: str | None = None,
) -> dict[str, Any] | None:
    """Select a speeds/feeds row based on material and operation."""

    records = _coerce_records(table)
    mm = normalize_material(material_label)
    canonical = (mm.get("canonical_material") if mm else None) or (material_label or "")
    iso = (mm.get("iso_group") if mm else "") or ""

    restricted_material = False
    canonical_lower = canonical.strip().lower()
    if canonical_lower:
        if any(term in canonical_lower for term in ("carbide", "ceramic")):
            if tool_description is None or "diamond" not in tool_description.lower():
                restricted_material = True

    selected: dict[str, Any] | None = None
    exact = _select_material_rows(records, operation, canonical)
    if exact:
        selected = exact[0]
    else:
        group_rows = _select_group_rows(records, operation, iso)
        if group_rows:
            selected = group_rows[0]
    if selected is None:
        op_rows = _select_op_rows(records, operation)
        if op_rows:
            selected = op_rows[0]
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

