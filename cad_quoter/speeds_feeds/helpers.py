"""Shared helpers for normalizing and selecting speeds/feeds data sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Any, Mapping as MappingType

from cad_quoter.domain_models import normalize_material_key as _normalize_material_key

__all__ = [
    "coerce_table_to_records",
    "material_label_from_records",
    "normalize_material_group_code",
    "normalize_operation",
    "normalize_records",
    "select_group_rows",
    "select_material_rows",
    "select_operation_rows",
]


def normalize_operation(operation: str | None) -> str:
    """Return a canonical identifier for an operation label."""

    return (
        str(operation or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_material_group_code(value: Any) -> str:
    """Return a canonical material group code (e.g., ``N1`` → ``N``)."""

    text = "" if value is None else str(value).strip().upper()
    if not text:
        return ""
    simplified = re.sub(r"[^A-Z0-9]+", "", text)
    if re.fullmatch(r"[A-Z]\d+", simplified or ""):
        return simplified[0]
    return simplified or text


def _clean_record(row: Mapping[str, Any]) -> dict[str, Any]:
    clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in dict(row).items()}
    op = normalize_operation(clean.get("operation"))
    if not op:
        return {}
    clean["_norm_operation"] = op

    material_value: str = str(clean.get("material", "") or "").strip()
    clean["_norm_material"] = material_value.lower()
    norm_key = _normalize_material_key(material_value) if material_value else ""
    clean["_norm_material_key"] = norm_key

    group_value = str(clean.get("material_group") or clean.get("iso_group") or "").strip()
    group_value = group_value.upper()
    clean["_iso_group"] = group_value
    clean["_iso_group_simple"] = normalize_material_group_code(group_value)

    return clean


def normalize_records(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return ``rows`` with canonical helper fields attached."""

    normalised: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        clean = _clean_record(row)
        if not clean:
            continue
        normalised.append(clean)
    return tuple(normalised)


def _iter_candidate_rows(table: Any) -> Iterable[Mapping[str, Any]]:
    if hasattr(table, "to_dict"):
        try:
            records = table.to_dict("records")  # type: ignore[attr-defined]
        except Exception:
            records = None
        if isinstance(records, list):
            for row in records:
                if isinstance(row, Mapping):
                    yield row
            return

    rows = getattr(table, "_rows", None)
    if isinstance(rows, list) and rows:
        for row in rows:
            if isinstance(row, Mapping):
                yield row
        return

    if isinstance(table, MappingType):
        yield table  # type: ignore[misc]
        return

    if isinstance(table, Sequence) and not isinstance(table, (str, bytes, bytearray)):
        for row in table:
            if isinstance(row, Mapping):
                yield row


def coerce_table_to_records(table: Any | None) -> tuple[dict[str, Any], ...]:
    """Return ``table`` normalised into a tuple of mapping records."""

    if table is None:
        return tuple()

    try:
        if getattr(table, "empty"):
            return tuple()
    except Exception:
        try:
            if len(table) == 0:  # type: ignore[arg-type]
                return tuple()
        except Exception:
            pass

    return normalize_records(_iter_candidate_rows(table))


def select_operation_rows(
    records: Sequence[Mapping[str, Any]],
    operation: str,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return rows matching ``operation`` with optional alias support."""

    target = normalize_operation(operation)
    if not target:
        return []

    variants: set[str] = {target}
    trimmed = target.rstrip("_")
    if trimmed:
        variants.add(trimmed)
    if target.endswith("ing") and len(target) > 3:
        variants.add(target[:-3])
    variants.add(target.replace("_", " "))

    if aliases:
        for key, options in aliases.items():
            norm_key = normalize_operation(key)
            if norm_key in variants:
                for option in options:
                    alias_norm = normalize_operation(option)
                    if alias_norm:
                        variants.add(alias_norm)

    matched = [row for row in records if row.get("_norm_operation") in variants]
    if matched:
        return matched

    prefix_matches = [
        row
        for row in records
        if isinstance(row, Mapping)
        and row.get("_norm_operation")
        and any(str(row.get("_norm_operation")).startswith(prefix) for prefix in variants if prefix)
    ]
    return prefix_matches


def select_group_rows(
    records: Sequence[Mapping[str, Any]],
    operation: str,
    group: str | None,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    if not group:
        return []
    target = str(group).strip().upper()
    target_simple = normalize_material_group_code(group)
    results: list[dict[str, Any]] = []
    for row in select_operation_rows(records, operation, aliases=aliases):
        row_group = str(row.get("_iso_group") or "").strip().upper()
        row_simple = normalize_material_group_code(row_group)
        if row_group == target or (
            target_simple and row_simple and row_simple == target_simple
        ):
            results.append(dict(row))
    return results


def select_material_rows(
    records: Sequence[Mapping[str, Any]],
    operation: str,
    material: str | None,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    if not material:
        return []
    lookup = str(material).strip().lower()
    lookup_key = _normalize_material_key(material) if material else ""
    results: list[dict[str, Any]] = []
    for row in select_operation_rows(records, operation, aliases=aliases):
        row_material = str(row.get("_norm_material") or "")
        row_key = str(row.get("_norm_material_key") or "")
        if (lookup and row_material == lookup) or (lookup_key and row_key == lookup_key):
            results.append(dict(row))
    return results


def material_label_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    normalized_lookup: str | None = None,
    material_group: str | None = None,
) -> str | None:
    """Return a canonical material label given normalized metadata."""

    lookup = str(normalized_lookup or "").strip().lower()
    group_simple = normalize_material_group_code(material_group)

    for row in records:
        label = (
            str(
                row.get("material")
                or row.get("material_name")
                or row.get("canonical_material")
                or ""
            )
            .strip()
        )
        if not label:
            continue
        row_key = str(row.get("_norm_material_key") or "")
        if lookup and row_key == lookup:
            return label
        row_group_simple = normalize_material_group_code(row.get("_iso_group"))
        if group_simple and row_group_simple == group_simple:
            return label
    return None
