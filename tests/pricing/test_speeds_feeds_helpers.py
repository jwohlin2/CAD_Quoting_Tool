from __future__ import annotations

from typing import Any

import pytest

from cad_quoter.speeds_feeds import (
    coerce_table_to_records,
    material_label_from_records,
    normalize_operation,
    normalize_records,
    select_group_rows,
    select_operation_rows,
)


@pytest.fixture()
def sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "operation": "Drill",
            "material": "6061 Aluminum",
            "material_group": "N1",
        },
        {
            "operation": "deep drilling",
            "material": "Stainless Steel 303",
            "material_group": "M2",
        },
    ]


def test_normalize_records_adds_helper_fields(sample_rows: list[dict[str, Any]]) -> None:
    normalized = normalize_records(sample_rows)
    assert normalized
    first = normalized[0]
    assert first["_norm_operation"] == "drill"
    assert first["_norm_material"] == "6061 aluminum"
    assert first["_iso_group"] == "N1"
    assert first["_iso_group_simple"] == "N"
    assert first["_norm_material_key"]


def test_coerce_table_to_records_handles_sequence(sample_rows: list[dict[str, Any]]) -> None:
    records = coerce_table_to_records(sample_rows)
    assert len(records) == len(sample_rows)
    assert {row["_norm_operation"] for row in records} == {"drill", "deep_drilling"}


def test_select_operation_rows_supports_aliases(sample_rows: list[dict[str, Any]]) -> None:
    records = normalize_records(sample_rows)
    matches = select_operation_rows(records, "deep_drill", aliases={"deep_drill": ("deep drilling",)})
    assert len(matches) == 1
    assert matches[0]["material_group"] == "M2"


def test_material_label_from_records_prefers_exact_key(sample_rows: list[dict[str, Any]]) -> None:
    records = normalize_records(sample_rows)
    label = material_label_from_records(
        records,
        normalized_lookup="stainless_steel_303",
        material_group="M2",
    )
    assert label == "Stainless Steel 303"


def test_select_group_rows_matches_group_code(sample_rows: list[dict[str, Any]]) -> None:
    records = normalize_records(sample_rows)
    matches = select_group_rows(records, "Drill", "N2")
    # N2 simplifies to N which matches the first record.
    assert len(matches) == 1
    assert matches[0]["material"] == "6061 Aluminum"


def test_normalize_operation_collapses_variants() -> None:
    assert normalize_operation("Deep Drill") == "deep_drill"
