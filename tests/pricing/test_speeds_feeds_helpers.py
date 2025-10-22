from __future__ import annotations

from typing import Any

import pytest

import pandas as pd

from cad_quoter.speeds_feeds import (
    coerce_table_to_records,
    ipm_from_rpm_ipt,
    lookup_mill_params,
    material_label_from_records,
    normalize_operation,
    normalize_records,
    rpm_from_sfm,
    select_group_rows,
    select_operation_rows,
)
from tests.data_loaders import load_speeds_feeds_samples


@pytest.fixture()
def sample_rows() -> list[dict[str, Any]]:
    return [dict(row) for row in load_speeds_feeds_samples()]


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


def test_lookup_mill_params_prefers_matches_and_nearest_diameter() -> None:
    table = pd.DataFrame(
        [
            {
                "material_group": "N",
                "operation": "rough mill",
                "tool_diam_in": 0.25,
                "sfm": 500,
                "ipt": 0.001,
                "flutes": 3,
                "max_ap_in": 0.05,
                "max_ae_in": 0.125,
                "stepover_pct": 0.4,
                "index_min_per_pass": 0.01,
                "toolchange_min": 1.5,
            },
            {
                "material_group": "N",
                "operation": "rough mill",
                "tool_diam_in": 0.5,
                "sfm": 600,
                "ipt": 0.002,
                "flutes": 4,
                "max_ap_in": 0.1,
                "max_ae_in": 0.25,
                "stepover_pct": 0.5,
                "index_min_per_pass": 0.015,
                "toolchange_min": 2.0,
            },
        ]
    )

    params = lookup_mill_params(table, "n", "Rough Mill", 0.45)

    assert params["sfm"] == 600
    assert params["flutes"] == 4
    assert params["max_ap_in"] == 0.1


def test_lookup_mill_params_uses_defaults_when_missing() -> None:
    table = pd.DataFrame(
        [
            {
                "operation": "finish mill",
                "tool_diam_in": 0.375,
                "sfm": "",  # missing value triggers defaults
                "flutes": None,
            }
        ]
    )

    params = lookup_mill_params(table, "unknown", "Finish Mill", 0.375)

    assert params["sfm"] == 800.0
    assert params["flutes"] == 3
    # default max_ae_in should be derived from tool diameter
    assert params["max_ae_in"] == pytest.approx(0.1875)


def test_rpm_and_ipm_helpers() -> None:
    rpm = rpm_from_sfm(500, 0.5)
    assert rpm == pytest.approx((500 * 12.0) / (3.141592653589793 * 0.5))

    ipm = ipm_from_rpm_ipt(rpm, 4, 0.002)
    assert ipm == pytest.approx(rpm * 4 * 0.002)
