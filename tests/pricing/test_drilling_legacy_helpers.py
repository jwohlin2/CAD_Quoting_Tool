from __future__ import annotations

from typing import Any

from cad_quoter.domain_models import normalize_material_key
from cad_quoter.estimators import drilling


def _make_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def test_material_label_from_table_uses_shared_lookup() -> None:
    table = _make_table(
        [
            {
                "operation": "Drill",
                "material": "6061 Aluminum",
                "material_group": "N1",
            }
        ]
    )
    normalized = normalize_material_key("6061 Aluminum")
    label = drilling._material_label_from_table(table, "N1", normalized)
    assert label == "6061 Aluminum"


def test_select_speeds_feeds_row_respects_aliases_and_group() -> None:
    table = _make_table(
        [
            {
                "operation": "deep drilling",
                "material": "Stainless Steel 303",
                "material_group": "M2",
            },
            {
                "operation": "Drill",
                "material": "Stainless Steel 303",
                "material_group": "M2",
            },
        ]
    )
    row = drilling._select_speeds_feeds_row(
        table,
        operation="deep_drill",
        material_key="stainless_steel_303",
        material_group="M2",
    )
    assert row is not None
    assert row.get("operation") == "deep drilling"
