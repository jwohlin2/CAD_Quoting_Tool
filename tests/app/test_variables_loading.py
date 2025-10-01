from pathlib import Path

import pandas as pd

from appV5 import CORE_COLS, read_variables_file


def test_master_variables_core_columns_populated():
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "Master_Variables.csv"

    core_df = read_variables_file(str(csv_path))

    assert core_df is not None

    required_columns = ["Item", "Data Type / Input Method", "Example Values / Options"]
    assert all(col in CORE_COLS for col in required_columns)

    items_to_check = {
        "Material Scrap / Remnant Value",
        "Masking Labor for Plating",
        "Final Inspection Labor (Manual)",
    }

    matched_rows = {item: None for item in items_to_check}

    for _, row in core_df.iterrows():
        item = row.get("Item")
        if item in matched_rows and matched_rows[item] is None:
            matched_rows[item] = row

    missing = [item for item, row in matched_rows.items() if row is None]
    assert not missing, f"Missing expected items: {missing}"

    for item, row in matched_rows.items():
        assert row is not None  # for type checkers
        for column in required_columns:
            value = row.get(column)
            assert pd.notna(value), f"{column} should not be NA for {item}"
            assert str(value).strip() != "", f"{column} should not be blank for {item}"
