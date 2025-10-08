from typing import Iterable, Tuple, Union

import pandas as pd
import pytest

import appV5


def _df(rows: Iterable[Tuple[str, Union[float, int, str], str]]) -> pd.DataFrame:
    normalized = [
        {
            "Item": item,
            "Example Values / Options": value,
            "Data Type / Input Method": dtype,
        }
        for item, value, dtype in rows
    ]
    return pd.DataFrame(normalized, columns=["Item", "Example Values / Options", "Data Type / Input Method"])


def test_programming_hours_override_caps_total_hours() -> None:
    df = _df(
        [
            ("Qty", 1, "number"),
            ("Material", "6061", "text"),
            ("Stock Thickness_mm", 25.4, "number"),
            ("Prog Max To Milling Ratio", 100.0, "number"),
            ("Programming Hours", 11.75, "number"),
            ("Programming Override Hr", 1.0, "number"),
        ]
    )

    result = appV5.compute_quote_from_df(
        df,
        llm_enabled=False,
        geo={"thickness_mm": 25.4, "material": "6061"},
    )

    prog_detail = result["breakdown"]["nre_detail"]["programming"]

    assert prog_detail["prog_hr"] == pytest.approx(1.0)
    assert prog_detail["override_applied"] is True
    assert prog_detail["auto_prog_hr"] == pytest.approx(11.75)


def test_programming_hours_without_override_unmodified() -> None:
    df = _df(
        [
            ("Qty", 1, "number"),
            ("Material", "6061", "text"),
            ("Stock Thickness_mm", 25.4, "number"),
            ("Prog Max To Milling Ratio", 100.0, "number"),
            ("Programming Hours", 11.75, "number"),
        ]
    )

    result = appV5.compute_quote_from_df(
        df,
        llm_enabled=False,
        geo={"thickness_mm": 25.4, "material": "6061"},
    )

    prog_detail = result["breakdown"]["nre_detail"]["programming"]

    assert prog_detail["prog_hr"] == pytest.approx(11.75)
    assert "override_applied" not in prog_detail
    assert "auto_prog_hr" not in prog_detail
