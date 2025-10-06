"""Tests for the decision tree quoting engine."""

from __future__ import annotations

import pandas as pd

from cad_quoter.config import load_default_params, load_default_rates
from cad_quoter.decision_tree import generate_decision_tree_quote


def _make_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Item", "Example Values / Options", "Data Type / Input Method"])


def test_generate_decision_tree_quote_returns_price_for_thin_plate() -> None:
    df = _make_df(
        [
            {"Item": "Qty", "Example Values / Options": 2, "Data Type / Input Method": "number"},
            {
                "Item": "Material Name",
                "Example Values / Options": "6061-T6 Aluminum",
                "Data Type / Input Method": "text",
            },
            {"Item": "Net Volume (cm^3)", "Example Values / Options": 80.0, "Data Type / Input Method": "number"},
            {"Item": "Thickness (in)", "Example Values / Options": 0.5, "Data Type / Input Method": "number"},
            {
                "Item": "Profile Perimeter (mm)",
                "Example Values / Options": 400.0,
                "Data Type / Input Method": "number",
            },
        ]
    )

    params = load_default_params()
    rates = load_default_rates()

    result = generate_decision_tree_quote(df, params, rates)

    assert result is not None
    assert result["scenario"] == "thin_plate"
    assert "cnc_rough_mill" in result["process_costs"]
    totals = result["totals"]
    assert totals["price_total"] > 0
    assert totals["unit_price"] == totals["price_total"] / df.loc[0, "Example Values / Options"]


def test_generate_decision_tree_quote_handles_heavy_plate() -> None:
    df = _make_df(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {"Item": "Material", "Example Values / Options": "Steel", "Data Type / Input Method": "text"},
            {"Item": "Net Volume (cm^3)", "Example Values / Options": 2400.0, "Data Type / Input Method": "number"},
            {"Item": "Thickness (in)", "Example Values / Options": 2.5, "Data Type / Input Method": "number"},
        ]
    )

    params = load_default_params()
    rates = load_default_rates()

    result = generate_decision_tree_quote(df, params, rates)

    assert result is not None
    assert result["scenario"] == "heavy_plate"
    assert "blanchard_grind_pre" in result["process_costs"]
    assert result["totals"]["price_total"] > 0

