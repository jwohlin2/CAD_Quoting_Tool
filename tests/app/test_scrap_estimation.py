import math

import appV5
import pytest


pd = pytest.importorskip("pandas")


def _stub_payload(*_args, **_kwargs):
    return {}


def _base_rows() -> list[dict[str, object]]:
    return [
        {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
        {
            "Item": "Material Name",
            "Example Values / Options": "6061-T6 Aluminum",
            "Data Type / Input Method": "text",
        },
        {"Item": "Net Volume (cm^3)", "Example Values / Options": 80.0, "Data Type / Input Method": "number"},
        {"Item": "Material Density", "Example Values / Options": 2.7, "Data Type / Input Method": "number"},
    ]


def test_scrap_defaults_to_guess_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    df = pd.DataFrame(_base_rows())

    result = appV5.compute_quote_from_df(df, llm_enabled=False)

    baseline = result["decision_state"]["baseline"]
    assert math.isclose(baseline["scrap_pct"], appV5.SCRAP_DEFAULT_GUESS, rel_tol=1e-9)

    material = result["breakdown"]["material"]
    assert math.isclose(material["scrap_pct"], appV5.SCRAP_DEFAULT_GUESS, rel_tol=1e-9)
    assert material.get("scrap_source") == "default_guess"


def test_scrap_uses_stock_plan_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    df = pd.DataFrame(_base_rows())
    geo = {
        "stock_plan_guess": {
            "net_volume_in3": 80.0,
            "stock_volume_in3": 96.0,
        }
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)

    expected_scrap = (96.0 - 80.0) / 80.0
    baseline = result["decision_state"]["baseline"]
    assert baseline["scrap_pct"] == pytest.approx(expected_scrap)

    material = result["breakdown"]["material"]
    assert material["scrap_pct"] == pytest.approx(expected_scrap)
    assert material.get("scrap_source") == "stock_plan_guess"


def test_scrap_respects_ui_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    rows = _base_rows() + [
        {"Item": "Scrap Percent (%)", "Example Values / Options": 12.0, "Data Type / Input Method": "number"},
    ]
    df = pd.DataFrame(rows)

    result = appV5.compute_quote_from_df(df, llm_enabled=False)

    expected_scrap = 0.12
    baseline = result["decision_state"]["baseline"]
    assert baseline["scrap_pct"] == pytest.approx(expected_scrap)

    material = result["breakdown"]["material"]
    assert material["scrap_pct"] == pytest.approx(expected_scrap)
    assert material.get("scrap_source") == "ui"
