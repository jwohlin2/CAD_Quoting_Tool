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


def test_scrap_hole_estimate_increases_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    df = pd.DataFrame(_base_rows())
    geo = {
        "derived": {
            "hole_diams_mm": [100.0, 100.0, 100.0, 100.0, 100.0],
            "bbox_mm": (100.0, 100.0),
        }
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)

    baseline = result["decision_state"]["baseline"]
    assert baseline["scrap_pct"] == pytest.approx(0.25)
    assert baseline.get("scrap_source_label") == "default_guess+holes"

    material = result["breakdown"]["material"]
    assert material["scrap_pct"] == pytest.approx(0.25)
    assert material.get("scrap_source") == "default_guess"
    assert material.get("scrap_source_label") == "default_guess+holes"


def test_scrap_credit_uses_wieland_price(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    monkeypatch.setattr(appV5, "_wieland_scrap_usd_per_lb", lambda _fam: 2.0)

    rows = _base_rows() + [
        {"Item": "Plate Length (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Plate Width (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Thickness (in)", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
    ]
    df = pd.DataFrame(rows)
    result = appV5.compute_quote_from_df(df, llm_enabled=False)

    material = result["breakdown"]["material"]
    scrap_credit = material.get("material_scrap_credit")
    assert scrap_credit is not None and scrap_credit > 0

    scrap_mass_g = material.get("scrap_mass_g")
    assert scrap_mass_g is not None and scrap_mass_g > 0
    scrap_mass_lb = float(scrap_mass_g) / 1000.0 * appV5.LB_PER_KG
    material_block = result["breakdown"]["material_block"]
    base_cost = float(material_block.get("material_cost_before_credit") or 0.0)
    expected_credit = round(
        min(base_cost, scrap_mass_lb * 2.0 * appV5.SCRAP_RECOVERY_DEFAULT),
        2,
    )
    assert scrap_credit == pytest.approx(expected_credit)

    net_material_cost = float(material_block.get("total_material_cost") or 0.0)
    assert base_cost > 0
    assert net_material_cost == pytest.approx(base_cost - scrap_credit)

    if material.get("material_cost") is not None:
        assert material.get("material_cost") == pytest.approx(net_material_cost)

    assert material.get("scrap_credit_unit_price_usd_per_lb") == pytest.approx(2.0)
    assert material.get("scrap_credit_recovery_pct") == pytest.approx(
        appV5.SCRAP_RECOVERY_DEFAULT
    )


def test_scrap_credit_falls_back_to_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    monkeypatch.setattr(appV5, "_wieland_scrap_usd_per_lb", lambda _fam: None)

    rows = _base_rows() + [
        {"Item": "Plate Length (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Plate Width (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Thickness (in)", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
    ]
    df = pd.DataFrame(rows)
    state = appV5.QuoteState()
    state.user_overrides = {
        "scrap_credit_unit_price_usd_per_lb": 1.25,
        "scrap_recovery_pct": 90.0,
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, quote_state=state)

    material = result["breakdown"]["material"]
    scrap_credit = float(material.get("material_scrap_credit") or 0.0)
    assert scrap_credit > 0

    scrap_mass_g = float(material.get("scrap_mass_g") or 0.0)
    assert scrap_mass_g > 0
    scrap_mass_lb = scrap_mass_g / 1000.0 * appV5.LB_PER_KG
    material_block = result["breakdown"]["material_block"]
    base_cost = float(material_block.get("material_cost_before_credit") or 0.0)
    expected_credit = round(min(base_cost, scrap_mass_lb * 1.25 * 0.90), 2)
    assert scrap_credit == pytest.approx(expected_credit)

    net_cost = float(material_block.get("total_material_cost") or 0.0)
    assert net_cost == pytest.approx(base_cost - scrap_credit)

    assert material.get("scrap_credit_unit_price_usd_per_lb") == pytest.approx(1.25)
    assert material.get("scrap_credit_recovery_pct") == pytest.approx(0.90)
    assert material.get("scrap_credit_source") == "override_unit_price"
