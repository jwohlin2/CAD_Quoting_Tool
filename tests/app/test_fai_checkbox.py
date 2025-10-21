"""Regression tests for FAIR checkbox parsing."""

import appV5
import pytest


pd = pytest.importorskip("pandas")


@pytest.mark.parametrize(
    ("token", "expected"),
    [("True", True), ("t", True), ("F", False)],
)
def test_fai_checkbox_tokens_set_baseline_and_features(
    monkeypatch, token: str, expected: bool
) -> None:
    captured: dict[str, dict] = {}

    def _capture_payload(geo, baseline, rates, bounds):
        captured["geo"] = geo
        return {}

    monkeypatch.setattr(appV5, "build_suggest_payload", _capture_payload)

    df = pd.DataFrame(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {
                "Item": "FAIR Required",
                "Example Values / Options": token,
                "Data Type / Input Method": "checkbox",
            },
            {"Item": "Material Name", "Example Values / Options": "6061-T6 Aluminum", "Data Type / Input Method": "text"},
            {"Item": "Net Volume (cm^3)", "Example Values / Options": 50, "Data Type / Input Method": "number"},
            {"Item": "Thickness (in)", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
        ]
    )

    result = appV5.compute_quote_from_df(df, llm_enabled=False)

    baseline = result["decision_state"]["baseline"]
    assert baseline.get("fai_required") is expected
    assert isinstance(baseline.get("fai_required"), bool)

    derived = captured["geo"]["derived"]
    assert derived.get("fai_required") is expected
    assert isinstance(derived.get("fai_required"), bool)
