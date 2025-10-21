import pytest

from appV5 import overrides_to_suggestions, suggestions_to_overrides


def test_overrides_to_suggestions_cleans_and_canonicalises() -> None:
    overrides = {
        "process_hour_multipliers": {"milling": "2.0", "turning": None},
        "process_hour_adders": {"inspection": "1.5", "invalid": ""},
        "add_pass_through": [
            {"label": "Hardware / BOM", "value": "12.5"},
            ("Finish", 7.2),
            {"label": "Hardware", "value": None},
        ],
        "scrap_pct_override": "0.15",
        "setups": "3",
        "fixture": "  Soft jaws  ",
        "notes": [" Use custom fixture ", ""],
        "operation_sequence": ["Mill", " Deburr  "],
        "dfm_risks": ["Thin walls", None],
        "drilling_strategy": {"multiplier": "1.2", "per_hole_floor_sec": "5"},
        "drilling_groups": [
            {"qty": "8", "dia_mm": "3.2"},
            {"qty": None, "dia_mm": "bad"},
        ],
        "stock_recommendation": {"stock_item": "Plate", "length_mm": "100"},
        "setup_recommendation": {"setups": "2"},
        "packaging_flat_cost": "12",
        "fai_required": "yes",
        "shipping_hint": " Foam inserts ",
    }

    suggestions = overrides_to_suggestions(overrides)

    assert suggestions["process_hour_multipliers"] == {"milling": 2.0}
    assert suggestions["process_hour_adders"] == {"inspection": 1.5}
    assert suggestions["add_pass_through"] == {"Hardware": 12.5, "Finish": 7.2}
    assert suggestions["scrap_pct"] == pytest.approx(0.15)
    assert suggestions["setups"] == 3
    assert suggestions["fixture"] == "Soft jaws"
    assert suggestions["notes"] == ["Use custom fixture"]
    assert suggestions["operation_sequence"] == ["Mill", "Deburr"]
    assert suggestions["dfm_risks"] == ["Thin walls"]
    assert suggestions["drilling_strategy"]["multiplier"] == pytest.approx(1.2)
    assert suggestions["drilling_strategy"]["per_hole_floor_sec"] == pytest.approx(5.0)
    assert suggestions["drilling_groups"][0]["qty"] == 8
    assert suggestions["drilling_groups"][0]["dia_mm"] == pytest.approx(3.2)
    assert suggestions["stock_recommendation"]["stock_item"] == "Plate"
    assert suggestions["stock_recommendation"]["length_mm"] == pytest.approx(100.0)
    assert suggestions["setup_recommendation"]["setups"] == 2
    assert suggestions["packaging_flat_cost"] == pytest.approx(12.0)
    assert suggestions["fai_required"] is True
    assert suggestions["shipping_hint"] == "Foam inserts"


def test_suggestions_to_overrides_filters_metadata_and_normalises() -> None:
    suggestions = {
        "process_hour_multipliers": {"milling": "1.5"},
        "process_hour_adders": {"inspection": "0.5"},
        "add_pass_through": [{"label": "Hardware", "amount": "18"}],
        "scrap_pct": 0.12,
        "setups": 2.7,
        "fixture": " Vise ",
        "notes": ["Keep tabs", ""],
        "dfm_risks": ["Thin walls"],
        "drilling_strategy": {"multiplier": 1.3, "note": "Peck"},
        "_meta": {"foo": "bar"},
        "no_change_reason": "n/a",
    }

    overrides = suggestions_to_overrides(suggestions)

    assert overrides["process_hour_multipliers"] == {"milling": 1.5}
    assert overrides["process_hour_adders"] == {"inspection": 0.5}
    assert overrides["add_pass_through"] == {"Hardware": 18.0}
    assert overrides["scrap_pct"] == pytest.approx(0.12)
    assert overrides["setups"] == 3
    assert overrides["fixture"] == "Vise"
    assert overrides["notes"] == ["Keep tabs"]
    assert overrides["dfm_risks"] == ["Thin walls"]
    assert overrides["drilling_strategy"]["multiplier"] == pytest.approx(1.3)
    assert overrides["drilling_strategy"]["note"] == "Peck"
    assert "_meta" not in overrides
    assert "no_change_reason" not in overrides


def test_overrides_to_suggestions_honours_bounds_when_provided() -> None:
    overrides = {
        "process_hour_multipliers": {"milling": "9.5"},
        "process_hour_adders": {"inspection": 12.0},
        "scrap_pct": 0.6,
    }

    bounds = {
        "mult_min": 0.8,
        "mult_max": 1.2,
        "adder_min_hr": 0.25,
        "adder_max_hr": 4.0,
        "scrap_min": 0.05,
        "scrap_max": 0.2,
    }

    suggestions = overrides_to_suggestions(overrides, bounds=bounds)

    assert suggestions["process_hour_multipliers"]["milling"] == pytest.approx(1.2)
    assert suggestions["process_hour_adders"]["inspection"] == pytest.approx(4.0)
    assert suggestions["scrap_pct"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("token", "expected"),
    [("t", True), ("on", True), ("f", False), ("off", False)],
)
def test_overrides_to_suggestions_handles_bool_tokens(token: str, expected: bool) -> None:
    suggestions = overrides_to_suggestions({"fai_required": token})
    assert suggestions.get("fai_required") is expected


@pytest.mark.parametrize(("token", "expected"), [("t", True), ("f", False)])
def test_suggestions_to_overrides_handles_bool_tokens(token: str, expected: bool) -> None:
    overrides = suggestions_to_overrides({"fai_required": token})
    assert overrides.get("fai_required") is expected
