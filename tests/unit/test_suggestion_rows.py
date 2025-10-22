import copy

from cad_quoter.ui.suggestions import build_suggestion_rows, iter_suggestion_rows
from cad_quoter.domain import QuoteState


def _clone_state(state: QuoteState) -> QuoteState:
    data = state.to_dict()
    return QuoteState.from_dict(copy.deepcopy(data))


def test_suggestion_rows_align_between_ui_and_legacy():
    state = QuoteState(
        baseline={
            "process_hours": {"mill": "1.5", "turn": 0},
            "pass_through": {"Hardware / BOM": 120.5},
            "scrap_pct": 5,
            "setups": None,
            "fixture": "Baseline fixture",
        },
        suggestions={
            "process_hour_multipliers": {"mill": 1.2},
            "process_hour_adders": {"mill": 0.5, "turn": 0.0},
            "add_pass_through": {"Hardware": 30, "Shipping": {"amount": 12}},
            "scrap_pct": 6,
            "setups": 2,
            "fixture": "LLM fixture",
        },
        user_overrides={
            "process_hour_multipliers": {"mill": 1.1},
            "process_hour_adders": {"turn": 0.25},
            "add_pass_through": [{"label": "hardware", "amount": 3}],
            "scrap_pct": 5.5,
        },
        effective={
            "process_hour_multipliers": {"mill": 1.15},
            "process_hour_adders": {"mill": 0.4},
            "add_pass_through": {"Hardware": 50, "Shipping": 15},
            "scrap_pct": 5.25,
            "fixture": "Effective fixture",
        },
        effective_sources={
            "process_hour_multipliers": {"mill": "user"},
            "process_hour_adders": {"mill": "llm"},
            "add_pass_through": {"Hardware": "user", "Shipping": "baseline"},
            "scrap_pct": "user",
            "fixture": "llm",
        },
        accept_llm={
            "process_hour_multipliers": {"mill": True},
            "process_hour_adders": {"turn": False},
            "add_pass_through": {"Hardware": True, "Shipping": False},
            "scrap_pct": True,
            "fixture": False,
        },
    )

    # exercise both implementations with separate state instances to avoid shared mutation
    rows_ui = build_suggestion_rows(_clone_state(state))
    rows_legacy = iter_suggestion_rows(_clone_state(state))

    assert rows_ui == rows_legacy


def test_suggestion_rows_empty_state_round_trip():
    state = QuoteState()
    state.accept_llm = True  # legacy UI can persist booleans here

    rows_ui = build_suggestion_rows(_clone_state(state))
    rows_legacy = iter_suggestion_rows(_clone_state(state))

    assert rows_ui == rows_legacy == []
