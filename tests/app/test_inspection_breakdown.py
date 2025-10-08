import pytest

from appV5 import compute_quote_from_df, default_variables_template
from cad_quoter.domain import QuoteState


def _base_df():
    return default_variables_template().copy()


def test_inspection_components_sum_to_baseline() -> None:
    df = _base_df()
    state = QuoteState()

    result = compute_quote_from_df(df, quote_state=state, llm_enabled=False)
    meta = result["breakdown"]["process_meta"]["inspection"]

    components = meta["components"]
    baseline_hr = meta["baseline_hr"]

    assert set(components.keys()) == {"in_process", "final", "cmm_programming", "cmm_run", "fair", "source"}
    assert baseline_hr == pytest.approx(sum(components.values()))


def test_inspection_overrides_record_adjustments() -> None:
    df = _base_df()
    baseline_state = QuoteState()
    baseline_result = compute_quote_from_df(df, quote_state=baseline_state, llm_enabled=False)
    baseline_meta = baseline_result["breakdown"]["process_meta"]["inspection"]
    baseline_hr = baseline_meta["hr"]

    target_hr = max(0.0, baseline_hr - 0.5)

    state = QuoteState()
    state.user_overrides["cmm_minutes"] = 90  # 1.5 hours
    state.user_overrides["inspection_total_hr"] = target_hr

    result = compute_quote_from_df(df, quote_state=state, llm_enabled=False)
    meta = result["breakdown"]["process_meta"]["inspection"]

    adjustments = meta["adjustments"]

    assert adjustments["cmm_run"] > 0
    assert state.effective_sources.get("inspection_total_hr") == "user"
