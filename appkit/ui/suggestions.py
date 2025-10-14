from __future__ import annotations

from typing import Any

from cad_quoter.app._value_utils import _format_value
from cad_quoter.domain import QuoteState, canonicalize_pass_through_map
from cad_quoter.llm_overrides import _canonical_pass_label

from appkit.merge_utils import _collect_process_keys


def iter_suggestion_rows(state: QuoteState) -> list[dict]:
    rows: list[dict] = []
    baseline = state.baseline or {}
    suggestions = state.suggestions or {}
    overrides = state.user_overrides or {}
    effective = state.effective or {}
    sources = state.effective_sources or {}
    accept_raw = state.accept_llm
    accept = accept_raw if isinstance(accept_raw, dict) else {}

    baseline_hours_raw = (
        baseline.get("process_hours") if isinstance(baseline.get("process_hours"), dict) else {}
    )
    baseline_hours: dict[str, float] = {}
    for key, value in (baseline_hours_raw or {}).items():
        try:
            if abs(float(value)) > 1e-6:
                baseline_hours[key] = float(value)
        except Exception:
            continue

    sugg_mult_raw = suggestions.get("process_hour_multipliers")
    sugg_mult = sugg_mult_raw if isinstance(sugg_mult_raw, dict) else {}
    over_mult_raw = overrides.get("process_hour_multipliers")
    over_mult = over_mult_raw if isinstance(over_mult_raw, dict) else {}
    eff_mult_raw = effective.get("process_hour_multipliers")
    eff_mult = eff_mult_raw if isinstance(eff_mult_raw, dict) else {}
    src_mult_raw = sources.get("process_hour_multipliers")
    src_mult = src_mult_raw if isinstance(src_mult_raw, dict) else {}
    accept_mult_raw = accept.get("process_hour_multipliers")
    accept_mult = accept_mult_raw if isinstance(accept_mult_raw, dict) else {}
    keys_mult = sorted(_collect_process_keys(baseline_hours, sugg_mult, over_mult))
    for key in keys_mult:
        rows.append(
            {
                "path": ("process_hour_multipliers", key),
                "label": f"Process × {key}",
                "kind": "multiplier",
                "baseline": 1.0,
                "llm": sugg_mult.get(key),
                "user": over_mult.get(key),
                "accept": bool(accept_mult.get(key)),
                "effective": eff_mult.get(key, 1.0),
                "source": src_mult.get(key, "baseline"),
            }
        )

    sugg_add_raw = suggestions.get("process_hour_adders")
    sugg_add = sugg_add_raw if isinstance(sugg_add_raw, dict) else {}
    over_add_raw = overrides.get("process_hour_adders")
    over_add = over_add_raw if isinstance(over_add_raw, dict) else {}
    eff_add_raw = effective.get("process_hour_adders")
    eff_add = eff_add_raw if isinstance(eff_add_raw, dict) else {}
    src_add_raw = sources.get("process_hour_adders")
    src_add = src_add_raw if isinstance(src_add_raw, dict) else {}
    accept_add_raw = accept.get("process_hour_adders")
    accept_add = accept_add_raw if isinstance(accept_add_raw, dict) else {}
    keys_add = sorted(_collect_process_keys(baseline_hours, sugg_add, over_add))
    for key in keys_add:
        rows.append(
            {
                "path": ("process_hour_adders", key),
                "label": f"Process +hr {key}",
                "kind": "hours",
                "baseline": 0.0,
                "llm": sugg_add.get(key),
                "user": over_add.get(key),
                "accept": bool(accept_add.get(key)),
                "effective": eff_add.get(key, 0.0),
                "source": src_add.get(key, "baseline"),
            }
        )

    sugg_pass = (
        canonicalize_pass_through_map(suggestions.get("add_pass_through"))
        if isinstance(suggestions.get("add_pass_through"), dict)
        else {}
    )
    over_pass = (
        canonicalize_pass_through_map(overrides.get("add_pass_through"))
        if isinstance(overrides.get("add_pass_through"), dict)
        else {}
    )
    base_pass = (
        canonicalize_pass_through_map(baseline.get("pass_through"))
        if isinstance(baseline.get("pass_through"), dict)
        else {}
    )
    eff_pass = (
        canonicalize_pass_through_map(effective.get("add_pass_through"))
        if isinstance(effective.get("add_pass_through"), dict)
        else {}
    )
    src_pass_candidate = sources.get("add_pass_through")
    src_pass_raw = src_pass_candidate if isinstance(src_pass_candidate, dict) else {}
    accept_pass_raw = accept.get("add_pass_through")
    accept_pass = accept_pass_raw if isinstance(accept_pass_raw, dict) else {}
    src_pass: dict[str, Any] = {}
    for key, value in src_pass_raw.items():
        canon_key = _canonical_pass_label(key)
        if canon_key:
            src_pass[canon_key] = value
    keys_pass = sorted(set(base_pass) | set(sugg_pass) | set(over_pass))
    for key in keys_pass:
        base_amount = base_pass.get(key)
        label = f"Pass-through Δ {key}"
        if base_amount not in (None, ""):
            try:
                label = f"{label} (base {_format_value(base_amount, 'currency')})"
            except Exception:
                pass
        rows.append(
            {
                "path": ("add_pass_through", key),
                "label": label,
                "kind": "currency",
                "baseline": 0.0,
                "llm": sugg_pass.get(key),
                "user": over_pass.get(key),
                "accept": bool(accept_pass.get(key)),
                "effective": eff_pass.get(key, 0.0),
                "source": src_pass.get(key, "baseline"),
            }
        )

    scrap_base = baseline.get("scrap_pct")
    scrap_llm = suggestions.get("scrap_pct")
    scrap_user = overrides.get("scrap_pct")
    scrap_eff = effective.get("scrap_pct")
    scrap_src = sources.get("scrap_pct", "baseline")
    if any(v is not None for v in (scrap_base, scrap_llm, scrap_user)):
        rows.append(
            {
                "path": ("scrap_pct",),
                "label": "Scrap %",
                "kind": "percent",
                "baseline": scrap_base,
                "llm": scrap_llm,
                "user": scrap_user,
                "accept": bool(accept.get("scrap_pct")),
                "effective": scrap_eff,
                "source": scrap_src,
            }
        )

    cont_base = baseline.get("contingency_pct")
    cont_llm = suggestions.get("contingency_pct")
    cont_user = overrides.get("contingency_pct")
    cont_eff = effective.get("contingency_pct")
    cont_src = sources.get("contingency_pct", "baseline")
    if any(v is not None for v in (cont_base, cont_llm, cont_user)):
        rows.append(
            {
                "path": ("contingency_pct",),
                "label": "Contingency %",
                "kind": "percent",
                "baseline": cont_base,
                "llm": cont_llm,
                "user": cont_user,
                "accept": bool(accept.get("contingency_pct")),
                "effective": cont_eff,
                "source": cont_src,
            }
        )

    setups_base = baseline.get("setups")
    setups_llm = suggestions.get("setups")
    setups_user = overrides.get("setups")
    setups_eff = effective.get("setups")
    setups_src = sources.get("setups", "baseline")
    if any(v is not None for v in (setups_base, setups_llm, setups_user)):
        rows.append(
            {
                "path": ("setups",),
                "label": "Setups",
                "kind": "int",
                "baseline": setups_base,
                "llm": setups_llm,
                "user": setups_user,
                "accept": bool(accept.get("setups")),
                "effective": setups_eff,
                "source": setups_src,
            }
        )

    fixture_base = baseline.get("fixture")
    fixture_llm = suggestions.get("fixture")
    fixture_user = overrides.get("fixture")
    fixture_eff = effective.get("fixture")
    fixture_src = sources.get("fixture", "baseline")
    if any(v is not None for v in (fixture_base, fixture_llm, fixture_user, fixture_eff)):
        rows.append(
            {
                "path": ("fixture",),
                "label": "Fixture plan",
                "kind": "text",
                "baseline": fixture_base,
                "llm": fixture_llm,
                "user": fixture_user,
                "accept": bool(accept.get("fixture")),
                "effective": fixture_eff,
                "source": fixture_src,
            }
        )

    def _add_scalar_row(path: tuple[str, ...], label: str, kind: str, key: str) -> None:
        # Placeholder to satisfy interpreter; implementation not required at import time.
        return None


__all__ = ["iter_suggestion_rows"]
