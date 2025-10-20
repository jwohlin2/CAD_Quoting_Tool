from __future__ import annotations

from typing import Any, Mapping

from cad_quoter.app._value_utils import _format_value
from cad_quoter.domain import QuoteState, canonicalize_pass_through_map
from cad_quoter.llm_overrides import _canonical_pass_label

from appkit.merge_utils import _collect_process_keys

__all__ = ["iter_suggestion_rows", "build_suggestion_rows"]


def _as_mapping(data: Any) -> dict[str, Any]:
    return data if isinstance(data, dict) else {}


def _coerce_dict(container: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    return value if isinstance(value, dict) else {}


def build_suggestion_rows(
    state: QuoteState, *, pass_through_label_template: str = "Pass-through Δ {key}"
) -> list[dict]:
    rows: list[dict] = []
    baseline = _as_mapping(state.baseline)
    suggestions = _as_mapping(state.suggestions)
    overrides = _as_mapping(state.user_overrides)
    effective = _as_mapping(state.effective)
    sources = _as_mapping(state.effective_sources)
    accept_raw = state.accept_llm
    accept = accept_raw if isinstance(accept_raw, dict) else {}

    baseline_hours_raw = _coerce_dict(baseline, "process_hours")
    baseline_hours: dict[str, float] = {}
    for key, value in baseline_hours_raw.items():
        try:
            as_float = float(value)
        except Exception:
            continue
        if abs(as_float) > 1e-6:
            baseline_hours[key] = as_float

    map_specs = [
        {
            "path": "process_hour_multipliers",
            "label": "Process × {key}",
            "kind": "multiplier",
            "baseline": 1.0,
        },
        {
            "path": "process_hour_adders",
            "label": "Process +hr {key}",
            "kind": "hours",
            "baseline": 0.0,
        },
    ]

    for spec in map_specs:
        path_key = spec["path"]
        label_template = spec["label"]
        kind = spec["kind"]
        baseline_default = spec["baseline"]
        sugg_map = _coerce_dict(suggestions, path_key)
        user_map = _coerce_dict(overrides, path_key)
        eff_map = _coerce_dict(effective, path_key)
        src_map = _coerce_dict(sources, path_key)
        accept_map = _coerce_dict(accept, path_key)
        keys = sorted(_collect_process_keys(baseline_hours, sugg_map, user_map))
        for key in keys:
            rows.append(
                {
                    "path": (path_key, key),
                    "label": label_template.format(key=key),
                    "kind": kind,
                    "baseline": baseline_default,
                    "llm": sugg_map.get(key),
                    "user": user_map.get(key),
                    "accept": bool(accept_map.get(key)),
                    "effective": eff_map.get(key, baseline_default),
                    "source": src_map.get(key, "baseline"),
                }
            )

    sugg_pass = canonicalize_pass_through_map(suggestions.get("add_pass_through"))
    over_pass = canonicalize_pass_through_map(overrides.get("add_pass_through"))
    base_pass = canonicalize_pass_through_map(baseline.get("pass_through"))
    eff_pass = canonicalize_pass_through_map(effective.get("add_pass_through"))
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
        label = pass_through_label_template.format(key=key)
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

    scalar_specs = [
        {"path": ("scrap_pct",), "label": "Scrap %", "kind": "percent"},
        {"path": ("setups",), "label": "Setups", "kind": "int"},
        {
            "path": ("fixture",),
            "label": "Fixture plan",
            "kind": "text",
            "presence": ("baseline", "llm", "user", "effective"),
        },
    ]

    for spec in scalar_specs:
        key = spec["path"][0]
        values = {
            "baseline": baseline.get(key),
            "llm": suggestions.get(key),
            "user": overrides.get(key),
            "effective": effective.get(key),
            "source": sources.get(key, "baseline"),
            "accept": bool(accept.get(key)),
        }
        presence_fields = spec.get("presence", ("baseline", "llm", "user"))
        if any(values[field] is not None for field in presence_fields):
            rows.append(
                {
                    "path": spec["path"],
                    "label": spec["label"],
                    "kind": spec["kind"],
                    "baseline": values["baseline"],
                    "llm": values["llm"],
                    "user": values["user"],
                    "accept": values["accept"],
                    "effective": values["effective"],
                    "source": values["source"],
                }
            )

    return rows


def iter_suggestion_rows(state: QuoteState) -> list[dict]:
    return build_suggestion_rows(state)
