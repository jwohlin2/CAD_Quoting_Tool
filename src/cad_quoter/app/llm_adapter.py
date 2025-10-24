"""Adapters and helpers for wiring the UI to the shared LLM integration."""

from __future__ import annotations

import re
from collections.abc import Mapping as _MappingABC
from typing import Any, Callable, Mapping, Protocol, TypeAlias, TYPE_CHECKING, cast

# Use the shared numeric coercion helper exposed by the domain layer.
from cad_quoter.domain import coerce_bounds
from cad_quoter.domain_models.values import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.llm_overrides import (
    clamp,
    get_llm_bound_defaults,
    get_llm_overrides,
)

from cad_quoter.geometry import upsert_var_row

try:
    # Optional runtime dependency; may be missing in some environments
    from cad_quoter.llm import run_llm_suggestions as _run_llm_suggestions  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _run_llm_suggestions = None  # type: ignore[assignment]

try:  # pragma: no cover - pandas is optional at runtime
    from pandas import DataFrame as PandasDataFrame  # type: ignore

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - used when pandas is unavailable
    PandasDataFrame = Any  # type: ignore
    _HAS_PANDAS = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cad_quoter.app.llm_helpers import LLMClient
else:  # pragma: no cover - runtime fallback
    LLMClient = Any  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Legacy surface area kept for backward compatibility with older call sites.


class LLMClientLike(Protocol):
    @property
    def model_path(self) -> str: ...

    @property
    def available(self) -> bool: ...

    def ask_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = ...,
        max_tokens: int = ...,
        context: Mapping[str, Any] | None = ...,
        params: Mapping[str, Any] | None = ...,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]: ...

    def close(self) -> None: ...


RunLLMSuggestions: TypeAlias = Callable[
    [LLMClientLike, dict[str, Any]], tuple[dict[str, Any], str, dict[str, Any]]
]

run_llm_suggestions: RunLLMSuggestions | None = cast(
    "RunLLMSuggestions | None", _run_llm_suggestions
)


# ---------------------------------------------------------------------------
# Newer helpers used by the modern Tk UI.


InferHoursDelegate = Callable[
    [dict[str, Any], dict | None, dict | None, LLMClient | None],
    dict[str, Any],
]

_infer_hours_delegate: InferHoursDelegate | None = None


def configure_llm_integration(delegate: Any) -> None:
    """Register the shared LLM integration used by the UI layer."""

    global _infer_hours_delegate

    candidate = getattr(delegate, "infer_hours_and_overrides_from_geo", None)
    if not callable(candidate):  # pragma: no cover - defensive guard
        raise ValueError("delegate must expose 'infer_hours_and_overrides_from_geo'")

    _infer_hours_delegate = candidate


def _require_delegate() -> InferHoursDelegate:
    if _infer_hours_delegate is None:  # pragma: no cover - configured during app start
        raise RuntimeError("LLM integration has not been configured")

    return _infer_hours_delegate


def infer_hours_and_overrides_from_geo(
    geo: dict,
    params: dict | None = None,
    rates: dict | None = None,
    *,
    client: "LLMClient" | None = None,
) -> dict[str, Any]:
    """Delegate to the configured LLM integration for hour estimation."""

    delegate = _require_delegate()
    return delegate(geo, params, rates, client)


_LLM_HOUR_ITEM_MAP: dict[str, str] = {
    "Programming_Hours": "Programming Hours",
    "CAM_Programming_Hours": "CAM Programming Hours",
    "Engineering_Hours": "Engineering (Docs/Fixture Design) Hours",
    "Fixture_Build_Hours": "Fixture Build Hours",
    "Roughing_Cycle_Time_hr": "Roughing Cycle Time",
    "Semi_Finish_Cycle_Time_hr": "Semi-Finish Cycle Time",
    "Finishing_Cycle_Time_hr": "Finishing Cycle Time",
    "InProcess_Inspection_Hours": "In-Process Inspection Hours",
    "Final_Inspection_Hours": "Final Inspection Hours",
    "CMM_Programming_Hours": "CMM Programming Hours",
    "CMM_RunTime_min": "CMM Run Time min",
    "Deburr_Hours": "Deburr Hours",
    "Tumble_Hours": "Tumbling Hours",
    "Blast_Hours": "Bead Blasting Hours",
    "Laser_Mark_Hours": "Laser Mark Hours",
    "Masking_Hours": "Masking Hours",
    "Saw_Waterjet_Hours": "Sawing Hours",
    "Assembly_Hours": "Assembly Hours",
    "Packaging_Labor_Hours": "Packaging Labor Hours",
}

_LLM_SETUP_ITEM_MAP: dict[str, str] = {
    "Milling_Setups": "Number of Milling Setups",
    "Setup_Hours_per_Setup": "Setup Hours / Setup",
}

_LLM_INSPECTION_ITEM_MAP: dict[str, str] = {
    "FAIR_Required": "FAIR Required",
    "Source_Inspection_Required": "Source Inspection Requirement",
}


def normalize_item_text(value: Any) -> str:
    """Return a normalized key for matching variables rows."""

    if value is None:
        text = ""
    else:
        text = str(value)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_item(value: Any) -> str:
    """Public wrapper for item normalization used across the editor."""

    return normalize_item_text(value)


def clamp_llm_hours(
    raw: Mapping[str, Any] | None,
    geo: Mapping[str, Any] | None,
    *,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sanitize LLM-derived hour estimates before applying them to the UI."""

    cleaned: dict[str, Any] = {}
    raw_map = cast(Mapping[str, Any], raw or {})
    params_map = cast(Mapping[str, Any], params or {})
    bounds_raw = params_map.get("bounds") if isinstance(params_map, _MappingABC) else None
    bounds_map = bounds_raw if isinstance(bounds_raw, _MappingABC) else None
    coerced_bounds = coerce_bounds(bounds_map)
    adder_min_bound = coerced_bounds["adder_min_hr"]
    adder_max_bound = coerced_bounds["adder_max_hr"]

    hours_out: dict[str, float] = {}
    hours_val = raw_map.get("hours")
    if isinstance(hours_val, _MappingABC):
        hours_src: Mapping[str, Any] = cast(Mapping[str, Any], hours_val)
    else:
        hours_src = {}
    for key, value in hours_src.items():
        val = _coerce_float_or_none(value)
        if val is None:
            continue
        upper = 48.0
        if str(key).endswith("_min"):
            upper = 2400.0
        hours_out[str(key)] = clamp(float(val), 0.0, upper, 0.0)
    if hours_out:
        cleaned["hours"] = hours_out

    setups_out: dict[str, Any] = {}
    setups_val = raw_map.get("setups")
    if isinstance(setups_val, _MappingABC):
        setups_src: Mapping[str, Any] = cast(Mapping[str, Any], setups_val)
    else:
        setups_src = {}
    if setups_src:
        count_raw = setups_src.get("Milling_Setups")
        if count_raw is not None:
            try:
                setups_out["Milling_Setups"] = max(1, min(6, int(round(float(count_raw)))))
            except Exception:  # pragma: no cover - defensive
                pass
        setup_hours = _coerce_float_or_none(setups_src.get("Setup_Hours_per_Setup"))
        if setup_hours is not None:
            setups_out["Setup_Hours_per_Setup"] = clamp(
                float(setup_hours), adder_min_bound, adder_max_bound, adder_min_bound
            )
    if setups_out:
        cleaned["setups"] = setups_out

    inspection_out: dict[str, bool] = {}
    inspection_val = raw_map.get("inspection")
    if isinstance(inspection_val, _MappingABC):
        inspection_src: Mapping[str, Any] = cast(Mapping[str, Any], inspection_val)
    else:
        inspection_src = {}
    for key in _LLM_INSPECTION_ITEM_MAP:
        if key in inspection_src:
            inspection_out[key] = bool(inspection_src.get(key))
    if inspection_out:
        cleaned["inspection"] = inspection_out

    notes_raw = raw_map.get("notes")
    if isinstance(notes_raw, list):
        cleaned["notes"] = [str(n).strip() for n in notes_raw if str(n).strip()][:8]

    meta_raw = raw_map.get("_meta")
    if isinstance(meta_raw, _MappingABC):
        cleaned["_meta"] = dict(meta_raw)

    for key, value in raw_map.items():
        if key in {"hours", "setups", "inspection", "notes", "_meta"}:
            continue
        cleaned.setdefault(str(key), value)

    return cleaned


def apply_llm_hours_to_variables(
    df: PandasDataFrame | None,
    estimates: Mapping[str, Any] | None,
    *,
    allow_overwrite_nonzero: bool = False,
    log: dict | None = None,
) -> PandasDataFrame | None:
    """Apply sanitized LLM hour estimates to a variables dataframe."""

    if not _HAS_PANDAS or df is None:
        return df

    estimates_map = cast(Mapping[str, Any], estimates or {})
    df_out = df.copy(deep=True)
    normalized_items = df_out["Item"].astype(str).apply(normalize_item_text)
    index_lookup = {norm: idx for idx, norm in zip(df_out.index, normalized_items)}

    def _write_value(label: str, value: Any, *, dtype: str = "number") -> None:
        nonlocal df_out, normalized_items, index_lookup
        if value is None:
            return
        normalized = normalize_item_text(label)
        idx = index_lookup.get(normalized)
        new_value = value
        if idx is None:
            df_out = upsert_var_row(df_out, label, new_value, dtype=dtype)
            normalized_items = df_out["Item"].astype(str).apply(normalize_item_text)
            index_lookup = {norm: idx for idx, norm in zip(df_out.index, normalized_items)}
            idx = index_lookup.get(normalized)
            previous = None
        else:
            previous = df_out.at[idx, "Example Values / Options"]
            if not allow_overwrite_nonzero:
                existing_val = _coerce_float_or_none(previous)
                if existing_val is not None and abs(existing_val) > 1e-9:
                    return
            df_out.at[idx, "Example Values / Options"] = new_value
            df_out.at[idx, "Data Type / Input Method"] = dtype
        if idx is None:
            return
        df_out.at[idx, "Example Values / Options"] = new_value
        df_out.at[idx, "Data Type / Input Method"] = dtype
        if log is not None:
            log.setdefault("llm_hours", []).append(
                {
                    "item": label,
                    "value": new_value,
                    "previous": previous,
                }
            )

    hours_val = estimates_map.get("hours")
    if isinstance(hours_val, _MappingABC):
        hours_src: Mapping[str, Any] = cast(Mapping[str, Any], hours_val)
    else:
        hours_src = {}
    for key, value in hours_src.items():
        label = _LLM_HOUR_ITEM_MAP.get(str(key))
        if not label:
            continue
        val = _coerce_float_or_none(value)
        if val is None:
            continue
        _write_value(label, float(val), dtype="number")

    setups_val = estimates_map.get("setups")
    if isinstance(setups_val, _MappingABC):
        setups_src: Mapping[str, Any] = cast(Mapping[str, Any], setups_val)
    else:
        setups_src = {}
    for key, value in setups_src.items():
        label = _LLM_SETUP_ITEM_MAP.get(str(key))
        if not label:
            continue
        if key == "Milling_Setups":
            try:
                numeric = max(1, min(6, int(round(float(value)))))
            except Exception:  # pragma: no cover - defensive
                continue
            _write_value(label, numeric, dtype="number")
        else:
            val = _coerce_float_or_none(value)
            if val is None:
                continue
            _write_value(label, float(val), dtype="number")

    inspection_val = estimates_map.get("inspection")
    if isinstance(inspection_val, _MappingABC):
        inspection_src: Mapping[str, Any] = cast(Mapping[str, Any], inspection_val)
    else:
        inspection_src = {}
    for key, value in inspection_src.items():
        label = _LLM_INSPECTION_ITEM_MAP.get(str(key))
        if not label:
            continue
        _write_value(label, "True" if bool(value) else "False", dtype="Checkbox")

    return df_out


def infer_shop_overrides_from_geo(
    geo: Mapping[str, Any] | None,
    *,
    params: Mapping[str, Any] | None = None,
    rates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return sanitized LLM output for the manual LLM tab."""

    estimates_raw = infer_hours_and_overrides_from_geo(
        dict(geo or {}),
        params=dict(params or {}),
        rates=dict(rates or {}),
    )
    cleaned = clamp_llm_hours(estimates_raw, geo or {}, params=params)
    return {
        "estimates": cleaned,
        "LLM_Adjustments": {},
    }


__all__ = [
    "LLMClientLike",
    "RunLLMSuggestions",
    "run_llm_suggestions",
    "get_llm_overrides",
    "get_llm_bound_defaults",
    "configure_llm_integration",
    "infer_hours_and_overrides_from_geo",
    "clamp_llm_hours",
    "apply_llm_hours_to_variables",
    "infer_shop_overrides_from_geo",
    "normalize_item",
    "normalize_item_text",
]
