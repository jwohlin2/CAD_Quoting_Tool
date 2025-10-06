"""Domain-level state containers for the CAD quoting tool."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping
import copy


def _to_plain_data(value: Any) -> Any:
    """Recursively convert values to JSON-serialisable Python primitives."""

    if isinstance(value, dict):
        return {str(key): _to_plain_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()  # type: ignore[no-any-return]
        except Exception:
            pass
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """Best-effort conversion of arbitrary mapping-like data to a plain dict."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): _to_plain_data(val) for key, val in value.items()}
    try:
        return {str(key): _to_plain_data(val) for key, val in dict(value).items()}
    except Exception:
        return {}


@dataclass
class QuoteState:
    """Container for quote-related state spanning UI, LLM and pricing layers."""

    geo: dict[str, Any] = field(default_factory=dict)
    ui_vars: dict[str, Any] = field(default_factory=dict)
    rates: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    llm_raw: dict[str, Any] = field(default_factory=dict)
    suggestions: dict[str, Any] = field(default_factory=dict)
    user_overrides: dict[str, Any] = field(default_factory=dict)
    effective: dict[str, Any] = field(default_factory=dict)
    effective_sources: dict[str, Any] = field(default_factory=dict)
    accept_llm: dict[str, Any] = field(default_factory=dict)
    bounds: dict[str, Any] = field(default_factory=dict)
    material_source: str | None = None
    guard_context: dict[str, Any] = field(default_factory=dict)
    process_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entire state tree to plain Python structures."""

        data: dict[str, Any] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            data[spec.name] = _to_plain_data(value)
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "QuoteState":
        """Construct a :class:`QuoteState` from persisted plain-data structures."""

        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("QuoteState.from_dict expects a mapping of field values")

        mapping_fields = {
            "geo",
            "ui_vars",
            "rates",
            "baseline",
            "llm_raw",
            "suggestions",
            "user_overrides",
            "effective",
            "effective_sources",
            "accept_llm",
            "bounds",
            "guard_context",
            "process_plan",
        }

        kwargs: dict[str, Any] = {}
        for spec in fields(cls):
            if spec.name not in raw:
                continue
            value = raw[spec.name]
            if spec.name in mapping_fields:
                kwargs[spec.name] = _to_plain_dict(value)
            elif spec.name == "material_source":
                kwargs[spec.name] = None if value in (None, "") else str(value)
            else:
                kwargs[spec.name] = copy.deepcopy(value)

        return cls(**kwargs)
