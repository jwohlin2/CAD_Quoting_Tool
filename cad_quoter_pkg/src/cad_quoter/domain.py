"""Domain utilities and thin proxies for UI-bound helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping as _MappingABC
from typing import Any, Callable, Mapping, TYPE_CHECKING, cast

from cad_quoter.config import logger
from cad_quoter.domain_models import QuoteState
from cad_quoter.domain_models.values import to_float
from cad_quoter.llm_suggest import build_suggest_payload as _build_suggest_payload
from cad_quoter.llm_overrides import (
    LLM_BOUND_DEFAULTS,
    _as_float_or_none,
    _default_llm_bounds_dict,
    coerce_bounds,
    get_llm_bound_defaults,
)
from cad_quoter.pass_labels import (
    HARDWARE_PASS_LABEL,
    LEGACY_HARDWARE_PASS_LABEL,
    _HARDWARE_LABEL_ALIASES,
    _canonical_pass_label,
)

if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    pass

__all__ = [
    "QuoteState",
    "merge_effective",
    "compute_effective_state",
    "effective_to_overrides",
    "overrides_to_suggestions",
    "suggestions_to_overrides",
    "apply_suggestions",
    "reprice_with_effective",
    "HARDWARE_PASS_LABEL",
    "LEGACY_HARDWARE_PASS_LABEL",
    "_canonical_pass_label",
    "_as_float_or_none",
    "_default_llm_bounds_dict",
    "canonicalize_pass_through_map",
    "coerce_bounds",
    "get_llm_bound_defaults",
    "LLM_BOUND_DEFAULTS",
    "build_suggest_payload",
]


def _effective_module():
    """Return the lazily-imported :mod:`appkit.effective` helpers."""

    from appkit import effective as _effective  # imported lazily to avoid cycles

    return _effective


def _merge_module():
    """Return the lazily-imported merge helpers."""

    from appkit import merge_utils as _merge_utils  # imported lazily to avoid cycles

    return _merge_utils


def _suggestions_module():
    """Return the lazily-imported suggestion helpers."""

    from appkit.ui import suggestions as _suggestions  # imported lazily to avoid cycles

    return _suggestions


def merge_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appkit.merge_utils.merge_effective` for test visibility."""

    merge_helpers = _merge_module()
    return merge_helpers.merge_effective(*args, **kwargs)


def compute_effective_state(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appkit.effective.compute_effective_state` for test visibility."""

    effective = _effective_module()
    return effective.compute_effective_state(*args, **kwargs)


def reprice_with_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appkit.effective.reprice_with_effective` for test visibility."""

    effective = _effective_module()
    return effective.reprice_with_effective(*args, **kwargs)


def effective_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appkit.effective.effective_to_overrides` for test visibility."""

    effective = _effective_module()
    return effective.effective_to_overrides(*args, **kwargs)


def overrides_to_suggestions(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appkit.llm_converters.overrides_to_suggestions`."""

    from appkit import llm_converters as _llm_converters

    return _llm_converters.overrides_to_suggestions(*args, **kwargs)


def suggestions_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appkit.llm_converters.suggestions_to_overrides`."""

    from appkit import llm_converters as _llm_converters

    return _llm_converters.suggestions_to_overrides(*args, **kwargs)


def apply_suggestions(baseline: Mapping[str, Any] | None, suggestions: Mapping[str, Any] | None) -> dict:
    """Apply sanitized LLM suggestions onto a baseline quote snapshot."""

    merged = merge_effective(dict(baseline or {}), dict(suggestions or {}), {})
    merged.pop("_source_tags", None)
    merged.pop("_clamp_notes", None)

    notes = list(suggestions.get("notes") or []) if isinstance(suggestions, Mapping) else []
    if isinstance(suggestions, Mapping) and suggestions.get("no_change_reason"):
        notes.append(f"no_change: {suggestions['no_change_reason']}")
    merged["_llm_notes"] = notes

    return merged


def _canonicalize_pass_through_map(data: Any) -> dict[str, float]:
    """Normalize a pass-through dictionary into ``{label: float}``."""

    result: dict[str, float] = {}

    def _add(label: Any, amount: Any) -> None:
        key = _canonical_pass_label(label)
        try:
            val = to_float(amount)
        except Exception:  # pragma: no cover - defensive
            val = None
        if key and val is not None and math.isfinite(float(val)):
            result[key] = result.get(key, 0.0) + float(val)

    if isinstance(data, _MappingABC):
        for key, value in data.items():
            if isinstance(value, _MappingABC):
                inner = value
                amount = inner.get("amount", inner.get("value", inner.get("cost", inner.get("price"))))
                _add(key, amount)
            else:
                _add(key, value)
        return result

    if isinstance(data, (list, tuple)):
        for entry in data:
            if isinstance(entry, _MappingABC):
                label = entry.get("label") or entry.get("name") or entry.get("key") or entry.get("type")
                amount = entry.get("amount", entry.get("value", entry.get("cost", entry.get("price"))))
                if label is None and len(entry) == 1:
                    key = next(iter(entry.keys()))
                    _add(key, entry.get(key))
                else:
                    _add(label, amount)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                _add(entry[0], entry[1])
        return result

    return result


def canonicalize_pass_through_map(data: Any) -> dict[str, float]:
    """Return a canonicalized pass-through map with defensive fallback."""

    canonicalizer_obj = globals().get("_canonicalize_pass_through_map")
    if callable(canonicalizer_obj):
        canonicalizer = cast(Callable[[Any], dict[str, float]], canonicalizer_obj)
        try:
            return canonicalizer(data)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to canonicalize pass-through map; using fallback")

    result: dict[str, float] = {}

    def _add(label: Any, amount: Any) -> None:
        key = _canonical_pass_label(label)
        try:
            val = float(amount)
        except Exception:  # pragma: no cover - defensive
            return
        if key and math.isfinite(val):
            result[key] = result.get(key, 0.0) + float(val)

    if isinstance(data, _MappingABC):
        for key, value in data.items():
            if isinstance(value, _MappingABC):
                inner = value
                amount = inner.get("amount") or inner.get("value") or inner.get("cost") or inner.get("price")
                _add(key, amount)
            else:
                _add(key, value)
    elif isinstance(data, (list, tuple)):
        for entry in data:
            if isinstance(entry, _MappingABC):
                label = entry.get("label") or entry.get("name") or entry.get("key") or entry.get("type")
                amount = entry.get("amount") or entry.get("value") or entry.get("cost") or entry.get("price")
                if label is None and len(entry) == 1:
                    key = next(iter(entry.keys()))
                    _add(key, entry.get(key))
                else:
                    _add(label, amount)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                _add(entry[0], entry[1])

    return result


def build_suggest_payload(*args, **kwargs):  # type: ignore[override]
    """Expose :func:`cad_quoter.llm_suggest.build_suggest_payload` for tests."""

    return _build_suggest_payload(*args, **kwargs)

def ensure_accept_flags(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.ensure_accept_flags` for test visibility."""

    app = _app_module()
    return app.ensure_accept_flags(*args, **kwargs)


def iter_suggestion_rows(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appkit.ui.suggestions.iter_suggestion_rows` for tests."""

    suggestions = _suggestions_module()
    return suggestions.iter_suggestion_rows(*args, **kwargs)

