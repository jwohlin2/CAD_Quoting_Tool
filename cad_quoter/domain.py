"""Domain utilities and thin proxies for UI-bound helpers."""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Mapping as _MappingABC
from types import MappingProxyType
from typing import Any, Callable, Mapping, TYPE_CHECKING, cast

from cad_quoter.coerce import to_float
from cad_quoter.config import logger
from cad_quoter.domain_models import DEFAULT_MATERIAL_DISPLAY, QuoteState

if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from appV5 import (  # pylint: disable=unused-import
        compute_effective_state as _compute_effective_state,
        effective_to_overrides as _effective_to_overrides,
        merge_effective as _merge_effective,
        reprice_with_effective as _reprice_with_effective,
    )

__all__ = [
    "QuoteState",
    "merge_effective",
    "compute_effective_state",
    "effective_to_overrides",
    "overrides_to_suggestions",
    "suggestions_to_overrides",
    "reprice_with_effective",
    "HARDWARE_PASS_LABEL",
    "LEGACY_HARDWARE_PASS_LABEL",
    "_canonical_pass_label",
    "_as_float_or_none",
    "canonicalize_pass_through_map",
    "coerce_bounds",
    "get_llm_bound_defaults",
    "LLM_BOUND_DEFAULTS",
    "build_suggest_payload",
]


def _app_module():
    """Return the lazily-imported :mod:`appV5` module."""

    module = sys.modules.get("appV5")
    if module is None:
        module = importlib.import_module("appV5")
    return module


def merge_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.merge_effective` for test visibility."""

    app = _app_module()
    return app.merge_effective(*args, **kwargs)


def compute_effective_state(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.compute_effective_state` for test visibility."""

    app = _app_module()
    return app.compute_effective_state(*args, **kwargs)


def reprice_with_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.reprice_with_effective` for test visibility."""

    app = _app_module()
    return app.reprice_with_effective(*args, **kwargs)


def effective_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.effective_to_overrides` for test visibility."""

    app = _app_module()
    return app.effective_to_overrides(*args, **kwargs)


HARDWARE_PASS_LABEL = "Hardware"
LEGACY_HARDWARE_PASS_LABEL = "Hardware / BOM"
_HARDWARE_LABEL_ALIASES = {
    HARDWARE_PASS_LABEL.lower(),
    LEGACY_HARDWARE_PASS_LABEL.lower(),
    "hardware/bom",
    "hardware bom",
}


def _canonical_pass_label(label: str | None) -> str:
    name = str(label or "").strip()
    if name.lower() in _HARDWARE_LABEL_ALIASES:
        return HARDWARE_PASS_LABEL
    return name


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


LLM_MULTIPLIER_MIN = 0.25
LLM_MULTIPLIER_MAX = 4.0
LLM_ADDER_MAX = 8.0


def _as_float_or_none(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            return float(cleaned)
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def coerce_bounds(bounds: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize LLM bounds into a canonical structure."""

    if bounds is None:
        bounds_map: Mapping[str, Any] = {}
    else:
        bounds_map = bounds

    mult_min = _as_float_or_none(bounds_map.get("mult_min"))
    if mult_min is None:
        mult_min = LLM_MULTIPLIER_MIN
    else:
        mult_min = max(LLM_MULTIPLIER_MIN, float(mult_min))

    mult_max = _as_float_or_none(bounds_map.get("mult_max"))
    if mult_max is None:
        mult_max = LLM_MULTIPLIER_MAX
    else:
        mult_max = min(LLM_MULTIPLIER_MAX, float(mult_max))
    mult_max = max(mult_max, mult_min)

    adder_min = _as_float_or_none(bounds_map.get("adder_min_hr"))
    if adder_min is None:
        adder_min = _as_float_or_none(bounds_map.get("add_hr_min"))
    adder_min = max(0.0, float(adder_min)) if adder_min is not None else 0.0

    adder_max = _as_float_or_none(bounds_map.get("adder_max_hr"))
    add_hr_cap = _as_float_or_none(bounds_map.get("add_hr_max"))
    if adder_max is None and add_hr_cap is not None:
        adder_max = float(add_hr_cap)
    elif adder_max is not None and add_hr_cap is not None:
        adder_max = min(float(adder_max), float(add_hr_cap))
    if adder_max is None:
        adder_max = LLM_ADDER_MAX
    adder_max = max(adder_min, min(LLM_ADDER_MAX, float(adder_max)))

    scrap_min = _as_float_or_none(bounds_map.get("scrap_min"))
    scrap_min = max(0.0, float(scrap_min)) if scrap_min is not None else 0.0

    scrap_max = _as_float_or_none(bounds_map.get("scrap_max"))
    scrap_max = float(scrap_max) if scrap_max is not None else 0.25
    scrap_max = max(scrap_max, scrap_min)

    bucket_caps_raw = bounds_map.get("adder_bucket_max") or bounds_map.get("add_hr_bucket_max")
    bucket_caps: dict[str, float] = {}
    if isinstance(bucket_caps_raw, _MappingABC):
        for key, raw in bucket_caps_raw.items():
            cap_val = _as_float_or_none(raw)
            if cap_val is None:
                continue
            bucket_caps[str(key).lower()] = max(adder_min, min(adder_max, float(cap_val)))

    return {
        "mult_min": mult_min,
        "mult_max": mult_max,
        "adder_min_hr": adder_min,
        "adder_max_hr": adder_max,
        "scrap_min": scrap_min,
        "scrap_max": scrap_max,
        "adder_bucket_max": bucket_caps,
    }


def _default_llm_bounds_dict() -> dict[str, Any]:
    """Return the sanitized default LLM guardrail bounds."""

    return coerce_bounds({})


def get_llm_bound_defaults() -> dict[str, Any]:
    """Return a mutable copy of the default LLM guardrail bounds."""

    return dict(coerce_bounds(LLM_BOUND_DEFAULTS))


LLM_BOUND_DEFAULTS: Mapping[str, Any] = MappingProxyType(_default_llm_bounds_dict())


def build_suggest_payload(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.build_suggest_payload` for test visibility."""

    app = _app_module()
    return app.build_suggest_payload(*args, **kwargs)

def ensure_accept_flags(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.ensure_accept_flags` for test visibility."""

    app = _app_module()
    return app.ensure_accept_flags(*args, **kwargs)


def iter_suggestion_rows(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.iter_suggestion_rows` for test visibility."""

    app = _app_module()
    return app.iter_suggestion_rows(*args, **kwargs)

