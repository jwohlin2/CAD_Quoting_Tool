"""Thin wrappers around domain utilities defined in :mod:`appV5`."""

from __future__ import annotations

from cad_quoter.domain_models import QuoteState
from appV5 import (
    compute_effective_state as _compute_effective_state,
    effective_to_overrides as _effective_to_overrides,
    merge_effective as _merge_effective,
    overrides_to_suggestions as _overrides_to_suggestions,
    reprice_with_effective as _reprice_with_effective,
    suggestions_to_overrides as _suggestions_to_overrides,
)

__all__ = [
    "QuoteState",
    "merge_effective",
    "compute_effective_state",
    "effective_to_overrides",
    "overrides_to_suggestions",
    "suggestions_to_overrides",
    "reprice_with_effective",
]


def merge_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.merge_effective` for test visibility."""

    return _merge_effective(*args, **kwargs)


def compute_effective_state(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.compute_effective_state` for test visibility."""

    return _compute_effective_state(*args, **kwargs)


def reprice_with_effective(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.reprice_with_effective` for test visibility."""

    return _reprice_with_effective(*args, **kwargs)


def effective_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.effective_to_overrides` for test visibility."""

    return _effective_to_overrides(*args, **kwargs)


def overrides_to_suggestions(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.overrides_to_suggestions` for test visibility."""

    return _overrides_to_suggestions(*args, **kwargs)


def suggestions_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.suggestions_to_overrides` for test visibility."""

    return _suggestions_to_overrides(*args, **kwargs)

