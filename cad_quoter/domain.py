"""Thin wrappers around domain utilities defined in :mod:`appV5`."""

from __future__ import annotations

from appV5 import (
    QuoteState,
    build_suggest_payload as _build_suggest_payload,
    compute_effective_state as _compute_effective_state,
    effective_to_overrides as _effective_to_overrides,
    ensure_accept_flags as _ensure_accept_flags,
    merge_effective as _merge_effective,
    reprice_with_effective as _reprice_with_effective,
)

__all__ = [
    "QuoteState",
    "merge_effective",
    "compute_effective_state",
    "reprice_with_effective",
    "ensure_accept_flags",
    "build_suggest_payload",
    "effective_to_overrides",
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


def ensure_accept_flags(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.ensure_accept_flags` for test visibility."""

    return _ensure_accept_flags(*args, **kwargs)


def build_suggest_payload(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.build_suggest_payload` for test visibility."""

    return _build_suggest_payload(*args, **kwargs)


def effective_to_overrides(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`appV5.effective_to_overrides` for test visibility."""

    return _effective_to_overrides(*args, **kwargs)

