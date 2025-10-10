"""Helpers for working with geometry context dictionaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _iter_geo_contexts(geo_context: Mapping[str, Any] | None) -> Iterable[Mapping[str, Any]]:
    if isinstance(geo_context, Mapping):
        yield geo_context
        inner = geo_context.get("geo")
        if isinstance(inner, Mapping):
            yield inner


def _collection_has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        for candidate in value.values():
            if _collection_has_text(candidate):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_collection_has_text(candidate) for candidate in value)
    return False


def _geo_mentions_outsourced(geo_context: Mapping[str, Any] | None) -> bool:
    for ctx in _iter_geo_contexts(geo_context):
        if _collection_has_text(ctx.get("finishes")):
            return True
        if _collection_has_text(ctx.get("finish_flags")):
            return True
    return False


def _should_include_outsourced_pass(
    outsourced_cost: float, geo_context: Mapping[str, Any] | None
) -> bool:
    try:
        cost_val = float(outsourced_cost)
    except Exception:
        cost_val = 0.0
    if abs(cost_val) > 1e-6:
        return True
    return _geo_mentions_outsourced(geo_context)


__all__ = [
    "_iter_geo_contexts",
    "_collection_has_text",
    "_geo_mentions_outsourced",
    "_should_include_outsourced_pass",
]
