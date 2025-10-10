"""Lightweight dictionary helpers used across the quoting tool."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def compact_dict(
    d: Mapping[K, V | None],
    *,
    drop_values: Iterable[Any] | None = (None,),
) -> dict[K, V]:
    """Return a copy of ``d`` without entries whose values are filtered out."""

    if drop_values is None:
        def _should_keep(value: Any) -> bool:
            return True
    else:
        drop = tuple(drop_values)

        def _should_keep(value: Any) -> bool:
            return value not in drop

    return {k: v for k, v in d.items() if _should_keep(v)}


def sdict(d: Mapping[Any, Any] | None) -> dict[str, str]:
    """Return a string-keyed/string-valued copy of ``d`` (ignoring ``None``)."""

    return {str(k): str(v) for k, v in (d or {}).items()}


__all__ = ["compact_dict", "sdict"]
