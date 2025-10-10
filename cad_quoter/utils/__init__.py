"""Shared utility helpers for the CAD quoter application."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def compact_dict(
    d: Mapping[K, V | None],
    *,
    drop_values: Mapping[Any, Any] | Iterable[Any] | None = (None,),
) -> dict[K, V]:
    """Return a copy of *d* without entries whose values are filtered out."""

    if drop_values is None:
        def _should_keep(value: Any) -> bool:
            return True
    else:
        drop = tuple(drop_values)

        def _should_keep(value: Any) -> bool:
            return value not in drop

    return {k: v for k, v in d.items() if _should_keep(v)}


def sdict(d: Mapping[Any, Any] | None) -> dict[str, str]:
    """Return a string-keyed/string-valued copy of *d* (ignoring ``None``)."""

    return {str(k): str(v) for k, v in (d or {}).items()}


T = TypeVar("T")


def _first_non_none(*vals: T | None) -> T | None:
    """Return the first value in *vals* that is not ``None``."""

    for val in vals:
        if val is not None:
            return val
    return None


def jdump(obj: Any, *, indent: int = 2, default: Any | None = str, **kwargs: Any) -> str:
    """Serialize *obj* to JSON using project-wide defaults."""

    dumps_kwargs = {"indent": indent, **kwargs}
    if default is not None:
        dumps_kwargs["default"] = default
    return json.dumps(obj, **dumps_kwargs)


__all__ = ["compact_dict", "sdict", "_first_non_none", "jdump"]
