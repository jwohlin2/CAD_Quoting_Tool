from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def coerce_bool(value: object, *, default: bool | None = ...) -> bool | None: ...


def compact_dict(
    d: Mapping[K, V | None],
    *,
    drop_values: Mapping[Any, Any] | Iterable[Any] | None = ...,
) -> dict[K, V]: ...


def sdict(d: Mapping[Any, Any] | None) -> dict[str, str]: ...


def json_safe_copy(
    obj: Any,
    *,
    max_depth: int = ...,
    _depth: int = ...,
    _seen: set[int] | None = ...,
) -> Any: ...


def jdump(obj: Any, *, indent: int = ..., default: Any | None = ..., **kwargs: Any) -> str: ...


__all__ = [
    "coerce_bool",
    "compact_dict",
    "jdump",
    "json_safe_copy",
    "sdict",
]

