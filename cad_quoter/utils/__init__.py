"""Utility functions for the cad_quoter package."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def _dict(value: Any) -> dict:
    """Return value if it's a dict, otherwise return an empty dict."""
    if isinstance(value, dict):
        return value
    return {}


def coerce_bool(value: object, *, default: bool | None = None) -> bool | None:
    """Coerce a value to a boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("true", "yes", "1", "y"):
            return True
        if lower in ("false", "no", "0", "n", ""):
            return False
    try:
        return bool(value)
    except Exception:
        return default


def compact_dict(
    d: Mapping[K, V | None],
    *,
    drop_values: Mapping[Any, Any] | Iterable[Any] | None = None,
) -> dict[K, V]:
    """Return a copy of the dict with None values removed.

    If drop_values is provided, also remove keys whose values are in that set.
    """
    if drop_values is None:
        drop_set: set[Any] = {None}
    elif isinstance(drop_values, Mapping):
        drop_set = set(drop_values.keys()) | {None}
    else:
        drop_set = set(drop_values) | {None}

    return {k: v for k, v in d.items() if v not in drop_set}  # type: ignore[return-value]


def sdict(d: Mapping[Any, Any] | None) -> dict[str, str]:
    """Convert a mapping to a dict with string keys and values."""
    if d is None:
        return {}
    return {str(k): str(v) for k, v in d.items()}


def json_safe_copy(
    obj: Any,
    *,
    max_depth: int = 50,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """Create a JSON-safe deep copy of an object."""
    if _depth > max_depth:
        return None

    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return None

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    _seen.add(obj_id)

    try:
        if isinstance(obj, dict):
            return {
                str(k): json_safe_copy(v, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return [
                json_safe_copy(item, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
                for item in obj
            ]
        if isinstance(obj, set):
            return [
                json_safe_copy(item, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
                for item in sorted(obj, key=lambda x: (type(x).__name__, str(x)))
            ]
        # Try to convert to a basic type
        try:
            return float(obj)
        except Exception:
            pass
        try:
            return str(obj)
        except Exception:
            return None
    finally:
        _seen.discard(obj_id)


def jdump(obj: Any, *, indent: int = 2, default: Any | None = None, **kwargs: Any) -> str:
    """JSON dumps with sensible defaults."""
    def _default(o: Any) -> Any:
        if default is not None:
            try:
                return default(o)
            except Exception:
                pass
        try:
            return float(o)
        except Exception:
            pass
        try:
            return str(o)
        except Exception:
            return None

    return json.dumps(obj, indent=indent, default=_default, ensure_ascii=False, **kwargs)


__all__ = [
    "_dict",
    "coerce_bool",
    "compact_dict",
    "jdump",
    "json_safe_copy",
    "sdict",
]
