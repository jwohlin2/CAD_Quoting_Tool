"""Shared utility helpers for the CAD quoter application."""
from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


_TRUTHY_STRINGS: frozenset[str] = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSY_STRINGS: frozenset[str] = frozenset({"0", "false", "f", "no", "n", "off"})


def coerce_bool(value: object, *, default: bool | None = None) -> bool | None:
    """Best-effort conversion of *value* into a boolean.

    ``None`` is returned when the input cannot be interpreted as a boolean and no
    *default* is provided. When *default* is supplied, it is returned for unknown
    values (including empty strings).
    """

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        if normalized in _TRUTHY_STRINGS:
            return True
        if normalized in _FALSY_STRINGS:
            return False
    return default


def _dict(value: Mapping[Any, Any] | None) -> dict[Any, Any]:
    """Return *value* if it is a ``dict``; otherwise return an empty dict."""

    return value if isinstance(value, dict) else {}


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


def _callable_label(func: Any) -> str:
    """Return a stable label for ``func`` when it is callable."""

    name = getattr(func, "__name__", None)
    if isinstance(name, str) and name:
        return f"<callable {name}>"
    qualname = getattr(func, "__qualname__", None)
    if isinstance(qualname, str) and qualname:
        return f"<callable {qualname}>"
    return repr(func)


def json_safe_copy(
    obj: Any,
    *,
    max_depth: int = 8,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """Return a JSON-serialisable copy of ``obj`` suitable for debugging."""

    if _seen is None:
        _seen = set()

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("utf-8", errors="replace")

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (datetime, date)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if (
        inspect.ismethod(obj)
        or inspect.isfunction(obj)
        or inspect.isbuiltin(obj)
        or inspect.ismethoddescriptor(obj)
    ):
        return _callable_label(obj)

    if callable(obj):
        return _callable_label(obj)

    if is_dataclass(obj):
        try:
            return json_safe_copy(asdict(obj), max_depth=max_depth, _depth=_depth, _seen=_seen)
        except Exception:
            return str(obj)

    if _depth >= max_depth:
        return str(obj)

    obj_id = id(obj)
    if obj_id in _seen:
        return "<recursion>"

    if isinstance(obj, Mapping):
        _seen.add(obj_id)
        try:
            return {
                str(key): json_safe_copy(value, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
                for key, value in obj.items()
            }
        finally:
            _seen.discard(obj_id)

    if isinstance(obj, (list, tuple, set)):
        _seen.add(obj_id)
        try:
            return [
                json_safe_copy(value, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
                for value in obj
            ]
        finally:
            _seen.discard(obj_id)

    try:
        return float(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return repr(obj)


def jdump(obj: Any, *, indent: int = 2, default: Any | None = str, **kwargs: Any) -> str:
    """Serialize *obj* to JSON using project-wide defaults."""

    dumps_kwargs = {"indent": indent, **kwargs}
    if default is not None:
        dumps_kwargs["default"] = default
    return json.dumps(obj, **dumps_kwargs)


__all__ = ["_dict", "compact_dict", "sdict", "_first_non_none", "jdump", "json_safe_copy"]
