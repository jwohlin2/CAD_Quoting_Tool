"""Shared utility helpers for the CAD quoter application."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def compact_dict(d: Mapping[K, V | None]) -> dict[K, V]:
    """Return a copy of *d* without keys whose values are ``None``."""

    return {k: v for k, v in d.items() if v is not None}


def sdict(d: Mapping[Any, Any] | None) -> dict[str, str]:
    """Return a string-keyed/string-valued copy of *d* (ignoring ``None``)."""

    return {str(k): str(v) for k, v in (d or {}).items()}


__all__ = ["compact_dict", "sdict"]
