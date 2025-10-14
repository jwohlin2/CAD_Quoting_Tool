"""Compatibility helpers for optional overhead index support."""
from __future__ import annotations

from dataclasses import asdict, fields as dataclass_fields, is_dataclass
from types import SimpleNamespace
from typing import Any, TypeAlias

from cad_quoter.pricing.time_estimator import (
    OverheadParams as _TimeOverheadParams,
)

__all__ = [
    "_TIME_OVERHEAD_SUPPORTS_INDEX_SEC",
    "OverheadLike",
    "_assign_overhead_index_attr",
    "_ensure_overhead_index_attr",
]


try:
    _TIME_OVERHEAD_FIELD_NAMES = {
        field.name for field in dataclass_fields(_TimeOverheadParams)
    }
except Exception:  # pragma: no cover - defensive against non-dataclass implementations
    _TIME_OVERHEAD_FIELD_NAMES = set()


def _detect_index_support() -> bool:
    """Return True when ``OverheadParams`` exposes ``index_sec_per_hole``."""

    if "index_sec_per_hole" in _TIME_OVERHEAD_FIELD_NAMES:
        return True

    try:
        probe = _TimeOverheadParams()
    except Exception:
        # If the constructor requires additional parameters we conservatively
        # assume the optional index field is unavailable.
        return False

    if hasattr(probe, "index_sec_per_hole"):
        return True

    for setter in (setattr, object.__setattr__):
        try:
            setter(probe, "index_sec_per_hole", None)
        except Exception:
            continue
        else:
            return hasattr(probe, "index_sec_per_hole")

    return False


_TIME_OVERHEAD_SUPPORTS_INDEX_SEC = _detect_index_support()


OverheadLike: TypeAlias = _TimeOverheadParams | SimpleNamespace


def _assign_overhead_index_attr(
    overhead: _TimeOverheadParams, index_value: float | None
) -> bool:
    """Attempt to assign ``index_sec_per_hole`` on ``overhead``.

    Returns ``True`` when the attribute exists (either pre-existing or after
    assignment) and ``False`` when the underlying dataclass does not support
    the field.
    """

    if overhead is None:
        return False

    if hasattr(overhead, "index_sec_per_hole"):
        try:
            setattr(overhead, "index_sec_per_hole", index_value)
        except Exception:
            try:
                object.__setattr__(overhead, "index_sec_per_hole", index_value)
            except Exception:
                # Attribute exists but cannot be mutated (e.g. frozen
                # dataclass). Treat as available so downstream callers rely on
                # the existing value.
                return True
        return True

    for setter in (setattr, object.__setattr__):
        try:
            setter(overhead, "index_sec_per_hole", index_value)
        except Exception:
            continue
        else:
            return True

    return False


def _ensure_overhead_index_attr(
    overhead: _TimeOverheadParams, index_value: float | None, *, assigned: bool = False
) -> OverheadLike:
    """Return an overhead object that always exposes ``index_sec_per_hole``."""

    if hasattr(overhead, "index_sec_per_hole"):
        if not assigned:
            _assign_overhead_index_attr(overhead, index_value)
        return overhead

    payload: dict[str, Any] = {}
    if is_dataclass(overhead):
        try:
            payload = asdict(overhead)
        except Exception:
            payload = {}

    if not payload:
        for name in (
            "toolchange_min",
            "approach_retract_in",
            "peck_penalty_min_per_in_depth",
            "dwell_min",
            "peck_min",
        ):
            if hasattr(overhead, name):
                payload[name] = getattr(overhead, name)

    payload.setdefault("index_sec_per_hole", index_value)
    try:
        return SimpleNamespace(**payload)
    except Exception:
        return overhead
