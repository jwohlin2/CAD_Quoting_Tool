"""Bucket helpers used by quote rendering."""

from __future__ import annotations

from typing import Any

try:  # Python 3.11+: ``collections.abc`` exports ``Mapping``
    from collections.abc import Mapping
except ImportError:  # pragma: no cover - fallback for older versions
    from typing import Mapping  # type: ignore[misc, assignment]

from .state import RenderState


def detect_planner_drilling(planner_buckets: Any) -> bool:
    """Return ``True`` when ``planner_buckets`` contains a drilling bucket."""

    def _walk(candidate: Any, seen: set[int]) -> bool:
        if not isinstance(candidate, Mapping):
            return False
        ident = id(candidate)
        if ident in seen:
            return False
        seen.add(ident)

        try:
            items_iter = candidate.items()
        except Exception:
            return False

        for raw_key, raw_value in items_iter:
            key_text = str(raw_key or "").strip().lower()
            if key_text == "drilling":
                return True
            if key_text == "buckets" and raw_value is not candidate:
                if _walk(raw_value, seen):
                    return True
        return False

    return _walk(planner_buckets, set())


def has_planner_drilling(state: RenderState | Mapping[str, Any] | None) -> bool:
    """Return whether ``state`` includes a drilling bucket allocation."""

    breakdown: Mapping[str, Any] | None
    if isinstance(state, RenderState):
        breakdown = state.breakdown
    elif isinstance(state, Mapping):
        breakdown = state
    else:
        breakdown = None

    if isinstance(breakdown, Mapping):
        bucket_view = breakdown.get("bucket_view")
        if detect_planner_drilling(bucket_view):
            return True
        planner_buckets = breakdown.get("planner_buckets")
        if detect_planner_drilling(planner_buckets):
            return True

    return False


__all__ = [
    "detect_planner_drilling",
    "has_planner_drilling",
]
