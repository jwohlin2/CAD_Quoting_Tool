"""Optional process planner wiring helpers."""
from __future__ import annotations

from typing import Any, Callable

try:
    from process_planner import (
        PLANNERS as _PROCESS_PLANNERS,
    )
    from process_planner import (
        choose_skims as _planner_choose_skims,
    )
    from process_planner import (
        choose_wire_size as _planner_choose_wire_size,
    )
    from process_planner import (
        needs_wedm_for_windows as _planner_needs_wedm_for_windows,
    )
    from process_planner import (
        plan_job as _process_plan_job,
    )
except Exception:  # pragma: no cover - planner is optional at runtime
    _process_plan_job = None
    _PROCESS_PLANNERS = {}
    _planner_choose_wire_size = None
    _planner_choose_skims = None
    _planner_needs_wedm_for_windows = None
else:  # pragma: no cover - defensive guard for unexpected exports
    if not isinstance(_PROCESS_PLANNERS, dict):
        _PROCESS_PLANNERS = {}


_PROCESS_PLANNER_HELPERS: dict[str, Callable[..., Any]] = {}
if "_planner_choose_wire_size" in globals() and callable(_planner_choose_wire_size):
    _PROCESS_PLANNER_HELPERS["choose_wire_size"] = _planner_choose_wire_size  # type: ignore[index]
if "_planner_choose_skims" in globals() and callable(_planner_choose_skims):
    _PROCESS_PLANNER_HELPERS["choose_skims"] = _planner_choose_skims  # type: ignore[index]
if "_planner_needs_wedm_for_windows" in globals() and callable(
    _planner_needs_wedm_for_windows
):
    _PROCESS_PLANNER_HELPERS["needs_wedm_for_windows"] = _planner_needs_wedm_for_windows  # type: ignore[index]

__all__ = [
    "_PROCESS_PLANNERS",
    "_PROCESS_PLANNER_HELPERS",
    "_process_plan_job",
]
