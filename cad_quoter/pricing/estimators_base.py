"""Shared dataclasses and helpers for estimator plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


class SpeedsFeedsUnavailableError(RuntimeError):
    """Raised when a machining estimator requires a speeds/feeds table."""


@dataclass(slots=True)
class EstimatorInput:
    """Normalized payload consumed by estimator plugins."""

    material_key: str
    geometry: Mapping[str, Any] = field(default_factory=dict)
    material_group: str | None = None
    tables: MutableMapping[str, Any] = field(default_factory=dict)
    coefficients: Mapping[str, Any] = field(default_factory=dict)
    machine_params: Any = None
    overhead_params: Any = None
    warnings: list[str] | None = None
    debug_lines: list[str] | None = None
    debug_summary: dict[str, dict[str, Any]] | None = None

