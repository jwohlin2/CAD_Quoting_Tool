"""Estimator modules exposed for backwards-compatible imports."""

from . import drilling_legacy
from .drilling_legacy import (
    EstimatorInput,
    SpeedsFeedsUnavailableError,
    estimate,
    legacy_estimate_drilling_hours,
)

__all__ = [
    "drilling_legacy",
    "estimate",
    "EstimatorInput",
    "legacy_estimate_drilling_hours",
    "SpeedsFeedsUnavailableError",
]
