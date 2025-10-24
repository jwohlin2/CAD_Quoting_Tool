"""Application-level helpers for the CAD quoting tool."""
from __future__ import annotations

from .container import ServiceContainer, create_default_container
from . import audit, runner, chart_lines, geo_helpers
from .geo_helpers import (
    _DRILL_REMOVAL_MINUTES_MAX,
    _DRILL_REMOVAL_MINUTES_MIN,
    _SIDE_BACK,
    _SIDE_BOTH,
    _SIDE_FRONT,
    RE_COUNTERDRILL,
    RE_JIG,
    RE_SPOT,
    _COUNTERDRILL_RE,
    _CENTER_OR_SPOT_RE,
    _log_geo_seed_debug,
    _merge_extra_bucket_entries,
    _normalize_extra_bucket_payload,
    _row_side,
    _seed_drill_bins_from_geo,
    _seed_drill_bins_from_geo__local,
    aggregate_ops_from_rows,
)

__all__ = [
    "ServiceContainer",
    "create_default_container",
    "audit",
    "runner",
    "chart_lines",
    "geo_helpers",
    "aggregate_ops_from_rows",
    "_row_side",
    "_seed_drill_bins_from_geo__local",
    "_seed_drill_bins_from_geo",
    "_log_geo_seed_debug",
    "_normalize_extra_bucket_payload",
    "_merge_extra_bucket_entries",
    "_DRILL_REMOVAL_MINUTES_MIN",
    "_DRILL_REMOVAL_MINUTES_MAX",
    "_SIDE_BOTH",
    "_SIDE_BACK",
    "_SIDE_FRONT",
    "RE_COUNTERDRILL",
    "RE_JIG",
    "RE_SPOT",
    "_COUNTERDRILL_RE",
    "_CENTER_OR_SPOT_RE",
]
