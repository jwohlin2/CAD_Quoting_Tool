"""Plugin adapter for the drilling time estimator."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from cad_quoter.estimators.base import EstimatorInput

import appV5 as _legacy


def _resolve_speeds_feeds_table(tables: Mapping[str, Any] | None) -> pd.DataFrame | None:
    if not tables:
        return None
    try:
        table = tables.get("speeds_feeds")
    except AttributeError:
        try:
            table = tables["speeds_feeds"]  # type: ignore[index]
        except Exception:
            table = None
    return table if isinstance(table, pd.DataFrame) else None


def estimate(input_data: EstimatorInput) -> float:
    """Estimate drilling hours using the legacy implementation."""

    geometry = dict(input_data.geometry or {})
    hole_diams_mm = list(geometry.get("hole_diams_mm") or [])
    thickness_raw = geometry.get("thickness_in", 0.0)
    try:
        thickness_in = float(thickness_raw)
    except (TypeError, ValueError):
        thickness_in = 0.0
    raw_groups = geometry.get("hole_groups")
    groups = None
    if isinstance(raw_groups, list):
        groups = [g for g in raw_groups if isinstance(g, Mapping)]
        if not groups:
            groups = None
    speeds_feeds_table = _resolve_speeds_feeds_table(input_data.tables)

    return _legacy._legacy_estimate_drilling_hours(
        hole_diams_mm,
        thickness_in,
        input_data.material_key,
        material_group=input_data.material_group,
        hole_groups=groups,
        speeds_feeds_table=speeds_feeds_table,
        machine_params=input_data.machine_params,
        overhead_params=input_data.overhead_params,
        warnings=input_data.warnings,
        debug_lines=input_data.debug_lines,
        debug_summary=input_data.debug_summary,
    )
