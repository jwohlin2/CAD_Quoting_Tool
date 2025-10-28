"""Guard helpers used during quote rendering."""

from __future__ import annotations

import logging

from collections.abc import Mapping, MutableMapping
from typing import Any, Callable


def render_drilling_guard(
    *,
    logger: logging.Logger,
    jdump: Callable[..., str],
    safe_float: Callable[..., float],
    breakdown: Mapping[str, Any] | None,
    process_plan_summary: Mapping[str, Any] | None,
    bucket_minutes_detail: Mapping[str, Any] | None,
    nre_detail: Mapping[str, Any] | None,
    ladder_subtotal: Any,
) -> None:
    """Log drilling sanity metrics extracted from the render context.

    The helper mirrors the inline guard previously in :func:`appV5.render_quote`
    while keeping the exact logging behaviour and broad ``try``/``except``
    protection.
    """

    try:
        process_plan_map: Mapping[str, Any] | None = None
        for candidate in (process_plan_summary, _resolve_process_plan(breakdown)):
            if isinstance(candidate, Mapping):
                process_plan_map = candidate
                break

        drilling_plan = (
            process_plan_map.get("drilling")
            if isinstance(process_plan_map, Mapping)
            else None
        )
        drill_min_card = safe_float(
            (drilling_plan or {}).get("total_minutes_billed"),
            default=0.0,
        )
        if drill_min_card <= 0.0 and isinstance(drilling_plan, Mapping):
            drill_min_card = safe_float(
                drilling_plan.get("total_minutes_with_toolchange"),
                default=0.0,
            )

        bucket_minutes_map: MutableMapping[str, Any] | None = (
            breakdown.get("bucket_minutes_detail")
            if isinstance(breakdown, Mapping)
            else None
        )
        if not isinstance(bucket_minutes_map, MutableMapping):
            bucket_minutes_map = (
                bucket_minutes_detail if isinstance(bucket_minutes_detail, dict) else {}
            )
        drill_min_row = safe_float(
            (bucket_minutes_map or {}).get("drilling"),
            default=0.0,
        )

        programming_hr = 0.0
        if isinstance(nre_detail, Mapping):
            programming_detail = nre_detail.get("programming")
            if isinstance(programming_detail, Mapping):
                programming_hr = safe_float(
                    programming_detail.get("prog_hr"),
                    default=0.0,
                )

        material_block_dbg = (
            breakdown.get("material") if isinstance(breakdown, Mapping) else None
        )
        material_cost = safe_float(
            (material_block_dbg or {}).get("total_cost"),
            default=0.0,
        )

        direct_costs = safe_float(
            (breakdown if isinstance(breakdown, Mapping) else {}).get(
                "total_direct_costs"
            ),
            default=0.0,
        )

        dbg = {
            "drill_min_card": float(drill_min_card),
            "drill_min_row": float(drill_min_row),
            "programming_hr": float(programming_hr),
            "material_cost": float(material_cost),
            "direct_costs": float(direct_costs),
            "ladder_subtotal": float(ladder_subtotal),
        }
        logger.debug("[render_quote] drill guard snapshot: %s", jdump(dbg, default=None))
    except Exception:
        logger.exception("Failed to run final drilling debug block")


def _resolve_process_plan(
    breakdown: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if isinstance(breakdown, Mapping):
        candidate = breakdown.get("process_plan")
        if isinstance(candidate, Mapping):
            return candidate
    return None
