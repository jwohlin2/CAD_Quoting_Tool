"""Utilities for working with GEO metadata harvested from CAD models."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from cad_quoter.utils.numeric import coerce_positive_float as _coerce_positive_float

logger = logging.getLogger(__name__)

# --- Ops aggregation from HOLE TABLE rows (minimal) -------------------------

_SIDE_BOTH = re.compile(r"\b(FRONT\s*&\s*BACK|BOTH\s+SIDES)\b", re.IGNORECASE)
_SIDE_BACK = re.compile(r"\b(?:FROM\s+)?BACK\b", re.IGNORECASE)
_SIDE_FRONT = re.compile(r"\b(?:FROM\s+)?FRONT\b", re.IGNORECASE)

RE_COUNTERDRILL = re.compile(
    r"\b(?:C[’']\s*DRILL|COUNTER[\s-]*DRILL|CTR\s*DRILL)\b",
    re.IGNORECASE,
)
RE_JIG = re.compile(r"\bJIG\s*GRIND\b", re.IGNORECASE)
RE_SPOT = re.compile(r"\b(?:SPOT|CENTER\s*DRILL)\b", re.IGNORECASE)

_COUNTERDRILL_RE = RE_COUNTERDRILL
_CENTER_OR_SPOT_RE = RE_SPOT


def _row_side(desc: str) -> str | None:
    """Return which side a HOLE TABLE row applies to based on the text."""

    U = (desc or "").upper()
    if _SIDE_BOTH.search(U):
        return "BOTH"
    if _SIDE_BACK.search(U):
        return "BACK"
    if _SIDE_FRONT.search(U):
        return "FRONT"
    return None


def aggregate_ops_from_rows(rows: list[dict]) -> dict:
    """Aggregate lightweight drilling/tapping counters from HOLE TABLE rows."""

    totals: defaultdict[str, int] = defaultdict(int)
    actions: defaultdict[str, int] = defaultdict(int)
    detail: list[dict[str, Any]] = []
    for r in rows or []:
        try:
            qty = int(float(r.get("qty") or 0))
        except Exception:
            qty = 0
        if qty <= 0:
            continue
        desc = str(r.get("desc", ""))
        side = _row_side(desc)
        U = desc.upper()

        if "TAP" in U:
            if side == "BACK":
                totals["tap_back"] += qty
                actions["tap_back"] += qty
            elif side == "BOTH":
                totals["tap_front"] += qty
                totals["tap_back"] += qty
                actions["tap_front"] += qty
                actions["tap_back"] += qty
            else:
                totals["tap_front"] += qty
                actions["tap_front"] += qty
            totals["tap"] += qty
            totals["drill"] += qty
            actions["drill"] += qty

        if (
            "CBORE" in U
            or "C'BORE" in U
            or "COUNTERBORE" in U
            or "COUNTER BORE" in U
        ):
            if side == "BACK":
                totals["counterbore_back"] += qty
                actions["counterbore_back"] += qty
            elif side == "BOTH":
                totals["counterbore_front"] += qty
                totals["counterbore_back"] += qty
                actions["counterbore_front"] += qty
                actions["counterbore_back"] += qty
            else:
                totals["counterbore_front"] += qty
                actions["counterbore_front"] += qty
            totals["counterbore"] += qty

        if (
            "C DRILL" in U
            or "C’DRILL" in U
            or "CENTER DRILL" in U
            or "SPOT DRILL" in U
            or "SPOT" in U
        ):
            if ("THRU" not in U) and ("TAP" not in U):
                totals["spot"] += qty
                actions["spot"] += qty

        if "JIG GRIND" in U:
            totals["jig_grind"] += qty
            actions["jig_grind"] += qty

        detail.append(
            {
                "hole": r.get("hole") or r.get("id") or "",
                "ref": r.get("ref", ""),
                "qty": qty,
                "desc": desc,
            }
        )

    back_ops_total = int(totals.get("counterbore_back", 0) + totals.get("tap_back", 0))
    return {
        "totals": dict(totals),
        "rows": detail,
        "actions_total": int(sum(actions.values())),
        "back_ops_total": back_ops_total,
        "flip_required": bool(back_ops_total > 0),
    }


# --- GEO helpers -------------------------------------------------------------


def _get_core_geo_map(geo_map: Mapping[str, Any] | dict[str, Any] | None) -> dict[str, Any]:
    """Return the dict that actually holds the hole lists / feature counts."""

    if isinstance(geo_map, dict):
        base: dict[str, Any] = geo_map
    elif isinstance(geo_map, Mapping):
        try:
            base = dict(geo_map)
        except Exception:
            base = {}
    else:
        base = {}

    if isinstance(base.get("geo"), dict):
        return base["geo"]

    nested = base.get("geo")
    if isinstance(nested, Mapping):
        try:
            return dict(nested)
        except Exception:
            return {}
    return base


def _seed_drill_bins_from_geo__local(geo_map) -> dict[float, int]:
    """Return {diam_in_rounded: count} from GEO (supports inch + mm + hole_sets)."""

    if not isinstance(geo_map, Mapping):
        return {}

    g = _get_core_geo_map(geo_map)
    diams_in: list[float] = []

    # inch sources
    for key in ("hole_diams_in", "hole_diams_in_precise", "hole_diams_inch"):
        vals = g.get(key)
        if isinstance(vals, (list, tuple)):
            for v in vals:
                try:
                    diams_in.append(float(v))
                except Exception:
                    pass

    # mm sources → inch
    if not diams_in:
        mm_list = g.get("hole_diams_mm_precise") or g.get("hole_diams_mm") or []
        if isinstance(mm_list, (list, tuple)):
            for v in mm_list:
                try:
                    diams_in.append(float(v) / 25.4)
                except Exception:
                    pass

    # hole_sets may carry per-set diameter (in or mm)
    for hs in (g.get("hole_sets") or []):
        if isinstance(hs, Mapping):
            d_in = None
            try:
                if "diam_in" in hs:
                    d_in = float(hs["diam_in"])
                elif "diam_mm" in hs:
                    d_in = float(hs["diam_mm"]) / 25.4
            except Exception:
                d_in = None
            if d_in:
                diams_in.append(d_in)

    bins: dict[float, int] = {}
    for d in diams_in:
        try:
            key = round(float(d), 3)
        except Exception:
            continue
        bins[key] = bins.get(key, 0) + 1

    return bins


def _seed_drill_bins_from_geo(geo: dict) -> dict[float, int]:
    """Return drill-diameter bins from GEO metadata, preferring family counts."""

    out: dict[float, int] = {}
    if not isinstance(geo, Mapping):
        return out

    core_geo = _get_core_geo_map(geo)

    for key in (
        "hole_diam_families_geom_in",
        "hole_diam_families_in",
        "hole_diam_families_geom",
        "hole_diam_families",
    ):
        fam = core_geo.get(key)
        if isinstance(fam, Mapping) and fam:
            for k, v in fam.items():
                try:
                    d = float(str(k).replace('"', "").strip())
                    q = int(v or 0)
                    if q > 0:
                        d = round(d, 4)
                        out[d] = out.get(d, 0) + q
                except Exception:
                    continue
            if out:
                return out

    fallback = _seed_drill_bins_from_geo__local(core_geo)
    if fallback:
        out.update(fallback)
    return out


def _log_geo_seed_debug(lines: list[str], geo: Mapping[str, Any] | dict[str, Any]) -> None:
    """Emit debugging context for GEO maps used to seed drill bins."""

    try:
        geo_obj: Mapping[str, Any]
        if isinstance(geo, Mapping):
            geo_obj = geo
        else:
            geo_obj = {}
    except Exception:
        geo_obj = {}

    def _append(target: list[str] | None, text: str) -> None:
        if isinstance(target, list):
            try:
                target.append(text)
            except Exception:
                logger.debug("geo debug append failed", exc_info=True)
        else:
            logger.debug(text)

    def _seq_len(value: Any) -> int:
        if isinstance(value, (list, tuple, set)):
            return len(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            try:
                return len(list(value))
            except Exception:
                return 0
        return 0

    try:
        try:
            geo_keys = sorted(geo_obj.keys())[:12]
        except Exception:
            geo_keys = sorted((str(key) for key in geo_obj.keys()))[:12]
        _append(lines, f"[DEBUG] geo keys={geo_keys}")
        _append(
            lines,
            f"[DEBUG] hole lists: in={_seq_len(geo_obj.get('hole_diams_in'))} "
            f"mm={_seq_len(geo_obj.get('hole_diams_mm'))}",
        )
    except Exception:
        logger.debug("geo debug emit failed", exc_info=True)


# --- Extra bucket helpers ----------------------------------------------------

_DRILL_REMOVAL_MINUTES_MIN = 0.0
_DRILL_REMOVAL_MINUTES_MAX = 600.0


def _normalize_extra_bucket_payload(
    payload: Mapping[str, Any] | Any,
    *,
    minutes: float | None = None,
) -> dict[str, Any] | None:
    """Normalize an extra bucket payload and enforce canonical fields."""

    if isinstance(payload, Mapping):
        data = dict(payload)
    else:
        try:
            data = dict(payload)  # type: ignore[arg-type]
        except Exception:
            return None

    name_text = str(data.get("name") or data.get("op") or "").strip()
    if not name_text:
        return None

    normalized: dict[str, Any] = {"name": name_text}

    side_raw = data.get("side")
    if side_raw is None:
        side_norm: str | None = None
    elif isinstance(side_raw, str):
        side_norm = side_raw.strip().lower() or None
    else:
        side_norm = str(side_raw).strip().lower() or None
    normalized["side"] = side_norm

    qty_raw = data.get("qty")
    qty_val = 0
    if qty_raw not in (None, ""):
        try:
            qty_val = int(round(float(qty_raw)))
        except Exception:
            try:
                qty_val = int(qty_raw)  # type: ignore[arg-type]
            except Exception:
                qty_val = 0
    normalized["qty"] = max(qty_val, 0)

    if minutes is None:
        minutes_raw = data.get("minutes")
        if minutes_raw in (None, ""):
            minutes_raw = data.get("mins")
        try:
            minutes_val = float(minutes_raw)
        except Exception:
            minutes_val = 0.0
    else:
        minutes_val = float(minutes)

    minutes_val = minutes_val if math.isfinite(minutes_val) else 0.0
    if minutes_val > 0.0:
        normalized["minutes"] = round(minutes_val, 3)

    def _extract_cost(*keys: str) -> float:
        for key in keys:
            cost_val = _coerce_positive_float(data.get(key))
            if cost_val is not None:
                return cost_val
        return 0.0

    machine_val = _extract_cost("machine", "machine_cost")
    labor_val = _extract_cost("labor", "labor_cost")
    total_val = _extract_cost("total", "total_cost")

    if machine_val > 0.0:
        normalized["machine"] = machine_val
    if labor_val > 0.0:
        normalized["labor"] = labor_val
    if total_val > 0.0:
        normalized["total"] = total_val

    return normalized


def _merge_extra_bucket_entries(
    existing: MutableMapping[str, Any],
    incoming: Mapping[str, Any],
) -> None:
    """Merge ``incoming`` values into ``existing`` without double-counting."""

    def _as_int(value: Any) -> int:
        try:
            return int(round(float(value)))
        except Exception:
            try:
                return int(value)  # type: ignore[arg-type]
            except Exception:
                return 0

    def _as_float(value: Any) -> float:
        try:
            numeric = float(value)
        except Exception:
            return 0.0
        return numeric if math.isfinite(numeric) else 0.0

    existing["name"] = str(incoming.get("name") or existing.get("name") or "").strip()

    incoming_side = incoming.get("side")
    if incoming_side is not None:
        existing["side"] = incoming_side

    incoming_qty = _as_int(incoming.get("qty"))
    if incoming_qty > 0:
        existing_qty = _as_int(existing.get("qty"))
        if incoming_qty > existing_qty:
            existing["qty"] = incoming_qty

    incoming_minutes = _as_float(incoming.get("minutes"))
    if incoming_minutes > 0.0:
        existing_minutes = _as_float(existing.get("minutes") or existing.get("mins"))
        if incoming_minutes > existing_minutes:
            existing["minutes"] = round(incoming_minutes, 3)

    for field in ("machine", "labor", "total"):
        incoming_val = _as_float(incoming.get(field))
        if incoming_val <= 0.0:
            continue
        existing_val = _as_float(existing.get(field))
        if incoming_val > existing_val:
            existing[field] = incoming_val

    for alias in ("mins", "machine_cost", "labor_cost", "total_cost"):
        if alias in existing:
            try:
                del existing[alias]
            except Exception:
                pass


__all__ = [
    "_SIDE_BOTH",
    "_SIDE_BACK",
    "_SIDE_FRONT",
    "RE_COUNTERDRILL",
    "RE_JIG",
    "RE_SPOT",
    "_COUNTERDRILL_RE",
    "_CENTER_OR_SPOT_RE",
    "_row_side",
    "aggregate_ops_from_rows",
    "_seed_drill_bins_from_geo__local",
    "_seed_drill_bins_from_geo",
    "_log_geo_seed_debug",
    "_normalize_extra_bucket_payload",
    "_merge_extra_bucket_entries",
    "_DRILL_REMOVAL_MINUTES_MIN",
    "_DRILL_REMOVAL_MINUTES_MAX",
]

