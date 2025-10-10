"""Planner pricing bucketization helpers.

This module provides a lightweight translation layer that maps detailed
planner pricing line items into the sales-facing buckets used by the quote
renderer.  The goal is to keep the planner output transparent while still
presenting totals that align with the traditional quoting experience.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

try:
    from cad_quoter.rates import OP_TO_LABOR, OP_TO_MACHINE, rate_for_role
except ImportError:  # pragma: no cover - legacy path when vendored separately
    from rates import OP_TO_LABOR, OP_TO_MACHINE, rate_for_role

# ---------------------------------------------------------------------------
# Bucket configuration
# ---------------------------------------------------------------------------

# Order matters: the rendered quote should display the buckets in this order.
BUCKETS: tuple[str, ...] = (
    "Programming",
    "Programming (amortized)",
    "Fixture Build",
    "Fixture Build (amortized)",
    "Milling",
    "Drilling",
    "Counterbore",
    "Countersink",
    "Tapping",
    "Saw Waterjet",
    "Wire EDM",
    "Sinker EDM",
    "Grinding",
    "Deburr",
    "Inspection",
)

# Explicit overrides for planner operation keys.  Any operation that does not
# appear here will be routed through the fallback heuristics in
# ``_resolve_bucket_for_op``.
OP_TO_BUCKET: Dict[str, str] = {
    # Milling family
    "cnc_rough_mill": "Milling",
    "finish_mill_windows": "Milling",
    "thread_mill": "Tapping",
    # Drilling family
    "drill_patterns": "Drilling",
    "drill_ream_bore": "Drilling",
    "drill_ream_dowel_press": "Drilling",
    "rigid_tap": "Tapping",
    "counterbore_holes": "Counterbore",
    # Saw / waterjet
    "waterjet_or_saw_blanks": "Saw Waterjet",
    # Grinding
    "surface_grind_faces": "Grinding",
    "surface_or_profile_grind_bearing": "Grinding",
    "profile_or_surface_grind_wear_faces": "Grinding",
    "profile_grind_cutting_edges_and_angles": "Grinding",
    "match_grind_set_for_gap_and_parallelism": "Grinding",
    "blanchard_grind_pre": "Grinding",
    "jig_bore_or_jig_grind_coaxial_bores": "Grinding",
    "jig_grind_ID_to_size_and_roundness": "Grinding",
    "visual_contour_grind": "Grinding",
    # EDM
    "wire_edm_windows": "Wire EDM",
    "wire_edm_outline": "Wire EDM",
    "sinker_edm_finish_burn": "Sinker EDM",
    # Deburr / finishing
    "edge_break": "Deburr",
    "lap_bearing_land": "Deburr",
    "lap_ID": "Deburr",
    "abrasive_flow_polish": "Deburr",
}

# Inspection heuristics.  These provide a simple, explainable estimate for
# inspection when the planner does not emit explicit inspection operations.
INSPECTION_BASE_MIN = 6.0
INSPECTION_PER_OP_MIN = 0.6
INSPECTION_PER_HOLE_MIN = 0.03
INSPECTION_FRACTION_OF_TOTAL = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion that tolerates ``None``/non-numeric values."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_rate_for_role(rates: Dict[str, Dict[str, float]], role: str) -> float:
    try:
        return rate_for_role(rates, role)
    except Exception:
        return float(rates.get("labor", {}).get(role, 0.0))


def _resolve_bucket_for_op(op: str) -> str:
    """Return the display bucket for a planner operation key."""

    if not op:
        return "Milling"

    bucket = OP_TO_BUCKET.get(op)
    if bucket:
        return bucket

    op_lower = op.lower()
    if any(token in op_lower for token in ("counterbore", "c'bore")):
        return "Counterbore"
    if any(token in op_lower for token in ("countersink", "csk")):
        return "Countersink"
    if any(
        token in op_lower
        for token in (
            "rigid_tap",
            "rigid tap",
            "thread_mill",
            "thread mill",
            "tap",
        )
    ):
        return "Tapping"

    machine = OP_TO_MACHINE.get(op, "").lower()
    if machine:
        if "grind" in machine:
            return "Grinding"
        if "edm" in machine:
            return "Wire EDM" if "wire" in machine else "Sinker EDM"
        if "counterbore" in machine:
            return "Counterbore"
        if "countersink" in machine or "csk" in machine:
            return "Countersink"
        if "drill" in machine:
            return "Drilling"
        if "tap" in machine:
            return "Tapping"
        if "waterjet" in machine or "saw" in machine:
            return "Saw Waterjet"

    role = OP_TO_LABOR.get(op, "").lower()
    if role:
        if "inspect" in role:
            return "Inspection"
        if any(token in role for token in ("deburr", "lap", "finish")):
            return "Deburr"

    return "Milling"


def _iter_geom_list(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set)):
        return value
    return ()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bucketize(
    planner_pricing: Dict[str, Any],
    rates_two_bucket: Dict[str, Dict[str, float]],
    nre: Dict[str, float],
    *,
    qty: int,
    geom: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate planner pricing into user-facing cost buckets."""

    qty_int = int(qty) if isinstance(qty, (int, float)) else 1
    if qty_int <= 0:
        qty_int = 1

    buckets: Dict[str, Dict[str, float]] = {
        name: {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0}
        for name in BUCKETS
    }

    def add(bucket: str, minutes: float, machine_cost: float, labor_cost: float) -> None:
        entry = buckets.setdefault(
            bucket, {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0}
        )
        entry["minutes"] += minutes
        entry["machine$"] += machine_cost
        entry["labor$"] += labor_cost
        entry["total$"] += machine_cost + labor_cost

    line_items = planner_pricing.get("line_items", [])
    for li in line_items if isinstance(line_items, list) else []:
        if not isinstance(li, dict):
            continue
        op_key = str(li.get("op") or "")
        minutes = _as_float(li.get("minutes"))
        machine_cost = _as_float(li.get("machine_cost"))
        labor_cost = _as_float(li.get("labor_cost"))
        bucket_name = _resolve_bucket_for_op(op_key)
        add(bucket_name, minutes, machine_cost, labor_cost)

    programming_min = _as_float(nre.get("programming_min"))
    fixture_min = _as_float(nre.get("fixture_min"))

    programmer_rate = (
        _safe_rate_for_role(rates_two_bucket, "Programmer")
        or _safe_rate_for_role(rates_two_bucket, "Engineer")
    )
    if programming_min > 0:
        labor_cost = programmer_rate * (programming_min / 60.0)
        add("Programming", programming_min, 0.0, labor_cost)
        if qty_int > 1:
            per_min = programming_min / qty_int
            add(
                "Programming (amortized)",
                per_min,
                0.0,
                programmer_rate * (per_min / 60.0),
            )

    fixture_rate = (
        _safe_rate_for_role(rates_two_bucket, "FixtureBuilder")
        or _safe_rate_for_role(rates_two_bucket, "Toolmaker")
        or _safe_rate_for_role(rates_two_bucket, "Machinist")
    )
    if fixture_min > 0:
        labor_cost = fixture_rate * (fixture_min / 60.0)
        add("Fixture Build", fixture_min, 0.0, labor_cost)
        if qty_int > 1:
            per_min = fixture_min / qty_int
            add(
                "Fixture Build (amortized)",
                per_min,
                0.0,
                fixture_rate * (per_min / 60.0),
            )

    hole_features = list(_iter_geom_list(geom.get("drill")))
    tapped_count = int(_as_float(geom.get("tapped_count")))
    cbore_entries = list(_iter_geom_list(geom.get("counterbore")))
    op_count = sum(1 for _ in line_items if isinstance(line_items, list))

    inspection_min = INSPECTION_BASE_MIN
    inspection_min += op_count * INSPECTION_PER_OP_MIN
    inspection_min += (len(hole_features) + tapped_count + len(cbore_entries)) * INSPECTION_PER_HOLE_MIN

    total_process_minutes = sum(entry["minutes"] for entry in buckets.values())
    floor_minutes = total_process_minutes * INSPECTION_FRACTION_OF_TOTAL
    inspection_min = max(inspection_min, floor_minutes)

    inspector_rate = _safe_rate_for_role(rates_two_bucket, "Inspector")
    add("Inspection", inspection_min, 0.0, inspector_rate * (inspection_min / 60.0))

    cleaned_buckets: Dict[str, Dict[str, float]] = {}
    totals = {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0, "total$": 0.0}

    for name in BUCKETS:
        entry = buckets.get(name)
        if not entry:
            continue
        if entry["minutes"] <= 0.01 and abs(entry["total$"]) <= 0.01:
            continue
        rounded = {
            "minutes": round(entry["minutes"], 2),
            "machine$": round(entry["machine$"], 2),
            "labor$": round(entry["labor$"], 2),
            "total$": round(entry["total$"], 2),
        }
        cleaned_buckets[name] = rounded
        totals["minutes"] += rounded["minutes"]
        totals["machine$"] += rounded["machine$"]
        totals["labor$"] += rounded["labor$"]
        totals["total$"] += rounded["total$"]

    totals = {key: round(value, 2) for key, value in totals.items()}

    return {"buckets": cleaned_buckets, "totals": totals}

