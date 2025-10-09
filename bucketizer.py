# bucketizer.py
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from rates import OP_TO_MACHINE, OP_TO_LABOR, rate_for_machine, rate_for_role

# ---- Sales-facing bucket names ----
BUCKETS = [
    "Fixture Build", "Programming",
    "Milling", "Drilling", "Counterbore", "Tapping",
    "Saw Waterjet", "Grinding",
    "Wire EDM", "Sinker EDM",               # (optional but very common)
    "Deburr", "Inspection",
    "Fixture Build Amortized", "Programming Amortized",
]

# ---- Map planner ops → buckets (extend as needed) ----
OP_TO_BUCKET = {
    # Milling family
    "cnc_rough_mill": "Milling",
    "finish_mill_windows": "Milling",
    "thread_mill": "Tapping",

    # Drilling family
    "drill_patterns": "Drilling",
    "drill_ream_bore": "Drilling",
    "drill_ream_dowel_press": "Drilling",
    "rigid_tap": "Tapping",
    # If you emit a dedicated counterbore op, map it:
    "counterbore_holes": "Counterbore",

    # Saw/Waterjet
    "waterjet_or_saw_blanks": "Saw Waterjet",

    # Grinding (roll surface/jig/profile/blanchard together)
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

# ---- Inspection model (transparent & tunable) ----
INSPECTION_BASE_MIN = 6.0            # job setup/first article paperwork
INSPECTION_PER_OP_MIN = 0.6          # per distinct operation in plan
INSPECTION_PER_HOLE_MIN = 0.03       # per hole feature (drill/tap/ream/cbore)
INSPECTION_FRACTION_OF_TOTAL = 0.05  # 5% of total minutes as a floor/ceiling helper

def _safe(n, d=0.0): return n if isinstance(n, (int, float)) else d

def bucketize(
    planner_pricing: Dict[str, Any],
    rates_two_bucket: Dict[str, Dict[str, float]],
    nre: Dict[str, float],
    qty: int,
    geom: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Roll planner line items into sales-facing buckets.
    Inputs:
      planner_pricing: result of price_with_planner(...). Must include "line_items".
      rates_two_bucket: {"labor": {...}, "machine": {...}} (migrate_flat_to_two_bucket output)
      nre: {"programming_min": float, "fixture_min": float}  # raw NRE minutes for the whole job
      qty: production quantity (for amortization)
      geom: geometry dict (for inspection estimation; uses drill/tap counts if present)
    Returns:
      {
        "buckets": { name: {"minutes": m, "labor$": L, "machine$": M, "total$": T}, ... },
        "totals":  {"minutes": m, "labor$": L, "machine$": M, "total$": T}
      }
    """

    buckets = {b: {"minutes": 0.0, "labor$": 0.0, "machine$": 0.0, "total$": 0.0} for b in BUCKETS}

    def add(bucket: str, minutes: float, machine_cost: float = 0.0, labor_cost: float = 0.0):
        b = buckets[bucket]
        b["minutes"] += minutes
        b["machine$"] += machine_cost
        b["labor$"]  += labor_cost
        b["total$"]  += machine_cost + labor_cost

    line_items: List[Dict[str, Any]] = planner_pricing.get("line_items", [])
    total_minutes = 0.0
    total_machine = 0.0
    total_labor = 0.0

    # 1) Allocate planner line items to buckets
    for li in line_items:
        op = li.get("op", "")
        minutes = float(li.get("minutes", 0.0))
        m$ = float(li.get("machine_cost", 0.0))
        l$ = float(li.get("labor_cost", 0.0))
        bucket = OP_TO_BUCKET.get(op)
        if not bucket:
            # Unmapped op: put under Grinding if machine is a grinder, else create a Misc bucket
            if op in ("jig_bore",): bucket = "Grinding"
            else: bucket = "Milling"  # safe generic
        add(bucket, minutes, m$, l$)
        total_minutes += minutes
        total_machine += m$
        total_labor += l$

    # 2) Add NRE (Programming & Fixture Build) buckets + Amortized buckets
    prog_min = float(nre.get("programming_min", 0.0))
    fixt_min = float(nre.get("fixture_min", 0.0))

    if prog_min > 0:
        prog_rate = rates_two_bucket["labor"].get("Programmer") or rates_two_bucket["labor"].get("Engineer")
        prog_cost = (prog_rate or 0.0) * (prog_min/60.0)
        add("Programming", prog_min, 0.0, prog_cost)

        if qty and qty > 0:
            per_min = prog_min / qty
            per_cost = (prog_rate or 0.0) * (per_min/60.0)
            add("Programming Amortized", per_min, 0.0, per_cost)

    if fixt_min > 0:
        tool_rate = rates_two_bucket["labor"].get("FixtureBuilder") or rates_two_bucket["labor"].get("Toolmaker") or rates_two_bucket["labor"].get("Machinist")
        fixt_cost = (tool_rate or 0.0) * (fixt_min/60.0)
        add("Fixture Build", fixt_min, 0.0, fixt_cost)

        if qty and qty > 0:
            per_min = fixt_min / qty
            per_cost = (tool_rate or 0.0) * (per_min/60.0)
            add("Fixture Build Amortized", per_min, 0.0, per_cost)

    # 3) Transparent Inspection estimate
    # If you already compute inspection ops elsewhere, you can skip this and add your own "Inspection" items.
    holes = geom.get("drill", []) or []
    tapped_n = int(geom.get("tapped_count", 0))
    cbore_n = len(geom.get("counterbore", [])) if isinstance(geom.get("counterbore"), list) else 0
    n_ops = len(line_items)
    insp_min = (
        INSPECTION_BASE_MIN
        + n_ops * INSPECTION_PER_OP_MIN
        + (len(holes) + tapped_n + cbore_n) * INSPECTION_PER_HOLE_MIN
    )
    # floor/ceiling helper as a fraction of total run minutes
    insp_min = max(insp_min, total_minutes * INSPECTION_FRACTION_OF_TOTAL)
    insp_rate = rates_two_bucket["labor"].get("Inspector", 0.0)
    insp_cost = insp_rate * (insp_min/60.0)
    add("Inspection", insp_min, 0.0, insp_cost)

    # 4) Totals
    out = {"buckets": {}, "totals": {}}
    # prune empty buckets to keep the quote clean
    for k, v in buckets.items():
        if v["minutes"] > 0.01 or v["total$"] > 0.01:
            out["buckets"][k] = {
                "minutes": round(v["minutes"], 2),
                "labor$": round(v["labor$"], 2),
                "machine$": round(v["machine$"], 2),
                "total$": round(v["total$"], 2),
            }

    out["totals"] = {
        "minutes": round(total_minutes + _safe(prog_min/qty,0) + _safe(fixt_min/qty,0) + round(insp_min,2), 2),
        "machine$": round(total_machine, 2),
        "labor$": round(total_labor + insp_cost
                        + (rates_two_bucket["labor"].get("Programmer",0.0) * ((prog_min/qty)/60.0) if qty else 0.0)
                        + ( (rates_two_bucket["labor"].get("FixtureBuilder",0.0) or rates_two_bucket["labor"].get("Toolmaker",0.0)) * ((fixt_min/qty)/60.0) if qty else 0.0)
                        , 2),
        "total$": 0.0  # fill in by caller if you also want to add NRE raw buckets into order total
    }
    out["totals"]["total$"] = round(out["totals"]["machine$"] + out["totals"]["labor$"], 2)
    return out
