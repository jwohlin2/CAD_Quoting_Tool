from typing import Dict, Any, Tuple
from process_planner import plan_job
from rates import OP_TO_MACHINE, OP_TO_LABOR, rate_for_machine, rate_for_role
import time_models as tm

ATTENDANCE = {  # operator attendance fraction of a labor role during machine ops (optional)
    "WireEDM": 0.20,
    "SinkerEDM": 0.40,
    "CNC_Mill": 0.25,
    "SurfaceGrind": 0.35,
    "JigGrind": 0.60,
    "Waterjet": 0.20,
    "Blanchard": 0.30,
    "DrillPress": 0.50,
}


def minutes_for_op(op: Dict[str, Any], geom: Dict[str, Any]) -> float:
    """Map a planner op to a minute model using geom hints."""
    name = op["op"]

    if name in ("wire_edm_windows", "wire_edm_outline"):
        d = dict(geom.get("wedm", {}))
        # let planner override passes/wire if provided
        if "passes" in op.get("params", op):
            d["passes"] = int(str(op.get("passes", "R+1S").split('+')[-1][0] or 1))
        if "wire_in" in op:
            d["wire_in"] = op["wire_in"]
        return tm.minutes_wedm(d)

    if name in (
        "surface_grind_faces",
        "surface_or_profile_grind_bearing",
        "profile_or_surface_grind_wear_faces",
    ):
        return tm.minutes_surface_grind(geom.get("sg", {}))

    if name == "blanchard_grind_pre":
        return tm.minutes_blanchard(geom.get("blanchard", {}))

    if name in ("cnc_rough_mill", "finish_mill_windows", "thread_mill"):
        return tm.minutes_mill(geom.get("milling", {}))

    if name in ("drill_patterns", "drill_ream_bore", "drill_ream_dowel_press"):
        return tm.minutes_drill(geom.get("drill", []))

    if name == "rigid_tap":
        return tm.minutes_tap(geom.get("tapped_count", 0))

    if name in (
        "jig_bore_or_jig_grind_coaxial_bores",
        "jig_grind_ID_to_size_and_roundness",
        "jig_bore",
    ):
        return tm.minutes_bores(geom.get("bores", []))

    if name in ("sinker_edm_finish_burn",):
        return tm.minutes_sinker(geom.get("sinker", []))

    if name in ("edge_break",):
        return tm.minutes_edgebreak(geom.get("length_ft_edges", 0.0))

    if name in ("lap_bearing_land", "lap_ID"):
        return tm.minutes_lap(geom.get("lap_area_sq_in", 0.0))

    # default small overhead if nothing matched
    return 0.5


def price_with_planner(
    family: str,
    params: Dict[str, Any],
    geom: Dict[str, Any],
    rates: Dict[str, Dict[str, float]],
    oee: float = 0.85,
) -> Dict[str, Any]:
    """
    Run the planner, compute per-op minutes, convert to dollars with machine & labor buckets.
    Returns {plan, line_items:[{op, min, machine$, labor$}], totals:{machine$, labor$, minutes}}
    """
    plan = plan_job(family, params)
    line_items = []
    total_machine = total_labor = total_min = 0.0

    for op in plan["ops"]:
        # uniform access to op params (we allowed dict merging earlier)
        if "params" not in op:
            op["params"] = {k: v for k, v in op.items() if k not in ("op",)}
        minutes = minutes_for_op(op, geom)
        total_min += minutes

        m_cost = l_cost = 0.0
        # Machine $ (OEE as availability penalty)
        if op["op"] in OP_TO_MACHINE:
            mname = OP_TO_MACHINE[op["op"]]
            rate = rate_for_machine(rates, mname)
            m_cost = rate * (minutes / 60.0) / max(0.10, oee)  # divide by OEE to inflate time

            # optional attended labor
            attend = ATTENDANCE.get(mname, 0.0)
            if attend > 0:
                try:
                    lrate = rate_for_role(rates, "EDMOperator" if "EDM" in mname else "Machinist")
                    l_cost += attend * lrate * (minutes / 60.0)
                except KeyError:
                    pass

        # Labor-only $
        if op["op"] in OP_TO_LABOR:
            role = OP_TO_LABOR[op["op"]]
            lrate = rate_for_role(rates, role)
            l_cost += lrate * (minutes / 60.0)

        line_items.append(
            {
                "op": op["op"],
                "minutes": minutes,
                "machine_cost": round(m_cost, 2),
                "labor_cost": round(l_cost, 2),
            }
        )
        total_machine += m_cost
        total_labor += l_cost

    return {
        "plan": plan,
        "line_items": line_items,
        "totals": {
            "minutes": round(total_min, 1),
            "machine_cost": round(total_machine, 2),
            "labor_cost": round(total_labor, 2),
            "total_cost": round(total_machine + total_labor, 2),
        },
    }
