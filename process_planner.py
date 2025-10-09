"""Process planning rules engine for machining families.

This module provides helper functions and planner entry points for several
process families used in the CAD quoting tool.  Each planner returns a
dictionary containing a list of operations (`ops`) as well as notes for
fixturing, quality assurance, and warnings.  The logic is based on the rules
supplied in the Composidie EDM and grinding reference playbook.
"""

from math import isfinite
from typing import Any, Dict, List, Optional

Op = Dict[str, Any]
Plan = Dict[str, Any]


# -----------------------------
# Shared helpers / guardrails
# -----------------------------


def _safe(v, default=None):
    return v if v is not None else default


def choose_wire_size(
    min_inside_radius: Optional[float], min_feature_width: Optional[float]
) -> float:
    """Return wire diameter (inches)."""

    mir = _safe(min_inside_radius, 1.0)
    mfw = _safe(min_feature_width, 1.0)
    wire = 0.010
    if mir <= 0.005 or mfw <= 0.020:
        wire = 0.008
    if mir <= 0.0035 or mfw <= 0.012:
        wire = 0.006
    return wire


def choose_skims(profile_tol: Optional[float]) -> int:
    """Return number of skim passes for WEDM."""

    pt = _safe(profile_tol, 0.0015)
    if pt <= 0.0002:
        return 3
    if pt <= 0.0003:
        return 2
    if pt <= 0.0005:
        return 1
    return 0


def needs_wedm_for_windows(
    windows_need_sharp: bool,
    window_corner_radius_req: Optional[float],
    profile_tol: Optional[float],
) -> bool:
    r = _safe(window_corner_radius_req, 999)
    t = _safe(profile_tol, 999)
    return bool(windows_need_sharp or r <= 0.030 or t <= 0.001)


def add(plan: Plan, op: str, **params) -> None:
    plan["ops"].append({"op": op, **({k: v for k, v in params.items() if v is not None})})


_DIRECT_KEYS = (
    "hardware",
    "outsourced",
    "utilities",
    "consumables_flat",
    "packaging_flat",
)


def base_plan() -> Plan:
    return {"ops": [], "fixturing": [], "qa": [], "warnings": [], "directs": {}}


def _init_directs(plan: Plan, **initial: bool) -> None:
    directs = {key: False for key in _DIRECT_KEYS}
    directs.update({k: bool(v) for k, v in initial.items() if k in directs})
    plan["directs"] = directs


def add_direct(plan: Plan, key: str, value=True) -> None:
    plan.setdefault("directs", {})[key] = value


def warn(plan: Plan, msg: str) -> None:
    plan["warnings"].append(msg)


def add_fixturing(plan: Plan, note: str) -> None:
    plan["fixturing"].append(note)


def add_qa(plan: Plan, note: str) -> None:
    plan["qa"].append(note)


# -----------------------------
# FAMILY 1: DIE PLATES / SHOES / FLATS / PROFILES
# -----------------------------


def plan_die_plate(p: Dict[str, Any]) -> Plan:
    """Generate a plan for die plates, shoes, flats, or profiles."""

    plan = base_plan()
    _init_directs(plan)

    material = p.get("material", "4140PH")
    L, W = p.get("plate_LxW", (0, 0))
    incoming = p.get("incoming_cut", "saw")
    flatness_spec = p.get("flatness_spec")
    parallelism_spec = p.get("parallelism_spec")
    profile_tol = p.get("profile_tol")
    windows_need_sharp = bool(p.get("windows_need_sharp", False))
    corner_r = p.get("window_corner_radius_req")
    stress_risk = p.get("stress_relief_risk", "low")

    # A) Stability
    if incoming == "flame" or stress_risk != "low":
        add(plan, "stress_relieve")
        add_fixturing(plan, "Use leveling blocks; record height change pre/post SR.")

    # B) Faces (Blanchard vs face-mill + final SG)
    large_plate = max(L, W) > 10
    tight_flat = (flatness_spec is not None) and (flatness_spec <= 0.001)
    if large_plate or tight_flat:
        add(plan, "blanchard_grind_pre", leave_total=0.010)
    else:
        add(plan, "face_mill_pre")

    # C) Rough CNC
    add(plan, "cnc_rough_mill", stock_wall=0.010, stock_floor=0.010)
    add(plan, "spot_drill_all")
    add(plan, "drill_patterns")
    add(plan, "interpolate_critical_bores", undersize=0.010)

    # D) Windows / profiles (WEDM vs finish-mill)
    if needs_wedm_for_windows(windows_need_sharp, corner_r, profile_tol):
        wire = choose_wire_size(corner_r, None)
        skims = choose_skims(profile_tol)
        add(plan, "wire_edm_windows", wire_in=wire, passes=f"R+{skims}S", tab_slugs=True)
        add_fixturing(plan, "Tab slugs; cut tabs after skim passes to minimize movement.")
    else:
        add(plan, "finish_mill_windows")

    # E) Bores & fits
    for h in p.get("hole_sets", []):
        htype = h.get("type")
        if htype in {"post_bore", "bushing_seat"}:
            tol = h.get("tol", h.get("od_tol", 0.001))
            if tol <= 0.0005 or h.get("coax_pair_id"):
                add(plan, "assemble_pair_on_fixture")
                add(plan, "jig_bore_or_jig_grind_coaxial_bores", tol=tol)
                add_qa(plan, "Check coaxiality/runout over assembled height.")
            else:
                add(plan, "drill_ream_bore", tol=tol)
        elif htype == "dowel_press":
            add(plan, "drill_ream_dowel_press")
        elif htype == "dowel_slip":
            add(plan, "ream_slip_in_assembly")
            add_qa(plan, "Verify slip-fit location after assembly ream.")
        elif htype == "tapped":
            dia = h.get("dia", 0.0)
            depth = h.get("depth", 0.0)
            use_tm = (dia >= 0.5) or (depth > 1.5 * dia)
            add(plan, "thread_mill" if use_tm else "rigid_tap", dia=dia, depth=depth)

    # F) Final faces
    if (flatness_spec is not None) or (parallelism_spec is not None):
        add(
            plan,
            "surface_grind_faces",
            flatness_spec=flatness_spec,
            parallelism_spec=parallelism_spec,
        )
        add_fixturing(plan, "Mag-chuck with ground backer; flip and relieve to avoid dish.")

    # G) Stress loop check
    add(plan, "stability_check_after_ops")
    add_qa(plan, "Granite flatness map; CMM datums A/B/C; plug/ID mics on bores.")
    add(plan, "edge_break", size=0.010)
    if p.get("marking", "laser") != "none":
        add(plan, "mark_id", method=p.get("marking", "laser"))

    hole_types = {h.get("type") for h in p.get("hole_sets", [])}
    if {"bushing_seat", "post_bore", "dowel_press", "dowel_slip"} & hole_types:
        add_direct(plan, "hardware", True)

    opnames = [o.get("op") for o in plan.get("ops", [])]
    if any(str(name).startswith("heat_treat") for name in opnames):
        add_direct(plan, "outsourced", True)

    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True

    return plan


# -----------------------------
# FAMILY 2: PUNCHES & INSERTS (steel/carbide; flat/form/step/comb)
# -----------------------------


def plan_punch(p: Dict[str, Any]) -> Plan:
    """Generate a plan for punches and inserts."""

    plan = base_plan()
    _init_directs(plan)

    mat = p.get("material", "tool_steel_annealed")
    overall_L = p.get("overall_length", 1.0)
    mfw = p.get("min_feature_width", 1.0)
    mir = p.get("min_inside_radius", 1.0)
    profile_tol = p.get("profile_tol", 0.0005)
    blind = bool(p.get("blind_relief", False))

    # Material route
    if mat == "carbide":
        add(plan, "start_ground_carbide_blank")
    elif mat == "tool_steel_annealed":
        add(plan, "saw_blank")
        add(plan, "cnc_mill_rough", stock=0.020)
        add(plan, "surface_grind_datums")
        add(plan, "heat_treat_to_spec")
    else:  # tool_steel_HT
        add(plan, "indicate_hardened_blank")

    # Slenderness & carrier
    slender = overall_L / max(mfw, 1e-6)
    if slender >= 20 or mfw <= 0.020:
        add_fixturing(plan, "Keep part tabbed to carrier for WEDM; support during grind.")
        add(plan, "prep_carrier_or_tab")

    # Wire EDM
    wire = choose_wire_size(mir, mfw)
    skims = choose_skims(profile_tol)
    add(
        plan,
        "wire_edm_outline",
        wire_in=wire,
        passes=f"R+{skims}S",
        tabbed=(slender >= 20 or mfw <= 0.020),
    )

    # Blind features via sinker
    if blind:
        elec = "copper" if (mir <= 0.005 or profile_tol <= 0.0003) else "graphite"
        add(plan, "machine_electrode", material=elec)
        add(plan, "sinker_edm_finish_burn")

    # Bearing lands
    bl = p.get("bearing_land_spec")
    if bl:
        add(
            plan,
            "surface_or_profile_grind_bearing",
            land_width=bl.get("width"),
            target_Ra=bl.get("Ra"),
        )
        add(plan, "lap_bearing_land", target_Ra=bl.get("Ra"))
    else:
        add(plan, "light_grind_cleanup")

    # Pilots & runout
    runout = p.get("runout_to_shank")
    if runout is not None and runout <= 0.0003:
        add(plan, "indicate_on_shank")
        add(plan, "profile_grind_pilot_OD_to_TIR", TIR=runout)
        add_qa(plan, "Check pilot OD TIR to shank on V-blocks or between centers.")

    # Edge & coat
    edge = p.get("edge_condition", "0.001R")
    if edge != "sharp":
        add(plan, "edge_prep", spec=edge)
    if p.get("coating") not in (None, "none"):
        add(plan, "clean_degas_for_coating")
        add(plan, "apply_coating", type=p["coating"])

    # QA
    add_qa(plan, "Comparator profile vs CAD; measure land width & size; hardness if steel.")

    opnames = [o.get("op") for o in plan.get("ops", [])]
    if any(str(name).startswith("heat_treat") for name in opnames):
        add_direct(plan, "outsourced", True)

    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True

    hardware_hints = (
        p.get("include_hardware"),
        p.get("hardware_items"),
        p.get("bom_items"),
    )
    if any(bool(hint) for hint in hardware_hints):
        add_direct(plan, "hardware", True)

    return plan


# -----------------------------
# FAMILY 3: PILOT PUNCHES (wrapper over punches with concentricity)
# -----------------------------


def plan_pilot_punch(p: Dict[str, Any]) -> Plan:
    p2 = dict(p)
    p2.setdefault("runout_to_shank", 0.0003)
    return plan_punch(p2)


# -----------------------------
# FAMILY 4: GUIDE BUSHINGS / RING GAUGES (ID-critical)
# -----------------------------


def plan_bushing_id_critical(p: Dict[str, Any]) -> Plan:
    """Generate a plan for ID critical guide bushings or ring gauges."""

    plan = base_plan()
    _init_directs(plan)
    if p.get("tight_od", True):
        add(plan, "purchase_OD_ground_blank")
        warn(
            plan,
            "No cylindrical/centerless grinder listed; outsource OD if OD is critical.",
        )
    else:
        add(plan, "turn_or_mill_OD")
        add(plan, "surface_or_profile_grind_OD_cleanup")

    if p.get("create_id_by_wire_first", False):
        add(plan, "wire_edm_open_ID", leave=0.005)
    else:
        add(plan, "drill_or_trepan_ID", leave=0.005)

    add(plan, "jig_grind_ID_to_size_and_roundness", tol=0.0002)
    if p.get("target_id_Ra", 16) <= 8:
        add(plan, "lap_ID", target_Ra=p.get("target_id_Ra", 8))

    add_qa(plan, "Calibrate ID with ring/plug masters; certify size & roundness.")
    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True

    hardware_flags = (
        p.get("include_hardware"),
        p.get("retainer_spec"),
        p.get("hardware_items"),
    )
    if any(bool(flag) for flag in hardware_flags):
        add_direct(plan, "hardware", True)
    return plan


# -----------------------------
# FAMILY 5: CAMS & HEMMING COMPONENTS
# -----------------------------


def plan_cam_or_hemmer(p: Dict[str, Any]) -> Plan:
    """Generate a plan for cams and hemming components."""

    plan = base_plan()
    _init_directs(plan)
    add(plan, "saw_or_mill_rough_blocks")
    if p.get("material", "tool_steel") in {"D2", "A2", "PM", "tool_steel"}:
        add(plan, "heat_treat_if_wear_part")

    if needs_wedm_for_windows(p.get("windows_need_sharp", False), 0.0, p.get("profile_tol")):
        wire = choose_wire_size(0.0, None)
        skims = choose_skims(p.get("profile_tol"))
        add(plan, "wire_edm_cam_slot_or_profile", wire_in=wire, passes=f"R+{skims}S")
    else:
        add(plan, "finish_mill_cam_slot_or_profile")

    add(plan, "profile_or_surface_grind_wear_faces")
    add(plan, "jig_bore_or_grind_pivot_bores", tol=0.0005)
    add_qa(plan, "Inspect cam path size/position; verify bore TIR and hardness.")

    opnames = [o.get("op") for o in plan.get("ops", [])]
    if any(str(name).startswith("heat_treat") for name in opnames):
        add_direct(plan, "outsourced", True)

    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True

    return plan


# -----------------------------
# FAMILY 6: SPECIALS
# -----------------------------


def plan_flat_die_chaser(p: Optional[Dict[str, Any]] = None) -> Plan:
    plan = base_plan()
    _init_directs(plan)
    add(plan, "mill_or_wire_rough_form")
    add(plan, "heat_treat")
    add(plan, "profile_grind_flanks_and_reliefs_to_spec")
    add(plan, "lap_edges")
    add_qa(plan, "Comparator flank angle/lead; hardness; edge condition.")

    opnames = [o.get("op") for o in plan.get("ops", [])]
    if any(str(name).startswith("heat_treat") for name in opnames):
        add_direct(plan, "outsourced", True)

    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True

    return plan


def plan_pm_compaction_die(p: Dict[str, Any]) -> Plan:
    plan = base_plan()
    _init_directs(plan)
    add(plan, "start_ground_carbide_ring")
    add(plan, "wire_edm_ID_leave", leave=0.005)
    add(plan, "jig_grind_ID_to_tenths_and_straightness", tol=0.0001)
    add(plan, "lap_bearing_land", target_Ra=8)
    add_qa(plan, "Measure taper/straightness over depth; Ra on land.")
    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True
    return plan


def plan_shear_blade(p: Optional[Dict[str, Any]] = None) -> Plan:
    plan = base_plan()
    _init_directs(plan)
    add(plan, "waterjet_or_saw_blanks")
    add(plan, "heat_treat", material="A2/D2/PM")
    add(plan, "profile_grind_cutting_edges_and_angles")
    add(plan, "match_grind_set_for_gap_and_parallelism")
    add(plan, "hone_edge")
    add_qa(plan, "Parallelism & edge angle match; hardness.")

    opnames = [o.get("op") for o in plan.get("ops", [])]
    if any(str(name).startswith("heat_treat") for name in opnames):
        add_direct(plan, "outsourced", True)

    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True

    return plan


def plan_extrude_hone(p: Dict[str, Any]) -> Plan:
    plan = base_plan()
    _init_directs(plan)
    add(plan, "verify_connected_passage_and_masking")
    add(plan, "abrasive_flow_polish", target_Ra=_safe(p.get("target_Ra"), 16))
    add(plan, "clean_and_flush_media")
    add_qa(plan, "Flow/pressure delta or Ra before/after report.")
    if plan.get("ops"):
        plan["directs"]["utilities"] = True
        plan["directs"]["consumables_flat"] = True
    return plan


# -----------------------------
# Convenience: registry & router
# -----------------------------


PLANNERS = {
    "die_plate": plan_die_plate,
    "punch": plan_punch,
    "pilot_punch": plan_pilot_punch,
    "bushing_id_critical": plan_bushing_id_critical,
    "cam_or_hemmer": plan_cam_or_hemmer,
    "flat_die_chaser": plan_flat_die_chaser,
    "pm_compaction_die": plan_pm_compaction_die,
    "shear_blade": plan_shear_blade,
    "extrude_hone": plan_extrude_hone,
}


def plan_job(family: str, params: Dict[str, Any]) -> Plan:
    fn = PLANNERS.get(family)
    if not fn:
        raise ValueError(f"Unknown family: {family}")
    return fn(params)


__all__ = [
    "plan_job",
    "plan_die_plate",
    "plan_punch",
    "plan_pilot_punch",
    "plan_bushing_id_critical",
    "plan_cam_or_hemmer",
    "plan_flat_die_chaser",
    "plan_pm_compaction_die",
    "plan_shear_blade",
    "plan_extrude_hone",
    "choose_wire_size",
    "choose_skims",
    "needs_wedm_for_windows",
]

