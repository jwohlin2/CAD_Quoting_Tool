"""Helpers for working with shop rate configuration data."""

from __future__ import annotations

from typing import Any, Dict

# ---- canonical names used by the planner ----
MACHINES = [
    "WireEDM",
    "SinkerEDM",
    "CNC_Mill",
    "DrillPress",
    "Lathe",
    "Waterjet",
    "Blanchard",
    "SurfaceGrind",
    "JigGrind",
    "ODIDGrind",
    "VisualContourGrind",
    "ExtrudeHone",
]


ROLES = [
    "Programmer",
    "Engineer",
    "ProjectManager",
    "Toolmaker",
    "Machinist",
    "EDMOperator",
    "Grinder",
    "Inspector",
    "Assembler",
    "FixtureBuilder",
    "Finisher",
    "Lapper",
    "Support",
]


# ---- mapping from current flat keys → new buckets ----
OLDKEY_TO_MACHINE = {
    "WireEDMRate": "WireEDM",
    "SinkerEDMRate": "SinkerEDM",
    "MillingRate": "CNC_Mill",
    "DrillingRate": "DrillPress",
    "TurningRate": "Lathe",
    "SawWaterjetRate": "Waterjet",
    "SurfaceGrindRate": "SurfaceGrind",
    "JigGrindRate": "JigGrind",
    "ODIDGrindRate": "ODIDGrind",
    # Not in the legacy UI but commonly used in workflows.
    # "BlanchardRate": "Blanchard",
    # "ExtrudeHoneRate": "ExtrudeHone",
    # "VisualContourGrindRate": "VisualContourGrind",
}


OLDKEY_TO_LABOR = {
    "ProgrammingRate": "Programmer",  # same as CAMRate below; we'll reconcile
    "CAMRate": "Programmer",
    "EngineerRate": "Engineer",
    "ProjectManagementRate": "ProjectManager",
    "ToolmakerSupportRate": "Toolmaker",
    "FixtureBuildRate": "FixtureBuilder",
    "FinishingRate": "Finisher",
    "LappingRate": "Lapper",
    "InspectionRate": "Inspector",
    "AssemblyRate": "Assembler",
    # Useful generic role for floor help / apprentices:
    # "SupportRate": "Support",
    # If you treat manual grinding as labor (not machine), you could also map to "Grinder"
}


# Preferred canonical role for overlapping keys (e.g., ProgrammingRate vs CAMRate)
PREFERRED_ROLE_FOR_DUPES = {
    "Programmer": ["ProgrammingRate", "CAMRate"],
}


def migrate_flat_to_two_bucket(old: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Convert flat overrides into ``{"labor": {...}, "machine": {...}}``."""

    labor: Dict[str, float] = {}
    machine: Dict[str, float] = {}

    # Resolve duplicate labor sources
    for role, keys in PREFERRED_ROLE_FOR_DUPES.items():
        for key in keys:
            if key in old:
                labor[role] = float(old[key])
                break

    # Map remaining labor keys
    for key, role in OLDKEY_TO_LABOR.items():
        if role == "Programmer":  # handled above
            continue
        if key in old:
            labor[role] = float(old[key])

    # Map machine keys
    for key, machine_name in OLDKEY_TO_MACHINE.items():
        if key in old:
            machine[machine_name] = float(old[key])

    # Optional logical aliases / defaults
    if "Blanchard" not in machine and "SurfaceGrind" in machine:
        machine["Blanchard"] = machine["SurfaceGrind"]
    if "VisualContourGrind" not in machine and "SurfaceGrind" in machine:
        machine["VisualContourGrind"] = machine["SurfaceGrind"]
    if "ExtrudeHone" not in machine and "Finisher" in labor:
        machine["ExtrudeHone"] = labor["Finisher"]

    return {"labor": labor, "machine": machine}


# ---- helpers for costing layers ----


def rate_for_machine(rates: Dict[str, Dict[str, float]], machine: str) -> float:
    return float(rates["machine"][machine])


def rate_for_role(rates: Dict[str, Dict[str, float]], role: str) -> float:
    return float(rates["labor"][role])


# ---- op → machine / labor maps that match process planner ops ----

OP_TO_MACHINE = {
    # die plates / flats
    "blanchard_grind_pre": "Blanchard",
    "surface_grind_faces": "SurfaceGrind",
    "finish_mill_windows": "CNC_Mill",
    "cnc_rough_mill": "CNC_Mill",
    "wire_edm_windows": "WireEDM",
    "wire_edm_outline": "WireEDM",
    "jig_bore_or_jig_grind_coaxial_bores": "JigGrind",
    "jig_grind_ID_to_size_and_roundness": "JigGrind",
    "drill_patterns": "DrillPress",
    "rigid_tap": "DrillPress",
    "thread_mill": "CNC_Mill",
    "waterjet_or_saw_blanks": "Waterjet",
    # punches / inserts
    "sinker_edm_finish_burn": "SinkerEDM",
    "surface_or_profile_grind_bearing": "SurfaceGrind",
    "profile_or_surface_grind_wear_faces": "SurfaceGrind",
    "profile_grind_pilot_OD_to_TIR": "SurfaceGrind",
    # specials
    "wire_edm_cam_slot_or_profile": "WireEDM",
    "profile_grind_cutting_edges_and_angles": "SurfaceGrind",
    "match_grind_set_for_gap_and_parallelism": "SurfaceGrind",
    "visual_contour_grind": "VisualContourGrind",
    "abrasive_flow_polish": "ExtrudeHone",
}


OP_TO_LABOR = {
    "program_estimate": "Engineer",
    "edge_break": "Finisher",
    "lap_bearing_land": "Lapper",
    "lap_ID": "Lapper",
    "clean_degas_for_coating": "Finisher",
    "apply_coating": "Finisher",
    "assemble_pair_on_fixture": "Toolmaker",
    "stability_check_after_ops": "Inspector",
    "spot_drill_all": "Machinist",
    "drill_ream_bore": "Machinist",
    "drill_ream_dowel_press": "Machinist",
    "ream_slip_in_assembly": "Toolmaker",
    "mark_id": "Assembler",
    "heat_treat_to_spec": "Engineer",
    "machine_electrode": "EDMOperator",
    "prep_carrier_or_tab": "EDMOperator",
    "surface_grind_datums": "Grinder",
    "indicate_on_shank": "Grinder",
    "light_grind_cleanup": "Grinder",
    "saw_blank": "Machinist",
    "face_mill_pre": "Machinist",
    "QA_generic": "Inspector",
}


_ROLE_TO_OLDKEY: Dict[str, str] = {}
for role, keys in PREFERRED_ROLE_FOR_DUPES.items():
    if keys:
        _ROLE_TO_OLDKEY[role] = keys[0]
for old_key, role in OLDKEY_TO_LABOR.items():
    _ROLE_TO_OLDKEY.setdefault(role, old_key)

_MACHINE_TO_OLDKEY: Dict[str, str] = {}
for old_key, machine in OLDKEY_TO_MACHINE.items():
    _MACHINE_TO_OLDKEY.setdefault(machine, old_key)


def two_bucket_to_flat(rates: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Best-effort conversion of a two-bucket rate tree back to flat keys."""

    flat: Dict[str, float] = {}

    machine_rates = rates.get("machine", {}) if isinstance(rates, dict) else {}
    if isinstance(machine_rates, dict):
        for machine_name, value in machine_rates.items():
            if value in (None, ""):
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            key = _MACHINE_TO_OLDKEY.get(machine_name, machine_name)
            flat[key] = numeric

    labor_rates = rates.get("labor", {}) if isinstance(rates, dict) else {}
    if isinstance(labor_rates, dict):
        for role, value in labor_rates.items():
            if value in (None, ""):
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            aliases = PREFERRED_ROLE_FOR_DUPES.get(role)
            if aliases:
                for alias in aliases:
                    flat[alias] = numeric
            key = _ROLE_TO_OLDKEY.get(role, role)
            flat[key] = numeric

    return flat


__all__ = [
    "MACHINES",
    "ROLES",
    "OLDKEY_TO_MACHINE",
    "OLDKEY_TO_LABOR",
    "PREFERRED_ROLE_FOR_DUPES",
    "OP_TO_MACHINE",
    "OP_TO_LABOR",
    "migrate_flat_to_two_bucket",
    "rate_for_machine",
    "rate_for_role",
    "two_bucket_to_flat",
]

