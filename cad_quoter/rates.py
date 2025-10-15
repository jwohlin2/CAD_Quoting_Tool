"""Helpers for working with shop rate configuration data."""

from __future__ import annotations

import math
from typing import Any

from cad_quoter.utils import _dict

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


LEGACY_PROGRAMMER_RATE_KEYS = (
    "ProgrammingRate",
    "CAMRate",
)


OLDKEY_TO_LABOR = {
    "ProgrammingRate": "Programmer",
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


# Preferred canonical role names when flattening
PREFERRED_ROLE_FOR_DUPES = {
    "Programmer": ["ProgrammingRate"],
}


def migrate_flat_to_two_bucket(old: dict[str, float]) -> dict[str, dict[str, float]]:
    """Convert flat overrides into ``{"labor": {...}, "machine": {...}}``."""

    numeric: dict[str, float] = {}
    for key, value in old.items():
        try:
            numeric[str(key)] = float(value)
        except Exception:
            continue

    labor: dict[str, float] = {}
    machine: dict[str, float] = {}

    labor_aliases: dict[str, float] = {}
    machine_aliases: dict[str, float] = {}

    # Resolve programmer rate (handle legacy CAMRate alias)
    for key in LEGACY_PROGRAMMER_RATE_KEYS:
        if key in numeric:
            value = numeric[key]
            labor["Programmer"] = value
            labor_aliases["ProgrammingRate"] = value
            labor_aliases[key] = value
            break

    # Map remaining labor keys
    for key, role in OLDKEY_TO_LABOR.items():
        if role == "Programmer":  # handled above
            continue
        if key in numeric:
            value = numeric[key]
            labor[role] = value
            labor_aliases[key] = value

    # Map machine keys
    for key, machine_name in OLDKEY_TO_MACHINE.items():
        if key in numeric:
            value = numeric[key]
            machine[machine_name] = value
            machine_aliases[key] = value

    # Optional logical aliases / defaults
    if "Blanchard" not in machine and "SurfaceGrind" in machine:
        machine["Blanchard"] = machine["SurfaceGrind"]
    if "VisualContourGrind" not in machine and "SurfaceGrind" in machine:
        machine["VisualContourGrind"] = machine["SurfaceGrind"]
    if "ExtrudeHone" not in machine and "Finisher" in labor:
        machine["ExtrudeHone"] = labor["Finisher"]

    labor_output: dict[str, float] = dict(labor)
    machine_output: dict[str, float] = dict(machine)

    for alias, value in labor_aliases.items():
        labor_output.setdefault(alias, value)
    if "Programmer" in labor_output:
        programmer_rate = labor_output["Programmer"]
        for alias in LEGACY_PROGRAMMER_RATE_KEYS:
            labor_output.setdefault(alias, programmer_rate)

    for alias, value in machine_aliases.items():
        machine_output.setdefault(alias, value)

    # Ensure canonical ↔ alias mappings are mirrored
    for old_key, role in OLDKEY_TO_LABOR.items():
        value = labor_output.get(role)
        if value is not None:
            labor_output.setdefault(old_key, value)
        else:
            alias_value = labor_output.get(old_key)
            if alias_value is not None:
                labor_output.setdefault(role, alias_value)

    for old_key, machine_name in OLDKEY_TO_MACHINE.items():
        value = machine_output.get(machine_name)
        if value is not None:
            machine_output.setdefault(old_key, value)
        else:
            alias_value = machine_output.get(old_key)
            if alias_value is not None:
                machine_output.setdefault(machine_name, alias_value)

    # Provide helpful logical aliases
    if "Finisher" in labor_output and "DeburrRate" not in labor_output:
        labor_output["DeburrRate"] = labor_output["Finisher"]

    # Collect candidate values for defaults
    def _mean(values: list[float]) -> float | None:
        filtered = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        if not filtered:
            return None
        return float(sum(filtered) / len(filtered))

    labor_candidates: list[float] = list({v for v in labor_output.values()})
    machine_candidates: list[float] = list({v for v in machine_output.values()})

    for fallback_key in ("ShopRate", "LaborRate"):
        if fallback_key in numeric:
            labor_candidates.append(numeric[fallback_key])
    for fallback_key in ("ShopRate", "MachineRate", "MachineShopRate"):
        if fallback_key in numeric:
            machine_candidates.append(numeric[fallback_key])

    labor_default = _mean(labor_candidates) or _mean(list(numeric.values())) or 90.0
    machine_default = _mean(machine_candidates) or labor_default or 90.0

    expected_labor_keys = {
        "ProgrammingRate",
        "InspectionRate",
        "FixtureBuildRate",
        "DeburrRate",
        "FinishingRate",
    }
    expected_machine_keys = set(OLDKEY_TO_MACHINE.keys()) | {
        "AbrasiveFlowRate",
    }

    for key in expected_labor_keys:
        if float(labor_output.get(key, 0.0) or 0.0) <= 0.0:
            labor_output[key] = labor_default
    if "Programmer" not in labor_output or labor_output["Programmer"] <= 0.0:
        labor_output["Programmer"] = labor_default
        labor_output.setdefault("ProgrammingRate", labor_default)

    for key in expected_machine_keys:
        if float(machine_output.get(key, 0.0) or 0.0) <= 0.0:
            machine_output[key] = machine_default
            alias = OLDKEY_TO_MACHINE.get(key)
            if alias:
                machine_output.setdefault(alias, machine_default)

    if float(machine_output.get("CNC_Mill", 0.0) or 0.0) <= 0.0:
        machine_output["CNC_Mill"] = machine_default
        machine_output.setdefault("MillingRate", machine_default)

    # Final cleanup: ensure strictly positive numeric values
    labor_clean = {
        key: float(value)
        for key, value in labor_output.items()
        if isinstance(value, (int, float)) and value > 0.0 and math.isfinite(float(value))
    }
    machine_clean = {
        key: float(value)
        for key, value in machine_output.items()
        if isinstance(value, (int, float)) and value > 0.0 and math.isfinite(float(value))
    }

    if not labor_clean:
        labor_clean = {"ProgrammingRate": labor_default, "Programmer": labor_default}
    if not machine_clean:
        machine_clean = {"MillingRate": machine_default, "CNC_Mill": machine_default}

    return {"labor": labor_clean, "machine": machine_clean}


# ---- helpers for costing layers ----


def rate_for_machine(rates: dict[str, dict[str, float]], machine: str) -> float:
    return float(rates["machine"][machine])


def rate_for_role(rates: dict[str, dict[str, float]], role: str) -> float:
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


_ROLE_TO_OLDKEY: dict[str, str] = {}
for role, keys in PREFERRED_ROLE_FOR_DUPES.items():
    if keys:
        _ROLE_TO_OLDKEY[role] = keys[0]
for old_key, role in OLDKEY_TO_LABOR.items():
    _ROLE_TO_OLDKEY.setdefault(role, old_key)

_MACHINE_TO_OLDKEY: dict[str, str] = {}
for old_key, machine in OLDKEY_TO_MACHINE.items():
    _MACHINE_TO_OLDKEY.setdefault(machine, old_key)


def two_bucket_to_flat(rates: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Best-effort conversion of a two-bucket rate tree back to flat keys."""

    flat: dict[str, float] = {}

    rates_map = _dict(rates)
    machine_rates = _dict(rates_map.get("machine"))
    for machine_name, value in machine_rates.items():
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        key = _MACHINE_TO_OLDKEY.get(machine_name, machine_name)
        flat[key] = numeric

    labor_rates = _dict(rates_map.get("labor"))
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

