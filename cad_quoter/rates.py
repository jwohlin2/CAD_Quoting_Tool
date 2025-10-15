"""Helpers for working with shop rate configuration data."""

from __future__ import annotations

import math
from typing import Any, Mapping

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


HARD_LABOR_FALLBACKS: Mapping[str, float] = {
    "Programmer": 80.0,
    "ProgrammingRate": 80.0,
    "Inspector": 75.0,
    "InspectionRate": 75.0,
}


HARD_MACHINE_FALLBACKS: Mapping[str, tuple[float, tuple[str, ...]]] = {
    "CNC_Mill": (110.0, ("MillingRate",)),
    "DrillPress": (95.0, ("DrillingRate",)),
    "SurfaceGrind": (95.0, ("SurfaceGrindRate", "GrindingRate")),
    "ODIDGrind": (95.0, ("ODIDGrindRate",)),
    "JigGrind": (95.0, ("JigGrindRate",)),
    "WireEDM": (130.0, ("WireEDMRate",)),
    "SinkerEDM": (130.0, ("SinkerEDMRate",)),
}


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

    return _finalize_two_bucket(labor_output, machine_output, numeric)


def _finalize_two_bucket(
    labor_output: dict[str, Any],
    machine_output: dict[str, Any],
    numeric: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    labor_output = dict(labor_output)
    machine_output = dict(machine_output)

    # Provide helpful logical aliases
    if "Finisher" in labor_output and "DeburrRate" not in labor_output:
        labor_output["DeburrRate"] = labor_output["Finisher"]

    def _mean(values: list[float]) -> float | None:
        filtered = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        if not filtered:
            return None
        return float(sum(filtered) / len(filtered))

    labor_candidates: list[float] = []
    for value in labor_output.values():
        if isinstance(value, (int, float)):
            labor_candidates.append(float(value))
    machine_candidates: list[float] = []
    for value in machine_output.values():
        if isinstance(value, (int, float)):
            machine_candidates.append(float(value))

    for fallback_key in ("ShopRate", "LaborRate"):
        try:
            labor_candidates.append(float(numeric[fallback_key]))
        except Exception:
            continue
    for fallback_key in ("ShopRate", "MachineRate", "MachineShopRate"):
        try:
            machine_candidates.append(float(numeric[fallback_key]))
        except Exception:
            continue

    numeric_values: list[float] = []
    for value in numeric.values():
        try:
            numeric_values.append(float(value))
        except Exception:
            continue

    labor_default = _mean(labor_candidates) or _mean(numeric_values) or 90.0
    machine_default = _mean(machine_candidates) or labor_default or 90.0

    expected_labor_keys = {
        "FixtureBuildRate",
        "DeburrRate",
        "FinishingRate",
    }
    hard_machine_aliases = {
        alias for _, alias_list in HARD_MACHINE_FALLBACKS.values() for alias in alias_list
    }
    expected_machine_keys = set(OLDKEY_TO_MACHINE.keys()) | {"AbrasiveFlowRate"}

    def _as_positive(value: Any) -> float:
        try:
            numeric_value = float(value or 0.0)
        except Exception:
            return 0.0
        if not math.isfinite(numeric_value) or numeric_value <= 0.0:
            return 0.0
        return numeric_value

    for key in expected_labor_keys:
        if _as_positive(labor_output.get(key)) <= 0.0:
            labor_output[key] = labor_default

    for canonical, fallback in HARD_LABOR_FALLBACKS.items():
        if _as_positive(labor_output.get(canonical)) <= 0.0:
            labor_output[canonical] = fallback
        if canonical == "Programmer":
            for alias in LEGACY_PROGRAMMER_RATE_KEYS:
                if _as_positive(labor_output.get(alias)) <= 0.0:
                    labor_output[alias] = labor_output[canonical]
        elif canonical == "Inspector":
            if _as_positive(labor_output.get("InspectionRate")) <= 0.0:
                labor_output["InspectionRate"] = labor_output[canonical]

    for key in expected_machine_keys:
        if key in hard_machine_aliases:
            continue
        existing = _as_positive(machine_output.get(key))
        if existing > 0.0:
            continue
        machine_output[key] = machine_default
        alias = OLDKEY_TO_MACHINE.get(key)
        if alias and _as_positive(machine_output.get(alias)) <= 0.0:
            machine_output[alias] = machine_default

    for canonical, (fallback_value, aliases) in HARD_MACHINE_FALLBACKS.items():
        if _as_positive(machine_output.get(canonical)) <= 0.0:
            machine_output[canonical] = fallback_value
        canonical_value = _as_positive(machine_output.get(canonical)) or fallback_value
        for alias in aliases:
            if _as_positive(machine_output.get(alias)) <= 0.0:
                machine_output[alias] = canonical_value

    if _as_positive(machine_output.get("MillingRate")) <= 0.0:
        machine_output["MillingRate"] = machine_output.get("CNC_Mill", machine_default)
    if _as_positive(machine_output.get("CNC_Mill")) <= 0.0:
        machine_output["CNC_Mill"] = _as_positive(machine_output.get("MillingRate")) or machine_default

    # Provide generic fallbacks expected by downstream renderers
    labor_rate_value = _as_positive(labor_output.get("LaborRate")) or labor_default
    if labor_rate_value > 0.0:
        labor_output["LaborRate"] = labor_rate_value
        labor_output.setdefault("DefaultLaborRate", labor_rate_value)

    programmer_value = _as_positive(labor_output.get("ProgrammingRate"))
    if programmer_value <= 0.0 and _as_positive(labor_output.get("Programmer")) > 0.0:
        programmer_value = float(labor_output["Programmer"])
        labor_output["ProgrammingRate"] = programmer_value
    if programmer_value > 0.0:
        for alias in LEGACY_PROGRAMMER_RATE_KEYS:
            labor_output.setdefault(alias, programmer_value)

    inspection_value = _as_positive(labor_output.get("InspectionRate"))
    if inspection_value <= 0.0 and _as_positive(labor_output.get("Inspector")) > 0.0:
        inspection_value = float(labor_output["Inspector"])
        labor_output["InspectionRate"] = inspection_value

    machine_rate_value = _as_positive(machine_output.get("MachineRate")) or machine_default
    if machine_rate_value > 0.0:
        machine_output["MachineRate"] = machine_rate_value
        machine_output.setdefault("DefaultMachineRate", machine_rate_value)

    drilling_value = _as_positive(machine_output.get("DrillingRate"))
    if drilling_value <= 0.0:
        drilling_value = _as_positive(machine_output.get("DrillPress"))
    if drilling_value <= 0.0:
        drilling_value = machine_rate_value
    if drilling_value > 0.0:
        machine_output.setdefault("DrillingRate", drilling_value)
        for alias in ("TappingRate", "CounterboreRate"):
            if _as_positive(machine_output.get(alias)) <= 0.0:
                machine_output[alias] = drilling_value

    saw_value = _as_positive(machine_output.get("SawWaterjetRate"))
    if saw_value <= 0.0:
        for candidate in ("Waterjet", "SawWaterjet", "Saw", "WaterjetRate"):
            saw_value = _as_positive(machine_output.get(candidate))
            if saw_value > 0.0:
                break
    if saw_value <= 0.0:
        saw_value = machine_rate_value
    if saw_value > 0.0:
        machine_output.setdefault("SawWaterjetRate", saw_value)
        machine_output.setdefault("WaterjetRate", saw_value)
        machine_output.setdefault("SawRate", saw_value)

    grinding_value = _as_positive(machine_output.get("GrindingRate"))
    if grinding_value <= 0.0:
        for candidate in (
            "SurfaceGrindRate",
            "SurfaceGrind",
            "ODIDGrindRate",
            "ODIDGrind",
            "JigGrindRate",
            "JigGrind",
        ):
            grinding_value = _as_positive(machine_output.get(candidate))
            if grinding_value > 0.0:
                break
    if grinding_value <= 0.0:
        grinding_value = machine_rate_value
    if grinding_value > 0.0:
        machine_output.setdefault("GrindingRate", grinding_value)

    wire_value = _as_positive(machine_output.get("WireEDMRate"))
    if wire_value <= 0.0:
        wire_value = _as_positive(machine_output.get("WireEDM"))
    if wire_value > 0.0:
        machine_output["WireEDMRate"] = wire_value
        machine_output.setdefault("WireEDM", wire_value)

    sinker_value = _as_positive(machine_output.get("SinkerEDMRate"))
    if sinker_value <= 0.0:
        sinker_value = _as_positive(machine_output.get("SinkerEDM"))
    if sinker_value > 0.0:
        machine_output["SinkerEDMRate"] = sinker_value
        machine_output.setdefault("SinkerEDM", sinker_value)

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
        fallback = HARD_LABOR_FALLBACKS.get("Programmer", labor_default)
        labor_clean = {"ProgrammingRate": fallback, "Programmer": fallback}
    if not machine_clean:
        fallback = HARD_MACHINE_FALLBACKS.get("CNC_Mill", (machine_default, ()))
        default_machine = fallback[0]
        machine_clean = {"MillingRate": default_machine, "CNC_Mill": default_machine}

    return {"labor": labor_clean, "machine": machine_clean}


def ensure_two_bucket_defaults(
    rates: Mapping[str, Mapping[str, Any]] | Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    """Ensure two-bucket rate mappings include deterministic fallbacks."""

    labor_raw = _dict(rates.get("labor")) if isinstance(rates, Mapping) else {}
    machine_raw = _dict(rates.get("machine")) if isinstance(rates, Mapping) else {}

    labor: dict[str, Any] = {}
    machine: dict[str, Any] = {}
    numeric: dict[str, Any] = {}

    for key, value in labor_raw.items():
        try:
            numeric_value = float(value)
        except Exception:
            continue
        labor[str(key)] = numeric_value
        numeric[str(key)] = numeric_value

    for key, value in machine_raw.items():
        try:
            numeric_value = float(value)
        except Exception:
            continue
        machine[str(key)] = numeric_value
        numeric[str(key)] = numeric_value

    return _finalize_two_bucket(labor, machine, numeric)


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
    "ensure_two_bucket_defaults",
    "rate_for_machine",
    "rate_for_role",
    "two_bucket_to_flat",
]

