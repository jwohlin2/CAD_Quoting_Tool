"""
Punch Time Estimator - Convert punch operations to machine/labor hours
======================================================================

Estimates manufacturing times for punch parts based on punch_planner operations.

Integrates with:
    - punch_planner.py for operation generation
    - QuoteDataHelper.py for quote data assembly

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class PunchMachineHours:
    """Machine hours breakdown for punch manufacturing."""
    # Turning/lathe operations
    rough_turning_min: float = 0.0
    finish_turning_min: float = 0.0

    # Grinding operations
    od_grinding_min: float = 0.0
    id_grinding_min: float = 0.0
    face_grinding_min: float = 0.0

    # Hole operations
    drilling_min: float = 0.0
    tapping_min: float = 0.0

    # Edge/form operations
    chamfer_min: float = 0.0
    polishing_min: float = 0.0
    edm_min: float = 0.0

    # Other
    sawing_min: float = 0.0
    inspection_min: float = 0.0

    # Totals
    total_minutes: float = 0.0
    total_hours: float = 0.0

    def calculate_totals(self):
        """Calculate total minutes and hours."""
        self.total_minutes = (
            self.rough_turning_min +
            self.finish_turning_min +
            self.od_grinding_min +
            self.id_grinding_min +
            self.face_grinding_min +
            self.drilling_min +
            self.tapping_min +
            self.chamfer_min +
            self.polishing_min +
            self.edm_min +
            self.sawing_min +
            self.inspection_min
        )
        self.total_hours = self.total_minutes / 60.0


@dataclass
class PunchLaborHours:
    """Labor hours breakdown for punch manufacturing."""
    # Setup times (per operation type)
    lathe_setup_min: float = 0.0
    grinder_setup_min: float = 0.0
    edm_setup_min: float = 0.0

    # Programming/planning
    cam_programming_min: float = 0.0

    # Handling
    handling_min: float = 0.0
    deburring_min: float = 0.0

    # Quality
    inspection_setup_min: float = 0.0
    first_article_min: float = 0.0

    # Totals
    total_minutes: float = 0.0
    total_hours: float = 0.0

    def calculate_totals(self):
        """Calculate total minutes and hours."""
        self.total_minutes = (
            self.lathe_setup_min +
            self.grinder_setup_min +
            self.edm_setup_min +
            self.cam_programming_min +
            self.handling_min +
            self.deburring_min +
            self.inspection_setup_min +
            self.first_article_min
        )
        self.total_hours = self.total_minutes / 60.0


# Time constants for punch operations (minutes per unit)
PUNCH_TIME_CONSTANTS = {
    # Turning/lathe (minutes per inch of diameter step)
    "rough_turning_per_diam": 8.0,       # 8 min per diameter section
    "finish_turning_per_diam": 5.0,      # 5 min per diameter section

    # Grinding (minutes per inch of length)
    "od_grinding_per_inch": 3.0,         # 3 min per inch of ground OD
    "id_grinding_per_inch": 6.0,         # 6 min per inch of ID (if present)
    "face_grinding_per_face": 4.0,       # 4 min per perpendicular face

    # Hole operations (minutes per hole)
    "drilling_per_hole": 2.0,            # 2 min per drilled hole
    "tapping_per_hole": 3.0,             # 3 min per tapped hole

    # Edge operations (minutes per feature)
    "chamfer_per_edge": 1.5,             # 1.5 min per chamfer
    "small_radius_per_edge": 2.0,        # 2 min per small radius

    # Form/polish operations
    "polish_contour_base": 30.0,         # 30 min base for polish contour
    "polish_per_sq_inch": 5.0,           # 5 min per sq inch of contour
    "form_complexity_multiplier": {      # Multiplier based on form level
        0: 1.0,
        1: 1.5,
        2: 2.0,
        3: 3.0,
    },

    # Misc operations
    "sawing_base": 5.0,                  # 5 min base sawing time
    "inspection_per_diam": 3.0,          # 3 min per diameter to inspect

    # Tolerance adjustments (multipliers)
    "tight_tolerance_multiplier": {      # Extra time for tight tolerances
        0.0001: 2.0,    # ±0.0001" - double time
        0.0002: 1.5,    # ±0.0002" - 50% more
        0.0005: 1.2,    # ±0.0005" - 20% more
        0.001: 1.0,     # ±0.001" - standard
    },

    # Setup times (minutes)
    "lathe_setup": 30.0,                 # 30 min lathe setup
    "grinder_setup": 20.0,               # 20 min grinder setup
    "edm_setup": 45.0,                   # 45 min EDM setup
    "cam_programming_base": 30.0,        # 30 min CAM base
    "cam_per_operation": 5.0,            # 5 min per additional op

    # Labor operations
    "handling_per_operation": 2.0,       # 2 min handling per op
    "deburring_per_edge": 1.0,           # 1 min per edge to deburr
    "inspection_setup": 10.0,            # 10 min inspection setup
    "first_article_base": 20.0,          # 20 min first article base
}


def estimate_punch_machine_hours(
    punch_plan: Dict[str, Any],
    punch_features: Dict[str, Any]
) -> PunchMachineHours:
    """
    Estimate machine hours for punch manufacturing.

    Args:
        punch_plan: Plan dict from create_punch_plan()
        punch_features: Feature dict from extract_punch_features_from_dxf()

    Returns:
        PunchMachineHours with time breakdown
    """
    hours = PunchMachineHours()
    tc = PUNCH_TIME_CONSTANTS

    # Extract feature values
    num_diams = punch_features.get("num_ground_diams", 1)
    ground_length = punch_features.get("total_ground_length_in", 0.0)
    tap_count = punch_features.get("tap_count", 0)
    num_chamfers = punch_features.get("num_chamfers", 0)
    num_radii = punch_features.get("num_small_radii", 0)
    has_polish = punch_features.get("has_polish_contour", False)
    has_3d = punch_features.get("has_3d_surface", False)
    has_perp_face = punch_features.get("has_perp_face_grind", False)
    form_level = punch_features.get("form_complexity_level", 0)
    min_dia_tol = punch_features.get("min_dia_tol_in")
    overall_length = punch_features.get("overall_length_in", 0.0)
    max_od = punch_features.get("max_od_or_width_in", 0.0)

    # Calculate tolerance multiplier
    tol_mult = 1.0
    if min_dia_tol is not None:
        for threshold, mult in sorted(tc["tight_tolerance_multiplier"].items()):
            if min_dia_tol <= threshold:
                tol_mult = mult
                break

    # Sawing
    hours.sawing_min = tc["sawing_base"]

    # Turning (rough and finish)
    if num_diams > 0:
        hours.rough_turning_min = num_diams * tc["rough_turning_per_diam"]
        hours.finish_turning_min = num_diams * tc["finish_turning_per_diam"] * tol_mult

    # OD Grinding
    if ground_length > 0:
        hours.od_grinding_min = ground_length * tc["od_grinding_per_inch"] * tol_mult

    # Face grinding
    if has_perp_face:
        hours.face_grinding_min = tc["face_grinding_per_face"] * 2  # Both ends

    # Hole operations
    if tap_count > 0:
        hours.drilling_min = tap_count * tc["drilling_per_hole"]
        hours.tapping_min = tap_count * tc["tapping_per_hole"]

    # Edge operations
    hours.chamfer_min = num_chamfers * tc["chamfer_per_edge"]
    hours.chamfer_min += num_radii * tc["small_radius_per_edge"]

    # Polish/form operations
    if has_polish or has_3d:
        contour_area = max_od * (overall_length * 0.3)  # Estimate contour area
        form_mult = tc["form_complexity_multiplier"].get(form_level, 1.0)
        hours.polishing_min = (
            tc["polish_contour_base"] +
            contour_area * tc["polish_per_sq_inch"]
        ) * form_mult

    # Inspection
    hours.inspection_min = num_diams * tc["inspection_per_diam"]

    # Calculate totals
    hours.calculate_totals()

    return hours


def estimate_punch_labor_hours(
    punch_plan: Dict[str, Any],
    punch_features: Dict[str, Any],
    machine_hours: PunchMachineHours
) -> PunchLaborHours:
    """
    Estimate labor hours for punch manufacturing.

    Args:
        punch_plan: Plan dict from create_punch_plan()
        punch_features: Feature dict from extract_punch_features_from_dxf()
        machine_hours: Machine hours breakdown

    Returns:
        PunchLaborHours with time breakdown
    """
    labor = PunchLaborHours()
    tc = PUNCH_TIME_CONSTANTS

    # Extract values
    ops = punch_plan.get("ops", [])
    num_ops = len(ops)
    num_chamfers = punch_features.get("num_chamfers", 0)
    num_radii = punch_features.get("num_small_radii", 0)
    tap_count = punch_features.get("tap_count", 0)
    has_polish = punch_features.get("has_polish_contour", False)

    # Setup times (assume one setup per machine type used)
    needs_lathe = machine_hours.rough_turning_min > 0 or machine_hours.finish_turning_min > 0
    needs_grinder = machine_hours.od_grinding_min > 0 or machine_hours.face_grinding_min > 0
    needs_edm = machine_hours.edm_min > 0

    if needs_lathe:
        labor.lathe_setup_min = tc["lathe_setup"]
    if needs_grinder:
        labor.grinder_setup_min = tc["grinder_setup"]
    if needs_edm:
        labor.edm_setup_min = tc["edm_setup"]

    # CAM programming
    labor.cam_programming_min = tc["cam_programming_base"] + num_ops * tc["cam_per_operation"]

    # Handling
    labor.handling_min = num_ops * tc["handling_per_operation"]

    # Deburring
    total_edges = num_chamfers + num_radii + tap_count * 2  # Taps have entry/exit edges
    labor.deburring_min = total_edges * tc["deburring_per_edge"]

    # Inspection
    labor.inspection_setup_min = tc["inspection_setup"]
    labor.first_article_min = tc["first_article_base"]

    # Extra time for polish
    if has_polish:
        labor.first_article_min *= 1.5  # 50% more for polish verification

    # Calculate totals
    labor.calculate_totals()

    return labor


def convert_punch_to_quote_machine_hours(
    machine_hours: PunchMachineHours,
    labor_hours: PunchLaborHours
) -> Dict[str, Any]:
    """
    Convert PunchMachineHours to QuoteDataHelper MachineHoursBreakdown format.

    Args:
        machine_hours: Punch machine hours
        labor_hours: Punch labor hours

    Returns:
        Dict compatible with MachineHoursBreakdown dataclass
    """
    return {
        # Map punch operations to standard categories
        "total_drill_minutes": machine_hours.drilling_min,
        "total_tap_minutes": machine_hours.tapping_min,
        "total_cbore_minutes": 0.0,
        "total_cdrill_minutes": 0.0,
        "total_jig_grind_minutes": 0.0,

        # Plan operations
        "total_milling_minutes": (
            machine_hours.rough_turning_min +
            machine_hours.finish_turning_min
        ),
        "total_grinding_minutes": (
            machine_hours.od_grinding_min +
            machine_hours.id_grinding_min +
            machine_hours.face_grinding_min
        ),
        "total_edm_minutes": machine_hours.edm_min,
        "total_other_minutes": (
            machine_hours.sawing_min +
            machine_hours.chamfer_min +
            machine_hours.polishing_min
        ),
        "total_cmm_minutes": machine_hours.inspection_min,

        # Overall totals
        "total_minutes": machine_hours.total_minutes,
        "total_hours": machine_hours.total_hours,
    }


def convert_punch_to_quote_labor_hours(
    labor_hours: PunchLaborHours
) -> Dict[str, Any]:
    """
    Convert PunchLaborHours to QuoteDataHelper LaborHoursBreakdown format.

    Args:
        labor_hours: Punch labor hours

    Returns:
        Dict compatible with LaborHoursBreakdown dataclass
    """
    return {
        # Setup times
        "total_setup_minutes": (
            labor_hours.lathe_setup_min +
            labor_hours.grinder_setup_min +
            labor_hours.edm_setup_min
        ),

        # Programming
        "cam_programming_minutes": labor_hours.cam_programming_min,

        # Operations
        "handling_minutes": labor_hours.handling_min,
        "deburring_minutes": labor_hours.deburring_min,

        # Quality
        "inspection_minutes": (
            labor_hours.inspection_setup_min +
            labor_hours.first_article_min
        ),

        # Totals
        "total_minutes": labor_hours.total_minutes,
        "total_hours": labor_hours.total_hours,
    }


def estimate_punch_times(
    punch_plan: Dict[str, Any],
    punch_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main entry point for punch time estimation.

    Args:
        punch_plan: Plan from create_punch_plan()
        punch_features: Features from extract_punch_features_from_dxf()

    Returns:
        Dict with machine_hours and labor_hours in QuoteData format
    """
    # Get detailed breakdowns
    machine = estimate_punch_machine_hours(punch_plan, punch_features)
    labor = estimate_punch_labor_hours(punch_plan, punch_features, machine)

    # Convert to QuoteData format
    return {
        "machine_hours": convert_punch_to_quote_machine_hours(machine, labor),
        "labor_hours": convert_punch_to_quote_labor_hours(labor),
        "punch_machine_breakdown": machine,  # Keep detailed breakdown
        "punch_labor_breakdown": labor,
    }
