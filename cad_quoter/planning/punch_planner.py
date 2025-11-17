"""
Punch Process Planner - Detailed planning for punch manufacturing
=================================================================

Creates detailed manufacturing plans for punches, pilot pins, and form punches
based on extracted DWG features.

Integrates with:
    - dwg_punch_extractor.py for feature extraction
    - process_planner.py for overall planning orchestration
    - time_estimator.py for operation time estimates

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PunchPlannerParams:
    """
    Parameters for punch planning.

    Can be populated from PunchFeatureSummary or provided directly.
    """
    # Classification
    family: str = "round_punch"
    shape_type: str = "round"

    # Dimensions
    overall_length_in: float = 0.0
    max_od_or_width_in: float = 0.0
    body_width_in: Optional[float] = None
    body_thickness_in: Optional[float] = None

    # Operations counts
    num_ground_diams: int = 0
    total_ground_length_in: float = 0.0
    tap_count: int = 0
    tap_summary: list = None
    num_chamfers: int = 0

    # Pain flags
    has_perp_face_grind: bool = False
    has_3d_surface: bool = False
    has_polish_contour: bool = False
    has_no_step_permitted: bool = False
    min_dia_tol_in: Optional[float] = None
    min_len_tol_in: Optional[float] = None

    # Material
    material: str = "A2"

    def __post_init__(self):
        if self.tap_summary is None:
            self.tap_summary = []


def create_punch_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a detailed manufacturing plan for a punch.

    This is the main entry point for punch planning. It can accept either:
    1. A PunchFeatureSummary dict (from dwg_punch_extractor)
    2. Manual params dict with individual fields

    Args:
        params: Dictionary with punch parameters

    Returns:
        Plan dict compatible with process_planner.py schema:
        {
            ops: List[Dict],      # operations with op name and params
            fixturing: List[str], # fixturing notes
            qa: List[str],        # quality checks
            warnings: List[str],  # warnings
            directs: Dict,        # cost flags
        }
    """
    # Extract parameters
    p = _extract_params(params)

    # Initialize plan
    plan = {
        "ops": [],
        "fixturing": [],
        "qa": [],
        "warnings": [],
        "directs": {
            "hardware": False,
            "outsourced": False,
            "utilities": False,
            "consumables_flat": False,
            "packaging_flat": True,
        }
    }

    # Add operations based on punch type and features
    _add_stock_ops(plan, p)
    _add_roughing_ops(plan, p)
    _add_heat_treat_ops(plan, p)
    _add_grinding_ops(plan, p)
    _add_hole_ops(plan, p)
    _add_edge_ops(plan, p)
    _add_form_ops(plan, p)
    _add_qa_checks(plan, p)
    _add_fixturing_notes(plan, p)

    return plan


def _extract_params(params: Dict[str, Any]) -> PunchPlannerParams:
    """
    Extract and normalize parameters from input dict.

    Handles both PunchFeatureSummary format and legacy format.
    """
    return PunchPlannerParams(
        family=params.get("family", "round_punch"),
        shape_type=params.get("shape_type", "round"),
        overall_length_in=float(params.get("overall_length_in", 0.0)),
        max_od_or_width_in=float(params.get("max_od_or_width_in", 0.0)),
        body_width_in=params.get("body_width_in"),
        body_thickness_in=params.get("body_thickness_in"),
        num_ground_diams=int(params.get("num_ground_diams", 0)),
        total_ground_length_in=float(params.get("total_ground_length_in", 0.0)),
        tap_count=int(params.get("tap_count", 0)),
        tap_summary=params.get("tap_summary", []),
        num_chamfers=int(params.get("num_chamfers", 0)),
        has_perp_face_grind=bool(params.get("has_perp_face_grind", False)),
        has_3d_surface=bool(params.get("has_3d_surface", False)),
        has_polish_contour=bool(params.get("has_polish_contour", False)),
        has_no_step_permitted=bool(params.get("has_no_step_permitted", False)),
        min_dia_tol_in=params.get("min_dia_tol_in"),
        min_len_tol_in=params.get("min_len_tol_in"),
        material=params.get("material", "A2"),
    )


def _add_stock_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add stock procurement operation."""
    if p.shape_type == "round":
        stock_size = f"{p.max_od_or_width_in:.3f}\" DIA x {p.overall_length_in:.2f}\" L"
    else:
        stock_size = f"{p.body_width_in:.3f}\" x {p.body_thickness_in:.3f}\" x {p.overall_length_in:.2f}\" L"

    plan["ops"].append({
        "op": "stock_procurement",
        "material": p.material,
        "stock_size": stock_size,
        "note": f"Order {p.material} stock: {stock_size}"
    })


def _add_roughing_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add roughing operations (sawing, rough turning/milling)."""
    # Saw to length (with allowance)
    plan["ops"].append({
        "op": "saw_to_length",
        "length_in": p.overall_length_in,
        "note": "Saw to length with HT allowance"
    })

    # Rough machining based on shape
    if p.shape_type == "round":
        if p.num_ground_diams > 0:
            plan["ops"].append({
                "op": "rough_turning",
                "num_diams": p.num_ground_diams,
                "note": f"Rough turn {p.num_ground_diams} diameter sections"
            })
    else:
        plan["ops"].append({
            "op": "rough_milling",
            "note": "Rough mill rectangular sections"
        })


def _add_heat_treat_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add heat treatment operation."""
    # Determine target hardness based on material
    hardness_map = {
        "A2": "60-62 RC",
        "D2": "58-60 RC",
        "M2": "62-64 RC",
        "O1": "60-62 RC",
        "S7": "54-56 RC",
        "CARBIDE": "N/A (as-sintered)",
    }
    target_hardness = hardness_map.get(p.material, "58-62 RC")

    if p.material != "CARBIDE":
        plan["ops"].append({
            "op": "heat_treat",
            "material": p.material,
            "target_hardness": target_hardness,
            "note": f"Heat treat to {target_hardness}"
        })
        plan["directs"]["outsourced"] = True


def _add_grinding_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add grinding operations for punch diameters/surfaces."""
    if p.shape_type == "round":
        # OD grinding for each diameter
        if p.num_ground_diams > 0:
            tol_factor = 1.0
            if p.min_dia_tol_in and p.min_dia_tol_in < 0.0002:
                tol_factor = 1.5  # Tight tolerance increases time

            if p.has_no_step_permitted:
                tol_factor *= 1.3  # No step adds complexity

            plan["ops"].append({
                "op": "od_grind",
                "num_diams": p.num_ground_diams,
                "total_length_in": p.total_ground_length_in,
                "tol_factor": tol_factor,
                "note": f"Grind {p.num_ground_diams} diameters, {p.total_ground_length_in:.2f}\" total length"
            })

        # Face grinding (perpendicular face)
        if p.has_perp_face_grind:
            plan["ops"].append({
                "op": "face_grind",
                "diameter": p.max_od_or_width_in,
                "note": "Grind face perpendicular to centerline"
            })

    else:
        # Rectangular parts: surface grinding
        plan["ops"].append({
            "op": "surface_grind",
            "width_in": p.body_width_in,
            "thickness_in": p.body_thickness_in,
            "length_in": p.overall_length_in,
            "note": "Surface grind all faces"
        })


def _add_hole_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add drilling and tapping operations."""
    if p.tap_count > 0:
        for tap in p.tap_summary:
            size = tap.get("size", "Unknown")
            depth = tap.get("depth_in")
            depth_str = f"{depth:.2f}\"" if depth else "full depth"

            plan["ops"].append({
                "op": "tap",
                "size": size,
                "depth_in": depth,
                "note": f"Tap {size} x {depth_str}"
            })


def _add_edge_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add chamfering and deburring operations."""
    if p.num_chamfers > 0:
        plan["ops"].append({
            "op": "chamfer",
            "count": p.num_chamfers,
            "note": f"Chamfer {p.num_chamfers} edges"
        })


def _add_form_ops(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add form/polish operations for contoured punches."""
    if p.has_3d_surface:
        # 3D milling or EDM for form
        plan["ops"].append({
            "op": "3d_mill_form",
            "note": "3D mill contoured nose section"
        })

    if p.has_polish_contour:
        plan["ops"].append({
            "op": "polish_contour",
            "note": "Polish contoured surface to spec"
        })


def _add_qa_checks(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add quality assurance checks."""
    qa = plan["qa"]

    # Always check overall dimensions
    qa.append(f"Verify OAL: {p.overall_length_in:.3f}\"")

    if p.shape_type == "round":
        qa.append(f"Verify max OD: {p.max_od_or_width_in:.4f}\"")
    else:
        qa.append(f"Verify width: {p.body_width_in:.3f}\" x thickness: {p.body_thickness_in:.3f}\"")

    # Tight tolerance checks
    if p.min_dia_tol_in and p.min_dia_tol_in < 0.0005:
        qa.append(f"Critical diameter tolerance: ±{p.min_dia_tol_in:.4f}\"")

    if p.min_len_tol_in and p.min_len_tol_in < 0.005:
        qa.append(f"Critical length tolerance: ±{p.min_len_tol_in:.4f}\"")

    # Perpendicularity check
    if p.has_perp_face_grind:
        qa.append("Verify face perpendicularity to centerline")

    # Polish/finish check
    if p.has_polish_contour:
        qa.append("Verify contour surface finish (8 µin or per spec)")

    # Hardness check
    qa.append(f"Verify hardness: {p.material} to spec")


def _add_fixturing_notes(plan: Dict[str, Any], p: PunchPlannerParams) -> None:
    """Add fixturing recommendations."""
    fix = plan["fixturing"]

    if p.shape_type == "round":
        fix.append("Use collet or 3-jaw chuck for turning")
        fix.append("Use centers for long/slender sections")

        if p.max_od_or_width_in > 0 and p.overall_length_in / p.max_od_or_width_in > 10:
            fix.append("Use steady rest for slender punch (L/D > 10)")
    else:
        fix.append("Use vise or fixture plate for milling")
        fix.append("Parallel supports for grinding")

    if p.has_perp_face_grind:
        fix.append("Use precision angle plate for perpendicular grinding")


# ============================================================================
# Integration function for process_planner.py
# ============================================================================


def planner_punches_enhanced(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced punch planner for process_planner.py integration.

    This function is designed to replace the stub planner_punches() in
    process_planner.py.

    Args:
        params: Dictionary with punch parameters (can include PunchFeatureSummary fields)

    Returns:
        Plan dict with ops, fixturing, qa, warnings, directs
    """
    # If we have minimal info, check if we can extract from DXF
    if not params.get("overall_length_in") and params.get("dxf_path"):
        try:
            from cad_quoter.geometry.dwg_punch_extractor import extract_punch_features
            summary = extract_punch_features(params["dxf_path"])

            # Merge summary into params
            summary_dict = {
                "family": summary.family,
                "shape_type": summary.shape_type,
                "overall_length_in": summary.overall_length_in,
                "max_od_or_width_in": summary.max_od_or_width_in,
                "body_width_in": summary.body_width_in,
                "body_thickness_in": summary.body_thickness_in,
                "num_ground_diams": summary.num_ground_diams,
                "total_ground_length_in": summary.total_ground_length_in,
                "tap_count": summary.tap_count,
                "tap_summary": summary.tap_summary,
                "num_chamfers": summary.num_chamfers,
                "has_perp_face_grind": summary.has_perp_face_grind,
                "has_3d_surface": summary.has_3d_surface,
                "has_polish_contour": summary.has_polish_contour,
                "has_no_step_permitted": summary.has_no_step_permitted,
                "min_dia_tol_in": summary.min_dia_tol_in,
                "min_len_tol_in": summary.min_len_tol_in,
                "material": summary.material_callout or params.get("material", "A2"),
            }
            params.update(summary_dict)

            if summary.warnings:
                if "warnings" not in params:
                    params["warnings"] = []
                params["warnings"].extend(summary.warnings)

        except Exception as e:
            # If extraction fails, continue with manual params
            if "warnings" not in params:
                params["warnings"] = []
            params["warnings"].append(f"DXF extraction failed: {str(e)}")

    return create_punch_plan(params)
