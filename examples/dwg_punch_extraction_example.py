"""
DWG Punch Extraction Example
============================

Demonstrates the complete workflow for extracting features from DWG punch
drawings and generating manufacturing plans.

Usage:
    python examples/dwg_punch_extraction_example.py

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from pathlib import Path
from pprint import pprint
from cad_quoter.geometry.dwg_punch_extractor import (
    extract_punch_features_from_dxf,
    PunchFeatureSummary,
)
from cad_quoter.planning.punch_planner import create_punch_plan
from cad_quoter.planning.process_planner import plan_job


def example_1_text_based_extraction():
    """
    Example 1: Extract features from text only (no DXF file).

    This demonstrates the text-based classification and feature detection
    without requiring an actual DXF file.
    """
    print("=" * 70)
    print("Example 1: Text-Based Feature Extraction")
    print("=" * 70)

    # Simulate text extracted from a punch drawing
    text_dump = """
    ROUND PUNCH - PART #RP-12345
    MATERIAL: A2 TOOL STEEL
    HEAT TREAT: 60-62 RC

    DIMENSIONS:
    6.990 ±.005 OAL
    Ø.7504 +.0000 -.0001 MAX DIA
    Ø.6250 +.0000 -.0002 SHANK
    Ø.5000 NOSE SECTION

    FEATURES:
    5/16-18 TAP X .80 DEEP
    Ø.125 COOLANT HOLE THRU

    EDGE WORK:
    (2) .010 X 45° CHAMFER

    NOTES:
    - GRIND ALL DIAMETERS
    - THIS SURFACE PERPENDICULAR TO CENTERLINE
    - NO STEP PERMITTED BETWEEN DIAMETERS
    """

    # Extract features
    dummy_path = Path("/tmp/dummy.dxf")
    summary = extract_punch_features_from_dxf(dummy_path, text_dump)

    print("\nExtracted Features:")
    print(f"  Family: {summary.family}")
    print(f"  Shape: {summary.shape_type}")
    print(f"  Material: {summary.material_callout}")
    print(f"  Overall Length: {summary.overall_length_in:.3f}\"")
    print(f"  Max OD: {summary.max_od_or_width_in:.4f}\"")
    print(f"  Ground Diameters: {summary.num_ground_diams}")
    print(f"  Tap Count: {summary.tap_count}")
    print(f"  Chamfers: {summary.num_chamfers}")
    print(f"  Has Perp Face: {summary.has_perp_face_grind}")
    print(f"  No Step: {summary.has_no_step_permitted}")
    print(f"  Confidence: {summary.confidence_score:.2f}")

    if summary.tap_summary:
        print("\n  Tap Details:")
        for tap in summary.tap_summary:
            print(f"    - {tap['size']} x {tap.get('depth_in', 'N/A')}\" deep")

    return summary


def example_2_create_manufacturing_plan(summary: PunchFeatureSummary):
    """
    Example 2: Create a manufacturing plan from extracted features.

    This demonstrates how the extracted features feed into the planning system.
    """
    print("\n" + "=" * 70)
    print("Example 2: Manufacturing Plan Generation")
    print("=" * 70)

    # Convert summary to params dict
    params = {
        "family": summary.family,
        "shape_type": summary.shape_type,
        "overall_length_in": summary.overall_length_in,
        "max_od_or_width_in": summary.max_od_or_width_in,
        "num_ground_diams": summary.num_ground_diams,
        "total_ground_length_in": summary.total_ground_length_in,
        "tap_count": summary.tap_count,
        "tap_summary": summary.tap_summary,
        "num_chamfers": summary.num_chamfers,
        "has_perp_face_grind": summary.has_perp_face_grind,
        "has_no_step_permitted": summary.has_no_step_permitted,
        "min_dia_tol_in": summary.min_dia_tol_in,
        "material": summary.material_callout or "A2",
    }

    # Generate plan
    plan = create_punch_plan(params)

    print("\nManufacturing Plan:")
    print("\nOperations:")
    for i, op in enumerate(plan["ops"], 1):
        print(f"  {i}. {op['op']}")
        if "note" in op:
            print(f"     Note: {op['note']}")

    print("\nFixturing:")
    for note in plan["fixturing"]:
        print(f"  - {note}")

    print("\nQA Checks:")
    for check in plan["qa"]:
        print(f"  - {check}")

    if plan["warnings"]:
        print("\nWarnings:")
        for warning in plan["warnings"]:
            print(f"  ⚠ {warning}")

    return plan


def example_3_integrated_planning():
    """
    Example 3: Use the integrated process planner.

    This demonstrates using plan_job() which automatically handles
    punch planning with feature extraction.
    """
    print("\n" + "=" * 70)
    print("Example 3: Integrated Process Planning")
    print("=" * 70)

    # Params for a pilot pin
    params = {
        "family": "pilot_pin",
        "shape_type": "round",
        "overall_length_in": 8.0,
        "max_od_or_width_in": 0.500,
        "num_ground_diams": 2,
        "total_ground_length_in": 6.0,
        "has_perp_face_grind": True,
        "min_dia_tol_in": 0.0001,
        "material": "M2",
    }

    # Use the integrated planner
    plan = plan_job("Punches", params)

    print("\nIntegrated Plan:")
    print("\nOperations:")
    for op in plan["ops"]:
        print(f"  - {op.get('op', 'unknown')}: {op.get('note', '')}")

    print("\nFixturing:")
    for note in plan["fixturing"]:
        print(f"  - {note}")

    print("\nQA:")
    for check in plan["qa"]:
        print(f"  - {check}")


def example_4_form_punch():
    """
    Example 4: Form punch with 3D surface.

    Demonstrates extraction and planning for a more complex form punch.
    """
    print("\n" + "=" * 70)
    print("Example 4: Form Punch with 3D Contour")
    print("=" * 70)

    text_dump = """
    FORM PUNCH
    MATERIAL: D2 TOOL STEEL

    DIMENSIONS:
    4.500 OAL
    Ø.875 MAX DIA
    Ø.750 SHANK

    CONTOUR:
    COIN PUNCH DETAIL
    POLISH CONTOUR TO 8 µin
    R.125
    R.250
    R.375
    R.500

    NOTES:
    NO STEP PERMITTED
    SHARP EDGES REQUIRED
    """

    summary = extract_punch_features_from_dxf(Path("/tmp/dummy.dxf"), text_dump)

    print("\nForm Punch Features:")
    print(f"  Family: {summary.family}")
    print(f"  Material: {summary.material_callout}")
    print(f"  Has 3D Surface: {summary.has_3d_surface}")
    print(f"  Has Polish: {summary.has_polish_contour}")
    print(f"  Form Complexity: {summary.form_complexity_level}")
    print(f"  Sharp Edges: {summary.has_sharp_edges}")

    # Generate plan
    params = {
        "family": summary.family,
        "overall_length_in": summary.overall_length_in,
        "max_od_or_width_in": summary.max_od_or_width_in,
        "has_3d_surface": summary.has_3d_surface,
        "has_polish_contour": summary.has_polish_contour,
        "has_no_step_permitted": summary.has_no_step_permitted,
        "material": summary.material_callout or "D2",
    }

    plan = create_punch_plan(params)

    print("\nForm Punch Operations:")
    for op in plan["ops"]:
        if op["op"] in ["3d_mill_form", "polish_contour"]:
            print(f"  ✓ {op['op']}: {op.get('note', '')}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "DWG PUNCH EXTRACTION EXAMPLES" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run examples
    summary = example_1_text_based_extraction()
    example_2_create_manufacturing_plan(summary)
    example_3_integrated_planning()
    example_4_form_punch()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
