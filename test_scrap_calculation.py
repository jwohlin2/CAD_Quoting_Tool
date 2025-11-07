"""
Test script for scrap calculation using DirectCostHelper.

This demonstrates the complete scrap calculation including:
1. Stock prep scrap (McMaster → Desired starting size)
2. Face milling scrap (Desired → Part envelope)
3. Hole drilling scrap (Holes removed from part)
"""

import sys
from pathlib import Path

# Add the cad_quoter module to path
sys.path.insert(0, str(Path(__file__).parent))

from cad_quoter.pricing.DirectCostHelper import (
    calculate_total_scrap,
    ScrapInfo
)


def test_scrap_calculation():
    """Test scrap calculation with a real CAD file."""

    # Test file
    cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dxf"

    print("\n" + "="*70)
    print("SCRAP CALCULATION TEST")
    print("="*70)

    # Test 1: Auto-lookup McMaster stock size
    print("\n1. Auto-lookup McMaster stock size")
    print("-" * 70)

    scrap = calculate_total_scrap(
        cad_file,
        material="aluminum MIC6",
        verbose=True
    )

    print("\nResults:")
    print(f"  McMaster stock: {scrap.mcmaster_length:.3f}\" x {scrap.mcmaster_width:.3f}\" x {scrap.mcmaster_thickness:.3f}\"")
    print(f"  McMaster volume: {scrap.mcmaster_volume:.4f} in³")
    print(f"  McMaster weight: {scrap.mcmaster_weight:.2f} lbs")
    print(f"\n  Desired stock: {scrap.desired_length:.3f}\" x {scrap.desired_width:.3f}\" x {scrap.desired_thickness:.3f}\"")
    print(f"  Desired volume: {scrap.desired_volume:.4f} in³")
    print(f"\n  Final part: {scrap.part_length:.3f}\" x {scrap.part_width:.3f}\" x {scrap.part_thickness:.3f}\"")
    print(f"  Part envelope volume: {scrap.part_envelope_volume:.4f} in³")
    print(f"  Part final volume (with holes): {scrap.part_final_volume:.4f} in³")
    print(f"  Final part weight: {scrap.final_part_weight:.2f} lbs")

    print(f"\n  SCRAP BREAKDOWN:")
    print(f"    Stock prep scrap: {scrap.stock_prep_scrap:.4f} in³")
    print(f"    Face milling scrap: {scrap.face_milling_scrap:.4f} in³")
    print(f"    Hole drilling scrap: {scrap.hole_drilling_scrap:.4f} in³")
    print(f"    TOTAL SCRAP: {scrap.total_scrap_volume:.4f} in³")
    print(f"    Total scrap weight: {scrap.total_scrap_weight:.2f} lbs")

    print(f"\n  PERCENTAGES:")
    print(f"    Material utilization: {scrap.utilization_percentage:.1f}%")
    print(f"    Scrap percentage: {scrap.scrap_percentage:.1f}%")

    # Verify the math
    print(f"\n  VERIFICATION:")
    print(f"    McMaster volume: {scrap.mcmaster_volume:.4f} in³")
    print(f"    - Total scrap: {scrap.total_scrap_volume:.4f} in³")
    print(f"    = Final part: {scrap.part_final_volume:.4f} in³")
    calculated_final = scrap.mcmaster_volume - scrap.total_scrap_volume
    print(f"    Calculated: {calculated_final:.4f} in³ (should match {scrap.part_final_volume:.4f} in³)")
    print(f"    Match: {'✓' if abs(calculated_final - scrap.part_final_volume) < 0.01 else '✗'}")

    # Test 2: Manual McMaster stock dimensions
    print("\n\n2. Manual McMaster stock dimensions")
    print("-" * 70)

    scrap2 = calculate_total_scrap(
        cad_file,
        material="P20 Tool Steel",
        mcmaster_length=18.0,
        mcmaster_width=14.0,
        mcmaster_thickness=2.5,
        verbose=True
    )

    print(f"\nWith larger stock:")
    print(f"  Stock prep scrap increases to: {scrap2.stock_prep_scrap:.4f} in³")
    print(f"  Total scrap: {scrap2.total_scrap_volume:.4f} in³")
    print(f"  Utilization: {scrap2.utilization_percentage:.1f}%")

    print("\n" + "="*70 + "\n")


def compare_materials():
    """Compare scrap for different materials."""

    cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dxf"

    print("\n" + "="*70)
    print("MATERIAL COMPARISON")
    print("="*70 + "\n")

    materials = [
        "aluminum MIC6",
        "P20 Tool Steel",
        "17-4 PH Stainless"
    ]

    print(f"{'Material':<25} {'Scrap Vol':<12} {'Scrap Wt':<12} {'Util %':<10}")
    print("-" * 70)

    for material in materials:
        scrap = calculate_total_scrap(
            cad_file,
            material=material,
            verbose=False
        )

        print(f"{material:<25} {scrap.total_scrap_volume:>10.2f} in³ {scrap.total_scrap_weight:>10.2f} lbs {scrap.utilization_percentage:>8.1f}%")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_scrap_calculation()
    compare_materials()
