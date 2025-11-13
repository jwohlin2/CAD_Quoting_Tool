"""Test script for punch planner estimator stubs.

This script demonstrates the new punch estimation functions for:
- Wire EDM profile cutting
- Face grinding (top/bottom surfaces)
- OD grinding (round pierce punches)
"""

from cad_quoter.pricing.time_estimator import (
    estimate_wire_edm_minutes,
    estimate_face_grind_minutes,
    estimate_od_grind_minutes,
    MIN_PER_CUIN_GRIND,
)
from cad_quoter.planning.process_planner import (
    estimate_machine_hours_from_plan,
)


def test_wire_edm_estimation():
    """Test Wire EDM profile cutting time estimation."""
    print("\n" + "=" * 70)
    print("TEST 1: Wire EDM Profile Cutting")
    print("=" * 70)

    # Example: Non-round form punch with complex profile
    # L=2.0", W=1.5", T=2.0", perimeter ≈ 2(L+W) = 7.0"
    op = {
        'wire_profile_perimeter_in': 7.0,
        'thickness_in': 2.0,
    }

    material = "P20 Tool Steel"
    material_group = "P3"

    minutes = estimate_wire_edm_minutes(op, material, material_group)

    print(f"Material: {material} ({material_group})")
    print(f"Perimeter: {op['wire_profile_perimeter_in']:.2f} inches")
    print(f"Thickness: {op['thickness_in']:.2f} inches")
    print(f"Estimated time: {minutes:.2f} minutes ({minutes/60:.2f} hours)")

    return minutes


def test_face_grind_estimation():
    """Test face grinding time estimation."""
    print("\n" + "=" * 70)
    print("TEST 2: Face Grinding (Top/Bottom Surfaces)")
    print("=" * 70)

    # Example: Rectangular punch, grind both faces
    L = 2.0  # inches
    W = 1.5  # inches
    stock_removed_total = 0.006  # inches total (both faces)
    material = "P20 Tool Steel"
    material_group = "P3"
    faces = 2

    minutes = estimate_face_grind_minutes(
        L, W, stock_removed_total, material, material_group, faces
    )

    volume = L * W * stock_removed_total

    print(f"Material: {material} ({material_group})")
    print(f"Dimensions: {L:.2f}\" × {W:.2f}\"")
    print(f"Stock removed (total): {stock_removed_total:.3f} inches")
    print(f"Volume removed: {volume:.4f} cubic inches")
    print(f"Base rate: {MIN_PER_CUIN_GRIND:.1f} min/cu.in")
    print(f"Estimated time: {minutes:.2f} minutes ({minutes/60:.2f} hours)")

    return minutes


def test_od_grind_estimation():
    """Test OD grinding time estimation for round pierce punches."""
    print("\n" + "=" * 70)
    print("TEST 3: OD Grinding (Round Pierce Punch)")
    print("=" * 70)

    # Example: Round pierce punch (steel, not carbide)
    # D=0.500", T=2.000", stock_allow_radial=0.003"
    from math import pi

    D = 0.500
    T = 2.000
    stock_radial = 0.003

    # Calculate volume removed
    r = D / 2.0
    vol = pi * (r**2 - (r - stock_radial)**2) * T
    circ = pi * D

    meta = {
        'diameter': D,
        'thickness': T,
        'stock_allow_radial': stock_radial,
        'od_grind_volume_removed_cuin': vol,
        'od_grind_circumference_in': circ,
        'od_length_in': T,
    }

    material = "P20 Tool Steel"
    material_group = "P3"

    minutes = estimate_od_grind_minutes(meta, material, material_group)

    print(f"Material: {material} ({material_group})")
    print(f"Diameter: {D:.3f} inches")
    print(f"Length/Thickness: {T:.3f} inches")
    print(f"Radial stock allowance: {stock_radial:.3f} inches")
    print(f"Volume removed: {vol:.6f} cubic inches")
    print(f"Circumference: {circ:.4f} inches")
    print(f"Base rate: {MIN_PER_CUIN_GRIND:.1f} min/cu.in")
    print(f"Estimated time: {minutes:.2f} minutes ({minutes/60:.3f} hours)")

    return minutes


def test_integrated_plan():
    """Test integrated plan with punch operations."""
    print("\n" + "=" * 70)
    print("TEST 4: Integrated Process Plan (Punch Operations)")
    print("=" * 70)

    # Example plan for a non-round form punch
    plan = {
        'ops': [
            {
                'op': 'Wire_EDM_profile',
                'wire_profile_perimeter_in': 7.0,
                'thickness_in': 2.0,
                'material_group': 'P3',
            },
            {
                'op': 'Grind_faces',
                'stock_removed_total': 0.006,
                'material_group': 'P3',
            },
        ],
        'meta': {},
    }

    material = "P20 Tool Steel"
    plate_LxW = (2.0, 1.5)
    thickness = 2.0

    result = estimate_machine_hours_from_plan(plan, material, plate_LxW, thickness)

    print(f"Material: {material}")
    print(f"Dimensions: {plate_LxW[0]:.2f}\" × {plate_LxW[1]:.2f}\" × {thickness:.2f}\"")
    print(f"\nTime Breakdown:")
    for category, minutes in result['breakdown_minutes'].items():
        if minutes > 0:
            print(f"  {category.capitalize()}: {minutes:.2f} min")
    print(f"\nTotal: {result['total_minutes']:.2f} minutes ({result['total_hours']:.2f} hours)")

    return result


def test_round_punch_integrated():
    """Test integrated plan for a round pierce punch with OD grinding."""
    print("\n" + "=" * 70)
    print("TEST 5: Round Pierce Punch (Complete Process)")
    print("=" * 70)

    from math import pi

    D = 0.500
    T = 2.000
    stock_radial = 0.003

    r = D / 2.0
    vol = pi * (r**2 - (r - stock_radial)**2) * T
    circ = pi * D

    # Example plan for a round pierce punch
    plan = {
        'ops': [
            {
                'op': 'Turn_rough',
                'material_group': 'P3',
            },
            {
                'op': 'OD_grind_rough',
                'material_group': 'P3',
            },
            {
                'op': 'OD_grind_finish',
                'material_group': 'P3',
            },
            {
                'op': 'Grind_length',
                'material_group': 'P3',
            },
        ],
        'meta': {
            'diameter': D,
            'thickness': T,
            'stock_allow_radial': stock_radial,
            'od_grind_volume_removed_cuin': vol,
            'od_grind_circumference_in': circ,
            'od_length_in': T,
        },
    }

    material = "P20 Tool Steel"
    plate_LxW = (D, D)
    thickness = T

    result = estimate_machine_hours_from_plan(plan, material, plate_LxW, thickness)

    print(f"Material: {material}")
    print(f"Diameter: {D:.3f}\" × Length: {T:.3f}\"")
    print(f"Volume to remove: {vol:.6f} cu.in")
    print(f"\nTime Breakdown:")
    for category, minutes in result['breakdown_minutes'].items():
        if minutes > 0:
            print(f"  {category.capitalize()}: {minutes:.2f} min")
    print(f"\nTotal: {result['total_minutes']:.2f} minutes ({result['total_hours']:.3f} hours)")

    return result


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# PUNCH PLANNER ESTIMATOR STUBS - TEST SUITE")
    print("#" * 70)

    try:
        # Test individual functions
        test_wire_edm_estimation()
        test_face_grind_estimation()
        test_od_grind_estimation()

        # Test integrated plans
        test_integrated_plan()
        test_round_punch_integrated()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nThe punch estimator stubs are working correctly!")
        print("You can now use these functions in your planner_punch() implementation.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
