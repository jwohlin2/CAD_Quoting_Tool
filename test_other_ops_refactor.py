#!/usr/bin/env python3
"""Test script for the refactored 'Other operations' bucket.

This script tests the new text parsers and time calculation functions
for explicit sub-operations like small undercuts, chamfers, lift specs, etc.
"""

from cad_quoter.geometry.dxf_enrich import (
    detect_small_undercut,
    detect_lift_spec,
    detect_waterjet_cleanup,
    detect_chamfer_details,
    detect_lead_in_notes
)
from cad_quoter.planning.process_planner import (
    calc_small_undercut_minutes,
    calc_lift_spec_minutes,
    calc_waterjet_cleanup_minutes,
    calc_chamfer_detail_minutes,
    calc_lead_in_minutes
)


def test_small_undercut_detection():
    """Test small undercut detection."""
    print("\n=== Testing Small Undercut Detection ===")

    # Test case 1: Small undercut with radius
    text1 = "SMALL UNDERCUT, R .020 MK'D \"A\""
    has_undercut, radius = detect_small_undercut(text1)
    print(f"Test 1: '{text1}'")
    print(f"  Result: has_undercut={has_undercut}, radius={radius:.3f}")
    assert has_undercut == True
    assert abs(radius - 0.020) < 0.001

    # Test case 2: Undercut without specific radius
    text2 = "UNDERCUT REQUIRED"
    has_undercut, radius = detect_small_undercut(text2)
    print(f"Test 2: '{text2}'")
    print(f"  Result: has_undercut={has_undercut}, radius={radius:.3f} (default)")
    assert has_undercut == True

    # Test case 3: No undercut
    text3 = "NO SPECIAL OPERATIONS"
    has_undercut, radius = detect_small_undercut(text3)
    print(f"Test 3: '{text3}'")
    print(f"  Result: has_undercut={has_undercut}")
    assert has_undercut == False

    print("✓ All small undercut detection tests passed!")


def test_lift_spec_detection():
    """Test lift spec detection."""
    print("\n=== Testing Lift Spec Detection ===")

    # Test case 1: Lift with dimension
    text1 = "LIFT: .0600"
    has_lift, dimension = detect_lift_spec(text1)
    print(f"Test 1: '{text1}'")
    print(f"  Result: has_lift={has_lift}, dimension={dimension:.4f}")
    assert has_lift == True
    assert abs(dimension - 0.0600) < 0.0001

    # Test case 2: Height specification
    text2 = "HEIGHT: 0.0450"
    has_lift, dimension = detect_lift_spec(text2)
    print(f"Test 2: '{text2}'")
    print(f"  Result: has_lift={has_lift}, dimension={dimension:.4f}")
    assert has_lift == True
    assert abs(dimension - 0.0450) < 0.0001

    # Test case 3: No lift spec
    text3 = "STANDARD DIMENSIONS"
    has_lift, dimension = detect_lift_spec(text3)
    print(f"Test 3: '{text3}'")
    print(f"  Result: has_lift={has_lift}")
    assert has_lift == False

    print("✓ All lift spec detection tests passed!")


def test_waterjet_cleanup_detection():
    """Test waterjet cleanup detection."""
    print("\n=== Testing Waterjet Cleanup Detection ===")

    # Test case 1: Waterjet cleanup
    text1 = "WATERJET CLEANUP REQUIRED"
    has_cleanup = detect_waterjet_cleanup(text1)
    print(f"Test 1: '{text1}'")
    print(f"  Result: has_cleanup={has_cleanup}")
    assert has_cleanup == True

    # Test case 2: Waterjet blend
    text2 = "BLEND WATERJET CHANNELS"
    has_cleanup = detect_waterjet_cleanup(text2)
    print(f"Test 2: '{text2}'")
    print(f"  Result: has_cleanup={has_cleanup}")
    assert has_cleanup == True

    # Test case 3: No cleanup
    text3 = "WATERJET ALL OPENINGS"  # This is cutting, not cleanup
    has_cleanup = detect_waterjet_cleanup(text3)
    print(f"Test 3: '{text3}'")
    print(f"  Result: has_cleanup={has_cleanup}")
    assert has_cleanup == False

    print("✓ All waterjet cleanup detection tests passed!")


def test_chamfer_details_detection():
    """Test chamfer details detection."""
    print("\n=== Testing Chamfer Details Detection ===")

    # Test case 1: Multiple chamfers with quantity
    text1 = "(4) .040 × 45°"
    chamfers = detect_chamfer_details(text1)
    print(f"Test 1: '{text1}'")
    print(f"  Result: {chamfers}")
    assert len(chamfers) == 1
    assert chamfers[0]['dimension'] == 0.040
    assert chamfers[0]['angle'] == 45
    assert chamfers[0]['quantity'] == 4

    # Test case 2: Multiple different chamfers
    text2 = "(2) .007 X 45° AND (4) .020 × 45°"
    chamfers = detect_chamfer_details(text2)
    print(f"Test 2: '{text2}'")
    print(f"  Result: {chamfers}")
    assert len(chamfers) == 2

    # Test case 3: Single chamfer without quantity
    text3 = ".030 × 45°"
    chamfers = detect_chamfer_details(text3)
    print(f"Test 3: '{text3}'")
    print(f"  Result: {chamfers}")
    assert len(chamfers) == 1
    assert chamfers[0]['quantity'] == 1

    print("✓ All chamfer details detection tests passed!")


def test_lead_in_detection():
    """Test lead-in notes detection."""
    print("\n=== Testing Lead-in Notes Detection ===")

    # Test case 1: Lead-in note
    text1 = "LEAD-IN REQUIRED"
    has_lead_in = detect_lead_in_notes(text1)
    print(f"Test 1: '{text1}'")
    print(f"  Result: has_lead_in={has_lead_in}")
    assert has_lead_in == True

    # Test case 2: Approach note
    text2 = "SPECIAL APPROACH ANGLE"
    has_lead_in = detect_lead_in_notes(text2)
    print(f"Test 2: '{text2}'")
    print(f"  Result: has_lead_in={has_lead_in}")
    assert has_lead_in == True

    # Test case 3: No lead-in
    text3 = "STANDARD MACHINING"
    has_lead_in = detect_lead_in_notes(text3)
    print(f"Test 3: '{text3}'")
    print(f"  Result: has_lead_in={has_lead_in}")
    assert has_lead_in == False

    print("✓ All lead-in detection tests passed!")


def test_time_calculations():
    """Test time calculation functions."""
    print("\n=== Testing Time Calculations ===")

    # Test small undercut time
    undercut_time = calc_small_undercut_minutes(
        has_small_undercut=True,
        undercut_radius=0.020,
        qty=10,
        material_group="TOOL_STEEL",
        operation_type="turn"
    )
    print(f"Small undercut time (10 parts, R.020, tool steel, turn): {undercut_time:.1f} min")
    assert undercut_time > 0

    # Test lift spec time
    lift_time = calc_lift_spec_minutes(
        has_lift_spec=True,
        lift_dimension=0.0600,
        qty=5
    )
    print(f"Lift spec time (5 parts, 0.0600\"): {lift_time:.1f} min")
    assert lift_time > 0

    # Test waterjet cleanup time
    cleanup_time = calc_waterjet_cleanup_minutes(
        has_waterjet_cleanup=True,
        qty=3,
        part_size_factor=1.5
    )
    print(f"Waterjet cleanup time (3 parts, medium-large): {cleanup_time:.1f} min")
    assert cleanup_time > 0

    # Test chamfer detail time
    chamfer_spec = {"dimension": 0.040, "angle": 45, "quantity": 4}
    chamfer_time = calc_chamfer_detail_minutes(chamfer_spec, "TOOL_STEEL")
    print(f"Chamfer time (4 × .040 × 45°, tool steel): {chamfer_time:.1f} min")
    assert chamfer_time > 0

    # Test lead-in time
    lead_in_time = calc_lead_in_minutes(has_lead_in=True, qty=8)
    print(f"Lead-in time (8 parts): {lead_in_time:.1f} min")
    assert lead_in_time > 0

    print("✓ All time calculation tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing 'Other Operations' Bucket Refactor")
    print("=" * 60)

    try:
        test_small_undercut_detection()
        test_lift_spec_detection()
        test_waterjet_cleanup_detection()
        test_chamfer_details_detection()
        test_lead_in_detection()
        test_time_calculations()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
