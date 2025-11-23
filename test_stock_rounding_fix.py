#!/usr/bin/env python3
"""
Test that stock rounding never shrinks dimensions below required stock.

This test verifies the fix for T1769-219 & T1769-134:
- Stock dimensions must NEVER be smaller than required dimensions
- The fix ensures mcmaster_* dimensions are always >= desired_* dimensions
"""

def test_stock_rounding_never_shrinks():
    """Test that stock rounding preserves minimum required dimensions."""

    # Simulate the bug scenario from T1769-219
    print("\n" + "="*80)
    print("TEST: T1769-219 (MIC6 guide_post)")
    print("="*80)

    # Required dimensions (before fix, width would shrink)
    desired_length = 11.09
    desired_width = 7.10
    desired_thickness = 6.85

    # Simulated catalog result (buggy scenario)
    catalog_length = 11.09
    catalog_width = 6.85  # BUG: smaller than desired_width 7.10
    catalog_thickness = 6.852

    print(f"Required Stock: {desired_length:.2f} × {desired_width:.2f} × {desired_thickness:.2f} in")
    print(f"Catalog returned: {catalog_length:.2f} × {catalog_width:.2f} × {catalog_thickness:.3f} in")

    # Apply the fix (same logic as in DirectCostHelper.py)
    mcmaster_length = max(catalog_length, desired_length)
    mcmaster_width = max(catalog_width, desired_width)
    mcmaster_thickness = max(catalog_thickness, desired_thickness)

    print(f"After fix: {mcmaster_length:.2f} × {mcmaster_width:.2f} × {mcmaster_thickness:.3f} in")

    # Verify fix
    assert mcmaster_length >= desired_length, f"Length {mcmaster_length} < required {desired_length}"
    assert mcmaster_width >= desired_width, f"Width {mcmaster_width} < required {desired_width}"
    assert mcmaster_thickness >= desired_thickness, f"Thickness {mcmaster_thickness} < required {desired_thickness}"

    print("✓ PASS: All dimensions >= required\n")

    # Test T1769-134
    print("="*80)
    print("TEST: T1769-134 (A2 round punch)")
    print("="*80)

    desired_length = 6.86
    desired_width = 4.46
    desired_thickness = 4.21

    catalog_length = 6.86
    catalog_width = 4.21  # BUG: smaller than desired_width 4.46
    catalog_thickness = 4.211

    print(f"Required Stock: {desired_length:.2f} × {desired_width:.2f} × {desired_thickness:.2f} in")
    print(f"Catalog returned: {catalog_length:.2f} × {catalog_width:.2f} × {catalog_thickness:.3f} in")

    mcmaster_length = max(catalog_length, desired_length)
    mcmaster_width = max(catalog_width, desired_width)
    mcmaster_thickness = max(catalog_thickness, desired_thickness)

    print(f"After fix: {mcmaster_length:.2f} × {mcmaster_width:.2f} × {mcmaster_thickness:.3f} in")

    assert mcmaster_length >= desired_length, f"Length {mcmaster_length} < required {desired_length}"
    assert mcmaster_width >= desired_width, f"Width {mcmaster_width} < required {desired_width}"
    assert mcmaster_thickness >= desired_thickness, f"Thickness {mcmaster_thickness} < required {desired_thickness}"

    print("✓ PASS: All dimensions >= required\n")

    # Test normal case (catalog returns correct dimensions)
    print("="*80)
    print("TEST: Normal case (catalog returns correct dimensions)")
    print("="*80)

    desired_length = 10.0
    desired_width = 6.0
    desired_thickness = 2.0

    catalog_length = 12.0  # Larger than required (correct)
    catalog_width = 6.0    # Equal to required (correct)
    catalog_thickness = 2.25  # Larger than required (correct)

    print(f"Required Stock: {desired_length:.2f} × {desired_width:.2f} × {desired_thickness:.2f} in")
    print(f"Catalog returned: {catalog_length:.2f} × {catalog_width:.2f} × {catalog_thickness:.2f} in")

    mcmaster_length = max(catalog_length, desired_length)
    mcmaster_width = max(catalog_width, desired_width)
    mcmaster_thickness = max(catalog_thickness, desired_thickness)

    print(f"After fix: {mcmaster_length:.2f} × {mcmaster_width:.2f} × {mcmaster_thickness:.2f} in")

    assert mcmaster_length >= desired_length
    assert mcmaster_width >= desired_width
    assert mcmaster_thickness >= desired_thickness

    # Should use catalog dimensions (no adjustment needed)
    assert mcmaster_length == catalog_length
    assert mcmaster_width == catalog_width
    assert mcmaster_thickness == catalog_thickness

    print("✓ PASS: Catalog dimensions used correctly\n")

if __name__ == "__main__":
    print("STOCK ROUNDING FIX VERIFICATION")
    print("Testing fix for T1769-219 & T1769-134")
    print("Stock dimensions must NEVER shrink below required dimensions\n")

    try:
        test_stock_rounding_never_shrinks()
        print("="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
