#!/usr/bin/env python3
"""Test script for polish contour operation detection and time calculation."""

import sys
from pathlib import Path

# Add cad_quoter to path
cad_quoter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(cad_quoter_dir))

from cad_quoter.geometry.dxf_enrich import detect_polish_contour_operation
from cad_quoter.planning.process_planner import calc_polish_contour_minutes


def test_polish_detection():
    """Test that polish contour patterns are detected correctly."""
    print("=" * 60)
    print("Testing Polish Contour Detection")
    print("=" * 60)

    # Test cases
    test_cases = [
        ("POLISH CONTOUR", True, "Standard text"),
        ("POLISH CONTOURED", True, "Past tense variant"),
        ("POLISHED CONTOUR", True, "Polished variant"),
        ("CONTOUR POLISH", True, "Reversed order"),
        ("POLISH FORM", True, "Polish form"),
        ("POLISH RADIUS", True, "Polish radius"),
        ("POLISH SURFACE", True, "Polish surface"),
        ("SOME OTHER TEXT", False, "No polish"),
        ("", False, "Empty string"),
        ("Text before\nPOLISH CONTOUR\nText after", True, "Multiline"),
        ("POLISH FLAT SURFACE", True, "Polish with flat"),
        ("NO POLISH", False, "Negation - should not match"),
        ("POLISHING COMPOUND", False, "Different context"),
    ]

    passed = 0
    failed = 0

    for text, expected, description in test_cases:
        result = detect_polish_contour_operation(text)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status}: {description:30s} | Expected: {expected:5} | Got: {result:5}")

    print(f"\nDetection Tests: {passed} passed, {failed} failed")
    print()


def test_polish_time_calculation():
    """Test that polish contour time calculation works correctly."""
    print("=" * 60)
    print("Testing Polish Contour Time Calculation")
    print("=" * 60)

    # Test cases: (has_polish, qty, length, width, description)
    test_cases = [
        (True, 1, None, None, "Single part, default dims (0.40×0.25)"),
        (True, 1, 0.40, 0.25, "Single part, explicit default dims"),
        (True, 10, None, None, "10 parts, default dims"),
        (True, 1, 0.80, 0.50, "Single part, larger contour (0.80×0.50)"),
        (True, 10, 0.80, 0.50, "10 parts, larger contour"),
        (True, 1, 0.20, 0.15, "Single part, smaller contour (0.20×0.15)"),
        (False, 10, None, None, "No polish requirement"),
        (True, 5, 0.50, 0.30, "5 parts, medium contour (0.50×0.30)"),
    ]

    for has_polish, qty, length, width, description in test_cases:
        time_min = calc_polish_contour_minutes(has_polish, qty, length, width)
        # Calculate area for display
        l = length if length is not None else 0.40
        w = width if width is not None else 0.25
        area = l * w if has_polish else 0.0
        print(f"{description:45s} | Qty: {qty:3d} | Area: {area:.4f} sq.in | Time: {time_min:6.2f} min")

    print()


def test_integration():
    """Test integration with plan."""
    print("=" * 60)
    print("Testing Integration")
    print("=" * 60)

    # Create a mock plan with polish contour requirement
    mock_plan = {
        'ops': [],
        'has_polish_contour': True,
        'text_dump': 'POLISH CONTOUR'
    }

    print(f"Mock plan has_polish_contour: {mock_plan.get('has_polish_contour')}")
    print(f"Text dump contains polish instruction: "
          f"{detect_polish_contour_operation(mock_plan['text_dump'])}")

    # Test time calculation
    from cad_quoter.planning.process_planner import estimate_machine_hours_from_plan

    L, W, T = 8.0, 4.0, 0.5  # 8" x 4" x 0.5" plate
    result = estimate_machine_hours_from_plan(
        mock_plan,
        material="P20 Tool Steel",
        plate_LxW=(L, W),
        thickness=T
    )

    polish_time = result.get('polish_minutes', 0.0)
    print(f"\nPlate dimensions: {L:.1f}\" x {W:.1f}\" x {T:.3f}\"")
    print(f"Polish contour time: {polish_time:.2f} minutes")

    if polish_time > 0:
        print("✓ PASS: Polish contour time calculated successfully")
    else:
        print("✗ FAIL: Polish contour time is zero")

    print()


def test_time_formula_accuracy():
    """Test that time formula matches expected values."""
    print("=" * 60)
    print("Testing Time Formula Accuracy")
    print("=" * 60)

    # Test formula: SETUP + qty * (BASE_PER_PART + area * MIN_PER_SQIN), min MIN_POLISH_TIME
    POLISH_SETUP_MIN = 2.0
    POLISH_MIN_PER_SQIN = 6.0
    POLISH_BASE_PER_PART = 0.5
    MIN_POLISH_TIME_PER_LOT = 5.0

    test_cases = [
        # (qty, length, width, expected_time)
        (1, 0.40, 0.25, max(POLISH_SETUP_MIN + 1 * (POLISH_BASE_PER_PART + 0.40*0.25*POLISH_MIN_PER_SQIN), MIN_POLISH_TIME_PER_LOT)),
        (10, 0.40, 0.25, max(POLISH_SETUP_MIN + 10 * (POLISH_BASE_PER_PART + 0.40*0.25*POLISH_MIN_PER_SQIN), MIN_POLISH_TIME_PER_LOT)),
        (1, 0.80, 0.50, max(POLISH_SETUP_MIN + 1 * (POLISH_BASE_PER_PART + 0.80*0.50*POLISH_MIN_PER_SQIN), MIN_POLISH_TIME_PER_LOT)),
        (5, 0.50, 0.30, max(POLISH_SETUP_MIN + 5 * (POLISH_BASE_PER_PART + 0.50*0.30*POLISH_MIN_PER_SQIN), MIN_POLISH_TIME_PER_LOT)),
    ]

    all_passed = True
    for qty, length, width, expected_time in test_cases:
        actual_time = calc_polish_contour_minutes(True, qty, length, width)
        matches = abs(actual_time - expected_time) < 0.01
        status = "✓ PASS" if matches else "✗ FAIL"
        if not matches:
            all_passed = False
        area = length * width
        print(f"{status}: Qty={qty:3d}, Area={area:.4f} sq.in | "
              f"Expected: {expected_time:6.2f} min | Actual: {actual_time:6.2f} min")

    if all_passed:
        print("\n✓ All formula accuracy tests passed!")
    else:
        print("\n✗ Some formula accuracy tests failed")

    print()


def test_default_dimensions():
    """Test that default dimensions are applied correctly."""
    print("=" * 60)
    print("Testing Default Dimensions")
    print("=" * 60)

    # Test with None dimensions (should use defaults: 0.40" × 0.25")
    time_with_none = calc_polish_contour_minutes(True, 1, None, None)
    time_with_defaults = calc_polish_contour_minutes(True, 1, 0.40, 0.25)

    print(f"Time with None dimensions:    {time_with_none:.2f} min")
    print(f"Time with explicit defaults:  {time_with_defaults:.2f} min")

    if abs(time_with_none - time_with_defaults) < 0.01:
        print("✓ PASS: Default dimensions applied correctly")
    else:
        print("✗ FAIL: Default dimensions not matching")

    print()


if __name__ == "__main__":
    try:
        test_polish_detection()
        test_polish_time_calculation()
        test_time_formula_accuracy()
        test_default_dimensions()
        test_integration()
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
