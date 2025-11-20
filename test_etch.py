#!/usr/bin/env python3
"""Test script for etch operation detection and time calculation."""

import sys
from pathlib import Path

# Add cad_quoter to path
cad_quoter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(cad_quoter_dir))

from cad_quoter.geometry.dxf_enrich import detect_etch_operation
from cad_quoter.planning.process_planner import calc_etch_minutes


def test_etch_detection():
    """Test that etch patterns are detected correctly."""
    print("=" * 60)
    print("Testing Etch Operation Detection")
    print("=" * 60)

    # Test cases
    test_cases = [
        ("ETCH ON DETAIL; VENDOR & DRAWING NO.", True, "Standard text"),
        ("ETCH ON DETAIL", True, "Short variant"),
        ("ETCH DETAIL", True, "Shorter variant"),
        ("ETCH VENDOR & DRAWING NO.", True, "Vendor & drawing"),
        ("ETCH DRAWING NO.", True, "Drawing number"),
        ("ETCH PART NUMBER", True, "Part number"),
        ("ETCH P/N", True, "P/N variant"),
        ("MARK ON DETAIL", True, "Mark variant"),
        ("LASER ETCH PART NO.", True, "Laser etch"),
        ("ELECTRO-ETCH SERIAL NO.", True, "Electro-etch"),
        ("SOME OTHER TEXT", False, "No etch"),
        ("", False, "Empty string"),
        ("Text before\nETCH ON DETAIL; VENDOR & DRAWING NO.\nText after", True, "Multiline"),
        ("SKETCH ON DETAIL", False, "Similar but not etch"),
    ]

    passed = 0
    failed = 0

    for text, expected, description in test_cases:
        result = detect_etch_operation(text)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status}: {description:30s} | Expected: {expected:5} | Got: {result:5}")

    print(f"\nDetection Tests: {passed} passed, {failed} failed")
    print()


def test_etch_time_calculation():
    """Test that etch time calculation works correctly."""
    print("=" * 60)
    print("Testing Etch Time Calculation")
    print("=" * 60)

    # Test cases: (has_etch_note, qty, details_with_etch, description)
    test_cases = [
        (True, 1, 1, "Single part, 1 mark"),
        (True, 10, 1, "10 parts, 1 mark each"),
        (True, 1, 2, "Single part, 2 marks"),
        (True, 10, 2, "10 parts, 2 marks each"),
        (True, 100, 1, "Large lot (100 parts)"),
        (False, 10, 1, "No etch requirement"),
        (True, 2, 1, "Small lot (2 parts)"),
    ]

    for has_etch, qty, details, description in test_cases:
        time_min = calc_etch_minutes(has_etch, qty, details)
        mark_count = qty * details if has_etch else 0
        print(f"{description:30s} | Qty: {qty:3d} | Marks/part: {details} | "
              f"Total marks: {mark_count:3d} | Time: {time_min:6.2f} min")

    print()


def test_integration():
    """Test integration with plan."""
    print("=" * 60)
    print("Testing Integration")
    print("=" * 60)

    # Create a mock plan with etch requirement
    mock_plan = {
        'ops': [],
        'has_etch': True,
        'text_dump': 'ETCH ON DETAIL; VENDOR & DRAWING NO.'
    }

    print(f"Mock plan has_etch: {mock_plan.get('has_etch')}")
    print(f"Text dump contains etch instruction: "
          f"{detect_etch_operation(mock_plan['text_dump'])}")

    # Test time calculation
    from cad_quoter.planning.process_planner import estimate_machine_hours_from_plan

    L, W, T = 8.0, 4.0, 0.5  # 8" x 4" x 0.5" plate
    result = estimate_machine_hours_from_plan(
        mock_plan,
        material="P20 Tool Steel",
        plate_LxW=(L, W),
        thickness=T
    )

    etch_time = result.get('etch_minutes', 0.0)
    print(f"\nPlate dimensions: {L:.1f}\" x {W:.1f}\" x {T:.3f}\"")
    print(f"Etch time: {etch_time:.2f} minutes")

    if etch_time > 0:
        print("✓ PASS: Etch time calculated successfully")
    else:
        print("✗ FAIL: Etch time is zero")

    print()


def test_time_formula_accuracy():
    """Test that time formula matches expected values."""
    print("=" * 60)
    print("Testing Time Formula Accuracy")
    print("=" * 60)

    # Test formula: ETCH_SETUP_MIN + mark_count * ETCH_MIN_PER_MARK, min MIN_ETCH_TIME_PER_LOT
    ETCH_SETUP_MIN = 4.0
    ETCH_MIN_PER_MARK = 0.75
    MIN_ETCH_TIME_PER_LOT = 5.0

    test_cases = [
        (1, 1, max(ETCH_SETUP_MIN + 1 * ETCH_MIN_PER_MARK, MIN_ETCH_TIME_PER_LOT)),
        (10, 1, max(ETCH_SETUP_MIN + 10 * ETCH_MIN_PER_MARK, MIN_ETCH_TIME_PER_LOT)),
        (1, 2, max(ETCH_SETUP_MIN + 2 * ETCH_MIN_PER_MARK, MIN_ETCH_TIME_PER_LOT)),
        (10, 2, max(ETCH_SETUP_MIN + 20 * ETCH_MIN_PER_MARK, MIN_ETCH_TIME_PER_LOT)),
    ]

    all_passed = True
    for qty, details, expected_time in test_cases:
        actual_time = calc_etch_minutes(True, qty, details)
        matches = abs(actual_time - expected_time) < 0.01
        status = "✓ PASS" if matches else "✗ FAIL"
        if not matches:
            all_passed = False
        mark_count = qty * details
        print(f"{status}: Qty={qty:3d}, Marks/part={details}, Total marks={mark_count:3d} | "
              f"Expected: {expected_time:6.2f} min | Actual: {actual_time:6.2f} min")

    if all_passed:
        print("\n✓ All formula accuracy tests passed!")
    else:
        print("\n✗ Some formula accuracy tests failed")

    print()


if __name__ == "__main__":
    try:
        test_etch_detection()
        test_etch_time_calculation()
        test_time_formula_accuracy()
        test_integration()
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
