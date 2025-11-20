#!/usr/bin/env python3
"""Test script for edge break operation detection and time calculation."""

import sys
from pathlib import Path

# Add cad_quoter to path
cad_quoter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(cad_quoter_dir))

from cad_quoter.geometry.dxf_enrich import detect_edge_break_operation
from cad_quoter.planning.process_planner import calc_edge_break_minutes


def test_edge_break_detection():
    """Test that edge break patterns are detected correctly."""
    print("=" * 60)
    print("Testing Edge Break Detection")
    print("=" * 60)

    # Test cases
    test_cases = [
        ("BREAK ALL OUTSIDE SHARP CORNERS", True, "Standard text"),
        ("BREAK ALL SHARP CORNERS", True, "Variant 1"),
        ("BREAK ALL OUTSIDE CORNERS", True, "Variant 2"),
        ("BREAK ALL EDGES", True, "Variant 3"),
        ("DEBURR ALL EDGES", True, "Deburr variant"),
        ("DEBURR ALL CORNERS", True, "Deburr variant 2"),
        ("SOME OTHER TEXT", False, "No edge break"),
        ("", False, "Empty string"),
        ("BREAK SHARP CORNERS", True, "Short variant"),
        ("Text before\nBREAK ALL OUTSIDE SHARP CORNERS\nText after", True, "Multiline"),
    ]

    passed = 0
    failed = 0

    for text, expected, description in test_cases:
        result = detect_edge_break_operation(text)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status}: {description:30s} | Expected: {expected:5} | Got: {result:5}")

    print(f"\nDetection Tests: {passed} passed, {failed} failed")
    print()


def test_edge_break_time_calculation():
    """Test that edge break time calculation works correctly."""
    print("=" * 60)
    print("Testing Edge Break Time Calculation")
    print("=" * 60)

    # Test cases: (perimeter, qty, material, description)
    test_cases = [
        (10.0, 1, "ALUMINUM", "Small aluminum part"),
        (10.0, 1, "TOOL_STEEL", "Small tool steel part"),
        (10.0, 10, "TOOL_STEEL", "10 tool steel parts"),
        (50.0, 1, "TOOL_STEEL", "Large tool steel part"),
        (10.0, 1, "STAINLESS", "Small stainless part"),
        (10.0, 1, "CARBIDE", "Small carbide part"),
        (0.0, 1, "TOOL_STEEL", "Zero perimeter"),
    ]

    for perim, qty, material, description in test_cases:
        time_min = calc_edge_break_minutes(perim, qty, material)
        edge_length = 2.0 * perim * qty
        print(f"{description:30s} | Perim: {perim:5.1f}\" | Qty: {qty:3d} | "
              f"Edge: {edge_length:6.1f}\" | Time: {time_min:6.2f} min")

    print()


def test_integration():
    """Test integration with plan."""
    print("=" * 60)
    print("Testing Integration")
    print("=" * 60)

    # Create a mock plan with edge break requirement
    mock_plan = {
        'ops': [],
        'has_edge_break': True,
        'text_dump': 'BREAK ALL OUTSIDE SHARP CORNERS'
    }

    print(f"Mock plan has_edge_break: {mock_plan.get('has_edge_break')}")
    print(f"Text dump contains edge break instruction: "
          f"{detect_edge_break_operation(mock_plan['text_dump'])}")

    # Test time calculation with typical plate dimensions
    from cad_quoter.planning.process_planner import estimate_machine_hours_from_plan

    L, W, T = 8.0, 4.0, 0.5  # 8" x 4" x 0.5" plate
    result = estimate_machine_hours_from_plan(
        mock_plan,
        material="P20 Tool Steel",
        plate_LxW=(L, W),
        thickness=T
    )

    edge_break_time = result.get('edge_break_minutes', 0.0)
    print(f"\nPlate dimensions: {L:.1f}\" x {W:.1f}\" x {T:.3f}\"")
    print(f"Edge break time: {edge_break_time:.2f} minutes")

    if edge_break_time > 0:
        print("✓ PASS: Edge break time calculated successfully")
    else:
        print("✗ FAIL: Edge break time is zero")

    print()


if __name__ == "__main__":
    try:
        test_edge_break_detection()
        test_edge_break_time_calculation()
        test_integration()
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
