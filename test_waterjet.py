#!/usr/bin/env python3
"""Test script for waterjet operation detection and time calculation."""

import sys
from pathlib import Path

# Add cad_quoter to path
cad_quoter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(cad_quoter_dir))

from cad_quoter.geometry.dxf_enrich import detect_waterjet_openings, detect_waterjet_profile
from cad_quoter.planning.process_planner import calc_waterjet_minutes


def test_waterjet_openings_detection():
    """Test that waterjet openings patterns are detected correctly."""
    print("=" * 70)
    print("Testing Waterjet Openings Detection")
    print("=" * 70)

    # Test cases: (text, expected_detected, expected_tolerance, description)
    test_cases = [
        ("WATERJET ALL OPENINGS ±.005", True, 0.005, "Standard openings with ±.005"),
        ("WATERJET ALL OPENINGS ±.003", True, 0.003, "Tight tolerance ±.003"),
        ("WATERJET OPENINGS ±0.005", True, 0.005, "Without 'ALL', with leading 0"),
        ("WATER JET ALL OPENINGS ±.005", True, 0.005, "Two-word variant"),
        ("WATER JET OPENINGS +/-.005", True, 0.005, "Plus/minus notation"),
        ("SOME OTHER TEXT", False, 0.0, "No waterjet"),
        ("", False, 0.0, "Empty string"),
        ("WATERJET ALL OPENINGS", True, 0.005, "No explicit tolerance (default)"),
        ("Text before\nWATERJET ALL OPENINGS ±.003\nText after", True, 0.003, "Multiline"),
        ("LASER CUT OPENINGS", False, 0.0, "Different process"),
    ]

    passed = 0
    failed = 0

    for text, expected_detected, expected_tol, description in test_cases:
        detected, tolerance = detect_waterjet_openings(text)

        # Check both detection and tolerance
        detection_ok = detected == expected_detected
        tolerance_ok = abs(tolerance - expected_tol) < 0.0001
        ok = detection_ok and tolerance_ok

        status = "✓ PASS" if ok else "✗ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        tol_str = f"±{tolerance:.3f}" if detected else "N/A"
        exp_tol_str = f"±{expected_tol:.3f}" if expected_detected else "N/A"
        print(f"{status}: {description:35s} | Expected: {str(expected_detected):5s} {exp_tol_str:8s} | Got: {str(detected):5s} {tol_str:8s}")

    print(f"\nDetection Tests: {passed} passed, {failed} failed")
    print()


def test_waterjet_profile_detection():
    """Test that waterjet profile patterns are detected correctly."""
    print("=" * 70)
    print("Testing Waterjet Profile Detection")
    print("=" * 70)

    # Test cases: (text, expected_detected, expected_tolerance, description)
    test_cases = [
        ("WATERJET TO ±.003", True, 0.003, "Standard profile with ±.003"),
        ("WATERJET TO ±.005", True, 0.005, "Standard profile with ±.005"),
        ("WATER JET TO ±.003", True, 0.003, "Two-word variant"),
        ("WATERJET TO +/-.003", True, 0.003, "Plus/minus notation"),
        ("WATERJET PROFILE", True, 0.003, "Profile keyword (default tol)"),
        ("WATER JET PROFILE", True, 0.003, "Two-word profile"),
        ("WATERJET CUT", True, 0.003, "Cut keyword"),
        ("WATER JET CUT ±.002", True, 0.002, "Cut with tolerance"),
        ("SOME OTHER TEXT", False, 0.0, "No waterjet"),
        ("", False, 0.0, "Empty string"),
        ("Text before\nWATERJET TO ±.003\nText after", True, 0.003, "Multiline"),
        ("LASER CUT PROFILE", False, 0.0, "Different process"),
    ]

    passed = 0
    failed = 0

    for text, expected_detected, expected_tol, description in test_cases:
        detected, tolerance = detect_waterjet_profile(text)

        # Check both detection and tolerance
        detection_ok = detected == expected_detected
        tolerance_ok = abs(tolerance - expected_tol) < 0.0001
        ok = detection_ok and tolerance_ok

        status = "✓ PASS" if ok else "✗ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        tol_str = f"±{tolerance:.3f}" if detected else "N/A"
        exp_tol_str = f"±{expected_tol:.3f}" if expected_detected else "N/A"
        print(f"{status}: {description:35s} | Expected: {str(expected_detected):5s} {exp_tol_str:8s} | Got: {str(detected):5s} {tol_str:8s}")

    print(f"\nDetection Tests: {passed} passed, {failed} failed")
    print()


def test_waterjet_time_calculation():
    """Test that waterjet time calculation works correctly."""
    print("=" * 70)
    print("Testing Waterjet Time Calculation")
    print("=" * 70)

    # Test cases: (length, thickness, tolerance, pierce_count, qty, description)
    test_cases = [
        # Basic scenarios
        (10.0, 0.5, 0.005, 4, 1, "Single part, 10\" cut, 0.5\" thick, std tol"),
        (10.0, 0.5, 0.003, 4, 1, "Single part, 10\" cut, tight tolerance"),
        (10.0, 0.5, 0.010, 4, 1, "Single part, 10\" cut, loose tolerance"),

        # Thickness variations
        (10.0, 0.25, 0.005, 4, 1, "Single part, thin material (0.25\")"),
        (10.0, 1.0, 0.005, 4, 1, "Single part, thick material (1.0\")"),
        (10.0, 2.0, 0.005, 4, 1, "Single part, very thick material (2.0\")"),

        # Quantity variations
        (10.0, 0.5, 0.005, 4, 10, "10 parts, 10\" cut each"),
        (10.0, 0.5, 0.005, 4, 100, "100 parts, 10\" cut each"),

        # Length variations
        (5.0, 0.5, 0.005, 2, 1, "Single part, short cut (5\")"),
        (50.0, 0.5, 0.005, 10, 1, "Single part, long cut (50\")"),

        # Pierce count variations
        (10.0, 0.5, 0.005, 1, 1, "Single part, 1 pierce"),
        (10.0, 0.5, 0.005, 20, 1, "Single part, 20 pierces"),

        # Edge cases
        (0.0, 0.5, 0.005, 4, 1, "Zero length (should return 0)"),
        (10.0, 0.5, 0.005, 4, 0, "Zero quantity (should return 0)"),
        (-10.0, 0.5, 0.005, 4, 1, "Negative length (should return 0)"),
        (10.0, 0.5, 0.005, 4, -1, "Negative quantity (should return 0)"),

        # Profile cutting (typically outer perimeter)
        (30.0, 0.75, 0.003, 1, 1, "Profile cut: 30\" perimeter, 0.75\" thick"),
        (30.0, 0.75, 0.003, 1, 50, "Profile cut: 50 parts"),
    ]

    print(f"{'Description':<45s} | {'Length':<6s} | {'Thk':<5s} | {'Tol':<6s} | {'Pierce':<6s} | {'Qty':<4s} | {'Time (min)':<10s}")
    print("-" * 110)

    for length, thickness, tolerance, pierces, qty, description in test_cases:
        time_min = calc_waterjet_minutes(length, thickness, tolerance, pierces, qty)
        print(f"{description:<45s} | {length:6.1f} | {thickness:5.2f} | {tolerance:6.3f} | {pierces:6d} | {qty:4d} | {time_min:10.2f}")

    print()


def test_waterjet_formula_accuracy():
    """Test specific formula components."""
    print("=" * 70)
    print("Testing Waterjet Formula Components")
    print("=" * 70)

    # Test minimum time enforcement
    print("\n1. Minimum Time Enforcement (MIN_WJ_MIN_PER_LOT = 10.0 min):")
    time_min = calc_waterjet_minutes(total_length_in=1.0, thickness_in=0.5,
                                     tol_plusminus=0.005, pierce_count=1, qty=1)
    print(f"   Small job (1\" cut, 1 pierce, 1 qty): {time_min:.2f} min")
    print(f"   Should be at least 10.0 min: {'✓ PASS' if time_min >= 10.0 else '✗ FAIL'}")

    # Test thickness factor (use larger jobs to exceed minimum floor)
    print("\n2. Thickness Factor (thicker = slower):")
    time_thin = calc_waterjet_minutes(50.0, 0.25, 0.005, 10, 10)
    time_ref = calc_waterjet_minutes(50.0, 0.50, 0.005, 10, 10)
    time_thick = calc_waterjet_minutes(50.0, 1.00, 0.005, 10, 10)
    print(f"   0.25\" thick: {time_thin:.2f} min (50\" × 10 parts)")
    print(f"   0.50\" thick: {time_ref:.2f} min (reference)")
    print(f"   1.00\" thick: {time_thick:.2f} min")
    print(f"   Thicker takes more time: {'✓ PASS' if time_thin < time_ref < time_thick else '✗ FAIL'}")

    # Test tolerance factor (use larger jobs to exceed minimum floor)
    print("\n3. Tolerance Factor (tighter = slower):")
    time_loose = calc_waterjet_minutes(50.0, 0.5, 0.010, 10, 10)
    time_std = calc_waterjet_minutes(50.0, 0.5, 0.005, 10, 10)
    time_tight = calc_waterjet_minutes(50.0, 0.5, 0.003, 10, 10)
    print(f"   ±0.010 (loose):    {time_loose:.2f} min (50\" × 10 parts)")
    print(f"   ±0.005 (standard): {time_std:.2f} min")
    print(f"   ±0.003 (tight):    {time_tight:.2f} min")
    print(f"   Tighter takes more time: {'✓ PASS' if time_loose < time_std < time_tight else '✗ FAIL'}")

    # Test setup time is included (use larger quantity)
    print("\n4. Setup Time (WJ_SETUP_MIN = 8.0 min):")
    time_1_part = calc_waterjet_minutes(10.0, 0.5, 0.005, 4, 10)
    time_2_parts = calc_waterjet_minutes(10.0, 0.5, 0.005, 4, 20)
    time_diff = time_2_parts - time_1_part
    per_part_time = time_diff / 10
    print(f"   10 parts: {time_1_part:.2f} min")
    print(f"   20 parts: {time_2_parts:.2f} min")
    print(f"   Difference: {time_diff:.2f} min for 10 additional parts")
    print(f"   Per-part time: {per_part_time:.3f} min/part")
    print(f"   Setup overhead included: {'✓ PASS' if time_diff < time_1_part else '✗ FAIL'}")

    # Test pierce time (use larger jobs to see the difference)
    print("\n5. Pierce Time (WJ_PIERCE_MIN = 0.15 min per pierce):")
    time_1_pierce = calc_waterjet_minutes(50.0, 0.5, 0.005, 1, 10)
    time_10_pierces = calc_waterjet_minutes(50.0, 0.5, 0.005, 10, 10)
    pierce_diff = time_10_pierces - time_1_pierce
    expected_diff = 9 * 0.15 * 10  # 9 additional pierces per part × 10 parts
    print(f"   1 pierce per part (10 parts):  {time_1_pierce:.2f} min")
    print(f"   10 pierces per part (10 parts): {time_10_pierces:.2f} min")
    print(f"   Difference: {pierce_diff:.2f} min (expected ~{expected_diff:.2f} min)")
    print(f"   Pierce time accurate: {'✓ PASS' if abs(pierce_diff - expected_diff) < 0.5 else '✗ FAIL'}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("WATERJET OPERATION TEST SUITE")
    print("=" * 70 + "\n")

    test_waterjet_openings_detection()
    test_waterjet_profile_detection()
    test_waterjet_time_calculation()
    test_waterjet_formula_accuracy()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
