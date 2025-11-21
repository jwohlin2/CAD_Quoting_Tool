#!/usr/bin/env python3
"""Test the tolerance regex fix for waterjet detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cad_quoter.geometry.dxf_enrich import detect_waterjet_openings, detect_waterjet_profile

print("=" * 80)
print("TOLERANCE REGEX FIX TEST")
print("=" * 80)

# Test cases for waterjet openings
test_cases_openings = [
    ("WATERJET ALL OPENINGS ±.005", 0.005),
    ("WATERJET ALL OPENINGS ±0.005", 0.005),
    ("WATERJET OPENINGS ±.003", 0.003),
    ("WATERJET ALL OPENINGS +/-.010", 0.010),
    ("WATERJET ALL OPENINGS ± .005", 0.005),  # with space
    ("WATERJET ALL OPENINGS", 0.005),  # no tolerance - use default
]

print("\nTest cases for WATERJET OPENINGS:")
print("-" * 80)
all_passed = True

for text, expected_tol in test_cases_openings:
    has_wj, tolerance = detect_waterjet_openings(text)
    status = "✓" if (has_wj and abs(tolerance - expected_tol) < 0.0001) else "✗"
    if status == "✗":
        all_passed = False
    print(f"{status} '{text}'")
    print(f"   Expected: {expected_tol:.3f}\"  Got: {tolerance:.3f}\"")

# Test cases for waterjet profile
test_cases_profile = [
    ("WATERJET TO ±.003", 0.003),
    ("WATERJET TO ±0.003", 0.003),
    ("WATERJET TO ±.005", 0.005),
    ("WATERJET PROFILE ±.002", 0.002),
    ("WATERJET CUT", 0.003),  # no tolerance - use default
]

print("\n" + "=" * 80)
print("Test cases for WATERJET PROFILE:")
print("-" * 80)

for text, expected_tol in test_cases_profile:
    has_wj, tolerance = detect_waterjet_profile(text)
    status = "✓" if (has_wj and abs(tolerance - expected_tol) < 0.0001) else "✗"
    if status == "✗":
        all_passed = False
    print(f"{status} '{text}'")
    print(f"   Expected: {expected_tol:.3f}\"  Got: {tolerance:.3f}\"")

# Test the actual bug case
print("\n" + "=" * 80)
print("ORIGINAL BUG TEST:")
print("-" * 80)
actual_text = "WATERJET ALL OPENINGS ±.005 (EXCEPT AS NOTED) SEE SHEET 2 OF 2 FOR HOLE CHART"
has_wj, tolerance = detect_waterjet_openings(actual_text)
print(f"Text: '{actual_text}'")
print(f"Detected: has_waterjet={has_wj}, tolerance={tolerance:.3f}\"")
if has_wj and abs(tolerance - 0.005) < 0.0001:
    print("✓ BUG FIXED! Correctly extracts ±.005\"")
else:
    print(f"✗ BUG STILL EXISTS! Expected 0.005\", got {tolerance:.3f}\"")
    all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED!")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED!")
    sys.exit(1)
