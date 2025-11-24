#!/usr/bin/env python3
"""Test to verify material factor is applied correctly for larger parts."""

import sys
from pathlib import Path

# Add cad_quoter to path
cad_quoter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(cad_quoter_dir))

from cad_quoter.planning.process_planner import calc_edge_break_minutes


def test_material_factor_application():
    """Test that material factor affects edge break time for larger parts."""
    print("=" * 70)
    print("Testing Material Factor Application for Larger Parts")
    print("=" * 70)

    # Larger part: 20" x 10" (perimeter = 60")
    perimeter = 60.0
    qty = 1

    materials = [
        ("ALUMINUM", 0.6),
        ("TOOL_STEEL", 1.0),
        ("52100", 1.4),
        ("STAINLESS", 1.3),
        ("CARBIDE", 2.5),
        ("CERAMIC", 3.5),
    ]

    print(f"\nLarge part - Perimeter: {perimeter}\"")
    print(f"Quantity: {qty}")
    print(f"Formula: 3.0 + (2 × {perimeter} × {qty} × 0.02 × material_factor)")
    print(f"Floor: max(result, 5.0)")
    print()

    results = []
    for material, expected_factor in materials:
        time_min = calc_edge_break_minutes(perimeter, qty, material)
        edge_length = 2.0 * perimeter * qty

        # Calculate expected time
        expected_time = 3.0 + edge_length * 0.02 * expected_factor
        expected_time = max(expected_time, 5.0)

        results.append((material, expected_factor, time_min, expected_time))

        diff = abs(time_min - expected_time)
        status = "✓" if diff < 0.01 else "✗"

        print(f"{status} {material:12s} (factor={expected_factor}x): {time_min:6.2f} min (expected: {expected_time:6.2f} min)")

    print()
    print("=" * 70)
    print("Material Factor Effect Comparison:")
    print("=" * 70)

    aluminum_time = results[0][2]

    for material, factor, time_min, _ in results:
        ratio = time_min / aluminum_time if aluminum_time > 0 else 0
        print(f"{material:12s}: {time_min:6.2f} min ({ratio:.2f}x vs Aluminum)")

    print()
    print("Expected ratios based on material factors:")
    print(f"  Carbide/Aluminum: {2.5/0.6:.2f}x")
    print(f"  Ceramic/Aluminum: {3.5/0.6:.2f}x")
    print(f"  Carbide/Tool Steel: {2.5/1.0:.2f}x")
    print()


if __name__ == "__main__":
    try:
        test_material_factor_application()
        print("=" * 70)
        print("Material factor test completed!")
        print("=" * 70)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
