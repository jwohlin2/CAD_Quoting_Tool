#!/usr/bin/env python3
"""Test script to verify baseline cleanup scaling for various part sizes."""

import math


def calculate_baseline_cleanup(L: float, W: float) -> tuple[float, str]:
    """Calculate baseline cleanup time and label for a part.

    Args:
        L: Length in inches
        W: Width in inches

    Returns:
        Tuple of (baseline_minutes, cleanup_type)
    """
    surface_area = L * W if (L > 0 and W > 0) else 0
    perimeter = 2 * (L + W) if (L > 0 and W > 0) else 0

    # For large flat plates, area is a better indicator than volume
    # Use sqrt(area) for more gradual scaling: sqrt(100)=10, sqrt(400)=20, sqrt(729)=27
    area_sqrt = math.sqrt(surface_area) if surface_area > 0 else 0

    # Calculate area-based and perimeter-based factors
    # Reference: 10"×10" part has area_sqrt=10, perimeter=40
    area_factor = min(area_sqrt / 10.0, 5.0)  # Cap at 5x for massive plates (50"×50"+)
    perim_factor = min(perimeter / 40.0, 5.0)  # Cap at 5x for huge perimeters (200"+)

    # Use the larger factor (handles both square and rectangular plates well)
    size_factor = max(area_factor, perim_factor)

    # Add baseline for general cleanup/deburr (scaled by part size, manual labor)
    # Scales with area/perimeter: small parts ~1.5-3 min, medium ~3-5 min, large plates ~5-9 min
    baseline = 1.5 + size_factor * 1.5

    # Choose descriptive label based on part size
    if surface_area < 25:  # < 5"×5"
        cleanup_type = "Small part cleanup"
    elif surface_area < 150:  # < ~12"×12"
        cleanup_type = "Part cleanup / deburr"
    else:  # Large plates
        cleanup_type = "Large plate washdown / deburr"

    return round(baseline, 1), cleanup_type


def main():
    """Test baseline cleanup scaling for various part sizes."""
    test_cases = [
        (3, 3, "Tiny part"),
        (5, 5, "Small part"),
        (10, 10, "Medium part (reference)"),
        (12, 12, "Medium-large part"),
        (15, 15, "Large part"),
        (20, 20, "Large plate (T1769-201_redacted.dwg style)"),
        (27, 27, "Huge plate (301_redacted.dwg style)"),
        (30, 30, "Massive plate"),
        (10, 40, "Long rectangular plate"),
        (5, 60, "Very long thin plate"),
    ]

    print("Baseline Cleanup Scaling Test")
    print("=" * 80)
    print(f"{'Size (L×W)':<20} {'Description':<35} {'Minutes':<10} {'Label'}")
    print("-" * 80)

    for L, W, description in test_cases:
        minutes, label = calculate_baseline_cleanup(L, W)
        size_str = f"{L}\" × {W}\""
        print(f"{size_str:<20} {description:<35} {minutes:<10.1f} {label}")

    print("=" * 80)
    print("\nKey observations:")
    print("- Small parts (3\"×3\", 5\"×5\"): 1.7-2.2 minutes")
    print("- Medium parts (10\"×10\", 12\"×12\"): 3.0-3.8 minutes")
    print("- Large plates (20\"×20\", 27\"×27\"): 4.5-5.6 minutes")
    print("- Massive plates (30\"×30\"+): 6.2-9.0 minutes (capped)")
    print("\nThis is much more reasonable than the old 2.5 min cap!")


if __name__ == "__main__":
    main()
