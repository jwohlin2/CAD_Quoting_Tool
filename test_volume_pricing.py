#!/usr/bin/env python3
"""
Test script for volume-based pricing estimation.
This tests the new fallback logic for large parts without direct McMaster pricing.
"""

from cad_quoter.pricing.DirectCostHelper import estimate_price_from_reference_part

# Test with a large aluminum MIC6 plate (like the user's 35.5" x 35.5" x 6.25")
print("Testing volume-based price estimation for large parts")
print("=" * 70)

# Test case: Large aluminum MIC6 plate
target_length = 35.25
target_width = 35.25
target_thickness = 6.125
material = "aluminum MIC6"

print(f"\nTarget part:")
print(f"  Material: {material}")
print(f"  Dimensions: {target_length} × {target_width} × {target_thickness} in")
print(f"  Volume: {target_length * target_width * target_thickness:.2f} in³")

print(f"\nAttempting volume-based price estimation...")
print("-" * 70)

try:
    estimated_price = estimate_price_from_reference_part(
        target_length=target_length,
        target_width=target_width,
        target_thickness=target_thickness,
        material=material,
        verbose=True
    )

    if estimated_price:
        print("\n" + "=" * 70)
        print(f"SUCCESS: Estimated price: ${estimated_price:,.2f}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("FAILED: Could not estimate price")
        print("=" * 70)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
