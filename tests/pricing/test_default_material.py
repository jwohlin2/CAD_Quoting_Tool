"""Test the new DEFAULT_MATERIAL setting."""

from cad_quoter.pricing.DirectCostHelper import (
    DEFAULT_MATERIAL,
    extract_part_info_from_cad,
    get_mcmaster_part_and_price
)

print("=" * 70)
print("TESTING DEFAULT MATERIAL SETTING")
print("=" * 70)
print(f"\nDEFAULT_MATERIAL is set to: '{DEFAULT_MATERIAL}'")
print("=" * 70)

# Test 1: Extract from CAD file (will auto-detect, then fall back to default)
print("\n[TEST 1] CAD File with auto-detect (should fall back to default)")
print("-" * 70)
cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dwg"
part_info = extract_part_info_from_cad(cad_file, verbose=True)

print(f"\nExtracted Part Info:")
print(f"  Dimensions: {part_info.length:.2f}\" x {part_info.width:.2f}\" x {part_info.thickness:.2f}\"")
print(f"  Material:   {part_info.material}")
print(f"  (Should be '{DEFAULT_MATERIAL}' since CAD has no specific material)")

# Test 2: Try to find McMaster part with default material
print(f"\n[TEST 2] McMaster lookup with smaller dimensions using default material")
print("-" * 70)
print(f"Looking up: 6x6x0.25 in '{DEFAULT_MATERIAL}'")
result = get_mcmaster_part_and_price(
    length=6.0,
    width=6.0,
    thickness=0.25,
    material=DEFAULT_MATERIAL
)

print(f"\nResult:")
print(f"  Part Number: {result['part_number']}")
print(f"  Unit Price:  ${result['price']:.2f}" if result['price'] else "  Unit Price:  N/A")
print(f"  Success:     {result['success']}")

if result['success']:
    print(f"\n  SUCCESS! Default material '{DEFAULT_MATERIAL}' works with McMaster catalog!")
else:
    print(f"\n  WARNING: Default material '{DEFAULT_MATERIAL}' may not be in catalog")

# Test 3: Try the actual 301.dwg dimensions
print(f"\n[TEST 3] McMaster lookup with actual 301.dwg dimensions")
print("-" * 70)
print(f"Looking up: {part_info.length:.2f}x{part_info.width:.2f}x{part_info.thickness:.2f} in '{part_info.material}'")
result = get_mcmaster_part_and_price(
    length=part_info.length,
    width=part_info.width,
    thickness=part_info.thickness,
    material=part_info.material
)

print(f"\nResult:")
print(f"  Part Number: {result['part_number']}")
print(f"  Unit Price:  ${result['price']:.2f}" if result['price'] else "  Unit Price:  N/A")
print(f"  Success:     {result['success']}")

if result['success']:
    print(f"\n  SUCCESS! Found part {result['part_number']} for ${result['price']:.2f}")
    print(f"  This is an 18\"x18\"x3\" plate that covers the 15.5\"x15.125\"x3\" part")
else:
    print(f"\n  No McMaster stock found for these dimensions")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
