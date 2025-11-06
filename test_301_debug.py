"""Debug why 301.dwg isn't finding the McMaster part."""

from cad_quoter.pricing.DirectCostHelper import (
    extract_part_info_from_cad,
    get_mcmaster_part_number,
    get_mcmaster_part_and_price
)

cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dwg"

print("=" * 70)
print("DEBUG: Why isn't 301.dwg finding McMaster part 86825K997?")
print("=" * 70)

# Extract part info
print("\nStep 1: Extract part info")
part_info = extract_part_info_from_cad(cad_file, verbose=False)
print(f"  Dimensions: {part_info.length}\" x {part_info.width}\" x {part_info.thickness}\"")
print(f"  Material: {part_info.material}")

# Try direct lookup
print("\nStep 2: Direct McMaster lookup")
print(f"  Looking for: L={part_info.length}, W={part_info.width}, T={part_info.thickness}")
print(f"  Material: {part_info.material}")

part_number = get_mcmaster_part_number(
    length=part_info.length,
    width=part_info.width,
    thickness=part_info.thickness,
    material=part_info.material
)
print(f"  Found part: {part_number}")

# Expected part from catalog: aluminum MIC6, 3" thick, 18x18, part 86825K997
print("\nStep 3: Verify catalog has the part")
print("  Expected: 86825K997 (18\"x18\"x3\" aluminum MIC6)")
print("  This should cover: 15.5\"x15.125\"x3\"")

# Try different material name variations
print("\nStep 4: Try material name variations")
materials = ["aluminum MIC6", "MIC6", "aluminum mic6", "Aluminum MIC6"]
for mat in materials:
    part = get_mcmaster_part_number(
        length=part_info.length,
        width=part_info.width,
        thickness=part_info.thickness,
        material=mat
    )
    print(f"  '{mat}': {part}")

# Get full result
print("\nStep 5: Full result with price")
result = get_mcmaster_part_and_price(
    length=part_info.length,
    width=part_info.width,
    thickness=part_info.thickness,
    material=part_info.material
)
print(f"  Part Number: {result['part_number']}")
print(f"  Price: ${result['price']:.2f}" if result['price'] else "  Price: N/A")
print(f"  Success: {result['success']}")

if result['part_number'] == '86825K997':
    print("\n  SUCCESS! Found the correct part!")
else:
    print(f"\n  ISSUE: Expected 86825K997, got {result['part_number']}")

print("\n" + "=" * 70)
