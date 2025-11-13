"""Test McMaster integration with specific material override."""

from cad_quoter.pricing.DirectCostHelper import (
    extract_part_info_from_cad,
    get_mcmaster_part_and_price,
)

# Test with the specified CAD file
cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dwg"

print("=" * 60)
print("Testing McMaster Integration with Material Override")
print("=" * 60)

# Extract part info from CAD file
print(f"\nExtracting part info from: {cad_file}")
part_info = extract_part_info_from_cad(
    cad_file,
    auto_detect_material=True,
    verbose=False
)

print(f"Detected dimensions: {part_info.length:.3f}\" x {part_info.width:.3f}\" x {part_info.thickness:.3f}\"")
print(f"Detected material: {part_info.material}")

# Test with different materials that exist in catalog
test_materials = [
    "aluminum 5083",
    "303 Stainless Steel",
    "aluminum MIC6"
]

# Also test with smaller dimensions that are more likely to be in catalog
test_dimensions = [
    (6.0, 6.0, 0.25, "6x6x0.25 plate"),
    (12.0, 12.0, 0.5, "12x12x0.5 plate"),
    (part_info.length, part_info.width, part_info.thickness, "Actual CAD dimensions")
]

print("\n" + "=" * 60)
print("Testing various material/size combinations:")
print("=" * 60)

for material in test_materials:
    print(f"\n--- Material: {material} ---")

    for length, width, thickness, desc in test_dimensions:
        print(f"\n  {desc} ({length}\" x {width}\" x {thickness}\"):")

        result = get_mcmaster_part_and_price(
            length=length,
            width=width,
            thickness=thickness,
            material=material,
            quantity=1
        )

        if result["success"]:
            print(f"    [SUCCESS] Part: {result['part_number']}")
            print(f"    [SUCCESS] Price: ${result['price']:.2f}")
        else:
            print(f"    [NO MATCH] No match found")
            if result['part_number']:
                print(f"              (Found part {result['part_number']}, but no price)")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
