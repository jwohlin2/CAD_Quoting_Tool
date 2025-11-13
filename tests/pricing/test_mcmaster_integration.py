"""Test McMaster integration with DirectCostHelper."""

from cad_quoter.pricing.DirectCostHelper import (
    extract_part_info_from_cad,
    get_mcmaster_part_number,
    get_mcmaster_price,
    get_mcmaster_part_and_price,
    get_material_cost_from_mcmaster
)

# Test with the specified CAD file
cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dwg"

print("=" * 60)
print("Testing McMaster Integration with DirectCostHelper")
print("=" * 60)

# Step 1: Extract part info from CAD file
print(f"\n1. Extracting part info from: {cad_file}")
try:
    part_info = extract_part_info_from_cad(
        cad_file,
        auto_detect_material=True,
        verbose=True
    )
    print(f"\n   Dimensions: {part_info.length:.3f}\" x {part_info.width:.3f}\" x {part_info.thickness:.3f}\"")
    print(f"   Material: {part_info.material}")
    print(f"   Volume: {part_info.volume:.3f} cubic inches")
    print(f"   Area: {part_info.area:.3f} square inches")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Get McMaster part number from catalog
print(f"\n2. Looking up McMaster part number...")
try:
    part_number = get_mcmaster_part_number(
        length=part_info.length,
        width=part_info.width,
        thickness=part_info.thickness,
        material=part_info.material
    )
    if part_number:
        print(f"   Found McMaster part: {part_number}")
    else:
        print(f"   No McMaster part found for this size/material combination")
        print(f"   (Material: {part_info.material}, Size: {part_info.length}x{part_info.width}x{part_info.thickness})")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Get price from McMaster API (if part number found)
if part_number:
    print(f"\n3. Fetching price from McMaster API for part {part_number}...")
    try:
        price = get_mcmaster_price(part_number, quantity=1)
        if price:
            print(f"   Unit price: ${price:.2f}")
        else:
            print(f"   Could not fetch price (API may have failed)")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n3. Skipping price lookup (no part number found)")

# Step 4: Combined function test
print(f"\n4. Testing combined get_mcmaster_part_and_price()...")
try:
    result = get_mcmaster_part_and_price(
        length=part_info.length,
        width=part_info.width,
        thickness=part_info.thickness,
        material=part_info.material,
        quantity=1
    )
    print(f"   Part Number: {result['part_number']}")
    print(f"   Price: ${result['price']:.2f}" if result['price'] else "   Price: N/A")
    print(f"   Success: {result['success']}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Get total material cost
print(f"\n5. Testing get_material_cost_from_mcmaster()...")
try:
    cost = get_material_cost_from_mcmaster(part_info, quantity=1)
    if cost:
        print(f"   Total material cost: ${cost:.2f}")
    else:
        print(f"   Could not calculate material cost")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
