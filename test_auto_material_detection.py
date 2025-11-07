"""Test auto-material detection in DirectCostHelper."""

from pathlib import Path
from cad_quoter.pricing.DirectCostHelper import extract_part_info_from_cad

cad_file = Path("Cad Files/301_redacted.dxf")

print("=" * 70)
print("AUTO-MATERIAL DETECTION TEST")
print("=" * 70)

# Test 1: Auto-detect material (will default to GENERIC if not found)
print("\n1. Extract part info with auto-detected material:")
print("-" * 70)
part_info = extract_part_info_from_cad(cad_file, verbose=True)

print(f"\n   Material: {part_info.material}")
print(f"   Dimensions: {part_info.length}\" x {part_info.width}\" x {part_info.thickness}\"")
print(f"   Volume: {part_info.volume:.2f} cubic inches")
print(f"   Area: {part_info.area:.2f} square inches")

# Test 2: Override with specific material
print("\n\n2. Extract part info with manual material override:")
print("-" * 70)
part_info_override = extract_part_info_from_cad(
    cad_file,
    material="17-4 PH Stainless Steel",
    verbose=True
)

print(f"\n   Material: {part_info_override.material}")
print(f"   Volume: {part_info_override.volume:.2f} cubic inches")

# Test 3: Disable auto-detection
print("\n\n3. Extract with auto-detection disabled (material=None):")
print("-" * 70)
part_info_no_auto = extract_part_info_from_cad(
    cad_file,
    material=None,
    auto_detect_material=False,
    verbose=True
)

print(f"\n   Material: {part_info_no_auto.material}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Auto-material detection:
- Searches CAD text for material keywords from material_map.csv
- Returns canonical material name if found
- Defaults to "GENERIC" if no material found
- Can be overridden by specifying material parameter
- Can be disabled with auto_detect_material=False
""")
