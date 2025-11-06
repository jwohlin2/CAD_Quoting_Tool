"""Test DirectCostHelper functionality."""

from pathlib import Path
from cad_quoter.pricing.DirectCostHelper import (
    extract_part_info_from_cad,
    extract_part_info_from_plan,
    extract_dimensions_with_paddle_ocr,
    get_part_dimensions,
    calculate_material_weight,
    get_material_density,
)
from cad_quoter.planning import plan_from_cad_file

cad_file = Path("Cad Files/301_redacted.dxf")

print("=" * 70)
print("DIRECT COST HELPER TEST")
print("=" * 70)

# Test 0: Extract dimensions using PaddleOCR directly
print("\n0. Extract dimensions with PaddleOCR:")
dims = extract_dimensions_with_paddle_ocr(cad_file)
print(f"   PaddleOCR extracted: L={dims['L']}\", W={dims['W']}\", T={dims['T']}\"")

# Test 1: Extract from CAD file directly (uses PaddleOCR internally)
print("\n1. Extract part info from CAD file (using PaddleOCR):")
material = "17-4 PH Stainless"
part_info = extract_part_info_from_cad(cad_file, material, use_paddle_ocr=True)

print(f"   Material: {part_info.material}")
print(f"   Dimensions: {part_info.length}\" x {part_info.width}\" x {part_info.thickness}\"")
print(f"   Volume: {part_info.volume:.2f} cubic inches")
print(f"   Area (LxW): {part_info.area:.2f} square inches")

# Test 2: Calculate weight
density = get_material_density(material)
weight = calculate_material_weight(part_info.volume, density)
print(f"\n2. Material calculations:")
print(f"   Density: {density:.3f} lb/in³")
print(f"   Weight: {weight:.2f} lbs")

# Test 3: Extract from existing plan
print("\n3. Extract from plan:")
plan = plan_from_cad_file(cad_file, verbose=False)
L, W, T = get_part_dimensions(plan)
print(f"   Dimensions: {L}\" x {W}\" x {T}\"")

part_info2 = extract_part_info_from_plan(plan, "Aluminum 6061-T6")
print(f"   Material: {part_info2.material}")
al_density = get_material_density(part_info2.material)
al_weight = calculate_material_weight(part_info2.volume, al_density)
print(f"   If Aluminum: {al_weight:.2f} lbs (vs Steel: {weight:.2f} lbs)")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
