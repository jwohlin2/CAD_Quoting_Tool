"""Debug test for catalog lookup."""

from cad_quoter.pricing.mcmaster_helpers import (
    load_mcmaster_catalog_rows,
    pick_mcmaster_plate_sku
)

# Load catalog
print("Loading catalog...")
catalog_rows = load_mcmaster_catalog_rows()
print(f"Loaded {len(catalog_rows)} rows from catalog")

# Show first few rows to verify structure
print("\nFirst 3 catalog rows:")
for i, row in enumerate(catalog_rows[:3]):
    print(f"  Row {i}: {row}")

# Test lookup with exact match from catalog
print("\n" + "="*60)
print("Test 1: Exact match (aluminum 5083, 6x6x0.25)")
print("="*60)
result = pick_mcmaster_plate_sku(
    need_L_in=6.0,
    need_W_in=6.0,
    need_T_in=0.25,
    material_key="aluminum 5083",
    catalog_rows=catalog_rows
)
print(f"Result: {result}")

# Test with slightly smaller dimensions (should round up)
print("\n" + "="*60)
print("Test 2: Round up (aluminum 5083, 5.5x5.5x0.25 -> should find 6x6x0.25)")
print("="*60)
result = pick_mcmaster_plate_sku(
    need_L_in=5.5,
    need_W_in=5.5,
    need_T_in=0.25,
    material_key="aluminum 5083",
    catalog_rows=catalog_rows
)
print(f"Result: {result}")

# Test 303 Stainless Steel
print("\n" + "="*60)
print("Test 3: 303 Stainless Steel (6x6x0.5)")
print("="*60)
result = pick_mcmaster_plate_sku(
    need_L_in=6.0,
    need_W_in=6.0,
    need_T_in=0.5,
    material_key="303 Stainless Steel",
    catalog_rows=catalog_rows
)
print(f"Result: {result}")

# Test material name variations
print("\n" + "="*60)
print("Test 4: Material name variations")
print("="*60)
for material in ["aluminum 5083", "aluminum5083", "5083", "MIC6", "aluminum MIC6"]:
    result = pick_mcmaster_plate_sku(
        need_L_in=6.0,
        need_W_in=6.0,
        need_T_in=0.25,
        material_key=material,
        catalog_rows=catalog_rows
    )
    print(f"  Material '{material}': {result['mcmaster_part'] if result else 'NO MATCH'}")
