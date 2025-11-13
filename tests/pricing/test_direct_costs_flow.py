"""Test the complete direct costs calculation flow."""
from pathlib import Path
import json

print("=" * 80)
print("TESTING DIRECT COSTS CALCULATION FLOW")
print("=" * 80)

# Step 1: Load cached dimensions
cad_file_path = r"Cad Files\301_redacted.dwg"
json_output = Path(__file__).parent / "debug" / f"{Path(cad_file_path).stem}_dims.json"

print(f"\n1. Loading dimensions from cached JSON...")
print(f"   File: {json_output}")

with open(json_output, 'r') as f:
    dims_data = json.load(f)

length = dims_data.get('length', 0.0)
width = dims_data.get('width', 0.0)
thickness = dims_data.get('thickness', 0.0)

print(f"   [OK] Length:    {length} in")
print(f"   [OK] Width:     {width} in")
print(f"   [OK] Thickness: {thickness} in")

# Step 2: Calculate desired stock dimensions
print(f"\n2. Calculating desired stock dimensions...")
desired_length = length + 0.25
desired_width = width + 0.25
desired_thickness = thickness + 0.125

print(f"   Before rounding: {desired_length:.3f} x {desired_width:.3f} x {desired_thickness:.3f}")

# Step 3: Round thickness to standard catalog thickness
standard_thicknesses = [0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 6.0]
desired_thickness = min(standard_thicknesses, key=lambda t: abs(t - desired_thickness) if t >= desired_thickness else float('inf'))

print(f"   After rounding:  {desired_length:.3f} x {desired_width:.3f} x {desired_thickness:.3f}")

# Step 4: Try McMaster lookup
print(f"\n3. Looking up McMaster stock...")

try:
    from cad_quoter.pricing.mcmaster_helpers import pick_mcmaster_plate_sku, load_mcmaster_catalog_rows
    from cad_quoter.resources import default_catalog_csv

    catalog_csv_path = str(default_catalog_csv())
    catalog_rows = load_mcmaster_catalog_rows(catalog_csv_path)
    print(f"   Loaded {len(catalog_rows)} rows from catalog")

    # Try multiple material keys
    material_keys_to_try = ["aluminum MIC6", "MIC6", "Aluminum 6061", "6061"]

    mcmaster_result = None
    for material_key in material_keys_to_try:
        print(f"   Trying material key: '{material_key}'...")
        mcmaster_result = pick_mcmaster_plate_sku(
            need_L_in=desired_length,
            need_W_in=desired_width,
            need_T_in=desired_thickness,
            material_key=material_key,
            catalog_rows=catalog_rows
        )
        if mcmaster_result:
            print(f"   [OK] Found match with '{material_key}'!")
            break

    if mcmaster_result:
        mcmaster_L = mcmaster_result.get('len_in', 0)
        mcmaster_W = mcmaster_result.get('wid_in', 0)
        mcmaster_T = mcmaster_result.get('thk_in', 0)
        mcmaster_part = mcmaster_result.get('mcmaster_part', 'N/A')

        print(f"\n   McMaster Stock Found:")
        print(f"   Part Number: {mcmaster_part}")
        print(f"   Dimensions:  {mcmaster_L} x {mcmaster_W} x {mcmaster_T} in")

        # Step 5: Calculate scrap
        print(f"\n4. Calculating scrap...")
        mcmaster_volume = mcmaster_L * mcmaster_W * mcmaster_T
        part_volume = length * width * thickness
        scrap_volume = mcmaster_volume - part_volume
        scrap_percentage = (scrap_volume / mcmaster_volume * 100) if mcmaster_volume > 0 else 0

        print(f"   McMaster volume: {mcmaster_volume:.2f} in³")
        print(f"   Part volume:     {part_volume:.2f} in³")
        print(f"   Scrap volume:    {scrap_volume:.2f} in³")
        print(f"   Scrap percent:   {scrap_percentage:.1f}%")

        # Step 6: Get material density
        print(f"\n5. Calculating weights...")
        from cad_quoter.pricing.DirectCostHelper import get_material_density

        density = get_material_density("aluminum MIC6")
        print(f"   Density: {density:.4f} lb/in³")

        mcmaster_weight = mcmaster_volume * density
        part_weight = part_volume * density
        scrap_weight = scrap_volume * density

        print(f"   McMaster weight: {mcmaster_weight:.2f} lb")
        print(f"   Part weight:     {part_weight:.2f} lb")
        print(f"   Scrap weight:    {scrap_weight:.2f} lb")

        print(f"\n{'=' * 80}")
        print(f"[SUCCESS] Complete direct costs flow test passed!")
        print(f"{'=' * 80}")
    else:
        print(f"\n[ERROR] Could not find McMaster stock for any material key")
        print(f"Tried: {', '.join(material_keys_to_try)}")

except Exception as e:
    import traceback
    print(f"\n[ERROR] Test failed: {e}")
    print(traceback.format_exc())
