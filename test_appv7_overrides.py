"""
Test script for AppV7 override functionality.

Tests all 8 override parameters:
1. Length, Width, Thickness (dimensions)
2. Material
3. Machine Rate
4. Labor Rate
5. Margin %
6. McMaster Price Override
7. Scrap Value Override
"""

from pathlib import Path
from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad

# Test CAD file
cad_file = Path("D:/CAD_Quoting_Tool/Cad Files/301_redacted.dwg")

if not cad_file.exists():
    print(f"ERROR: CAD file not found: {cad_file}")
    print("Please update the path to an existing CAD file.")
    exit(1)

print("=" * 80)
print("TEST 1: Baseline Quote (No Overrides)")
print("=" * 80)

baseline = extract_quote_data_from_cad(
    cad_file_path=cad_file,
    verbose=True
)

print(f"\n[OK] Baseline Quote Generated")
print(f"  Material: {baseline.material_info.material_name}")
print(f"  Dimensions: {baseline.part_dimensions.length} x {baseline.part_dimensions.width} x {baseline.part_dimensions.thickness}")
print(f"  McMaster Price: ${baseline.stock_info.mcmaster_price:.2f}" if baseline.stock_info.mcmaster_price else "  McMaster Price: N/A")
print(f"  Scrap Value: ${baseline.scrap_info.scrap_value:.2f}")
print(f"  Machine Cost: ${baseline.machine_hours.machine_cost:.2f}")
print(f"  Labor Cost: ${baseline.labor_hours.labor_cost:.2f}")
print(f"  Total Cost: ${baseline.cost_summary.total_cost:.2f}")
print(f"  Margin: {baseline.cost_summary.margin_rate:.0%}")
print(f"  Final Price: ${baseline.cost_summary.final_price:.2f}")

print("\n" + "=" * 80)
print("TEST 2: Dimension Override")
print("=" * 80)

dim_override = extract_quote_data_from_cad(
    cad_file_path=cad_file,
    dimension_override=(10.0, 8.0, 0.5),
    verbose=True
)

print(f"\n[OK] Dimension Override Applied")
print(f"  Dimensions: {dim_override.part_dimensions.length} x {dim_override.part_dimensions.width} x {dim_override.part_dimensions.thickness}")
assert dim_override.part_dimensions.length == 10.0
assert dim_override.part_dimensions.width == 8.0
assert dim_override.part_dimensions.thickness == 0.5
print(f"  [OK] Dimensions match override values")

print("\n" + "=" * 80)
print("TEST 3: Material Override")
print("=" * 80)

material_override = extract_quote_data_from_cad(
    cad_file_path=cad_file,
    material_override="17-4 PH Stainless Steel",
    dimension_override=(10.0, 8.0, 0.5),
    verbose=True
)

print(f"\n[OK] Material Override Applied")
print(f"  Material: {material_override.material_info.material_name}")
assert material_override.material_info.material_name == "17-4 PH Stainless Steel"
print(f"  [OK] Material matches override value")

print("\n" + "=" * 80)
print("TEST 4: Rate Overrides (Machine, Labor, Margin)")
print("=" * 80)

rate_override = extract_quote_data_from_cad(
    cad_file_path=cad_file,
    dimension_override=(10.0, 8.0, 0.5),
    machine_rate=120.0,
    labor_rate=120.0,
    margin_rate=0.20,  # 20%
    verbose=True
)

print(f"\n[OK] Rate Overrides Applied")
print(f"  Machine Rate: $120/hr")
print(f"  Labor Rate: $120/hr")
print(f"  Margin Rate: 20%")
print(f"  Machine Cost: ${rate_override.machine_hours.machine_cost:.2f}")
print(f"  Labor Cost: ${rate_override.labor_hours.labor_cost:.2f}")
print(f"  Margin Amount: ${rate_override.cost_summary.margin_amount:.2f}")
print(f"  Final Price: ${rate_override.cost_summary.final_price:.2f}")

# Verify rates were applied
expected_machine_cost = rate_override.machine_hours.total_hours * 120.0
assert abs(rate_override.machine_hours.machine_cost - expected_machine_cost) < 0.01
print(f"  [OK] Machine rate correctly applied")

expected_labor_cost = rate_override.labor_hours.total_hours * 120.0
assert abs(rate_override.labor_hours.labor_cost - expected_labor_cost) < 0.01
print(f"  [OK] Labor rate correctly applied")

assert rate_override.cost_summary.margin_rate == 0.20
print(f"  [OK] Margin rate correctly applied")

print("\n" + "=" * 80)
print("TEST 5: McMaster Price Override")
print("=" * 80)

price_override = extract_quote_data_from_cad(
    cad_file_path=cad_file,
    dimension_override=(10.0, 8.0, 0.5),
    mcmaster_price_override=100.00,
    verbose=True
)

print(f"\n[OK] McMaster Price Override Applied")
print(f"  McMaster Price: ${price_override.stock_info.mcmaster_price:.2f}")
assert price_override.stock_info.mcmaster_price == 100.00
print(f"  [OK] Price matches override value")
print(f"  Direct Cost: ${price_override.direct_cost_breakdown.net_material_cost:.2f}")

print("\n" + "=" * 80)
print("TEST 6: Scrap Value Override")
print("=" * 80)

scrap_override = extract_quote_data_from_cad(
    cad_file_path=cad_file,
    dimension_override=(10.0, 8.0, 0.5),
    scrap_value_override=5.00,
    verbose=True
)

print(f"\n[OK] Scrap Value Override Applied")
print(f"  Scrap Value: ${scrap_override.scrap_info.scrap_value:.2f}")
assert scrap_override.scrap_info.scrap_value == 5.00
print(f"  [OK] Scrap value matches override value")
print(f"  Scrap Credit: ${scrap_override.direct_cost_breakdown.scrap_credit:.2f}")

print("\n" + "=" * 80)
print("TEST 7: Multiple Overrides Combined")
print("=" * 80)

combined = extract_quote_data_from_cad(
    cad_file_path=cad_file,
    dimension_override=(12.0, 10.0, 0.75),
    material_override="Aluminum 6061",
    machine_rate=150.0,
    labor_rate=150.0,
    margin_rate=0.25,  # 25%
    mcmaster_price_override=200.00,
    scrap_value_override=10.00,
    verbose=True
)

print(f"\n[OK] All Overrides Applied Together")
print(f"  Dimensions: {combined.part_dimensions.length} x {combined.part_dimensions.width} x {combined.part_dimensions.thickness}")
print(f"  Material: {combined.material_info.material_name}")
print(f"  McMaster Price: ${combined.stock_info.mcmaster_price:.2f}")
print(f"  Scrap Value: ${combined.scrap_info.scrap_value:.2f}")
print(f"  Machine Cost: ${combined.machine_hours.machine_cost:.2f}")
print(f"  Labor Cost: ${combined.labor_hours.labor_cost:.2f}")
print(f"  Margin: {combined.cost_summary.margin_rate:.0%}")
print(f"  Final Price: ${combined.cost_summary.final_price:.2f}")

# Verify all overrides
assert combined.part_dimensions.length == 12.0
assert combined.part_dimensions.width == 10.0
assert combined.part_dimensions.thickness == 0.75
assert combined.material_info.material_name == "Aluminum 6061"
assert combined.stock_info.mcmaster_price == 200.00
assert combined.scrap_info.scrap_value == 10.00
assert combined.cost_summary.margin_rate == 0.25

print(f"\n  [OK] All overrides verified!")

print("\n" + "=" * 80)
print("ALL TESTS PASSED! [OK]")
print("=" * 80)
print("\nOverride functionality is working correctly.")
print("You can now use AppV7 with all 8 override parameters.")
