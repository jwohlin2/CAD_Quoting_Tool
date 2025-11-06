"""Test the machine hours report generation in AppV7."""
from pathlib import Path

# Test the machine hours report generation
cad_file = Path("Cad Files/301_redacted.dxf")
if not cad_file.exists():
    print(f"ERROR: Test file not found: {cad_file}")
    exit(1)

print("=" * 80)
print("TESTING MACHINE HOURS REPORT GENERATION (AppV7 Integration)")
print("=" * 80)

# Simulate what AppV7 does
from cad_quoter.planning import (
    extract_hole_operations_from_cad,
    estimate_hole_table_times,
    plan_from_cad_file,
)

# Extract hole operations
print(f"\n1. Extracting hole operations from: {cad_file.name}")
hole_table = extract_hole_operations_from_cad(cad_file)
print(f"   Found {len(hole_table)} hole operations")

# Get dimensions
print(f"\n2. Extracting dimensions...")
plan = plan_from_cad_file(cad_file, verbose=False)
dims = plan.get('extracted_dims', {})
thickness = dims.get('T', 0)

# Try cached JSON if needed
if thickness == 0:
    json_output = Path(__file__).parent / "debug" / f"{cad_file.stem}_dims.json"
    if json_output.exists():
        print(f"   Loading from cached JSON: {json_output.name}")
        import json
        with open(json_output, 'r') as f:
            dims_data = json.load(f)
        thickness = dims_data.get('thickness', 0.0)
        print(f"   Thickness from cache: {thickness}\"")
    else:
        print(f"   WARNING: No cached dimensions found")
else:
    print(f"   Thickness from plan: {thickness}\"")

# Calculate times
print(f"\n3. Calculating machine times...")
material = "17-4 PH Stainless"
times = estimate_hole_table_times(hole_table, material, thickness)

# Format the report (same as AppV7)
def format_drill_group(group):
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | {group['sfm']:.0f} sfm | "
            f"{group['ipr']:.4f} ipr | t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def format_jig_grind_group(group):
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def format_tap_group(group):
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | {group['tpi']} TPI | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def format_cbore_group(group):
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | {group['sfm']:.0f} sfm | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def format_cdrill_group(group):
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

# Build and print the report
print("\n" + "=" * 80)
print("MACHINE HOURS ESTIMATION - DETAILED HOLE TABLE BREAKDOWN")
print("=" * 80)
print(f"Material: {material}")
print(f"Thickness: {thickness:.3f}\"")
print(f"Hole entries: {len(hole_table)}")
print()

# TIME PER HOLE - DRILL GROUPS
if times['drill_groups']:
    print("TIME PER HOLE - DRILL GROUPS")
    print("-" * 80)
    for group in times['drill_groups']:
        print(format_drill_group(group))
    print(f"\nTotal Drilling Time: {times['total_drill_minutes']:.2f} minutes\n")

# TIME PER HOLE - JIG GRIND
if times['jig_grind_groups']:
    print("TIME PER HOLE - JIG GRIND")
    print("-" * 80)
    for group in times['jig_grind_groups']:
        print(format_jig_grind_group(group))
    print(f"\nTotal Jig Grind Time: {times['total_jig_grind_minutes']:.2f} minutes\n")

# TIME PER HOLE - TAP
if times['tap_groups']:
    print("TIME PER HOLE - TAP")
    print("-" * 80)
    for group in times['tap_groups']:
        print(format_tap_group(group))
    print(f"\nTotal Tapping Time: {times['total_tap_minutes']:.2f} minutes\n")

# TIME PER HOLE - C'BORE
if times['cbore_groups']:
    print("TIME PER HOLE - C'BORE")
    print("-" * 80)
    for group in times['cbore_groups']:
        print(format_cbore_group(group))
    print(f"\nTotal Counterbore Time: {times['total_cbore_minutes']:.2f} minutes\n")

# TIME PER HOLE - CDRILL
if times['cdrill_groups']:
    print("TIME PER HOLE - CDRILL")
    print("-" * 80)
    for group in times['cdrill_groups']:
        print(format_cdrill_group(group))
    print(f"\nTotal Center Drill Time: {times['total_cdrill_minutes']:.2f} minutes\n")

# Summary
print("=" * 80)
print(f"TOTAL MACHINE TIME: {times['total_minutes']:.2f} minutes ({times['total_hours']:.2f} hours)")
print("=" * 80)

print("\n[SUCCESS] Machine hours report test completed!")
print("\nThis output will now appear in AppV7's Output tab when you click 'Generate Quote'")
