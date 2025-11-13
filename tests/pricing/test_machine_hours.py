"""Test machine hours estimation from CAD file with detailed hole-by-hole breakdown."""

from pathlib import Path
from cad_quoter.planning import (
    plan_from_cad_file,
    extract_hole_operations_from_cad,
    estimate_hole_table_times,
)

cad_file = Path("Cad Files/301_redacted.dxf")
if not cad_file.exists():
    print(f"ERROR: Test file not found: {cad_file}")
    exit(1)

print("=" * 70)
print("MACHINE HOURS ESTIMATION - DETAILED HOLE TABLE BREAKDOWN")
print("=" * 70)

# Load CAD file and extract data
print(f"\nLoading CAD file: {cad_file.name}")
plan = plan_from_cad_file(cad_file, verbose=False)

# Use expanded operations instead of compressed hole table
hole_table = extract_hole_operations_from_cad(cad_file)

# Extract dimensions from plan
dims = plan.get('extracted_dims', {})
L = dims.get('L', 0)
W = dims.get('W', 0)
T = dims.get('T', 0)

print(f"Dimensions: L={L}\" x W={W}\" x T={T}\"")
print(f"Hole entries: {len(hole_table)}")
print()

# Calculate times for hole table
material = "17-4 PH Stainless"
times = estimate_hole_table_times(hole_table, material, T)

# Display detailed breakdowns
def print_drill_group(group):
    """Print a drill group in the requested format."""
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | {group['sfm']:.0f} sfm | "
            f"{group['ipr']:.4f} ipr | t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def print_jig_grind_group(group):
    """Print a jig grind group."""
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def print_tap_group(group):
    """Print a tap group."""
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | {group['tpi']} TPI | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def print_cbore_group(group):
    """Print a counterbore group."""
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | {group['sfm']:.0f} sfm | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

def print_cdrill_group(group):
    """Print a center drill group."""
    return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
            f"depth {group['depth']:.3f}\" | "
            f"t/hole {group['time_per_hole']:.2f} min | "
            f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

# TIME PER HOLE - DRILL GROUPS
if times['drill_groups']:
    print("TIME PER HOLE - DRILL GROUPS")
    print("-" * 70)
    for group in times['drill_groups']:
        print(print_drill_group(group))
    print(f"\nTotal Drilling Time: {times['total_drill_minutes']:.2f} minutes\n")

# TIME PER HOLE - JIG GRIND
if times['jig_grind_groups']:
    print("TIME PER HOLE - JIG GRIND")
    print("-" * 70)
    for group in times['jig_grind_groups']:
        print(print_jig_grind_group(group))
    print(f"\nTotal Jig Grind Time: {times['total_jig_grind_minutes']:.2f} minutes\n")

# TIME PER HOLE - TAP
if times['tap_groups']:
    print("TIME PER HOLE - TAP")
    print("-" * 70)
    for group in times['tap_groups']:
        print(print_tap_group(group))
    print(f"\nTotal Tapping Time: {times['total_tap_minutes']:.2f} minutes\n")

# TIME PER HOLE - C'BORE
if times['cbore_groups']:
    print("TIME PER HOLE - C'BORE")
    print("-" * 70)
    for group in times['cbore_groups']:
        print(print_cbore_group(group))
    print(f"\nTotal Counterbore Time: {times['total_cbore_minutes']:.2f} minutes\n")

# TIME PER HOLE - CDRILL
if times['cdrill_groups']:
    print("TIME PER HOLE - CDRILL")
    print("-" * 70)
    for group in times['cdrill_groups']:
        print(print_cdrill_group(group))
    print(f"\nTotal Center Drill Time: {times['total_cdrill_minutes']:.2f} minutes\n")

# Summary
print("=" * 70)
print(f"TOTAL MACHINE TIME: {times['total_minutes']:.2f} minutes ({times['total_hours']:.2f} hours)")
print(f"Material: {material}")
print("=" * 70)
