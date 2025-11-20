#!/usr/bin/env python3
"""Compare small blank vs normal blank overage impact."""

from cad_quoter.planning.process_planner import planner_die_plate, estimate_machine_hours_from_plan

def analyze_part(L, W, T, material):
    """Analyze and display details for a part."""
    params = {"L": L, "W": W, "T": T, "material": material, "family": "die_plate"}

    plan = planner_die_plate(params)
    plan_dict = {'ops': plan.ops, 'fixturing': plan.fixturing, 'qa': plan.qa, 'warnings': plan.warnings}

    machine_hours = estimate_machine_hours_from_plan(
        plan=plan_dict,
        material=material,
        plate_LxW=(L, W),
        thickness=T,
        stock_thickness=T + (0.125 if max(L, W) < 3.0 else 0.25)
    )

    for op in machine_hours.get('milling_operations', []):
        if op.get('op_name') == 'full_square_up_mill':
            return op
    return None

print("="*80)
print("SMALL BLANK vs NORMAL BLANK COMPARISON")
print("="*80)

# Small blank
print("\n1. SMALL BLANK (2.0\" x 1.5\" x 0.5\")")
print("-" * 80)
small = analyze_part(2.0, 1.5, 0.5, "P20 Tool Steel")
if small:
    print(f"Max dimension: {max(2.0, 1.5):.2f}\" < 3.0\" → Small blank overage")
    print(f"Stock overage: +0.25\" L/W, +0.125\" T")
    print(f"Stock dimensions: {small['stock_length']:.3f}\" x {small['stock_width']:.3f}\" x {small['stock_thickness']:.3f}\"")
    print(f"Finished dimensions: {small['length']:.3f}\" x {small['width']:.3f}\" x {small['thickness']:.3f}\"")
    print(f"\nVolume breakdown:")
    print(f"  - Thickness: {small['volume_thickness']:.4f} in³")
    print(f"  - Length trim: {small['volume_length_trim']:.4f} in³")
    print(f"  - Width trim: {small['volume_width_trim']:.4f} in³")
    print(f"  - TOTAL: {small['volume_removed_cuin']:.4f} in³")
    print(f"\nTime: {small['time_minutes']:.2f} min")

# Normal blank (same proportions, scaled to 4x)
print("\n\n2. NORMAL BLANK (8.0\" x 6.0\" x 2.0\") - 4x scale")
print("-" * 80)
normal = analyze_part(8.0, 6.0, 2.0, "P20 Tool Steel")
if normal:
    print(f"Max dimension: {max(8.0, 6.0):.2f}\" ≥ 3.0\" → Normal blank overage")
    print(f"Stock overage: +0.50\" L/W, +0.25\" T")
    print(f"Stock dimensions: {normal['stock_length']:.3f}\" x {normal['stock_width']:.3f}\" x {normal['stock_thickness']:.3f}\"")
    print(f"Finished dimensions: {normal['length']:.3f}\" x {normal['width']:.3f}\" x {normal['thickness']:.3f}\"")
    print(f"\nVolume breakdown:")
    print(f"  - Thickness: {normal['volume_thickness']:.4f} in³")
    print(f"  - Length trim: {normal['volume_length_trim']:.4f} in³")
    print(f"  - Width trim: {normal['volume_width_trim']:.4f} in³")
    print(f"  - TOTAL: {normal['volume_removed_cuin']:.4f} in³")
    print(f"\nTime: {normal['time_minutes']:.2f} min")

# Comparison
if small and normal:
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Volume ratio (normal/small): {normal['volume_removed_cuin'] / small['volume_removed_cuin']:.2f}x")
    print(f"Time ratio (normal/small): {normal['time_minutes'] / small['time_minutes']:.2f}x")
    print(f"\nNote: The smaller overage for small blanks reduces unnecessary material removal")
    print(f"      and machining time for tiny parts.")
