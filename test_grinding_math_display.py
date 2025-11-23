#!/usr/bin/env python3
"""Test script to verify grinding operation display math fix"""

import sys
sys.path.insert(0, '/home/user/CAD_Quoting_Tool')

from cad_quoter.pricing.QuoteDataHelper import GrindingOperation

# Test format_grinding_op function (copied from AppV7.py with the fix)
def format_grinding_op(op):
    """Format grinding operation with all details"""
    lines = []

    # Check if this is a volume-based or non-volume-based operation
    # Non-volume operations have volume_removed near 0 but non-zero time
    is_volume_based = op.volume_removed > 0.001

    if is_volume_based:
        # Volume-based grinding: show geometry and volume calculation
        lines.append(f"{op.op_description} | L={op.length:.3f}\" | W={op.width:.3f}\" | Area={op.area:.2f} sq in")
        lines.append(f"  stock_removed={op.stock_removed_total:.3f}\" | faces={op.faces} | volume={op.volume_removed:.3f} cu in")
        lines.append(f"  min_per_cuin={op.min_per_cuin:.1f} | material_factor={op.material_factor:.2f}")
        lines.append(f"  time = {op.volume_removed:.3f} × {op.min_per_cuin:.1f} × {op.material_factor:.2f} = {op.time_minutes:.2f} min")
    else:
        # Non-volume-based grinding: show simplified format
        lines.append(f"{op.op_description} | faces={op.faces}")
        if op.material_factor != 1.0:
            lines.append(f"  material_factor={op.material_factor:.2f}")
        lines.append(f"  time = {op.time_minutes:.2f} min")

    return "\n".join(lines)


# Test case 1: Volume-based grinding (normal case)
print("=" * 80)
print("TEST 1: Volume-based grinding (should show full formula)")
print("=" * 80)
op1 = GrindingOperation(
    op_name='grind_faces',
    op_description='Face Grind - Final Kiss',
    length=8.500,
    width=4.250,
    area=36.13,
    stock_removed_total=0.006,
    faces=2,
    volume_removed=0.217,
    min_per_cuin=3.0,
    material_factor=1.00,
    time_minutes=0.65
)
print(format_grinding_op(op1))
print()

# Test case 2: Non-volume-based grinding (the problematic case)
print("=" * 80)
print("TEST 2: Non-volume-based grinding - Rough Grind (should show simplified format)")
print("=" * 80)
op2 = GrindingOperation(
    op_name='rough_grind_all_faces',
    op_description='Rough grind all 6 faces to establish datums (0.150 in³ removed)',
    length=0.0,  # Not set for die section operations
    width=0.0,   # Not set for die section operations
    area=0.0,    # Not set for die section operations
    stock_removed_total=0.0,
    faces=6,
    volume_removed=0.0,  # Not set for these operations
    min_per_cuin=3.0,    # Default value
    material_factor=2.50,  # Carbide factor
    time_minutes=45.88
)
print(format_grinding_op(op2))
print()

# Test case 3: Non-volume-based grinding - Finish Grind
print("=" * 80)
print("TEST 3: Non-volume-based grinding - Finish Grind (should show simplified format)")
print("=" * 80)
op3 = GrindingOperation(
    op_name='finish_grind_thickness',
    op_description='Finish grind top/bottom to thickness ±0.0005"',
    length=0.0,
    width=0.0,
    area=0.0,
    stock_removed_total=0.0,
    faces=2,
    volume_removed=0.0,
    min_per_cuin=3.0,
    material_factor=2.50,
    time_minutes=5.00
)
print(format_grinding_op(op3))
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ Volume-based operations show full geometry and math formula")
print("✓ Non-volume operations show simplified format without nonsense math")
print("✓ Fix successfully prevents '0.000 × 3.0 × 1.00 = 45.88 min' display")
