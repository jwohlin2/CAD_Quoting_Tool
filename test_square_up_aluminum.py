#!/usr/bin/env python3
"""Test square-up with aluminum (should have lower material factor)."""

from cad_quoter.planning.process_planner import planner_die_plate, estimate_machine_hours_from_plan

params = {
    "L": 8.72,
    "W": 2.50,
    "T": 1.0,
    "material": "Aluminum 6061-T6",
    "family": "die_plate",
}

print(f"Testing with: {params['material']}")
plan = planner_die_plate(params)
plan_dict = {'ops': plan.ops, 'fixturing': plan.fixturing, 'qa': plan.qa, 'warnings': plan.warnings}

machine_hours = estimate_machine_hours_from_plan(
    plan=plan_dict,
    material=params['material'],
    plate_LxW=(params['L'], params['W']),
    thickness=params['T'],
    stock_thickness=params['T'] + 0.25
)

for op in machine_hours.get('milling_operations', []):
    if op.get('op_name') == 'full_square_up_mill':
        print(f"\nMaterial: {params['material']}")
        print(f"Material factor: {op.get('material_factor'):.2f}")
        print(f"Volume: {op.get('volume_removed_cuin'):.4f} in³")
        print(f"Time: {op.get('time_minutes'):.2f} min")
        break
