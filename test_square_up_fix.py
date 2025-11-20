#!/usr/bin/env python3
"""
Test the new full-blank square-up logic.

This test creates a die plate similar to the example in the issue:
- Small width (2.5"), but significant length (8.72")
- Should now use fixed stock overage and full-blank square-up calculation
- Expected time: ~6-7 minutes instead of 2.72 minutes
"""

from cad_quoter.planning.process_planner import planner_die_plate, estimate_machine_hours_from_plan

def test_full_blank_square_up():
    """Test full-blank square-up with A2 tool steel plate."""

    # Create a die plate similar to the example in the issue
    # 305 A2 plate: ~8.72" L x 2.5" W x some thickness
    params = {
        "L": 8.72,
        "W": 2.50,
        "T": 1.0,  # Assume 1" thick
        "material": "A2 Tool Steel",
        "family": "die_plate",
    }

    print("=" * 80)
    print("TEST: Full-Blank Square-Up Fix")
    print("=" * 80)
    print(f"Part: {params['L']:.2f}\" L x {params['W']:.2f}\" W x {params['T']:.2f}\" T")
    print(f"Material: {params['material']}")
    print()

    # Generate plan
    plan = planner_die_plate(params)

    # Convert plan to dict for estimate_machine_hours_from_plan
    plan_dict = {
        'ops': plan.ops,
        'fixturing': plan.fixturing,
        'qa': plan.qa,
        'warnings': plan.warnings,
    }

    # Estimate machine hours
    machine_hours = estimate_machine_hours_from_plan(
        plan=plan_dict,
        material=params['material'],
        plate_LxW=(params['L'], params['W']),
        thickness=params['T'],
        stock_thickness=params['T'] + 0.25
    )

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    # Check for full_square_up_mill operation
    milling_ops = machine_hours.get('milling_operations', [])
    square_up_op = None
    for op in milling_ops:
        if op.get('op_name') == 'full_square_up_mill':
            square_up_op = op
            break

    if square_up_op:
        print("\n✓ Full square-up operation found!")
        print(f"  Operation: {square_up_op.get('op_description')}")
        print(f"  Finished dimensions: L={square_up_op.get('length'):.3f}\", W={square_up_op.get('width'):.3f}\", T={square_up_op.get('thickness'):.3f}\"")
        print(f"  Stock dimensions: L={square_up_op.get('stock_length'):.3f}\", W={square_up_op.get('stock_width'):.3f}\", T={square_up_op.get('stock_thickness'):.3f}\"")
        print(f"  Volume removed: {square_up_op.get('volume_removed_cuin'):.4f} in³")
        print(f"    - Thickness: {square_up_op.get('volume_thickness'):.4f} in³")
        print(f"    - Length trim: {square_up_op.get('volume_length_trim'):.4f} in³")
        print(f"    - Width trim: {square_up_op.get('volume_width_trim'):.4f} in³")
        print(f"  Material factor: {square_up_op.get('material_factor'):.2f}")
        print(f"  Time: {square_up_op.get('time_minutes'):.2f} minutes")

        # Verify expectations
        time_min = square_up_op.get('time_minutes', 0)
        if time_min >= 6.0 and time_min <= 8.0:
            print(f"\n✓ Time is in expected range (6-7 min): {time_min:.2f} min")
        elif time_min > 0:
            print(f"\n⚠ Time is outside expected range: {time_min:.2f} min (expected ~6-7 min)")
        else:
            print(f"\n✗ Time is zero or invalid: {time_min:.2f} min")

        # Check stock overage
        stock_L = square_up_op.get('stock_length', 0)
        stock_W = square_up_op.get('stock_width', 0)
        stock_T = square_up_op.get('stock_thickness', 0)
        fin_L = square_up_op.get('length', 0)
        fin_W = square_up_op.get('width', 0)
        fin_T = square_up_op.get('thickness', 0)

        overage_L = stock_L - fin_L
        overage_W = stock_W - fin_W
        overage_T = stock_T - fin_T

        if abs(overage_L - 0.50) < 0.01 and abs(overage_W - 0.50) < 0.01 and abs(overage_T - 0.25) < 0.01:
            print(f"✓ Stock overage is correct: +{overage_L:.2f}\" L, +{overage_W:.2f}\" W, +{overage_T:.2f}\" T")
        else:
            print(f"✗ Stock overage is incorrect: +{overage_L:.2f}\" L, +{overage_W:.2f}\" W, +{overage_T:.2f}\" T")
            print(f"  Expected: +0.50\" L, +0.50\" W, +0.25\" T")
    else:
        print("\n✗ No full_square_up_mill operation found!")
        print("Available milling operations:")
        for op in milling_ops:
            print(f"  - {op.get('op_name')}: {op.get('op_description')}")

    # Check total milling time
    total_milling = machine_hours.get('breakdown_minutes', {}).get('milling', 0)
    print(f"\nTotal milling time: {total_milling:.2f} min")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_full_blank_square_up()
