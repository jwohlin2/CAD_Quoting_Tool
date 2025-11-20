#!/usr/bin/env python3
"""
Test size-based stock overage tiers.

Tests that:
- Small blanks (max_dim < 3.0") get +0.25" L/W, +0.125" T
- Normal blanks (max_dim ≥ 3.0") get +0.50" L/W, +0.25" T
"""

from cad_quoter.planning.process_planner import planner_die_plate, estimate_machine_hours_from_plan

def test_part(description, L, W, T, material, expected_over_L, expected_over_W, expected_over_T):
    """Test a single part and verify stock overage."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")

    params = {
        "L": L,
        "W": W,
        "T": T,
        "material": material,
        "family": "die_plate",
    }

    print(f"Part: {L:.2f}\" L x {W:.2f}\" W x {T:.2f}\" T")
    print(f"Max dimension: {max(L, W):.2f}\"")
    print(f"Material: {material}")

    # Generate plan
    plan = planner_die_plate(params)
    plan_dict = {
        'ops': plan.ops,
        'fixturing': plan.fixturing,
        'qa': plan.qa,
        'warnings': plan.warnings,
    }

    # Estimate machine hours
    machine_hours = estimate_machine_hours_from_plan(
        plan=plan_dict,
        material=material,
        plate_LxW=(L, W),
        thickness=T,
        stock_thickness=T + expected_over_T
    )

    # Check for full_square_up_mill operation
    milling_ops = machine_hours.get('milling_operations', [])
    square_up_op = None
    for op in milling_ops:
        if op.get('op_name') == 'full_square_up_mill':
            square_up_op = op
            break

    if square_up_op:
        stock_L = square_up_op.get('stock_length', 0)
        stock_W = square_up_op.get('stock_width', 0)
        stock_T = square_up_op.get('stock_thickness', 0)
        fin_L = square_up_op.get('length', 0)
        fin_W = square_up_op.get('width', 0)
        fin_T = square_up_op.get('thickness', 0)

        actual_over_L = stock_L - fin_L
        actual_over_W = stock_W - fin_W
        actual_over_T = stock_T - fin_T

        print(f"\nActual overage: +{actual_over_L:.3f}\" L, +{actual_over_W:.3f}\" W, +{actual_over_T:.3f}\" T")
        print(f"Expected overage: +{expected_over_L:.3f}\" L, +{expected_over_W:.3f}\" W, +{expected_over_T:.3f}\" T")

        # Check if overage matches expectations
        L_match = abs(actual_over_L - expected_over_L) < 0.01
        W_match = abs(actual_over_W - expected_over_W) < 0.01
        T_match = abs(actual_over_T - expected_over_T) < 0.01

        if L_match and W_match and T_match:
            print("✓ Stock overage is CORRECT")
        else:
            print("✗ Stock overage is INCORRECT")
            if not L_match:
                print(f"  Length: expected {expected_over_L:.3f}\", got {actual_over_L:.3f}\"")
            if not W_match:
                print(f"  Width: expected {expected_over_W:.3f}\", got {actual_over_W:.3f}\"")
            if not T_match:
                print(f"  Thickness: expected {expected_over_T:.3f}\", got {actual_over_T:.3f}\"")

        # Show volume and time
        volume = square_up_op.get('volume_removed_cuin', 0)
        time = square_up_op.get('time_minutes', 0)
        print(f"\nVolume removed: {volume:.4f} in³")
        print(f"Square-up time: {time:.2f} min")

        return L_match and W_match and T_match
    else:
        print("\n✗ No full_square_up_mill operation found!")
        return False

if __name__ == "__main__":
    print("="*80)
    print("STOCK OVERAGE TIER TESTS")
    print("="*80)

    results = []

    # Test 1: Small blank (max_dim < 3.0") - tiny part
    results.append(test_part(
        description="Small blank - 2.0\" x 1.5\" x 0.5\" (max_dim=2.0\")",
        L=2.0,
        W=1.5,
        T=0.5,
        material="P20 Tool Steel",
        expected_over_L=0.25,
        expected_over_W=0.25,
        expected_over_T=0.125
    ))

    # Test 2: Small blank at boundary (max_dim = 2.99")
    results.append(test_part(
        description="Small blank - 2.99\" x 2.0\" x 0.75\" (max_dim=2.99\")",
        L=2.99,
        W=2.0,
        T=0.75,
        material="Aluminum 6061-T6",
        expected_over_L=0.25,
        expected_over_W=0.25,
        expected_over_T=0.125
    ))

    # Test 3: Normal blank at boundary (max_dim = 3.0")
    results.append(test_part(
        description="Normal blank - 3.0\" x 2.5\" x 1.0\" (max_dim=3.0\")",
        L=3.0,
        W=2.5,
        T=1.0,
        material="P20 Tool Steel",
        expected_over_L=0.50,
        expected_over_W=0.50,
        expected_over_T=0.25
    ))

    # Test 4: Normal blank - original example (8.72" x 2.5")
    results.append(test_part(
        description="Normal blank - 8.72\" x 2.5\" x 1.0\" (max_dim=8.72\")",
        L=8.72,
        W=2.50,
        T=1.0,
        material="A2 Tool Steel",
        expected_over_L=0.50,
        expected_over_W=0.50,
        expected_over_T=0.25
    ))

    # Test 5: Normal blank - large part
    results.append(test_part(
        description="Normal blank - 12.0\" x 8.0\" x 2.0\" (max_dim=12.0\")",
        L=12.0,
        W=8.0,
        T=2.0,
        material="P20 Tool Steel",
        expected_over_L=0.50,
        expected_over_W=0.50,
        expected_over_T=0.25
    ))

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    print(f"{'='*80}")
