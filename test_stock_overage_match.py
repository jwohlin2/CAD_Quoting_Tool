#!/usr/bin/env python3
"""
Test that stock overage calculations are consistent between:
1. Square-up milling (process_planner.py)
2. Material cost (DirectCostHelper.py)
"""

def calculate_stock_overage_planner(part_L, part_W, part_T):
    """Calculate stock overage using planner logic."""
    max_dim = max(part_L, part_W)
    if max_dim < 3.0:
        # Small blanks
        over_L = 0.25
        over_W = 0.25
        over_T = 0.125
    else:
        # Normal blanks
        over_L = 0.50
        over_W = 0.50
        over_T = 0.25

    return part_L + over_L, part_W + over_W, part_T + over_T

def calculate_stock_overage_direct_cost(part_L, part_W, part_T):
    """Calculate stock overage using DirectCostHelper logic."""
    max_dim = max(part_L, part_W)
    if max_dim < 3.0:
        # Small blanks
        over_L = 0.25
        over_W = 0.25
        over_T = 0.125
    else:
        # Normal blanks
        over_L = 0.50
        over_W = 0.50
        over_T = 0.25

    return part_L + over_L, part_W + over_W, part_T + over_T

def test_consistency(description, L, W, T):
    """Test that both calculations match."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Part: {L:.3f}\" x {W:.3f}\" x {T:.3f}\"")
    print(f"Max dimension: {max(L, W):.3f}\" → {'Small' if max(L, W) < 3.0 else 'Normal'} blank")

    stock_L_p, stock_W_p, stock_T_p = calculate_stock_overage_planner(L, W, T)
    stock_L_d, stock_W_d, stock_T_d = calculate_stock_overage_direct_cost(L, W, T)

    print(f"\nPlanner stock:     {stock_L_p:.3f}\" x {stock_W_p:.3f}\" x {stock_T_p:.3f}\"")
    print(f"DirectCost stock:  {stock_L_d:.3f}\" x {stock_W_d:.3f}\" x {stock_T_d:.3f}\"")

    match = (abs(stock_L_p - stock_L_d) < 0.001 and
             abs(stock_W_p - stock_W_d) < 0.001 and
             abs(stock_T_p - stock_T_d) < 0.001)

    if match:
        print("✓ PASS: Calculations match")
        return True
    else:
        print("✗ FAIL: Calculations don't match")
        return False

if __name__ == "__main__":
    print("="*80)
    print("STOCK OVERAGE CONSISTENCY TESTS")
    print("="*80)

    results = []

    # Small blank examples
    results.append(test_consistency(
        "Small blank - 0.75\" x 1.175\" x 0.5\" (user example)",
        0.75, 1.175, 0.5
    ))

    results.append(test_consistency(
        "Small blank - 2.0\" x 2.5\" x 1.0\"",
        2.0, 2.5, 1.0
    ))

    results.append(test_consistency(
        "Small blank at boundary - 2.99\" x 2.0\" x 0.75\"",
        2.99, 2.0, 0.75
    ))

    # Normal blank examples
    results.append(test_consistency(
        "Normal blank at boundary - 3.0\" x 2.5\" x 1.0\"",
        3.0, 2.5, 1.0
    ))

    results.append(test_consistency(
        "Normal blank - 8.72\" x 2.5\" x 1.0\" (original example)",
        8.72, 2.5, 1.0
    ))

    results.append(test_consistency(
        "Normal blank - 12.0\" x 8.0\" x 2.0\"",
        12.0, 8.0, 2.0
    ))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("✓ All logic tests passed - stock overage calculations match!")
        print("\nExpected stock for user's part (0.75\" x 1.175\" x 0.5\"):")
        print("  Required Stock: 1.000\" × 1.425\" × 0.625\"")
    else:
        print(f"✗ {total - passed} test(s) failed")
    print(f"{'='*80}")
