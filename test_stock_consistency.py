#!/usr/bin/env python3
"""
Test that stock dimensions are consistent between:
1. Square-up milling calculation (process_planner.py)
2. Material cost calculation (DirectCostHelper.py)
"""

from cad_quoter.planning.process_planner import planner_die_plate, estimate_machine_hours_from_plan
from cad_quoter.pricing.DirectCostHelper import calculate_machining_scrap_from_cad
from pathlib import Path
import tempfile
import ezdxf

def create_test_dxf(length, width, thickness):
    """Create a simple test DXF with specified dimensions."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Draw a simple rectangle
    msp.add_lwpolyline([
        (0, 0),
        (length, 0),
        (length, width),
        (0, width),
        (0, 0)
    ])

    # Add dimension text
    msp.add_text(
        f'{length:.4f}x{width:.4f}x{thickness:.4f}',
        dxfattribs={'height': 0.1}
    ).set_placement((0, -0.5))

    # Save to temp file
    temp = tempfile.NamedTemporaryFile(suffix='.dxf', delete=False)
    doc.saveas(temp.name)
    return Path(temp.name)

def test_stock_consistency(description, L, W, T, expected_stock_L, expected_stock_W, expected_stock_T):
    """Test that both calculations produce the same stock dimensions."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Part: {L:.3f}\" x {W:.3f}\" x {T:.3f}\"")
    print(f"Max dimension: {max(L, W):.3f}\"")

    # Test 1: Square-up calculation
    params = {"L": L, "W": W, "T": T, "material": "P20 Tool Steel", "family": "die_plate"}
    plan = planner_die_plate(params)
    plan_dict = {'ops': plan.ops, 'fixturing': plan.fixturing, 'qa': plan.qa, 'warnings': plan.warnings}

    machine_hours = estimate_machine_hours_from_plan(
        plan=plan_dict,
        material="P20 Tool Steel",
        plate_LxW=(L, W),
        thickness=T,
        stock_thickness=T + (0.125 if max(L, W) < 3.0 else 0.25)
    )

    # Get square-up stock dimensions
    square_up_stock_L = None
    square_up_stock_W = None
    square_up_stock_T = None
    for op in machine_hours.get('milling_operations', []):
        if op.get('op_name') == 'full_square_up_mill':
            square_up_stock_L = op.get('stock_length')
            square_up_stock_W = op.get('stock_width')
            square_up_stock_T = op.get('stock_thickness')
            break

    # Test 2: Material cost calculation
    dxf_path = create_test_dxf(L, W, T)
    try:
        scrap_info = calculate_machining_scrap_from_cad(
            dxf_path,
            part_length=L,
            part_width=W,
            part_thickness=T,
            verbose=False
        )
        material_stock_L = scrap_info['desired_length']
        material_stock_W = scrap_info['desired_width']
        material_stock_T = scrap_info['desired_thickness']
    finally:
        dxf_path.unlink()  # Clean up temp file

    # Compare
    print(f"\nSquare-up calculation stock: {square_up_stock_L:.3f}\" x {square_up_stock_W:.3f}\" x {square_up_stock_T:.3f}\"")
    print(f"Material cost stock:         {material_stock_L:.3f}\" x {material_stock_W:.3f}\" x {material_stock_T:.3f}\"")
    print(f"Expected stock:              {expected_stock_L:.3f}\" x {expected_stock_W:.3f}\" x {expected_stock_T:.3f}\"")

    # Check consistency
    sq_match = (abs(square_up_stock_L - material_stock_L) < 0.001 and
                abs(square_up_stock_W - material_stock_W) < 0.001 and
                abs(square_up_stock_T - material_stock_T) < 0.001)

    expected_match = (abs(material_stock_L - expected_stock_L) < 0.001 and
                      abs(material_stock_W - expected_stock_W) < 0.001 and
                      abs(material_stock_T - expected_stock_T) < 0.001)

    if sq_match and expected_match:
        print("\n✓ PASS: Stock dimensions are consistent")
        return True
    else:
        print("\n✗ FAIL: Stock dimensions mismatch")
        if not sq_match:
            print("  Square-up and material cost calculations don't match")
        if not expected_match:
            print("  Material cost doesn't match expected values")
        return False

if __name__ == "__main__":
    print("="*80)
    print("STOCK DIMENSION CONSISTENCY TESTS")
    print("="*80)

    results = []

    # Small blank examples
    results.append(test_stock_consistency(
        description="Small blank - 0.75\" x 1.175\" x 0.5\" (your example)",
        L=0.75,
        W=1.175,
        T=0.5,
        expected_stock_L=1.0,
        expected_stock_W=1.425,
        expected_stock_T=0.625
    ))

    results.append(test_stock_consistency(
        description="Small blank - 2.0\" x 2.5\" x 1.0\"",
        L=2.0,
        W=2.5,
        T=1.0,
        expected_stock_L=2.25,
        expected_stock_W=2.75,
        expected_stock_T=1.125
    ))

    # Normal blank examples
    results.append(test_stock_consistency(
        description="Normal blank - 4.0\" x 3.5\" x 1.5\"",
        L=4.0,
        W=3.5,
        T=1.5,
        expected_stock_L=4.5,
        expected_stock_W=4.0,
        expected_stock_T=1.75
    ))

    results.append(test_stock_consistency(
        description="Normal blank - 8.72\" x 2.5\" x 1.0\" (original example)",
        L=8.72,
        W=2.5,
        T=1.0,
        expected_stock_L=9.22,
        expected_stock_W=3.0,
        expected_stock_T=1.25
    ))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("✓ All tests passed - stock dimensions are consistent!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    print(f"{'='*80}")
