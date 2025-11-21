#!/usr/bin/env python3
"""Debug script to check waterjet detection for a specific part."""

import sys
from pathlib import Path
from cad_quoter.planning.process_planner import plan_from_cad_file
from cad_quoter.geometry.dxf_enrich import detect_waterjet_openings, detect_waterjet_profile

def debug_waterjet(part_path: str):
    """Debug waterjet detection for a part file."""
    print(f"="*80)
    print(f"Debugging waterjet detection for: {part_path}")
    print(f"="*80)

    # Load the plan
    try:
        plan = plan_from_cad_file(part_path)
    except Exception as e:
        print(f"Error loading part: {e}")
        return

    # Check plan flags
    print(f"\n1. Plan waterjet flags:")
    print(f"   has_waterjet_openings: {plan.get('has_waterjet_openings', False)}")
    print(f"   has_waterjet_profile: {plan.get('has_waterjet_profile', False)}")
    print(f"   waterjet_openings_tolerance: {plan.get('waterjet_openings_tolerance', 'N/A')}")
    print(f"   waterjet_profile_tolerance: {plan.get('waterjet_profile_tolerance', 'N/A')}")

    # Check text_dump
    text_dump = plan.get('text_dump', '')
    print(f"\n2. Text dump analysis:")
    print(f"   text_dump length: {len(text_dump)} characters")

    if 'waterjet' in text_dump.lower():
        print(f"   ✓ Found 'waterjet' in text_dump")
        print(f"\n   Lines containing 'waterjet':")
        for i, line in enumerate(text_dump.split('\n'), 1):
            if 'waterjet' in line.lower():
                print(f"      Line {i}: {line.strip()}")
    else:
        print(f"   ✗ 'waterjet' NOT found in text_dump")

    # Test detection functions directly
    print(f"\n3. Direct detection test:")
    has_openings, tol_openings = detect_waterjet_openings(text_dump)
    has_profile, tol_profile = detect_waterjet_profile(text_dump)
    print(f"   detect_waterjet_openings(): {has_openings}, tolerance={tol_openings}")
    print(f"   detect_waterjet_profile(): {has_profile}, tolerance={tol_profile}")

    # Check machine hours calculation
    print(f"\n4. Machine hours calculation:")
    try:
        from cad_quoter.planning.process_planner import estimate_machine_hours_from_plan

        # Get part dimensions (you may need to adjust these)
        L = plan.get('length', 0) or 10.0
        W = plan.get('width', 0) or 10.0
        T = plan.get('thickness', 0) or 0.5
        material = plan.get('material', 'TOOL STEEL')

        result = estimate_machine_hours_from_plan(
            plan,
            material=material,
            plate_LxW=(L, W),
            thickness=T,
            stock_thickness=T
        )

        waterjet_ops = result.get('waterjet_operations', [])
        waterjet_mins = result.get('breakdown_minutes', {}).get('waterjet', 0.0)

        print(f"   waterjet_operations count: {len(waterjet_ops)}")
        print(f"   total waterjet minutes: {waterjet_mins}")

        if waterjet_ops:
            print(f"\n   Waterjet operations:")
            for i, op in enumerate(waterjet_ops, 1):
                print(f"      Op {i}: {op.get('op_description', 'N/A')}")
                print(f"         Time: {op.get('time_min', 0)} min")

    except Exception as e:
        print(f"   Error in machine hours calculation: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_waterjet_201.py <path_to_part_file>")
        print("\nExample:")
        print("  python debug_waterjet_201.py test_parts/201.dxf")
        print("  python debug_waterjet_201.py test_parts/201.pdf")
        sys.exit(1)

    part_path = sys.argv[1]
    if not Path(part_path).exists():
        print(f"Error: File not found: {part_path}")
        sys.exit(1)

    debug_waterjet(part_path)
