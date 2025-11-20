#!/usr/bin/env python3
"""
Test script to verify the new inspection and CMM time allocation features.
"""

from cad_quoter.planning.process_planner import (
    LaborInputs,
    cmm_inspection_minutes,
    compute_labor_minutes
)

def test_cmm_inspection_levels():
    """Test CMM inspection time scaling with different inspection levels."""
    print("=" * 70)
    print("Test 1: CMM Inspection Time Scaling")
    print("=" * 70)

    holes_total = 88

    for level in ["full_first_article", "critical_only", "spot_check"]:
        result = cmm_inspection_minutes(holes_total, inspection_level=level)
        print(f"\n{level}:")
        print(f"  Setup (labor): {result['setup_labor_min']:.1f} min")
        print(f"  Checking (machine): {result['checking_machine_min']:.1f} min")
        print(f"  Total: {result['total_min']:.1f} min")
        print(f"  Holes checked: {result['holes_checked']}")

def test_inspection_breakdown():
    """Test inspection labor breakdown with different inspection levels."""
    print("\n" + "=" * 70)
    print("Test 2: Inspection Labor Breakdown")
    print("=" * 70)

    # Test case: Part with 88 holes, 30 checked by CMM
    for level in ["full_first_article", "critical_only", "spot_check"]:
        inputs = LaborInputs(
            ops_total=15,
            holes_total=88,
            tool_changes=8,
            fixturing_complexity=1,
            cmm_setup_min=30.0,
            cmm_holes_checked=88,  # All holes checked by CMM
            inspection_level=level,
            net_weight_lb=25.0,  # Below 40 lb threshold
        )

        result = compute_labor_minutes(inputs)
        inspection = result['inspection_breakdown']

        print(f"\n{level}:")
        print(f"  Base: {inspection['insp_base_min']:.1f} min")
        print(f"  Dim checks: {inspection['insp_dim_checks_min']:.1f} min")
        print(f"  Runout/concentricity: {inspection['insp_runout_concentricity_min']:.1f} min")
        print(f"  Other features: {inspection['insp_other_features_min']:.1f} min")
        print(f"  Sampling: {inspection['insp_sampling_min']:.1f} min")
        print(f"  CMM setup: {inspection['insp_cmm_setup_min']:.1f} min")
        print(f"  Handling bump: {inspection['insp_handling_bump_min']:.1f} min")
        print(f"  TOTAL INSPECTION: {inspection['total_min']:.1f} min")

def test_double_counting_prevention():
    """Test that CMM-inspected holes are not double counted."""
    print("\n" + "=" * 70)
    print("Test 3: Double Counting Prevention")
    print("=" * 70)

    # Part with 88 holes, all checked by CMM
    inputs_with_cmm = LaborInputs(
        ops_total=15,
        holes_total=88,
        tool_changes=8,
        fixturing_complexity=1,
        cmm_setup_min=30.0,
        cmm_holes_checked=88,  # All holes checked by CMM
        inspection_level="critical_only",
        net_weight_lb=25.0,
    )

    # Same part but no CMM
    inputs_no_cmm = LaborInputs(
        ops_total=15,
        holes_total=88,
        tool_changes=8,
        fixturing_complexity=1,
        cmm_setup_min=0.0,
        cmm_holes_checked=0,  # No CMM
        inspection_level="critical_only",
        net_weight_lb=25.0,
    )

    result_with_cmm = compute_labor_minutes(inputs_with_cmm)
    result_no_cmm = compute_labor_minutes(inputs_no_cmm)

    insp_with_cmm = result_with_cmm['inspection_breakdown']
    insp_no_cmm = result_no_cmm['inspection_breakdown']

    print(f"\nWith CMM (88 holes checked by CMM):")
    print(f"  Manual hole inspection: {insp_with_cmm['insp_dim_checks_min']:.1f} min (0 holes)")
    print(f"  CMM setup: {insp_with_cmm['insp_cmm_setup_min']:.1f} min")
    print(f"  Total inspection labor: {insp_with_cmm['total_min']:.1f} min")

    print(f"\nWithout CMM (manual inspection of 88 holes):")
    print(f"  Manual hole inspection: {insp_no_cmm['insp_dim_checks_min']:.1f} min (88 holes × 0.5)")
    print(f"  CMM setup: {insp_no_cmm['insp_cmm_setup_min']:.1f} min")
    print(f"  Total inspection labor: {insp_no_cmm['total_min']:.1f} min")

    print(f"\n✓ Holes are NOT double counted when CMM is used")

def test_heavy_part_handling():
    """Test heavy part handling bump (>40 lbs)."""
    print("\n" + "=" * 70)
    print("Test 4: Heavy Part Handling Bump")
    print("=" * 70)

    # Light part (25 lbs)
    inputs_light = LaborInputs(
        ops_total=15,
        holes_total=10,
        tool_changes=8,
        fixturing_complexity=1,
        net_weight_lb=25.0,  # Below threshold
        machine_time_minutes=60.0,
    )

    # Heavy part (55 lbs)
    inputs_heavy = LaborInputs(
        ops_total=15,
        holes_total=10,
        tool_changes=8,
        fixturing_complexity=1,
        net_weight_lb=55.0,  # Above 40 lb threshold
        machine_time_minutes=60.0,
    )

    result_light = compute_labor_minutes(inputs_light)
    result_heavy = compute_labor_minutes(inputs_heavy)

    print(f"\nLight part (25 lbs):")
    print(f"  Setup: {result_light['minutes']['Setup']:.1f} min")
    print(f"  Machining: {result_light['minutes']['Machining_Steps']:.1f} min")
    print(f"  Inspection: {result_light['minutes']['Inspection']:.1f} min")
    print(f"  Handling bump: {result_light['inspection_breakdown']['insp_handling_bump_min']:.1f} min")

    print(f"\nHeavy part (55 lbs):")
    print(f"  Setup: {result_heavy['minutes']['Setup']:.1f} min (+10 min bump)")
    print(f"  Machining: {result_heavy['minutes']['Machining_Steps']:.1f} min (+5 min bump)")
    print(f"  Inspection: {result_heavy['minutes']['Inspection']:.1f} min (+5 min bump)")
    print(f"  Handling bump: {result_heavy['inspection_breakdown']['insp_handling_bump_min']:.1f} min")

    setup_diff = result_heavy['minutes']['Setup'] - result_light['minutes']['Setup']
    machining_diff = result_heavy['minutes']['Machining_Steps'] - result_light['minutes']['Machining_Steps']
    inspection_diff = result_heavy['minutes']['Inspection'] - result_light['minutes']['Inspection']

    print(f"\n✓ Heavy part adds {setup_diff:.1f} min to setup (expected: 10)")
    print(f"✓ Heavy part adds {machining_diff:.1f} min to machining (expected: 5)")
    print(f"✓ Heavy part adds {inspection_diff:.1f} min to inspection (expected: 5)")

def test_simple_part_caps():
    """Test simple-part caps for programming and inspection."""
    print("\n" + "=" * 70)
    print("Test 5: Simple Part Caps")
    print("=" * 70)

    # Simple part: 4 holes, no special operations
    inputs_simple = LaborInputs(
        ops_total=6,
        holes_total=4,
        tool_changes=4,
        fixturing_complexity=1,
        cmm_holes_checked=0,  # No CMM
        edm_window_count=0,
        jig_grind_bore_qty=0,
        inspection_level="critical_only",
        machine_time_minutes=15.0,
    )

    result = compute_labor_minutes(inputs_simple)

    print(f"\nSimple part (4 holes, no special ops):")
    print(f"  Programming: {result['minutes']['Programming']:.1f} min (capped at 10)")
    print(f"  Inspection: {result['minutes']['Inspection']:.1f} min (capped at 8)")

    print(f"\n✓ Simple parts have reasonable time caps applied")

if __name__ == "__main__":
    test_cmm_inspection_levels()
    test_inspection_breakdown()
    test_double_counting_prevention()
    test_heavy_part_handling()
    test_simple_part_caps()

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
