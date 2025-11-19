"""Test script for simple punch programming cap and critical OD tolerances.

This script tests:
1. Programming cap for simple round punches (15-30 min instead of 70+ min)
2. Critical OD tolerance handling (extra grinding and inspection time)
"""

from cad_quoter.pricing.time_estimator import (
    estimate_punch_machine_hours,
    estimate_punch_labor_hours,
    convert_punch_to_quote_machine_hours,
    PUNCH_TIME_CONSTANTS,
)


def test_simple_punch_programming_cap():
    """Test that simple round punches have capped programming time.

    Part 134 scenario: Simple turned part with:
    - One diameter step (Ø.250 pilot to Ø.375 body)
    - A flange/shoulder
    - A couple of small undercuts/chamfers
    - No holes, no threads

    Expected: Programming should be 15-30 min, not 70+ min
    """
    print("\n" + "=" * 70)
    print("TEST 1: Simple Round Punch Programming Cap")
    print("=" * 70)

    # Simple round punch features (like part 134)
    punch_features = {
        "shape_type": "round",
        "num_ground_diams": 2,  # Pilot + body (Ø.250, Ø.375)
        "total_ground_length_in": 1.5,
        "tap_count": 0,  # No taps
        "num_chamfers": 4,  # A few chamfers
        "num_small_radii": 2,  # Small undercuts/radii
        "has_polish_contour": False,
        "has_3d_surface": False,
        "overall_length_in": 2.0,
        "max_od_or_width_in": 0.375,
        "min_dia_tol_in": 0.001,  # Standard tolerance
    }

    # Create a simple plan with typical punch operations
    punch_plan = {
        "ops": [
            {"op": "stock_procurement"},
            {"op": "saw_to_length"},
            {"op": "rough_turning"},
            {"op": "OD_grind_rough"},
            {"op": "OD_grind_finish"},
            {"op": "Grind_length"},
            {"op": "chamfer"},
        ]
    }

    # Estimate machine hours first (needed for labor calculation)
    machine = estimate_punch_machine_hours(punch_plan, punch_features)
    labor = estimate_punch_labor_hours(punch_plan, punch_features, machine)

    print(f"Simple round punch characteristics:")
    print(f"  - Shape: round")
    print(f"  - Diameters: {punch_features['num_ground_diams']}")
    print(f"  - Chamfers: {punch_features['num_chamfers']}")
    print(f"  - Small radii: {punch_features['num_small_radii']}")
    print(f"  - Taps: {punch_features['tap_count']}")
    print(f"  - 3D surface: {punch_features['has_3d_surface']}")
    print(f"  - Polish: {punch_features['has_polish_contour']}")
    print()
    print(f"Number of operations: {len(punch_plan['ops'])}")
    print(f"Old formula: 30 + {len(punch_plan['ops'])} × 5 = {30 + len(punch_plan['ops']) * 5} min")
    print(f"New formula (simple punch): {labor.cam_programming_min:.2f} min")
    print()

    # Verify programming time is within cap
    max_cap = PUNCH_TIME_CONSTANTS["simple_punch_programming_cap"]
    assert labor.cam_programming_min <= max_cap, \
        f"Programming time {labor.cam_programming_min:.2f} exceeds cap {max_cap:.2f}"

    # Verify it's less than the old formula would produce
    old_formula = PUNCH_TIME_CONSTANTS["cam_programming_base"] + \
                  len(punch_plan['ops']) * PUNCH_TIME_CONSTANTS["cam_per_operation"]
    print(f"Programming reduced from {old_formula:.2f} to {labor.cam_programming_min:.2f} min")
    print(f"Savings: {old_formula - labor.cam_programming_min:.2f} min ({((old_formula - labor.cam_programming_min) / old_formula * 100):.1f}%)")

    return labor.cam_programming_min


def test_complex_punch_keeps_original_formula():
    """Test that complex punches (with holes, polish, etc.) use original formula."""
    print("\n" + "=" * 70)
    print("TEST 2: Complex Punch Uses Original Formula")
    print("=" * 70)

    # Complex punch with taps (not simple)
    punch_features = {
        "shape_type": "round",
        "num_ground_diams": 2,
        "total_ground_length_in": 1.5,
        "tap_count": 2,  # Has taps - NOT simple
        "num_chamfers": 4,
        "num_small_radii": 2,
        "has_polish_contour": False,
        "has_3d_surface": False,
        "overall_length_in": 2.0,
        "max_od_or_width_in": 0.375,
        "min_dia_tol_in": 0.001,
    }

    punch_plan = {
        "ops": [
            {"op": "stock_procurement"},
            {"op": "saw_to_length"},
            {"op": "rough_turning"},
            {"op": "OD_grind_rough"},
            {"op": "OD_grind_finish"},
            {"op": "Grind_length"},
            {"op": "chamfer"},
            {"op": "drill"},
            {"op": "tap"},
        ]
    }

    machine = estimate_punch_machine_hours(punch_plan, punch_features)
    labor = estimate_punch_labor_hours(punch_plan, punch_features, machine)

    expected = PUNCH_TIME_CONSTANTS["cam_programming_base"] + \
               len(punch_plan['ops']) * PUNCH_TIME_CONSTANTS["cam_per_operation"]

    print(f"Complex punch (has taps): {punch_features['tap_count']} taps")
    print(f"Expected programming: {expected:.2f} min")
    print(f"Actual programming: {labor.cam_programming_min:.2f} min")

    assert abs(labor.cam_programming_min - expected) < 0.01, \
        f"Complex punch should use original formula: {expected:.2f}, got {labor.cam_programming_min:.2f}"

    return labor.cam_programming_min


def test_critical_od_tolerance():
    """Test critical OD tolerance handling (≤0.0005" band).

    Part 134 scenario: Ø.2497 +0/-0.0002 = 0.0002" total band
    This is gage-pin territory and needs extra grinding and inspection.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Critical OD Tolerance Handling")
    print("=" * 70)

    # Punch with critical OD tolerance
    punch_features = {
        "shape_type": "round",
        "num_ground_diams": 2,
        "total_ground_length_in": 1.5,
        "tap_count": 0,
        "num_chamfers": 4,
        "num_small_radii": 2,
        "has_polish_contour": False,
        "has_3d_surface": False,
        "overall_length_in": 2.0,
        "max_od_or_width_in": 0.375,
        "min_dia_tol_in": 0.0002,  # Critical: +0/-0.0002
    }

    punch_plan = {"ops": []}

    machine = estimate_punch_machine_hours(punch_plan, punch_features)
    quote_machine = convert_punch_to_quote_machine_hours(machine, None)

    print(f"Critical tolerance: {punch_features['min_dia_tol_in']:.4f}\" (gage-pin territory)")
    print(f"Number of ground diameters: {punch_features['num_ground_diams']}")
    print()
    print(f"Critical OD tracking:")
    print(f"  - Critical OD tolerance: {machine.critical_od_tolerance_in:.4f}\"")
    print(f"  - Extra grinding time: {machine.critical_od_grinding_min:.2f} min")
    print(f"  - Extra inspection time: {machine.critical_od_inspection_min:.2f} min")
    print()

    # Verify critical OD info is tracked
    assert machine.critical_od_tolerance_in > 0, "Critical OD tolerance should be tracked"
    assert machine.critical_od_grinding_min > 0, "Critical OD grinding should be added"
    assert machine.critical_od_inspection_min > 0, "Critical OD inspection should be added"

    # Verify it appears in quote breakdown
    assert "critical_od_tolerance_in" in quote_machine, \
        "Critical OD should appear in quote breakdown"
    assert quote_machine["critical_od_grinding_minutes"] == machine.critical_od_grinding_min
    assert quote_machine["critical_od_inspection_minutes"] == machine.critical_od_inspection_min

    print(f"Quote breakdown includes:")
    print(f"  - critical_od_tolerance_in: {quote_machine['critical_od_tolerance_in']:.4f}\"")
    print(f"  - critical_od_grinding_minutes: {quote_machine['critical_od_grinding_minutes']:.2f} min")
    print(f"  - critical_od_inspection_minutes: {quote_machine['critical_od_inspection_minutes']:.2f} min")

    return machine


def test_standard_tolerance_no_extra():
    """Test that standard tolerances don't get critical OD treatment."""
    print("\n" + "=" * 70)
    print("TEST 4: Standard Tolerance (No Extra Critical OD Time)")
    print("=" * 70)

    # Punch with standard tolerance
    punch_features = {
        "shape_type": "round",
        "num_ground_diams": 2,
        "total_ground_length_in": 1.5,
        "tap_count": 0,
        "num_chamfers": 4,
        "num_small_radii": 2,
        "has_polish_contour": False,
        "has_3d_surface": False,
        "overall_length_in": 2.0,
        "max_od_or_width_in": 0.375,
        "min_dia_tol_in": 0.001,  # Standard tolerance
    }

    punch_plan = {"ops": []}

    machine = estimate_punch_machine_hours(punch_plan, punch_features)
    quote_machine = convert_punch_to_quote_machine_hours(machine, None)

    print(f"Standard tolerance: {punch_features['min_dia_tol_in']:.4f}\" (not critical)")
    print(f"Critical OD tracking:")
    print(f"  - Critical OD tolerance: {machine.critical_od_tolerance_in:.4f}\"")
    print(f"  - Extra grinding time: {machine.critical_od_grinding_min:.2f} min")
    print(f"  - Extra inspection time: {machine.critical_od_inspection_min:.2f} min")

    # Verify no critical OD info is tracked
    assert machine.critical_od_tolerance_in == 0, "Standard tolerance shouldn't be marked critical"
    assert machine.critical_od_grinding_min == 0, "No extra grinding for standard tolerance"
    assert machine.critical_od_inspection_min == 0, "No extra inspection for standard tolerance"

    # Verify it doesn't appear in quote breakdown
    assert "critical_od_tolerance_in" not in quote_machine, \
        "Critical OD shouldn't appear in quote breakdown for standard tolerance"

    print("No extra critical OD time added - correct!")

    return machine


def test_programming_scaling_with_features():
    """Test that simple punch programming scales correctly with features."""
    print("\n" + "=" * 70)
    print("TEST 5: Programming Scaling with Turned Features")
    print("=" * 70)

    tc = PUNCH_TIME_CONSTANTS
    base_features = {
        "shape_type": "round",
        "total_ground_length_in": 1.5,
        "tap_count": 0,
        "has_polish_contour": False,
        "has_3d_surface": False,
        "overall_length_in": 2.0,
        "max_od_or_width_in": 0.375,
        "min_dia_tol_in": 0.001,
    }

    punch_plan = {"ops": [{"op": "stock"}] * 5}

    # Test 1: Minimal features (1 diam, 0 chamfers, 0 radii)
    features1 = {**base_features, "num_ground_diams": 1, "num_chamfers": 0, "num_small_radii": 0}
    machine1 = estimate_punch_machine_hours(punch_plan, features1)
    labor1 = estimate_punch_labor_hours(punch_plan, features1, machine1)
    expected1 = tc["simple_punch_programming_base"]

    # Test 2: More features (3 diams, 4 chamfers, 2 radii)
    features2 = {**base_features, "num_ground_diams": 3, "num_chamfers": 4, "num_small_radii": 2}
    machine2 = estimate_punch_machine_hours(punch_plan, features2)
    labor2 = estimate_punch_labor_hours(punch_plan, features2, machine2)
    uncapped2 = (tc["simple_punch_programming_base"] +
                 2 * tc["simple_punch_per_extra_diam"] +  # 2 extra diameters
                 4 * tc["simple_punch_per_chamfer"] +     # 4 chamfers
                 2 * tc["simple_punch_per_radius"])       # 2 radii
    expected2 = min(uncapped2, tc["simple_punch_programming_cap"])  # Apply cap

    print(f"Minimal features (1 diam, 0 chamfers, 0 radii):")
    print(f"  Expected: {expected1:.2f} min, Actual: {labor1.cam_programming_min:.2f} min")
    print()
    print(f"More features (3 diams, 4 chamfers, 2 radii):")
    print(f"  Expected: {expected2:.2f} min, Actual: {labor2.cam_programming_min:.2f} min")
    print()

    assert abs(labor1.cam_programming_min - expected1) < 0.01, \
        f"Minimal features: expected {expected1:.2f}, got {labor1.cam_programming_min:.2f}"
    assert abs(labor2.cam_programming_min - expected2) < 0.01, \
        f"More features: expected {expected2:.2f}, got {labor2.cam_programming_min:.2f}"

    print(f"Feature scaling:")
    print(f"  Per extra diameter: +{tc['simple_punch_per_extra_diam']:.1f} min")
    print(f"  Per chamfer: +{tc['simple_punch_per_chamfer']:.1f} min")
    print(f"  Per small radius: +{tc['simple_punch_per_radius']:.1f} min")

    return labor1.cam_programming_min, labor2.cam_programming_min


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# SIMPLE PUNCH PROGRAMMING & CRITICAL OD TOLERANCE TESTS")
    print("#" * 70)

    try:
        # Test programming cap
        prog_time = test_simple_punch_programming_cap()
        test_complex_punch_keeps_original_formula()
        test_programming_scaling_with_features()

        # Test critical OD tolerance
        test_critical_od_tolerance()
        test_standard_tolerance_no_extra()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("Summary of improvements:")
        print(f"1. Simple punch programming capped at {PUNCH_TIME_CONSTANTS['simple_punch_programming_cap']:.0f} min")
        print(f"   (was 70+ min for part 134 scenario, now {prog_time:.2f} min)")
        print()
        print(f"2. Critical OD tolerances (<={PUNCH_TIME_CONSTANTS['critical_od_tolerance_threshold']:.4f}\") now tracked")
        print(f"   - Extra grinding: +{PUNCH_TIME_CONSTANTS['critical_od_grinding_adder']:.0f} min per diameter")
        print(f"   - Extra inspection: +{PUNCH_TIME_CONSTANTS['critical_od_inspection_adder']:.0f} min per diameter")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
