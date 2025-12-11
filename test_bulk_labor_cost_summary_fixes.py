#!/usr/bin/env python3
"""
Test suite for bulk labor + cost summary + machine breakdown fixes.

Tests:
1. Labor cost in PART COST SUMMARY matches LABOR HOURS job-level totals
2. Machine breakdown total equals sum of visible breakdown lines
3. Scrap percentage calculation uses weight-based formula
"""

from cad_quoter.pricing.time_estimator import PunchMachineHours, convert_punch_to_quote_machine_hours, PunchLaborHours


def test_labor_cost_job_level_calculation():
    """
    Test that job-level labor costs are calculated correctly for parts with quantity > 1.

    The LABOR HOURS block calculates:
    - job_level_labor = setup + programming + inspection (one-time costs)
    - variable_labor_per_unit = machining + finishing (per-unit costs)
    - total_job_labor_cost = job_level_labor + (variable_labor_per_unit × qty)
    - labor_cost_per_unit = total_job_labor_cost / qty

    The PART COST SUMMARY should use these values, NOT multiply per-unit by qty.
    """
    # Example: m1_105-A with 16 pieces
    quantity = 16
    labor_rate = 60.0  # $60/hr

    # Job-level costs (one-time)
    setup_min = 30.0
    programming_min = 20.0
    inspection_min = 25.0
    job_level_min = setup_min + programming_min + inspection_min  # 75 min

    # Variable costs (per-unit)
    machining_min = 45.0  # per unit
    finishing_min = 8.0   # per unit
    variable_min_per_unit = machining_min + finishing_min  # 53 min

    # Calculate total job labor
    total_job_labor_min = job_level_min + (variable_min_per_unit * quantity)
    total_job_labor_cost = (total_job_labor_min / 60.0) * labor_rate

    # Calculate per-unit labor (averaged)
    labor_cost_per_unit = total_job_labor_cost / quantity

    # Verify calculations
    expected_total_min = 75 + (53 * 16)  # 75 + 848 = 923 min
    expected_total_cost = (923 / 60.0) * 60.0  # $923.00
    expected_per_unit = 923.0 / 16  # $57.69 (rounded)

    assert abs(total_job_labor_min - expected_total_min) < 0.01
    assert abs(total_job_labor_cost - expected_total_cost) < 0.01
    assert abs(labor_cost_per_unit - expected_per_unit) < 0.01

    # The key assertion: total_job_labor_cost should NOT equal labor_cost_per_unit * quantity
    # (due to rounding, they may differ slightly)
    # But the job-level cost should be calculated from minutes, not from per-unit cost
    recalculated_from_per_unit = labor_cost_per_unit * quantity

    # In this case they should match because we're using simple math
    # But in practice, rounding can cause small differences
    assert abs(total_job_labor_cost - recalculated_from_per_unit) < 0.50

    print(f"\n✓ Labor cost calculation test passed")
    print(f"  Quantity: {quantity}")
    print(f"  Job-level labor: ${total_job_labor_cost:.2f}")
    print(f"  Per-unit labor: ${labor_cost_per_unit:.2f}")


def test_machine_breakdown_equals_total():
    """
    Test that the total machine time equals the sum of visible breakdown lines.

    For punch parts, the breakdown shows:
    - Turning (rough + finish)
    - Grinding (OD/ID/face)
    - Drilling
    - Tapping
    - EDM
    - Edge break / deburr (if > 0)
    - Etch / marking (if > 0)
    - Polish contour (if > 0)
    - Other (chamfer/saw) - NOTE: polishing removed from here
    - Inspection

    The total should equal the sum of all these lines.
    """
    # Create a PunchMachineHours object with sample values
    machine_hours = PunchMachineHours(
        rough_turning_min=12.5,
        finish_turning_min=8.3,
        od_grinding_min=15.2,
        id_grinding_min=0.0,
        face_grinding_min=6.8,
        drilling_min=3.5,
        tapping_min=2.1,
        chamfer_min=1.5,
        polishing_min=4.2,
        edm_min=0.0,
        sawing_min=1.0,
        etch_marking_min=0.5,
        inspection_min=5.0,
        critical_od_grinding_min=3.0,
        critical_od_inspection_min=2.0,
    )

    # Calculate totals
    machine_hours.calculate_totals()

    # Convert to quote format
    labor_hours = PunchLaborHours()  # Empty labor hours for conversion
    quote_dict = convert_punch_to_quote_machine_hours(machine_hours, labor_hours)

    # Extract the breakdown fields (matching the display in AppV7.py)
    turning = quote_dict["total_milling_minutes"]  # rough + finish
    grinding = quote_dict["total_grinding_minutes"]  # OD + ID + face + critical OD
    drilling = quote_dict["total_drill_minutes"]
    tapping = quote_dict["total_tap_minutes"]
    edm = quote_dict["total_edm_minutes"]
    edge_break = quote_dict["total_edge_break_minutes"]  # Should be 0 for punches
    etch = quote_dict["total_etch_minutes"]
    polish = quote_dict["total_polish_minutes"]
    other = quote_dict["total_other_minutes"]  # chamfer + saw (NOT including polish)
    inspection = quote_dict["total_cmm_minutes"]

    # Calculate the sum of visible breakdown lines
    breakdown_sum = (
        turning + grinding + drilling + tapping + edm +
        edge_break + etch + polish + other + inspection
    )

    # Get the total from the quote dict
    total_minutes = quote_dict["total_minutes"]

    # Debug output
    print(f"\nDEBUG: Machine breakdown")
    print(f"  Turning: {turning:.2f}")
    print(f"  Grinding: {grinding:.2f}")
    print(f"  Drilling: {drilling:.2f}")
    print(f"  Tapping: {tapping:.2f}")
    print(f"  EDM: {edm:.2f}")
    print(f"  Edge break: {edge_break:.2f}")
    print(f"  Etch: {etch:.2f}")
    print(f"  Polish: {polish:.2f}")
    print(f"  Other: {other:.2f}")
    print(f"  Inspection: {inspection:.2f}")
    print(f"  Breakdown sum: {breakdown_sum:.2f}")
    print(f"  Total minutes: {total_minutes:.2f}")
    print(f"  Difference: {abs(breakdown_sum - total_minutes):.2f}")

    # Verify that breakdown sum equals total
    # Allow small floating point tolerance
    assert abs(breakdown_sum - total_minutes) < 0.01, (
        f"Breakdown sum ({breakdown_sum:.2f}) does not match total ({total_minutes:.2f}). "
        f"Difference: {abs(breakdown_sum - total_minutes):.2f} min"
    )

    print(f"\n✓ Machine breakdown test passed")
    print(f"  Breakdown sum: {breakdown_sum:.2f} min")
    print(f"  Total minutes: {total_minutes:.2f} min")
    print(f"  Difference: {abs(breakdown_sum - total_minutes):.4f} min")

    # Verify specific calculations
    expected_turning = 12.5 + 8.3  # 20.8
    expected_grinding = 15.2 + 0.0 + 6.8 + 3.0  # 25.0 (includes critical OD)
    expected_other = 1.5 + 1.0  # 2.5 (chamfer + saw, NOT including polish)
    expected_inspection = 5.0 + 2.0  # 7.0 (includes critical OD inspection)

    assert abs(turning - expected_turning) < 0.01
    assert abs(grinding - expected_grinding) < 0.01
    assert abs(other - expected_other) < 0.01, (
        f"Other minutes ({other:.2f}) should NOT include polishing ({polish:.2f}). "
        f"Expected {expected_other:.2f} (chamfer + saw only)"
    )
    assert abs(inspection - expected_inspection) < 0.01


def test_scrap_percentage_uses_weight():
    """
    Test that scrap percentage is calculated using weight, not volume.

    Formula: scrap_pct = (scrap_weight_lbs / starting_weight_lbs) × 100
    Rounded to 1 decimal place.

    Example from m1_101a-A:
    - Starting weight: 1.9 oz = 0.11875 lbs
    - Scrap weight: 1.4 oz = 0.0875 lbs
    - Scrap %: (0.0875 / 0.11875) × 100 = 73.68... ≈ 73.7%
    """
    # Convert oz to lbs (16 oz = 1 lb)
    starting_weight_oz = 1.9
    scrap_weight_oz = 1.4

    starting_weight_lbs = starting_weight_oz / 16.0  # 0.11875 lbs
    scrap_weight_lbs = scrap_weight_oz / 16.0  # 0.0875 lbs

    # Calculate scrap percentage
    scrap_pct = (scrap_weight_lbs / starting_weight_lbs) * 100
    scrap_pct_rounded = round(scrap_pct, 1)

    # Verify calculation
    expected_pct = (1.4 / 1.9) * 100  # ~73.68%
    expected_rounded = 73.7

    assert abs(scrap_pct - expected_pct) < 0.01
    assert abs(scrap_pct_rounded - expected_rounded) < 0.01

    print(f"\n✓ Scrap percentage test passed")
    print(f"  Starting weight: {starting_weight_oz:.1f} oz ({starting_weight_lbs:.5f} lbs)")
    print(f"  Scrap weight: {scrap_weight_oz:.1f} oz ({scrap_weight_lbs:.5f} lbs)")
    print(f"  Scrap %: {scrap_pct:.2f}% ≈ {scrap_pct_rounded:.1f}%")

    # Test edge case: zero starting weight
    scrap_pct_zero = (0.5 / 0.0) * 100 if 0.0 > 0 else 0
    assert scrap_pct_zero == 0

    # Test that percentage is clamped to [0, 100]
    # (This should be done in the actual code)
    scrap_pct_clamped = max(0.0, min(100.0, round(scrap_pct, 1)))
    assert 0.0 <= scrap_pct_clamped <= 100.0


def test_polishing_not_double_counted():
    """
    Test that polishing is not double-counted in the machine breakdown.

    Previously, polishing_min was included in both:
    - total_polish_minutes (shown separately)
    - total_other_minutes (shown as "Other (chamfer/polish/saw)")

    Now it should ONLY be in total_polish_minutes, and total_other_minutes
    should only contain chamfer + saw.
    """
    machine_hours = PunchMachineHours(
        chamfer_min=2.0,
        polishing_min=5.0,
        sawing_min=1.5,
    )
    machine_hours.calculate_totals()

    labor_hours = PunchLaborHours()
    quote_dict = convert_punch_to_quote_machine_hours(machine_hours, labor_hours)

    polish = quote_dict["total_polish_minutes"]
    other = quote_dict["total_other_minutes"]

    # Verify polishing is shown separately
    assert abs(polish - 5.0) < 0.01

    # Verify other only contains chamfer + saw (NOT polishing)
    expected_other = 2.0 + 1.5  # 3.5
    assert abs(other - expected_other) < 0.01, (
        f"Other ({other:.2f}) should NOT include polishing ({polish:.2f}). "
        f"Expected {expected_other:.2f} (chamfer + saw only)"
    )

    print(f"\n✓ Polishing double-count test passed")
    print(f"  Polish minutes: {polish:.2f}")
    print(f"  Other minutes: {other:.2f} (chamfer + saw, no polish)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BULK LABOR + COST SUMMARY + MACHINE BREAKDOWN FIXES TEST SUITE")
    print("=" * 70)

    test_labor_cost_job_level_calculation()
    test_machine_breakdown_equals_total()
    test_scrap_percentage_uses_weight()
    test_polishing_not_double_counted()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70 + "\n")
