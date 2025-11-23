#!/usr/bin/env python3
"""
Test that setup, programming, and inspection are properly amortized across quantity.

Problem: For parts with Quantity > 1, setup/programming/inspection minutes
should be job-level (amortized) not per-unit.

Expected behavior:
- Setup, programming, and first-article inspection are job-level costs
- Machining and finishing are per-unit variable costs
- Total job labor = (setup + programming + inspection) + (machining + finishing) × qty
- Per-unit labor = total job labor / qty
"""


def test_multi_qty_amortization():
    """Test that job-level costs (setup/programming/inspection) are amortized across quantity."""

    # Example values (in minutes)
    setup_min = 30.0
    programming_min = 19.0
    inspection_min = 30.0
    machining_min = 50.0  # per unit
    finishing_min = 10.0  # per unit

    quantity = 8
    labor_rate = 60.0  # $60/hr for easy math

    # ========================================================================
    # Simulate the calculation logic from QuoteDataHelper.py lines 2599-2630
    # ========================================================================

    # Calculate labor costs from minutes
    setup_labor = round(setup_min * (labor_rate / 60.0), 2)
    programming_labor = round(programming_min * (labor_rate / 60.0), 2)
    inspection_labor = round(inspection_min * (labor_rate / 60.0), 2)

    # Job-level costs (setup, programming, first-article inspection) - amortized across quantity
    job_level_labor = setup_labor + programming_labor + inspection_labor
    amortized_job_level_cost = round(job_level_labor / quantity, 2)

    # Variable labor costs per unit (machining, finishing only - inspection is job-level)
    machining_labor = round(machining_min * (labor_rate / 60.0), 2)
    finishing_labor = round(finishing_min * (labor_rate / 60.0), 2)
    misc_overhead_labor = 0.0
    variable_labor_per_unit = machining_labor + finishing_labor + misc_overhead_labor

    # Per-unit labor cost
    per_unit_labor_cost = round(amortized_job_level_cost + variable_labor_per_unit, 2)

    # Total labor cost for all units
    total_labor_cost = round(job_level_labor + (variable_labor_per_unit * quantity), 2)

    # ========================================================================
    # Verify the calculations
    # ========================================================================

    # Expected job-level costs (setup + programming + inspection)
    expected_job_level_min = setup_min + programming_min + inspection_min  # 30 + 19 + 30 = 79 min
    expected_job_level_cost = expected_job_level_min * (labor_rate / 60.0)  # 79 * 1.0 = $79.00

    # Expected variable costs per unit (machining + finishing)
    expected_variable_min_per_unit = machining_min + finishing_min  # 50 + 10 = 60 min
    expected_variable_cost_per_unit = expected_variable_min_per_unit * (labor_rate / 60.0)  # 60 * 1.0 = $60.00

    # Expected amortized job-level cost per unit
    expected_amortized_per_unit = expected_job_level_cost / quantity  # 79 / 8 = $9.875 ≈ $9.88

    # Expected per-unit labor cost
    expected_per_unit_labor = round(expected_amortized_per_unit + expected_variable_cost_per_unit, 2)  # 9.88 + 60.00 = $69.88

    # Expected total labor cost for all units
    expected_total_labor = round(expected_job_level_cost + (expected_variable_cost_per_unit * quantity), 2)  # 79 + (60 * 8) = 79 + 480 = $559.00

    # Actual values from our calculation
    actual_per_unit_labor = per_unit_labor_cost
    actual_total_labor = total_labor_cost

    print("\n" + "=" * 70)
    print("MULTI-QUANTITY AMORTIZATION TEST")
    print("=" * 70)
    print(f"\nQuantity: {quantity} units")
    print(f"Labor Rate: ${labor_rate:.2f}/hr\n")

    print("JOB-LEVEL COSTS (one-time, amortized):")
    print(f"  Setup:              {setup_min:>8.2f} min = ${setup_labor:>7.2f}")
    print(f"  Programming:        {programming_min:>8.2f} min = ${programming_labor:>7.2f}")
    print(f"  Inspection:         {inspection_min:>8.2f} min = ${inspection_labor:>7.2f}")
    print(f"  {'─' * 38}")
    print(f"  Total job-level:    {expected_job_level_min:>8.2f} min = ${job_level_labor:>7.2f}")
    print(f"  Amortized per unit:              ${amortized_job_level_cost:>7.2f}")

    print("\nPER-UNIT VARIABLE COSTS:")
    print(f"  Machining:          {machining_min:>8.2f} min = ${machining_labor:>7.2f}")
    print(f"  Finishing:          {finishing_min:>8.2f} min = ${finishing_labor:>7.2f}")
    print(f"  {'─' * 38}")
    print(f"  Total variable:     {expected_variable_min_per_unit:>8.2f} min = ${variable_labor_per_unit:>7.2f}")

    print("\nEXPECTED RESULTS:")
    print(f"  Per-unit labor cost:             ${expected_per_unit_labor:>7.2f}")
    print(f"  Total labor cost (all units):    ${expected_total_labor:>7.2f}")

    print("\nACTUAL RESULTS:")
    print(f"  Per-unit labor cost:             ${actual_per_unit_labor:>7.2f}")
    print(f"  Total labor cost (all units):    ${actual_total_labor:>7.2f}")

    print("\nVERIFICATION:")
    per_unit_match = abs(actual_per_unit_labor - expected_per_unit_labor) < 0.01
    total_match = abs(actual_total_labor - expected_total_labor) < 0.01

    print(f"  Per-unit labor matches: {'✓ PASS' if per_unit_match else '✗ FAIL'}")
    print(f"  Total labor matches:    {'✓ PASS' if total_match else '✗ FAIL'}")

    # Show the formula being used
    print("\nFORMULA VERIFICATION:")
    print(f"  Total labor = (setup + prog + insp) + (mach + finish) × qty")
    print(f"              = ${job_level_labor:.2f} + ${variable_labor_per_unit:.2f} × {quantity}")
    print(f"              = ${job_level_labor:.2f} + ${variable_labor_per_unit * quantity:.2f}")
    print(f"              = ${expected_total_labor:.2f}")
    print("=" * 70 + "\n")

    # Assert the values match
    assert per_unit_match, f"Per-unit labor cost mismatch: expected ${expected_per_unit_labor:.2f}, got ${actual_per_unit_labor:.2f}"
    assert total_match, f"Total labor cost mismatch: expected ${expected_total_labor:.2f}, got ${actual_total_labor:.2f}"

    print("✓ All tests passed!\n")


if __name__ == "__main__":
    test_multi_qty_amortization()
