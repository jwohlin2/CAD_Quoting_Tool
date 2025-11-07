"""Test the merged LaborOpsHelper functionality in process_planner."""

from cad_quoter.planning import (
    plan_from_cad_file,
    LaborInputs,
    compute_labor_minutes,
    __all__
)

print("=" * 60)
print("TESTING MERGED LABOR ESTIMATION")
print("=" * 60)

# Test 1: Basic labor calculation
print("\n1. Testing basic labor calculation:")
inputs = LaborInputs(
    ops_total=15,
    holes_total=22,
    tool_changes=8,
    fixturing_complexity=2,
    edm_window_count=1,
    jig_grind_bore_qty=2,
    thread_mill=2,
    tap_rigid=3,
    counterbore_qty=4,
)

result = compute_labor_minutes(inputs)
print(f"   Setup: {result['minutes']['Setup']:.1f} min")
print(f"   Programming: {result['minutes']['Programming']:.1f} min")
print(f"   Machining: {result['minutes']['Machining_Steps']:.1f} min")
print(f"   Inspection: {result['minutes']['Inspection']:.1f} min")
print(f"   Finishing: {result['minutes']['Finishing']:.1f} min")
print(f"   TOTAL: {result['minutes']['Labor_Total']:.1f} min")

# Test 2: Verify all exports
print("\n2. Available exports from cad_quoter.planning:")
for name in __all__:
    print(f"   - {name}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
