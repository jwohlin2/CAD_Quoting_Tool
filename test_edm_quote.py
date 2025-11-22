"""
Test script to verify EDM time estimation for T1769-143_redacted.dwg
Run this to check:
1. EDM minutes > 0.00 for carbide die sections
2. profile_process decision in plan metadata
"""
import sys
import io
from pathlib import Path
from pprint import pprint

# Set UTF-8 encoding for stdout to handle Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import the quote generation logic
from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad

def test_edm_quote():
    """Run quote for test file and verify EDM timing."""
    test_file = Path("Cad Files/T1769-143_redacted.dwg")

    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return False

    print(f"Testing quote for: {test_file.name}")
    print("=" * 60)

    # Generate the quote data (verbose=True to get raw_plan)
    try:
        quote_data = extract_quote_data_from_cad(str(test_file), verbose=True)
    except Exception as e:
        print(f"ERROR generating quote: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get the raw plan data
    plan = quote_data.raw_plan if quote_data.raw_plan else {}

    # Check metadata for profile_process decision
    print("\n1. Checking plan metadata for profile_process...")
    if "meta" in plan and "profile_process" in plan["meta"]:
        profile_process = plan["meta"]["profile_process"]
        print(f"   [OK] profile_process: {profile_process}")
    else:
        print("   [FAIL] profile_process NOT FOUND in plan metadata")
        if "meta" in plan:
            print(f"   Available metadata keys: {list(plan['meta'].keys())}")
        else:
            print("   No metadata found in plan")

    # Check machine hours breakdown for EDM operations
    print("\n2. Checking for EDM operations with time > 0...")
    found_edm = False

    # Check the machine hours breakdown
    if quote_data.machine_hours and hasattr(quote_data.machine_hours, 'edm_operations'):
        edm_ops = quote_data.machine_hours.edm_operations
        if edm_ops:
            for op in edm_ops:
                print(f"   Found EDM operation: {op.operation}")
                print(f"      Minutes: {op.minutes}")
                if op.minutes > 0:
                    print(f"      [OK] Time is > 0.00")
                    found_edm = True
                else:
                    print(f"      [FAIL] WARNING: Time is 0.00")
        else:
            print("   No EDM operations found in machine hours")

    # Also check the raw plan operations
    if "ops" in plan:
        for op in plan["ops"]:
            op_name = op.get("op", "")
            # Try both 'minutes' and 'time_minutes' keys
            minutes = op.get("minutes", op.get("time_minutes", 0.0))

            # Look for EDM-related operations
            if "edm" in op_name.lower() or "wire_edm" in op_name.lower() or "form_grind" in op_name.lower():
                print(f"   Found in plan: {op_name}")
                print(f"      Minutes: {minutes}")
                if minutes > 0:
                    print(f"      [OK] Time is > 0.00")
                    found_edm = True
                else:
                    print(f"      [FAIL] WARNING: Time is 0.00")

    if not found_edm:
        print("   [FAIL] No EDM operations with time > 0 found")

    # Print full plan for debugging
    print("\n" + "=" * 60)
    print("RAW PLAN:")
    print("=" * 60)
    if plan:
        pprint(plan)
    else:
        print("No raw plan data available")

    # Print machine hours info
    print("\n" + "=" * 60)
    print("MACHINE HOURS BREAKDOWN:")
    print("=" * 60)
    if quote_data.machine_hours:
        print(f"EDM operations count: {len(quote_data.machine_hours.edm_operations) if quote_data.machine_hours.edm_operations else 0}")
        if quote_data.machine_hours.edm_operations:
            for op in quote_data.machine_hours.edm_operations:
                print(f"  - {op.operation}: {op.minutes} min")
        # Print grinding and milling info
        if quote_data.machine_hours.grinding_operations:
            print(f"\nGrinding operations:")
            for op in quote_data.machine_hours.grinding_operations:
                print(f"  - {op.operation}: {op.minutes} min")

    return found_edm

if __name__ == "__main__":
    success = test_edm_quote()
    sys.exit(0 if success else 1)
