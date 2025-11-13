"""Test the extract_dimensions_from_cad function that AppV7 uses"""

import sys
from pathlib import Path

# Test the actual function AppV7 calls
from cad_quoter.planning.process_planner import extract_dimensions_from_cad

# Test file
test_file = Path("Cad Files/301_redacted.dwg")

if not test_file.exists():
    print(f"Test file not found: {test_file}")
    sys.exit(1)

print(f"Testing extract_dimensions_from_cad() with: {test_file}")
print("=" * 70)

result = extract_dimensions_from_cad(test_file)

print("=" * 70)
if result:
    length, width, thickness = result
    print(f"✓ SUCCESS!")
    print(f"  Length:    {length} in")
    print(f"  Width:     {width} in")
    print(f"  Thickness: {thickness} in")
else:
    print(f"✗ FAILED - returned None")
    print(f"\nThis function has verbose=False, so we can't see details.")
    print(f"The function is in: cad_quoter/planning/process_planner.py:527")
