"""Debug CSV loading."""

import csv
import os

csv_path = r"D:\CAD_Quoting_Tool\cad_quoter\resources\catalog.csv"

print(f"Testing CSV load from: {csv_path}")
print(f"File exists: {os.path.exists(csv_path)}")
print(f"File size: {os.path.getsize(csv_path)} bytes")

# Test direct CSV reading
print("\n--- Direct CSV reading ---")
try:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader if row]
        print(f"Loaded {len(rows)} rows")
        if rows:
            print(f"First row: {rows[0]}")
            print(f"Keys: {list(rows[0].keys())}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test using the actual function
print("\n--- Using load_mcmaster_catalog_rows ---")
try:
    from cad_quoter.pricing.mcmaster_helpers import load_mcmaster_catalog_rows
    rows = load_mcmaster_catalog_rows(csv_path)
    print(f"Loaded {len(rows)} rows")
    if rows:
        print(f"First row: {rows[0]}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test using default path
print("\n--- Using load_mcmaster_catalog_rows with default path ---")
try:
    from cad_quoter.pricing.mcmaster_helpers import load_mcmaster_catalog_rows
    rows = load_mcmaster_catalog_rows()
    print(f"Loaded {len(rows)} rows")
    if rows:
        print(f"First row: {rows[0]}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
