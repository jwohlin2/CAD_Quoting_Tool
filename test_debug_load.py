"""Debug the load function step by step."""

import csv
import os
from cad_quoter.resources import default_catalog_csv

def debug_load_mcmaster_catalog_rows(path=None):
    """Debug version of load_mcmaster_catalog_rows."""
    print(f"Input path: {path}")
    print(f"Input path type: {type(path)}")

    csv_path = path or os.getenv("CATALOG_CSV_PATH") or str(default_catalog_csv())
    print(f"Resolved csv_path: {csv_path}")
    print(f"Resolved csv_path type: {type(csv_path)}")

    if not csv_path:
        print("csv_path is empty/None - returning []")
        return []

    print(f"File exists: {os.path.exists(csv_path)}")

    try:
        print(f"Attempting to open: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            print("File opened successfully")
            reader = csv.DictReader(handle)
            print(f"DictReader created")
            rows = [dict(row) for row in reader if row]
            print(f"Read {len(rows)} rows")
            return rows
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return []
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return []

print("=" * 60)
print("Testing with path=None (default)")
print("=" * 60)
rows = debug_load_mcmaster_catalog_rows(None)
print(f"Result: {len(rows)} rows\n")

print("=" * 60)
print("Testing with explicit path")
print("=" * 60)
rows = debug_load_mcmaster_catalog_rows(r"D:\CAD_Quoting_Tool\cad_quoter\resources\catalog.csv")
print(f"Result: {len(rows)} rows")
