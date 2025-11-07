"""Test script to verify the JSON caching mechanism works."""
from pathlib import Path
import json

# Simulate what AppV7 does
cad_file_path = r"Cad Files\301_redacted.dwg"
json_output = Path(__file__).parent / "debug" / f"{Path(cad_file_path).stem}_dims.json"

print(f"Looking for cached JSON at: {json_output}")
print(f"File exists: {json_output.exists()}")

if json_output.exists():
    print(f"\n[OK] Found existing dims JSON: {json_output}")
    try:
        with open(json_output, 'r') as f:
            dims_data = json.load(f)

        length = dims_data.get('length', 0.0)
        width = dims_data.get('width', 0.0)
        thickness = dims_data.get('thickness', 0.0)

        print(f"\n[OK] Successfully loaded dimensions from JSON:")
        print(f"  Length:    {length} in")
        print(f"  Width:     {width} in")
        print(f"  Thickness: {thickness} in")

        if length > 0 and width > 0 and thickness > 0:
            print(f"\n[OK] All dimensions are valid (non-zero)")
            print(f"\n[SUCCESS] JSON caching mechanism is ready to use!")
        else:
            print(f"\n[ERROR] Some dimensions are zero")
    except Exception as e:
        print(f"\n[ERROR] Failed to load JSON: {e}")
else:
    print(f"\n[ERROR] Cached JSON file not found")
    print(f"Expected location: {json_output.absolute()}")
