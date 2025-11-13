"""Test material detection with GENERIC default."""

from pathlib import Path
from cad_quoter.pricing.KeywordDetector import detect_material_in_cad

cad_file = Path("Cad Files/301_redacted.dxf")

print("=" * 70)
print("MATERIAL DETECTION WITH DEFAULT TEST")
print("=" * 70)

# Test: Detect material, defaulting to GENERIC if not found
print(f"\nDetecting material in: {cad_file.name}")
print("-" * 70)

material = detect_material_in_cad(cad_file)

print(f"\nDetected Material: {material}")

if material == "GENERIC":
    print("  (No specific material found in CAD file, using default)")
else:
    print(f"  (Found material specification in CAD file)")

print("\n" + "=" * 70)
print("USAGE EXAMPLE")
print("=" * 70)
print("""
# Quick material detection with default:
from cad_quoter.pricing.KeywordDetector import detect_material_in_cad

material = detect_material_in_cad("part.dxf")
# Returns canonical material name or "GENERIC" if not found

# Custom default material:
material = detect_material_in_cad("part.dxf", default_material="Low Carbon Steel")
""")
