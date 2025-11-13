"""Test material detection from material_map.csv in CAD file text."""

from pathlib import Path
from cad_quoter.pricing.KeywordDetector import KeywordDetector
from cad_quoter.planning import extract_all_text_from_cad

cad_file = Path("Cad Files/301_redacted.dxf")
material_csv = Path("cad_quoter/pricing/resources/material_map.csv")

print("=" * 70)
print("MATERIAL DETECTION TEST")
print("=" * 70)

# Step 1: Load materials from CSV
print("\n1. Loading material keywords from material_map.csv:")
print("-" * 70)
detector = KeywordDetector()
detector.load_materials_from_csv(material_csv)
print(f"Loaded {len(detector.keyword_categories['MATERIAL'])} material keywords")
print(f"Examples: {', '.join(detector.keyword_categories['MATERIAL'][:5])}")

# Step 2: Extract all text from CAD file
print("\n2. Extracting text from CAD file:")
print("-" * 70)
all_text = extract_all_text_from_cad(cad_file)
print(f"Found {len(all_text)} text entries")

# Step 3: Run material detection
print("\n3. Searching for material keywords in CAD text:")
print("-" * 70)
result = detector.detect_from_cad_file(cad_file)

# Filter to only material matches
material_matches = result.get_matches_by_category("MATERIAL")

if material_matches:
    print(f"Found {len(material_matches)} material keyword matches:\n")

    # Group by canonical material
    materials_found = {}
    for match in material_matches:
        canonical = match.canonical_material
        if canonical not in materials_found:
            materials_found[canonical] = []
        materials_found[canonical].append(match)

    print("Materials detected (grouped by canonical name):")
    print("-" * 70)
    for canonical, matches in materials_found.items():
        print(f"\n{canonical}:")
        for match in matches:
            context_safe = match.context.encode('ascii', 'replace').decode('ascii')
            print(f"  - Found '{match.keyword}' in: \"{context_safe}\"")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Detected {len(materials_found)} unique material(s)")
    print("=" * 70)
    for canonical in sorted(materials_found.keys()):
        print(f"  * {canonical}")
else:
    print("No material keywords found in CAD file")
    print("\nShowing first 20 text entries for debugging:")
    for i, text in enumerate(all_text[:20], 1):
        text_safe = text.encode('ascii', 'replace').decode('ascii')
        print(f"  {i}. {text_safe}")

print("\n" + "=" * 70)
