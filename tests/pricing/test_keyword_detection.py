"""Test keyword detection infrastructure with CAD file."""

from pathlib import Path
from cad_quoter.pricing.KeywordDetector import (
    KeywordDetector,
    detect_keywords_in_cad,
)

cad_file = Path("Cad Files/301_redacted.dxf")

print("=" * 70)
print("KEYWORD DETECTION TEST")
print("=" * 70)

# Test 1: Extract all text first to see what we're working with
print("\n1. Extracting all text from CAD file:")
print("-" * 70)
from cad_quoter.planning import extract_all_text_from_cad
all_text = extract_all_text_from_cad(cad_file)
print(f"Found {len(all_text)} text entries in CAD file")
print("\nFirst 10 text entries:")
for i, text in enumerate(all_text[:10], 1):
    print(f"  {i}. {text}")

# Test 2: Set up keyword detector with empty categories (ready for keywords to be added later)
print("\n\n2. Setting up keyword detector with category infrastructure:")
print("-" * 70)
detector = KeywordDetector()
print("Default categories configured:")
for category in detector.keyword_categories.keys():
    print(f"  - {category}")

# Test 3: Add some example keywords to demonstrate functionality
print("\n\n3. Adding example keywords to demonstrate infrastructure:")
print("-" * 70)

# Example keywords - these can be modified/expanded later
example_keywords = {
    "FINISH": ["POLISH", "GRIND", "BEAD BLAST", "SURFACE FINISH"],
    "HEAT_TREAT": ["HARDEN", "TEMPER", "ANNEAL", "HEAT TREAT"],
    "COATING": ["PLATE", "COAT", "ANODIZE", "BLACK OXIDE"],
    "TOLERANCE": ["±", "TOLERANCE", "+/-"],
    "MATERIAL": ["STAINLESS", "ALUMINUM", "STEEL", "CARBIDE"],
    "INSPECTION": ["INSPECT", "CMM", "MEASURE"],
    "SPECIAL_PROCESS": ["JIG GRIND", "EDM", "WIRE EDM"],
}

for category, keywords in example_keywords.items():
    detector.add_keywords(category, keywords)
    print(f"{category}: {', '.join(keywords)}")

# Test 4: Run keyword detection
print("\n\n4. Running keyword detection on CAD file:")
print("-" * 70)
result = detector.detect_from_cad_file(cad_file)

if result.matches:
    print(f"Found {len(result.matches)} keyword matches:\n")

    # Group matches by category
    summary = result.summary()
    for category, keywords_found in summary.items():
        print(f"\n{category}:")
        matches = result.get_matches_by_category(category)
        for match in matches:
            # Handle Unicode characters in context
            context_safe = match.context.encode('ascii', 'replace').decode('ascii')
            print(f"  [X] '{match.keyword}' found in: \"{context_safe}\"")

    # Show which categories were detected
    print(f"\n\nCategories detected: {', '.join(sorted(result.categories_found))}")
else:
    print("No keyword matches found with current keyword list")

# Test 5: Demonstrate quick detection function
print("\n\n5. Testing quick detection function:")
print("-" * 70)
quick_keywords = {
    "MATERIAL": ["STAINLESS", "ALUMINUM"],
    "SPECIAL_PROCESS": ["JIG GRIND"],
}
quick_result = detect_keywords_in_cad(cad_file, quick_keywords)
print(f"Quick detection found {len(quick_result.matches)} matches")
if quick_result.matches:
    for match in quick_result.matches:
        print(f"  - {match.category}: {match.keyword}")

print("\n" + "=" * 70)
print("KEYWORD DETECTION INFRASTRUCTURE READY")
print("=" * 70)
print("\nTo add keywords later, use:")
print("  detector.add_keywords('CATEGORY_NAME', ['keyword1', 'keyword2'])")
print("\nExample:")
print("  detector.add_keywords('FINISH', ['MIRROR FINISH', 'RA 32'])")
