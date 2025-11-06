"""Dump all text from CAD file to see what's available."""

from pathlib import Path
from cad_quoter.planning import extract_all_text_from_cad

cad_file = Path("Cad Files/301_redacted.dxf")

print("=" * 70)
print(f"ALL TEXT FROM: {cad_file.name}")
print("=" * 70)

all_text = extract_all_text_from_cad(cad_file)
print(f"\nTotal entries: {len(all_text)}\n")

for i, text in enumerate(all_text, 1):
    # Handle Unicode characters
    text_safe = text.encode('ascii', 'replace').decode('ascii')
    print(f"{i:3d}. {text_safe}")

print("\n" + "=" * 70)
