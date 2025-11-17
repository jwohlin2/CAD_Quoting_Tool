"""Test script to extract text from a DXF file."""

import sys
from pathlib import Path

# Add project root to Python path so imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dxf_text import extract_text_lines_from_dxf

# Extract text from a DXF file
dxf_path = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dxf"
print(f"Extracting text from: {dxf_path}\n")

text_lines = extract_text_lines_from_dxf(dxf_path)

print(f"Found {len(text_lines)} text lines:\n")
print("=" * 60)
for line in text_lines:
    print(line)
print("=" * 60)
