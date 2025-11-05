"""
dump_all_text.py
================
Simple script to dump all text from a CAD file to a text file.

Usage:
    python dump_all_text.py "path/to/file.dxf"
    python dump_all_text.py "path/to/file.dxf" "output.txt"
"""

import sys
from pathlib import Path
from cad_quoter.geo_dump import extract_all_text_from_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python dump_all_text.py <input_file.dxf> [output_file.txt]")
        print("\nExample:")
        print('  python dump_all_text.py "Cad Files/301.dxf"')
        print('  python dump_all_text.py "Cad Files/301.dxf" "output.txt"')
        return 1

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return 1

    # Determine output file
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    else:
        output_file = Path("debug") / f"{input_file.stem}_all_text.txt"
        output_file.parent.mkdir(exist_ok=True)

    print(f"Extracting text from: {input_file}")

    try:
        # Extract all text
        all_text = extract_all_text_from_file(input_file)

        # Save to file (one line per text entity)
        with open(output_file, "w", encoding="utf-8") as f:
            for record in all_text:
                f.write(f"{record['text']}\n")

        print(f"✓ Saved {len(all_text)} text records to: {output_file}")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
