"""
extract_text_demo.py
=====================
Simple script to extract all text from a CAD file.

Usage:
    python extract_text_demo.py "path/to/file.dxf"
"""

import sys
from pathlib import Path
from cad_quoter.geo_dump import extract_all_text_from_file


def main():
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Try some common test files
        test_files = [
            Path("Cad Files/301_redacted.dxf"),
            Path("Cad Files/301.dxf"),
            Path("debug/301.dxf"),
        ]
        file_path = None
        for f in test_files:
            if f.exists():
                file_path = f
                break

        if file_path is None:
            print("Usage: python extract_text_demo.py <path_to_cad_file.dxf>")
            print("\nExample:")
            print('  python extract_text_demo.py "Cad Files/301.dxf"')
            return 1

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    print(f"Extracting text from: {file_path}")
    print("=" * 70)

    # Extract all text
    try:
        all_text = extract_all_text_from_file(file_path)
    except Exception as e:
        print(f"Error extracting text: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\nFound {len(all_text)} text records\n")

    # Display results in organized way
    print("Layout Distribution:")
    layouts = {}
    for record in all_text:
        layout = record["layout"]
        layouts[layout] = layouts.get(layout, 0) + 1
    for layout, count in sorted(layouts.items()):
        print(f"  {layout}: {count} records")

    print("\n" + "=" * 70)
    print("First 20 text records:")
    print("=" * 70)

    for i, record in enumerate(all_text[:20], 1):
        text = record["text"]
        # Truncate long text
        if len(text) > 60:
            text = text[:57] + "..."

        print(f"{i:3d}. {text}")
        print(f"      Layout: {record['layout']:15s} Layer: {record['layer']:15s} Type: {record['etype']}")

        # Show position if available
        if record["x"] or record["y"]:
            print(f"      Position: ({record['x']:.2f}, {record['y']:.2f})")

        print()

    # Save full dump to file
    output_file = Path("debug") / f"{file_path.stem}_all_text.txt"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Text extraction from: {file_path}\n")
        f.write(f"Total records: {len(all_text)}\n")
        f.write("=" * 70 + "\n\n")

        for i, record in enumerate(all_text, 1):
            f.write(f"Record {i}\n")
            f.write(f"  Text: {record['text']}\n")
            f.write(f"  Layout: {record['layout']}\n")
            f.write(f"  Layer: {record['layer']}\n")
            f.write(f"  Type: {record['etype']}\n")
            f.write(f"  Position: ({record['x']:.2f}, {record['y']:.2f})\n")
            f.write(f"  Height: {record['height']:.2f}\n")
            f.write(f"  In Block: {record['in_block']}\n")
            if record['block_path']:
                f.write(f"  Block Path: {' > '.join(record['block_path'])}\n")
            f.write("\n")

    print("=" * 70)
    print(f"Full text dump saved to: {output_file}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
