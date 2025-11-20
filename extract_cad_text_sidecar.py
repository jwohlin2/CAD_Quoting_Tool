#!/usr/bin/env python3
"""
Standalone CAD Text Extraction Sidecar
======================================

Simple standalone script for extracting all text from DWG/DXF files.
Used for testing and debugging text extraction.

Usage:
    python extract_cad_text_sidecar.py [path_to_cad_file.dxf|dwg]

    # If no file path is provided, you will be prompted interactively
    python extract_cad_text_sidecar.py

    # Examples with file path:
    python extract_cad_text_sidecar.py "Cad Files/301_redacted.dxf"
    python extract_cad_text_sidecar.py test.dwg --format csv
    python extract_cad_text_sidecar.py test.dxf --text-only

Default behavior:
    - Automatically saves output as JSON file alongside the CAD file
    - Output filename: {input_name}_text_extraction.json
    - Use --output to specify a different location
    - Use --format to change output format (human, json, csv)
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cad_quoter import geo_extractor


def extract_all_text(cad_file_path: str, max_block_depth: int = 5):
    """
    Extract all text from a CAD file using the same method as the main application.

    Args:
        cad_file_path: Path to DXF or DWG file
        max_block_depth: How deep to recurse into blocks (default: 5)

    Returns:
        List of dictionaries containing text records with metadata
    """
    file_path = Path(cad_file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Open the CAD file (handles both DXF and DWG)
    print(f"Opening: {file_path}", file=sys.stderr)
    doc = geo_extractor.open_doc(file_path)

    # Extract all text from all layouts
    print(f"Extracting text (block depth={max_block_depth})...", file=sys.stderr)
    text_records = geo_extractor.collect_all_text(
        doc,
        layouts=None,  # All layouts
        include_layers=None,  # All layers
        exclude_layers=None,
        max_block_depth=max_block_depth,
        min_height=0.0,  # All text sizes
    )

    # Decode unicode characters (e.g., \U+2205 → Ø)
    for record in text_records:
        if "text" in record and isinstance(record["text"], str):
            # Decode special characters
            text = record["text"]
            # Handle unicode escape sequences
            import re
            def decode_uplus(s: str) -> str:
                pattern = re.compile(r"\\U\+([0-9A-Fa-f]{4})")
                return pattern.sub(lambda m: chr(int(m.group(1), 16)), s or "")
            record["text"] = decode_uplus(text)

    print(f"Found {len(text_records)} text records", file=sys.stderr)
    return text_records


def format_human_readable(records):
    """Format text records in a human-readable way."""
    output = []
    output.append("=" * 80)
    output.append(f"CAD TEXT EXTRACTION - {len(records)} records found")
    output.append("=" * 80)
    output.append("")

    # Group by layout
    by_layout = {}
    for r in records:
        layout = r.get("layout", "Unknown")
        if layout not in by_layout:
            by_layout[layout] = []
        by_layout[layout].append(r)

    for layout_name, layout_records in by_layout.items():
        output.append(f"\n{'─' * 80}")
        output.append(f"Layout: {layout_name} ({len(layout_records)} records)")
        output.append('─' * 80)

        for i, r in enumerate(layout_records, 1):
            text = r.get("text", "")
            layer = r.get("layer", "")
            etype = r.get("etype", "")
            x = r.get("x", 0.0)
            y = r.get("y", 0.0)
            in_block = r.get("in_block", False)
            depth = r.get("depth", 0)

            output.append(f"\n[{i}] {etype} on layer '{layer}'")
            if in_block:
                block_path = r.get("block_path", [])
                output.append(f"    Block: {' -> '.join(block_path)} (depth {depth})")
            output.append(f"    Position: ({x:.2f}, {y:.2f})")
            output.append(f"    Text: {text}")

    output.append("\n" + "=" * 80)
    return "\n".join(output)


def format_text_only(records):
    """Return just the text strings, one per line."""
    return "\n".join(r.get("text", "") for r in records if r.get("text", "").strip())


def format_csv(records):
    """Format as CSV."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["layout", "layer", "etype", "text", "x", "y", "height", "rotation", "in_block", "depth"],
        extrasaction='ignore'
    )
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Extract all text from a CAD file (DXF or DWG) and save as JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive mode - prompts for file
  %(prog)s drawing.dxf                   # Saves as drawing_text_extraction.json
  %(prog)s drawing.dwg -o output.json    # Custom output location
  %(prog)s drawing.dxf --format csv      # Save as CSV instead
  %(prog)s drawing.dxf --block-depth 10  # Increase block recursion depth
        """
    )

    parser.add_argument(
        "cad_file",
        nargs="?",  # Make it optional
        help="Path to CAD file (DXF or DWG)"
    )

    parser.add_argument(
        "--format",
        choices=["human", "json", "csv"],
        default="human",
        help="Output format (default: json when auto-saving, human otherwise)"
    )

    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Output only text strings, one per line (no metadata)"
    )

    parser.add_argument(
        "--block-depth",
        type=int,
        default=5,
        help="Maximum block nesting depth to explore (default: 5)"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: auto-generated as {input_name}_text_extraction.json)"
    )

    args = parser.parse_args()

    # If no file was provided, prompt the user
    if not args.cad_file:
        print("CAD Text Extraction Tool", file=sys.stderr)
        print("=" * 40, file=sys.stderr)
        cad_file = input("Enter path to CAD file (DXF or DWG): ").strip()

        # Remove surrounding quotes if present
        if cad_file.startswith('"') and cad_file.endswith('"'):
            cad_file = cad_file[1:-1]
        elif cad_file.startswith("'") and cad_file.endswith("'"):
            cad_file = cad_file[1:-1]

        if not cad_file:
            print("ERROR: No file path provided", file=sys.stderr)
            return 1

        args.cad_file = cad_file

    try:
        # Extract text
        records = extract_all_text(args.cad_file, max_block_depth=args.block_depth)

        # Format output
        if args.text_only:
            output = format_text_only(records)
        elif args.format == "json":
            output = json.dumps(records, indent=2, ensure_ascii=False)
        elif args.format == "csv":
            output = format_csv(records)
        else:  # human
            output = format_human_readable(records)

        # Auto-generate output filename if not specified
        if not args.output:
            input_path = Path(args.cad_file)
            # Default to JSON format and save alongside the CAD file
            output_filename = f"{input_path.stem}_text_extraction.json"
            args.output = str(input_path.parent / output_filename)
            # Override format to JSON when auto-generating filename
            if not args.text_only:
                output = json.dumps(records, indent=2, ensure_ascii=False)

        # Write output
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\n✓ Extraction complete!", file=sys.stderr)
        print(f"✓ Output written to: {args.output}", file=sys.stderr)
        print(f"✓ Found {len(records)} text records", file=sys.stderr)

        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
