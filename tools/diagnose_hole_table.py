#!/usr/bin/env python3
"""
diagnose_hole_table.py
======================
Diagnostic tool to investigate why a hole table isn't being detected.

This tool extracts all text from a DWG/DXF file and analyzes it to find
potential hole table content that might not be getting picked up.

Usage:
    python -m tools.diagnose_hole_table path/to/file.dwg
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Add parent directory to path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cad_quoter import geo_extractor
from cad_quoter.geo_dump import _find_hole_table_chunks, _decode_uplus


def diagnose_hole_table(file_path: str, output_dir: str = None) -> dict:
    """
    Run comprehensive diagnostics on hole table detection.

    Args:
        file_path: Path to DWG/DXF file
        output_dir: Optional directory for output files

    Returns:
        Dictionary with diagnostic results
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    results = {
        "file": str(path),
        "total_text_records": 0,
        "entity_type_counts": {},
        "hole_table_detected": False,
        "potential_hole_content": [],
        "all_text_samples": [],
    }

    # Open the document
    try:
        doc = geo_extractor.open_doc(path)
    except Exception as e:
        return {"error": f"Failed to open file: {e}"}

    # Collect all text with deep block exploration
    text_records = geo_extractor.collect_all_text(doc, max_block_depth=8)

    # Decode unicode characters
    for r in text_records:
        if "text" in r and isinstance(r["text"], str):
            r["text"] = _decode_uplus(r["text"])

    results["total_text_records"] = len(text_records)

    # Count entity types
    for r in text_records:
        etype = r.get("etype", "UNKNOWN")
        results["entity_type_counts"][etype] = results["entity_type_counts"].get(etype, 0) + 1

    # Check if standard detection works
    header_chunks, body_chunks = _find_hole_table_chunks(text_records)
    results["hole_table_detected"] = len(header_chunks) > 0

    if header_chunks:
        results["detected_header_chunks"] = header_chunks[:5]  # First 5
        results["detected_body_chunks_count"] = len(body_chunks)

    # Search for potential hole-related content
    hole_keywords = [
        "HOLE", "TABLE", "REF", "QTY", "DESCRIPTION", "DESC",
        "THRU", "TAP", "DRILL", "C'BORE", "CBORE", "DEEP",
        "JIG GRIND", "FROM FRONT", "FROM BACK"
    ]

    for i, r in enumerate(text_records):
        text = r.get("text", "")
        if not text:
            continue

        text_upper = text.upper()

        # Check for any hole-related keywords
        matches = [kw for kw in hole_keywords if kw in text_upper]

        if matches:
            results["potential_hole_content"].append({
                "index": i,
                "etype": r.get("etype"),
                "layer": r.get("layer"),
                "text": text[:500] if len(text) > 500 else text,
                "keywords_found": matches,
                "in_block": r.get("in_block"),
                "block_path": r.get("block_path", []),
            })

        # Also look for diameter symbols
        if any(sym in text for sym in ["Ø", "∅", "⌀"]):
            if not any(kw in text_upper for kw in hole_keywords):  # Don't duplicate
                results["potential_hole_content"].append({
                    "index": i,
                    "etype": r.get("etype"),
                    "layer": r.get("layer"),
                    "text": text[:500] if len(text) > 500 else text,
                    "keywords_found": ["DIAMETER_SYMBOL"],
                    "in_block": r.get("in_block"),
                    "block_path": r.get("block_path", []),
                })

    # Collect samples of all text for manual review
    for r in text_records[:100]:  # First 100
        text = r.get("text", "")
        if text:
            results["all_text_samples"].append({
                "etype": r.get("etype"),
                "layer": r.get("layer"),
                "text": text[:200] if len(text) > 200 else text,
            })

    # Analysis summary
    results["analysis"] = analyze_detection_issues(results, text_records)

    # Output to file if requested
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Write full diagnostic report
        report_path = out_path / f"{path.stem}_hole_diagnostic.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Write all text records for manual inspection
        all_text_path = out_path / f"{path.stem}_all_text.jsonl"
        with all_text_path.open("w", encoding="utf-8") as f:
            for r in text_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        results["output_files"] = {
            "diagnostic_report": str(report_path),
            "all_text": str(all_text_path),
        }

    return results


def analyze_detection_issues(results: dict, text_records: list) -> list:
    """Analyze potential reasons for detection failure."""

    issues = []

    if results["hole_table_detected"]:
        issues.append("✓ Hole table WAS detected by standard detection")
        return issues

    issues.append("✗ Hole table was NOT detected by standard detection")

    # Check for "HOLE TABLE" text in any form
    hole_table_variants = []
    for r in text_records:
        text = r.get("text", "").upper()
        if "HOLE" in text and "TABLE" in text:
            hole_table_variants.append({
                "text": r.get("text"),
                "etype": r.get("etype"),
            })

    if hole_table_variants:
        issues.append(f"Found {len(hole_table_variants)} text(s) with both 'HOLE' and 'TABLE':")
        for v in hole_table_variants[:3]:
            etype = v["etype"]
            # Check if entity type is in the detection list
            if etype not in ("PROXYTEXT", "MTEXT", "TEXT"):
                issues.append(f"  - Entity type '{etype}' is NOT in detection list!")
                issues.append(f"    Text: {v['text'][:100]}...")
            else:
                # Check for exact "HOLE TABLE" match
                if "HOLE TABLE" not in v["text"].upper():
                    issues.append(f"  - 'HOLE TABLE' not as exact phrase in: {v['text'][:100]}...")
                else:
                    issues.append(f"  - Should have been detected: {v['text'][:100]}...")
    else:
        issues.append("No text found containing both 'HOLE' and 'TABLE'")

        # Check if we have any hole-related content at all
        if results["potential_hole_content"]:
            issues.append(f"Found {len(results['potential_hole_content'])} potential hole-related texts")

            # Check entity types
            etypes_found = set(p["etype"] for p in results["potential_hole_content"])
            non_standard = etypes_found - {"PROXYTEXT", "MTEXT", "TEXT"}
            if non_standard:
                issues.append(f"  - Some hole content is in non-standard entity types: {non_standard}")
        else:
            issues.append("No hole-related content found at all - file may not contain hole table")

    # Check for TABLE entity type
    tablecell_count = results["entity_type_counts"].get("TABLECELL", 0)
    if tablecell_count > 0:
        issues.append(f"Found {tablecell_count} TABLECELL entities - hole table might be in native TABLE format")
        issues.append("  Current detection doesn't check TABLECELL entities!")

    return issues


def print_diagnostic_report(results: dict):
    """Print a human-readable diagnostic report."""

    print("\n" + "=" * 70)
    print("HOLE TABLE DIAGNOSTIC REPORT")
    print("=" * 70)

    print(f"\nFile: {results.get('file', 'N/A')}")

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    print(f"Total text records: {results['total_text_records']}")
    print(f"\nEntity type counts:")
    for etype, count in sorted(results.get("entity_type_counts", {}).items()):
        print(f"  {etype}: {count}")

    print(f"\nHole table detected by standard method: {'YES' if results['hole_table_detected'] else 'NO'}")

    if results.get("detected_header_chunks"):
        print(f"\nDetected header chunks:")
        for chunk in results["detected_header_chunks"]:
            print(f"  {chunk[:100]}...")

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print("=" * 70)

    for issue in results.get("analysis", []):
        print(f"  {issue}")

    if results.get("potential_hole_content"):
        print(f"\n{'=' * 70}")
        print(f"POTENTIAL HOLE CONTENT ({len(results['potential_hole_content'])} items)")
        print("=" * 70)

        for i, item in enumerate(results["potential_hole_content"][:20], 1):
            print(f"\n[{i}] Entity: {item['etype']}, Layer: {item['layer']}")
            print(f"    Keywords: {', '.join(item['keywords_found'])}")
            if item.get("in_block"):
                print(f"    Block path: {'/'.join(item['block_path'])}")
            text_preview = item['text'].replace('\n', '\\n')
            if len(text_preview) > 100:
                text_preview = text_preview[:100] + "..."
            print(f"    Text: {text_preview}")

    if results.get("output_files"):
        print(f"\n{'=' * 70}")
        print("OUTPUT FILES")
        print("=" * 70)
        for name, path in results["output_files"].items():
            print(f"  {name}: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose hole table detection issues in DWG/DXF files"
    )
    parser.add_argument(
        "filepath",
        help="Path to DWG/DXF file"
    )
    parser.add_argument(
        "--output-dir",
        default="debug",
        help="Directory for output files (default: debug)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON only"
    )

    args = parser.parse_args()

    results = diagnose_hole_table(args.filepath, args.output_dir)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_diagnostic_report(results)


if __name__ == "__main__":
    main()
