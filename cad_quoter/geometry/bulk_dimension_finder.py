#!/usr/bin/env python3
"""
Bulk Dimension Finder - Run dimension extraction on multiple DWG/DXF files.

This script processes a list of CAD files and extracts bounding box dimensions.
For DWG files, it will look for corresponding mtext_results.json files or
convert to DXF first.

Usage:
    python bulk_dimension_finder.py
    python bulk_dimension_finder.py --with-expected

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cad_quoter.geometry.dimension_finder import DimensionFinder, analyze_file

# List of DWG files to process
DWG_FILES = [
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-130.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-134.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-139.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\Tl 769-157.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-202.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-219.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-326.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-334.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-339.dwg",
    r"C:\Users\John Wohlin\Downloads\composinonimizer\T1769-348.dwg",
    r"D:\CAD_Quoting_Tool\Cad Files\T1769-201_redacted.dwg",
    r"D:\CAD_Quoting_Tool\Cad Files\T1769-143_redacted.dwg",
    r"D:\CAD_Quoting_Tool\Cad Files\T1769-108_redacted.dwg",
    r"D:\CAD_Quoting_Tool\Cad Files\T1769-104_redacted.dwg",
    r"D:\CAD_Quoting_Tool\Cad Files\316A_redacted.dwg",
    r"D:\CAD_Quoting_Tool\Cad Files\305_redacted.dwg",
    r"D:\CAD_Quoting_Tool\Cad Files\301 _redacted.dwg",
]

# Expected dimensions for validation (optional)
# Format: file_pattern -> "LxWxH"
EXPECTED_DIMS = {
    "130": "2.2248x1.8098x.4000",
    "134": "3.23x0.29x0.375",
    "139": "1.74x1.990x.1000",
    "157": "1.25x.68x2.250",
    "202": "4.47x2.5x.4370",
    "326": ".2700x.5000x.7000",
    "334": "1x3.2x.290",
    "339": ".7500x1.1750x.5005",
    "348": "1.25x.68x2.325",
    "201": "19x11.5x1.125",
    "143": ".665x2.758x.7811",
    "108": "1.25x6.5x.250",
    "104": "5x7.72x.75",
    "316": ".148x.445x2",
    "305": "8.72x2.5x5.005",
    "301": "15.5x12x2",
}


def find_json_results(dwg_path: Path) -> Path | None:
    """
    Find corresponding mtext_results.json file for a DWG.

    Looks in common locations for the results file.
    """
    stem = dwg_path.stem

    # Try same directory
    json_file = dwg_path.parent / f"{stem}.mtext_results.json"
    if json_file.exists():
        return json_file

    # Try Cad Files directory
    cad_files_dir = Path(__file__).parent.parent.parent / "Cad Files"
    if cad_files_dir.exists():
        # Look for matching files
        for f in cad_files_dir.glob("*mtext_results.json"):
            # Extract part number from filename
            if stem.replace("_redacted", "") in f.stem:
                return f
            # Handle variations like "T1769-130" matching "T1769-130.mtext_results"
            for part in stem.split("-"):
                if part.isdigit() and len(part) == 3 and part in f.stem:
                    return f

    return None


def find_dxf_file(dwg_path: Path) -> Path | None:
    """Find corresponding DXF file for a DWG."""
    dxf_path = dwg_path.with_suffix(".dxf")
    if dxf_path.exists():
        return dxf_path

    # Try without _redacted suffix
    stem = dwg_path.stem.replace("_redacted", "")
    dxf_path = dwg_path.parent / f"{stem}.dxf"
    if dxf_path.exists():
        return dxf_path

    return None


def get_expected_dims(filepath: str) -> tuple | None:
    """Get expected dimensions for a file based on its name."""
    for pattern, dims_str in EXPECTED_DIMS.items():
        if pattern in filepath:
            dims = [float(d) for d in dims_str.split('x')]
            return tuple(dims)
    return None


def process_file(filepath: str, use_expected: bool = False) -> dict:
    """
    Process a single DWG file.

    Returns dict with results or error information.
    """
    dwg_path = Path(filepath)
    result = {
        "file": filepath,
        "filename": dwg_path.name,
        "status": "unknown",
        "dimensions": [],
        "bbox_candidates": [],
    }

    # Check if file exists (may not on this system)
    if not dwg_path.exists():
        # Try to find mtext_results.json
        json_file = find_json_results(dwg_path)
        if json_file:
            result["source"] = str(json_file)
            result["status"] = "json_results"
        else:
            result["status"] = "file_not_found"
            result["error"] = f"DWG not found and no JSON results available"
            return result
    else:
        # Try DXF first, then look for JSON results
        dxf_file = find_dxf_file(dwg_path)
        if dxf_file:
            result["source"] = str(dxf_file)
            result["status"] = "dxf"
        else:
            json_file = find_json_results(dwg_path)
            if json_file:
                result["source"] = str(json_file)
                result["status"] = "json_results"
            else:
                result["status"] = "no_source"
                result["error"] = "No DXF or JSON results found"
                return result

    # Get expected dimensions if requested
    expected = None
    if use_expected:
        expected = get_expected_dims(filepath)
        if expected:
            result["expected"] = expected

    # Process the file
    try:
        source_path = Path(result["source"])
        analysis = analyze_file(source_path, expected)

        result["total_dimensions"] = analysis["total_dimensions"]
        result["unique_values"] = analysis["unique_values"]
        result["bbox_candidates"] = analysis["inferred_bbox"]

        # Get top 3 inferred dimensions
        top_3 = [val for val, _ in analysis["inferred_bbox"][:3]]
        result["top_3_inferred"] = top_3

        if expected and "comparison" in analysis:
            result["comparison"] = analysis["comparison"]
            result["match_rate"] = analysis["comparison"]["match_rate"]

            # Calculate how well top 3 inferred match expected
            # Sort both for comparison
            exp_sorted = sorted(expected, reverse=True)
            inf_sorted = sorted(top_3, reverse=True) if len(top_3) >= 3 else top_3

            matches = 0
            max_diff_pct = 0
            for i, exp_val in enumerate(exp_sorted):
                if i < len(inf_sorted):
                    inf_val = inf_sorted[i]
                    diff_pct = abs(inf_val - exp_val) / exp_val * 100 if exp_val else 0
                    max_diff_pct = max(max_diff_pct, diff_pct)
                    # Consider a match if within 5%
                    if diff_pct <= 5:
                        matches += 1

            result["inferred_match_count"] = matches
            result["max_diff_pct"] = max_diff_pct

        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    """Main entry point for bulk dimension finder."""
    use_expected = "--with-expected" in sys.argv

    print("=" * 70)
    print("Bulk Dimension Finder")
    print("=" * 70)
    print()

    if use_expected:
        print("Mode: With expected dimensions comparison")
    else:
        print("Mode: Dimension extraction only")
    print()

    results = []
    success_count = 0
    error_count = 0

    for filepath in DWG_FILES:
        result = process_file(filepath, use_expected)
        results.append(result)

        # Print result
        print(f"{'─' * 70}")
        print(f"File: {result['filename']}")

        if result["status"] == "success":
            success_count += 1

            # Show inferred vs expected comparison
            if result.get("expected") and result.get("top_3_inferred"):
                exp = result["expected"]
                inf = result["top_3_inferred"]

                # Sort both for comparison
                exp_sorted = sorted(exp, reverse=True)
                inf_sorted = sorted(inf, reverse=True)

                print(f"Expected (sorted): {exp_sorted[0]:.4f} x {exp_sorted[1]:.4f} x {exp_sorted[2]:.4f}")
                if len(inf_sorted) >= 3:
                    print(f"Inferred (top 3):  {inf_sorted[0]:.4f} x {inf_sorted[1]:.4f} x {inf_sorted[2]:.4f}")
                else:
                    print(f"Inferred (top 3):  {inf_sorted}")

                # Show match quality
                match_count = result.get("inferred_match_count", 0)
                max_diff = result.get("max_diff_pct", 0)
                print(f"Matches: {match_count}/3 (max diff: {max_diff:.1f}%)")

                # Detailed comparison
                for i in range(3):
                    if i < len(inf_sorted):
                        diff_pct = abs(inf_sorted[i] - exp_sorted[i]) / exp_sorted[i] * 100 if exp_sorted[i] else 0
                        status = "OK" if diff_pct <= 5 else "MISS"
                        print(f"  [{status}] {exp_sorted[i]:.4f} -> {inf_sorted[i]:.4f} ({diff_pct:+.1f}%)")
            else:
                # Just show inferred if no expected
                if result.get("bbox_candidates"):
                    print("Top 3 inferred dimensions:")
                    for val, score in result["bbox_candidates"][:3]:
                        print(f"  {val:.4f}\"")

        elif result["status"] == "file_not_found":
            error_count += 1
            print(f"Status: FILE NOT FOUND")
            print(f"Note: {result.get('error', 'Unknown error')}")

        else:
            error_count += 1
            print(f"Status: {result['status'].upper()}")
            if result.get("error"):
                print(f"Error: {result['error']}")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(DWG_FILES)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors/Not found: {error_count}")

    if use_expected:
        # Calculate inferred bbox match statistics
        perfect_matches = 0
        partial_matches = 0
        no_matches = 0

        for r in results:
            match_count = r.get("inferred_match_count", 0)
            if match_count == 3:
                perfect_matches += 1
            elif match_count > 0:
                partial_matches += 1
            elif r.get("status") == "success":
                no_matches += 1

        print()
        print("INFERRED BBOX ACCURACY:")
        print(f"  Perfect (3/3): {perfect_matches}")
        print(f"  Partial (1-2/3): {partial_matches}")
        print(f"  None (0/3): {no_matches}")

        # Calculate average max diff
        max_diffs = [r.get("max_diff_pct", 0) for r in results if r.get("max_diff_pct") is not None]
        if max_diffs:
            avg_max_diff = sum(max_diffs) / len(max_diffs)
            print(f"  Average max diff: {avg_max_diff:.1f}%")

    return results


if __name__ == "__main__":
    main()
