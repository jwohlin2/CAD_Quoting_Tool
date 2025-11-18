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
    r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dwg",
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


def compare_dimensions(found_candidates: list, expected: tuple, tolerance: float = 0.05) -> dict:
    """
    Compare found dimension candidates against expected dimensions.

    Args:
        found_candidates: List of (value, score) tuples from dimension finder
        expected: Tuple of expected (L, W, H) dimensions
        tolerance: Relative tolerance for matching (default 5%)

    Returns:
        dict with comparison results
    """
    found_values = [val for val, _ in found_candidates] if found_candidates else []

    results = {
        "expected": expected,
        "found_values": found_values[:10],  # Top 10 candidates
        "matches": [],
        "missing": [],
        "pass": False,
        "match_count": 0,
        "total_expected": len(expected),
    }

    matched_indices = set()

    for exp_dim in expected:
        best_match = None
        best_diff = float('inf')
        best_idx = None

        for idx, found_val in enumerate(found_values):
            if idx in matched_indices:
                continue

            # Calculate relative difference
            if exp_dim == 0:
                diff = abs(found_val)
            else:
                diff = abs(found_val - exp_dim) / exp_dim

            if diff < best_diff:
                best_diff = diff
                best_match = found_val
                best_idx = idx

        if best_match is not None and best_diff <= tolerance:
            results["matches"].append({
                "expected": exp_dim,
                "found": best_match,
                "diff_pct": best_diff * 100,
                "status": "PASS"
            })
            results["match_count"] += 1
            if best_idx is not None:
                matched_indices.add(best_idx)
        else:
            # Check if we found something close but outside tolerance
            if best_match is not None:
                results["missing"].append({
                    "expected": exp_dim,
                    "closest": best_match,
                    "diff_pct": best_diff * 100,
                    "status": "FAIL"
                })
            else:
                results["missing"].append({
                    "expected": exp_dim,
                    "closest": None,
                    "diff_pct": None,
                    "status": "FAIL"
                })

    # Overall pass if all expected dimensions were found
    results["pass"] = results["match_count"] == results["total_expected"]
    results["match_rate"] = results["match_count"] / results["total_expected"] if results["total_expected"] > 0 else 0

    return results


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

        # Capture the top inferred bbox values for later reporting
        top_3 = [val for val, _ in analysis["inferred_bbox"][:3]]
        result["top_3_inferred"] = top_3

        # Perform our own comparison if expected dimensions provided
        if expected:
            result["expected"] = expected
            comparison = compare_dimensions(analysis["inferred_bbox"], expected)
            result["comparison"] = comparison
            result["match_rate"] = comparison["match_rate"]
            result["pass"] = comparison["pass"]

            # Quick quality check: compare expected vs inferred top values
            exp_sorted = sorted(expected, reverse=True)
            inf_sorted = sorted(top_3, reverse=True)

            matches = 0
            max_diff_pct = 0.0
            for i, exp_val in enumerate(exp_sorted):
                if i < len(inf_sorted):
                    inf_val = inf_sorted[i]
                    diff_pct = abs(inf_val - exp_val) / exp_val * 100 if exp_val else 0
                    max_diff_pct = max(max_diff_pct, diff_pct)
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
            print(f"Source: {result.get('source', 'N/A')}")
            print(f"Total dimensions: {result.get('total_dimensions', 0)}")
            print(f"Unique values: {len(result.get('unique_values', []))}")

            # Show top bbox candidates
            if result.get("bbox_candidates"):
                print("Bounding box candidates:")
                for val, score in result["bbox_candidates"][:3]:
                    print(f"  {val:.4f}\" (score: {score:.2f})")

            # Show comparison if available
            if result.get("comparison"):
                comp = result["comparison"]
                pass_fail = "PASS" if result.get("pass") else "FAIL"
                print(f"\nComparison Result: [{pass_fail}]")
                print(f"Expected dimensions: {result.get('expected')}")
                print(f"Match rate: {comp['match_rate']*100:.0f}% ({comp['match_count']}/{comp['total_expected']})")

                print("\nDimension matching:")
                for m in comp.get("matches", []):
                    print(f"  [PASS] {m['expected']:.4f} -> {m['found']:.4f} (diff: {m['diff_pct']:.1f}%)")

                for m in comp.get("missing", []):
                    if m.get("closest") is not None:
                        print(f"  [FAIL] {m['expected']:.4f} -> closest: {m['closest']:.4f} (diff: {m['diff_pct']:.1f}%)")
                    else:
                        print(f"  [FAIL] {m['expected']:.4f} -> NOT FOUND")

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

        max_diffs = [
            r.get("max_diff_pct")
            for r in results
            if r.get("max_diff_pct") is not None
        ]
        if max_diffs:
            avg_max_diff = sum(max_diffs) / len(max_diffs)
            print(f"  Average max diff: {avg_max_diff:.1f}%")

        # Calculate pass/fail statistics
        passed = [r for r in results if r.get("pass") is True]
        failed = [r for r in results if r.get("pass") is False]

        print(f"\nComparison Results:")
        print(f"  Passed: {len(passed)}")
        print(f"  Failed: {len(failed)}")

        # Calculate overall match rate
        match_rates = [r.get("match_rate", 0) for r in results if r.get("match_rate") is not None]
        if match_rates:
            avg_match = sum(match_rates) / len(match_rates)
            print(f"  Average match rate: {avg_match*100:.1f}%")

        # Results table
        print("\n" + "=" * 80)
        print("RESULTS TABLE")
        print("=" * 80)
        print(f"{'File':<28} {'Status':<8} {'Rate':<8} {'Max Diff':<10} {'Expected'}")
        print("-" * 80)

        for r in results:
            filename = r['filename'][:26] if len(r['filename']) > 26 else r['filename']
            if r.get("pass") is not None:
                status = "PASS" if r["pass"] else "FAIL"
                rate = f"{r.get('match_rate', 0)*100:.0f}%"
                expected = r.get("expected", "N/A")
                if expected != "N/A":
                    expected = f"{expected[0]}x{expected[1]}x{expected[2]}"

                # Calculate max diff from matches and missing
                comp = r.get("comparison", {})
                all_diffs = []
                for m in comp.get("matches", []):
                    all_diffs.append(m["diff_pct"])
                for m in comp.get("missing", []):
                    if m.get("diff_pct") is not None:
                        all_diffs.append(m["diff_pct"])

                if all_diffs:
                    max_diff = f"{max(all_diffs):.1f}%"
                else:
                    max_diff = "N/A"
            else:
                status = "ERROR"
                rate = "N/A"
                expected = "N/A"
                max_diff = "N/A"

            print(f"{filename:<28} {status:<8} {rate:<8} {max_diff:<10} {expected}")

        # Show failed files details
        if failed:
            print("\n" + "=" * 70)
            print("FAILED FILES DETAILS")
            print("=" * 70)
            for r in failed:
                print(f"\n{r['filename']}:")
                comp = r.get("comparison", {})
                for m in comp.get("missing", []):
                    if m.get("closest") is not None:
                        print(f"  Missing: {m['expected']:.4f} (closest found: {m['closest']:.4f}, diff: {m['diff_pct']:.1f}%)")
                    else:
                        print(f"  Missing: {m['expected']:.4f} (not found)")

    return results


if __name__ == "__main__":
    main()
