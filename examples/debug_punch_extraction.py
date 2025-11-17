"""
DWG Punch Extraction Debug/Validation Script
=============================================

This script helps debug and validate punch feature extraction from DWG/DXF files.
Use it to diagnose issues with extraction for specific parts (e.g., 316A).

Usage:
    python examples/debug_punch_extraction.py path/to/file.dxf

Output:
    - Full text dump to debug/[filename]_text_dump.txt
    - Extraction results in JSON format
    - Detailed diagnostics and warnings

Author: CAD Quoting Tool
Date: 2025-11-17
"""

import sys
import json
from pathlib import Path
from pprint import pprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_quoter.geometry.dwg_punch_extractor import (
    extract_punch_features,
    extract_punch_features_from_dxf,
    classify_punch_family,
    detect_material,
    detect_ops_features,
    detect_pain_flags,
    parse_holes_from_text,
    parse_tolerances_from_text,
)


def debug_extraction(dxf_path: str, save_debug: bool = True):
    """
    Run punch extraction with detailed debugging output.

    Args:
        dxf_path: Path to DXF file
        save_debug: Whether to save debug output to files
    """
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        print(f"ERROR: File not found: {dxf_path}")
        return

    print("=" * 70)
    print(f"DWG PUNCH EXTRACTION DEBUG: {dxf_path.name}")
    print("=" * 70)

    # Extract text from DXF
    print("\n1. Extracting text from DXF...")
    try:
        from cad_quoter.geo_extractor import open_doc, collect_all_text

        doc = open_doc(dxf_path)
        text_records = list(collect_all_text(doc))
        text_lines = [rec.text for rec in text_records if rec.text]
        text_dump = "\n".join(text_lines)

        print(f"   ✓ Extracted {len(text_records)} text records")
        print(f"   ✓ Total text lines: {len(text_lines)}")
        print(f"   ✓ Total text chars: {len(text_dump)}")

        # Save text dump
        if save_debug:
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            text_file = debug_dir / f"{dxf_path.stem}_text_dump.txt"
            text_file.write_text(text_dump)
            print(f"   ✓ Saved text dump to: {text_file}")

    except Exception as e:
        print(f"   ✗ Error extracting text: {e}")
        text_lines = []
        text_dump = ""

    # Show sample text
    print("\n2. Text Sample (first 500 chars):")
    print("-" * 70)
    print(text_dump[:500])
    print("-" * 70)

    # Test individual extraction functions
    print("\n3. Testing Individual Extractors:")
    print("-" * 70)

    # Classification
    family, shape = classify_punch_family(text_dump)
    print(f"   Family: {family}")
    print(f"   Shape: {shape}")

    # Material
    material = detect_material(text_dump)
    print(f"   Material: {material}")

    # Operations features
    ops_features = detect_ops_features(text_dump)
    print(f"   Chamfers: {ops_features['num_chamfers']}")
    print(f"   Small Radii: {ops_features['num_small_radii']}")
    print(f"   Has 3D Surface: {ops_features['has_3d_surface']}")
    print(f"   Has Perp Face: {ops_features['has_perp_face_grind']}")
    print(f"   Form Complexity: {ops_features['form_complexity_level']}")

    # Pain flags
    pain_flags = detect_pain_flags(text_dump)
    print(f"   Polish: {pain_flags['has_polish_contour']}")
    print(f"   No Step: {pain_flags['has_no_step_permitted']}")
    print(f"   Sharp Edges: {pain_flags['has_sharp_edges']}")
    print(f"   GD&T: {pain_flags['has_gdt']}")

    # Holes/taps
    hole_data = parse_holes_from_text(text_dump)
    print(f"   Tap Count: {hole_data['tap_count']}")
    if hole_data['tap_summary']:
        for tap in hole_data['tap_summary']:
            print(f"      - {tap['size']} x {tap.get('depth_in', 'N/A')}\"")

    # Tolerance examples (from text)
    print("\n4. Tolerance Detection Examples:")
    print("-" * 70)
    # Find lines with common tolerance patterns
    tolerance_lines = [
        line for line in text_lines
        if any(pattern in line.upper() for pattern in ['+', '±', '/-'])
    ][:5]  # Show first 5
    for line in tolerance_lines:
        tols = parse_tolerances_from_text(line)
        if tols:
            print(f"   Line: {line[:60]}")
            print(f"   Tolerances: {tols}")

    # Run full extraction
    print("\n5. Running Full Extraction:")
    print("-" * 70)
    try:
        summary = extract_punch_features_from_dxf(dxf_path, text_dump)

        # Convert to dict for JSON serialization
        summary_dict = {
            "family": summary.family,
            "shape_type": summary.shape_type,
            "overall_length_in": summary.overall_length_in,
            "max_od_or_width_in": summary.max_od_or_width_in,
            "body_width_in": summary.body_width_in,
            "body_thickness_in": summary.body_thickness_in,
            "num_ground_diams": summary.num_ground_diams,
            "total_ground_length_in": summary.total_ground_length_in,
            "has_perp_face_grind": summary.has_perp_face_grind,
            "has_3d_surface": summary.has_3d_surface,
            "form_complexity_level": summary.form_complexity_level,
            "tap_count": summary.tap_count,
            "tap_summary": summary.tap_summary,
            "num_chamfers": summary.num_chamfers,
            "num_small_radii": summary.num_small_radii,
            "min_dia_tol_in": summary.min_dia_tol_in,
            "min_len_tol_in": summary.min_len_tol_in,
            "has_polish_contour": summary.has_polish_contour,
            "has_no_step_permitted": summary.has_no_step_permitted,
            "has_sharp_edges": summary.has_sharp_edges,
            "has_gdt": summary.has_gdt,
            "material_callout": summary.material_callout,
            "confidence_score": summary.confidence_score,
            "warnings": summary.warnings,
        }

        print("\n✓ Extraction successful!")
        print("\nExtracted Features:")
        pprint(summary_dict, width=100)

        # Save JSON
        if save_debug:
            debug_dir = Path("debug")
            json_file = debug_dir / f"{dxf_path.stem}_extraction_results.json"
            with open(json_file, 'w') as f:
                json.dump(summary_dict, f, indent=2)
            print(f"\n✓ Saved results to: {json_file}")

        # Validation checks
        print("\n6. Validation Checks:")
        print("-" * 70)
        checks = []

        if summary.overall_length_in > 0:
            checks.append(("✓", f"Overall length: {summary.overall_length_in:.3f}\""))
        else:
            checks.append(("✗", "Overall length is 0"))

        if summary.max_od_or_width_in > 0:
            checks.append(("✓", f"Max OD/width: {summary.max_od_or_width_in:.4f}\""))
        else:
            checks.append(("✗", "Max OD/width is 0"))

        if summary.material_callout:
            checks.append(("✓", f"Material: {summary.material_callout}"))
        else:
            checks.append(("⚠", "No material detected"))

        if summary.num_ground_diams > 0:
            checks.append(("✓", f"Ground diameters: {summary.num_ground_diams}"))
        else:
            checks.append(("⚠", "No ground diameters found"))

        if summary.confidence_score >= 0.7:
            checks.append(("✓", f"Confidence: {summary.confidence_score:.2f} (good)"))
        elif summary.confidence_score >= 0.4:
            checks.append(("⚠", f"Confidence: {summary.confidence_score:.2f} (moderate)"))
        else:
            checks.append(("✗", f"Confidence: {summary.confidence_score:.2f} (poor)"))

        for symbol, message in checks:
            print(f"   {symbol} {message}")

        # Warnings
        if summary.warnings:
            print("\n7. Warnings:")
            print("-" * 70)
            for warning in summary.warnings:
                print(f"   ⚠ {warning}")

        return summary

    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python debug_punch_extraction.py <dxf_file>")
        print("\nExample:")
        print("  python examples/debug_punch_extraction.py parts/316A.dxf")
        sys.exit(1)

    dxf_file = sys.argv[1]
    debug_extraction(dxf_file, save_debug=True)


if __name__ == "__main__":
    main()
