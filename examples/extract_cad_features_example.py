#!/usr/bin/env python3
"""
Example: Comprehensive CAD Feature Extraction for Quoting
==========================================================

This script demonstrates how to use the comprehensive CAD feature extractor
to extract all the detailed features needed for accurate quoting.

Usage:
    python extract_cad_features_example.py <cad_file_path>

Example:
    python extract_cad_features_example.py ../test_data/T1769-219.step
    python extract_cad_features_example.py ../test_data/316A.dxf
"""

import sys
import json
from pathlib import Path
from pprint import pprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_quoter.cad_feature_extractor import (
    ComprehensiveFeatureExtractor,
    extract_features_from_cad_file,
    features_to_dict,
    features_to_quoting_variables,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demonstrate_with_text_examples():
    """Demonstrate feature extraction with text examples from real drawings."""

    print_section("EXAMPLE 1: T1769-219 (Round Part with Multiple Diameters)")

    text_records = [
        "Ø.7504 ±.0001",
        "Ø.5021 ±.0002",
        "Ø.149",
        "Ø.145",
        "5/16-18 TAP X .80 DEEP",
        "6.99 ±1/32 OAL",
        ".62 STRAIGHT TYP",
        ".76 STRAIGHT TYP",
        "NO STEP PERMITTED",
        "PERPENDICULAR TO CENTERLINE",
    ]

    extractor = ComprehensiveFeatureExtractor()
    features = extractor.extract_all_features(text_records=text_records)

    print("\n📋 Extracted Features:")
    print(f"   Part Type: {features.stock_geometry.part_type}")
    length_str = f"{features.stock_geometry.overall_length:.3f}\"" if features.stock_geometry.overall_length else "N/A"
    print(f"   Overall Length: {length_str}")
    diameter_str = f"{features.stock_geometry.max_diameter:.4f}\"" if features.stock_geometry.max_diameter else "N/A"
    print(f"   Max Diameter: {diameter_str}")
    print(f"\n🔧 Machining Features:")
    print(f"   Ground Diameters: {features.num_ground_diameters}")
    print(f"   Total Ground Length: {features.sum_ground_length:.3f}\"")
    print(f"   Perpendicular Face Grind: {features.has_perp_face_grind}")
    print(f"   Tapped Holes: {features.tap_count}")
    print(f"\n📏 Quality Features:")
    print(f"   Tight Tolerances: {features.tight_tolerance_count}")
    print(f"   Requires Polish: {features.requires_polish}")
    print(f"\n📊 Complexity Scores:")
    print(f"   Machining Complexity: {features.machining_complexity_score:.1f}/100")
    print(f"   Inspection Complexity: {features.inspection_complexity_score:.1f}/100")

    # Convert to quoting variables
    variables = features_to_quoting_variables(features)
    print(f"\n💰 Quoting Variables (sample):")
    for key, value in list(variables.items())[:10]:
        print(f"   {key}: {value}")

    # -------------------------------------------------------------------------

    print_section("EXAMPLE 2: 316A (Form Punch with Contour)")

    text_records = [
        "(2) .4997 +.0000/-.0002",
        "POLISH CONTOUR TO 8 µin",
        ".7000 × .5000 SHANK",
        "2.0430 OAL",
        "FORM AREA",
        "R.005 BLEND",
        "R.025 CORNER",
        "(3) .040 × 45° CHAMFER",
    ]

    features = extractor.extract_all_features(text_records=text_records)

    print("\n📋 Extracted Features:")
    print(f"   Part Type: {features.stock_geometry.part_type}")
    shank_str = f"{features.stock_geometry.body_width:.4f} × {features.stock_geometry.body_height:.4f}\"" if (features.stock_geometry.body_width and features.stock_geometry.body_height) else "N/A"
    print(f"   Shank: {shank_str}")
    length_str = f"{features.stock_geometry.centerline_length:.4f}\"" if features.stock_geometry.centerline_length else "N/A"
    print(f"   Overall Length: {length_str}")
    print(f"\n✨ Contour Features:")
    print(f"   Has Polish Contour: {features.contour_features.has_polish_contour}")
    print(f"   Minimum Radius: {features.contour_features.min_radius or 'N/A'}")
    print(f"   Radii Count: {len(features.contour_features.radii_list)}")
    print(f"\n🔨 Edge Features:")
    print(f"   Chamfers: {features.chamfer_count}")
    print(f"   Small Radii (EDM/Grind): {features.small_radius_count}")
    print(f"\n⏱️  Polish Work:")
    print(f"   Estimated Handwork: {features.polish_handwork_minutes:.1f} minutes")

    # -------------------------------------------------------------------------

    print_section("EXAMPLE 3: 1769-326 (Complex with Undercuts)")

    text_records = [
        ".7811 OVER R OAL",
        "(3) .040 × 45° CHAMFER",
        "(2) SMALL UNDERCUT",
        "R.005 MAX",
        "R.007 BLEND",
        "R.09 CORNER",
        "Ø.500 ±.0001 NO STEP PERMITTED",
        "WIRE EDM PROFILE",
        "COIN PUNCH FORM",
    ]

    features = extractor.extract_all_features(text_records=text_records)

    print("\n📋 Extracted Features:")
    print(f"   Overall Length: {features.stock_geometry.overall_length or 'N/A'}")
    print(f"\n🔧 Special Features:")
    print(f"   Chamfers: {features.chamfer_count}")
    print(f"   Undercuts: {features.undercut_count}")
    print(f"   Small Radii (≤.010): {features.small_radius_count}")
    print(f"\n✨ Form Features:")
    print(f"   Coin Punch: {features.contour_features.has_coin_punch}")
    print(f"   Wire EDM Profile: {features.contour_features.has_2d_wire_edm_profile}")
    print(f"\n📊 Complexity:")
    print(f"   Machining: {features.machining_complexity_score:.1f}/100")
    print(f"   Inspection: {features.inspection_complexity_score:.1f}/100")


def demonstrate_with_geometry():
    """Demonstrate feature extraction with 3D geometry data."""

    print_section("EXAMPLE 4: STEP File Geometry Analysis")

    # Simulated geometry features from a STEP file
    geo_features = {
        "GEO-01_Length_mm": 177.546,  # 6.99"
        "GEO-02_Width_mm": 19.0602,   # 0.75"
        "GEO-03_Height_mm": 19.0602,  # 0.75"
        "GEO_Turning_Score_0to1": 0.85,  # Cylindrical part
        "GEO_MaxOD_mm": 19.0602,
        "GEO-Volume_mm3": 50000.0,
        "GEO-SurfaceArea_mm2": 15000.0,
        "Feature_Face_Count": 24,
        "GEO_Hole_Groups": [
            {"dia_mm": 7.938, "depth_mm": 20.32, "through": False, "count": 1},  # 5/16" tap hole
            {"dia_mm": 3.78, "depth_mm": 12.7, "through": False, "count": 2},    # Small holes
        ],
        "GEO_Area_Freeform_mm2": 200.0,  # Has some freeform surface
        "GEO_WEDM_PathLen_mm": 0.0,
        "GEO_Complexity_0to100": 45.0,
    }

    extractor = ComprehensiveFeatureExtractor()
    features = extractor.extract_all_features(geo_features=geo_features)

    print("\n📐 Geometry Analysis:")
    print(f"   Part Type: {features.stock_geometry.part_type}")
    print(f"   Dimensions: {features.stock_geometry.overall_length:.2f}\" × Ø{features.stock_geometry.max_diameter:.3f}\"")
    print(f"   Estimated Volume: {features.stock_geometry.estimated_volume:.2f} cu in")
    print(f"   Estimated Weight: {features.stock_geometry.estimated_weight:.2f} lb (steel)")
    print(f"\n🔍 Detected Features:")
    print(f"   Holes Detected: {len(features.hole_features)}")
    for i, hole in enumerate(features.hole_features, 1):
        print(f"     {i}. Ø{hole.diameter:.3f}\" × {hole.depth:.3f}\" deep, qty: {hole.count}")
    print(f"\n✨ Surface Analysis:")
    print(f"   Has 3D Contour: {features.contour_features.has_3d_contoured_nose}")
    print(f"   Form Area: {features.contour_features.form_area_sq_in or 'N/A'}")


def demonstrate_combined_extraction():
    """Demonstrate combining text and geometry extraction."""

    print_section("EXAMPLE 5: Combined Text + Geometry Extraction")

    # Text annotations from drawing
    text_records = [
        "Ø.7504 ±.0001",
        "Ø.5021 ±.0002",
        "5/16-18 TAP X .80 DEEP",
        "(2) .040 × 45° CHAMFER",
        "POLISH CONTOUR",
    ]

    # Geometry from STEP file
    geo_features = {
        "GEO-01_Length_mm": 177.546,
        "GEO_Turning_Score_0to1": 0.85,
        "GEO_MaxOD_mm": 19.0602,
        "GEO-Volume_mm3": 50000.0,
        "GEO_Hole_Groups": [
            {"dia_mm": 7.938, "depth_mm": 20.32, "through": False, "count": 1},
        ],
    }

    extractor = ComprehensiveFeatureExtractor()
    features = extractor.extract_all_features(
        text_records=text_records,
        geo_features=geo_features
    )

    print("\n🔄 Combined Analysis:")
    print(f"   Text Records Processed: {len(text_records)}")
    print(f"   Geometry Features: {len(geo_features)}")
    print(f"\n📊 Results:")
    print(f"   Part Type: {features.stock_geometry.part_type}")
    print(f"   Cylindrical Sections: {len(features.cylindrical_sections)}")
    print(f"   Holes (text): {features.tap_count}")
    print(f"   Holes (geometry): {len([h for h in features.hole_features if h.hole_type != 'threaded'])}")
    print(f"   Chamfers: {features.chamfer_count}")
    print(f"   Requires Polish: {features.requires_polish}")
    print(f"\n💰 Quoting Impact:")
    print(f"   Machining Complexity: {features.machining_complexity_score:.1f}/100")
    print(f"   Inspection Complexity: {features.inspection_complexity_score:.1f}/100")
    print(f"   Polish Time: {features.polish_handwork_minutes:.1f} min")


def export_to_json_example():
    """Demonstrate exporting features to JSON."""

    print_section("EXAMPLE 6: Export to JSON")

    text_records = [
        "Ø.7504 ±.0001",
        "6.99 OAL",
        "5/16-18 TAP X .80 DEEP",
    ]

    extractor = ComprehensiveFeatureExtractor()
    features = extractor.extract_all_features(text_records=text_records)

    # Convert to dictionary (JSON-serializable)
    features_dict = features_to_dict(features)

    print("\n📄 JSON Output (partial):")
    print(json.dumps({
        "stock_geometry": features_dict["stock_geometry"],
        "num_ground_diameters": features_dict["num_ground_diameters"],
        "tap_count": features_dict["tap_count"],
        "machining_complexity_score": features_dict["machining_complexity_score"],
    }, indent=2))

    print("\n💾 Full output would be saved as:")
    print("   features.json")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 80)
    print("  COMPREHENSIVE CAD FEATURE EXTRACTION - DEMONSTRATIONS")
    print("█" * 80)

    demonstrate_with_text_examples()
    demonstrate_with_geometry()
    demonstrate_combined_extraction()
    export_to_json_example()

    print("\n" + "█" * 80)
    print("  ALL DEMONSTRATIONS COMPLETE")
    print("█" * 80)
    print("\n✅ The comprehensive feature extractor can now extract:")
    print("   • Stock/envelope geometry (round & rectangular parts)")
    print("   • Cylindrical sections with tight tolerances")
    print("   • Contours, forms, and 3D surfaces")
    print("   • Holes, taps, and internal features")
    print("   • Chamfers, radii, and undercuts")
    print("   • Tolerance and finish requirements")
    print("   • Machining & inspection complexity scores")
    print("\n🚀 Ready to integrate with your quoting system!\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Extract from actual CAD file if provided
        cad_file = Path(sys.argv[1])
        if not cad_file.exists():
            print(f"Error: File not found: {cad_file}")
            sys.exit(1)

        print(f"\n🔍 Extracting features from: {cad_file}")
        features = extract_features_from_cad_file(cad_file)

        print("\n📊 Extraction Results:")
        print(f"   Part Type: {features.stock_geometry.part_type}")
        print(f"   Cylindrical Sections: {len(features.cylindrical_sections)}")
        print(f"   Holes: {len(features.hole_features)}")
        print(f"   Edge Features: {len(features.edge_features)}")
        print(f"   Machining Complexity: {features.machining_complexity_score:.1f}/100")
        print(f"   Inspection Complexity: {features.inspection_complexity_score:.1f}/100")
    else:
        # Run demonstrations
        main()
