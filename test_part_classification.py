#!/usr/bin/env python3
"""
Test script to verify part classification and planner routing.
Tests that the part_family field matches the planner that was used.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad


def test_part_classification(cad_file_path: str):
    """Test that part classification matches planner routing."""
    print(f"\n{'='*80}")
    print(f"Testing: {Path(cad_file_path).name}")
    print(f"{'='*80}\n")

    try:
        # Extract quote data
        quote_data = extract_quote_data_from_cad(
            cad_file_path=cad_file_path,
            machine_rate=85.0,
            labor_rate=45.0,
            margin_rate=0.15,
            verbose=True
        )

        # Display results
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Part Family: {quote_data.part_family}")

        if quote_data.raw_plan:
            planner = quote_data.raw_plan.get('planner', 'Unknown')
            print(f"Planner Used: {planner}")

            # Check if they match (or are compatible)
            if quote_data.part_family == planner:
                print(f"✓ PASS: Part family matches planner")
            elif quote_data.part_family in ['Guide Post', 'Punch', 'Pilot Pin', 'Bushing', 'Form Punch', 'Spring Punch'] and planner == 'punch_planner':
                print(f"✓ PASS: Punch sub-type correctly identified")
            else:
                print(f"✗ FAIL: Mismatch - Part family '{quote_data.part_family}' != Planner '{planner}'")

        print(f"\nPart Dimensions:")
        print(f"  Length: {quote_data.part_dimensions.length:.3f}\"")
        print(f"  Width: {quote_data.part_dimensions.width:.3f}\"")
        print(f"  Thickness: {quote_data.part_dimensions.thickness:.3f}\"")
        if quote_data.part_dimensions.is_cylindrical:
            print(f"  Diameter: {quote_data.part_dimensions.diameter:.3f}\"")
            print(f"  Is Cylindrical: Yes")

        print(f"\nMaterial: {quote_data.material_info.material_name}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test files (using DXF where available since ODA converter may not be set up)
    test_files = [
        "Cad Files/301_redacted.dxf",         # Test with available DXF file
        "Cad Files/zeus1.dxf",                # Test with available DXF file
    ]

    results = []
    for test_file in test_files:
        file_path = Path(test_file)
        if file_path.exists():
            success = test_part_classification(str(file_path))
            results.append((test_file, success))
        else:
            print(f"\n✗ SKIP: File not found: {test_file}")
            results.append((test_file, None))

    # Summary
    print(f"\n\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    for test_file, success in results:
        if success is None:
            status = "SKIP"
        elif success:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"{status}: {Path(test_file).name}")
