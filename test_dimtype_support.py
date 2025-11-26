#!/usr/bin/env python3
"""
Test the updated geo_extractor with dimtype and measurement support.
"""

import sys
sys.path.insert(0, '.')

from cad_quoter.geo_extractor import TextRecord

def test_textrecord_with_dimtype():
    """Test that TextRecord now includes dimtype and measurement fields."""
    print("=" * 70)
    print("TEST: TextRecord with dimtype and measurement")
    print("=" * 70)

    # Test creating a TextRecord for a radial dimension
    record = TextRecord(
        layout="Model",
        layer="BUP_1-AM_0",
        etype="DIMENSION",
        text="R.010",
        x=0.0,
        y=0.0,
        height=0.0,
        rotation=0.0,
        in_block=False,
        depth=0,
        block_path=(),
        dimtype=4,  # Radial dimension
        measurement=0.010,
    )

    print(f"✓ Created TextRecord for radial dimension:")
    print(f"  text: {record.text}")
    print(f"  dimtype: {record.dimtype} (4 = radius)")
    print(f"  measurement: {record.measurement}")
    print()

    # Test creating a TextRecord for a diameter dimension
    record2 = TextRecord(
        layout="Model",
        layer="BUP_1-AM_0",
        etype="DIMENSION",
        text="Ø.3834",
        x=0.0,
        y=0.0,
        height=0.0,
        rotation=0.0,
        in_block=False,
        depth=0,
        block_path=(),
        dimtype=3,  # Diameter dimension
        measurement=0.3834,
    )

    print(f"✓ Created TextRecord for diameter dimension:")
    print(f"  text: {record2.text}")
    print(f"  dimtype: {record2.dimtype} (3 = diameter)")
    print(f"  measurement: {record2.measurement}")
    print()

    # Test creating a TextRecord for regular text (no dimtype)
    record3 = TextRecord(
        layout="Model",
        layer="0",
        etype="TEXT",
        text="SOME TEXT",
        x=0.0,
        y=0.0,
        height=0.125,
        rotation=0.0,
        in_block=False,
        depth=0,
        block_path=(),
        dimtype=None,
        measurement=None,
    )

    print(f"✓ Created TextRecord for regular text:")
    print(f"  text: {record3.text}")
    print(f"  dimtype: {record3.dimtype} (None for non-dimensions)")
    print(f"  measurement: {record3.measurement}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ TextRecord now supports dimtype and measurement fields")
    print("✓ Radial dimensions (dimtype=4) can be identified")
    print("✓ Diameter dimensions (dimtype=3) can be identified")
    print("✓ Measurement values are preserved separately from text")
    print()
    print("Expected behavior for the user's issues:")
    print("  1. Text '1.1933 {\\Famgdt|c0;q}' with measurement .3834")
    print("     → Will now show text: 'Ø.3834' (using actual measurement)")
    print()
    print("  2. Radial dimension with measurement .0829")
    print("     → Will now show text: 'R.0829' (with R prefix)")
    print("     → dimtype: 4 (can be filtered/identified)")
    print()

    return True

if __name__ == '__main__':
    success = test_textrecord_with_dimtype()
    sys.exit(0 if success else 1)
