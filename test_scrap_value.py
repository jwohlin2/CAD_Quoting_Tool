"""
Test script for scrap value calculation using Wieland scrap prices.

This demonstrates calculating the dollar value of scrap material.
"""

import sys
from pathlib import Path

# Add the cad_quoter module to path
sys.path.insert(0, str(Path(__file__).parent))

from cad_quoter.pricing.DirectCostHelper import (
    calculate_total_scrap_with_value,
    calculate_scrap_value
)


def test_scrap_value_calculation():
    """Test scrap value calculation with Wieland prices."""

    # Test file
    cad_file = r"D:\CAD_Quoting_Tool\Cad Files\301_redacted.dxf"

    print("\n" + "="*70)
    print("SCRAP VALUE CALCULATION TEST")
    print("="*70)

    # Test 1: Aluminum MIC6
    print("\n1. Aluminum MIC6")
    print("-" * 70)

    result = calculate_total_scrap_with_value(
        cad_file,
        material="aluminum MIC6",
        verbose=True
    )

    scrap = result['scrap_info']
    value = result['scrap_value_info']

    print(f"\nSummary:")
    print(f"  Total scrap: {scrap.total_scrap_volume:.2f} in³ ({scrap.total_scrap_weight:.2f} lbs)")
    print(f"  Scrap price: ${value['scrap_price_per_lb']:.4f}/lb ({value['price_source']})")
    print(f"  Scrap value: ${value['scrap_value']:.2f}")
    print(f"  Material utilization: {scrap.utilization_percentage:.1f}%")

    # Test 2: P20 Tool Steel
    print("\n\n2. P20 Tool Steel")
    print("-" * 70)

    result2 = calculate_total_scrap_with_value(
        cad_file,
        material="P20 Tool Steel",
        verbose=True
    )

    scrap2 = result2['scrap_info']
    value2 = result2['scrap_value_info']

    print(f"\nSummary:")
    print(f"  Total scrap: {scrap2.total_scrap_volume:.2f} in³ ({scrap2.total_scrap_weight:.2f} lbs)")
    print(f"  Scrap price: ${value2['scrap_price_per_lb']:.4f}/lb ({value2['price_source']})")
    print(f"  Scrap value: ${value2['scrap_value']:.2f}")
    print(f"  Material utilization: {scrap2.utilization_percentage:.1f}%")

    # Test 3: 17-4 PH Stainless
    print("\n\n3. 17-4 PH Stainless")
    print("-" * 70)

    result3 = calculate_total_scrap_with_value(
        cad_file,
        material="17-4 PH Stainless",
        verbose=True
    )

    scrap3 = result3['scrap_info']
    value3 = result3['scrap_value_info']

    print(f"\nSummary:")
    print(f"  Total scrap: {scrap3.total_scrap_volume:.2f} in³ ({scrap3.total_scrap_weight:.2f} lbs)")
    print(f"  Scrap price: ${value3['scrap_price_per_lb']:.4f}/lb ({value3['price_source']})")
    print(f"  Scrap value: ${value3['scrap_value']:.2f}")
    print(f"  Material utilization: {scrap3.utilization_percentage:.1f}%")

    print("\n" + "="*70 + "\n")


def test_standalone_scrap_value():
    """Test standalone scrap value calculation."""

    print("\n" + "="*70)
    print("STANDALONE SCRAP VALUE TEST")
    print("="*70 + "\n")

    materials = [
        ("aluminum MIC6", 21.19),
        ("P20 Tool Steel", 59.88),
        ("17-4 PH Stainless", 59.29),
        ("Copper", 30.0),
        ("Brass", 25.0),
    ]

    print(f"{'Material':<25} {'Weight (lbs)':<15} {'Price ($/lb)':<15} {'Value ($)':<15}")
    print("-" * 70)

    for material, weight in materials:
        value_info = calculate_scrap_value(weight, material, verbose=False)

        if value_info['scrap_price_per_lb']:
            price_str = f"${value_info['scrap_price_per_lb']:.4f}"
            value_str = f"${value_info['scrap_value']:.2f}"
        else:
            price_str = "N/A"
            value_str = "N/A"

        print(f"{material:<25} {weight:<15.2f} {price_str:<15} {value_str:<15}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_scrap_value_calculation()
    test_standalone_scrap_value()
