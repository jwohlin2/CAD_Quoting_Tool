#!/usr/bin/env python3
"""Test script for waterjet metrics calculation from hole table."""

import math
import sys
from pathlib import Path

# Add cad_quoter to path
sys.path.insert(0, str(Path(__file__).parent))

from cad_quoter.planning.process_planner import calc_waterjet_metrics_from_hole_table

# Example hole table from user (matching the provided data)
hole_table = [
    {'HOLE': 'A', 'REF_DIAM': 'Ø1.5048', 'QTY': 4},
    {'HOLE': 'B', 'REF_DIAM': 'Ø.5313', 'QTY': 2},
    {'HOLE': 'C', 'REF_DIAM': 'Ø1/2', 'QTY': 14},
    {'HOLE': 'D', 'REF_DIAM': 'Ø.2610', 'QTY': 12},
    {'HOLE': 'E', 'REF_DIAM': 'Ø9/32', 'QTY': 12},
    {'HOLE': 'F', 'REF_DIAM': 'Ø7/32', 'QTY': 1},
    {'HOLE': 'G', 'REF_DIAM': 'Ø3/16', 'QTY': 2},
    {'HOLE': 'H', 'REF_DIAM': 'Ø.1875', 'QTY': 33},
    {'HOLE': 'J', 'REF_DIAM': 'Ø.1875', 'QTY': 4},
    {'HOLE': 'K', 'REF_DIAM': 'Ø1/4', 'QTY': 4},
    {'HOLE': 'L', 'REF_DIAM': 'Ø.1360', 'QTY': 14},
    {'HOLE': 'M', 'REF_DIAM': 'Ø.25', 'QTY': 12},
    {'HOLE': 'N', 'REF_DIAM': 'Ø.1250', 'QTY': 2},
]

# Expected values from user's example
expected_pierce_count = 116
expected_total_length = 107.675103688769

print("Testing waterjet metrics calculation...")
print("=" * 60)

# Calculate metrics
metrics = calc_waterjet_metrics_from_hole_table(hole_table)

print(f"\nResults:")
print(f"  Pierce count: {metrics['pierce_count']}")
print(f"  Total length: {metrics['total_length_in']:.12f} inches")

print(f"\nExpected (from user example):")
print(f"  Pierce count: {expected_pierce_count}")
print(f"  Total length: {expected_total_length:.12f} inches")

print(f"\nComparison:")
pierce_match = metrics['pierce_count'] == expected_pierce_count
length_diff = abs(metrics['total_length_in'] - expected_total_length)
length_match = length_diff < 0.001  # Within 0.001 inches

print(f"  Pierce count match: {'✓' if pierce_match else '✗'}")
print(f"  Total length match: {'✓' if length_match else '✗'} (diff: {length_diff:.6f}\")")

# Show detailed breakdown
print("\n" + "=" * 60)
print("Detailed breakdown:")
print("=" * 60)
print(f"{'HOLE':<6} {'Diameter':<10} {'QTY':<5} {'Circumference':<15} {'QTY × Circ':<15}")
print("-" * 60)

total_check = 0.0
qty_check = 0

for entry in hole_table:
    hole = entry['HOLE']
    ref_diam = entry['REF_DIAM']
    qty = entry['QTY']

    # Parse diameter
    diam_str = ref_diam.replace('Ø', '').replace('∅', '').strip()
    if '/' in diam_str:
        from fractions import Fraction
        diameter = float(Fraction(diam_str))
    else:
        diameter = float(diam_str)

    circumference = math.pi * diameter
    qty_x_circ = circumference * qty

    print(f"{hole:<6} {diameter:<10.4f} {qty:<5} {circumference:<15.9f} {qty_x_circ:<15.9f}")

    total_check += qty_x_circ
    qty_check += qty

print("-" * 60)
print(f"{'TOTALS':<6} {'':<10} {qty_check:<5} {'':<15} {total_check:<15.9f}")
print("=" * 60)

if pierce_match and length_match:
    print("\n✓ All tests passed!")
    sys.exit(0)
else:
    print("\n✗ Tests failed!")
    sys.exit(1)
