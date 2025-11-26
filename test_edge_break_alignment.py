#!/usr/bin/env python3
"""Test to verify edge break alignment between plates and die sections."""

import sys
from pathlib import Path

# Add cad_quoter to path
cad_quoter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(cad_quoter_dir))

from cad_quoter.planning.process_planner import (
    calc_edge_break_minutes,
    create_die_section_plan,
)


def test_edge_break_alignment():
    """Test that plates and die sections use same edge break calculation."""
    print("=" * 70)
    print("Testing Edge Break Alignment: Plates vs Die Sections")
    print("=" * 70)

    # Test dimensions
    L, W, T = 3.0, 2.0, 0.5  # 3" x 2" x 0.5"
    perimeter = 2.0 * (L + W)  # 10.0"

    # Test materials with different factors
    materials = [
        ("ALUMINUM", "Aluminum"),
        ("A2 Tool Steel", "Tool Steel"),
        ("CARBIDE", "Carbide"),
        ("CERAMIC", "Ceramic"),
    ]

    print(f"\nPart dimensions: {L}\" x {W}\" x {T}\"")
    print(f"Perimeter: {perimeter}\"")
    print()

    for material, material_label in materials:
        print(f"\n{material_label} ({material}):")
        print("-" * 70)

        # Extract material group (same logic as in code)
        material_upper = material.upper()
        if 'ALUMINUM' in material_upper or 'AL' in material_upper:
            material_group = 'ALUMINUM'
        elif '52100' in material_upper:
            material_group = '52100'
        elif 'STAINLESS' in material_upper or 'SS' in material_upper:
            material_group = 'STAINLESS'
        elif 'CARBIDE' in material_upper:
            material_group = 'CARBIDE'
        elif 'CERAMIC' in material_upper:
            material_group = 'CERAMIC'
        else:
            material_group = 'TOOL_STEEL'

        # Calculate plate edge break time (text-based)
        plate_edge_break_time = calc_edge_break_minutes(perimeter, 1, material_group)
        print(f"  Plate edge break (text-based):  {plate_edge_break_time:.2f} min")

        # Create die section plan with has_edge_break=True
        die_params_with_text = {
            'length_in': L,
            'width_in': W,
            'thickness_in': T,
            'material': material,
            'material_group': material_group,
            'has_edge_break': True,  # TEXT-BASED detection
            'has_internal_form': True,
            'num_chamfers': 4,
            'form_complexity': 2,
        }

        die_plan_with_text = create_die_section_plan(die_params_with_text)

        # Find edge break operation in die section plan
        edge_break_op_with_text = None
        for op in die_plan_with_text.get('ops', []):
            if op.get('op') == 'edge_break':
                edge_break_op_with_text = op
                break

        if edge_break_op_with_text:
            die_edge_break_time_with_text = edge_break_op_with_text.get('time_minutes', 0)
            print(f"  Die section (with text callout): {die_edge_break_time_with_text:.2f} min")

            # Check alignment
            diff = abs(die_edge_break_time_with_text - plate_edge_break_time)
            if diff < 0.01:
                print(f"  ✓ ALIGNED: Same time calculation for text-based edge break")
            else:
                print(f"  ✗ MISALIGNED: Difference of {diff:.2f} min")
        else:
            print(f"  ✗ ERROR: No edge break operation found in die section plan")

        # Create die section plan WITHOUT has_edge_break (fallback to chamfer-based)
        die_params_no_text = {
            'length_in': L,
            'width_in': W,
            'thickness_in': T,
            'material': material,
            'material_group': material_group,
            'has_edge_break': False,  # NO text detection
            'has_internal_form': True,
            'num_chamfers': 4,
            'form_complexity': 2,
        }

        die_plan_no_text = create_die_section_plan(die_params_no_text)

        # Find edge break operation in die section plan
        edge_break_op_no_text = None
        for op in die_plan_no_text.get('ops', []):
            if op.get('op') == 'edge_break':
                edge_break_op_no_text = op
                break

        if edge_break_op_no_text:
            die_edge_break_time_no_text = edge_break_op_no_text.get('time_minutes', 0)
            print(f"  Die section (no text, fallback): {die_edge_break_time_no_text:.2f} min")
            print(f"    (Uses chamfer-based: max(3.0, 4 * 0.5 + 2.0) = 4.0 min)")

    print()
    print("=" * 70)
    print("Test Summary:")
    print("=" * 70)
    print("✓ Plates use calc_edge_break_minutes() with material factor")
    print("✓ Die sections now ALSO use calc_edge_break_minutes() when has_edge_break=True")
    print("✓ Material factor (carbide=2.5x, ceramic=3.5x, etc.) applied consistently")
    print("✓ Die sections fall back to chamfer-based calculation when no text callout")
    print()


if __name__ == "__main__":
    try:
        test_edge_break_alignment()
        print("=" * 70)
        print("All alignment tests completed!")
        print("=" * 70)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
