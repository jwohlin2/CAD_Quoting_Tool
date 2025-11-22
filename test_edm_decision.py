#!/usr/bin/env python3
"""Test script for GRIND vs WIRE EDM decision logic."""

from cad_quoter.planning.process_planner import determine_profile_process, _normalize_material_group

def test_edm_decision_logic():
    """Test the profile process decision tree."""
    print("Testing GRIND vs WIRE EDM Decision Logic")
    print("=" * 60)

    # Test 1: Carbide material → WIRE EDM
    print("\nTest 1: Carbide material")
    mat_group = _normalize_material_group("CARBIDE", "H1")
    result = determine_profile_process(
        material_group=mat_group,
        smallest_inside_radius=0.050,
        has_undercut=False,
        min_section_width=0.5,
        overall_height=2.0,
        num_radius_dims=1,
        num_sc_dims=0,
        num_chord_dims=0
    )
    print(f"  Material: CARBIDE (H1) → {result}")
    assert result == "wire_edm", f"Expected 'wire_edm', got '{result}'"
    print("  ✓ PASS")

    # Test 2: Small inside radius < 0.020" → WIRE EDM
    print("\nTest 2: Small inside radius (0.015\")")
    mat_group = _normalize_material_group("A2", "P2")
    result = determine_profile_process(
        material_group=mat_group,
        smallest_inside_radius=0.015,
        has_undercut=False,
        min_section_width=0.5,
        overall_height=2.0,
        num_radius_dims=1,
        num_sc_dims=0,
        num_chord_dims=0
    )
    print(f"  Radius: 0.015\" → {result}")
    assert result == "wire_edm", f"Expected 'wire_edm', got '{result}'"
    print("  ✓ PASS")

    # Test 3: Undercut → WIRE EDM
    print("\nTest 3: Has undercut")
    result = determine_profile_process(
        material_group="P2",
        smallest_inside_radius=0.050,
        has_undercut=True,
        min_section_width=0.5,
        overall_height=2.0,
        num_radius_dims=1,
        num_sc_dims=0,
        num_chord_dims=0
    )
    print(f"  Has undercut: True → {result}")
    assert result == "wire_edm", f"Expected 'wire_edm', got '{result}'"
    print("  ✓ PASS")

    # Test 4: Slender section (min_width < 0.10 * height) → WIRE EDM
    print("\nTest 4: Slender section")
    result = determine_profile_process(
        material_group="P2",
        smallest_inside_radius=0.050,
        has_undercut=False,
        min_section_width=0.15,  # 0.15 < 0.10 * 2.0 = 0.20
        overall_height=2.0,
        num_radius_dims=1,
        num_sc_dims=0,
        num_chord_dims=0
    )
    print(f"  Min width: 0.15\", Height: 2.0\" → {result}")
    assert result == "wire_edm", f"Expected 'wire_edm', got '{result}'"
    print("  ✓ PASS")

    # Test 5: High complexity (feature_count > 2) → WIRE EDM
    print("\nTest 5: High complexity (3 radius dims)")
    result = determine_profile_process(
        material_group="P2",
        smallest_inside_radius=0.050,
        has_undercut=False,
        min_section_width=0.5,
        overall_height=2.0,
        num_radius_dims=3,
        num_sc_dims=0,
        num_chord_dims=0
    )
    print(f"  Feature count: 3 → {result}")
    assert result == "wire_edm", f"Expected 'wire_edm', got '{result}'"
    print("  ✓ PASS")

    # Test 6: Simple profile → GRIND
    print("\nTest 6: Simple profile (should use grinding)")
    result = determine_profile_process(
        material_group="P2",
        smallest_inside_radius=0.050,
        has_undercut=False,
        min_section_width=0.5,
        overall_height=2.0,
        num_radius_dims=1,
        num_sc_dims=0,
        num_chord_dims=0
    )
    print(f"  Simple profile (1 radius dim, no special features) → {result}")
    assert result == "grind", f"Expected 'grind', got '{result}'"
    print("  ✓ PASS")

    # Test 7: Ceramic material → WIRE EDM
    print("\nTest 7: Ceramic material")
    mat_group = _normalize_material_group("CERAMIC", "C1")
    result = determine_profile_process(
        material_group=mat_group,
        smallest_inside_radius=0.050,
        has_undercut=False,
        min_section_width=0.5,
        overall_height=2.0,
        num_radius_dims=1,
        num_sc_dims=0,
        num_chord_dims=0
    )
    print(f"  Material: CERAMIC (C1) → {result}")
    assert result == "wire_edm", f"Expected 'wire_edm', got '{result}'"
    print("  ✓ PASS")

    print("\n" + "=" * 60)
    print("All tests PASSED! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_edm_decision_logic()
