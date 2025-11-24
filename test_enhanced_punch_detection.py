#!/usr/bin/env python3
"""
Test enhanced punch detection heuristics.

This script demonstrates how the new multi-heuristic detection system
identifies punches vs plates using text, geometry, and feature signals.
"""

from pathlib import Path
import sys

sys.path.insert(0, '/home/user/CAD_Quoting_Tool')

from cad_quoter.pricing.QuoteDataHelper import detect_punch_drawing


def test_detection(name: str, filename: str, text_dump: str = None, plan: dict = None, expected: bool = None):
    """Test punch detection with given inputs."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Filename: {filename}")

    # Temporarily enable debug output
    import cad_quoter.pricing.QuoteDataHelper as qdh
    original_debug = False

    # Monkey-patch to enable debug for this test
    original_code = qdh.detect_punch_drawing.__code__

    cad_path = Path(filename)
    result = detect_punch_drawing(cad_path, text_dump, plan)

    print(f"\n{'RESULT:':<12} {'PUNCH ✓' if result else 'PLATE ✗'}")
    if expected is not None:
        status = "PASS ✓" if result == expected else "FAIL ✗"
        print(f"{'EXPECTED:':<12} {'PUNCH' if expected else 'PLATE'} → {status}")

    return result


def run_tests():
    """Run comprehensive detection tests."""

    print("\n" + "="*70)
    print("ENHANCED PUNCH DETECTION TEST SUITE")
    print("="*70)

    # ========================================================================
    # Test 1: Text-based detection (filename)
    # ========================================================================

    test_detection(
        name="Filename with PUNCH keyword",
        filename="T1769-PUNCH.dwg",
        expected=True
    )

    test_detection(
        name="Filename with PILOT keyword",
        filename="316A-PILOT-PIN.dwg",
        expected=True
    )

    test_detection(
        name="Filename with PLATE (exclusion)",
        filename="T1769-PUNCH-PLATE.dwg",
        expected=False
    )

    # ========================================================================
    # Test 2: Text-based detection (drawing text)
    # ========================================================================

    test_detection(
        name="Text contains 'FORM PUNCH'",
        filename="T1769-134.dwg",
        text_dump="MATERIAL: A2 TOOL STEEL\nPART NAME: FORM PUNCH\nHARDNESS: 58-62 HRC",
        expected=True
    )

    test_detection(
        name="Text contains 'PUNCH PLATE' (exclusion)",
        filename="T1769-201.dwg",
        text_dump="MATERIAL: A2\nPART NAME: PUNCH PLATE\nTHICKNESS: 0.500",
        expected=False
    )

    # ========================================================================
    # Test 3: Part number detection (NEW)
    # ========================================================================

    test_detection(
        name="Part number callout (PART 2)",
        filename="T1769-134.dwg",
        text_dump="QTY: 2\nPART 2\nMATERIAL: D2",
        expected=True
    )

    test_detection(
        name="Multiple small part numbers",
        filename="T1769-219.dwg",
        text_dump="DETAIL 6\nITEM 7\nPART 14\nQTY: 1 EA",
        expected=True
    )

    # ========================================================================
    # Test 4: Geometry-based detection (NEW)
    # ========================================================================

    # Round punch: 1.5" diameter × 6" long
    test_detection(
        name="Round punch geometry (1.5\" × 1.5\" × 6\")",
        filename="T1769-134.dwg",
        plan={
            'extracted_dims': {'L': 6.0, 'W': 1.5, 'T': 1.5},
        },
        expected=True
    )

    # Small round pin: 0.25" diameter × 2" long
    test_detection(
        name="Small pilot pin (0.25\" × 0.25\" × 2\")",
        filename="T1769-326.dwg",
        plan={
            'extracted_dims': {'L': 2.0, 'W': 0.25, 'T': 0.25},
        },
        expected=True
    )

    # Plate: 10" × 8" × 0.5"
    test_detection(
        name="Plate geometry (10\" × 8\" × 0.5\")",
        filename="T1769-201.dwg",
        plan={
            'extracted_dims': {'L': 10.0, 'W': 8.0, 'T': 0.5},
        },
        expected=False
    )

    # ========================================================================
    # Test 5: Feature-based detection (NEW)
    # ========================================================================

    # Simple punch: turned part with grinding, no holes
    test_detection(
        name="Turned part with grinding, no holes",
        filename="T1769-219.dwg",
        plan={
            'extracted_dims': {'L': 4.0, 'W': 1.2, 'T': 1.2},
            'ops': [
                {'op': 'rough_turn'},
                {'op': 'finish_turn'},
                {'op': 'od_grind'},
            ],
            'hole_sets': [],
            'windows': [],
        },
        expected=True
    )

    # Plate with EDM windows and many holes
    test_detection(
        name="Plate with EDM windows and many holes",
        filename="T1769-201.dwg",
        plan={
            'extracted_dims': {'L': 10.0, 'W': 8.0, 'T': 0.5},
            'ops': [
                {'op': 'square_up_mill'},
                {'op': 'drill_patterns'},
                {'op': 'wedm_windows'},
            ],
            'hole_sets': [
                {'qty': 88},
            ],
            'windows': [
                {'perimeter': 12.5},
                {'perimeter': 8.3},
            ],
        },
        expected=False
    )

    # ========================================================================
    # Test 6: Combined signals (realistic scenarios)
    # ========================================================================

    # Guide post: no text keywords, but geometry + features suggest punch
    test_detection(
        name="Guide post (geometry + features, no text)",
        filename="T1769-104.dwg",
        plan={
            'extracted_dims': {'L': 8.5, 'W': 1.75, 'T': 1.75},  # Round, L/D = 4.9
            'ops': [
                {'op': 'rough_turn'},
                {'op': 'finish_turn'},
                {'op': 'od_grind'},
                {'op': 'face_grind'},
            ],
            'hole_sets': [{'qty': 1}],  # One tapped hole
            'windows': [],
        },
        expected=True
    )

    # Form punch: part number + geometry
    test_detection(
        name="Form punch (part # + geometry)",
        filename="T1769-326.dwg",
        text_dump="PART 14\nMATERIAL: A2\nHARDNESS: 58-60 HRC",
        plan={
            'extracted_dims': {'L': 3.2, 'W': 0.75, 'T': 0.75},
            'ops': [
                {'op': 'rough_turn'},
                {'op': 'finish_turn'},
            ],
            'hole_sets': [],
            'windows': [],
        },
        expected=True
    )

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nThe enhanced detection uses confidence scoring:")
    print("  • Text signals: +2 to +5 points")
    print("  • Geometry signals: +1 to +2 points each")
    print("  • Feature signals: +1 point or -2/-3 points")
    print("  • Threshold: ≥3 points → classify as PUNCH")
    print()


if __name__ == "__main__":
    run_tests()
