"""Acceptance tests for punch datum/square-up heuristics.

Tests the new punch planning logic:
- Round punches: OD + end faces only (no side square-up)
- Non-round/form punches: Grind_reference_faces before profiling
- Carbide punches: only grind + WEDM (no milling)
- Very small parts: prefer Grind_reference_faces over mill
"""

try:
    import pytest
except ImportError:
    pytest = None

from cad_quoter.planning.process_planner import (
    create_punch_plan,
    estimate_machine_hours_from_plan,
    render_punch_datum_block,
    render_punch_grind_block,
    _is_carbide,
)
from cad_quoter.pricing.time_estimator import (
    estimate_face_grind_minutes,
    estimate_wire_edm_minutes,
)


class TestRoundPierceSteel:
    """Test case: Round pierce punch in steel.

    Expected: emits OD_grind_rough → OD_grind_finish → Grind_length
    No square_up_* operations.
    """

    def test_round_steel_operations(self):
        """Round steel punch should emit OD grind and Grind_length ops."""
        params = {
            "family": "round_punch",
            "shape_type": "round",
            "overall_length_in": 3.0,
            "max_od_or_width_in": 0.500,
            "num_ground_diams": 2,
            "total_ground_length_in": 2.5,
            "material": "A2",
            "material_group": "",
            "has_flats": False,
            "has_wire_profile": False,
        }

        plan = create_punch_plan(params)
        ops = plan["ops"]
        op_names = [op["op"] for op in ops]

        # Should include OD grind operations
        assert "OD_grind_rough" in op_names, "Should include OD_grind_rough"
        assert "OD_grind_finish" in op_names, "Should include OD_grind_finish"
        assert "Grind_length" in op_names, "Should include Grind_length"

        # Should NOT include square_up operations
        assert "square_up_rough_sides" not in op_names, "Should not include square_up_rough_sides"
        assert "square_up_rough_faces" not in op_names, "Should not include square_up_rough_faces"

        # Should NOT include Grind_reference_faces for round punches
        assert "Grind_reference_faces" not in op_names, "Round punches should not have Grind_reference_faces"

    def test_round_steel_estimator(self):
        """Estimator should calculate grinding time for round steel punch."""
        from math import pi

        D = 0.500
        T = 3.0
        stock_radial = 0.003

        r = D / 2.0
        vol = pi * (r**2 - (r - stock_radial)**2) * T

        plan = {
            'ops': [
                {'op': 'OD_grind_rough', 'num_diams': 2, 'total_length_in': 2.5},
                {'op': 'OD_grind_finish', 'num_diams': 2, 'total_length_in': 2.5},
                {'op': 'Grind_length', 'diameter': D, 'length_in': T, 'stock_removed_total': 0.006, 'faces': 2},
            ],
            'meta': {
                'diameter': D,
                'thickness': T,
                'stock_allow_radial': stock_radial,
                'od_grind_volume_removed_cuin': vol,
            },
        }

        result = estimate_machine_hours_from_plan(plan, "A2", (D, D), T)

        # Should have grinding time
        assert result['breakdown_minutes']['grinding'] > 0, "Should have grinding time"
        # Should have no milling time
        assert result['breakdown_minutes']['milling'] == 0, "Should not have milling time"

    def test_round_steel_renderer(self):
        """Renderer should show OD/length grinds only for round steel punch."""
        grinding_ops = [
            {
                'op_name': 'od_grind_rough',
                'op_description': 'OD Grind - Rough',
                'num_diams': 2,
                'total_length_in': 2.5,
                'grind_material_factor': 1.0,
                'time_minutes': 0.5,
            },
            {
                'op_name': 'od_grind_finish',
                'op_description': 'OD Grind - Finish',
                'num_diams': 2,
                'total_length_in': 2.5,
                'grind_material_factor': 1.0,
                'time_minutes': 0.5,
            },
            {
                'op_name': 'grind_length',
                'op_description': 'Grind to Length',
                'length': 0.5,
                'width': 0.5,
                'stock_removed_total': 0.006,
                'grind_material_factor': 1.0,
                'time_minutes': 0.1,
            },
        ]

        lines = render_punch_grind_block(grinding_ops)
        output = "\n".join(lines)

        # Should show OD grind operations
        assert "OD Grind" in output, "Should show OD grind operations"
        assert "Grind Length" in output, "Should show length grind"

        # Should NOT show DATUM line (that's for non-round)
        datum_lines = render_punch_datum_block(grinding_ops)
        assert len(datum_lines) == 0, "Round punches should not have DATUM line"


class TestRoundCarbideWithFlats:
    """Test case: Round carbide punch with flats.

    Expected: includes Wire_EDM_profile before grind.
    Estimator uses wire perimeter × mins/in.
    """

    def test_carbide_with_flats_operations(self):
        """Round carbide punch with flats should include Wire_EDM_profile."""
        params = {
            "family": "round_punch",
            "shape_type": "round",
            "overall_length_in": 2.5,
            "max_od_or_width_in": 0.375,
            "num_ground_diams": 1,
            "total_ground_length_in": 2.0,
            "material": "CARBIDE",
            "material_group": "",
            "has_flats": True,
            "has_wire_profile": False,
        }

        plan = create_punch_plan(params)
        ops = plan["ops"]
        op_names = [op["op"] for op in ops]

        # Should include Wire_EDM_profile for flats
        assert "Wire_EDM_profile" in op_names, "Should include Wire_EDM_profile for flats"

        # Wire EDM should come before OD grind
        wire_idx = op_names.index("Wire_EDM_profile")
        if "OD_grind_rough" in op_names:
            od_idx = op_names.index("OD_grind_rough")
            assert wire_idx < od_idx, "Wire_EDM_profile should come before OD_grind_rough"

        # Should also include OD grind and length grind
        assert "OD_grind_rough" in op_names, "Should include OD_grind_rough"
        assert "Grind_length" in op_names, "Should include Grind_length"

    def test_carbide_with_flats_estimator(self):
        """Estimator should use wire perimeter for carbide with flats."""
        D = 0.375
        T = 2.5
        # Approximate perimeter for round with flats
        perimeter = 3.14159 * D + (D * 0.5)

        plan = {
            'ops': [
                {
                    'op': 'Wire_EDM_profile',
                    'wire_profile_perimeter_in': perimeter,
                    'thickness_in': T,
                },
                {'op': 'OD_grind_rough', 'num_diams': 1, 'total_length_in': 2.0},
                {'op': 'OD_grind_finish', 'num_diams': 1, 'total_length_in': 2.0},
                {'op': 'Grind_length', 'diameter': D, 'length_in': T, 'stock_removed_total': 0.006},
            ],
            'meta': {
                'diameter': D,
                'thickness': T,
            },
        }

        result = estimate_machine_hours_from_plan(plan, "CARBIDE", (D, D), T)

        # Should have EDM time
        assert result['breakdown_minutes']['edm'] > 0, "Should have EDM time for wire profile"
        # Should have grinding time
        assert result['breakdown_minutes']['grinding'] > 0, "Should have grinding time"


class TestNonRoundForm:
    """Test case: Non-round form punch.

    Expected: emits Grind_reference_faces → Wire_EDM_profile → Grind_length
    Renderer shows DATUM line.
    """

    def test_nonround_form_operations(self):
        """Non-round form punch should emit Grind_reference_faces before Wire_EDM_profile."""
        params = {
            "family": "form_punch",
            "shape_type": "rectangular",
            "overall_length_in": 3.0,
            "max_od_or_width_in": 1.5,
            "body_width_in": 1.5,
            "body_thickness_in": 1.0,
            "material": "D2",
            "material_group": "",
            "has_flats": False,
            "has_wire_profile": True,
            "wire_profile_perimeter_in": 9.0,
        }

        plan = create_punch_plan(params)
        ops = plan["ops"]
        op_names = [op["op"] for op in ops]

        # Should include Grind_reference_faces
        assert "Grind_reference_faces" in op_names, "Should include Grind_reference_faces"

        # Should include Wire_EDM_profile
        assert "Wire_EDM_profile" in op_names, "Should include Wire_EDM_profile"

        # Should include Grind_length
        assert "Grind_length" in op_names, "Should include Grind_length"

        # Grind_reference_faces should come before Wire_EDM_profile
        ref_idx = op_names.index("Grind_reference_faces")
        wire_idx = op_names.index("Wire_EDM_profile")
        assert ref_idx < wire_idx, "Grind_reference_faces should come before Wire_EDM_profile"

        # Should NOT include square_up operations
        assert "square_up_rough_sides" not in op_names
        assert "square_up_rough_faces" not in op_names

    def test_nonround_form_estimator(self):
        """Estimator should calculate times for non-round form punch."""
        plan = {
            'ops': [
                {
                    'op': 'Grind_reference_faces',
                    'stock_removed_total': 0.006,
                    'faces': 2,
                    'length_in': 3.0,
                    'width_in': 1.5,
                },
                {
                    'op': 'Wire_EDM_profile',
                    'wire_profile_perimeter_in': 9.0,
                    'thickness_in': 1.0,
                },
                {
                    'op': 'Grind_length',
                    'width_in': 1.5,
                    'thickness_in': 1.0,
                    'length_in': 3.0,
                    'stock_removed_total': 0.004,
                },
            ],
            'meta': {},
        }

        result = estimate_machine_hours_from_plan(plan, "D2", (1.5, 1.0), 3.0)

        # Should have EDM time
        assert result['breakdown_minutes']['edm'] > 0, "Should have EDM time"
        # Should have grinding time
        assert result['breakdown_minutes']['grinding'] > 0, "Should have grinding time"

        # Check that grinding_operations contains the datum op
        datum_ops = [op for op in result['grinding_operations']
                     if op.get('op_name') == 'grind_reference_faces']
        assert len(datum_ops) == 1, "Should have one Grind_reference_faces operation"
        assert datum_ops[0].get('is_datum') is True, "Datum op should be marked as datum"

    def test_nonround_form_renderer_datum_line(self):
        """Renderer should show DATUM line for non-round form punch."""
        grinding_ops = [
            {
                'op_name': 'grind_reference_faces',
                'op_description': 'Grind Reference Faces (Datum)',
                'length': 3.0,
                'width': 1.5,
                'stock_removed_total': 0.006,
                'faces': 2,
                'grind_material_factor': 1.2,
                'time_minutes': 0.324,
                'is_datum': True,
            },
            {
                'op_name': 'grind_length',
                'op_description': 'Grind to Length',
                'length': 1.5,
                'width': 1.0,
                'stock_removed_total': 0.004,
                'grind_material_factor': 1.2,
                'time_minutes': 0.216,
            },
        ]

        lines = render_punch_datum_block(grinding_ops)

        assert len(lines) > 0, "Should render DATUM line"
        output = lines[0]

        # Check DATUM line format
        assert "DATUM:" in output, "Should start with DATUM:"
        assert "Grind 2 faces" in output, "Should show number of faces"
        assert "factor" in output, "Should show grind factor"


class TestTinyNonRound:
    """Test case: Very small non-round punch (0.75" × 0.5").

    Expected: planner uses Grind_reference_faces (no mill).
    Estimator computes via face-grind rule.
    """

    def test_tiny_nonround_operations(self):
        """Tiny non-round punch should use Grind_reference_faces, not mill."""
        params = {
            "family": "form_punch",
            "shape_type": "rectangular",
            "overall_length_in": 2.0,
            "max_od_or_width_in": 0.75,
            "body_width_in": 0.75,
            "body_thickness_in": 0.5,
            "material": "A2",
            "material_group": "",
            "has_flats": False,
            "has_wire_profile": False,  # No wire profile needed
        }

        plan = create_punch_plan(params)
        ops = plan["ops"]
        op_names = [op["op"] for op in ops]

        # Should include Grind_reference_faces for tiny parts
        assert "Grind_reference_faces" in op_names, "Tiny parts should use Grind_reference_faces"

        # Should include Grind_length
        assert "Grind_length" in op_names, "Should include Grind_length"

        # Should NOT include milling operations for datum
        assert "square_up_rough_sides" not in op_names, "Tiny parts should not use mill for datum"
        assert "square_up_rough_faces" not in op_names, "Tiny parts should not use mill for datum"

    def test_tiny_nonround_estimator_face_grind_rule(self):
        """Estimator should use face-grind rule for tiny non-round punch."""
        L = 2.0
        W = 0.75
        T = 0.5
        stock = 0.006

        plan = {
            'ops': [
                {
                    'op': 'Grind_reference_faces',
                    'stock_removed_total': stock,
                    'faces': 2,
                    'length_in': L,
                    'width_in': W,
                },
                {
                    'op': 'Grind_length',
                    'width_in': W,
                    'thickness_in': T,
                    'length_in': L,
                    'stock_removed_total': 0.004,
                },
            ],
            'meta': {},
        }

        result = estimate_machine_hours_from_plan(plan, "A2", (W, T), L)

        # Should have grinding time
        assert result['breakdown_minutes']['grinding'] > 0, "Should have grinding time"

        # Verify the formula: minutes = (L×W×stock_removed_total) × 3.0 × grind_factor
        # For A2, factor should be ~1.0-1.2
        expected_volume = L * W * stock
        min_expected = expected_volume * 3.0 * 0.8  # Allow some margin
        max_expected = expected_volume * 3.0 * 2.0  # Allow some margin

        # Get the datum op time specifically
        datum_ops = [op for op in result['grinding_operations']
                     if op.get('op_name') == 'grind_reference_faces']
        if datum_ops:
            datum_time = datum_ops[0].get('time_minutes', 0)
            assert min_expected <= datum_time <= max_expected, \
                f"Datum grind time {datum_time} should be based on face-grind formula"


class TestCarbideDisallowsMilling:
    """Test that carbide punches do not use milling for datum faces."""

    def test_is_carbide_helper(self):
        """Test _is_carbide helper function."""
        assert _is_carbide("CARBIDE", "") is True
        assert _is_carbide("carbide", "") is True
        assert _is_carbide("A2", "") is False
        assert _is_carbide("A2", "CARBIDE") is True
        assert _is_carbide("", "Carbide Group") is True

    def test_carbide_nonround_uses_grind_datum(self):
        """Carbide non-round punch should use Grind_reference_faces, not mill."""
        params = {
            "family": "form_punch",
            "shape_type": "rectangular",
            "overall_length_in": 2.0,
            "max_od_or_width_in": 1.5,  # Not tiny
            "body_width_in": 1.5,
            "body_thickness_in": 1.0,
            "material": "CARBIDE",
            "material_group": "",
            "has_flats": False,
            "has_wire_profile": True,
        }

        plan = create_punch_plan(params)
        ops = plan["ops"]
        op_names = [op["op"] for op in ops]

        # Carbide should use Grind_reference_faces even if not tiny
        assert "Grind_reference_faces" in op_names, "Carbide should use Grind_reference_faces"

        # Should NOT include milling operations
        assert "square_up_rough_sides" not in op_names
        assert "square_up_rough_faces" not in op_names


class TestGrindReferencesFacesSchema:
    """Test that Grind_reference_faces operation follows the expected schema."""

    def test_grind_reference_faces_schema(self):
        """Grind_reference_faces should have correct schema."""
        params = {
            "family": "form_punch",
            "shape_type": "rectangular",
            "overall_length_in": 3.0,
            "max_od_or_width_in": 0.75,
            "body_width_in": 0.75,
            "body_thickness_in": 0.5,
            "material": "A2",
            "has_wire_profile": False,
        }

        plan = create_punch_plan(params)

        # Find Grind_reference_faces op
        grind_ref_ops = [op for op in plan["ops"] if op["op"] == "Grind_reference_faces"]
        assert len(grind_ref_ops) == 1, "Should have exactly one Grind_reference_faces op"

        op = grind_ref_ops[0]

        # Check required fields per spec
        assert op.get("stock_removed_total") == 0.006, "Should have stock_removed_total of 0.006"
        assert op.get("faces") == 2, "Should have faces=2"
        assert "note" in op, "Should have note field"
        assert "datum" in op["note"].lower(), "Note should mention datum"


def test_renderer_integration():
    """Test that renderer functions are properly integrated."""
    # This is a smoke test to ensure the functions can be imported and called

    grinding_ops = [
        {
            'op_name': 'grind_reference_faces',
            'length': 2.0,
            'width': 1.0,
            'stock_removed_total': 0.006,
            'faces': 2,
            'grind_material_factor': 1.0,
            'time_minutes': 0.1,
            'is_datum': True,
        }
    ]

    # These should not raise
    datum_lines = render_punch_datum_block(grinding_ops)
    grind_lines = render_punch_grind_block(grinding_ops)

    assert isinstance(datum_lines, list)
    assert isinstance(grind_lines, list)


def run_tests():
    """Run all tests manually if pytest is not available."""
    import traceback

    test_classes = [
        TestRoundPierceSteel,
        TestRoundCarbideWithFlats,
        TestNonRoundForm,
        TestTinyNonRound,
        TestCarbideDisallowsMilling,
        TestGrindReferencesFacesSchema,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                try:
                    method()
                    print(f"  PASS: {cls.__name__}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {cls.__name__}.{method_name}")
                    print(f"        {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ERROR: {cls.__name__}.{method_name}")
                    print(f"        {e}")
                    traceback.print_exc()
                    failed += 1

    # Run standalone test functions
    standalone_tests = [test_renderer_integration]
    for test_func in standalone_tests:
        try:
            test_func()
            print(f"  PASS: {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_func.__name__}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_func.__name__}")
            print(f"        {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    if pytest:
        pytest.main([__file__, "-v"])
    else:
        print("=" * 60)
        print("PUNCH DATUM HEURISTICS - ACCEPTANCE TESTS")
        print("=" * 60)
        print()
        success = run_tests()
        import sys
        sys.exit(0 if success else 1)
