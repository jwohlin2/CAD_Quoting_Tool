"""
Unit tests for the simple-part setup time guardrail.

Tests the logic that prevents excessive setup time relative to machine time
for simple parts (few ops, few holes, no complex operations).
"""

import pytest
from cad_quoter.planning.process_planner import LaborInputs, setup_minutes


class TestSetupGuardrail:
    """Tests for the setup time guardrail on simple parts."""

    def test_simple_part_setup_reduction(self):
        """
        Test that setup time is reduced for simple parts with high setup/machine ratio.

        Part 139 scenario: 6 thru-holes, machine time 7.22 min
        Without guardrail: ~53 min setup
        With guardrail: should reduce to ~21.66 min (3 × 7.22)
        """
        # Simulate part 139: 6 ops, 6 holes, machine time 7.22 min
        # tool_changes = len(ops) * 2 = 12
        # Raw setup = 15 + 3*12 + 8*1 = 15 + 36 + 8 = 59 min
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=6,
            tool_changes=12,  # 6 ops × 2
            fixturing_complexity=1,
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Should be reduced to 3 × 7.22 = 21.66 min
        # (since raw 59 min > 4 × 7.22 = 28.88 min)
        expected_setup = 3.0 * 7.22  # 21.66 min
        assert abs(setup - expected_setup) < 0.01, \
            f"Expected {expected_setup:.2f} min, got {setup:.2f} min"

    def test_simple_part_no_reduction_when_below_threshold(self):
        """Test that setup time is NOT reduced when below 4× machine time."""
        # 8 ops, 8 holes, machine time 20 min
        # Raw setup = 15 + 3*16 + 8*1 = 71 min
        # 4 × 20 = 80 min, so 71 < 80, no reduction
        labor_inputs = LaborInputs(
            ops_total=8,
            holes_total=8,
            tool_changes=16,
            fixturing_complexity=1,
            machine_time_minutes=20.0
        )

        setup = setup_minutes(labor_inputs)

        # Raw setup = 15 + 48 + 8 = 71 min, should not be reduced
        expected_raw = 15 + 3*16 + 8*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_complex_part_no_reduction(self):
        """Test that complex parts (with EDM, jig grind, etc.) are not reduced."""
        # Part with EDM operations - should NOT be considered simple
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=6,
            tool_changes=12,
            fixturing_complexity=1,
            edm_window_count=2,  # This makes it complex
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Raw setup = 15 + 36 + 8 + 6 = 65 min
        # Should NOT be reduced because edm_window_count > 0
        expected_raw = 15 + 3*12 + 8*1 + 3*2
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_many_holes_not_simple(self):
        """Test that parts with many holes (>12) are not considered simple."""
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=15,  # More than 12 holes
            tool_changes=12,
            fixturing_complexity=1,
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Raw setup = 15 + 36 + 8 = 59 min
        # Should NOT be reduced because holes > 12
        expected_raw = 15 + 3*12 + 8*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_many_ops_not_simple(self):
        """Test that parts with many ops (>8) are not considered simple."""
        labor_inputs = LaborInputs(
            ops_total=10,  # More than 8 ops
            holes_total=6,
            tool_changes=20,
            fixturing_complexity=1,
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Raw setup = 15 + 60 + 8 = 83 min
        # Should NOT be reduced because ops > 8
        expected_raw = 15 + 3*20 + 8*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_setup_floor_minimum(self):
        """Test that setup never goes below the 10-minute floor."""
        # Very short machine time - setup should be at least 10 min
        labor_inputs = LaborInputs(
            ops_total=2,
            holes_total=2,
            tool_changes=4,
            fixturing_complexity=1,
            machine_time_minutes=2.0  # Very short
        )

        setup = setup_minutes(labor_inputs)

        # 3 × 2.0 = 6.0 min, but floor is 10 min
        # Raw setup = 15 + 12 + 8 = 35 min > 4 × 2 = 8 min
        # So it should reduce to max(10, 6) = 10 min
        assert setup == 10.0, \
            f"Expected 10 min floor, got {setup} min"

    def test_no_machine_time_no_reduction(self):
        """Test that no reduction is applied when machine time is not provided."""
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=6,
            tool_changes=12,
            fixturing_complexity=1,
            machine_time_minutes=0.0  # Not provided
        )

        setup = setup_minutes(labor_inputs)

        # Raw setup = 15 + 36 + 8 = 59 min
        # Should NOT be reduced because machine_time = 0
        expected_raw = 15 + 3*12 + 8*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_jig_grind_not_simple(self):
        """Test that parts with jig grinding are not considered simple."""
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=6,
            tool_changes=12,
            fixturing_complexity=1,
            jig_grind_bore_qty=1,  # Jig grinding
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Should NOT be reduced
        expected_raw = 15 + 3*12 + 8*1 + 3*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_thread_mill_not_simple(self):
        """Test that parts with thread milling are not considered simple."""
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=6,
            tool_changes=12,
            fixturing_complexity=1,
            thread_mill=1,
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Should NOT be reduced
        expected_raw = 15 + 3*12 + 8*1 + 2*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_npt_tapping_not_simple(self):
        """Test that parts with NPT tapping are not considered simple."""
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=6,
            tool_changes=12,
            fixturing_complexity=1,
            tap_npt=1,
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Should NOT be reduced
        expected_raw = 15 + 3*12 + 8*1 + 3*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"

    def test_outsource_not_simple(self):
        """Test that parts with outsource touches are not considered simple."""
        labor_inputs = LaborInputs(
            ops_total=6,
            holes_total=6,
            tool_changes=12,
            fixturing_complexity=1,
            outsource_touches=1,  # Heat treat, coating, etc.
            machine_time_minutes=7.22
        )

        setup = setup_minutes(labor_inputs)

        # Should NOT be reduced
        expected_raw = 15 + 3*12 + 8*1 + 6*1
        assert setup == expected_raw, \
            f"Expected raw setup {expected_raw} min, got {setup} min"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
