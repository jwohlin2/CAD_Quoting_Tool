"""
Test hole table pipeline enforcement.

Tests that:
1. Every tap operation has a matching pre-drill group (at minor diameter)
2. Every jig-grind operation has a matching pre-drill group
3. Correct diameter references are shown (tap nominal vs minor)
4. No operations are ignored (all holes have time > 0)
"""
import pytest
from cad_quoter.planning.process_planner import estimate_hole_table_times
from cad_quoter.geometry.hole_table_parser import THREAD_MAJOR_DIAMETERS


class TestTapPredrill:
    """Test that tap operations get proper pre-drill groups."""

    def test_tap_gets_predrill_at_minor_diameter(self):
        """Test that a tap operation creates a pre-drill at minor diameter."""
        # 4-40 TAP example
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.112',  # #4 major diameter
                'QTY': 6,
                'DESCRIPTION': '4-40 TAP X .50 DEEP'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="6061", thickness=0.5)

        # Should have both drill and tap groups
        assert len(times['drill_groups']) >= 1, "Expected at least one drill group for tap pre-drill"
        assert len(times['tap_groups']) == 1, "Expected exactly one tap group"

        # Check pre-drill diameter
        # For 4-40: major = 0.112", TPI = 40
        # Minor (tap drill) = major - (1/TPI) = 0.112 - 0.025 = 0.087"
        drill_group = times['drill_groups'][0]
        expected_minor_dia = THREAD_MAJOR_DIAMETERS.get('#4', 0.112) - (1.0 / 40)
        assert abs(drill_group['diameter'] - expected_minor_dia) < 0.002, \
            f"Expected tap drill diameter ~{expected_minor_dia:.4f}\", got {drill_group['diameter']:.4f}\""

        # Check that drill group has correct quantity
        assert drill_group['qty'] == 6, "Tap pre-drill should have same qty as tap"

        # Check that tap group uses tap diameter (major)
        tap_group = times['tap_groups'][0]
        assert abs(tap_group['diameter'] - THREAD_MAJOR_DIAMETERS.get('#4', 0.112)) < 0.002, \
            "Tap group should use tap major diameter"

    def test_fractional_tap_gets_predrill(self):
        """Test that a fractional tap (e.g., 5/16-18) gets proper pre-drill."""
        hole_table = [
            {
                'HOLE': 'B',
                'REF_DIAM': 'Ø5/16',  # 5/16 = 0.3125"
                'QTY': 4,
                'DESCRIPTION': '5/16-18 TAP THRU'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="steel", thickness=0.5)

        # Should have both drill and tap groups
        assert len(times['drill_groups']) >= 1
        assert len(times['tap_groups']) == 1

        # For 5/16-18: major = 0.3125", TPI = 18
        # Minor (tap drill) = 0.3125 - (1/18) = 0.3125 - 0.0556 = 0.2569"
        drill_group = times['drill_groups'][0]
        expected_minor_dia = 0.3125 - (1.0 / 18)
        assert abs(drill_group['diameter'] - expected_minor_dia) < 0.002, \
            f"Expected tap drill diameter ~{expected_minor_dia:.4f}\", got {drill_group['diameter']:.4f}\""

    def test_multiple_taps_get_multiple_predrills(self):
        """Test that multiple tap operations each get their own pre-drill."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.112',
                'QTY': 6,
                'DESCRIPTION': '4-40 TAP X .50 DEEP'
            },
            {
                'HOLE': 'B',
                'REF_DIAM': 'Ø5/16',
                'QTY': 4,
                'DESCRIPTION': '5/16-18 TAP THRU'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="steel", thickness=0.5)

        # Should have 2 drill groups (one for each tap) and 2 tap groups
        assert len(times['drill_groups']) >= 2, "Expected at least 2 drill groups for 2 taps"
        assert len(times['tap_groups']) == 2, "Expected 2 tap groups"

    def test_tap_predrill_in_text_table(self):
        """Test that tap pre-drill operations appear in drill time breakdown."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.112',
                'QTY': 6,
                'DESCRIPTION': '4-40 TAP X .50 DEEP'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="6061", thickness=0.5)

        # Check that drill group has non-zero time
        drill_group = times['drill_groups'][0]
        assert drill_group['time_per_hole'] > 0, "Tap pre-drill should have non-zero time"
        assert drill_group['total_time'] > 0, "Tap pre-drill total time should be non-zero"

        # Check that total drill minutes includes tap pre-drill time
        assert times['total_drill_minutes'] > 0, "Total drill minutes should include tap pre-drill"


class TestJigGrindPredrill:
    """Test that jig-grind operations get proper pre-drill groups."""

    def test_jig_grind_gets_predrill(self):
        """Test that a jig-grind operation creates a pre-drill."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.2500',
                'QTY': 2,
                'DESCRIPTION': 'THRU (JIG GRIND)'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="A2", thickness=0.5)

        # Should have both drill and jig_grind groups
        assert len(times['drill_groups']) >= 1, "Expected at least one drill group for jig-grind pre-drill"
        assert len(times['jig_grind_groups']) == 1, "Expected exactly one jig-grind group"

        # Check pre-drill diameter (should be slightly under final diameter)
        # Typically 0.005-0.010" under final diameter
        drill_group = times['drill_groups'][0]
        final_diameter = 0.2500
        expected_predrill_undersize = 0.007  # Default undersize
        expected_predrill_dia = final_diameter - expected_predrill_undersize
        assert abs(drill_group['diameter'] - expected_predrill_dia) < 0.002, \
            f"Expected jig-grind pre-drill ~{expected_predrill_dia:.4f}\", got {drill_group['diameter']:.4f}\""

        # Check that drill group has correct quantity
        assert drill_group['qty'] == 2, "Jig-grind pre-drill should have same qty as jig-grind"

    def test_jig_grind_predrill_depth_matches(self):
        """Test that jig-grind pre-drill has same depth as final grind."""
        hole_table = [
            {
                'HOLE': 'B',
                'REF_DIAM': 'Ø.375',
                'QTY': 4,
                'DESCRIPTION': 'THRU (JIG GRIND)'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="D2", thickness=0.75)

        # Check that pre-drill depth matches jig-grind depth
        drill_group = times['drill_groups'][0]
        jig_grind_group = times['jig_grind_groups'][0]
        assert abs(drill_group['depth'] - jig_grind_group['depth']) < 0.01, \
            "Jig-grind pre-drill depth should match final grind depth"

    def test_multiple_jig_grinds_get_multiple_predrills(self):
        """Test that multiple jig-grind operations each get their own pre-drill."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.2500',
                'QTY': 2,
                'DESCRIPTION': 'THRU (JIG GRIND)'
            },
            {
                'HOLE': 'B',
                'REF_DIAM': 'Ø.375',
                'QTY': 3,
                'DESCRIPTION': 'THRU (JIG GRIND)'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="carbide", thickness=0.5)

        # Should have 2 drill groups (one for each jig-grind) and 2 jig-grind groups
        assert len(times['drill_groups']) >= 2, "Expected at least 2 drill groups for 2 jig-grinds"
        assert len(times['jig_grind_groups']) == 2, "Expected 2 jig-grind groups"


class TestMixedOperations:
    """Test combinations of operations to ensure proper pipeline."""

    def test_drill_tap_cbore_jig_grind_pipeline(self):
        """Test a complex hole table with all operation types."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.250',
                'QTY': 10,
                'DESCRIPTION': 'THRU'  # Plain drill - no pre-drill needed
            },
            {
                'HOLE': 'B',
                'REF_DIAM': 'Ø.190',
                'QTY': 5,
                'DESCRIPTION': '#10-32 TAP X .50 DEEP'  # Tap - needs pre-drill
            },
            {
                'HOLE': 'C',
                'REF_DIAM': 'Ø.500',
                'QTY': 4,
                'DESCRIPTION': "Ø.750 C'BORE X .25 DEEP"  # C'bore - no pre-drill in this fix
            },
            {
                'HOLE': 'D',
                'REF_DIAM': 'Ø.3125',
                'QTY': 2,
                'DESCRIPTION': 'THRU (JIG GRIND)'  # Jig-grind - needs pre-drill
            }
        ]

        times = estimate_hole_table_times(hole_table, material="steel", thickness=0.5)

        # Check that we have the expected groups:
        # Drill groups: plain drill (A) + tap pre-drill (B) + jig-grind pre-drill (D) = 3
        # Tap groups: 1 (B)
        # C'bore groups: 1 (C)
        # Jig-grind groups: 1 (D)
        assert len(times['drill_groups']) >= 3, \
            f"Expected at least 3 drill groups, got {len(times['drill_groups'])}"
        assert len(times['tap_groups']) == 1, "Expected 1 tap group"
        assert len(times['cbore_groups']) == 1, "Expected 1 cbore group"
        assert len(times['jig_grind_groups']) == 1, "Expected 1 jig-grind group"

    def test_no_zero_time_operations(self):
        """Test that all operations have non-zero time (fixes pressure pad 130 issue)."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.250',
                'QTY': 10,
                'DESCRIPTION': 'THRU'
            },
            {
                'HOLE': 'B',
                'REF_DIAM': 'Ø.190',
                'QTY': 5,
                'DESCRIPTION': '#10-32 TAP THRU'
            },
            {
                'HOLE': 'C',
                'REF_DIAM': 'Ø.500',
                'QTY': 4,
                'DESCRIPTION': "Ø.750 C'BORE X .25 DEEP"
            },
            {
                'HOLE': 'D',
                'REF_DIAM': 'Ø.3125',
                'QTY': 2,
                'DESCRIPTION': 'THRU (JIG GRIND)'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="steel", thickness=0.5)

        # Check that all groups have non-zero time
        for drill_group in times['drill_groups']:
            assert drill_group['time_per_hole'] > 0, \
                f"Drill group {drill_group['hole_id']} has zero time per hole"
            assert drill_group['total_time'] > 0, \
                f"Drill group {drill_group['hole_id']} has zero total time"

        for tap_group in times['tap_groups']:
            assert tap_group['time_per_hole'] > 0, \
                f"Tap group {tap_group['hole_id']} has zero time per hole"
            assert tap_group['total_time'] > 0, \
                f"Tap group {tap_group['hole_id']} has zero total time"

        for cbore_group in times['cbore_groups']:
            assert cbore_group['time_per_hole'] > 0, \
                f"C'bore group {cbore_group['hole_id']} has zero time per hole"
            assert cbore_group['total_time'] > 0, \
                f"C'bore group {cbore_group['hole_id']} has zero total time"

        for jig_grind_group in times['jig_grind_groups']:
            assert jig_grind_group['time_per_hole'] > 0, \
                f"Jig-grind group {jig_grind_group['hole_id']} has zero time per hole"
            assert jig_grind_group['total_time'] > 0, \
                f"Jig-grind group {jig_grind_group['hole_id']} has zero total time"

        # Check totals
        assert times['total_drill_minutes'] > 0, "Total drill minutes should be non-zero"
        assert times['total_tap_minutes'] > 0, "Total tap minutes should be non-zero"
        assert times['total_cbore_minutes'] > 0, "Total cbore minutes should be non-zero"
        assert times['total_jig_grind_minutes'] > 0, "Total jig-grind minutes should be non-zero"


class TestDiameterReferences:
    """Test that correct diameters are shown in drill vs tap operations."""

    def test_tap_shows_nominal_tap_diameter_shows_minor(self):
        """Test that tap operations show nominal diameter, pre-drill shows minor."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø5/16',  # 5/16 = 0.3125"
                'QTY': 4,
                'DESCRIPTION': '5/16-18 TAP THRU'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="steel", thickness=0.5)

        # Tap group should show TAP NOMINAL diameter (major = 0.3125")
        tap_group = times['tap_groups'][0]
        assert abs(tap_group['diameter'] - 0.3125) < 0.001, \
            f"Tap group should show nominal (major) diameter 0.3125\", got {tap_group['diameter']:.4f}\""

        # Drill group should show MINOR diameter (tap drill)
        # For 5/16-18: minor = 0.3125 - (1/18) = 0.2569"
        drill_group = times['drill_groups'][0]
        expected_minor = 0.3125 - (1.0 / 18)
        assert abs(drill_group['diameter'] - expected_minor) < 0.002, \
            f"Drill group should show minor diameter ~{expected_minor:.4f}\", got {drill_group['diameter']:.4f}\""

    def test_jig_grind_shows_final_diameter_predrill_shows_undersize(self):
        """Test that jig-grind shows final diameter, pre-drill shows undersize."""
        hole_table = [
            {
                'HOLE': 'A',
                'REF_DIAM': 'Ø.2500',
                'QTY': 2,
                'DESCRIPTION': 'THRU (JIG GRIND)'
            }
        ]

        times = estimate_hole_table_times(hole_table, material="A2", thickness=0.5)

        # Jig-grind group should show FINAL diameter (0.2500")
        jig_grind_group = times['jig_grind_groups'][0]
        assert abs(jig_grind_group['diameter'] - 0.2500) < 0.0001, \
            f"Jig-grind group should show final diameter 0.2500\", got {jig_grind_group['diameter']:.4f}\""

        # Drill group should show UNDERSIZE diameter (0.2500 - 0.007 = 0.2430")
        drill_group = times['drill_groups'][0]
        expected_undersize = 0.2500 - 0.007
        assert abs(drill_group['diameter'] - expected_undersize) < 0.0001, \
            f"Drill group should show undersize diameter {expected_undersize:.4f}\", got {drill_group['diameter']:.4f}\""
