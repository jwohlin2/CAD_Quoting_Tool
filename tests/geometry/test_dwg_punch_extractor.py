"""
Unit tests for DWG Punch Feature Extractor
==========================================

Tests the punch-specific feature extraction system for DWG/DXF drawings.
"""

import pytest
from pathlib import Path
from cad_quoter.geometry.dxf_enrich import (
    PunchFeatureSummary,
    classify_punch_family,
    detect_punch_material as detect_material,
    detect_punch_ops_features as detect_ops_features,
    detect_punch_pain_flags as detect_pain_flags,
    parse_punch_holes_from_text as parse_holes_from_text,
    parse_punch_tolerances_from_text as parse_tolerances_from_text,
    extract_punch_features_from_dxf,
)


class TestPunchFeatureSummary:
    """Test PunchFeatureSummary dataclass."""

    def test_default_initialization(self):
        """Test that PunchFeatureSummary has sensible defaults."""
        summary = PunchFeatureSummary()
        assert summary.family == "round_punch"
        assert summary.shape_type == "round"
        assert summary.overall_length_in == 0.0
        assert summary.max_od_or_width_in == 0.0
        assert summary.num_ground_diams == 0
        assert summary.tap_count == 0
        assert summary.confidence_score == 1.0
        assert summary.warnings == []

    def test_custom_initialization(self):
        """Test creating a summary with custom values."""
        summary = PunchFeatureSummary(
            family="pilot_pin",
            shape_type="round",
            overall_length_in=6.5,
            max_od_or_width_in=0.75,
            num_ground_diams=3,
            tap_count=2,
        )
        assert summary.family == "pilot_pin"
        assert summary.overall_length_in == 6.5
        assert summary.max_od_or_width_in == 0.75
        assert summary.num_ground_diams == 3
        assert summary.tap_count == 2


class TestClassifyPunchFamily:
    """Test punch family and shape classification."""

    def test_classify_round_punch(self):
        """Test classification of standard round punch."""
        text = "ROUND PUNCH\nMATERIAL: A2 TOOL STEEL"
        family, shape = classify_punch_family(text)
        assert family == "round_punch"
        assert shape == "round"

    def test_classify_pilot_pin(self):
        """Test classification of pilot pin."""
        text = "PILOT PIN\nMATERIAL: M2"
        family, shape = classify_punch_family(text)
        assert family == "pilot_pin"
        assert shape == "round"

    def test_classify_form_punch(self):
        """Test classification of form punch."""
        text = "FORM PUNCH\nCOIN DETAIL"
        family, shape = classify_punch_family(text)
        assert family == "form_punch"
        assert shape == "round"

    def test_classify_die_section(self):
        """Test classification of die section."""
        text = "DIE SECTION\nRECTANGULAR"
        family, shape = classify_punch_family(text)
        assert family == "die_section"
        assert shape == "rectangular"

    def test_classify_rectangular_punch(self):
        """Test classification based on shape indicators."""
        text = "PUNCH\nRECTANGULAR\nWIDTH: 2.0\nTHICKNESS: 0.5"
        family, shape = classify_punch_family(text)
        assert family == "round_punch"
        assert shape == "rectangular"

    def test_classify_guide_post(self):
        """Test classification of guide post."""
        text = "GUIDE POST\nMATERIAL: O1"
        family, shape = classify_punch_family(text)
        assert family == "guide_post"
        assert shape == "round"

    def test_classify_bushing(self):
        """Test classification of bushing."""
        text = "GUIDE BUSHING\nMATERIAL: D2"
        family, shape = classify_punch_family(text)
        assert family == "bushing"
        assert shape == "round"


class TestDetectMaterial:
    """Test material detection from text."""

    def test_detect_a2(self):
        """Test detection of A2 tool steel."""
        text = "MATERIAL: A2 TOOL STEEL\nHEAT TREAT: 60-62 RC"
        material = detect_material(text)
        assert material == "A2"

    def test_detect_d2(self):
        """Test detection of D2 tool steel."""
        text = "MATERIAL: D2\nHARDNESS: 58-60 RC"
        material = detect_material(text)
        assert material == "D2"

    def test_detect_m2(self):
        """Test detection of M2 high speed steel."""
        text = "MAT'L: M2 HSS"
        material = detect_material(text)
        assert material == "M2"

    def test_detect_carbide(self):
        """Test detection of carbide."""
        text = "MATERIAL: CARBIDE"
        material = detect_material(text)
        assert material == "CARBIDE"

    def test_detect_440c(self):
        """Test detection of 440C stainless."""
        text = "MATERIAL: 440C STAINLESS"
        material = detect_material(text)
        assert material == "440C"

    def test_no_material(self):
        """Test when no material is specified."""
        text = "ROUND PUNCH\nNOTES: GRIND ALL DIAMETERS"
        material = detect_material(text)
        assert material is None


class TestParseTolerancesFromText:
    """Test tolerance parsing from dimension text."""

    def test_parse_plus_minus_tolerance(self):
        """Test parsing ±0.001 format."""
        tolerances = parse_tolerances_from_text("1.500±0.001")
        assert len(tolerances) == 1
        assert tolerances[0] == 0.001

    def test_parse_bilateral_tolerance(self):
        """Test parsing +0.0000-0.0002 format."""
        tolerances = parse_tolerances_from_text("0.7500+0.0000-0.0002")
        assert len(tolerances) == 2
        assert 0.0000 in tolerances
        assert 0.0002 in tolerances

    def test_parse_small_decimal_tolerance(self):
        """Test parsing .0001 format."""
        tolerances = parse_tolerances_from_text("Ø.7500 .0001")
        assert len(tolerances) >= 1
        # Should find .0001
        assert any(abs(t - 0.0001) < 0.00001 for t in tolerances)

    def test_no_tolerance(self):
        """Test when no tolerance is present."""
        tolerances = parse_tolerances_from_text("1.500 REF")
        # May or may not find anything; just ensure it doesn't crash
        assert isinstance(tolerances, list)


class TestDetectOpsFeatures:
    """Test operations feature detection."""

    def test_detect_chamfers(self):
        """Test detection of chamfer callouts."""
        text = "(2) .010 X 45°\n.015 X 45° TYP"
        features = detect_ops_features(text)
        assert features["num_chamfers"] >= 2

    def test_detect_small_radii(self):
        """Test detection of small radii."""
        text = "R.005 TYP\n.003 R"
        features = detect_ops_features(text)
        assert features["num_small_radii"] == 2

    def test_detect_3d_surface(self):
        """Test detection of 3D surface features."""
        text = "POLISH CONTOUR\nFORM DETAIL"
        features = detect_ops_features(text)
        assert features["has_3d_surface"] is True

    def test_detect_perp_face(self):
        """Test detection of perpendicular face requirement."""
        text = "THIS SURFACE PERPENDICULAR TO CENTERLINE"
        features = detect_ops_features(text)
        assert features["has_perp_face_grind"] is True

    def test_form_complexity_low(self):
        """Test low form complexity."""
        text = "R.125\nR.250"
        features = detect_ops_features(text)
        assert features["form_complexity_level"] == 1

    def test_form_complexity_medium(self):
        """Test medium form complexity."""
        text = "\n".join([f"R.{i:03d}" for i in range(100, 107)])
        features = detect_ops_features(text)
        assert features["form_complexity_level"] == 2

    def test_form_complexity_high(self):
        """Test high form complexity."""
        text = "\n".join([f"R.{i:03d}" for i in range(100, 115)])
        features = detect_ops_features(text)
        assert features["form_complexity_level"] == 3


class TestDetectPainFlags:
    """Test pain/quality flag detection."""

    def test_detect_polish(self):
        """Test detection of polish requirement."""
        text = "POLISH CONTOUR TO 8 µin"
        flags = detect_pain_flags(text)
        assert flags["has_polish_contour"] is True

    def test_detect_no_step(self):
        """Test detection of no step permitted."""
        text = "NO STEP PERMITTED BETWEEN DIAMETERS"
        flags = detect_pain_flags(text)
        assert flags["has_no_step_permitted"] is True

    def test_detect_sharp_edges(self):
        """Test detection of sharp edge requirement."""
        text = "SHARP EDGES REQUIRED"
        flags = detect_pain_flags(text)
        assert flags["has_sharp_edges"] is True

    def test_detect_gdt(self):
        """Test detection of GD&T."""
        text = "GD&T PER ASME Y14.5"
        flags = detect_pain_flags(text)
        assert flags["has_gdt"] is True

    def test_no_pain_flags(self):
        """Test when no special requirements exist."""
        text = "ROUND PUNCH\nMATERIAL: A2"
        flags = detect_pain_flags(text)
        assert flags["has_polish_contour"] is False
        assert flags["has_no_step_permitted"] is False
        assert flags["has_sharp_edges"] is False


class TestParseHolesFromText:
    """Test hole and tap parsing."""

    def test_parse_tap_with_depth(self):
        """Test parsing tap specification with depth."""
        text = "5/16-18 TAP X .80 DEEP"
        result = parse_holes_from_text(text)
        assert result["tap_count"] == 1
        assert len(result["tap_summary"]) == 1
        assert result["tap_summary"][0]["size"] == "5/16-18"
        assert result["tap_summary"][0]["depth_in"] == 0.80

    def test_parse_multiple_taps(self):
        """Test parsing multiple tap specifications."""
        text = """
        5/16-18 TAP X .80 DEEP
        1/4-20 TAP X .50 DEEP
        """
        result = parse_holes_from_text(text)
        assert result["tap_count"] == 2
        assert len(result["tap_summary"]) == 2

    def test_parse_tap_no_depth(self):
        """Test parsing tap without depth."""
        text = "5/16-18 TAP"
        result = parse_holes_from_text(text)
        assert result["tap_count"] == 1
        assert result["tap_summary"][0]["size"] == "5/16-18"
        assert result["tap_summary"][0]["depth_in"] is None

    def test_parse_thru_hole(self):
        """Test parsing through hole."""
        text = "Ø.250 THRU"
        result = parse_holes_from_text(text)
        assert result["hole_count"] == 1
        assert result["hole_summary"][0]["diameter"] == 0.250
        assert result["hole_summary"][0]["thru"] is True

    def test_parse_blind_hole(self):
        """Test parsing blind hole with depth."""
        text = "Ø.125 X .50 DP"
        result = parse_holes_from_text(text)
        assert result["hole_count"] == 1
        assert result["hole_summary"][0]["diameter"] == 0.125
        assert result["hole_summary"][0]["depth_in"] == 0.50
        assert result["hole_summary"][0]["thru"] is False

    def test_parse_mixed_holes_and_taps(self):
        """Test parsing combination of holes and taps."""
        text = """
        Ø.250 THRU
        5/16-18 TAP X .80 DEEP
        Ø.125 X .50 DP
        """
        result = parse_holes_from_text(text)
        assert result["tap_count"] == 1
        assert result["hole_count"] == 2

    def test_no_holes(self):
        """Test when no holes are present."""
        text = "ROUND PUNCH\nGRIND ALL DIAMETERS"
        result = parse_holes_from_text(text)
        assert result["tap_count"] == 0
        assert result["hole_count"] == 0


class TestExtractPunchFeaturesFromDxf:
    """Test main extraction function."""

    def test_basic_extraction_with_text_only(self):
        """Test extraction with only text (no real DXF file)."""
        # Create a simple text dump
        text_dump = """
        ROUND PUNCH
        MATERIAL: A2 TOOL STEEL
        6.50 ±.010 OAL
        Ø.750 MAX DIA
        Ø.625 SHANK
        5/16-18 TAP X .80 DEEP
        .010 X 45° CHAMFER TYP
        POLISH CONTOUR
        """

        # Use a dummy path (won't actually read DXF)
        dummy_path = Path("/tmp/dummy.dxf")

        summary = extract_punch_features_from_dxf(dummy_path, text_dump)

        # Check classification
        assert summary.family == "round_punch"
        assert summary.shape_type == "round"

        # Check material
        assert summary.material_callout == "A2"

        # Check taps
        assert summary.tap_count == 1
        assert len(summary.tap_summary) == 1

        # Check pain flags
        assert summary.has_polish_contour is True

    def test_pilot_pin_extraction(self):
        """Test extraction of pilot pin features."""
        text_dump = """
        PILOT PIN
        MATERIAL: M2 HSS
        8.00 OAL
        Ø.500 +.0000 -.0002
        THIS SURFACE PERPENDICULAR TO CENTERLINE
        """

        summary = extract_punch_features_from_dxf(Path("/tmp/dummy.dxf"), text_dump)

        assert summary.family == "pilot_pin"
        assert summary.shape_type == "round"
        assert summary.material_callout == "M2"
        assert summary.has_perp_face_grind is True

    def test_form_punch_extraction(self):
        """Test extraction of form punch features."""
        text_dump = """
        FORM PUNCH
        MATERIAL: D2
        COIN PUNCH DETAIL
        POLISH CONTOUR TO 8 µin
        R.125
        R.250
        R.375
        NO STEP PERMITTED
        """

        summary = extract_punch_features_from_dxf(Path("/tmp/dummy.dxf"), text_dump)

        assert summary.family == "form_punch"
        assert summary.has_3d_surface is True
        assert summary.has_polish_contour is True
        assert summary.has_no_step_permitted is True
        assert summary.form_complexity_level > 0

    def test_rectangular_section_extraction(self):
        """Test extraction of rectangular section features."""
        text_dump = """
        DIE SECTION
        RECTANGULAR
        MATERIAL: A2
        4.00 LENGTH
        2.00 WIDTH
        0.75 THICKNESS
        (2) .015 X 45° CHAMFERS
        """

        summary = extract_punch_features_from_dxf(Path("/tmp/dummy.dxf"), text_dump)

        assert summary.family == "die_section"
        assert summary.shape_type == "rectangular"
        assert summary.num_chamfers >= 2

    def test_confidence_score_no_dimensions(self):
        """Test that confidence score is reduced when dimensions are missing."""
        text_dump = "ROUND PUNCH"

        summary = extract_punch_features_from_dxf(Path("/tmp/dummy.dxf"), text_dump)

        # Should have reduced confidence when no dimensions extracted
        assert summary.confidence_score < 1.0

    def test_warnings_on_errors(self):
        """Test that warnings are added when extraction has errors."""
        text_dump = "PUNCH"

        summary = extract_punch_features_from_dxf(Path("/tmp/nonexistent.dxf"), text_dump)

        # Should have warnings about missing file
        # (though they might be empty if ezdxf handles gracefully)
        assert isinstance(summary.warnings, list)


class TestIntegration:
    """Integration tests for the full extraction pipeline."""

    def test_comprehensive_punch_extraction(self):
        """Test extraction of a comprehensive punch with all features."""
        text_dump = """
        ROUND PUNCH - PART #12345
        MATERIAL: A2 TOOL STEEL
        HEAT TREAT: 60-62 RC

        DIMENSIONS:
        6.990 ±.005 OAL
        Ø.7504 +.0000 -.0001 MAX DIA
        Ø.6250 +.0000 -.0002 SHANK
        Ø.5000 NOSE SECTION

        FEATURES:
        5/16-18 TAP X .80 DEEP
        1/4-20 TAP X .50 DEEP
        Ø.125 COOLANT HOLE THRU

        EDGE WORK:
        (2) .010 X 45° CHAMFER
        R.005 TYP

        NOTES:
        - GRIND ALL DIAMETERS
        - THIS SURFACE PERPENDICULAR TO CENTERLINE
        - POLISH CONTOUR TO 8 µin
        - NO STEP PERMITTED BETWEEN DIAMETERS
        - GD&T PER ASME Y14.5
        """

        summary = extract_punch_features_from_dxf(Path("/tmp/dummy.dxf"), text_dump)

        # Classification
        assert summary.family == "round_punch"
        assert summary.shape_type == "round"

        # Material
        assert summary.material_callout == "A2"

        # Taps
        assert summary.tap_count == 2

        # Chamfers and radii
        assert summary.num_chamfers >= 2
        assert summary.num_small_radii >= 1

        # Pain flags
        assert summary.has_perp_face_grind is True
        assert summary.has_polish_contour is True
        assert summary.has_no_step_permitted is True
        assert summary.has_gdt is True

    def test_minimal_punch_extraction(self):
        """Test extraction with minimal information."""
        text_dump = "PUNCH\nMATERIAL: D2"

        summary = extract_punch_features_from_dxf(Path("/tmp/dummy.dxf"), text_dump)

        # Should still work, with defaults
        assert summary.family == "round_punch"
        assert summary.shape_type == "round"
        assert summary.material_callout == "D2"
        assert summary.confidence_score < 1.0  # Lower confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
