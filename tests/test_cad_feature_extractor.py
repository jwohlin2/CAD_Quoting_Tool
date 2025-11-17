"""
Unit tests for CAD Feature Extractor
=====================================

Tests the comprehensive feature extraction system for CAD quoting.
"""

import pytest
from cad_quoter.cad_feature_extractor import (
    ComprehensiveFeatureExtractor,
    StockGeometryExtractor,
    CylindricalSectionExtractor,
    ContourFeatureExtractor,
    HoleFeatureExtractor,
    ChamferRadiusExtractor,
    ToleranceFinishExtractor,
    features_to_quoting_variables,
)


class TestStockGeometryExtractor:
    """Test stock/envelope geometry extraction."""

    def test_extract_round_part_from_geometry(self):
        """Test extraction of round part dimensions from STEP/IGES geometry."""
        geo_features = {
            "GEO-01_Length_mm": 177.546,  # ~6.99"
            "GEO-02_Width_mm": 19.0602,   # ~0.75"
            "GEO-03_Height_mm": 19.0602,
            "GEO_Turning_Score_0to1": 0.85,  # High turning score = round part
            "GEO_MaxOD_mm": 19.0602,
            "GEO-Volume_mm3": 50000.0,
        }

        extractor = StockGeometryExtractor()
        stock = extractor.extract_from_geometry(geo_features)

        assert stock.part_type == "round"
        assert stock.overall_length == pytest.approx(6.99, abs=0.01)
        assert stock.max_diameter == pytest.approx(0.75, abs=0.01)
        assert stock.estimated_volume is not None
        assert stock.estimated_weight is not None

    def test_extract_rectangular_part_from_geometry(self):
        """Test extraction of rectangular part from geometry."""
        geo_features = {
            "GEO-01_Length_mm": 50.8,  # 2.0"
            "GEO-02_Width_mm": 17.78,  # 0.7"
            "GEO-03_Height_mm": 12.7,  # 0.5"
            "GEO_Turning_Score_0to1": 0.2,  # Low turning score = rectangular
            "GEO-Volume_mm3": 10000.0,
        }

        extractor = StockGeometryExtractor()
        stock = extractor.extract_from_geometry(geo_features)

        assert stock.part_type == "rectangular"
        assert stock.body_width == pytest.approx(0.7, abs=0.01)
        assert stock.body_height == pytest.approx(0.5, abs=0.01)
        assert stock.centerline_length == pytest.approx(2.0, abs=0.01)

    def test_extract_dimensions_from_text(self):
        """Test extraction of dimensions from drawing text."""
        text_records = [
            "Ø.7504 MAX DIA",
            "6.99 ±1/32 OAL",
            ".7000 × .5000 SHANK",
        ]

        extractor = StockGeometryExtractor()
        stock = extractor.extract_from_text(text_records)

        assert stock.max_diameter == pytest.approx(0.7504)
        assert stock.overall_length == pytest.approx(6.99)
        assert stock.body_width == pytest.approx(0.7)
        assert stock.body_height == pytest.approx(0.5)


class TestCylindricalSectionExtractor:
    """Test cylindrical section extraction."""

    def test_extract_diameter_with_symmetric_tolerance(self):
        """Test extraction of diameter with ± tolerance."""
        text_records = [
            "Ø.7504 ±.0001",
            "Ø.5021 ±.0002",
        ]

        extractor = CylindricalSectionExtractor()
        sections = extractor.extract_from_text(text_records)

        assert len(sections) == 2
        assert sections[0].diameter == pytest.approx(0.7504)
        assert sections[0].tolerance_upper == pytest.approx(0.0001)
        assert sections[0].tolerance_lower == pytest.approx(0.0001)
        assert sections[0].is_critical is True  # ±.0001 is tight

    def test_extract_diameter_with_asymmetric_tolerance(self):
        """Test extraction with +/- tolerance."""
        text_records = [
            "Ø.5021 +.0000/-.0002",
            ".4997 +.0000/-.0002",
        ]

        extractor = CylindricalSectionExtractor()
        sections = extractor.extract_from_text(text_records)

        assert len(sections) >= 1
        assert sections[0].tolerance_upper == pytest.approx(0.0)
        assert sections[0].tolerance_lower == pytest.approx(0.0002)
        assert sections[0].is_critical is True

    def test_extract_straight_section_with_length(self):
        """Test extraction of STRAIGHT TYP callouts."""
        text_records = [
            ".62 STRAIGHT TYP",
            ".76 STRAIGHT TYP",
        ]

        extractor = CylindricalSectionExtractor()
        sections = extractor.extract_from_text(text_records)

        straight_sections = [s for s in sections if s.is_straight]
        assert len(straight_sections) >= 1

    def test_special_callouts(self):
        """Test detection of special callouts."""
        text_records = [
            "Ø.500 ±.0001 NO STEP PERMITTED",
            "Ø.750 PERPENDICULAR TO CENTERLINE",
        ]

        extractor = CylindricalSectionExtractor()
        sections = extractor.extract_from_text(text_records)

        # Check for no_step_permitted
        no_step_sections = [s for s in sections if s.no_step_permitted]
        assert len(no_step_sections) >= 1

        # Check for perpendicular face
        perp_sections = [s for s in sections if s.has_perp_face]
        assert len(perp_sections) >= 1

    def test_aggregate_sections(self):
        """Test aggregation of section metrics."""
        sections = []
        extractor = CylindricalSectionExtractor()

        # Create mock sections
        from cad_quoter.cad_feature_extractor import CylindricalSection
        sections.append(CylindricalSection(diameter=0.75, length=1.0, is_critical=True))
        sections.append(CylindricalSection(diameter=0.50, length=0.5, is_critical=True, has_perp_face=True))
        sections.append(CylindricalSection(diameter=0.25, length=0.3, is_critical=False))

        agg = extractor.aggregate_sections(sections)

        assert agg["num_ground_diameters"] == 2  # Only critical ones
        assert agg["sum_ground_length"] == pytest.approx(1.8)
        assert agg["has_perp_face_grind"] is True


class TestContourFeatureExtractor:
    """Test contour and form feature extraction."""

    def test_detect_polish_contour(self):
        """Test detection of POLISH CONTOUR keyword."""
        text_records = [
            "POLISH CONTOUR TO DRAWING",
            "R.025 BLEND",
        ]

        extractor = ContourFeatureExtractor()
        contour = extractor.extract_from_text(text_records)

        assert contour.has_polish_contour is True
        assert "POLISH CONTOUR" in contour.keywords

    def test_detect_coin_punch(self):
        """Test detection of COIN keyword."""
        text_records = [
            "COIN PUNCH FORM",
        ]

        extractor = ContourFeatureExtractor()
        contour = extractor.extract_from_text(text_records)

        assert contour.has_coin_punch is True

    def test_extract_radii(self):
        """Test extraction of radius values."""
        text_records = [
            "R.005 MAX",
            "R.025 BLEND",
            "R.09 CORNER",
        ]

        extractor = ContourFeatureExtractor()
        contour = extractor.extract_from_text(text_records)

        assert len(contour.radii_list) == 3
        assert contour.min_radius == pytest.approx(0.005)
        assert pytest.approx(0.025) in contour.radii_list

    def test_extract_wire_edm_profile(self):
        """Test detection of wire EDM profile."""
        text_records = [
            "WIRE EDM PROFILE",
        ]

        extractor = ContourFeatureExtractor()
        contour = extractor.extract_from_text(text_records)

        assert contour.has_2d_wire_edm_profile is True

    def test_extract_from_geometry(self):
        """Test contour extraction from 3D geometry."""
        geo_features = {
            "GEO_Area_Freeform_mm2": 500.0,  # Has freeform surface
            "GEO_WEDM_PathLen_mm": 100.0,    # Wire EDM path
            "GEO-01_Length_mm": 50.0,
            "GEO-02_Width_mm": 30.0,
            "GEO-03_Height_mm": 20.0,
        }

        extractor = ContourFeatureExtractor()
        contour = extractor.extract_from_geometry(geo_features)

        assert contour.has_3d_contoured_nose is True
        assert contour.has_2d_wire_edm_profile is True
        assert contour.form_area_sq_in is not None


class TestHoleFeatureExtractor:
    """Test hole and tap feature extraction."""

    def test_extract_tapped_hole(self):
        """Test extraction of tapped hole specification."""
        text_records = [
            "5/16-18 TAP X .80 DEEP",
            "(2) 1/4-20 TAP X .50 DEEP",
        ]

        extractor = HoleFeatureExtractor()
        holes = extractor.extract_from_text(text_records)

        assert len(holes) >= 1
        assert holes[0].hole_type == "threaded"
        assert holes[0].thread_spec == "5/16-18"
        assert holes[0].depth == pytest.approx(0.80)

    def test_extract_drilled_hole(self):
        """Test extraction of drilled hole."""
        text_records = [
            "Ø.250 X .500 DEEP",
        ]

        extractor = HoleFeatureExtractor()
        holes = extractor.extract_from_text(text_records)

        assert len(holes) >= 1
        assert holes[0].hole_type == "blind"
        assert holes[0].diameter == pytest.approx(0.250)
        assert holes[0].depth == pytest.approx(0.500)

    def test_extract_through_holes(self):
        """Test extraction of through holes."""
        text_records = [
            "(3) THRU HOLES",
        ]

        extractor = HoleFeatureExtractor()
        holes = extractor.extract_from_text(text_records)

        assert len(holes) >= 1
        assert holes[0].is_through is True
        assert holes[0].count == 3

    def test_extract_from_geometry(self):
        """Test hole extraction from 3D geometry."""
        geo_features = {
            "GEO_Hole_Groups": [
                {"dia_mm": 6.35, "depth_mm": 12.7, "through": False, "count": 2},
                {"dia_mm": 8.0, "depth_mm": 20.0, "through": True, "count": 4},
            ]
        }

        extractor = HoleFeatureExtractor()
        holes = extractor.extract_from_geometry(geo_features)

        assert len(holes) == 2
        assert holes[0].diameter == pytest.approx(0.25, abs=0.01)  # 6.35mm ≈ 0.25"
        assert holes[0].is_through is False
        assert holes[1].is_through is True


class TestChamferRadiusExtractor:
    """Test chamfer, radius, and undercut extraction."""

    def test_extract_chamfer(self):
        """Test extraction of chamfer callout."""
        text_records = [
            "(3) .040 × 45°",
            ".020 X 45",
        ]

        extractor = ChamferRadiusExtractor()
        features = extractor.extract_from_text(text_records)

        chamfers = [f for f in features if f.feature_type == "chamfer"]
        assert len(chamfers) >= 1
        assert chamfers[0].chamfer_size == pytest.approx(0.040)
        assert chamfers[0].chamfer_angle == pytest.approx(45.0)
        assert chamfers[0].count == 3

    def test_extract_small_radius(self):
        """Test extraction of small radii."""
        text_records = [
            "R.005 MAX",
            "R.007 CORNER",
        ]

        extractor = ChamferRadiusExtractor()
        features = extractor.extract_from_text(text_records)

        radii = [f for f in features if f.feature_type == "radius"]
        assert len(radii) >= 1
        small_radii = [f for f in radii if f.is_small_radius]
        assert len(small_radii) >= 1

    def test_extract_undercut(self):
        """Test extraction of undercut callout."""
        text_records = [
            "(2) SMALL UNDERCUT",
        ]

        extractor = ChamferRadiusExtractor()
        features = extractor.extract_from_text(text_records)

        undercuts = [f for f in features if f.is_undercut]
        assert len(undercuts) >= 1
        assert undercuts[0].count == 2


class TestToleranceFinishExtractor:
    """Test tolerance and finish extraction."""

    def test_extract_tight_symmetric_tolerance(self):
        """Test extraction of tight ± tolerance."""
        text_records = [
            "Ø.7504 ±.0001",
        ]

        extractor = ToleranceFinishExtractor()
        tolerances = extractor.extract_from_text(text_records)

        assert len(tolerances) >= 1
        assert tolerances[0].nominal_value == pytest.approx(0.7504)
        assert tolerances[0].tolerance_upper == pytest.approx(0.0001)
        assert tolerances[0].tolerance_class == "tight"
        assert tolerances[0].inspection_minutes_multiplier > 1.0

    def test_extract_very_tight_tolerance(self):
        """Test classification of very tight tolerance."""
        text_records = [
            "Ø.5000 ±.00005",
        ]

        extractor = ToleranceFinishExtractor()
        tolerances = extractor.extract_from_text(text_records)

        assert len(tolerances) >= 1
        assert tolerances[0].tolerance_class == "very_tight"
        assert tolerances[0].inspection_minutes_multiplier >= 3.0

    def test_extract_asymmetric_tolerance(self):
        """Test extraction of +/- tolerance."""
        text_records = [
            ".2497 +.0000/-.0002",
        ]

        extractor = ToleranceFinishExtractor()
        tolerances = extractor.extract_from_text(text_records)

        assert len(tolerances) >= 1
        assert tolerances[0].tolerance_upper == pytest.approx(0.0)
        assert tolerances[0].tolerance_lower == pytest.approx(0.0002)

    def test_extract_surface_finish(self):
        """Test extraction of surface finish requirements."""
        text_records = [
            "8 µin FINISH",
            "16 RA",
        ]

        extractor = ToleranceFinishExtractor()
        tolerances = extractor.extract_from_text(text_records)

        finishes = [t for t in tolerances if t.feature_type == "surface_finish"]
        assert len(finishes) >= 1

    def test_extract_polish_requirement(self):
        """Test detection of POLISH requirement."""
        text_records = [
            "POLISH CONTOUR TO 8 µin",
        ]

        extractor = ToleranceFinishExtractor()
        tolerances = extractor.extract_from_text(text_records)

        polish_features = [t for t in tolerances if "POLISH" in (t.surface_finish or "")]
        assert len(polish_features) >= 1


class TestComprehensiveFeatureExtractor:
    """Test the main orchestrator."""

    def test_extract_all_features_from_text(self):
        """Test comprehensive extraction from text records."""
        text_records = [
            "Ø.7504 ±.0001",
            "6.99 ±1/32 OAL",
            "5/16-18 TAP X .80 DEEP",
            "(3) .040 × 45°",
            "POLISH CONTOUR",
            "R.005 BLEND",
        ]

        extractor = ComprehensiveFeatureExtractor()
        features = extractor.extract_all_features(text_records=text_records)

        # Check stock geometry
        assert features.stock_geometry.overall_length == pytest.approx(6.99)

        # Check cylindrical sections
        assert len(features.cylindrical_sections) >= 1
        assert features.num_ground_diameters >= 1

        # Check holes
        assert features.tap_count >= 1

        # Check edge features
        assert features.chamfer_count >= 3

        # Check tolerances
        assert features.tight_tolerance_count >= 1
        assert features.requires_polish is True

        # Check complexity scores
        assert features.machining_complexity_score > 0
        assert features.inspection_complexity_score > 0

    def test_extract_all_features_from_geometry(self):
        """Test comprehensive extraction from geometry."""
        geo_features = {
            "GEO-01_Length_mm": 177.546,
            "GEO-02_Width_mm": 19.0602,
            "GEO-03_Height_mm": 19.0602,
            "GEO_Turning_Score_0to1": 0.85,
            "GEO_MaxOD_mm": 19.0602,
            "GEO-Volume_mm3": 50000.0,
            "GEO_Hole_Groups": [
                {"dia_mm": 6.35, "depth_mm": 12.7, "through": False, "count": 2},
            ],
            "GEO_Area_Freeform_mm2": 500.0,
            "GEO_WEDM_PathLen_mm": 100.0,
        }

        extractor = ComprehensiveFeatureExtractor()
        features = extractor.extract_all_features(geo_features=geo_features)

        assert features.stock_geometry.part_type == "round"
        assert features.stock_geometry.max_diameter is not None
        assert len(features.hole_features) >= 1
        assert features.contour_features.has_3d_contoured_nose is True

    def test_features_to_quoting_variables(self):
        """Test conversion to quoting variables."""
        text_records = [
            "Ø.7504 ±.0001",
            "6.99 OAL",
            "5/16-18 TAP X .80 DEEP",
        ]

        extractor = ComprehensiveFeatureExtractor()
        features = extractor.extract_all_features(text_records=text_records)
        variables = features_to_quoting_variables(features)

        assert "STOCK_LENGTH" in variables
        assert "NUM_GROUND_DIAMETERS" in variables
        assert "TAP_COUNT" in variables
        assert "MACHINING_COMPLEXITY" in variables
        assert variables["TAP_COUNT"] >= 1.0

    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        from cad_quoter.cad_feature_extractor import (
            ComprehensiveFeatures,
            CylindricalSection,
            HoleFeature,
        )

        features = ComprehensiveFeatures()
        features.num_ground_diameters = 5
        features.tap_count = 3
        features.tight_tolerance_count = 4
        features.requires_polish = True

        extractor = ComprehensiveFeatureExtractor()
        machining_score = extractor._calculate_machining_complexity(features)
        inspection_score = extractor._calculate_inspection_complexity(features)

        assert machining_score > 0
        assert inspection_score > 0
        assert machining_score <= 100.0
        assert inspection_score <= 100.0


class TestRealWorldExamples:
    """Test with real-world part examples from the checklist."""

    def test_t1769_219_round_part(self):
        """Test extraction for T1769-219 (round part with multiple diameters)."""
        text_records = [
            "Ø.7504 ±.0001",
            "Ø.5021 ±.0002",
            "Ø.149",
            "Ø.145",
            "5/16-18 TAP X .80 DEEP",
            "6.99 ±1/32 OAL",
            ".62 STRAIGHT TYP",
        ]

        extractor = ComprehensiveFeatureExtractor()
        features = extractor.extract_all_features(text_records=text_records)

        # Should detect multiple ground diameters
        assert features.num_ground_diameters >= 2

        # Should detect tap
        assert features.tap_count >= 1

        # Should have overall length
        assert features.stock_geometry.overall_length == pytest.approx(6.99)

    def test_316a_form_punch(self):
        """Test extraction for 316A (form punch with contour)."""
        text_records = [
            "(2) .4997 +.0000/-.0002",
            "POLISH CONTOUR",
            ".7000 × .5000 SHANK",
            "FORM AREA",
        ]

        extractor = ComprehensiveFeatureExtractor()
        features = extractor.extract_all_features(text_records=text_records)

        # Should be rectangular
        assert features.stock_geometry.part_type == "rectangular"

        # Should require polish
        assert features.requires_polish is True

        # Should have contour
        assert features.contour_features.has_polish_contour is True

    def test_t1769_326_with_chamfers_and_undercuts(self):
        """Test extraction for 1769-326 (chamfers and undercuts)."""
        text_records = [
            "(3) .040 × 45°",
            "(2) SMALL UNDERCUT",
            "R.005 BLEND",
            "R.025 CORNER",
        ]

        extractor = ComprehensiveFeatureExtractor()
        features = extractor.extract_all_features(text_records=text_records)

        # Should detect chamfers
        assert features.chamfer_count >= 3

        # Should detect undercuts
        assert features.undercut_count >= 2

        # Should detect small radii
        assert features.small_radius_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
