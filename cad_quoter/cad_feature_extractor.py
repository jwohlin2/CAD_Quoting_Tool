"""
CAD Feature Extractor - Comprehensive Quoting Feature Analysis
===============================================================

This module extracts detailed manufacturing features from CAD files and text
to drive quoting calculations for stock sizing, machining operations, and
inspection requirements.

Based on the comprehensive checklist for extracting:
    A. Stock/envelope geometry (round & rectangular parts)
    B. Feature & tolerance drivers:
        1. Cylindrical and straight sections
        2. Contours, forms, and 3D surfaces
        3. Holes, taps, and internal features
        4. Chamfers, corner radii, and undercuts
        5. Tight tolerances and finish/polish requirements

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StockGeometry:
    """Stock/envelope geometry for material selection and rough machining."""

    # Part type classification
    part_type: str = "unknown"  # "round", "rectangular", "form_punch", "irregular"

    # Round part dimensions
    overall_length: Optional[float] = None  # inches
    max_diameter: Optional[float] = None  # inches
    min_diameter: Optional[float] = None  # inches
    slender_section_diameter: Optional[float] = None  # inches
    slender_section_length: Optional[float] = None  # inches
    taper_length: Optional[float] = None  # inches

    # Rectangular/form punch dimensions
    body_width: Optional[float] = None  # inches
    body_height: Optional[float] = None  # inches
    cross_section_area: Optional[float] = None  # sq in
    centerline_length: Optional[float] = None  # inches
    form_area_width: Optional[float] = None  # inches
    form_area_height: Optional[float] = None  # inches
    form_area_depth: Optional[float] = None  # inches

    # Material estimation
    estimated_volume: Optional[float] = None  # cubic inches
    estimated_weight: Optional[float] = None  # pounds (depends on material)
    stock_size_recommendation: Optional[str] = None  # e.g., "0.75 x 6.0 rod"


@dataclass
class CylindricalSection:
    """A distinct diameter or width section that must be machined."""
    diameter: float  # inches
    tolerance_upper: float = 0.0  # +tolerance
    tolerance_lower: float = 0.0  # -tolerance
    length: float = 0.0  # length of this section
    is_critical: bool = False  # tight tolerance or special callout
    has_perp_face: bool = False  # perpendicular face requirement
    no_step_permitted: bool = False
    is_straight: bool = False  # "STRAIGHT TYP" callout
    surface_finish: Optional[str] = None  # e.g., "8 µin"


@dataclass
class ContourFeature:
    """Complex 2D/3D contoured surfaces and forms."""
    has_2d_wire_edm_profile: bool = False
    has_3d_contoured_nose: bool = False
    has_coin_punch: bool = False
    has_polish_contour: bool = False

    # Bounding box of contoured zone
    contour_width: Optional[float] = None
    contour_depth: Optional[float] = None
    contour_length: Optional[float] = None

    # Radii in contoured area
    radii_list: List[float] = field(default_factory=list)
    min_radius: Optional[float] = None

    # EDM characteristics
    wire_edm_profile_length: Optional[float] = None  # linear inches
    form_area_sq_in: Optional[float] = None

    # Keywords found
    keywords: List[str] = field(default_factory=list)  # POLISH, COIN, FORM, etc.


@dataclass
class HoleFeature:
    """Threaded holes, through holes, and internal features."""
    hole_type: str = "unknown"  # "threaded", "through", "blind", "coolant", "cross"
    diameter: Optional[float] = None  # inches
    depth: Optional[float] = None  # inches
    thread_spec: Optional[str] = None  # e.g., "5/16-18"
    is_through: bool = False
    count: int = 1
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z) if available


@dataclass
class ChamferRadiusFeature:
    """Edge features: chamfers, radii, undercuts."""
    feature_type: str = "unknown"  # "chamfer", "radius", "undercut"

    # Chamfer details
    chamfer_size: Optional[float] = None  # e.g., 0.040
    chamfer_angle: Optional[float] = None  # degrees, e.g., 45

    # Radius details
    radius_size: Optional[float] = None

    # Special flags
    is_small_radius: bool = False  # R.005, R.007 - requires grinding/EDM
    is_undercut: bool = False

    count: int = 1  # how many of this feature


@dataclass
class ToleranceFeature:
    """Tight tolerances and finish requirements."""
    feature_type: str = "unknown"  # "diameter", "length", "surface_finish", "position"
    nominal_value: Optional[float] = None
    tolerance_upper: float = 0.0
    tolerance_lower: float = 0.0

    # Tolerance tightness classification
    tolerance_class: str = "normal"  # "tight" (±.0001), "very_tight" (±.00005), "normal"

    # Surface finish
    surface_finish: Optional[str] = None  # "8 µin", "POLISH CONTOUR"

    # Special callouts
    no_step_permitted: bool = False
    center_permitted: bool = True
    must_be_perpendicular: bool = False

    # Inspection impact
    inspection_minutes_multiplier: float = 1.0  # e.g., 2.0 for very tight
    grind_time_multiplier: float = 1.0


@dataclass
class ComprehensiveFeatures:
    """Complete feature extraction results for quoting."""

    # A. Stock/envelope geometry
    stock_geometry: StockGeometry = field(default_factory=StockGeometry)

    # B.1. Cylindrical and straight sections
    cylindrical_sections: List[CylindricalSection] = field(default_factory=list)
    num_ground_diameters: int = 0
    sum_ground_length: float = 0.0
    has_perp_face_grind: bool = False

    # B.2. Contours, forms, and 3D surfaces
    contour_features: ContourFeature = field(default_factory=ContourFeature)

    # B.3. Holes, taps, and internal features
    hole_features: List[HoleFeature] = field(default_factory=list)
    tap_count: int = 0
    through_hole_count: int = 0
    deep_hole_count: int = 0

    # B.4. Chamfers, corner radii, and undercuts
    edge_features: List[ChamferRadiusFeature] = field(default_factory=list)
    chamfer_count: int = 0
    small_radius_count: int = 0
    undercut_count: int = 0

    # B.5. Tight tolerances and finish/polish
    tolerance_features: List[ToleranceFeature] = field(default_factory=list)
    tight_tolerance_count: int = 0
    requires_polish: bool = False
    polish_handwork_minutes: float = 0.0

    # Summary metrics for quoting
    machining_complexity_score: float = 0.0  # 0-100
    inspection_complexity_score: float = 0.0  # 0-100

    # Raw text for reference
    raw_text_annotations: List[str] = field(default_factory=list)


# ============================================================================
# A. STOCK / ENVELOPE GEOMETRY EXTRACTION
# ============================================================================

class StockGeometryExtractor:
    """Extract stock envelope geometry from CAD geometry and text."""

    @staticmethod
    def extract_from_geometry(geo_features: Dict[str, Any]) -> StockGeometry:
        """
        Extract stock geometry from STEP/IGES geometry features.

        Args:
            geo_features: Dictionary of GEO-* features from geometry/__init__.py

        Returns:
            StockGeometry with envelope dimensions
        """
        stock = StockGeometry()

        # Get bounding box dimensions (convert mm to inches)
        length_mm = geo_features.get("GEO-01_Length_mm", 0.0)
        width_mm = geo_features.get("GEO-02_Width_mm", 0.0)
        height_mm = geo_features.get("GEO-03_Height_mm", 0.0)

        length_in = length_mm / 25.4
        width_in = width_mm / 25.4
        height_in = height_mm / 25.4

        # Determine part type from turning score and geometry
        turning_score = geo_features.get("GEO_Turning_Score_0to1", 0.0)

        if turning_score > 0.7:
            # Likely a round part
            stock.part_type = "round"
            stock.overall_length = length_in
            stock.max_diameter = geo_features.get("GEO_MaxOD_mm", 0.0) / 25.4

            # Estimate min diameter from hole groups or assume 50% of max
            hole_groups = geo_features.get("GEO_Hole_Groups", [])
            if hole_groups:
                min_dia_mm = min(h.get("dia_mm", stock.max_diameter * 25.4) for h in hole_groups)
                stock.min_diameter = min_dia_mm / 25.4
            else:
                stock.min_diameter = stock.max_diameter * 0.5 if stock.max_diameter else None

        else:
            # Likely rectangular or form punch
            stock.part_type = "rectangular"
            stock.body_width = width_in
            stock.body_height = height_in
            stock.centerline_length = length_in
            stock.cross_section_area = width_in * height_in if (width_in and height_in) else None

        # Volume and weight estimation
        volume_mm3 = geo_features.get("GEO-Volume_mm3", 0.0)
        stock.estimated_volume = volume_mm3 / (25.4 ** 3)  # convert to cubic inches

        # Weight depends on material density - assume steel (0.283 lb/in³) as default
        if stock.estimated_volume:
            stock.estimated_weight = stock.estimated_volume * 0.283

        return stock

    @staticmethod
    def extract_from_text(text_records: List[str]) -> StockGeometry:
        """
        Extract stock geometry dimensions from drawing text annotations.

        Args:
            text_records: List of text strings from DXF extraction

        Returns:
            StockGeometry with dimensions found in text
        """
        stock = StockGeometry()

        # Patterns for extracting dimensions
        # Match decimal numbers like .7504 or 0.7504 (prefer leading dot)
        diameter_pattern = r'Ø\s*(\.?\d+\.?\d*)\s*(?:±\s*(\d+\.?\d*))?'
        length_pattern = r'(\d+\.\d+)\s*(?:±[^O]*)?(?:LONG|LENGTH|OAL|O\.?A\.?L\.?)'
        # Cross section pattern - must have decimal points in both numbers to avoid matching degrees
        cross_section_pattern = r'(\.\d+|\d+\.\d+)\s*[×xX]\s*(\.\d+|\d+\.\d+)(?!\s*°)'

        for text in text_records:
            text_upper = text.upper()

            # Look for diameter callouts
            dia_match = re.search(diameter_pattern, text)
            if dia_match:
                dia_val = float(dia_match.group(1))
                if stock.max_diameter is None or dia_val > stock.max_diameter:
                    stock.max_diameter = dia_val

            # Look for length dimensions
            len_match = re.search(length_pattern, text_upper)
            if len_match:
                len_val = float(len_match.group(1))
                if stock.overall_length is None or len_val > stock.overall_length:
                    stock.overall_length = len_val

            # Look for cross-section dimensions (e.g., ".7000 × .5000")
            cross_match = re.search(cross_section_pattern, text)
            if cross_match:
                width_val = float(cross_match.group(1))
                height_val = float(cross_match.group(2))
                stock.body_width = width_val
                stock.body_height = height_val
                stock.cross_section_area = width_val * height_val
                stock.part_type = "rectangular"

        return stock


# ============================================================================
# B.1. CYLINDRICAL AND STRAIGHT SECTIONS
# ============================================================================

class CylindricalSectionExtractor:
    """Extract distinct diameters, widths, and critical lands."""

    @staticmethod
    def extract_from_text(text_records: List[str]) -> List[CylindricalSection]:
        """
        Parse text for diameter/width callouts with tolerances.

        Examples:
            - "Ø.7504 ±.0001"
            - "Ø.5021 +.0000/-.0002"
            - ".4997 +.0000/-.0002"
            - ".62 STRAIGHT TYP"
        """
        sections = []

        # Enhanced pattern for diameter with tolerances
        patterns = [
            # Ø.7504 ±.0001
            r'Ø\s*(\.?\d+\.?\d*)\s*±\s*(\.?\d+\.?\d*)',
            # Ø.5021 +.0000/-.0002
            r'Ø\s*(\.?\d+\.?\d*)\s*\+\s*(\.?\d+\.?\d*)\s*/\s*-\s*(\.?\d+\.?\d*)',
            # .4997 +.0000/-.0002 (no diameter symbol)
            r'(?:^|\s)(\.?\d+\.?\d*)\s*\+\s*(\.?\d+\.?\d*)\s*/\s*-\s*(\.?\d+\.?\d*)',
            # Simple diameter
            r'Ø\s*(\.?\d+\.?\d*)',
        ]

        for text in text_records:
            section = None
            text_upper = text.upper()

            # Try symmetric tolerance (±)
            match = re.search(patterns[0], text)
            if match:
                section = CylindricalSection(
                    diameter=float(match.group(1)),
                    tolerance_upper=float(match.group(2)),
                    tolerance_lower=float(match.group(2)),
                    is_critical=(float(match.group(2)) <= 0.0002)
                )

            # Try asymmetric tolerance (+/-)
            if not section:
                match = re.search(patterns[1], text)
                if match:
                    section = CylindricalSection(
                        diameter=float(match.group(1)),
                        tolerance_upper=float(match.group(2)),
                        tolerance_lower=float(match.group(3)),
                        is_critical=(float(match.group(2)) <= 0.0002 or float(match.group(3)) <= 0.0002)
                    )

            # Check for special callouts
            if section:
                if "STRAIGHT" in text_upper and "TYP" in text_upper:
                    section.is_straight = True
                    # Extract length if present
                    len_match = re.search(r'\.?(\d+\.?\d*)\s+STRAIGHT', text)
                    if len_match:
                        section.length = float(len_match.group(1))

                if "NO STEP" in text_upper:
                    section.no_step_permitted = True
                    section.is_critical = True

                if "PERPENDICULAR" in text_upper and "CENTERLINE" in text_upper:
                    section.has_perp_face = True
                    section.is_critical = True

                # Surface finish
                finish_match = re.search(r'(\d+)\s*[µu]?in', text_upper)
                if finish_match:
                    section.surface_finish = f"{finish_match.group(1)} µin"

                sections.append(section)

        return sections

    @staticmethod
    def aggregate_sections(sections: List[CylindricalSection]) -> Dict[str, Any]:
        """Aggregate section data for quoting metrics."""
        if not sections:
            return {
                "num_ground_diameters": 0,
                "sum_ground_length": 0.0,
                "has_perp_face_grind": False,
            }

        return {
            "num_ground_diameters": len([s for s in sections if s.is_critical]),
            "sum_ground_length": sum(s.length for s in sections),
            "has_perp_face_grind": any(s.has_perp_face for s in sections),
        }


# ============================================================================
# B.2. CONTOURS, FORMS, AND 3D SURFACES
# ============================================================================

class ContourFeatureExtractor:
    """Identify 2D/3D contoured surfaces, wire EDM profiles, coin punches."""

    # Keywords that indicate contoured/formed features
    CONTOUR_KEYWORDS = [
        "POLISH CONTOUR", "COIN", "FORM", "CONTOUR", "BLEND",
        "PROFILE", "WIRE EDM", "WEDM", "FREEFORM"
    ]

    @staticmethod
    def extract_from_text(text_records: List[str]) -> ContourFeature:
        """Detect contoured features from text annotations."""
        contour = ContourFeature()
        radii_found = []

        for text in text_records:
            text_upper = text.upper()

            # Check for contour keywords
            for keyword in ContourFeatureExtractor.CONTOUR_KEYWORDS:
                if keyword in text_upper:
                    contour.keywords.append(keyword)

                    if "POLISH" in keyword or "POLISH" in text_upper:
                        contour.has_polish_contour = True
                    if "COIN" in keyword:
                        contour.has_coin_punch = True
                    if "WIRE" in keyword or "EDM" in keyword:
                        contour.has_2d_wire_edm_profile = True
                    if "FORM" in keyword or "FREEFORM" in keyword or "BLEND" in keyword:
                        contour.has_3d_contoured_nose = True

            # Extract radii (R.005, R.025, R.09, etc.)
            radius_pattern = r'R\s*(\.?\d+\.?\d*)'
            for match in re.finditer(radius_pattern, text):  # Use original text to preserve decimals
                radius_val = float(match.group(1))
                radii_found.append(radius_val)

        # Process radii
        if radii_found:
            contour.radii_list = sorted(radii_found)
            contour.min_radius = min(radii_found)

        return contour

    @staticmethod
    def extract_from_geometry(geo_features: Dict[str, Any]) -> ContourFeature:
        """Extract contour characteristics from 3D geometry analysis."""
        contour = ContourFeature()

        # Check for freeform surfaces
        freeform_area = geo_features.get("GEO_Area_Freeform_mm2", 0.0)
        if freeform_area > 0:
            contour.has_3d_contoured_nose = True
            contour.form_area_sq_in = freeform_area / (25.4 ** 2)

        # Wire EDM path length
        wedm_path = geo_features.get("GEO_WEDM_PathLen_mm", 0.0)
        if wedm_path > 0:
            contour.wire_edm_profile_length = wedm_path / 25.4
            contour.has_2d_wire_edm_profile = True

        # Estimate contour bounding box from overall dimensions
        # (refined analysis would need more detailed surface segmentation)
        length_mm = geo_features.get("GEO-01_Length_mm", 0.0)
        width_mm = geo_features.get("GEO-02_Width_mm", 0.0)
        height_mm = geo_features.get("GEO-03_Height_mm", 0.0)

        if freeform_area > 0:
            # Rough estimate: form area is ~30% of total dimensions
            contour.contour_width = width_mm / 25.4 * 0.3
            contour.contour_depth = height_mm / 25.4 * 0.3
            contour.contour_length = length_mm / 25.4 * 0.3

        return contour


# ============================================================================
# B.3. HOLES, TAPS, AND INTERNAL FEATURES
# ============================================================================

class HoleFeatureExtractor:
    """Extract hole, tap, and thread specifications."""

    @staticmethod
    def extract_from_text(text_records: List[str]) -> List[HoleFeature]:
        """
        Parse text for hole specifications.

        Examples:
            - "5/16-18 TAP X .80 DEEP"
            - "(3) THRU HOLES"
            - "Ø.250 X .500 DEEP"
        """
        holes = []

        # Pattern for tapped holes: 5/16-18 TAP X .80 DEEP
        tap_pattern = r'(\d+/\d+-\d+|M\d+(?:x\d+\.?\d*)?)\s*TAP\s*(?:X\s*\.?(\d+\.?\d*)\s*DEEP)?'

        # Pattern for drilled holes: Ø.250 X .500 DEEP
        drill_pattern = r'Ø\s*\.?(\d+\.?\d*)\s*X\s*\.?(\d+\.?\d*)\s*DEEP'

        # Pattern for through holes
        thru_pattern = r'\((\d+)\)\s*(?:THRU|THROUGH)\s*HOLE'

        for text in text_records:
            text_upper = text.upper()

            # Check for tapped holes
            tap_match = re.search(tap_pattern, text_upper)
            if tap_match:
                hole = HoleFeature(
                    hole_type="threaded",
                    thread_spec=tap_match.group(1),
                    depth=float(tap_match.group(2)) if tap_match.group(2) else None,
                )

                # Extract quantity if present
                qty_match = re.search(r'\((\d+)\)', text)
                if qty_match:
                    hole.count = int(qty_match.group(1))

                holes.append(hole)

            # Check for drilled holes
            drill_match = re.search(drill_pattern, text)
            if drill_match:
                hole = HoleFeature(
                    hole_type="blind",
                    diameter=float(drill_match.group(1)),
                    depth=float(drill_match.group(2)),
                )

                qty_match = re.search(r'\((\d+)\)', text)
                if qty_match:
                    hole.count = int(qty_match.group(1))

                holes.append(hole)

            # Check for through holes
            thru_match = re.search(thru_pattern, text_upper)
            if thru_match:
                hole = HoleFeature(
                    hole_type="through",
                    is_through=True,
                    count=int(thru_match.group(1)),
                )
                holes.append(hole)

        return holes

    @staticmethod
    def extract_from_geometry(geo_features: Dict[str, Any]) -> List[HoleFeature]:
        """Extract holes from 3D geometry analysis."""
        holes = []

        hole_groups = geo_features.get("GEO_Hole_Groups", [])
        for group in hole_groups:
            hole = HoleFeature(
                hole_type="through" if group.get("through", False) else "blind",
                diameter=group.get("dia_mm", 0.0) / 25.4,  # convert to inches
                depth=group.get("depth_mm", 0.0) / 25.4,
                is_through=group.get("through", False),
                count=group.get("count", 1),
            )
            holes.append(hole)

        return holes


# ============================================================================
# B.4. CHAMFERS, CORNER RADII, AND UNDERCUTS
# ============================================================================

class ChamferRadiusExtractor:
    """Extract edge features: chamfers, radii, undercuts."""

    @staticmethod
    def extract_from_text(text_records: List[str]) -> List[ChamferRadiusFeature]:
        """
        Parse text for chamfers, radii, and undercuts.

        Examples:
            - "(3) .040 × 45°"
            - "R.005"
            - "(2) SMALL UNDERCUT"
        """
        features = []

        # Chamfer pattern: .040 × 45° or .040 X 45
        chamfer_pattern = r'(\.?\d+\.?\d*)\s*[×xX]\s*(\d+)°?'

        # Radius pattern: R.005, R.025
        radius_pattern = r'R\s*(\.?\d+\.?\d*)'

        # Undercut pattern
        undercut_pattern = r'UNDERCUT'

        for text in text_records:
            text_upper = text.upper()

            # Extract quantity prefix if present: (3), (2), etc.
            qty_match = re.search(r'\((\d+)\)', text)
            qty = int(qty_match.group(1)) if qty_match else 1

            # Check for chamfers
            chamfer_match = re.search(chamfer_pattern, text)
            if chamfer_match:
                feature = ChamferRadiusFeature(
                    feature_type="chamfer",
                    chamfer_size=float(chamfer_match.group(1)),
                    chamfer_angle=float(chamfer_match.group(2)),
                    count=qty,
                )
                features.append(feature)

            # Check for radii
            radius_match = re.search(radius_pattern, text_upper)
            if radius_match:
                radius_val = float(radius_match.group(1))
                feature = ChamferRadiusFeature(
                    feature_type="radius",
                    radius_size=radius_val,
                    is_small_radius=(radius_val <= 0.010),  # R.010 or smaller
                    count=qty,
                )
                features.append(feature)

            # Check for undercuts
            if undercut_pattern in text_upper:
                feature = ChamferRadiusFeature(
                    feature_type="undercut",
                    is_undercut=True,
                    count=qty,
                )
                features.append(feature)

        return features


# ============================================================================
# B.5. TIGHT TOLERANCES AND FINISH/POLISH
# ============================================================================

class ToleranceFinishExtractor:
    """Extract tolerance and surface finish requirements."""

    @staticmethod
    def extract_from_text(text_records: List[str]) -> List[ToleranceFeature]:
        """
        Parse text for tight tolerances and finish requirements.

        Examples:
            - "Ø.7504 ±.0001"
            - ".2497 +.0000/-.0002"
            - "POLISH CONTOUR"
            - "8 µin finish"
            - "NO STEP PERMITTED"
        """
        tolerances = []

        # Symmetric tolerance pattern: ±.0001
        sym_tol_pattern = r'\.?(\d+\.?\d*)\s*±\s*\.?(\d+\.?\d*)'

        # Asymmetric tolerance: +.0000/-.0002
        asym_tol_pattern = r'\.?(\d+\.?\d*)\s*\+\s*\.?(\d+\.?\d*)\s*/\s*-\s*\.?(\d+\.?\d*)'

        # Surface finish: 8 µin, 16 RA, etc.
        finish_pattern = r'(\d+)\s*(?:µin|[µu]in|RA|RMS)'

        for text in text_records:
            text_upper = text.upper()

            # Check symmetric tolerance
            match = re.search(sym_tol_pattern, text)
            if match:
                nominal = float(match.group(1))
                tol = float(match.group(2))

                # Classify tolerance tightness
                if tol <= 0.00005:
                    tol_class = "very_tight"
                    inspect_mult = 3.0
                    grind_mult = 2.0
                elif tol <= 0.0002:
                    tol_class = "tight"
                    inspect_mult = 2.0
                    grind_mult = 1.5
                else:
                    tol_class = "normal"
                    inspect_mult = 1.0
                    grind_mult = 1.0

                feature = ToleranceFeature(
                    feature_type="diameter",
                    nominal_value=nominal,
                    tolerance_upper=tol,
                    tolerance_lower=tol,
                    tolerance_class=tol_class,
                    inspection_minutes_multiplier=inspect_mult,
                    grind_time_multiplier=grind_mult,
                )

                # Check for special callouts
                if "NO STEP" in text_upper:
                    feature.no_step_permitted = True
                if "CENTER PERMITTED" in text_upper:
                    feature.center_permitted = True
                elif "CENTER NOT PERMITTED" in text_upper:
                    feature.center_permitted = False

                tolerances.append(feature)

            # Check asymmetric tolerance
            match = re.search(asym_tol_pattern, text)
            if match:
                nominal = float(match.group(1))
                tol_upper = float(match.group(2))
                tol_lower = float(match.group(3))
                max_tol = max(tol_upper, tol_lower)

                if max_tol <= 0.00005:
                    tol_class = "very_tight"
                    inspect_mult = 3.0
                    grind_mult = 2.0
                elif max_tol <= 0.0002:
                    tol_class = "tight"
                    inspect_mult = 2.0
                    grind_mult = 1.5
                else:
                    tol_class = "normal"
                    inspect_mult = 1.0
                    grind_mult = 1.0

                feature = ToleranceFeature(
                    feature_type="diameter",
                    nominal_value=nominal,
                    tolerance_upper=tol_upper,
                    tolerance_lower=tol_lower,
                    tolerance_class=tol_class,
                    inspection_minutes_multiplier=inspect_mult,
                    grind_time_multiplier=grind_mult,
                )
                tolerances.append(feature)

            # Check for surface finish
            finish_match = re.search(finish_pattern, text_upper)
            if finish_match:
                finish_val = int(finish_match.group(1))
                feature = ToleranceFeature(
                    feature_type="surface_finish",
                    surface_finish=f"{finish_val} µin",
                    inspection_minutes_multiplier=1.5 if finish_val <= 16 else 1.0,
                    grind_time_multiplier=1.5 if finish_val <= 16 else 1.0,
                )
                tolerances.append(feature)

            # Check for polish requirement
            if "POLISH" in text_upper:
                feature = ToleranceFeature(
                    feature_type="surface_finish",
                    surface_finish="POLISH",
                    inspection_minutes_multiplier=2.0,
                    grind_time_multiplier=2.0,
                )
                tolerances.append(feature)

        return tolerances


# ============================================================================
# MAIN EXTRACTION ORCHESTRATOR
# ============================================================================

class ComprehensiveFeatureExtractor:
    """Main orchestrator for extracting all features from CAD files."""

    def __init__(self):
        self.stock_extractor = StockGeometryExtractor()
        self.cylindrical_extractor = CylindricalSectionExtractor()
        self.contour_extractor = ContourFeatureExtractor()
        self.hole_extractor = HoleFeatureExtractor()
        self.chamfer_extractor = ChamferRadiusExtractor()
        self.tolerance_extractor = ToleranceFinishExtractor()

    def extract_all_features(
        self,
        text_records: Optional[List[str]] = None,
        geo_features: Optional[Dict[str, Any]] = None,
    ) -> ComprehensiveFeatures:
        """
        Extract all features from CAD text and geometry.

        Args:
            text_records: List of text strings extracted from DXF/drawing
            geo_features: Dictionary of GEO-* features from STEP/IGES analysis

        Returns:
            ComprehensiveFeatures with all extracted data
        """
        features = ComprehensiveFeatures()

        text_records = text_records or []
        geo_features = geo_features or {}

        # A. Stock/envelope geometry
        if geo_features:
            features.stock_geometry = self.stock_extractor.extract_from_geometry(geo_features)
        if text_records:
            text_stock = self.stock_extractor.extract_from_text(text_records)
            # Merge text dimensions into geometry-based stock
            self._merge_stock_geometry(features.stock_geometry, text_stock)

        # B.1. Cylindrical and straight sections
        if text_records:
            features.cylindrical_sections = self.cylindrical_extractor.extract_from_text(text_records)
            agg = self.cylindrical_extractor.aggregate_sections(features.cylindrical_sections)
            features.num_ground_diameters = agg["num_ground_diameters"]
            features.sum_ground_length = agg["sum_ground_length"]
            features.has_perp_face_grind = agg["has_perp_face_grind"]

        # B.2. Contours, forms, and 3D surfaces
        if text_records:
            features.contour_features = self.contour_extractor.extract_from_text(text_records)
        if geo_features:
            geo_contour = self.contour_extractor.extract_from_geometry(geo_features)
            self._merge_contour_features(features.contour_features, geo_contour)

        # B.3. Holes, taps, and internal features
        if text_records:
            features.hole_features.extend(self.hole_extractor.extract_from_text(text_records))
        if geo_features:
            features.hole_features.extend(self.hole_extractor.extract_from_geometry(geo_features))

        # Aggregate hole counts
        features.tap_count = len([h for h in features.hole_features if h.hole_type == "threaded"])
        features.through_hole_count = len([h for h in features.hole_features if h.is_through])
        features.deep_hole_count = len([h for h in features.hole_features
                                       if h.depth and h.depth > (h.diameter * 5 if h.diameter else 0)])

        # B.4. Chamfers, corner radii, and undercuts
        if text_records:
            features.edge_features = self.chamfer_extractor.extract_from_text(text_records)
            features.chamfer_count = sum(f.count for f in features.edge_features if f.feature_type == "chamfer")
            features.small_radius_count = sum(f.count for f in features.edge_features if f.is_small_radius)
            features.undercut_count = sum(f.count for f in features.edge_features if f.is_undercut)

        # B.5. Tight tolerances and finish/polish
        if text_records:
            features.tolerance_features = self.tolerance_extractor.extract_from_text(text_records)
            features.tight_tolerance_count = len([t for t in features.tolerance_features
                                                  if t.tolerance_class in ["tight", "very_tight"]])
            features.requires_polish = any("POLISH" in (t.surface_finish or "")
                                          for t in features.tolerance_features)

            # Estimate polish handwork time
            if features.requires_polish:
                # Base: 10 min for polish setup + 5 min per tight tolerance
                features.polish_handwork_minutes = 10.0 + (features.tight_tolerance_count * 5.0)

        # Calculate complexity scores
        features.machining_complexity_score = self._calculate_machining_complexity(features)
        features.inspection_complexity_score = self._calculate_inspection_complexity(features)

        # Store raw text for reference
        features.raw_text_annotations = text_records[:100]  # Keep first 100 for debugging

        return features

    def _merge_stock_geometry(self, base: StockGeometry, overlay: StockGeometry):
        """Merge text-based dimensions into geometry-based stock."""
        if overlay.overall_length and not base.overall_length:
            base.overall_length = overlay.overall_length
        if overlay.max_diameter and not base.max_diameter:
            base.max_diameter = overlay.max_diameter
        if overlay.body_width and not base.body_width:
            base.body_width = overlay.body_width
        if overlay.body_height and not base.body_height:
            base.body_height = overlay.body_height

    def _merge_contour_features(self, base: ContourFeature, overlay: ContourFeature):
        """Merge geometry-based contour features into text-based."""
        if overlay.has_2d_wire_edm_profile:
            base.has_2d_wire_edm_profile = True
        if overlay.has_3d_contoured_nose:
            base.has_3d_contoured_nose = True
        if overlay.wire_edm_profile_length and not base.wire_edm_profile_length:
            base.wire_edm_profile_length = overlay.wire_edm_profile_length
        if overlay.form_area_sq_in and not base.form_area_sq_in:
            base.form_area_sq_in = overlay.form_area_sq_in

    def _calculate_machining_complexity(self, features: ComprehensiveFeatures) -> float:
        """Calculate machining complexity score (0-100)."""
        score = 0.0

        # Base complexity from number of operations
        score += features.num_ground_diameters * 5.0
        score += features.tap_count * 3.0
        score += features.chamfer_count * 1.0
        score += features.small_radius_count * 5.0  # Small radii are hard
        score += features.undercut_count * 8.0  # Undercuts are very hard

        # Contour complexity
        if features.contour_features.has_3d_contoured_nose:
            score += 20.0
        if features.contour_features.has_2d_wire_edm_profile:
            score += 15.0

        # Tight tolerance penalty
        score += features.tight_tolerance_count * 3.0

        # Polish requirement
        if features.requires_polish:
            score += 10.0

        return min(score, 100.0)  # Cap at 100

    def _calculate_inspection_complexity(self, features: ComprehensiveFeatures) -> float:
        """Calculate inspection complexity score (0-100)."""
        score = 0.0

        # Tight tolerances drive inspection time
        score += features.tight_tolerance_count * 8.0

        # Each critical dimension to measure
        score += len(features.cylindrical_sections) * 2.0

        # Surface finish requirements
        finish_count = len([t for t in features.tolerance_features
                           if t.feature_type == "surface_finish"])
        score += finish_count * 5.0

        # Perpendicular face requirements
        if features.has_perp_face_grind:
            score += 10.0

        # Contoured surfaces are hard to inspect
        if features.contour_features.has_3d_contoured_nose:
            score += 15.0

        return min(score, 100.0)  # Cap at 100


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def extract_features_from_cad_file(
    cad_file_path: Path,
    text_records: Optional[List[str]] = None,
) -> ComprehensiveFeatures:
    """
    High-level function to extract all features from a CAD file.

    Args:
        cad_file_path: Path to CAD file (STEP, IGES, DXF, etc.)
        text_records: Optional pre-extracted text records from DXF

    Returns:
        ComprehensiveFeatures with all extracted data
    """
    from cad_quoter.geometry import extract_features_with_occ

    extractor = ComprehensiveFeatureExtractor()

    # Extract geometry features if STEP/IGES file
    geo_features = {}
    if cad_file_path.suffix.lower() in ['.step', '.stp', '.iges', '.igs', '.brep']:
        try:
            geo_features = extract_features_with_occ(cad_file_path)
        except Exception as e:
            print(f"Warning: Could not extract geometry features: {e}")

    # Extract text if not provided and file is DXF
    if not text_records and cad_file_path.suffix.lower() == '.dxf':
        try:
            from cad_quoter.geo_extractor import iter_text_records, open_doc
            doc = open_doc(cad_file_path)
            text_records = [rec.text for rec in iter_text_records(doc)]
        except Exception as e:
            print(f"Warning: Could not extract text records: {e}")
            text_records = []

    return extractor.extract_all_features(
        text_records=text_records,
        geo_features=geo_features,
    )


def features_to_dict(features: ComprehensiveFeatures) -> Dict[str, Any]:
    """Convert ComprehensiveFeatures to dictionary for JSON serialization."""
    from dataclasses import asdict
    return asdict(features)


def features_to_quoting_variables(features: ComprehensiveFeatures) -> Dict[str, float]:
    """
    Convert extracted features to quoting system variables.

    Returns a dictionary of variable_name: value pairs that can be used
    directly in the quoting formulas.
    """
    variables = {}

    # Stock geometry
    if features.stock_geometry.max_diameter:
        variables["STOCK_MAX_DIAMETER"] = features.stock_geometry.max_diameter
    if features.stock_geometry.overall_length:
        variables["STOCK_LENGTH"] = features.stock_geometry.overall_length
    if features.stock_geometry.estimated_volume:
        variables["STOCK_VOLUME_CU_IN"] = features.stock_geometry.estimated_volume
    if features.stock_geometry.estimated_weight:
        variables["STOCK_WEIGHT_LB"] = features.stock_geometry.estimated_weight

    # Machining metrics
    variables["NUM_GROUND_DIAMETERS"] = float(features.num_ground_diameters)
    variables["SUM_GROUND_LENGTH"] = features.sum_ground_length
    variables["TAP_COUNT"] = float(features.tap_count)
    variables["THROUGH_HOLE_COUNT"] = float(features.through_hole_count)
    variables["DEEP_HOLE_COUNT"] = float(features.deep_hole_count)
    variables["CHAMFER_COUNT"] = float(features.chamfer_count)
    variables["SMALL_RADIUS_COUNT"] = float(features.small_radius_count)
    variables["UNDERCUT_COUNT"] = float(features.undercut_count)

    # Complexity scores
    variables["MACHINING_COMPLEXITY"] = features.machining_complexity_score
    variables["INSPECTION_COMPLEXITY"] = features.inspection_complexity_score

    # Tolerance and finish
    variables["TIGHT_TOLERANCE_COUNT"] = float(features.tight_tolerance_count)
    variables["POLISH_HANDWORK_MIN"] = features.polish_handwork_minutes

    # Contour features
    variables["HAS_3D_CONTOUR"] = 1.0 if features.contour_features.has_3d_contoured_nose else 0.0
    variables["HAS_WIRE_EDM"] = 1.0 if features.contour_features.has_2d_wire_edm_profile else 0.0
    variables["HAS_COIN_PUNCH"] = 1.0 if features.contour_features.has_coin_punch else 0.0

    if features.contour_features.wire_edm_profile_length:
        variables["WIRE_EDM_LENGTH_IN"] = features.contour_features.wire_edm_profile_length
    if features.contour_features.form_area_sq_in:
        variables["FORM_AREA_SQ_IN"] = features.contour_features.form_area_sq_in

    # Special flags
    variables["HAS_PERP_FACE_GRIND"] = 1.0 if features.has_perp_face_grind else 0.0
    variables["REQUIRES_POLISH"] = 1.0 if features.requires_polish else 0.0

    return variables
