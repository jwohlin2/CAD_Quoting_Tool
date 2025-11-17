"""
DWG Punch Feature Extractor - Automated Feature Extraction from 2D Drawings
===========================================================================

Extracts manufacturing features from DWG/DXF punch drawings for automated quoting.
Handles punch drawings that have NO STEP file by extracting geometry, dimensions,
and text to build a structured feature summary.

Based on: docs/DWG_Punch_Extraction_Readme.md

Key capabilities:
    - Geometry extraction (bbox, envelope) using ezdxf
    - Dimension mining (measurements + tolerances)
    - Text-based classification (family, shape, material)
    - Operations detection (grinding, tapping, chamfers, etc.)
    - Pain flags (tight tolerances, polish, GD&T, etc.)

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import ezdxf
    from ezdxf.bbox import extents
except ImportError:
    ezdxf = None  # type: ignore
    extents = None  # type: ignore


# ============================================================================
# MTEXT NORMALIZATION AND DIMENSION TEXT HELPERS
# ============================================================================


def normalize_acad_mtext(line: str) -> str:
    """
    Normalize AutoCAD MTEXT formatting codes into simpler plain text.

    Handles:
    - Strip outer {...}
    - Remove \\Hxx; (height) and \\Cxx; (color)
    - Convert stacked text \\S+.005^ -.000; -> '+.005/-.000'
    - Remove leftover '{}' braces

    Examples:
        "{\\H0.71x;\\C3;\\S+.005^ -.000;}" -> "+.005/-.000"
        "{\\H1.0x;TEXT}" -> "TEXT"

    Args:
        line: Raw MTEXT string with formatting codes

    Returns:
        Normalized plain text string
    """
    if not line:
        return ""

    # Strip single outer braces
    if line.startswith("{") and line.endswith("}"):
        line = line[1:-1]

    # Remove height codes like \H0.71x;
    line = re.sub(r"\\H[0-9.]+x;", "", line)

    # Remove color codes like \C3;
    line = re.sub(r"\\C\d+;", "", line)

    # Convert stacked text: \S top ^ bottom ;
    def repl_stack(m):
        top = m.group(1).strip()
        bot = m.group(2).strip()
        return f"{top}/{bot}"

    line = re.sub(r"\\S([^\\^]+)\^([^;]+);", repl_stack, line)

    # Remove empty {} blocks
    line = line.replace("{}", "").strip()

    return line


def units_to_inch_factor(insunits: int) -> float:
    """
    Convert DXF $INSUNITS code to inch conversion factor.

    Args:
        insunits: Value from $INSUNITS header variable
            0 = Unitless
            1 = Inches
            2 = Feet
            4 = Millimeters
            5 = Centimeters
            6 = Meters

    Returns:
        Multiplication factor to convert to inches
    """
    units_factors = {
        0: 1.0,          # Unitless - assume inches
        1: 1.0,          # Inches
        2: 12.0,         # Feet -> inches
        4: 1.0 / 25.4,   # Millimeters -> inches
        5: 1.0 / 2.54,   # Centimeters -> inches
        6: 39.3701,      # Meters -> inches
    }
    return units_factors.get(insunits, 1.0)


def resolved_dimension_text(dim, unit_factor: float) -> str:
    """
    Given an ezdxf DIMENSION entity and a unit conversion factor,
    return a resolved text string with:
    - <> placeholder replaced by numeric measurement
    - MTEXT formatting normalized

    Examples:
        dim.dxf.text = "(2) <>"
        meas = 0.1480 (in drawing units)
        unit_factor = 1.0 (inches)
        Result: "(2) .148"

        dim.dxf.text = "<> {\\H0.71x;\\C3;\\S+.0000^ -.0002;}"
        meas = 0.4997
        Result: ".4997 +.0000/-.0002"

    Args:
        dim: ezdxf DIMENSION entity
        unit_factor: Multiplication factor to convert to inches

    Returns:
        Resolved dimension text string
    """
    raw_text = dim.dxf.text if hasattr(dim.dxf, 'text') else ""

    # Get numeric measurement
    try:
        meas = dim.get_measurement()
        if meas is None:
            meas = 0

        # Handle Vec3 objects
        if hasattr(meas, 'magnitude'):
            meas = meas.magnitude
        elif hasattr(meas, 'x'):
            meas = abs(meas.x)

        meas = float(meas)
    except Exception:
        meas = 0

    # Convert to inches
    value_in = meas * unit_factor

    # Format nominal value
    # Use 4 decimal places, strip trailing zeros
    nominal_str = f"{value_in:.4f}".rstrip("0").rstrip(".")

    # If less than 1.0 and starts with "0.", convert to ".XXX" style
    if nominal_str.startswith("0.") and value_in < 1.0:
        nominal_str = nominal_str[1:]  # ".148" instead of "0.148"
    elif not nominal_str or nominal_str == ".":
        nominal_str = "0"

    # First normalize MTEXT formatting
    text = normalize_acad_mtext(raw_text) if raw_text else ""

    # Replace <> placeholder with numeric value
    if "<>" in text and nominal_str:
        text = text.replace("<>", nominal_str)
    elif not text and nominal_str:
        # No override text at all; just use the numeric string
        text = nominal_str

    return text.strip()


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class PunchFeatureSummary:
    """
    Comprehensive feature summary for punch parts extracted from DWG/DXF.

    This structure provides everything needed for punch quoting:
    - Classification (family, shape)
    - Size/envelope (length, diameter/width)
    - Operations drivers (grinding, tapping, form work)
    - Pain/quality flags (tolerances, polish, GD&T)
    """

    # === CLASSIFICATION ===
    family: str = "round_punch"  # round_punch, pilot_pin, form_punch, die_section, etc.
    shape_type: str = "round"  # round or rectangular

    # === SIZE / STOCK (envelope) ===
    overall_length_in: float = 0.0
    max_od_or_width_in: float = 0.0
    body_width_in: Optional[float] = None  # for rectangular punches
    body_thickness_in: Optional[float] = None  # for rectangular punches
    form_length_in: Optional[float] = None  # length of contoured nose region

    # === OPS DRIVERS - Grinding / Turning ===
    num_ground_diams: int = 0  # count of distinct ground diameters
    total_ground_length_in: float = 0.0
    has_perp_face_grind: bool = False  # "THIS SURFACE PERPENDICULAR TO CENTERLINE"

    # === OPS DRIVERS - Form nose ===
    has_3d_surface: bool = False  # form/coin/polish contour
    form_complexity_level: int = 0  # 0-3 scale

    # === OPS DRIVERS - Holes & undercuts ===
    tap_count: int = 0
    tap_summary: List[Dict[str, Any]] = field(default_factory=list)  # {size, depth_in}
    num_undercuts: int = 0

    # === OPS DRIVERS - Edge work ===
    num_chamfers: int = 0
    num_small_radii: int = 0  # tiny R that are slow to grind/EDM

    # === PAIN / QUALITY LEVEL ===
    min_dia_tol_in: Optional[float] = None  # tightest diameter tolerance
    min_len_tol_in: Optional[float] = None  # tightest length tolerance
    has_polish_contour: bool = False
    has_no_step_permitted: bool = False
    has_sharp_edges: bool = False
    has_gdt: bool = False  # any GD&T frames present

    # === MATERIAL (for reference) ===
    material_callout: Optional[str] = None  # e.g., "A2", "D2", "M2", "CARBIDE"

    # === METADATA ===
    extraction_source: str = "dxf_geometry_and_text"  # vs "step_model"
    confidence_score: float = 1.0  # 0-1, quality of extraction
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# GEOMETRY EXTRACTION (BBOX)
# ============================================================================


def extract_geometry_envelope(dxf_path: Path) -> Dict[str, Any]:
    """
    Extract bounding box and envelope dimensions from DXF geometry.

    Args:
        dxf_path: Path to DXF file

    Returns:
        Dict with:
            - overall_length_in: float
            - overall_width_in: float
            - overall_height_in: float (for 3D, usually 0 for 2D)
            - bbox_min: tuple (x, y, z)
            - bbox_max: tuple (x, y, z)
            - units: str ("inches" or "mm")
    """
    if ezdxf is None:
        return {
            "overall_length_in": 0.0,
            "overall_width_in": 0.0,
            "overall_height_in": 0.0,
            "bbox_min": (0, 0, 0),
            "bbox_max": (0, 0, 0),
            "units": "inches",
            "error": "ezdxf not available"
        }

    try:
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()

        # Get outline entities (exclude text)
        outline_entities = msp.query("LINE ARC CIRCLE LWPOLYLINE POLYLINE SPLINE")

        if not outline_entities:
            return {
                "overall_length_in": 0.0,
                "overall_width_in": 0.0,
                "overall_height_in": 0.0,
                "bbox_min": (0, 0, 0),
                "bbox_max": (0, 0, 0),
                "units": "inches",
                "error": "no geometry entities found"
            }

        # Compute bounding box
        bbox = extents(outline_entities)
        min_pt = bbox.extmin
        max_pt = bbox.extmax

        # Calculate dimensions
        length = max_pt.x - min_pt.x
        width = max_pt.y - min_pt.y
        height = max_pt.z - min_pt.z if len(max_pt) > 2 else 0.0

        # Determine units from DXF header
        insunits = doc.header.get("$INSUNITS", 1)  # 1=inches, 4=mm, 0=unitless
        measurement = doc.header.get("$MEASUREMENT", 0)  # 0=Imperial, 1=Metric

        units_map = {
            0: "unitless",
            1: "inches",
            2: "feet",
            4: "mm",
            5: "cm",
            6: "meters",
        }
        units_name = units_map.get(insunits, "inches")

        # Robust units detection: if $MEASUREMENT says metric, trust that over $INSUNITS
        # Also use heuristic: if dimensions are very large (>50), likely in mm
        is_metric = False
        if measurement == 1:  # $MEASUREMENT indicates metric
            is_metric = True
        elif insunits == 4:  # Explicitly mm
            is_metric = True
        elif insunits == 0 or insunits == 1:  # Unitless or inches - use heuristic
            # If dimensions are suspiciously large, assume mm
            if max(length, width) > 50:  # Unlikely to have 50+ inch punch
                is_metric = True

        # Convert to inches if metric
        if is_metric:
            length /= 25.4
            width /= 25.4
            height /= 25.4
            units_name = "mm (converted)"
        elif units_name == "feet":
            length *= 12
            width *= 12
            height *= 12

        return {
            "overall_length_in": length,
            "overall_width_in": width,
            "overall_height_in": height,
            "bbox_min": tuple(min_pt),
            "bbox_max": tuple(max_pt),
            "units": units_name,
        }

    except Exception as e:
        return {
            "overall_length_in": 0.0,
            "overall_width_in": 0.0,
            "overall_height_in": 0.0,
            "bbox_min": (0, 0, 0),
            "bbox_max": (0, 0, 0),
            "units": "inches",
            "error": str(e)
        }


# ============================================================================
# DIMENSION MINING (MEASUREMENTS + TOLERANCES)
# ============================================================================


def extract_dimensions(dxf_path: Path) -> Dict[str, Any]:
    """
    Extract dimension measurements and tolerances from DIMENSION entities.

    Args:
        dxf_path: Path to DXF file

    Returns:
        Dict with:
            - linear_dims: List[Dict] with measurement, text, type
            - diameter_dims: List[Dict] with measurement, text
            - max_linear_dim: float (inches)
            - max_diameter_dim: float (inches)
            - min_dia_tol: float or None
            - min_len_tol: float or None
            - all_tolerances: List[float]
    """
    if ezdxf is None:
        return {
            "linear_dims": [],
            "diameter_dims": [],
            "resolved_dim_texts": [],
            "max_linear_dim": 0.0,
            "max_diameter_dim": 0.0,
            "min_dia_tol": None,
            "min_len_tol": None,
            "all_tolerances": [],
            "error": "ezdxf not available"
        }

    try:
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()

        # Determine units - robust detection
        insunits = doc.header.get("$INSUNITS", 1)
        measurement = doc.header.get("$MEASUREMENT", 0)  # 0=Imperial, 1=Metric

        # Get unit conversion factor to inches
        unit_factor = units_to_inch_factor(insunits)

        # Also check $MEASUREMENT for additional validation
        is_metric = False
        if measurement == 1:  # $MEASUREMENT indicates metric
            is_metric = True
            if insunits not in [4, 5, 6]:  # Not already metric
                unit_factor = 1.0 / 25.4
        elif insunits == 4:  # Explicitly mm
            is_metric = True

        linear_dims = []
        diameter_dims = []
        all_tolerances = []
        resolved_dim_texts = []  # For tolerance/pattern detection

        for dim in msp.query("DIMENSION"):
            try:
                # Get resolved dimension text (handles <> and MTEXT)
                text_resolved = resolved_dimension_text(dim, unit_factor)
                resolved_dim_texts.append(text_resolved)

                # Get raw text for diameter detection
                raw_text = dim.dxf.text if hasattr(dim.dxf, 'text') else ""

                # Get measurement value (already converted to inches in resolved_dimension_text)
                meas = dim.get_measurement()
                if meas is None:
                    continue

                # Handle Vec3 objects (convert to scalar)
                if hasattr(meas, 'magnitude'):
                    meas = meas.magnitude
                elif hasattr(meas, 'x'):
                    meas = abs(meas.x)

                meas = float(meas)
                meas_in = meas * unit_factor

                # Get dimension type
                # dimtype: 0=linear, 1=aligned, 3=diameter, 4=radius, etc.
                dimtype = dim.dimtype

                # Classify as linear or diameter
                # Check: dimtype, %%c (old DXF diameter symbol), Ø, DIA
                is_diameter = (
                    dimtype == 3 or
                    "%%c" in raw_text.lower() or
                    "Ø" in raw_text or
                    "Ø" in text_resolved or
                    "DIA" in raw_text.upper()
                )

                if is_diameter:
                    diameter_dims.append({
                        "measurement": meas_in,
                        "text": text_resolved,
                        "raw_text": raw_text,
                        "type": "diameter"
                    })
                else:
                    linear_dims.append({
                        "measurement": meas_in,
                        "text": text_resolved,
                        "raw_text": raw_text,
                        "type": dimtype
                    })

                # Extract tolerances from resolved text
                tolerances = parse_tolerances_from_text(text_resolved)
                all_tolerances.extend(tolerances)

            except Exception as e:
                # Skip dimensions that can't be processed
                continue

        # Find max dimensions
        max_linear = max([d["measurement"] for d in linear_dims], default=0.0)
        max_diameter = max([d["measurement"] for d in diameter_dims], default=0.0)

        # Heuristic check: if dimensions are very large and we didn't convert, do it now
        # This catches cases where $INSUNITS and $MEASUREMENT are wrong
        if not is_metric and (max_linear > 50 or max_diameter > 10):
            # Suspiciously large dimensions - likely mm that wasn't converted
            for d in linear_dims:
                d["measurement"] /= 25.4
            for d in diameter_dims:
                d["measurement"] /= 25.4
            max_linear /= 25.4
            max_diameter /= 25.4

        # Find tightest tolerances
        min_tol = min(all_tolerances, default=None) if all_tolerances else None

        # Separate diameter and length tolerances (simple heuristic)
        dia_tolerances = []
        len_tolerances = []

        for dim in diameter_dims:
            tols = parse_tolerances_from_text(dim["text"])
            dia_tolerances.extend(tols)

        for dim in linear_dims:
            tols = parse_tolerances_from_text(dim["text"])
            len_tolerances.extend(tols)

        min_dia_tol = min(dia_tolerances, default=None) if dia_tolerances else None
        min_len_tol = min(len_tolerances, default=None) if len_tolerances else None

        return {
            "linear_dims": linear_dims,
            "diameter_dims": diameter_dims,
            "resolved_dim_texts": resolved_dim_texts,  # For additional pattern matching
            "max_linear_dim": max_linear,
            "max_diameter_dim": max_diameter,
            "min_dia_tol": min_dia_tol,
            "min_len_tol": min_len_tol,
            "all_tolerances": all_tolerances,
        }

    except Exception as e:
        return {
            "linear_dims": [],
            "diameter_dims": [],
            "resolved_dim_texts": [],
            "max_linear_dim": 0.0,
            "max_diameter_dim": 0.0,
            "min_dia_tol": None,
            "min_len_tol": None,
            "all_tolerances": [],
            "error": str(e)
        }


def parse_tolerances_from_text(text: str) -> List[float]:
    """
    Parse tolerance values from dimension text.

    Examples:
        "±0.001" -> [0.001]
        "+0.0000-0.0002" -> [0.0000, 0.0002]
        "+.0000/-.0002" -> [0.0000, 0.0002]
        "+ .0002 / - .0000" -> [0.0002, 0.0000]
        "1.234±.0005" -> [0.0005]

    Args:
        text: Dimension text string

    Returns:
        List of tolerance values (absolute, in inches)
    """
    tolerances = []
    matched_ranges = []  # Track matched spans to avoid duplicates

    def add_match(start, end, values):
        """Add tolerance values if this range doesn't overlap existing matches."""
        for s, e in matched_ranges:
            if not (end <= s or start >= e):  # Overlaps
                return False
        matched_ranges.append((start, end))
        tolerances.extend(values)
        return True

    # Pattern: ±0.000X or ± 0.000X
    pm_pattern = r'±\s*(\d*\.?\d+)'
    for match in re.finditer(pm_pattern, text):
        tol = float(match.group(1))
        add_match(match.start(), match.end(), [abs(tol)])

    # Pattern: +.0000/-.0002 or + .0000 / - .0002 (with slash separator) - PRIORITY
    slash_pattern = r'\+\s*(\d*\.?\d+)\s*/\s*-\s*(\d*\.?\d+)'
    for match in re.finditer(slash_pattern, text):
        tol_plus = float(match.group(1))
        tol_minus = float(match.group(2))
        add_match(match.start(), match.end(), [abs(tol_plus), abs(tol_minus)])

    # Pattern: +0.000X -0.000Y or +0.000X-0.000Y (without slash) - check for non-overlap
    # Only match if not already matched by slash pattern
    plus_minus_pattern = r'\+\s*(\d*\.?\d+)\s*-\s*(\d*\.?\d+)'
    for match in re.finditer(plus_minus_pattern, text):
        # Skip if this was already matched by slash pattern
        if any(s <= match.start() < e or s < match.end() <= e for s, e in matched_ranges):
            continue
        tol_plus = float(match.group(1))
        tol_minus = float(match.group(2))
        add_match(match.start(), match.end(), [abs(tol_plus), abs(tol_minus)])

    return tolerances


# ============================================================================
# TEXT-BASED CLASSIFICATION & FEATURE DETECTION
# ============================================================================


def classify_punch_family(text_dump: str) -> Tuple[str, str]:
    """
    Classify punch family and shape type from text.

    Priority order (check most specific first):
    1. Specific multi-word terms (PILOT PIN, FORM PUNCH, etc.)
    2. Form/contour indicators (COIN, FORM, INSERT)
    3. Generic PUNCH (only if no other indicators)

    Args:
        text_dump: Combined text from drawing (title block, notes, etc.)

    Returns:
        Tuple of (family, shape_type)

    Families:
        - round_punch
        - pilot_pin
        - form_punch
        - die_section
        - guide_post
        - bushing
        - die_insert

    Shapes:
        - round
        - rectangular
    """
    text_upper = text_dump.upper()

    # Determine family based on keywords (most specific first)
    family = None

    # Check for specific multi-word terms first
    if "PILOT PIN" in text_upper or "PILOT-PIN" in text_upper:
        family = "pilot_pin"
    elif "GUIDE POST" in text_upper:
        family = "guide_post"
    elif "GUIDE BUSHING" in text_upper:
        family = "bushing"
    elif "FORM PUNCH" in text_upper or "COIN PUNCH" in text_upper:
        family = "form_punch"
    elif "DIE SECTION" in text_upper:
        family = "die_section"

    # If not matched, check for form/insert indicators (these suggest NOT a simple round punch)
    if family is None:
        if "INSERT" in text_upper or "COIN" in text_upper:
            # Check if it's punch-like or more of an insert/die component
            if "PUNCH" in text_upper:
                family = "form_punch"
            else:
                family = "die_insert"
        elif "FORM" in text_upper and ("PUNCH" in text_upper or "DETAIL" in text_upper):
            family = "form_punch"
        elif "SECTION" in text_upper:
            family = "die_section"

    # If still not matched, check for simple terms
    if family is None:
        if "BUSHING" in text_upper:
            family = "bushing"
        elif "PUNCH" in text_upper:
            # Default to round_punch only if we found "PUNCH" keyword
            family = "round_punch"
        else:
            # No clear indicator, use most generic
            family = "round_punch"

    # Determine shape
    shape = "round"  # default

    # Look for rectangular indicators
    if "RECTANGULAR" in text_upper or "SQUARE" in text_upper:
        shape = "rectangular"
    # If we see thickness AND width dimensions, likely rectangular
    elif ("THICKNESS" in text_upper or "THK" in text_upper) and ("WIDTH" in text_upper or " W " in text_upper):
        shape = "rectangular"
    # Form punches with contours are often not simple rounds
    elif family == "form_punch" and ("CONTOUR" in text_upper or "PROFILE" in text_upper):
        # Keep as round unless explicitly rectangular
        pass

    return family, shape


def detect_material(text_dump: str) -> Optional[str]:
    """
    Detect material callout from text.

    Common materials:
        - A2, A-2, A6, A10
        - D2, D-2, D3
        - M2, M-2, M4
        - O1, S7, H13
        - CARBIDE
        - 440C, 17-4

    Args:
        text_dump: Combined text from drawing

    Returns:
        Material string or None (normalized without hyphens)
    """
    text_upper = text_dump.upper()

    # Common tool steel patterns (with and without hyphens)
    materials = [
        (r'\bA-?2\b', 'A2'),
        (r'\bA-?6\b', 'A6'),
        (r'\bA-?10\b', 'A10'),
        (r'\bD-?2\b', 'D2'),
        (r'\bD-?3\b', 'D3'),
        (r'\bM-?2\b', 'M2'),
        (r'\bM-?4\b', 'M4'),
        (r'\bO-?1\b', 'O1'),
        (r'\bS-?7\b', 'S7'),
        (r'\bH-?13\b', 'H13'),
        (r'\bCARBIDE\b', 'CARBIDE'),
        (r'\b440-?C\b', '440C'),
        (r'\b17-4\b', '17-4'),
        (r'\b4140\b', '4140'),
        (r'\b4340\b', '4340'),
    ]

    for pattern, normalized in materials:
        match = re.search(pattern, text_upper)
        if match:
            return normalized

    return None


def detect_ops_features(text_dump: str) -> Dict[str, Any]:
    """
    Detect operations-driving features from text.

    Args:
        text_dump: Combined text from drawing

    Returns:
        Dict with counts and flags:
            - num_chamfers
            - num_small_radii
            - has_3d_surface
            - has_perp_face_grind
            - form_complexity_level
    """
    text_upper = text_dump.upper()

    features = {
        "num_chamfers": 0,
        "num_small_radii": 0,
        "has_3d_surface": False,
        "has_perp_face_grind": False,
        "form_complexity_level": 0,
    }

    # Chamfers with quantity: (2) .010 X 45°, (3) 0.040 X 45, etc.
    # Allow optional leading zero, spaces, degree symbol
    chamfer_qty_pattern = r'\((\d+)\)\s*(?:0)?\.?\d+\s*X\s*45'
    for match in re.finditer(chamfer_qty_pattern, text_upper):
        qty = int(match.group(1))
        features["num_chamfers"] += qty

    # Individual chamfer callouts without quantity
    # Matches: .040 X 45, 0.040X45, .010 X 45°, etc.
    single_chamfer = r'(?:0)?\.?\d+\s*X\s*45'
    single_count = len(re.findall(single_chamfer, text_upper))
    if single_count > features["num_chamfers"]:
        features["num_chamfers"] = single_count

    # Small radii: R.005, R .005, 0.005 R, .005R, etc.
    # Allow optional leading zero, optional space after R
    small_radius_patterns = [
        r'R\s*(?:0)?\.00\d+',      # R.005, R .005, R0.005
        r'(?:0)?\.00\d+\s*R',      # .005 R, 0.005R
    ]
    radius_count = 0
    for pattern in small_radius_patterns:
        radius_count += len(re.findall(pattern, text_upper))
    features["num_small_radii"] = radius_count

    # 3D surface / form
    if any(kw in text_upper for kw in ["POLISH", "FORM", "COIN", "CONTOUR"]):
        features["has_3d_surface"] = True

    # Perpendicular face grind
    if "PERPENDICULAR" in text_upper or "PERP" in text_upper:
        features["has_perp_face_grind"] = True

    # Form complexity (simple heuristic based on radius count)
    # Count all radii including larger ones
    all_radius_pattern = r'R\s*(?:0)?\.?\d+'
    total_radius_count = len(re.findall(all_radius_pattern, text_upper))
    if total_radius_count > 10:
        features["form_complexity_level"] = 3
    elif total_radius_count > 5:
        features["form_complexity_level"] = 2
    elif total_radius_count > 2:
        features["form_complexity_level"] = 1

    return features


def detect_pain_flags(text_dump: str) -> Dict[str, bool]:
    """
    Detect quality/pain flags from text.

    Args:
        text_dump: Combined text from drawing (including raw MTEXT)

    Returns:
        Dict with boolean flags:
            - has_polish_contour
            - has_no_step_permitted
            - has_sharp_edges
            - has_gdt
    """
    text_upper = text_dump.upper()

    # Polish detection (multiple variations)
    has_polish = any(kw in text_upper for kw in [
        "POLISH CONTOUR",
        "POLISH CONTOURED",
        "POLISHED",
        "POLISH TO",
        " POLISH "
    ])

    # No step permitted detection
    has_no_step = any(kw in text_upper for kw in [
        "NO STEP PERMITTED",
        "NO STEP",
        "NO STEPS",
        "NO-STEP"
    ])

    # Sharp edge detection
    has_sharp = any(kw in text_upper for kw in [
        "SHARP EDGE",
        "SHARP EDGES",
        " SHARP "
    ])

    # GD&T detection (symbols, callouts, font codes, or explicit mentions)
    # Check for \Famgdt (AutoCAD GD&T font code) in raw text
    has_gdt_font = "\\Famgdt" in text_dump or "\\FAMGDT" in text_upper

    # Check for GD&T symbols and keywords
    has_gdt_symbols = bool(re.search(
        r'[⏥⌭⏄⌯⊕⌖]|GD&T|PERPENDICULARITY|FLATNESS|POSITION|CONCENTRICITY|RUNOUT|TIR',
        text_upper
    ))

    has_gdt = has_gdt_font or has_gdt_symbols

    return {
        "has_polish_contour": has_polish,
        "has_no_step_permitted": has_no_step,
        "has_sharp_edges": has_sharp,
        "has_gdt": has_gdt,
    }


# ============================================================================
# HOLE/TAP PARSING FROM FREE TEXT
# ============================================================================


def parse_holes_from_text(text_dump: str) -> Dict[str, Any]:
    """
    Parse hole and tap specifications from free text.

    Patterns:
        - "5/16-18 TAP X .80 DEEP"
        - "Ø.250 THRU"
        - "Ø.125 X .50 DP"
        - "#7 DRILL (.201) X .75 DEEP"

    Args:
        text_dump: Combined text from drawing

    Returns:
        Dict with:
            - tap_count: int
            - tap_summary: List[Dict] with size and depth_in
            - hole_count: int
            - hole_summary: List[Dict]
    """
    taps = []
    holes = []

    # TAP pattern: 5/16-18 TAP X .80 DEEP
    tap_pattern = r'(\d+/\d+-\d+)\s+TAP\s+X\s+([\d\.]+)\s+DEEP'
    for match in re.finditer(tap_pattern, text_dump, re.IGNORECASE):
        size = match.group(1)
        depth = float(match.group(2))
        taps.append({"size": size, "depth_in": depth})

    # TAP pattern without depth: 5/16-18 TAP
    tap_pattern_no_depth = r'(\d+/\d+-\d+)\s+TAP'
    for match in re.finditer(tap_pattern_no_depth, text_dump, re.IGNORECASE):
        size = match.group(1)
        # Check if this tap was already captured with depth
        if not any(t["size"] == size for t in taps):
            taps.append({"size": size, "depth_in": None})

    # HOLE pattern: Ø.250 THRU
    hole_thru_pattern = r'Ø\s*([\d\.]+)\s+THRU'
    for match in re.finditer(hole_thru_pattern, text_dump, re.IGNORECASE):
        dia = float(match.group(1))
        holes.append({"diameter": dia, "depth_in": None, "thru": True})

    # HOLE pattern: Ø.125 X .50 DP
    hole_depth_pattern = r'Ø\s*([\d\.]+)\s+X\s+([\d\.]+)\s+(?:DP|DEEP)'
    for match in re.finditer(hole_depth_pattern, text_dump, re.IGNORECASE):
        dia = float(match.group(1))
        depth = float(match.group(2))
        holes.append({"diameter": dia, "depth_in": depth, "thru": False})

    return {
        "tap_count": len(taps),
        "tap_summary": taps,
        "hole_count": len(holes),
        "hole_summary": holes,
    }


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================


def extract_punch_features_from_dxf(
    dxf_path: Path,
    text_dump: str
) -> PunchFeatureSummary:
    """
    Main function to extract punch features from DXF + text.

    This implements the full extraction pipeline:
        1. Classification & material (text only)
        2. Size from geometry + dimensions
        3. Ops & pain flags (text + dims)
        4. Holes/taps from free text

    Args:
        dxf_path: Path to DXF file
        text_dump: Combined text content from drawing

    Returns:
        PunchFeatureSummary with all extracted features
    """
    summary = PunchFeatureSummary()
    warnings = []

    # === PASS 1: CLASSIFICATION & MATERIAL ===
    family, shape = classify_punch_family(text_dump)
    summary.family = family
    summary.shape_type = shape

    material = detect_material(text_dump)
    summary.material_callout = material

    # === PASS 2: SIZE FROM GEOMETRY + DIMENSIONS ===

    # Get geometry envelope
    geo_envelope = extract_geometry_envelope(dxf_path)
    if "error" in geo_envelope:
        warnings.append(f"Geometry extraction: {geo_envelope['error']}")

    # Get dimension data
    dim_data = extract_dimensions(dxf_path)
    if "error" in dim_data:
        warnings.append(f"Dimension extraction: {dim_data['error']}")

    # Use the larger of geometry bbox or max dimension
    geo_length = geo_envelope.get("overall_length_in", 0.0)
    geo_width = geo_envelope.get("overall_width_in", 0.0)
    dim_length = dim_data.get("max_linear_dim", 0.0)
    dim_diameter = dim_data.get("max_diameter_dim", 0.0)

    summary.overall_length_in = max(geo_length, dim_length)

    if summary.shape_type == "round":
        summary.max_od_or_width_in = max(geo_width, dim_diameter)
    else:
        summary.max_od_or_width_in = geo_width
        summary.body_width_in = geo_width
        summary.body_thickness_in = geo_envelope.get("overall_height_in", None)

    # Estimate num_ground_diams from distinct diameter dimensions
    diameter_dims = dim_data.get("diameter_dims", [])
    unique_diameters = set(round(d["measurement"], 4) for d in diameter_dims)
    summary.num_ground_diams = len(unique_diameters)

    # Sanity check: if we have dimensions but num_ground_diams is 0, use fallback
    if summary.num_ground_diams == 0 and (summary.max_od_or_width_in > 0 or dim_diameter > 0):
        # At least one ground diameter must exist if we detected any diameter
        summary.num_ground_diams = 1
        warnings.append("No distinct diameters found from dimensions; defaulting to 1 ground diameter")

    # Rough estimate of total ground length
    # For round punches, assume most of the length is ground
    # For form punches, use a smaller fraction
    if summary.family == "form_punch" or summary.has_3d_surface:
        ground_fraction = 0.3  # Form punches have less cylindrical ground area
    elif summary.num_ground_diams > 2:
        ground_fraction = 0.7  # Multiple diameters suggest most is ground
    else:
        ground_fraction = 0.5  # Default

    summary.total_ground_length_in = summary.overall_length_in * ground_fraction

    # Ensure total_ground_length_in is not zero if we have ground diameters
    if summary.num_ground_diams > 0 and summary.total_ground_length_in == 0 and summary.overall_length_in > 0:
        summary.total_ground_length_in = summary.overall_length_in * 0.5

    # Tolerances
    summary.min_dia_tol_in = dim_data.get("min_dia_tol")
    summary.min_len_tol_in = dim_data.get("min_len_tol")

    # === PASS 3: OPS & PAIN FLAGS ===

    # Combine text_dump with resolved dimension texts for comprehensive pattern detection
    # This ensures dimension text overrides like "(2) <>" resolved to "(2) .148" are detected
    resolved_dim_texts = dim_data.get("resolved_dim_texts", [])
    combined_text = text_dump
    if resolved_dim_texts:
        combined_text = text_dump + "\n" + "\n".join(resolved_dim_texts)

    # Operations features
    ops_features = detect_ops_features(combined_text)
    summary.num_chamfers = ops_features["num_chamfers"]
    summary.num_small_radii = ops_features["num_small_radii"]
    summary.has_3d_surface = ops_features["has_3d_surface"]
    summary.has_perp_face_grind = ops_features["has_perp_face_grind"]
    summary.form_complexity_level = ops_features["form_complexity_level"]

    # Pain flags
    pain_flags = detect_pain_flags(combined_text)
    summary.has_polish_contour = pain_flags["has_polish_contour"]
    summary.has_no_step_permitted = pain_flags["has_no_step_permitted"]
    summary.has_sharp_edges = pain_flags["has_sharp_edges"]
    summary.has_gdt = pain_flags["has_gdt"]

    # Holes and taps
    hole_data = parse_holes_from_text(combined_text)
    summary.tap_count = hole_data["tap_count"]
    summary.tap_summary = hole_data["tap_summary"]

    # === METADATA ===
    summary.warnings = warnings

    # Add warning if text dump is suspiciously small
    if len(text_dump) < 100:
        warnings.append(f"Text dump is very small ({len(text_dump)} chars) - may be incomplete")
        summary.warnings = warnings

    # Confidence score based on how much data we extracted
    confidence = 1.0

    # Major penalties for missing critical data
    if summary.overall_length_in == 0:
        confidence -= 0.3
        warnings.append("Overall length is 0 - dimension extraction may have failed")

    if summary.max_od_or_width_in == 0:
        confidence -= 0.3
        warnings.append("Max OD/width is 0 - dimension extraction may have failed")

    # Minor penalties for missing optional data
    if not summary.material_callout:
        confidence -= 0.1

    # Bonus for successfully extracting detailed features
    if summary.num_chamfers > 0 or summary.tap_count > 0:
        confidence += 0.05  # Found specific features

    if summary.min_dia_tol_in is not None or summary.min_len_tol_in is not None:
        confidence += 0.05  # Found tolerances

    # Ensure confidence is in valid range
    summary.confidence_score = max(0.0, min(1.0, confidence))
    summary.warnings = warnings

    return summary


# ============================================================================
# CONVENIENCE FUNCTION FOR INTEGRATION
# ============================================================================


def extract_punch_features(
    dxf_path: str | Path,
    text_lines: Optional[List[str]] = None
) -> PunchFeatureSummary:
    """
    Convenience function for extracting punch features.

    If text_lines is not provided, will attempt to extract text from DXF.

    Args:
        dxf_path: Path to DXF file (string or Path)
        text_lines: Optional list of text lines from drawing

    Returns:
        PunchFeatureSummary
    """
    dxf_path = Path(dxf_path)

    # If no text provided, try to extract it
    if text_lines is None:
        # Import here to avoid circular dependency
        try:
            from cad_quoter.geo_extractor import open_doc, collect_all_text
            doc = open_doc(dxf_path)
            text_records = list(collect_all_text(doc))
            # collect_all_text returns list of dicts, not TextRecord objects
            text_lines = [rec["text"] for rec in text_records if rec.get("text")]
        except Exception as e:
            text_lines = []

    # Combine text lines into single dump
    text_dump = "\n".join(text_lines)

    return extract_punch_features_from_dxf(dxf_path, text_dump)
