"""
MTEXT Normalization and Dimension Text Resolution
==================================================

Standalone module for handling AutoCAD MTEXT formatting codes and resolving
dimension text placeholders in DXF/DWG files.

This module provides utilities to:
1. Strip MTEXT formatting codes (\\H, \\C, \\S, etc.)
2. Convert stacked text to readable format
3. Replace <> placeholders with actual measurements
4. Convert between DXF unit systems

Usage:
    from mtext_normalizer import (
        normalize_acad_mtext,
        units_to_inch_factor,
        resolved_dimension_text,
        resolve_all_dimensions,
    )

    # Normalize MTEXT formatting
    text = normalize_acad_mtext("{\\H0.71x;\\C3;\\S+.005^ -.000;}")
    # Result: "+.005/-.000"

    # Resolve dimension text with measurement
    import ezdxf
    doc = ezdxf.readfile("drawing.dxf")
    unit_factor = units_to_inch_factor(doc.header.get("$INSUNITS", 1))

    for dim in doc.modelspace().query("DIMENSION"):
        resolved = resolved_dimension_text(dim, unit_factor)
        print(resolved)

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Optional ezdxf import
try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


# ============================================================================
# MTEXT NORMALIZATION
# ============================================================================


def normalize_acad_mtext(line: str) -> str:
    """
    Normalize AutoCAD MTEXT formatting codes into simpler plain text.

    Handles the following MTEXT formatting codes:
    - Strip outer {...} braces
    - Remove \\Hxx; (text height scaling)
    - Remove \\Cxx; (color codes)
    - Convert stacked text \\S+.005^ -.000; -> '+.005/-.000'
    - Remove empty braces {}
    - Remove \\fArial|... font specifications

    Args:
        line: Raw MTEXT string with formatting codes

    Returns:
        Normalized plain text string

    Examples:
        >>> normalize_acad_mtext("{\\H0.71x;\\C3;\\S+.005^ -.000;}")
        '+.005/-.000'

        >>> normalize_acad_mtext("{\\H1.0x;POLISH CONTOUR}")
        'POLISH CONTOUR'

        >>> normalize_acad_mtext("{\\S1/4^ -20;} TAP")
        '1/4/-20 TAP'

        >>> normalize_acad_mtext("(2) <>")
        '(2) <>'

        >>> normalize_acad_mtext("")
        ''
    """
    if not line:
        return ""

    # Strip outer braces if present
    if line.startswith("{") and line.endswith("}"):
        line = line[1:-1]

    # Remove height scaling codes: \H0.71x; \H1.0x; etc.
    line = re.sub(r"\\H[0-9.]+x;", "", line)

    # Remove color codes: \C3; \C256; etc.
    line = re.sub(r"\\C\d+;", "", line)

    # Remove font specifications: \fArial|b0|i0|c0|p34;
    line = re.sub(r"\\f[^;]+;", "", line)

    # Remove alignment codes: \A1; \A0; etc.
    line = re.sub(r"\\A\d+;", "", line)

    # Remove paragraph/line spacing codes: \P
    line = re.sub(r"\\P", "\n", line)

    # Convert stacked text: \S+.005^ -.000; -> +.005/-.000
    # Pattern: \S<top>^<bottom>;
    def repl_stack(m):
        top = m.group(1).strip()
        bot = m.group(2).strip()
        return f"{top}/{bot}"

    line = re.sub(r"\\S([^\\^]+)\^([^;]+);", repl_stack, line)

    # Remove any remaining braces that were around stacked text
    line = re.sub(r"\{([^{}]+)\}", r"\1", line)

    # Remove empty braces
    line = line.replace("{}", "")

    # Remove escape sequences for special characters
    line = line.replace("\\~", " ")  # Non-breaking space
    line = line.replace("\\\\", "\\")  # Escaped backslash

    # Clean up multiple spaces
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def strip_mtext_codes(text: str) -> str:
    """
    Alias for normalize_acad_mtext() for backward compatibility.

    Args:
        text: Raw MTEXT string

    Returns:
        Normalized plain text
    """
    return normalize_acad_mtext(text)


# ============================================================================
# UNIT CONVERSION
# ============================================================================


def units_to_inch_factor(insunits: int) -> float:
    """
    Convert DXF $INSUNITS code to inch conversion factor.

    The $INSUNITS header variable specifies the drawing units:
    - 0 = Unitless (assume inches)
    - 1 = Inches
    - 2 = Feet
    - 3 = Miles
    - 4 = Millimeters
    - 5 = Centimeters
    - 6 = Meters
    - 7 = Kilometers
    - 8 = Microinches
    - 9 = Mils
    - 10 = Yards
    - 11 = Angstroms
    - 12 = Nanometers
    - 13 = Microns
    - 14 = Decimeters
    - 15 = Decameters
    - 16 = Hectometers
    - 17 = Gigameters
    - 18 = Astronomical units
    - 19 = Light years
    - 20 = Parsecs

    Args:
        insunits: DXF $INSUNITS value

    Returns:
        Multiplication factor to convert to inches

    Examples:
        >>> units_to_inch_factor(1)   # Inches
        1.0
        >>> units_to_inch_factor(4)   # Millimeters
        0.03937007874015748
        >>> units_to_inch_factor(2)   # Feet
        12.0
    """
    units_factors = {
        0: 1.0,              # Unitless - assume inches
        1: 1.0,              # Inches
        2: 12.0,             # Feet -> inches
        3: 63360.0,          # Miles -> inches
        4: 1.0 / 25.4,       # Millimeters -> inches
        5: 1.0 / 2.54,       # Centimeters -> inches
        6: 39.3701,          # Meters -> inches
        7: 39370.1,          # Kilometers -> inches
        8: 1.0e-6,           # Microinches -> inches
        9: 0.001,            # Mils -> inches
        10: 36.0,            # Yards -> inches
        13: 1.0 / 25400.0,   # Microns -> inches
        14: 3.93701,         # Decimeters -> inches
    }
    return units_factors.get(insunits, 1.0)


def get_unit_name(insunits: int) -> str:
    """
    Get human-readable name for DXF $INSUNITS value.

    Args:
        insunits: DXF $INSUNITS value

    Returns:
        Unit name string

    Examples:
        >>> get_unit_name(1)
        'inches'
        >>> get_unit_name(4)
        'millimeters'
    """
    unit_names = {
        0: "unitless",
        1: "inches",
        2: "feet",
        3: "miles",
        4: "millimeters",
        5: "centimeters",
        6: "meters",
        7: "kilometers",
        8: "microinches",
        9: "mils",
        10: "yards",
        13: "microns",
        14: "decimeters",
    }
    return unit_names.get(insunits, "unknown")


# ============================================================================
# DIMENSION TEXT RESOLUTION
# ============================================================================


def resolved_dimension_text(dim, unit_factor: float) -> str:
    """
    Resolve dimension text by replacing <> placeholder and normalizing MTEXT.

    This function takes an ezdxf DIMENSION entity and:
    1. Gets the raw text (which may contain <> placeholder and MTEXT codes)
    2. Gets the numeric measurement value
    3. Converts measurement to inches using unit_factor
    4. Normalizes MTEXT formatting codes
    5. Replaces <> with the formatted numeric value

    Args:
        dim: ezdxf DIMENSION entity
        unit_factor: Conversion factor from drawing units to inches

    Returns:
        Resolved and normalized dimension text string

    Examples:
        # For a dimension with text "(2) <>" and measurement 3.76mm:
        >>> resolved_dimension_text(dim, 1/25.4)
        '(2) .148'

        # For a dimension with tolerance MTEXT:
        # Raw: "<> {\\H0.71x;\\S+.0000^ -.0002;}"
        # Measurement: 12.69mm
        >>> resolved_dimension_text(dim, 1/25.4)
        '.4996 +.0000/-.0002'
    """
    # Get raw dimension text
    raw_text = ""
    if hasattr(dim, 'dxf') and hasattr(dim.dxf, 'text'):
        raw_text = dim.dxf.text or ""

    # Get numeric measurement
    meas = dim.get_measurement()
    if meas is None:
        meas = 0.0

    # Handle Vec3 objects (ezdxf may return Vec3 for some dimensions)
    if hasattr(meas, 'magnitude'):
        meas = meas.magnitude
    elif hasattr(meas, 'x'):
        # For Vec3, use the magnitude or absolute value of x
        meas = abs(meas.x)

    meas = float(meas)

    # Convert to inches
    value_in = meas * unit_factor

    # Format the numeric value
    # Use 4 decimal places, strip trailing zeros
    nominal_str = f"{value_in:.4f}".rstrip("0").rstrip(".")

    # For values < 1.0, use ".XXX" format instead of "0.XXX"
    if nominal_str.startswith("0.") and value_in < 1.0:
        nominal_str = nominal_str[1:]  # ".148" instead of "0.148"
    elif not nominal_str or nominal_str == ".":
        nominal_str = "0"

    # Normalize MTEXT formatting codes
    text = normalize_acad_mtext(raw_text) if raw_text else ""

    # Replace <> placeholder with numeric value
    if "<>" in text and nominal_str:
        text = text.replace("<>", nominal_str)
    elif not text and nominal_str:
        # No override text at all; just use the numeric string
        text = nominal_str

    return text.strip()


def resolve_dimension_with_info(dim, unit_factor: float) -> Dict[str, Any]:
    """
    Resolve dimension text and return detailed information.

    Args:
        dim: ezdxf DIMENSION entity
        unit_factor: Conversion factor to inches

    Returns:
        Dict with:
            - resolved_text: Normalized and resolved text
            - raw_text: Original text from entity
            - measurement: Numeric value in drawing units
            - measurement_in: Numeric value in inches
            - dimtype: Dimension type code
            - is_diameter: Whether this is a diameter dimension
    """
    # Get raw text
    raw_text = ""
    if hasattr(dim, 'dxf') and hasattr(dim.dxf, 'text'):
        raw_text = dim.dxf.text or ""

    # Get measurement
    meas = dim.get_measurement()
    if meas is None:
        meas = 0.0

    # Handle Vec3
    if hasattr(meas, 'magnitude'):
        meas = meas.magnitude
    elif hasattr(meas, 'x'):
        meas = abs(meas.x)
    meas = float(meas)

    # Get dimtype
    dimtype = dim.dimtype if hasattr(dim, 'dimtype') else 0

    # Check if diameter
    is_diameter = (
        dimtype == 3 or
        "%%c" in raw_text.lower() or
        "Ø" in raw_text or
        "DIA" in raw_text.upper()
    )

    # Get ordinate direction for ordinate dimensions (dimtype 6)
    # In DXF, ordinate_type flag: 0 = X-type, 1 = Y-type
    ordinate_direction = None
    if dimtype == 6:
        try:
            # Check flag in dimension type value (bit 6 = Y-type if set)
            # dimtype value for ordinate is 6, with flag 64 (0x40) for Y-type
            full_dimtype = dim.dxf.dimtype if hasattr(dim.dxf, 'dimtype') else 0
            if full_dimtype & 64:  # Bit 6 set = Y-type ordinate
                ordinate_direction = "Y"
            else:
                ordinate_direction = "X"
        except:
            pass

    # Resolve text
    resolved = resolved_dimension_text(dim, unit_factor)

    result = {
        "resolved_text": resolved,
        "raw_text": raw_text,
        "measurement": meas,
        "measurement_in": meas * unit_factor,
        "dimtype": dimtype,
        "is_diameter": is_diameter,
    }

    if ordinate_direction:
        result["ordinate_direction"] = ordinate_direction

    return result


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def resolve_all_dimensions(dxf_path: Path) -> List[Dict[str, Any]]:
    """
    Resolve all DIMENSION entities in a DXF file.

    Args:
        dxf_path: Path to DXF file

    Returns:
        List of dicts with resolved dimension information

    Raises:
        ImportError: If ezdxf is not available
        FileNotFoundError: If DXF file doesn't exist

    Example:
        >>> results = resolve_all_dimensions(Path("drawing.dxf"))
        >>> for dim in results:
        ...     print(f"{dim['resolved_text']} ({dim['measurement_in']:.4f}\")")
    """
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf is required for resolve_all_dimensions()")

    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    # Read DXF file
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Get unit conversion factor
    insunits = doc.header.get("$INSUNITS", 1)
    measurement = doc.header.get("$MEASUREMENT", 0)
    unit_factor = units_to_inch_factor(insunits)

    # Override if $MEASUREMENT indicates metric
    if measurement == 1 and insunits not in [4, 5, 6]:
        unit_factor = 1.0 / 25.4

    # Process all dimensions
    results = []
    for dim in msp.query("DIMENSION"):
        try:
            info = resolve_dimension_with_info(dim, unit_factor)
            results.append(info)
        except Exception as e:
            results.append({
                "resolved_text": "",
                "raw_text": "",
                "measurement": 0.0,
                "measurement_in": 0.0,
                "dimtype": 0,
                "is_diameter": False,
                "error": str(e),
            })

    return results


def get_dimension_texts(dxf_path: Path) -> List[str]:
    """
    Get list of resolved dimension texts from a DXF file.

    Convenience function that returns just the text strings.

    Args:
        dxf_path: Path to DXF file

    Returns:
        List of resolved dimension text strings

    Example:
        >>> texts = get_dimension_texts(Path("drawing.dxf"))
        >>> for text in texts:
        ...     print(text)
    """
    results = resolve_all_dimensions(dxf_path)
    return [r["resolved_text"] for r in results if r.get("resolved_text")]


# ============================================================================
# TOLERANCE EXTRACTION
# ============================================================================


def extract_tolerance_from_text(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract tolerance values from dimension text.

    Looks for patterns like:
    - +.0000/-.0002
    - ±0.001
    - +0.005 -0.000

    Args:
        text: Resolved dimension text

    Returns:
        Tuple of (plus_tolerance, minus_tolerance) or None if not found

    Examples:
        >>> extract_tolerance_from_text(".4997 +.0000/-.0002")
        (0.0, 0.0002)

        >>> extract_tolerance_from_text("1.234 ±0.001")
        (0.001, 0.001)

        >>> extract_tolerance_from_text("6.990")
        None
    """
    # Pattern: ±0.000X
    pm_match = re.search(r'±\s*(\d*\.?\d+)', text)
    if pm_match:
        tol = float(pm_match.group(1))
        return (tol, tol)

    # Pattern: +.0000/-.0002
    slash_match = re.search(r'\+\s*(\d*\.?\d+)\s*/\s*-\s*(\d*\.?\d+)', text)
    if slash_match:
        plus_tol = float(slash_match.group(1))
        minus_tol = float(slash_match.group(2))
        return (plus_tol, minus_tol)

    # Pattern: +0.000X -0.000Y
    pm2_match = re.search(r'\+\s*(\d*\.?\d+)\s+-\s*(\d*\.?\d+)', text)
    if pm2_match:
        plus_tol = float(pm2_match.group(1))
        minus_tol = float(pm2_match.group(2))
        return (plus_tol, minus_tol)

    return None


def get_tolerance_band(text: str) -> Optional[float]:
    """
    Get the total tolerance band (max of plus and minus) from dimension text.

    Args:
        text: Resolved dimension text

    Returns:
        Maximum tolerance value or None

    Examples:
        >>> get_tolerance_band(".4997 +.0000/-.0002")
        0.0002

        >>> get_tolerance_band("1.234 ±0.001")
        0.001
    """
    result = extract_tolerance_from_text(text)
    if result:
        return max(result[0], result[1])
    return None


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================


def main():
    """Command-line interface for testing MTEXT normalization."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mtext_normalizer.py <dxf_file>")
        print("       python mtext_normalizer.py --test")
        sys.exit(1)

    if sys.argv[1] == "--test":
        # Run tests
        print("Testing MTEXT normalization...")

        test_cases = [
            ("{\\H0.71x;\\C3;\\S+.005^ -.000;}", "+.005/-.000"),
            ("{\\H1.0x;POLISH CONTOUR}", "POLISH CONTOUR"),
            ("(2) <>", "(2) <>"),
            ("{\\S1/4^ -20;} TAP", "1/4/-20 TAP"),
            ("", ""),
        ]

        passed = 0
        for input_text, expected in test_cases:
            result = normalize_acad_mtext(input_text)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            print(f"  {status} Input: {input_text!r}")
            print(f"    Expected: {expected!r}")
            print(f"    Got:      {result!r}")
            print()

        print(f"Passed {passed}/{len(test_cases)} tests")

        # Test unit conversion
        print("\nTesting unit conversion...")
        print(f"  Inches (1): {units_to_inch_factor(1)}")
        print(f"  Millimeters (4): {units_to_inch_factor(4)}")
        print(f"  Feet (2): {units_to_inch_factor(2)}")

        sys.exit(0 if passed == len(test_cases) else 1)

    else:
        # Process DXF file
        dxf_path = Path(sys.argv[1])

        if not dxf_path.exists():
            print(f"Error: File not found: {dxf_path}")
            sys.exit(1)

        if not EZDXF_AVAILABLE:
            print("Error: ezdxf is required. Install with: pip install ezdxf")
            sys.exit(1)

        print(f"Processing: {dxf_path}")
        print()

        results = resolve_all_dimensions(dxf_path)

        print(f"Found {len(results)} DIMENSION entities:\n")

        for i, dim in enumerate(results, 1):
            if dim.get("error"):
                print(f"{i}. ERROR: {dim['error']}")
            else:
                resolved = dim["resolved_text"]
                raw = dim["raw_text"]
                meas_in = dim["measurement_in"]
                is_dia = "Ø" if dim["is_diameter"] else ""

                print(f"{i}. {is_dia}{resolved}")
                if raw and raw != resolved:
                    print(f"   Raw: {raw!r}")
                print(f"   Measurement: {meas_in:.4f}\"")

                # Show tolerance if present
                tol = get_tolerance_band(resolved)
                if tol:
                    print(f"   Tolerance: ±{tol:.4f}\"")
                print()


if __name__ == "__main__":
    main()
