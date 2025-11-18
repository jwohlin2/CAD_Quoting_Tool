"""
backup_hole_finder.py
=====================
Backup hole finder that validates hole extraction against the CAD feature extractor.

This module provides a secondary method for extracting and validating hole features
from DWG/DXF files. It compares results against expected hole descriptions to ensure
accuracy in hole detection.

Usage:
    python -m tools.backup_hole_finder path/to/file.dwg
    python -m tools.backup_hole_finder --part 108

Expected hole format uses AutoCAD MTEXT notation:
    - (2) <> THRU\\X(JIG GRIND)           # 2 holes, through, with jig grind
    - \\A1;(3) ∅7/32 THRU; ∅11/32 C'BORE\\PX .100 DEEP FROM FRONT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import ezdxf
    from ezdxf.entities import DXFEntity
except ImportError:
    ezdxf = None  # type: ignore

# Import from existing modules
try:
    from cad_quoter.geometry.dwg_punch_extractor import normalize_acad_mtext, units_to_inch_factor
    from cad_quoter.geometry.hole_table_parser import (
        parse_drill_token,
        INCH_TO_MM,
        HoleRow,
        _parse_description,
    )
except ImportError:
    # Fallback definitions if imports fail
    INCH_TO_MM = 25.4

    def normalize_acad_mtext(line: str) -> str:
        if not line:
            return ""
        if line.startswith("{") and line.endswith("}"):
            line = line[1:-1]
        line = re.sub(r"\\H[0-9.]+x;", "", line)
        line = re.sub(r"\\C\d+;", "", line)
        line = re.sub(r"\\S([^\\^]+)\^([^;]+);", lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", line)
        line = line.replace("{}", "").strip()
        return line

    def parse_drill_token(tok: str) -> Optional[float]:
        return None


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BackupHoleFeature:
    """Represents a hole feature extracted by the backup finder."""

    qty: int = 1
    diameter: Optional[str] = None  # Raw diameter string (e.g., "7/32", ".375")
    diameter_mm: Optional[float] = None
    is_thru: bool = False
    depth_in: Optional[float] = None
    operations: List[str] = field(default_factory=list)  # JIG GRIND, C'BORE, etc.
    from_face: Optional[str] = None  # FRONT, BACK, or None
    raw_text: str = ""
    cbore_diameter: Optional[str] = None
    cbore_depth_in: Optional[float] = None


@dataclass
class HoleValidationResult:
    """Result of comparing extracted holes against expected."""

    part_number: str
    file_path: str
    extracted_holes: List[BackupHoleFeature]
    expected_holes: List[BackupHoleFeature]
    matches: bool
    discrepancies: List[str] = field(default_factory=list)


# =============================================================================
# EXPECTED HOLE DATA
# =============================================================================

# Expected holes for known parts
# Format: part_number -> list of (raw_mtext_description, parsed_features)
EXPECTED_HOLES: Dict[str, List[str]] = {
    "108": [
        r"(2) <> THRU\X(JIG GRIND)",
        r"\A1;(3) ∅7/32 THRU; ∅11/32 C'BORE\PX .100 DEEP FROM FRONT",
    ],
}


# =============================================================================
# MTEXT PARSING
# =============================================================================

def parse_mtext_hole_description(mtext: str) -> BackupHoleFeature:
    """
    Parse an AutoCAD MTEXT hole description into a BackupHoleFeature.

    Examples:
        "(2) <> THRU\\X(JIG GRIND)" -> qty=2, thru=True, ops=[JIG GRIND]
        "\\A1;(3) ∅7/32 THRU; ∅11/32 C'BORE\\PX .100 DEEP FROM FRONT"
            -> qty=3, dia=7/32, thru=True, cbore_dia=11/32, cbore_depth=.100

    Args:
        mtext: Raw MTEXT string with formatting codes

    Returns:
        Parsed BackupHoleFeature
    """
    feature = BackupHoleFeature(raw_text=mtext)

    # Normalize the text for easier parsing
    text = mtext

    # Remove alignment codes like \A1;
    text = re.sub(r"\\A\d+;", "", text)

    # Replace \X with space (line break)
    text = text.replace(r"\X", " ")

    # Replace \P with space (paragraph)
    text = re.sub(r"\\P[A-Z]?", " ", text)

    # Normalize other MTEXT codes
    text = normalize_acad_mtext(text) if callable(normalize_acad_mtext) else text

    # Extract quantity: (2), (3), etc.
    qty_match = re.search(r"\((\d+)\)", text)
    if qty_match:
        feature.qty = int(qty_match.group(1))

    # Check for THRU
    if re.search(r"\bTHRU\b", text, re.IGNORECASE):
        feature.is_thru = True

    # Extract main diameter
    # Patterns: ∅7/32, Ø.375, <> (placeholder)
    dia_patterns = [
        r"[∅Ø]\s*(\d+\s*/\s*\d+)",  # Fraction: ∅7/32
        r"[∅Ø]\s*(\.\d+)",          # Decimal: Ø.375
        r"[∅Ø]\s*(\d+\.\d+)",       # Decimal with leading: Ø0.375
    ]

    for pattern in dia_patterns:
        dia_match = re.search(pattern, text)
        if dia_match:
            feature.diameter = dia_match.group(1).replace(" ", "")
            # Convert to mm
            if "/" in feature.diameter:
                try:
                    frac = Fraction(feature.diameter)
                    feature.diameter_mm = float(frac) * INCH_TO_MM
                except (ValueError, ZeroDivisionError):
                    pass
            else:
                try:
                    feature.diameter_mm = float(feature.diameter) * INCH_TO_MM
                except ValueError:
                    pass
            break

    # Check for <> placeholder (diameter from dimension)
    if "<>" in text and not feature.diameter:
        feature.diameter = "<>"

    # Extract JIG GRIND
    if re.search(r"\bJIG\s*GRIND\b", text, re.IGNORECASE):
        feature.operations.append("JIG GRIND")

    # Extract C'BORE
    cbore_match = re.search(
        r"[∅Ø]\s*(\d+\s*/\s*\d+|\.\d+|\d+\.\d+)\s*C['']?BORE",
        text,
        re.IGNORECASE
    )
    if cbore_match:
        feature.cbore_diameter = cbore_match.group(1).replace(" ", "")
        feature.operations.append("C'BORE")

    # Extract C'BORE depth
    cbore_depth_match = re.search(
        r"C['']?BORE.*?(\.\d+|\d+\.\d+)\s*DEEP",
        text,
        re.IGNORECASE
    )
    if cbore_depth_match:
        try:
            feature.cbore_depth_in = float(cbore_depth_match.group(1))
        except ValueError:
            pass

    # Also check for depth pattern without C'BORE prefix
    if not feature.cbore_depth_in:
        depth_match = re.search(r"(\.\d+|\d+\.\d+)\s*DEEP", text, re.IGNORECASE)
        if depth_match and "C'BORE" in text.upper():
            try:
                feature.cbore_depth_in = float(depth_match.group(1))
            except ValueError:
                pass

    # Extract FROM FRONT/BACK
    face_match = re.search(r"FROM\s+(FRONT|BACK)", text, re.IGNORECASE)
    if face_match:
        feature.from_face = face_match.group(1).upper()

    return feature


# =============================================================================
# DWG/DXF HOLE EXTRACTION
# =============================================================================

def extract_holes_from_text_records(text_records: List[Dict[str, Any]]) -> List[BackupHoleFeature]:
    """
    Extract hole features from geo_extractor text records.

    This is designed to work with text records from geo_dump/geo_extractor
    when no formal HOLE TABLE is found. It looks for dimension annotations
    that describe holes.

    Args:
        text_records: List of text record dicts with 'text' and 'etype' keys

    Returns:
        List of extracted BackupHoleFeature objects
    """
    holes: List[BackupHoleFeature] = []

    for record in text_records:
        text = record.get("text", "")
        if not text:
            continue

        # Check if this looks like a hole description
        if _is_hole_description(text):
            feature = parse_mtext_hole_description(text)
            if feature.diameter or feature.is_thru or feature.operations:
                holes.append(feature)

    return holes


def convert_to_hole_operations(holes: List[BackupHoleFeature]) -> List[List[str]]:
    """
    Convert BackupHoleFeature list to hole operations format.

    This matches the format expected by explode_rows_to_operations output:
    [HOLE_LETTER, REF_DIAM, QTY, OPERATION]

    Args:
        holes: List of BackupHoleFeature objects

    Returns:
        List of [hole_letter, ref_diam, qty, operation] lists
    """
    operations: List[List[str]] = []
    hole_letter = ord('A')

    for hole in holes:
        letter = chr(hole_letter)
        hole_letter += 1

        # Format diameter
        if hole.diameter and hole.diameter != "<>":
            ref_diam = f"Ø{hole.diameter}"
        elif hole.diameter_mm:
            inches = hole.diameter_mm / INCH_TO_MM
            ref_diam = f"Ø{inches:.4f}".rstrip("0").rstrip(".")
        else:
            ref_diam = "Ø?"

        qty = str(hole.qty)

        # Build operation description
        op_parts = []

        if hole.is_thru:
            op_parts.append("THRU")

        for op in hole.operations:
            if op == "JIG GRIND":
                op_parts.append("(JIG GRIND)")
            elif op == "C'BORE" and hole.cbore_diameter:
                cbore_desc = f"Ø{hole.cbore_diameter} C'BORE"
                if hole.cbore_depth_in:
                    cbore_desc += f" X {hole.cbore_depth_in} DEEP"
                if hole.from_face:
                    cbore_desc += f" FROM {hole.from_face}"
                op_parts.append(cbore_desc)

        operation = " ".join(op_parts) if op_parts else hole.raw_text

        operations.append([letter, ref_diam, qty, operation])

    return operations


def extract_holes_from_dwg(filepath: str) -> List[BackupHoleFeature]:
    """
    Extract hole features from a DWG/DXF file using ezdxf.

    This provides a backup method independent of the main feature extractor.

    Args:
        filepath: Path to the DWG/DXF file

    Returns:
        List of extracted BackupHoleFeature objects
    """
    if ezdxf is None:
        raise ImportError("ezdxf is required for DWG/DXF extraction")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    holes: List[BackupHoleFeature] = []

    try:
        doc = ezdxf.readfile(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read DWG/DXF file: {e}")

    # Get unit conversion factor
    try:
        insunits = doc.header.get("$INSUNITS", 0)
        unit_factor = units_to_inch_factor(insunits) if callable(units_to_inch_factor) else 1.0
    except Exception:
        unit_factor = 1.0

    # Collect all text entities that might contain hole information
    hole_texts: List[str] = []

    for entity in doc.modelspace():
        text = _extract_entity_text(entity, unit_factor)
        if text:
            hole_texts.append(text)

    # Parse hole descriptions from collected text
    for text in hole_texts:
        # Check if this looks like a hole description
        if _is_hole_description(text):
            feature = parse_mtext_hole_description(text)
            if feature.diameter or feature.is_thru or feature.operations:
                holes.append(feature)

    return holes


def _extract_entity_text(entity: DXFEntity, unit_factor: float) -> str:
    """Extract text content from a DXF entity."""

    dxftype = entity.dxftype()

    if dxftype == "TEXT":
        return getattr(entity.dxf, "text", "")

    if dxftype == "MTEXT":
        return getattr(entity.dxf, "text", "")

    if dxftype == "DIMENSION":
        raw_text = getattr(entity.dxf, "text", "")

        # Handle <> placeholder
        if "<>" in raw_text:
            try:
                meas = entity.get_measurement()
                if meas is not None:
                    if hasattr(meas, "magnitude"):
                        meas = meas.magnitude
                    meas_in = float(meas) * unit_factor
                    meas_str = f"{meas_in:.4f}".rstrip("0").rstrip(".")
                    if meas_str.startswith("0."):
                        meas_str = meas_str[1:]
                    raw_text = raw_text.replace("<>", meas_str)
            except Exception:
                pass

        return raw_text

    return ""


def _is_hole_description(text: str) -> bool:
    """Check if text appears to be a hole description."""

    upper = text.upper()

    # Must have a quantity marker like (2), (3), etc.
    has_qty = bool(re.search(r"\(\d+\)", text))
    if not has_qty:
        return False

    # Must have a diameter indicator (Ø, ∅) or placeholder (<>)
    has_diameter = bool(re.search(r"[∅Ø]", text) or "<>" in text)

    # Must have a hole operation keyword
    hole_op_keywords = [
        "THRU",
        "JIG GRIND",
        "C'BORE",
        "CBORE",
        "C'DRILL",
        "TAP",
    ]
    has_op_keyword = any(kw in upper for kw in hole_op_keywords)

    # Require both diameter/placeholder AND operation keyword
    return has_diameter and has_op_keyword


# =============================================================================
# VALIDATION
# =============================================================================

def validate_holes(
    part_number: str,
    filepath: str,
    extracted: Optional[List[BackupHoleFeature]] = None,
) -> HoleValidationResult:
    """
    Validate extracted holes against expected values for a part.

    Args:
        part_number: Part identifier (e.g., "108")
        filepath: Path to the DWG/DXF file
        extracted: Pre-extracted holes (if None, will extract from file)

    Returns:
        HoleValidationResult with comparison details
    """
    # Get expected holes for this part
    expected_mtext = EXPECTED_HOLES.get(part_number, [])
    expected: List[BackupHoleFeature] = [
        parse_mtext_hole_description(mt) for mt in expected_mtext
    ]

    # Extract holes from file if not provided
    if extracted is None:
        try:
            extracted = extract_holes_from_dwg(filepath)
        except Exception as e:
            return HoleValidationResult(
                part_number=part_number,
                file_path=filepath,
                extracted_holes=[],
                expected_holes=expected,
                matches=False,
                discrepancies=[f"Extraction failed: {e}"],
            )

    # Compare extracted vs expected
    discrepancies: List[str] = []

    # Group by quantity for comparison
    extracted_by_qty = _group_holes_by_qty(extracted)
    expected_by_qty = _group_holes_by_qty(expected)

    # Check for matching quantities
    for qty, exp_holes in expected_by_qty.items():
        ext_holes = extracted_by_qty.get(qty, [])

        if len(ext_holes) != len(exp_holes):
            discrepancies.append(
                f"Qty {qty}: expected {len(exp_holes)} hole group(s), "
                f"found {len(ext_holes)}"
            )
            continue

        # Compare individual holes
        for i, (exp, ext) in enumerate(zip(exp_holes, ext_holes)):
            hole_diffs = _compare_holes(exp, ext)
            if hole_diffs:
                discrepancies.append(f"Qty {qty} hole {i+1}: {'; '.join(hole_diffs)}")

    # Check for unexpected holes
    for qty in extracted_by_qty:
        if qty not in expected_by_qty:
            discrepancies.append(f"Unexpected hole group with qty={qty}")

    matches = len(discrepancies) == 0

    return HoleValidationResult(
        part_number=part_number,
        file_path=filepath,
        extracted_holes=extracted,
        expected_holes=expected,
        matches=matches,
        discrepancies=discrepancies,
    )


def _group_holes_by_qty(holes: List[BackupHoleFeature]) -> Dict[int, List[BackupHoleFeature]]:
    """Group holes by their quantity."""
    groups: Dict[int, List[BackupHoleFeature]] = {}
    for hole in holes:
        if hole.qty not in groups:
            groups[hole.qty] = []
        groups[hole.qty].append(hole)
    return groups


def _compare_holes(expected: BackupHoleFeature, extracted: BackupHoleFeature) -> List[str]:
    """Compare two hole features and return list of differences."""

    diffs: List[str] = []

    # Compare THRU
    if expected.is_thru != extracted.is_thru:
        diffs.append(f"THRU: expected={expected.is_thru}, got={extracted.is_thru}")

    # Compare operations
    exp_ops = set(expected.operations)
    ext_ops = set(extracted.operations)

    missing_ops = exp_ops - ext_ops
    extra_ops = ext_ops - exp_ops

    if missing_ops:
        diffs.append(f"Missing operations: {', '.join(missing_ops)}")
    if extra_ops:
        diffs.append(f"Extra operations: {', '.join(extra_ops)}")

    # Compare diameter (if specified)
    if expected.diameter and expected.diameter != "<>":
        if not extracted.diameter:
            diffs.append(f"Missing diameter: expected {expected.diameter}")
        elif expected.diameter_mm and extracted.diameter_mm:
            # Compare with tolerance
            if abs(expected.diameter_mm - extracted.diameter_mm) > 0.1:  # 0.1mm tolerance
                diffs.append(
                    f"Diameter mismatch: expected {expected.diameter}, "
                    f"got {extracted.diameter}"
                )

    # Compare C'BORE
    if expected.cbore_diameter:
        if not extracted.cbore_diameter:
            diffs.append(f"Missing C'BORE diameter: expected {expected.cbore_diameter}")

    if expected.cbore_depth_in:
        if not extracted.cbore_depth_in:
            diffs.append(f"Missing C'BORE depth: expected {expected.cbore_depth_in}")
        elif abs(expected.cbore_depth_in - extracted.cbore_depth_in) > 0.001:
            diffs.append(
                f"C'BORE depth mismatch: expected {expected.cbore_depth_in}, "
                f"got {extracted.cbore_depth_in}"
            )

    # Compare FROM face
    if expected.from_face and expected.from_face != extracted.from_face:
        diffs.append(f"Face mismatch: expected {expected.from_face}, got {extracted.from_face}")

    return diffs


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def find_holes_from_text(text_descriptions: List[str]) -> List[BackupHoleFeature]:
    """
    Parse holes from a list of MTEXT descriptions.

    Args:
        text_descriptions: List of raw MTEXT hole descriptions

    Returns:
        List of parsed BackupHoleFeature objects
    """
    return [parse_mtext_hole_description(desc) for desc in text_descriptions]


def validate_from_text(
    part_number: str,
    text_descriptions: List[str],
) -> HoleValidationResult:
    """
    Validate hole descriptions against expected values.

    Args:
        part_number: Part identifier
        text_descriptions: List of raw MTEXT hole descriptions

    Returns:
        HoleValidationResult
    """
    extracted = find_holes_from_text(text_descriptions)
    return validate_holes(part_number, "<text_input>", extracted)


def find_holes_backup(
    filepath: Optional[str] = None,
    part_number: Optional[str] = None,
    text_input: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for backup hole finding.

    Args:
        filepath: Path to DWG/DXF file (optional if text_input provided)
        part_number: Optional part number for validation
        text_input: Optional list of raw MTEXT descriptions

    Returns:
        Dictionary with extraction results and validation (if applicable)
    """
    result: Dict[str, Any] = {
        "file": filepath or "<text_input>",
        "holes": [],
        "validation": None,
    }

    # Extract holes from text or file
    try:
        if text_input:
            holes = find_holes_from_text(text_input)
        elif filepath:
            holes = extract_holes_from_dwg(filepath)
        else:
            raise ValueError("Either filepath or text_input must be provided")

        result["holes"] = [
            {
                "qty": h.qty,
                "diameter": h.diameter,
                "diameter_mm": h.diameter_mm,
                "is_thru": h.is_thru,
                "depth_in": h.depth_in,
                "operations": h.operations,
                "from_face": h.from_face,
                "cbore_diameter": h.cbore_diameter,
                "cbore_depth_in": h.cbore_depth_in,
                "raw_text": h.raw_text,
            }
            for h in holes
        ]
    except Exception as e:
        result["error"] = str(e)
        return result

    # Validate against expected if part number provided
    if part_number and part_number in EXPECTED_HOLES:
        validation = validate_holes(part_number, filepath or "<text_input>", holes)
        result["validation"] = {
            "matches": validation.matches,
            "expected_count": len(validation.expected_holes),
            "extracted_count": len(validation.extracted_holes),
            "discrepancies": validation.discrepancies,
        }

    return result


def print_expected_holes(part_number: str) -> None:
    """Print the expected holes for a part in a readable format."""

    if part_number not in EXPECTED_HOLES:
        print(f"No expected holes defined for part {part_number}")
        return

    print(f"\nExpected holes for part {part_number}:")
    print("=" * 60)

    for i, mtext in enumerate(EXPECTED_HOLES[part_number], 1):
        print(f"\nHole group {i}:")
        print(f"  Raw MTEXT: {mtext}")

        feature = parse_mtext_hole_description(mtext)
        print(f"  Parsed:")
        print(f"    Quantity: {feature.qty}")
        print(f"    Diameter: {feature.diameter or 'N/A'}")
        print(f"    Through: {feature.is_thru}")
        print(f"    Operations: {', '.join(feature.operations) or 'None'}")
        if feature.cbore_diameter:
            print(f"    C'BORE dia: {feature.cbore_diameter}")
        if feature.cbore_depth_in:
            print(f"    C'BORE depth: {feature.cbore_depth_in}")
        if feature.from_face:
            print(f"    From face: {feature.from_face}")


def main():
    """Command-line interface for backup hole finder."""

    parser = argparse.ArgumentParser(
        description="Backup hole finder with CAD feature extractor validation"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Path to DWG/DXF file"
    )
    parser.add_argument(
        "--part",
        help="Part number for validation (e.g., 108)"
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Show expected holes for the specified part"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--text",
        nargs="+",
        help="Raw MTEXT hole descriptions to parse (instead of file)"
    )
    parser.add_argument(
        "--validate-expected",
        action="store_true",
        help="Validate expected holes against themselves (self-test)"
    )

    args = parser.parse_args()

    # Show expected holes
    if args.show_expected:
        if args.part:
            print_expected_holes(args.part)
        else:
            print("Available parts with expected holes:")
            for part in EXPECTED_HOLES:
                print(f"  - {part}")
        return

    # Validate expected holes (self-test)
    if args.validate_expected and args.part:
        expected_mtext = EXPECTED_HOLES.get(args.part, [])
        result = find_holes_backup(part_number=args.part, text_input=expected_mtext)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"\nSelf-test validation for part {args.part}:")
            print("=" * 60)
            for i, hole in enumerate(result["holes"], 1):
                print(f"\n  Hole group {i}:")
                print(f"    Qty: {hole['qty']}")
                print(f"    Diameter: {hole['diameter'] or 'N/A'}")
                print(f"    Through: {hole['is_thru']}")
                print(f"    Operations: {', '.join(hole['operations']) or 'None'}")
                if hole.get("cbore_diameter"):
                    print(f"    C'BORE: {hole['cbore_diameter']}")
                if hole.get("cbore_depth_in"):
                    print(f"    C'BORE depth: {hole['cbore_depth_in']}")
                if hole.get("from_face"):
                    print(f"    From face: {hole['from_face']}")
            if result.get("validation"):
                val = result["validation"]
                print(f"\n  Validation: {'PASS' if val['matches'] else 'FAIL'}")
        return

    # If text input provided, use that
    if args.text:
        result = find_holes_backup(part_number=args.part, text_input=args.text)
    else:
        # Determine filepath
        filepath = args.filepath
        if not filepath and args.part:
            # Try to find the file for this part
            part_files = {
                "108": "/home/user/CAD_Quoting_Tool/Cad Files/T1769-108_redacted.dwg",
            }
            filepath = part_files.get(args.part)
            if not filepath:
                print(f"No default file path for part {args.part}")
                sys.exit(1)

        if not filepath:
            parser.print_help()
            sys.exit(1)

        # Run backup hole finder
        result = find_holes_backup(filepath=filepath, part_number=args.part)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\nBackup Hole Finder Results")
        print(f"File: {result['file']}")
        print("=" * 60)

        if "error" in result:
            print(f"\nError: {result['error']}")
        else:
            print(f"\nExtracted {len(result['holes'])} hole group(s):")
            for i, hole in enumerate(result["holes"], 1):
                print(f"\n  Hole group {i}:")
                print(f"    Qty: {hole['qty']}")
                print(f"    Diameter: {hole['diameter'] or 'N/A'}")
                print(f"    Through: {hole['is_thru']}")
                print(f"    Operations: {', '.join(hole['operations']) or 'None'}")
                if hole.get("cbore_diameter"):
                    print(f"    C'BORE: {hole['cbore_diameter']}")
                if hole.get("cbore_depth_in"):
                    print(f"    C'BORE depth: {hole['cbore_depth_in']}")
                if hole.get("from_face"):
                    print(f"    From face: {hole['from_face']}")

        # Show validation results
        if result.get("validation"):
            val = result["validation"]
            print(f"\nValidation Results:")
            print(f"  Matches: {'YES' if val['matches'] else 'NO'}")
            print(f"  Expected: {val['expected_count']} group(s)")
            print(f"  Extracted: {val['extracted_count']} group(s)")

            if val["discrepancies"]:
                print(f"\n  Discrepancies:")
                for disc in val["discrepancies"]:
                    print(f"    - {disc}")


if __name__ == "__main__":
    main()
