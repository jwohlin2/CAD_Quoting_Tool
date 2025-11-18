"""
Dimension Finder - Find Part Bounding Box Dimensions from Extracted Text
=========================================================================

Standalone module for identifying the bounding box dimensions (L x W x H)
from mtext_results.json files produced by mtext_normalizer.py.

This module provides utilities to:
1. Parse mtext_results.json files
2. Extract all numeric dimension values
3. Match dimensions with tolerance ("close enough" matching)
4. Identify likely bounding box dimensions using heuristics

Usage:
    from dimension_finder import DimensionFinder

    finder = DimensionFinder()
    finder.load_results("path/to/mtext_results.json")

    # Find specific dimensions with tolerance
    matches = finder.find_dimension(2.5, tolerance=0.01)

    # Get all extracted dimensions
    dims = finder.get_all_dimensions()

    # Attempt to identify bounding box dimensions
    bbox = finder.find_bounding_box()

Author: CAD Quoting Tool
Date: 2025-11-17
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

# Optional ezdxf import for DXF file parsing
try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

# Import mtext normalizer if available
try:
    from .mtext_normalizer import normalize_acad_mtext, units_to_inch_factor
except ImportError:
    try:
        from mtext_normalizer import normalize_acad_mtext, units_to_inch_factor
    except ImportError:
        def normalize_acad_mtext(text):
            return text
        def units_to_inch_factor(units):
            return 1.0


@dataclass
class DimensionMatch:
    """Represents a matched dimension value."""
    value: float
    source_text: str
    measurement_in: float
    dimtype: int
    is_diameter: bool
    confidence: float  # 0.0 to 1.0


class DimensionFinder:
    """
    Find and analyze dimensions from mtext_results.json files.
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.dimensions: List[Dict[str, Any]] = []
        self._numeric_values: List[Tuple[float, str, Dict]] = []  # (value, source, dim_info)

    def load_results(self, json_path: Path | str) -> None:
        """
        Load mtext_results.json file.

        Args:
            json_path: Path to the JSON file
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Results file not found: {json_path}")

        with open(json_path, 'r') as f:
            self.results = json.load(f)

        self.dimensions = self.results.get("dimensions", [])
        self._extract_numeric_values()

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load results from a dictionary.

        Args:
            data: Dictionary with mtext results structure
        """
        self.results = data
        self.dimensions = data.get("dimensions", [])
        self._extract_numeric_values()

    def load_dxf(self, dxf_path: Path | str) -> None:
        """
        Load dimensions directly from a DXF file.

        Extracts text from:
        - DIMENSION entities (with resolved measurements)
        - TEXT entities
        - MTEXT entities

        Args:
            dxf_path: Path to DXF file
        """
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf is required for DXF parsing. Install with: pip install ezdxf")

        dxf_path = Path(dxf_path)
        if not dxf_path.exists():
            raise FileNotFoundError(f"DXF file not found: {dxf_path}")

        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()

        # Get unit conversion factor
        insunits = doc.header.get("$INSUNITS", 1)
        measurement = doc.header.get("$MEASUREMENT", 0)
        unit_factor = units_to_inch_factor(insunits)

        # Override if $MEASUREMENT indicates metric
        if measurement == 1 and insunits not in [4, 5, 6]:
            unit_factor = 1.0 / 25.4

        self.results = {
            "source_file": str(dxf_path),
            "unit_factor": unit_factor,
            "text_entities": [],
            "mtext_entities": [],
            "dimensions": []
        }

        # Extract TEXT entities
        for text in msp.query("TEXT"):
            text_content = text.dxf.text if hasattr(text.dxf, 'text') else ""
            if text_content:
                self.results["text_entities"].append({
                    "text": text_content,
                    "type": "TEXT"
                })

        # Extract MTEXT entities
        for mtext in msp.query("MTEXT"):
            text_content = mtext.text if hasattr(mtext, 'text') else ""
            normalized = normalize_acad_mtext(text_content)
            if normalized:
                self.results["mtext_entities"].append({
                    "text": normalized,
                    "raw_text": text_content,
                    "type": "MTEXT"
                })

        # Extract DIMENSION entities
        for dim in msp.query("DIMENSION"):
            try:
                raw_text = ""
                if hasattr(dim, 'dxf') and hasattr(dim.dxf, 'text'):
                    raw_text = dim.dxf.text or ""

                meas = dim.get_measurement()
                if meas is None:
                    meas = 0.0

                if hasattr(meas, 'magnitude'):
                    meas = meas.magnitude
                elif hasattr(meas, 'x'):
                    meas = abs(meas.x)
                meas = float(meas)

                value_in = meas * unit_factor
                dimtype = dim.dimtype if hasattr(dim, 'dimtype') else 0

                # Format nominal string
                nominal_str = f"{value_in:.4f}".rstrip("0").rstrip(".")
                if nominal_str.startswith("0.") and value_in < 1.0:
                    nominal_str = nominal_str[1:]
                elif not nominal_str or nominal_str == ".":
                    nominal_str = "0"

                # Resolve text
                text = normalize_acad_mtext(raw_text) if raw_text else ""
                if "<>" in text and nominal_str:
                    text = text.replace("<>", nominal_str)
                elif not text and nominal_str:
                    text = nominal_str

                is_diameter = (
                    dimtype == 3 or
                    "%%c" in raw_text.lower() or
                    "Ø" in raw_text or
                    "DIA" in raw_text.upper()
                )

                self.results["dimensions"].append({
                    "resolved_text": text.strip(),
                    "raw_text": raw_text,
                    "measurement": meas,
                    "measurement_in": value_in,
                    "dimtype": dimtype,
                    "is_diameter": is_diameter
                })
            except Exception as e:
                continue

        self.dimensions = self.results.get("dimensions", [])
        self._extract_numeric_values()

        # Also extract from TEXT and MTEXT entities
        self._extract_from_text_entities()

    def _extract_from_text_entities(self) -> None:
        """Extract numeric values from TEXT and MTEXT entities."""
        # From TEXT entities
        for item in self.results.get("text_entities", []):
            text = item.get("text", "")
            values = self._extract_numbers_from_text(text)
            for val in values:
                if val > 0:
                    self._numeric_values.append((
                        val,
                        f"TEXT: {text}",
                        {"dimtype": -1, "is_diameter": False, "measurement_in": val}
                    ))

        # From MTEXT entities
        for item in self.results.get("mtext_entities", []):
            text = item.get("text", "")
            values = self._extract_numbers_from_text(text)
            for val in values:
                if val > 0:
                    self._numeric_values.append((
                        val,
                        f"MTEXT: {text}",
                        {"dimtype": -1, "is_diameter": False, "measurement_in": val}
                    ))

    def _extract_numeric_values(self) -> None:
        """Extract all numeric values from the dimensions."""
        self._numeric_values = []

        for dim in self.dimensions:
            resolved_text = dim.get("resolved_text", "")
            measurement_in = dim.get("measurement_in", 0.0)

            # Extract numeric value from measurement_in
            if measurement_in > 0:
                self._numeric_values.append((
                    measurement_in,
                    f"measurement: {measurement_in}",
                    dim
                ))

            # Also extract numbers from resolved_text
            # This catches values that might be formatted differently
            text_values = self._extract_numbers_from_text(resolved_text)
            for val in text_values:
                if val > 0:
                    self._numeric_values.append((
                        val,
                        f"text: {resolved_text}",
                        dim
                    ))

    def _extract_numbers_from_text(self, text: str) -> List[float]:
        """
        Extract all numeric values from text string.

        Handles formats like:
        - "2.5"
        - ".5005"
        - "8.7209"
        - "2.5+.0000/-.0002" (extracts 2.5)
        - "(2) .148" (extracts .148)

        Args:
            text: Text to parse

        Returns:
            List of extracted float values
        """
        values = []

        # Remove tolerance patterns to get base value
        # Pattern: value+tol/-tol or value±tol
        text_clean = re.sub(r'\+[.\d]+/-[.\d]+', '', text)
        text_clean = re.sub(r'±[.\d]+', '', text_clean)

        # Find all numeric patterns
        # Matches: .123, 0.123, 123, 123.456
        patterns = [
            r'(?<![.\d])(\d+\.?\d*)',  # Integer or decimal
            r'(?<![.\d])(\.\d+)',       # Decimal starting with .
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text_clean)
            for match in matches:
                try:
                    val = float(match)
                    # Filter out likely non-dimension values
                    # (like count prefixes, angles in degrees > 90)
                    if 0 < val < 1000 and val not in values:
                        values.append(val)
                except ValueError:
                    continue

        return values

    def get_all_dimensions(self) -> List[float]:
        """
        Get all unique dimension values.

        Returns:
            Sorted list of unique dimension values
        """
        values = set()
        for val, _, _ in self._numeric_values:
            # Round to avoid floating point duplicates
            rounded = round(val, 4)
            if 0.001 < rounded < 1000:  # Filter out very small and very large
                values.add(rounded)

        return sorted(values)

    def find_dimension(
        self,
        target: float,
        tolerance: float = 0.01,
        relative_tolerance: float = 0.005
    ) -> List[DimensionMatch]:
        """
        Find dimensions matching a target value within tolerance.

        Uses both absolute and relative tolerance for matching.

        Args:
            target: Target dimension value
            tolerance: Absolute tolerance (e.g., 0.01 for ±0.01")
            relative_tolerance: Relative tolerance (e.g., 0.005 for ±0.5%)

        Returns:
            List of matching dimensions sorted by confidence
        """
        matches = []

        # Calculate effective tolerance
        abs_tol = tolerance
        rel_tol = target * relative_tolerance
        eff_tol = max(abs_tol, rel_tol)

        for val, source, dim_info in self._numeric_values:
            diff = abs(val - target)

            if diff <= eff_tol:
                # Calculate confidence based on how close the match is
                if diff == 0:
                    confidence = 1.0
                else:
                    confidence = max(0, 1.0 - (diff / eff_tol))

                match = DimensionMatch(
                    value=val,
                    source_text=source,
                    measurement_in=dim_info.get("measurement_in", 0),
                    dimtype=dim_info.get("dimtype", 0),
                    is_diameter=dim_info.get("is_diameter", False),
                    confidence=confidence
                )
                matches.append(match)

        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)

        # Remove duplicates (same value)
        seen_values = set()
        unique_matches = []
        for m in matches:
            rounded = round(m.value, 4)
            if rounded not in seen_values:
                seen_values.add(rounded)
                unique_matches.append(m)

        return unique_matches

    def find_bounding_box(
        self,
        expected: Optional[Tuple[float, float, float]] = None,
        tolerance: float = 0.02
    ) -> List[Tuple[float, float]]:
        """
        Attempt to identify bounding box dimensions.

        If expected values are provided, finds best matches.
        Otherwise, uses heuristics to identify likely bounding box dims.

        Args:
            expected: Optional tuple of expected (L, W, H) dimensions
            tolerance: Tolerance for matching expected values

        Returns:
            List of (matched_value, confidence) tuples
        """
        if expected:
            return self._match_expected_bbox(expected, tolerance)
        else:
            return self._infer_bounding_box()

    def _match_expected_bbox(
        self,
        expected: Tuple[float, float, float],
        tolerance: float
    ) -> List[Tuple[float, float]]:
        """Match expected bounding box dimensions."""
        results = []

        for exp_dim in expected:
            matches = self.find_dimension(exp_dim, tolerance=tolerance)
            if matches:
                best = matches[0]
                results.append((best.value, best.confidence))
            else:
                results.append((None, 0.0))

        return results

    def _infer_bounding_box(self) -> List[Tuple[float, float]]:
        """
        Infer bounding box dimensions using heuristics.

        Key insight: Bounding box dimensions often have tight tolerances
        because they're critical for fitment. Ordinate dimensions (dimtype 6)
        without tolerances are usually positions from a datum, not overall sizes.

        Strategy:
        1. Strongly prefer dimensions with tolerances
        2. Prefer linear (dimtype 0) over ordinate (dimtype 6)
        3. Ordinate dims without tolerances are likely positions - deprioritize
        4. Size still matters but is secondary to tolerance presence
        """
        candidates = []

        for dim in self.dimensions:
            resolved_text = dim.get("resolved_text", "")
            measurement = dim.get("measurement_in", 0)
            dimtype = dim.get("dimtype", 0)

            # Skip angles (dimtype 2)
            if dimtype == 2:
                continue

            # Skip very small values (chamfers, fillets)
            if measurement < 0.05:
                continue

            # Skip diameter dimensions (usually holes, not bounding box)
            if dim.get("is_diameter", False):
                continue

            # Skip radii (dimtype 4)
            if dimtype == 4:
                continue

            # Calculate score based on heuristics
            score = 0.0

            # Check for tolerance (critical indicator!)
            has_tolerance = ('+' in resolved_text or '±' in resolved_text or
                           '-.00' in resolved_text or '+.00' in resolved_text)

            if has_tolerance:
                score += 3.0  # Strong bonus for toleranced dimensions

            # Dimension type scoring
            if dimtype == 0:
                # Linear dimensions - these are direct measurements
                score += 1.0
            elif dimtype == 6:
                # Ordinate dimensions - positions from datum
                if has_tolerance:
                    score += 0.5  # Still good if toleranced
                else:
                    score -= 1.0  # Penalize untolerance ordinate dims

            # Size factor (smaller bonus than before)
            if measurement > 0.5:
                score += 0.2
            if measurement > 2.0:
                score += 0.1

            # Penalize reference dimensions and scale markers
            if 'REF' in resolved_text.upper():
                score -= 2.0
            if ' sc' in resolved_text.lower():
                score -= 2.0
            if 'TYP' in resolved_text.upper():
                score -= 0.5  # Slightly penalize typical dims

            # Penalize multiplicity prefix less - could still be bbox
            if re.match(r'^\(\d+\)', resolved_text):
                score -= 0.2

            candidates.append((measurement, score, dim))

        # Sort by SCORE first, then by value for ties
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # Get unique values, preserving score order
        seen = set()
        top_dims = []
        for val, score, _ in candidates:
            rounded = round(val, 4)
            if rounded not in seen:
                seen.add(rounded)
                top_dims.append((val, score))
                if len(top_dims) >= 10:  # Return top 10 candidates
                    break

        return top_dims

    def get_top_3_dimensions(self) -> List[float]:
        """
        Get the 3 largest dimensions as likely bounding box.

        Returns:
            List of 3 largest unique dimension values, sorted descending
        """
        bbox_candidates = self._infer_bounding_box()
        return [val for val, _ in bbox_candidates[:3]]

    def compare_with_expected(
        self,
        expected_dims: Tuple[float, float, float],
        tolerance: float = 0.02
    ) -> Dict[str, Any]:
        """
        Compare extracted dimensions with expected bounding box.

        Args:
            expected_dims: Expected (L, W, H) dimensions
            tolerance: Tolerance for matching

        Returns:
            Dict with comparison results including matches and missing dims
        """
        all_dims = self.get_all_dimensions()

        result = {
            "expected": expected_dims,
            "all_extracted": all_dims,
            "matches": [],
            "missing": [],
            "match_rate": 0.0
        }

        matched_count = 0
        for exp in expected_dims:
            matches = self.find_dimension(exp, tolerance=tolerance)
            if matches:
                best = matches[0]
                result["matches"].append({
                    "expected": exp,
                    "found": best.value,
                    "confidence": best.confidence,
                    "source": best.source_text
                })
                matched_count += 1
            else:
                # Try with looser tolerance
                loose_matches = self.find_dimension(exp, tolerance=tolerance * 3)
                if loose_matches:
                    best = loose_matches[0]
                    result["matches"].append({
                        "expected": exp,
                        "found": best.value,
                        "confidence": best.confidence * 0.5,  # Lower confidence
                        "source": best.source_text,
                        "note": "loose match"
                    })
                    matched_count += 0.5
                else:
                    result["missing"].append(exp)

        result["match_rate"] = matched_count / len(expected_dims)

        return result


def analyze_file(
    file_path: Path | str,
    expected: Optional[Tuple[float, float, float]] = None,
    tolerance: float = 0.02
) -> Dict[str, Any]:
    """
    Analyze a single mtext_results.json or DXF file.

    Args:
        file_path: Path to the JSON or DXF file
        expected: Optional expected dimensions
        tolerance: Tolerance for matching

    Returns:
        Analysis results
    """
    finder = DimensionFinder()
    file_path = Path(file_path)

    # Determine file type and load accordingly
    if file_path.suffix.lower() == '.dxf':
        finder.load_dxf(file_path)
    else:
        finder.load_results(file_path)

    # Count text entities if available
    text_count = len(finder.results.get("text_entities", []))
    mtext_count = len(finder.results.get("mtext_entities", []))

    result = {
        "file": str(file_path),
        "total_dimensions": len(finder.dimensions),
        "text_entities": text_count,
        "mtext_entities": mtext_count,
        "unique_values": finder.get_all_dimensions(),
    }

    if expected:
        result["comparison"] = finder.compare_with_expected(expected, tolerance)
        result["inferred_bbox"] = finder.find_bounding_box()[:5]  # Top 5 candidates
    else:
        result["inferred_bbox"] = finder.find_bounding_box()[:5]

    return result


def main():
    """Command-line interface for dimension finder."""
    import sys
    import os

    if len(sys.argv) < 2:
        print("Dimension Finder - Find bounding box dimensions from CAD files")
        print()
        print("Usage:")
        print("  python dimension_finder.py <json_or_dxf_file> [expected_dims]")
        print("  python dimension_finder.py --analyze-all <directory>")
        print()
        print("Examples:")
        print("  python dimension_finder.py results.json")
        print("  python dimension_finder.py drawing.dxf 8.72x2.5x5.005")
        print("  python dimension_finder.py results.json 8.72x2.5x5.005")
        print("  python dimension_finder.py --analyze-all './Cad Files'")
        sys.exit(1)

    if sys.argv[1] == "--analyze-all":
        # Analyze all mtext_results.json files in a directory
        if len(sys.argv) < 3:
            print("Error: Please provide a directory path")
            sys.exit(1)

        directory = Path(sys.argv[2])
        json_files = list(directory.glob("*mtext_results.json"))

        if not json_files:
            print(f"No mtext_results.json files found in {directory}")
            sys.exit(1)

        print(f"Found {len(json_files)} files\n")

        for json_file in sorted(json_files):
            print(f"{'='*60}")
            print(f"File: {json_file.name}")
            print(f"{'='*60}")

            try:
                result = analyze_file(json_file)

                print(f"Total dimensions: {result['total_dimensions']}")
                print(f"Unique values: {len(result['unique_values'])}")
                print()

                # Show top dimension candidates
                print("Top bounding box candidates:")
                for val, score in result['inferred_bbox']:
                    print(f"  {val:.4f}\" (score: {score:.2f})")
                print()

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                print()

    else:
        # Analyze single file
        input_path = Path(sys.argv[1])

        expected = None
        if len(sys.argv) >= 3:
            # Parse expected dimensions (format: "8.72x2.5x5.005")
            dims_str = sys.argv[2]
            try:
                dims = [float(d) for d in dims_str.split('x')]
                if len(dims) == 3:
                    expected = tuple(dims)
                else:
                    print(f"Warning: Expected 3 dimensions, got {len(dims)}")
            except ValueError:
                print(f"Warning: Could not parse dimensions '{dims_str}'")

        result = analyze_file(input_path, expected)

        print(f"File: {result['file']}")
        print(f"Total dimensions: {result['total_dimensions']}")
        if result.get('text_entities', 0) > 0:
            print(f"TEXT entities: {result['text_entities']}")
        if result.get('mtext_entities', 0) > 0:
            print(f"MTEXT entities: {result['mtext_entities']}")
        print(f"Unique values: {len(result['unique_values'])}")
        print()

        # Show all extracted values
        print("All extracted dimension values:")
        dims = result['unique_values']
        # Group into rows of 8
        for i in range(0, len(dims), 8):
            row = dims[i:i+8]
            print("  " + "  ".join(f"{v:.4f}" for v in row))
        print()

        if expected and "comparison" in result:
            comp = result["comparison"]
            print(f"Expected: {comp['expected']}")
            print(f"Match rate: {comp['match_rate']*100:.0f}%")
            print()

            if comp["matches"]:
                print("Matches found:")
                for m in comp["matches"]:
                    note = f" ({m.get('note', '')})" if m.get('note') else ""
                    print(f"  {m['expected']} -> {m['found']:.4f} "
                          f"(confidence: {m['confidence']:.2f}){note}")

            if comp["missing"]:
                print()
                print("Missing dimensions:")
                for m in comp["missing"]:
                    print(f"  {m}")

        print()
        print("Top bounding box candidates (inferred):")
        for val, score in result['inferred_bbox']:
            print(f"  {val:.4f}\" (score: {score:.2f})")


if __name__ == "__main__":
    main()
