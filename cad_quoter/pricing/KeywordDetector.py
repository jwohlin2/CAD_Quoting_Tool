"""Keyword detection from CAD file text - for identifying special requirements and features."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Optional, Any


@dataclass
class KeywordMatch:
    """Represents a keyword match found in CAD text."""
    keyword: str
    category: str
    matched_text: str  # The actual text that matched (preserves case)
    context: str = ""  # Surrounding text for context
    canonical_material: Optional[str] = None  # Canonical material name (for materials)

    def __repr__(self):
        return f"KeywordMatch(keyword='{self.keyword}', category='{self.category}')"


@dataclass
class KeywordDetectionResult:
    """Results from keyword detection in CAD file."""
    matches: List[KeywordMatch] = field(default_factory=list)
    categories_found: Set[str] = field(default_factory=set)
    all_text: List[str] = field(default_factory=list)

    def has_keyword(self, keyword: str) -> bool:
        """Check if a specific keyword was found."""
        return any(m.keyword.upper() == keyword.upper() for m in self.matches)

    def has_category(self, category: str) -> bool:
        """Check if any keyword from a category was found."""
        return category in self.categories_found

    def get_matches_by_category(self, category: str) -> List[KeywordMatch]:
        """Get all matches for a specific category."""
        return [m for m in self.matches if m.category == category]

    def summary(self) -> Dict[str, List[str]]:
        """Get summary of matches grouped by category."""
        summary = {}
        for match in self.matches:
            if match.category not in summary:
                summary[match.category] = []
            summary[match.category].append(match.keyword)
        return summary


class KeywordDetector:
    """Detects keywords in CAD file text for process planning and pricing."""

    def __init__(self):
        """Initialize with default keyword categories."""
        # Keywords organized by category
        # Add more keywords to these lists as needed
        self.keyword_categories: Dict[str, List[str]] = {
            # Surface finish requirements
            "FINISH": [],

            # Heat treatment
            "HEAT_TREAT": [],

            # Coating/plating
            "COATING": [],

            # Tolerance requirements
            "TOLERANCE": [],

            # Material specifications
            "MATERIAL": [],

            # Inspection requirements
            "INSPECTION": [],

            # Special processes
            "SPECIAL_PROCESS": [],

            # Assembly notes
            "ASSEMBLY": [],
        }

        # Material keyword to canonical name mapping
        self.material_mapping: Dict[str, str] = {}

        # Case-insensitive search enabled by default
        self.case_sensitive = False

    def add_keywords(self, category: str, keywords: List[str]) -> None:
        """
        Add keywords to a category.

        Args:
            category: Category name (e.g., "FINISH", "HEAT_TREAT")
            keywords: List of keywords to add

        Example:
            >>> detector = KeywordDetector()
            >>> detector.add_keywords("FINISH", ["POLISH", "GRIND", "BEAD BLAST"])
        """
        if category not in self.keyword_categories:
            self.keyword_categories[category] = []

        self.keyword_categories[category].extend(keywords)

    def load_materials_from_csv(self, csv_path: str | Path) -> None:
        """
        Load material keywords from material_map.csv.

        This reads the CSV and adds all input_label values as searchable keywords
        in the MATERIAL category, mapping them to their canonical_material names.

        Args:
            csv_path: Path to material_map.csv

        Example:
            >>> detector = KeywordDetector()
            >>> detector.load_materials_from_csv("cad_quoter/pricing/resources/material_map.csv")
        """
        csv_path = Path(csv_path)
        material_keywords = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_label = row['input_label'].strip()
                canonical = row['canonical_material'].strip()

                # Add to keyword list
                material_keywords.append(input_label)

                # Map keyword to canonical name (case-insensitive key)
                self.material_mapping[input_label.upper()] = canonical

        # Add all materials to MATERIAL category
        self.add_keywords("MATERIAL", material_keywords)

    def detect_from_text(self, text_list: List[str]) -> KeywordDetectionResult:
        """
        Search for keywords in extracted CAD text.

        Args:
            text_list: List of text strings from CAD file

        Returns:
            KeywordDetectionResult with all matches found

        Example:
            >>> detector = KeywordDetector()
            >>> detector.add_keywords("FINISH", ["POLISH", "BEAD BLAST"])
            >>> text = ["POLISH ALL SURFACES", "HOLE NOTES", "BEAD BLAST AFTER HEAT TREAT"]
            >>> result = detector.detect_from_text(text)
            >>> print(result.summary())
        """
        result = KeywordDetectionResult(all_text=text_list)

        for text_entry in text_list:
            # Search for each keyword in each category
            for category, keywords in self.keyword_categories.items():
                for keyword in keywords:
                    if self._text_contains_keyword(text_entry, keyword):
                        # Get canonical material if this is a material keyword
                        canonical_mat = None
                        if category == "MATERIAL":
                            canonical_mat = self.material_mapping.get(keyword.upper())

                        match = KeywordMatch(
                            keyword=keyword,
                            category=category,
                            matched_text=self._extract_matched_text(text_entry, keyword),
                            context=text_entry,
                            canonical_material=canonical_mat
                        )
                        result.matches.append(match)
                        result.categories_found.add(category)

        return result

    def detect_from_cad_file(self, cad_file_path: str | Path) -> KeywordDetectionResult:
        """
        Extract text from CAD file and detect keywords.

        Args:
            cad_file_path: Path to CAD file (DXF/DWG)

        Returns:
            KeywordDetectionResult with all matches found

        Example:
            >>> detector = KeywordDetector()
            >>> detector.add_keywords("FINISH", ["POLISH", "GRIND"])
            >>> result = detector.detect_from_cad_file("part.dxf")
            >>> if result.has_category("FINISH"):
            ...     print("Special finish required")
        """
        from cad_quoter.planning import extract_all_text_from_cad

        text_list = extract_all_text_from_cad(cad_file_path)
        return self.detect_from_text(text_list)

    def _text_contains_keyword(self, text: str, keyword: str) -> bool:
        """Check if text contains keyword (case-insensitive by default)."""
        if self.case_sensitive:
            return keyword in text
        else:
            return keyword.upper() in text.upper()

    def _extract_matched_text(self, text: str, keyword: str) -> str:
        """Extract the actual matched text preserving original case."""
        if self.case_sensitive:
            return keyword

        # Find the keyword in text preserving case
        text_upper = text.upper()
        keyword_upper = keyword.upper()

        idx = text_upper.find(keyword_upper)
        if idx != -1:
            return text[idx:idx + len(keyword)]

        return keyword


# Convenience function for quick keyword detection
def detect_keywords_in_cad(
    cad_file_path: str | Path,
    keyword_dict: Optional[Dict[str, List[str]]] = None
) -> KeywordDetectionResult:
    """
    Quick keyword detection in CAD file.

    Args:
        cad_file_path: Path to CAD file
        keyword_dict: Dictionary of {category: [keywords]} to search for

    Returns:
        KeywordDetectionResult

    Example:
        >>> keywords = {
        ...     "FINISH": ["POLISH", "GRIND"],
        ...     "HEAT_TREAT": ["HARDEN", "TEMPER"]
        ... }
        >>> result = detect_keywords_in_cad("part.dxf", keywords)
        >>> print(result.summary())
    """
    detector = KeywordDetector()

    if keyword_dict:
        for category, keywords in keyword_dict.items():
            detector.add_keywords(category, keywords)

    return detector.detect_from_cad_file(cad_file_path)


def detect_material_in_cad(
    cad_file_path: str | Path,
    material_csv_path: Optional[str | Path] = None,
    default_material: str = "GENERIC"
) -> str:
    """
    Detect material from CAD file text, defaulting to GENERIC if not found.

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)
        material_csv_path: Path to material_map.csv (defaults to standard location)
        default_material: Material to return if none found (default: "GENERIC")

    Returns:
        Canonical material name (e.g., "17-4 PH Stainless Steel", "GENERIC")

    Example:
        >>> material = detect_material_in_cad("part.dxf")
        >>> print(f"Material: {material}")
        Material: GENERIC
    """
    # Use default material_map.csv location if not specified
    if material_csv_path is None:
        import os
        script_dir = Path(__file__).parent
        material_csv_path = script_dir / "resources" / "material_map.csv"

    # Set up detector with materials from CSV
    detector = KeywordDetector()
    detector.load_materials_from_csv(material_csv_path)

    # Run detection
    result = detector.detect_from_cad_file(cad_file_path)

    # Get material matches
    material_matches = result.get_matches_by_category("MATERIAL")

    if material_matches:
        # Return the canonical name of the first match
        return material_matches[0].canonical_material
    else:
        # No material found, return default
        return default_material
