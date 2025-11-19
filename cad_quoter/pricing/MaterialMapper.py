"""
Centralized material mapping service.

This module provides a unified interface for material name normalization and mapping
across different vendor systems (McMaster, Wieland, speeds/feeds databases).
"""

import csv
from pathlib import Path
from typing import Optional, Dict, List
import re


class MaterialMapper:
    """
    Centralized material mapping service that loads material_map.csv and provides
    standardized material lookups for use throughout the application.
    """

    # Singleton instance
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MaterialMapper, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the MaterialMapper by loading material_map.csv."""
        if self._initialized:
            return

        self._initialized = True
        self._load_material_map()

    def _load_material_map(self):
        """Load the material mapping CSV file."""
        # Find material_map.csv relative to this file
        csv_path = Path(__file__).parent / "resources" / "material_map.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Material map CSV not found at: {csv_path}")

        # Initialize mapping dictionaries
        self._input_to_canonical: Dict[str, str] = {}
        self._canonical_to_mcmaster: Dict[str, str] = {}
        self._canonical_to_wieland: Dict[str, str] = {}
        self._canonical_to_speeds: Dict[str, str] = {}
        self._canonical_to_density: Dict[str, float] = {}
        self._dropdown_options: List[str] = []

        # Load CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_term = row['input_term'].strip()
                canonical = row['canonical_material'].strip()
                mcmaster_key = row['mcmaster_key'].strip()
                wieland_key = row['wieland_key'].strip()
                speeds_key = row['speeds_key'].strip()
                density_str = row.get('density_lb_in3', '').strip()

                # Skip empty rows
                if not input_term or not canonical:
                    continue

                # Map input term to canonical material
                self._input_to_canonical[input_term] = canonical

                # Map canonical to vendor keys
                if mcmaster_key:
                    self._canonical_to_mcmaster[canonical] = mcmaster_key
                if wieland_key:
                    self._canonical_to_wieland[canonical] = wieland_key
                if speeds_key:
                    self._canonical_to_speeds[canonical] = speeds_key

                # Map canonical to density
                if density_str:
                    try:
                        density = float(density_str)
                        self._canonical_to_density[canonical] = density
                    except ValueError:
                        # Skip invalid density values
                        pass

                # Add to dropdown options (use input_term as display name)
                if input_term not in self._dropdown_options:
                    self._dropdown_options.append(input_term)

        # Sort dropdown options alphabetically, but keep GENERIC at the end
        generic_items = [item for item in self._dropdown_options if item.upper() == 'GENERIC']
        non_generic_items = [item for item in self._dropdown_options if item.upper() != 'GENERIC']
        self._dropdown_options = sorted(non_generic_items) + generic_items

    def get_canonical_material(self, input_material: Optional[str]) -> str:
        """
        Get the canonical material name for a given input.

        Args:
            input_material: User-provided material name

        Returns:
            Canonical material name, or "GENERIC" if not found
        """
        if not input_material:
            return "GENERIC"

        # Clean input
        cleaned = input_material.strip()

        # Check if input is already a canonical material name
        # (exists in the canonical-to-vendor mappings)
        if cleaned in self._canonical_to_mcmaster:
            return cleaned

        # Try exact match in input terms
        if cleaned in self._input_to_canonical:
            return self._input_to_canonical[cleaned]

        # Try case-insensitive match in canonical names
        for canonical_name in self._canonical_to_mcmaster.keys():
            if canonical_name.lower() == cleaned.lower():
                return canonical_name

        # Try case-insensitive match in input terms
        for input_term, canonical in self._input_to_canonical.items():
            if input_term.lower() == cleaned.lower():
                return canonical

        # Try fuzzy matching (normalize and compare)
        normalized_input = self._normalize_material_key(cleaned)

        # Check canonical names first
        for canonical_name in self._canonical_to_mcmaster.keys():
            if self._normalize_material_key(canonical_name) == normalized_input:
                return canonical_name

        # Then check input terms
        for input_term, canonical in self._input_to_canonical.items():
            if self._normalize_material_key(input_term) == normalized_input:
                return canonical

        # Default to GENERIC if no match found
        return "GENERIC"

    def get_mcmaster_key(self, material: str) -> Optional[str]:
        """
        Get the McMaster catalog lookup key for a material.

        Args:
            material: Material name (can be input term or canonical)

        Returns:
            McMaster key, or None if not found
        """
        canonical = self.get_canonical_material(material)
        return self._canonical_to_mcmaster.get(canonical)

    def get_wieland_key(self, material: str) -> Optional[str]:
        """
        Get the Wieland scrap pricing key for a material.

        Args:
            material: Material name (can be input term or canonical)

        Returns:
            Wieland key (e.g., "AL", "SS", "TI"), or None if not found
        """
        canonical = self.get_canonical_material(material)
        return self._canonical_to_wieland.get(canonical)

    def get_speeds_key(self, material: str) -> Optional[str]:
        """
        Get the speeds/feeds lookup key for a material.

        Args:
            material: Material name (can be input term or canonical)

        Returns:
            Speeds/feeds key, or None if not found
        """
        canonical = self.get_canonical_material(material)
        return self._canonical_to_speeds.get(canonical)

    def get_density_lb_in3(self, material: str) -> float:
        """
        Get the density in lb/in³ for a material.

        Args:
            material: Material name (can be input term or canonical)

        Returns:
            Density in lb/in³. Returns 0.283 (steel) as default if not found.
        """
        canonical = self.get_canonical_material(material)
        return self._canonical_to_density.get(canonical, 0.283)

    def get_dropdown_options(self) -> List[str]:
        """
        Get the list of material options for the dropdown UI.

        Returns:
            Sorted list of material display names
        """
        return self._dropdown_options.copy()

    def is_valid_material(self, material: str) -> bool:
        """
        Check if a material is valid (exists in the mapping).

        Args:
            material: Material name to check

        Returns:
            True if material is valid, False otherwise
        """
        if not material:
            return False
        canonical = self.get_canonical_material(material)
        return canonical != "GENERIC" or material.upper() == "GENERIC"

    @staticmethod
    def _normalize_material_key(value: str) -> str:
        """
        Normalize a material name for fuzzy matching.

        Args:
            value: Material name to normalize

        Returns:
            Normalized string (lowercase, alphanumeric + spaces)
        """
        cleaned = re.sub(r"[^0-9a-zA-Z]+", " ", value.strip().lower())
        return re.sub(r"\s+", " ", cleaned).strip()


# Singleton instance for easy import
material_mapper = MaterialMapper()
