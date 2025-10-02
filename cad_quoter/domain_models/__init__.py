"""Domain models for the CAD quoting tool."""

from .materials import (
    DEFAULT_MATERIAL_DISPLAY,
    DEFAULT_MATERIAL_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEY,
    MATERIAL_DENSITY_G_CC_BY_KEYWORD,
    MATERIAL_DISPLAY_BY_KEY,
    MATERIAL_DROPDOWN_OPTIONS,
    MATERIAL_KEYWORDS,
    MATERIAL_MAP,
    MATERIAL_OTHER_KEY,
    canonical_material_key,
    is_material_match,
    normalize_material_key,
)
from .state import QuoteState
from .values import coerce_float_or_none

__all__ = [
    "DEFAULT_MATERIAL_DISPLAY",
    "DEFAULT_MATERIAL_KEY",
    "MATERIAL_DENSITY_G_CC_BY_KEY",
    "MATERIAL_DENSITY_G_CC_BY_KEYWORD",
    "MATERIAL_DISPLAY_BY_KEY",
    "MATERIAL_DROPDOWN_OPTIONS",
    "MATERIAL_KEYWORDS",
    "MATERIAL_MAP",
    "MATERIAL_OTHER_KEY",
    "QuoteState",
    "canonical_material_key",
    "coerce_float_or_none",
    "is_material_match",
    "normalize_material_key",
]
