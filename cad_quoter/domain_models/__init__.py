"""Domain model value utilities."""

from .values import (
    coerce_float_or_none,
    parse_mixed_fraction,
    safe_float,
    to_float,
    to_int,
    to_positive_float,
)

# Re-export material constants for backwards compatibility
from cad_quoter.materials import (
    DEFAULT_MATERIAL_DISPLAY,
    DEFAULT_MATERIAL_KEY,
    MATERIAL_OTHER_KEY,
    normalize_material_key,
)

__all__ = [
    "coerce_float_or_none",
    "parse_mixed_fraction",
    "safe_float",
    "to_float",
    "to_int",
    "to_positive_float",
    "DEFAULT_MATERIAL_DISPLAY",
    "DEFAULT_MATERIAL_KEY",
    "MATERIAL_OTHER_KEY",
    "normalize_material_key",
]
