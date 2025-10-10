"""Domain primitives describing supported material selections."""
from __future__ import annotations

from typing import Dict, Set


def normalize_material_key(value: str) -> str:
    """Return a canonical lookup key for material identifiers."""

    import re

    cleaned = re.sub(r"[^0-9a-z]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


MATERIAL_DROPDOWN_OPTIONS = [
    "Aluminum",
    "Aluminum 5083",
    "Aluminum MIC6",
    "Berylium Copper",
    "Bismuth",
    "Brass",
    "Carbide",
    "Ceramic",
    "Cobalt",
    "Copper",
    "Gold",
    "Inconel",
    "Indium",
    "Lead",
    "Nickel",
    "Nickel Silver",
    "Palladium",
    "Phosphor Bronze",
    "Phosphorus",
    "Silver",
    "Stainless Steel",
    "Stainless Steel 303",
    "Steel",
    "Steel Low-Carbon",
    "Steel Mild",
    "Tin",
    "Titanium",
    "Tool Steel",
    "Tool Steel A2",
    "Tungsten",
    "Other (enter custom price)",
]

DEFAULT_MATERIAL_DISPLAY = "Aluminum MIC6"
DEFAULT_MATERIAL_KEY = normalize_material_key(DEFAULT_MATERIAL_DISPLAY)

_MATERIAL_ADDITIONAL_KEYWORDS: Dict[str, Set[str]] = {
    "Aluminum": {"aluminium"},
    "Aluminum 5083": {"5083", "al 5083"},
    "Aluminum MIC6": {"mic6", "mic-6"},
    "Berylium Copper": {"beryllium copper", "c172", "cube"},
    "Brass": {"c360", "c260"},
    "Copper": {"cu"},
    "Inconel": {"in718", "in625"},
    "Nickel": {"ni"},
    "Nickel Silver": {"german silver"},
    "Phosphor Bronze": {"phosphorbronze"},
    "Stainless Steel": {"stainless"},
    "Stainless Steel 303": {"ss303", "303"},
    "Steel": {"carbon steel"},
    "Steel Low-Carbon": {"low carbon", "low carbon steel"},
    "Steel Mild": {"mild steel"},
    "Titanium": {"ti", "ti6al4v"},
    "Tool Steel": {"h13", "o1", "d2"},
    "Tool Steel A2": {"a2", "a2 tool steel"},
}

MATERIAL_KEYWORDS: Dict[str, Set[str]] = {}
for display in MATERIAL_DROPDOWN_OPTIONS:
    key = normalize_material_key(display)
    if not key:
        continue
    extras = {normalize_material_key(term) for term in _MATERIAL_ADDITIONAL_KEYWORDS.get(display, set())}
    extras.discard("")
    MATERIAL_KEYWORDS[key] = {key} | extras

MATERIAL_DISPLAY_BY_KEY: Dict[str, str] = {}
for display in MATERIAL_DROPDOWN_OPTIONS:
    key = normalize_material_key(display)
    if key:
        MATERIAL_DISPLAY_BY_KEY[key] = display

MATERIAL_OTHER_KEY = normalize_material_key(MATERIAL_DROPDOWN_OPTIONS[-1])

_MATERIAL_DENSITY_G_CC_BY_DISPLAY: Dict[str, float | None] = {
    "Aluminum": 2.70,
    "Aluminum 5083": 2.66,
    "Aluminum MIC6": 2.70,
    "Berylium Copper": 8.25,
    "Bismuth": 9.78,
    "Brass": 8.40,
    "Carbide": 15.60,
    "Ceramic": 3.90,
    "Cobalt": 8.90,
    "Copper": 8.96,
    "Gold": 19.32,
    "Inconel": 8.44,
    "Indium": 7.31,
    "Lead": 11.34,
    "Nickel": 8.90,
    "Nickel Silver": 8.70,
    "Palladium": 12.02,
    "Phosphor Bronze": 8.80,
    "Phosphorus": 1.82,
    "Silver": 10.49,
    "Stainless Steel": 7.90,
    "Stainless Steel 303": 8.00,
    "Steel": 7.85,
    "Steel Low-Carbon": 7.85,
    "Steel Mild": 7.87,
    "Tin": 7.31,
    "Titanium": 4.43,
    "Tool Steel": 7.80,
    "Tool Steel A2": 7.86,
    "Tungsten": 19.30,
    "Other (enter custom price)": None,
}

MATERIAL_DENSITY_G_CC_BY_KEY: Dict[str, float] = {}
MATERIAL_DENSITY_G_CC_BY_KEYWORD: Dict[str, float] = {}

for display, density in _MATERIAL_DENSITY_G_CC_BY_DISPLAY.items():
    key = normalize_material_key(display)
    if not key or density is None:
        continue
    MATERIAL_DENSITY_G_CC_BY_KEY[key] = density

for key, keywords in MATERIAL_KEYWORDS.items():
    density = MATERIAL_DENSITY_G_CC_BY_KEY.get(key)
    if density is None:
        continue
    for token in keywords:
        if not token:
            continue
        MATERIAL_DENSITY_G_CC_BY_KEYWORD[token] = density

MATERIAL_MAP: Dict[str, Dict[str, float | str]] = {
    # Aluminum alloys → Aluminum cash index
    "6061": {"symbol": "XAL", "basis": "index_usd_per_tonne", "loss_factor": 0.0, "wieland_key": "6061"},
    "6061-T6": {"symbol": "XAL", "basis": "index_usd_per_tonne", "wieland_key": "6061-T6"},
    "7075": {"symbol": "XAL", "basis": "index_usd_per_tonne", "wieland_key": "7075"},

    # Stainless → approximate with Nickel + premium or use vendor CSV override
    "304": {"symbol": "XNI", "basis": "index_usd_per_tonne", "premium_usd_per_kg": 1.20, "wieland_key": "304"},
    "316": {"symbol": "XNI", "basis": "index_usd_per_tonne", "premium_usd_per_kg": 1.80, "wieland_key": "316"},

    # Copper alloys → Copper
    "C110": {"symbol": "XCU", "basis": "index_usd_per_tonne", "wieland_key": "C110"},

    # Precious (if you ever quote)
    "AU-9999": {"symbol": "XAU", "basis": "usd_per_troy_oz"},
}


def canonical_material_key(value: str | None, default: str = DEFAULT_MATERIAL_KEY) -> str:
    """Return a canonical key for the given display label or fallback."""

    if not value:
        return default
    key = normalize_material_key(value)
    return key or default


def is_material_match(display: str, keyword: str) -> bool:
    """Return ``True`` if the keyword matches the display entry."""

    canon = normalize_material_key(display)
    return keyword in MATERIAL_KEYWORDS.get(canon, set())


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
    "canonical_material_key",
    "is_material_match",
    "normalize_material_key",
]
