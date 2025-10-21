"""Shared material density lookups and normalization helpers."""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict

from appkit.data import load_json


LB_PER_IN3_PER_GCC = float(
    # 1 in = 2.54 cm (exact) and 1 lb = 453.59237 g (exact).
    # Therefore 1 g/cm^3 = (2.54^3 / 453.59237) lb/in^3.
    Fraction(2048383, 125000) * Fraction(100000, 45359237)
)


def density_g_cc_to_lb_in3(density_g_cc: float | int | None) -> float | None:
    """Convert a density in g/cc to lb/in^3 while preserving ``None``."""

    if density_g_cc is None:
        return None
    return float(density_g_cc) * LB_PER_IN3_PER_GCC


def normalize_material_key(value: object) -> str:
    """Return a canonical lookup key for material identifiers."""

    cleaned = re.sub(r"[^0-9a-z]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


_MATERIAL_DATA = load_json("materials.json")

_DENSITY_BY_DISPLAY_RAW = _MATERIAL_DATA.get("density_g_cc_by_display", {})

MATERIAL_DENSITY_G_CC_BY_DISPLAY: Dict[str, float] = {}
for display, density in _DENSITY_BY_DISPLAY_RAW.items():
    if density is None:
        continue
    try:
        MATERIAL_DENSITY_G_CC_BY_DISPLAY[str(display)] = float(density)
    except Exception:
        continue

MATERIAL_DENSITY_G_CC_BY_KEY: Dict[str, float] = {}
MATERIAL_DENSITY_G_CC_BY_KEYWORD: Dict[str, float] = {}

_ADDITIONAL_KEYWORDS = _MATERIAL_DATA.get("additional_keywords", {})

for display, density in MATERIAL_DENSITY_G_CC_BY_DISPLAY.items():
    key = normalize_material_key(display)
    if not key:
        continue
    MATERIAL_DENSITY_G_CC_BY_KEY[key] = density

    keywords = {key}
    extras = _ADDITIONAL_KEYWORDS.get(display, [])
    if isinstance(extras, (list, tuple, set)):
        keywords.update(normalize_material_key(term) for term in extras)

    for token in keywords:
        if not token:
            continue
        MATERIAL_DENSITY_G_CC_BY_KEYWORD[token] = density


_ADDITIONAL_DENSITY_ALIASES = {
    # Aluminum family
    "6061": 2.70,
    "6061-t6": 2.70,
    "7075": 2.81,
    "7050": 2.83,
    "2024": 2.78,
    "5052": 2.68,
    "6063": 2.70,
    # Stainless steels
    "17-4": 7.90,
    "17 4": 7.90,
    "15-5": 7.90,
    "15 5": 7.90,
    "303": 8.00,
    "304": 8.00,
    "316": 8.00,
    "410": 7.75,
    "420": 7.75,
    # Tool and alloy steels
    "a2": 7.86,
    "d2": 7.70,
    "o1": 7.81,
    "s7": 7.83,
    "h13": 7.80,
    "4140": 7.85,
    "4340": 7.85,
    "4130": 7.80,
    "1045": 7.87,
    "8620": 7.85,
    # Copper/brass family
    "c110": 8.96,
    "c172": 8.25,
    "c360": 8.50,
    "c260": 8.50,
    # Titanium
    "ti-6al-4v": 4.43,
    "ti64": 4.43,
    "grade 5": 4.43,
    # Plastics and light materials
    "delrin": 1.41,
    "acetal": 1.42,
    "nylon": 1.14,
    "uhmw": 0.94,
    "peek": 1.32,
    "abs": 1.05,
    "ptfe": 2.20,
    "pvc": 1.40,
}

for alias, density in _ADDITIONAL_DENSITY_ALIASES.items():
    key = normalize_material_key(alias)
    if not key or density is None:
        continue
    MATERIAL_DENSITY_G_CC_BY_KEY.setdefault(key, float(density))
    MATERIAL_DENSITY_G_CC_BY_KEYWORD.setdefault(key, float(density))


DEFAULT_MATERIAL_DENSITY_G_CC = MATERIAL_DENSITY_G_CC_BY_KEY.get("aluminum", 7.85)


def density_for_material(material: object | None, default: float | None = None) -> float:
    """Return a density estimate (g/cc) for ``material``."""

    raw = str(material or "").strip()
    if not raw:
        if default is None:
            return DEFAULT_MATERIAL_DENSITY_G_CC
        return float(default)

    normalized = normalize_material_key(raw)
    collapsed = normalized.replace(" ", "")

    for token in (normalized, collapsed):
        density = MATERIAL_DENSITY_G_CC_BY_KEYWORD.get(token)
        if density is not None:
            return float(density)

    for token, density in MATERIAL_DENSITY_G_CC_BY_KEYWORD.items():
        if not token:
            continue
        if token in normalized or token in collapsed:
            return float(density)

    lowered = raw.lower()
    if any(tag in lowered for tag in ("plastic", "uhmw", "delrin", "acetal", "peek", "abs", "nylon")):
        return 1.45
    if any(tag in lowered for tag in ("foam", "poly", "composite")):
        return 1.10
    if any(tag in lowered for tag in ("magnesium", "az31", "az61")):
        return 1.80
    if "graphite" in lowered:
        return 1.85

    if default is not None:
        return float(default)
    return DEFAULT_MATERIAL_DENSITY_G_CC


__all__ = [
    "LB_PER_IN3_PER_GCC",
    "DEFAULT_MATERIAL_DENSITY_G_CC",
    "MATERIAL_DENSITY_G_CC_BY_DISPLAY",
    "MATERIAL_DENSITY_G_CC_BY_KEY",
    "MATERIAL_DENSITY_G_CC_BY_KEYWORD",
    "density_for_material",
    "density_g_cc_to_lb_in3",
    "normalize_material_key",
]
