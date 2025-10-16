"""Domain primitives describing supported material selections."""
from __future__ import annotations

from typing import Dict, Set

from appkit.data import load_json


def normalize_material_key(value: str) -> str:
    """Return a canonical lookup key for material identifiers."""

    import re

    cleaned = re.sub(r"[^0-9a-z]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


_MATERIAL_DATA = load_json("materials.json")

_raw_dropdowns = _MATERIAL_DATA.get("dropdown_options", [])
MATERIAL_DROPDOWN_OPTIONS = [
    str(option).strip()
    for option in _raw_dropdowns
    if str(option or "").strip()
]

default_display_raw = _MATERIAL_DATA.get("default_display")
if default_display_raw:
    DEFAULT_MATERIAL_DISPLAY = str(default_display_raw).strip()
elif MATERIAL_DROPDOWN_OPTIONS:
    DEFAULT_MATERIAL_DISPLAY = MATERIAL_DROPDOWN_OPTIONS[0]
else:
    DEFAULT_MATERIAL_DISPLAY = "Aluminum MIC6"
DEFAULT_MATERIAL_KEY = normalize_material_key(DEFAULT_MATERIAL_DISPLAY)

_MATERIAL_ADDITIONAL_KEYWORDS: Dict[str, Set[str]] = {}
for display, keywords in _MATERIAL_DATA.get("additional_keywords", {}).items():
    if not isinstance(keywords, (list, tuple, set)):
        continue
    extras = {str(term).strip() for term in keywords if str(term or "").strip()}
    if extras:
        _MATERIAL_ADDITIONAL_KEYWORDS[str(display)] = extras

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

_MATERIAL_DISPLAY_OVERRIDES = {
    "aluminum": "Aluminum MIC6",
    "tool_steel_a2": "Tool Steel A2",
    "ss_303": "303 Stainless Steel",
    "mild_steel": "Mild Steel / Low-Carbon Steel",
}

for lookup, display in _MATERIAL_DISPLAY_OVERRIDES.items():
    key = normalize_material_key(lookup)
    if not key:
        continue
    target = MATERIAL_DISPLAY_BY_KEY.get(normalize_material_key(display), display)
    MATERIAL_DISPLAY_BY_KEY[key] = target

MATERIAL_OTHER_KEY = normalize_material_key(MATERIAL_DROPDOWN_OPTIONS[-1])

_MATERIAL_DENSITY_G_CC_BY_DISPLAY: Dict[str, float | None] = {}
for display, density in _MATERIAL_DATA.get("density_g_cc_by_display", {}).items():
    if density is None:
        _MATERIAL_DENSITY_G_CC_BY_DISPLAY[str(display)] = None
        continue
    try:
        _MATERIAL_DENSITY_G_CC_BY_DISPLAY[str(display)] = float(density)
    except Exception:
        continue

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

MATERIAL_MAP: Dict[str, Dict[str, float | str]] = {}
for key, meta in _MATERIAL_DATA.get("material_map", {}).items():
    if not isinstance(meta, dict):
        continue
    cleaned: Dict[str, float | str] = {}
    for meta_key, meta_value in meta.items():
        if isinstance(meta_value, (int, float)):
            cleaned[str(meta_key)] = float(meta_value)
        elif meta_value is None:
            continue
        else:
            cleaned[str(meta_key)] = str(meta_value)
    MATERIAL_MAP[str(key)] = cleaned


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
