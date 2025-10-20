"""Shared helpers for normalising hardware pass labels."""
from __future__ import annotations

from typing import Any

HARDWARE_PASS_LABEL = "Hardware"
LEGACY_HARDWARE_PASS_LABEL = "Hardware / BOM"
_HARDWARE_LABEL_ALIASES = {
    HARDWARE_PASS_LABEL.lower(),
    LEGACY_HARDWARE_PASS_LABEL.lower(),
    "hardware/bom",
    "hardware bom",
}


def canonical_pass_label(label: Any) -> str:
    """Return the canonical pass label for ``label``."""

    name = str(label or "").strip()
    if name.lower() in _HARDWARE_LABEL_ALIASES:
        return HARDWARE_PASS_LABEL
    return name


def _canonical_pass_label(label: Any) -> str:
    """Backwards-compatible alias for :func:`canonical_pass_label`."""

    return canonical_pass_label(label)


__all__ = [
    "HARDWARE_PASS_LABEL",
    "LEGACY_HARDWARE_PASS_LABEL",
    "_HARDWARE_LABEL_ALIASES",
    "canonical_pass_label",
    "_canonical_pass_label",
]
