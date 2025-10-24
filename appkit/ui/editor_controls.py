"""Compatibility shim that re-exports editor control helpers."""

from __future__ import annotations

import cad_quoter_pkg  # noqa: F401  # ensures ``cad_quoter`` is importable without installation

from cad_quoter.ui.editor_controls import (
    EditorControlSpec,
    coerce_checkbox_state,
    derive_editor_control_spec,
)

__all__ = [
    "EditorControlSpec",
    "coerce_checkbox_state",
    "derive_editor_control_spec",
]
