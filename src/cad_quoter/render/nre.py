"""Stub non-recurring engineering section renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from . import RenderState


def render_nre(state: "RenderState") -> list[str]:
    """Return the legacy NRE section lines."""

    return []
