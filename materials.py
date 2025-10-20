"""Compatibility shim for legacy imports of :mod:`cad_quoter.pricing.materials`."""

from __future__ import annotations

from cad_quoter.pricing import materials as _shared_materials

__all__ = [name for name in dir(_shared_materials) if not name.startswith("__")]
for name in __all__:
    globals()[name] = getattr(_shared_materials, name)
