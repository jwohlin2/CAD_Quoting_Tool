"""Runtime shim that re-exports the packaged rate defaults helper."""
from __future__ import annotations

from cad_quoter_pkg.src.cad_quoter.pricing.rate_defaults import (
    DEFAULT_ROLE_MODE_FALLBACKS,
    fallback_keys_for_mode,
    fallback_rate_for_bucket,
    fallback_rate_for_role,
)

__all__ = [
    "DEFAULT_ROLE_MODE_FALLBACKS",
    "fallback_keys_for_mode",
    "fallback_rate_for_bucket",
    "fallback_rate_for_role",
]
