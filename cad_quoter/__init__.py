"""CAD Quoter shared utilities."""

from .config import APP_ENV, AppEnvironment, describe_runtime_environment
from .domain import (
    QuoteState,
    build_suggest_payload,
    ensure_scrap_pct,
    match_items_contains,
    normalize_lookup_key,
)

__all__ = [
    "APP_ENV",
    "AppEnvironment",
    "QuoteState",
    "build_suggest_payload",
    "describe_runtime_environment",
    "ensure_scrap_pct",
    "match_items_contains",
    "normalize_lookup_key",
]
