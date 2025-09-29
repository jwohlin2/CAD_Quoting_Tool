"""Utilities and data models shared across CAD Quoter components."""

from .config import APP_ENV, AppEnvironment, describe_runtime_environment
from .domain import (
    QuoteState,
    _as_float_or_none,
    _ensure_scrap_pct,
    _normalize_lookup_key,
    build_suggest_payload,
)

__all__ = [
    "APP_ENV",
    "AppEnvironment",
    "QuoteState",
    "_as_float_or_none",
    "_ensure_scrap_pct",
    "_normalize_lookup_key",
    "build_suggest_payload",
    "describe_runtime_environment",
]
