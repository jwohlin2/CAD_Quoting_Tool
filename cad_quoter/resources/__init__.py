"""Utilities for accessing packaged resource files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .loading import load_json, load_text

_RESOURCE_ROOT = Path(__file__).resolve().parent


def _build_path(parts: Iterable[str]) -> Path:
    path = _RESOURCE_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Resource not found: {path}")
    return path


def resource_path(*parts: str) -> Path:
    """Return the path to a resource stored alongside the package.

    Parameters
    ----------
    parts:
        One or more path components relative to the resources directory.
    """

    return _build_path(parts)


def default_master_variables_csv() -> Path:
    """Return the bundled master variables CSV file."""

    return resource_path("Master_Variables.csv")


def default_app_settings_json() -> Path:
    """Return the default application settings JSON file."""

    return resource_path("app_settings.json")


def default_catalog_csv() -> Path:
    """Return the bundled McMaster-Carr stock catalog CSV file."""

    return resource_path("catalog.csv")


__all__ = [
    "resource_path",
    "default_master_variables_csv",
    "default_app_settings_json",
    "default_catalog_csv",
    "load_json",
    "load_text",
]
