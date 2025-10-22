"""Deployment utilities for the CAD Quoting Tool.

This package consolidates the ad-hoc shell helpers that previously lived in
various directories (for example the deprecated ``git-auto-pull`` submodule).
The new interface exposes a small Python-based CLI so recurring operational
flows can be scripted and documented in a single place.
"""

from __future__ import annotations

__all__ = ["get_repo_root"]

from pathlib import Path


def get_repo_root() -> Path:
    """Return the absolute path to the repository root.

    The helper resolves the directory relative to this module so callers do not
    need to rely on their current working directory when invoking the CLI from
    automation.
    """

    return Path(__file__).resolve().parent.parent
