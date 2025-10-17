"""Minimal stub of :mod:`bs4` for tests.

This project only requires the module to exist so the runtime dependency
checker passes inside the lightweight CI environment.  The full
BeautifulSoup implementation is not needed for the unit tests, but a
placeholder ``BeautifulSoup`` class helps satisfy imports from legacy
modules that expect the real package.
"""

from __future__ import annotations


class BeautifulSoup:  # pragma: no cover - stub implementation
    """Tiny stand-in that mimics the BeautifulSoup constructor signature."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - mimic API
        self.args = args
        self.kwargs = kwargs


__all__ = ["BeautifulSoup"]

