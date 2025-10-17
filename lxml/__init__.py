"""Minimal stub package for :mod:`lxml` used in tests.

The production application depends on the third-party lxml package for
XML parsing.  Installing the compiled dependency inside the execution
environment is outside the scope of the unit tests, so we provide a tiny
shim that exposes the modules accessed by the runtime dependency check.
"""

from __future__ import annotations


__all__: list[str] = []

