"""CAD Quoter core package.

This lightweight package exposes configuration, geometry, pricing and
language-model helpers used by the Tkinter front-end in ``appV5.py``.
The goal of the package is to centralise implementation details away
from the legacy script so the entry point can focus on wiring
components together.
"""

from __future__ import annotations

from .config import AppEnvironment, describe_runtime_environment, parse_cli_args
from .domain import QuoteApplication, ApplicationServices, build_application_services

__all__ = [
    "AppEnvironment",
    "describe_runtime_environment",
    "parse_cli_args",
    "QuoteApplication",
    "ApplicationServices",
    "build_application_services",
]
