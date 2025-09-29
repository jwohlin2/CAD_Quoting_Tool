"""Runtime configuration helpers for the CAD Quoter application."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_DEFAULT_DEBUG_DIR = Path(__file__).resolve().parent / ".." / "llm_debug"


@dataclass(frozen=True)
class AppEnvironment:
    """Immutable configuration loaded from environment variables."""

    llm_debug_enabled: bool
    llm_debug_dir: Path

    @classmethod
    def from_env(cls) -> "AppEnvironment":
        """Initialise configuration from process environment variables."""

        debug_enabled_raw = os.getenv("LLM_DEBUG", "1")
        try:
            debug_enabled = bool(int(debug_enabled_raw))
        except ValueError:
            debug_enabled = debug_enabled_raw.lower() in {"true", "t", "yes", "y"}

        debug_dir_raw = os.getenv("LLM_DEBUG_DIR")
        if debug_dir_raw:
            debug_dir = Path(debug_dir_raw).expanduser()
        else:
            debug_dir = _DEFAULT_DEBUG_DIR
        debug_dir.mkdir(parents=True, exist_ok=True)

        return cls(llm_debug_enabled=debug_enabled, llm_debug_dir=debug_dir)


def describe_runtime_environment(env: AppEnvironment) -> dict[str, str]:
    """Return a serialisable snapshot of key runtime configuration."""

    summary: dict[str, str] = {
        "llm_debug_enabled": "1" if env.llm_debug_enabled else "0",
        "llm_debug_dir": str(env.llm_debug_dir),
    }

    for key in ("QWEN_GGUF_PATH", "ODA_CONVERTER_EXE", "DWG2DXF_EXE", "METALS_API_KEY"):
        value = os.getenv(key)
        if not value:
            summary[key.lower()] = ""
            continue
        if key.endswith("_KEY"):
            summary[key.lower()] = "<redacted>"
        else:
            summary[key.lower()] = value
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser used by the entry point."""

    parser = argparse.ArgumentParser(description="Launch the CAD Quoter UI")
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print a JSON summary of runtime configuration and exit.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Initialise services without starting the Tkinter event loop.",
    )
    return parser


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the application."""

    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args


def format_environment_summary(env: AppEnvironment) -> str:
    """Return a human-readable JSON dump of :func:`describe_runtime_environment`."""

    summary = describe_runtime_environment(env)
    return json.dumps(summary, indent=2, sort_keys=True)


def ensure_debug_directory(env: AppEnvironment) -> None:
    """Guarantee that the debug directory exists on disk."""

    env.llm_debug_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    "AppEnvironment",
    "describe_runtime_environment",
    "build_argument_parser",
    "parse_cli_args",
    "format_environment_summary",
    "ensure_debug_directory",
]
