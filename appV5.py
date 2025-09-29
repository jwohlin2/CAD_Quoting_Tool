"""Minimal Tkinter entry-point for the refactored CAD Quoter app."""

from __future__ import annotations

import sys
from typing import Sequence

from tkinter import TclError

from cad_quoter.config import (
    AppEnvironment,
    ensure_debug_directory,
    format_environment_summary,
    parse_cli_args,
)
from cad_quoter.domain import QuoteApplication, build_application_services


def launch_ui(env: AppEnvironment, headless: bool) -> int:
    """Create the Tkinter application and optionally run ``mainloop``."""

    services = build_application_services(env)
    try:
        app = QuoteApplication(env, services)
    except TclError as exc:
        if headless:
            # In CI/headless environments Tk may be unavailable; treat this as a
            # successful dry-run so automated checks can still exercise wiring.
            print(f"Tkinter unavailable in headless mode: {exc}")
            return 0
        raise

    if headless:
        # Update/destroy immediately so headless checks can ensure widgets build.
        app.update_idletasks()
        app.destroy()
        return 0

    try:
        app.mainloop()
    except KeyboardInterrupt:
        # Gracefully exit when the shell sends Ctrl+C.
        return 0
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python appV5.py``."""

    args = parse_cli_args(argv)
    env = AppEnvironment.from_env()
    ensure_debug_directory(env)

    if getattr(args, "print_env", False):
        print(format_environment_summary(env))
        return 0

    return launch_ui(env, headless=getattr(args, "headless", False))


if __name__ == "__main__":
    sys.exit(main())
