"""Command-line entry point for launching the desktop quoting UI."""

from __future__ import annotations

import argparse
import os
import json
import sys
from collections.abc import Mapping as _MappingABC
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from cad_quoter.config import AppEnvironment, configure_logging, describe_runtime_environment, logger
from cad_quoter.utils import jdump


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD Quoting Tool UI")
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print a JSON dump of relevant environment configuration and exit.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Initialise subsystems but do not launch the Tkinter GUI.",
    )
    parser.add_argument(
        "--debug-removal",
        action="store_true",
        help="Force-enable material removal debug output (shows Material Removal Debug table).",
    )
    parser.add_argument(
        "--geo-json",
        type=str,
        default=None,
        help="Bypass CAD and load a prebuilt geometry JSON for debugging.",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    app_cls: type | None = None,
    pricing_engine_cls: type | None = None,
    pricing_registry_factory: Callable[[], Any] | None = None,
    app_env: AppEnvironment | None = None,
    env_setter: Callable[[AppEnvironment], None] | None = None,
) -> int:
    """Run the Tkinter application entry point with command-line options."""

    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if pricing_registry_factory is None or pricing_engine_cls is None or app_cls is None or app_env is None:
        from cad_quoter.pricing import PricingEngine, create_default_registry
        import appV5 as app_module

        if pricing_registry_factory is None:
            pricing_registry_factory = create_default_registry
        if pricing_engine_cls is None:
            pricing_engine_cls = PricingEngine
        if app_cls is None:
            app_cls = app_module.App
        if app_env is None:
            app_env = app_module.APP_ENV
        if env_setter is None:
            env_setter = lambda env: setattr(app_module, "APP_ENV", env)
    if pricing_registry_factory is None or pricing_engine_cls is None or app_cls is None or app_env is None:
        raise RuntimeError("CLI dependencies not provided and could not be resolved")

    # Reset debug.log at start with ASCII to avoid garbled content from prior runs
    try:
        import datetime as _dt

        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_log_path = os.path.join(debug_dir, "debug.log")

        with open(debug_log_path, "w", encoding="ascii", errors="replace") as _dbg:
            _dbg.write(f"[app] start { _dt.datetime.now().isoformat() }\n")
    except Exception:
        pass

    # CLI override: force-enable removal debug output
    if getattr(args, "debug_removal", False):
        updated_env = replace(app_env, llm_debug_enabled=True)
        if env_setter is not None:
            env_setter(updated_env)
        app_env = updated_env

    if args.print_env:
        logger.info("Runtime environment:\n%s", jdump(describe_runtime_environment(), default=None))
        return 0

    geo_json_payload: Mapping[str, Any] | None = None
    geo_json_path = getattr(args, "geo_json", None)
    if geo_json_path:
        path_obj = Path(str(geo_json_path))
        try:
            with path_obj.open("r", encoding="utf-8") as handle:
                raw_payload = json.load(handle)
        except FileNotFoundError:
            logger.error("Geometry JSON not found: %s", path_obj)
            return 1
        except json.JSONDecodeError as exc:
            logger.error("Geometry JSON is invalid (%s): %s", path_obj, exc)
            return 1
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to read geometry JSON %s: %s", path_obj, exc)
            return 1
        if isinstance(raw_payload, _MappingABC):
            geo_json_payload = raw_payload  # type: ignore[assignment]
        else:
            logger.error(
                "Geometry JSON must contain an object at the top level: %s",
                path_obj,
            )
            return 1

    if args.no_gui:
        if geo_json_payload is not None:
            logger.warning("--geo-json is ignored when --no-gui is supplied.")
        return 0

    pricing_registry = pricing_registry_factory()
    pricing_engine = pricing_engine_cls(pricing_registry)

    app = None
    try:
        app = app_cls(pricing_engine)
    except RuntimeError as exc:  # pragma: no cover - headless guard
        logger.error("Unable to start the GUI: %s", exc)
        return 1

    if geo_json_payload is not None:
        try:
            app.apply_geometry_payload(geo_json_payload, source=geo_json_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to apply geometry JSON %s: %s", geo_json_path, exc)
            return 1

    app.mainloop()

    return 0


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.exit(main())
    except Exception as exc:
        try:
            print(jdump({"ok": False, "error": str(exc)}))
        except Exception:
            pass
        sys.exit(1)
