"""Unified command-line runner and runtime utilities for the desktop UI."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from collections.abc import Mapping as _MappingABC
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from cad_quoter.config import (
    AppEnvironment,
    configure_logging,
    describe_runtime_environment,
    logger,
)
from cad_quoter.resources import default_catalog_csv
from cad_quoter.utils import jdump

# ---------------------------------------------------------------------------
# Environment initialisation
# ---------------------------------------------------------------------------

# Ensure the McMaster stock selector uses the bundled catalog unless callers
# explicitly override it via the environment. This matches the historical
# Windows installer behaviour where ``catalog.csv`` lived alongside the app.
os.environ.setdefault("CATALOG_CSV_PATH", str(default_catalog_csv()))

# Placeholder for llama-cpp bindings so tests can monkeypatch the loader without
# importing the optional dependency at module import time.
Llama = None  # type: ignore[assignment]

# Runtime dependencies required when the desktop UI launches. These are kept
# here so they can be reused in tests without importing the enormous Tk UI
# module.
REQUIRED_RUNTIME_PACKAGES: dict[str, str] = {
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
}

# Historical Windows installs placed the GGUF files in this directory. We keep
# it in the search path so the upgraded runtime can still discover the models
# automatically for those environments.
PREFERRED_MODEL_DIRS: tuple[Path, ...] = (
    Path(r"D:\\CAD_Quoting_Tool\\models"),
)

# Legacy filenames from earlier builds. The discovery helpers will fall back to
# these names if no explicit paths are provided.
LEGACY_VL_MODEL = Path(r"D:\\CAD_Quoting_Tool\\models\\qwen2.5-vl-7b-instruct-q4_k_m.gguf")
LEGACY_MM_PROJ = Path(r"D:\\CAD_Quoting_Tool\\models\\mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")

DEFAULT_VL_MODEL_NAMES = (
    LEGACY_VL_MODEL.name,
    "qwen2.5-vl-7b-instruct-q4_0.gguf",
    "qwen2.5-vl-7b-instruct-q4_k_s.gguf",
    "qwen2-vl-7b-instruct-q4_0.gguf",
)

DEFAULT_MM_PROJ_NAMES = (
    LEGACY_MM_PROJ.name,
    "mmproj-Qwen2.5-VL-3B-Instruct-Q4_0.gguf",
    "mmproj-Qwen2-VL-7B-Instruct-Q4_0.gguf",
)


# ---------------------------------------------------------------------------
# Command-line interface helpers
# ---------------------------------------------------------------------------


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

        with open("debug.log", "w", encoding="ascii", errors="replace") as _dbg:
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


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def ensure_runtime_dependencies() -> None:
    """Raise :class:`ImportError` if optional-but-required packages are missing."""

    missing = [
        (module, dist_name)
        for module, dist_name in REQUIRED_RUNTIME_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return

    missing_dists = ", ".join(dist_name for _, dist_name in missing)
    raise ImportError(
        "Required runtime dependencies are missing. Install them with "
        "`pip install -r requirements.txt` before launching appV5.py "
        f"(missing: {missing_dists})."
    )


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for raw in paths:
        try:
            candidate = raw.expanduser()
        except Exception:
            continue
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _collect_model_dirs(*paths: str | Path | None) -> list[Path]:
    dirs: list[Path] = []
    for value in paths:
        if not value:
            continue
        try:
            candidate = Path(value).expanduser()
        except Exception:
            continue
        if candidate.is_file():
            dirs.append(candidate.parent)
        elif candidate.is_dir():
            dirs.append(candidate)
        else:
            dirs.append(candidate.parent)
    for env_var in ("QWEN_MODELS_DIR", "QWEN_VL_MODELS_DIR"):
        env_value = os.environ.get(env_var)
        if env_value:
            try:
                dirs.append(Path(env_value).expanduser())
            except Exception:
                continue
    dirs.extend(PREFERRED_MODEL_DIRS)
    try:
        dirs.append(Path(__file__).resolve().parent / "models")
    except Exception:
        pass
    dirs.append(Path.cwd() / "models")
    return [d for d in _dedupe_paths(dirs) if str(d)]


def _find_weight_file(
    names: Sequence[str],
    directories: Sequence[Path],
    glob: str,
) -> str:
    for directory in directories:
        try:
            if not directory.exists():
                continue
        except Exception:
            continue
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
        try:
            matches = list(directory.glob(glob))
        except Exception:
            matches = []
        if matches:
            try:
                matches.sort(key=lambda p: (-p.stat().st_size, p.name))
            except Exception:
                matches.sort()
            return str(matches[0])
    return ""


def discover_qwen_vl_assets(
    *,
    model_path: str | None = None,
    mmproj_path: str | None = None,
) -> tuple[str, str]:
    """Locate Qwen vision model + projector weights on disk."""

    model_candidates = [
        model_path,
        os.environ.get("QWEN_VL_GGUF_PATH"),
        os.environ.get("QWEN_GGUF_PATH"),
        str(LEGACY_VL_MODEL),
    ]
    mmproj_candidates = [
        mmproj_path,
        os.environ.get("QWEN_VL_MMPROJ_PATH"),
        os.environ.get("QWEN_MMPROJ_PATH"),
        str(LEGACY_MM_PROJ),
    ]

    def _first_existing(paths: Sequence[str | None]) -> str:
        for value in paths:
            if not value:
                continue
            try:
                candidate = Path(value).expanduser()
            except Exception:
                continue
            if candidate.is_file():
                return str(candidate)
        return ""

    model_file = _first_existing(model_candidates)
    mmproj_file = _first_existing(mmproj_candidates)

    search_dirs = _collect_model_dirs(*(model_candidates + mmproj_candidates))

    if not model_file:
        model_file = _find_weight_file(DEFAULT_VL_MODEL_NAMES, search_dirs, "*vl*.gguf")
    if not mmproj_file:
        mmproj_file = _find_weight_file(DEFAULT_MM_PROJ_NAMES, search_dirs, "*mmproj*.gguf")

    if not model_file or not mmproj_file:
        searched = ", ".join(str(d) for d in search_dirs if d)
        raise RuntimeError(
            "Vision LLM weights not found. Set QWEN_VL_GGUF_PATH and "
            "QWEN_VL_MMPROJ_PATH (or place matching *.gguf files in one of: "
            f"{searched or 'the known model directories'})."
        )

    return model_file, mmproj_file


def load_qwen_vl(
    n_ctx: int = 8192,
    n_gpu_layers: int = 20,
    n_threads: int | None = None,
    *,
    model_path: str | None = None,
    mmproj_path: str | None = None,
):
    """Load Qwen2.5-VL with vision projector configured for llama.cpp."""

    # Lazy import so environments without llama-cpp-python can still launch the UI
    global Llama
    llama_cls = Llama
    if llama_cls is None:
        try:
            from llama_cpp import Llama as _ImportedLlama  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Vision LLM support requires llama-cpp-python. Install it via "
                "`pip install -r requirements.txt` (or `pip install llama-cpp-python`) "
                "before enabling LLM features."
            ) from exc
        llama_cls = _ImportedLlama
        Llama = llama_cls

    if n_threads is None:
        cpu_count = os.cpu_count() or 8
        n_threads = max(4, cpu_count // 2)

    model_file, mmproj_file = discover_qwen_vl_assets(
        model_path=model_path,
        mmproj_path=mmproj_path,
    )

    attempted_chat_formats: tuple[str, ...] = ("qwen2.5_vl", "qwen")
    last_invalid_chat_error: Exception | None = None

    for chat_format in attempted_chat_formats:
        try:
            llm = llama_cls(
                model_path=model_file,
                mmproj_path=mmproj_file,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                verbose=False,
            )
            _ = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Return JSON {\"ok\":true}."},
                    {"role": "user", "content": "ping"},
                ],
                max_tokens=16,
                temperature=0,
            )
            return llm
        except Exception as exc:
            message = str(exc)
            if "Invalid chat handler" in message:
                last_invalid_chat_error = exc
                continue
            if n_ctx > 4096:
                gc.collect()
                return load_qwen_vl(
                    n_ctx=4096,
                    n_gpu_layers=0,
                    n_threads=n_threads,
                    model_path=model_file,
                    mmproj_path=mmproj_file,
                )
            raise

    assert last_invalid_chat_error is not None  # for type checkers
    raise RuntimeError(
        "Vision LLM unavailable: llama.cpp build is missing Qwen chat handlers. "
        "Upgrade llama.cpp to a version that includes qwen2.5_vl support."
    ) from last_invalid_chat_error


def _pick_best_gguf(paths: Iterable[str]) -> str:
    candidates = [Path(p) for p in paths]
    filtered = [p for p in candidates if p.is_file() and p.suffix.lower() == ".gguf"]
    if not filtered:
        return ""
    preferred = [
        p
        for p in filtered
        if "qwen" in p.name.lower() and "instr" in p.name.lower()
    ]
    pool = preferred or filtered
    try:
        best = max(pool, key=lambda p: p.stat().st_size)
    except Exception:
        best = pool[0]
    return str(best)


def find_default_qwen_model() -> str:
    """Best-effort discovery for the primary GGUF model path."""

    envp = os.environ.get("QWEN_GGUF_PATH", "")
    if envp:
        candidate = Path(envp).expanduser()
        if candidate.is_file():
            return str(candidate)

    for directory in PREFERRED_MODEL_DIRS:
        if not directory.is_dir():
            continue
        choice = _pick_best_gguf(str(p) for p in directory.glob("*.gguf"))
        if choice:
            return choice

    try:
        script_dir = Path(__file__).resolve().parent
        choice = _pick_best_gguf(str(p) for p in (script_dir / "models").glob("*.gguf"))
        if choice:
            return choice
    except Exception:
        pass

    cwd_choice = _pick_best_gguf(str(p) for p in (Path.cwd() / "models").glob("*.gguf"))
    if cwd_choice:
        return cwd_choice

    return ""


__all__ = [
    "build_arg_parser",
    "main",
    "DEFAULT_MM_PROJ_NAMES",
    "DEFAULT_VL_MODEL_NAMES",
    "LEGACY_MM_PROJ",
    "LEGACY_VL_MODEL",
    "PREFERRED_MODEL_DIRS",
    "REQUIRED_RUNTIME_PACKAGES",
    "discover_qwen_vl_assets",
    "ensure_runtime_dependencies",
    "find_default_qwen_model",
    "load_qwen_vl",
]


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
