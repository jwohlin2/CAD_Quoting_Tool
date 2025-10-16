"""Runtime helpers shared by the desktop UI entrypoint."""
from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path
from typing import Iterable, Sequence

from cad_quoter.resources import default_catalog_csv

# Ensure the McMaster stock selector uses the bundled catalog unless callers
# explicitly override it via the environment.  This matches the historical
# Windows installer behaviour where ``catalog.csv`` lived alongside the app.
os.environ.setdefault("CATALOG_CSV_PATH", str(default_catalog_csv()))

# Placeholder for llama-cpp bindings so tests can monkeypatch the loader without
# importing the optional dependency at module import time.
Llama = None  # type: ignore[assignment]

# Note: Avoid importing llama_cpp at module import time so the desktop UI can
# launch in environments without the optional LLM runtime installed. We import
# it lazily inside load_qwen_vl().

# Runtime dependencies required when the desktop UI launches.  These are kept
# here so they can be reused in tests without importing the enormous Tk UI
# module.
REQUIRED_RUNTIME_PACKAGES: dict[str, str] = {
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
}

# Historical Windows installs placed the GGUF files in this directory.  We keep
# it in the search path so the upgraded runtime can still discover the models
# automatically for those environments.
PREFERRED_MODEL_DIRS: tuple[Path, ...] = (
    Path(r"D:\CAD_Quoting_Tool\models"),
)

# Legacy filenames from earlier builds.  The discovery helpers will fall back to
# these names if no explicit paths are provided.
LEGACY_VL_MODEL = Path(r"D:\CAD_Quoting_Tool\models\qwen2.5-vl-7b-instruct-q4_k_m.gguf")
LEGACY_MM_PROJ = Path(r"D:\CAD_Quoting_Tool\models\mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")

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
