"""Configuration helpers for the CAD Quoter application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppEnvironment:
    """Runtime configuration extracted from environment variables."""

    llm_debug_enabled: bool = False
    llm_debug_dir: Path = field(default_factory=Path)

    @classmethod
    def from_env(cls) -> "AppEnvironment":
        debug_enabled = bool(int(os.getenv("LLM_DEBUG", "1")))
        debug_dir_raw = os.getenv("LLM_DEBUG_DIR")
        if debug_dir_raw:
            debug_dir = Path(debug_dir_raw)
        else:
            debug_dir = Path(__file__).resolve().parent.parent / "llm_debug"
        debug_dir.mkdir(exist_ok=True)
        return cls(llm_debug_enabled=debug_enabled, llm_debug_dir=debug_dir)


APP_ENV = AppEnvironment.from_env()


def describe_runtime_environment() -> dict[str, str]:
    """Return a redacted snapshot of runtime configuration for auditors."""

    info = {"llm_debug_enabled": str(APP_ENV.llm_debug_enabled)}
    info["llm_debug_dir"] = str(APP_ENV.llm_debug_dir)
    for key in ("QWEN_GGUF_PATH", "ODA_CONVERTER_EXE", "DWG2DXF_EXE", "METALS_API_KEY"):
        value = os.getenv(key)
        if not value:
            info[key.lower()] = ""
            continue
        if key.endswith("_KEY"):
            info[key.lower()] = "<redacted>"
        else:
            info[key.lower()] = value
    return info


__all__ = ["APP_ENV", "AppEnvironment", "describe_runtime_environment"]
