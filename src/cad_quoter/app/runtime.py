from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import logging


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration derived from environment variables for CLI runs."""

    # Feature flags
    llm_enabled: bool = False
    pdf_enabled: bool = False
    pandas_enabled: bool = False

    # Logging
    log_level: str = "INFO"

    # Execution
    workdir: Path = Path.cwd()

    # App-specific tunables (extend as needed)
    default_rate_file: Path | None = None
    default_speeds_feeds_csv: Path | None = None


def _to_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip() in {"1", "true", "True", "YES", "yes", "on", "On"}


def build_config(env: dict[str, str] | None = None, *, workdir: Path | None = None) -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` from ``env`` and optionally ``workdir``."""

    e = env or os.environ
    cfg = RuntimeConfig(
        llm_enabled=_to_bool(e.get("APP_LLM"), False),
        pdf_enabled=_to_bool(e.get("APP_PDF"), False),
        pandas_enabled=_to_bool(e.get("APP_PANDAS"), False),
        log_level=e.get("APP_LOG_LEVEL", "INFO"),
        workdir=workdir or Path(e.get("APP_WORKDIR") or Path.cwd()),
        default_rate_file=Path(e["APP_RATE_FILE"]).resolve() if e.get("APP_RATE_FILE") else None,
        default_speeds_feeds_csv=Path(e["APP_SF_CSV"]).resolve() if e.get("APP_SF_CSV") else None,
    )
    _configure_logging(cfg.log_level)
    return cfg


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s %(name)s: %(message)s")
