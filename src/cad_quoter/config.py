"""Configuration helpers for the CAD quoter application."""
from __future__ import annotations

import json
import os
import copy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Mapping

from cad_quoter.rates import (
    ensure_two_bucket_defaults,
    migrate_flat_to_two_bucket,
    shared_two_bucket_rate_defaults,
)

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_VERSION = 1
APP_SETTINGS_ENV_VAR = "CAD_QUOTER_APP_SETTINGS"
_APP_SETTINGS_CACHE: dict[str, Any] | None = None


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Return a boolean from the environment with tolerant parsing."""

    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if not normalized:
        return default

    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    if normalized in truthy:
        return True
    if normalized in falsy:
        return False

    try:
        return bool(int(normalized))
    except Exception:
        return default


@dataclass(frozen=True)
class AppEnvironment:
    """Runtime configuration extracted from environment variables."""

    llm_debug_enabled: bool = False
    llm_debug_dir: Path = field(default_factory=Path)

    @classmethod
    def from_env(cls) -> "AppEnvironment":
        debug_enabled = _env_flag("LLM_DEBUG", default=False)
        debug_dir_raw = os.getenv("LLM_DEBUG_DIR")
        if debug_dir_raw:
            debug_dir = Path(debug_dir_raw)
        else:
            debug_dir = Path(__file__).resolve().parent.parent / "llm_debug"
        debug_dir.mkdir(exist_ok=True, parents=True)
        return cls(llm_debug_enabled=debug_enabled, llm_debug_dir=debug_dir)


def describe_runtime_environment() -> dict[str, str]:
    """Return a redacted snapshot of runtime configuration for auditors."""

    env = AppEnvironment.from_env()
    info: dict[str, str] = {
        "llm_debug_enabled": str(env.llm_debug_enabled),
        "llm_debug_dir": str(env.llm_debug_dir),
    }
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


class ConfigError(RuntimeError):
    """Raised when configuration data cannot be loaded or validated."""

def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Malformed JSON in {path.name}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration root must be an object in {path.name}")

    return dict(raw)


def _merge_mappings(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], Mapping) and isinstance(value, Mapping):
            base[key] = _merge_mappings(dict(base[key]), value)
        else:
            base[key] = value  # type: ignore[assignment]
    return base


def _load_app_settings_raw() -> dict[str, Any]:
    base_path = RESOURCE_DIR / "app_settings.json"
    base = _load_json_mapping(base_path)

    override_raw = os.getenv(APP_SETTINGS_ENV_VAR)
    if override_raw:
        override_path = Path(override_raw).expanduser()
        if override_path.exists():
            try:
                override = _load_json_mapping(override_path)
            except ConfigError as exc:
                raise ConfigError(f"Failed to load override settings: {exc}") from exc
            base = _merge_mappings(base, override)
        else:
            logger.warning("Override settings path does not exist: %s", override_path)

    return base


def load_app_settings(*, reload: bool = False) -> dict[str, Any]:
    """Return the merged application settings, applying optional overrides."""

    global _APP_SETTINGS_CACHE
    if reload or _APP_SETTINGS_CACHE is None:
        _APP_SETTINGS_CACHE = _load_app_settings_raw()

    return copy.deepcopy(_APP_SETTINGS_CACHE)


def load_named_config(name: str, version: int = DEFAULT_VERSION) -> dict[str, Any]:
    """Load a named configuration bundle from the merged application settings."""

    if version != DEFAULT_VERSION:
        raise ConfigError(
            f"Unsupported version requested: {version!r}; expected {DEFAULT_VERSION}"
        )

    settings = load_app_settings()
    pricing_defaults = settings.get("pricing_defaults")
    if not isinstance(pricing_defaults, Mapping):
        raise ConfigError("'pricing_defaults' section missing from app settings")

    section = pricing_defaults.get(name)
    if not isinstance(section, Mapping):
        raise ConfigError(f"Missing configuration section in app settings: {name}")

    return dict(section)
def _ensure_two_bucket_rates(raw: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    from cad_quoter.domain_models.values import to_float

    labor_raw = raw.get("labor") if isinstance(raw, Mapping) else None
    machine_raw = raw.get("machine") if isinstance(raw, Mapping) else None

    if isinstance(labor_raw, Mapping) and isinstance(machine_raw, Mapping):
        labor: dict[str, float] = {}
        for key, value in labor_raw.items():
            numeric = to_float(value)
            if numeric is None:
                continue
            labor[str(key)] = numeric

        machine: dict[str, float] = {}
        for key, value in machine_raw.items():
            numeric = to_float(value)
            if numeric is None:
                continue
            machine[str(key)] = numeric

        return ensure_two_bucket_defaults({"labor": labor, "machine": machine})

    flat: dict[str, float] = {}
    for key, value in raw.items():
        numeric = to_float(value)
        if numeric is None:
            continue
        flat[str(key)] = numeric

    return ensure_two_bucket_defaults(migrate_flat_to_two_bucket(flat))


def load_default_rates() -> dict[str, dict[str, float]]:
    """Return the canonical two-bucket shop rate configuration."""

    return shared_two_bucket_rate_defaults()


def load_default_params() -> dict[str, Any]:
    """Return the default quoting parameter configuration."""

    return load_named_config("params", DEFAULT_VERSION)


def save_named_config(
    data: Mapping[str, Any],
    path: str | Path,
    *,
    version: int = DEFAULT_VERSION,
    indent: int = 2,
) -> Path:
    """Persist configuration data with version metadata.

    The output format is inferred from the destination suffix. JSON is used when
    the suffix is unrecognised.
    """

    destination = Path(path)
    payload: dict[str, Any] = {"version": version, "data": dict(data)}

    suffix = destination.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ConfigError(
                "PyYAML is required to save YAML configuration files"
            ) from exc
        text = yaml.safe_dump(payload, sort_keys=False)  # type: ignore[name-defined]
    else:
        text = json.dumps(payload, indent=indent, sort_keys=True)

    destination.write_text(text, encoding="utf-8")
    return destination


LOGGER_NAME = "cad_quoter"


def get_logger(*names: str) -> logging.Logger:
    """Return a logger under the shared CAD Quoter namespace."""

    if not names:
        return logging.getLogger(LOGGER_NAME)
    qualified = ".".join((LOGGER_NAME, *names))
    return logging.getLogger(qualified)


logger = get_logger()


def append_debug_log(*lines: str) -> None:
    """Append diagnostic lines to ``debug.log`` using ASCII encoding."""

    if not lines:
        return

    try:
        with open("debug.log", "a", encoding="ascii", errors="replace") as log:
            for line in lines:
                text = str(line)
                log.write(text)
                if not text.endswith("\n"):
                    log.write("\n")
    except Exception:
        logger.debug("Failed to append to debug.log", exc_info=True)


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """Initialise a basic logging configuration if none is present."""

    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


__all__ = [
    "APP_SETTINGS_ENV_VAR",
    "AppEnvironment",
    "ConfigError",
    "DEFAULT_VERSION",
    "LOGGER_NAME",
    "RESOURCE_DIR",
    "append_debug_log",
    "configure_logging",
    "describe_runtime_environment",
    "get_logger",
    "load_app_settings",
    "load_named_config",
    "load_default_rates",
    "load_default_params",
    "logger",
    "save_named_config",
]
