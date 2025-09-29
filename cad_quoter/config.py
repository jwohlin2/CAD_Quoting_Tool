"""Configuration helpers for the CAD quoter application."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_VERSION = 1


class ConfigError(RuntimeError):
    """Raised when configuration data cannot be loaded or validated."""


def _resource_candidates(name: str, version: int) -> list[Path]:
    base = f"{name}_v{version}"
    return [
        RESOURCE_DIR / f"{base}.json",
        RESOURCE_DIR / f"{base}.yaml",
        RESOURCE_DIR / f"{base}.yml",
    ]


def _load_from_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        try:
            raw: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Malformed JSON in {path.name}: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ConfigError(
                "PyYAML is required to load YAML configuration files"
            ) from exc
        try:
            raw = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigError(f"Malformed YAML in {path.name}: {exc}") from exc
    else:
        raise ConfigError(f"Unsupported configuration format: {path.suffix}")

    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration root must be an object in {path.name}")

    version = raw.get("version")
    if version != DEFAULT_VERSION:
        raise ConfigError(
            f"Unsupported version in {path.name}: {version!r}; expected {DEFAULT_VERSION}"
        )

    data = raw.get("data")
    if not isinstance(data, Mapping):
        raise ConfigError(f"Configuration 'data' must be an object in {path.name}")

    return dict(data)


def load_named_config(name: str, version: int = DEFAULT_VERSION) -> dict[str, Any]:
    """Load a named configuration bundle from the packaged resources."""

    for candidate in _resource_candidates(name, version):
        if candidate.exists():
            return _load_from_path(candidate)
    raise ConfigError(
        f"No configuration resource found for '{name}' version {version} in {RESOURCE_DIR}"
    )


def load_default_rates() -> dict[str, float]:
    """Return the default shop rate configuration."""

    data = load_named_config("rates", DEFAULT_VERSION)
    return {str(k): float(v) for k, v in data.items()}


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


__all__ = [
    "ConfigError",
    "DEFAULT_VERSION",
    "RESOURCE_DIR",
    "load_named_config",
    "load_default_rates",
    "load_default_params",
    "save_named_config",
]
