import json
from pathlib import Path
from typing import Any

import pytest

from cad_quoter import config
from cad_quoter import rates as rate_helpers


def test_app_environment_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    monkeypatch.delenv("LLM_DEBUG_DIR", raising=False)

    env = config.AppEnvironment.from_env()

    assert env.llm_debug_enabled is False
    assert env.llm_debug_dir.name == "llm_debug"
    assert env.llm_debug_dir.exists()


def test_app_environment_from_env_parses_boolean_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DEBUG", "TrUe")
    env_true = config.AppEnvironment.from_env()
    assert env_true.llm_debug_enabled is True

    monkeypatch.setenv("LLM_DEBUG", "off")
    env_false = config.AppEnvironment.from_env()
    assert env_false.llm_debug_enabled is False


def test_describe_runtime_environment_redacts_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_DEBUG", "0")
    monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path / "custom_debug"))
    monkeypatch.setenv("METALS_API_KEY", "secret")

    info = config.describe_runtime_environment()

    assert info["llm_debug_enabled"] == "False"
    assert Path(info["llm_debug_dir"]).name == "custom_debug"
    assert info["metals_api_key"] == "<redacted>"


def test_load_default_rates_returns_two_buckets() -> None:
    rates = config.load_default_rates()

    assert set(rates.keys()) == {"labor", "machine"}
    expected = rate_helpers.shared_two_bucket_rate_defaults()
    assert rates == expected


def test_load_default_rates_ignores_app_settings_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_settings() -> dict[str, Any]:
        return {
            "pricing_defaults": {
                "rates": {
                    "ProgrammingRate": 110.0,
                    "WireEDMRate": 150.0,
                    "SurfaceGrindRate": 120.0,
                }
            }
        }

    monkeypatch.setattr(config, "load_app_settings", lambda reload=False: _fake_settings())

    migrated = config.load_default_rates()

    expected = rate_helpers.shared_two_bucket_rate_defaults()
    assert migrated == expected


def test_load_app_settings_applies_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    override = tmp_path / "override.json"
    override.write_text(
        json.dumps(
            {
                "pricing_defaults": {
                    "params": {"MarginPct": 0.5},
                    "rates": {"labor": {"Programmer": 123.0}},
                }
            }
        )
    )

    monkeypatch.setenv(config.APP_SETTINGS_ENV_VAR, str(override))

    merged = config.load_app_settings(reload=True)

    assert merged["pricing_defaults"]["params"]["MarginPct"] == pytest.approx(0.5)
    assert merged["pricing_defaults"]["rates"]["labor"]["Programmer"] == pytest.approx(123.0)

    monkeypatch.delenv(config.APP_SETTINGS_ENV_VAR, raising=False)
