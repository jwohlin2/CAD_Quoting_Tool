from pathlib import Path

import pytest

from cad_quoter import config


def test_app_environment_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_DEBUG", raising=False)
    monkeypatch.delenv("LLM_DEBUG_DIR", raising=False)

    env = config.AppEnvironment.from_env()

    assert env.llm_debug_enabled is True
    assert env.llm_debug_dir.name == "llm_debug"
    assert env.llm_debug_dir.exists()


def test_describe_runtime_environment_redacts_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_DEBUG", "0")
    monkeypatch.setenv("LLM_DEBUG_DIR", str(tmp_path / "custom_debug"))
    monkeypatch.setenv("METALS_API_KEY", "secret")

    info = config.describe_runtime_environment()

    assert info["llm_debug_enabled"] == "False"
    assert Path(info["llm_debug_dir"]).name == "custom_debug"
    assert info["metals_api_key"] == "<redacted>"
