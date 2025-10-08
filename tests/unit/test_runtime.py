from __future__ import annotations

from pathlib import Path

import pytest

from cad_quoter.app import runtime


class _DummyLlama:
    last_chat_format: str | None = None

    def __init__(self, *_, chat_format: str | None = None, **__):
        _DummyLlama.last_chat_format = chat_format
        if chat_format == "qwen2.5_vl":
            raise RuntimeError(
                "Invalid chat handler: qwen2.5_vl (valid formats: ['llama-2', 'qwen'])"
            )

    def create_chat_completion(self, *_, **__):
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


@pytest.fixture(autouse=True)
def patch_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "Llama", _DummyLlama)


def _touch(path: Path) -> Path:
    path.write_text("stub", encoding="utf-8")
    return path


def test_load_qwen_vl_falls_back_to_legacy_chat(tmp_path: Path) -> None:
    model = _touch(tmp_path / "model.gguf")
    mmproj = _touch(tmp_path / "mmproj.gguf")

    llm = runtime.load_qwen_vl(
        n_ctx=2048,
        n_gpu_layers=0,
        n_threads=2,
        model_path=str(model),
        mmproj_path=str(mmproj),
    )

    assert isinstance(llm, _DummyLlama)
    assert _DummyLlama.last_chat_format == "qwen"
