"""LLM façade used by the CAD Quoter UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from .config import AppEnvironment


@dataclass
class LLMInsight:
    """Represents a single string returned from an LLM prompt."""

    message: str


class LLMService:
    """Small wrapper that pretends to talk to a language model.

    In the real system this module would load GGUF models and stream
    responses.  Here we emit deterministic placeholder insights so the UI can
    be exercised without bundling heavyweight dependencies.
    """

    def __init__(self, env: AppEnvironment) -> None:
        self._env = env

    @property
    def debug_directory(self) -> Path:
        return self._env.llm_debug_dir

    def generate_insights(self, prompt: str) -> list[LLMInsight]:
        prompt = prompt.strip()
        if not prompt:
            return [LLMInsight("No prompt provided. Enter context to receive suggestions.")]
        notes = [
            "LLM placeholder response.",
            "Debugging is {}.".format("enabled" if self._env.llm_debug_enabled else "disabled"),
            f"Prompt length: {len(prompt)} characters.",
        ]
        return [LLMInsight(message=n) for n in notes]


__all__ = ["LLMService", "LLMInsight"]
