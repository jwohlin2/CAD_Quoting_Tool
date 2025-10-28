from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol


EventHook = Callable[[str, Dict[str, Any]], None]
ErrorHook = Callable[[Exception, Dict[str, Any]], None]


class LLMClient:
    model_path: str
    available: bool

    def __init__(
        self,
        model_path: str | None,
        *,
        debug_enabled: bool = ...,
        debug_dir: Path | None = ...,
        on_event: EventHook | None = ...,
        on_error: ErrorHook | None = ...,
    ) -> None: ...

    def ask_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = ...,
        max_tokens: int = ...,
        context: Optional[Dict[str, Any]] = ...,
        params: Optional[Dict[str, Any]] = ...,
    ) -> tuple[dict, str, dict]: ...

    def close(self) -> None: ...


def parse_llm_json(text: str) -> dict[str, Any]: ...


class LLMIntegration(Protocol):
    system_suggest: str
    sugg_to_editor: Callable[[str], str]
    editor_to_sugg: Callable[[str], str]
    editor_from_ui: Callable[[str], str]
    llm_client: type[LLMClient]
    parse_llm_json: Callable[[str], dict[str, Any]]
    explain_quote: Callable[[Mapping[str, Any]], str]


def init_llm_integration(system_suggest: str | None = ...) -> LLMIntegration: ...


def configure_llm_integration(integration: LLMIntegration) -> None: ...


def get_llm_quote_explanation(*args: Any, **kwargs: Any) -> str: ...

