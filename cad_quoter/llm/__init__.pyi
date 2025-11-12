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
    sugg_to_editor: Mapping[str, Any]
    editor_to_sugg: Mapping[str, Any]
    editor_from_ui: Mapping[str, Any]
    llm_client: type[LLMClient]
    infer_hours_and_overrides_from_geo: Callable[..., Dict[str, Any]]
    parse_llm_json: Callable[[str], Dict[str, Any]]
    explain_quote: Callable[..., str]


def init_llm_integration(system_suggest: str | None = ...) -> LLMIntegration: ...

