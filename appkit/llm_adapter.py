from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, TypeAlias, cast

# Public protocol describing the LLM client surface appV5 relies on
class LLMClientLike(Protocol):
    @property
    def model_path(self) -> str: ...

    @property
    def available(self) -> bool: ...

    def ask_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = ...,
        max_tokens: int = ...,
        context: Mapping[str, Any] | None = ...,
        params: Mapping[str, Any] | None = ...,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]: ...

    def close(self) -> None: ...


RunLLMSuggestions: TypeAlias = Callable[
    [LLMClientLike, dict[str, Any]], tuple[dict[str, Any], str, dict[str, Any]]
]

try:
    # Optional runtime dependency; may be missing in some environments
    from cad_quoter.llm import run_llm_suggestions as _run_llm_suggestions  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _run_llm_suggestions = None  # type: ignore[assignment]

run_llm_suggestions: RunLLMSuggestions | None = cast(
    "RunLLMSuggestions | None", _run_llm_suggestions
)

# Re-export guardrails/bounds helpers from the existing module
from cad_quoter.llm_overrides import (  # noqa: E402
    get_llm_overrides,
    get_llm_bound_defaults,
)

__all__ = [
    "LLMClientLike",
    "RunLLMSuggestions",
    "run_llm_suggestions",
    "get_llm_overrides",
    "get_llm_bound_defaults",
]

