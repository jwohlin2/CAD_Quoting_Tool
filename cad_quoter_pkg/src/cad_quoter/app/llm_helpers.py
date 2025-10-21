"""Helper utilities for optional LLM integrations used by ``appV5``.

This module centralizes the defensive imports and compatibility aliases that
keep the GUI functional when the optional ``cad_quoter.llm`` package is not
installed.  The UI only needs a handful of helpers – primarily the LLM client
type and a few mapping tables – so the logic here focuses on returning safe
fallbacks when those helpers are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from cad_quoter.app import runtime as _runtime

# Runtime helpers that callers historically imported from ``appV5``.  Expose
# them here so the large UI module can stay focused on presentation logic.
ensure_runtime_dependencies = _runtime.ensure_runtime_dependencies
find_default_qwen_model = _runtime.find_default_qwen_model
load_qwen_vl = _runtime.load_qwen_vl

DEFAULT_VL_MODEL_NAMES = _runtime.DEFAULT_VL_MODEL_NAMES
DEFAULT_MM_PROJ_NAMES = _runtime.DEFAULT_MM_PROJ_NAMES
VL_MODEL = str(_runtime.LEGACY_VL_MODEL)
MM_PROJ = str(_runtime.LEGACY_MM_PROJ)
LEGACY_VL_MODEL = VL_MODEL
LEGACY_MM_PROJ = MM_PROJ


@dataclass(frozen=True)
class LLMIntegration:
    """Container describing the optional LLM helpers used by the UI."""

    system_suggest: str
    sugg_to_editor: Dict[str, Any]
    editor_to_sugg: Dict[str, Any]
    editor_from_ui: Dict[str, Any]
    llm_client: type
    infer_hours_and_overrides_from_geo: Callable[..., Dict[str, Any]]
    parse_llm_json: Callable[[str], Dict[str, Any]]
    explain_quote: Callable[..., str]


class _FallbackLLMClient:
    """Placeholder used when the optional LLM helpers are unavailable."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
        raise RuntimeError("LLM integration is not available in this environment.")

    @property
    def model_path(self) -> str:  # pragma: no cover - defensive
        return ""

    @property
    def available(self) -> bool:  # pragma: no cover - defensive
        return False

    def ask_json(self, *args: Any, **kwargs: Any) -> tuple[dict, str, dict]:  # pragma: no cover - defensive
        raise RuntimeError("LLM integration is not available in this environment.")

    def close(self) -> None:  # pragma: no cover - defensive
        return None


def init_llm_integration(default_system_suggest: str) -> LLMIntegration:
    """Return LLM helpers with graceful fallbacks when unavailable."""

    try:  # pragma: no cover - defensive guard when optional LLM helpers are absent
        from cad_quoter import llm as _cad_llm  # type: ignore
    except Exception:  # pragma: no cover - tolerate partial installs
        _cad_llm = None

    system_suggest = default_system_suggest
    sugg_to_editor: Dict[str, Any] = {}
    editor_to_sugg: Dict[str, Any] = {}
    editor_from_ui: Dict[str, Any] = {}
    llm_client: type = _FallbackLLMClient
    infer_hours_and_overrides_from_geo: Callable[..., Dict[str, Any]] = (
        lambda *args, **kwargs: {}
    )
    parse_llm_json: Callable[[str], Dict[str, Any]] = lambda _text: {}
    explain_quote: Callable[..., str] = (
        lambda *args, **kwargs: "LLM explanation unavailable."
    )

    if _cad_llm is not None:
        system_suggest = getattr(_cad_llm, "SYSTEM_SUGGEST", default_system_suggest)
        sugg_to_editor = getattr(_cad_llm, "SUGG_TO_EDITOR", {})
        editor_to_sugg = getattr(_cad_llm, "EDITOR_TO_SUGG", {})
        editor_from_ui = getattr(_cad_llm, "EDITOR_FROM_UI", {})

        maybe_client = getattr(_cad_llm, "LLMClient", None)
        if isinstance(maybe_client, type):
            llm_client = maybe_client

        infer_hours_and_overrides_from_geo = getattr(
            _cad_llm,
            "infer_hours_and_overrides_from_geo",
            infer_hours_and_overrides_from_geo,
        )
        parse_llm_json = getattr(_cad_llm, "parse_llm_json", parse_llm_json)
        explain_quote = getattr(_cad_llm, "explain_quote", explain_quote)

    return LLMIntegration(
        system_suggest=system_suggest,
        sugg_to_editor=sugg_to_editor,
        editor_to_sugg=editor_to_sugg,
        editor_from_ui=editor_from_ui,
        llm_client=llm_client,
        infer_hours_and_overrides_from_geo=infer_hours_and_overrides_from_geo,
        parse_llm_json=parse_llm_json,
        explain_quote=explain_quote,
    )


__all__ = [
    "DEFAULT_MM_PROJ_NAMES",
    "DEFAULT_VL_MODEL_NAMES",
    "LEGACY_MM_PROJ",
    "LEGACY_VL_MODEL",
    "LLMIntegration",
    "MM_PROJ",
    "VL_MODEL",
    "ensure_runtime_dependencies",
    "find_default_qwen_model",
    "init_llm_integration",
    "load_qwen_vl",
]
