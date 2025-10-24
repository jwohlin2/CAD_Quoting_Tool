"""LLM configuration helpers for the desktop UI."""

from __future__ import annotations

import os
import time
import tkinter as tk
from pathlib import Path
from typing import Any, Callable, Optional

try:  # pragma: no cover - optional dependency guard
    from cad_quoter.llm import LLMClient as LLMClientType
except Exception:  # pragma: no cover - fallback when llama-cpp bindings missing
    LLMClientType = Any  # type: ignore[assignment]

from cad_quoter.ui.services import LLMServices, UIConfiguration


StatusSetter = Callable[[str], None]
IdleCallback = Callable[[], None]
EventLogger = Callable[[str, dict[str, Any]], None]
ErrorHandler = Callable[[Exception, dict[str, Any]], None]


class LLMControls:
    """Encapsulate Tk variables and helpers for LLM configuration."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        llm_services: LLMServices,
        configuration: UIConfiguration,
        settings: dict[str, Any],
        settings_path: Path | None,
        debug_enabled: bool = False,
        debug_dir: str | os.PathLike[str] | None = None,
        on_event: EventLogger | None = None,
        on_error: ErrorHandler | None = None,
        idle_callback: IdleCallback | None = None,
        status_setter: StatusSetter | None = None,
    ) -> None:
        self._master = master
        self.llm_services = llm_services
        self.configuration = configuration
        self.settings = settings
        self.settings_path = settings_path
        self._debug_enabled = debug_enabled
        self._debug_dir = debug_dir
        self._on_event = on_event
        self._on_error = on_error
        self._idle_callback = idle_callback
        self._status_setter = status_setter or (lambda _msg: None)

        self.llm_enabled = tk.BooleanVar(master=master, value=configuration.llm_enabled_default)
        self.apply_llm_adj = tk.BooleanVar(
            master=master, value=configuration.apply_llm_adjustments_default
        )

        default_model = configuration.default_llm_model_path or llm_services.default_model_path()
        default_model = default_model or ""
        if default_model:
            os.environ["QWEN_GGUF_PATH"] = default_model
        self.model_path = tk.StringVar(master=master, value=default_model)

        thread_setting = str(settings.get("llm_thread_limit", "") or "").strip()
        env_thread_setting = os.environ.get("QWEN_N_THREADS", "").strip()
        initial_thread_setting = thread_setting or env_thread_setting
        self.thread_limit = tk.StringVar(master=master, value=initial_thread_setting)
        self._thread_limit_applied: int | None = None

        if hasattr(self.thread_limit, "trace_add"):
            self.thread_limit.trace_add("write", self._on_thread_limit_changed)
        elif hasattr(self.thread_limit, "trace"):
            self.thread_limit.trace("w", lambda *_: self._on_thread_limit_changed())

        self._client_cache: LLMClientType | None = None
        self._vision_model: Any | None = None
        self._load_attempted = False
        self._load_error: Exception | None = None

    # ------------------------------------------------------------------
    # Callback wiring
    # ------------------------------------------------------------------

    def register_status_setter(self, setter: StatusSetter) -> None:
        self._status_setter = setter

    def register_idle_callback(self, callback: IdleCallback) -> None:
        self._idle_callback = callback

    # ------------------------------------------------------------------
    # Thread limit helpers
    # ------------------------------------------------------------------

    def current_thread_limit(self) -> int | None:
        raw = self.thread_limit.get()
        text = str(raw).strip()
        if not text:
            return None
        try:
            value = int(text, 10)
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    def sync_thread_limit(self, *, persist: bool) -> int | None:
        limit = self.current_thread_limit()
        if limit != self._thread_limit_applied:
            self._thread_limit_applied = limit
            self._invalidate_client_cache()

        updated = self.llm_services.apply_thread_limit_env(
            limit,
            settings=self.settings,
            persist=persist,
            settings_path=self.settings_path if persist else None,
        )
        if isinstance(updated, dict) and updated is not self.settings:
            self.settings.clear()
            self.settings.update(updated)
        return limit

    def _on_thread_limit_changed(self, *_: object) -> None:
        self.sync_thread_limit(persist=True)

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _invalidate_client_cache(self) -> None:
        cached = self._client_cache
        if cached is not None:
            try:
                cached.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._client_cache = None

    def get_client(self, model_path: str | None = None) -> Optional[LLMClientType]:
        path = (model_path or "").strip()
        if not path and self.model_path.get():
            path = self.model_path.get().strip()
        if not path:
            path = os.environ.get("QWEN_GGUF_PATH", "").strip()
        if not path:
            return None

        self.sync_thread_limit(persist=False)

        cached = self._client_cache
        if cached is not None and getattr(cached, "model_path", None) == path:
            return cached
        if cached is not None:
            try:
                cached.close()  # type: ignore[attr-defined]
            except Exception:
                pass

        client = LLMClientType(  # type: ignore[call-arg]
            path,
            debug_enabled=self._debug_enabled,
            debug_dir=self._debug_dir,
            on_event=self._on_event,
            on_error=self._on_error,
        )
        self._client_cache = client
        return client

    # ------------------------------------------------------------------
    # Vision model helpers
    # ------------------------------------------------------------------

    @property
    def vision_model(self) -> Any | None:
        return self._vision_model

    def ensure_loaded(self) -> Any | None:
        if self._vision_model is not None:
            return self._vision_model
        if self._load_attempted and self._load_error is not None:
            return None

        self._load_attempted = True
        start = time.perf_counter()

        limit = None
        try:
            limit = self.sync_thread_limit(persist=False)
            status = "Loading Vision LLM (GPU)…"
            if limit:
                status = f"Loading Vision LLM (GPU, {limit} CPU threads)…"
            self._status_setter(status)
            if self._idle_callback is not None:
                self._idle_callback()
        except Exception:
            pass

        try:
            self._vision_model = self.llm_services.load_vision_model(
                n_ctx=8192,
                n_gpu_layers=20,
                n_threads=limit,
            )
        except Exception as exc:
            self._load_error = exc
            try:
                limit = self.sync_thread_limit(persist=False)
                message = f"Vision LLM GPU load failed ({exc}); retrying CPU mode…"
                if limit:
                    message = f"{message[:-1]} with {limit} CPU threads…)"
                self._status_setter(message)
                if self._idle_callback is not None:
                    self._idle_callback()
            except Exception:
                pass
            try:
                self._vision_model = self.llm_services.load_vision_model(
                    n_ctx=4096,
                    n_gpu_layers=0,
                    n_threads=limit,
                )
            except Exception as exc2:
                self._load_error = exc2
                self._status_setter(f"Vision LLM unavailable: {exc2}")
                return None
        else:
            self._load_error = None

        duration = time.perf_counter() - start
        try:
            self._status_setter(f"Vision LLM ready in {duration:.1f}s")
        except Exception:
            pass
        return self._vision_model
