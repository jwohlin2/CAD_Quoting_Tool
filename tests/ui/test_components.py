import tkinter as tk
from types import SimpleNamespace

import pandas as pd
import pytest
from tkinter import ttk

from cad_quoter.ui.llm_controls import LLMControls
from cad_quoter.ui.output_pane import OutputPane
from cad_quoter.ui.quote_editor import QuoteEditorView
from cad_quoter.ui.services import UIConfiguration


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tkinter is unavailable in this environment")
    root.withdraw()
    yield root
    root.destroy()


def test_output_pane_initialises_text_widgets(tk_root):
    frame = ttk.Frame(tk_root)
    frame.pack()
    pane = OutputPane(frame)

    widget = pane.get_text_widget("simplified")
    assert widget is not None
    assert "rcol" in widget.tag_names()


class _DummyLLMServices:
    def __init__(self):
        self.saved_settings = None
        self.applied_limit = None

    def default_model_path(self) -> str:
        return ""

    def load_vision_model(self, **_kwargs):
        return object()

    def apply_thread_limit_env(self, limit, *, settings=None, persist=True, settings_path=None):
        self.applied_limit = limit
        if persist and isinstance(settings, dict):
            settings["llm_thread_limit"] = str(limit) if limit is not None else ""
        return settings

    def save_settings(self, path, settings):
        self.saved_settings = (path, dict(settings))


def test_llm_controls_syncs_thread_limit(tk_root):
    services = _DummyLLMServices()
    settings = {}
    controls = LLMControls(
        tk_root,
        llm_services=services,
        configuration=UIConfiguration(),
        settings=settings,
        settings_path=None,
        on_event=None,
        on_error=None,
    )
    controls.register_status_setter(lambda _msg: None)

    controls.thread_limit.set("4")
    assert controls.sync_thread_limit(persist=True) == 4
    assert services.applied_limit == 4
    assert settings.get("llm_thread_limit") == "4"


class _StubLLMServices:
    def save_settings(self, *_args, **_kwargs):
        return None


class _StubQuoteState:
    def __init__(self):
        self.user_overrides = {}
        self.bounds = {}


def test_quote_editor_view_populate_creates_fields(tk_root):
    app = SimpleNamespace()
    app.params = {"Quantity": 1}
    app.rates = {}
    app.default_material_display = "Aluminum"
    app.settings = {}
    app.llm_services = _StubLLMServices()
    app.settings_path = None
    app.quote_state = _StubQuoteState()
    app.default_params_template = {}
    app.default_rates_template = {}
    app.status_var = tk.StringVar(master=tk_root)
    app._reprice_in_progress = False
    app.vars_df = None
    app.vars_df_full = pd.DataFrame(
        {"Item": ["Material"], "Data Type / Input Method": ["text"], "Example Values / Options": ["Aluminum"]}
    )

    def _set_user_override_value(path, value):
        app.last_override = (path, value)

    def _reprice(**_kwargs):
        app.last_reprice = True

    app._set_user_override_value = _set_user_override_value
    app.reprice = _reprice

    df = pd.DataFrame(
        {
            "Item": ["Material"],
            "Example Values / Options": ["Aluminum"],
            "Data Type / Input Method": ["Text"],
        }
    )

    view = QuoteEditorView(
        app,
        ttk.Frame(tk_root),
        coerce_df_fn=lambda frame: frame,
        material_price_updater=lambda *args, **kwargs: False,
        sugg_to_editor={},
        editor_to_sugg={},
        editor_from_ui={},
    )

    view.populate(df)
    assert "Material" in view.quote_vars
    assert view.quote_vars["Material"].get() == "Aluminum"
    assert app.vars_df is df


def test_llm_controls_cpu_fallback_without_tk(monkeypatch):
    from types import SimpleNamespace

    import cad_quoter.ui.llm_controls as mod

    statuses: list[str] = []
    idle_calls: list[int] = []

    class _Var:
        def __init__(self, master=None, value=None):
            self._value = value

        def get(self):
            return self._value

        def set(self, value):
            self._value = value

        def trace_add(self, *_args):
            return None

    class _BooleanVar(_Var):
        pass

    class _StringVar(_Var):
        pass

    fake_tk = SimpleNamespace(BooleanVar=_BooleanVar, StringVar=_StringVar)
    monkeypatch.setattr(mod, "tk", fake_tk)

    class _Services:
        def __init__(self):
            self.env_calls: list[tuple[int | None, bool]] = []

        def default_model_path(self) -> str:
            return ""

        def apply_thread_limit_env(self, limit, *, settings=None, persist=True, settings_path=None):
            self.env_calls.append((limit, persist))
            return settings or {}

        def load_vision_model(self, *, n_ctx, n_gpu_layers, n_threads):
            if n_gpu_layers > 0:
                raise RuntimeError("gpu unavailable")
            return {
                "n_ctx": n_ctx,
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads,
            }

    services = _Services()
    config = UIConfiguration()
    settings: dict[str, str] = {}

    controls = mod.LLMControls(
        SimpleNamespace(),
        llm_services=services,
        configuration=config,
        settings=settings,
        settings_path=None,
        on_event=None,
        on_error=None,
        idle_callback=lambda: idle_calls.append(1),
        status_setter=statuses.append,
    )

    controls.thread_limit.set("6")
    model = controls.ensure_loaded()

    assert services.env_calls and services.env_calls[0] == (6, False)
    assert model == {"n_ctx": 4096, "n_gpu_layers": 0, "n_threads": 6}
    assert any("retrying CPU mode" in msg for msg in statuses)
    assert idle_calls  # idle callback should have been invoked
