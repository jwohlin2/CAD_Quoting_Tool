"""Modern Tkinter application window mirroring the legacy desktop UI style.

The historical :mod:`appV5` module bundles quoting logic, service wiring, and
Tk widget construction in a single 12k-line script.  This module reimplements
the user-interface layer from scratch while delegating heavy lifting (pricing
and rendering) back to the proven helpers.  The goal is to provide a
maintainable, testable surface that matches the ergonomics of the legacy UI so
future refactors can migrate behaviour a subsystem at a time.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from cad_quoter.config import logger
from cad_quoter.domain_models import QuoteState
from cad_quoter.utils import jdump, json_safe_copy

from cad_quoter.ui.tk_compat import (
    _ensure_tk,
    filedialog,
    messagebox,
    scrolledtext,
    tk,
    ttk,
)
from cad_quoter.ui.widgets import ScrollableFrame
from cad_quoter.ui import editor_controls, llm_panel, session_io
from cad_quoter.ui.services import (
    GeometryLoader,
    LLMServices,
    PricingRegistry,
    QuoteConfiguration,
    UIConfiguration,
)

from cad_quoter.app.optional_loaders import pd
from cad_quoter.app.variables import CORE_COLS, read_variables_file, sanitize_vars_df


@dataclass(slots=True)
class _LegacySupport:
    """Container for optional helpers sourced from :mod:`appV5`."""

    params: dict[str, Any]
    rates: dict[str, Any]
    material_display: str
    pricing_engine: Any
    default_variables_template: Callable[[], Any] | None
    coerce_vars_df: Callable[[Any], Any] | None
    render_quote: Callable[..., str] | None
    compute_quote_from_df: Callable[..., dict[str, Any]] | None


def _load_legacy_support() -> _LegacySupport:
    """Best-effort import of the legacy quoting helpers from :mod:`appV5`."""

    params: dict[str, Any] = {}
    rates: dict[str, Any] = {}
    material_display = "Aluminum MIC6"
    pricing_engine = None
    default_variables_template: Callable[[], Any] | None = None
    coerce_vars_df: Callable[[Any], Any] | None = None
    render_quote: Callable[..., str] | None = None
    compute_quote_from_df: Callable[..., dict[str, Any]] | None = None

    try:  # pragma: no cover - defensive guard for optional dependency
        import appV5 as legacy
    except Exception as exc:  # pragma: no cover - legacy module unavailable
        logger.debug("Legacy UI module unavailable: %s", exc)
    else:
        params = copy.deepcopy(getattr(legacy, "PARAMS_DEFAULT", {}) or {})
        rates = copy.deepcopy(getattr(legacy, "RATES_DEFAULT", {}) or {})
        material_display = getattr(legacy, "DEFAULT_MATERIAL_DISPLAY", material_display)
        pricing_engine = getattr(legacy, "_DEFAULT_PRICING_ENGINE", None)
        default_variables_template = getattr(legacy, "default_variables_template", None)
        coerce_vars_df = getattr(legacy, "coerce_or_make_vars_df", None)
        render_quote = getattr(legacy, "render_quote", None)
        compute_quote_from_df = getattr(legacy, "compute_quote_from_df", None)

    return _LegacySupport(
        params=params,
        rates=rates,
        material_display=material_display,
        pricing_engine=pricing_engine,
        default_variables_template=default_variables_template,
        coerce_vars_df=coerce_vars_df,
        render_quote=render_quote,
        compute_quote_from_df=compute_quote_from_df,
    )


class QuoteApp(tk.Tk):
    """Tkinter window matching the layout of the historical desktop app."""

    def __init__(
        self,
        pricing: Any | None = None,
        *,
        configuration: UIConfiguration | None = None,
        geometry_loader: GeometryLoader | None = None,
        pricing_registry: PricingRegistry | None = None,
        llm_services: LLMServices | None = None,
    ) -> None:
        _ensure_tk()
        super().__init__()

        self.legacy = _load_legacy_support()

        if configuration is None:
            configuration = UIConfiguration(
                default_params=copy.deepcopy(self.legacy.params),
                default_material_display=self.legacy.material_display,
            )
        if pricing_registry is None:
            pricing_registry = PricingRegistry(
                default_params=copy.deepcopy(self.legacy.params),
                default_rates=copy.deepcopy(self.legacy.rates),
            )
        if geometry_loader is None:
            geometry_loader = GeometryLoader()
        if llm_services is None:
            llm_services = LLMServices()

        self.configuration = configuration
        self.pricing_registry = pricing_registry
        self.geometry_loader = geometry_loader
        self.llm_services = llm_services
        self.pricing = pricing or self.legacy.pricing_engine

        self.title(self.configuration.title)
        if self.configuration.window_geometry:
            self.geometry(self.configuration.window_geometry)

        self.quote_state = QuoteState()
        self.vars_df = None
        self.vars_df_full = None
        self.geo: dict[str, Any] | None = None
        self.geo_context: dict[str, Any] = {}

        self.params = self.configuration.create_params()
        self.default_params_template = copy.deepcopy(self.params)
        self.rates = self.pricing_registry.create_rates()
        self.default_rates_template = copy.deepcopy(self.rates)

        self.default_material_display = (
            self.configuration.default_material_display or self.legacy.material_display
        )
        self.quote_config = QuoteConfiguration(
            default_params=copy.deepcopy(self.default_params_template),
            default_material_display=self.default_material_display,
        )

        self.settings_path = self.configuration.settings_path
        self.settings = self.llm_services.load_settings(self.settings_path)

        self.status_var = tk.StringVar(value="Ready")
        self.llm_enabled = tk.BooleanVar(value=self.configuration.llm_enabled_default)
        self.apply_llm_adj = tk.BooleanVar(value=self.configuration.apply_llm_adjustments_default)
        self.llm_model_path = tk.StringVar(
            value=self._initial_llm_model_path(self.settings)
        )
        self.llm_thread_limit = tk.StringVar(
            value=str(self.settings.get("llm_thread_limit", "")) if isinstance(self.settings, dict) else ""
        )

        self.quote_vars: dict[str, tk.StringVar] = {}
        self.param_vars: dict[str, tk.StringVar] = {}
        self.rate_vars: dict[str, tk.StringVar] = {}
        self.editor_specs: dict[str, editor_controls.EditorControlSpec] = {}

        self.auto_reprice_enabled = False
        self._reprice_in_progress = False
        self._quote_dirty = False
        self._quote_dirty_hint = ""

        self._build_menus()
        self._build_layout()
        self._load_default_variables()

    # ------------------------------------------------------------------ UI ----
    def _build_menus(self) -> None:
        self.option_add("*tearOff", False)
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar)
        file_menu.add_command(label="Load Variables…", command=self.load_variables_dialog)
        file_menu.add_separator()
        file_menu.add_command(
            label="Import Session…",
            command=lambda: session_io.import_quote_session(self),
        )
        file_menu.add_command(
            label="Export Session…",
            command=lambda: session_io.export_quote_session(self),
        )
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        quote_menu = tk.Menu(menubar)
        quote_menu.add_command(label="Generate Quote", command=self.gen_quote)
        quote_menu.add_command(label="Apply Overrides", command=self.apply_overrides)
        quote_menu.add_command(label="Reset Overrides", command=self.reset_overrides)
        menubar.add_cascade(label="Quote", menu=quote_menu)

        llm_menu = tk.Menu(menubar)
        llm_menu.add_command(label="Run LLM", command=self.run_llm)
        llm_menu.add_command(label="Open LLM Inspector", command=self.open_llm_inspector)
        menubar.add_cascade(label="LLM", menu=llm_menu)

        self.config(menu=menubar)

    def _build_layout(self) -> None:
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.nb = ttk.Notebook(container)
        self.nb.grid(row=0, column=0, sticky="nsew")

        self.tab_geo = ttk.Frame(self.nb)
        self.tab_editor = ttk.Frame(self.nb)
        self.tab_overrides = ttk.Frame(self.nb)
        self.tab_llm = ttk.Frame(self.nb)
        self.tab_output = ttk.Frame(self.nb)

        self.nb.add(self.tab_geo, text="GEO")
        self.nb.add(self.tab_editor, text="Quote Editor")
        self.nb.add(self.tab_overrides, text="Overrides")
        self.nb.add(self.tab_llm, text="LLM")
        self.nb.add(self.tab_output, text="Output")

        self._build_geo_tab()
        self._build_editor_tab()
        self._build_overrides_tab()
        self._build_llm_tab()
        self._build_output_tab()

        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.grid(row=1, column=0, sticky="ew", padx=4, pady=(2, 4))

    def _build_geo_tab(self) -> None:
        toolbar = ttk.Frame(self.tab_geo)
        toolbar.pack(side="top", fill="x", padx=8, pady=4)
        ttk.Button(toolbar, text="Load CAD…", command=self.load_geometry_dialog).pack(
            side="left"
        )
        ttk.Button(toolbar, text="Apply JSON Payload…", command=self.import_geometry_json).pack(
            side="left", padx=6
        )

        self.geo_txt = scrolledtext.ScrolledText(self.tab_geo, wrap="word", height=20)
        self.geo_txt.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.geo_txt.configure(state="disabled")

    def _build_editor_tab(self) -> None:
        self.editor_scroll = ScrollableFrame(self.tab_editor)
        self.editor_scroll.pack(fill="both", expand=True, padx=8, pady=8)
        self.editor_scroll.inner.columnconfigure(1, weight=1)

    def _build_overrides_tab(self) -> None:
        container = ttk.Frame(self.tab_overrides)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        param_frame = ttk.Labelframe(container, text="Pricing Parameters", padding=(10, 8))
        param_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        rate_frame = ttk.Labelframe(container, text="Hourly Rates ($/hr)", padding=(10, 8))
        rate_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))

        for idx, (key, value) in enumerate(sorted(self.params.items())):
            label = ttk.Label(param_frame, text=key)
            label.grid(row=idx, column=0, sticky="e", padx=4, pady=3)
            var = tk.StringVar(value=self._format_numeric(value))
            entry = ttk.Entry(param_frame, textvariable=var, width=18)
            entry.grid(row=idx, column=1, sticky="w", padx=4, pady=3)
            self.param_vars[key] = var

        for idx, (key, value) in enumerate(sorted(self.rates.items())):
            label = ttk.Label(rate_frame, text=key)
            label.grid(row=idx, column=0, sticky="e", padx=4, pady=3)
            var = tk.StringVar(value=self._format_numeric(value))
            entry = ttk.Entry(rate_frame, textvariable=var, width=18)
            entry.grid(row=idx, column=1, sticky="w", padx=4, pady=3)
            self.rate_vars[key] = var

        buttons = ttk.Frame(container)
        buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)

        ttk.Button(buttons, text="Apply Overrides", command=self.apply_overrides).grid(
            row=0, column=0, sticky="ew", padx=4
        )
        ttk.Button(buttons, text="Reset", command=self.reset_overrides).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(buttons, text="Generate Quote", command=self.gen_quote).grid(
            row=0, column=2, sticky="ew", padx=4
        )

    def _build_llm_tab(self) -> None:
        llm_panel.build_llm_tab(self, self.tab_llm)

    def _build_output_tab(self) -> None:
        self.output_nb = ttk.Notebook(self.tab_output)
        self.output_nb.pack(fill="both", expand=True, padx=8, pady=8)

        simplified_tab = ttk.Frame(self.output_nb)
        full_tab = ttk.Frame(self.output_nb)
        self.output_nb.add(simplified_tab, text="Simplified")
        self.output_nb.add(full_tab, text="Full Detail")

        self.output_text_widgets: dict[str, tk.Text] = {}
        for name, parent in ("simplified", simplified_tab), ("full", full_tab):
            text = tk.Text(parent, wrap="word")
            text.pack(fill="both", expand=True)
            try:
                text.tag_configure("rcol", tabs=("4.8i right",), tabstyle="tabular")
            except tk.TclError:
                text.tag_configure("rcol", tabs=("4.8i right",))
            self.output_text_widgets[name] = text

    # ------------------------------------------------------------ Data I/O ----
    def _load_default_variables(self) -> None:
        if pd is None:
            return
        df = None
        if self.legacy.default_variables_template is not None:
            try:
                df = self.legacy.default_variables_template()
            except Exception:
                df = None
        if df is None:
            columns = CORE_COLS if CORE_COLS else ["Item", "Example Values / Options", "Data Type / Input Method"]
            df = pd.DataFrame(columns=columns)
        self.set_variables_dataframe(df, notify=False)

    def load_variables_dialog(self) -> None:
        if pd is None:
            messagebox.showerror("Variables", "pandas is required to load a variables sheet.")
            return
        path = filedialog.askopenfilename(
            title="Load Variables",
            filetypes=[("Spreadsheet", "*.xlsx"), ("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        self.load_variables_from_path(path)

    def load_variables_from_path(self, path: str | Path) -> None:
        if pd is None:
            raise RuntimeError("pandas is required to load variables")
        try:
            core, full = read_variables_file(str(path), return_full=True)
        except Exception as exc:
            messagebox.showerror("Variables", f"Failed to load variables:\n{exc}")
            logger.exception("Failed to load variables from %s", path)
            return
        self.vars_df_full = full
        self.set_variables_dataframe(core)
        self.status_var.set(f"Loaded variables from {path}")

    def set_variables_dataframe(self, df: Any, *, notify: bool = True) -> None:
        if pd is not None and isinstance(df, pd.DataFrame):
            core_df = sanitize_vars_df(df)
            if self.legacy.coerce_vars_df is not None:
                try:
                    core_df = self.legacy.coerce_vars_df(core_df)
                except Exception:
                    pass
            self.vars_df = core_df.copy()
        else:
            self.vars_df = df
        try:
            self._populate_editor_tab(self.vars_df)
        except Exception as exc:
            logger.exception("Failed to populate editor from dataframe")
            messagebox.showerror("Quote Editor", f"Unable to populate editor widgets:\n{exc}")
        else:
            if notify:
                self.status_var.set("Quote variables ready.")

    # --------------------------------------------------------- Editor pane ----
    def _populate_editor_tab(self, df: Any) -> None:
        parent = self.editor_scroll.inner
        for child in parent.winfo_children():
            child.destroy()

        self.quote_vars.clear()
        self.editor_specs.clear()

        if pd is not None and isinstance(df, pd.DataFrame):
            iterator = df.iterrows()
        elif isinstance(df, list):
            iterator = enumerate(df)
        else:
            iterator = enumerate([])

        row = 0
        for _, raw_row in iterator:
            if pd is not None and hasattr(raw_row, "to_dict"):
                row_data = raw_row.to_dict()
            elif isinstance(raw_row, Mapping):
                row_data = dict(raw_row)
            else:
                continue
            item = str(row_data.get("Item", "")).strip()
            if not item:
                continue
            dtype = row_data.get("Data Type / Input Method", "")
            example = row_data.get("Example Values / Options", "")
            spec = editor_controls.derive_editor_control_spec(dtype, example)
            self.editor_specs[item] = spec

            label = ttk.Label(parent, text=spec.display_label or item)
            label.grid(row=row, column=0, sticky="e", padx=6, pady=4)

            if spec.control == "checkbox":
                string_var = tk.StringVar(value="True" if spec.checkbox_state else "False")
                bool_var = tk.BooleanVar(value=spec.checkbox_state)

                def _sync(var=string_var, source=bool_var) -> None:
                    var.set("True" if source.get() else "False")

                bool_var.trace_add("write", lambda *_: _sync())
                _sync()
                widget = ttk.Checkbutton(parent, text=spec.checkbox_label, variable=bool_var)
                widget.grid(row=row, column=1, sticky="w", padx=6, pady=4)
            elif spec.control == "dropdown":
                string_var = tk.StringVar(value=spec.entry_value)
                widget = ttk.Combobox(
                    parent,
                    textvariable=string_var,
                    values=list(spec.options),
                    width=40,
                    state="readonly" if spec.options else "normal",
                )
                widget.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
            else:
                string_var = tk.StringVar(value=spec.entry_value)
                widget = ttk.Entry(parent, textvariable=string_var, width=45)
                widget.grid(row=row, column=1, sticky="ew", padx=6, pady=4)

            self.quote_vars[item] = string_var
            row += 1

        parent.grid_columnconfigure(1, weight=1)

    # ------------------------------------------------------------ Overrides ----
    def _format_numeric(self, value: Any) -> str:
        if value in (None, ""):
            return ""
        try:
            num = float(value)
        except Exception:
            return str(value)
        text = f"{num:.6f}".rstrip("0").rstrip(".")
        return text or "0"

    def apply_overrides(self, notify: bool = False) -> None:
        for key, var in self.param_vars.items():
            raw = var.get()
            if raw in (None, ""):
                continue
            try:
                value = float(str(raw))
            except Exception:
                self.params[key] = raw
            else:
                self.params[key] = value

        for key, var in self.rate_vars.items():
            raw = var.get()
            if raw in (None, ""):
                continue
            try:
                self.rates[key] = float(str(raw))
            except Exception:
                self.rates[key] = raw

        if notify:
            messagebox.showinfo("Overrides", "Overrides applied.")
        self.status_var.set("Overrides applied.")

    def reset_overrides(self) -> None:
        self.params = copy.deepcopy(self.default_params_template)
        self.rates = copy.deepcopy(self.default_rates_template)
        for key, var in self.param_vars.items():
            var.set(self._format_numeric(self.params.get(key, "")))
        for key, var in self.rate_vars.items():
            var.set(self._format_numeric(self.rates.get(key, "")))
        self.status_var.set("Overrides reset to defaults.")

    # -------------------------------------------------------------- Geometry ----
    def load_geometry_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Load CAD",
            filetypes=[
                ("CAD Files", "*.step *.stp *.iges *.igs *.stl"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        self.load_geometry_from_path(Path(path))

    def load_geometry_from_path(self, path: Path) -> None:
        payload: dict[str, Any] = {"source_path": str(path)}
        try:
            shape = self.geometry_loader.read_cad_any(path)
        except Exception as exc:
            logger.warning("Failed to read CAD %s: %s", path, exc)
        else:
            try:
                enriched = self.geometry_loader.enrich_geo_occ(shape)
            except Exception:
                enriched = None
            if isinstance(enriched, Mapping):
                payload.update(json_safe_copy(enriched))
        self.apply_geometry_payload(payload, source=str(path))

    def import_geometry_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Geometry JSON",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            messagebox.showerror("Geometry", f"Failed to load geometry JSON:\n{exc}")
            logger.exception("Failed to load geometry JSON from %s", path)
            return
        self.apply_geometry_payload(payload, source=str(path))

    def apply_geometry_payload(
        self, payload: Mapping[str, Any], *, source: str | None = None
    ) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Geometry payload must be a mapping")

        def _clone(mapping: Mapping[str, Any]) -> dict[str, Any]:
            return {str(key): value for key, value in mapping.items()}

        geo_payload: dict[str, Any]
        if isinstance(payload.get("geo"), Mapping):
            geo_payload = _clone(payload["geo"])
        elif isinstance(payload.get("geom"), Mapping):
            geo_payload = _clone(payload["geom"])
        else:
            geo_payload = _clone(payload)

        geo_context: dict[str, Any]
        if isinstance(payload.get("geo_context"), Mapping):
            geo_context = _clone(payload["geo_context"])
        else:
            geo_context = dict(geo_payload)
            extra = payload.get("geo_read_more")
            if isinstance(extra, Mapping):
                geo_context["geo_read_more"] = _clone(extra)

        self.geo = json_safe_copy(geo_payload)
        self.geo_context = json_safe_copy(geo_context)
        self.quote_state.geo = json_safe_copy(self.geo)

        status = "Geometry payload loaded." if not source else f"Geometry loaded from {source}"
        self.status_var.set(status)
        self.nb.select(self.tab_geo)

        self.geo_txt.configure(state="normal")
        self.geo_txt.delete("1.0", "end")
        self.geo_txt.insert("end", jdump(self.geo_context, default=None))
        self.geo_txt.configure(state="disabled")

    # ------------------------------------------------------------- LLM tab ----
    def _initial_llm_model_path(self, settings: Mapping[str, Any] | None) -> str:
        if isinstance(settings, Mapping):
            raw = settings.get("llm_model_path")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        try:
            return self.llm_services.default_model_path()
        except Exception:
            return ""

    def _validate_thread_limit(self, proposed: str) -> bool:
        text = str(proposed).strip()
        return text.isdigit() or text == ""

    def _current_llm_thread_limit(self) -> int | None:
        text = str(self.llm_thread_limit.get() or "").strip()
        if not text:
            return None
        try:
            value = int(text, 10)
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    def _sync_llm_thread_limit(self, persist: bool = True) -> None:
        limit = self._current_llm_thread_limit()
        self.settings = self.llm_services.apply_thread_limit_env(
            limit,
            settings=dict(self.settings or {}),
            persist=persist,
            settings_path=self.settings_path,
        )

    def _pick_model(self) -> None:
        llm_panel.pick_llm_model(self)

    def run_llm(self) -> None:
        llm_panel.run_llm(self)

    def open_llm_inspector(self) -> None:
        llm_panel.open_llm_inspector(self)

    # ------------------------------------------------------------- Quoting ----
    def _collect_ui_vars(self) -> dict[str, Any]:
        return {label: var.get() for label, var in self.quote_vars.items()}

    def gen_quote(self, reuse_suggestions: bool = False) -> None:
        if self._reprice_in_progress:
            return
        if pd is None:
            messagebox.showerror("Quote", "pandas is required to generate a quote.")
            return
        if self.vars_df is None:
            messagebox.showinfo("Quote", "Load a variables sheet before generating a quote.")
            return

        try:
            df_local = self.vars_df.copy()
        except Exception:
            df_local = self.vars_df

        try:
            values = self._collect_ui_vars()
            if pd is not None and isinstance(df_local, pd.DataFrame):
                items = df_local["Item"].astype(str)
                for label, string_var in values.items():
                    mask = items == label
                    if mask.any():
                        df_local.loc[mask, "Example Values / Options"] = string_var
        except Exception:
            logger.exception("Failed to merge UI values into dataframe")

        self.apply_overrides()

        if self.legacy.compute_quote_from_df is None or self.legacy.render_quote is None:
            messagebox.showerror("Quote", "Legacy quoting helpers are unavailable.")
            return

        self._reprice_in_progress = True
        try:
            self.status_var.set("Generating quote…")
            self.update_idletasks()

            try:
                result = self.legacy.compute_quote_from_df(
                    df_local,
                    params=self.params,
                    rates=self.rates,
                    geo=self.geo,
                    ui_vars=values,
                    quote_state=self.quote_state,
                    llm_enabled=bool(self.llm_enabled.get()),
                    llm_model_path=self.llm_model_path.get().strip() or None,
                    reuse_suggestions=reuse_suggestions,
                    cfg=self.quote_config,
                )
            except Exception as exc:
                logger.exception("compute_quote_from_df failed")
                messagebox.showerror("Quote", f"Quote generation failed:\n{exc}")
                return

            state_payload = result.get("quote_state") if isinstance(result, Mapping) else None
            if isinstance(state_payload, QuoteState):
                self.quote_state = state_payload
            elif isinstance(state_payload, Mapping):
                try:
                    self.quote_state = QuoteState.from_dict(state_payload)
                except Exception:
                    pass

            simplified = self.legacy.render_quote(
                result,
                currency="$",
                show_zeros=False,
                cfg=self.quote_config,
                geometry=self.geo_context,
            )
            full = self.legacy.render_quote(
                result,
                currency="$",
                show_zeros=True,
                cfg=self.quote_config,
                geometry=self.geo_context,
            )

            for name, text in ("simplified", simplified), ("full", full):
                widget = self.output_text_widgets.get(name)
                if widget is None:
                    continue
                widget.delete("1.0", "end")
                widget.insert("end", text, "rcol")
                try:
                    widget.mark_set("insert", "1.0")
                    widget.see("1.0")
                except Exception:
                    pass

            self.output_nb.select(0)
            self.nb.select(self.tab_output)
            price_text = result.get("price") if isinstance(result, Mapping) else None
            if price_text is None:
                self.status_var.set("Quote generated.")
            else:
                try:
                    price_value = float(price_text)
                except Exception:
                    self.status_var.set("Quote generated.")
                else:
                    self.status_var.set(f"Quote generated — ${price_value:,.2f}")
        finally:
            self._reprice_in_progress = False

    # --------------------------------------------------------------- Session ----
    def apply_overrides_and_mark_dirty(self) -> None:
        self.apply_overrides()
        self._mark_quote_dirty("overrides")

    def _mark_quote_dirty(self, hint: str | None = None) -> None:
        self._quote_dirty = True
        if hint:
            self._quote_dirty_hint = hint
        self.status_var.set("Quote needs regeneration.")

    def _clear_quote_dirty(self) -> None:
        self._quote_dirty = False
        self._quote_dirty_hint = ""


__all__ = ["QuoteApp"]

