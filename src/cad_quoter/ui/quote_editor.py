"""Quote Editor view helpers extracted from the Tk application."""

from __future__ import annotations

import math
import re
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import typing

try:  # pragma: no cover - optional pandas dependency
    import pandas as pd
    from pandas import DataFrame as PandasDataFrame
    from pandas import Index as PandasIndex
    from pandas import Series as PandasSeries
    _HAS_PANDAS = True
except Exception:  # pragma: no cover - pandas optional
    pd = None
    PandasDataFrame = typing.Any  # type: ignore[assignment]
    PandasIndex = typing.Any  # type: ignore[assignment]
    PandasSeries = typing.Any  # type: ignore[assignment]
    _HAS_PANDAS = False

from cad_quoter.app.llm_adapter import coerce_bounds, normalize_item_text
from cad_quoter.app.variables import _load_master_variables
from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.domain_models import normalize_material_key as _normalize_lookup_key
from cad_quoter.domain_models.materials import (
    MATERIAL_DISPLAY_BY_KEY,
    MATERIAL_DROPDOWN_OPTIONS,
    MATERIAL_KEYWORDS,
    MATERIAL_OTHER_KEY,
)
from cad_quoter.geometry import upsert_var_row as geometry_upsert_var_row
from cad_quoter.llm_overrides import clamp
from cad_quoter.pricing import price_value_to_per_gram as _price_value_to_per_gram
from cad_quoter.rates import LABOR_RATE_KEYS, MACHINE_RATE_KEYS
from cad_quoter.ui.editor_controls import coerce_checkbox_state, derive_editor_control_spec
from cad_quoter.ui.services import infer_geo_override_defaults
from cad_quoter.ui.widgets import CreateToolTip, ScrollableFrame
from cad_quoter.utils.text_rules import PROC_MULT_TARGETS


class QuoteEditorView:
    """Manage the Quote Editor tab and associated Tk variables."""

    def __init__(
        self,
        app: Any,
        parent: tk.Widget,
        *,
        coerce_df_fn: Callable[[PandasDataFrame | None], PandasDataFrame],
        material_price_updater: Callable[[Any, Any, dict[str, float]], bool],
        sugg_to_editor: Mapping[str, Any],
        editor_to_sugg: Mapping[str, Any],
        editor_from_ui: Mapping[str, Any],
    ) -> None:
        self.app = app
        self._coerce_df = coerce_df_fn
        self._material_price_updater = material_price_updater
        self.sugg_to_editor = dict(sugg_to_editor)
        self.editor_to_sugg = dict(editor_to_sugg)
        self.editor_from_ui = dict(editor_from_ui)

        self.scroll = ScrollableFrame(parent)
        self.scroll.pack(fill="both", expand=True)

        self.quote_vars: dict[str, tk.Variable] = {}
        self.param_vars: dict[str, tk.Variable] = {}
        self.rate_vars: dict[str, tk.Variable] = {}
        self.editor_vars: dict[str, tk.Variable] = {}
        self.editor_label_widgets: dict[str, ttk.Label] = {}
        self.editor_label_base: dict[str, str] = {}
        self.editor_value_sources: dict[str, str] = {}
        self.editor_widgets_frame: tk.Widget | None = None
        self.var_material: tk.StringVar | None = None

        settings = getattr(app, "settings", {})
        saved_rate_mode = str(settings.get("rate_mode", "") or "").strip().lower()
        if saved_rate_mode not in {"simple", "detailed"}:
            saved_rate_mode = "detailed"
        self.rate_mode = tk.StringVar(master=app, value=saved_rate_mode)
        self.simple_labor_rate_var = tk.StringVar(master=app)
        self.simple_machine_rate_var = tk.StringVar(master=app)
        self._updating_simple_rates = False

        if hasattr(self.rate_mode, "trace_add"):
            self.rate_mode.trace_add("write", self._on_rate_mode_changed)
        elif hasattr(self.rate_mode, "trace"):
            self.rate_mode.trace("w", lambda *_: self._on_rate_mode_changed())

        def _bind_simple(var: tk.StringVar, kind: str) -> None:
            if hasattr(var, "trace_add"):
                var.trace_add("write", lambda *_: self._on_simple_rate_changed(kind))
            elif hasattr(var, "trace"):
                var.trace("w", lambda *_: self._on_simple_rate_changed(kind))

        _bind_simple(self.simple_labor_rate_var, "labor")
        _bind_simple(self.simple_machine_rate_var, "machine")

        self._editor_set_depth = 0
        self._building_editor = False

    # ------------------------------------------------------------------
    # Populate + interaction helpers
    # ------------------------------------------------------------------

    def populate(self, df: PandasDataFrame) -> None:
        app = self.app
        df = self._coerce_df(df)
        if app.vars_df_full is None:
            _, master_full = _load_master_variables()
            if master_full is not None:
                app.vars_df_full = master_full

        def _ensure_row(
            dataframe: PandasDataFrame, item: str, value: Any, dtype: str = "number"
        ) -> PandasDataFrame:
            mask = dataframe["Item"].astype(str).str.fullmatch(item, case=False)
            if mask.any():
                return dataframe
            return geometry_upsert_var_row(dataframe, item, value, dtype=dtype)

        df = _ensure_row(df, "Scrap Percent (%)", 15.0, dtype="number")
        df = _ensure_row(df, "Plate Length (in)", 12.0, dtype="number")
        df = _ensure_row(df, "Plate Width (in)", 14.0, dtype="number")
        df = _ensure_row(df, "Thickness (in)", 2.0, dtype="number")
        df = _ensure_row(df, "Hole Count (override)", 0, dtype="number")
        df = _ensure_row(df, "Avg Hole Diameter (mm)", 0.0, dtype="number")
        df = _ensure_row(df, "Material", app.default_material_display, dtype="text")
        app.vars_df = df

        parent = self.scroll.inner
        for child in parent.winfo_children():
            child.destroy()

        self.quote_vars.clear()
        self.param_vars.clear()
        self.rate_vars.clear()
        self.editor_vars.clear()
        self.editor_label_widgets.clear()
        self.editor_label_base.clear()
        self.editor_value_sources.clear()

        self._building_editor = True

        self.editor_widgets_frame = parent
        self.editor_widgets_frame.grid_columnconfigure(0, weight=1)

        items_series = df["Item"].astype(str)
        normalized_items = items_series.apply(normalize_item_text)
        qty_mask = normalized_items.isin({"quantity", "qty", "lot size"})
        if qty_mask.any():
            qty_column = typing.cast(PandasSeries, df.loc[qty_mask, "Example Values / Options"])
            qty_raw = qty_column.iloc[0]
            try:
                qty_value = float(str(qty_raw).strip())
            except Exception:
                qty_value = float(app.params.get("Quantity", 1) or 1)
            if math.isnan(qty_value):
                qty_value = float(app.params.get("Quantity", 1) or 1)
            app.params["Quantity"] = max(1, int(round(qty_value)))

        raw_skip_items = {
            "Profit Margin %",
            "Profit Margin",
            "Margin %",
            "Margin",
            "Expedite %",
            "Expedite",
            "Insurance %",
            "Insurance",
            "Vendor Markup %",
            "Vendor Markup",
            "Min Lot Charge",
            "Programmer $/hr",
            "CAM Programmer $/hr",
            "Milling $/hr",
            "Inspection $/hr",
            "Deburr $/hr",
            "Packaging $/hr",
            "Quantity",
            "Qty",
            "Lot Size",
        }
        skip_items = {normalize_item_text(item) for item in raw_skip_items}

        material_lookup: Dict[str, float] = {}
        for _, row_data in df.iterrows():
            item_label = str(row_data.get("Item", ""))
            raw_value = _coerce_float_or_none(row_data.get("Example Values / Options"))
            if raw_value is None:
                continue
            per_g = _price_value_to_per_gram(raw_value, item_label)
            if per_g is None:
                continue
            normalized_label = _normalize_lookup_key(item_label)
            for canonical_key, keywords in MATERIAL_KEYWORDS.items():
                if canonical_key == MATERIAL_OTHER_KEY:
                    continue
                if any((kw and kw in normalized_label) for kw in keywords):
                    material_lookup[canonical_key] = per_g
                    break

        current_row = 0

        quote_frame = ttk.Labelframe(
            self.editor_widgets_frame,
            text="Quote-Specific Variables",
            padding=(10, 5),
        )
        quote_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1

        row_index = 0
        material_choice_var: tk.StringVar | None = None
        material_price_var: tk.StringVar | None = None
        self.var_material = None

        def update_material_price(*_args: Any) -> None:
            self._material_price_updater(
                material_choice_var,
                material_price_var,
                material_lookup,
            )

        def _resolve_column(name: str) -> str:
            def _norm_col(s: str) -> str:
                s = str(s).replace("\u00A0", " ")
                s = re.sub(r"\s+", " ", s).strip().lower()
                return re.sub(r"[^a-z0-9]", "", s)

            target = _norm_col(name)
            column_sources: list[PandasIndex] = []
            if app.vars_df_full is not None:
                column_sources.append(app.vars_df_full.columns)
            column_sources.append(df.columns)
            for columns in column_sources:
                for column in columns:
                    if _norm_col(column) == target:
                        return str(column)
            return name

        full_lookup: dict[str, Any] = {}
        if app.vars_df_full is not None and "Item" in app.vars_df_full.columns:
            full_items = app.vars_df_full["Item"].astype(str)
            for idx, label in enumerate(full_items):
                normalized = normalize_item_text(label)
                full_lookup[normalized] = app.vars_df_full.iloc[idx]

        for _, row_data in df.iterrows():
            item_name = str(row_data.get("Item", "") or "").strip()
            if not item_name:
                continue
            normalized_name = normalize_item_text(item_name)
            if normalized_name in skip_items:
                continue

            dtype_col_name = _resolve_column("Data Type / Input Method")
            value_col_name = _resolve_column("Example Values / Options")

            full_row = full_lookup.get(normalized_name)

            dtype_source = row_data.get(dtype_col_name, "")
            if full_row is not None:
                dtype_source = full_row.get(dtype_col_name, dtype_source)

            initial_raw = row_data.get(value_col_name, "")
            if full_row is not None:
                initial_raw = full_row.get(value_col_name, initial_raw)
            is_missing = False
            if pd is not None:
                try:
                    is_missing = bool(pd.isna(initial_raw))
                except Exception:
                    is_missing = False
            initial_value = "" if is_missing else "" if initial_raw is None else str(initial_raw)

            control_spec = derive_editor_control_spec(dtype_source, initial_raw)
            label_text = item_name
            if full_row is not None and "Variable ID" in full_row:
                var_id = str(full_row.get("Variable ID", "") or "").strip()
                if var_id:
                    label_text = f"{var_id} • {label_text}"
            display_hint = control_spec.display_label.strip()
            if display_hint and display_hint.lower() not in {"number", "text"}:
                label_text = f"{label_text}\n[{display_hint}]"

            row_container = ttk.Frame(quote_frame)
            row_container.grid(
                row=row_index,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=5,
                pady=4,
            )
            row_container.grid_columnconfigure(1, weight=1)

            label_widget = ttk.Label(row_container, text=label_text, wraplength=400)
            label_widget.grid(row=0, column=0, sticky="w", padx=(0, 6))

            control_row = 0
            info_indicator: ttk.Label | None = None
            info_tooltip: CreateToolTip | None = None
            info_text_parts: list[str] = []

            control_container = ttk.Frame(row_container)
            control_container.grid(row=control_row, column=1, sticky="ew", padx=5)
            control_container.grid_columnconfigure(0, weight=1)
            control_container.grid_columnconfigure(1, weight=1)
            control_container.grid_columnconfigure(2, weight=0)

            def _add_info_label(text: str) -> None:
                nonlocal info_indicator, info_tooltip

                text = text.strip()
                if not text:
                    return
                info_text_parts.append(text)
                combined_text = "\n\n".join(info_text_parts)

                if info_indicator is None:
                    info_indicator = ttk.Label(
                        control_container,
                        text="ⓘ",
                        padding=(4, 0),
                        cursor="question_arrow",
                        takefocus=0,
                    )
                    info_indicator.grid(row=control_row, column=2, sticky="nw", padx=(6, 0))
                    info_tooltip = CreateToolTip(info_indicator, combined_text, wraplength=360)
                elif info_tooltip is not None:
                    info_tooltip.update_text(combined_text)

            if normalized_name in {"material"}:
                var = tk.StringVar(master=app, value=app.default_material_display)
                if initial_value:
                    var.set(initial_value)
                normalized_initial = _normalize_lookup_key(var.get())
                for canonical_key, keywords in MATERIAL_KEYWORDS.items():
                    if canonical_key == MATERIAL_OTHER_KEY:
                        continue
                    if any(kw and kw in normalized_initial for kw in keywords):
                        display = MATERIAL_DISPLAY_BY_KEY.get(canonical_key)
                        if display:
                            var.set(display)
                        break
                combo = ttk.Combobox(
                    control_container,
                    textvariable=var,
                    values=MATERIAL_DROPDOWN_OPTIONS,
                    width=32,
                )
                combo.grid(row=0, column=0, sticky="ew")
                combo.bind("<<ComboboxSelected>>", update_material_price)
                var.trace_add("write", update_material_price)
                material_choice_var = var
                self.var_material = var
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            elif re.search(
                r"(Material\s*Price.*(per\s*gram|per\s*g|/g)|Unit\s*Price\s*/\s*g)",
                item_name,
                flags=re.IGNORECASE,
            ):
                var = tk.StringVar(master=app, value=initial_value)
                ttk.Entry(control_container, textvariable=var, width=30).grid(
                    row=0, column=0, sticky="w"
                )
                material_price_var = var
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            elif control_spec.control == "dropdown":
                options = list(control_spec.options) or (
                    [control_spec.entry_value] if control_spec.entry_value else []
                )
                selected = control_spec.entry_value or (options[0] if options else "")
                var = tk.StringVar(master=app, value=selected)
                combo = ttk.Combobox(
                    control_container,
                    textvariable=var,
                    values=options,
                    width=28,
                    state="readonly" if options else "normal",
                )
                combo.grid(row=0, column=0, sticky="ew")
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            elif control_spec.control == "checkbox":
                initial_bool = control_spec.checkbox_state
                bool_var = tk.BooleanVar(master=app, value=bool(initial_bool))
                var = tk.StringVar(master=app, value="True" if initial_bool else "False")

                def _sync_string_from_bool(*_args: Any) -> None:
                    value = "True" if bool_var.get() else "False"
                    if var.get() != value:
                        var.set(value)

                def _sync_bool_from_string(*_args: Any) -> None:
                    current = var.get()
                    parsed = coerce_checkbox_state(current, bool_var.get())
                    if bool_var.get() != parsed:
                        bool_var.set(parsed)

                if hasattr(bool_var, "trace_add"):
                    bool_var.trace_add("write", _sync_string_from_bool)
                else:  # pragma: no cover - legacy Tk fallback
                    bool_var.trace("w", lambda *_: _sync_string_from_bool())

                if hasattr(var, "trace_add"):
                    var.trace_add("write", _sync_bool_from_string)
                else:  # pragma: no cover - legacy Tk fallback
                    var.trace("w", lambda *_: _sync_bool_from_string())

                ttk.Checkbutton(
                    control_container,
                    variable=bool_var,
                    text=control_spec.checkbox_label or "Enabled",
                ).grid(row=0, column=0, sticky="w")
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)
            else:
                entry_value = control_spec.entry_value
                if not entry_value and control_spec.control != "formula":
                    entry_value = initial_value
                var = tk.StringVar(master=app, value=entry_value)
                ttk.Entry(control_container, textvariable=var, width=30).grid(
                    row=0, column=0, sticky="w"
                )
                base_text = (
                    control_spec.base_text.strip()
                    if isinstance(control_spec.base_text, str)
                    else ""
                )
                if base_text:
                    _add_info_label(f"Based on: {base_text}")
                self.quote_vars[item_name] = var
                self._register_editor_field(item_name, var, label_widget)

            why_text = ""
            if full_row is not None and "Why it Matters" in full_row:
                why_text = str(full_row.get("Why it Matters", "") or "").strip()
            elif "Why it Matters" in row_data:
                why_text = str(row_data.get("Why it Matters", "") or "").strip()
            if why_text:
                _add_info_label(why_text)

            swing_text = ""
            if full_row is not None and "Typical Price Swing*" in full_row:
                swing_text = str(full_row.get("Typical Price Swing*", "") or "").strip()
            elif "Typical Price Swing*" in row_data:
                swing_text = str(row_data.get("Typical Price Swing*", "") or "").strip()
            if swing_text:
                _add_info_label(f"Typical swing: {swing_text}")

            row_index += 1

        if material_choice_var is not None and material_price_var is not None:
            existing = _coerce_float_or_none(material_price_var.get())
            if existing is None or abs(existing) < 1e-9:
                update_material_price()

        def create_global_entries(
            parent_frame: tk.Widget,
            keys: Sequence[str],
            data_source: Mapping[str, Any],
            var_dict: dict[str, tk.StringVar],
            columns: int = 2,
        ) -> None:
            for i, key in enumerate(keys):
                row, col = divmod(i, columns)
                label_widget = ttk.Label(parent_frame, text=key)
                label_widget.grid(row=row, column=col * 2, sticky="e", padx=5, pady=2)
                var = tk.StringVar(master=app, value=str(data_source.get(key, "")))
                entry = ttk.Entry(parent_frame, textvariable=var, width=15)
                if "Path" in key:
                    entry.config(width=50)
                entry.grid(row=row, column=col * 2 + 1, sticky="w", padx=5, pady=2)
                var_dict[key] = var
                self._register_editor_field(key, var, label_widget)

        comm_frame = ttk.Labelframe(
            self.editor_widgets_frame,
            text="Global Overrides: Commercial & General",
            padding=(10, 5),
        )
        comm_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1
        comm_keys = [
            "MarginPct",
            "ExpeditePct",
            "VendorMarkupPct",
            "InsurancePct",
            "MinLotCharge",
            "Quantity",
        ]
        create_global_entries(comm_frame, comm_keys, app.params, self.param_vars)

        rates_frame = ttk.Labelframe(
            self.editor_widgets_frame,
            text="Global Overrides: Hourly Rates ($/hr)",
            padding=(10, 5),
        )
        rates_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1

        mode_frame = ttk.Frame(rates_frame)
        mode_frame.grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(mode_frame, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Radiobutton(
            mode_frame,
            text="Simple",
            value="simple",
            variable=self.rate_mode,
        ).grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Radiobutton(
            mode_frame,
            text="Detailed",
            value="detailed",
            variable=self.rate_mode,
        ).grid(row=0, column=2, sticky="w")

        rates_container = ttk.Frame(rates_frame)
        rates_container.grid(row=1, column=0, sticky="ew")
        rates_container.grid_columnconfigure(0, weight=1)

        if self._simple_rate_mode_active():
            self._build_simple_rate_entries(rates_container)
        else:
            create_global_entries(
                rates_container,
                sorted(app.rates.keys()),
                app.rates,
                self.rate_vars,
                columns=2,
            )

        self._building_editor = False

    # ------------------------------------------------------------------
    # Rate helpers
    # ------------------------------------------------------------------

    def _simple_rate_mode_active(self) -> bool:
        mode = str(self.rate_mode.get() or "").strip().lower()
        return mode == "simple"

    def simple_rate_mode_active(self) -> bool:
        """Return ``True`` when the simplified rate mode is active."""

        return self._simple_rate_mode_active()

    def _format_rate_value(self, value: Any) -> str:
        if value in (None, ""):
            return ""
        try:
            num = float(value)
        except Exception:
            return str(value)
        text = f"{num:.3f}".rstrip("0").rstrip(".")
        return text or "0"

    def format_rate_value(self, value: Any) -> str:
        """Public wrapper that mirrors :meth:`_format_rate_value`."""

        return self._format_rate_value(value)

    def _build_simple_rate_entries(self, parent: tk.Widget) -> None:
        app = self.app
        known_keys = set(app.rates.keys()) | LABOR_RATE_KEYS | MACHINE_RATE_KEYS
        for key in sorted(known_keys):
            formatted = self._format_rate_value(app.rates.get(key, ""))
            self.rate_vars[key] = tk.StringVar(master=app, value=formatted)

        container = ttk.Frame(parent)
        container.grid(row=0, column=0, sticky="w")

        ttk.Label(container, text="Labor Rate").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(container, textvariable=self.simple_labor_rate_var, width=15).grid(
            row=0,
            column=1,
            sticky="w",
            padx=5,
            pady=2,
        )
        ttk.Label(container, text="Machine Rate").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(container, textvariable=self.simple_machine_rate_var, width=15).grid(
            row=1,
            column=1,
            sticky="w",
            padx=5,
            pady=2,
        )

        self._sync_simple_rate_fields()

    def _sync_simple_rate_fields(self) -> None:
        if not self._simple_rate_mode_active():
            return

        app = self.app
        self._updating_simple_rates = True
        try:
            for key, var in self.rate_vars.items():
                var.set(self._format_rate_value(app.rates.get(key, "")))

            labor_value = next((app.rates.get(k) for k in LABOR_RATE_KEYS if k in app.rates), None)
            machine_value = next(
                (app.rates.get(k) for k in MACHINE_RATE_KEYS if k in app.rates),
                None,
            )

            self.simple_labor_rate_var.set(self._format_rate_value(labor_value))
            self.simple_machine_rate_var.set(self._format_rate_value(machine_value))
        finally:
            self._updating_simple_rates = False

    def sync_simple_rate_fields(self) -> None:
        """Refresh the simplified rate entry widgets from ``app.rates``."""

        self._sync_simple_rate_fields()

    def _update_rate_group(self, keys: Iterable[str], value: float) -> bool:
        app = self.app
        changed = False
        formatted = self._format_rate_value(value)
        for key in keys:
            previous = _coerce_float_or_none(app.rates.get(key))
            if previous is None or abs(previous - value) > 1e-6:
                changed = True
            app.rates[key] = float(value)
            var = self.rate_vars.get(key)
            if var is None:
                self.rate_vars[key] = tk.StringVar(master=app, value=formatted)
            elif var.get() != formatted:
                var.set(formatted)
        return changed

    def _apply_simple_rates(
        self,
        hint: str | None = None,
        *,
        trigger_reprice: bool = True,
    ) -> None:
        if not self._simple_rate_mode_active() or self._updating_simple_rates:
            return

        labor_value = _coerce_float_or_none(self.simple_labor_rate_var.get())
        machine_value = _coerce_float_or_none(self.simple_machine_rate_var.get())

        changed = False
        if labor_value is not None:
            changed |= self._update_rate_group(LABOR_RATE_KEYS, float(labor_value))
        if machine_value is not None:
            changed |= self._update_rate_group(MACHINE_RATE_KEYS, float(machine_value))

        if changed and trigger_reprice:
            self.app.reprice(hint=hint or "Updated hourly rates.")

    def apply_simple_rates(
        self,
        hint: str | None = None,
        *,
        trigger_reprice: bool = True,
    ) -> None:
        """Public wrapper around :meth:`_apply_simple_rates`."""

        self._apply_simple_rates(hint=hint, trigger_reprice=trigger_reprice)

    def _on_simple_rate_changed(self, kind: str) -> None:
        hint = None
        if kind == "labor":
            hint = "Updated labor rates."
        elif kind == "machine":
            hint = "Updated machine rates."
        self._apply_simple_rates(hint=hint)

    def _on_rate_mode_changed(self, *_: Any) -> None:
        if getattr(self, "_building_editor", False):
            return

        mode = str(self.rate_mode.get() or "").strip().lower()
        app = self.app
        if isinstance(app.settings, dict):
            app.settings["rate_mode"] = mode
            app.llm_services.save_settings(app.settings_path, app.settings)

        try:
            if app.vars_df is not None:
                self.populate(typing.cast(PandasDataFrame, app.vars_df))
            else:
                self.populate(self._coerce_df(None))
        except Exception:
            self.populate(self._coerce_df(None))

    # ------------------------------------------------------------------
    # Editor field helpers
    # ------------------------------------------------------------------

    def _register_editor_field(
        self, label: str, var: tk.Variable, label_widget: ttk.Label | None
    ) -> None:
        if not isinstance(label, str) or not isinstance(var, tk.Variable):
            return
        self.editor_vars[label] = var
        if label_widget is not None:
            self.editor_label_widgets[label] = label_widget
            self.editor_label_base[label] = label
        else:
            self.editor_label_base.setdefault(label, label)
        self.editor_value_sources.pop(label, None)
        self._mark_label_source(label, None)
        self._bind_editor_var(label, var)

    def _bind_editor_var(self, label: str, var: tk.Variable) -> None:
        def _on_write(*_args: object) -> None:
            if self._building_editor or self._editor_set_depth > 0:
                return
            self._update_editor_override_from_label(label, var.get())
            self._mark_label_source(label, "User")
            self.app.reprice(hint=f"Updated {label}.")

        var.trace_add("write", _on_write)

    def mark_dirty(self, hint: str | None = None) -> None:
        self.app._quote_dirty = True
        message = "Quote editor updated."
        if isinstance(hint, str):
            cleaned = hint.strip()
            if cleaned:
                message = cleaned.splitlines()[0]
        try:
            self.app.status_var.set(
                f"{message} Click Generate Quote to refresh totals."
            )
        except Exception:
            pass

    def clear_dirty(self) -> None:
        self.app._quote_dirty = False

    def _mark_label_source(self, label: str, src: str | None) -> None:
        widget = self.editor_label_widgets.get(label)
        base = self.editor_label_base.get(label, label)
        if widget is not None:
            if src == "LLM":
                widget.configure(text=f"{base}  (LLM)", foreground="#1463FF")
            elif src == "User":
                widget.configure(text=f"{base}  (User)", foreground="#22863a")
            else:
                widget.configure(text=base, foreground="")
        if src:
            self.editor_value_sources[label] = src
        else:
            self.editor_value_sources.pop(label, None)

    def _set_editor(self, label: str, value: Any, source_tag: str = "LLM") -> None:
        if self.editor_value_sources.get(label) == "User":
            return
        var = self.editor_vars.get(label)
        if var is None:
            return
        if isinstance(value, float):
            text_value = f"{value:.3f}"
        else:
            text_value = str(value)
        self._editor_set_depth += 1
        try:
            var.set(text_value)
        finally:
            self._editor_set_depth -= 1
        self._mark_label_source(label, source_tag)

    def _fill_editor_if_blank(self, label: str, value: Any, source: str = "GEO") -> None:
        if value is None:
            return
        if self.editor_value_sources.get(label) == "User":
            return
        var = self.editor_vars.get(label)
        if var is None:
            return
        raw = var.get()
        current = str(raw).strip() if raw is not None else ""
        fill = False
        if not current:
            fill = True
        else:
            try:
                fill = float(current) == 0.0
            except Exception:
                fill = False
        if not fill:
            return
        if isinstance(value, (int, float)):
            txt = f"{float(value):.3f}"
        else:
            txt = str(value)
        self._editor_set_depth += 1
        try:
            var.set(txt)
        finally:
            self._editor_set_depth -= 1
        self._mark_label_source(label, source)

    def apply_geo_defaults(self, geo_data: dict[str, Any] | None) -> None:
        defaults = infer_geo_override_defaults(geo_data)
        if not defaults:
            return

        for label, value in defaults.items():
            if value is None:
                continue
            if isinstance(value, str):
                var = self.editor_vars.get(label)
                if var is None:
                    continue
                current = str(var.get() or "").strip()
                if label == "Material":
                    if current and current != self.app.default_material_display:
                        continue
                elif current:
                    continue
                self._set_editor(label, value, "GEO")
                continue

            self._fill_editor_if_blank(label, value, source="GEO")

    def _update_editor_override_from_label(self, label: str, raw_value: str) -> None:
        key = self.editor_to_sugg.get(label)
        if key is None:
            return
        converter = self.editor_from_ui.get(label)
        value: Any | None = None
        if converter is not None:
            try:
                value = converter(raw_value)
            except Exception:
                value = None
        path = key if isinstance(key, tuple) else (key,)
        if value is None:
            self.app._set_user_override_value(path, None)
        else:
            self.app._set_user_override_value(path, value)

    # ------------------------------------------------------------------
    # LLM application helpers
    # ------------------------------------------------------------------

    def apply_llm_to_editor(self, sugg: dict, baseline_ctx: dict) -> None:
        if not isinstance(sugg, dict) or not isinstance(baseline_ctx, dict):
            return
        for key, spec in self.sugg_to_editor.items():
            label, to_ui, _ = spec
            if isinstance(key, tuple):
                root, sub = key
                val = (sugg.get(root) or {}).get(sub)
            else:
                val = sugg.get(key)
            if val is None:
                continue
            try:
                ui_val = to_ui(val)
            except Exception:
                continue
            self._set_editor(label, ui_val, "LLM")

        mults = sugg.get("process_hour_multipliers") or {}
        eff_hours: dict[str, float] = {}
        base_hours = baseline_ctx.get("process_hours")
        if isinstance(base_hours, dict):
            for proc, hours in base_hours.items():
                val = _coerce_float_or_none(hours)
                if val is not None:
                    eff_hours[str(proc)] = float(val)
        bounds_src = None
        if isinstance(self.app.quote_state.bounds, dict):
            bounds_src = self.app.quote_state.bounds
        elif isinstance(baseline_ctx.get("_bounds"), Mapping):
            bounds_src = baseline_ctx.get("_bounds")
        coerced_bounds = coerce_bounds(bounds_src if isinstance(bounds_src, Mapping) else None)
        mult_min_bound = coerced_bounds["mult_min"]
        mult_max_bound = coerced_bounds["mult_max"]
        for proc, mult in mults.items():
            if proc in eff_hours:
                try:
                    base_val = float(eff_hours[proc])
                except Exception:
                    continue
                clamped_mult = clamp(mult, mult_min_bound, mult_max_bound, 1.0)
                try:
                    eff_hours[proc] = base_val * float(clamped_mult)
                except Exception:
                    eff_hours[proc] = base_val
        for proc, (label, scale) in PROC_MULT_TARGETS.items():
            if proc in eff_hours:
                try:
                    derived = eff_hours[proc] * float(scale)
                except Exception:
                    continue
                self._set_editor(label, derived, "LLM")

        app = self.app
        app.effective_process_hours = eff_hours
        app.effective_scrap = float(
            sugg.get("scrap_pct", baseline_ctx.get("scrap_pct", 0.0)) or 0.0
        )
        app.effective_setups = int(sugg.get("setups", baseline_ctx.get("setups", 1)) or 1)
        app.effective_fixture = str(
            sugg.get("fixture", baseline_ctx.get("fixture", "standard")) or "standard"
        )

        if not app._reprice_in_progress:
            app.reprice(hint="LLM adjustments applied.")
