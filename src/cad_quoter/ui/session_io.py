"""Helper functions for saving and loading quote sessions from the UI."""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from cad_quoter.app.effective import ensure_accept_flags
from cad_quoter.ui.tk_compat import filedialog, messagebox
from cad_quoter.app.optional_loaders import pd
from cad_quoter.domain import QuoteState


def _exportable_vars_records(app: Any) -> list[dict[str, Any]]:
    """Build a JSON-serialisable snapshot of the current variables dataframe."""

    vars_df = getattr(app, "vars_df", None)
    if vars_df is None:
        return []

    try:
        df_snapshot = vars_df.copy(deep=True)  # type: ignore[call-arg]
    except TypeError:
        df_snapshot = vars_df.copy()  # type: ignore[call-arg]
    except Exception:
        df_snapshot = vars_df

    try:
        quote_vars = getattr(app, "quote_vars", {})
        for item_name, string_var in quote_vars.items():
            mask = df_snapshot["Item"] == item_name
            if mask.any():
                df_snapshot.loc[mask, "Example Values / Options"] = string_var.get()
    except Exception:
        # The UI state is best-effort; if something goes wrong we still want
        # to produce a payload that roughly reflects the current dataframe.
        pass

    records: list[dict[str, Any]] = []
    for _, row in df_snapshot.iterrows():
        record: dict[str, Any] = {}
        for column, value in row.items():
            column_name = str(column)
            value_is_missing = False
            if pd is not None and hasattr(pd, "isna"):
                try:
                    value_is_missing = bool(pd.isna(value))
                except Exception:
                    value_is_missing = False
            if value_is_missing:
                record[column_name] = None
            elif hasattr(value, "item"):
                try:
                    record[column_name] = value.item()
                except Exception:
                    record[column_name] = value
            else:
                record[column_name] = value
        records.append(record)
    return records


def export_quote_session(app: Any) -> None:
    """Persist the current quote session to disk using a save-as dialog."""

    if getattr(app, "vars_df", None) is None:
        messagebox.showinfo("Export Quote Session", "Load a quote before exporting the session.")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("Quote Session", "*.json"), ("JSON", "*.json"), ("All", "*.*")],
        initialfile="quote_session.json",
    )
    if not path:
        return

    status_var = getattr(app, "status_var", None)
    previous_status = status_var.get() if status_var is not None else ""
    try:
        app.apply_overrides()
    finally:
        if status_var is not None:
            status_var.set(previous_status)

    ui_vars = {label: var.get() for label, var in getattr(app, "quote_vars", {}).items()}
    quote_state = getattr(app, "quote_state", None)
    if quote_state is not None and ui_vars:
        quote_state.ui_vars = dict(ui_vars)
    if quote_state is not None:
        quote_state.rates = dict(getattr(app, "rates", {}))
        geo = getattr(app, "geo", None)
        if geo:
            quote_state.geo = dict(geo)

    session_payload = {
        "version": 1,
        "exported_at": time.time(),
        "params": dict(getattr(app, "params", {})),
        "rates": dict(getattr(app, "rates", {})),
        "geo": dict(getattr(app, "geo", {}) or {}),
        "geo_context": dict(getattr(app, "geo_context", {}) or {}),
        "vars_df": _exportable_vars_records(app),
        "quote_state": quote_state.to_dict() if quote_state is not None else {},
        "llm": {
            "enabled": bool(getattr(app, "llm_enabled", None).get() if getattr(app, "llm_enabled", None) else False),
            "apply_adjustments": bool(getattr(app, "apply_llm_adj", None).get() if getattr(app, "apply_llm_adj", None) else False),
            "model_path": getattr(app, "llm_model_path", None).get().strip() if getattr(app, "llm_model_path", None) else "",
            "thread_limit": getattr(app, "llm_thread_limit", None).get().strip() if getattr(app, "llm_thread_limit", None) else "",
        },
        "status": status_var.get() if status_var is not None else "",
    }

    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(session_payload, handle, indent=2)
        messagebox.showinfo("Export Quote Session", f"Session saved to:\n{path}")
        if status_var is not None:
            status_var.set(f"Quote session exported to {path}")
    except Exception as exc:
        messagebox.showerror("Export Quote Session", f"Failed to export session:\n{exc}")
        if status_var is not None:
            status_var.set("Failed to export quote session.")


def import_quote_session(app: Any) -> None:
    """Load a previously exported quote session into the running application."""

    path = filedialog.askopenfilename(
        title="Import Quote Session",
        filetypes=[("Quote Session", "*.json"), ("JSON", "*.json"), ("All", "*.*")],
    )
    if not path:
        return

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        messagebox.showerror("Import Quote Session", f"Failed to read session:\n{exc}")
        status_var = getattr(app, "status_var", None)
        if status_var is not None:
            status_var.set("Failed to import quote session.")
        return

    if not isinstance(payload, dict):
        messagebox.showerror("Import Quote Session", "Session file is not a valid JSON object.")
        status_var = getattr(app, "status_var", None)
        if status_var is not None:
            status_var.set("Invalid quote session file.")
        return

    params_template = copy.deepcopy(getattr(app, "default_params_template", {}))
    if not isinstance(params_template, dict):
        params_template = {}
    params_payload = payload.get("params")
    params = dict(params_template)
    if isinstance(params_payload, dict):
        params.update(params_payload)
    app.params = params

    rates_template = copy.deepcopy(getattr(app, "default_rates_template", {}))
    if not isinstance(rates_template, dict):
        rates_template = {}
    rates_payload = payload.get("rates")
    rates = dict(rates_template)
    if isinstance(rates_payload, dict):
        rates.update(rates_payload)
    app.rates = rates

    llm_payload = payload.get("llm") or {}
    if isinstance(llm_payload, dict):
        llm_enabled = getattr(app, "llm_enabled", None)
        if llm_enabled is not None:
            llm_enabled.set(bool(llm_payload.get("enabled", True)))
        apply_adj = getattr(app, "apply_llm_adj", None)
        if apply_adj is not None:
            apply_adj.set(bool(llm_payload.get("apply_adjustments", True)))
        model_path = llm_payload.get("model_path")
        llm_model_path = getattr(app, "llm_model_path", None)
        if isinstance(model_path, str) and llm_model_path is not None:
            llm_model_path.set(model_path)
        thread_limit = llm_payload.get("thread_limit")
        llm_thread_limit = getattr(app, "llm_thread_limit", None)
        if llm_thread_limit is not None:
            if isinstance(thread_limit, str):
                llm_thread_limit.set(thread_limit)
            elif isinstance(thread_limit, (int, float)):
                try:
                    llm_thread_limit.set(str(int(thread_limit)))
                except Exception:
                    llm_thread_limit.set("")
        if hasattr(app, "_sync_llm_thread_limit"):
            try:
                app._sync_llm_thread_limit(persist=True)
            except Exception:
                pass

    geo_payload = payload.get("geo")
    app.geo = dict(geo_payload) if isinstance(geo_payload, dict) else {}
    geo_context_payload = payload.get("geo_context")
    if isinstance(geo_context_payload, dict):
        app.geo_context = dict(geo_context_payload)
    else:
        app.geo_context = dict(app.geo)

    vars_payload = payload.get("vars_df")
    has_records = isinstance(vars_payload, list) and len(vars_payload) > 0
    if isinstance(vars_payload, list) and pd is not None and hasattr(pd, "DataFrame"):
        try:
            frame_builder = getattr(pd.DataFrame, "from_records", None)
            if callable(frame_builder):
                app.vars_df = frame_builder(vars_payload)
            else:
                app.vars_df = pd.DataFrame(vars_payload)
        except Exception:
            app.vars_df = None
    else:
        app.vars_df = None

    quote_state_payload = payload.get("quote_state")
    try:
        app.quote_state = QuoteState.from_dict(quote_state_payload)
    except Exception as exc:
        messagebox.showerror("Import Quote Session", f"Quote state invalid:\n{exc}")
        status_var = getattr(app, "status_var", None)
        if status_var is not None:
            status_var.set("Failed to import quote session.")
        return

    ensure_accept_flags(app.quote_state)
    app.quote_state.rates = dict(app.rates)
    if not getattr(app.quote_state, "geo", None) and app.geo:
        app.quote_state.geo = dict(app.geo)

    if app.geo:
        try:
            app._log_geo(app.geo)
        except Exception:
            pass

    try:
        if getattr(app, "vars_df", None) is not None:
            app._populate_editor_tab(app.vars_df)
        else:
            app._populate_editor_tab(None)
    except Exception:
        pass

    for key, var in getattr(app, "param_vars", {}).items():
        try:
            var.set(str(app.params.get(key, "")))
        except Exception:
            continue

    if hasattr(app, "_simple_rate_mode_active") and app._simple_rate_mode_active():
        if hasattr(app, "_sync_simple_rate_fields"):
            try:
                app._sync_simple_rate_fields()
            except Exception:
                pass
    else:
        for key, var in getattr(app, "rate_vars", {}).items():
            try:
                var.set(app._format_rate_value(app.rates.get(key, "")))
            except Exception:
                continue

    status_var = getattr(app, "status_var", None)
    if status_var is not None:
        status_var.set(f"Quote session imported from {path}")

    if has_records and getattr(app, "vars_df", None) is not None:
        try:
            empty = getattr(app.vars_df, "empty", True)
        except Exception:
            empty = False
        if not empty and hasattr(app, "gen_quote"):
            try:
                app.gen_quote(reuse_suggestions=True)
            except Exception:
                pass
