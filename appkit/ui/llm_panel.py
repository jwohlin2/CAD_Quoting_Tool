"""UI helpers for the LLM tab and inspector widgets."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from appkit.llm_adapter import infer_shop_overrides_from_geo
from appkit.ui.tk_compat import (
    _ensure_tk,
    filedialog,
    messagebox,
    scrolledtext,
    tk,
    ttk,
)
from cad_quoter.config import logger
from cad_quoter.utils import jdump


def build_llm_tab(app: Any, parent: tk.Misc) -> None:
    """Populate the LLM tab widgets inside ``parent``."""

    row = 0
    ttk.Checkbutton(
        parent,
        text="Enable LLM (Qwen via llama-cpp, offline)",
        variable=app.llm_enabled,
    ).grid(row=row, column=0, sticky="w", pady=(6, 2))
    row += 1

    ttk.Label(parent, text="Qwen GGUF model path").grid(
        row=row, column=0, sticky="e", padx=5, pady=3
    )
    ttk.Entry(parent, textvariable=app.llm_model_path, width=80).grid(
        row=row, column=1, sticky="w", padx=5, pady=3
    )
    ttk.Button(parent, text="Browse...", command=app._pick_model).grid(
        row=row, column=2, padx=5
    )
    row += 1

    validate_cmd = app.register(app._validate_thread_limit)
    ttk.Label(parent, text="Max CPU threads (blank = auto)").grid(
        row=row, column=0, sticky="e", padx=5, pady=3
    )
    ttk.Entry(
        parent,
        textvariable=app.llm_thread_limit,
        width=12,
        validate="key",
        validatecommand=(validate_cmd, "%P"),
    ).grid(row=row, column=1, sticky="w", padx=5, pady=3)
    ttk.Label(
        parent,
        text="Lower this if llama.cpp overwhelms your machine.",
    ).grid(row=row, column=2, sticky="w", padx=5, pady=3)
    row += 1

    ttk.Checkbutton(
        parent,
        text="Apply LLM adjustments to params",
        variable=app.apply_llm_adj,
    ).grid(row=row, column=0, sticky="w", pady=(0, 6))
    row += 1

    ttk.Button(
        parent,
        text="Run LLM on current GEO",
        command=app.run_llm,
    ).grid(row=row, column=0, sticky="w", padx=5, pady=6)
    row += 1

    app.llm_txt = tk.Text(parent, wrap="word", height=24)
    app.llm_txt.grid(row=row, column=0, columnspan=3, sticky="nsew")
    parent.grid_columnconfigure(1, weight=1)
    parent.grid_rowconfigure(row, weight=1)


def pick_llm_model(app: Any) -> None:
    """Prompt the user to select a GGUF model path."""

    path = filedialog.askopenfilename(
        title="Choose Qwen *.gguf",
        filetypes=[("GGUF", "*.gguf"), ("All", "*.*")],
    )
    if not path:
        return

    app.llm_model_path.set(path)
    os.environ["QWEN_GGUF_PATH"] = path


def run_llm(app: Any) -> None:
    """Invoke the LLM helper for the current GEO context."""

    app.llm_txt.delete("1.0", "end")
    if not app.llm_enabled.get():
        app.llm_txt.insert("end", "LLM disabled (toggle ON to use it).\n")
        return
    if not app.geo:
        messagebox.showinfo("LLM", "Load a CAD first so we have GEO context.")
        return

    model_path = app.llm_model_path.get().strip() or os.environ.get("QWEN_GGUF_PATH", "")
    if not (model_path and Path(model_path).is_file() and model_path.lower().endswith(".gguf")):
        app.llm_txt.insert(
            "end",
            "No GGUF model found. Put one in D:\\CAD_Quoting_Tool\\models or set QWEN_GGUF_PATH.\n",
        )
        return

    os.environ["QWEN_GGUF_PATH"] = model_path
    try:
        output = infer_shop_overrides_from_geo(
            app.geo,
            params=app.params,
            rates=app.rates,
        )
    except Exception as exc:  # pragma: no cover - UI level guard
        logger.exception("LLM override inference failed")
        app.llm_txt.insert("end", f"LLM error: {exc}\n")
        return

    app.llm_txt.insert("end", jdump(output, default=None))
    if not app.apply_llm_adj.get() or not isinstance(output, dict):
        return

    adjustments = output.get("LLM_Adjustments", {})
    try:
        app.params["MarginPct"] += float(adjustments.get("MarginPct_add", 0.0) or 0.0)
        app.params["ConsumablesFlat"] += float(
            adjustments.get("ConsumablesFlat_add", 0.0) or 0.0
        )
        for key, var in app.param_vars.items():
            var.set(str(app.params.get(key, "")))
        messagebox.showinfo("LLM", "Applied LLM adjustments to parameters.")
    except Exception:
        pass


def open_llm_inspector(app: Any) -> None:
    """Open the LLM debug snapshot viewer."""

    try:
        _ensure_tk("LLM Inspector")
    except RuntimeError as exc:  # pragma: no cover - headless guard
        logger.error("Cannot open LLM Inspector: %s", exc)
        return

    debug_dir = Path(__file__).with_name("llm_debug")
    files = sorted(debug_dir.glob("llm_snapshot_*.json"))
    if not files:
        messagebox.showinfo("LLM Inspector", "No snapshots yet.")
        return

    latest = files[-1]
    try:
        raw = latest.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            shown = jdump(data, default=None)
        except Exception:
            shown = raw
    except Exception as exc:
        messagebox.showerror("LLM Inspector", f"Failed to read: {exc}")
        return

    window = tk.Toplevel(app)
    window.title(f"LLM Inspector — {latest.name}")
    window.geometry("900x700")

    text = scrolledtext.ScrolledText(window, wrap="word")
    text.pack(fill="both", expand=True)
    text.insert("1.0", shown)
    text.configure(state="disabled")


__all__ = ["build_llm_tab", "pick_llm_model", "run_llm", "open_llm_inspector"]
