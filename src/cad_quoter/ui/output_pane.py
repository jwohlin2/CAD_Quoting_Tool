"""Output notebook utilities for the CAD Quoter UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class OutputPane:
    """Manage the Output tab and associated ``Text`` widgets."""

    def __init__(self, parent: tk.Widget) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self._tabs: dict[str, ttk.Frame] = {}
        self.text_widgets: dict[str, tk.Text] = {}

        self._create_text_tab("simplified", "Simplified")
        self._create_text_tab("full", "Full Detail")
        self.select("simplified")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def select(self, name: str) -> None:
        tab = self._tabs.get(name)
        if tab is not None:
            self.notebook.select(tab)

    def get_text_widget(self, name: str) -> tk.Text | None:
        return self.text_widgets.get(name)

    def get_tab(self, name: str) -> ttk.Frame | None:
        return self._tabs.get(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_text_tab(self, name: str, label: str) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=label)
        widget = tk.Text(frame, wrap="word")
        widget.pack(fill="both", expand=True)
        try:
            widget.tag_configure("rcol", tabs=("4.8i right",), tabstyle="tabular")
        except tk.TclError:
            widget.tag_configure("rcol", tabs=("4.8i right",))
        self._tabs[name] = frame
        self.text_widgets[name] = widget
