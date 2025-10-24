"""Layout helpers that assemble the primary Tk widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class MainNotebook:
    """Create the root notebook and expose commonly used tabs."""

    def __init__(self, master: tk.Misc) -> None:
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)

        self.geo_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.geo_tab, text="GEO")

        self.editor_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.editor_tab, text="Quote Editor")

        self.output_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.output_tab, text="Output")

        # LLM tab is not attached by default but we keep the frame available for
        # helpers that expect to populate it.
        self.llm_tab = ttk.Frame(master)

    def select(self, tab: tk.Widget) -> None:
        """Proxy to ``ttk.Notebook.select`` for convenience."""

        self.notebook.select(tab)
