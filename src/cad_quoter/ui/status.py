"""Status bar helper widgets for the CAD Quoter UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class StatusBar:
    """Create and manage a status bar with a ``StringVar`` backing it."""

    def __init__(self, master: tk.Misc, *, initial: str = "Ready") -> None:
        self.variable = tk.StringVar(master=master, value=initial)
        self.label = ttk.Label(
            master,
            textvariable=self.variable,
            relief=tk.SUNKEN,
            anchor="w",
            padding=5,
        )
        self.label.pack(side="bottom", fill="x")

    def set(self, text: str) -> None:
        """Update the status bar text."""

        self.variable.set(text)

    def get(self) -> str:
        """Return the current status bar text."""

        return self.variable.get()
