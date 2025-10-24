"""Main menu construction helpers for the desktop UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import cad_quoter.geometry as geometry
from cad_quoter.config import append_debug_log
from cad_quoter.ui import session_io


class MainMenu:
    """Construct the top-level menu bar for the application."""

    def __init__(self, app: tk.Misc) -> None:
        self.app = app
        self.menubar = tk.Menu(app)

    def build(self) -> tk.Menu:
        """Populate and attach the menu bar to the application window."""

        self.app.config(menu=self.menubar)
        self._build_file_menu()
        self._build_help_menu()
        self._build_tools_menu()
        return self.menubar

    # ------------------------------------------------------------------
    # Menu construction helpers
    # ------------------------------------------------------------------

    def _build_file_menu(self) -> None:
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Overrides...", command=self.app.load_overrides)
        file_menu.add_command(label="Save Overrides...", command=self.app.save_overrides)
        file_menu.add_separator()
        file_menu.add_command(
            label="Import Quote Session...",
            command=lambda: session_io.import_quote_session(self.app),
        )
        file_menu.add_command(
            label="Export Quote Session...",
            command=lambda: session_io.export_quote_session(self.app),
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Set Material Vendor CSV...",
            command=self.app.set_material_vendor_csv,
        )
        file_menu.add_command(
            label="Clear Material Vendor CSV",
            command=self.app.clear_material_vendor_csv,
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.app.quit)

    def _build_help_menu(self) -> None:
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="Diagnostics",
            command=lambda: messagebox.showinfo(
                "Diagnostics", geometry.get_import_diagnostics_text()
            ),
        )

    def _build_tools_menu(self) -> None:
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=tools_menu)

        def _trigger_generate_quote() -> None:
            append_debug_log("", "[UI] Tools > Generate Quote (debug)")
            try:
                self.app.status_var.set("Generating quote (via Tools menu)...")
            except Exception:
                pass
            self.app.gen_quote()

        tools_menu.add_command(
            label="Generate Quote (debug)",
            command=_trigger_generate_quote,
            accelerator="Ctrl+G",
        )
        try:
            self.app.bind_all("<Control-g>", lambda *_: _trigger_generate_quote())
        except Exception:
            pass
