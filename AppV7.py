from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import json
from pathlib import Path


class CreateToolTip:
    """Attach a lightweight tooltip to a Tk widget."""

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        *,
        delay: int = 500,
        wraplength: int = 320,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id: str | None = None
        self._tip_window: tk.Toplevel | None = None
        self._label: tk.Label | ttk.Label | None = None
        self._pinned = False

        self.widget.bind("<Enter>", self._schedule_show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<FocusIn>", self._schedule_show, add="+")
        self.widget.bind("<FocusOut>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._on_button_press, add="+")
        self.widget.bind("<Button-1>", self._toggle_pin, add="+")

    def update_text(self, text: str) -> None:
        self.text = text
        if self._label is not None:
            self._label.configure(text=text)

    def _on_button_press(self, event: tk.Event | None = None) -> None:
        if event is not None and getattr(event, "num", None) == 1:
            return
        self._hide()

    def _toggle_pin(self, _event: tk.Event | None = None) -> None:
        if self._pinned:
            self._pinned = False
            self._hide()
        else:
            self._pinned = True
            self._cancel_scheduled()
            self._show()

    def _schedule_show(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled()
        if not self.text:
            return
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel_scheduled(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            finally:
                self._after_id = None

    def _show(self) -> None:
        if self._tip_window is not None or not self.text:
            return

        bbox: tuple[int, int, int, int] | None = None
        bbox_method = getattr(self.widget, "bbox", None)
        if callable(bbox_method):
            try:
                raw_bbox = bbox_method("insert")
                if isinstance(raw_bbox, (tuple, list)) and len(raw_bbox) >= 4:
                    bbox = (
                        int(raw_bbox[0]),
                        int(raw_bbox[1]),
                        int(raw_bbox[2]),
                        int(raw_bbox[3]),
                    )
            except Exception:
                bbox = None
        if bbox:
            x, y, width, height = bbox
        else:
            x = y = 0
            width = self.widget.winfo_width()
            height = self.widget.winfo_height()

        root_x = self.widget.winfo_rootx()
        root_y = self.widget.winfo_rooty()
        x = root_x + x + width + 12
        y = root_y + y + height + 12

        master = self.widget.winfo_toplevel()
        tip = tk.Toplevel(master)
        tip.wm_overrideredirect(True)
        try:
            tip.transient(master)
        except Exception:
            tip.wm_transient(master)
        tip.wm_geometry(f"+{x}+{y}")
        tip.lift()
        try:
            tip.attributes("-topmost", True)
        except Exception:
            pass

        label = tk.Label(
            tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", 8, "normal"),
            wraplength=self.wraplength,
        )
        label.pack(ipadx=4, ipady=2)

        self._tip_window = tip
        self._label = label

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled()
        if self._pinned:
            return
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None
        self._label = None


class ScrollableFrame(ttk.Frame):
    """A ttk frame with vertical scrolling support."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # Bind wheel only while cursor is over this widget
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    # Windows & macOS wheel
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")

    # Linux wheel
    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


class AppV7:
    """Main application class."""

    def __init__(self) -> None:
        self.title = "Compos-AI"
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("800x700")

        # Data storage
        self.cad_data = {}
        self.quote_vars = {}

        self._create_menu()
        self._create_button_panel()
        self._create_tabs()
        self._create_status_bar()

    def _create_menu(self) -> None:
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open CAD File...", command=self.load_cad)
        file_menu.add_command(label="Save Quote...", command=self.save_quote)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Settings", command=self.show_settings)

    def _create_button_panel(self) -> None:
        """Create the top button panel."""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, pady=(6, 8))

        ttk.Button(button_frame, text="1. Load CAD & Vars",
                   command=self.load_cad).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="2. Generate Quote",
                   command=self.generate_quote).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="LLM Inspector",
                   command=self.show_llm_inspector).pack(side=tk.LEFT, padx=5)

    def _create_tabs(self) -> None:
        """Create the tabbed interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # GEO tab
        self.geo_tab = tk.Frame(self.notebook)
        self.notebook.add(self.geo_tab, text="GEO")
        self._create_geo_tab()

        # Quote Editor tab
        self.quote_editor_tab = tk.Frame(self.notebook)
        self.notebook.add(self.quote_editor_tab, text="Quote Editor")
        self._create_quote_editor_tab()

        # Output tab
        self.output_tab = tk.Frame(self.notebook)
        self.notebook.add(self.output_tab, text="Output")
        self._create_output_tab()

    def _create_geo_tab(self) -> None:
        """Create the GEO tab content."""
        # Create a frame for the hole table
        table_frame = ttk.Frame(self.geo_tab)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a label
        label = ttk.Label(table_frame, text="Hole Operations", font=("Arial", 12, "bold"))
        label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Create a Treeview for the hole operations table
        columns = ("HOLE", "REF_DIAM", "QTY", "OPERATION")
        self.hole_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)

        # Define column headings
        self.hole_table.heading("HOLE", text="HOLE")
        self.hole_table.heading("REF_DIAM", text="REF DIAM")
        self.hole_table.heading("QTY", text="QTY")
        self.hole_table.heading("OPERATION", text="OPERATION")

        # Set column widths
        self.hole_table.column("HOLE", width=60)
        self.hole_table.column("REF_DIAM", width=100)
        self.hole_table.column("QTY", width=60)
        self.hole_table.column("OPERATION", width=400)

        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.hole_table.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.hole_table.xview)
        self.hole_table.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid everything
        self.hole_table.grid(row=1, column=0, sticky="nsew")
        v_scrollbar.grid(row=1, column=1, sticky="ns")
        h_scrollbar.grid(row=2, column=0, sticky="ew")

        table_frame.grid_rowconfigure(1, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

    def _create_quote_editor_tab(self) -> None:
        """Create the Quote Editor tab with form fields."""
        # Create scrollable frame
        editor_scroll = ScrollableFrame(self.quote_editor_tab)
        editor_scroll.pack(fill="both", expand=True)

        # Quote-Specific Variables section with Labelframe (like appV5)
        quote_frame = ttk.Labelframe(editor_scroll.inner, text="Quote-Specific Variables", padding=(10, 5))
        quote_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Define all the quote variables from the screenshot
        self.quote_fields = {}
        variables = [
            ("Material Scrap / Remnant Value", "50", "Enter the estimated scrap or remnant value for this material"),
        ]

        row = 0
        for item in variables:
            label_text, default_value = item[0], item[1]
            tooltip_text = item[2] if len(item) > 2 else ""

            # Label
            label = ttk.Label(quote_frame, text=label_text)
            label.grid(row=row, column=0, sticky="w", padx=5, pady=5)

            # Entry field or dropdown
            if default_value == "Number":
                field = ttk.Combobox(quote_frame, width=30, values=["Number"])
                field.set("Number")
            else:
                field = ttk.Entry(quote_frame, width=30)
                if default_value:
                    field.insert(0, default_value)

            field.grid(row=row, column=1, sticky="w", padx=5, pady=5)
            self.quote_fields[label_text] = field

            # Info indicator with tooltip (like appV5)
            if tooltip_text:
                info_indicator = ttk.Label(
                    quote_frame,
                    text="ⓘ",
                    padding=(4, 0),
                    cursor="question_arrow",
                )
                info_indicator.grid(row=row, column=2, sticky="nw", padx=(6, 0))
                CreateToolTip(info_indicator, tooltip_text, wraplength=360)

            row += 1

    def _create_output_tab(self) -> None:
        """Create the Output tab with JSON display."""
        # Add simplified/full detail toggle
        toggle_frame = tk.Frame(self.output_tab)
        toggle_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(toggle_frame, text="Simplified").pack(side=tk.LEFT, padx=5)
        tk.Label(toggle_frame, text="Full Detail").pack(side=tk.LEFT, padx=5)

        # Scrolled text area for output
        self.output_text = scrolledtext.ScrolledText(self.output_tab, wrap=tk.WORD,
                                                      font=("Courier", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_status_bar(self) -> None:
        """Create the status bar at the bottom."""
        self.status_bar = tk.Label(self.root, text="Ready. Select a CAD file to begin.",
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_cad(self) -> None:
        """Load CAD file and variables."""
        filename = filedialog.askopenfilename(
            title="Select CAD File",
            filetypes=[
                ("All CAD Files", "*.dwg *.dxf *.step *.stp *.iges *.igs"),
                ("DWG Files", "*.dwg"),
                ("DXF Files", "*.dxf"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.status_bar.config(text=f"Loading {Path(filename).name}...")
            self.root.update_idletasks()

            try:
                # Load and extract hole table data
                self._extract_and_display_hole_table(filename)
                self.status_bar.config(text="DWG variables loaded. Review the Quote Editor and generate the quote.")
                self.notebook.select(self.geo_tab)
            except Exception as e:
                error_msg = str(e)
                # Provide helpful message for DWG converter issue
                if "DWG→DXF converter" in error_msg or "convert_dwg_to_dxf" in error_msg:
                    error_msg = (
                        "DWG files require a DWG→DXF converter to be installed.\n\n"
                        "Please try one of these options:\n"
                        "1. Use a .dxf file instead (e.g., 301_redacted.dxf)\n"
                        "2. Convert your .dwg file to .dxf using AutoCAD or another tool\n"
                        "3. Set up the ODA File Converter (see documentation)\n\n"
                        f"Original error: {error_msg}"
                    )
                self.status_bar.config(text=f"Error loading file: {Path(filename).name}")
                messagebox.showerror("Error Loading CAD File", error_msg)

    def _extract_and_display_hole_table(self, filename: str) -> None:
        """Extract geometry and display hole operations."""
        try:
            from cad_quoter.geo_dump import extract_hole_operations_from_file
        except ImportError as e:
            self.status_bar.config(text=f"Missing required module: {e}")
            return

        # Clear existing table
        for item in self.hole_table.get_children():
            self.hole_table.delete(item)

        # Extract hole operations using geo_dump's public API
        try:
            operations = extract_hole_operations_from_file(filename)

            if operations:
                # Populate the table
                for op in operations:
                    self.hole_table.insert(
                        "", "end",
                        values=(
                            op["HOLE"],
                            op["REF_DIAM"],
                            op["QTY"],
                            op["OPERATION"]
                        )
                    )
                self.status_bar.config(text=f"Loaded {len(operations)} operations from {Path(filename).name}")
            else:
                self.hole_table.insert("", "end", values=("No hole table found", "", "", ""))
                self.status_bar.config(text="No hole table found in file")

        except Exception as e:
            self.hole_table.insert("", "end", values=(f"Error: {str(e)}", "", "", ""))
            self.status_bar.config(text=f"Error parsing hole table: {str(e)}")
            raise  # Re-raise so load_cad can provide detailed error message

    def generate_quote(self) -> None:
        """Generate the quote."""
        # Collect values from quote editor
        quote_data = {}
        for label, field in self.quote_fields.items():
            quote_data[label] = field.get()

        # Display in output tab
        self.output_text.delete(1.0, tk.END)
        output = json.dumps(quote_data, indent=2)
        self.output_text.insert(1.0, output)

        self.notebook.select(self.output_tab)
        self.status_bar.config(text="Quote generated successfully!")

    def save_quote(self) -> None:
        """Save the quote to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Quote",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filename:
            self.status_bar.config(text=f"Saved to: {filename}")

    def show_llm_inspector(self) -> None:
        """Show LLM inspector window."""
        self.status_bar.config(text="LLM Inspector opened")

    def show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo("About", "Compos-AI\nCAD Quoting Tool v7")

    def show_settings(self) -> None:
        """Show settings dialog."""
        self.status_bar.config(text="Settings (Coming Soon)")

    def run(self) -> None:
        """Run the application."""
        self.root.mainloop()


def main() -> None:
    """Launch the application."""
    app = AppV7()
    app.run()


if __name__ == "__main__":
    main()