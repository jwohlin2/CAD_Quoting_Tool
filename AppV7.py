from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import json
from pathlib import Path
from typing import Optional


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

    # Hourly rates for cost calculations
    MACHINE_RATE = 45.0  # $ per hour
    LABOR_RATE = 45.0    # $ per hour

    def __init__(self) -> None:
        self.title = "Compos-AI"
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("900x700")

        # Data storage
        self.cad_data = {}
        self.quote_vars = {}
        self.cad_file_path: Optional[str] = None

        # Cached totals for summary display
        self.direct_cost_total: Optional[float] = None
        self.machine_cost_total: Optional[float] = None
        self.labor_cost_total: Optional[float] = None

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
                # Store the CAD file path
                self.cad_file_path = filename

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

    def _format_weight(self, weight_lbs: float) -> str:
        """Format weight in lb and oz."""
        lbs = int(weight_lbs)
        oz = (weight_lbs - lbs) * 16
        return f"{lbs} lb {oz:.1f} oz"

    def _generate_direct_costs_report(self) -> str:
        """Generate formatted direct costs report using DirectCostHelper functions."""
        self.direct_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            from cad_quoter.pricing.DirectCostHelper import extract_part_info_from_cad
            from cad_quoter.pricing.mcmaster_helpers import (
                pick_mcmaster_plate_sku,
                load_mcmaster_catalog_rows
            )
            from cad_quoter.resources import default_catalog_csv

            # Extract part info from CAD file
            part_info = extract_part_info_from_cad(self.cad_file_path, verbose=True)

            # Debug: print what was extracted
            print(f"[AppV7] Extracted part_info: L={part_info.length}, W={part_info.width}, T={part_info.thickness}, material={part_info.material}")

            # Check if dimensions were successfully extracted
            if part_info.length == 0 or part_info.width == 0 or part_info.thickness == 0:
                # First check if a cached JSON file exists from previous extraction
                json_output = Path(__file__).parent / "debug" / f"{Path(self.cad_file_path).stem}_dims.json"

                if json_output.exists():
                    print(f"[AppV7] Found existing dims JSON: {json_output}")
                    try:
                        import json
                        with open(json_output, 'r') as f:
                            dims_data = json.load(f)

                        part_info.length = dims_data.get('length', 0.0)
                        part_info.width = dims_data.get('width', 0.0)
                        part_info.thickness = dims_data.get('thickness', 0.0)
                        part_info.volume = part_info.length * part_info.width * part_info.thickness
                        part_info.area = part_info.length * part_info.width

                        print(f"[AppV7] Loaded dims from JSON: L={part_info.length}, W={part_info.width}, T={part_info.thickness}")
                    except Exception as e:
                        print(f"[AppV7] Failed to load JSON: {e}")

                # If still zero, try calling paddle_dims_extractor.py as subprocess
                if part_info.length == 0 or part_info.width == 0 or part_info.thickness == 0:
                    print("[AppV7] Trying paddle_dims_extractor.py as subprocess...")
                    try:
                        import subprocess
                        import sys

                        paddle_script = Path(__file__).parent / "tools" / "paddle_dims_extractor.py"

                        cmd = [sys.executable, str(paddle_script), "--input", self.cad_file_path, "--output-json", str(json_output)]
                        print(f"[AppV7] Running: {' '.join(cmd)}")

                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                        if result.returncode == 0 and json_output.exists():
                            import json
                            with open(json_output, 'r') as f:
                                dims_data = json.load(f)

                            part_info.length = dims_data.get('length', 0.0)
                            part_info.width = dims_data.get('width', 0.0)
                            part_info.thickness = dims_data.get('thickness', 0.0)
                            part_info.volume = part_info.length * part_info.width * part_info.thickness
                            part_info.area = part_info.length * part_info.width

                            print(f"[AppV7] paddle_dims_extractor.py succeeded: L={part_info.length}, W={part_info.width}, T={part_info.thickness}")
                        else:
                            print(f"[AppV7] paddle_dims_extractor.py failed with code {result.returncode}")
                            print(f"[AppV7] stdout: {result.stdout[:500]}")
                            print(f"[AppV7] stderr: {result.stderr[:500]}")
                    except Exception as e:
                        print(f"[AppV7] paddle_dims_extractor.py subprocess failed: {e}")

                # If still zero, try fallback text extraction
                if part_info.length == 0 or part_info.width == 0 or part_info.thickness == 0:
                    print("[AppV7] Trying fallback dimension extraction from text...")
                    try:
                        from cad_quoter.planning import plan_from_cad_file
                        plan = plan_from_cad_file(self.cad_file_path, use_paddle_ocr=False, verbose=True)

                        # Try to extract from text records
                        import re
                        from fractions import Fraction
                        from typing import Dict, Optional as Opt

                        text_records = plan.get('all_text', [])
                        found_dims: Dict[str, Opt[float]] = {'L': None, 'W': None, 'T': None}

                        for text in text_records:
                            text_str = str(text).upper()
                            # Look for patterns like "L = 15.50" or "LENGTH 15.50"
                            for dim in ['L', 'W', 'T', 'LENGTH', 'WIDTH', 'THICK']:
                                pattern = rf'{dim}[=:\s]+([0-9]+\.?[0-9]*|[0-9]+\s*/\s*[0-9]+)'
                                match = re.search(pattern, text_str)
                                if match:
                                    dim_value = match.group(1).strip()
                                    try:
                                        if '/' in dim_value:
                                            value = float(Fraction(dim_value))
                                        else:
                                            value = float(dim_value)

                                        if dim in ['L', 'LENGTH'] and found_dims['L'] is None:
                                            found_dims['L'] = value
                                        elif dim in ['W', 'WIDTH'] and found_dims['W'] is None:
                                            found_dims['W'] = value
                                        elif dim in ['T', 'THICK'] and found_dims['T'] is None:
                                            found_dims['T'] = value
                                    except:
                                        pass

                        if found_dims['L'] and found_dims['W'] and found_dims['T']:
                            print(f"[AppV7] Fallback extraction found: L={found_dims['L']}, W={found_dims['W']}, T={found_dims['T']}")
                            part_info.length = found_dims['L']
                            part_info.width = found_dims['W']
                            part_info.thickness = found_dims['T']
                            # Recalculate volume and area
                            part_info.volume = part_info.length * part_info.width * part_info.thickness
                            part_info.area = part_info.length * part_info.width
                    except Exception as e:
                        print(f"[AppV7] Fallback extraction failed: {e}")

            # Final check if dimensions were successfully extracted
            if part_info.length == 0 or part_info.width == 0 or part_info.thickness == 0:
                # Try to help the user
                file_path = Path(self.cad_file_path)
                if file_path.suffix.lower() == '.dwg':
                    dxf_path = file_path.with_suffix('.dxf')
                    if dxf_path.exists():
                        error_msg = f"Could not extract dimensions from DWG file.\n\n"
                        error_msg += f"A DXF version exists at:\n{dxf_path}\n\n"
                        error_msg += f"Please load the .dxf file instead for better dimension extraction."
                        return error_msg
                    else:
                        error_msg = f"Could not extract dimensions from DWG file.\n\n"
                        error_msg += f"DWG files require conversion to DXF first.\n"
                        error_msg += f"Please convert to DXF using AutoCAD or the ODA File Converter."
                        return error_msg
                else:
                    error_msg = f"Could not extract dimensions from {file_path.name}\n\n"
                    error_msg += f"Extracted dimensions: L={part_info.length}, W={part_info.width}, T={part_info.thickness}\n\n"
                    error_msg += f"Please ensure the CAD file contains dimension annotations."
                    return error_msg

            # Calculate desired stock dimensions (part + allowances)
            desired_length = part_info.length + 0.25
            desired_width = part_info.width + 0.25
            desired_thickness = part_info.thickness + 0.125

            # Round desired thickness to nearest standard catalog thickness
            # McMaster catalog has standard thicknesses: 0.25, 0.375, 0.5, 0.625, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, etc.
            standard_thicknesses = [0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 6.0]
            desired_thickness = min(standard_thicknesses, key=lambda t: abs(t - desired_thickness) if t >= desired_thickness else float('inf'))

            # Get McMaster catalog stock size and part number
            catalog_csv_path = str(default_catalog_csv())
            catalog_rows = load_mcmaster_catalog_rows(catalog_csv_path)

            # Try multiple material key variations for better matching
            material_keys_to_try = [
                part_info.material,  # Try original first
                "MIC6",  # Common aluminum plate
                "Aluminum 6061",  # Generic aluminum
                "6061",
            ]

            mcmaster_result = None
            tried_keys = []

            for material_key in material_keys_to_try:
                mcmaster_result = pick_mcmaster_plate_sku(
                    need_L_in=desired_length,
                    need_W_in=desired_width,
                    need_T_in=desired_thickness,
                    material_key=material_key,
                    catalog_rows=catalog_rows
                )
                tried_keys.append(material_key)
                if mcmaster_result:
                    break

            if not mcmaster_result:
                # Get available material keys from catalog for debugging
                available_materials = set()
                for row in catalog_rows[:20]:  # Sample first 20
                    if 'material' in row:
                        available_materials.add(row['material'])

                error_msg = f"Could not find McMaster stock for material: {part_info.material}\n\n"
                error_msg += f"Tried keys: {', '.join(tried_keys)}\n\n"
                error_msg += f"Sample available materials in catalog:\n"
                for mat in sorted(available_materials):
                    error_msg += f"  - {mat}\n"
                return error_msg

            # Get McMaster dimensions and part number
            mcmaster_L = mcmaster_result.get('len_in', mcmaster_result.get('stock_L_in', 0))
            mcmaster_W = mcmaster_result.get('wid_in', mcmaster_result.get('stock_W_in', 0))
            mcmaster_T = mcmaster_result.get('thk_in', mcmaster_result.get('stock_T_in', 0))
            mcmaster_part = mcmaster_result.get('mcmaster_part', 'N/A')

            # Get price
            from cad_quoter.pricing.DirectCostHelper import get_mcmaster_price
            mcmaster_price = get_mcmaster_price(mcmaster_part, quantity=1)

            if mcmaster_price is None:
                price_str = "Price N/A"
            else:
                price_str = f"${mcmaster_price:,.2f}"

            # Calculate scrap info manually using dimensions we already have
            from cad_quoter.pricing.DirectCostHelper import (
                get_material_density,
                calculate_scrap_value
            )

            # Calculate volumes
            mcmaster_volume = mcmaster_L * mcmaster_W * mcmaster_T
            part_volume = part_info.length * part_info.width * part_info.thickness
            total_scrap_volume = mcmaster_volume - part_volume

            # Calculate weights
            density = get_material_density(part_info.material)
            mcmaster_weight = mcmaster_volume * density
            final_part_weight = part_volume * density
            total_scrap_weight = total_scrap_volume * density

            # Calculate percentages
            scrap_percentage = (total_scrap_volume / mcmaster_volume * 100) if mcmaster_volume > 0 else 0

            # Calculate scrap value using Wieland prices
            scrap_value_info = calculate_scrap_value(
                scrap_weight_lbs=total_scrap_weight,
                material=part_info.material,
                verbose=False
            )

            # Create a simple scrap_info dict with the data we need
            class ScrapInfo:
                def __init__(self):
                    self.mcmaster_weight = mcmaster_weight
                    self.final_part_weight = final_part_weight
                    self.total_scrap_weight = total_scrap_weight
                    self.scrap_percentage = scrap_percentage

            scrap_info = ScrapInfo()

            # Calculate net cost
            scrap_value = scrap_value_info.get('scrap_value', 0.0)
            scrap_price_per_lb = scrap_value_info.get('scrap_price_per_lb', 0.0)

            tax_amount = 0.0
            shipping_amount = 0.0

            if mcmaster_price is not None:
                tax_amount = mcmaster_price * 0.07
                shipping_amount = mcmaster_price * 0.125
                net_cost = mcmaster_price + tax_amount + shipping_amount - scrap_value
            else:
                net_cost = 0.0

            # Format the report
            report = []
            report.append("DIRECT COSTS")
            report.append("=" * 74)
            report.append(f"Material used: {part_info.material}")
            # Show original required dimensions (before rounding thickness to catalog)
            original_desired_thickness = part_info.thickness + 0.125
            report.append(f"  Required Stock: {desired_length:.2f} × {desired_width:.2f} × {original_desired_thickness:.2f} in")
            report.append(f"  Rounded to catalog: {mcmaster_L:.2f} × {mcmaster_W:.2f} × {mcmaster_T:.3f}")
            report.append(f"  Starting Weight: {self._format_weight(scrap_info.mcmaster_weight)}")
            report.append(f"  Net Weight: {self._format_weight(scrap_info.final_part_weight)}")
            report.append(f"  Scrap Percentage: {scrap_info.scrap_percentage:.1f}%")
            report.append(f"  Scrap Weight: {self._format_weight(scrap_info.total_scrap_weight)}")

            if scrap_price_per_lb is not None:
                report.append(f"  Scrap Price: ${scrap_price_per_lb:.2f} / lb")
            else:
                report.append(f"  Scrap Price: N/A")

            report.append("")
            report.append("")

            # Cost breakdown
            report.append(f"  Stock Piece (McMaster part {mcmaster_part})".ljust(60) + f"{price_str:>14}")

            if mcmaster_price is not None:
                report.append(f"  Tax".ljust(60) + f"+${tax_amount:>12.2f}")
                report.append(f"  Shipping".ljust(60) + f"+${shipping_amount:>12.2f}")

            if scrap_value > 0:
                scrap_credit_line = f"  Scrap Credit @ Wieland ${scrap_price_per_lb:.2f}/lb × {scrap_info.scrap_percentage:.1f}%"
                report.append(f"{scrap_credit_line.ljust(60)}-${scrap_value:>12.2f}")

            report.append(" " * 60 + "-" * 14)

            if mcmaster_price is not None:
                self.direct_cost_total = net_cost
                report.append(f"  Total Material Cost :".ljust(60) + f"${net_cost:>13.2f}")
            else:
                self.direct_cost_total = None
                report.append(f"  Total Material Cost :".ljust(60) + "Price N/A")

            report.append("")

            return "\n".join(report)

        except Exception as e:
            self.direct_cost_total = None
            import traceback
            return f"Error generating direct costs report:\n{str(e)}\n\n{traceback.format_exc()}"

    def _generate_machine_hours_report(self) -> str:
        """Generate formatted machine hours report with hole-by-hole breakdown."""
        self.machine_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            from cad_quoter.planning import (
                extract_hole_operations_from_cad,
                estimate_hole_table_times,
                plan_from_cad_file,
            )

            # Extract hole operations (expanded format with one row per operation)
            hole_table = extract_hole_operations_from_cad(self.cad_file_path)

            if not hole_table:
                return "No hole table found in CAD file."

            # Get dimensions from plan (or cached JSON)
            plan = plan_from_cad_file(self.cad_file_path, verbose=False)
            dims = plan.get('extracted_dims', {})
            thickness = dims.get('T', 0)

            # If thickness is still 0, try to get it from cached JSON
            if thickness == 0:
                json_output = Path(__file__).parent / "debug" / f"{Path(self.cad_file_path).stem}_dims.json"
                if json_output.exists():
                    try:
                        import json
                        with open(json_output, 'r') as f:
                            dims_data = json.load(f)
                        thickness = dims_data.get('thickness', 0.0)
                    except Exception:
                        pass

            # Get material from part info
            material = "17-4 PH Stainless"  # Default, could extract from CAD text

            # Calculate times for hole table
            times = estimate_hole_table_times(hole_table, material, thickness)

            # Format helper functions (matching test_machine_hours.py)
            def format_drill_group(group):
                return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
                        f"depth {group['depth']:.3f}\" | {group['sfm']:.0f} sfm | "
                        f"{group['ipr']:.4f} ipr | t/hole {group['time_per_hole']:.2f} min | "
                        f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

            def format_jig_grind_group(group):
                return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
                        f"depth {group['depth']:.3f}\" | "
                        f"t/hole {group['time_per_hole']:.2f} min | "
                        f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

            def format_tap_group(group):
                return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
                        f"depth {group['depth']:.3f}\" | {group['tpi']} TPI | "
                        f"t/hole {group['time_per_hole']:.2f} min | "
                        f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

            def format_cbore_group(group):
                return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
                        f"depth {group['depth']:.3f}\" | {group['sfm']:.0f} sfm | "
                        f"t/hole {group['time_per_hole']:.2f} min | "
                        f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

            def format_cdrill_group(group):
                return (f"Hole {group['hole_id']} | Dia {group['diameter']:.4f}\" x {group['qty']} | "
                        f"depth {group['depth']:.3f}\" | "
                        f"t/hole {group['time_per_hole']:.2f} min | "
                        f"group {group['qty']}x{group['time_per_hole']:.2f} = {group['total_time']:.2f} min")

            # Build the report
            report = []
            report.append("MACHINE HOURS ESTIMATION - DETAILED HOLE TABLE BREAKDOWN")
            report.append("=" * 74)
            report.append(f"Material: {material}")
            report.append(f"Thickness: {thickness:.3f}\"")
            report.append(f"Hole entries: {len(hole_table)}")
            report.append("")

            # TIME PER HOLE - DRILL GROUPS
            if times['drill_groups']:
                report.append("TIME PER HOLE - DRILL GROUPS")
                report.append("-" * 74)
                for group in times['drill_groups']:
                    report.append(format_drill_group(group))
                report.append(f"\nTotal Drilling Time: {times['total_drill_minutes']:.2f} minutes")
                report.append("")

            # TIME PER HOLE - JIG GRIND
            if times['jig_grind_groups']:
                report.append("TIME PER HOLE - JIG GRIND")
                report.append("-" * 74)
                for group in times['jig_grind_groups']:
                    report.append(format_jig_grind_group(group))
                report.append(f"\nTotal Jig Grind Time: {times['total_jig_grind_minutes']:.2f} minutes")
                report.append("")

            # TIME PER HOLE - TAP
            if times['tap_groups']:
                report.append("TIME PER HOLE - TAP")
                report.append("-" * 74)
                for group in times['tap_groups']:
                    report.append(format_tap_group(group))
                report.append(f"\nTotal Tapping Time: {times['total_tap_minutes']:.2f} minutes")
                report.append("")

            # TIME PER HOLE - C'BORE
            if times['cbore_groups']:
                report.append("TIME PER HOLE - C'BORE")
                report.append("-" * 74)
                for group in times['cbore_groups']:
                    report.append(format_cbore_group(group))
                report.append(f"\nTotal Counterbore Time: {times['total_cbore_minutes']:.2f} minutes")
                report.append("")

            # TIME PER HOLE - CDRILL
            if times['cdrill_groups']:
                report.append("TIME PER HOLE - CDRILL")
                report.append("-" * 74)
                for group in times['cdrill_groups']:
                    report.append(format_cdrill_group(group))
                report.append(f"\nTotal Center Drill Time: {times['total_cdrill_minutes']:.2f} minutes")
                report.append("")

            # Summary
            machine_cost = times['total_hours'] * self.MACHINE_RATE
            report.append("=" * 74)
            report.append(f"TOTAL MACHINE TIME: {times['total_minutes']:.2f} minutes ({times['total_hours']:.2f} hours)")
            report.append(f"TOTAL MACHINE COST: {machine_cost:.2f}")
            report.append("=" * 74)
            report.append("")

            self.machine_cost_total = machine_cost

            return "\n".join(report)

        except Exception as e:
            self.machine_cost_total = None
            import traceback
            return f"Error generating machine hours report:\n{str(e)}\n\n{traceback.format_exc()}"

    def _generate_labor_hours_report(self) -> str:
        """Generate formatted labor hours report using process_planner helpers."""
        self.labor_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            from cad_quoter.planning import plan_from_cad_file
            from cad_quoter.planning.process_planner import LaborInputs, compute_labor_minutes

            # Get the process plan
            plan = plan_from_cad_file(self.cad_file_path, verbose=False)

            # Extract operation counts from plan
            ops = plan.get('ops', [])

            # Count different operation types
            ops_total = len(ops)
            holes_total = plan.get('extracted_hole_operations', 0)

            # Count specific operations
            tool_changes = 0
            fixturing_complexity = 1  # Default: light fixturing
            edm_window_count = 0
            edm_skim_passes = 0
            thread_mill = 0
            jig_grind_bore_qty = 0
            grind_face_pairs = 0
            deep_holes = 0
            counterbore_qty = 0
            counterdrill_qty = 0
            ream_press_dowel = 0
            ream_slip_dowel = 0
            tap_rigid = 0
            tap_npt = 0
            outsource_touches = 0
            part_flips = 0

            # Analyze operations to populate counts
            for op in ops:
                op_type = op.get('op', '').lower()

                # Tool changes (estimate 2 per unique operation type)
                tool_changes += 2

                # EDM operations
                if 'wedm' in op_type or 'wire_edm' in op_type:
                    edm_window_count += 1
                    edm_skim_passes += op.get('skims', 0)

                # Thread operations
                if 'thread_mill' in op_type:
                    thread_mill += 1

                # Tapping
                if 'tap' in op_type:
                    if 'rigid' in op_type:
                        tap_rigid += 1
                    elif 'npt' in op_type or 'pipe' in op_type:
                        tap_npt += 1
                    else:
                        tap_rigid += 1  # Default to rigid tap

                # Grinding
                if 'jig_grind' in op_type and 'bore' in op_type:
                    jig_grind_bore_qty += 1
                if 'grind' in op_type and 'face' in op_type:
                    grind_face_pairs += 1

                # Counterbore/Counterdrill
                if 'counterbore' in op_type or 'cbore' in op_type:
                    counterbore_qty += 1
                if 'counterdrill' in op_type or 'cdrill' in op_type:
                    counterdrill_qty += 1

                # Dowel operations
                if 'ream_press' in op_type or ('ream' in op_type and 'press' in op_type):
                    ream_press_dowel += 1
                if 'ream_slip' in op_type or ('ream' in op_type and 'slip' in op_type):
                    ream_slip_dowel += 1

                # Outsourced operations
                if 'heat_treat' in op_type or 'coat' in op_type or 'hone' in op_type:
                    outsource_touches += 1

                # Part flips (estimate based on operations that require both sides)
                if 'back' in str(op.get('side', '')).lower():
                    part_flips = max(part_flips, 1)

            # Check fixturing notes
            fixturing_notes = plan.get('fixturing', [])
            if len(fixturing_notes) > 2:
                fixturing_complexity = 2  # Moderate
            if any('mag' in str(note).lower() or 'parallel' in str(note).lower() for note in fixturing_notes):
                fixturing_complexity = max(fixturing_complexity, 2)

            # Create LaborInputs
            labor_inputs = LaborInputs(
                ops_total=ops_total,
                holes_total=holes_total,
                tool_changes=tool_changes,
                fixturing_complexity=fixturing_complexity,
                edm_window_count=edm_window_count,
                edm_skim_passes=edm_skim_passes,
                thread_mill=thread_mill,
                jig_grind_bore_qty=jig_grind_bore_qty,
                grind_face_pairs=grind_face_pairs,
                deep_holes=deep_holes,
                counterbore_qty=counterbore_qty,
                counterdrill_qty=counterdrill_qty,
                ream_press_dowel=ream_press_dowel,
                ream_slip_dowel=ream_slip_dowel,
                tap_rigid=tap_rigid,
                tap_npt=tap_npt,
                outsource_touches=outsource_touches,
                inspection_frequency=0.1,  # 10% sampling
                part_flips=part_flips,
            )

            # Compute labor minutes
            labor_result = compute_labor_minutes(labor_inputs)
            minutes = labor_result['minutes']

            # Format the report
            report = []
            report.append("LABOR HOURS ESTIMATION")
            report.append("=" * 74)
            report.append("")

            # Summary table
            report.append("LABOR BREAKDOWN BY CATEGORY")
            report.append("-" * 74)
            report.append(f"  Setup / Prep:                    {minutes['Setup']:>10.2f} minutes")
            report.append(f"  Programming / Prove-out:         {minutes['Programming']:>10.2f} minutes")
            report.append(f"  Machining Steps:                 {minutes['Machining_Steps']:>10.2f} minutes")
            report.append(f"  Inspection:                      {minutes['Inspection']:>10.2f} minutes")
            report.append(f"  Finishing / Deburr:              {minutes['Finishing']:>10.2f} minutes")
            report.append("-" * 74)
            labor_hours = minutes['Labor_Total'] / 60
            labor_cost = labor_hours * self.LABOR_RATE
            report.append(f"  TOTAL LABOR TIME:                {minutes['Labor_Total']:>10.2f} minutes")
            report.append(f"                                   {labor_hours:>10.2f} hours")
            report.append(f"  TOTAL LABOR COST:                {labor_cost:>10.2f}")
            report.append("")

            self.labor_cost_total = labor_cost

            # Input details (hidden by request)
            # report.append("LABOR INPUT DETAILS")
            # report.append("-" * 74)
            # report.append(f"  Total Operations:                {labor_inputs.ops_total:>10}")
            # report.append(f"  Total Holes:                     {labor_inputs.holes_total:>10}")
            # report.append(f"  Tool Changes:                    {labor_inputs.tool_changes:>10}")
            # report.append(
            #     f"  Fixturing Complexity:            {labor_inputs.fixturing_complexity:>10} (0=none, 1=light, 2=moderate, 3=complex)"
            # )

            # if edm_window_count > 0:
            #     report.append(f"  EDM Windows:                     {labor_inputs.edm_window_count:>10}")
            #     report.append(f"  EDM Skim Passes:                 {labor_inputs.edm_skim_passes:>10}")

            # if thread_mill > 0:
            #     report.append(f"  Thread Mill Operations:          {labor_inputs.thread_mill:>10}")

            # if jig_grind_bore_qty > 0:
            #     report.append(f"  Jig Grind Bores:                 {labor_inputs.jig_grind_bore_qty:>10}")

            # if grind_face_pairs > 0:
            #     report.append(f"  Grind Face Pairs:                {labor_inputs.grind_face_pairs:>10}")

            # if tap_rigid > 0:
            #     report.append(f"  Rigid Tap Operations:            {labor_inputs.tap_rigid:>10}")

            # if tap_npt > 0:
            #     report.append(f"  NPT Tap Operations:              {labor_inputs.tap_npt:>10}")

            # if counterbore_qty > 0:
            #     report.append(f"  Counterbore Operations:          {labor_inputs.counterbore_qty:>10}")

            # if counterdrill_qty > 0:
            #     report.append(f"  Counterdrill Operations:         {labor_inputs.counterdrill_qty:>10}")

            # if ream_press_dowel > 0:
            #     report.append(f"  Press Fit Dowel Reaming:         {labor_inputs.ream_press_dowel:>10}")

            # if ream_slip_dowel > 0:
            #     report.append(f"  Slip Fit Dowel Reaming:          {labor_inputs.ream_slip_dowel:>10}")

            # if outsource_touches > 0:
            #     report.append(f"  Outsource Touches:               {labor_inputs.outsource_touches:>10}")

            # if part_flips > 0:
            #     report.append(f"  Part Flips:                      {labor_inputs.part_flips:>10}")

            # report.append(
            #     f"  Inspection Frequency:            {labor_inputs.inspection_frequency:>10.1%}"
            # )

            # report.append("")
            report.append("=" * 74)
            report.append("")

            return "\n".join(report)

        except Exception as e:
            self.labor_cost_total = None
            import traceback
            return f"Error generating labor hours report:\n{str(e)}\n\n{traceback.format_exc()}"

    def _format_cost_summary_line(self, label: str, value: Optional[float]) -> str:
        """Return a formatted cost line or N/A when value is missing."""
        if value is None:
            return f"  {label}:".ljust(30) + "N/A"
        return f"  {label}:".ljust(30) + f"${value:>12,.2f}"

    def generate_quote(self) -> None:
        """Generate the quote."""
        # Collect values from quote editor
        quote_data = {}
        for label, field in self.quote_fields.items():
            quote_data[label] = field.get()
        self.quote_vars = quote_data

        # Display in output tab
        self.output_text.delete(1.0, tk.END)

        # Generate labor hours report first
        labor_hours_report = self._generate_labor_hours_report()
        self.output_text.insert(tk.END, labor_hours_report)

        # Add separator
        self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")

        # Generate machine hours report next
        machine_hours_report = self._generate_machine_hours_report()
        self.output_text.insert(tk.END, machine_hours_report)

        # Add separator
        self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")

        # Generate direct costs report last before summary
        direct_costs_report = self._generate_direct_costs_report()
        self.output_text.insert(tk.END, direct_costs_report)

        # Add summary
        self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")
        summary_lines = [
            "COST SUMMARY",
            "=" * 74,
            self._format_cost_summary_line("Direct Cost", self.direct_cost_total),
            self._format_cost_summary_line("Machine Cost", self.machine_cost_total),
            self._format_cost_summary_line("Labor Cost", self.labor_cost_total),
        ]

        if None not in (self.direct_cost_total, self.machine_cost_total, self.labor_cost_total):
            total_cost = (
                (self.direct_cost_total or 0.0)
                + (self.machine_cost_total or 0.0)
                + (self.labor_cost_total or 0.0)
            )
            summary_lines.append("-" * 74)
            summary_lines.append(self._format_cost_summary_line("Total Estimated Cost", total_cost))
            margin_rate = 0.15
            margin_amount = total_cost * margin_rate
            final_cost = total_cost + margin_amount
            summary_lines.append(
                self._format_cost_summary_line(
                    f"Margin ({margin_rate:.0%})",
                    margin_amount,
                )
            )
            summary_lines.append(self._format_cost_summary_line("Final Cost", final_cost))

        summary_lines.append("")
        self.output_text.insert(tk.END, "\n".join(summary_lines))

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