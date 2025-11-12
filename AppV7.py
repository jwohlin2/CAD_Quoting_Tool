from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import json
import os
import subprocess
import platform
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
    MACHINE_RATE = 90.0  # $ per hour
    LABOR_RATE = 90.0    # $ per hour

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

        # Cached CAD extraction results using QuoteDataHelper (to avoid redundant ODA/OCR calls)
        self._cached_quote_data = None

        # Default profit margin applied to the final price
        self.margin_rate: float = 0.15

        # Store path to drawing image file
        self.drawing_image_path: Optional[str] = None

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

        ttk.Button(button_frame, text="Drawing Preview",
                   command=self.open_drawing_preview).pack(side=tk.LEFT, padx=5)

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

    def open_drawing_preview(self) -> None:
        """Open the drawing preview image file with the system default viewer."""
        if not self.drawing_image_path:
            messagebox.showinfo(
                "No Drawing Image",
                "No drawing image found.\n\n"
                "Load a CAD file first, and make sure there's a corresponding image file\n"
                "(PNG, JPG, etc.) with the same name in the same directory.\n\n"
                "Or use File > Load Drawing Image... to select an image manually."
            )
            return

        if not os.path.exists(self.drawing_image_path):
            messagebox.showerror(
                "File Not Found",
                f"Drawing image file not found:\n{self.drawing_image_path}"
            )
            return

        try:
            # Open file with system default viewer
            if platform.system() == 'Windows':
                os.startfile(self.drawing_image_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', self.drawing_image_path])
            else:  # Linux
                subprocess.run(['xdg-open', self.drawing_image_path])

            self.status_bar.config(text=f"Opened drawing: {Path(self.drawing_image_path).name}")
        except Exception as e:
            messagebox.showerror(
                "Error Opening File",
                f"Failed to open drawing image:\n{str(e)}"
            )

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
            # Part Dimensions (optional - leave blank for OCR auto-detection)
            ("Length (in)", "", "Part length in inches (optional - leave blank for OCR auto-detection)"),
            ("Width (in)", "", "Part width in inches (optional - leave blank for OCR auto-detection)"),
            ("Thickness (in)", "", "Part thickness in inches (optional - leave blank for OCR auto-detection)"),

            # Material Override
            ("Material", "", "Override auto-detected material (e.g., '17-4 PH Stainless Steel', 'Aluminum 6061'). Leave blank for auto-detection."),

            # Rate Overrides
            ("Machine Rate ($/hr)", "90", "Override machine hourly rate (default: $90/hr)"),
            ("Labor Rate ($/hr)", "90", "Override labor hourly rate (default: $90/hr)"),
            ("Margin (%)", "15", "Override profit margin percentage (default: 15%)"),

            # Price Overrides (optional)
            ("McMaster Price Override ($)", "", "Manual stock price - skips McMaster API lookup (optional)"),
            ("Scrap Value Override ($)", "", "Manual scrap value - skips automatic scrap value calculation (optional)"),
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

    def _clear_cad_cache(self) -> None:
        """Clear cached CAD extraction results (QuoteData)."""
        self._cached_quote_data = None
        print("[AppV7] Cleared CAD extraction cache")

    def _get_or_create_quote_data(self):
        """
        Get cached QuoteData or extract it once using QuoteDataHelper.

        This replaces the old separate caching of plan, part_info, and hole_operations
        with a unified QuoteData structure that contains everything.

        Returns:
            QuoteData with all extraction results
        """
        if not self.cad_file_path:
            raise ValueError("No CAD file loaded")

        if self._cached_quote_data is None:
            print("[AppV7] Extracting complete quote data (ODA + OCR will run once)...")
            from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad

            # Read all overrides from Quote Editor
            dimension_override = self._get_manual_dimensions()
            material_override = self._get_field_string("Material")
            machine_rate = self._get_field_float("Machine Rate ($/hr)", self.MACHINE_RATE)
            labor_rate = self._get_field_float("Labor Rate ($/hr)", self.LABOR_RATE)
            margin_percent = self._get_field_float("Margin (%)", 15.0)
            margin_rate = (margin_percent / 100.0) if margin_percent is not None else 0.15  # Convert percentage to decimal
            mcmaster_price_override = self._get_field_float("McMaster Price Override ($)")
            scrap_value_override = self._get_field_float("Scrap Value Override ($)")

            try:
                self._cached_quote_data = extract_quote_data_from_cad(
                    cad_file_path=self.cad_file_path,
                    machine_rate=machine_rate,
                    labor_rate=labor_rate,
                    margin_rate=margin_rate,
                    material_override=material_override,
                    dimension_override=dimension_override,
                    mcmaster_price_override=mcmaster_price_override,
                    scrap_value_override=scrap_value_override,
                    verbose=True
                )
                if dimension_override:
                    print(f"[AppV7] Quote data cached (using manual dimensions: {dimension_override})")
                else:
                    print("[AppV7] Quote data cached for reuse")
            except ValueError as e:
                error_msg = str(e)
                # Check if it's a dimension extraction failure
                if "Could not extract dimensions" in error_msg:
                    print(f"[AppV7 ERROR] OCR dimension extraction failed: {e}")
                    messagebox.showerror(
                        "OCR Dimension Extraction Failed",
                        "Could not extract dimensions from the CAD file.\n\n"
                        "Please enter the part dimensions manually in the Quote Editor tab:\n"
                        "- Length (in)\n"
                        "- Width (in)\n"
                        "- Thickness (in)\n\n"
                        "Then click 'Generate Quote' again."
                    )
                    # Switch to Quote Editor tab so user can see the fields
                    self.notebook.select(self.quote_editor_tab)
                    raise
                else:
                    print(f"[AppV7 ERROR] Failed to extract quote data: {e}")
                    raise
            except Exception as e:
                print(f"[AppV7 ERROR] Failed to extract quote data: {e}")
                raise
        else:
            print("[AppV7] Using cached quote data (no ODA/OCR)")

        return self._cached_quote_data

    def _get_manual_dimensions(self):
        """
        Read manual dimensions from Quote Editor fields.

        Returns:
            tuple of (length, width, thickness) or None if not provided
        """
        try:
            length_str = self.quote_fields.get("Length (in)", None)
            width_str = self.quote_fields.get("Width (in)", None)
            thickness_str = self.quote_fields.get("Thickness (in)", None)

            if not length_str or not width_str or not thickness_str:
                return None

            length_val = length_str.get().strip()
            width_val = width_str.get().strip()
            thickness_val = thickness_str.get().strip()

            # If any field is empty, don't use manual dimensions
            if not length_val or not width_val or not thickness_val:
                return None

            # Parse values
            length = float(length_val)
            width = float(width_val)
            thickness = float(thickness_val)

            # Validate
            if length <= 0 or width <= 0 or thickness <= 0:
                messagebox.showerror(
                    "Invalid Dimensions",
                    "All dimensions must be positive numbers."
                )
                return None

            return (length, width, thickness)

        except ValueError:
            messagebox.showerror(
                "Invalid Dimensions",
                "Please enter valid numbers for Length, Width, and Thickness."
            )
            return None
        except Exception:
            # If fields don't exist or other error, just return None
            return None

    def _get_field_float(self, label: str, default: Optional[float] = None) -> Optional[float]:
        """
        Read a float value from Quote Editor field.

        Args:
            label: Field label text
            default: Default value if field is empty or invalid (can be None)

        Returns:
            Float value from field or default
        """
        try:
            field = self.quote_fields.get(label, None)
            if not field:
                return default

            value_str = field.get().strip()
            if not value_str:
                return default

            value = float(value_str)

            # Validate positive (only if we got a value)
            if value <= 0:
                if default is not None:
                    print(f"[AppV7] Warning: {label} must be positive. Using default: {default}")
                else:
                    print(f"[AppV7] Warning: {label} must be positive. Ignoring value.")
                return default

            return value

        except ValueError:
            if default is not None:
                print(f"[AppV7] Warning: Invalid number for {label}. Using default: {default}")
            return default
        except Exception:
            return default

    def _get_field_string(self, label: str, default: Optional[str] = None) -> Optional[str]:
        """
        Read a string value from Quote Editor field.

        Args:
            label: Field label text
            default: Default value if field is empty

        Returns:
            String value from field or default (None if not provided)
        """
        try:
            field = self.quote_fields.get(label, None)
            if not field:
                return default

            value = field.get().strip()
            if not value:
                return default

            return value

        except Exception:
            return default

    def load_drawing_image_manual(self) -> None:
        """Manually select a drawing image file."""
        filename = filedialog.askopenfilename(
            title="Select Drawing Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.drawing_image_path = filename
            self.status_bar.config(text=f"Drawing image set: {Path(filename).name}")
            messagebox.showinfo(
                "Drawing Image Set",
                f"Drawing image set to:\n{Path(filename).name}\n\n"
                f"Click the 'Drawing Preview' button to open it."
            )

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

                # Clear cached CAD extraction results
                self._clear_cad_cache()

                # Load and extract hole table data
                self._extract_and_display_hole_table(filename)

                # Try to load or generate a corresponding drawing image
                self.status_bar.config(text="Generating drawing preview...")
                self.root.update_idletasks()
                self._try_load_drawing_image(filename)

                status_msg = "CAD file loaded. Review the Quote Editor and generate the quote."
                if self.drawing_image_path:
                    status_msg += " (Drawing preview available)"
                self.status_bar.config(text=status_msg)
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

    def _try_load_drawing_image(self, cad_filename: str) -> None:
        """Try to find or create a drawing image for the CAD file."""
        cad_path = Path(cad_filename)
        base_name = cad_path.stem

        # Common image extensions to check
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']

        # Check for existing image file in the same directory
        for ext in image_extensions:
            image_path = cad_path.parent / f"{base_name}{ext}"
            if image_path.exists():
                self.drawing_image_path = str(image_path)
                print(f"[AppV7] Found existing drawing image: {image_path}")
                return

        # If no existing image found, try to generate one
        print(f"[AppV7] No existing drawing image found for {base_name}, generating PNG...")

        try:
            from tools.paddle_dims_extractor import DrawingRenderer

            # Generate PNG in the same directory as the CAD file
            output_png = str(cad_path.parent / f"{base_name}.png")

            # Create renderer and generate image
            renderer = DrawingRenderer(verbose=False)
            renderer.render(str(cad_path), output_png)

            self.drawing_image_path = output_png
            print(f"[AppV7] Successfully generated drawing image: {output_png}")

        except Exception as e:
            print(f"[AppV7] Failed to generate drawing image: {e}")
            self.drawing_image_path = None

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
        """Generate formatted direct costs report using QuoteData."""
        self.direct_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            # Get cached QuoteData (avoids redundant ODA/OCR)
            quote_data = self._get_or_create_quote_data()

            # Extract data from QuoteData
            part_dims = quote_data.part_dimensions
            material_info = quote_data.material_info
            stock_info = quote_data.stock_info
            scrap_info = quote_data.scrap_info
            cost_breakdown = quote_data.direct_cost_breakdown

            # Check if overrides were used
            material_override = self._get_field_string("Material")
            mcmaster_price_override = self._get_field_float("McMaster Price Override ($)")
            scrap_value_override = self._get_field_float("Scrap Value Override ($)")

            # Check if we have valid price data
            if stock_info.mcmaster_price is None:
                price_str = "Price N/A"
            else:
                price_str = f"${stock_info.mcmaster_price:,.2f}"
                if mcmaster_price_override is not None:
                    price_str += " (MANUAL)"

            # Format the report
            report = []
            report.append("DIRECT COSTS")
            report.append("=" * 74)
            material_label = f"Material used: {material_info.material_name}"
            if material_override:
                material_label += " (OVERRIDDEN)"
            report.append(material_label)
            report.append(f"  Required Stock: {stock_info.desired_length:.2f} × {stock_info.desired_width:.2f} × {stock_info.desired_thickness:.2f} in")
            report.append(f"  Rounded to catalog: {stock_info.mcmaster_length:.2f} × {stock_info.mcmaster_width:.2f} × {stock_info.mcmaster_thickness:.3f}")
            report.append(f"  Starting Weight: {self._format_weight(stock_info.mcmaster_weight)}")
            report.append(f"  Net Weight: {self._format_weight(stock_info.final_part_weight)}")
            report.append(f"  Scrap Percentage: {scrap_info.scrap_percentage:.1f}%")
            report.append(f"  Scrap Weight: {self._format_weight(scrap_info.total_scrap_weight)}")

            if scrap_info.scrap_price_per_lb is not None:
                report.append(f"  Scrap Price: ${scrap_info.scrap_price_per_lb:.2f} / lb")
            else:
                report.append(f"  Scrap Price: N/A")

            report.append("")
            report.append("")

            # Cost breakdown
            report.append(f"  Stock Piece (McMaster part {stock_info.mcmaster_part_number or 'N/A'})".ljust(60) + f"{price_str:>14}")

            if stock_info.mcmaster_price is not None:
                report.append(f"  Tax".ljust(60) + f"+${cost_breakdown.tax:>12.2f}")
                report.append(f"  Shipping".ljust(60) + f"+${cost_breakdown.shipping:>12.2f}")

            if cost_breakdown.scrap_credit > 0:
                scrap_credit_line = f"  Scrap Credit @ Wieland ${scrap_info.scrap_price_per_lb:.2f}/lb × {scrap_info.scrap_percentage:.1f}%"
                if scrap_value_override is not None:
                    scrap_credit_line += " (MANUAL)"
                report.append(f"{scrap_credit_line.ljust(60)}-${cost_breakdown.scrap_credit:>12.2f}")

            report.append(" " * 60 + "-" * 14)

            if stock_info.mcmaster_price is not None:
                self.direct_cost_total = cost_breakdown.net_material_cost
                report.append(f"  Total Material Cost :".ljust(60) + f"${cost_breakdown.net_material_cost:>13.2f}")
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
        """Generate formatted machine hours report using QuoteData."""
        self.machine_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            # Get cached QuoteData (avoids redundant ODA/OCR)
            quote_data = self._get_or_create_quote_data()

            machine_hours = quote_data.machine_hours

            if not machine_hours.drill_operations and not machine_hours.tap_operations:
                return "No hole operations found in CAD file."

            # Format helper functions
            def format_drill_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | {op.sfm:.0f} sfm | "
                        f"{op.ipr:.4f} ipr | t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_jig_grind_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_tap_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | {op.tpi} TPI | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_cbore_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | {op.sfm:.0f} sfm | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_cdrill_group(op):
                return (f"Hole {op.hole_id} | Dia {op.diameter:.4f}\" x {op.qty} | "
                        f"depth {op.depth:.3f}\" | "
                        f"t/hole {op.time_per_hole:.2f} min | "
                        f"group {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            # Build the report
            report = []
            report.append("MACHINE HOURS ESTIMATION - DETAILED HOLE TABLE BREAKDOWN")
            report.append("=" * 74)
            report.append(f"Material: {quote_data.material_info.material_name}")
            report.append(f"Thickness: {quote_data.part_dimensions.thickness:.3f}\"")
            report.append(f"Hole entries: {len(machine_hours.drill_operations + machine_hours.tap_operations + machine_hours.cbore_operations)}")
            report.append("")

            # TIME PER HOLE - DRILL GROUPS
            if machine_hours.drill_operations:
                report.append("TIME PER HOLE - DRILL GROUPS")
                report.append("-" * 74)
                for op in machine_hours.drill_operations:
                    report.append(format_drill_group(op))
                report.append(f"\nTotal Drilling Time: {machine_hours.total_drill_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - JIG GRIND
            if machine_hours.jig_grind_operations:
                report.append("TIME PER HOLE - JIG GRIND")
                report.append("-" * 74)
                for op in machine_hours.jig_grind_operations:
                    report.append(format_jig_grind_group(op))
                report.append(f"\nTotal Jig Grind Time: {machine_hours.total_jig_grind_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - TAP
            if machine_hours.tap_operations:
                report.append("TIME PER HOLE - TAP")
                report.append("-" * 74)
                for op in machine_hours.tap_operations:
                    report.append(format_tap_group(op))
                report.append(f"\nTotal Tapping Time: {machine_hours.total_tap_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - C'BORE
            if machine_hours.cbore_operations:
                report.append("TIME PER HOLE - C'BORE")
                report.append("-" * 74)
                for op in machine_hours.cbore_operations:
                    report.append(format_cbore_group(op))
                report.append(f"\nTotal Counterbore Time: {machine_hours.total_cbore_minutes:.2f} minutes")
                report.append("")

            # TIME PER HOLE - CDRILL
            if machine_hours.cdrill_operations:
                report.append("TIME PER HOLE - CDRILL")
                report.append("-" * 74)
                for op in machine_hours.cdrill_operations:
                    report.append(format_cdrill_group(op))
                report.append(f"\nTotal Center Drill Time: {machine_hours.total_cdrill_minutes:.2f} minutes")
                report.append("")

            # Summary
            machine_rate = self._get_field_float("Machine Rate ($/hr)", self.MACHINE_RATE)
            machine_rate_label = f"@ ${machine_rate:.2f}/hr"
            if machine_rate != self.MACHINE_RATE:
                machine_rate_label += " (OVERRIDDEN)"

            report.append("=" * 74)
            report.append(f"TOTAL MACHINE TIME: {machine_hours.total_minutes:.2f} minutes ({machine_hours.total_hours:.2f} hours)")
            report.append(f"TOTAL MACHINE COST: ${machine_hours.machine_cost:.2f} {machine_rate_label}")
            report.append("=" * 74)
            report.append("")

            self.machine_cost_total = machine_hours.machine_cost

            return "\n".join(report)

        except Exception as e:
            self.machine_cost_total = None
            import traceback
            return f"Error generating machine hours report:\n{str(e)}\n\n{traceback.format_exc()}"

    def _generate_labor_hours_report(self) -> str:
        """Generate formatted labor hours report using QuoteData."""
        self.labor_cost_total = None
        if not self.cad_file_path:
            return "No CAD file loaded. Please load a CAD file first."

        try:
            # Get cached QuoteData (avoids redundant ODA/OCR)
            quote_data = self._get_or_create_quote_data()

            labor_hours = quote_data.labor_hours

            # Format the report
            report = []
            report.append("LABOR HOURS ESTIMATION")
            report.append("=" * 74)
            report.append("")

            # Summary table
            report.append("LABOR BREAKDOWN BY CATEGORY")
            report.append("-" * 74)
            report.append(f"  Setup / Prep:                    {labor_hours.setup_minutes:>10.2f} minutes")
            report.append(f"  Programming / Prove-out:         {labor_hours.programming_minutes:>10.2f} minutes")
            report.append(f"  Machining Steps:                 {labor_hours.machining_steps_minutes:>10.2f} minutes")
            report.append(f"  Inspection:                      {labor_hours.inspection_minutes:>10.2f} minutes")
            report.append(f"  Finishing / Deburr:              {labor_hours.finishing_minutes:>10.2f} minutes")
            report.append("-" * 74)
            report.append(f"  TOTAL LABOR TIME:                {labor_hours.total_minutes:>10.2f} minutes")
            report.append(f"                                   {labor_hours.total_hours:>10.2f} hours")

            # Show labor rate with override indicator
            labor_rate = self._get_field_float("Labor Rate ($/hr)", self.LABOR_RATE)
            labor_rate_label = f"@ ${labor_rate:.2f}/hr"
            if labor_rate != self.LABOR_RATE:
                labor_rate_label += " (OVERRIDDEN)"

            report.append(f"  TOTAL LABOR COST:                ${labor_hours.labor_cost:>9.2f} {labor_rate_label}")
            report.append("")
            report.append("=" * 74)
            report.append("")

            self.labor_cost_total = labor_hours.labor_cost

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

    def _format_quick_margin_line(
        self, margin: float, total_cost: float, *, is_current: bool = False
    ) -> str:
        """Format a single Quick What-If margin line."""

        safe_margin = max(0.0, margin)
        margin_text = f"{safe_margin:.0%}"
        if is_current:
            margin_text += " (current)"
        final_price = total_cost * (1.0 + safe_margin)
        price_text = f"${final_price:,.2f}"
        return f" {margin_text:<17}{price_text:>15}"

    def _build_quick_margin_section(self, total_cost: float, margin_rate: float) -> list[str]:
        """Return the QUICK WHAT-IFS margin table for the summary output."""

        base_margin = max(0.0, margin_rate)
        margins = [
            max(0.0, base_margin - 0.05),
            base_margin,
            base_margin + 0.05,
            base_margin + 0.10,
        ]

        seen: set[float] = set()
        lines = [
            "QUICK WHAT-IFS",
            "=" * 74,
            "Margin slider",
            f" {'Margin':<17}{'Final Price':>15}",
        ]

        for margin in margins:
            key = round(max(0.0, margin), 4)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                self._format_quick_margin_line(
                    margin,
                    total_cost,
                    is_current=abs(margin - base_margin) < 1e-9,
                )
            )

        lines.append("")
        return lines

    def generate_quote(self) -> None:
        """Generate the quote."""
        # Clear the cache to ensure we use the latest overrides from Quote Editor
        self._clear_cad_cache()

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

        # Get margin rate with override check
        margin_percent = self._get_field_float("Margin (%)", 15.0)
        margin_rate = (margin_percent / 100.0) if margin_percent is not None else 0.15
        margin_overridden = (margin_percent is not None and margin_percent != 15.0)

        quick_margin_lines: list[str] = []
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
            quick_margin_lines = self._build_quick_margin_section(total_cost, margin_rate)
            summary_lines.append("-" * 74)
            summary_lines.append(self._format_cost_summary_line("Total Estimated Cost", total_cost))
            margin_amount = total_cost * margin_rate
            final_cost = total_cost + margin_amount

            # Add margin line with override indicator
            margin_label = f"Margin ({margin_rate:.0%})"
            if margin_overridden:
                margin_label += " (OVERRIDDEN)"
            summary_lines.append(
                self._format_cost_summary_line(margin_label, margin_amount)
            )
            summary_lines.append(self._format_cost_summary_line("Final Cost", final_cost))

        if quick_margin_lines:
            self.output_text.insert(tk.END, "\n".join(quick_margin_lines))

        summary_lines.append("")
        self.output_text.insert(tk.END, "\n".join(summary_lines))

        self.notebook.select(self.output_tab)
        self.status_bar.config(text="Quote generated successfully!")

    def save_quote(self) -> None:
        """Save the quote to JSON file."""
        if not self.cad_file_path:
            messagebox.showwarning("No Quote", "Please load a CAD file and generate a quote first.")
            return

        if self._cached_quote_data is None:
            messagebox.showwarning("No Quote", "Please generate a quote first before saving.")
            return

        # Suggest filename based on CAD file
        from pathlib import Path
        suggested_name = Path(self.cad_file_path).stem + "_quote.json"

        filename = filedialog.asksaveasfilename(
            title="Save Quote",
            defaultextension=".json",
            initialfile=suggested_name,
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if filename:
            try:
                from cad_quoter.pricing.QuoteDataHelper import save_quote_data
                save_quote_data(self._cached_quote_data, filename)
                self.status_bar.config(text=f"Quote saved to: {filename}")
                messagebox.showinfo("Success", f"Quote saved successfully to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save quote:\n{str(e)}")
                self.status_bar.config(text="Error saving quote")

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