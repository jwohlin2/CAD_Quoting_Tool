from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import json
import os
import subprocess
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Import MaterialMapper for material dropdown
from cad_quoter.pricing.MaterialMapper import material_mapper


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

        # Multi-part order support
        from cad_quoter.pricing.QuoteDataHelper import OrderData
        self.current_order = OrderData()  # Current order (can hold multiple parts)
        self.active_part_index: Optional[int] = None  # Index of currently selected part

        # Cached totals for summary display
        self.direct_cost_total: Optional[float] = None
        self.machine_cost_total: Optional[float] = None
        self.labor_cost_total: Optional[float] = None

        # Cached CAD extraction results using QuoteDataHelper (to avoid redundant ODA/OCR calls)
        self._cached_quote_data = None

        # Track previous quote inputs for smart cache invalidation
        self._previous_quote_inputs: Optional[dict] = None

        # Default profit margin applied to the final price
        self.margin_rate: float = 0.15

        # Store path to drawing image file
        self.drawing_image_path: Optional[str] = None

        # Background drawing generation (threading)
        self._drawing_generation_thread: Optional[threading.Thread] = None
        self._drawing_generation_in_progress: bool = False
        self._drawing_generation_success: bool = False

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
        file_menu.add_command(label="Load Order...", command=self.load_order)
        file_menu.add_separator()
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

        ttk.Button(button_frame, text="Remove Overrides",
                   command=self.remove_overrides).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Drawing Preview",
                   command=self.open_drawing_preview).pack(side=tk.LEFT, padx=5)

    def _create_tabs(self) -> None:
        """Create the tabbed interface with parts list sidebar."""
        # Create main container with grid layout
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)  # Notebook (expandable)

        # Create notebook
        self.notebook = ttk.Notebook(main_container)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Order Parts tab (new)
        self.order_parts_tab = tk.Frame(self.notebook)
        self.notebook.add(self.order_parts_tab, text="Order Parts")
        self._create_order_parts_tab()

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

    def _create_order_parts_tab(self) -> None:
        """Create the Order Parts tab with parts list and management buttons."""
        # Main container for the tab
        parts_container = ttk.Frame(self.order_parts_tab, padding=10)
        parts_container.pack(fill=tk.BOTH, expand=True)

        # Parts listbox with scrollbar
        listbox_frame = ttk.Frame(parts_container)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.parts_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            font=("TkDefaultFont", 10)
        )
        self.parts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.parts_listbox.yview)

        # Bind selection event
        self.parts_listbox.bind('<<ListboxSelect>>', self._on_part_selected)

        # Buttons for managing parts
        btn_frame = ttk.Frame(parts_container)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            btn_frame,
            text="Add Part(s)...",
            command=self.add_parts_to_order
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Remove Part",
            command=self.remove_part_from_order
        ).pack(side=tk.LEFT, padx=5)

        # Info label showing current selection
        self.part_info_label = ttk.Label(
            parts_container,
            text="No parts in order",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=5
        )
        self.part_info_label.pack(fill=tk.X)

    def _refresh_parts_list(self) -> None:
        """Refresh the parts listbox and dropdown to reflect current order."""
        self.parts_listbox.delete(0, tk.END)

        if not self.current_order.parts:
            self.part_info_label.config(text="No parts in order")
            # Clear the dropdown
            if hasattr(self, 'part_selector'):
                self.part_selector['values'] = []
                self.part_selector_var.set("")
            return

        # Populate both listbox and dropdown with the same format
        part_options = []
        for i, part in enumerate(self.current_order.parts):
            # Format: "[1] filename.dwg · Qty: 5"
            display_text = f"[{i+1}] {part.cad_file_name} · Qty: {part.quantity}"
            self.parts_listbox.insert(tk.END, display_text)
            part_options.append(display_text)

        # Update dropdown with all parts
        if hasattr(self, 'part_selector'):
            self.part_selector['values'] = part_options
            # Set current selection if there's an active part
            if self.active_part_index is not None and 0 <= self.active_part_index < len(part_options):
                self.part_selector_var.set(part_options[self.active_part_index])

        # Update info label
        total_parts = len(self.current_order.parts)
        self.part_info_label.config(text=f"{total_parts} part(s) in order")

    def _on_part_selected(self, event) -> None:
        """Handle part selection from the listbox."""
        selection = self.parts_listbox.curselection()
        if not selection:
            return

        part_index = selection[0]
        if part_index != self.active_part_index:
            self.active_part_index = part_index
            self._load_active_part_to_ui()

    def _on_part_selector_changed(self, event) -> None:
        """Handle part selection from the dropdown in Quote Editor tab."""
        if not hasattr(self, 'part_selector') or not self.part_selector_var.get():
            return

        # Extract the part index from the dropdown selection
        # Format: "[1] filename.dwg · Qty: 5"
        selected_text = self.part_selector_var.get()
        if selected_text.startswith("["):
            try:
                part_index = int(selected_text.split("]")[0][1:]) - 1
                if part_index != self.active_part_index and 0 <= part_index < len(self.current_order.parts):
                    self.active_part_index = part_index
                    self._load_active_part_to_ui()
                    # Update the listbox selection to match
                    self.parts_listbox.selection_clear(0, tk.END)
                    self.parts_listbox.selection_set(part_index)
                    self.parts_listbox.see(part_index)
            except (ValueError, IndexError):
                pass

    def _load_active_part_to_ui(self) -> None:
        """Load the active part's data into the UI (Quote Editor, GEO tab, etc.)."""
        if self.active_part_index is None:
            return

        part = self.current_order.get_part(self.active_part_index)
        if not part:
            return

        # Update the cached quote data to match the active part
        self._cached_quote_data = part

        # Update the CAD file path
        self.cad_file_path = part.cad_file_path

        # Populate Quote Editor fields from the active part's data
        self._populate_quote_fields_from_part(part)

        # Update info label
        self.part_info_label.config(
            text=f"Active: Part {self.active_part_index + 1} - {part.cad_file_name}"
        )

        # Update the part selector dropdown
        if hasattr(self, 'part_selector') and hasattr(self, 'part_selector_var'):
            part_options = self.part_selector['values']
            if part_options and 0 <= self.active_part_index < len(part_options):
                self.part_selector_var.set(part_options[self.active_part_index])

        # Update status bar
        self.status_bar.config(
            text=f"Viewing Part {self.active_part_index + 1}: {part.cad_file_name}"
        )

    def _populate_quote_fields_from_part(self, part) -> None:
        """Populate Quote Editor fields from a QuoteData object."""
        # Family
        if hasattr(self, 'family_var'):
            family_value = getattr(part, 'part_family', '')
            if family_value:
                self.family_var.set(family_value)

        # Dimensions
        if part.part_dimensions:
            if hasattr(self, 'quote_fields'):
                if 'length' in self.quote_fields:
                    self.quote_fields['length'].delete(0, tk.END)
                    if part.part_dimensions.length:
                        self.quote_fields['length'].insert(0, str(part.part_dimensions.length))

                if 'width' in self.quote_fields:
                    self.quote_fields['width'].delete(0, tk.END)
                    if part.part_dimensions.width:
                        self.quote_fields['width'].insert(0, str(part.part_dimensions.width))

                if 'thickness' in self.quote_fields:
                    self.quote_fields['thickness'].delete(0, tk.END)
                    if part.part_dimensions.thickness:
                        self.quote_fields['thickness'].insert(0, str(part.part_dimensions.thickness))

                if 'diameter1' in self.quote_fields:
                    self.quote_fields['diameter1'].delete(0, tk.END)
                    if part.part_dimensions.diameter_major:
                        self.quote_fields['diameter1'].insert(0, str(part.part_dimensions.diameter_major))

                if 'diameter2' in self.quote_fields:
                    self.quote_fields['diameter2'].delete(0, tk.END)
                    if part.part_dimensions.diameter_minor:
                        self.quote_fields['diameter2'].insert(0, str(part.part_dimensions.diameter_minor))

        # Quantity
        if hasattr(self, 'qty_var'):
            self.qty_var.set(str(part.quantity))

        # Material
        if hasattr(self, 'material_var') and part.material_info:
            if part.material_info.material_name:
                self.material_var.set(part.material_info.material_name)

        # CMM level
        # (Note: This might need to be stored in QuoteData if not already)

        # Margin
        if hasattr(self, 'margin_var') and part.cost_summary:
            margin_pct = part.cost_summary.margin_rate * 100
            self.margin_var.set(str(margin_pct))

    def add_parts_to_order(self) -> None:
        """Add one or more CAD files as parts to the current order."""
        filenames = filedialog.askopenfilenames(
            title="Select CAD File(s) to Add",
            filetypes=[
                ("All CAD Files", "*.dwg *.dxf *.step *.stp *.iges *.igs"),
                ("DWG Files", "*.dwg"),
                ("DXF Files", "*.dxf"),
                ("All Files", "*.*")
            ]
        )

        if not filenames:
            return

        self.status_bar.config(text=f"Adding {len(filenames)} part(s) to order...")
        self.root.update_idletasks()

        added_count = 0
        for filename in filenames:
            try:
                # Load this file as a part (lightweight - just store the path initially)
                # Full extraction happens when user generates the quote
                from cad_quoter.pricing.QuoteDataHelper import QuoteData
                part = QuoteData()
                part.cad_file_path = filename
                part.cad_file_name = Path(filename).name
                from datetime import datetime
                part.extraction_timestamp = datetime.now().isoformat()

                # Add to order
                part_index = self.current_order.add_part(part)
                added_count += 1

                print(f"[AppV7] Added part {part_index + 1}: {part.cad_file_name}")

            except Exception as e:
                messagebox.showerror(
                    "Error Adding Part",
                    f"Failed to add {Path(filename).name}:\n{str(e)}"
                )
                continue

        # Refresh the parts list
        self._refresh_parts_list()

        # Auto-select the first part if nothing was selected before
        if self.active_part_index is None and self.current_order.parts:
            self.parts_listbox.selection_set(0)
            self.active_part_index = 0
            self._load_active_part_to_ui()

        self.status_bar.config(text=f"Added {added_count} part(s) to order. Total: {len(self.current_order.parts)} parts.")

    def remove_part_from_order(self) -> None:
        """Remove the selected part from the order."""
        if self.active_part_index is None:
            messagebox.showinfo("No Selection", "Please select a part to remove.")
            return

        part = self.current_order.get_part(self.active_part_index)
        if not part:
            return

        # Confirm removal
        result = messagebox.askyesno(
            "Confirm Removal",
            f"Remove part {self.active_part_index + 1} ({part.cad_file_name})?"
        )

        if not result:
            return

        # Remove the part
        self.current_order.remove_part(self.active_part_index)

        # Refresh the list
        self._refresh_parts_list()

        # Update active part index
        if not self.current_order.parts:
            # No parts left
            self.active_part_index = None
            self.cad_file_path = None
            self._cached_quote_data = None
            self.part_info_label.config(text="No parts in order")
            self.status_bar.config(text="Order is now empty")
        else:
            # Select the previous part, or the first part if we removed index 0
            if self.active_part_index >= len(self.current_order.parts):
                self.active_part_index = len(self.current_order.parts) - 1

            self.parts_listbox.selection_set(self.active_part_index)
            self._load_active_part_to_ui()

    def open_drawing_preview(self) -> None:
        """Open the drawing preview image file with the system default viewer.

        If background generation is in progress, waits for it to complete.
        If no image exists, generates on-demand (fallback for edge cases).
        Background generation makes this nearly instant in most cases.
        """
        # Check if background generation is in progress
        if self._drawing_generation_in_progress:
            self.status_bar.config(text="Drawing preview is being generated in background, please wait...")
            self.root.update_idletasks()

            # Wait for background thread to complete (with timeout)
            if self._drawing_generation_thread and self._drawing_generation_thread.is_alive():
                self._drawing_generation_thread.join(timeout=20.0)  # Max 20 second wait

            # Check if generation succeeded
            if not self._drawing_generation_success:
                messagebox.showerror(
                    "Generation Failed",
                    "Background drawing generation failed.\n\n"
                    "Please try loading the CAD file again."
                )
                self.status_bar.config(text="Drawing preview generation failed")
                return

            self.status_bar.config(text="Drawing preview ready!")
            self.root.update_idletasks()

        # If still no image path set, try to generate on-demand (fallback)
        if not self.drawing_image_path:
            if not self.cad_file_path:
                messagebox.showinfo(
                    "No CAD File",
                    "No CAD file loaded.\n\n"
                    "Please load a CAD file first using 'Load CAD & Vars' button."
                )
                return

            # Synchronous fallback generation (should rarely happen)
            self.status_bar.config(text="Generating drawing preview (this may take 5-15 seconds)...")
            self.root.update_idletasks()

            success = self._generate_drawing_image(self.cad_file_path)

            if not success:
                messagebox.showerror(
                    "Generation Failed",
                    "Failed to generate drawing preview.\n\n"
                    "Please check that the CAD file is valid and the DrawingRenderer is configured correctly."
                )
                self.status_bar.config(text="Drawing preview generation failed")
                return

            self.status_bar.config(text="Drawing preview generated successfully!")
            self.root.update_idletasks()

        # Verify image file exists
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

        # Header with label and copy button
        header_frame = ttk.Frame(table_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header_frame.columnconfigure(0, weight=1)

        label = ttk.Label(header_frame, text="Hole Operations", font=("Arial", 12, "bold"))
        label.grid(row=0, column=0, sticky="w")

        copy_button = ttk.Button(
            header_frame,
            text="Copy Table",
            command=self._copy_hole_table_to_clipboard
        )
        copy_button.grid(row=0, column=1, sticky="e")

        # Create a Treeview for the hole operations table
        columns = ("HOLE", "REF_DIAM", "QTY", "OPERATION")
        self.hole_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        self.hole_table.bind("<Control-c>", self._copy_hole_table_to_clipboard)
        self.hole_table.bind("<Control-C>", self._copy_hole_table_to_clipboard)

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

        # Part selector dropdown
        selector_frame = ttk.Frame(editor_scroll.inner)
        selector_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 5))

        ttk.Label(selector_frame, text="Select Part to Edit:", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.part_selector_var = tk.StringVar()
        self.part_selector = ttk.Combobox(
            selector_frame,
            textvariable=self.part_selector_var,
            state="readonly",
            width=60
        )
        self.part_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.part_selector.bind('<<ComboboxSelected>>', self._on_part_selector_changed)

        # Quote-Specific Variables section with Labelframe (like appV5)
        quote_frame = ttk.Labelframe(editor_scroll.inner, text="Quote-Specific Variables", padding=(10, 5))
        quote_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Define all the quote variables from the screenshot
        self.quote_fields = {}

        # Part Family mapping (internal key -> display name)
        self.part_families = {
            "Plates": "Plates (Die Plate / Shoe / Retainer / Stripper)",
            "Punches": "Punches (Punch / Pilot / Guide Post / Spring Pin)",
            "bushing_id_critical": "Guide Bushing / Ring Gauge (ID Critical)",
            "Sections_blocks": "Sections & Blocks (Cam / Hemmer / Die Chase / Sensor Block)",
            "Special_processes": "Special Processes (PM Die / Shear Blade / Extrude Hone)",
        }

        variables = [
            # Part Family (dropdown)
            ("Part Family", "Plates", "Select the part family type for process planning"),

            # Part Dimensions (optional - leave blank for OCR auto-detection)
            ("Length (in)", "", "Part length in inches (optional - leave blank for OCR auto-detection)"),
            ("Width (in)", "", "Part width in inches (optional - leave blank for OCR auto-detection)"),
            ("Thickness (in)", "", "Part thickness in inches (optional - leave blank for OCR auto-detection)"),

            # Diameter overrides for cylindrical parts (e.g., tapered punches)
            ("Diameter 1 (in)", "", "Primary diameter for cylindrical parts - detected from ⌀ symbol in CAD (optional override)"),
            ("Diameter 2 (in)", "", "Secondary diameter for tapered cylindrical parts like punches (optional override)"),

            # Quantity
            ("Quantity", "1", "Number of parts to quote - setup costs will be amortized across quantity, and material quantity pricing will be applied"),

            # Material Override
            ("Material", "", "Select material from dropdown to override auto-detection. Leave blank to use auto-detected material from CAD file."),

            # CMM Inspection Level (dropdown)
            ("CMM Inspection Level", "Full Inspection", "CMM inspection intensity: Full Inspection (1.0 min/hole), Critical Only (0.5 min/hole), or Spot Check (0.3 min/hole)"),

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

            # Special handling for Part Family dropdown
            if label_text == "Part Family":
                # Create combobox with display names
                display_names = list(self.part_families.values())
                field = ttk.Combobox(quote_frame, width=30, values=display_names, state="readonly")
                # Set default to the display name
                if default_value in self.part_families:
                    default_display = self.part_families[default_value]
                    field.set(default_display)
                else:
                    field.set(display_names[0])
            # Special handling for Material dropdown
            elif label_text == "Material":
                # Create combobox with material options from MaterialMapper
                material_options = material_mapper.get_dropdown_options()
                field = ttk.Combobox(quote_frame, width=30, values=material_options, state="readonly")
                # Leave empty by default (auto-detection)
                field.set("")
            # Special handling for CMM Inspection Level dropdown
            elif label_text == "CMM Inspection Level":
                # Create combobox with inspection level options
                inspection_levels = ["Full Inspection", "Critical Only", "Spot Check"]
                field = ttk.Combobox(quote_frame, width=30, values=inspection_levels, state="readonly")
                field.set(default_value)  # Default to "Full Inspection"
            # Entry field or dropdown
            elif default_value == "Number":
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
        """Clear cached CAD extraction results (QuoteData and DXF path)."""
        self._cached_quote_data = None
        self._cached_dxf_path = None  # Clear cached DXF conversion
        print("[AppV7] Cleared CAD extraction cache")

    def _get_ocr_cache_path(self, cad_file_path: str) -> Path:
        """
        Get the path to the OCR cache file for a given CAD file.

        Args:
            cad_file_path: Path to CAD file

        Returns:
            Path to the .ocr_cache.json sidecar file
        """
        cad_path = Path(cad_file_path)
        cache_path = cad_path.parent / f".{cad_path.name}.ocr_cache.json"
        return cache_path

    def _load_ocr_cache(self, cad_file_path: str) -> Optional[dict]:
        """
        Load cached OCR results from sidecar file.

        Args:
            cad_file_path: Path to CAD file

        Returns:
            Dictionary with cached OCR data or None if no cache exists
        """
        cache_path = self._get_ocr_cache_path(cad_file_path)

        if not cache_path.exists():
            print(f"[AppV7] No OCR cache found at {cache_path.name}")
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # Validate cache structure
            if 'dimensions' in cache_data and 'material' in cache_data:
                print(f"[AppV7] Loaded OCR cache from {cache_path.name}")
                return cache_data
            else:
                print(f"[AppV7] Invalid OCR cache format in {cache_path.name}")
                return None

        except Exception as e:
            print(f"[AppV7] Failed to load OCR cache: {e}")
            return None

    def _save_ocr_cache(self, cad_file_path: str, dimensions: tuple, material: str) -> None:
        """
        Save OCR results to sidecar cache file.

        Args:
            cad_file_path: Path to CAD file
            dimensions: Tuple of (length, width, thickness) in inches
            material: Detected material name
        """
        cache_path = self._get_ocr_cache_path(cad_file_path)

        try:
            import time
            cache_data = {
                'dimensions': {
                    'length': dimensions[0],
                    'width': dimensions[1],
                    'thickness': dimensions[2]
                },
                'material': material,
                'timestamp': Path(cad_file_path).stat().st_mtime,
                'cached_at': time.time()
            }

            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)

            print(f"[AppV7] Saved OCR cache to {cache_path.name}")

        except Exception as e:
            print(f"[AppV7] Failed to save OCR cache: {e}")

    def remove_overrides(self) -> None:
        """
        Clear all user overrides and remove the OCR cache file.

        This method:
        1. Clears all quote field entries (dimensions, material, rates, etc.)
        2. Deletes the OCR cache file if it exists
        3. Clears in-memory cached data
        """
        try:
            # Clear all quote field entries
            cleared_fields = []
            for label, field in self.quote_fields.items():
                # Skip Part Family as it should have a default value
                if label == "Part Family":
                    continue

                # Clear the field if it has content
                if isinstance(field, ttk.Entry):
                    current_value = field.get()
                    if current_value:
                        field.delete(0, tk.END)
                        cleared_fields.append(label)
                elif isinstance(field, ttk.Combobox):
                    current_value = field.get()
                    if current_value:
                        field.set('')
                        cleared_fields.append(label)

            # Delete OCR cache file if it exists
            cache_deleted = False
            if self.cad_file_path:
                cache_path = self._get_ocr_cache_path(self.cad_file_path)
                if cache_path.exists():
                    cache_path.unlink()
                    cache_deleted = True
                    print(f"[AppV7] Deleted OCR cache file: {cache_path.name}")

            # Clear in-memory caches
            self._clear_cad_cache()
            self._previous_quote_inputs = None

            # Show success message
            message_parts = []
            if cleared_fields:
                message_parts.append(f"Cleared {len(cleared_fields)} override field(s)")
            if cache_deleted:
                message_parts.append("Deleted OCR cache file")

            if message_parts:
                messagebox.showinfo("Overrides Removed", "\n".join(message_parts))
            else:
                messagebox.showinfo("Overrides Removed", "No overrides to clear")

            print("[AppV7] Successfully removed all overrides")

        except Exception as e:
            error_msg = f"Failed to remove overrides: {str(e)}"
            print(f"[AppV7] {error_msg}")
            messagebox.showerror("Error", error_msg)

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
            diameter_overrides = self._get_diameter_overrides()
            material_override = self._get_field_string("Material")
            cmm_inspection_level_override = self._get_cmm_inspection_level()

            # Try to load OCR cache if no manual dimensions provided (saves ~43 seconds!)
            ocr_cache_used = False
            if dimension_override is None:
                ocr_cache = self._load_ocr_cache(self.cad_file_path)
                if ocr_cache:
                    dims = ocr_cache['dimensions']
                    dimension_override = (dims['length'], dims['width'], dims['thickness'])
                    ocr_cache_used = True
                    print(f"[AppV7] Using cached OCR dimensions: {dimension_override} (saves ~43 seconds!)")

            machine_rate = self._get_field_float("Machine Rate ($/hr)", self.MACHINE_RATE)
            labor_rate = self._get_field_float("Labor Rate ($/hr)", self.LABOR_RATE)
            margin_percent = self._get_field_float("Margin (%)", 15.0)
            margin_rate = (margin_percent / 100.0) if margin_percent is not None else 0.15  # Convert percentage to decimal
            mcmaster_price_override = self._get_field_float("McMaster Price Override ($)")
            scrap_value_override = self._get_field_float("Scrap Value Override ($)")
            quantity = self._get_quantity()
            family_override = self._get_part_family()

            try:
                # Use cached DXF path if available to avoid redundant ODA conversion
                cad_path_for_quote = self._cached_dxf_path if hasattr(self, '_cached_dxf_path') and self._cached_dxf_path else self.cad_file_path

                self._cached_quote_data = extract_quote_data_from_cad(
                    cad_file_path=cad_path_for_quote,
                    machine_rate=machine_rate,
                    labor_rate=labor_rate,
                    margin_rate=margin_rate,
                    material_override=material_override,
                    dimension_override=dimension_override,
                    diameter_overrides=diameter_overrides,
                    mcmaster_price_override=mcmaster_price_override,
                    scrap_value_override=scrap_value_override,
                    quantity=quantity,
                    family_override=family_override,
                    cmm_inspection_level_override=cmm_inspection_level_override,
                    verbose=True
                )

                # Save OCR results to cache for next time (if we actually ran OCR)
                if not ocr_cache_used and dimension_override is None:
                    # OCR was just performed, save results to cache
                    dims = self._cached_quote_data.part_dimensions
                    material = self._cached_quote_data.material_info.material_name
                    self._save_ocr_cache(
                        self.cad_file_path,
                        (dims.length, dims.width, dims.thickness),
                        material
                    )

                if dimension_override:
                    print(f"[AppV7] Quote data cached (using manual/cached dimensions: {dimension_override})")
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

    def _get_diameter_overrides(self):
        """
        Read diameter overrides from Quote Editor fields for cylindrical parts.

        Returns:
            tuple of (diameter_1, diameter_2) or None if not provided.
            Either value can be None if only one is provided.
        """
        try:
            diameter_1_field = self.quote_fields.get("Diameter 1 (in)", None)
            diameter_2_field = self.quote_fields.get("Diameter 2 (in)", None)

            if not diameter_1_field or not diameter_2_field:
                return None

            diameter_1_val = diameter_1_field.get().strip()
            diameter_2_val = diameter_2_field.get().strip()

            # If both are empty, return None
            if not diameter_1_val and not diameter_2_val:
                return None

            # Parse values (allow one to be empty/None)
            diameter_1 = float(diameter_1_val) if diameter_1_val else None
            diameter_2 = float(diameter_2_val) if diameter_2_val else None

            # Log what was read
            if diameter_1 or diameter_2:
                print(f"[AppV7] Read diameter overrides from Quote Editor: Diameter 1={diameter_1}, Diameter 2={diameter_2}")

            # Validate positive values
            if diameter_1 is not None and diameter_1 <= 0:
                messagebox.showerror(
                    "Invalid Diameter",
                    "Diameter 1 must be a positive number."
                )
                return None
            if diameter_2 is not None and diameter_2 <= 0:
                messagebox.showerror(
                    "Invalid Diameter",
                    "Diameter 2 must be a positive number."
                )
                return None

            return (diameter_1, diameter_2)

        except ValueError:
            messagebox.showerror(
                "Invalid Diameter",
                "Please enter valid numbers for diameters."
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

    def _get_quantity(self) -> int:
        """
        Get the quantity value from Quote Editor field.

        Returns:
            Integer quantity value (minimum 1, default 1)
        """
        try:
            quantity = self._get_field_float("Quantity", 1.0)
            if quantity is None or quantity < 1:
                return 1
            return int(quantity)
        except Exception:
            return 1

    def _get_part_family(self) -> str:
        """
        Get the selected part family, converting from display name to internal key.

        Returns:
            Part family internal key (e.g., "Plates", "Punches", etc.)
        """
        try:
            field = self.quote_fields.get("Part Family", None)
            if not field:
                return "Plates"  # Default

            display_name = field.get().strip()
            if not display_name:
                return "Plates"  # Default

            # Convert display name back to internal key
            for key, display in self.part_families.items():
                if display == display_name:
                    return key

            return "Plates"  # Default if not found

        except Exception:
            return "Plates"  # Default on error

    def _get_cmm_inspection_level(self) -> Optional[str]:
        """
        Get the selected CMM inspection level, converting from display name to internal format.

        Returns:
            CMM inspection level in lowercase with underscores (e.g., "full_inspection")
            or None if not set
        """
        try:
            field = self.quote_fields.get("CMM Inspection Level", None)
            if not field:
                return None

            display_name = field.get().strip()
            if not display_name:
                return None

            # Convert display name to internal format
            level_mapping = {
                "Full Inspection": "full_inspection",
                "Critical Only": "critical_only",
                "Spot Check": "spot_check"
            }

            return level_mapping.get(display_name, None)

        except Exception:
            return None  # Return None on error to use default

    def _get_current_quote_inputs(self) -> dict:
        """
        Capture all quote inputs that affect CAD extraction and pricing.

        Returns a dictionary of current input values for comparison.
        Used to determine if cache should be invalidated.
        """
        return {
            'material': self._get_field_string("Material", ""),
            'length': self._get_field_string("Length (in)", ""),
            'width': self._get_field_string("Width (in)", ""),
            'thickness': self._get_field_string("Thickness (in)", ""),
            'diameter_1': self._get_field_string("Diameter 1 (in)", ""),
            'diameter_2': self._get_field_string("Diameter 2 (in)", ""),
            'machine_rate': self._get_field_string("Machine Rate ($/hr)", "90"),
            'labor_rate': self._get_field_string("Labor Rate ($/hr)", "90"),
            'margin': self._get_field_string("Margin (%)", "15"),
            'mcmaster_override': self._get_field_string("McMaster Price Override ($)", ""),
            'scrap_override': self._get_field_string("Scrap Value Override ($)", ""),
            'quantity': self._get_field_string("Quantity", "1"),
            'part_family': self._get_part_family(),
        }

    def _quote_inputs_changed(self) -> bool:
        """
        Check if any quote inputs have changed since last generation.

        Returns:
            True if inputs changed (cache should be cleared)
            False if inputs unchanged (cache can be reused)
        """
        current_inputs = self._get_current_quote_inputs()

        # First time generating quote - consider it changed
        if self._previous_quote_inputs is None:
            self._previous_quote_inputs = current_inputs
            return True

        # Compare current inputs to previous
        if current_inputs != self._previous_quote_inputs:
            if hasattr(self, 'status_bar'):
                self.status_bar.config(text="Input changed - regenerating quote data...")
            self._previous_quote_inputs = current_inputs
            return True

        # Inputs unchanged - can reuse cache
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text="Using cached quote data (inputs unchanged)...")
        return False

    def _find_existing_drawing_image(self, cad_filename: str) -> bool:
        """
        Check if a drawing image already exists for the CAD file.

        This is a fast operation (<1ms) that only checks for existing files.
        Does NOT generate a new image if not found.

        Args:
            cad_filename: Path to CAD file

        Returns:
            True if existing image found, False otherwise
        """
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
                return True

        # No existing image found
        self.drawing_image_path = None
        print(f"[AppV7] No existing drawing image found for {base_name}")
        return False

    def _generate_drawing_image(self, cad_filename: str) -> bool:
        """
        Generate a drawing preview image for the CAD file.

        This is a slow operation (5-15 seconds) that renders the CAD file to PNG.
        Only called when user explicitly requests preview and no cached image exists.

        Args:
            cad_filename: Path to CAD file

        Returns:
            True if generation succeeded, False otherwise
        """
        cad_path = Path(cad_filename)
        base_name = cad_path.stem

        print(f"[AppV7] Generating drawing preview for {base_name}...")

        try:
            from tools.paddle_dims_extractor import DrawingRenderer

            # Generate PNG in the same directory as the CAD file
            output_png = str(cad_path.parent / f"{base_name}.png")

            # Create renderer and generate image
            renderer = DrawingRenderer(verbose=False)
            renderer.render(str(cad_path), output_png)

            self.drawing_image_path = output_png
            print(f"[AppV7] Successfully generated drawing image: {output_png}")
            return True

        except Exception as e:
            print(f"[AppV7] Failed to generate drawing image: {e}")
            self.drawing_image_path = None
            return False

    def _generate_drawing_image_background(self, cad_filename: str) -> None:
        """
        Generate drawing image in background thread (non-blocking).

        This allows the UI to remain responsive while the image is being generated.
        Sets internal flags that can be checked by open_drawing_preview().

        Args:
            cad_filename: Path to CAD file
        """
        def _background_worker():
            """Worker function that runs in background thread."""
            self._drawing_generation_in_progress = True
            self._drawing_generation_success = False

            print(f"[AppV7] Starting background drawing generation...")

            # Call the synchronous generation method
            success = self._generate_drawing_image(cad_filename)

            self._drawing_generation_success = success
            self._drawing_generation_in_progress = False

            if success:
                print(f"[AppV7] Background drawing generation completed successfully")
            else:
                print(f"[AppV7] Background drawing generation failed")

        # Create and start background thread
        self._drawing_generation_thread = threading.Thread(
            target=_background_worker,
            daemon=True,
            name="DrawingGenerationThread"
        )
        self._drawing_generation_thread.start()
        print(f"[AppV7] Drawing generation started in background (non-blocking)")

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

                # Reset previous quote inputs so cache will be regenerated
                self._previous_quote_inputs = None

                # Pre-convert DWG to DXF once to avoid multiple ODA converter invocations
                # This cached path will be used for hole table extraction AND quote generation
                file_for_extraction = filename
                if filename.lower().endswith('.dwg'):
                    try:
                        from cad_quoter.geometry import convert_dwg_to_dxf
                        self.status_bar.config(text=f"Converting DWG to DXF (one-time)...")
                        self.root.update_idletasks()
                        dxf_path = convert_dwg_to_dxf(filename)
                        if dxf_path:
                            self._cached_dxf_path = dxf_path
                            file_for_extraction = dxf_path
                            print(f"[AppV7] Cached DXF conversion: {Path(dxf_path).name}")
                    except Exception as e:
                        print(f"[AppV7] DWG conversion failed, will retry per-function: {e}")
                        # Fall back to original - each function will try its own conversion

                # Load and extract hole table data using cached DXF if available
                self._extract_and_display_hole_table(file_for_extraction)

                # Check for existing drawing image (fast, <1ms)
                has_existing_image = self._find_existing_drawing_image(filename)

                # If no existing image, start background generation (non-blocking)
                # This allows UI to remain responsive while image is being generated
                if not has_existing_image:
                    self._generate_drawing_image_background(filename)
                    status_msg = "CAD file loaded. Review the Quote Editor and generate the quote. (Drawing preview generating in background...)"
                else:
                    status_msg = "CAD file loaded. Review the Quote Editor and generate the quote. (Drawing preview available)"

                # Add this file as a part to the order (for backward compatibility)
                # Clear existing order if there's only one old part
                if len(self.current_order.parts) <= 1:
                    from cad_quoter.pricing.QuoteDataHelper import QuoteData, OrderData
                    self.current_order = OrderData()  # Reset order

                    # Create a new part
                    part = QuoteData()
                    part.cad_file_path = filename
                    part.cad_file_name = Path(filename).name
                    from datetime import datetime
                    part.extraction_timestamp = datetime.now().isoformat()

                    # Add to order
                    self.current_order.add_part(part)
                    self.active_part_index = 0

                    # Refresh parts list
                    self._refresh_parts_list()
                    self.parts_listbox.selection_set(0)

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

    def _copy_hole_table_to_clipboard(self, event: tk.Event | None = None) -> str | None:
        """Copy selected rows (or the whole table) to the clipboard."""
        if not hasattr(self, "hole_table"):
            return "break" if event else None

        selected_items = self.hole_table.selection()
        if not selected_items:
            selected_items = self.hole_table.get_children()
            if not selected_items:
                return "break" if event else None

        headers = ("HOLE", "REF DIAM", "QTY", "OPERATION")
        rows = ["\t".join(headers)]

        for item in selected_items:
            values = self.hole_table.item(item, "values")
            rows.append("\t".join(str(value) for value in values))

        clipboard_text = "\n".join(rows)

        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(clipboard_text)
            self.status_bar.config(text=f"Copied {len(selected_items)} hole rows to clipboard.")
        except Exception as e:
            messagebox.showerror("Clipboard Error", f"Unable to copy hole table:\n{e}")

        return "break" if event else None

    def _format_weight(self, weight_lbs: float) -> str:
        """Format weight in lb and oz."""
        lbs = int(weight_lbs)
        oz = (weight_lbs - lbs) * 16
        return f"{lbs} lb {oz:.1f} oz"

    def _abbreviate_scrap_source(self, source_label: str) -> str:
        """Abbreviate long scrap price source names for display."""
        if "ScrapMetalBuyers" in source_label:
            return "SMB"
        elif "Wieland" in source_label:
            return "Wieland"
        elif "house_rate" in source_label.lower():
            return "house rate"
        return source_label

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

            # Get quantity for total cost calculation (from temp variable set in main thread)
            quantity = getattr(self, '_temp_quantity', 1)

            # Check if overrides were used (read from temp variables set in main thread)
            material_override = getattr(self, '_temp_material_override', None)
            mcmaster_price_override = getattr(self, '_temp_mcmaster_override', None)
            scrap_value_override = getattr(self, '_temp_scrap_override', None)

            # Check if we have valid price data
            if stock_info.mcmaster_price is None:
                price_str = "Price N/A"
            else:
                if mcmaster_price_override is not None:
                    price_str = f"(MANUAL) ${stock_info.mcmaster_price:,.2f}"
                elif hasattr(stock_info, 'price_is_estimated') and stock_info.price_is_estimated:
                    price_str = f"(ESTIMATED) ${stock_info.mcmaster_price:,.2f}"
                else:
                    price_str = f"${stock_info.mcmaster_price:,.2f}"

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
                source_label = scrap_info.scrap_price_source if scrap_info.scrap_price_source else "Scrap"
                source_label = self._abbreviate_scrap_source(source_label)

                report.append(f"  Scrap Price: ${scrap_info.scrap_price_per_lb:.2f} / lb @ {source_label}")
            else:
                report.append(f"  Scrap Price: N/A")

            report.append("")
            report.append("")

            # Cost breakdown
            # Format price with label handling - keep alignment consistent
            if stock_info.mcmaster_price is None:
                formatted_price = price_str  # "Price N/A"
            else:
                # Separate label from price for proper alignment
                if mcmaster_price_override is not None:
                    formatted_price = f"(MANUAL) ${stock_info.mcmaster_price:>8,.2f}"
                elif hasattr(stock_info, 'price_is_estimated') and stock_info.price_is_estimated:
                    formatted_price = f"(ESTIMATED) ${stock_info.mcmaster_price:>5,.2f}"
                else:
                    formatted_price = f"${stock_info.mcmaster_price:>13,.2f}"

            # Format stock piece label - only show McMaster part number if we have a real one
            if stock_info.mcmaster_part_number and stock_info.mcmaster_part_number != "N/A":
                stock_label = f"  Stock Piece (McMaster part {stock_info.mcmaster_part_number})"
            else:
                stock_label = "  Stock Piece (no McMaster match)"

            # Show per-unit or total costs based on quantity
            if quantity > 1:
                report.append(f"{stock_label} × {quantity}".ljust(50) + f"{formatted_price:>24}")
            else:
                report.append(f"{stock_label}".ljust(50) + f"{formatted_price:>24}")

            if stock_info.mcmaster_price is not None:
                if quantity > 1:
                    report.append(f"  Tax (${cost_breakdown.tax:.2f} × {quantity})".ljust(50) + f"+${cost_breakdown.tax * quantity:>22.2f}")
                    report.append(f"  Shipping".ljust(50) + f"+${cost_breakdown.shipping:>22.2f}")
                else:
                    report.append(f"  Tax".ljust(50) + f"+${cost_breakdown.tax:>22.2f}")
                    report.append(f"  Shipping".ljust(50) + f"+${cost_breakdown.shipping:>22.2f}")

            if cost_breakdown.scrap_credit > 0:
                # Use the scrap price source from scrap_info (could be Wieland, ScrapMetalBuyers, or house_rate)
                source_label = scrap_info.scrap_price_source if scrap_info.scrap_price_source else "Scrap"
                source_label = self._abbreviate_scrap_source(source_label)

                scrap_weight_formatted = self._format_weight(scrap_info.total_scrap_weight)
                scrap_credit_line = f"  Scrap Credit @ {source_label} ${scrap_info.scrap_price_per_lb:.2f}/lb × {scrap_weight_formatted}"
                if scrap_value_override is not None:
                    scrap_credit_line += " (MANUAL)"

                if quantity > 1:
                    scrap_credit_line += f" × {quantity}"
                    report.append(f"{scrap_credit_line.ljust(50)}-${cost_breakdown.scrap_credit * quantity:>22.2f}")
                else:
                    report.append(f"{scrap_credit_line.ljust(50)}-${cost_breakdown.scrap_credit:>22.2f}")

            report.append(" " * 50 + "-" * 24)

            if stock_info.mcmaster_price is not None:
                self.direct_cost_total = cost_breakdown.net_material_cost
                if quantity > 1:
                    report.append(f"  Total Material Cost (per unit):".ljust(50) + f"${cost_breakdown.net_material_cost:>23.2f}")
                    report.append(f"  Total Material Cost ({quantity} units):".ljust(50) + f"${cost_breakdown.net_material_cost * quantity:>23.2f}")
                else:
                    report.append(f"  Total Material Cost :".ljust(50) + f"${cost_breakdown.net_material_cost:>23.2f}")
            else:
                self.direct_cost_total = None
                report.append(f"  Total Material Cost :".ljust(50) + "Price N/A".rjust(24))

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

            # Check if this is a punch part
            is_punch = quote_data.raw_plan and quote_data.raw_plan.get("is_punch", False)

            # For punch parts, show punch-specific machine hours report
            if is_punch:
                # Get machine rate for display
                machine_rate = getattr(self, '_temp_machine_rate', self.MACHINE_RATE)
                machine_rate_label = f"@ ${machine_rate:.2f}/hr"
                if machine_rate != self.MACHINE_RATE:
                    machine_rate_label += " (OVERRIDDEN)"

                report = []
                report.append("MACHINE HOURS ESTIMATION - PUNCH PART")
                report.append("=" * 74)
                report.append(f"Material: {quote_data.material_info.material_name}")
                report.append(f"Part type: {quote_data.raw_plan.get('punch_features', {}).get('family', 'punch')}")
                report.append(f"Shape: {quote_data.raw_plan.get('punch_features', {}).get('shape_type', 'round')}")
                report.append("")

                report.append("MACHINE TIME BREAKDOWN")
                report.append("-" * 74)
                report.append(f"  Turning (rough + finish):        {machine_hours.total_milling_minutes:>10.2f} minutes")
                report.append(f"  Grinding (OD/ID/face):           {machine_hours.total_grinding_minutes:>10.2f} minutes")
                report.append(f"  Drilling:                        {machine_hours.total_drill_minutes:>10.2f} minutes")
                report.append(f"  Tapping:                         {machine_hours.total_tap_minutes:>10.2f} minutes")
                report.append(f"  EDM:                             {machine_hours.total_edm_minutes:>10.2f} minutes")
                if machine_hours.total_edge_break_minutes > 0:
                    report.append(f"  Edge break / deburr:             {machine_hours.total_edge_break_minutes:>10.2f} minutes")
                if machine_hours.total_etch_minutes > 0:
                    report.append(f"  Etch / marking:                  {machine_hours.total_etch_minutes:>10.2f} minutes")
                if machine_hours.total_polish_minutes > 0:
                    report.append(f"  Polish contour:                  {machine_hours.total_polish_minutes:>10.2f} minutes")
                report.append(f"  Other (chamfer/polish/saw):      {machine_hours.total_other_minutes:>10.2f} minutes")
                report.append(f"  Inspection:                      {machine_hours.total_cmm_minutes:>10.2f} minutes")
                report.append("-" * 74)
                report.append(f"  TOTAL MACHINE TIME:              {machine_hours.total_minutes:>10.2f} minutes")
                report.append(f"                                   {machine_hours.total_hours:>10.2f} hours")
                report.append("")
                report.append("=" * 74)
                report.append(f"TOTAL MACHINE COST: ${machine_hours.machine_cost:.2f} {machine_rate_label}")
                report.append("=" * 74)
                report.append("")

                self.machine_cost_total = machine_hours.machine_cost
                return "\n".join(report)

            # Check if this is a die section (form-only, no drilled holes expected)
            is_die_section = (quote_data.raw_plan and
                              quote_data.raw_plan.get("meta", {}).get("family") == "Sections_blocks")
            is_carbide_die_section = (quote_data.raw_plan and
                                       quote_data.raw_plan.get("meta", {}).get("is_carbide_die_section", False))

            # Check if there are ANY hole operations (drill, tap, cbore, cdrill, jig grind, or EDM)
            has_any_hole_ops = (machine_hours.drill_operations or
                                machine_hours.tap_operations or
                                machine_hours.cbore_operations or
                                machine_hours.cdrill_operations or
                                machine_hours.jig_grind_operations or
                                machine_hours.edm_operations)

            if not has_any_hole_ops:
                # For die sections, provide appropriate messaging instead of implying "no machining"
                if is_die_section or is_carbide_die_section:
                    # Get machine rate for display
                    machine_rate = getattr(self, '_temp_machine_rate', self.MACHINE_RATE)
                    machine_rate_label = f"@ ${machine_rate:.2f}/hr"
                    if machine_rate != self.MACHINE_RATE:
                        machine_rate_label += " (OVERRIDDEN)"

                    # Generate die section machine hours report
                    report = []
                    report.append("MACHINE HOURS ESTIMATION - DIE SECTION / FORM BLOCK")
                    report.append("=" * 74)
                    report.append(f"Material: {quote_data.material_info.material_name}")
                    if quote_data.raw_plan:
                        meta = quote_data.raw_plan.get("meta", {})
                        report.append(f"Part type: {meta.get('sub_type', 'die_section')}")
                        if meta.get("is_carbide_die_section"):
                            report.append("Classification: Carbide Form Die Section")
                        report.append(f"Complexity: {meta.get('complexity_level', 'medium').title()}")
                    report.append("")

                    report.append("NOTE: This is a form die section (no drilled/tapped holes)")
                    report.append("Machining operations: grinding, wire EDM, form cutting")
                    report.append("")

                    report.append("MACHINE TIME BREAKDOWN")
                    report.append("-" * 74)
                    report.append(f"  Grinding (square-up/finish):     {machine_hours.total_grinding_minutes:>10.2f} minutes")
                    report.append(f"  EDM (form cutting):              {machine_hours.total_edm_minutes:>10.2f} minutes")
                    if machine_hours.total_edge_break_minutes > 0:
                        report.append(f"  Edge break / deburr:             {machine_hours.total_edge_break_minutes:>10.2f} minutes")
                    if machine_hours.total_etch_minutes > 0:
                        report.append(f"  Etch / marking:                  {machine_hours.total_etch_minutes:>10.2f} minutes")
                    if machine_hours.total_polish_minutes > 0:
                        report.append(f"  Polish contour:                  {machine_hours.total_polish_minutes:>10.2f} minutes")
                    report.append(f"  Other (chamfer/polish):          {machine_hours.total_other_minutes:>10.2f} minutes")
                    report.append(f"  Inspection:                      {machine_hours.total_cmm_minutes:>10.2f} minutes")
                    report.append("-" * 74)
                    report.append(f"  TOTAL MACHINE TIME:              {machine_hours.total_minutes:>10.2f} minutes")
                    report.append(f"                                   {machine_hours.total_hours:>10.2f} hours")
                    report.append("")
                    report.append("=" * 74)
                    report.append(f"TOTAL MACHINE COST: ${machine_hours.machine_cost:.2f} {machine_rate_label}")
                    report.append("=" * 74)
                    report.append("")

                    self.machine_cost_total = machine_hours.machine_cost
                    return "\n".join(report)
                else:
                    # Check if there's machine time from plan operations (milling, grinding, EDM)
                    # even without drilled/tapped holes
                    has_plan_machine_time = (
                        machine_hours.total_milling_minutes > 0 or
                        machine_hours.total_grinding_minutes > 0 or
                        machine_hours.total_edm_minutes > 0 or
                        machine_hours.total_other_minutes > 0
                    )

                    if has_plan_machine_time:
                        # Get machine rate for display
                        machine_rate = getattr(self, '_temp_machine_rate', self.MACHINE_RATE)
                        machine_rate_label = f"@ ${machine_rate:.2f}/hr"
                        if machine_rate != self.MACHINE_RATE:
                            machine_rate_label += " (OVERRIDDEN)"

                        # Generate machine hours report for parts with plan operations but no holes
                        report = []
                        report.append("MACHINE HOURS ESTIMATION - FORM / MACHINED PART")
                        report.append("=" * 74)
                        report.append(f"Material: {quote_data.material_info.material_name}")
                        report.append("")

                        report.append("INFO: No hole operations found in CAD file.")
                        report.append("Machine time calculated from turning/grinding/EDM operations.")
                        report.append("")

                        report.append("MACHINE TIME BREAKDOWN")
                        report.append("-" * 74)
                        report.append(f"  Milling/Turning:                 {machine_hours.total_milling_minutes:>10.2f} minutes")
                        report.append(f"  Grinding (OD/face/form):         {machine_hours.total_grinding_minutes:>10.2f} minutes")
                        report.append(f"  EDM:                             {machine_hours.total_edm_minutes:>10.2f} minutes")
                        if machine_hours.total_edge_break_minutes > 0:
                            report.append(f"  Edge break / deburr:             {machine_hours.total_edge_break_minutes:>10.2f} minutes")
                        if machine_hours.total_etch_minutes > 0:
                            report.append(f"  Etch / marking:                  {machine_hours.total_etch_minutes:>10.2f} minutes")
                        if machine_hours.total_polish_minutes > 0:
                            report.append(f"  Polish contour:                  {machine_hours.total_polish_minutes:>10.2f} minutes")
                        report.append(f"  Other (chamfer/polish):          {machine_hours.total_other_minutes:>10.2f} minutes")
                        report.append(f"  Inspection:                      {machine_hours.total_cmm_minutes:>10.2f} minutes")
                        report.append("-" * 74)
                        report.append(f"  TOTAL MACHINE TIME:              {machine_hours.total_minutes:>10.2f} minutes")
                        report.append(f"                                   {machine_hours.total_hours:>10.2f} hours")
                        report.append("")
                        report.append("=" * 74)
                        report.append(f"TOTAL MACHINE COST: ${machine_hours.machine_cost:.2f} {machine_rate_label}")
                        report.append("=" * 74)
                        report.append("")

                        self.machine_cost_total = machine_hours.machine_cost
                        return "\n".join(report)
                    else:
                        # Truly no machine operations - set cost to 0 with warning
                        self.machine_cost_total = 0.0
                        return ("WARNING: No hole operations or machining operations found in CAD file.\n"
                                "Machine Cost: $0.00\n\n"
                                "Note: If this part requires machining, verify:\n"
                                "  - Part family classification (e.g., Punches, Sections_blocks)\n"
                                "  - Material detection\n"
                                "  - Dimension extraction")

            # Format helper functions
            def format_drill_group(op):
                return (f"Hole {op.hole_id} | Ø {op.diameter:.4f}\" x {op.qty:>3} | "
                        f"depth {op.depth:.3f}\" | {op.sfm:.0f} sfm | "
                        f"{op.ipr:.4f} ipr | {op.time_per_hole:>5.2f} min/h | "
                        f"tot. {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_jig_grind_group(op):
                return (f"Hole {op.hole_id} | Ø {op.diameter:.4f}\" x {op.qty:>3} | "
                        f"depth {op.depth:.3f}\" | "
                        f"{op.time_per_hole:>5.2f} min/h | "
                        f"tot. {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_tap_group(op):
                return (f"Hole {op.hole_id} | Ø {op.diameter:.4f}\" x {op.qty:>3} | "
                        f"depth {op.depth:.3f}\" | {op.tpi} TPI | "
                        f"{op.time_per_hole:>5.2f} min/h | "
                        f"tot. {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_cbore_group(op):
                # Add side indicator if available (front/back)
                if op.side:
                    hole_label = f"Hole {op.hole_id} – C'BORE {op.side.upper():5}"
                else:
                    hole_label = f"Hole {op.hole_id}"
                return (f"{hole_label} | Ø {op.diameter:.4f}\" x {op.qty:>3} | "
                        f"depth {op.depth:.3f}\" | {op.sfm:.0f} sfm | "
                        f"{op.time_per_hole:>5.2f} min/h | "
                        f"tot. {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_cdrill_group(op):
                return (f"Hole {op.hole_id} | Ø {op.diameter:.4f}\" x {op.qty:>3} | "
                        f"depth {op.depth:.3f}\" | "
                        f"{op.time_per_hole:>5.2f} min/h | "
                        f"tot. {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            def format_edm_group(op):
                """Format EDM (wire EDM profile) operation."""
                return (f"Hole {op.hole_id} | Starter Ø {op.diameter:.4f}\" x {op.qty:>3} | "
                        f"thickness {op.depth:.3f}\" | "
                        f"{op.time_per_hole:>5.2f} min/window | "
                        f"tot. {op.qty}x{op.time_per_hole:.2f} = {op.total_time:.2f} min")

            MILLING_DESC_WIDTH = 28
            MILLING_SEPARATOR_LENGTH = 106

            def format_milling_op(op):
                """Format a milling operation into a compact single line."""

                desc_raw = (op.op_description or "").strip()
                desc_short = desc_raw.replace("Square Up", "SQ UP")
                desc = desc_short[:MILLING_DESC_WIDTH].ljust(MILLING_DESC_WIDTH)

                width = op.width if op.width is not None else 0.0
                length = op.length if op.length is not None else 0.0
                tool_dia = op.tool_diameter if op.tool_diameter is not None else 0.0
                path_length = op.path_length if op.path_length is not None else 0.0
                time_minutes = op.time_minutes if op.time_minutes is not None else 0.0

                w_str = f"{width:.3f}"
                l_str = f"{length:.3f}"
                tool_str = f"{tool_dia:.3f}"
                path_str = f"{path_length:.1f}"
                time_str = f"{time_minutes:.2f}"

                line = (
                    f"{desc} | W {w_str} | L {l_str} | "
                    f"Tool Ø {tool_str} | Path {path_str} | Time (min) {time_str}"
                )

                return line

            def format_grinding_op(op):
                """Format grinding operation with all details"""
                lines = []
                lines.append(f"{op.op_description} | L={op.length:.3f}\" | W={op.width:.3f}\" | Area={op.area:.2f} sq in")
                lines.append(f"  stock_removed={op.stock_removed_total:.3f}\" | faces={op.faces} | volume={op.volume_removed:.3f} cu in")
                lines.append(f"  min_per_cuin={op.min_per_cuin:.1f} | material_factor={op.material_factor:.2f}")
                lines.append(f"  time = {op.volume_removed:.3f} × {op.min_per_cuin:.1f} × {op.material_factor:.2f} = {op.time_minutes:.2f} min")
                return "\n".join(lines)

            # Build the report
            report = []
            report.append("MACHINE HOURS ESTIMATION - DETAILED HOLE TABLE BREAKDOWN")
            report.append("=" * 74)
            report.append(f"Material: {quote_data.material_info.material_name}")
            report.append(f"Thickness: {quote_data.part_dimensions.thickness:.3f}\"")
            report.append(f"Hole entries: {machine_hours.hole_entries}")
            report.append(f"Total holes: {machine_hours.holes_total}")
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

            # WIRE EDM PROFILE - Time to cut profiles from starter holes
            if machine_hours.edm_operations:
                report.append("WIRE EDM PROFILE")
                report.append("-" * 74)
                for op in machine_hours.edm_operations:
                    report.append(format_edm_group(op))
                report.append(f"\nTotal Wire EDM Time: {machine_hours.total_edm_minutes:.2f} minutes")
                report.append("")

            # SQUARE-UP BLOCK - Enhanced rendering for square-up operations
            from cad_quoter.planning.process_planner import render_square_up_block
            # Convert dataclass objects to dicts for render_square_up_block
            milling_ops_dicts = []
            for op in (machine_hours.milling_operations or []):
                milling_ops_dicts.append({
                    'op_name': op.op_name,
                    'op_description': op.op_description,
                    'length': op.length,
                    'width': op.width,
                    'perimeter': op.perimeter,
                    'tool_diameter': op.tool_diameter,
                    'passes': op.passes,
                    'stepover': op.stepover,
                    'radial_stock': op.radial_stock,
                    'axial_step': op.axial_step,
                    'axial_passes': op.axial_passes,
                    'radial_passes': op.radial_passes,
                    'path_length': op.path_length,
                    'feed_rate': op.feed_rate,
                    'time_minutes': op.time_minutes,
                    '_used_override': op._used_override,
                    'override_time_minutes': op.override_time_minutes,
                })

            grinding_ops_dicts = []
            for op in (machine_hours.grinding_operations or []):
                grinding_ops_dicts.append({
                    'op_name': op.op_name,
                    'op_description': op.op_description,
                    'length': op.length,
                    'width': op.width,
                    'area': op.area,
                    'stock_removed_total': op.stock_removed_total,
                    'faces': op.faces,
                    'volume_removed': op.volume_removed,
                    'min_per_cuin': op.min_per_cuin,
                    'material_factor': op.material_factor,
                    'grind_material_factor': op.grind_material_factor,
                    'time_minutes': op.time_minutes,
                    '_used_override': op._used_override,
                })

            # Render the square-up block
            square_up_lines = render_square_up_block(
                plan=quote_data.raw_plan or {},
                milling_ops=milling_ops_dicts,
                grinding_ops=grinding_ops_dicts,
                setup_time_min=0.0,
                flip_time_min=0.0
            )

            if square_up_lines:
                report.append("")
                for line in square_up_lines:
                    report.append(line)
                report.append("")

            # TIME PER OP - MILLING (additional milling ops not in square-up block)
            non_square_up_milling = [
                op for op in (machine_hours.milling_operations or [])
                if op.op_name not in ('square_up_rough_sides', 'square_up_rough_faces', 'full_square_up_mill')
            ]
            if non_square_up_milling:
                report.append("TIME PER OP - MILLING (Other)")
                report.append("-" * MILLING_SEPARATOR_LENGTH)
                for op in non_square_up_milling:
                    report.append(format_milling_op(op))
                report.append("")

            # Show total milling time
            if machine_hours.milling_operations:
                report.append(
                    f"Total Milling Time (Square/Finish): {machine_hours.total_milling_minutes:.2f} minutes"
                )
                report.append("")

            # TIME PER OP - GRINDING (non-square-up grinding)
            non_square_up_grinding = [
                op for op in (machine_hours.grinding_operations or [])
                if op.op_name != 'wet_grind_square_all'
            ]
            if non_square_up_grinding:
                report.append("TIME PER OP - GRINDING (Other)")
                report.append("-" * 74)
                for op in non_square_up_grinding:
                    report.append(format_grinding_op(op))
                    report.append("")

            # Show total grinding time
            if machine_hours.grinding_operations:
                report.append(f"Total Wet Grind Time: {machine_hours.total_grinding_minutes:.2f} minutes")
                report.append("")

            # TIME PER OP - WATERJET (NEW: promoted to first-class operation)
            if machine_hours.waterjet_operations and len(machine_hours.waterjet_operations) > 0:
                report.append("TIME PER OP - WATERJET")
                report.append("-" * 74)
                for op in machine_hours.waterjet_operations:
                    desc = op.get('op_description', 'Waterjet operation')
                    length = op.get('length_in', 0.0)
                    thickness = op.get('thickness_in', 0.0)
                    pierce_count = op.get('pierce_count', 0)
                    time_min = op.get('time_min', 0.0)

                    # Format waterjet operation details
                    report.append(f"  {desc}")
                    report.append(f"    Cut length: {length:.1f}\" | Thickness: {thickness:.3f}\" | Pierces: {pierce_count}")
                    report.append(f"    Time: {time_min:.2f} min")
                    report.append("")

                report.append(f"Total Waterjet Time: {machine_hours.total_waterjet_minutes:.2f} minutes")
                report.append("")

            # CMM INSPECTION (if applicable)
            if machine_hours.total_cmm_minutes > 0:
                report.append("CMM INSPECTION")
                report.append("-" * 74)
                report.append(f"Holes to inspect: {machine_hours.cmm_holes_checked}")
                # Calculate actual minutes per hole from the total
                minutes_per_hole = machine_hours.total_cmm_minutes / machine_hours.cmm_holes_checked
                inspection_type = "full inspection" if minutes_per_hole >= 0.9 else ("spot check" if minutes_per_hole < 0.4 else "critical dimensions")
                report.append(f"Time per hole: {minutes_per_hole:.1f} min ({inspection_type})")
                report.append(f"Total CMM time: {machine_hours.cmm_holes_checked} holes × {minutes_per_hole:.1f} min = {machine_hours.total_cmm_minutes:.2f} min")
                report.append("")
                report.append(f"Note: CMM run time ({machine_hours.total_cmm_minutes:.2f} min) is billed as MACHINE TIME.")
                report.append(f"      CMM setup (30 min) is included in Labor → Inspection.")
                report.append("")

            # MACHINE TIME BREAKDOWN
            # Show all operation types individually for full transparency
            report.append("MACHINE TIME BREAKDOWN")
            report.append("-" * 74)
            if machine_hours.total_drill_minutes > 0:
                report.append(f"  Drilling:                         {machine_hours.total_drill_minutes:>10.2f} min")
            if machine_hours.total_jig_grind_minutes > 0:
                report.append(f"  Jig grind:                        {machine_hours.total_jig_grind_minutes:>10.2f} min")
            if machine_hours.total_tap_minutes > 0:
                report.append(f"  Tap:                              {machine_hours.total_tap_minutes:>10.2f} min")
            if machine_hours.total_cdrill_minutes > 0:
                report.append(f"  Center drill:                     {machine_hours.total_cdrill_minutes:>10.2f} min")
            if machine_hours.total_cbore_minutes > 0:
                report.append(f"  Counterbore:                      {machine_hours.total_cbore_minutes:>10.2f} min")
            if machine_hours.total_milling_minutes > 0:
                report.append(f"  Milling (square-up, face):        {machine_hours.total_milling_minutes:>10.2f} min")
            if machine_hours.total_grinding_minutes > 0:
                report.append(f"  Wet grind:                        {machine_hours.total_grinding_minutes:>10.2f} min")
            if machine_hours.total_edm_minutes > 0:
                report.append(f"  EDM:                              {machine_hours.total_edm_minutes:>10.2f} min")
            if machine_hours.total_waterjet_minutes > 0:
                report.append(f"  Waterjet:                         {machine_hours.total_waterjet_minutes:>10.2f} min")
            if machine_hours.total_edge_break_minutes > 0:
                report.append(f"  Edge break / deburr:              {machine_hours.total_edge_break_minutes:>10.2f} min")
            if machine_hours.total_etch_minutes > 0:
                report.append(f"  Etch / marking:                   {machine_hours.total_etch_minutes:>10.2f} min")
            if machine_hours.total_polish_minutes > 0:
                report.append(f"  Polish contour:                   {machine_hours.total_polish_minutes:>10.2f} min")

            # NEW: Detailed "Other Operations" breakdown (if available)
            if machine_hours.total_other_minutes > 0:
                # Check if detailed breakdown is available
                if machine_hours.other_ops_detail and len(machine_hours.other_ops_detail) > 0:
                    report.append(f"  Other operations (Total):         {machine_hours.total_other_minutes:>10.2f} min")
                    # Show detail breakdown with indentation - align minutes with main items
                    for detail in machine_hours.other_ops_detail:
                        label = detail.get('label', 'Unknown operation')
                        minutes = detail.get('minutes', 0.0)
                        # Use 36 char width for label to align with main items (2 spaces + bullet + space + label)
                        report.append(f"    • {label:<36} {minutes:>10.1f} min")
                else:
                    # Fallback to simple total if no detail available (backward compatibility)
                    report.append(f"  Other operations:                 {machine_hours.total_other_minutes:>10.2f} min")

            if machine_hours.total_cmm_minutes > 0:
                report.append(f"  CMM (machine time):               {machine_hours.total_cmm_minutes:>10.2f} min")
            report.append("-" * 74)

            # Summary (read from temp variable set in main thread to avoid Tkinter widget access)
            machine_rate = getattr(self, '_temp_machine_rate', self.MACHINE_RATE)
            machine_rate_label = f"@ ${machine_rate:.2f}/hr"
            if machine_rate != self.MACHINE_RATE:
                machine_rate_label += " (OVERRIDDEN)"

            report.append(f"  TOTAL MACHINE TIME:               {machine_hours.total_minutes:>10.2f} min ({machine_hours.total_hours:.2f} hours)")
            report.append("")
            report.append("=" * 74)
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

            # Finishing / Deburr with detailed breakdown
            if labor_hours.finishing_detail and len(labor_hours.finishing_detail) > 0:
                report.append(f"  Finishing / Deburr (Total):      {labor_hours.finishing_minutes:>10.2f} minutes")
                # Show detail breakdown with indentation - align minutes with main items
                for detail in labor_hours.finishing_detail:
                    label = detail.get('label', 'Unknown operation')
                    minutes = detail.get('minutes', 0.0)
                    # Use 37 char width for label to align with main items (4 spaces + bullet + space + label)
                    report.append(f"    • {label:<33} {minutes:>10.1f} minutes")
            else:
                report.append(f"  Finishing / Deburr:              {labor_hours.finishing_minutes:>10.2f} minutes")

            # Show misc overhead if non-zero
            if abs(labor_hours.misc_overhead_minutes) > 0.01:
                report.append(f"  Misc / Overhead:                 {labor_hours.misc_overhead_minutes:>10.2f} minutes")
            report.append("-" * 74)
            report.append(f"  TOTAL LABOR TIME:                {labor_hours.total_minutes:>10.2f} minutes")
            report.append(f"                                   {labor_hours.total_hours:>10.2f} hours")

            # Show labor rate with override indicator (read from temp variable set in main thread)
            labor_rate = getattr(self, '_temp_labor_rate', self.LABOR_RATE)
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

    def _generate_multi_part_order_report(self) -> None:
        """Generate a report for a multi-part order."""
        self.output_text.delete(1.0, tk.END)

        # Order header
        header_lines = [
            "=" * 74,
            "MULTI-PART ORDER QUOTE",
            "=" * 74,
            f"Order ID: {self.current_order.order_id}",
            f"Total Parts: {len(self.current_order.parts)}",
            f"Generated: {self.current_order.order_timestamp}",
            "=" * 74,
            "",
            ""
        ]
        self.output_text.insert(tk.END, "\n".join(header_lines))

        # Generate report for each part
        all_parts_data = []
        for part_idx, part in enumerate(self.current_order.parts):
            try:
                # Part header
                part_header = [
                    "",
                    "█" * 74,
                    f"PART {part_idx + 1} OF {len(self.current_order.parts)} – {part.cad_file_name}",
                    "█" * 74,
                    ""
                ]
                self.output_text.insert(tk.END, "\n".join(part_header))

                # Set this part as active temporarily
                old_active_index = self.active_part_index
                self.active_part_index = part_idx

                # Extract quote data for this part if not already extracted
                if not part.cost_summary or not part.cost_summary.final_price:
                    # Need to extract - set up temporary vars from part data
                    self.cad_file_path = part.cad_file_path

                    # Clear cache to force fresh extraction (critical fix!)
                    self._cached_quote_data = None

                    # Get overrides from part data or use defaults
                    self._temp_machine_rate = self.MACHINE_RATE
                    self._temp_labor_rate = self.LABOR_RATE
                    self._temp_material_override = part.material_info.material_name if part.material_info else ""
                    self._temp_mcmaster_override = None
                    self._temp_scrap_override = None

                    # Extract quote data
                    from cad_quoter.pricing.QuoteDataHelper import extract_quote_data_from_cad
                    part_quote = self._get_or_create_quote_data()

                    # Update the part in the order with the extracted data
                    self.current_order.parts[part_idx] = part_quote
                    part = part_quote
                else:
                    # Costs already extracted, use this part as cache
                    self._cached_quote_data = part

                # Generate the three reports for this part
                labor_report = self._generate_labor_hours_report()
                machine_report = self._generate_machine_hours_report()
                direct_report = self._generate_direct_costs_report()

                # Insert part reports
                self.output_text.insert(tk.END, labor_report)
                self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")
                self.output_text.insert(tk.END, machine_report)
                self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")
                self.output_text.insert(tk.END, direct_report)

                # Part cost summary
                if part.cost_summary:
                    part_summary = [
                        "",
                        "",
                        "=" * 74,
                        f"PART {part_idx + 1} COST SUMMARY",
                        "=" * 74,
                        self._format_cost_summary_line("Direct Cost (per unit)", part.cost_summary.direct_cost or 0),
                        self._format_cost_summary_line("Machine Cost (per unit)", part.cost_summary.machine_cost or 0),
                        self._format_cost_summary_line("Labor Cost (per unit)", part.cost_summary.labor_cost or 0),
                        "-" * 74,
                        self._format_cost_summary_line("Total Cost (per unit)", part.cost_summary.total_cost or 0),
                        self._format_cost_summary_line(f"Margin ({part.cost_summary.margin_rate:.0%})", part.cost_summary.margin_amount or 0),
                        self._format_cost_summary_line("Final Price (per unit)", part.cost_summary.final_price or 0),
                    ]

                    if part.quantity > 1:
                        part_summary.extend([
                            "",
                            f"Quantity: {part.quantity} units",
                            "-" * 74,
                            self._format_cost_summary_line("Total Part Cost", part.cost_summary.total_total_cost or 0),
                            self._format_cost_summary_line("Total Part Price", part.cost_summary.total_final_price or 0),
                        ])

                    part_summary.append("")
                    self.output_text.insert(tk.END, "\n".join(part_summary))

                # Store part data for order summary
                all_parts_data.append(part)

                # Restore active index
                self.active_part_index = old_active_index

            except Exception as e:
                error_msg = f"\n\nERROR generating quote for Part {part_idx + 1}: {str(e)}\n\n"
                self.output_text.insert(tk.END, error_msg)
                print(f"[AppV7] Error generating part {part_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Generate order-level summary
        self._generate_order_level_summary(all_parts_data)

        # Show output and update status
        self.notebook.select(self.output_tab)
        self.status_bar.config(text=f"Multi-part order quote generated ({len(all_parts_data)} parts)!")

    def _generate_order_level_summary(self, parts_data) -> None:
        """Generate the order-level summary section."""
        summary_lines = [
            "",
            "",
            "█" * 74,
            "█" * 74,
            "TOTAL ORDER COSTS",
            "█" * 74,
            "█" * 74,
            ""
        ]

        # Calculate order totals
        parts_subtotal_no_shipping = 0.0
        total_parts_cost = 0.0
        total_weight_lb = 0.0

        for part in parts_data:
            if part.cost_summary:
                # Get part total (excluding shipping)
                if part.quantity > 1:
                    part_total = part.cost_summary.total_final_price
                else:
                    part_total = part.cost_summary.final_price

                # Subtract per-part shipping to get cost without shipping
                if part.direct_cost_breakdown and part.direct_cost_breakdown.shipping:
                    part_shipping = part.direct_cost_breakdown.shipping * part.quantity
                    parts_subtotal_no_shipping += (part_total - part_shipping)
                else:
                    parts_subtotal_no_shipping += part_total

                # Also track total with per-part shipping for comparison
                total_parts_cost += part_total

            # Calculate weight
            if part.stock_info and part.stock_info.mcmaster_weight:
                total_weight_lb += part.stock_info.mcmaster_weight * part.quantity

        # Calculate order-level shipping
        from cad_quoter.pricing.mcmaster_helpers import estimate_mcmaster_shipping
        order_shipping = estimate_mcmaster_shipping(total_weight_lb)

        # Calculate grand total
        order_total = parts_subtotal_no_shipping + order_shipping

        # Build summary
        summary_lines.append(f"Number of parts: {len(parts_data)}")
        summary_lines.append("")
        summary_lines.append("-" * 74)
        summary_lines.append(self._format_cost_summary_line("Parts Subtotal (excl. shipping)", parts_subtotal_no_shipping))
        summary_lines.append("")
        summary_lines.append(f"Total Order Weight: {total_weight_lb:.2f} lb")
        summary_lines.append(self._format_cost_summary_line("Order Shipping Cost", order_shipping))
        summary_lines.append("")
        summary_lines.append("=" * 74)
        summary_lines.append(self._format_cost_summary_line("ORDER TOTAL", order_total))
        summary_lines.append("=" * 74)
        summary_lines.append("")

        self.output_text.insert(tk.END, "\n".join(summary_lines))

    def generate_quote(self) -> None:
        """Generate the quote."""
        # Check if we have a multi-part order
        if len(self.current_order.parts) > 1:
            self._generate_multi_part_order_report()
            return
        elif len(self.current_order.parts) == 0:
            messagebox.showinfo(
                "No Parts",
                "Please add at least one part using 'Add Part(s)...' button before generating a quote."
            )
            return

        # Single-part flow (backward compatible with original behavior)
        # Smart cache invalidation - only clear if inputs changed
        # This saves 40+ seconds by avoiding redundant ODA/OCR extraction
        if self._quote_inputs_changed():
            self._clear_cad_cache()
            print("[AppV7] Cache cleared - quote inputs changed")
        else:
            print("[AppV7] Reusing cached quote data - inputs unchanged (saves ~40+ seconds)")

        # Collect values from quote editor
        quote_data = {}
        for label, field in self.quote_fields.items():
            quote_data[label] = field.get()
        self.quote_vars = quote_data

        # CRITICAL: Read all widget values in main thread BEFORE parallel execution
        # Tkinter widgets are NOT thread-safe - must only be accessed from main thread
        # Store values as instance variables that worker threads can safely read
        self._temp_machine_rate = self._get_field_float("Machine Rate ($/hr)", self.MACHINE_RATE)
        self._temp_labor_rate = self._get_field_float("Labor Rate ($/hr)", self.LABOR_RATE)
        self._temp_material_override = self._get_field_string("Material")
        self._temp_mcmaster_override = self._get_field_float("McMaster Price Override ($)")
        self._temp_scrap_override = self._get_field_float("Scrap Value Override ($)")
        self._temp_quantity = self._get_quantity()

        # Display in output tab
        self.output_text.delete(1.0, tk.END)

        # Add quote header with file name and part family
        from pathlib import Path
        cad_filename = Path(self.cad_file_path).name if self.cad_file_path else "Unknown File"
        part_family = quote_data.get("Part Family", "Unknown")
        quantity = quote_data.get("Quantity", "1")
        material = quote_data.get("Material", "")

        header_lines = [
            "=" * 74,
            "QUOTE SUMMARY",
            "=" * 74,
            f"CAD File: {cad_filename}",
            f"Part Type: {part_family}",
        ]

        # Add quantity if > 1
        if quantity and str(quantity) != "1":
            header_lines.append(f"Quantity: {quantity} part(s)")

        # Add material if overridden
        if material:
            header_lines.append(f"Material: {material}")

        header_lines.extend([
            "=" * 74,
            "",
            ""
        ])

        self.output_text.insert(tk.END, "\n".join(header_lines))

        # CRITICAL: Extract quote data ONCE before parallel report generation
        # This prevents race condition where all 3 threads try to extract simultaneously
        try:
            # This ensures _cached_quote_data is populated before threading
            _ = self._get_or_create_quote_data()
        except Exception as e:
            # If extraction fails, show error and abort
            error_msg = f"Error extracting quote data:\n{str(e)}"
            self.output_text.insert(tk.END, error_msg)
            self.status_bar.config(text="Quote generation failed - see Output tab")
            return

        # Generate all three reports in parallel for 10-20 second speedup
        # The reports are independent and can run concurrently
        # NOTE: Quote data is already cached, so no race conditions
        # NOTE: All widget values read above, so threads don't access Tkinter widgets
        print("[AppV7] Generating reports in parallel...")

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ReportGen") as executor:
            # Submit all three report generation tasks concurrently
            future_labor = executor.submit(self._generate_labor_hours_report)
            future_machine = executor.submit(self._generate_machine_hours_report)
            future_direct = executor.submit(self._generate_direct_costs_report)

            # Wait for all reports to complete and collect results
            labor_hours_report = future_labor.result()
            machine_hours_report = future_machine.result()
            direct_costs_report = future_direct.result()

        print("[AppV7] All reports generated (parallel execution complete)")

        # Clean up temporary variables
        del self._temp_machine_rate
        del self._temp_labor_rate
        del self._temp_material_override
        del self._temp_mcmaster_override
        del self._temp_scrap_override
        del self._temp_quantity

        # Insert reports in the correct order
        self.output_text.insert(tk.END, labor_hours_report)
        self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")
        self.output_text.insert(tk.END, machine_hours_report)
        self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")
        self.output_text.insert(tk.END, direct_costs_report)

        # Add summary
        self.output_text.insert(tk.END, "\n\n" + "=" * 74 + "\n\n")

        # Get margin rate with override check
        margin_percent = self._get_field_float("Margin (%)", 15.0)
        margin_rate = (margin_percent / 100.0) if margin_percent is not None else 0.15
        margin_overridden = (margin_percent is not None and margin_percent != 15.0)

        # Get quantity for display
        quantity = self._get_quantity()
        quantity_overridden = quantity > 1

        quick_margin_lines: list[str] = []
        summary_lines = [
            "COST SUMMARY",
            "=" * 74,
        ]

        # Show quantity if > 1
        if quantity_overridden:
            summary_lines.append(f"Quantity: {quantity} parts")
            summary_lines.append("-" * 74)
            summary_lines.append("PER-UNIT COSTS:")

        summary_lines.extend([
            self._format_cost_summary_line("Direct Cost", self.direct_cost_total),
            self._format_cost_summary_line("Machine Cost", self.machine_cost_total),
            self._format_cost_summary_line("Labor Cost", self.labor_cost_total),
        ])

        # Check for missing costs and warn user
        missing_costs = []
        if self.direct_cost_total is None:
            missing_costs.append("Direct Cost")
        if self.machine_cost_total is None:
            missing_costs.append("Machine Cost")
        if self.labor_cost_total is None:
            missing_costs.append("Labor Cost")

        if missing_costs:
            summary_lines.append("-" * 74)
            summary_lines.append("WARNING: Some costs could not be calculated:")
            for cost_name in missing_costs:
                summary_lines.append(f"  - {cost_name} is N/A")
            summary_lines.append("Quote may be incomplete - manual review required.")
            summary_lines.append("-" * 74)

        # Calculate total even with missing costs (treat N/A as $0.00)
        total_cost = (
            (self.direct_cost_total or 0.0)
            + (self.machine_cost_total or 0.0)
            + (self.labor_cost_total or 0.0)
        )

        if total_cost > 0 or not missing_costs:

            # Use cost summary from quote_data for accurate per-unit and total costs
            margin_amount = total_cost * margin_rate
            final_cost = total_cost + margin_amount

            quote_data = self._cached_quote_data
            if quote_data and quote_data.cost_summary:
                total_cost = quote_data.cost_summary.total_cost
                margin_amount = quote_data.cost_summary.margin_amount
                final_cost = quote_data.cost_summary.final_price

            quick_margin_lines = self._build_quick_margin_section(total_cost, margin_rate)
            summary_lines.append("-" * 74)
            summary_lines.append(self._format_cost_summary_line("Total Estimated Cost", total_cost))

            # Add margin line with override indicator
            margin_label = f"Margin ({margin_rate:.0%})"
            if margin_overridden:
                margin_label += " (OVERRIDDEN)"
            summary_lines.append(
                self._format_cost_summary_line(margin_label, margin_amount)
            )
            summary_lines.append(self._format_cost_summary_line("Final Price (per unit)", final_cost))

            # Show total costs if quantity > 1
            if quantity_overridden and quote_data and quote_data.cost_summary:
                summary_lines.append("")
                summary_lines.append("=" * 74)
                summary_lines.append("TOTAL ORDER COSTS:")
                summary_lines.append(self._format_cost_summary_line("Total Direct Cost", quote_data.cost_summary.total_direct_cost))
                summary_lines.append(self._format_cost_summary_line("Total Machine Cost", quote_data.cost_summary.total_machine_cost))
                summary_lines.append(self._format_cost_summary_line("Total Labor Cost", quote_data.cost_summary.total_labor_cost))
                summary_lines.append("-" * 74)
                summary_lines.append(self._format_cost_summary_line("Total Order Cost", quote_data.cost_summary.total_total_cost))
                summary_lines.append(self._format_cost_summary_line(f"Total Margin ({margin_rate:.0%})", quote_data.cost_summary.total_total_cost * margin_rate))
                summary_lines.append(self._format_cost_summary_line("Total Order Price", quote_data.cost_summary.total_final_price))

        if quick_margin_lines:
            self.output_text.insert(tk.END, "\n".join(quick_margin_lines))

        summary_lines.append("")
        self.output_text.insert(tk.END, "\n".join(summary_lines))

        self.notebook.select(self.output_tab)
        self.status_bar.config(text="Quote generated successfully!")

    def save_quote(self) -> None:
        """Save the quote or order to JSON file."""
        # Check if we have a multi-part order
        if len(self.current_order.parts) > 1:
            # Save as multi-part order
            self._save_order()
            return
        elif len(self.current_order.parts) == 0:
            # Fallback to old behavior for backward compatibility
            if not self.cad_file_path:
                messagebox.showwarning("No Quote", "Please load a CAD file and generate a quote first.")
                return

            if self._cached_quote_data is None:
                messagebox.showwarning("No Quote", "Please generate a quote first before saving.")
                return

        # Single-part save (backward compatible)
        from pathlib import Path
        suggested_name = Path(self.cad_file_path).stem + "_quote.json" if self.cad_file_path else "quote.json"

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

    def _save_order(self) -> None:
        """Save a multi-part order to JSON file."""
        suggested_name = f"order_{self.current_order.order_id}.json"

        filename = filedialog.asksaveasfilename(
            title="Save Order",
            defaultextension=".json",
            initialfile=suggested_name,
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if filename:
            try:
                # Save using OrderData's to_json method
                self.current_order.to_json(filepath=filename)
                self.status_bar.config(text=f"Order saved to: {filename}")
                messagebox.showinfo("Success", f"Order with {len(self.current_order.parts)} parts saved successfully to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save order:\n{str(e)}")
                self.status_bar.config(text="Error saving order")

    def load_order(self) -> None:
        """Load a multi-part order from JSON file."""
        filename = filedialog.askopenfilename(
            title="Load Order",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if not filename:
            return

        try:
            from cad_quoter.pricing.QuoteDataHelper import OrderData

            # Load the order
            self.current_order = OrderData.from_json(filename)

            # Refresh the parts list UI
            self._refresh_parts_list()

            # Auto-select the first part if available
            if self.current_order.parts:
                self.parts_listbox.selection_set(0)
                self.active_part_index = 0
                self._load_active_part_to_ui()

            self.status_bar.config(text=f"Order loaded: {len(self.current_order.parts)} parts")
            messagebox.showinfo(
                "Success",
                f"Order loaded successfully!\n\n"
                f"Order ID: {self.current_order.order_id}\n"
                f"Parts: {len(self.current_order.parts)}"
            )

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load order:\n{str(e)}")
            self.status_bar.config(text="Error loading order")
            import traceback
            traceback.print_exc()

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
