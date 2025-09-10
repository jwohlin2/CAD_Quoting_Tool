# app.py

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os

# Import the "brain" from our other files
from quoting_logic import generate_quote, MATERIAL_DATA
from feature_extractor import analyze_step_file, analyze_dxf_file, analyze_stl_file

class QuotingApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Quoting Tool")
        self.geometry("500x650")

        # --- Instance Variables ---
        self.part_params = {}

        # --- UI Variables ---
        self.unit_var = tk.StringVar(value='mm')
        self.material_var = tk.StringVar(value='aluminum_6061')
        self.quantity_var = tk.StringVar(value="1")
        self.tight_tol_var = tk.BooleanVar()
        self.edm_var = tk.BooleanVar()
        self.grinding_var = tk.BooleanVar()
        self.honing_var = tk.BooleanVar()

        self.result_price_var = tk.StringVar(value="$0.00 Total")
        self.result_per_part_var = tk.StringVar(value="($0.00 each)")
        self.result_dims_var = tk.StringVar()
        self.result_details_var = tk.StringVar(value="Details will appear here...")

        # --- Create UI ---
        self._create_widgets()

    def _create_widgets(self):
        # --- Main Frame ---
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Controls ---
        controls_frame = ttk.LabelFrame(main_frame, text="Quote Parameters", padding="15")
        controls_frame.pack(fill=tk.X)

        # --- Basic Parameters ---
        basic_params_frame = ttk.Frame(controls_frame)
        basic_params_frame.pack(fill=tk.X)

        ttk.Label(basic_params_frame, text="CAD Units:").grid(row=0, column=0, sticky=tk.W, pady=5)
        unit_menu = ttk.OptionMenu(basic_params_frame, self.unit_var, 'mm', 'mm', 'cm', 'in', 'm', command=self.update_quote)
        unit_menu.grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(basic_params_frame, text="Material:").grid(row=1, column=0, sticky=tk.W, pady=5)
        combo_material = ttk.Combobox(basic_params_frame, textvariable=self.material_var, values=list(MATERIAL_DATA.keys()), state="readonly")
        combo_material.grid(row=1, column=1, sticky=tk.EW, pady=5)
        combo_material.bind("<<ComboboxSelected>>", self.update_quote)

        ttk.Label(basic_params_frame, text="Quantity:").grid(row=2, column=0, sticky=tk.W, pady=5)
        entry_quantity = ttk.Entry(basic_params_frame, textvariable=self.quantity_var)
        entry_quantity.grid(row=2, column=1, sticky=tk.EW, pady=5)
        entry_quantity.bind("<KeyRelease>", self.update_quote)

        basic_params_frame.columnconfigure(1, weight=1)

        # --- Advanced Processes ---
        adv_process_frame = ttk.LabelFrame(controls_frame, text="Additional Processes", padding="15")
        adv_process_frame.pack(fill=tk.X, pady=10)

        check_tight_tol = ttk.Checkbutton(adv_process_frame, text="Tight Tolerances (±0.002mm)", variable=self.tight_tol_var, command=self.update_quote)
        check_tight_tol.pack(anchor=tk.W)

        check_edm = ttk.Checkbutton(adv_process_frame, text="Internal Sharp Corners (EDM)", variable=self.edm_var, command=self.update_quote)
        check_edm.pack(anchor=tk.W)

        check_grinding = ttk.Checkbutton(adv_process_frame, text="Surface Grinding (Ra ≤ 0.4 µm)", variable=self.grinding_var, command=self.update_quote)
        check_grinding.pack(anchor=tk.W)

        check_honing = ttk.Checkbutton(adv_process_frame, text="Edge Honing (0.03–0.05 mm)", variable=self.honing_var, command=self.update_quote)
        check_honing.pack(anchor=tk.W)

        # --- Action Buttons ---
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=20)

        load_button = ttk.Button(buttons_frame, text="Load CAD & Get Quote", command=self.load_file_and_generate_quote)
        load_button.pack(side=tk.LEFT, padx=10)

        # --- Results Display ---
        result_frame = ttk.LabelFrame(main_frame, text="Quote Result", padding="20")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(result_frame, text="Quote Price:", font=("Helvetica", 12)).pack()
        ttk.Label(result_frame, textvariable=self.result_price_var, font=("Helvetica", 20, "bold")).pack()
        ttk.Label(result_frame, textvariable=self.result_per_part_var, font=("Helvetica", 12)).pack(pady=(0, 10))

        ttk.Label(result_frame, textvariable=self.result_dims_var, font=("Helvetica", 10, "italic")).pack(pady=5)

        ttk.Label(result_frame, textvariable=self.result_details_var, justify=tk.LEFT).pack(fill=tk.X, pady=5)

    def update_quote(self, event=None):
        if not self.part_params:
            return

        try:
            self.part_params['material'] = self.material_var.get()
            self.part_params['quantity'] = int(self.quantity_var.get())
            self.part_params['tight_tolerances'] = self.tight_tol_var.get()
            self.part_params['internal_sharp_corners'] = self.edm_var.get()
            self.part_params['surface_grinding'] = self.grinding_var.get()
            self.part_params['edge_honing'] = self.honing_var.get()

            self.handle_quote_generation(self.part_params)
        except (ValueError, tk.TclError):
            # Ignore errors during typing
            pass

    def handle_quote_generation(self, part_params):
        try:
            quote = generate_quote(part_params)

            self.result_price_var.set(f"${quote['final_quote_price']:,.2f} Total")
            self.result_per_part_var.set(f"(${quote['price_per_part']:,.2f} each)")
            self.result_dims_var.set(
                f"L: {part_params['length_mm']:.2f}mm x "
                f"W: {part_params['width_mm']:.2f}mm x "
                f"H: {part_params['height_mm']:.2f}mm"
            )
            self.result_details_var.set(
                f"Material Cost: ${quote['details']['total_material_cost']:,.2f}\n"
                f"Labor Cost: ${quote['details']['labor_cost']:,.2f} ({quote['details']['total_labor_hours']:.1f} hrs)\n"
                f"Consumables: ${quote['details']['consumables_cost']:,.2f}\n"
                f"Margin: ${quote['details']['margin_amount']:,.2f}\n"
                f"Discount: -${quote['details']['discount_amount']:,.2f}\n"
                f"--- \n"
                f"{quote['notes']}"
            )

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your inputs.\n\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def load_file_and_generate_quote(self):
        file_path = filedialog.askopenfilename(
            filetypes=([
                ("CAD Files", ".step .stp .dxf .stl"),
                ("All files", "*.*")
            ])
        )

        if not file_path:
            return

        file_ext = os.path.splitext(file_path)[1].lower()
        extracted_data, message = None, None

        if file_ext in (".step", ".stp"):
            extracted_data, message = analyze_step_file(file_path)
        elif file_ext == ".dxf":
            extracted_data, message = analyze_dxf_file(file_path)
        elif file_ext == ".stl":
            extracted_data, message = analyze_stl_file(file_path)
        else:
            messagebox.showerror("Unsupported File", f"Unsupported file type: {file_ext}")
            return

        if not extracted_data:
            messagebox.showerror("Analysis Error", message)
            return

        # --- Unit Conversion ---
        selected_unit = self.unit_var.get()
        conversion_factor = 1.0
        if selected_unit == 'in':
            conversion_factor = 25.4
        elif selected_unit == 'cm':
            conversion_factor = 10.0
        elif selected_unit == 'm':
            conversion_factor = 1000.0

        self.part_params = {
            'length_mm': extracted_data.get('length_mm', 0.0) * conversion_factor,
            'width_mm': extracted_data.get('width_mm', 0.0) * conversion_factor,
            'height_mm': extracted_data.get('height_mm', 0.0) * conversion_factor,
            'surface_area_mm2': extracted_data.get('surface_area_mm2', 0.0) * (conversion_factor**2),
        }

        self.update_quote()

if __name__ == "__main__":
    app = QuotingApp()
    app.mainloop()
