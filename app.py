# app.py

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os

# Import the "brain" from our other files
from quoting_logic import generate_quote
from feature_extractor import analyze_step_file, analyze_dxf_file, analyze_stl_file

# --- UI Functions ---
def handle_quote_generation(part_params):
    """
    Calls the quoting logic and displays the result.
    """
    try:
        quote = generate_quote(part_params)

        result_price_var.set(f"${quote['final_quote_price']:.2f}")
        result_dims_var.set(
            f"L: {part_params['length_mm']:.2f}mm x "
            f"W: {part_params['width_mm']:.2f}mm x "
            f"H: {part_params['height_mm']:.2f}mm"
        )
        result_details_var.set(
            f"Material Cost: ${quote['details']['material_cost']:.2f}\n"
            f"Labor Cost: ${quote['details']['labor_cost']:.2f} ({quote['details']['total_labor_hours']:.1f} hrs)\n"
            f"Margin: ${quote['margin_amount']:.2f}\n"
            f"--- \n"
            f"Notes: {quote['notes']}"
        )

    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

def load_file_and_generate_quote():
    """
    Loads a CAD file, extracts dimensions, and then generates a quote.
    """
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
    selected_unit = unit_var.get()
    conversion_factor = 1.0
    if selected_unit == 'in':
        conversion_factor = 25.4
    elif selected_unit == 'cm':
        conversion_factor = 10.0
    elif selected_unit == 'm':
        conversion_factor = 1000.0

    converted_params = {
        'length_mm': extracted_data.get('length_mm', 0.0) * conversion_factor,
        'width_mm': extracted_data.get('width_mm', 0.0) * conversion_factor,
        'height_mm': extracted_data.get('height_mm', 0.0) * conversion_factor,
    }

    handle_quote_generation(converted_params)

def open_manual_input_window():
    """
    Opens a new window for manually entering part dimensions.
    """
    manual_window = tk.Toplevel(root)
    selected_unit = unit_var.get()
    manual_window.title(f"Manual Entry ({selected_unit})")
    manual_window.geometry("350x200")

    input_frame = ttk.Frame(manual_window, padding="20")
    input_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(input_frame, text=f"Length ({selected_unit}):").grid(row=0, column=0, sticky=tk.W, pady=5)
    entry_length = ttk.Entry(input_frame)
    entry_length.grid(row=0, column=1, sticky=tk.EW)

    ttk.Label(input_frame, text=f"Width ({selected_unit}):").grid(row=1, column=0, sticky=tk.W, pady=5)
    entry_width = ttk.Entry(input_frame)
    entry_width.grid(row=1, column=1, sticky=tk.EW)

    ttk.Label(input_frame, text=f"Height ({selected_unit}):").grid(row=2, column=0, sticky=tk.W, pady=5)
    entry_height = ttk.Entry(input_frame)
    entry_height.grid(row=2, column=1, sticky=tk.EW)

    input_frame.columnconfigure(1, weight=1)

    def manual_quote_callback():
        try:
            # --- Unit Conversion ---
            conversion_factor = 1.0
            if selected_unit == 'in':
                conversion_factor = 25.4
            elif selected_unit == 'cm':
                conversion_factor = 10.0
            elif selected_unit == 'm':
                conversion_factor = 1000.0

            part_params = {
                'length_mm': float(entry_length.get()) * conversion_factor,
                'width_mm': float(entry_width.get()) * conversion_factor,
                'height_mm': float(entry_height.get()) * conversion_factor
            }
            handle_quote_generation(part_params)
            manual_window.destroy()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for all dimensions.", parent=manual_window)

    quote_button = ttk.Button(input_frame, text="Generate Quote", command=manual_quote_callback)
    quote_button.grid(row=3, column=0, columnspan=2, pady=20)

# --- Main Application Setup ---
root = tk.Tk()
root.title("CAD Quoting Tool")
root.geometry("450x500")

# --- Main Frame ---
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Unit Selection ---
unit_frame = ttk.Frame(main_frame)
unit_frame.pack(pady=10)
ttk.Label(unit_frame, text="Select CAD File Units:").pack(side=tk.LEFT, padx=5)
unit_var = tk.StringVar(value='mm')
unit_menu = ttk.OptionMenu(unit_frame, unit_var, 'mm', 'mm', 'cm', 'in', 'm')
unit_menu.pack(side=tk.LEFT)

# --- Buttons ---
load_button = ttk.Button(main_frame, text="Load CAD File and Generate Quote", command=load_file_and_generate_quote)
load_button.pack(pady=10)

manual_button = ttk.Button(main_frame, text="Enter Dimensions Manually", command=open_manual_input_window)
manual_button.pack(pady=10)

# --- Results Display ---
result_frame = ttk.LabelFrame(main_frame, text="Quote Result", padding="20")
result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

result_price_var = tk.StringVar(value="$0.00")
ttk.Label(result_frame, text="Final Quote Price:", font=("Helvetica", 12)).pack()
ttk.Label(result_frame, textvariable=result_price_var, font=("Helvetica", 20, "bold")).pack(pady=5)

result_dims_var = tk.StringVar()
ttk.Label(result_frame, textvariable=result_dims_var, font=("Helvetica", 10, "italic")).pack(pady=5)

result_details_var = tk.StringVar(value="Details will appear here...")
ttk.Label(result_frame, textvariable=result_details_var, justify=tk.LEFT).pack(fill=tk.X, pady=5)

root.mainloop()