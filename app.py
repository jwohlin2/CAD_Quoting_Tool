# app.py

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os
import pandas as pd

# Import the new quoting logic and feature extractor
from quoting_logic import compute_full_quote, DEFAULT_RATES, DEFAULT_PARAMS
from feature_extractor import analyze_cad_file
from llm_integration import create_llm_prompt, call_llm_api

# Create a dummy excel file for the compute_full_quote function
if not os.path.exists("dummy_quote_sheet.xlsx"):
    pd.DataFrame({
        'Data Type / Input Method': ['number', 'number'],
        'Item': ['Example Item', 'Example Item 2'],
        'Example Values / Options': [1, 2]
    }).to_excel("dummy_quote_sheet.xlsx", index=False)

class QuotingApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Full Coverage Quoting Tool")
        self.geometry("600x800")

        # --- Instance Variables ---
        self.geo_context = {}

        # --- UI Variables ---
        self.user_choices = {
            'dropdown': {},
            'checkbox': {}
        }

        # --- Create UI ---
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Dropdowns ---
        dropdown_frame = ttk.LabelFrame(main_frame, text="Dropdown Selections", padding="15")
        dropdown_frame.pack(fill=tk.X, pady=5)
        dropdown_frame.columnconfigure(1, weight=1)

        self.dropdown_vars = {}
        dropdown_options = {
            "Primary Milling Machine Rate": ["3-Axis Mill", "5-Axis Mill"],
            "Payment Terms": ["Net 30", "Net 60", "Net 90"],
            "Customer Relationship": ["New", "Existing", "Key Account"],
            "Manual Deburring / Edge Break Labor": ["None", "Standard Edge Break", "Full Cosmetic"],
            "Outsourced Plating / Coating Cost": ["None", "Anodize", "Black Oxide", "Nickel Plate"],
            "Outsourced Heat Treat Cost": ["None", "Harden & Temper", "Anneal", "Nitride"],
            "Packaging Method": ["Standard", "ESD Bagging", "Custom Crate"],
            "Shipment Speed": ["Standard", "Rush"],
            "Material Starting Condition": ["As-is", "Pre-Hardened"],
            "Solid Model Quality": ["Good", "Needs Repair"],
            "Fixture Design Hours": ["No", "Yes"],
            "RFQ Completeness": ["Complete", "Incomplete"],
            "Revision Control Complexity": ["Low", "High"],
            "Required Certifications (ITAR, AS9100)": ["None", "ITAR", "AS9100"]
        }

        for i, (label, options) in enumerate(dropdown_options.items()):
            var = tk.StringVar(value=options[0])
            self.dropdown_vars[label] = var
            ttk.Label(dropdown_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=3)
            combo = ttk.Combobox(dropdown_frame, textvariable=var, values=options, state="readonly")
            combo.grid(row=i, column=1, sticky=tk.EW, pady=3)

        # --- Checkboxes ---
        checkbox_frame = ttk.LabelFrame(main_frame, text="Checkbox Selections", padding="15")
        checkbox_frame.pack(fill=tk.X, pady=5)

        self.checkbox_vars = {}
        checkbox_options = [
            "Expedite Request",
            "Customer-Furnished Material (CFM)",
            "First Article Inspection Report (FAIR) Labor",
            "Source Inspection Requirement",
            "Gauge / Check Fixture NRE",
            "Precision Fitting Labor (Toolmaker)",
            "Complex Assembly Documentation",
            "Laser Marking / Engraving Time",
            "Passivation / Cleaning",
            "Tumbling / Vibratory Finishing Time",
            "Live Tooling / Mill-Turn Ops Time",
            "Sub-Spindle Utilization",
            "Form Tool Requirement"
        ]

        for i, label in enumerate(checkbox_options):
            var = tk.BooleanVar()
            self.checkbox_vars[label] = var
            cb = ttk.Checkbutton(checkbox_frame, text=label, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W)

        # --- Action Buttons ---
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=20)

        load_button = ttk.Button(buttons_frame, text="Load CAD", command=self.load_cad_file)
        load_button.pack(side=tk.LEFT, padx=10)

        llm_button = ttk.Button(buttons_frame, text="Analyze with LLM", command=self.analyze_with_llm)
        llm_button.pack(side=tk.LEFT, padx=10)

        quote_button = ttk.Button(buttons_frame, text="Generate Quote", command=self.generate_quote_from_gui)
        quote_button.pack(side=tk.LEFT, padx=10)

        # --- Results Display ---
        self.result_text = tk.Text(main_frame, height=15, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def generate_quote_from_gui(self):
        # --- Gather user choices ---
        for label, var in self.dropdown_vars.items():
            self.user_choices['dropdown'][label] = var.get()
        for label, var in self.checkbox_vars.items():
            self.user_choices['checkbox'][label] = var.get()

        # --- Generate Quote ---
        try:
            quote = compute_full_quote(
                xlsx_path="dummy_quote_sheet.xlsx",
                user_choices=self.user_choices,
                geo_context=self.geo_context
            )
            self.display_quote(quote)
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def load_cad_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=([
                ("CAD Files", ".step .stp .stl"),
                ("All files", "*.*")
            ])
        )

        if not file_path:
            return

        extracted_data, message = analyze_cad_file(file_path)

        if not extracted_data:
            messagebox.showerror("Analysis Error", message)
            return

        self.geo_context = extracted_data
        self.generate_quote_from_gui()

    def analyze_with_llm(self):
        file_path = filedialog.askopenfilename(
            filetypes=([
                ("CAD Files", ".step .stp .stl"),
                ("All files", "*.*")
            ])
        )

        if not file_path:
            return

        geometric_features = analyze_cad_file(file_path)

        if geometric_features:
            llm_prompt = create_llm_prompt(geometric_features)
            quote_variables = call_llm_api(llm_prompt)
            self.update_gui_from_llm(quote_variables)

    def update_gui_from_llm(self, quote_variables):
        # This is a simplified mapping. A real application would need a more robust way to map these.
        self.dropdown_vars["Primary Milling Machine Rate"].set(quote_variables.get("PM-01_Quote_Priority", "Standard"))
        self.checkbox_vars["Live Tooling / Mill-Turn Ops Time"].set(quote_variables.get("TRN-03_Live_Tooling_Required", False))
        self.checkbox_vars["Gauge / Check Fixture NRE"].set(quote_variables.get("ENG-05_Custom_Fixture_Required", False) == "Yes")
        self.dropdown_vars["Manual Deburring / Edge Break Labor"].set(quote_variables.get("FIN-01_Manual_Deburring_Level", "Standard Edge Break"))
        self.checkbox_vars["First Article Inspection Report (FAIR) Labor"].set(quote_variables.get("QC-05_FAIR_Report_Required", False))

    def display_quote(self, quote):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"**Final Quote: ${quote['price']:,.2f}**\n\n")
        self.result_text.insert(tk.END, "**Quote Breakdown:**\n")
        for key, value in quote['components'].items():
            if isinstance(value, float):
                self.result_text.insert(tk.END, f"{key}: ${value:,.2f}\n")
            else:
                self.result_text.insert(tk.END, f"{key}: {value}\n")
        
        self.result_text.insert(tk.END, "\n**Cost Buckets:**\n")
        for key, value in quote['cost_buckets'].items():
            self.result_text.insert(tk.END, f"{key}: ${value:,.2f}\n")

if __name__ == "__main__":
    app = QuotingApp()
    app.mainloop()