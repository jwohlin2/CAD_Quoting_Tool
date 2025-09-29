"""Domain-level wiring for the CAD Quoter Tkinter application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from .config import AppEnvironment
from .geometry import GeometryService
from .llm import LLMService
from .pricing import PricingBreakdown, PricingService


@dataclass
class ApplicationServices:
    """Collection of services used by :class:`QuoteApplication`."""

    geometry: GeometryService
    pricing: PricingService
    llm: LLMService


def build_application_services(env: AppEnvironment) -> ApplicationServices:
    """Instantiate default application services."""

    geometry = GeometryService()
    pricing = PricingService()
    llm = LLMService(env)
    return ApplicationServices(geometry=geometry, pricing=pricing, llm=llm)


class QuoteApplication(tk.Tk):
    """Tkinter front-end that orchestrates the quoting helpers."""

    def __init__(self, env: AppEnvironment, services: ApplicationServices) -> None:
        super().__init__()
        self.env = env
        self.services = services
        self.title("CAD Quoter")
        self.geometry("820x520")
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Geometry tab -------------------------------------------------
        geometry_frame = ttk.Frame(notebook)
        notebook.add(geometry_frame, text="Geometry")

        ttk.Label(geometry_frame, text="CAD file path:").grid(row=0, column=0, sticky="w")
        self.geometry_path_var = tk.StringVar()
        path_entry = ttk.Entry(geometry_frame, textvariable=self.geometry_path_var, width=60)
        path_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        load_button = ttk.Button(geometry_frame, text="Load", command=self._on_load_geometry)
        load_button.grid(row=1, column=1, sticky="ew")

        geometry_frame.columnconfigure(0, weight=1)

        self.geometry_output = tk.Text(geometry_frame, height=12, wrap="word", state="disabled")
        self.geometry_output.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        geometry_frame.rowconfigure(2, weight=1)

        # Pricing tab --------------------------------------------------
        pricing_frame = ttk.Frame(notebook)
        notebook.add(pricing_frame, text="Pricing")

        ttk.Label(pricing_frame, text="Material weight (lbs)").grid(row=0, column=0, sticky="w")
        ttk.Label(pricing_frame, text="Machining hours").grid(row=0, column=1, sticky="w")

        self.material_var = tk.DoubleVar(value=2.5)
        self.machining_var = tk.DoubleVar(value=1.0)
        ttk.Entry(pricing_frame, textvariable=self.material_var).grid(row=1, column=0, sticky="ew", padx=(0, 12))
        ttk.Entry(pricing_frame, textvariable=self.machining_var).grid(row=1, column=1, sticky="ew")

        quote_btn = ttk.Button(pricing_frame, text="Compute quote", command=self._on_compute_quote)
        quote_btn.grid(row=2, column=0, columnspan=2, pady=(12, 0))

        self.pricing_summary = ttk.Label(pricing_frame, text="Enter values and compute a quote")
        self.pricing_summary.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        pricing_frame.columnconfigure(0, weight=1)
        pricing_frame.columnconfigure(1, weight=1)

        # LLM tab ------------------------------------------------------
        llm_frame = ttk.Frame(notebook)
        notebook.add(llm_frame, text="Insights")

        ttk.Label(llm_frame, text="Prompt").grid(row=0, column=0, sticky="w")
        self.prompt_text = tk.Text(llm_frame, height=6, wrap="word")
        self.prompt_text.grid(row=1, column=0, sticky="nsew")

        llm_button = ttk.Button(llm_frame, text="Generate", command=self._on_generate_insights)
        llm_button.grid(row=2, column=0, pady=8, sticky="e")

        self.llm_output = tk.Text(llm_frame, height=10, wrap="word", state="disabled")
        self.llm_output.grid(row=3, column=0, sticky="nsew")

        llm_frame.rowconfigure(1, weight=1)
        llm_frame.rowconfigure(3, weight=1)
        llm_frame.columnconfigure(0, weight=1)

    # ---------------------------------------------------------- Callbacks
    def _on_load_geometry(self) -> None:
        path = Path(self.geometry_path_var.get())
        summary = self.services.geometry.try_load(path)
        if summary is None:
            messagebox.showerror("Geometry", f"Could not locate '{path}'.")
            return
        self._write_text_widget(self.geometry_output, summary.as_display_text())

    def _on_compute_quote(self) -> None:
        try:
            material = float(self.material_var.get())
            hours = float(self.machining_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Pricing", "Please enter valid numeric values.")
            return
        breakdown = self.services.pricing.quote_basic_part(material, hours)
        text = self._format_pricing(breakdown)
        self.pricing_summary.config(text=text)

    def _on_generate_insights(self) -> None:
        prompt = self.prompt_text.get("1.0", tk.END)
        insights = self.services.llm.generate_insights(prompt)
        text = "\n\n".join(insight.message for insight in insights)
        self._write_text_widget(self.llm_output, text)

    # ------------------------------------------------------- Util helpers
    @staticmethod
    def _format_pricing(breakdown: PricingBreakdown) -> str:
        return (
            f"Material: ${breakdown.material_cost:,.2f} — "
            f"Machining: ${breakdown.machining_cost:,.2f} — "
            f"Overhead: ${breakdown.overhead_cost:,.2f} — "
            f"Total: ${breakdown.total:,.2f}"
        )

    @staticmethod
    def _write_text_widget(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")


__all__ = [
    "ApplicationServices",
    "build_application_services",
    "QuoteApplication",
]
