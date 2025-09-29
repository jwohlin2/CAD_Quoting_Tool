"""Pricing calculators for the CAD Quoter application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PricingBreakdown:
    """Container for cost components presented to the user."""

    material_cost: float
    machining_cost: float
    overhead_cost: float

    @property
    def total(self) -> float:
        return self.material_cost + self.machining_cost + self.overhead_cost

    def to_dict(self) -> Dict[str, float]:
        return {
            "material_cost": self.material_cost,
            "machining_cost": self.machining_cost,
            "overhead_cost": self.overhead_cost,
            "total": self.total,
        }


class PricingService:
    """Very small pricing helper used by the UI demo."""

    def quote_basic_part(self, material_weight_lbs: float, machining_hours: float) -> PricingBreakdown:
        """Return a deterministic but simple pricing structure."""

        material_cost = 7.5 * max(material_weight_lbs, 0.0)
        machining_cost = 95.0 * max(machining_hours, 0.0)
        overhead_cost = 0.15 * (material_cost + machining_cost)
        return PricingBreakdown(material_cost, machining_cost, overhead_cost)


__all__ = ["PricingService", "PricingBreakdown"]
