"""Costing helpers that bridge planner output to shop rate data."""

from __future__ import annotations

from typing import Any, Dict

from cad_quoter.rates import OP_TO_LABOR, OP_TO_MACHINE, default_process_rate


def op_cost(op: Dict[str, Any], rates: Dict[str, Dict[str, float]], minutes: float) -> float:
    """Compute the direct cost for a single operation."""

    name = op["op"]
    cost = 0.0

    if name in OP_TO_MACHINE:
        machine = OP_TO_MACHINE[name]
        machine_rate = rates.get("machine", {}).get(machine)
        if machine_rate is None:
            machine_rate = default_process_rate("machine", machine)
        else:
            machine_rate = float(machine_rate)
        cost += machine_rate * (minutes / 60.0)

    if name in OP_TO_LABOR:
        role = OP_TO_LABOR[name]
        labor_rate = rates.get("labor", {}).get(role)
        if labor_rate is None:
            labor_rate = default_process_rate("labor", role)
        else:
            labor_rate = float(labor_rate)
        cost += labor_rate * (minutes / 60.0)

    return cost


__all__ = ["op_cost"]

