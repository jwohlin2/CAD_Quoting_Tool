"""Costing helpers that bridge planner output to shop rate data."""

from __future__ import annotations

from typing import Any, Dict

from cad_quoter.rates import OP_TO_LABOR, OP_TO_MACHINE, rate_for_machine, rate_for_role


def op_cost(op: Dict[str, Any], rates: Dict[str, Dict[str, float]], minutes: float) -> float:
    """Compute the direct cost for a single operation."""

    name = op["op"]
    cost = 0.0

    if name in OP_TO_MACHINE:
        machine = OP_TO_MACHINE[name]
        cost += rate_for_machine(rates, machine) * (minutes / 60.0)

    if name in OP_TO_LABOR:
        role = OP_TO_LABOR[name]
        cost += rate_for_role(rates, role) * (minutes / 60.0)

    return cost


__all__ = ["op_cost"]

