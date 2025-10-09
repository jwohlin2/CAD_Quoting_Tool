import pytest

from cad_quoter.pricing.process_cost_renderer import (
    ORDER,
    canonicalize_costs,
    render_process_costs,
)


class TableCollector:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add_row(self, **row: object) -> None:
        self.rows.append(row)


def test_canonicalize_costs_groups_aliases_and_skips_planner_total() -> None:
    costs = {
        "Milling": 100.0,
        "machining": 20.0,
        "Deburr": 10.0,
        "Finishing": 5.0,
        "Planner Total": 900.0,
        "Planner Machine": 300.0,
        "Planner Labor": 200.0,
        "Wire EDM": 7.0,
    }

    canon = canonicalize_costs(costs)

    assert canon["milling"] == pytest.approx(120.0)
    assert canon["finishing_deburr"] == pytest.approx(15.0)
    assert canon["misc"] == pytest.approx(507.0)
    assert "planner_total" not in canon


def test_render_process_costs_orders_rows_and_rates() -> None:
    table = TableCollector()
    process_costs = {
        "milling": 100.0,
        "Deburr": 30.0,
        "Countersink": 5.0,
        "Assembly": 10.0,
    }
    minutes_detail = {
        "milling": 120.0,
        "deburr": 90.0,
        "Countersink": 15.0,
        "Assembly": 30.0,
    }
    rates = {
        "MillingRate": 60.0,
        "FinishingRate": 45.0,
        "DrillingRate": 55.0,
    }

    total = render_process_costs(table, process_costs, rates, minutes_detail)

    labels = [row["label"] for row in table.rows]
    assert labels == [
        "Milling",
        "Drilling",
        "Finishing/Deburr",
        "Misc",
    ]
    assert total == pytest.approx(145.0)

    expected_hours = {
        "Milling": 2.0,
        "Drilling": 0.25,
        "Finishing/Deburr": 1.5,
        "Misc": 0.5,
    }
    expected_rates = {
        "Milling": 60.0,
        "Drilling": 55.0,
        "Finishing/Deburr": 45.0,
        "Misc": 20.0,
    }
    expected_costs = {
        "Milling": 100.0,
        "Drilling": 5.0,
        "Finishing/Deburr": 30.0,
        "Misc": 10.0,
    }

    for row in table.rows:
        label = row["label"]
        assert row["hours"] == pytest.approx(expected_hours[label])
        assert row["rate"] == pytest.approx(expected_rates[label])
        assert row["cost"] == pytest.approx(expected_costs[label])

    assert all(label in ORDER for label in canonicalize_costs(process_costs))
