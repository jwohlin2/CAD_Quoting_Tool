import pytest

from cad_quoter.pricing import ORDER, canonicalize_costs, render_process_costs


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
        "Wire EDM Windows": 11.0,
        "Wire-EDM": 3.0,
        "WireEDM": 2.0,
        "Sinker EDM Finish Burn": 13.0,
        "Ram EDM": 5.0,
        "RamEDM": 4.0,
    }

    canon = canonicalize_costs(costs)

    assert canon["milling"] == pytest.approx(120.0)
    assert canon["finishing_deburr"] == pytest.approx(15.0)
    assert canon["misc"] == pytest.approx(500.0)
    assert canon["wire_edm"] == pytest.approx(23.0)
    assert canon["sinker_edm"] == pytest.approx(22.0)
    assert "planner_total" not in canon


def test_render_process_costs_orders_rows_and_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG_MISC", raising=False)

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
    ]
    assert total == pytest.approx(201.25)

    expected_hours = {
        "Milling": 2.0,
        "Drilling": 0.25,
        "Finishing/Deburr": 1.5,
    }
    expected_rates = {
        "Milling": 60.0,
        "Drilling": 55.0,
        "Finishing/Deburr": 45.0,
    }
    expected_costs = {
        "Milling": 120.0,
        "Drilling": 13.75,
        "Finishing/Deburr": 67.5,
    }

    for row in table.rows:
        label = row["label"]
        assert row["hours"] == pytest.approx(expected_hours[label])
        assert row["rate"] == pytest.approx(expected_rates[label])
        assert row["cost"] == pytest.approx(expected_costs[label])

    assert all(label in ORDER for label in canonicalize_costs(process_costs))


def test_render_process_costs_backfills_rate_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG_MISC", raising=False)

    table = TableCollector()
    process_costs = {"Wire EDM": 120.0}
    minutes_detail = {"Wire EDM": 240.0}

    total = render_process_costs(table, process_costs, rates={}, minutes_detail=minutes_detail)

    assert total == pytest.approx(120.0)
    assert len(table.rows) == 1
    row = table.rows[0]
    assert row["label"] == "Wire EDM"
    assert row["hours"] == pytest.approx(4.0)
    assert row["rate"] == pytest.approx(30.0)
    assert row["cost"] == pytest.approx(120.0)


def test_render_process_costs_hides_misc_even_when_significant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEBUG_MISC", raising=False)

    table = TableCollector()
    process_costs = {"Misc": 75.0}

    total = render_process_costs(table, process_costs, rates={}, minutes_detail={})

    assert table.rows == []
    assert total == pytest.approx(0.0)


def test_render_process_costs_hides_misc_even_when_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBUG_MISC", "1")

    table = TableCollector()
    process_costs = {"Misc": 10.0}

    total = render_process_costs(table, process_costs, rates={}, minutes_detail={})

    assert table.rows == []
    assert total == pytest.approx(0.0)

    monkeypatch.delenv("DEBUG_MISC", raising=False)
