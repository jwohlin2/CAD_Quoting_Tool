import pytest

from appkit.ui import planner_render


def test_planner_render_canonicalize_filters_meta_and_misc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEBUG_MISC", raising=False)

    costs = {
        "Milling": 60.0,
        "Planner Machine": 125.0,
        "Planner Labor": 80.0,
        "Planner Total": 205.0,
        "Misc": 10.0,
        "Saw/Waterjet": 12.5,
    }

    canon = planner_render.canonicalize_costs(costs)

    assert set(canon) == {"milling", "saw_waterjet"}
    assert canon["milling"] == pytest.approx(60.0)
    assert canon["saw_waterjet"] == pytest.approx(12.5)
