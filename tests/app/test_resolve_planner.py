import importlib

import pytest


@pytest.fixture(autouse=True)
def reload_appv5(monkeypatch):
    geometry = importlib.import_module("cad_quoter.geometry")
    if not hasattr(geometry, "contours_from_polylines"):
        monkeypatch.setattr(
            geometry,
            "contours_from_polylines",
            lambda *args, **kwargs: [],
            raising=False,
        )

    module = importlib.import_module("appV5")
    monkeypatch.setattr(module, "FORCE_PLANNER", False)
    yield module
    importlib.reload(module)


def test_resolve_planner_defaults_to_auto(reload_appv5):
    module = reload_appv5
    used, mode = module.resolve_planner(None, None)
    assert used is False
    assert mode == "auto"


def test_resolve_planner_handles_planner_mode_signal(reload_appv5, monkeypatch):
    module = reload_appv5
    monkeypatch.setattr(module, "FORCE_PLANNER", False)
    used, mode = module.resolve_planner(
        {"PlannerMode": "planner"},
        {"totals_present": True},
    )
    assert used is True
    assert mode == "planner"


def test_resolve_planner_legacy_mode_requires_signals(reload_appv5):
    module = reload_appv5
    used, mode = module.resolve_planner(
        {"PlannerMode": "legacy"},
        {"recognized_line_items": 0},
    )
    assert used is False
    assert mode == "legacy"


def test_resolve_planner_legacy_mode_recognized_items(reload_appv5):
    module = reload_appv5
    used, mode = module.resolve_planner(
        {"PlannerMode": "legacy"},
        {"recognized_line_items": "3"},
    )
    assert used is True
    assert mode == "legacy"


def test_resolve_planner_invalid_mode_falls_back(reload_appv5):
    module = reload_appv5
    used, mode = module.resolve_planner(
        {"PlannerMode": "   "},
        {"pricing_result": {"totals": {}}},
    )
    assert used is True
    assert mode == "auto"


def test_resolve_planner_force_flag_overrides_mode(monkeypatch, reload_appv5):
    module = reload_appv5
    monkeypatch.setattr(module, "FORCE_PLANNER", True)
    used, mode = module.resolve_planner({"PlannerMode": "legacy"}, None)
    assert used is False
    assert mode == "planner"


def test_no_internal_resolve_helpers_exposed(reload_appv5):
    module = reload_appv5
    assert not hasattr(module, "_resolve_planner_mode")
    assert not hasattr(module, "_resolve_planner_usage")


def test_resolve_planner_line_items_force_usage(reload_appv5):
    module = reload_appv5
    used, mode = module.resolve_planner(
        {"PlannerMode": "auto"},
        {"line_items": [{"op": "test"}]},
    )
    assert used is True
    assert mode == "auto"
