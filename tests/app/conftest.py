from __future__ import annotations

import importlib
from collections.abc import Generator

import pytest


@pytest.fixture
def reload_appv5(monkeypatch: pytest.MonkeyPatch) -> Generator[object, None, None]:
    geometry = importlib.import_module("cad_quoter.geometry")
    if not hasattr(geometry, "contours_from_polylines"):
        monkeypatch.setattr(
            geometry,
            "contours_from_polylines",
            lambda *args, **kwargs: [],
            raising=False,
        )

    module = importlib.import_module("appV5")
    planner_adapter = importlib.import_module("cad_quoter.app.planner_adapter")
    monkeypatch.setattr(module, "FORCE_PLANNER", False)
    monkeypatch.setattr(planner_adapter, "FORCE_PLANNER", False)
    monkeypatch.setattr(module, "FORCE_ESTIMATOR", False, raising=False)
    monkeypatch.setattr(planner_adapter, "FORCE_ESTIMATOR", False, raising=False)

    try:
        yield module
    finally:
        importlib.reload(module)


@pytest.fixture
def disable_speeds_feeds_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    import appV5

    monkeypatch.setattr(appV5, "_load_speeds_feeds_table_from_path", lambda _path: (None, False))
    monkeypatch.setattr(appV5, "FORCE_ESTIMATOR", False, raising=False)
