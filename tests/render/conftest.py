from __future__ import annotations

import importlib
from types import ModuleType

from cad_quoter import render as render_package
from cad_quoter.render.config import apply_render_overrides, ensure_mutable_breakdown
from cad_quoter.render.guards import render_drilling_guard
from cad_quoter.render.writer import QuoteWriter

import pytest

from tests.conftest import _install_runtime_dep_stubs


@pytest.fixture(scope="module")
def appv5_module() -> ModuleType:
    """Load ``appV5`` with runtime dependency stubs for headless execution."""

    _install_runtime_dep_stubs()
    import appV5

    module = importlib.reload(appV5)

    # ``appV5`` expects these helpers in its global namespace; provide them here so
    # the legacy entry points behave the same way in a headless test harness.
    module.apply_render_overrides = apply_render_overrides
    module.ensure_mutable_breakdown = ensure_mutable_breakdown
    module.QuoteWriter = QuoteWriter

    for name in getattr(render_package, "__all__", []):
        if not hasattr(module, name):
            setattr(module, name, getattr(render_package, name))

    if not hasattr(module, "render_state_has_planner_drilling"):
        module.render_state_has_planner_drilling = render_package.has_planner_drilling
    if not hasattr(module, "render_drilling_guard"):
        module.render_drilling_guard = render_drilling_guard

    return module
