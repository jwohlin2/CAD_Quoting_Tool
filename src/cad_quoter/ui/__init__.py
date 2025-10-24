"""UI helpers and Tk widgets for the CAD Quoting tool."""

from importlib import import_module
from types import ModuleType

__all__ = [
    "editor_controls",
    "layout",
    "llm_controls",
    "llm_panel",
    "menus",
    "output_pane",
    "planner_render",
    "quote_editor",
    "services",
    "session_io",
    "status",
    "suggestions",
    "tk_compat",
    "widgets",
]

_LAZY_SUBMODULES = {name: f"{__name__}.{name}" for name in __all__}


def __getattr__(name: str) -> ModuleType:
    """Lazily import UI submodules on first access."""

    if name in _LAZY_SUBMODULES:
        module = import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
