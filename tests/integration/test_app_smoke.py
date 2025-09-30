from __future__ import annotations

import pytest

import appV5


class _DummyVar:
    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummyWidget:
    def __init__(self, *args, **kwargs):
        self.children = []

    def pack(self, *args, **kwargs):
        return self

    def grid(self, *args, **kwargs):
        return self

    def place(self, *args, **kwargs):
        return self

    def bind(self, *args, **kwargs):
        return self

    def bind_all(self, *args, **kwargs):
        return self

    def unbind_all(self, *args, **kwargs):
        return self

    def config(self, *args, **kwargs):
        return self

    configure = config

    def create_window(self, *args, **kwargs):
        return 1

    def insert(self, *args, **kwargs):
        return self

    def delete(self, *args, **kwargs):
        return self

    def see(self, *args, **kwargs):
        return self

    def add(self, *args, **kwargs):
        self.children.append(args[0] if args else None)
        return self

    def add_command(self, *args, **kwargs):
        return self

    def add_cascade(self, *args, **kwargs):
        return self

    def add_separator(self, *args, **kwargs):
        return self

    def set(self, *args, **kwargs):
        return self

    def yview(self, *args, **kwargs):
        return self

    def yview_moveto(self, *args, **kwargs):
        return self

    def destroy(self):
        return None


class _DummyNotebook(_DummyWidget):
    def select(self, *args, **kwargs):
        return self.children[0] if self.children else None


class _DummyStyle:
    def configure(self, *args, **kwargs):
        return None

    def map(self, *args, **kwargs):
        return None


class _DummyCanvas(_DummyWidget):
    def bbox(self, *args, **kwargs):
        return (0, 0, 10, 10)


class _DummyText(_DummyWidget):
    def get(self, *args, **kwargs):
        return ""

    def tag_configure(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def _patch_app_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appV5.tk.Tk, "__init__", lambda self, *a, **k: None, raising=False)
    for name in ("title", "geometry", "config", "quit", "destroy", "update_idletasks"):
        monkeypatch.setattr(appV5.tk.Tk, name, lambda self, *a, **k: None, raising=False)

    monkeypatch.setattr(appV5.tk, "Menu", _DummyWidget, raising=False)
    monkeypatch.setattr(appV5.tk, "Canvas", _DummyCanvas, raising=False)
    monkeypatch.setattr(appV5.tk, "Text", _DummyText, raising=False)
    for var in ("BooleanVar", "StringVar", "IntVar", "DoubleVar"):
        monkeypatch.setattr(appV5.tk, var, _DummyVar, raising=False)

    for attr in (
        "Frame",
        "Button",
        "Label",
        "Scrollbar",
        "Treeview",
        "Checkbutton",
        "Combobox",
        "Separator",
        "LabelFrame",
        "Entry",
        "Progressbar",
    ):
        monkeypatch.setattr(appV5.ttk, attr, _DummyWidget, raising=False)

    monkeypatch.setattr(appV5.ttk, "Notebook", _DummyNotebook, raising=False)
    monkeypatch.setattr(appV5.ttk, "Style", lambda *a, **k: _DummyStyle(), raising=False)

    monkeypatch.setattr(appV5.messagebox, "showinfo", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(appV5.App, "_load_settings", lambda self: {}, raising=False)
    monkeypatch.setattr(appV5, "find_default_qwen_model", lambda: "", raising=False)

    class _DummyScrollable:
        def __init__(self, *args, **kwargs):
            self.canvas = _DummyCanvas()
            self.scrollable_frame = _DummyWidget()

        def pack(self, *args, **kwargs):
            return self

    monkeypatch.setattr(appV5, "ScrollableFrame", _DummyScrollable, raising=False)


def test_app_instantiation_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    app = appV5.App()
    try:
        assert isinstance(app.quote_state, appV5.QuoteState)
        assert isinstance(app.llm_enabled.get(), (type(None), bool))
        app.status_var.set("Ready")
    finally:
        app.destroy()


def test_geo_read_more_hook_is_optional() -> None:
    assert hasattr(appV5, "build_geo_from_dxf")
    hook = appV5.build_geo_from_dxf
    assert callable(hook)


def test_geo_read_more_hook_override_roundtrip() -> None:
    captured: list[str] = []

    def _fake_loader(path: str) -> dict:
        captured.append(path)
        return {"ok": True, "path": path}

    appV5.set_build_geo_from_dxf_hook(_fake_loader)
    try:
        result = appV5.build_geo_from_dxf("/tmp/file.dxf")
        assert result["ok"] is True
        assert captured == ["/tmp/file.dxf"]
    finally:
        appV5.set_build_geo_from_dxf_hook(None)
