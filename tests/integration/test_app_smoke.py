from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

if "cad_quoter.geometry" not in sys.modules:
    geom_stub = types.ModuleType("cad_quoter.geometry")
    sys.modules["cad_quoter.geometry"] = geom_stub
else:  # pragma: no cover - ensure attribute exists when module already imported
    geom_stub = sys.modules["cad_quoter.geometry"]

setattr(geom_stub, "FACE_OF", getattr(geom_stub, "FACE_OF", {}))


def _ensure_face(face, *_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return face


def _face_surface(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return None


def _iter_faces(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return iter(())


def _linear_properties(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return {}


def _map_shapes_and_ancestors(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return {}


setattr(geom_stub, "ensure_face", getattr(geom_stub, "ensure_face", _ensure_face))
setattr(geom_stub, "face_surface", getattr(geom_stub, "face_surface", _face_surface))
setattr(geom_stub, "iter_faces", getattr(geom_stub, "iter_faces", _iter_faces))
setattr(geom_stub, "linear_properties", getattr(geom_stub, "linear_properties", _linear_properties))
setattr(
    geom_stub,
    "map_shapes_and_ancestors",
    getattr(geom_stub, "map_shapes_and_ancestors", _map_shapes_and_ancestors),
)

import appV5
from cad_quoter.app import runtime as app_runtime


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

def test_discover_qwen_vl_assets_prefers_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "custom-model.gguf"
    model.write_text("model", encoding="utf-8")
    mmproj = tmp_path / "custom-mmproj.gguf"
    mmproj.write_text("proj", encoding="utf-8")

    monkeypatch.setenv("QWEN_VL_GGUF_PATH", str(model))
    monkeypatch.setenv("QWEN_VL_MMPROJ_PATH", str(mmproj))
    monkeypatch.setenv("QWEN_GGUF_PATH", "")
    monkeypatch.setenv("QWEN_MMPROJ_PATH", "")

    found_model, found_mmproj = app_runtime.discover_qwen_vl_assets()
    assert Path(found_model) == model
    assert Path(found_mmproj) == mmproj


def test_discover_qwen_vl_assets_scans_known_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("QWEN_VL_GGUF_PATH", raising=False)
    monkeypatch.delenv("QWEN_VL_MMPROJ_PATH", raising=False)
    monkeypatch.delenv("QWEN_GGUF_PATH", raising=False)
    monkeypatch.delenv("QWEN_MMPROJ_PATH", raising=False)

    monkeypatch.setattr(app_runtime, "PREFERRED_MODEL_DIRS", (tmp_path,), raising=False)

    model = tmp_path / app_runtime.DEFAULT_VL_MODEL_NAMES[0]
    mmproj = tmp_path / app_runtime.DEFAULT_MM_PROJ_NAMES[0]
    model.write_text("model", encoding="utf-8")
    mmproj.write_text("proj", encoding="utf-8")

    found_model, found_mmproj = app_runtime.discover_qwen_vl_assets()
    assert Path(found_model) == model
    assert Path(found_mmproj) == mmproj


def test_discover_qwen_vl_assets_errors_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("QWEN_VL_GGUF_PATH", raising=False)
    monkeypatch.delenv("QWEN_VL_MMPROJ_PATH", raising=False)
    monkeypatch.delenv("QWEN_GGUF_PATH", raising=False)
    monkeypatch.delenv("QWEN_MMPROJ_PATH", raising=False)
    monkeypatch.setattr(app_runtime, "PREFERRED_MODEL_DIRS", (tmp_path,), raising=False)

    with pytest.raises(RuntimeError) as exc:
        app_runtime.discover_qwen_vl_assets()

    message = str(exc.value)
    assert "QWEN_VL_GGUF_PATH" in message
    assert "QWEN_VL_MMPROJ_PATH" in message

def test_validate_quote_blocks_low_hole_cost_for_legacy_buckets() -> None:
    geo = {"hole_diams_mm": [5.0] * 12}
    pass_through = {"Material": 25.0}
    process_costs = {"drilling": 10.0, "milling": 0.0}

    with pytest.raises(ValueError) as exc:
        appV5.validate_quote_before_pricing(geo, process_costs, pass_through, {})

    assert "Unusually low machining time" in str(exc.value)


def test_validate_quote_requires_material_hints_when_cost_low() -> None:
    geo: dict[str, object] = {}
    pass_through = {"Material": 0.0}
    process_costs = {"drilling": 0.0, "milling": 0.0}

    with pytest.raises(ValueError) as exc:
        appV5.validate_quote_before_pricing(geo, process_costs, pass_through, {})

    assert "Material cost is near zero" in str(exc.value)


def test_validate_quote_allows_low_material_when_thickness_provided() -> None:
    geo = {"GEO-03_Height_mm": 6.0, "material": "6061"}
    pass_through = {"Material": 2.0}
    process_costs = {"drilling": 0.0, "milling": 0.0}

    try:
        appV5.validate_quote_before_pricing(geo, process_costs, pass_through, {})
    except ValueError as exc:  # pragma: no cover - should not raise
        pytest.fail(f"unexpected validation error: {exc}")

