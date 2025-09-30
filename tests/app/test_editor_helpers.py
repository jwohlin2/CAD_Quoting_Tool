import appV5


class DummyVar:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _make_stub_app():
    app = appV5.App.__new__(appV5.App)
    app.editor_vars = {
        "Hole Count (override)": DummyVar(""),
        "Avg Hole Diameter (mm)": DummyVar(""),
    }
    app.editor_value_sources = {}
    app.editor_label_widgets = {}
    app.editor_label_base = {}
    app._editor_set_depth = 0
    return app


def test_update_material_price_field_uses_fallback(monkeypatch):
    choice_var = DummyVar("Custom Alloy")
    price_var = DummyVar("")
    material_lookup = {}

    monkeypatch.setattr(
        appV5,
        "_resolve_material_unit_price",
        lambda choice, unit="kg": (5.5, "backup_csv:material_price_backup.csv"),
    )

    changed = appV5._update_material_price_field(choice_var, price_var, material_lookup)

    assert changed is True
    assert price_var.get() == "0.0055"


def test_apply_geo_defaults_populates_hole_fields():
    app = _make_stub_app()

    geo = {
        "hole_count": 6,
        "hole_diams_mm": [3.0, 5.0, 4.0],
        "hole_bins": {2.0: 10},
        "feature_counts": {"hole_count": 6},
    }

    app._apply_geo_defaults(geo)

    assert app.editor_vars["Hole Count (override)"].get() == "6.000"
    assert app.editor_vars["Avg Hole Diameter (mm)"].get() == "4.000"


def test_apply_geo_defaults_falls_back_to_hole_bins():
    app = _make_stub_app()

    geo = {
        "hole_bins": {3.0: 2, 5.0: 4},
    }

    app._apply_geo_defaults(geo)

    assert app.editor_vars["Hole Count (override)"].get() == "6.000"
    assert app.editor_vars["Avg Hole Diameter (mm)"].get() == "4.333"
