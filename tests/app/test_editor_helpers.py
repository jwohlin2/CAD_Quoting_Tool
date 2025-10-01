import appV5
import pytest


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
    app.default_material_display = ""
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


def test_infer_geo_override_defaults_basic_fields():
    geo = {
        "plate_len_in": 10.0,
        "plate_wid_in": 5.0,
        "thickness_mm": 12.7,
        "hole_diams_mm": [10.0, 5.0, 5.0],
        "tap_qty": 4,
        "cbore_qty": 2,
        "csk_qty": 1,
        "material": "Aluminum 6061",
        "fai_required": True,
        "setups": 3,
    }

    defaults = appV5.infer_geo_override_defaults(geo)

    assert defaults["Plate Length (in)"] == pytest.approx(10.0)
    assert defaults["Plate Width (in)"] == pytest.approx(5.0)
    assert defaults["Thickness (in)"] == pytest.approx(12.7 / 25.4)
    assert defaults["Number of Milling Setups"] == 3
    assert defaults["Material"] == "Aluminum 6061"
    assert defaults["FAIR Required"] == 1
    assert defaults["Tap Qty (LLM/GEO)"] == 4
    assert defaults["Cbore Qty (LLM/GEO)"] == 2
    assert defaults["Csk Qty (LLM/GEO)"] == 1
    assert defaults["Hole Count (override)"] == 3
    assert defaults["Avg Hole Diameter (mm)"] == pytest.approx((10.0 + 5.0 + 5.0) / 3.0)


def test_infer_geo_override_defaults_uses_bins_and_back_face():
    geo = {
        "meta": {"needs_back_face": True},
        "derived": {"hole_bins": {"0.25 in": 4, "6 mm": 2}},
        "scrap_pct": 0.08,
    }

    defaults = appV5.infer_geo_override_defaults(geo)

    expected_avg = ((0.25 * 25.4) * 4 + 6.0 * 2) / 6.0
    assert defaults["Hole Count (override)"] == 6
    assert defaults["Avg Hole Diameter (mm)"] == pytest.approx(expected_avg)
    assert defaults["Number of Milling Setups"] == 2
    assert defaults["Scrap Percent (%)"] == pytest.approx(8.0)


def test_apply_geo_defaults_populates_thickness(monkeypatch):
    app = _make_stub_app()
    thickness_var = DummyVar("0.0")
    app.editor_vars["Thickness (in)"] = thickness_var

    monkeypatch.setattr(
        appV5,
        "infer_geo_override_defaults",
        lambda geo: {"Thickness (in)": 1.0},
    )

    app._apply_geo_defaults({})

    assert thickness_var.get() == "1.000"
    assert app.editor_value_sources["Thickness (in)"] == "GEO"

