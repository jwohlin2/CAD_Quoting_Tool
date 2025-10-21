"""Unit tests for editor control classification helpers."""

import pytest

try:
    from appV5 import default_variables_template
except ImportError:  # pragma: no cover - optional dependency for tests
    default_variables_template = None

from cad_quoter.ui.editor_controls import coerce_checkbox_state, derive_editor_control_spec


def test_number_control_from_declared_dtype():
    spec = derive_editor_control_spec("Number", "0.25")
    assert spec.control == "number"
    assert spec.entry_value == "0.25"
    assert spec.display_label.lower() == "number"


def test_dropdown_control_uses_options_list():
    spec = derive_editor_control_spec("Dropdown", "Auto, Manual, Override")
    assert spec.control == "dropdown"
    assert spec.options == ("Auto", "Manual", "Override")
    assert spec.entry_value == "Auto"
    assert not spec.guessed_dropdown


def test_checkbox_detection_handles_boolean_pairs():
    spec = derive_editor_control_spec("Checkbox", "True / False")
    assert spec.control == "checkbox"
    assert spec.entry_value in {"True", "False"}
    assert spec.display_label.lower() == "checkbox"


def test_formula_fields_surface_base_text():
    spec = derive_editor_control_spec("Lookup Value (Percentage)", "MachiningCost * 0.04")
    assert spec.control == "formula"
    assert spec.entry_value == ""
    assert spec.base_text == "MachiningCost * 0.04"


def test_formula_numeric_values_keep_default():
    spec = derive_editor_control_spec("Lookup Value (Rate)", "$80.00")
    assert spec.control == "formula"
    assert spec.entry_value == "80"
    assert spec.base_text == "$80.00"


def test_options_without_dtype_are_promoted_to_dropdown():
    spec = derive_editor_control_spec("", "Low, Medium, High")
    assert spec.control == "dropdown"
    assert spec.guessed_dropdown
    assert spec.options == ("Low", "Medium", "High")


@pytest.mark.parametrize(
    "value, default, expected",
    [
        (True, False, True),
        (0, True, False),
        (1.0, False, True),
        (float("nan"), True, True),
        ("Yes", False, True),
        ("no", True, False),
        (" y ", False, True),
        ("N", True, False),
        ("Yes / No", False, True),
        ("Off / On", True, False),
        ("maybe", True, True),
        ("", False, False),
    ],
)
def test_coerce_checkbox_state_handles_common_inputs(value, default, expected):
    assert coerce_checkbox_state(value, default) is expected


@pytest.mark.skipif(default_variables_template is None, reason="default template unavailable")
def test_default_template_flags_render_as_checkboxes():
    df = default_variables_template()
    for item in ("FAIR Required", "Source Inspection Requirement"):
        row = next((row for _, row in df.iterrows() if row["Item"] == item), None)
        assert row is not None, f"Missing {item} in default template"
        dtype = row["Data Type / Input Method"]
        example = row["Example Values / Options"]
        spec = derive_editor_control_spec(dtype, example)
        assert spec.control == "checkbox"
        assert spec.checkbox_state is False
