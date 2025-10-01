"""Unit tests for editor control classification helpers."""

from appV5 import derive_editor_control_spec


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
