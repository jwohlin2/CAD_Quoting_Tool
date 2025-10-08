from appV5 import _estimate_inprocess_default_from_tolerance


def test_inspection_hours_scale_with_tolerance_precision():
    loose = _estimate_inprocess_default_from_tolerance({"Default": "±0.005"})
    medium = _estimate_inprocess_default_from_tolerance({"Default": "±0.001"})
    tight = _estimate_inprocess_default_from_tolerance({"Default": "±0.0002"})

    assert loose < medium < tight


def test_multiple_tight_callouts_add_sublinear_premium():
    single = _estimate_inprocess_default_from_tolerance({"Tol A": "±0.0006"})
    stacked = _estimate_inprocess_default_from_tolerance(
        {"Tol A": "±0.0006", "Tol B": "±0.0004", "Tol C": "±0.0004"}
    )

    assert stacked - single < single  # premium should not double the base
