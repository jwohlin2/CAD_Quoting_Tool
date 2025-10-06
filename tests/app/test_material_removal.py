import math

from appV5 import compute_mass_and_scrap_after_removal


def test_compute_mass_and_scrap_after_removal_preserves_effective_mass() -> None:
    net_after, scrap_after, eff_after = compute_mass_and_scrap_after_removal(1000.0, 0.1, 100.0)

    expected_scrap = (1000.0 * 0.1 + 100.0) / 900.0

    assert math.isclose(net_after, 900.0)
    assert math.isclose(scrap_after, expected_scrap, rel_tol=1e-9)
    assert math.isclose(eff_after, 1100.0, rel_tol=1e-9)


def test_compute_mass_and_scrap_after_removal_respects_scrap_bounds() -> None:
    net_after, scrap_after, eff_after = compute_mass_and_scrap_after_removal(
        1000.0,
        0.2,
        400.0,
        scrap_min=0.0,
        scrap_max=0.25,
    )

    assert math.isclose(net_after, 600.0)
    assert math.isclose(scrap_after, 0.25)
    assert math.isclose(eff_after, 600.0 * 1.25)


def test_compute_mass_and_scrap_after_removal_no_change_when_zero_removal() -> None:
    net_after, scrap_after, eff_after = compute_mass_and_scrap_after_removal(500.0, 0.05, 0.0)

    assert math.isclose(net_after, 500.0)
    assert math.isclose(scrap_after, 0.05)
    assert math.isclose(eff_after, 500.0 * 1.05)
