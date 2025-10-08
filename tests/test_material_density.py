import math

import pytest

from appV5 import _density_for_material, _material_family, net_mass_kg


def test_net_mass_kg_uses_copper_density():
    density = _density_for_material("Copper")
    assert math.isclose(density, 8.96, rel_tol=0.02)

    length_in = 4.0
    width_in = 3.0
    thickness_in = 0.5
    expected_mass = (length_in * width_in * thickness_in * 16.387064 * density) / 1000.0

    mass = net_mass_kg(length_in, width_in, thickness_in, [], density)
    assert math.isclose(mass, expected_mass, rel_tol=1e-9)

    assert _material_family("Copper") == "copper"


def test_net_mass_kg_optionally_returns_removed_mass() -> None:
    density = _density_for_material("Aluminum")
    length_in = 6.0
    width_in = 2.0
    thickness_in = 0.5
    hole_diam_mm = 12.7  # 0.5 in

    net_mass, removed_mass = net_mass_kg(
        length_in,
        width_in,
        thickness_in,
        [hole_diam_mm],
        density,
        return_removed_mass=True,
    )

    assert net_mass is not None
    assert removed_mass is not None

    volume_plate_in3 = length_in * width_in * thickness_in
    radius_mm = hole_diam_mm / 2.0
    height_mm = thickness_in * 25.4
    hole_volume_mm3 = math.pi * (radius_mm**2) * height_mm
    hole_volume_in3 = hole_volume_mm3 / 16387.064
    expected_removed_g = hole_volume_in3 * 16.387064 * density

    assert removed_mass == pytest.approx(expected_removed_g, rel=1e-9)
    expected_net_mass = (volume_plate_in3 - hole_volume_in3) * 16.387064 * density / 1000.0
    assert net_mass == pytest.approx(expected_net_mass, rel=1e-9)
