import math

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
