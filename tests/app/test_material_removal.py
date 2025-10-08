import math

import pytest

import appV5
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


def test_plate_scrap_pct_tracks_hole_volume() -> None:
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {"Item": "Material", "Example Values / Options": "6061-T6 Aluminum", "Data Type / Input Method": "text"},
            {"Item": "Material Name", "Example Values / Options": "6061-T6 Aluminum", "Data Type / Input Method": "text"},
            {"Item": "Thickness (in)", "Example Values / Options": 0.5, "Data Type / Input Method": "number"},
            {"Item": "Plate Length (in)", "Example Values / Options": 6.0, "Data Type / Input Method": "number"},
            {"Item": "Plate Width (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
            {"Item": "Roughing Cycle Time", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
        ]
    )

    geo = {
        "kind": "2d",
        "material": "6061-T6 Aluminum",
        "thickness_mm": 12.7,
        "plate_length_mm": 152.4,
        "plate_width_mm": 101.6,
        "hole_diams_mm": [12.7, 12.7],
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)
    material_info = result["breakdown"]["material"]
    scrap_pct = material_info.get("scrap_pct")

    assert scrap_pct is not None

    density = appV5._density_for_material("6061")
    thickness_in = 0.5
    length_in = 6.0
    width_in = 4.0
    plate_volume_in3 = length_in * width_in * thickness_in
    radius_mm = 12.7 / 2.0
    height_mm = thickness_in * 25.4
    hole_volume_mm3 = math.pi * (radius_mm**2) * height_mm * 2
    hole_volume_in3 = hole_volume_mm3 / 16387.064
    removed_mass = hole_volume_in3 * 16.387064 * density
    net_mass = (plate_volume_in3 - hole_volume_in3) * 16.387064 * density

    expected_scrap = min(0.25, removed_mass / net_mass)
    assert scrap_pct == pytest.approx(expected_scrap, rel=1e-6)
