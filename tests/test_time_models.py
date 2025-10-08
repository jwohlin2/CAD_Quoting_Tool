import math

import pytest

from time_models import minutes_surface_grind, minutes_wedm


def test_minutes_wedm_returns_minutes_without_extra_scaling():
    """A 10" perimeter at 0.6 ipm should take ~17 minutes including ancillaries."""
    params = {
        "perimeter_in": 10.0,
        "starts": 1,
        "tabs": 0,
        "passes": 1,
        "wire_in": 0.010,
    }

    minutes = minutes_wedm(params)

    # Cutting time is 10 / 0.6 = 16.666..., with 0.6 minutes ancillary start/stop.
    expected = 10.0 / 0.60 + 0.6
    assert math.isclose(minutes, expected, rel_tol=1e-6)


def test_minutes_surface_grind_returns_minutes_without_extra_scaling():
    params = {
        "area_sq_in": 20.0,
        "stock_in": 0.002,
    }

    minutes = minutes_surface_grind(params)

    # Stock removal requires 4 passes. Each pass sweeps 20/2 = 10 in at 60 ipm => 0.1666... min.
    expected = 6.0 + 4 * ((20.0 / 2.0) / 60.0)
    assert math.isclose(minutes, expected, rel_tol=1e-6)
