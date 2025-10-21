from __future__ import annotations

import math

import pytest

from appV5 import _density_for_material, render_quote
from cad_quoter.domain_models import MATERIAL_DROPDOWN_OPTIONS
from cad_quoter.material_density import (
    MATERIAL_DENSITY_G_CC_BY_KEY,
    density_for_material,
    normalize_material_key,
)


@pytest.mark.parametrize(
    "display",
    [m for m in MATERIAL_DROPDOWN_OPTIONS if "other" not in m.lower()],
)
def test_density_matches_dropdown_entries(display: str) -> None:
    key = normalize_material_key(display)
    expected = MATERIAL_DENSITY_G_CC_BY_KEY.get(key)
    assert expected is not None, f"Missing density mapping for {display}"
    lookup_density = density_for_material(display)
    assert math.isclose(lookup_density, expected, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        _density_for_material(display), lookup_density, rel_tol=0.0, abs_tol=1e-6
    )


@pytest.mark.parametrize(
    "alias, expected_key",
    [
        ("ti-6al-4v", "titanium"),
        ("c172", "berylium copper"),
        ("phosphor bronze", "phosphor bronze"),
        ("nickel silver", "nickel silver"),
    ],
)
def test_density_handles_common_aliases(alias: str, expected_key: str) -> None:
    expected = MATERIAL_DENSITY_G_CC_BY_KEY[normalize_material_key(expected_key)]
    lookup_density = density_for_material(alias)
    assert math.isclose(lookup_density, expected, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(
        _density_for_material(alias), lookup_density, rel_tol=0.0, abs_tol=1e-6
    )


def test_render_quote_displays_weight_in_pounds_ounces() -> None:
    breakdown = {
        "totals": {"labor_cost": 0.0, "direct_costs": 0.0},
        "material": {
            "mass_g": 6000.0,
            "mass_g_net": 5800.0,
            "scrap_pct": 0.05,
        },
        "qty": 1,
    }
    result = {"breakdown": breakdown, "price": 0.0, "ui_vars": {}}

    text = render_quote(result)

    assert "Weight Reference: Start 13 lb 3.6 oz" in text
    assert "Net 12 lb 12.6 oz" in text
    assert "Scrap 7.1 oz" in text
    weight_lines = "\n".join(
        line for line in text.splitlines() if "Weight" in line or "Mass" in line
    )
    assert " g" not in weight_lines
