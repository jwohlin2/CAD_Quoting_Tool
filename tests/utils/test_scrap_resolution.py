"""Tests for :mod:`cad_quoter.utils.scrap` helper functions."""

from __future__ import annotations

import math

import pytest

from cad_quoter.utils.scrap import resolve_material_scrap_state
import cad_quoter.pricing.materials as materials


def _density_lookup(_: str) -> float:
    return 2.7


def test_resolve_material_scrap_state_prefers_ui_value() -> None:
    state = resolve_material_scrap_state(
        value_map={"Scrap Percent (%)": "12"},
        geo_payload={},
        material_text="6061",
        scrap_value="12",
        density_lookup=_density_lookup,
    )

    assert math.isclose(state.scrap_fraction, 0.12, rel_tol=0.0, abs_tol=1e-9)
    assert state.scrap_source == "ui"
    assert state.scrap_source_label == "ui"


def test_hole_scrap_bumps_fraction_above_ui() -> None:
    geo_payload = {
        "derived": {
            "hole_diams_mm": [20.0, 20.0, 20.0],
            "bbox_mm": (100.0, 100.0),
        },
        "thickness_mm": 25.4,
        "plate_len_mm": 100.0,
        "plate_wid_mm": 100.0,
    }

    state = resolve_material_scrap_state(
        value_map={"Scrap Percent (%)": 0.02},
        geo_payload=geo_payload,
        material_text="6061",
        scrap_value=0.02,
        density_lookup=_density_lookup,
    )

    assert state.scrap_fraction > 0.02
    assert state.scrap_source_label.endswith("+holes")


def test_finalize_credit_uses_wieland_price_when_available() -> None:
    state = resolve_material_scrap_state(
        value_map={},
        geo_payload={},
        material_text="6061",
        scrap_value=None,
        density_lookup=_density_lookup,
    )

    state.finalize_credit(
        material_block={"scrap_weight_lb": 10},
        overrides={},
        geo_context={"material_group": "Aluminum"},
        material_display="6061",
        material_group_display="Aluminum",
        wieland_lookup=lambda _: 2.5,
        default_recovery_fraction=materials.SCRAP_RECOVERY_DEFAULT,
        default_scrap_price_usd_per_lb=materials.SCRAP_PRICE_FALLBACK_USD_PER_LB,
    )

    credit = state.scrap_credit
    assert credit.source == "wieland"
    assert credit.unit_price_usd_per_lb == pytest.approx(2.5)
    assert credit.amount_usd == pytest.approx(
        10 * 2.5 * materials.SCRAP_RECOVERY_DEFAULT
    )


def test_finalize_credit_handles_zero_credit_gracefully() -> None:
    state = resolve_material_scrap_state(
        value_map={},
        geo_payload={},
        material_text="6061",
        scrap_value=None,
        density_lookup=_density_lookup,
    )

    state.finalize_credit(
        material_block={},
        overrides=None,
        geo_context={},
        material_display="6061",
        material_group_display="Aluminum",
        wieland_lookup=lambda _: None,
        default_recovery_fraction=materials.SCRAP_RECOVERY_DEFAULT,
        default_scrap_price_usd_per_lb=materials.SCRAP_PRICE_FALLBACK_USD_PER_LB,
    )

    credit = state.scrap_credit
    assert credit.amount_usd is None
    assert credit.source is None

