import math

import appV5
import cad_quoter.pricing.materials as materials


def test_pick_mcmaster_plate_handles_quoted_thickness_strings() -> None:
    sku = appV5._pick_mcmaster_plate_sku(
        13.01,
        13.01,
        2.0,
        material_key="aluminum mic6",
    )

    assert sku is not None
    assert math.isclose(sku["thk_in"], 2.0, abs_tol=1e-6)
    dims = sorted((sku["len_in"], sku["wid_in"]))
    assert dims[0] >= 13.01
    assert dims[1] >= 13.01
    assert sku["mcmaster_part"]


def test_resolve_plate_for_quote_falls_back_to_existing_stock_dims() -> None:
    catalog_rows = [
        {
            "material": "Aluminum MIC6",
            "thickness_in": "2",
            "length_in": "18",
            "width_in": "18",
            "part": "86825K956",
        }
    ]

    sku = appV5._resolve_mcmaster_plate_for_quote(
        None,
        None,
        None,
        material_key="aluminum mic6",
        stock_L_in=18.0,
        stock_W_in=18.0,
        stock_T_in=2.0,
        catalog_rows=catalog_rows,
    )

    assert sku is not None
    assert sku["mcmaster_part"] == "86825K956"
