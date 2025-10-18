import math

import appV5
import materials


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


def test_compute_material_block_promotes_mcmaster_fallback(monkeypatch) -> None:
    geo_ctx = {
        "material_display": "Aluminum MIC6",
        "outline_bbox": {"plate_len_in": 13.01, "plate_wid_in": 13.01},
        "thickness_in": 2.0,
    }

    monkeypatch.setattr(materials, "pick_stock_from_mcmaster", lambda *args, **kwargs: None)
    monkeypatch.setattr(materials, "_pick_plate_from_mcmaster", lambda *args, **kwargs: None)

    stdgrid_pick = {
        "vendor": "StdGrid",
        "len_in": 18.0,
        "wid_in": 18.0,
        "thk_in": 2.0,
    }
    monkeypatch.setattr(
        materials,
        "_pick_from_stdgrid",
        lambda *args, **kwargs: dict(stdgrid_pick),
    )

    fallback_candidate = {
        "len_in": 18.0,
        "wid_in": 18.0,
        "thk_in": 2.0,
        "mcmaster_part": "86825K956",
        "source": "mcmaster-catalog",
    }
    monkeypatch.setattr(
        materials,
        "resolve_mcmaster_plate_for_quote",
        lambda *args, **kwargs: dict(fallback_candidate),
    )

    block = materials._compute_material_block(geo_ctx, "aluminum mic6", 2.70, 0.1)

    assert block["part_no"] == "86825K956"
    assert block["stock_vendor"] == "McMaster"
    assert block["stock_source_tag"] == "mcmaster-catalog"
