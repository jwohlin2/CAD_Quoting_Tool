import math

import appV5


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
