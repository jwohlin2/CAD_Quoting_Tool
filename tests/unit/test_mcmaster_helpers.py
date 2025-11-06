from cad_quoter.pricing.mcmaster_helpers import collect_available_plate_thicknesses


def test_collect_available_plate_thicknesses_parses_and_sorts_unique_values():
    rows = [
        {"thickness_in": "0.500"},
        {"T_in": "1/4"},
        {"thk_in": " 0.375 "},
        {"thickness": 0.375},  # duplicate value should be deduped
        {"thickness_in": ""},  # ignored blank
        {"thickness_in": None},
        {"thickness_in": "bad"},
    ]

    assert collect_available_plate_thicknesses(rows) == [0.25, 0.375, 0.5]
