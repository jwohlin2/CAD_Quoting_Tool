import appV5


def test_coerce_overhead_dataclass_adds_index_attribute() -> None:
    legacy = {
        "toolchange_min": 0.5,
        "approach_retract_in": 0.25,
        "peck_penalty_min_per_in_depth": 0.03,
        "dwell_min": None,
        "peck_min": None,
        "index_sec_per_hole": "18.0",
    }

    compat = appV5._coerce_overhead_dataclass(legacy)

    assert hasattr(compat, "index_sec_per_hole")
    assert getattr(compat, "index_sec_per_hole") == 18.0
    assert getattr(compat, "toolchange_min") == legacy["toolchange_min"]


def test_coerce_overhead_dataclass_preserves_existing_dataclass() -> None:
    overhead = appV5._TimeOverheadParams(index_sec_per_hole=12.0)

    compat = appV5._coerce_overhead_dataclass(overhead)

    assert compat is overhead
    assert getattr(compat, "index_sec_per_hole") == 12.0
