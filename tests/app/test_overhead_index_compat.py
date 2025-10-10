from dataclasses import dataclass

import appV5


def test_ensure_overhead_index_attr_adds_missing_attribute() -> None:
    @dataclass(slots=True)
    class LegacyOverhead:
        toolchange_min: float | None = 0.5
        approach_retract_in: float | None = 0.25
        peck_penalty_min_per_in_depth: float | None = 0.03
        dwell_min: float | None = None
        peck_min: float | None = None

    legacy = LegacyOverhead()

    compat = appV5._ensure_overhead_index_attr(legacy, 18.0)

    assert hasattr(compat, "index_sec_per_hole")
    assert getattr(compat, "index_sec_per_hole") == 18.0
    assert getattr(compat, "toolchange_min") == legacy.toolchange_min


def test_ensure_overhead_index_attr_preserves_existing_attribute() -> None:
    overhead = appV5._TimeOverheadParams(index_sec_per_hole=12.0)

    compat = appV5._ensure_overhead_index_attr(overhead, 24.0)

    assert compat is overhead
    assert getattr(compat, "index_sec_per_hole") == 12.0
