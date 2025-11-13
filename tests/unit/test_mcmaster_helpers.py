from cad_quoter.pricing.mcmaster_helpers import (
    collect_available_plate_thicknesses,
    get_qty_one_tier,
    compute_price_per_cubic_inch,
    estimate_price_for_part_from_volume,
)


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


def test_get_qty_one_tier_finds_tier_with_min_qty_one():
    tiers = [
        {"MinimumQuantity": 1, "Amount": 10.50, "UnitOfMeasure": "Each"},
        {"MinimumQuantity": 10, "Amount": 9.00, "UnitOfMeasure": "Each"},
        {"MinimumQuantity": 100, "Amount": 7.50, "UnitOfMeasure": "Each"},
    ]

    result = get_qty_one_tier(tiers)
    assert result is not None
    assert result["MinimumQuantity"] == 1
    assert result["Amount"] == 10.50


def test_get_qty_one_tier_finds_tier_with_min_qty_less_than_one():
    tiers = [
        {"MinimumQuantity": 0, "Amount": 12.00, "UnitOfMeasure": "Each"},
        {"MinimumQuantity": 10, "Amount": 9.00, "UnitOfMeasure": "Each"},
    ]

    result = get_qty_one_tier(tiers)
    assert result is not None
    assert result["MinimumQuantity"] == 0
    assert result["Amount"] == 12.00


def test_get_qty_one_tier_returns_none_when_no_tier_available():
    tiers = [
        {"MinimumQuantity": 10, "Amount": 20.00, "UnitOfMeasure": "Each"},
        {"MinimumQuantity": 25, "Amount": 18.00, "UnitOfMeasure": "Each"},
    ]

    result = get_qty_one_tier(tiers)
    assert result is None


def test_get_qty_one_tier_returns_none_for_empty_list():
    assert get_qty_one_tier([]) is None


def test_compute_price_per_cubic_inch_with_qty_one_tier():
    tiers = [
        {"MinimumQuantity": 1, "Amount": 20.00, "UnitOfMeasure": "Each"},
        {"MinimumQuantity": 10, "Amount": 18.00, "UnitOfMeasure": "Each"},
    ]
    stock_volume = 336.0  # 12 x 14 x 2 inches

    result = compute_price_per_cubic_inch(tiers, stock_volume)
    assert result is not None
    assert abs(result - 0.0595238) < 0.0001  # 20 / 336 ≈ 0.0595


def test_compute_price_per_cubic_inch_without_qty_one_tier():
    tiers = [
        {"MinimumQuantity": 10, "Amount": 20.00, "UnitOfMeasure": "Each"},
        {"MinimumQuantity": 25, "Amount": 18.00, "UnitOfMeasure": "Each"},
    ]
    stock_volume = 336.0

    result = compute_price_per_cubic_inch(tiers, stock_volume)
    assert result is not None
    assert abs(result - 0.0595238) < 0.0001  # Uses first tier: 20 / 336


def test_compute_price_per_cubic_inch_returns_none_for_zero_volume():
    tiers = [{"MinimumQuantity": 1, "Amount": 20.00, "UnitOfMeasure": "Each"}]

    assert compute_price_per_cubic_inch(tiers, 0) is None
    assert compute_price_per_cubic_inch(tiers, -5) is None


def test_compute_price_per_cubic_inch_returns_none_for_empty_tiers():
    assert compute_price_per_cubic_inch([], 336.0) is None


def test_estimate_price_for_part_from_volume():
    tiers = [
        {"MinimumQuantity": 1, "Amount": 20.00, "UnitOfMeasure": "Each"},
    ]
    stock_volume = 336.0  # 12 x 14 x 2
    part_volume = 48.0    # 8 x 6 x 1

    result = estimate_price_for_part_from_volume(tiers, stock_volume, part_volume)
    assert result is not None
    # Expected: (20 / 336) * 48 ≈ 2.857
    assert abs(result - 2.857) < 0.01


def test_estimate_price_for_part_from_volume_with_min_qty_tier():
    tiers = [
        {"MinimumQuantity": 10, "Amount": 20.00, "UnitOfMeasure": "Each"},
        {"MinimumQuantity": 25, "Amount": 18.00, "UnitOfMeasure": "Each"},
    ]
    stock_volume = 336.0
    part_volume = 48.0

    result = estimate_price_for_part_from_volume(tiers, stock_volume, part_volume)
    assert result is not None
    # Should use smallest MinQty tier (highest price): (20 / 336) * 48
    assert abs(result - 2.857) < 0.01


def test_estimate_price_for_part_from_volume_returns_none_for_zero_part_volume():
    tiers = [{"MinimumQuantity": 1, "Amount": 20.00, "UnitOfMeasure": "Each"}]
    stock_volume = 336.0

    assert estimate_price_for_part_from_volume(tiers, stock_volume, 0) is None
    assert estimate_price_for_part_from_volume(tiers, stock_volume, -10) is None


def test_estimate_price_for_part_from_volume_returns_none_for_invalid_stock():
    tiers = [{"MinimumQuantity": 1, "Amount": 20.00, "UnitOfMeasure": "Each"}]
    part_volume = 48.0

    assert estimate_price_for_part_from_volume(tiers, 0, part_volume) is None
    assert estimate_price_for_part_from_volume([], 336.0, part_volume) is None
