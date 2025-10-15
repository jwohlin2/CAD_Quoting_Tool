import pytest

pd = pytest.importorskip("pandas")

import appV5


def test_fallback_populates_drilling_groups() -> None:
    df = pd.DataFrame(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {
                "Item": "Material",
                "Example Values / Options": "6061-T6 Aluminum",
                "Data Type / Input Method": "text",
            },
        ]
    )

    geo = {"hole_diams_mm": [6.35, 6.35, 12.7], "thickness_in": 0.75}

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)

    drilling_meta = result.get("breakdown", {}).get("drilling_meta", {})
    groups = drilling_meta.get("bins_list") or []
    assert groups, "expected fallback drill groups to be generated"

    total_qty = sum(int(group.get("qty", 0)) for group in groups)
    assert total_qty == len(geo["hole_diams_mm"])
    assert drilling_meta.get("hole_count") == total_qty
