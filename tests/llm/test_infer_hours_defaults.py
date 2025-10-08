import pytest

from cad_quoter.llm import infer_hours_and_overrides_from_geo


def test_infer_hours_programming_defaults_to_one_hour_without_llm() -> None:
    geo = {
        "GEO__Face_Count": 512,
        "GEO__MaxDim_mm": 100.0,
        "GEO__MinWall_mm": 5.0,
        "GEO__ThinWall_Present": False,
        "GEO_Deburr_EdgeLen_mm": 0.0,
    }

    result = infer_hours_and_overrides_from_geo(geo, params={}, rates={}, client=None)

    hours = result.get("hours") or {}
    assert pytest.approx(hours.get("Programming_Hours"), rel=1e-6) == 1.0
