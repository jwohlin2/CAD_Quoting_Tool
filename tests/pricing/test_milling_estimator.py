import pytest

from cad_quoter.pricing.milling_estimator import estimate_milling_minutes_from_geometry
from cad_quoter.speeds_feeds import ipm_from_rpm_ipt, rpm_from_sfm


@pytest.fixture
def aluminum_profile_row() -> dict:
    return {
        "material_group": "N1",
        "operation": "Endmill_Profile",
        "sfm_start": 800.0,
        "feed_type": "fz",
        "fz_ipr_0_5in": 0.010,
        "stepover_pct": 0.5,
    }


def test_finish_feed_uses_sfm_ipt_when_linear_rate_missing(aluminum_profile_row: dict) -> None:
    geom = {
        "edge_len_in": 120.0,
        "thickness_in": 1.0,
        "plate_area_in2": 0.0,
        "pocket_area_in2": 0.0,
        "finish_tool_diam_in": 0.5,
    }

    result = estimate_milling_minutes_from_geometry(
        geom=geom,
        sf_df=[aluminum_profile_row],
        material_group="N1",
        rates=None,
    )

    assert result is not None
    rpm = rpm_from_sfm(800.0, 0.5)
    finish_ipm = ipm_from_rpm_ipt(rpm, 4, 0.010)
    expected_minutes = (geom["edge_len_in"] / finish_ipm) * 60.0
    assert result["minutes"] == pytest.approx(expected_minutes, rel=1e-6, abs=1e-6)


def test_face_stepover_scales_with_tool_diameter(aluminum_profile_row: dict) -> None:
    geom = {
        "plate_area_in2": 144.0,
        "thickness_in": 1.0,
        "edge_len_in": 0.0,
        "pocket_area_in2": 0.0,
        "finish_tool_diam_in": 0.5,
        "face_tool_diam_in": 2.0,
    }

    result = estimate_milling_minutes_from_geometry(
        geom=geom,
        sf_df=[aluminum_profile_row],
        material_group="N1",
        rates=None,
    )

    assert result is not None
    rpm = rpm_from_sfm(800.0, 0.5)
    finish_ipm = ipm_from_rpm_ipt(rpm, 4, 0.010)
    rpm_face = rpm_from_sfm(800.0, geom["face_tool_diam_in"])
    face_ipm = ipm_from_rpm_ipt(rpm_face, 6, 0.010)
    face_feed = face_ipm * 0.75
    stepover_in = aluminum_profile_row["stepover_pct"] * geom["face_tool_diam_in"]
    effective_length = geom["plate_area_in2"] / stepover_in
    expected_minutes = (effective_length / face_feed) * 60.0
    assert result["minutes"] == pytest.approx(expected_minutes, rel=1e-6, abs=1e-6)
