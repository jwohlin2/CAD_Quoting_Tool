"""Unit tests for manual dimension overrides in CAD helpers."""

from pathlib import Path

import pytest

from cad_quoter.planning import plan_from_cad_file
from cad_quoter.pricing.DirectCostHelper import extract_part_info_from_cad


CAD_PATH = Path("Cad Files/301_redacted.dxf")


@pytest.mark.skipif(not CAD_PATH.exists(), reason="Test CAD file not available")
def test_plan_from_cad_file_applies_overrides():
    overrides = {"L": "10.5", "W": 8.25, "T": 0.75}

    plan = plan_from_cad_file(
        CAD_PATH,
        use_paddle_ocr=False,
        verbose=False,
        override_dims=overrides,
    )

    dims = plan.get("extracted_dims")
    assert dims, "Plan should expose extracted dimensions when overrides are supplied"
    assert dims["L"] == pytest.approx(10.5)
    assert dims["W"] == pytest.approx(8.25)
    assert dims["T"] == pytest.approx(0.75)


@pytest.mark.skipif(not CAD_PATH.exists(), reason="Test CAD file not available")
def test_extract_part_info_from_cad_applies_overrides():
    overrides = {"l": 12.0, "W": 6.0, "T": "0.5"}

    part_info = extract_part_info_from_cad(
        CAD_PATH,
        material="Aluminum 6061-T6",
        use_paddle_ocr=False,
        auto_detect_material=False,
        verbose=False,
        override_dims=overrides,
    )

    assert part_info.length == pytest.approx(12.0)
    assert part_info.width == pytest.approx(6.0)
    assert part_info.thickness == pytest.approx(0.5)
    assert part_info.area == pytest.approx(12.0 * 6.0)
    assert part_info.volume == pytest.approx(12.0 * 6.0 * 0.5)
