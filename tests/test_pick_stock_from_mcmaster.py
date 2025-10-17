import sys
import types
from types import SimpleNamespace

import pytest

# Provide a lightweight requests stub for optional dependencies during import time.
requests_stub = types.ModuleType("requests")
class _DummySession:
    def __init__(self, *args, **kwargs):
        pass

requests_stub.Session = _DummySession
sys.modules.setdefault("requests", requests_stub)

from materials import pick_stock_from_mcmaster  # noqa: E402


@pytest.mark.parametrize(
    "length,width,thickness,expected_part",
    [
        (12.0, 24.0, 3.5, "86825K626"),
    ],
)
def test_pick_stock_prefers_smallest_plate_when_scrap_blocks_exact(
    length: float, width: float, thickness: float, expected_part: str
) -> None:
    cfg = SimpleNamespace(enforce_exact_thickness=True, allow_thickness_upsize=False)
    result = pick_stock_from_mcmaster("Aluminum MIC6", length, width, thickness, cfg=cfg)
    assert result is not None
    assert pytest.approx(24.0) == result["len_in"]
    assert pytest.approx(12.0) == result["wid_in"]
    assert pytest.approx(thickness) == result["thk_in"]
    assert result.get("mcmaster_part") == expected_part
    assert pytest.approx(24.0) == result["required_blank_len_in"]
    assert pytest.approx(12.0) == result["required_blank_wid_in"]
