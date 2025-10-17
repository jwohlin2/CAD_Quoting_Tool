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

import materials  # noqa: E402
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


def test_pick_stock_rejects_api_thickness_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    if materials._mc is None:  # pragma: no cover - optional dependency missing
        pytest.skip("McMaster helpers not available")

    cfg = SimpleNamespace(enforce_exact_thickness=True, allow_thickness_upsize=False)
    need_thickness = 2.0

    def fake_lookup(material: str, length_mm: float, width_mm: float, thickness_mm: float, qty: int = 1):
        length_in = length_mm / 25.4
        width_in = width_mm / 25.4
        return (
            "WRONGSKU",
            99.0,
            "Each",
            (length_in, width_in, need_thickness + 1.5),
        )

    monkeypatch.setattr(materials._mc, "lookup_sku_and_price_for_mm", fake_lookup)

    result = pick_stock_from_mcmaster(
        "Aluminum MIC6",
        12.0,
        12.0,
        need_thickness,
        cfg=cfg,
    )

    assert result is not None
    assert pytest.approx(need_thickness) == result["thk_in"]
    assert result.get("stock_piece_api_price") is None
    assert result.get("stock_piece_price_usd") is None
    assert result.get("stock_piece_source") is None
