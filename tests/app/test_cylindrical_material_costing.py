"""Tests for cylindrical parts material costing (guide posts, round punches, etc.)."""
from __future__ import annotations

import importlib.machinery
import sys
import types
import pytest

# Mock dependencies
for _module_name in ("requests", "bs4", "lxml"):
    if _module_name not in sys.modules:
        stub = types.ModuleType(_module_name)
        stub.__spec__ = importlib.machinery.ModuleSpec(_module_name, loader=None)
        sys.modules[_module_name] = stub

import cad_quoter.pricing.materials as materials


def test_cylindrical_guide_post_uses_bar_stock(monkeypatch):
    """Test that a cylindrical guide post uses bar stock, not plate stock."""
    # Mock price resolver
    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        return 5.0, "test_resolver"

    monkeypatch.setattr(materials, "_resolve_price_per_lb", _fake_resolver, raising=False)

    # Geo context for a guide post (cylindrical, round)
    geo_ctx = {
        "material_display": "Aluminum MIC6",
        "is_cylindrical": True,
        "shape_type": "round",
        "family": "guide_post",
        "diameter_in": 3.0,  # 3" diameter guide post
        "length_in": 8.0,    # 8" length
    }

    block = materials._compute_material_block(geo_ctx, "aluminum", 2.70, 0.1)

    # Verify it used cylindrical stock
    assert block.get("is_cylindrical") == True
    assert block["stock_diam_in"] is not None
    assert block["stock_L_in"] is not None

    # Verify it calculated using cylindrical volume (π×r²×L), not plate (L×W×T)
    # For a 3" diameter × 8" cylinder with allowances:
    # - Diameter: 3.0 + 0.125 = 3.125" → rounds to 3.125"
    # - Length: 8.0 + 0.5 = 8.5" → rounds to 12"
    # Volume = π × (3.125/2)² × 12 ≈ 91.97 in³
    # Weight @ 2.70 g/cc = 91.97 × 2.70 × 0.0361273 ≈ 8.98 lb

    start_lb = block["start_lb"]
    assert start_lb > 0, "Start weight should be > 0"
    assert start_lb < 15.0, "Cylindrical bar should be < 15 lb (not a 22 lb plate!)"

    # Verify scrap is reasonable (not 57% like the bug)
    scrap_pct_actual = block["scrap_lb"] / start_lb if start_lb > 0 else 0
    assert scrap_pct_actual < 0.40, f"Scrap should be < 40%, got {scrap_pct_actual:.1%}"


def test_square_form_punch_uses_plate_stock(monkeypatch):
    """Test that a square form punch uses plate stock, not bar stock."""
    # Mock price resolver
    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        return 5.0, "test_resolver"

    monkeypatch.setattr(materials, "_resolve_price_per_lb", _fake_resolver, raising=False)

    # Geo context for a square form punch (NOT cylindrical, rectangular)
    geo_ctx = {
        "material_display": "Aluminum MIC6",
        "is_cylindrical": False,  # Square punch is not cylindrical
        "shape_type": "rectangular",
        "family": "form_punch",
        "thickness_in": 2.0,
        "outline_bbox": {"plate_len_in": 4.0, "plate_wid_in": 4.0},
    }

    block = materials._compute_material_block(geo_ctx, "aluminum", 2.70, 0.1)

    # Verify it used plate stock (NOT cylindrical)
    assert block.get("is_cylindrical") is None or block.get("is_cylindrical") == False
    assert block["stock_L_in"] is not None
    assert block["stock_W_in"] is not None
    assert block["stock_T_in"] is not None

    # Verify it calculated using plate volume (L×W×T)
    start_lb = block["start_lb"]
    assert start_lb > 0


def test_round_punch_round_shape_uses_bar_stock(monkeypatch):
    """Test that a round punch with round shape uses bar stock."""
    # Mock price resolver
    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        return 5.0, "test_resolver"

    monkeypatch.setattr(materials, "_resolve_price_per_lb", _fake_resolver, raising=False)

    # Geo context for a round punch (cylindrical AND round)
    geo_ctx = {
        "material_display": "A2 Tool Steel",
        "is_cylindrical": True,
        "shape_type": "round",
        "family": "round_punch",
        "diameter_in": 1.5,  # 1.5" diameter
        "length_in": 6.0,    # 6" length
    }

    block = materials._compute_material_block(geo_ctx, "A2", 7.85, 0.15)

    # Verify it used cylindrical stock
    assert block.get("is_cylindrical") == True
    assert block["stock_diam_in"] is not None
    assert block["stock_L_in"] is not None

    start_lb = block["start_lb"]
    assert start_lb > 0
    assert start_lb < 10.0, "Small round punch bar should be < 10 lb"


def test_cylindrical_part_scrap_calculation():
    """Test that cylindrical parts have realistic scrap percentages."""
    geo_ctx = {
        "material_display": "Aluminum MIC6",
        "is_cylindrical": True,
        "shape_type": "round",
        "family": "guide_post",
        "diameter_in": 3.0,
        "length_in": 8.0,
    }

    block = materials._compute_material_block(geo_ctx, "aluminum", 2.70, 0.10)

    # Calculate actual scrap percentage
    start_lb = block["start_lb"]
    scrap_lb = block["scrap_lb"]
    scrap_pct = (scrap_lb / start_lb) * 100 if start_lb > 0 else 0

    # Scrap should be reasonable for bar stock (not 57% like the bug)
    assert scrap_pct < 40, f"Scrap percentage {scrap_pct:.1f}% is too high (bug: using plate logic)"
    assert scrap_pct >= 10, f"Scrap percentage {scrap_pct:.1f}% should include material removal"


def test_cylindrical_part_without_shape_fallsback_to_plate():
    """Test that parts marked cylindrical but without shape info fall back gracefully."""
    geo_ctx = {
        "material_display": "Steel",
        "is_cylindrical": True,
        # Missing shape_type - should fail cylindrical check and use plate
        "thickness_in": 1.0,
        "outline_bbox": {"plate_len_in": 3.0, "plate_wid_in": 3.0},
    }

    block = materials._compute_material_block(geo_ctx, "steel", 7.85, 0.1)

    # Should fall back to plate logic
    assert block["start_lb"] > 0
