from __future__ import annotations

from cad_quoter.pricing.machining_report import render_drilling_section


def _build_overheads() -> dict[str, float]:
    return {
        "index_per_hole_min": 0.08,
        "peck_per_hole_min": 0.02,
        "toolchange_deep_min": 0.5,
        "toolchange_std_min": 0.25,
    }


def test_render_drilling_section_uses_diameter_depth_ranges() -> None:
    groups = [
        {
            "dia": 0.125,
            "qty": 7,
            "depth_in": 0.75,
            "sfm": 120.0,
            "ipr": 0.0125,
            "t_per_hole_min": 0.2,
        },
        {
            "dia": 0.25,
            "qty": 3,
            "depth_in": 0.5,
            "sfm": 150.0,
            "ipr": 0.015,
            "t_per_hole_min": 0.18,
        },
    ]

    section = render_drilling_section(
        material="6061-T6",
        block_thickness=1.0,
        drill_groups=groups,
        overheads=_build_overheads(),
    )

    lines = section.splitlines()

    assert "MATERIAL REMOVAL – DRILLING" in lines[0]
    assert "Deep-Drill (L/D ≥ 3), Drill" in lines[4]
    assert "7 deep + 3 std  = 10" in lines[5]
    assert "Diameter range .... 0.125–0.250\"" in lines[6]
    assert "Depth per hole .... 0.50–0.75 in" in lines[7]


def test_render_drilling_section_handles_all_standard_holes() -> None:
    groups = [
        {
            "dia": 0.5,
            "qty": 4,
            "depth_in": 1.0,
            "sfm": 90.0,
            "ipr": 0.02,
            "t_per_hole_min": 0.25,
        }
    ]

    section = render_drilling_section(
        material="1018",
        block_thickness=2.0,
        drill_groups=groups,
        overheads=_build_overheads(),
    )

    lines = section.splitlines()

    assert "Operations ........ Drill" in lines[4]
    assert "0 deep + 4 std  = 4" in lines[5]
    assert "Diameter range .... 0.500\"" in lines[6]
    assert "Depth per hole .... 1.00 in" in lines[7]


def test_render_drilling_section_prefers_adjusted_counts_when_available() -> None:
    groups = [
        {
            "dia": 0.125,
            "qty": 7,
            "depth_in": 0.75,
            "sfm": 120.0,
            "ipr": 0.0125,
            "t_per_hole_min": 0.2,
            "counts_by_diam": {0.125: 5, 0.25: 1},
        },
        {
            "dia": 0.25,
            "qty": 3,
            "depth_in": 0.5,
            "sfm": 150.0,
            "ipr": 0.015,
            "t_per_hole_min": 0.18,
        },
    ]

    section = render_drilling_section(
        material="6061-T6",
        block_thickness=1.0,
        drill_groups=groups,
        overheads=_build_overheads(),
    )

    lines = section.splitlines()

    assert "5 deep + 1 std  = 6" in lines[5]
    assert 'Dia 0.125" × 5  | depth 0.750"' in section
    assert 'Dia 0.250" × 1  | depth 0.500"' in section
    assert "Subtotal (per-hole × qty) . 1.18 min" in section
