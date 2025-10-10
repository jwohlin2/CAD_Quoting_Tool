from __future__ import annotations

import math
import re

import pytest


def test_die_plate_deep_drill_regression() -> None:
    import appV5

    if not hasattr(appV5, "_canonical_amortized_label"):
        appV5._canonical_amortized_label = lambda label: (str(label), False)
    if not hasattr(appV5, "_fold_buckets"):
        appV5._fold_buckets = lambda mapping, *_args, **_kwargs: mapping
    if not hasattr(appV5, "notes_order"):
        appV5.notes_order = []
    if not hasattr(appV5, "pricing_source_text"):
        appV5.pricing_source_text = ""
    if not hasattr(appV5, "_apply_drilling_per_hole_bounds"):
        def _apply_drilling_per_hole_bounds(
            hours: float,
            *,
            hole_count_hint: int | None = None,
            material_group: str | None = None,
            depth_in: float | None = None,
            **_: object,
        ) -> float:
            hole_count = int(hole_count_hint or 0)
            return appV5._apply_drill_minutes_clamp(
                hours,
                hole_count,
                material_group=material_group,
                depth_in=depth_in,
            )

        appV5._apply_drilling_per_hole_bounds = _apply_drilling_per_hole_bounds

    hole_count = 163
    thickness_in = 2.0
    hole_dia_mm = 13.815
    hole_dia_in = hole_dia_mm / 25.4

    ld_ratio = thickness_in / hole_dia_in
    assert pytest.approx(ld_ratio, rel=1e-6) == ld_ratio
    assert ld_ratio > 3.0, "L/D ratio should trigger the deep drilling cycle"

    breakthrough_in = max(0.04, 0.2 * hole_dia_in)
    depth_for_bounds = thickness_in + breakthrough_in

    min_hours = appV5._apply_drilling_per_hole_bounds(
        1e-6,
        hole_count_hint=hole_count,
        material_group="Stainless Steel",
        depth_in=depth_for_bounds,
    )
    max_hours = appV5._apply_drilling_per_hole_bounds(
        100.0,
        hole_count_hint=hole_count,
        material_group="Stainless Steel",
        depth_in=depth_for_bounds,
    )

    min_per_hole, max_per_hole = appV5._drill_minutes_per_hole_bounds(
        "Stainless Steel",
        depth_in=depth_for_bounds,
    )

    expected_min_hours = (hole_count * min_per_hole) / 60.0
    expected_max_hours = (hole_count * max_per_hole) / 60.0
    clamp_floor_hr = expected_min_hours
    clamp_cap_hr = expected_max_hours

    assert math.isclose(min_hours, expected_min_hours, rel_tol=1e-9)
    assert math.isclose(max_hours, expected_max_hours, rel_tol=1e-9)

    bounded_hours = appV5._apply_drilling_per_hole_bounds(
        4.8,
        hole_count_hint=hole_count,
        material_group="Stainless Steel",
        depth_in=depth_for_bounds,
    )
    assert expected_min_hours <= bounded_hours <= expected_max_hours

    process_cost = bounded_hours * 90.0

    breakdown = {
        "pricing_source": "planner",
        "process_costs": {"drilling": process_cost},
        "process_meta": {
            "drilling": {
                "hr": bounded_hours,
                "basis": [
                    "Planner: Deep_Drill cycle with per-hole clamp",
                ],
            }
        },
        "process_breakdown": {
            "drilling": {
                "hr": bounded_hours,
                "basis": ["Planner clamp bounds"],
                "why": [
                    "Deep drilling cycle triggered by L/D 3.68",
                    f"Clamp keeps drilling time between {clamp_floor_hr:.2f} and {clamp_cap_hr:.2f} hr",
                ],
            }
        },
        "process_hours": {"drilling": bounded_hours},
        "process_minutes": {"drilling": bounded_hours * 60.0},
        "labor_costs": {"Drilling": process_cost},
        "labor_cost_details": {"Drilling": "Bounded by per-hole clamp"},
        "pass_through": {"Consumables": 40.0},
        "direct_cost_details": {"Consumables": "Cutting oil allowance"},
        "applied_pcts": {
            "MarginPct": 0.20,
            "OverheadPct": 0.10,
            "GA_Pct": 0.05,
            "ContingencyPct": 0.00,
        },
        "totals": {
            "subtotal": process_cost + 40.0,
            "labor_cost": process_cost,
            "with_overhead": (process_cost + 40.0) * 1.10,
            "with_ga": (process_cost + 40.0) * 1.10 * 1.05,
            "with_contingency": (process_cost + 40.0) * 1.10 * 1.05,
            "with_expedite": (process_cost + 40.0) * 1.10 * 1.05,
            "price": (process_cost + 40.0) * 1.10 * 1.05 / (1 - 0.20),
        },
        "qty": 1,
        "geo_context": {
            "hole_count": hole_count,
            "hole_diams_mm": [hole_dia_mm] * hole_count,
            "thickness_mm": thickness_in * 25.4,
        },
        "rates": {"labor": {"DrillingRate": 90.0}},
    }

    result = {
        "price": breakdown["totals"]["price"],
        "breakdown": breakdown,
        "narrative": (
            "Deep_Drill selected because L/D ≈ 3.68. "
            f"Clamp bounds drilling hours between {clamp_floor_hr:.2f} and {clamp_cap_hr:.2f}."
        ),
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)

    assert rendered.count("Labor Hour Summary") == 1
    assert rendered.count("Why this price") == 1
    assert "deep_drill" in rendered.lower()


def test_steel_die_plate_deep_drill_runtime_floor() -> None:
    import pandas as pd
    import appV5

    hole_count = 163
    thickness_in = 2.0
    hole_dia_mm = 13.815

    speeds = pd.DataFrame(
        [
            {
                "operation": "Drill",
                "material": "Stainless Steel",
                "sfm_start": 300,
                "fz_ipr_0_125in": 0.0025,
                "fz_ipr_0_25in": 0.0035,
                "fz_ipr_0_5in": 0.0050,
            },
            {
                "operation": "Deep_Drill",
                "material": "Stainless Steel",
                "sfm_start": 75,
                "fz_ipr_0_125in": 0.0010,
                "fz_ipr_0_25in": 0.0015,
                "fz_ipr_0_5in": 0.0020,
            },
        ]
    )

    debug_lines: list[str] = []
    hours = appV5.estimate_drilling_hours(
        [hole_dia_mm] * hole_count,
        thickness_in,
        "Stainless Steel",
        hole_groups=[
            {"dia_mm": hole_dia_mm, "depth_mm": thickness_in * 25.4, "count": hole_count}
        ],
        speeds_feeds_table=speeds,
        debug_lines=debug_lines,
    )

    assert hours >= 3.0, f"Expected conservative deep drill time, got {hours:.2f} hr"
    assert hours == pytest.approx(13.5833333333, rel=1e-6)

    summary = next((line for line in debug_lines if line.startswith("Drill calc")), "")
    assert summary, "Expected deep drill debug summary"
    assert "op=Deep_Drill" in summary
    assert "index" in summary.lower()

    rpm_match = re.search(r"RPM (\d+(?:\.\d+)?)(?:[–-](\d+(?:\.\d+)?))?", summary)
    ipm_match = re.search(r"IPM (\d+(?:\.\d+)?)(?:[–-](\d+(?:\.\d+)?))?", summary)
    index_match = re.search(r"index (\d+(?:\.\d+)?) s/hole", summary)

    assert rpm_match is not None
    assert ipm_match is not None
    assert index_match is not None

    rpm_low = float(rpm_match.group(1))
    rpm_high = float(rpm_match.group(2) or rpm_match.group(1))
    ipm_low = float(ipm_match.group(1))
    ipm_high = float(ipm_match.group(2) or ipm_match.group(1))
    index_seconds = float(index_match.group(1))

    assert rpm_low <= 560 and rpm_high >= 420, f"RPM range {rpm_low:.0f}–{rpm_high:.0f} misses deep drill target"
    assert ipm_high >= 0.8 and ipm_low <= 1.2, f"Feed range {ipm_low:.2f}–{ipm_high:.2f} IPM outside expected deep drill range"
    assert index_seconds >= 120.0, f"Index {index_seconds:.1f}s per hole too low for deep drill"
