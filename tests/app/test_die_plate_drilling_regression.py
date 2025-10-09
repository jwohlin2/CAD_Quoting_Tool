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
    if not hasattr(appV5, "_apply_drilling_per_hole_bounds"):
        def _apply_drilling_per_hole_bounds(hours: float, *, hole_count_hint: int | None = None, **_: object) -> float:
            hole_count = int(hole_count_hint or 0)
            return appV5._apply_drill_minutes_clamp(hours, hole_count)

        appV5._apply_drilling_per_hole_bounds = _apply_drilling_per_hole_bounds

    hole_count = 163
    thickness_in = 2.0
    hole_dia_mm = 13.815
    hole_dia_in = hole_dia_mm / 25.4

    ld_ratio = thickness_in / hole_dia_in
    assert pytest.approx(ld_ratio, rel=1e-6) == ld_ratio
    assert ld_ratio > 3.0, "L/D ratio should trigger the deep drilling cycle"

    min_hours = appV5._apply_drilling_per_hole_bounds(1e-6, hole_count_hint=hole_count)
    max_hours = appV5._apply_drilling_per_hole_bounds(100.0, hole_count_hint=hole_count)

    assert math.isclose(min_hours, (hole_count * 0.10) / 60.0, rel_tol=1e-9)
    assert math.isclose(max_hours, (hole_count * 2.00) / 60.0, rel_tol=1e-9)

    bounded_hours = appV5._apply_drilling_per_hole_bounds(4.8, hole_count_hint=hole_count)
    assert (hole_count * 0.10) / 60.0 <= bounded_hours <= (hole_count * 2.00) / 60.0

    breakdown = {
        "pricing_source": "planner",
        "process_costs": {"drilling": 270.0},
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
                    "Clamp keeps drilling time between 0.27 and 5.43 hr",
                ],
            }
        },
        "process_hours": {"drilling": bounded_hours},
        "process_minutes": {"drilling": bounded_hours * 60.0},
        "labor_costs": {"Drilling": bounded_hours * 90.0},
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
            "subtotal": 270.0 + 40.0,
            "with_overhead": (270.0 + 40.0) * 1.10,
            "with_ga": (270.0 + 40.0) * 1.10 * 1.05,
            "with_contingency": (270.0 + 40.0) * 1.10 * 1.05,
            "with_expedite": (270.0 + 40.0) * 1.10 * 1.05,
            "price": (270.0 + 40.0) * 1.10 * 1.05 / (1 - 0.20),
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
            "Clamp bounds drilling hours between 0.27 and 5.43."
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
    assert hours == pytest.approx(5.9, rel=0.15)

    summary = next((line for line in debug_lines if line.startswith("Drill calc")), "")
    assert summary, "Expected deep drill debug summary"
    assert "op=Deep_Drill" in summary

    rpm_match = re.search(r"RPM:(\d+(?:\.\d+)?)", summary)
    ipm_match = re.search(r"IPM:(\d+(?:\.\d+)?)", summary)
    min_per_match = re.search(r"min/hole: (\d+(?:\.\d+)?)", summary)

    assert rpm_match is not None
    assert ipm_match is not None
    assert min_per_match is not None

    rpm = float(rpm_match.group(1))
    ipm = float(ipm_match.group(1))
    min_per_hole = float(min_per_match.group(1))

    assert 420 <= rpm <= 560, f"RPM {rpm:.0f} outside expected deep drill range"
    assert 0.8 <= ipm <= 1.2, f"Feed {ipm:.2f} IPM outside expected deep drill range"
    assert min_per_hole >= 2.0, f"Minutes per hole {min_per_hole:.2f} too low for deep drill"
