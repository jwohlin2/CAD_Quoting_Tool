"""Aggregated smoke and unit tests for application helpers."""

from __future__ import annotations

import ast
import logging
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

APP_PATH = PROJECT_ROOT / "appV5.py"
TARGET_DEBUG_HELPERS = {"_jsonify_debug_value", "_jsonify_debug_summary"}

_requests_stub = sys.modules.setdefault("requests", types.ModuleType("requests"))
_requests_stub.Session = getattr(
    _requests_stub,
    "Session",
    type("Session", (), {"__init__": lambda self, *args, **kwargs: None}),
)

from appV5 import (  # noqa: E402  # pylint: disable=wrong-import-position
    _aggregate_hole_entries,
    _dedupe_hole_entries,
    _density_for_material,
    _material_family,
    _parse_hole_line,
    classify_concentric,
    hole_count_from_geometry,
    net_mass_kg,
    render_quote,
    summarize_hole_chart_lines,
    hole_rows_to_ops,
)

try:  # noqa: E402  # pylint: disable=wrong-import-position
    from appV5 import default_variables_template
except ImportError:  # pragma: no cover - optional dependency for tests
    default_variables_template = None

from cad_quoter.app.chart_lines import (  # noqa: E402  # pylint: disable=wrong-import-position
    RE_DEPTH,
    RE_TAP,
    build_ops_rows_from_lines_fallback,
)
from cad_quoter.geometry.hole_table_parser import (  # noqa: E402  # pylint: disable=wrong-import-position
    parse_hole_table_lines,
)
from cad_quoter.material_density import density_for_material  # noqa: E402  # pylint: disable=wrong-import-position
from cad_quoter import llm  # noqa: E402  # pylint: disable=wrong-import-position


def test_geometry_service_imports() -> None:
    geometry = pytest.importorskip("cad_quoter.geometry")

    service = geometry.GeometryService()

    assert hasattr(service, "load_model")
    assert isinstance(geometry.HAS_TRIMESH, bool)


class FakeCircle:
    def __init__(self, center: tuple[float, float, float], radius: float, layer: str = "0") -> None:
        self.dxf = SimpleNamespace(center=center, radius=radius, layer=layer)


class FakeSpace:
    def __init__(self, circles: list[FakeCircle] | None = None) -> None:
        self._circles = circles or []

    def query(self, types: str):  # pragma: no cover - trivial wrapper
        if types == "CIRCLE":
            return list(self._circles)
        return []


class FakeLayouts:
    def __init__(self, spaces: dict[str, FakeSpace] | None = None) -> None:
        self._spaces = spaces or {}

    def names_in_taborder(self):  # pragma: no cover - trivial wrapper
        return list(self._spaces)

    def get(self, name: str):  # pragma: no cover - trivial wrapper
        return SimpleNamespace(entity_space=self._spaces[name])


class FakeDoc:
    def __init__(
        self,
        model_circles: list[FakeCircle] | None = None,
        layout_circles: list[list[FakeCircle]] | None = None,
    ) -> None:
        self._model = FakeSpace(model_circles)
        layouts = {
            f"Layout{i}": FakeSpace(circles)
            for i, circles in enumerate(layout_circles or [])
        }
        self.layouts = FakeLayouts(layouts)

    def modelspace(self):  # pragma: no cover - trivial wrapper
        return self._model


@pytest.mark.parametrize(
    "model_circles, layout_circles, expected_count, expected_families",
    [
        (
            [FakeCircle((0.0, 0.0, 0.0), 0.25)],
            [[FakeCircle((10.0, 5.0, 0.0), 0.25)]],
            1,
            {0.5: 1},
        ),
        (
            [],
            [[FakeCircle((0.0, 0.0, 0.0), 0.2), FakeCircle((1.0, 0.0, 0.0), 0.2)]],
            2,
            {0.4: 2},
        ),
    ],
)
def test_hole_count_handles_model_and_layout_views(
    model_circles: list[FakeCircle],
    layout_circles: list[list[FakeCircle]],
    expected_count: int,
    expected_families: dict[float, int],
) -> None:
    doc = FakeDoc(model_circles=model_circles, layout_circles=layout_circles)

    count, families = hole_count_from_geometry(doc, to_in=1.0)

    assert count == expected_count
    assert families == expected_families


def test_classify_concentric_ignores_duplicate_layout_views() -> None:
    doc = FakeDoc(
        model_circles=[
            FakeCircle((0.0, 0.0, 0.0), 0.5),
            FakeCircle((0.0, 0.0, 0.0), 0.65),
        ],
        layout_circles=[[FakeCircle((10.0, 10.0, 0.0), 0.5), FakeCircle((10.0, 10.0, 0.0), 0.65)]],
    )

    result = classify_concentric(doc, to_in=1.0)

    total_pairs = result.get("cbore_pairs_geom", 0) + result.get("csk_pairs_geom", 0)
    assert total_pairs == 1


def test_net_mass_kg_uses_copper_density() -> None:
    density_lookup = density_for_material("Copper")
    assert math.isclose(density_lookup, 8.96, rel_tol=0.02)
    density = _density_for_material("Copper")
    assert math.isclose(density, density_lookup, rel_tol=0.0, abs_tol=1e-6)

    length_in = 4.0
    width_in = 3.0
    thickness_in = 0.5
    expected_mass = (length_in * width_in * thickness_in * 16.387064 * density) / 1000.0

    mass = net_mass_kg(length_in, width_in, thickness_in, [], density)
    assert math.isclose(mass, expected_mass, rel_tol=1e-9)

    assert _material_family("Copper") == "copper"


def test_net_mass_kg_optionally_returns_removed_mass() -> None:
    density_lookup = density_for_material("Aluminum")
    density = _density_for_material("Aluminum")
    assert math.isclose(density, density_lookup, rel_tol=0.0, abs_tol=1e-6)
    length_in = 6.0
    width_in = 2.0
    thickness_in = 0.5
    hole_diam_mm = 12.7  # 0.5 in

    net_mass, removed_mass = net_mass_kg(
        length_in,
        width_in,
        thickness_in,
        [hole_diam_mm],
        density,
        return_removed_mass=True,
    )

    assert net_mass is not None
    assert removed_mass is not None

    volume_plate_in3 = length_in * width_in * thickness_in
    radius_mm = hole_diam_mm / 2.0
    height_mm = thickness_in * 25.4
    hole_volume_mm3 = math.pi * (radius_mm**2) * height_mm
    hole_volume_in3 = hole_volume_mm3 / 16387.064
    expected_removed_g = hole_volume_in3 * 16.387064 * density

    assert removed_mass == pytest.approx(expected_removed_g, rel=1e-9)
    expected_net_mass = (volume_plate_in3 - hole_volume_in3) * 16.387064 * density / 1000.0
    assert net_mass == pytest.approx(expected_net_mass, rel=1e-9)


@pytest.mark.parametrize(
    "hole_line, source, expected",
    [
        ("TAP THRU (FROM BACK)", "LEADER", {"side": "BACK", "from_back": True}),
        ("0.250 C'BORE FRONT & BACK", "LEADER", {"double_sided": True}),
    ],
)
def test_parse_hole_line_extracts_side_information(hole_line: str, source: str, expected: dict[str, object]) -> None:
    entry = _parse_hole_line(hole_line, 1.0, source=source)
    assert entry is not None
    for key, value in expected.items():
        assert entry.get(key) == value


def test_leader_entries_duplicate_table_rows_are_ignored() -> None:
    table_entry = _parse_hole_line("QTY 8 Ø0.201 THRU", 1.0, source="TABLE")
    assert table_entry is not None
    leader_entry = _parse_hole_line("qty 8 0.201 thru", 1.0, source="LEADER")
    assert leader_entry is not None

    unique_leaders = _dedupe_hole_entries([table_entry], [leader_entry])

    assert unique_leaders == []


def test_leader_entries_with_new_information_are_preserved() -> None:
    table_entry = _parse_hole_line("QTY 4 1/4-20 TAP", 1.0, source="TABLE")
    assert table_entry is not None
    leader_entry = _parse_hole_line("QTY 2 C'BORE .500", 1.0, source="LEADER")
    assert leader_entry is not None

    unique_leaders = _dedupe_hole_entries([table_entry], [leader_entry])

    assert len(unique_leaders) == 1
    agg = _aggregate_hole_entries([table_entry] + unique_leaders)

    assert agg["hole_count"] == 6
    assert agg["cbore_qty"] == 2


def test_aggregate_flags_back_side_from_hint() -> None:
    entry = _parse_hole_line("QTY 2 TAP THRU (FROM BACK)", 1.0, source="LEADER")
    assert entry is not None
    agg = _aggregate_hole_entries([entry])
    assert agg["from_back"] is True


def test_chart_summary_marks_back_side_without_depth() -> None:
    summary = summarize_hole_chart_lines(["TAP THRU (FROM BACK)"])
    assert summary["from_back"] is True


def _find_feature(row, feature_type):
    for feature in row.features:
        if feature.get("type") == feature_type:
            return feature
    raise AssertionError(f"Feature {feature_type!r} not found in row {row.ref}")


def test_parse_hole_table_handles_leading_decimal_tokens() -> None:
    lines = [
        "HOLE   REF Ø   QTY   DESCRIPTION",
        "A      Ø.7500  2     Ø.7500 ±.0001 THRU (JIG GRIND); 1.78Ø C'BORE X .38 DEEP FROM BACK",
        "C              4     1/4-20 TAP (JIG GRIND); .623 C'BORE X .62 DEEP FROM FRONT",
    ]

    rows = parse_hole_table_lines(lines)
    assert len(rows) == 2

    row_a = next(row for row in rows if row.ref == "A")
    drill = _find_feature(row_a, "drill")
    assert math.isclose(drill["dia_mm"], 0.75 * 25.4, rel_tol=1e-6)
    assert drill["thru"] is True
    assert drill["from_face"] == "back"

    cbore_a = _find_feature(row_a, "cbore")
    assert math.isclose(cbore_a["depth_mm"], 0.38 * 25.4, rel_tol=1e-6)
    assert math.isclose(cbore_a["dia_mm"], 1.78 * 25.4, rel_tol=1e-6)

    row_c = next(row for row in rows if row.ref == "C")
    cbore_c = _find_feature(row_c, "cbore")
    assert math.isclose(cbore_c["depth_mm"], 0.62 * 25.4, rel_tol=1e-6)


def test_parser_rules_v2_to_ops_matches_spec(caplog: pytest.LogCaptureFixture) -> None:
    lines = [
        "HOLE   REF Ø   QTY   DESCRIPTION",
        "K      Ø.201  8     Ø.201 THRU",
        "L      #7     4     1/4-20 TAP .50 DEEP FROM BACK",
        "C      Ø1.000 2     Ø1.000 C'BORE X .25 DEEP FROM FRONT",
        "D      Ø1.000 2     Ø1.000 C'BORE X .25 DEEP FROM BACK",
        "H              3     Ø.250 SPOT DRILL",
        "B      Ø.159  6     #21 10-32 TAP THRU",
    ]

    with caplog.at_level(logging.INFO):
        rows = parse_hole_table_lines(lines, rules_v2=True, block_thickness_in=1.0)

    assert len(rows) == 6

    ops = hole_rows_to_ops(rows)
    assert len(ops) == 10

    ops_by_key = {(op["ref"], op["type"], op["side"]): op for op in ops}

    drill_k = ops_by_key[("K", "drill", "front")]
    assert drill_k["qty"] == 8
    assert drill_k["thread"] is None
    assert drill_k["depth_in"] == pytest.approx(1.05, rel=1e-6)
    assert drill_k["ref_dia"] == pytest.approx(0.201, rel=1e-6)

    drill_l = ops_by_key[("L", "drill", "back")]
    assert drill_l["qty"] == 4
    assert drill_l["depth_in"] == pytest.approx(0.5, rel=1e-6)
    assert drill_l["ref_dia"] == pytest.approx(0.201, rel=1e-6)

    tap_l = ops_by_key[("L", "tap", "back")]
    assert tap_l["qty"] == 4
    assert tap_l["thread"] == "1/4-20"
    assert tap_l["depth_in"] == pytest.approx(0.5, rel=1e-6)
    assert tap_l["ref_dia"] == pytest.approx(0.25, rel=1e-6)

    cbore_c = ops_by_key[("C", "cbore", "front")]
    assert cbore_c["qty"] == 2
    assert cbore_c["depth_in"] == pytest.approx(0.25, rel=1e-6)
    assert cbore_c["ref_dia"] == pytest.approx(1.0, rel=1e-6)

    drill_c = ops_by_key[("C", "drill", "front")]
    assert drill_c["qty"] == 2
    assert drill_c["depth_in"] is None
    assert drill_c["ref_dia"] == pytest.approx(1.0, rel=1e-6)

    cbore_d = ops_by_key[("D", "cbore", "back")]
    assert cbore_d["qty"] == 2
    assert cbore_d["depth_in"] == pytest.approx(0.25, rel=1e-6)

    drill_d = ops_by_key[("D", "drill", "back")]
    assert drill_d["qty"] == 2
    assert drill_d["depth_in"] is None
    assert drill_d["ref_dia"] == pytest.approx(1.0, rel=1e-6)

    spot_h = ops_by_key[("H", "spot", "front")]
    assert spot_h["qty"] == 3
    assert spot_h["ref_dia"] == pytest.approx(0.25, rel=1e-6)
    assert spot_h["depth_in"] is None

    drill_b = ops_by_key[("B", "drill", "front")]
    assert drill_b["qty"] == 6
    assert drill_b["depth_in"] == pytest.approx(1.05, rel=1e-6)
    assert drill_b["ref_dia"] == pytest.approx(0.159, rel=1e-6)

    tap_b = ops_by_key[("B", "tap", "front")]
    assert tap_b["qty"] == 6
    assert tap_b["thread"] == "10-32"
    assert tap_b["depth_in"] == pytest.approx(1.05, rel=1e-6)

    assert any("[rules]" in record.getMessage() for record in caplog.records)


def test_build_ops_rows_from_lines_fallback_extracts_common_ops() -> None:
    lines = [
        "(2) 1/4-20 TAP",
        "THRU",
        "0.25 DEEP FROM BACK",
        "(3) COUNTERBORE Ø0.750",
        "X 0.25 DEEP FROM FRONT",
        "3/8 - NPT",
    ]

    rows = build_ops_rows_from_lines_fallback(lines)

    assert len(rows) == 3

    tap_row, cbore_row, npt_row = rows

    assert tap_row == {
        "hole": "",
        "ref": "",
        "qty": 2,
        "desc": '1/4-20 TAP THRU × 0.25" FROM BACK',
        "t_per_hole_min": 0.085,
        "feed_fmt": "0.0500 ipr | 917 rpm | 45.840 ipm",
    }

    assert cbore_row == {
        "hole": "",
        "ref": "",
        "qty": 3,
        "desc": '0.7500 C’BORE × 0.25" FROM FRONT',
        "t_per_hole_min": 0.158,
        "feed_fmt": "0.0050 ipr | 764 rpm | 3.820 ipm",
    }

    assert npt_row == {"hole": "", "ref": "", "qty": 1, "desc": "3/8 - NPT"}


@pytest.mark.parametrize(
    "text, thread",
    [
        ("(4) #10-32 TAP", "#10-32"),
        ("1/4-20 TAP", "1/4-20"),
    ],
)
def test_re_tap_exposes_thread_spec(text: str, thread: str) -> None:
    match = RE_TAP.search(text)
    assert match is not None
    assert match.group(2).replace(" ", "") == thread


def test_re_depth_captures_numeric_and_side() -> None:
    match = RE_DEPTH.search("0.50 DEEP FROM BACK")
    assert match is not None
    assert match.group(1) == "0.50"
    assert match.group(2) == "BACK"


def _load_module_ast() -> ast.AST:
    source = APP_PATH.read_text(encoding="utf-8")
    if source.startswith("\ufeff"):
        source = source.lstrip("\ufeff")
    return ast.parse(source, filename=str(APP_PATH))


def test_single_debug_helper_definitions() -> None:
    module = _load_module_ast()
    counts = {name: 0 for name in TARGET_DEBUG_HELPERS}
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in counts:
            counts[node.name] += 1
    assert counts == {name: 1 for name in TARGET_DEBUG_HELPERS}


def test_no_dynamic_debug_helper_lookups() -> None:
    module = _load_module_ast()

    def _is_dynamic_lookup(call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "get":
            base = func.value
            if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
                if base.func.id in {"globals", "locals"}:
                    if call.args and isinstance(call.args[0], ast.Constant):
                        return call.args[0].value in TARGET_DEBUG_HELPERS
        if isinstance(func, ast.Name) and func.id == "getattr":
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
                return call.args[1].value in TARGET_DEBUG_HELPERS
        return False

    dynamic_calls: list[ast.Call] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and _is_dynamic_lookup(node):
            dynamic_calls.append(node)

    assert dynamic_calls == []


def test_render_quote_sanitizes_special_characters() -> None:
    data = {
        "summary": {
            "qty": 2,
            "final_price": 100,
            "unit_price": 50,
            "subtotal_before_margin": 80,
            "margin_pct": 0.25,
        },
        "price_drivers": [
            {
                "label": "Cycle – roughing",
                "detail": "Contains\tcolor \x1b[31mred\x1b[0m text",
            }
        ],
        "cost_breakdown": {"Labor": 60, "Material": 20},
    }

    rendered = render_quote(data)

    assert "\t" not in rendered
    assert "\x1b" not in rendered
    allowed_unicode = {"×", "–", "≥", "≤"}
    assert all(ord(ch) < 128 or ch in allowed_unicode for ch in rendered)
    assert "Cycle – roughing" in rendered
    assert "color red text" in rendered


@pytest.mark.parametrize(
    "value, default, expected",
    [
        (True, False, True),
        (0, True, False),
        (1.0, False, True),
        (float("nan"), True, True),
        ("Yes", False, True),
        ("no", True, False),
        (" y ", False, True),
        ("N", True, False),
        ("Yes / No", False, True),
        ("Off / On", True, False),
        ("maybe", True, True),
        ("", False, False),
    ],
)
def test_coerce_checkbox_state_handles_common_inputs(value, default, expected):
    from appkit.ui.editor_controls import coerce_checkbox_state

    assert coerce_checkbox_state(value, default) is expected


def test_editor_control_helpers_detect_dropdowns() -> None:
    from appkit.ui.editor_controls import derive_editor_control_spec

    spec = derive_editor_control_spec("Dropdown", "Auto, Manual, Override")
    assert spec.control == "dropdown"
    assert spec.options == ("Auto", "Manual", "Override")
    assert spec.entry_value == "Auto"
    assert not spec.guessed_dropdown


def test_editor_control_helpers_detect_numbers() -> None:
    from appkit.ui.editor_controls import derive_editor_control_spec

    spec = derive_editor_control_spec("Number", "0.25")
    assert spec.control == "number"
    assert spec.entry_value == "0.25"
    assert spec.display_label.lower() == "number"


def test_editor_control_helpers_detect_formulas() -> None:
    from appkit.ui.editor_controls import derive_editor_control_spec

    spec = derive_editor_control_spec("Lookup Value (Percentage)", "MachiningCost * 0.04")
    assert spec.control == "formula"
    assert spec.entry_value == ""
    assert spec.base_text == "MachiningCost * 0.04"


def test_editor_control_helpers_preserve_formula_defaults() -> None:
    from appkit.ui.editor_controls import derive_editor_control_spec

    spec = derive_editor_control_spec("Lookup Value (Rate)", "$80.00")
    assert spec.control == "formula"
    assert spec.entry_value == "80"
    assert spec.base_text == "$80.00"


def test_editor_control_helpers_promote_dropdowns() -> None:
    from appkit.ui.editor_controls import derive_editor_control_spec

    spec = derive_editor_control_spec("", "Low, Medium, High")
    assert spec.control == "dropdown"
    assert spec.guessed_dropdown
    assert spec.options == ("Low", "Medium", "High")


@pytest.mark.skipif(default_variables_template is None, reason="default template unavailable")
def test_default_template_flags_render_as_checkboxes() -> None:
    from appkit.ui.editor_controls import derive_editor_control_spec

    df = default_variables_template()
    for item in ("FAIR Required", "Source Inspection Requirement"):
        row = next((row for _, row in df.iterrows() if row["Item"] == item), None)
        assert row is not None, f"Missing {item} in default template"
        dtype = row["Data Type / Input Method"]
        example = row["Example Values / Options"]
        spec = derive_editor_control_spec(dtype, example)
        assert spec.control == "checkbox"
        assert spec.checkbox_state is False


def test_explain_quote_notes_drilling_from_plan_info() -> None:
    breakdown = {
        "totals": {"price": 100, "qty": 1},
        "process_costs": {},
    }
    plan_info = {
        "bucket_state_extra": {"drill_total_minutes": 18.0},
        "bucket_view": {"buckets": {"drilling": {"total$": 120.0}}},
    }

    explanation = llm.explain_quote(breakdown, plan_info=plan_info)

    assert "Main cost drivers: Drilling $120.00." in explanation
    assert "Main cost drivers derive from planner buckets; none dominate." not in explanation


def test_explain_quote_ignores_non_numeric_drilling_minutes() -> None:
    breakdown = {"totals": {"price": 120}}
    render_state = {"extra": {"removal_drilling_hours": "NaN"}}

    explanation = llm.explain_quote(breakdown, render_state=render_state)

    assert "Main cost drivers derive from planner buckets; none dominate." in explanation
    assert "Main cost drivers" not in explanation


def test_explain_quote_uses_qty_string_for_piece_label() -> None:
    breakdown = {"totals": {"price": 250, "qty": "3"}}

    explanation = llm.explain_quote(breakdown)

    assert "for 3 pieces" in explanation
