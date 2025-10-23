from __future__ import annotations

import importlib
import math
import re
import sys
import types
from dataclasses import dataclass
from typing import Callable, Iterable, Union

import pytest


CaseRunner = Callable[[pytest.FixtureRequest], None]


@dataclass
class Case:
    name: str
    run: CaseRunner


# ----- helpers reused across domains -----


def _ensure_geometry_stubs() -> None:
    if "cad_quoter.geometry" not in sys.modules:
        geom_stub = types.ModuleType("cad_quoter.geometry")
        sys.modules["cad_quoter.geometry"] = geom_stub
    else:  # pragma: no cover - already imported
        geom_stub = sys.modules["cad_quoter.geometry"]

    setattr(geom_stub, "FACE_OF", getattr(geom_stub, "FACE_OF", {}))
    setattr(geom_stub, "ensure_face", getattr(geom_stub, "ensure_face", lambda face, *_a, **_kw: face))
    setattr(geom_stub, "face_surface", getattr(geom_stub, "face_surface", lambda *_a, **_kw: None))
    setattr(geom_stub, "iter_faces", getattr(geom_stub, "iter_faces", lambda *_a, **_kw: iter(())))
    setattr(geom_stub, "linear_properties", getattr(geom_stub, "linear_properties", lambda *_a, **_kw: {}))
    setattr(
        geom_stub,
        "map_shapes_and_ancestors",
        getattr(geom_stub, "map_shapes_and_ancestors", lambda *_a, **_kw: {}),
    )

    if "cad_quoter.geometry.dxf_enrich" not in sys.modules:
        enrich_stub = types.ModuleType("cad_quoter.geometry.dxf_enrich")
        enrich_stub.detect_units_scale = lambda *_a, **_kw: (1.0, "inch")
        enrich_stub.iter_spaces = lambda *_a, **_kw: []
        enrich_stub.iter_table_entities = lambda *_a, **_kw: []
        enrich_stub.iter_table_text = lambda *_a, **_kw: []
        sys.modules["cad_quoter.geometry.dxf_enrich"] = enrich_stub


_ensure_geometry_stubs()


from cad_quoter.utils import sheet_helpers

TIME_RE = re.compile(r"\b(?:hours?|hrs?|hr|time|min(?:ute)?s?)\b", re.IGNORECASE)
MONEY_RE = re.compile(r"(?:rate|/hr|per\s*hour|per\s*hr|price|cost|\$)", re.IGNORECASE)


def _build_series(values: list[str]) -> list[str]:
    return list(values)


def _list_sum_time_from_sequence(
    items: list[str],
    values: list[str],
    data_types: list[str],
    mask: list[bool],
    *,
    default: float,
    exclude_mask: list[bool] | None = None,
) -> float:
    matched_indices: list[int] = []
    for idx, include in enumerate(mask):
        if not include:
            continue
        item_text = str(items[idx] or "")
        if not TIME_RE.search(item_text):
            continue
        type_text = str(data_types[idx] or "")
        looks_money = bool(MONEY_RE.search(item_text))
        typed_money = bool(re.search(r"(?:rate|currency|price|cost)", type_text, re.IGNORECASE))
        if exclude_mask is not None and idx < len(exclude_mask) and exclude_mask[idx]:
            continue
        if looks_money or typed_money:
            continue
        matched_indices.append(idx)

    if not matched_indices:
        return float(default)

    total = 0.0
    found_numeric = False
    for idx in matched_indices:
        try:
            value = float(values[idx])
        except Exception:
            continue
        found_numeric = True
        item_text = str(items[idx] or "")
        if re.search(r"\bmin(?:ute)?s?\b", item_text, re.IGNORECASE):
            total += value / 60.0
        else:
            total += value

    if not found_numeric:
        return float(default)
    return float(total)


def _sum_time(
    items: list[str],
    values: list[str],
    types: list[str],
    pattern: str,
    *,
    default: float,
) -> float:
    regex = re.compile(pattern, re.IGNORECASE)

    def matcher(sequence: list[str], _pat: str) -> list[bool]:
        return [bool(regex.search(str(item) or "")) for item in sequence]

    return sheet_helpers.sum_time(
        items,
        values,
        types,
        pattern,
        matcher=matcher,
        sum_time_func=_list_sum_time_from_sequence,
        default=default,
    )


# ----- Service container domain -----


def _run_service_container_caches(request: pytest.FixtureRequest) -> None:
    from dataclasses import dataclass

    from cad_quoter.app.container import ServiceContainer

    @dataclass
    class DummyEngine:
        cleared: bool = False

        def clear_cache(self) -> None:  # pragma: no cover - behaviour not needed
            self.cleared = True

    factory_calls: list[DummyEngine] = []

    def factory() -> DummyEngine:
        engine = DummyEngine()
        factory_calls.append(engine)
        return engine

    container = ServiceContainer(
        load_params=lambda: {"ok": True},
        load_rates=lambda: {"labor": {"Programmer": 1.0}, "machine": {}},
        pricing_engine_factory=factory,
    )

    first = container.get_pricing_engine()
    second = container.get_pricing_engine()

    assert first is second
    assert len(factory_calls) == 1


def _run_service_container_create(request: pytest.FixtureRequest) -> None:
    from dataclasses import dataclass

    from cad_quoter.app.container import ServiceContainer

    @dataclass
    class DummyEngine:
        cleared: bool = False

        def clear_cache(self) -> None:  # pragma: no cover - behaviour not needed
            self.cleared = True

    container = ServiceContainer(
        load_params=lambda: {},
        load_rates=lambda: {"labor": {}, "machine": {}},
        pricing_engine_factory=DummyEngine,
    )

    assert container.create_pricing_engine() is not container.create_pricing_engine()


@pytest.mark.parametrize(
    "case",
    [
        Case("cache_pricing_engine", _run_service_container_caches),
        Case("create_engine_instances", _run_service_container_create),
    ],
    ids=lambda case: case.name,
)
def test_service_container_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Environment utilities domain -----


def _run_coerce_bool_tokens(request: pytest.FixtureRequest) -> None:
    from cad_quoter.utils import coerce_bool

    for token, expected in [
        (True, True),
        (False, False),
        ("t", True),
        ("on", True),
        ("f", False),
        ("off", False),
    ]:
        assert coerce_bool(token) is expected


def _run_coerce_env_bool_tokens(request: pytest.FixtureRequest) -> None:
    module = importlib.import_module("cad_quoter.app.env_flags")
    importlib.reload(module)

    for value, expected in [
        ("true", True),
        ("t", True),
        ("1", True),
        ("false", False),
        ("f", False),
        ("0", False),
        (None, False),
        ("", False),
        ("maybe", False),
    ]:
        assert module._coerce_env_bool(value) is expected


@pytest.mark.parametrize(
    "case",
    [
        Case("coerce_bool_tokens", _run_coerce_bool_tokens),
        Case("coerce_env_bool_tokens", _run_coerce_env_bool_tokens),
    ],
    ids=lambda case: case.name,
)
def test_environment_utilities(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Vendor utility alignment domain -----


def _run_vendor_lead_time_alignment(_: pytest.FixtureRequest) -> None:
    from cad_quoter.pricing import vendor_lead_times, vendor_utils

    for raw in ["3-5 days", "rush", 10, None]:
        assert vendor_utils.coerce_lead_time_days(raw) == vendor_lead_times.coerce_lead_time_days(raw)


def _run_vendor_lead_time_adjustment(_: pytest.FixtureRequest) -> None:
    from cad_quoter.pricing import vendor_lead_times, vendor_utils

    for base, includes_weekends, rush in [
        (10, False, False),
        (10, True, True),
        ("2 weeks", True, False),
        (None, False, False),
    ]:
        assert vendor_utils.apply_lead_time_adjustments(
            base, includes_weekends=includes_weekends, rush=rush
        ) == vendor_lead_times.apply_lead_time_adjustments(
            base, includes_weekends=includes_weekends, rush=rush
        )


@pytest.mark.parametrize(
    "case",
    [
        Case("lead_time_alignment", _run_vendor_lead_time_alignment),
        Case("lead_time_adjustment_alignment", _run_vendor_lead_time_adjustment),
    ],
    ids=lambda case: case.name,
)
def test_vendor_lead_time_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Editor helper domain -----


class DummyVar:
    def __init__(self, value: str | float | int | None = "") -> None:
        self._value = value

    def get(self) -> str | float | int | None:
        return self._value

    def set(self, value: str | float | int | None) -> None:
        self._value = value


def _make_stub_app():
    import appV5

    app = appV5.App.__new__(appV5.App)
    app.editor_vars = {
        "Hole Count (override)": DummyVar(""),
        "Avg Hole Diameter (mm)": DummyVar(""),
    }
    app.editor_value_sources = {}
    app.editor_label_widgets = {}
    app.editor_label_base = {}
    app._editor_set_depth = 0
    app.default_material_display = ""
    return app


def _run_update_material_price(request: pytest.FixtureRequest) -> None:
    import appV5

    monkeypatch = request.getfixturevalue("monkeypatch")

    choice_var = DummyVar("Custom Alloy")
    price_var = DummyVar("")
    material_lookup: dict[str, float] = {}

    monkeypatch.setattr(
        appV5,
        "_resolve_material_unit_price",
        lambda choice, unit="kg": (5.5, "backup_csv:material_price_backup.csv"),
    )

    changed = appV5._update_material_price_field(choice_var, price_var, material_lookup)

    assert changed is True
    assert price_var.get() == "0.0055"


def _run_infer_geo_defaults_basic(_: pytest.FixtureRequest) -> None:
    import appV5

    geo = {
        "plate_len_in": 10.0,
        "plate_wid_in": 5.0,
        "thickness_mm": 12.7,
        "hole_diams_mm": [10.0, 5.0, 5.0],
        "tap_qty": 4,
        "cbore_qty": 2,
        "csk_qty": 1,
        "material": "Aluminum 6061",
        "fai_required": True,
        "setups": 3,
    }

    defaults = appV5.infer_geo_override_defaults(geo)

    assert defaults["Plate Length (in)"] == pytest.approx(10.0)
    assert defaults["Plate Width (in)"] == pytest.approx(5.0)
    assert defaults["Thickness (in)"] == pytest.approx(12.7 / 25.4)
    assert defaults["Number of Milling Setups"] == 3
    assert defaults["Material"] == "Aluminum 6061"
    assert defaults["FAIR Required"] == 1
    assert defaults["Tap Qty (LLM/GEO)"] == 4
    assert defaults["Cbore Qty (LLM/GEO)"] == 2
    assert defaults["Csk Qty (LLM/GEO)"] == 1
    assert defaults["Hole Count (override)"] == 3
    assert defaults["Avg Hole Diameter (mm)"] == pytest.approx((10.0 + 5.0 + 5.0) / 3.0)


def _run_infer_geo_defaults_bins(_: pytest.FixtureRequest) -> None:
    import appV5

    geo = {
        "meta": {"needs_back_face": True},
        "derived": {"hole_bins": {"0.25 in": 4, "6 mm": 2}},
        "scrap_pct": 0.08,
    }

    defaults = appV5.infer_geo_override_defaults(geo)

    expected_avg = ((0.25 * 25.4) * 4 + 6.0 * 2) / 6.0
    assert defaults["Hole Count (override)"] == 6
    assert defaults["Avg Hole Diameter (mm)"] == pytest.approx(expected_avg)
    assert defaults["Number of Milling Setups"] == 2
    assert defaults["Scrap Percent (%)"] == pytest.approx(8.0)


def _run_apply_geo_defaults(request: pytest.FixtureRequest) -> None:
    import appV5

    monkeypatch = request.getfixturevalue("monkeypatch")

    app = _make_stub_app()
    thickness_var = DummyVar("0.0")
    app.editor_vars["Thickness (in)"] = thickness_var

    monkeypatch.setattr(
        appV5,
        "infer_geo_override_defaults",
        lambda geo: {"Thickness (in)": 1.0},
    )

    app._apply_geo_defaults({})

    assert thickness_var.get() == "1.000"
    assert app.editor_value_sources["Thickness (in)"] == "GEO"


@pytest.mark.parametrize(
    "case",
    [
        Case("update_material_price_field", _run_update_material_price),
        Case("infer_geo_defaults_basic", _run_infer_geo_defaults_basic),
        Case("infer_geo_defaults_bins", _run_infer_geo_defaults_bins),
        Case("apply_geo_defaults", _run_apply_geo_defaults),
    ],
    ids=lambda case: case.name,
)
def test_editor_helper_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Overhead compatibility domain -----


def _run_overhead_index_adds_attribute(_: pytest.FixtureRequest) -> None:
    import appV5

    legacy = {
        "toolchange_min": 0.5,
        "approach_retract_in": 0.25,
        "peck_penalty_min_per_in_depth": 0.03,
        "dwell_min": None,
        "peck_min": None,
        "index_sec_per_hole": "18.0",
    }

    compat = appV5._coerce_overhead_dataclass(legacy)

    assert hasattr(compat, "index_sec_per_hole")
    assert getattr(compat, "index_sec_per_hole") == 18.0
    assert getattr(compat, "toolchange_min") == legacy["toolchange_min"]


def _run_overhead_index_preserves(_: pytest.FixtureRequest) -> None:
    import appV5

    overhead = appV5._TimeOverheadParams(index_sec_per_hole=12.0)

    compat = appV5._coerce_overhead_dataclass(overhead)

    assert compat is overhead
    assert getattr(compat, "index_sec_per_hole") == 12.0


@pytest.mark.parametrize(
    "case",
    [
        Case("add_index_attribute", _run_overhead_index_adds_attribute),
        Case("preserve_dataclass", _run_overhead_index_preserves),
    ],
    ids=lambda case: case.name,
)
def test_overhead_index_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Variables loading domain -----


def _run_variables_core_columns(_: pytest.FixtureRequest) -> None:
    import pandas as pd

    from appV5 import CORE_COLS, read_variables_file
    from cad_quoter.resources import default_master_variables_csv

    core_df = read_variables_file(str(default_master_variables_csv()))

    assert core_df is not None

    required_columns = ["Item", "Data Type / Input Method", "Example Values / Options"]
    assert all(col in CORE_COLS for col in required_columns)

    items_to_check = {
        "Material Scrap / Remnant Value",
        "Masking Labor for Plating",
        "Final Inspection Labor (Manual)",
    }

    matched_rows = {item: None for item in items_to_check}

    for _, row in core_df.iterrows():
        item = row.get("Item")
        if item in matched_rows and matched_rows[item] is None:
            matched_rows[item] = row

    missing = [item for item, row in matched_rows.items() if row is None]
    assert not missing, f"Missing expected items: {missing}"

    for item, row in matched_rows.items():
        assert row is not None
        for column in required_columns:
            value = row.get(column)
            assert pd.notna(value), f"{column} should not be NA for {item}"
            assert str(value).strip() != "", f"{column} should not be blank for {item}"


@pytest.mark.parametrize("case", [Case("variables_core_columns", _run_variables_core_columns)], ids=lambda c: c.name)
def test_variables_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Material density domain -----


def _run_density_matches_dropdown(_: pytest.FixtureRequest) -> None:
    import appV5
    from cad_quoter.domain_models import MATERIAL_DROPDOWN_OPTIONS
    from cad_quoter.material_density import MATERIAL_DENSITY_G_CC_BY_KEY, density_for_material, normalize_material_key

    for display in [m for m in MATERIAL_DROPDOWN_OPTIONS if "other" not in m.lower()]:
        key = normalize_material_key(display)
        expected = MATERIAL_DENSITY_G_CC_BY_KEY.get(key)
        assert expected is not None, f"Missing density mapping for {display}"
        lookup_density = density_for_material(display)
        assert math.isclose(lookup_density, expected, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(appV5._density_for_material(display), lookup_density, rel_tol=0.0, abs_tol=1e-6)


def _run_density_handles_aliases(_: pytest.FixtureRequest) -> None:
    import appV5
    from cad_quoter.material_density import MATERIAL_DENSITY_G_CC_BY_KEY, density_for_material, normalize_material_key

    for alias, expected_key in [
        ("ti-6al-4v", "titanium"),
        ("c172", "berylium copper"),
        ("phosphor bronze", "phosphor bronze"),
        ("nickel silver", "nickel silver"),
    ]:
        expected = MATERIAL_DENSITY_G_CC_BY_KEY[normalize_material_key(expected_key)]
        lookup_density = density_for_material(alias)
        assert math.isclose(lookup_density, expected, rel_tol=0.0, abs_tol=1e-6)
        assert math.isclose(appV5._density_for_material(alias), lookup_density, rel_tol=0.0, abs_tol=1e-6)


def _run_render_quote_weight(_: pytest.FixtureRequest) -> None:
    from appV5 import render_quote

    breakdown = {
        "totals": {"labor_cost": 0.0, "direct_costs": 0.0},
        "material": {
            "mass_g": 6000.0,
            "mass_g_net": 5800.0,
            "scrap_pct": 0.05,
        },
        "qty": 1,
    }
    result = {"breakdown": breakdown, "price": 0.0, "ui_vars": {}}

    text = render_quote(result)

    assert "Starting Weight: 13 lb 3.6 oz" in text
    assert "Net Weight: 12 lb 12.6 oz" in text
    assert "Scrap Weight: 7.1 oz" in text
    assert "Weight Reference" not in text
    weight_lines = "\n".join(line for line in text.splitlines() if "Weight" in line or "Mass" in line)
    assert " g" not in weight_lines


@pytest.mark.parametrize(
    "case",
    [
        Case("density_matches_dropdown", _run_density_matches_dropdown),
        Case("density_aliases", _run_density_handles_aliases),
        Case("render_quote_weight", _run_render_quote_weight),
    ],
    ids=lambda case: case.name,
)
def test_material_density_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Operations summary domain -----


def _run_ops_summary_rows(_: pytest.FixtureRequest) -> None:
    import appV5

    rows = [
        {
            "hole": "H1",
            "ref": "0.201",
            "qty": "4X",
            "desc": "4X TAP 1/4-20 THRU",
        }
    ]
    ops_entries = [
        appV5._parse_hole_line("QTY 4 0.201 1/4-20 TAP THRU", 1.0, source="TABLE")
    ]

    summary = appV5.aggregate_ops(rows, ops_entries=ops_entries)

    assert summary["rows"] == [
        {"hole": "H1", "ref": "0.201", "qty": 4, "desc": "4X TAP 1/4-20 THRU"}
    ]
    detail_entry = summary["rows_detail"][0]
    assert detail_entry["type"] == "tap"
    assert detail_entry["qty"] == 4
    assert detail_entry["sides"] == ["FRONT"]
    grouped = summary["group_totals"].get("tap")
    assert grouped is not None
    assert grouped["1/4-20"]["FRONT"]["qty"] == 4


@pytest.mark.parametrize("case", [Case("ops_summary_rows", _run_ops_summary_rows)], ids=lambda c: c.name)
def test_ops_summary_case(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Quote operations cards domain -----


def _run_emit_ops_cards_sections(_: pytest.FixtureRequest) -> None:
    import appV5

    lines: list[str] = []
    geo = {
        "ops_summary": {
            "rows": [
                {"qty": 6, "desc": "6X TAP 1/4-20 THRU FROM FRONT", "ref": ""},
                {"qty": 3, "desc": "Ø0.500 CBORE × .250 DEEP FROM BACK", "ref": ""},
                {"qty": 2, "desc": "CENTER DRILL .187 DIA × .060 DEEP", "ref": ""},
                {"qty": 1, "desc": "JIG GRIND Ø.375", "ref": ""},
            ]
        }
    }

    appV5._emit_hole_table_ops_cards(lines, geo=geo, material_group="aluminum", speeds_csv=None)

    joined = "\n".join(lines)

    assert "MATERIAL REMOVAL – TAPPING" in joined

    tap_line = next(
        line for line in lines if line and not line.startswith("[DEBUG]") and "×" in line
    )
    assert ("0.05" in tap_line) or ("0.0500" in tap_line)
    assert "rpm" in tap_line and "ipm" in tap_line

    debug_line = next(line for line in lines if line.startswith("[DEBUG]"))
    assert "tapping_bucket" in debug_line


def _run_emit_ops_cards_bucket_view(_: pytest.FixtureRequest) -> None:
    import appV5

    lines: list[str] = []
    geo = {
        "ops_summary": {
            "rows": [
                {"qty": 3, "desc": "1/4-20 TAP THRU", "ref": ""},
                {"qty": 2, "desc": "Ø0.3125 CBORE × .25 DEEP", "ref": ""},
                {"qty": 4, "desc": "CENTER DRILL .187 DIA × .060 DEEP", "ref": ""},
                {"qty": 1, "desc": "JIG GRIND Ø.375", "ref": ""},
            ],
            "totals": {
                "tap_front": 3,
                "cbore_front": 2,
                "spot_front": 4,
                "jig_grind": 1,
            },
        }
    }
    rates = {
        "TappingRate": 120.0,
        "CounterboreRate": 100.0,
        "DrillingRate": 90.0,
        "GrindingRate": 70.0,
        "LaborRate": 60.0,
    }
    breakdown: dict[str, object] = {"bucket_view": {}, "rates": rates}

    appV5._emit_hole_table_ops_cards(
        lines,
        geo=geo,
        material_group="aluminum",
        speeds_csv=None,
        breakdown=breakdown,
        rates=rates,
    )

    bucket_view = breakdown["bucket_view"]
    assert isinstance(bucket_view, dict)
    buckets = bucket_view.get("buckets", {})
    assert isinstance(buckets, dict)

    tapping = buckets.get("tapping")
    assert isinstance(tapping, dict)
    assert tapping["minutes"] == pytest.approx(0.5)
    assert tapping["machine$"] == pytest.approx(1.0)
    assert tapping["labor$"] == pytest.approx(0.5)
    assert tapping["total$"] == pytest.approx(1.5)

    ops = bucket_view.get("bucket_ops")
    if isinstance(ops, dict):
        assert any(entry.get("name") == "Tapping ops" for entry in ops.get("tapping", []))


def _run_aggregate_ops_sets_built(_: pytest.FixtureRequest) -> None:
    import appV5

    rows = [
        {"hole": "A1", "ref": "Ø0.257", "qty": 4, "desc": "Ø0.257 THRU"},
        {"hole": "A2", "ref": "", "qty": 2, "desc": "1/4-20 TAP THRU"},
    ]

    ops_entries = [
        appV5._parse_hole_line("QTY 4 Ø0.257 THRU", 1.0, source="TABLE"),
        appV5._parse_hole_line("QTY 2 1/4-20 TAP THRU", 1.0, source="TABLE"),
    ]

    summary = appV5.aggregate_ops(rows, ops_entries=ops_entries)

    assert summary.get("built_rows") == 2


def test_parser_rules_v2_ops_summary_and_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    import appV5

    monkeypatch.setenv("PARSER_RULES_V2", "1")

    rows = [
        {"hole": "D1", "ref": "Ø0.531", "qty": 22, "desc": "Ø0.531 THRU"},
        {"hole": "D2", "ref": "Ø0.201", "qty": 61, "desc": "Ø0.201 THRU"},
        {"hole": "D3", "ref": "Ø0.159", "qty": 2, "desc": "Ø0.159 THRU"},
        {"hole": "T1", "ref": "0.201", "qty": 21, "desc": "1/4-20 TAP THRU"},
        {"hole": "C1", "ref": "Ø0.750", "qty": 30, "desc": "Ø0.750 CBORE × 0.25\" FRONT & BACK"},
    ]

    ops_entries = [
        {"type": "drill", "qty": 22, "dia_in": 0.531, "depth_in": 2.0, "ref": "Ø0.531"},
        {"type": "drill", "qty": 61, "dia_in": 0.201, "depth_in": 2.0, "ref": "Ø0.201"},
        {"type": "drill", "qty": 2, "dia_in": 0.159, "depth_in": 2.0, "ref": "Ø0.159"},
        {"type": "tap", "qty": 21, "thread": "1/4-20", "ref": "0.201", "depth_in": 0.5, "side": "FRONT"},
        {
            "type": "cbore",
            "qty": 30,
            "dia_in": 0.75,
            "depth_in": 0.25,
            "double_sided": True,
            "ref": "Ø0.750",
        },
        {"type": "spot", "qty": 10, "depth_in": 0.1, "ref": "Ø0.531"},
    ]

    summary = appV5.aggregate_ops(rows, ops_entries=ops_entries)
    totals = summary["totals"]

    assert totals["drill"] == 85
    assert totals["tap_front"] == 21
    assert totals["cbore_front"] == 30
    assert totals["cbore_back"] == 30
    assert totals["cbore_total"] == 60
    assert totals["spot_total"] == 10

    geo = {"ops_summary": summary}
    drilling_meta = {
        "index_min_per_hole": 2.12608593,
        "peck_min_per_hole_vals": [0.0, 0.0],
        "toolchange_min_deep": 0.0,
        "toolchange_min_std": 0.0,
        "bins_list": [
            {"op": "drill", "op_name": "drill", "diameter_in": 0.531, "depth_in": 2.0, "qty": 1},
            {"op": "drill", "op_name": "drill", "diameter_in": 0.201, "depth_in": 2.0, "qty": 1},
            {"op": "drill", "op_name": "drill", "diameter_in": 0.159, "depth_in": 2.0, "qty": 1},
        ],
    }

    subtotal_min, tool_min, total_min, detail = appV5._estimate_drilling_minutes_from_meta(
        drilling_meta, geo
    )

    assert subtotal_min == pytest.approx(207.18, rel=1e-3)
    assert tool_min == pytest.approx(0.0, abs=1e-6)
    assert total_min == pytest.approx(207.18, rel=1e-3)


def test_aggregate_ops_tap_pilot_claims() -> None:
    _ensure_material_pricing_stubs()
    import appV5

    rows = [
        {
            "hole": "P1",
            "ref": "Ø0.201",
            "qty": 4,
            "desc": "Ø0.201 DRILL THRU 1/4-20 TAP",
            "diameter_in": 0.201,
        }
    ]
    ops_entries = [
        {
            "type": "tap",
            "qty": 4,
            "ref_dia_in": 0.201,
            "side": "FRONT",
            "thread": "1/4-20",
            "source": "CHART",
        },
        {
            "type": "drill",
            "qty": 4,
            "ref_dia_in": 0.201,
            "side": "FRONT",
            "claimed_by_tap": True,
            "pilot_for_thread": "1/4-20",
            "source": "CHART",
        },
    ]

    summary = appV5.aggregate_ops(rows, ops_entries=ops_entries)

    totals = summary["totals"]
    assert totals["drill"] == 0
    assert totals["tap_front"] == 4

    claims = summary.get("claims", {})
    assert claims
    assert claims.get("claimed_pilot_diams") == [pytest.approx(0.201, rel=1e-6)] * 4

    detail = summary.get("rows_detail") or []
    assert all(entry.get("type") != "drill" for entry in detail)


def test_aggregate_ops_claimed_pilot_without_tap_diameter() -> None:
    _ensure_material_pricing_stubs()
    import appV5

    rows = [
        {
            "hole": "P1",
            "ref": "Ø0.201",
            "qty": 2,
            "desc": "Ø0.201 DRILL THRU 1/4-20 TAP",
            "diameter_in": 0.201,
        }
    ]
    ops_entries = [
        {
            "type": "tap",
            "qty": 2,
            "side": "FRONT",
            "thread": "1/4-20",
            "source": "CHART",
        },
        {
            "type": "drill",
            "qty": 2,
            "ref_dia_in": 0.201,
            "side": "FRONT",
            "claimed_by_tap": True,
            "pilot_for_thread": "1/4-20",
            "source": "CHART",
        },
    ]

    summary = appV5.aggregate_ops(rows, ops_entries=ops_entries)

    totals = summary["totals"]
    assert totals["drill"] == 0
    assert totals["tap_front"] == 2

    claims = summary.get("claims", {})
    assert claims
    assert claims.get("claimed_pilot_diams") == [pytest.approx(0.201, rel=1e-6)] * 2

    detail = summary.get("rows_detail") or []
    assert all(entry.get("type") != "drill" for entry in detail)


@pytest.mark.parametrize(
    "case",
    [
        Case("emit_ops_cards_sections", _run_emit_ops_cards_sections),
        Case("emit_ops_cards_bucket_view", _run_emit_ops_cards_bucket_view),
        Case("aggregate_ops_sets_built_rows", _run_aggregate_ops_sets_built),
    ],
    ids=lambda case: case.name,
)
def test_quote_ops_cards_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Sum time helper domain -----


def _run_sum_time_default(_: pytest.FixtureRequest) -> None:
    items = _build_series(["In-Process Inspection Hours"])
    vals = _build_series([""])
    types = _build_series(["number"])

    result = _sum_time(items, vals, types, r"In-Process Inspection", default=1.0)

    assert result == pytest.approx(1.0, rel=1e-6)


def _run_sum_time_respects_zero(_: pytest.FixtureRequest) -> None:
    items = _build_series(["In-Process Inspection Hours"])
    vals = _build_series(["0"])
    types = _build_series(["number"])

    result = _sum_time(items, vals, types, r"In-Process Inspection", default=1.0)

    assert result == pytest.approx(0.0, rel=1e-6)


def _run_sum_time_minutes(_: pytest.FixtureRequest) -> None:
    items = _build_series(["Inspection Minutes"])
    vals = _build_series(["30"])
    types = _build_series(["number"])

    result = _sum_time(items, vals, types, r"Inspection", default=0.0)

    assert result == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        Case("sum_time_default", _run_sum_time_default),
        Case("sum_time_respects_zero", _run_sum_time_respects_zero),
        Case("sum_time_minutes", _run_sum_time_minutes),
    ],
    ids=lambda case: case.name,
)
def test_sum_time_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- FAIR checkbox domain -----


def _run_fai_checkbox_tokens(request: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")
    monkeypatch = request.getfixturevalue("monkeypatch")

    captured: dict[str, dict] = {}

    def _capture_payload(geo, baseline, rates, bounds):
        captured["geo"] = geo
        return {}

    monkeypatch.setattr(appV5, "build_suggest_payload", _capture_payload)

    expectations = {"True": True, "t": False, "F": False}
    for token, expected in expectations.items():
        df = pd.DataFrame(
            [
                {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
                {
                    "Item": "FAIR Required",
                    "Example Values / Options": token,
                    "Data Type / Input Method": "checkbox",
                },
                {
                    "Item": "Material Name",
                    "Example Values / Options": "6061-T6 Aluminum",
                    "Data Type / Input Method": "text",
                },
                {
                    "Item": "Net Volume (cm^3)",
                    "Example Values / Options": 50,
                    "Data Type / Input Method": "number",
                },
                {
                    "Item": "Thickness (in)",
                    "Example Values / Options": 1.0,
                    "Data Type / Input Method": "number",
                },
            ]
        )

        result = appV5.compute_quote_from_df(df, llm_enabled=False)

        baseline = result["decision_state"]["baseline"]
        assert baseline.get("fai_required") is expected
        assert isinstance(baseline.get("fai_required"), bool)

        derived = captured["geo"]["derived"]
        assert derived.get("fai_required") is expected
        assert isinstance(derived.get("fai_required"), bool)


@pytest.mark.parametrize("case", [Case("fai_checkbox_tokens", _run_fai_checkbox_tokens)], ids=lambda c: c.name)
def test_fai_checkbox_case(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Inspection breakdown domain -----


def _base_variables_df():
    import appV5

    return appV5.default_variables_template().copy()


def _run_inspection_components(_: pytest.FixtureRequest) -> None:
    import appV5
    from cad_quoter.domain import QuoteState

    df = _base_variables_df()
    state = QuoteState()

    result = appV5.compute_quote_from_df(df, quote_state=state, llm_enabled=False)
    meta = result["breakdown"]["process_meta"]["inspection"]

    components = meta["components"]
    baseline_hr = meta["baseline_hr"]

    assert set(components.keys()) == {
        "in_process",
        "final",
        "cmm_programming",
        "cmm_run",
        "fair",
        "source",
    }
    assert baseline_hr == pytest.approx(sum(components.values()))


def _run_inspection_overrides(_: pytest.FixtureRequest) -> None:
    import appV5
    from cad_quoter.domain import QuoteState

    df = _base_variables_df()
    baseline_state = QuoteState()
    baseline_result = appV5.compute_quote_from_df(df, quote_state=baseline_state, llm_enabled=False)
    baseline_meta = baseline_result["breakdown"]["process_meta"]["inspection"]
    baseline_hr = baseline_meta["hr"]

    target_hr = max(0.0, baseline_hr - 0.5)

    state = QuoteState()
    state.user_overrides["cmm_minutes"] = 90
    state.user_overrides["inspection_total_hr"] = target_hr

    result = appV5.compute_quote_from_df(df, quote_state=state, llm_enabled=False)
    meta = result["breakdown"]["process_meta"]["inspection"]

    adjustments = meta["adjustments"]

    assert adjustments["cmm_run"] > 0
    assert state.effective_sources.get("inspection_total_hr") == "user"


@pytest.mark.parametrize(
    "case",
    [
        Case("inspection_components", _run_inspection_components),
        Case("inspection_overrides", _run_inspection_overrides),
    ],
    ids=lambda case: case.name,
)
def test_inspection_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Scrap estimation domain -----


def _scrap_base_rows() -> list[dict[str, object]]:
    return [
        {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
        {
            "Item": "Material Name",
            "Example Values / Options": "6061-T6 Aluminum",
            "Data Type / Input Method": "text",
        },
        {"Item": "Net Volume (cm^3)", "Example Values / Options": 80.0, "Data Type / Input Method": "number"},
        {"Item": "Material Density", "Example Values / Options": 2.7, "Data Type / Input Method": "number"},
    ]


def _stub_payload(*_args, **_kwargs):
    return {}


def _run_scrap_defaults(monkey_request: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")
    monkeypatch = monkey_request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    df = pd.DataFrame(_scrap_base_rows())

    result = appV5.compute_quote_from_df(df, llm_enabled=False)

    baseline = result["decision_state"]["baseline"]
    assert math.isclose(baseline["scrap_pct"], appV5.SCRAP_DEFAULT_GUESS, rel_tol=1e-9)

    material = result["breakdown"]["material"]
    assert math.isclose(material["scrap_pct"], appV5.SCRAP_DEFAULT_GUESS, rel_tol=1e-9)
    assert material.get("scrap_source") == "default_guess"


def _run_scrap_stock_plan(monkey_request: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")
    monkeypatch = monkey_request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    df = pd.DataFrame(_scrap_base_rows())
    geo = {"stock_plan_guess": {"net_volume_in3": 80.0, "stock_volume_in3": 96.0}}

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)

    expected_scrap = (96.0 - 80.0) / 80.0
    baseline = result["decision_state"]["baseline"]
    assert baseline["scrap_pct"] == pytest.approx(expected_scrap)

    material = result["breakdown"]["material"]
    assert material["scrap_pct"] == pytest.approx(expected_scrap)
    assert material.get("scrap_source") == "stock_plan_guess"


def _run_scrap_ui_override(monkey_request: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")
    monkeypatch = monkey_request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    rows = _scrap_base_rows() + [
        {"Item": "Scrap Percent (%)", "Example Values / Options": 12.0, "Data Type / Input Method": "number"}
    ]
    df = pd.DataFrame(rows)

    result = appV5.compute_quote_from_df(df, llm_enabled=False)

    expected_scrap = 0.12
    baseline = result["decision_state"]["baseline"]
    assert baseline["scrap_pct"] == pytest.approx(expected_scrap)

    material = result["breakdown"]["material"]
    assert material["scrap_pct"] == pytest.approx(expected_scrap)
    assert material.get("scrap_source") == "ui"


def _run_scrap_hole_estimate(monkey_request: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")
    monkeypatch = monkey_request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    df = pd.DataFrame(_scrap_base_rows())
    geo = {
        "derived": {
            "hole_diams_mm": [100.0, 100.0, 100.0, 100.0, 100.0],
            "bbox_mm": (100.0, 100.0),
        }
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)

    baseline = result["decision_state"]["baseline"]
    assert baseline["scrap_pct"] == pytest.approx(0.25)
    assert baseline.get("scrap_source_label") == "default_guess+holes"

    material = result["breakdown"]["material"]
    assert material["scrap_pct"] == pytest.approx(0.25)
    assert material.get("scrap_source") == "default_guess"
    assert material.get("scrap_source_label") == "default_guess+holes"


def _run_scrap_credit_wieland(monkey_request: pytest.FixtureRequest) -> None:
    import appV5
    from cad_quoter.pricing import materials as materials_pricing

    pd = pytest.importorskip("pandas")
    monkeypatch = monkey_request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    monkeypatch.setattr(appV5, "_wieland_scrap_usd_per_lb", lambda _fam: 2.0)

    rows = _scrap_base_rows() + [
        {"Item": "Plate Length (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Plate Width (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Thickness (in)", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
    ]
    df = pd.DataFrame(rows)
    result = appV5.compute_quote_from_df(df, llm_enabled=False)

    material = result["breakdown"]["material"]
    scrap_credit = material.get("computed_scrap_credit_usd")
    assert scrap_credit is not None and scrap_credit > 0
    assert not material.get("material_scrap_credit")
    assert material.get("scrap_price_source") == "wieland"

    scrap_mass_g = material.get("scrap_mass_g")
    assert scrap_mass_g is not None and scrap_mass_g > 0
    scrap_mass_lb = float(scrap_mass_g) / 1000.0 * appV5.LB_PER_KG
    material_block = result["breakdown"]["material_block"]
    base_cost = float(material_block.get("material_cost_before_credit") or 0.0)
    expected_credit = round(
        min(base_cost, scrap_mass_lb * 2.0 * materials_pricing.SCRAP_RECOVERY_DEFAULT),
        2,
    )
    assert scrap_credit == pytest.approx(expected_credit)

    net_material_cost = float(material_block.get("total_material_cost") or 0.0)
    stock_price = float(
        material_block.get("stock_price$")
        or material.get("stock_price$")
        or base_cost
    )
    assert base_cost > 0
    assert net_material_cost == pytest.approx(stock_price)

    assert material.get("scrap_credit_unit_price_usd_per_lb") == pytest.approx(2.0)
    assert material.get("scrap_credit_recovery_pct") == pytest.approx(materials_pricing.SCRAP_RECOVERY_DEFAULT)


def _run_scrap_credit_overrides(monkey_request: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")
    monkeypatch = monkey_request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "build_suggest_payload", _stub_payload)
    monkeypatch.setattr(appV5, "_wieland_scrap_usd_per_lb", lambda _fam: None)

    rows = _scrap_base_rows() + [
        {"Item": "Plate Length (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Plate Width (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        {"Item": "Thickness (in)", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
    ]
    df = pd.DataFrame(rows)
    state = appV5.QuoteState()
    state.user_overrides = {
        "scrap_credit_unit_price_usd_per_lb": 1.25,
        "scrap_recovery_pct": 90.0,
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, quote_state=state)

    material = result["breakdown"]["material"]
    scrap_credit = float(material.get("material_scrap_credit") or 0.0)
    assert scrap_credit > 0

    scrap_mass_g = float(material.get("scrap_mass_g") or 0.0)
    assert scrap_mass_g > 0
    scrap_mass_lb = scrap_mass_g / 1000.0 * appV5.LB_PER_KG
    material_block = result["breakdown"]["material_block"]
    base_cost = float(material_block.get("material_cost_before_credit") or 0.0)
    expected_credit = round(min(base_cost, scrap_mass_lb * 1.25 * 0.90), 2)
    assert scrap_credit == pytest.approx(expected_credit)

    net_cost = float(material_block.get("total_material_cost") or 0.0)
    stock_price = float(
        material_block.get("stock_price$")
        or material.get("stock_price$")
        or base_cost
    )
    assert net_cost == pytest.approx(stock_price)

    assert material.get("scrap_credit_unit_price_usd_per_lb") == pytest.approx(1.25)
    assert material.get("scrap_credit_recovery_pct") == pytest.approx(0.90)
    assert material.get("scrap_credit_source") == "override_unit_price"


@pytest.mark.parametrize(
    "case",
    [
        Case("scrap_defaults", _run_scrap_defaults),
        Case("scrap_stock_plan", _run_scrap_stock_plan),
        Case("scrap_ui_override", _run_scrap_ui_override),
        Case("scrap_hole_estimate", _run_scrap_hole_estimate),
        Case("scrap_credit_wieland", _run_scrap_credit_wieland),
        Case("scrap_credit_overrides", _run_scrap_credit_overrides),
    ],
    ids=lambda case: case.name,
)
def test_scrap_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Material removal domain -----


def _run_mass_scrap_preserves(_: pytest.FixtureRequest) -> None:
    from appV5 import compute_mass_and_scrap_after_removal

    net_after, scrap_after, eff_after = compute_mass_and_scrap_after_removal(1000.0, 0.1, 100.0)

    expected_scrap = (1000.0 * 0.1 + 100.0) / 900.0

    assert math.isclose(net_after, 900.0)
    assert math.isclose(scrap_after, expected_scrap, rel_tol=1e-9)
    assert math.isclose(eff_after, 1100.0, rel_tol=1e-9)


def _run_mass_scrap_bounds(_: pytest.FixtureRequest) -> None:
    from appV5 import compute_mass_and_scrap_after_removal

    net_after, scrap_after, eff_after = compute_mass_and_scrap_after_removal(
        1000.0,
        0.2,
        400.0,
        scrap_min=0.0,
        scrap_max=0.25,
    )

    assert math.isclose(net_after, 600.0)
    assert math.isclose(scrap_after, 0.25)
    assert math.isclose(eff_after, 600.0 * 1.25)


def _run_mass_scrap_no_change(_: pytest.FixtureRequest) -> None:
    from appV5 import compute_mass_and_scrap_after_removal

    net_after, scrap_after, eff_after = compute_mass_and_scrap_after_removal(500.0, 0.05, 0.0)

    assert math.isclose(net_after, 500.0)
    assert math.isclose(scrap_after, 0.05)
    assert math.isclose(eff_after, 500.0 * 1.05)


def _run_plate_scrap_tracks(_: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {"Item": "Material", "Example Values / Options": "6061-T6 Aluminum", "Data Type / Input Method": "text"},
            {"Item": "Material Name", "Example Values / Options": "6061-T6 Aluminum", "Data Type / Input Method": "text"},
            {"Item": "Thickness (in)", "Example Values / Options": 0.5, "Data Type / Input Method": "number"},
            {"Item": "Plate Length (in)", "Example Values / Options": 6.0, "Data Type / Input Method": "number"},
            {"Item": "Plate Width (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
            {"Item": "Roughing Cycle Time", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
        ]
    )

    geo = {
        "kind": "2d",
        "material": "6061-T6 Aluminum",
        "thickness_mm": 12.7,
        "plate_length_mm": 152.4,
        "plate_width_mm": 101.6,
        "hole_diams_mm": [12.7, 12.7],
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)
    material_info = result["breakdown"]["material"]
    scrap_pct = material_info.get("scrap_pct")

    assert scrap_pct is not None

    density = appV5._density_for_material("6061")
    thickness_in = 0.5
    length_in = 6.0
    width_in = 4.0
    plate_volume_in3 = length_in * width_in * thickness_in
    radius_mm = 12.7 / 2.0
    height_mm = thickness_in * 25.4
    hole_volume_mm3 = math.pi * (radius_mm**2) * height_mm * 2
    hole_volume_in3 = hole_volume_mm3 / 16387.064
    removed_mass = hole_volume_in3 * 16.387064 * density
    net_mass = (plate_volume_in3 - hole_volume_in3) * 16.387064 * density

    expected_scrap = min(0.25, removed_mass / net_mass)
    assert scrap_pct == pytest.approx(expected_scrap, rel=1e-6)


def _run_geo_context_fields(_: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {"Item": "Material", "Example Values / Options": "A2 Tool Steel", "Data Type / Input Method": "text"},
            {"Item": "Material Name", "Example Values / Options": "A2 Tool Steel", "Data Type / Input Method": "text"},
            {"Item": "Thickness (in)", "Example Values / Options": 0.5, "Data Type / Input Method": "number"},
            {"Item": "Plate Length (in)", "Example Values / Options": 6.0, "Data Type / Input Method": "number"},
            {"Item": "Plate Width (in)", "Example Values / Options": 4.0, "Data Type / Input Method": "number"},
        ]
    )

    geo = {
        "kind": "2d",
        "plate_len_in": 6.0,
        "plate_wid_in": 4.0,
        "thickness_in": 0.5,
        "hole_groups": [
            {"dia_mm": 6.35, "count": 2},
            {"dia_mm": 12.7, "count": 1},
        ],
        "hole_diams_mm": [6.35, 6.35, 12.7],
    }

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)
    geo_context = result["breakdown"]["geo_context"]

    assert geo_context["thickness_mm"] == pytest.approx(0.5 * 25.4, rel=1e-6)

    expected_area = (6.0 * 25.4) * (4.0 * 25.4)
    assert geo_context["plate_bbox_area_mm2"] == pytest.approx(expected_area, rel=1e-6)

    hole_sets = geo_context.get("hole_sets")
    assert isinstance(hole_sets, list)
    assert len(hole_sets) == 2
    assert hole_sets[0]["qty"] == 2
    assert hole_sets[0]["dia_mm"] == pytest.approx(6.35, rel=1e-6)
    assert hole_sets[1]["qty"] == 1
    assert hole_sets[1]["dia_mm"] == pytest.approx(12.7, rel=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        Case("mass_scrap_preserves", _run_mass_scrap_preserves),
        Case("mass_scrap_bounds", _run_mass_scrap_bounds),
        Case("mass_scrap_no_change", _run_mass_scrap_no_change),
        Case("plate_scrap_tracks", _run_plate_scrap_tracks),
        Case("geo_context_fields", _run_geo_context_fields),
    ],
    ids=lambda case: case.name,
)
def test_material_removal_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Drilling groups domain -----


def _run_drilling_groups_fallback(_: pytest.FixtureRequest) -> None:
    import appV5

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {"Item": "Material", "Example Values / Options": "6061-T6 Aluminum", "Data Type / Input Method": "text"},
        ]
    )

    geo = {"hole_diams_mm": [6.35, 6.35, 12.7], "thickness_in": 0.75}

    result = appV5.compute_quote_from_df(df, llm_enabled=False, geo=geo)

    drilling_meta = result.get("breakdown", {}).get("drilling_meta", {})
    groups = drilling_meta.get("bins_list") or []
    assert groups, "expected fallback drill groups to be generated"

    total_qty = sum(int(group.get("qty", 0)) for group in groups)
    assert total_qty == len(geo["hole_diams_mm"])
    assert drilling_meta.get("hole_count") == total_qty


@pytest.mark.parametrize("case", [Case("drilling_groups_fallback", _run_drilling_groups_fallback)], ids=lambda c: c.name)
def test_drilling_groups_case(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Drilling time estimation domain -----


def _run_estimate_drilling_hours_uses_table(_: pytest.FixtureRequest) -> None:
    import pandas as pd

    from appV5 import SpeedsFeedsUnavailableError, estimate_drilling_hours
    from cad_quoter.pricing.time_estimator import MachineParams, OverheadParams

    table = pd.DataFrame(
        [
            {
                "operation": "Drill",
                "material": "Aluminum",
                "sfm_start": 120,
                "fz_ipr_0_125in": 0.002,
                "fz_ipr_0_25in": 0.004,
                "fz_ipr_0_5in": 0.008,
            }
        ]
    )

    machine = MachineParams(rapid_ipm=200)
    overhead = OverheadParams(
        toolchange_min=0.5,
        approach_retract_in=0.25,
        peck_penalty_min_per_in_depth=0.02,
    )

    hours = estimate_drilling_hours(
        [6.35, 6.35, 12.7],
        0.5,
        "Aluminum",
        hole_groups=[
            {"dia_mm": 6.35, "depth_mm": 12.7, "count": 2},
            {"dia_mm": 12.7, "depth_mm": 12.7, "count": 1},
        ],
        speeds_feeds_table=table,
        machine_params=machine,
        overhead_params=overhead,
    )

    assert hours == pytest.approx(0.0222651727, rel=1e-6)


def _run_estimate_drilling_hours_requires_table(_: pytest.FixtureRequest) -> None:
    from appV5 import SpeedsFeedsUnavailableError, estimate_drilling_hours

    with pytest.raises(SpeedsFeedsUnavailableError):
        estimate_drilling_hours([5.0, 5.0], 0.25, "Steel")


def _run_estimate_drilling_hours_deep_drill(_: pytest.FixtureRequest) -> None:
    import pandas as pd

    from appV5 import estimate_drilling_hours
    from cad_quoter.pricing.time_estimator import MachineParams, OverheadParams

    table = pd.DataFrame(
        [
            {
                "operation": "Deep_Drill",
                "material": "Aluminum 6061-T6",
                "material_group": "N1",
                "sfm_start": 80,
                "fz_ipr_0_125in": 0.001,
                "fz_ipr_0_25in": 0.0015,
                "fz_ipr_0_5in": 0.002,
            }
        ]
    )

    machine = MachineParams(rapid_ipm=200)
    overhead = OverheadParams(
        toolchange_min=0.5,
        approach_retract_in=0.25,
        peck_penalty_min_per_in_depth=0.02,
    )

    hours = estimate_drilling_hours(
        [6.35],
        19.05,
        "Aluminum",
        hole_groups=[{"dia_mm": 6.35, "depth_mm": 19.05, "count": 1}],
        speeds_feeds_table=table,
        machine_params=machine,
        overhead_params=overhead,
    )

    assert hours == pytest.approx(0.0274080019, rel=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        Case("drilling_hours_uses_table", _run_estimate_drilling_hours_uses_table),
        Case("drilling_hours_requires_table", _run_estimate_drilling_hours_requires_table),
        Case("drilling_hours_deep_drill", _run_estimate_drilling_hours_deep_drill),
    ],
    ids=lambda case: case.name,
)
def test_drilling_time_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Die plate drilling regression domain -----


def _run_die_plate_deep_drill_regression(_: pytest.FixtureRequest) -> None:
    import math

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
    assert ld_ratio > 3.0

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
                "basis": ["Planner: Deep_Drill cycle with per-hole clamp"],
            }
        },
        "process_breakdown": {
            "drilling": {
                "hr": bounded_hours,
                "basis": ["Planner clamp bounds"],
                "why": [
                    "Deep drilling cycle triggered by L/D 3.68",
                    f"Clamp keeps drilling time between {expected_min_hours:.2f} and {expected_max_hours:.2f} hr",
                ],
            }
        },
        "process_hours": {"drilling": bounded_hours},
        "process_minutes": {"drilling": bounded_hours * 60.0},
        "labor_costs": {"Drilling": process_cost},
        "labor_cost_details": {"Drilling": "Bounded by per-hole clamp"},
        "pass_through": {"Consumables": 40.0},
        "direct_cost_details": {"Consumables": "Cutting oil allowance"},
        "applied_pcts": {"MarginPct": 0.20},
        "totals": {
            "subtotal": process_cost + 40.0,
            "labor_cost": process_cost,
            "with_expedite": process_cost + 40.0,
            "with_margin": (process_cost + 40.0) * 1.20,
            "price": (process_cost + 40.0) * 1.20,
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
            f"Clamp bounds drilling hours between {expected_min_hours:.2f} and {expected_max_hours:.2f}."
        ),
    }

    rendered = appV5.render_quote(result, currency="$", show_zeros=False)
    payload = breakdown.get("render_payload")
    assert isinstance(payload, dict)

    assert "Quote Summary" in rendered
    drivers = payload.get("price_drivers", [])
    assert any("deep_drill" in driver.get("detail", "").lower() for driver in drivers)


def _run_steel_die_plate_runtime(_: pytest.FixtureRequest) -> None:
    import math
    import re

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

    assert hours >= 3.0
    assert hours == pytest.approx(14.0042194610, rel=1e-6)

    summary = next((line for line in debug_lines if line.startswith("Drill calc")), "")
    assert summary, "Expected deep drill debug summary"
    assert "op=Deep_Drill" in summary
    assert "index" in summary.lower()

    rpm_match = re.search(r"RPM (\d+(?:\.\d+)?)(?:[–-](\d+(?:\.\d+)?))?", summary)
    assert rpm_match is not None


@pytest.mark.parametrize(
    "case",
    [
        Case("die_plate_deep_drill_regression", _run_die_plate_deep_drill_regression),
        Case("steel_die_plate_runtime", _run_steel_die_plate_runtime),
    ],
    ids=lambda case: case.name,
)
def test_die_plate_drilling_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Programming override domain -----


def _programming_df(rows: Iterable[tuple[str, Union[float, int, str], str]]):
    import pandas as pd

    normalized = [
        {
            "Item": item,
            "Example Values / Options": value,
            "Data Type / Input Method": dtype,
        }
        for item, value, dtype in rows
    ]
    return pd.DataFrame(normalized, columns=["Item", "Example Values / Options", "Data Type / Input Method"])


BASE_GEO = {
    "thickness_mm": 25.4,
    "material": "6061",
    "GEO__Face_Count": 6,
    "GEO__Unique_Normal_Count": 3,
    "GEO_Complexity_0to100": 25,
    "GEO__MaxDim_mm": 100,
}


def _run_programming_override_caps(_: pytest.FixtureRequest) -> None:
    import appV5

    df = _programming_df(
        [
            ("Qty", 1, "number"),
            ("Material", "6061", "text"),
            ("Stock Thickness_mm", 25.4, "number"),
            ("Programming Override Hr", 1.0, "number"),
        ]
    )

    result = appV5.compute_quote_from_df(
        df,
        llm_enabled=False,
        geo=BASE_GEO,
    )

    prog_detail = result["breakdown"]["nre_detail"]["programming"]

    assert prog_detail["prog_hr"] == pytest.approx(1.0)
    assert prog_detail["override_applied"] is True
    assert prog_detail["auto_prog_hr"] > 0.0


def _run_programming_without_override(_: pytest.FixtureRequest) -> None:
    import appV5

    df = _programming_df(
        [
            ("Qty", 1, "number"),
            ("Material", "6061", "text"),
            ("Stock Thickness_mm", 25.4, "number"),
        ]
    )

    result = appV5.compute_quote_from_df(
        df,
        llm_enabled=False,
        geo=BASE_GEO,
    )

    prog_detail = result["breakdown"]["nre_detail"]["programming"]

    assert prog_detail["auto_prog_hr"] > 0.0
    assert prog_detail["prog_hr"] == pytest.approx(prog_detail["auto_prog_hr"], rel=1e-6)
    assert "override_applied" not in prog_detail


@pytest.mark.parametrize(
    "case",
    [
        Case("programming_override_caps", _run_programming_override_caps),
        Case("programming_without_override", _run_programming_without_override),
    ],
    ids=lambda case: case.name,
)
def test_programming_override_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Planner render domain -----


def _run_planner_render_canonicalize(request: pytest.FixtureRequest) -> None:
    import pytest

    from cad_quoter.ui import planner_render

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.delenv("DEBUG_MISC", raising=False)

    costs = {
        "Milling": 60.0,
        "Planner Machine": 125.0,
        "Planner Labor": 80.0,
        "Planner Total": 205.0,
        "Misc": 10.0,
        "Saw/Waterjet": 12.5,
    }

    canon = planner_render.canonicalize_costs(costs)

    assert set(canon) == {"milling", "saw_waterjet"}
    assert canon["milling"] == pytest.approx(60.0)
    assert canon["saw_waterjet"] == pytest.approx(12.5)


@pytest.mark.parametrize("case", [Case("planner_render_canonicalize", _run_planner_render_canonicalize)], ids=lambda c: c.name)
def test_planner_render_case(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Planner resolve domain -----


def _run_resolve_defaults(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    used, mode = module.resolve_planner(None, None)
    assert used is False
    assert mode == "planner"


def _run_resolve_planner_mode(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(module, "FORCE_PLANNER", False)
    used, mode = module.resolve_planner(
        {"PlannerMode": "planner"},
        {"totals_present": True},
    )
    assert used is True
    assert mode == "planner"


def _run_resolve_legacy_requires(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    used, mode = module.resolve_planner(
        {"PlannerMode": "legacy"},
        {"recognized_line_items": 0},
    )
    assert used is False
    assert mode == "legacy"


def _run_resolve_legacy_recognized(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    used, mode = module.resolve_planner(
        {"PlannerMode": "legacy"},
        {"recognized_line_items": "3"},
    )
    assert used is True
    assert mode == "legacy"


def _run_resolve_invalid_mode(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    used, mode = module.resolve_planner(
        {"PlannerMode": "   "},
        {"pricing_result": {"totals": {}}},
    )
    assert used is True
    assert mode == "planner"


def _run_resolve_force_flag(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    monkeypatch = request.getfixturevalue("monkeypatch")
    planner_adapter = importlib.import_module("cad_quoter.app.planner_adapter")
    monkeypatch.setattr(module, "FORCE_PLANNER", True)
    monkeypatch.setattr(planner_adapter, "FORCE_PLANNER", True)
    used, mode = module.resolve_planner({"PlannerMode": "legacy"}, None)
    assert used is True
    assert mode == "planner"


def _run_resolve_force_estimator(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    monkeypatch = request.getfixturevalue("monkeypatch")
    planner_adapter = importlib.import_module("cad_quoter.app.planner_adapter")
    monkeypatch.setattr(module, "FORCE_PLANNER", False)
    monkeypatch.setattr(planner_adapter, "FORCE_PLANNER", False)
    monkeypatch.setattr(module, "FORCE_ESTIMATOR", True, raising=False)
    monkeypatch.setattr(planner_adapter, "FORCE_ESTIMATOR", True, raising=False)

    used, mode = module.resolve_planner({"PlannerMode": "planner"}, {"line_items": [1]})

    assert used is False
    assert mode == "estimator"


def _run_resolve_no_internal(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    assert not hasattr(module, "_resolve_planner_mode")
    assert not hasattr(module, "_resolve_planner_usage")


def _run_resolve_line_items_force(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    used, mode = module.resolve_planner(
        {"PlannerMode": "auto"},
        {"line_items": [{"op": "test"}]},
    )
    assert used is True
    assert mode == "auto"


def _run_count_recognized_ops(request: pytest.FixtureRequest) -> None:
    module = request.getfixturevalue("reload_appv5")
    assert module._count_recognized_ops(None) == 0
    assert module._count_recognized_ops({"ops": None}) == 0

    plan_summary = {
        "ops": [
            {"name": "rough mill"},
            {"name": "finish mill"},
            "drill",
            None,
            0,
        ]
    }

    assert module._count_recognized_ops(plan_summary) == 3


@pytest.mark.parametrize(
    "case",
    [
        Case("resolve_defaults", _run_resolve_defaults),
        Case("resolve_planner_mode", _run_resolve_planner_mode),
        Case("resolve_legacy_requires", _run_resolve_legacy_requires),
        Case("resolve_legacy_recognized", _run_resolve_legacy_recognized),
        Case("resolve_invalid_mode", _run_resolve_invalid_mode),
        Case("resolve_force_flag", _run_resolve_force_flag),
        Case("resolve_force_estimator", _run_resolve_force_estimator),
        Case("resolve_no_internal", _run_resolve_no_internal),
        Case("resolve_line_items_force", _run_resolve_line_items_force),
        Case("count_recognized_ops", _run_count_recognized_ops),
    ],
    ids=lambda case: case.name,
)
def test_resolve_planner_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Planner minutes domain -----


def _ensure_planner_fixtures(request: pytest.FixtureRequest) -> None:
    request.getfixturevalue("disable_speeds_feeds_loader")


def _run_planner_compute_quote(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 25.4,
    }

    plan_called: dict[str, bool] = {"plan": False}
    price_called: dict[str, bool] = {"price": False}

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        plan_called["plan"] = True
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "Roughing"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        price_called["price"] = True
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {"op": "Machine Time", "minutes": 30.0, "machine_cost": 90.0, "labor_cost": 0.0},
                {"op": "Handwork", "minutes": 15.0, "machine_cost": 0.0, "labor_cost": 45.0},
            ],
            "totals": {"minutes": 45.0, "machine_cost": 90.0, "labor_cost": 45.0},
        }

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9},
        geo=geo,
        ui_vars={},
    )

    assert plan_called["plan"] is True
    assert price_called["price"] is True

    baseline = result["decision_state"]["baseline"]
    assert str(baseline["pricing_source"]).lower() == "planner", baseline["pricing_source"]
    assert baseline["process_plan_pricing"]["totals"]["machine_cost"] == 90.0

    breakdown = result["breakdown"]
    plan_pricing = breakdown.get("process_plan_pricing")
    assert plan_pricing is not None, "expected planner pricing to be present"
    assert plan_pricing.get("line_items"), "expected planner pricing line items"
    assert isinstance(plan_pricing["line_items"][0], dict)
    assert str(breakdown["pricing_source"]).lower() == "planner", breakdown["pricing_source"]
    assert breakdown["process_costs"] == {"Machine": 90.0, "Labor": 45.0}
    assert breakdown["process_minutes"] == 45.0
    planner_meta = breakdown["process_meta"]
    assert "planner_machine" in planner_meta
    assert "planner_labor" in planner_meta
    assert plan_pricing["totals"]["minutes"] == 45.0


def _run_planner_totals_no_flag(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 25.4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "Roughing"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {
                    "op": "Machine Time",
                    "minutes": 30.0,
                    "machine_cost": 90.0,
                    "labor_cost": 0.0,
                },
                {
                    "op": "Handwork",
                    "minutes": 15.0,
                    "machine_cost": 0.0,
                    "labor_cost": 45.0,
                },
            ],
            "totals": {"minutes": 45.0, "machine_cost": 90.0, "labor_cost": 45.0},
        }

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    red_flags = breakdown.get("red_flags") or []
    assert all("Planner totals drifted" not in flag for flag in red_flags)

    plan_pricing = breakdown.get("process_plan_pricing") or {}
    planner_totals = plan_pricing.get("totals", {})
    planner_labor_cost = float(planner_totals.get("labor_cost", 0.0) or 0.0)
    assert planner_labor_cost == pytest.approx(45.0)

    process_costs = breakdown["process_costs"]
    assert "Labor" in process_costs

    totals = breakdown["totals"]
    labor_rendered = float(breakdown.get("labor_cost_rendered", 0.0) or 0.0)
    assert totals["labor_cost"] == pytest.approx(labor_rendered)
    assert labor_rendered > 0.0


def _run_planner_bucket_minutes_from_line_items(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "drill_ream_bore"}, {"op": "cnc_rough_mill"}, {"op": "deburr"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {"op": "drill_ream_bore", "minutes": 30.0, "machine_cost": 300.0, "labor_cost": 0.0},
                {"op": "cnc_rough_mill", "minutes": 20.0, "machine_cost": 200.0, "labor_cost": 0.0},
                {"op": "deburr", "minutes": 10.0, "machine_cost": 0.0, "labor_cost": 100.0},
            ],
            "totals": {"minutes": 60.0, "machine_cost": 500.0, "labor_cost": 100.0},
        }

    def fake_bucketize(*_args, **_kwargs):
        return {"buckets": {}, "totals": {"minutes": 0.0, "machine$": 0.0, "labor$": 0.0, "total$": 0.0}}

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "bucketize", fake_bucketize)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.95},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert str(breakdown["pricing_source"]).lower() == "planner"

    bucket_view = breakdown.get("bucket_view") or {}
    buckets = bucket_view.get("buckets") or {}
    assert buckets, "expected bucket minutes derived from planner line items"
    assert buckets["drilling"]["minutes"] == pytest.approx(30.0, abs=0.01)
    assert buckets["milling"]["minutes"] == pytest.approx(20.0, abs=0.01)
    assert buckets["finishing_deburr"]["minutes"] == pytest.approx(10.0, abs=0.01)

    planner_meta = breakdown["process_meta"]
    assert planner_meta["planner_total"]["minutes"] == pytest.approx(60.0, abs=0.01)
    assert planner_meta["planner_machine"]["minutes"] == pytest.approx(50.0, abs=0.01)
    assert planner_meta["planner_labor"]["minutes"] == pytest.approx(10.0, abs=0.01)


def _run_planner_seed_bucket_minutes(_: pytest.FixtureRequest) -> None:
    from cad_quoter.ui import planner_render

    breakdown: dict[str, object] = {
        "bucket_view": {"buckets": {}},
        "rates": {"TappingRate": 90.0, "LaborRate": 120.0},
    }

    planner_render._seed_bucket_minutes(
        breakdown,
        tapping_min=12.0,
    )

    bucket_view = breakdown["bucket_view"]
    assert isinstance(bucket_view, dict)
    buckets = bucket_view.get("buckets")
    assert isinstance(buckets, dict)
    tapping = buckets.get("tapping")
    assert isinstance(tapping, dict)
    assert tapping["minutes"] == pytest.approx(12.0)
    assert tapping["machine$"] == pytest.approx(18.0)
    assert tapping["labor$"] == pytest.approx(24.0)
    assert tapping["total$"] == pytest.approx(42.0)


def _run_planner_milling_defaults(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "cnc_rough_mill"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {
                    "op": "cnc_rough_mill",
                    "minutes": 60.0,
                    "machine_cost": 45.0,
                    "labor_cost": 45.0,
                }
            ],
            "totals": {"minutes": 60.0, "machine_cost": 45.0, "labor_cost": 45.0},
        }

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 1.0},
        rates={"MillingRate": 90.0, "LaborRate": 45.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    bucket_view = breakdown.get("bucket_view") or {}
    milling_entry = (
        bucket_view.get("milling")
        or (bucket_view.get("buckets") or {}).get("milling")
        or {}
    )
    machine_component = float(
        milling_entry.get("machine$")
        or milling_entry.get("machine_cost")
        or 0.0
    )
    labor_component = float(
        milling_entry.get("labor$")
        or milling_entry.get("labor_cost")
        or 0.0
    )
    total_component = float(
        milling_entry.get("total$")
        or milling_entry.get("total_cost")
        or machine_component + labor_component
    )

    assert machine_component == pytest.approx(90.0, abs=0.01)
    assert labor_component == pytest.approx(0.0, abs=1e-6)
    assert total_component == pytest.approx(machine_component, abs=0.01)


def _run_planner_no_legacy_buckets(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
        "hole_diams_mm": [6.0, 8.0, 10.0, 6.0],
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "drill_ream_bore"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "line_items": [
                {"op": "drill_ream_bore", "minutes": 28.0, "machine_cost": 280.0, "labor_cost": 0.0},
                {"op": "cnc_rough_mill", "minutes": 15.0, "machine_cost": 225.0, "labor_cost": 0.0},
            ],
            "totals": {"minutes": 43.0, "machine_cost": 505.0, "labor_cost": 0.0},
        }

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9, "DrillingRate": 75.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert str(breakdown["pricing_source"]).lower() == "planner"

    bucket_view = breakdown["bucket_view"]
    assert "drilling" not in bucket_view
    assert "milling" not in bucket_view

    drill_meta = breakdown.get("drilling_meta") or {}
    assert "estimator_hours_for_planner" not in drill_meta


def _run_planner_fallback_no_line_items(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
        "hole_diams_mm": [6.0, 8.0],
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": []}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {"line_items": [], "totals": {}}

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9, "DrillingRate": 75.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert breakdown["pricing_source"] == "legacy"
    red_flags = breakdown.get("red_flags") or []
    assert any("Planner recognized no operations" in flag for flag in red_flags)

    bucket_view = breakdown.get("bucket_view") or {}
    drilling_bucket = bucket_view.get("drilling")
    assert drilling_bucket is not None, "expected drilling bucket for fallback"

    estimator_hours = drilling_bucket["minutes"] / 60.0
    assert estimator_hours > 0
    expected_minutes = estimator_hours * 60.0
    assert drilling_bucket["minutes"] == pytest.approx(expected_minutes, abs=0.05)

    process_meta = breakdown["process_meta"]
    drilling_meta_entry = process_meta.get("drilling") or {}
    assert drilling_meta_entry.get("hr") == pytest.approx(expected_minutes / 60.0, abs=1e-6)
    assert drilling_bucket["machine_cost"] == pytest.approx((expected_minutes / 60.0) * 75.0, abs=0.05)
    expected_labor_cost = (expected_minutes / 60.0) * 45.0
    assert drilling_bucket.get("labor_cost", 0.0) == pytest.approx(expected_labor_cost, abs=0.05)


def _run_planner_zero_totals(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "steel",
        "thickness_mm": 25.4,
        "hole_diams_mm": [8.0] * 4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "drill"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "recognized_line_items": 1,
            "line_items": [
                {"op": "Machine Time", "minutes": 30.0, "machine_cost": 0.0, "labor_cost": 0.0}
            ],
            "totals": {"minutes": 30.0, "machine_cost": 0.0, "labor_cost": 0.0},
        }

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9, "DrillingRate": 75.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert breakdown["pricing_source"] == "legacy"
    red_flags = breakdown.get("red_flags") or []
    assert any("Planner produced zero machine/labor cost" in flag for flag in red_flags)
    quote_log = breakdown.get("quote_log") or []
    assert any("Planner produced zero machine/labor cost" in entry for entry in quote_log)

    app_meta = result.get("app_meta") or {}
    assert not app_meta.get("used_planner", False)


def _run_planner_milling_backfill(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(
        [
            {
                "Item": "Roughing Cycle Time (hr)",
                "Example Values / Options": 1.65,
                "Data Type / Input Method": "Number",
            },
            {
                "Item": "Setup Time per Setup (hr)",
                "Example Values / Options": 0.0,
                "Data Type / Input Method": "Number",
            },
        ]
    )

    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 25.4,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": []}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {"line_items": [], "totals": {}}

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 1.0},
        rates={"MillingRate": 100.0, "DrillingRate": 75.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert breakdown["pricing_source"] == "legacy"
    red_flags = breakdown.get("red_flags") or []
    assert any("Planner recognized no operations" in flag for flag in red_flags)
    bucket_view = breakdown["bucket_view"]
    milling_bucket = bucket_view.get("milling")
    assert milling_bucket is not None, "expected milling bucket backfill"
    assert milling_bucket["minutes"] == pytest.approx(99.0, abs=0.1)
    assert milling_bucket["machine_cost"] == pytest.approx(165.0, abs=0.1)
    assert milling_bucket.get("labor_cost", 0.0) == pytest.approx(0.0, abs=1e-6)

    process_meta = breakdown["process_meta"]
    milling_meta = process_meta.get("milling")
    assert milling_meta is not None, "expected milling meta entry"
    assert milling_meta.get("hr") == pytest.approx(1.65, abs=0.01)
    basis = milling_meta.get("basis") or []
    assert any("planner_milling_backfill" in str(item) for item in basis)


def _run_planner_die_plate_totals(request: pytest.FixtureRequest) -> None:
    _ensure_planner_fixtures(request)
    import appV5
    import cad_quoter.pricing.planner as planner_pricing

    pd = pytest.importorskip("pandas")

    df = pd.DataFrame(columns=["Item", "Example Values / Options", "Data Type / Input Method"])
    geo = {
        "process_planner_family": "die_plate",
        "material": "tool steel",
        "thickness_mm": 50.8,
        "hole_diams_mm": [13.8] * 163,
    }

    def fake_plan_job(family: str, inputs: dict[str, object]) -> dict[str, object]:
        assert family == "die_plate"
        assert isinstance(inputs, dict)
        return {"ops": [{"op": "Deep_Drill"}]}

    def fake_price_with_planner(
        family: str,
        inputs: dict[str, object],
        geom_payload: dict[str, object],
        rates: dict[str, object],
        *,
        oee: float,
    ) -> dict[str, object]:
        assert family == "die_plate"
        assert geom_payload, "expected non-empty geom payload"
        assert isinstance(rates, dict)
        assert oee > 0
        return {
            "recognized_line_items": 2,
            "line_items": [
                {
                    "op": "Planner Machine",
                    "minutes": 420.0,
                    "machine_cost": 2400.0,
                    "labor_cost": 600.0,
                },
                {
                    "op": "Programming (per part)",
                    "minutes": 0.0,
                    "machine_cost": 0.0,
                    "labor_cost": 320.0,
                },
            ],
            "totals": {"minutes": 420.0, "machine_cost": 2400.0, "labor_cost": 920.0},
        }

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(appV5, "_process_plan_job", fake_plan_job)
    monkeypatch.setattr(planner_pricing, "price_with_planner", fake_price_with_planner)
    monkeypatch.setattr(appV5, "FORCE_PLANNER", False)

    result = appV5.compute_quote_from_df(
        df,
        params={"OEE_EfficiencyPct": 0.9, "DrillingRate": 90.0},
        geo=geo,
        ui_vars={},
    )

    breakdown = result["breakdown"]
    assert str(breakdown["pricing_source"]).lower() == "planner"
    process_costs = breakdown["process_costs"]
    assert process_costs["Machine"] > 0
    assert process_costs["Labor"] > 0

    planner_meta = breakdown["process_meta"]
    planner_total = planner_meta.get("planner_total") or {}
    assert planner_total.get("machine_cost", 0.0) > 0
    assert planner_total.get("labor_cost", 0.0) > 0
    assert planner_total.get("amortized_programming", 0.0) > 0

    planner_machine = planner_meta.get("planner_machine") or {}
    assert planner_machine.get("cost", 0.0) > 0

    planner_labor = planner_meta.get("planner_labor") or {}
    assert planner_labor.get("cost", 0.0) > 0
    assert planner_labor.get("amortized_programming", 0.0) > 0


@pytest.mark.parametrize(
    "case",
    [
        Case("planner_compute_quote", _run_planner_compute_quote),
        Case("planner_totals_no_flag", _run_planner_totals_no_flag),
        Case("planner_bucket_minutes", _run_planner_bucket_minutes_from_line_items),
        Case("planner_seed_bucket_minutes", _run_planner_seed_bucket_minutes),
        Case("planner_milling_defaults", _run_planner_milling_defaults),
        Case("planner_no_legacy_buckets", _run_planner_no_legacy_buckets),
        Case("planner_fallback_no_line_items", _run_planner_fallback_no_line_items),
        Case("planner_zero_totals", _run_planner_zero_totals),
        Case("planner_milling_backfill", _run_planner_milling_backfill),
        Case("planner_die_plate_totals", _run_planner_die_plate_totals),
    ],
    ids=lambda case: case.name,
)
def test_planner_minutes_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)


# ----- Material pricing domain -----


def _ensure_material_pricing_stubs() -> None:
    import importlib.machinery

    for module_name in ("requests", "bs4", "lxml"):
        if module_name not in sys.modules:
            stub = types.ModuleType(module_name)
            stub.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
            sys.modules[module_name] = stub


def _run_resolve_price_prefers_mcmaster(request: pytest.FixtureRequest) -> None:
    _ensure_material_pricing_stubs()
    import cad_quoter.pricing.materials as materials

    monkeypatch = request.getfixturevalue("monkeypatch")
    module = types.ModuleType("metals_api")

    def _raise(_material_key: str) -> float:
        raise RuntimeError("metals api unavailable")

    module.price_per_lb_for_material = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    def _fake_mcmaster(name: str, *, unit: str = "kg") -> tuple[float, str]:
        assert unit == "lb"
        return 123.45, "mcmaster:aluminum"

    def _fail_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        raise AssertionError("resolver should not run when McMaster succeeds")

    monkeypatch.setattr(materials, "get_mcmaster_unit_price", _fake_mcmaster, raising=False)
    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fail_resolver, raising=False)

    price, source = materials._resolve_price_per_lb("aluminum", "Aluminum")

    assert price == pytest.approx(123.45)
    assert source == "mcmaster:aluminum"


def _run_resolve_price_uses_resolver(request: pytest.FixtureRequest) -> None:
    _ensure_material_pricing_stubs()
    import cad_quoter.pricing.materials as materials

    monkeypatch = request.getfixturevalue("monkeypatch")
    module = types.ModuleType("metals_api")
    module.price_per_lb_for_material = lambda _material_key: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    wieland_module = types.ModuleType("cad_quoter.pricing.wieland_scraper")

    def _fail_live_price(*_args, **_kwargs):
        return None

    wieland_module.get_live_material_price = _fail_live_price  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cad_quoter.pricing.wieland_scraper", wieland_module)

    fallback_calls: list[tuple[str, str]] = []

    def _failing_mcmaster(_name: str, *, unit: str = "kg") -> tuple[float | None, str]:
        return None, ""

    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        fallback_calls.append((name, unit))
        return 321.0, "backup_csv"

    monkeypatch.setattr(materials, "get_mcmaster_unit_price", _failing_mcmaster, raising=False)
    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fake_resolver, raising=False)

    price, source = materials._resolve_price_per_lb("6061", "6061")

    assert fallback_calls == [("6061", "lb")]
    assert price == pytest.approx(321.0)
    assert source == "backup_csv"


def _run_material_price_helper(request: pytest.FixtureRequest) -> None:
    _ensure_material_pricing_stubs()
    import cad_quoter.pricing.materials as materials

    monkeypatch = request.getfixturevalue("monkeypatch")

    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        assert unit == "kg"
        return 400.0, "backup_csv"

    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fake_resolver, raising=False)

    price_per_g, source = materials._material_price_per_g_from_choice("Stainless Steel", {})

    assert price_per_g == pytest.approx(0.4)
    assert source == "backup_csv"


def _run_compute_material_block_uses_resolver(request: pytest.FixtureRequest) -> None:
    _ensure_material_pricing_stubs()
    import cad_quoter.pricing.materials as materials

    monkeypatch = request.getfixturevalue("monkeypatch")
    module = types.ModuleType("metals_api")

    def _raise(_material_key: str) -> float:
        raise RuntimeError("metals api offline")

    module.price_per_lb_for_material = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    calls: list[tuple[str, str]] = []

    def _fake_resolver(name: str, *, unit: str = "kg") -> tuple[float, str]:
        calls.append((name, unit))
        assert unit == "lb"
        return 10.0, "resolver"

    monkeypatch.setattr(materials, "_resolve_material_unit_price", _fake_resolver, raising=False)

    geo_ctx = {
        "material_display": "Aluminum 6061",
        "thickness_in": 1.0,
        "outline_bbox": {"plate_len_in": 2.0, "plate_wid_in": 3.0},
    }

    block = materials._compute_material_block(geo_ctx, "aluminum", 2.70, 0.1)

    assert calls == [("Aluminum 6061", "lb")]
    assert block["price_per_lb"] == pytest.approx(10.0)
    assert block["price_source"] == "resolver"

    start_lb = block["start_lb"]
    scrap_lb = block["scrap_lb"]
    assert scrap_lb >= start_lb * 0.1 - 1e-6
    assert block["net_lb"] == pytest.approx(start_lb - scrap_lb)
    assert block["total_material_cost"] == pytest.approx((start_lb - scrap_lb) * 10.0)


def _run_compute_material_block_applies_supplier_min(request: pytest.FixtureRequest) -> None:
    _ensure_material_pricing_stubs()
    import cad_quoter.pricing.materials as materials

    monkeypatch = request.getfixturevalue("monkeypatch")
    module = types.ModuleType("metals_api")
    module.price_per_lb_for_material = lambda _material_key: 2.0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metals_api", module)

    geo_ctx = {
        "material_display": "Steel",
        "thickness_in": 0.5,
        "outline_bbox": {"plate_len_in": 1.0, "plate_wid_in": 1.0},
        "supplier_min$": 75.0,
    }

    block = materials._compute_material_block(geo_ctx, "steel", 7.85, 0.2)

    assert block["supplier_min$"] == pytest.approx(75.0)
    assert block["price_per_lb"] == pytest.approx(2.0)
    assert block["total_material_cost"] == pytest.approx(75.0)


def _run_material_cost_components_prefers_stock(_: pytest.FixtureRequest) -> None:
    import cad_quoter.pricing.materials as materials

    block = {
        "starting_mass_g": 1000.0,
        "scrap_mass_g": 200.0,
        "unit_price_per_lb_usd": 5.0,
        "unit_price_per_lb_source": "resolver",
        "stock_piece_price_usd": 20.0,
        "stock_piece_source": "McMaster API (qty=1, part=4936K451)",
        "material_tax_usd": 1.5,
        "scrap_price_usd_per_lb": 0.8,
    }
    overrides = {"scrap_recovery_pct": 0.9}

    components = materials._material_cost_components(block, overrides=overrides, cfg=None)

    assert components["base_usd"] == pytest.approx(20.0)
    assert components["base_source"] == "McMaster API (qty=1, part=4936K451)"
    assert components["stock_piece_usd"] == pytest.approx(20.0)
    assert components["stock_source"] == "McMaster API (qty=1, part=4936K451)"
    assert components["tax_usd"] == pytest.approx(1.5)
    assert components["scrap_credit_usd"] == pytest.approx(0.32)
    assert components["scrap_rate_text"] == "$0.80/lb × 90%"
    assert components["total_usd"] == pytest.approx(21.18)


def _run_material_cost_components_per_lb(_: pytest.FixtureRequest) -> None:
    import cad_quoter.pricing.materials as materials

    block = {
        "start_mass_g": 500.0,
        "unit_price_per_lb_usd": 6.0,
        "unit_price_source": "resolver",
        "material_tax": 0.75,
    }

    components = materials._material_cost_components(block, overrides={}, cfg=None)

    start_lb = 500.0 * 0.00220462262
    expected_base = round(start_lb * 6.0, 2)
    expected_total = round(expected_base + 0.75, 2)

    assert components["base_usd"] == pytest.approx(expected_base)
    assert components["base_source"] == "resolver @ $6.00/lb"
    assert components["tax_usd"] == pytest.approx(0.75)
    assert components["scrap_credit_usd"] == 0.0
    assert components["scrap_rate_text"] is None
    assert components["total_usd"] == pytest.approx(expected_total)


def _run_material_cost_components_explicit_credit(_: pytest.FixtureRequest) -> None:
    import cad_quoter.pricing.materials as materials

    block = {
        "starting_mass_g": 1000.0,
        "material_tax_usd": 0.0,
        "stock_piece_price_usd": 50.0,
        "stock_piece_source": "Stock piece",
        "material_scrap_credit": 12.34,
    }

    components = materials._material_cost_components(block, overrides={}, cfg=None)

    assert components["scrap_credit_usd"] == pytest.approx(12.34)
    assert components["total_usd"] == pytest.approx(37.66)


def _run_plan_stock_blank_respects_tol(_: pytest.FixtureRequest) -> None:
    import cad_quoter.pricing.materials as materials

    plan = materials.plan_stock_blank(12.0, 6.0, 2.0, None, None)

    assert plan["round_tol_in"] == pytest.approx(0.05)
    assert plan["need_len_in"] == pytest.approx(12.05)
    assert plan["need_wid_in"] == pytest.approx(6.05)
    assert plan["stock_len_in"] == pytest.approx(12.0)
    assert plan["stock_wid_in"] == pytest.approx(6.0)
    assert plan["stock_thk_in"] == pytest.approx(2.0)


def _run_plan_stock_blank_uses_configured(_: pytest.FixtureRequest) -> None:
    import types

    import cad_quoter.pricing.materials as materials

    cfg = types.SimpleNamespace(round_tol_in=0.1)

    plan = materials.plan_stock_blank(18.0, 8.0, 2.0, None, None, cfg=cfg)

    assert plan["round_tol_in"] == pytest.approx(0.1)
    assert plan["need_len_in"] == pytest.approx(18.1)
    assert plan["stock_len_in"] == pytest.approx(18.0)


def _run_vendor_catalog_prefers_thickness(request: pytest.FixtureRequest) -> None:
    import cad_quoter.pricing.materials as materials
    from cad_quoter.pricing import vendor_csv

    rows = [
        {
            "material": "Aluminum MIC6",
            "thickness_in": "3.5",
            "length_in": "12",
            "width_in": "24",
            "vendor": "McMaster",
            "part": "86825K626",
        },
        {
            "material": "Aluminum MIC6",
            "thickness_in": "2",
            "length_in": "12",
            "width_in": "24",
            "vendor": "McMaster",
            "part": "86825K954",
        },
    ]

    calls: list[float] = []

    def fake_pick(L, W, T, *, material_key, catalog_rows):
        calls.append(T)
        assert catalog_rows is rows
        if pytest.approx(T, abs=1e-6) == pytest.approx(2.0):
            return {
                "len_in": 24.0,
                "wid_in": 12.0,
                "thk_in": T,
                "mcmaster_part": "86825K954",
            }
        return None

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setattr(vendor_csv, "load_mcmaster_catalog_rows", lambda _path=None: rows)
    monkeypatch.setattr(vendor_csv, "pick_mcmaster_plate_sku", fake_pick)

    picked = vendor_csv.pick_plate_from_mcmaster(
        "Aluminum MIC6",
        12.0,
        12.0,
        2.0,
        scrap_fraction=0.0,
        allow_thickness_upsize=False,
        thickness_tolerance=0.02,
    )

    assert calls, "helper should be consulted"
    assert picked is not None
    assert picked["thk_in"] == pytest.approx(2.0)
    assert picked["part_no"] == "86825K954"
    assert picked["len_in"] == pytest.approx(24.0)
    assert picked["wid_in"] == pytest.approx(12.0)


@pytest.mark.parametrize(
    "case",
    [
        Case("resolve_price_prefers_mcmaster", _run_resolve_price_prefers_mcmaster),
        Case("resolve_price_uses_resolver", _run_resolve_price_uses_resolver),
        Case("material_price_helper", _run_material_price_helper),
        Case("compute_material_block_resolver", _run_compute_material_block_uses_resolver),
        Case("compute_material_block_supplier_min", _run_compute_material_block_applies_supplier_min),
        Case("material_cost_components_stock", _run_material_cost_components_prefers_stock),
        Case("material_cost_components_per_lb", _run_material_cost_components_per_lb),
        Case("material_cost_components_explicit_credit", _run_material_cost_components_explicit_credit),
        Case("plan_stock_blank_respects_tol", _run_plan_stock_blank_respects_tol),
        Case("plan_stock_blank_uses_configured", _run_plan_stock_blank_uses_configured),
        Case("vendor_catalog_prefers_thickness", _run_vendor_catalog_prefers_thickness),
    ],
    ids=lambda case: case.name,
)
def test_material_pricing_cases(case: Case, request: pytest.FixtureRequest) -> None:
    case.run(request)
