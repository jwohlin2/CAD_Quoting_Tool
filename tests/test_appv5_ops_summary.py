from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest


EZDXF_AVAILABLE = ("ezdxf" in sys.modules) or (importlib.util.find_spec("ezdxf") is not None)


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, attrs: dict[str, object] | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.skipif(not EZDXF_AVAILABLE, reason="ezdxf is required to load sample DXF")
def test_adapter_ops_summary_short_circuits_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_path = Path(__file__).resolve().parents[1] / "Cad Files" / "zeus1.dxf"
    if not sample_path.exists():
        pytest.skip("Sample DXF is not available")

    # Ensure a clean import surface before installing temporary stubs.
    for name in [
        "appV5",
        "cad_quoter.geometry.hole_table_adapter",
        "cad_quoter.geo_dump",
        "requests",
        "bs4",
        "lxml",
        "pandas",
        "mcmaster_api",
    ]:
        sys.modules.pop(name, None)

    _stub_module(monkeypatch, "requests")
    _stub_module(monkeypatch, "bs4")
    _stub_module(monkeypatch, "lxml")

    pandas_stub = _stub_module(monkeypatch, "pandas")
    pandas_stub.DataFrame = type("DataFrame", (), {})

    class _DummyMcMaster:
        def __init__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - defensive
            pass

    _stub_module(
        monkeypatch,
        "mcmaster_api",
        {
            "McMasterAPI": _DummyMcMaster,
            "load_env": lambda: {},
        },
    )

    _stub_module(
        monkeypatch,
        "cad_quoter.geo_dump",
        {
            "_find_hole_table_chunks": lambda _rows: ([], []),
            "_parse_header": lambda _chunks: ([], [], []),
            "_split_descriptions": lambda _body, _diam: [],
        },
    )

    import appV5

    monkeypatch.setattr(appV5, "_parse_hole_table_lines", None, raising=False)

    def _parser_guard(_lines: list[str]) -> list[dict[str, object]]:  # pragma: no cover - defensive
        raise AssertionError("Legacy chart_lines parser should not run when adapter ops are present")

    monkeypatch.setattr(appV5.geometry, "parse_hole_table_lines", _parser_guard, raising=False)

    adapter_module = importlib.import_module("cad_quoter.geometry.hole_table_adapter")

    structured_rows = [
        {"HOLE": "A", "REF_DIAM": "Ø0.257", "QTY": "4", "DESCRIPTION": "1/4-20 TAP FROM FRONT"},
        {"HOLE": "B", "REF_DIAM": "Ø0.201", "QTY": "4", "DESCRIPTION": "Ø0.201 DRILL THRU FROM FRONT"},
    ]
    adapter_ops = [
        ("A", "Ø0.257", 4, "1/4-20 TAP FROM FRONT"),
        ("B", "Ø0.201", 4, "Ø0.201 DRILL THRU FROM FRONT"),
    ]

    adapter_calls: list[object] = []

    def _adapter_stub(doc: object) -> tuple[list[dict[str, str]], list[tuple[str, str, int, str]]]:
        adapter_calls.append(doc)
        return structured_rows, adapter_ops

    monkeypatch.setattr(adapter_module, "extract_hole_table_from_doc", _adapter_stub, raising=False)

    expected_summary = appV5.aggregate_ops(
        [
            {"hole": "A", "ref": "Ø0.257", "qty": 4, "desc": "1/4-20 TAP FROM FRONT"},
            {"hole": "B", "ref": "Ø0.201", "qty": 4, "desc": "Ø0.201 DRILL THRU FROM FRONT"},
        ]
    )

    result = appV5.extract_2d_features_from_dxf_or_dwg(sample_path)

    assert adapter_calls, "Adapter stub was not invoked"

    geo = result["geo"]
    ops_summary = geo["ops_summary"]

    assert ops_summary["rows"] == expected_summary["rows"]
    assert ops_summary.get("ops_rows") == expected_summary["rows"]
    assert ops_summary["totals"] == expected_summary["totals"]
    assert ops_summary["actions_total"] == expected_summary["actions_total"]
    assert ops_summary["rows_detail"] == expected_summary["rows_detail"]
    assert ops_summary["built_rows"] == expected_summary["built_rows"]
    assert ops_summary["flip_required"] == expected_summary["flip_required"]

    assert geo["hole_table_ops"] == [
        {"HOLE": "A", "REF_DIAM": "Ø0.257", "QTY": 4, "DESCRIPTION/DEPTH": "1/4-20 TAP FROM FRONT"},
        {"HOLE": "B", "REF_DIAM": "Ø0.201", "QTY": 4, "DESCRIPTION/DEPTH": "Ø0.201 DRILL THRU FROM FRONT"},
    ]

