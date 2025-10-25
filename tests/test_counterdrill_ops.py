import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - test bootstrap helper
    import cad_quoter.geometry.dxf_enrich  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - load shim for tests
    geometry_dir = PROJECT_ROOT / "src" / "cad_quoter" / "geometry"

    for _module_name in ("dxf_text", "dxf_enrich"):
        spec = importlib.util.spec_from_file_location(
            f"cad_quoter.geometry.{_module_name}",
            geometry_dir / f"{_module_name}.py",
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]

    import cad_quoter.geometry as _geometry_pkg  # type: ignore[import-not-found]

    for _module_name in ("dxf_text", "dxf_enrich"):
        module = sys.modules.get(f"cad_quoter.geometry.{_module_name}")
        if module is not None:
            setattr(_geometry_pkg, _module_name, module)

from appV5 import (  # noqa: E402  # pylint: disable=wrong-import-position
    _append_counterbore_spot_jig_cards,
    _count_counterdrill,
    _count_jig,
    _count_spot_and_jig,
)
from cad_quoter.ui.bucket_ops import (  # noqa: E402  # pylint: disable=wrong-import-position
    _append_counterdrill_extra,
    _publish_extra_bucket_op,
    COUNTERDRILL_MIN_PER_SIDE_MIN,
    JIG_GRIND_MIN_PER_FEATURE,
)
from cad_quoter.app.op_parser import (  # noqa: E402  # pylint: disable=wrong-import-position
    _parse_ops_and_claims,
)
from cad_quoter.app.chart_lines import (  # noqa: E402  # pylint: disable=wrong-import-position
    build_ops_rows_from_lines_fallback as _build_chart_rows_from_lines,
)


@pytest.mark.parametrize(
    "lines, expected",
    [
        (["(3) COUNTERDRILL"], 3),
        (["(2) C DRILL"], 0),
        (["(4) Center Drill", "(1) Spot Drill"], 0),
        (["COUNTERDRILL"], 1),
        (["(5) COUNTER DRILL", "(2) center drill"], 5),
        (["(2) C’ DRILL"], 2),
        (["(2) Counter-Drill"], 2),
        (["(2) CTR DRILL"], 2),
        (["DRILL THRU"], 0),
        (["C DRILL THRU"], 0),
        (["Counterdrill spot drill"], 0),
    ],
)
def test_count_counterdrill(lines: list[str], expected: int) -> None:
    assert _count_counterdrill(lines) == expected


@pytest.mark.parametrize(
    "lines, expected",
    [
        (["(2) JIG GRIND"], 2),
        (["JIG GRIND"], 1),
        (["(3) jig grind", "(5) spot"], 3),
    ],
)
def test_count_jig(lines: list[str], expected: int) -> None:
    assert _count_jig(lines) == expected


def test_parse_ops_and_claims_counterdrill_and_jig() -> None:
    lines = ["(4) Counterdrill", "(2) Jig Grind"]
    claims = _parse_ops_and_claims(lines)
    assert claims["counterdrill"] == 4
    assert claims["jig"] == 2


def test_parse_ops_and_claims_skips_center_for_counterdrill() -> None:
    lines = ["(2) Center Drill", "(1) Counterdrill"]
    claims = _parse_ops_and_claims(lines)
    assert claims["counterdrill"] == 1
    assert claims["spot"] >= 2


def test_count_spot_and_jig_skips_counterdrill_rows() -> None:
    rows = [
        {"qty": 1, "desc": "Counterdrill"},
        {"qty": 2, "desc": "Spot Drill"},
        {"qty": 3, "desc": "Jig Grind"},
    ]
    spot, jig = _count_spot_and_jig(rows)
    assert spot == 2
    assert jig == 3


def test_append_cards_publish_spot_without_counterbore() -> None:
    lines_out: list[str] = []
    breakdown: dict[str, object] = {}
    appended = _append_counterbore_spot_jig_cards(
        lines_out=lines_out,
        chart_lines=None,
        rows=[{"qty": 4, "desc": "Spot Drill"}],
        breakdown_mutable=breakdown,
        rates={},
    )
    assert appended >= 1
    extra_bucket_ops = breakdown.get("extra_bucket_ops")
    assert isinstance(extra_bucket_ops, dict)
    assert not extra_bucket_ops.get("counterbore")
    spot_entries = extra_bucket_ops.get("spot") if isinstance(extra_bucket_ops, dict) else None
    assert isinstance(spot_entries, list)
    assert spot_entries[-1]["qty"] == 4
    assert spot_entries[-1]["minutes"] == pytest.approx(0.2)


def test_append_cards_publish_jig_minutes() -> None:
    lines_out: list[str] = []
    breakdown: dict[str, object] = {}
    appended = _append_counterbore_spot_jig_cards(
        lines_out=lines_out,
        chart_lines=None,
        rows=[{"qty": 2, "desc": "Jig Grind"}],
        breakdown_mutable=breakdown,
        rates={},
    )
    assert appended >= 1
    extra_bucket_ops = breakdown.get("extra_bucket_ops")
    assert isinstance(extra_bucket_ops, dict)
    jig_entries = extra_bucket_ops.get("jig-grind") if isinstance(extra_bucket_ops, dict) else None
    assert isinstance(jig_entries, list)
    jig_entry = jig_entries[-1]
    assert jig_entry["qty"] == 2
    assert "minutes" not in jig_entry

    bucket_view = breakdown.get("bucket_view")
    assert isinstance(bucket_view, dict)
    buckets = bucket_view.get("buckets") if isinstance(bucket_view, dict) else None
    assert isinstance(buckets, dict)
    grinding_bucket = buckets.get("grinding") if isinstance(buckets, dict) else None
    assert isinstance(grinding_bucket, dict)
    per_jig = float(JIG_GRIND_MIN_PER_FEATURE or 0.75)
    assert grinding_bucket.get("minutes") == pytest.approx(2 * per_jig)


def test_append_counterdrill_extra_sets_bucket_minutes() -> None:
    breakdown: dict[str, object] = {}
    minutes = _append_counterdrill_extra(breakdown, 3, rates={})
    per_counterdrill = float(COUNTERDRILL_MIN_PER_SIDE_MIN or 0.12)
    assert minutes == pytest.approx(3 * per_counterdrill)

    extra_bucket_ops = breakdown.get("extra_bucket_ops")
    assert isinstance(extra_bucket_ops, dict)
    counterdrill_entries = extra_bucket_ops.get("counterdrill")
    assert isinstance(counterdrill_entries, list)
    counterdrill_entry = counterdrill_entries[-1]
    assert counterdrill_entry["qty"] == 3
    assert "minutes" not in counterdrill_entry

    bucket_view = breakdown.get("bucket_view")
    assert isinstance(bucket_view, dict)
    buckets = bucket_view.get("buckets") if isinstance(bucket_view, dict) else None
    assert isinstance(buckets, dict)
    counterdrill_bucket = buckets.get("counterdrill") if isinstance(buckets, dict) else None
    assert isinstance(counterdrill_bucket, dict)
    assert counterdrill_bucket.get("minutes") == pytest.approx(minutes)


def test_publish_extra_bucket_op_merges_minutes() -> None:
    ebo: dict[str, list[dict]] = {}
    _publish_extra_bucket_op(
        ebo,
        "counterdrill",
        {"name": "Counterdrill", "qty": 2, "side": "front"},
        minutes=0.24,
    )
    _publish_extra_bucket_op(
        ebo,
        "counterdrill",
        {"name": "Counterdrill", "qty": 3, "side": "front"},
        minutes=0.36,
    )
    entries = ebo.get("counterdrill")
    assert isinstance(entries, list)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["qty"] == 3
    assert entry["minutes"] == pytest.approx(0.36)


def test_chart_fallback_extracts_counterdrill_rows() -> None:
    rows = _build_chart_rows_from_lines(["(3) Counterdrill"])
    assert any(row.get("desc", "").upper().startswith("COUNTERDRILL") for row in rows)
    assert any(row.get("qty") == 3 for row in rows)
