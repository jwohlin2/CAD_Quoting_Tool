import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PKG_SRC = PROJECT_ROOT / "cad_quoter_pkg" / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

try:  # pragma: no cover - test bootstrap helper
    import cad_quoter.geometry.dxf_enrich  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - load shim for tests
    spec = importlib.util.spec_from_file_location(
        "cad_quoter.geometry.dxf_enrich",
        PROJECT_ROOT
        / "cad_quoter_pkg"
        / "src"
        / "cad_quoter"
        / "geometry"
        / "dxf_enrich.py",
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        import cad_quoter.geometry as _geometry_pkg  # type: ignore[import-not-found]

        setattr(_geometry_pkg, "dxf_enrich", module)

from appV5 import _count_counterdrill, _count_jig  # noqa: E402  # pylint: disable=wrong-import-position
from cad_quoter.app.op_parser import (  # noqa: E402  # pylint: disable=wrong-import-position
    _parse_ops_and_claims,
)


@pytest.mark.parametrize(
    "lines, expected",
    [
        (["(3) COUNTERDRILL"], 3),
        (["(2) C DRILL"], 2),
        (["(4) Center Drill", "(1) Spot Drill"], 0),
        (["COUNTERDRILL"], 1),
        (["(5) COUNTER DRILL", "(2) center drill"], 5),
        (["(2) C’ DRILL"], 2),
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
