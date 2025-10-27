from __future__ import annotations

import copy
import importlib
import sys
import types
from typing import Iterable

import pytest

from tests.pricing.test_dummy_quote_acceptance import _dummy_quote_payload


EXPECTED_MATERIAL_SECTION: list[str] = [
    "Material & Stock",
    "--------------------------------------------------------------------------",
    "  Material used:  Aluminum MIC6",
    "  Starting Weight: 217 lb 2.5 oz",
    "  Net Weight: 201 lb 11.6 oz",
    "  Scrap Weight: 15 lb 3.2 oz",
    "  Scrap Percentage: 7.0% (computed)",
    "  Scrap % (geometry hint): 7.0%",
    "  Base Material @ per-lb @ $0.00/lb                                $420.00",
    "                                                                   -------",
    "  Total Material Cost :                                            $420.00",
    "",
]


def _extract_material_section(text: str) -> list[str]:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == "Material & Stock":
            start = index
            break
    if start is None:
        return []
    collected: list[str] = []
    for line in lines[start:]:
        collected.append(line)
        if line.strip() == "" and len(collected) > 1:
            break
    return collected


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide stub modules expected by :mod:`appV5`."""

    def _stub_module(name: str, attrs: Iterable[tuple[str, object]] = ()) -> None:
        module = types.ModuleType(name)
        module.__spec__ = types.SimpleNamespace()  # type: ignore[attr-defined]
        for attr_name, value in attrs:
            setattr(module, attr_name, value)
        monkeypatch.setitem(sys.modules, name, module)

    class _DummySession:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial stub
            pass

        def mount(self, *args, **kwargs) -> None:  # pragma: no cover - trivial stub
            pass

    _stub_module("requests", (("Session", _DummySession),))
    _stub_module("bs4")
    _stub_module("lxml")
    _stub_module("requests_pkcs12", (("Pkcs12Adapter", lambda **_kwargs: None),))


def _load_appv5(monkeypatch: pytest.MonkeyPatch):
    _install_runtime_stubs(monkeypatch)
    import appV5  # noqa: F401 - imported for side effects

    return importlib.reload(sys.modules["appV5"])


def test_material_section_matches_legacy_output(monkeypatch: pytest.MonkeyPatch) -> None:
    appV5 = _load_appv5(monkeypatch)
    payload = _dummy_quote_payload()
    rendered = appV5.render_quote(copy.deepcopy(payload), currency="$", show_zeros=False)
    section = _extract_material_section(rendered)
    assert section == EXPECTED_MATERIAL_SECTION


def test_material_section_absent_when_no_material(monkeypatch: pytest.MonkeyPatch) -> None:
    appV5 = _load_appv5(monkeypatch)
    payload = _dummy_quote_payload()
    payload["breakdown"]["material"] = {}
    payload["breakdown"].pop("material_block", None)
    payload["breakdown"].pop("material_selected", None)
    payload["breakdown"].pop("materials", None)
    rendered = appV5.render_quote(copy.deepcopy(payload), currency="$", show_zeros=False)
    section = _extract_material_section(rendered)
    assert section == []
