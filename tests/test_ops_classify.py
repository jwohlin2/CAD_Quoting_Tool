"""Tests for the public operations classifier helpers."""

from __future__ import annotations

from typing import Any

import pytest

ops_classify = pytest.importorskip("cad_quoter.ops.classify")

_CLASSIFY_OPERATION = getattr(ops_classify, "classify_operation", None)

if _CLASSIFY_OPERATION is None:
    pytest.skip("operation classifier helper unavailable", allow_module_level=True)


def _extract_label(result: Any) -> str:
    if hasattr(result, "label"):
        label = getattr(result, "label")
        if isinstance(label, str):
            return label
    if isinstance(result, tuple) and result:
        candidate = result[0]
        if isinstance(candidate, str):
            return candidate
    if isinstance(result, str):
        return result
    raise AssertionError(f"Unrecognised classifier result: {result!r}")


def _extract_normalized(result: Any) -> str | None:
    if hasattr(result, "normalized"):
        normalized = getattr(result, "normalized")
        if isinstance(normalized, str):
            return normalized
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], str):
        return result[1]
    return None


@pytest.mark.parametrize(
    "desc, expected",
    [
        ("1/4-20 TAP THRU", "Tap"),
        ("Ø0.250 DRILL THRU", "Drill"),
        ("Ø0.500 C’BORE FROM BACK", "C'Bore"),
        ("SPOT DRILL Ø0.312", "C'Drill/CSink"),
        ("FINAL JIG GRIND", "Jig Grind"),
    ],
)
def test_operation_classifier_labels(desc: str, expected: str) -> None:
    result = _CLASSIFY_OPERATION(desc)

    label = _extract_label(result)
    normalized = _extract_normalized(result)

    assert label == expected
    if normalized is not None:
        assert isinstance(normalized, str)
        assert normalized
