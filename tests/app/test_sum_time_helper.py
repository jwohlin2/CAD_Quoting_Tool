import re
import sys
import types

import pytest

if "cad_quoter.geometry" not in sys.modules:
    geom_stub = types.ModuleType("cad_quoter.geometry")
    sys.modules["cad_quoter.geometry"] = geom_stub
else:  # pragma: no cover - ensure attribute exists when module already imported
    geom_stub = sys.modules["cad_quoter.geometry"]

setattr(geom_stub, "FACE_OF", getattr(geom_stub, "FACE_OF", {}))


def _ensure_face(face, *_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return face


def _face_surface(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return None


def _iter_faces(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return iter(())


def _linear_properties(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return {}


def _map_shapes_and_ancestors(*_args, **_kwargs):  # pragma: no cover - simple shim for tests
    return {}


setattr(geom_stub, "ensure_face", getattr(geom_stub, "ensure_face", _ensure_face))
setattr(geom_stub, "face_surface", getattr(geom_stub, "face_surface", _face_surface))
setattr(geom_stub, "iter_faces", getattr(geom_stub, "iter_faces", _iter_faces))
setattr(geom_stub, "linear_properties", getattr(geom_stub, "linear_properties", _linear_properties))
setattr(
    geom_stub,
    "map_shapes_and_ancestors",
    getattr(geom_stub, "map_shapes_and_ancestors", _map_shapes_and_ancestors),
)

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


def test_sum_time_uses_default_when_only_blank_values() -> None:
    items = _build_series(["In-Process Inspection Hours"])
    vals = _build_series([""])
    types = _build_series(["number"])

    result = _sum_time(items, vals, types, r"In-Process Inspection", default=1.0)

    assert result == pytest.approx(1.0, rel=1e-6)


def test_sum_time_respects_explicit_zero_values() -> None:
    items = _build_series(["In-Process Inspection Hours"])
    vals = _build_series(["0"])
    types = _build_series(["number"])

    result = _sum_time(items, vals, types, r"In-Process Inspection", default=1.0)

    assert result == pytest.approx(0.0, rel=1e-6)


def test_sum_time_converts_minutes_to_hours() -> None:
    items = _build_series(["Inspection Minutes"])
    vals = _build_series(["30"])
    types = _build_series(["number"])

    result = _sum_time(items, vals, types, r"Inspection", default=0.0)

    assert result == pytest.approx(0.5, rel=1e-6)
