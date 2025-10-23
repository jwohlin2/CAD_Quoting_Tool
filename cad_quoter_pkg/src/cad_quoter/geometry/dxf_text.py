"""DXF text harvesting helpers shared across the application."""

from __future__ import annotations

import re
from typing import Iterator, List, Sequence, Tuple

from cad_quoter.vendors import ezdxf as _ezdxf_vendor

HOLE_TOKENS = re.compile(
    r"(?:\bTAP\b|C[’']?\s*BORE|CBORE|COUNTER\s*BORE|"
    r"C[’']?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|"
    r"\bJIG\s*GRIND\b|\bDRILL\s+THRU\b|\bN\.?P\.?T\.?\b)",
    re.IGNORECASE,
)


# --- Clean DXF MTEXT escapes (alignment, symbols, etc.) ----------------------
_MT_ALIGN_RE = re.compile(r"\\A\d;")      # \A1; \A0; etc.
_MT_BREAK_RE = re.compile(r"\\P", re.I)   # \P = line break
_MT_SYMS = {
    "%%C": "Ø", "%%c": "Ø",               # diameter
    "%%D": "°", "%%d": "°",               # degree
    "%%P": "±", "%%p": "±",               # plus/minus
}


def _clean_mtext(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for token, replacement in _MT_SYMS.items():
        s = s.replace(token, replacement)
    s = _MT_ALIGN_RE.sub("", s)
    s = _MT_BREAK_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _extract_insert(entity: object) -> Tuple[float, float] | None:
    try:
        insert = entity.dxf.insert  # type: ignore[attr-defined]
    except Exception:
        return None
    try:
        x = float(getattr(insert, "x", insert[0]))
        y = float(getattr(insert, "y", insert[1]))
    except Exception:
        return None
    return x, y


def _iter_segments(entity: object) -> Iterator[tuple[str, Tuple[float, float] | None]]:
    try:
        kind = entity.dxftype()
    except Exception:
        return

    if kind == "TEXT":
        try:
            text = entity.dxf.text  # type: ignore[attr-defined]
        except Exception:
            text = ""
        normalized = _normalize_text(text)
        if normalized:
            yield normalized, _extract_insert(entity)
    elif kind == "MTEXT":
        raw = ""
        try:
            raw = entity.text  # type: ignore[attr-defined]
        except Exception:
            pass
        if not raw and hasattr(entity, "plain_text"):
            try:
                raw = entity.plain_text()
            except Exception:
                raw = ""
        for segment in re.split(r"\\P|\n", raw or ""):
            cleaned = _clean_mtext(segment)
            if cleaned:
                yield cleaned, _extract_insert(entity)
    elif kind == "INSERT":
        yield from _iter_insert(entity)


def _iter_insert(entity: object) -> Iterator[tuple[str, Tuple[float, float] | None]]:
    try:
        virtual_entities = entity.virtual_entities()
    except Exception:
        virtual_entities = []

    for child in virtual_entities:
        try:
            child_kind = child.dxftype()
        except Exception:
            continue
        if child_kind == "INSERT":
            yield from _iter_insert(child)
        else:
            yield from _iter_segments(child)


def _iter_table_strings(table: object) -> Iterator[str]:
    try:
        cells = getattr(table, "cells")
    except Exception:
        cells = None

    if cells is not None:
        for cell in cells:  # type: ignore[assignment]
            try:
                text = getattr(cell, "text")
            except Exception:
                text = ""
            if callable(text):
                try:
                    text = text()
                except Exception:
                    text = ""
            normalized = _normalize_text(text)
            if normalized:
                yield normalized
        return

    try:
        nrows = int(getattr(table, "nrows", 0))
        ncols = int(getattr(table, "ncols", 0))
    except Exception:
        nrows = ncols = 0

    for row in range(max(nrows, 0)):
        for col in range(max(ncols, 0)):
            try:
                cell = table.cell(row, col)  # type: ignore[call-arg]
            except Exception:
                continue
            try:
                text = getattr(cell, "text")
            except Exception:
                text = ""
            if callable(text):
                try:
                    text = text()
                except Exception:
                    text = ""
            normalized = _normalize_text(text)
            if normalized:
                yield normalized


def _harvest_layout(
    layout: object,
    layout_index: int,
    *,
    include_tables: bool,
) -> List[tuple[Tuple[int, float, float, int], str]]:
    entries: List[tuple[Tuple[int, float, float, int], str]] = []
    counter = 0

    def _make_key(coords: Tuple[float, float] | None, order: int) -> Tuple[int, float, float, int]:
        if coords is None:
            return (layout_index, float(order + 1), float(order + 1), order)
        x, y = coords
        try:
            key_y = -float(y)
        except Exception:
            key_y = float(order)
        try:
            key_x = float(x)
        except Exception:
            key_x = float(order)
        return (layout_index, key_y, key_x, order)

    for entity in getattr(layout, "query", lambda *_: [])("TEXT"):  # type: ignore[misc]
        for text, coords in _iter_segments(entity):
            entries.append((_make_key(coords, counter), text))
            counter += 1

    for entity in getattr(layout, "query", lambda *_: [])("MTEXT"):  # type: ignore[misc]
        for text, coords in _iter_segments(entity):
            entries.append((_make_key(coords, counter), text))
            counter += 1

    for entity in getattr(layout, "query", lambda *_: [])("INSERT"):  # type: ignore[misc]
        for text, coords in _iter_insert(entity):
            entries.append((_make_key(coords, counter), text))
            counter += 1

    if include_tables:
        for table in getattr(layout, "query", lambda *_: [])("TABLE"):  # type: ignore[misc]
            for text in _iter_table_strings(table):
                entries.append((_make_key(None, counter), text))
                counter += 1

    return entries


def harvest_text_lines(doc: object, *, include_tables: bool) -> List[str]:
    lines_with_keys: List[tuple[Tuple[int, float, float, int], str]] = []

    try:
        modelspace = doc.modelspace()
    except Exception:
        modelspace = None

    if modelspace is not None:
        lines_with_keys.extend(_harvest_layout(modelspace, 0, include_tables=include_tables))

    layout_names: Sequence[str]
    try:
        layout_names = list(doc.layouts.names())
    except Exception:
        layout_names = []

    layout_order = 1
    for name in layout_names:
        if isinstance(name, str) and name.lower() == "model":
            continue
        try:
            layout = doc.layouts.get(name)
        except Exception:
            continue
        if layout is None:
            continue
        lines_with_keys.extend(
            _harvest_layout(layout, layout_order, include_tables=include_tables)
        )
        layout_order += 1

    lines_with_keys.sort(key=lambda item: item[0])

    seen: set[str] = set()
    ordered_lines: List[str] = []
    for _, text in lines_with_keys:
        if text not in seen:
            seen.add(text)
            ordered_lines.append(text)

    return ordered_lines


def extract_text_lines_from_dxf(path: str, *, include_tables: bool = False) -> List[str]:
    """Return a list of text fragments extracted from a DXF document."""

    doc = _ezdxf_vendor.read_document(path)
    return harvest_text_lines(doc, include_tables=include_tables)


__all__ = ["extract_text_lines_from_dxf", "harvest_text_lines", "HOLE_TOKENS"]

