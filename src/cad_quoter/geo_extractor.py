"""Isolated GEO extraction helpers for DWG/DXF sources."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
import inspect
import math
from functools import lru_cache
import os
from pathlib import Path
import re
import statistics
from typing import Any, Callable

from cad_quoter import geometry
from cad_quoter.geometry import convert_dwg_to_dxf
from cad_quoter.vendors import ezdxf as _ezdxf_vendor

_HAS_ODAFC = bool(getattr(geometry, "HAS_ODAFC", False))

_DEFAULT_LAYER_ALLOWLIST = frozenset({"BALLOON"})
_PREFERRED_BLOCK_NAME_RE = re.compile(r"HOLE.*(?:CHART|TABLE)", re.IGNORECASE)

_LAST_ACAD_TABLE_SCAN: dict[str, Any] | None = None


def _normalize_block_allowlist(
    block_allowlist: Iterable[str] | None,
) -> set[str]:
    if block_allowlist is None:
        return set()
    normalized: set[str] = set()
    for value in block_allowlist:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        normalized.add(text.upper())
    return normalized


def _compile_block_name_patterns(
    block_patterns: Iterable[str] | str | None,
) -> list[re.Pattern[str]]:
    if block_patterns is None:
        return []
    if isinstance(block_patterns, str):
        candidates: Iterable[str] = [block_patterns]
    else:
        candidates = block_patterns
    compiled: list[re.Pattern[str]] = []
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            compiled.append(re.compile(text, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _gather_entity_points(entity: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def _append_value(value: Any) -> None:
        if value is None:
            return
        if hasattr(value, "x") and hasattr(value, "y"):
            try:
                points.append((float(value.x), float(value.y)))
            except Exception:
                return
            return
        if isinstance(value, (tuple, list)):
            if len(value) >= 2:
                try:
                    points.append((float(value[0]), float(value[1])))
                except Exception:
                    pass
            for item in value:
                _append_value(item)

    for source in (getattr(entity, "dxf", None), entity):
        if source is None:
            continue
        for attr in (
            "insert",
            "alignment_point",
            "start",
            "end",
            "center",
            "defpoint",
            "base_point",
            "location",
        ):
            _append_value(getattr(source, attr, None))
    try:
        iterator = iter(entity)
    except Exception:
        iterator = None
    if iterator is not None:
        for item in iterator:
            _append_value(getattr(item, "dxf", None))
            _append_value(item)
    return points


def _compute_entity_bbox(
    entity: Any,
    *,
    include_virtual: bool = False,
    virtual_entities: Iterable[Any] | None = None,
) -> tuple[float, float, float, float] | None:
    points = _gather_entity_points(entity)
    if include_virtual:
        if virtual_entities is None:
            try:
                virtual_entities = list(entity.virtual_entities())
            except Exception:
                virtual_entities = []
        for child in virtual_entities or []:
            points.extend(_gather_entity_points(child))
    if not points:
        return None
    xs = [pt[0] for pt in points if isinstance(pt[0], (int, float))]
    ys = [pt[1] for pt in points if isinstance(pt[1], (int, float))]
    if not xs or not ys:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def _normalize_layer_allowlist(
    layer_allowlist: Iterable[str] | None,
) -> set[str] | None:
    if layer_allowlist is None:
        return None
    special_tokens = {"ALL", "*", "<ALL>"}
    normalized: set[str] = set()
    for value in layer_allowlist:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        upper = text.upper()
        if upper in special_tokens:
            return None
        normalized.add(upper)
    if not normalized:
        return set()
    return normalized


@lru_cache(maxsize=1)
def _load_app_module():
    import importlib

    return importlib.import_module("appV5")


@lru_cache(maxsize=None)
def _resolve_app_callable(name: str) -> Callable[..., Any] | None:
    try:
        module = _load_app_module()
    except Exception:
        return None
    return getattr(module, name, None)


def _describe_helper(helper: Any) -> str:
    if helper is None:
        return "None"
    name = getattr(helper, "__name__", None)
    if isinstance(name, str):
        return name
    return repr(helper)


def _print_helper_debug(tag: str, helper: Any) -> None:
    try:
        helper_desc = _describe_helper(helper)
    except Exception:
        helper_desc = repr(helper)
    print(f"[EXTRACT] {tag} helper: {helper_desc}")


def _debug_entities_enabled() -> bool:
    value = os.environ.get("CAD_QUOTER_DEBUG_ENTITIES", "").strip().lower()
    if not value:
        return False
    return value not in {"0", "false", "no"}


def _split_mtext_plain_text(text: Any) -> list[str]:
    if text is None:
        return []
    try:
        raw = str(text)
    except Exception:
        raw = text if isinstance(text, str) else ""
    if not raw:
        return []
    candidate = raw.replace("\r\n", "\n").replace("\r", "\n")
    candidate = _MTEXT_BREAK_RE.sub("\n", candidate)
    parts = []
    for piece in candidate.split("\n"):
        cleaned = piece.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


_HOLE_ACTION_TOKEN_PATTERN = (
    r"(Ø|⌀|C['’]?BORE|COUNTER\s*BORE|DRILL|TAP|N\.?P\.?T|NPT|THRU|JIG\s*GRIND)"
)
_ROW_QUANTITY_PATTERNS = [
    re.compile(r"^\(\s*(\d+)\s*\)", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*[x×]\b", re.IGNORECASE),
    re.compile(r"^\s*(?:QTY|QTY\.|QTY:)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*(?:REQD|REQUIRED|RE'?D)\b", re.IGNORECASE),
    re.compile(rf"^\s*(\d+)\b(?=.*{_HOLE_ACTION_TOKEN_PATTERN})", re.IGNORECASE),
]
_ROW_QUANTITY_FLEX_PATTERNS = [
    re.compile(r"\b(\d+)\s*[x×]\b", re.IGNORECASE),
    re.compile(r"\b(?:QTY|QTY\.|QTY:)\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*(?:REQD|REQUIRED|RE'?D)\b", re.IGNORECASE),
    re.compile(r"\(\s*(\d+)\s*\)"),
]
_LETTER_CODE_ROW_RE = re.compile(r"^\s*[A-Z]\s*(?:[-.:|]|$)")
_HOLE_ACTION_TOKEN_RE = re.compile(_HOLE_ACTION_TOKEN_PATTERN, re.IGNORECASE)
_DIAMETER_PREFIX_RE = re.compile(
    r"(?:Ø|⌀|DIA(?:\.\b|\b))\s*(\d+\s*/\s*\d+|\d*\.\d+|\.\d+|\d+)",
    re.IGNORECASE,
)
_DIAMETER_SUFFIX_RE = re.compile(
    r"(\d+\s*/\s*\d+|\d*\.\d+|\.\d+|\d+)\s*(?:Ø|⌀|DIA(?:\.\b|\b))",
    re.IGNORECASE,
)
_MTEXT_ALIGN_RE = re.compile(r"\\A\d;", re.IGNORECASE)
_MTEXT_BREAK_RE = re.compile(r"\\P", re.IGNORECASE)
_CANDIDATE_TOKEN_RE = re.compile(
    r"(TAP\b|DRILL\b|THRU\b|N\.P\.T\b|NPT\b|C['’]?BORE\b|COUNTER\s*BORE\b|"
    r"JIG\s+GRIND\b|AS\s+SHOWN\b|FROM\s+BACK\b|FROM\s+FRONT\b|BOTH\s+SIDES\b)",
    re.IGNORECASE,
)
_COLUMN_TOKEN_RE = re.compile(
    r"(TAP|DRILL|THRU|C['’]?BORE|COUNTER\s*BORE|N\.?P\.?T|NPT|Ø|JIG)",
    re.IGNORECASE,
)
_QSTRIPE_CANDIDATE_RE = re.compile(
    r"(^\(?\d{1,3}\)?$|^\d{1,3}[x×]$|^QTY[:.]?$)",
    re.IGNORECASE,
)
_ROI_ANCHOR_RE = re.compile(
    r"(HOLE\s+CHART|HOLE\s+TABLE|QTY|SIZE|DIA|Ø|⌀|TAP|DRILL|THRU|C['’]?BORE|"
    r"COUNTER\s*BORE|N\.?P\.?T|JIG)",
    re.IGNORECASE,
)
_TITLE_AXIS_DROP_RE = re.compile(
    r"(GENTITLE|TITLE|DRAWING|SHEET|SCALE|REV|DWG|DATE)",
    re.IGNORECASE,
)
_SEE_SHEET_DROP_RE = re.compile(r"(SEE\s+SHEET|SEE\s+DETAIL)", re.IGNORECASE)
_AXIS_ZERO_PAIR_RE = re.compile(r"^[A-Z]\s+[A-Z]\s+0\.0{3,}\b")
_AXIS_ZERO_SINGLE_RE = re.compile(r"^0\.0{3,}\s+[XY]\b", re.IGNORECASE)
_SMALL_INT_TOKEN_RE = re.compile(r"\b\d+\b")
_FRACTION_RE = re.compile(r"\b\d+\s*/\s*\d+\b")
_DECIMAL_RE = re.compile(r"\b(?:\d+\.\d+|\.\d+)\b")
_DECIMAL_3PLUS_RE = re.compile(r"\b\d+\.\d{3,}\b")
_BAND_KEEP_TOKEN_RE = re.compile(
    r"(Ø|⌀|TAP|DRILL|C['’]?BORE|COUNTER\s*BORE|CSINK|N\.?P\.?T|THREAD|#\d+-\d+|\d/\d|\d\.\d{3,})",
    re.IGNORECASE,
)
_MAX_INSERT_DEPTH = 3

_BAND_QTY_FALLBACK_PATTERNS = [
    re.compile(r"^\(\s*(?P<qty>\d+)\s*\)"),
    re.compile(r"(^|\s)(?P<qty>\d+)\s*(?:X|×)(\s|$)", re.IGNORECASE),
    re.compile(r"(^|\s)QTY[:=\s]*(?P<qty>\d+)(\s|$)", re.IGNORECASE),
    re.compile(r"(\s|^)RE(?:Q'D|QD|QUIRED)[:=\s]*(?P<qty>\d+)(\s|$)", re.IGNORECASE),
]

_LAST_TEXT_TABLE_DEBUG: dict[str, Any] | None = None


def _score_table(info: Mapping[str, Any] | None) -> tuple[int, int]:
    if not isinstance(info, Mapping):
        return (0, 0)
    rows = info.get("rows") or []
    return (_sum_qty(rows), len(rows))


def _sum_qty(rows: Iterable[Mapping[str, Any]] | None) -> int:
    total = 0
    if not rows:
        return total
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        qty_val = row.get("qty")
        try:
            total += int(float(qty_val or 0))
        except Exception:
            continue
    return total


def read_acad_table(
    doc, layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST
) -> dict[str, Any]:
    helper = _resolve_app_callable("hole_count_from_acad_table")
    _print_helper_debug("acad", helper)
    if callable(helper):
        try:
            result = helper(doc) or {}
        except Exception as exc:
            print(f"[EXTRACT] acad helper error: {exc}")
            raise
        if isinstance(result, Mapping):
            return dict(result)
        return {}

    allowlist = _normalize_layer_allowlist(layer_allowlist)
    global _LAST_ACAD_TABLE_SCAN

    layouts_scanned: set[str] = set()
    blocks_scanned: set[str] = set()
    layout_total_count = 0
    block_total_count = 0
    seen_table_handles: set[str] = set()
    tables_found = 0
    table_candidates: list[dict[str, Any]] = []
    scan_tables: list[dict[str, Any]] = []

    if doc is None:
        print("[ACAD-TABLE] scanned layouts=0 blocks=0 tables_found=0")
        _LAST_ACAD_TABLE_SCAN = {
            "layouts": [],
            "blocks": [],
            "tables_found": 0,
            "tables": [],
        }
        return {}

    if doc is not None:
        layouts_attr = getattr(doc, "layouts", None)
        if layouts_attr is not None:
            try:
                layout_total_count = len(layouts_attr)  # type: ignore[arg-type]
            except Exception:
                layout_total_count = 0
                names_attr = getattr(layouts_attr, "names", None)
                try:
                    if callable(names_attr):
                        layout_total_count = len(list(names_attr()))
                    elif names_attr is not None:
                        layout_total_count = len(list(names_attr))
                except Exception:
                    layout_total_count = 0
        blocks_attr = getattr(doc, "blocks", None)
        if blocks_attr is not None:
            try:
                block_total_count = len(blocks_attr)  # type: ignore[arg-type]
            except Exception:
                block_total_count = 0
                keys_attr = getattr(blocks_attr, "keys", None)
                try:
                    if callable(keys_attr):
                        block_total_count = len(list(keys_attr()))
                    elif keys_attr is not None:
                        block_total_count = len(list(keys_attr))
                except Exception:
                    try:
                        block_total_count = len(list(blocks_attr))
                    except Exception:
                        block_total_count = 0

    def _iter_layout_containers() -> Iterable[tuple[str, Any, bool]]:
        seen_layouts: set[int] = set()
        modelspace = getattr(doc, "modelspace", None)
        if callable(modelspace):
            try:
                layout_obj = modelspace()
            except Exception:
                layout_obj = None
            if layout_obj is not None:
                marker = id(layout_obj)
                if marker not in seen_layouts:
                    seen_layouts.add(marker)
                    yield ("Model", layout_obj, False)

        layouts_manager = getattr(doc, "layouts", None)
        if layouts_manager is not None:
            try:
                raw_names = getattr(layouts_manager, "names", None)
                if callable(raw_names):
                    names_iter = raw_names()
                else:
                    names_iter = raw_names
                layout_names = list(names_iter or [])
            except Exception:
                layout_names = []
            get_layout = getattr(layouts_manager, "get", None)
            for raw_name in layout_names:
                if not isinstance(raw_name, str):
                    continue
                name = raw_name.strip() or raw_name
                if name.lower() == "model":
                    continue
                layout_obj = None
                if callable(get_layout):
                    try:
                        layout_obj = get_layout(raw_name)
                    except Exception:
                        layout_obj = None
                if layout_obj is None:
                    continue
                marker = id(layout_obj)
                if marker in seen_layouts:
                    continue
                seen_layouts.add(marker)
                yield (name, layout_obj, False)

        blocks = getattr(doc, "blocks", None)
        if blocks is None:
            return
        try:
            keys = getattr(blocks, "keys", None)
            if callable(keys):
                block_names = list(keys())
            else:
                block_names = list(keys or [])
        except Exception:
            block_names = []
        if not block_names:
            try:
                block_names = [block.name for block in blocks]  # type: ignore[attr-defined]
            except Exception:
                block_names = []
        get_block = getattr(blocks, "get", None)
        for raw_name in block_names:
            if not isinstance(raw_name, str):
                continue
            name = raw_name.strip()
            if not name:
                continue
            upper = name.upper()
            if name.startswith("*") or name.startswith("|") or upper.startswith("AC$"):
                continue
            block_layout = None
            if callable(get_block):
                try:
                    block_layout = get_block(raw_name)
                except Exception:
                    block_layout = None
            if block_layout is None:
                block_layout = getattr(blocks, name, None)
            if block_layout is None:
                continue
            marker = id(block_layout)
            if marker in seen_layouts:
                continue
            seen_layouts.add(marker)
            yield (name, block_layout, True)

    def _iter_tables(container: Any) -> Iterable[Any]:
        seen_entities: set[int] = set()
        query = getattr(container, "query", None)
        if callable(query):
            for spec in ("ACAD_TABLE", "TABLE"):
                try:
                    candidates = list(query(spec))
                except Exception:
                    candidates = []
                for entity in candidates:
                    if entity is None:
                        continue
                    marker = id(entity)
                    if marker in seen_entities:
                        continue
                    seen_entities.add(marker)
                    yield entity
        try:
            iterator = iter(container)
        except Exception:
            iterator = None
        if iterator is not None:
            for entity in iterator:
                if entity is None:
                    continue
                try:
                    dxftype = entity.dxftype()
                except Exception:
                    dxftype = None
                if str(dxftype or "").upper() not in {"ACAD_TABLE", "TABLE"}:
                    continue
                marker = id(entity)
                if marker in seen_entities:
                    continue
                seen_entities.add(marker)
                yield entity

    qty_digit_re = re.compile(r"^\d{1,3}$")
    qty_decimal_re = re.compile(r"\d+\.\d+")

    def _parse_qty_cell_text(text: str) -> int | None:
        candidate = (text or "").strip()
        if not candidate:
            return None
        for pattern in _BAND_QTY_FALLBACK_PATTERNS:
            match = pattern.search(candidate)
            if not match:
                continue
            groupdict = match.groupdict()
            qty_text = groupdict.get("qty") if groupdict else None
            if not qty_text and match.groups():
                qty_text = match.group(1)
            if not qty_text:
                continue
            try:
                return int(qty_text)
            except Exception:
                continue
        stripped = re.sub(r"^\s*(?:QTY[:.=]?\s*)", "", candidate, flags=re.IGNORECASE)
        stripped = re.sub(r"[()\[\]]", "", stripped)
        stripped = re.sub(r"\b(?:EA|EACH|HOLES?|REQD|REQUIRED)\b", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"[X×]\s*$", "", stripped, flags=re.IGNORECASE).strip()
        if qty_digit_re.match(stripped):
            try:
                value = int(stripped)
            except Exception:
                value = None
            if value is not None and value > 0:
                return value
        if qty_decimal_re.search(candidate):
            return None
        loose_match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", candidate)
        if loose_match:
            try:
                return int(loose_match.group(1))
            except Exception:
                return None
        return None

    def _table_dimension(entity: Any, names: tuple[str, ...]) -> int | None:
        dxf_obj = getattr(entity, "dxf", None)
        for name in names:
            for source in (entity, dxf_obj):
                if source is None:
                    continue
                value = getattr(source, name, None)
                if value is None:
                    continue
                try:
                    return int(float(value))
                except Exception:
                    continue
        return None

    def _cell_text(entity: Any, row: int, col: int) -> str:
        text_value = ""
        for method_name in ("text_cell_content", "cell_content"):
            method = getattr(entity, method_name, None)
            if not callable(method):
                continue
            try:
                candidate = method(row, col)
            except Exception:
                continue
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)):
                candidate = " ".join(str(part) for part in candidate if part is not None)
            try:
                text_value = str(candidate)
            except Exception:
                text_value = candidate if isinstance(candidate, str) else ""
            if text_value:
                return _normalize_table_fragment(text_value)

        get_cell = getattr(entity, "get_cell", None)
        cell_obj = None
        if callable(get_cell):
            try:
                cell_obj = get_cell(row, col)
            except Exception:
                cell_obj = None
        if cell_obj is not None:
            for attr in ("get_text", "get_plain_text", "get_text_string"):
                method = getattr(cell_obj, attr, None)
                if not callable(method):
                    continue
                try:
                    candidate = method() or ""
                except Exception:
                    continue
                if isinstance(candidate, (list, tuple)):
                    candidate = " ".join(str(part) for part in candidate if part is not None)
                try:
                    text_value = str(candidate)
                except Exception:
                    text_value = candidate if isinstance(candidate, str) else ""
                if text_value:
                    break
            if not text_value:
                for attr in ("text", "plain_text", "value", "content"):
                    raw = getattr(cell_obj, attr, None)
                    if raw is None:
                        continue
                    if callable(raw):
                        try:
                            raw = raw()
                        except Exception:
                            continue
                    try:
                        text_value = str(raw)
                    except Exception:
                        text_value = raw if isinstance(raw, str) else ""
                    if text_value:
                        break
        if not text_value:
            for method_name in (
                "get_cell_text",
                "get_display_text",
                "get_text_with_formatting",
                "get_text",
                "cell_text",
            ):
                method = getattr(entity, method_name, None)
                if not callable(method):
                    continue
                try:
                    candidate = method(row, col) or ""
                except Exception:
                    continue
                if isinstance(candidate, (list, tuple)):
                    candidate = " ".join(str(part) for part in candidate if part is not None)
                try:
                    text_value = str(candidate)
                except Exception:
                    text_value = candidate if isinstance(candidate, str) else ""
                if text_value:
                    break
        return _normalize_table_fragment(text_value)

    header_token_re = re.compile(
        r"(QTY|QUANTITY|DESC|DESCRIPTION|REF|DIA|Ø|⌀|HOLE|ID|SIDE)",
        re.IGNORECASE,
    )

    def _detect_header_hits(cells: list[str]) -> dict[str, int]:
        hits: dict[str, int] = {}
        for idx, cell in enumerate(cells):
            if not cell:
                continue
            upper = cell.upper()
            if "QTY" in upper or "QUANTITY" in upper:
                hits.setdefault("qty", idx)
            if "DESC" in upper or "DESCRIPTION" in upper:
                hits.setdefault("desc", idx)
            if any(token in upper for token in ("Ø", "⌀", "DIA", "REF")):
                hits.setdefault("ref", idx)
            if "HOLE" in upper or re.search(r"\bID\b", upper):
                hits.setdefault("hole", idx)
            if "SIDE" in upper or "FACE" in upper:
                hits.setdefault("side", idx)
        if not hits:
            combined = " ".join(cells)
            if not header_token_re.search(combined):
                return {}
        return hits

    def _compute_table_bbox(entity: Any) -> tuple[float, float, float, float] | None:
        try:
            virtual_entities = list(entity.virtual_entities())
        except Exception:
            virtual_entities = []
        return _compute_entity_bbox(
            entity,
            include_virtual=True,
            virtual_entities=virtual_entities,
        )

    def _estimate_text_height(entity: Any, n_rows: int) -> float:
        heights: list[float] = []
        dxf_obj = getattr(entity, "dxf", None)
        if dxf_obj is not None:
            for attr in ("text_height", "char_height", "height"):
                value = getattr(dxf_obj, attr, None)
                if value is None:
                    continue
                try:
                    height_val = float(value)
                except Exception:
                    continue
                if height_val > 0:
                    heights.append(height_val)
        get_row_height = getattr(entity, "get_row_height", None)
        if callable(get_row_height):
            sample_rows = min(max(int(n_rows or 0), 0), 20)
            for idx in range(sample_rows):
                try:
                    row_height = get_row_height(idx)
                except Exception:
                    continue
                try:
                    height_val = float(row_height)
                except Exception:
                    continue
                if height_val > 0:
                    heights.append(height_val)
        if heights:
            try:
                return float(statistics.median(heights))
            except Exception:
                pass
        return 0.0

    for layout_name, layout_obj, is_block in _iter_layout_containers():
        if layout_obj is None:
            continue
        owner_label = layout_name
        if is_block:
            owner_label = f"BLOCK:{layout_name}"
            blocks_scanned.add(layout_name)
        else:
            layouts_scanned.add(layout_name)
        for table_entity in _iter_tables(layout_obj):
            if table_entity is None:
                continue
            dxf_obj = getattr(table_entity, "dxf", None)
            handle = getattr(dxf_obj, "handle", None) if dxf_obj is not None else None
            handle_str = str(handle or "") or hex(id(table_entity))
            marker = handle_str
            if marker in seen_table_handles:
                continue
            seen_table_handles.add(marker)
            tables_found += 1
            layer_name = ""
            if dxf_obj is not None:
                layer_name = str(getattr(dxf_obj, "layer", "") or "").strip()
            n_rows = _table_dimension(table_entity, ("n_rows", "row_count", "rows"))
            n_cols = _table_dimension(table_entity, ("n_cols", "column_count", "cols"))
            table_entry: dict[str, Any] = {
                "owner": owner_label,
                "layer": layer_name,
                "handle": handle_str,
                "rows": int(n_rows or 0),
                "cols": int(n_cols or 0),
                "header_tokens": False,
            }
            scan_tables.append(table_entry)
            if not n_rows or not n_cols:
                continue
            try:
                if n_rows <= 0 or n_cols <= 0:
                    continue
            except Exception:
                continue

            get_cell_extents = getattr(table_entity, "get_cell_extents", None)
            get_column_width = getattr(table_entity, "get_column_width", None)
            get_row_height = getattr(table_entity, "get_row_height", None)
            base_insert = getattr(dxf_obj, "insert", None) if dxf_obj is not None else None
            base_x: float | None = None
            base_y: float | None = None
            if base_insert is not None:
                try:
                    base_x = float(getattr(base_insert, "x", None))
                except Exception:
                    base_x = None
                try:
                    base_y = float(getattr(base_insert, "y", None))
                except Exception:
                    base_y = None

            fallback_col_edges: list[float] | None = None
            fallback_row_edges: list[float] | None = None
            if not callable(get_cell_extents):
                if callable(get_column_width):
                    edges: list[float] = [0.0]
                    total = 0.0
                    for col_idx in range(int(n_cols)):
                        width_val = 0.0
                        try:
                            width_val = float(get_column_width(col_idx) or 0.0)
                        except Exception:
                            width_val = 0.0
                        if not math.isfinite(width_val) or width_val < 0:
                            width_val = 0.0
                        total += width_val
                        edges.append(total)
                    fallback_col_edges = edges if len(edges) == int(n_cols) + 1 else None
                if callable(get_row_height):
                    edges_y: list[float] = [0.0]
                    total_y = 0.0
                    for row_idx in range(int(n_rows)):
                        height_val = 0.0
                        try:
                            height_val = float(get_row_height(row_idx) or 0.0)
                        except Exception:
                            height_val = 0.0
                        if not math.isfinite(height_val) or height_val < 0:
                            height_val = 0.0
                        total_y += height_val
                        edges_y.append(total_y)
                    fallback_row_edges = edges_y if len(edges_y) == int(n_rows) + 1 else None

            def _cell_center_from_extents(row_idx: int, col_idx: int) -> tuple[float, float] | None:
                if callable(get_cell_extents):
                    try:
                        extents = get_cell_extents(row_idx, col_idx)
                    except Exception:
                        extents = None
                    if extents and len(extents) >= 4:
                        try:
                            x_min = float(extents[0])
                            y_min = float(extents[1])
                            x_max = float(extents[2])
                            y_max = float(extents[3])
                        except Exception:
                            pass
                        else:
                            if math.isfinite(x_min) and math.isfinite(x_max) and math.isfinite(y_min) and math.isfinite(y_max):
                                return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
                if (
                    base_x is not None
                    and base_y is not None
                    and fallback_col_edges
                    and fallback_row_edges
                    and col_idx + 1 < len(fallback_col_edges)
                    and row_idx + 1 < len(fallback_row_edges)
                ):
                    x_min = fallback_col_edges[col_idx]
                    x_max = fallback_col_edges[col_idx + 1]
                    y_top = fallback_row_edges[row_idx]
                    y_bottom = fallback_row_edges[row_idx + 1]
                    if math.isfinite(x_min) and math.isfinite(x_max) and math.isfinite(y_top) and math.isfinite(y_bottom):
                        x_center = base_x + (x_min + x_max) / 2.0
                        y_center = base_y - (y_top + y_bottom) / 2.0
                        if math.isfinite(x_center) and math.isfinite(y_center):
                            return (x_center, y_center)
                return None

            table_cells: list[list[str]] = []
            table_centers: list[list[tuple[float, float] | None]] = []
            for row_idx in range(int(n_rows)):
                row_cells: list[str] = []
                row_centers: list[tuple[float, float] | None] = []
                for col_idx in range(int(n_cols)):
                    try:
                        text_value = _cell_text(table_entity, row_idx, col_idx)
                    except Exception:
                        text_value = ""
                    row_cells.append(text_value)
                    try:
                        center_val = _cell_center_from_extents(row_idx, col_idx)
                    except Exception:
                        center_val = None
                    row_centers.append(center_val)
                if any(cell.strip() for cell in row_cells):
                    table_cells.append(row_cells)
                    table_centers.append(row_centers)
            if not table_cells:
                continue
            if table_centers:
                table_entry["cell_centers"] = table_centers
            table_entry["row_count"] = len(table_cells)

            header_map: dict[str, int] = {}
            header_row_idx: int | None = None
            header_tokens_hit = False
            for idx, row_cells in enumerate(table_cells):
                hits = _detect_header_hits(row_cells)
                if not hits:
                    continue
                header_map = dict(hits)
                header_row_idx = idx
                combined_upper = " ".join(cell.upper() for cell in row_cells if cell)
                for token in ("QTY", "Ø", "⌀", "DIA", "DIAM", "HOLE", "DESCRIPTION", "DESC"):
                    if token in combined_upper:
                        header_tokens_hit = True
                        break
                break
            if "desc" not in header_map and table_cells:
                candidate_indices = list(range(len(table_cells[0])))
                for used in header_map.values():
                    if used in candidate_indices:
                        candidate_indices.remove(used)
                if candidate_indices:
                    header_map["desc"] = max(candidate_indices)
            if header_tokens_hit:
                table_entry["header_tokens"] = True

            table_rows: list[dict[str, Any]] = []
            families: dict[str, int] = {}
            for idx, row_cells in enumerate(table_cells):
                if header_row_idx is not None and idx <= header_row_idx:
                    continue
                hits = _detect_header_hits(row_cells)
                if hits:
                    continue
                combined_text = " ".join(cell.strip() for cell in row_cells if cell).strip()
                if not combined_text:
                    continue
                fallback_desc = ""
                qty_val = None
                qty_idx = header_map.get("qty")
                if qty_idx is not None and qty_idx < len(row_cells):
                    qty_val = _parse_qty_cell_text(row_cells[qty_idx])
                combined_qty, combined_remainder = _extract_row_quantity_and_remainder(
                    combined_text
                )
                if qty_val is None and combined_qty is not None and combined_qty > 0:
                    qty_val = combined_qty
                    fallback_desc = combined_remainder.strip()
                elif qty_val is not None and combined_qty is not None and qty_val == combined_qty:
                    fallback_desc = combined_remainder.strip()
                if qty_val is None or qty_val <= 0:
                    continue

                desc_idx = header_map.get("desc")
                desc_text = ""
                if desc_idx is not None and desc_idx < len(row_cells):
                    desc_text = row_cells[desc_idx]
                if not desc_text:
                    desc_text = fallback_desc
                if not desc_text:
                    excluded = {idx for idx in header_map.values() if idx is not None}
                    desc_parts = [
                        row_cells[col].strip()
                        for col in range(len(row_cells))
                        if col not in excluded and row_cells[col].strip()
                    ]
                    desc_text = " ".join(desc_parts)
                desc_text = " ".join((desc_text or "").split())
                if not desc_text:
                    continue

                ref_idx = header_map.get("ref")
                ref_cell_text = (
                    row_cells[ref_idx] if ref_idx is not None and ref_idx < len(row_cells) else ""
                )
                ref_cell_ref = _extract_row_reference(ref_cell_text) if ref_cell_text else ("", None)
                hole_idx = header_map.get("hole")
                hole_text = ""
                if hole_idx is not None and hole_idx < len(row_cells):
                    raw_hole = row_cells[hole_idx]
                    if isinstance(raw_hole, str):
                        upper_hole = raw_hole.upper()
                        match = re.search(r"\b([A-Z]{1,3})\b", upper_hole)
                        if match:
                            hole_text = match.group(1)
                        else:
                            hole_text = raw_hole.strip()
                side_idx = header_map.get("side")
                side_cell_text = (
                    row_cells[side_idx]
                    if side_idx is not None and side_idx < len(row_cells)
                    else ""
                )
                base_side = _detect_row_side(" ".join([side_cell_text, desc_text]))

                fragments = [frag.strip() for frag in desc_text.split(";") if frag.strip()]
                if not fragments:
                    fragments = [desc_text]
                for fragment in fragments:
                    fragment_desc = " ".join(fragment.split())
                    if not fragment_desc:
                        continue
                    ref_text, ref_value = _extract_row_reference(fragment_desc)
                    if not ref_text and ref_cell_ref[0]:
                        ref_text, ref_value = ref_cell_ref
                    elif not ref_text and ref_cell_text:
                        ref_text = " ".join(ref_cell_text.split())
                        ref_value = None
                    side_value = _detect_row_side(" ".join([fragment_desc, side_cell_text]))
                    if not side_value:
                        side_value = base_side
                    row_entry: dict[str, Any] = {
                        "hole": hole_text,
                        "qty": qty_val,
                        "desc": fragment_desc,
                        "ref": ref_text,
                    }
                    if side_value:
                        row_entry["side"] = side_value
                    table_rows.append(row_entry)
                    if ref_value is not None:
                        key = f"{ref_value:.4f}".rstrip("0").rstrip(".")
                        families[key] = families.get(key, 0) + qty_val

            if not table_rows:
                continue

            bbox = _compute_table_bbox(table_entity)
            median_height = _estimate_text_height(table_entity, n_rows)
            pad = 2.0 * median_height if median_height > 0 else 6.0
            roi_hint = None
            if bbox is not None:
                xmin, xmax, ymin, ymax = bbox
                roi_hint = {
                    "source": "ACAD_TABLE",
                    "handle": handle_str,
                    "layer": layer_name,
                    "bbox": [xmin, xmax, ymin, ymax],
                    "pad": pad,
                    "median_height": median_height,
                }

            sum_qty = _sum_qty(table_rows)
            table_entry["row_count"] = len(table_rows)
            table_entry["sum_qty"] = sum_qty

            candidate = {
                "rows": table_rows,
                "families": families,
                "layer": layer_name,
                "layer_upper": layer_name.upper() if layer_name else "",
                "owner": owner_label,
                "handle": handle_str,
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "row_count": len(table_rows),
                "sum_qty": sum_qty,
                "roi_hint": roi_hint,
                "median_height": median_height,
                "header_tokens": header_tokens_hit,
                "cell_centers": table_entry.get("cell_centers"),
            }
            table_candidates.append(candidate)

    _LAST_ACAD_TABLE_SCAN = {
        "layouts": sorted(layouts_scanned),
        "blocks": sorted(blocks_scanned),
        "tables_found": tables_found,
        "tables": scan_tables,
        "allow_layers": None if allowlist is None else sorted(allowlist),
        "layouts_total": layout_total_count,
        "blocks_total": block_total_count,
    }

    layouts_display = layout_total_count or len(layouts_scanned)
    blocks_display = block_total_count or len(blocks_scanned)
    print(
        f"[ACAD-TABLE] scanned layouts={layouts_display} blocks={blocks_display} "
        f"tables_found={tables_found}"
    )
    preview_limit = 8
    for entry in scan_tables[:preview_limit]:
        owner = str(entry.get("owner") or "-")
        layer = str(entry.get("layer") or "-")
        handle = str(entry.get("handle") or "-")
        try:
            rows_val = int(entry.get("rows") or entry.get("row_count") or 0)
        except Exception:
            rows_val = 0
        try:
            cols_val = int(entry.get("cols") or 0)
        except Exception:
            cols_val = 0
        try:
            qty_val = int(entry.get("sum_qty") or 0)
        except Exception:
            qty_val = 0
        print(
            "[ACAD-TABLE] hit owner={owner} layer={layer} handle={handle} rows={rows} cols={cols} qty_sum={qty}".format(
                owner=owner,
                layer=layer,
                handle=handle,
                rows=rows_val,
                cols=cols_val,
                qty=qty_val,
            )
        )

    if not table_candidates:
        return {}

    preferred_candidates = []
    if allowlist is not None:
        preferred_candidates = [
            cand for cand in table_candidates if cand.get("layer_upper") in allowlist
        ]
    else:
        preferred_candidates = list(table_candidates)
    if not preferred_candidates:
        preferred_candidates = table_candidates

    def _priority(candidate: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
        layer_upper = str(candidate.get("layer_upper") or "")
        allow_hit = 1
        if allowlist:
            allow_hit = 1 if layer_upper in allowlist else 0
        header_bonus = 1 if candidate.get("header_tokens") else 0
        return (
            allow_hit,
            int(candidate.get("sum_qty") or 0),
            int(candidate.get("row_count") or 0),
            header_bonus,
            int(candidate.get("n_rows") or 0),
            int(candidate.get("n_cols") or 0),
        )

    best_candidate = max(preferred_candidates, key=_priority)
    if int(best_candidate.get("row_count") or 0) < 5:
        return {}

    best_rows = list(best_candidate.get("rows") or [])
    if not best_rows:
        return {}
    hole_count = _sum_qty(best_rows)
    result: dict[str, Any] = {
        "rows": best_rows,
        "hole_count": hole_count,
        "sum_qty": hole_count,
        "provenance_holes": "HOLE TABLE",
        "layer": best_candidate.get("layer"),
        "owner": best_candidate.get("owner"),
        "handle": best_candidate.get("handle"),
        "n_rows": best_candidate.get("n_rows"),
        "n_cols": best_candidate.get("n_cols"),
    }
    families_map = best_candidate.get("families")
    if isinstance(families_map, Mapping) and families_map:
        result["hole_diam_families_in"] = dict(families_map)
    roi_hint = best_candidate.get("roi_hint")
    if isinstance(roi_hint, Mapping):
        result["roi_hint"] = dict(roi_hint)
    centers_value = best_candidate.get("cell_centers")
    if isinstance(centers_value, list) and centers_value:
        result["cell_centers"] = centers_value

    print(
        "[ACAD-TABLE] chosen handle={handle} layer={layer} owner={owner} rows={rows} qty_sum={qty}".format(
            handle=result.get("handle"),
            layer=result.get("layer") or "-",
            owner=result.get("owner") or "-",
            rows=len(best_rows),
            qty=hole_count,
        )
    )

    return result


def _collect_table_text_lines(doc: Any) -> list[str]:
    lines: list[str] = []
    if doc is None:
        return lines

    spaces: list[Any] = []
    modelspace = getattr(doc, "modelspace", None)
    if callable(modelspace):
        try:
            space = modelspace()
        except Exception:
            space = None
        if space is not None:
            spaces.append(space)

    for space in spaces:
        query = getattr(space, "query", None)
        if not callable(query):
            continue
        try:
            entities = list(query("TEXT, MTEXT"))
        except Exception:
            continue
        for entity in entities:
            fragments = list(_iter_entity_text_fragments(entity))
            for fragment, _ in fragments:
                normalized = _normalize_table_fragment(fragment)
                if normalized:
                    lines.append(normalized)
    return lines


def _normalize_table_fragment(fragment: str) -> str:
    if not isinstance(fragment, str):
        fragment = str(fragment)
    cleaned = fragment.replace("%%C", "Ø").replace("%%c", "Ø")
    cleaned = _MTEXT_ALIGN_RE.sub("", cleaned)
    cleaned = _MTEXT_BREAK_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("|", " |")
    cleaned = cleaned.replace("\\~", "~")
    cleaned = cleaned.replace("\\`", "`")
    cleaned = cleaned.replace("\\", " ")
    return " ".join(cleaned.split())


def _iter_entity_text_fragments(entity: Any) -> Iterable[tuple[str, bool]]:
    dxftype = None
    try:
        dxftype = entity.dxftype()
    except Exception:
        dxftype = None
    kind = str(dxftype or "").upper()
    if kind == "MTEXT":
        plain_text = getattr(entity, "plain_text", None)
        content = None
        if callable(plain_text):
            try:
                content = plain_text()
            except Exception:
                content = None
        if content is None:
            content = getattr(entity, "text", "")
        for piece in _split_mtext_plain_text(content):
            yield (piece, True)
    elif kind == "TEXT":
        dxf_obj = getattr(entity, "dxf", None)
        raw_text = getattr(dxf_obj, "text", "") if dxf_obj is not None else ""
        if not raw_text:
            raw_text = getattr(entity, "text", "")
        try:
            base = str(raw_text)
        except Exception:
            base = raw_text if isinstance(raw_text, str) else ""
        for piece in base.splitlines():
            if piece.strip():
                yield (piece, False)
    else:
        raw_text = getattr(entity, "text", "")
        if not raw_text:
            return
        try:
            base = str(raw_text)
        except Exception:
            base = raw_text if isinstance(raw_text, str) else ""
        for piece in base.splitlines():
            if piece.strip():
                yield (piece, False)


def _parse_number_token(token: str) -> float | None:
    text = (token or "").strip()
    if not text:
        return None
    if "/" in text:
        try:
            return float(Fraction(text))
        except Exception:
            return None
    if text.startswith("."):
        text = "0" + text
    try:
        return float(text)
    except Exception:
        return None


def _format_ref_value(value: float) -> str:
    return f"{value:.4f}\""


def _has_candidate_token(text: str) -> bool:
    if not text:
        return False
    if _CANDIDATE_TOKEN_RE.search(text):
        return True
    if "Ø" in text or "⌀" in text:
        return True
    if '"' in text:
        return True
    if _FRACTION_RE.search(text):
        return True
    if _DECIMAL_RE.search(text):
        return True
    return False


def _match_row_quantity(text: str) -> re.Match[str] | None:
    candidate = text or ""
    for pattern in _ROW_QUANTITY_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match
    return None


def _search_flexible_quantity(text: str) -> re.Match[str] | None:
    candidate = text or ""
    for pattern in _ROW_QUANTITY_FLEX_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match
    return None


def _is_letter_code_row_start(text: str, next_text: str | None = None) -> bool:
    if not text:
        return False
    match = _LETTER_CODE_ROW_RE.match(text)
    if not match:
        return False
    remainder = text[match.end() :]
    if _HOLE_ACTION_TOKEN_RE.search(remainder):
        return True
    if next_text and _HOLE_ACTION_TOKEN_RE.search(next_text):
        return True
    return False


def _is_row_start(text: str, *, next_text: str | None = None) -> bool:
    if _match_row_quantity(text):
        return True
    return _is_letter_code_row_start(text, next_text)


def _extract_row_quantity_and_remainder(text: str) -> tuple[int | None, str]:
    base = (text or "").strip()
    if not base:
        return (None, "")

    def _strip_span(source: str, span: tuple[int, int]) -> str:
        start, end = span
        return (source[:start] + " " + source[end:]).strip()

    primary_match = _match_row_quantity(base)
    if primary_match:
        qty_text = primary_match.group(1)
        try:
            qty_val = int(qty_text)
        except Exception:
            qty_val = None
        remainder = base[primary_match.end() :].strip()
        return (qty_val, remainder)

    letter_match = _LETTER_CODE_ROW_RE.match(base)
    if letter_match:
        remainder_body = base[letter_match.end() :].lstrip(" -.:|")
        remainder_match = _match_row_quantity(remainder_body)
        if remainder_match:
            qty_text = remainder_match.group(1)
            try:
                qty_val = int(qty_text)
            except Exception:
                qty_val = None
            remainder = remainder_body[remainder_match.end() :].strip()
            return (qty_val, remainder)
        flexible = _search_flexible_quantity(remainder_body)
        if flexible:
            qty_text = flexible.group(1)
            try:
                qty_val = int(qty_text)
            except Exception:
                qty_val = None
            remainder = _strip_span(remainder_body, flexible.span())
            return (qty_val, remainder)

    flexible_match = _search_flexible_quantity(base)
    if flexible_match:
        qty_text = flexible_match.group(1)
        try:
            qty_val = int(qty_text)
        except Exception:
            qty_val = None
        remainder = _strip_span(base, flexible_match.span())
        return (qty_val, remainder)

    bare_match = re.match(r"^\s*(\d+)\b", base)
    if bare_match and _HOLE_ACTION_TOKEN_RE.search(base):
        qty_text = bare_match.group(1)
        try:
            qty_val = int(qty_text)
        except Exception:
            qty_val = None
        remainder = base[bare_match.end() :].strip()
        return (qty_val, remainder)

    return (None, base)


def _extract_band_quantity(text: str) -> tuple[int | None, str]:
    candidate = " ".join((text or "").split())
    if not candidate:
        return (None, "")
    for pattern in _BAND_QTY_FALLBACK_PATTERNS:
        match = pattern.search(candidate)
        if not match:
            continue
        qty_text = match.group("qty") if "qty" in match.groupdict() else None
        if not qty_text:
            continue
        try:
            qty_val = int(qty_text)
        except Exception:
            continue
        start, end = match.span()
        remainder = (candidate[:start] + " " + candidate[end:]).strip()
        return (qty_val, remainder)
    return (None, candidate)


def _extract_row_reference(desc: str) -> tuple[str, float | None]:
    diameter = _extract_diameter(desc)
    if diameter is not None and diameter > 0 and diameter <= 10:
        return (_format_ref_value(diameter), diameter)
    search_space = desc or ""
    for match in _FRACTION_RE.finditer(search_space):
        value = _parse_number_token(match.group(0))
        if value is not None and 0 < value <= 10:
            return (_format_ref_value(value), value)
    for match in _DECIMAL_RE.finditer(search_space):
        value = _parse_number_token(match.group(0))
        if value is not None and 0 < value <= 10:
            return (_format_ref_value(value), value)
    return ("", None)


def _detect_row_side(desc: str) -> str:
    upper = (desc or "").upper()
    if "BOTH SIDES" in upper or ("FRONT" in upper and "BACK" in upper):
        return "both"
    if "FROM BACK" in upper:
        return "back"
    if "FROM FRONT" in upper:
        return "front"
    return ""


def _merge_table_lines(lines: Iterable[str]) -> list[str]:
    merged: list[str] = []
    current: list[str] | None = None
    buffer: list[str] = []
    for raw_line in lines:
        candidate = (raw_line or "").strip()
        if candidate:
            buffer.append(candidate)
    for index, line in enumerate(buffer):
        next_line = buffer[index + 1] if index + 1 < len(buffer) else None
        if _is_row_start(line, next_text=next_line):
            if current:
                merged.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        merged.append(" ".join(current))
    return merged


def _extract_diameter(text: str) -> float | None:
    search_space = text or ""
    match = _DIAMETER_PREFIX_RE.search(search_space)
    if not match:
        match = _DIAMETER_SUFFIX_RE.search(search_space)
    if not match:
        return None
    return _parse_number_token(match.group(1))



def _truncate_cell_preview(text: str, limit: int = 60) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _cell_has_ref_marker(text: str) -> bool:
    if not text:
        return False
    candidate = text.strip()
    if "Ø" in candidate or "⌀" in candidate or '"' in candidate:
        return True
    if _FRACTION_RE.search(candidate):
        return True
    if _DECIMAL_RE.search(candidate):
        return True
    return False


def _build_columnar_table_from_entries(
    entries: list[dict[str, Any]],
    *,
    roi_hint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[int(position)]
        lower_val = ordered[lower]
        upper_val = ordered[upper]
        return lower_val + (upper_val - lower_val) * (position - lower)

    records: list[dict[str, Any]] = []
    for entry in entries:
        text_value = (entry.get("normalized_text") or entry.get("text") or "").strip()
        if not text_value:
            continue
        x_val = entry.get("x")
        y_val = entry.get("y")
        try:
            x_float = float(x_val)
            y_float = float(y_val)
        except Exception:
            continue
        record = {
            "layout": entry.get("layout_name"),
            "from_block": bool(entry.get("from_block")),
            "x": x_float,
            "y": y_float,
            "text": text_value,
            "height": entry.get("height"),
        }
        records.append(record)

    if not records:
        return (
            None,
            {
                "bands": [],
                "band_cells": [],
                "rows_txt_fallback": [],
                "qty_col": None,
                "ref_col": None,
                "desc_col": None,
                "roi": None,
            },
        )

    base_records = list(records)
    records_all = list(base_records)
    roi_bounds: dict[str, float] | None = None
    roi_info: dict[str, Any] | None = None
    roi_median_height = 0.0

    if isinstance(roi_hint, Mapping):
        bbox = roi_hint.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                xmin = float(bbox[0])
                xmax = float(bbox[1])
                ymin = float(bbox[2])
                ymax = float(bbox[3])
            except Exception:
                xmin = xmax = ymin = ymax = 0.0
            else:
                pad = 0.0
                try:
                    pad = float(roi_hint.get("pad") or 0.0)
                except Exception:
                    pad = 0.0
                expanded_xmin = xmin - pad
                expanded_xmax = xmax + pad
                expanded_ymin = ymin - pad
                expanded_ymax = ymax + pad
                filtered = [
                    rec
                    for rec in base_records
                    if expanded_xmin <= rec["x"] <= expanded_xmax
                    and expanded_ymin <= rec["y"] <= expanded_ymax
                ]
                if filtered:
                    records_all = filtered
                roi_bounds = {
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "dx": pad,
                    "dy": pad,
                    "clusters": 1,
                    "anchors": 0,
                }
                kept_count = len(filtered)
                source = str(roi_hint.get("source") or "ACAD_TABLE")
                roi_info = {
                    "source": source,
                    "bbox": [xmin, xmax, ymin, ymax],
                    "pad": pad,
                    "kept": kept_count,
                }
                try:
                    roi_median_height = float(roi_hint.get("median_height") or 0.0)
                except Exception:
                    roi_median_height = 0.0
                handle = roi_hint.get("handle")
                layer = roi_hint.get("layer")
                block_name = roi_hint.get("name")
                if block_name is not None:
                    roi_info["name"] = block_name
                if source.upper() == "BLOCK":
                    print(
                        "[ROI] seeded_from=BLOCK name={name} layer={layer} "
                        "box=[{xmin:.1f}..{xmax:.1f}, {ymin:.1f}..{ymax:.1f}]".format(
                            name=block_name or handle or "-",
                            layer=layer or "-",
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax,
                        )
                    )
                else:
                    print(
                        "[ROI] seeded_from={src} handle={handle} layer={layer} "
                        "box=[{xmin:.1f}..{xmax:.1f}, {ymin:.1f}..{ymax:.1f}]".format(
                            src=source,
                            handle=handle,
                            layer=layer or "-",
                            xmin=xmin,
                            xmax=xmax,
                            ymin=ymin,
                            ymax=ymax,
                        )
                    )

    all_height_values = [
        float(rec["height"])
        for rec in records_all
        if isinstance(rec.get("height"), (int, float)) and float(rec["height"]) > 0
    ]
    median_height_all = (
        statistics.median(all_height_values) if all_height_values else 0.0
    )
    if roi_median_height <= 0:
        roi_median_height = median_height_all
    anchor_lines = [rec for rec in records_all if _ROI_ANCHOR_RE.search(rec["text"])]
    filtered_records = records_all
    if roi_bounds is None and anchor_lines:
        anchor_count = len(anchor_lines)
        sorted_anchors = sorted(anchor_lines, key=lambda rec: -rec["y"])
        clusters: list[list[dict[str, Any]]] = []
        if sorted_anchors:
            if anchor_count >= 4:
                height_values = [
                    float(rec["height"])
                    for rec in sorted_anchors
                    if isinstance(rec.get("height"), (int, float))
                    and float(rec["height"]) > 0
                ]
                anchor_y_diffs = [
                    abs(sorted_anchors[idx]["y"] - sorted_anchors[idx - 1]["y"])
                    for idx in range(1, len(sorted_anchors))
                    if abs(sorted_anchors[idx]["y"] - sorted_anchors[idx - 1]["y"]) > 0
                ]
                if height_values:
                    median_height = statistics.median(height_values)
                    y_anchor_eps = 1.8 * median_height if median_height > 0 else 0.0
                    roi_median_height = median_height
                elif anchor_y_diffs:
                    median_diff = statistics.median(anchor_y_diffs)
                    y_anchor_eps = 0.5 * median_diff if median_diff > 0 else 0.0
                else:
                    y_anchor_eps = 0.0
                y_anchor_eps = max(6.0, y_anchor_eps)
                current_cluster: list[dict[str, Any]] | None = None
                prev_anchor: dict[str, Any] | None = None
                for anchor in sorted_anchors:
                    if current_cluster is None:
                        current_cluster = [anchor]
                        clusters.append(current_cluster)
                        prev_anchor = anchor
                        continue
                    prev_y = prev_anchor["y"] if prev_anchor is not None else None
                    if prev_y is not None and abs(anchor["y"] - prev_y) <= y_anchor_eps:
                        current_cluster.append(anchor)
                    else:
                        current_cluster = [anchor]
                        clusters.append(current_cluster)
                    prev_anchor = anchor
            if not clusters:
                clusters = [sorted_anchors]

        def _cluster_span(cluster: list[dict[str, Any]]) -> float:
            if not cluster:
                return 0.0
            y_vals = [rec["y"] for rec in cluster]
            return max(y_vals) - min(y_vals) if len(y_vals) > 1 else 0.0

        chosen_cluster = clusters[0] if clusters else []
        best_size = len(chosen_cluster)
        best_span = _cluster_span(chosen_cluster)
        for cluster in clusters[1:]:
            size = len(cluster)
            span = _cluster_span(cluster)
            if size > best_size or (size == best_size and span < best_span):
                chosen_cluster = cluster
                best_size = size
                best_span = span

        if not chosen_cluster:
            chosen_cluster = sorted_anchors
        cluster_xmin = min(rec["x"] for rec in chosen_cluster)
        cluster_xmax = max(rec["x"] for rec in chosen_cluster)
        cluster_ymin = min(rec["y"] for rec in chosen_cluster)
        cluster_ymax = max(rec["y"] for rec in chosen_cluster)
        base_dx = 18.0 * median_height_all if median_height_all > 0 else 0.0
        base_dy = 24.0 * median_height_all if median_height_all > 0 else 0.0
        dx = max(40.0, base_dx)
        dy = max(50.0, base_dy)
        if roi_median_height and roi_median_height > 0:
            dx = max(dx, 18.0 * roi_median_height)
            dy = max(dy, 24.0 * roi_median_height)
        expanded_xmin = cluster_xmin - dx
        expanded_xmax = cluster_xmax + dx
        expanded_ymin = cluster_ymin - dy
        expanded_ymax = cluster_ymax + dy
        filtered_records = [
            rec
            for rec in records_all
            if expanded_xmin <= rec["x"] <= expanded_xmax
            and expanded_ymin <= rec["y"] <= expanded_ymax
        ]
        clusters_count = len(clusters) or 1
        roi_bounds = {
            "xmin": cluster_xmin,
            "xmax": cluster_xmax,
            "ymin": cluster_ymin,
            "ymax": cluster_ymax,
            "dx": dx,
            "dy": dy,
            "clusters": clusters_count,
            "anchors": anchor_count,
        }
        roi_info = {
            "anchors": anchor_count,
            "clusters": clusters_count,
            "bbox": [cluster_xmin, cluster_xmax, cluster_ymin, cluster_ymax],
            "total": len(records_all),
        }
    records = list(filtered_records)

    def _prepare_records(values: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
        ordered = list(values)
        ordered.sort(key=lambda item: (-item["y"], item["x"]))
        height_vals = [
            float(rec["height"])
            for rec in ordered
            if isinstance(rec.get("height"), (int, float)) and float(rec["height"]) > 0
        ]
        y_offsets = [
            abs(ordered[idx]["y"] - ordered[idx - 1]["y"])
            for idx in range(1, len(ordered))
            if abs(ordered[idx]["y"] - ordered[idx - 1]["y"]) > 0
        ]
        median_val = statistics.median(height_vals) if height_vals else 0.0
        if (median_val is None or median_val <= 0) and roi_median_height > 0:
            median_val = roi_median_height
        if (median_val is None or median_val <= 0) and median_height_all > 0:
            median_val = median_height_all
        if (median_val is None or median_val <= 0) and y_offsets:
            median_val = statistics.median(y_offsets)
        if median_val is None or median_val <= 0:
            median_val = 4.0
        return ordered, median_val

    records, median_h = _prepare_records(records)

    if roi_bounds is not None:
        desired_dx = max(roi_bounds["dx"], 18.0 * median_h)
        desired_dy = max(roi_bounds["dy"], 24.0 * median_h)
        if desired_dx > roi_bounds["dx"] + 1e-6 or desired_dy > roi_bounds["dy"] + 1e-6:
            expanded_xmin = roi_bounds["xmin"] - desired_dx
            expanded_xmax = roi_bounds["xmax"] + desired_dx
            expanded_ymin = roi_bounds["ymin"] - desired_dy
            expanded_ymax = roi_bounds["ymax"] + desired_dy
            filtered_records = [
                rec
                for rec in records_all
                if expanded_xmin <= rec["x"] <= expanded_xmax
                and expanded_ymin <= rec["y"] <= expanded_ymax
            ]
            roi_bounds["dx"] = desired_dx
            roi_bounds["dy"] = desired_dy
            records, median_h = _prepare_records(filtered_records)
        expanded_xmin = roi_bounds["xmin"] - roi_bounds["dx"]
        expanded_xmax = roi_bounds["xmax"] + roi_bounds["dx"]
        expanded_ymin = roi_bounds["ymin"] - roi_bounds["dy"]
        expanded_ymax = roi_bounds["ymax"] + roi_bounds["dy"]
        kept_count = len(records)
        if roi_info is None:
            roi_info = {}
        roi_info.update(
            {
                "expanded": [expanded_xmin, expanded_xmax, expanded_ymin, expanded_ymax],
                "kept": kept_count,
                "median_h": median_h,
                "anchors": int(roi_bounds.get("anchors", 0.0)),
                "clusters": int(roi_bounds.get("clusters", 0.0)) or 1,
            }
        )
        print(
            "[ROI] anchors={count} clusters={clusters} chosen_span=[{ymax:.1f}..{ymin:.1f}] "
            "bbox=[{xmin:.1f}..{xmax:.1f}] expanded=[{xmin_exp:.1f}..{xmax_exp:.1f},{ymin_exp:.1f}..{ymax_exp:.1f}]".format(
                count=int(roi_bounds.get("anchors", 0.0)),
                clusters=int(roi_bounds.get("clusters", 0.0)) or 1,
                ymax=roi_bounds["ymax"],
                ymin=roi_bounds["ymin"],
                xmin=roi_bounds["xmin"],
                xmax=roi_bounds["xmax"],
                xmin_exp=expanded_xmin,
                xmax_exp=expanded_xmax,
                ymin_exp=expanded_ymin,
                ymax_exp=expanded_ymax,
            )
        )
        print(
            f"[ROI] median_h={median_h:.2f} expand=({roi_bounds['dx']:.1f},{roi_bounds['dy']:.1f})"
        )
        print(
            f"[ROI] raw_lines -> roi_lines: {len(records_all)} -> {kept_count}"
        )

    y_gap_limit = 0.75 * median_h if median_h > 0 else 4.0
    if y_gap_limit <= 0:
        y_gap_limit = 4.0

    raw_bands: list[list[dict[str, Any]]] = []
    current_band: list[dict[str, Any]] = []
    current_sum_y = 0.0
    prev_y: float | None = None
    for record in records:
        y_val = record["y"]
        if not current_band:
            current_band = [record]
            current_sum_y = y_val
            prev_y = y_val
            continue
        band_center = current_sum_y / len(current_band)
        direct_gap = abs(y_val - prev_y) if prev_y is not None else 0.0
        if direct_gap > y_gap_limit:
            raw_bands.append(current_band)
            current_band = [record]
            current_sum_y = y_val
            prev_y = y_val
            continue
        if abs(y_val - band_center) > y_gap_limit:
            raw_bands.append(current_band)
            current_band = [record]
            current_sum_y = y_val
            prev_y = y_val
            continue
        proposed_sum = current_sum_y + y_val
        proposed_count = len(current_band) + 1
        proposed_center = proposed_sum / proposed_count
        if (
            abs(proposed_center - band_center) > y_gap_limit
            or abs(y_val - proposed_center) > y_gap_limit
        ):
            raw_bands.append(current_band)
            current_band = [record]
            current_sum_y = y_val
            prev_y = y_val
            continue
        current_band.append(record)
        current_sum_y = proposed_sum
        prev_y = y_val
    if current_band:
        raw_bands.append(current_band)

    bands: list[list[dict[str, Any]]] = []
    for raw_index, band in enumerate(raw_bands):
        if len(band) < 2:
            continue
        combined = " ".join(item["text"] for item in band)
        if not any(char.isdigit() for char in combined) and not _COLUMN_TOKEN_RE.search(combined):
            continue
        drop_band = False
        combined_upper = combined.upper()
        drop_reason = ""
        trimmed = combined.strip()
        contains_hole_token = bool(_HOLE_ACTION_TOKEN_RE.search(combined))
        keep_marker = bool(_BAND_KEEP_TOKEN_RE.search(combined_upper))
        decimal_hits = len(_DECIMAL_RE.findall(combined))
        fraction_hits = len(_FRACTION_RE.findall(combined))
        numeric_guard = (decimal_hits + fraction_hits) >= 2
        if _TITLE_AXIS_DROP_RE.search(combined_upper):
            drop_band = not (keep_marker or numeric_guard)
            drop_reason = "title/axis" if drop_band else ""
        elif _SEE_SHEET_DROP_RE.search(combined_upper):
            drop_band = not (keep_marker or numeric_guard)
            drop_reason = "title/axis" if drop_band else ""
        elif _AXIS_ZERO_PAIR_RE.match(trimmed):
            drop_band = not (keep_marker or numeric_guard)
            drop_reason = "title/axis" if drop_band else ""
        elif _AXIS_ZERO_SINGLE_RE.match(trimmed):
            drop_band = not (keep_marker or numeric_guard)
            drop_reason = "title/axis" if drop_band else ""
        else:
            small_ints = []
            for token in _SMALL_INT_TOKEN_RE.findall(combined):
                try:
                    value = int(token)
                except Exception:
                    continue
                if 1 <= value <= 16:
                    small_ints.append(value)
            if len(small_ints) >= 10:
                drop_band = not (keep_marker or numeric_guard)
                drop_reason = "title/axis" if drop_band else ""
            elif len(small_ints) >= 12 and len(set(small_ints)) <= 16:
                drop_band = not (keep_marker or numeric_guard)
                drop_reason = "title/axis" if drop_band else ""
        if drop_band and contains_hole_token:
            drop_band = False
            drop_reason = ""
        if drop_band:
            preview = _truncate_cell_preview(trimmed or combined, 80).replace("\"", "\\\"")
            print(
                f"[TABLE-Y] drop band#{raw_index} reason={drop_reason} lines={len(band)} text=\"{preview}\""
            )
            continue
        bands.append(band)

    print(f"[TABLE-Y] bands_total={len(bands)} median_h={median_h:.2f}")

    header_token_patterns = [
        (re.compile(r"\bHOLE\b", re.IGNORECASE), "hole"),
        (re.compile(r"\bQTY\b|\bQUANTITY\b", re.IGNORECASE), "qty"),
        (re.compile(r"\bDESC(?:RIPTION)?\b", re.IGNORECASE), "desc"),
        (re.compile(r"\bSIDE\b|\bFACE\b", re.IGNORECASE), "side"),
        (re.compile(r"\bREF\b", re.IGNORECASE), "ref"),
        (re.compile(r"Ø|⌀|DIA|DIAM", re.IGNORECASE), "ref"),
    ]

    def _header_windows_from_band(
        band_items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        positions: list[float] = []
        field_positions: dict[str, list[float]] = defaultdict(list)
        for item in band_items:
            text_value = str(item.get("text") or "")
            if not text_value:
                continue
            try:
                x_val = float(item.get("x"))
            except Exception:
                continue
            positions.append(x_val)
            upper_text = text_value.upper()
            for pattern, field in header_token_patterns:
                if pattern.search(upper_text):
                    field_positions.setdefault(field, []).append(x_val)
        if not positions:
            return ([], {})
        unique_positions = sorted({float(pos) for pos in positions})
        diffs = [
            abs(unique_positions[idx + 1] - unique_positions[idx])
            for idx in range(len(unique_positions) - 1)
            if abs(unique_positions[idx + 1] - unique_positions[idx]) > 0
        ]
        gap_baseline = statistics.median(diffs) if diffs else 0.0
        if not gap_baseline or gap_baseline <= 0:
            gap_baseline = 6.0 * median_h if median_h > 0 else 12.0
        default_half = max(6.0, 0.5 * gap_baseline)
        windows: list[dict[str, Any]] = []
        for idx, center in enumerate(unique_positions):
            if idx == 0:
                left = center - default_half
            else:
                left = (unique_positions[idx - 1] + center) / 2.0
            if idx + 1 < len(unique_positions):
                right = (center + unique_positions[idx + 1]) / 2.0
            else:
                right = center + default_half
            windows.append({"center": center, "left": left, "right": right, "field": None})

        field_centers = {
            field: statistics.mean(values)
            for field, values in field_positions.items()
            if values
        }
        assigned_fields: set[str] = set()
        used_indices: set[int] = set()
        for field, center_val in sorted(field_centers.items(), key=lambda item: item[1]):
            if field in assigned_fields:
                continue
            candidate_indices = list(range(len(windows)))
            best_idx = min(
                candidate_indices,
                key=lambda idx: (abs(windows[idx]["center"] - center_val), idx),
            )
            if windows[best_idx].get("field") not in (None, field):
                free_candidates = [
                    idx
                    for idx in candidate_indices
                    if idx not in used_indices
                    and windows[idx].get("field") in (None, field)
                ]
                if free_candidates:
                    best_idx = min(
                        free_candidates,
                        key=lambda idx: (abs(windows[idx]["center"] - center_val), idx),
                    )
            windows[best_idx]["field"] = field
            assigned_fields.add(field)
            used_indices.add(best_idx)

        for idx, window in enumerate(windows):
            if window.get("field"):
                continue
            if idx == 0:
                window["field"] = "hole"
            elif idx == len(windows) - 1:
                window["field"] = "desc"
            elif idx == 1:
                window["field"] = "ref"
            elif idx == 2:
                window["field"] = "qty"
            else:
                window["field"] = "desc"
        if windows and not windows[-1].get("field"):
            windows[-1]["field"] = "desc"

        seed_info: dict[str, dict[str, Any]] = {}
        for idx, window in enumerate(windows):
            field = window.get("field")
            if isinstance(field, str) and field not in seed_info:
                seed_info[field] = {
                    "col": idx,
                    "center": window.get("center"),
                    "span": (window.get("left"), window.get("right")),
                }
        return (windows, seed_info)

    column_windows: list[dict[str, Any]] = []
    header_seed_info: dict[str, dict[str, Any]] = {}
    header_band_index: int | None = None

    for idx, band in enumerate(bands):
        combined_upper = " ".join(
            str(item.get("text") or "").upper() for item in band if item.get("text")
        )
        if (
            "HOLE" in combined_upper
            and ("DESC" in combined_upper or "DESCRIPTION" in combined_upper)
            and any(
                token in combined_upper
                for token in ("REF", "Ø", "⌀", "DIA", "QUANTITY", "QTY")
            )
        ):
            header_band_index = idx
            column_windows, header_seed_info = _header_windows_from_band(band)
            break

    if header_band_index is None:
        for idx, band in enumerate(bands):
            non_empty = [
                item
                for item in band
                if str(item.get("text") or "").strip()
            ]
            if len(non_empty) >= 3:
                header_band_index = idx
                column_windows, header_seed_info = _header_windows_from_band(band)
                if column_windows:
                    break
        if header_band_index is None:
            header_band_index = 0 if bands else None

    if column_windows:
        if "hole" not in header_seed_info and column_windows:
            header_seed_info["hole"] = {
                "col": 0,
                "center": column_windows[0].get("center"),
                "span": (
                    column_windows[0].get("left"),
                    column_windows[0].get("right"),
                ),
            }
        if "ref" not in header_seed_info and len(column_windows) >= 2:
            header_seed_info["ref"] = {
                "col": 1,
                "center": column_windows[1].get("center"),
                "span": (
                    column_windows[1].get("left"),
                    column_windows[1].get("right"),
                ),
            }
        if "qty" not in header_seed_info and len(column_windows) >= 3:
            qty_idx = 2 if len(column_windows) >= 3 else len(column_windows) - 2
            qty_idx = max(1, min(qty_idx, len(column_windows) - 2))
            header_seed_info["qty"] = {
                "col": qty_idx,
                "center": column_windows[qty_idx].get("center"),
                "span": (
                    column_windows[qty_idx].get("left"),
                    column_windows[qty_idx].get("right"),
                ),
            }
        if "desc" not in header_seed_info and column_windows:
            last_idx = len(column_windows) - 1
            header_seed_info["desc"] = {
                "col": last_idx,
                "center": column_windows[last_idx].get("center"),
                "span": (
                    column_windows[last_idx].get("left"),
                    column_windows[last_idx].get("right"),
                ),
            }

    header_initial_field_info: dict[str, dict[str, Any]] = {}
    if header_seed_info and header_band_index is not None:
        for field, info in header_seed_info.items():
            info_copy = dict(info)
            info_copy["band"] = header_band_index
            header_initial_field_info[field] = info_copy

    y_eps = y_gap_limit

    qty_x: float | None = None
    qty_groups: list[dict[str, Any]] = []
    selected_qty_group: dict[str, Any] | None = None
    qty_candidates: list[dict[str, Any]] = []
    for band in bands:
        for item in band:
            text_candidate = (item.get("text") or "").strip()
            if not text_candidate:
                continue
            if _QSTRIPE_CANDIDATE_RE.match(text_candidate):
                qty_candidates.append({"x": item.get("x"), "text": text_candidate})

    if qty_candidates:
        valid_candidates = [
            {"x": float(candidate["x"]), "text": candidate["text"]}
            for candidate in qty_candidates
            if isinstance(candidate.get("x"), (int, float))
        ]
        sorted_candidates = sorted(valid_candidates, key=lambda rec: rec["x"])
        x_diffs = [
            abs(sorted_candidates[idx]["x"] - sorted_candidates[idx - 1]["x"])
            for idx in range(1, len(sorted_candidates))
            if abs(sorted_candidates[idx]["x"] - sorted_candidates[idx - 1]["x"]) > 0
        ]
        if x_diffs:
            median_diff = statistics.median(x_diffs)
            x_stripe_eps = 0.6 * median_diff if median_diff > 0 else 0.0
        else:
            x_stripe_eps = 0.0
        x_stripe_eps = max(4.0, x_stripe_eps)
        for candidate in sorted_candidates:
            x_pos = candidate["x"]
            if not qty_groups:
                qty_groups.append(
                    {
                        "items": [candidate],
                        "count": 1,
                        "sum_x": x_pos,
                        "center": x_pos,
                    }
                )
                continue
            group = qty_groups[-1]
            if abs(x_pos - group["center"]) <= x_stripe_eps:
                group["items"].append(candidate)
                group["count"] += 1
                group["sum_x"] += x_pos
                group["center"] = group["sum_x"] / group["count"]
            else:
                qty_groups.append(
                    {
                        "items": [candidate],
                        "count": 1,
                        "sum_x": x_pos,
                        "center": x_pos,
                    }
                )
        if qty_groups:
            chosen_group = max(
                qty_groups,
                key=lambda group: (
                    group.get("count", 0),
                    -group.get("center", float("inf")),
                ),
            )
            qty_x = chosen_group.get("center")
            selected_qty_group = chosen_group
    qty_groups_count = len(qty_groups)
    qty_group_size = 0
    qty_x_display = "-"
    if qty_groups:
        if selected_qty_group is None:
            selected_qty_group = max(
                qty_groups,
                key=lambda group: (
                    group.get("count", 0),
                    -group.get("center", 0.0),
                ),
            )
        qty_group_size = selected_qty_group.get("count", 0)
        if qty_x is None:
            qty_x = selected_qty_group.get("center")
        if isinstance(qty_x, (int, float)):
            qty_x_display = f"{float(qty_x):.1f}"
    print(
        f"[QSTRIPE] groups={qty_groups_count} chosen_x≈{qty_x_display} count={qty_group_size}"
    )

    if not bands:
        return (
            None,
            {
                "bands": [],
                "band_cells": [],
                "rows_txt_fallback": [],
                "qty_col": None,
                "ref_col": None,
                "desc_col": None,
                "y_eps": y_eps,
                "roi": roi_info,
                "column_windows": column_windows,
                "header_band": header_band_index,
            },
        )

    for idx, band in enumerate(bands[:10]):
        mean_y = sum(item["y"] for item in band) / len(band)
        print(f"[TABLE-Y] band#{idx} y≈{mean_y:.2f} lines={len(band)}")

    band_summaries: list[dict[str, Any]] = []
    band_cells_dump: list[dict[str, Any]] = []
    band_results: list[dict[str, Any]] = []
    window_fields_master = [
        window.get("field") if isinstance(window, Mapping) else None
        for window in column_windows
    ] if column_windows else []

    for band_index, band in enumerate(bands):
        mean_y = sum(item["y"] for item in band) / len(band)
        band_summaries.append(
            {"index": band_index, "y_mean": mean_y, "line_count": len(band)}
        )
        sorted_band = sorted(band, key=lambda item: item["x"])
        columns: list[dict[str, Any]] = []
        x_eps_value = y_eps
        if column_windows:
            assignments: list[list[dict[str, Any]]] = [
                [] for _ in column_windows
            ]
            for item in sorted_band:
                x_val = item.get("x")
                try:
                    x_float = float(x_val)
                except Exception:
                    continue
                best_idx: int | None = None
                best_score: tuple[int, float, int] | None = None
                for col_idx, window in enumerate(column_windows):
                    center_val = float(window.get("center") or 0.0)
                    left_val = window.get("left")
                    right_val = window.get("right")
                    try:
                        left_float = float(left_val)
                    except Exception:
                        left_float = center_val
                    try:
                        right_float = float(right_val)
                    except Exception:
                        right_float = center_val
                    width = abs(right_float - left_float)
                    margin = max(1.0, 0.1 * width, 0.4 * median_h)
                    if margin <= 0:
                        margin = max(1.0, 0.4 * median_h, 2.0)
                    in_window = (left_float - margin) <= x_float <= (right_float + margin)
                    distance = abs(x_float - center_val)
                    score = (0 if in_window else 1, distance, col_idx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = col_idx
                if best_idx is None:
                    continue
                assignments[best_idx].append(item)
            for col_idx, window in enumerate(column_windows):
                col_items = sorted(assignments[col_idx], key=lambda itm: itm["x"])
                if col_items:
                    text_parts = [
                        str(itm.get("text") or "").strip()
                        for itm in col_items
                        if str(itm.get("text") or "").strip()
                    ]
                    cell_text = " ".join(text_parts).strip()
                    sum_x = sum(float(itm["x"]) for itm in col_items)
                    sum_y = sum(float(itm["y"]) for itm in col_items)
                    count = len(col_items)
                    center_x = sum_x / count if count else float(window.get("center") or 0.0)
                    center_y = sum_y / count if count else mean_y
                    min_x = min(float(itm["x"]) for itm in col_items)
                    max_x = max(float(itm["x"]) for itm in col_items)
                else:
                    cell_text = ""
                    center_x = float(window.get("center") or 0.0)
                    center_y = mean_y
                    try:
                        min_x = float(window.get("left"))
                    except Exception:
                        min_x = center_x
                    try:
                        max_x = float(window.get("right"))
                    except Exception:
                        max_x = center_x
                columns.append(
                    {
                        "items": col_items,
                        "center_x": center_x,
                        "center_y": center_y,
                        "min_x": min_x,
                        "max_x": max_x,
                        "text": cell_text,
                        "field": window_fields_master[col_idx]
                        if col_idx < len(window_fields_master)
                        else None,
                        "span_hint": (
                            float(column_windows[col_idx].get("left"))
                            if column_windows[col_idx].get("left") is not None
                            else min_x,
                            float(column_windows[col_idx].get("right"))
                            if column_windows[col_idx].get("right") is not None
                            else max_x,
                        ),
                    }
                )
        else:
            x_values = [item["x"] for item in sorted_band]
            x_gaps = [
                abs(x_values[pos + 1] - x_values[pos])
                for pos in range(len(x_values) - 1)
                if abs(x_values[pos + 1] - x_values[pos]) > 0
            ]
            gap_med = statistics.median(x_gaps) if x_gaps else 0.0
            gap_p75 = _percentile(x_gaps, 0.75) if x_gaps else gap_med
            if gap_p75 <= 0:
                gap_p75 = gap_med
            x_eps = max(6.0, 0.8 * gap_p75 if gap_p75 > 0 else 0.0)
            if x_eps <= 0:
                x_eps = 6.0
            x_eps_value = x_eps
            for item in sorted_band:
                x_pos = item["x"]
                if not columns:
                    columns.append(
                        {
                            "items": [item],
                            "sum_x": x_pos,
                            "sum_y": item["y"],
                            "count": 1,
                            "center_x": x_pos,
                            "center_y": item["y"],
                            "min_x": x_pos,
                            "max_x": x_pos,
                            "field": None,
                        }
                    )
                    continue
                column = columns[-1]
                if abs(x_pos - column["center_x"]) <= x_eps:
                    column["items"].append(item)
                    column["sum_x"] += x_pos
                    column["sum_y"] += item["y"]
                    column["count"] += 1
                    column["center_x"] = column["sum_x"] / column["count"]
                    column["center_y"] = column["sum_y"] / column["count"]
                    column["min_x"] = min(column.get("min_x", x_pos), x_pos)
                    column["max_x"] = max(column.get("max_x", x_pos), x_pos)
                    continue
                columns.append(
                    {
                        "items": [item],
                        "sum_x": x_pos,
                        "sum_y": item["y"],
                        "count": 1,
                        "center_x": x_pos,
                        "center_y": item["y"],
                        "min_x": x_pos,
                        "max_x": x_pos,
                        "field": None,
                    }
                )
            for column in columns:
                column.setdefault("span_hint", (
                    float(column.get("min_x", column.get("center_x", 0.0))),
                    float(column.get("max_x", column.get("center_x", 0.0))),
                ))

        cell_texts: list[str] = []
        preview_parts: list[str] = []
        for col_index, column in enumerate(columns):
            if column.get("items") is not None and "sum_x" in column:
                # legacy path; recompute text
                sorted_items = sorted(column.get("items", []), key=lambda itm: itm["x"])
                cell_text = " ".join(item["text"] for item in sorted_items).strip()
            else:
                cell_text = column.get("text", "") or ""
            if not isinstance(cell_text, str):
                cell_text = str(cell_text)
            cell_text = cell_text.strip()
            cell_texts.append(cell_text)
            cell_entry = {
                "band": band_index,
                "col": col_index,
                "x_center": column.get("center_x"),
                "y_center": column.get("center_y"),
                "text": cell_text,
            }
            field_name = column.get("field")
            if field_name:
                cell_entry["field"] = field_name
            band_cells_dump.append(cell_entry)
            preview_parts.append(
                f'C{col_index}="{_truncate_cell_preview(cell_text)}"'
            )

        if band_index < 10:
            preview_body = " | ".join(preview_parts)
            print(f"[TABLE-X] band#{band_index} cols={len(columns)} | {preview_body}")

        spans = []
        for column in columns:
            span_hint = column.get("span_hint")
            if isinstance(span_hint, tuple) and len(span_hint) == 2:
                try:
                    span_min = float(span_hint[0])
                    span_max = float(span_hint[1])
                except Exception:
                    span_min = float(column.get("min_x", column.get("center_x", 0.0)))
                    span_max = float(column.get("max_x", column.get("center_x", 0.0)))
            else:
                span_min = float(column.get("min_x", column.get("center_x", 0.0)))
                span_max = float(column.get("max_x", column.get("center_x", 0.0)))
            spans.append((span_min, span_max))

        band_results.append(
            {
                "cells": cell_texts,
                "y_mean": mean_y,
                "line_count": len(band),
                "centers": [column.get("center_x") for column in columns],
                "x_eps": x_eps_value,
                "spans": spans,
                "window_fields": window_fields_master if column_windows else None,
            }
        )

    if not band_results:
        return (
            None,
            {
                "bands": band_summaries,
                "band_cells": band_cells_dump,
                "rows_txt_fallback": [],
                "qty_col": None,
                "ref_col": None,
                "desc_col": None,
                "y_eps": y_eps,
                "roi": roi_info,
                "column_windows": column_windows,
                "header_band": header_band_index,
            },
        )

    header_field_info: dict[str, dict[str, Any]] = dict(header_initial_field_info)
    header_band_indices: set[int] = {
        int(info.get("band"))
        for info in header_initial_field_info.values()
        if isinstance(info.get("band"), int)
    }

    def _header_hits(cells: list[str]) -> dict[str, int]:
        hits: dict[str, int] = {}
        for idx, cell_text in enumerate(cells):
            upper = cell_text.upper()
            if not upper:
                continue
            if "QTY" in upper or "QUANTITY" in upper:
                hits.setdefault("qty", idx)
            if "DESC" in upper or "DESCRIPTION" in upper:
                hits.setdefault("desc", idx)
            if "SIDE" in upper or "FACE" in upper:
                hits.setdefault("side", idx)
            if any(token in upper for token in ("Ø", "⌀", "DIA", "REF")):
                hits.setdefault("ref", idx)
            if "HOLE" in upper or re.search(r"\bID\b", upper):
                hits.setdefault("hole", idx)
        return hits

    for band_index, band in enumerate(band_results):
        cells = band.get("cells", [])
        if not cells:
            continue
        hits = _header_hits(cells)
        if not hits:
            continue
        if "qty" not in hits and len(hits) < 2:
            continue
        header_band_indices.add(band_index)
        centers = band.get("centers") or []
        spans = band.get("spans") or []
        for field, col_idx in hits.items():
            if field in header_field_info:
                continue
            span_value = spans[col_idx] if col_idx < len(spans) else None
            center_value = centers[col_idx] if col_idx < len(centers) else None
            header_field_info[field] = {
                "band": band_index,
                "col": col_idx,
                "center": center_value,
                "span": span_value,
            }

    if header_field_info and "desc" not in header_field_info and header_band_indices:
        header_idx = min(header_band_indices)
        sample_band = band_results[header_idx]
        cells = sample_band.get("cells", [])
        centers = sample_band.get("centers") or []
        spans = sample_band.get("spans") or []
        used_cols = {info.get("col") for info in header_field_info.values() if isinstance(info.get("col"), int)}
        candidates = [idx for idx in range(len(cells)) if idx not in used_cols]
        if candidates:
            best_idx = max(candidates, key=lambda idx: len(cells[idx]))
            header_field_info["desc"] = {
                "band": header_idx,
                "col": best_idx,
                "center": centers[best_idx] if best_idx < len(centers) else None,
                "span": spans[best_idx] if best_idx < len(spans) else None,
            }

    def _match_header_index(field: str, band: dict[str, Any]) -> int | None:
        info = header_field_info.get(field)
        if not info:
            return None
        centers = band.get("centers") or []
        spans = band.get("spans") or []
        if not centers:
            return None
        target_center = info.get("center")
        header_col = info.get("col")
        span_hint = info.get("span")
        width = None
        if isinstance(span_hint, tuple) and len(span_hint) == 2:
            try:
                span_min = float(span_hint[0])
                span_max = float(span_hint[1])
                width = abs(span_max - span_min)
            except Exception:
                span_min = span_max = None
        else:
            span_min = span_max = None
        tolerance = max(6.0, (width or 0.0) * 0.6)
        best_idx: int | None = None
        best_score: tuple[float, float] | None = None
        for idx, center in enumerate(centers):
            if not isinstance(center, (int, float)):
                continue
            if span_min is not None and center < span_min - tolerance:
                continue
            if span_max is not None and center > span_max + tolerance:
                continue
            if isinstance(target_center, (int, float)):
                distance = abs(center - float(target_center))
            else:
                distance = 0.0
            if isinstance(header_col, int):
                column_delta = abs(idx - header_col)
            else:
                column_delta = float("inf")
            score = (distance, column_delta)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None and isinstance(header_col, int) and header_col < len(centers):
            best_idx = header_col
        return best_idx

    if header_field_info:
        for band in band_results:
            header_indices: dict[str, int] = {}
            for field in header_field_info:
                idx = _match_header_index(field, band)
                if idx is not None:
                    header_indices[field] = idx
            if header_indices:
                band["header_indices"] = header_indices

    column_count = max(len(band["cells"]) for band in band_results)
    header_cols: dict[str, int] = {}
    if header_field_info:
        for key in ("qty", "ref", "desc", "side", "hole"):
            value = header_field_info.get(key, {}).get("col")
            if isinstance(value, int) and 0 <= value < column_count:
                header_cols[key] = value
    if column_windows:
        window_fields = [
            window.get("field") if isinstance(window, Mapping) else None
            for window in column_windows
        ]
        for key in ("hole", "ref", "qty", "side", "desc"):
            if key in header_cols:
                continue
            try:
                idx = window_fields.index(key)  # type: ignore[arg-type]
            except ValueError:
                continue
            if isinstance(idx, int) and 0 <= idx < column_count:
                header_cols[key] = idx
    qty_digit_re = re.compile(r"^\d{1,3}$")
    qty_suffix_re = re.compile(r"\b(REQD|REQUIRED|RE'?D|EA|EACH|HOLES?)\b$", re.IGNORECASE)

    def _normalize_qty_cell(text: str) -> str:
        candidate = (text or "").strip()
        if not candidate:
            return ""
        normalized = candidate.upper()
        normalized = re.sub(r"^QTY[:.]?", "", normalized).strip()
        normalized = normalized.strip("() ")
        normalized = re.sub(r"[X×]+$", "", normalized).strip()
        while True:
            cleaned = qty_suffix_re.sub("", normalized).strip()
            if cleaned == normalized:
                break
            normalized = cleaned
        return normalized

    def _parse_qty_cell_value(cell_text: str) -> int | None:
        normalized = _normalize_qty_cell(cell_text)
        if not normalized:
            return None
        if qty_digit_re.match(normalized):
            try:
                value = int(normalized)
            except Exception:
                return None
            if 0 < value <= 999:
                return value
        return None
    qty_header_re = re.compile(r"^QTY[:.]?$", re.IGNORECASE)
    ref_value_re = re.compile(r"(Ø|⌀|\"|\d+\s*/\s*\d+|(?<!\d)\.\d+|\d+\.\d+)")
    ref_header_re = re.compile(r"^REF\b", re.IGNORECASE)

    bands_for_stats = band_results[:5]

    metrics: list[dict[str, Any]] = []
    for col_index in range(column_count):
        non_empty = 0
        numeric_hits = 0
        ref_hits = 0
        token_hits = 0
        total_len = 0
        header_hit = False
        ref_header_hit = False
        for band in bands_for_stats:
            cells = band.get("cells", [])
            if col_index >= len(cells):
                continue
            cell_text = cells[col_index].strip()
            if not cell_text:
                continue
            non_empty += 1
            total_len += len(cell_text)
            upper_text = cell_text.upper()
            if qty_header_re.match(upper_text):
                header_hit = True
            if ref_header_re.match(upper_text):
                ref_header_hit = True
            qty_candidate = _parse_qty_cell_value(cell_text)
            if qty_candidate is not None:
                numeric_hits += 1
            if ref_value_re.search(cell_text):
                ref_hits += 1
            if _HOLE_ACTION_TOKEN_RE.search(cell_text):
                token_hits += 1
        avg_len = (total_len / non_empty) if non_empty else 0.0
        numeric_ratio = (numeric_hits / non_empty) if non_empty else 0.0
        ref_ratio = (ref_hits / non_empty) if non_empty else 0.0
        action_ratio = (token_hits / non_empty) if non_empty else 0.0
        metrics.append(
            {
                "non_empty": non_empty,
                "numeric_hits": numeric_hits,
                "ref_hits": ref_hits,
                "token_hits": token_hits,
                "avg_len": avg_len,
                "header_qty": header_hit,
                "header_ref": ref_header_hit,
                "ref_ratio": ref_ratio,
                "numeric_ratio": numeric_ratio,
                "action_ratio": action_ratio,
            }
        )

    fallback_qty_col: int | None = header_cols.get("qty") if header_cols else None
    qty_candidates = [
        idx
        for idx, info in enumerate(metrics)
        if info["header_qty"] or (info["non_empty"] > 0 and info["numeric_ratio"] >= 0.6)
    ]
    if qty_candidates and fallback_qty_col is None:
        fallback_qty_col = min(qty_candidates)

    if fallback_qty_col is None:
        relaxed_candidates = [
            idx
            for idx, info in enumerate(metrics)
            if info["non_empty"] > 0 and info["numeric_ratio"] >= 0.4
        ]
        if relaxed_candidates:
            fallback_qty_col = min(relaxed_candidates)
        else:
            numericish_candidates = [
                idx
                for idx, info in enumerate(metrics)
                if info["numeric_hits"] > 0 or info["numeric_ratio"] > 0
            ]
            if numericish_candidates:
                fallback_qty_col = min(numericish_candidates)
            elif column_count > 0:
                fallback_qty_col = 0

    qty_votes: dict[int, int] = defaultdict(int)
    qty_index_counts: dict[int, int] = defaultdict(int)
    for band_idx, band in enumerate(band_results):
        cells = band.get("cells", [])
        centers = band.get("centers") or []
        x_eps_band = float(band.get("x_eps") or 0.0)
        qty_idx: int | None = None
        qty_cell_text = ""
        header_indices = band.get("header_indices") or {}
        header_qty_idx = header_indices.get("qty")
        if isinstance(header_qty_idx, int) and header_qty_idx < len(cells):
            qty_idx = header_qty_idx
            qty_cell_text = cells[header_qty_idx].strip()
        if qty_idx is None and qty_x is not None and centers:
            center_pairs = [
                (col_index, float(center))
                for col_index, center in enumerate(centers)
                if isinstance(center, (int, float))
            ]
            if center_pairs:
                closest_index, closest_center = min(
                    center_pairs,
                    key=lambda pair: (abs(pair[1] - qty_x), pair[0]),
                )
                qty_idx = closest_index
                if qty_idx < len(cells):
                    qty_cell_text = cells[qty_idx].strip()
                if _parse_qty_cell_value(qty_cell_text) is None:
                    within_eps: list[tuple[int, float]] = []
                    for col_index, center_value in center_pairs:
                        if abs(center_value - qty_x) <= x_eps_band + 1e-6:
                            if col_index < len(cells):
                                candidate_text = cells[col_index].strip()
                                if _parse_qty_cell_value(candidate_text) is not None:
                                    within_eps.append(
                                        (col_index, abs(center_value - qty_x))
                                    )
                    if within_eps:
                        best_index, _ = min(
                            within_eps, key=lambda pair: (pair[1], pair[0])
                        )
                        qty_idx = best_index
                        qty_cell_text = cells[qty_idx].strip() if qty_idx < len(cells) else ""
        if qty_idx is None and fallback_qty_col is not None:
            if fallback_qty_col < len(cells):
                qty_idx = fallback_qty_col
                qty_cell_text = cells[qty_idx].strip()
        band["qty_index"] = qty_idx
        if qty_idx is not None:
            qty_index_counts[qty_idx] += 1
            if (
                band_idx < len(bands_for_stats)
                and _parse_qty_cell_value(qty_cell_text or "") is not None
            ):
                qty_votes[qty_idx] += 1

    qty_col: int | None = None
    if qty_votes:
        qty_col = max(qty_votes.items(), key=lambda item: (item[1], -item[0]))[0]
    elif qty_index_counts:
        qty_col = max(qty_index_counts.items(), key=lambda item: (item[1], -item[0]))[0]
    else:
        qty_col = fallback_qty_col
    if "qty" in header_cols:
        qty_col = header_cols["qty"]

    ref_candidates = [
        idx
        for idx, info in enumerate(metrics)
        if info["header_ref"] or (info["non_empty"] > 0 and info["ref_ratio"] >= 0.4)
    ]
    ref_col = max(ref_candidates) if ref_candidates else None
    if "ref" in header_cols:
        ref_col = header_cols["ref"]

    desc_candidates = [idx for idx in range(column_count) if idx != qty_col and idx != ref_col]
    if not desc_candidates:
        desc_candidates = list(range(column_count))

    def _desc_score(index: int) -> tuple[float, float, int]:
        data = metrics[index]
        return (float(data["action_ratio"]), float(data["avg_len"]), -index)

    desc_col = max(desc_candidates, key=_desc_score)
    if "desc" in header_cols:
        desc_col = header_cols["desc"]

    print(
        "[TABLE-ID] qty_x≈{qx} qty_col={qty} ref_col={ref} desc_col={desc} bands_checked={bands}".format(
            qx=qty_x_display,
            qty=qty_col if qty_col is not None else "None",
            ref=ref_col if ref_col is not None else "None",
            desc=desc_col,
            bands=len(bands_for_stats),
        )
    )

    def _cell_at(cells: list[str], index: int | None) -> str:
        if index is None:
            return ""
        if index < 0 or index >= len(cells):
            return ""
        return cells[index].strip()

    rows: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    for band_index, band in enumerate(band_results):
        cells = band.get("cells", [])
        header_indices = band.get("header_indices") or {}

        def _resolved_index(default_idx: int | None, field: str) -> int | None:
            idx_val = header_indices.get(field)
            if isinstance(idx_val, int):
                return idx_val
            idx_val = header_cols.get(field)
            if isinstance(idx_val, int):
                return idx_val
            return default_idx

        qty_idx = _resolved_index(band.get("qty_index"), "qty")
        if qty_idx is None:
            qty_idx = qty_col
        qty_text = _cell_at(cells, qty_idx)
        qty_val = None
        if qty_text:
            qty_val = _parse_qty_cell_value(qty_text)
        used_qty_fallback = False
        fallback_desc = ""
        combined_row_text = " ".join(
            cell.strip() for cell in cells if cell.strip()
        )
        fallback_qty: int | None = None
        fallback_remainder = combined_row_text
        if combined_row_text:
            fallback_qty, fallback_remainder = _extract_band_quantity(
                combined_row_text
            )
            if fallback_qty is None or fallback_qty <= 0:
                fallback_qty, fallback_remainder = _extract_row_quantity_and_remainder(
                    combined_row_text
                )
        if (
            fallback_qty is not None
            and fallback_qty > 0
            and (qty_groups_count == 0 or qty_val is None or qty_val <= 0)
        ):
            qty_val = fallback_qty
            fallback_desc = fallback_remainder.strip()
            used_qty_fallback = True
            print(f"[QTY-FALLBACK] band#{band_index} qty={qty_val}")
        if qty_val is None or qty_val <= 0:
            continue
        desc_idx = _resolved_index(desc_col, "desc")
        desc_text = _cell_at(cells, desc_idx)
        if used_qty_fallback:
            desc_text = fallback_desc or desc_text
        if not desc_text:
            exclude_index = qty_idx if qty_idx is not None else qty_col
            fallback_parts = [
                cell.strip()
                for idx, cell in enumerate(cells)
                if idx != exclude_index and cell.strip()
            ]
            desc_text = " ".join(fallback_parts)
        desc_text = " ".join((desc_text or "").split())
        ref_idx_resolved = _resolved_index(ref_col, "ref")
        ref_text_candidate = _cell_at(cells, ref_idx_resolved)
        ref_cell_ref = ("", None)
        if ref_text_candidate:
            ref_cell_ref = _extract_row_reference(ref_text_candidate)
        hole_idx = _resolved_index(None, "hole")
        hole_text = _cell_at(cells, hole_idx)
        side_idx = _resolved_index(header_cols.get("side"), "side")
        side_cell_text = _cell_at(cells, side_idx)
        fragments = [frag.strip() for frag in desc_text.split(";") if frag.strip()]
        if not fragments:
            fragments = [desc_text]
        for frag_index, fragment in enumerate(fragments):
            fragment_desc = " ".join(fragment.split())
            if not fragment_desc:
                continue
            ref_text, ref_value = _extract_row_reference(fragment_desc)
            if not ref_text and ref_cell_ref[0]:
                ref_text, ref_value = ref_cell_ref
            combined_side_text = " ".join(
                part for part in (side_cell_text, fragment_desc) if part
            )
            side = _detect_row_side(combined_side_text)
            if not side:
                side = _detect_row_side(fragment_desc)
            if not side and side_cell_text:
                upper_side = side_cell_text.upper()
                if "BOTH" in upper_side:
                    side = "both"
                elif "BACK" in upper_side and "FRONT" not in upper_side:
                    side = "back"
                elif "FRONT" in upper_side and "BACK" not in upper_side:
                    side = "front"
            row_dict: dict[str, Any] = {
                "hole": hole_text or "",
                "qty": qty_val,
                "desc": fragment_desc,
                "ref": ref_text,
            }
            if side:
                row_dict["side"] = side
            rows.append(row_dict)
            if ref_value is not None:
                key = f"{ref_value:.4f}".rstrip("0").rstrip(".")
                families[key] = families.get(key, 0) + qty_val
            if band_index < 10:
                band_label = (
                    f"{band_index}.{frag_index}" if len(fragments) > 1 else str(band_index)
                )
                ref_display = ref_text or "-"
                side_display = side or "-"
                desc_display = _truncate_cell_preview(fragment_desc or "", 80)
                print(
                    f"[TABLE-R] band#{band_label} qty={qty_val} ref={ref_display} "
                    f'side={side_display} desc="{desc_display}"'
                )

    if len(rows) < 8:
        try:
            raw_path = Path("raw_lines.tsv")
            with raw_path.open("w", encoding="utf-8") as handle:
                handle.write("layout\tin_block\tx\ty\ttext\n")
                for rec in records_all:
                    layout_val = str(rec.get("layout") or "")
                    in_block = "1" if rec.get("from_block") else "0"
                    x_val = rec.get("x")
                    y_val = rec.get("y")
                    if isinstance(x_val, (int, float)):
                        x_text = f"{float(x_val):.3f}"
                    else:
                        x_text = ""
                    if isinstance(y_val, (int, float)):
                        y_text = f"{float(y_val):.3f}"
                    else:
                        y_text = ""
                    text_val = rec.get("text") or ""
                    handle.write(
                        f"{layout_val}\t{in_block}\t{x_text}\t{y_text}\t{text_val}\n"
                    )
        except Exception as exc:
            print(f"[COLUMN] failed to write raw_lines.tsv: {exc}")
        try:
            roi_path = Path("roi_cells.tsv")
            with roi_path.open("w", encoding="utf-8") as handle:
                handle.write("band\tcol\tx_center\ty_center\ttext\n")
                for cell in band_cells_dump:
                    band_idx = cell.get("band")
                    col_idx = cell.get("col")
                    x_center = cell.get("x_center")
                    y_center = cell.get("y_center")
                    if isinstance(x_center, (int, float)):
                        x_center_text = f"{float(x_center):.3f}"
                    else:
                        x_center_text = ""
                    if isinstance(y_center, (int, float)):
                        y_center_text = f"{float(y_center):.3f}"
                    else:
                        y_center_text = ""
                    text_val = cell.get("text") or ""
                    handle.write(
                        f"{band_idx}\t{col_idx}\t{x_center_text}\t{y_center_text}\t{text_val}\n"
                    )
        except Exception as exc:
            print(f"[COLUMN] failed to write roi_cells.tsv: {exc}")

    if not rows:
        return (
            None,
            {
                "bands": band_summaries,
                "band_cells": band_cells_dump,
                "rows_txt_fallback": [],
                "qty_col": qty_col,
                "ref_col": ref_col,
                "desc_col": desc_col,
                "y_eps": y_eps,
                "median_h": median_h,
                "roi": roi_info,
            },
        )

    total_qty = sum(row["qty"] for row in rows)
    table_info: dict[str, Any] = {
        "rows": rows,
        "hole_count": total_qty,
        "provenance_holes": "HOLE TABLE",
    }
    if families:
        table_info["hole_diam_families_in"] = families

    debug_info = {
        "bands": band_summaries,
        "band_cells": band_cells_dump,
        "rows_txt_fallback": rows,
        "qty_col": qty_col,
        "ref_col": ref_col,
        "desc_col": desc_col,
        "y_eps": y_eps,
        "median_h": median_h,
        "column_windows": column_windows,
        "header_band": header_band_index,
        "bands_checked": len(bands_for_stats),
        "qty_x": qty_x,
    }
    if roi_info is not None:
        debug_info["roi"] = roi_info
    else:
        debug_info["roi"] = {
            "anchors": 0,
            "bbox": None,
            "expanded": None,
            "total": len(records_all),
            "kept": len(records),
            "median_h": median_h,
        }
    return (table_info, debug_info)


def _fallback_text_table(lines: Iterable[str]) -> dict[str, Any]:
    merged = _merge_table_lines(lines)
    rows: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    total_qty = 0

    for entry in merged:
        qty_val, remainder = _extract_row_quantity_and_remainder(entry)
        if qty_val is None or qty_val <= 0:
            continue
        normalized_desc = " ".join(entry.split())
        if not normalized_desc:
            continue
        rows.append({"hole": "", "ref": "", "qty": qty_val, "desc": normalized_desc})
        total_qty += qty_val

        ref_text, ref_value = _extract_row_reference(remainder)
        if ref_text:
            rows[-1]["ref"] = ref_text
        side = _detect_row_side(normalized_desc)
        if side:
            rows[-1]["side"] = side
        if ref_value is not None:
            key = f"{ref_value:.4f}".rstrip("0").rstrip(".")
            families[key] = families.get(key, 0) + qty_val

    if not rows:
        return {}

    result: dict[str, Any] = {"rows": rows, "hole_count": total_qty}
    if families:
        result["hole_diam_families_in"] = families
    result["provenance_holes"] = "HOLE TABLE (TEXT_FALLBACK)"
    return result


def read_text_table(
    doc,
    *,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    roi_hint: Mapping[str, Any] | None = None,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    helper = _resolve_app_callable("extract_hole_table_from_text")
    _print_helper_debug("text", helper)
    global _LAST_TEXT_TABLE_DEBUG
    _LAST_TEXT_TABLE_DEBUG = {
        "candidates": [],
        "band_cells": [],
        "bands": [],
        "rows": [],
        "raw_lines": [],
        "roi_hint": roi_hint,
        "roi": None,
        "preferred_blocks": [],
    }
    roi_hint_effective: Mapping[str, Any] | None = roi_hint
    resolved_allowlist = _normalize_layer_allowlist(layer_allowlist)
    normalized_block_allow = _normalize_block_allowlist(block_name_allowlist)
    block_regex_patterns = _compile_block_name_patterns(block_name_regex)
    allowlist_display = (
        "None"
        if resolved_allowlist is None
        else "{" + ",".join(sorted(resolved_allowlist) or []) + "}"
    )
    table_lines: list[str] | None = None
    fallback_candidate: Mapping[str, Any] | None = None
    best_candidate: Mapping[str, Any] | None = None
    best_score: tuple[int, int] = (0, 0)
    text_rows_info: dict[str, Any] | None = None
    merged_rows: list[str] = []
    parsed_rows: list[dict[str, Any]] = []
    columnar_table_info: dict[str, Any] | None = None
    columnar_debug_info: dict[str, Any] | None = None

    def _analyze_helper_signature(func: Callable[..., Any]) -> tuple[bool, bool]:
        needs_lines = False
        allows_lines = False
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return (needs_lines, allows_lines)
        positional: list[inspect.Parameter] = []
        for parameter in signature.parameters.values():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                allows_lines = True
                continue
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                positional.append(parameter)
        if len(positional) >= 2:
            allows_lines = True
            required = [
                param
                for param in positional
                if param.default is inspect._empty
            ]
            if len(required) >= 2:
                needs_lines = True
        return (needs_lines, allows_lines)

    def ensure_lines() -> list[str]:
        nonlocal table_lines, text_rows_info, merged_rows, parsed_rows
        nonlocal columnar_table_info, columnar_debug_info, roi_hint_effective
        if table_lines is not None:
            return table_lines

        collected_entries: list[dict[str, Any]] = []
        merged_rows = []
        parsed_rows = []
        text_rows_info = None
        rows_txt_initial = 0
        hint_logged = False
        attrib_count = 0
        preferred_block_names: list[str] = []
        preferred_block_rois: list[dict[str, Any]] = []
        block_height_samples: defaultdict[str, list[float]] = defaultdict(list)

        if doc is None:
            table_lines = []
            return table_lines

        def _iter_layouts() -> list[tuple[str, Any]]:
            layouts: list[tuple[str, Any]] = []
            modelspace = getattr(doc, "modelspace", None)
            if callable(modelspace):
                try:
                    layout_obj = modelspace()
                except Exception:
                    layout_obj = None
                if layout_obj is not None:
                    layouts.append(("Model", layout_obj))

            layouts_manager = getattr(doc, "layouts", None)
            if layouts_manager is None:
                return layouts
            names: list[Any]
            try:
                raw_names = getattr(layouts_manager, "names", None)
                if callable(raw_names):
                    names_iter = raw_names()
                else:
                    names_iter = raw_names
                names = list(names_iter or [])
            except Exception:
                names = []
            get_layout = getattr(layouts_manager, "get", None)
            for name in names:
                if not isinstance(name, str):
                    continue
                if name.lower() == "model":
                    continue
                layout_obj = None
                if callable(get_layout):
                    try:
                        layout_obj = get_layout(name)
                    except Exception:
                        layout_obj = None
                if layout_obj is not None:
                    layouts.append((name, layout_obj))
            return layouts

        def _extract_coords(entity: Any) -> tuple[float | None, float | None]:
            insert = None
            dxf_obj = getattr(entity, "dxf", None)
            if dxf_obj is not None:
                insert = getattr(dxf_obj, "insert", None)
            if insert is None:
                insert = getattr(entity, "insert", None)
            x_val: float | None = None
            y_val: float | None = None
            if insert is not None:
                x_val = getattr(insert, "x", None)
                y_val = getattr(insert, "y", None)
                if (x_val is None or y_val is None) and hasattr(insert, "__iter__"):
                    try:
                        parts = list(insert)
                    except Exception:
                        parts = []
                    if x_val is None and len(parts) >= 1:
                        x_val = parts[0]
                    if y_val is None and len(parts) >= 2:
                        y_val = parts[1]
            try:
                x_float = float(x_val) if x_val is not None else None
            except Exception:
                x_float = None
            try:
                y_float = float(y_val) if y_val is not None else None
            except Exception:
                y_float = None
            return (x_float, y_float)

        def _extract_text_height(entity: Any) -> float | None:
            dxf_obj = getattr(entity, "dxf", None)
            height_candidates: list[Any] = []
            if dxf_obj is not None:
                for attr in ("height", "char_height", "text_height", "thickness"):
                    height_candidates.append(getattr(dxf_obj, attr, None))
            height_candidates.append(getattr(entity, "height", None))
            for candidate in height_candidates:
                if candidate is None:
                    continue
                try:
                    value = float(candidate)
                except Exception:
                    continue
                if value > 0:
                    return value
            return None

        def _extract_layer(entity: Any) -> str:
            dxf_obj = getattr(entity, "dxf", None)
            candidates: list[Any] = []
            if dxf_obj is not None:
                candidates.append(getattr(dxf_obj, "layer", None))
            candidates.append(getattr(entity, "layer", None))
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    text = str(candidate).strip()
                except Exception:
                    continue
                if text:
                    return text
            return ""

        debug_enabled = _debug_entities_enabled()

        for layout_index, (layout_name, layout_obj) in enumerate(_iter_layouts()):
            query = getattr(layout_obj, "query", None)
            base_entities: list[Any] = []
            if callable(query):
                try:
                    base_entities = list(query("TEXT, MTEXT, INSERT"))
                except Exception:
                    base_entities = []
                if not base_entities:
                    for spec in ("TEXT", "MTEXT", "INSERT"):
                        try:
                            base_entities.extend(list(query(spec)))
                        except Exception:
                            continue
            if not base_entities:
                try:
                    base_entities = list(layout_obj)
                except Exception:
                    base_entities = []
            if not base_entities:
                continue

            seen_entities: set[int] = set()
            text_fragments = 0
            mtext_fragments = 0
            kept_count = 0
            from_blocks_count = 0
            counter = 0
            visited_blocks: set[str] = set()

            def _iter_insert_attributes(entity: Any) -> Iterable[Any]:
                attr_seen: set[int] = set()
                for attr_name in ("attribs", "attribs_raw"):
                    attr_value = getattr(entity, attr_name, None)
                    if attr_value is None:
                        continue
                    if callable(attr_value):
                        try:
                            attr_iterable = attr_value()
                        except Exception:
                            continue
                    else:
                        attr_iterable = attr_value
                    if attr_iterable is None:
                        continue
                    try:
                        iterator = list(attr_iterable)
                    except Exception:
                        iterator = [attr_iterable]
                    for attr_entity in iterator:
                        if attr_entity is None:
                            continue
                        marker = id(attr_entity)
                        if marker in attr_seen:
                            continue
                        attr_seen.add(marker)
                        yield attr_entity

            def _process_entity(
                entity: Any,
                *,
                depth: int,
                from_block: bool,
                parent_effective_layer: str | None,
                active_block: str | None,
            ) -> None:
                nonlocal text_fragments, mtext_fragments, kept_count, from_blocks_count, counter
                nonlocal attrib_count
                nonlocal hint_logged
                if depth > _MAX_INSERT_DEPTH:
                    return
                dxftype = None
                try:
                    dxftype = entity.dxftype()
                except Exception:
                    dxftype = None
                kind = str(dxftype or "").upper()
                layer_name = _extract_layer(entity)
                layer_upper = layer_name.upper() if layer_name else ""
                effective_layer = layer_name
                effective_layer_upper = layer_upper
                if not effective_layer_upper or effective_layer_upper == "0":
                    candidate = parent_effective_layer or layer_name or ""
                    effective_layer = candidate
                    effective_layer_upper = candidate.upper() if candidate else ""
                if kind in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}:
                    coords = _extract_coords(entity)
                    text_height = _extract_text_height(entity)
                    for fragment, is_mtext in _iter_entity_text_fragments(entity):
                        normalized = _normalize_table_fragment(fragment)
                        if not normalized:
                            continue
                        if kind in {"ATTRIB", "ATTDEF"}:
                            attrib_count += 1
                        if (
                            not hint_logged
                            and "SEE SHEET 2 FOR HOLE CHART" in normalized.upper()
                        ):
                            print(
                                "[HINT] Chart may live on an alternate sheet/block; ensure its INSERT is present and not on a frozen/off layer."
                            )
                            hint_logged = True
                        entry = {
                            "layout_index": layout_index,
                            "layout_name": layout_name,
                            "text": normalized,
                            "x": coords[0],
                            "y": coords[1],
                            "order": counter,
                            "from_block": from_block,
                            "height": text_height,
                            "layer": layer_name,
                            "layer_upper": layer_upper,
                            "effective_layer": effective_layer,
                            "effective_layer_upper": effective_layer_upper,
                            "block_name": active_block,
                        }
                        counter += 1
                        collected_entries.append(entry)
                        kept_count += 1
                        if (
                            active_block
                            and isinstance(text_height, (int, float))
                            and float(text_height) > 0
                        ):
                            block_height_samples[active_block].append(float(text_height))
                        if is_mtext:
                            mtext_fragments += 1
                        else:
                            text_fragments += 1
                        if from_block:
                            from_blocks_count += 1
                elif kind == "INSERT" and depth < _MAX_INSERT_DEPTH:
                    block_name = None
                    dxf_obj = getattr(entity, "dxf", None)
                    if dxf_obj is not None:
                        block_name = getattr(dxf_obj, "name", None)
                    if block_name is None:
                        block_name = getattr(entity, "name", None)
                    name_str = block_name if isinstance(block_name, str) else None
                    if name_str and name_str in visited_blocks:
                        return
                    if name_str:
                        visited_blocks.add(name_str)
                    try:
                        virtual_entities = list(entity.virtual_entities())
                    except Exception:
                        virtual_entities = []
                    child_parent_layer = effective_layer or parent_effective_layer
                    is_preferred_block = False
                    if name_str:
                        name_upper = name_str.upper()
                        if name_upper in normalized_block_allow:
                            is_preferred_block = True
                        elif any(pattern.search(name_str) for pattern in block_regex_patterns):
                            is_preferred_block = True
                        elif _PREFERRED_BLOCK_NAME_RE.search(name_str):
                            is_preferred_block = True
                    if is_preferred_block and name_str:
                        if name_str not in preferred_block_names:
                            preferred_block_names.append(name_str)
                        bbox = _compute_entity_bbox(
                            entity,
                            include_virtual=True,
                            virtual_entities=virtual_entities,
                        )
                        if bbox is not None:
                            preferred_block_rois.append(
                                {
                                    "name": name_str,
                                    "layer": effective_layer or layer_name,
                                    "bbox": bbox,
                                }
                            )
                    if virtual_entities:
                        for child in virtual_entities:
                            _process_entity(
                                child,
                                depth=depth + 1,
                                from_block=True,
                                parent_effective_layer=child_parent_layer,
                                active_block=name_str or active_block,
                            )
                    else:
                        blocks = getattr(doc, "blocks", None)
                        block_layout = None
                        if blocks is not None and name_str:
                            get_block = getattr(blocks, "get", None)
                            if callable(get_block):
                                try:
                                    block_layout = get_block(name_str)
                                except Exception:
                                    block_layout = None
                        if block_layout is not None and depth + 1 <= _MAX_INSERT_DEPTH:
                            for child in block_layout:
                                _process_entity(
                                    child,
                                    depth=depth + 1,
                                    from_block=True,
                                    parent_effective_layer=child_parent_layer,
                                    active_block=name_str or active_block,
                                )
                    for attribute in _iter_insert_attributes(entity):
                        _process_entity(
                            attribute,
                            depth=depth + 1,
                            from_block=True,
                            parent_effective_layer=child_parent_layer,
                            active_block=name_str or active_block,
                        )
                    if name_str:
                        visited_blocks.discard(name_str)

            for entity in base_entities:
                marker = id(entity)
                if marker in seen_entities:
                    continue
                seen_entities.add(marker)
                _process_entity(
                    entity,
                    depth=0,
                    from_block=False,
                    parent_effective_layer=None,
                    active_block=None,
                )

            print(
                f"[TEXT-SCAN] layout={layout_name} text={text_fragments} "
                f"mtext={mtext_fragments} kept={kept_count} from_blocks={from_blocks_count}"
            )

        if preferred_block_names:
            print(f"[TEXT-SCAN] preferred_blocks={preferred_block_names}")
        _LAST_TEXT_TABLE_DEBUG["preferred_blocks"] = list(preferred_block_names)
        print(
            f"[TEXT-SCAN] attrib_lines={attrib_count} depth_max={_MAX_INSERT_DEPTH} "
            f"allow_layers={allowlist_display}"
        )

        def _format_layer_summary(counts: Mapping[str, int]) -> str:
            if not counts:
                return "{}"
            top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            return "{" + ", ".join(f"{name or '-'}:{count}" for name, count in top) + "}"

        layer_counts_pre: dict[str, int] = defaultdict(int)
        for entry in collected_entries:
            layer_key = str(
                entry.get("effective_layer")
                or entry.get("layer")
                or ""
            ).strip()
            layer_counts_pre[layer_key] += 1
        print(f"[TEXT-SCAN] kept_by_layer(pre)={_format_layer_summary(layer_counts_pre)}")

        if resolved_allowlist is not None:
            filtered_entries = [
                entry
                for entry in collected_entries
                if not (entry.get("effective_layer_upper") or "")
                or (entry.get("effective_layer_upper") or "") in resolved_allowlist
            ]
        else:
            filtered_entries = list(collected_entries)

        if (
            resolved_allowlist is not None
            and not filtered_entries
            and collected_entries
        ):
            print("[TEXT-SCAN] layer-allowlist emptied set; falling back to no layer filter")
            filtered_entries = list(collected_entries)

        layer_counts_post: dict[str, int] = defaultdict(int)
        for entry in filtered_entries:
            layer_key = str(
                entry.get("effective_layer")
                or entry.get("layer")
                or ""
            ).strip()
            layer_counts_post[layer_key] += 1
        print(
            f"[TEXT-SCAN] kept_by_layer(post-allow)={_format_layer_summary(layer_counts_post)}"
        )

        collected_entries = filtered_entries

        if roi_hint_effective is None and preferred_block_rois:
            block_hint: Mapping[str, Any] | None = None
            for block_info in preferred_block_rois:
                bbox = block_info.get("bbox")
                if not bbox:
                    continue
                name = block_info.get("name")
                layer = block_info.get("layer")
                heights = block_height_samples.get(str(name) if name else "")
                median_height = (
                    statistics.median(heights)
                    if heights
                    else 0.0
                )
                pad = 2.0 * median_height if median_height > 0 else 6.0
                block_hint = {
                    "source": "BLOCK",
                    "name": name,
                    "layer": layer,
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ],
                    "pad": pad,
                    "median_height": median_height,
                }
                break
            if block_hint is not None:
                roi_hint_effective = block_hint
                _LAST_TEXT_TABLE_DEBUG["roi_hint"] = dict(block_hint)

        if not collected_entries:
            table_lines = []
            print("[TEXT-SCAN] rows_txt count=0")
            print("[TEXT-SCAN] parsed rows: 0")
            return table_lines

        def _entry_sort_key(entry: dict[str, Any]) -> tuple[float, float, int, int]:
            x_val = entry.get("x")
            y_val = entry.get("y")
            try:
                y_key = -float(y_val) if y_val is not None else float("inf")
            except Exception:
                y_key = float("inf")
            try:
                x_key = float(x_val) if x_val is not None else float("inf")
            except Exception:
                x_key = float("inf")
            return (y_key, x_key, int(entry.get("layout_index", 0)), int(entry.get("order", 0)))

        collected_entries.sort(key=_entry_sort_key)

        candidate_entries: list[dict[str, Any]] = []
        row_active = False
        continuation_budget = 0
        for idx, entry in enumerate(collected_entries):
            stripped = entry.get("text", "").strip()
            if not stripped:
                row_active = False
                continuation_budget = 0
                continue
            next_text = (
                collected_entries[idx + 1].get("text", "")
                if idx + 1 < len(collected_entries)
                else None
            )
            row_start = _is_row_start(stripped, next_text=next_text)
            token_hit = _has_candidate_token(stripped)
            keep_line = False
            if row_start:
                row_active = True
                continuation_budget = 3
                keep_line = True
            elif token_hit:
                keep_line = True
                if row_active:
                    continuation_budget = max(continuation_budget, 1)
                row_active = row_active or token_hit
            elif row_active and continuation_budget > 0:
                keep_line = True
                continuation_budget -= 1
            else:
                row_active = False
                continuation_budget = 0
            if keep_line:
                candidate_entries.append(entry)

        if debug_enabled and candidate_entries:
            limit = min(40, len(candidate_entries))
            print(f"[TEXT-SCAN] candidates[0..{limit - 1}]:")
            for idx, entry in enumerate(candidate_entries[:40]):
                x_val = entry.get("x")
                y_val = entry.get("y")
                if isinstance(x_val, (int, float)):
                    x_display = f"{float(x_val):.3f}"
                else:
                    x_display = "-"
                if isinstance(y_val, (int, float)):
                    y_display = f"{float(y_val):.3f}"
                else:
                    y_display = "-"
                print(
                    f"  [{idx:02d}] (x={x_display} y={y_display}) text=\"{entry.get('text', '')}\""
                )

        normalized_entries: list[dict[str, Any]] = []
        normalized_lines: list[str] = []
        for entry in candidate_entries:
            raw_line = str(entry.get("text", ""))
            match = _match_row_quantity(raw_line)
            if match:
                prefix = raw_line[: match.start()].strip(" |")
                suffix = raw_line[match.end() :].strip()
                row_token = match.group(0).strip()
                parts = [row_token]
                if prefix:
                    parts.append(prefix)
                if suffix:
                    parts.append(suffix)
                normalized_line = " ".join(parts)
            else:
                normalized_line = raw_line
            normalized_line = " ".join(normalized_line.split())
            entry_copy = dict(entry)
            entry_copy["normalized_text"] = normalized_line
            normalized_entries.append(entry_copy)
            normalized_lines.append(normalized_line)

        candidate_entries = normalized_entries
        table_lines = normalized_lines

        debug_candidates: list[dict[str, Any]] = []
        for entry in candidate_entries:
            debug_candidates.append(
                {
                    "layout": entry.get("layout_name"),
                    "in_block": bool(entry.get("from_block")),
                    "x": entry.get("x"),
                    "y": entry.get("y"),
                    "text": entry.get("normalized_text")
                    or entry.get("text")
                    or "",
                }
            )
        _LAST_TEXT_TABLE_DEBUG["candidates"] = debug_candidates

        current_row: list[str] = []
        for idx, entry in enumerate(candidate_entries):
            line = entry.get("normalized_text", "").strip()
            if not line:
                continue
            next_line = (
                candidate_entries[idx + 1].get("normalized_text", "")
                if idx + 1 < len(candidate_entries)
                else None
            )
            if _is_row_start(line, next_text=next_line):
                if current_row:
                    merged_rows.append(" ".join(current_row))
                current_row = [line]
            elif current_row:
                current_row.append(line)
        if current_row:
            merged_rows.append(" ".join(current_row))

        rows_txt_initial = len(merged_rows)
        print(f"[TEXT-SCAN] rows_txt count={len(merged_rows)}")
        for idx, row_text in enumerate(merged_rows[:10]):
            print(f"  [{idx:02d}] {row_text}")

        def _parse_rows(row_texts: list[str]) -> tuple[list[dict[str, Any]], dict[str, int], int]:
            families: dict[str, int] = {}
            parsed: list[dict[str, Any]] = []
            total = 0
            for row_text in row_texts:
                text_value = " ".join((row_text or "").split()).strip()
                if not text_value:
                    continue
                qty_val, remainder = _extract_row_quantity_and_remainder(text_value)
                if qty_val is None or qty_val <= 0:
                    continue
                ref_text, ref_value = _extract_row_reference(remainder)
                side = _detect_row_side(text_value)
                row_dict: dict[str, Any] = {
                    "hole": "",
                    "qty": qty_val,
                    "desc": text_value,
                    "ref": ref_text,
                }
                if side:
                    row_dict["side"] = side
                parsed.append(row_dict)
                total += qty_val
                if ref_value is not None:
                    key = f"{ref_value:.4f}".rstrip("0").rstrip(".")
                    families[key] = families.get(key, 0) + qty_val
            return (parsed, families, total)

        parsed_rows, families, total_qty = _parse_rows(merged_rows)

        def _cluster_entries_by_y(
            entries: list[dict[str, Any]]
        ) -> list[list[dict[str, Any]]]:
            valid = [entry for entry in entries if entry.get("normalized_text")]
            if not valid:
                return []

            def _estimate_eps(values: list[dict[str, Any]]) -> float:
                y_values: list[float] = []
                for item in values:
                    y_val = item.get("y")
                    if isinstance(y_val, (int, float)):
                        y_values.append(float(y_val))
                if len(y_values) >= 2:
                    diffs = [abs(y_values[i] - y_values[i + 1]) for i in range(len(y_values) - 1)]
                    diffs = [diff for diff in diffs if diff > 0]
                    if diffs:
                        median_diff = statistics.median(diffs)
                        if median_diff > 0:
                            return max(4.0, median_diff * 0.75)
                return 8.0

            eps = _estimate_eps(valid)
            for _ in range(3):
                clusters: list[list[dict[str, Any]]] = []
                current: list[dict[str, Any]] | None = None
                prev_y: float | None = None
                for entry in valid:
                    y_val = entry.get("y")
                    y_float = float(y_val) if isinstance(y_val, (int, float)) else None
                    if current is None:
                        current = [entry]
                        clusters.append(current)
                        prev_y = y_float
                        continue
                    if y_float is None or prev_y is None or abs(y_float - prev_y) > eps:
                        current = [entry]
                        clusters.append(current)
                    else:
                        current.append(entry)
                    prev_y = y_float if y_float is not None else prev_y
                if not clusters:
                    return []
                avg_cluster_size = len(valid) / len(clusters)
                if avg_cluster_size >= 1.5 or eps >= 24.0:
                    return clusters
                eps *= 1.5
            return clusters

        def _clusters_to_rows(clusters: list[list[dict[str, Any]]]) -> list[str]:
            rows: list[str] = []
            for cluster in clusters:
                def _x_key(value: Any) -> float:
                    try:
                        return float(value)
                    except Exception:
                        return float("inf")

                ordered = sorted(
                    cluster,
                    key=lambda item: (
                        _x_key(item.get("x")),
                        int(item.get("order", 0)),
                    ),
                )
                parts = [str(item.get("normalized_text", "")).strip() for item in ordered]
                row_text = " ".join(part for part in parts if part)
                row_text = " ".join(row_text.split())
                if not row_text:
                    continue
                if not _HOLE_ACTION_TOKEN_RE.search(row_text):
                    continue
                rows.append(row_text)
            return rows

        if len(parsed_rows) < 8:
            clusters = _cluster_entries_by_y(candidate_entries)
            fallback_rows = _clusters_to_rows(clusters)
            fallback_rows = [row for row in fallback_rows if row]
            fallback_parsed, fallback_families, fallback_qty = _parse_rows(fallback_rows)
            print(
                f"[TEXT-SCAN] fallback clusters={len(clusters)} "
                f"chosen_rows={len(fallback_parsed)} qty_sum={fallback_qty}"
            )
            if fallback_parsed and (
                (fallback_qty, len(fallback_parsed))
                > (total_qty, len(parsed_rows))
            ):
                merged_rows = fallback_rows
                parsed_rows = fallback_parsed
                families = fallback_families
                total_qty = fallback_qty

        if rows_txt_initial < 8:
            chart_lines: list[dict[str, Any]] = []
            sheet_lines: list[dict[str, Any]] = []
            model_lines: list[dict[str, Any]] = []
            other_lines: list[dict[str, Any]] = []
            for entry in collected_entries:
                text_value = str(entry.get("text") or "").strip()
                if not text_value:
                    continue
                record = {
                    "layout_name": entry.get("layout_name"),
                    "from_block": bool(entry.get("from_block")),
                    "x": entry.get("x"),
                    "y": entry.get("y"),
                    "height": entry.get("height"),
                    "text": text_value,
                    "normalized_text": text_value,
                }
                layout_name = str(entry.get("layout_name") or "")
                lower_name = layout_name.lower()
                if "chart" in lower_name:
                    chart_lines.append(record)
                elif "sheet" in lower_name:
                    sheet_lines.append(record)
                elif lower_name == "model":
                    model_lines.append(record)
                else:
                    other_lines.append(record)
            raw_lines = chart_lines + sheet_lines + model_lines
            if not raw_lines:
                raw_lines = list(other_lines)
            _LAST_TEXT_TABLE_DEBUG["raw_lines"] = [
                {
                    "layout": item.get("layout_name"),
                    "in_block": bool(item.get("from_block")),
                    "x": item.get("x"),
                    "y": item.get("y"),
                    "text": item.get("text"),
                }
                for item in raw_lines
            ]
            block_count = sum(1 for item in raw_lines if item.get("from_block"))
            print(
                "[COLUMN] raw_lines total={total} (chart={chart} sheet={sheet} "
                "model={model}) blocks={blocks}".format(
                    total=len(raw_lines),
                    chart=len(chart_lines),
                    sheet=len(sheet_lines),
                    model=len(model_lines),
                    blocks=block_count,
                )
            )
            if raw_lines:
                table_candidate, debug_payload = _build_columnar_table_from_entries(
                    raw_lines, roi_hint=roi_hint_effective
                )
                columnar_table_info = table_candidate
                columnar_debug_info = debug_payload
                if isinstance(debug_payload, Mapping):
                    _LAST_TEXT_TABLE_DEBUG["bands"] = list(
                        debug_payload.get("bands", [])
                    )
                    _LAST_TEXT_TABLE_DEBUG["band_cells"] = list(
                        debug_payload.get("band_cells", [])
                    )
                    _LAST_TEXT_TABLE_DEBUG["rows"] = list(
                        debug_payload.get("rows_txt_fallback", [])
                    )
                    if "roi" in debug_payload:
                        _LAST_TEXT_TABLE_DEBUG["roi"] = debug_payload.get("roi")

        print(f"[TEXT-SCAN] parsed rows: {len(parsed_rows)}")
        for idx, row in enumerate(parsed_rows[:20]):
            ref_val = row.get("ref") or ""
            side_val = row.get("side") or ""
            desc_val = row.get("desc") or ""
            if len(desc_val) > 80:
                desc_val = desc_val[:77] + "..."
            print(
                f"  [{idx:02d}] qty={row.get('qty')} ref={ref_val or '-'} "
                f"side={side_val or '-'} desc={desc_val}"
            )

        if parsed_rows:
            text_rows_info = {
                "rows": parsed_rows,
                "hole_count": total_qty,
                "provenance_holes": "HOLE TABLE",
            }
            if families:
                text_rows_info["hole_diam_families_in"] = families
        else:
            text_rows_info = None

        return table_lines

    def _log_and_normalize(label: str, result: Any) -> tuple[dict[str, Any] | None, tuple[int, int]]:
        rows_list: list[Any] = []
        candidate_map: dict[str, Any] | None = None
        if isinstance(result, Mapping):
            candidate_map = dict(result)
            rows_value = candidate_map.get("rows")
            if isinstance(rows_value, list):
                rows_list = rows_value
            elif isinstance(rows_value, Iterable) and not isinstance(
                rows_value, (str, bytes, bytearray)
            ):
                rows_list = list(rows_value)
                candidate_map["rows"] = rows_list
            else:
                rows_list = []
                candidate_map["rows"] = rows_list
        qty_total = _sum_qty(rows_list)
        row_count = len(rows_list)
        print(f"[TEXT-SCAN] helper={label} rows={row_count} qty={qty_total}")
        return candidate_map, (qty_total, row_count)

    lines = ensure_lines()

    if isinstance(text_rows_info, Mapping):
        fallback_candidate = text_rows_info
        scan_score = _score_table(text_rows_info)
        if scan_score[1] > 0 and scan_score > best_score:
            best_candidate = text_rows_info
            best_score = scan_score

    if callable(helper):
        needs_lines, allows_lines = _analyze_helper_signature(helper)
        use_lines = needs_lines or allows_lines
        args: list[Any] = [doc]
        if use_lines:
            args.append(lines)
        try:
            helper_result = helper(*args)
        except TypeError as exc:
            if use_lines and allows_lines and not needs_lines:
                try:
                    helper_result = helper(doc)
                    use_lines = False
                except Exception as inner_exc:
                    print(f"[EXTRACT] text helper error: {inner_exc}")
                    raise
            else:
                print(f"[EXTRACT] text helper error: {exc}")
                raise
        except Exception as exc:
            print(f"[EXTRACT] text helper error: {exc}")
            raise
        helper_map, helper_score = _log_and_normalize(
            "extract_hole_table_from_text", helper_result or {}
        )
        if helper_map is not None:
            if fallback_candidate is None:
                fallback_candidate = helper_map
            if helper_score[1] > 0 and helper_score > best_score:
                best_candidate = helper_map
                best_score = helper_score

    legacy_helper = _resolve_app_callable("hole_count_from_text_table")
    _print_helper_debug("text_alt", legacy_helper)
    if callable(legacy_helper):
        needs_lines, allows_lines = _analyze_helper_signature(legacy_helper)
        use_lines = needs_lines or allows_lines
        args: list[Any] = [doc]
        if use_lines:
            args.append(lines)
        try:
            legacy_result = legacy_helper(*args)
        except TypeError as exc:
            if use_lines and allows_lines and not needs_lines:
                try:
                    legacy_result = legacy_helper(doc)
                    use_lines = False
                except Exception as inner_exc:
                    print(f"[EXTRACT] text helper error: {inner_exc}")
                    raise
            else:
                print(f"[EXTRACT] text helper error: {exc}")
                raise
        except Exception as exc:
            print(f"[EXTRACT] text helper error: {exc}")
            raise
        legacy_map, legacy_score = _log_and_normalize(
            "hole_count_from_text_table", legacy_result or {}
        )
        if legacy_map is not None:
            if fallback_candidate is None:
                fallback_candidate = legacy_map
            if legacy_score[1] > 0 and legacy_score > best_score:
                best_candidate = legacy_map
                best_score = legacy_score

    primary_result: dict[str, Any] | None = None
    if isinstance(best_candidate, Mapping):
        primary_result = dict(best_candidate)
    elif isinstance(text_rows_info, Mapping):
        primary_result = dict(text_rows_info)
    elif isinstance(fallback_candidate, Mapping):
        primary_result = dict(fallback_candidate)

    columnar_result: dict[str, Any] | None = None
    if isinstance(columnar_table_info, Mapping):
        columnar_result = dict(columnar_table_info)

    column_selected = False
    if columnar_result:
        existing_score = _score_table(primary_result)
        fallback_score = _score_table(columnar_result)
        if fallback_score[1] > 0 and fallback_score > existing_score:
            rows_count = fallback_score[1]
            qty_sum = fallback_score[0]
            print(
                f"[EXTRACT] promoted table rows={rows_count} qty_sum={qty_sum} "
                "source=text_table (column-mode+stripe)"
            )
            primary_result = columnar_result
            column_selected = True

    if primary_result is None:
        fallback = _fallback_text_table(lines)
        if fallback:
            _LAST_TEXT_TABLE_DEBUG["rows"] = list(fallback.get("rows", []))
            return fallback
        _LAST_TEXT_TABLE_DEBUG["rows"] = []
        return {}

    if column_selected:
        _LAST_TEXT_TABLE_DEBUG["rows"] = list(primary_result.get("rows", []))
        return primary_result

    _LAST_TEXT_TABLE_DEBUG["rows"] = list(primary_result.get("rows", []))
    return primary_result


def _normalize_table_rows(rows_value: Any) -> list[dict[str, Any]]:
    if isinstance(rows_value, list):
        source = rows_value
    elif isinstance(rows_value, Iterable) and not isinstance(rows_value, (str, bytes, bytearray)):
        source = list(rows_value)
    else:
        return []
    normalized: list[dict[str, Any]] = []
    for row in source:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
    return normalized


def _qty_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return str(value)
    if isinstance(value, (int,)):
        return str(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = int(round(value))
        if abs(rounded - value) < 1e-6:
            return str(rounded)
        return str(value)
    text = str(value).strip()
    return text or None


def _format_chart_lines_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    chart_lines: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        desc = str(row.get("desc") or "").strip()
        if desc:
            cleaned_desc = " ".join(desc.split())
        else:
            cleaned_desc = ""
        qty_text = _qty_to_text(row.get("qty"))
        if cleaned_desc:
            if not cleaned_desc.startswith("(") and qty_text:
                line = f"({qty_text}) {cleaned_desc}"
            else:
                line = cleaned_desc
        else:
            parts: list[str] = []
            if qty_text:
                parts.append(f"({qty_text})")
            hole_val = str(row.get("hole") or "").strip()
            if hole_val:
                parts.append(hole_val)
            ref_val = str(row.get("ref") or "").strip()
            if ref_val:
                parts.append(ref_val)
            side_val = str(row.get("side") or "").strip()
            if side_val:
                parts.append(side_val)
            extra_desc = str(row.get("desc") or "").strip()
            if extra_desc:
                parts.append(extra_desc)
            line = " ".join(part for part in parts if part)
        line = " ".join(line.split())
        if line:
            chart_lines.append(line)
    return chart_lines


def read_geo(doc) -> dict[str, Any]:
    acad_info_raw = read_acad_table(doc) or {}
    acad_info = dict(acad_info_raw) if isinstance(acad_info_raw, Mapping) else {}
    acad_rows = _normalize_table_rows(acad_info.get("rows"))
    if acad_rows:
        acad_info["rows"] = acad_rows

    best_info: dict[str, Any] = dict(acad_info) if acad_rows else {}
    text_info: dict[str, Any] = {}

    if acad_rows:
        text_info_raw = read_text_table(doc) or {}
        if isinstance(text_info_raw, Mapping):
            text_info = dict(text_info_raw)
            text_rows = _normalize_table_rows(text_info.get("rows"))
            if text_rows:
                text_info["rows"] = text_rows
            chosen = choose_better_table(acad_info, text_info)
            if isinstance(chosen, Mapping):
                best_info = dict(chosen)
    else:
        text_info_raw = read_text_table(doc) or {}
        if isinstance(text_info_raw, Mapping):
            text_info = dict(text_info_raw)
            text_rows = _normalize_table_rows(text_info.get("rows"))
            if text_rows:
                text_info["rows"] = text_rows
            best_info = dict(text_info)

    rows = _normalize_table_rows(best_info.get("rows"))
    if not rows and text_info:
        rows = _normalize_table_rows(text_info.get("rows"))
        if rows:
            best_info = dict(text_info)

    if rows:
        best_info["rows"] = rows

    hole_count_val: Any = best_info.get("hole_count")
    if hole_count_val is None:
        hole_count = _sum_qty(rows)
    else:
        try:
            hole_count = int(float(hole_count_val))
        except Exception:
            hole_count = _sum_qty(rows)

    provenance = best_info.get("provenance_holes")
    if not provenance and text_info:
        provenance = text_info.get("provenance_holes")
    if not provenance:
        provenance = "HOLE TABLE" if rows else "HOLE TABLE (TEXT_FALLBACK)"

    families_val = best_info.get("hole_diam_families_in")
    if not isinstance(families_val, Mapping) and text_info:
        families_val = text_info.get("hole_diam_families_in")
    families = dict(families_val) if isinstance(families_val, Mapping) else None

    chart_lines = _format_chart_lines_from_rows(rows)

    result: dict[str, Any] = {
        "rows": rows,
        "hole_count": hole_count,
        "provenance_holes": provenance,
        "chart_lines": chart_lines,
    }
    if families is not None:
        result["hole_diam_families_in"] = families

    return result


def choose_better_table(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> Mapping[str, Any]:
    helper = _resolve_app_callable("_choose_better")
    if callable(helper):
        try:
            chosen = helper(a, b)
        except Exception:
            chosen = None
        if isinstance(chosen, Mapping):
            return chosen
        if isinstance(chosen, list):
            return {"rows": list(chosen)}
    score_a = _score_table(a)
    score_b = _score_table(b)
    candidate = a if score_a >= score_b else b
    if isinstance(candidate, Mapping):
        return candidate
    return {}


def _format_chart_line(row: Mapping[str, Any]) -> str:
    qty_val = row.get("qty") if isinstance(row, Mapping) else None
    try:
        qty = int(float(qty_val or 0))
    except Exception:
        qty = 0
    ref_raw = row.get("ref") if isinstance(row, Mapping) else None
    ref_text = str(ref_raw) if ref_raw not in (None, "") else "-"
    side_raw = row.get("side") if isinstance(row, Mapping) else None
    if isinstance(side_raw, str) and side_raw.strip():
        side_text = side_raw.strip().upper()
    else:
        side_text = "-"
    desc_source = None
    if isinstance(row, Mapping):
        for key in ("desc", "description", "text", "hole"):
            value = row.get(key)
            if value:
                desc_source = value
                break
    desc_text = "-"
    if desc_source is not None:
        desc_text = " ".join(str(desc_source).split()) or "-"
    return f"qty={qty} ref={ref_text} side={side_text} desc={desc_text}"


def _normalize_table_info(info: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        return {}
    normalized: dict[str, Any] = dict(info)
    rows_raw = normalized.get("rows")
    if isinstance(rows_raw, list):
        rows_list = [dict(row) if isinstance(row, Mapping) else row for row in rows_raw]
    elif isinstance(rows_raw, Iterable) and not isinstance(rows_raw, (str, bytes, bytearray)):
        rows_list = [dict(row) if isinstance(row, Mapping) else row for row in rows_raw]
    else:
        rows_list = []
    normalized["rows"] = rows_list

    hole_count_raw = normalized.get("hole_count")
    try:
        hole_count = int(float(hole_count_raw))
    except Exception:
        hole_count = 0
    if hole_count <= 0:
        hole_count = _sum_qty(row for row in rows_list if isinstance(row, Mapping))
    normalized["hole_count"] = hole_count

    families = normalized.get("hole_diam_families_in")
    if isinstance(families, Mapping):
        normalized["hole_diam_families_in"] = dict(families)
    else:
        normalized["hole_diam_families_in"] = {}

    provenance = normalized.get("provenance_holes")
    if hole_count and not provenance:
        normalized["provenance_holes"] = "HOLE TABLE"

    chart_lines_raw = normalized.get("chart_lines")
    chart_lines: list[str] = []
    if isinstance(chart_lines_raw, Iterable) and not isinstance(
        chart_lines_raw, (str, bytes, bytearray)
    ):
        for entry in chart_lines_raw:
            text = str(entry).strip()
            if text:
                chart_lines.append(text)
    if not chart_lines:
        chart_lines = [
            _format_chart_line(row)
            for row in rows_list
            if isinstance(row, Mapping)
        ]
    normalized["chart_lines"] = chart_lines

    return normalized


def read_geo(doc) -> dict[str, Any]:
    acad_info = read_acad_table(doc) or {}
    text_info = read_text_table(doc) or {}
    best_info = choose_better_table(acad_info, text_info)

    candidates: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    for candidate in (best_info, text_info, acad_info):
        if isinstance(candidate, Mapping):
            key = id(candidate)
            if key not in seen_ids:
                seen_ids.add(key)
                candidates.append(candidate)

    for candidate in candidates:
        normalized = _normalize_table_info(candidate)
        if normalized.get("rows"):
            return normalized

    return {}


def promote_table_to_geo(geo: dict[str, Any], table_info: Mapping[str, Any], source_tag: str) -> None:
    helper = _resolve_app_callable("_persist_rows_and_totals")
    if callable(helper):
        try:
            helper(geo, table_info, src=source_tag)
            return
        except Exception:
            pass
    if not isinstance(table_info, Mapping):
        return
    rows = table_info.get("rows") or []
    if not rows:
        return
    ops_summary = geo.setdefault("ops_summary", {})
    ops_summary["rows"] = list(rows)
    ops_summary["source"] = source_tag
    totals = defaultdict(int)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            qty = int(float(row.get("qty") or 0))
        except Exception:
            qty = 0
        desc = str(row.get("desc") or "").upper()
        if qty <= 0:
            continue
        if "TAP" in desc:
            totals["tap"] += qty
            totals["drill"] += qty
            if "BACK" in desc and "FRONT" not in desc:
                totals["tap_back"] += qty
            elif "FRONT" in desc and "BACK" in desc:
                totals["tap_front"] += qty
                totals["tap_back"] += qty
            else:
                totals["tap_front"] += qty
        if any(marker in desc for marker in ("CBORE", "COUNTERBORE", "C'BORE")):
            totals["counterbore"] += qty
            if "BACK" in desc and "FRONT" not in desc:
                totals["counterbore_back"] += qty
            elif "FRONT" in desc and "BACK" in desc:
                totals["counterbore_front"] += qty
                totals["counterbore_back"] += qty
            else:
                totals["counterbore_front"] += qty
        if "JIG GRIND" in desc:
            totals["jig_grind"] += qty
        if (
            "SPOT" in desc
            or "CENTER DRILL" in desc
            or "C DRILL" in desc
            or "C’DRILL" in desc
        ) and "TAP" not in desc and "THRU" not in desc:
            totals["spot"] += qty
    if totals:
        ops_summary["totals"] = dict(totals)
    preferred_hole_count = _sum_qty(rows)
    hole_count = table_info.get("hole_count")
    if preferred_hole_count > 0:
        hole_count = preferred_hole_count
    try:
        geo["hole_count"] = int(hole_count)
    except Exception:
        pass
    provenance = geo.setdefault("provenance", {})
    provenance["holes"] = "HOLE TABLE"


def extract_geometry(doc) -> dict[str, Any]:
    helper = _resolve_app_callable("_build_geo_from_ezdxf_doc")
    if callable(helper):
        try:
            geo = helper(doc)
        except Exception:
            geo = None
        if isinstance(geo, Mapping):
            return dict(geo)
    return {}


def _load_doc_for_path(path: Path, *, use_oda: bool, out_ver: str | None = None) -> Any:
    ezdxf_mod = geometry.require_ezdxf()
    readfile = getattr(ezdxf_mod, "readfile", None)
    if not callable(readfile):
        raise AttributeError("ezdxf module does not expose a callable readfile")
    lower_suffix = path.suffix.lower()
    if lower_suffix == ".dwg":
        if use_oda and _HAS_ODAFC:
            odafc_mod = None
            try:
                odafc_mod = _ezdxf_vendor.require_odafc()
            except Exception:
                odafc_mod = None
            if odafc_mod is not None:
                odaread = getattr(odafc_mod, "readfile", None)
                if callable(odaread):
                    return odaread(str(path))
        if out_ver:
            dxf_path = convert_dwg_to_dxf(str(path), out_ver=out_ver)
        else:
            dxf_path = convert_dwg_to_dxf(str(path))
        return readfile(dxf_path)
    return readfile(str(path))


def _ensure_ops_summary_map(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {}


def _best_geo_hole_count(geo: Mapping[str, Any]) -> int | None:
    for key in ("hole_count", "hole_count_geom", "hole_count_geom_dedup", "hole_count_geom_raw"):
        value = geo.get(key) if isinstance(geo, Mapping) else None
        try:
            val_int = int(float(value))
        except Exception:
            val_int = 0
        if val_int > 0:
            return val_int
    return None


def read_geo(
    doc: Any,
    *,
    prefer_table: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Process a loaded DXF/DWG document into GEO payload details."""

    del feature_flags  # placeholder for future feature toggles
    geo = extract_geometry(doc)
    if not isinstance(geo, dict):
        geo = {}

    use_tables = bool(prefer_table)

    existing_ops_summary = geo.get("ops_summary") if isinstance(geo, Mapping) else {}
    provenance = geo.get("provenance") if isinstance(geo, Mapping) else {}
    provenance_holes = None
    if isinstance(provenance, Mapping):
        provenance_holes = provenance.get("holes")
    existing_source = ""
    if isinstance(existing_ops_summary, Mapping):
        existing_source = str(existing_ops_summary.get("source") or "")
    existing_is_table = bool(
        use_tables
        and (
            (existing_source and "table" in existing_source.lower())
            or (isinstance(provenance_holes, str) and provenance_holes.upper() == "HOLE TABLE")
        )
    )
    if existing_is_table and isinstance(existing_ops_summary, Mapping):
        current_table_info = dict(existing_ops_summary)
        rows = current_table_info.get("rows")
        if isinstance(rows, Iterable) and not isinstance(rows, list):
            current_table_info["rows"] = list(rows)
    else:
        current_table_info = {}

    try:
        acad_info = read_acad_table(doc, layer_allowlist=layer_allowlist) or {}
    except TypeError as exc:
        if "layer_allowlist" in str(exc):
            try:
                acad_info = read_acad_table(doc) or {}
            except Exception:
                acad_info = {}
        else:
            raise
    except Exception:
        acad_info = {}
    acad_roi_hint: Mapping[str, Any] | None = None
    if isinstance(acad_info, Mapping):
        roi_candidate = acad_info.get("roi_hint")
        acad_roi_hint = roi_candidate if isinstance(roi_candidate, Mapping) else None
    try:
        text_info = read_text_table(
            doc,
            layer_allowlist=layer_allowlist,
            roi_hint=acad_roi_hint,
            block_name_allowlist=block_name_allowlist,
            block_name_regex=block_name_regex,
        ) or {}
    except TypeError as exc:
        if "layer_allowlist" in str(exc) or "roi_hint" in str(exc):
            try:
                text_info = read_text_table(doc) or {}
            except Exception:
                text_info = {}
        else:
            raise
    except Exception:
        text_info = {}

    acad_rows = len((acad_info.get("rows") or [])) if isinstance(acad_info, Mapping) else 0
    text_rows = len((text_info.get("rows") or [])) if isinstance(text_info, Mapping) else 0
    print(f"[EXTRACT] acad_rows={acad_rows} text_rows={text_rows}")

    best_table = choose_better_table(acad_info, text_info)
    score_a = _score_table(acad_info)
    score_b = _score_table(text_info)
    table_used = False
    source_tag = None
    existing_score = _score_table(current_table_info)
    best_score = _score_table(best_table)
    if (
        use_tables
        and isinstance(best_table, Mapping)
        and best_table.get("rows")
        and best_score > existing_score
    ):
        source_tag = "acad_table" if score_a >= score_b else "text_table"
        promote_table_to_geo(geo, best_table, source_tag)
        table_used = True

    ops_summary = _ensure_ops_summary_map(geo.get("ops_summary"))
    geo["ops_summary"] = ops_summary
    rows = ops_summary.get("rows")
    if not isinstance(rows, list):
        if isinstance(rows, Iterable):
            rows = list(rows)
        else:
            rows = []
    if not table_used and existing_is_table:
        table_used = bool(rows)
    if table_used:
        qty_sum = _sum_qty(rows)
    else:
        if rows:
            ops_summary.pop("rows", None)
            rows = []
        qty_sum = 0
        ops_summary["source"] = "geom"
    if not table_used:
        hole_count = _best_geo_hole_count(geo)
        if hole_count:
            geo["hole_count"] = hole_count

    if table_used and source_tag:
        ops_summary["source"] = source_tag
    totals = ops_summary.get("totals")
    if isinstance(totals, Mapping):
        ops_summary["totals"] = dict(totals)

    rows_for_log = rows
    if table_used:
        rows_for_log = ops_summary.get("rows") or []
        if not isinstance(rows_for_log, list) and isinstance(rows_for_log, Iterable):
            rows_for_log = list(rows_for_log)
        qty_sum = _sum_qty(rows_for_log)
    print(
        f"[EXTRACT] published rows={len(rows_for_log)} qty_sum={qty_sum} "
        f"source={ops_summary.get('source')}"
    )
    provenance_holes = None
    provenance = geo.get("provenance")
    if isinstance(provenance, Mapping):
        provenance_holes = provenance.get("holes")
    print(f"[EXTRACT] provenance={provenance_holes}")

    debug_payload = get_last_text_table_debug() or {}
    hole_count_val = None
    try:
        hole_count_val = geo.get("hole_count") if isinstance(geo, Mapping) else None
    except Exception:
        hole_count_val = None
    if hole_count_val in (None, ""):
        hole_count_val = _best_geo_hole_count(geo) if isinstance(geo, Mapping) else None

    payload_rows: list[Mapping[str, Any]] = []
    if isinstance(rows_for_log, list):
        payload_rows = rows_for_log
    elif isinstance(rows_for_log, Iterable):
        payload_rows = list(rows_for_log)

    families_map: dict[str, int] | None = None
    for candidate in (best_table, text_info, acad_info, current_table_info):
        if not isinstance(candidate, Mapping):
            continue
        families_val = candidate.get("hole_diam_families_in")
        if isinstance(families_val, Mapping) and families_val:
            normalized_families: dict[str, int] = {}
            for key, value in families_val.items():
                try:
                    normalized_families[str(key)] = int(value)
                except Exception:
                    continue
            if normalized_families:
                families_map = normalized_families
                break

    chart_lines = [
        _format_chart_line(row) for row in payload_rows if isinstance(row, Mapping)
    ]

    result_payload = {
        "geo": geo,
        "ops_summary": ops_summary,
        "rows": payload_rows,
        "qty_sum": qty_sum,
        "hole_count": hole_count_val,
        "provenance_holes": provenance_holes,
        "table_used": table_used,
        "source": ops_summary.get("source") if isinstance(ops_summary, Mapping) else None,
        "debug_payload": debug_payload,
        "chart_lines": chart_lines,
    }

    if families_map is not None:
        result_payload["hole_diam_families_in"] = families_map

    return result_payload


def _read_geo_payload_from_path(
    path_obj: Path,
    *,
    prefer_table: bool = True,
    use_oda: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    try:
        doc = _load_doc_for_path(path_obj, use_oda=use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[EXTRACT] failed to load document: {exc}")
        return {"error": str(exc)}

    payload = read_geo(
        doc,
        prefer_table=prefer_table,
        feature_flags=feature_flags,
        layer_allowlist=layer_allowlist,
        block_name_allowlist=block_name_allowlist,
        block_name_regex=block_name_regex,
    )

    scan_info = get_last_acad_table_scan() or {}
    tables_found = 0
    try:
        tables_found = int(scan_info.get("tables_found", 0))  # type: ignore[arg-type]
    except Exception:
        tables_found = 0
    if tables_found == 0 and path_obj.suffix.lower() == ".dwg":
        fallback_versions = [
            "ACAD2000",
            "ACAD2004",
            "ACAD2007",
            "ACAD2013",
            "ACAD2018",
        ]
        for version in fallback_versions:
            print(f"[ACAD-TABLE] trying DXF fallback version={version}")
            try:
                fallback_doc = _load_doc_for_path(
                    path_obj, use_oda=use_oda, out_ver=version
                )
            except Exception as exc:
                print(f"[ACAD-TABLE] DXF fallback {version} failed: {exc}")
                continue
            payload = read_geo(
                fallback_doc,
                prefer_table=prefer_table,
                feature_flags=feature_flags,
                layer_allowlist=layer_allowlist,
                block_name_allowlist=block_name_allowlist,
                block_name_regex=block_name_regex,
            )
            scan_info = get_last_acad_table_scan() or {}
            try:
                tables_found = int(scan_info.get("tables_found", 0))
            except Exception:
                tables_found = 0
            if tables_found:
                break
    return payload


def extract_geo_from_path(
    path: str,
    *,
    prefer_table: bool = True,
    use_oda: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Load DWG/DXF at ``path`` and return a GEO dictionary."""

    path_obj = Path(path)
    payload = _read_geo_payload_from_path(
        path_obj,
        prefer_table=prefer_table,
        use_oda=use_oda,
        feature_flags=feature_flags,
        layer_allowlist=layer_allowlist,
        block_name_allowlist=block_name_allowlist,
        block_name_regex=block_name_regex,
    )
    if "error" in payload:
        return {"error": payload["error"]}
    geo = payload.get("geo")
    if isinstance(geo, dict):
        return geo
    return {}


def extract_geo_from_path(
    path: str,
    *,
    prefer_table: bool = True,
    use_oda: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Load DWG/DXF at ``path`` and return a GEO dictionary."""

    path_obj = Path(path)
    return _read_geo_payload_from_path(
        path_obj,
        prefer_table=prefer_table,
        use_oda=use_oda,
        feature_flags=feature_flags,
        layer_allowlist=layer_allowlist,
        block_name_allowlist=block_name_allowlist,
        block_name_regex=block_name_regex,
    )


def get_last_text_table_debug() -> dict[str, Any] | None:
    if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
        return _LAST_TEXT_TABLE_DEBUG
    return None


def get_last_acad_table_scan() -> dict[str, Any] | None:
    if isinstance(_LAST_ACAD_TABLE_SCAN, Mapping):
        scan: dict[str, Any] = dict(_LAST_ACAD_TABLE_SCAN)
        tables = scan.get("tables")
        if isinstance(tables, list):
            normalized: list[dict[str, Any]] = []
            for entry in tables:
                if isinstance(entry, Mapping):
                    normalized.append(dict(entry))
                else:
                    normalized.append({"value": entry})
            scan["tables"] = normalized
        return scan
    return None


__all__ = [
    "read_geo",
    "extract_geo_from_path",
    "read_acad_table",
    "read_geo",
    "read_text_table",
    "read_geo",
    "choose_better_table",
    "promote_table_to_geo",
    "extract_geometry",
    "read_geo",
    "get_last_text_table_debug",
    "get_last_acad_table_scan",
]
