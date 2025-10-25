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


def read_acad_table(doc) -> dict[str, Any]:
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
    return {}


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
    entries: list[dict[str, Any]]
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
            {"bands": [], "band_cells": [], "rows_txt_fallback": [], "qty_col": None, "ref_col": None, "desc_col": None},
        )

    records_all = list(records)
    all_height_values = [
        float(rec["height"])
        for rec in records_all
        if isinstance(rec.get("height"), (int, float)) and float(rec["height"]) > 0
    ]
    median_height_all = (
        statistics.median(all_height_values) if all_height_values else 0.0
    )
    roi_median_height = median_height_all
    roi_info: dict[str, Any] | None = None
    anchor_lines = [rec for rec in records_all if _ROI_ANCHOR_RE.search(rec["text"])]
    filtered_records = records_all
    if anchor_lines:
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
        print(
            "[ROI] anchors={count} clusters={clusters} chosen_span=[{ymax:.1f}..{ymin:.1f}] "
            "bbox=[{xmin:.1f}..{xmax:.1f}] expanded=[{xmin_exp:.1f}..{xmax_exp:.1f},{ymin_exp:.1f}..{ymax_exp:.1f}]".format(
                count=anchor_count,
                clusters=clusters_count,
                ymax=cluster_ymax,
                ymin=cluster_ymin,
                xmin=cluster_xmin,
                xmax=cluster_xmax,
                xmin_exp=expanded_xmin,
                xmax_exp=expanded_xmax,
                ymin_exp=expanded_ymin,
                ymax_exp=expanded_ymax,
            )
        )
        print(
            f"[ROI] median_h={roi_median_height:.2f} expand=({dx:.1f},{dy:.1f})"
        )
        print(
            f"[ROI] raw_lines -> roi_lines: {len(records_all)} -> {len(filtered_records)}"
        )
        roi_info = {
            "anchors": anchor_count,
            "clusters": clusters_count,
            "bbox": [cluster_xmin, cluster_xmax, cluster_ymin, cluster_ymax],
            "expanded": [expanded_xmin, expanded_xmax, expanded_ymin, expanded_ymax],
            "total": len(records_all),
            "kept": len(filtered_records),
            "median_h": roi_median_height,
        }
    records = list(filtered_records)

    records.sort(key=lambda item: (-item["y"], item["x"]))

    height_values = [
        float(rec["height"])
        for rec in records
        if isinstance(rec.get("height"), (int, float)) and float(rec["height"]) > 0
    ]
    y_diffs = [
        abs(records[idx]["y"] - records[idx - 1]["y"])
        for idx in range(1, len(records))
        if abs(records[idx]["y"] - records[idx - 1]["y"]) > 0
    ]

    median_h = statistics.median(height_values) if height_values else 0.0
    if (median_h is None or median_h <= 0) and roi_median_height > 0:
        median_h = roi_median_height
    if (median_h is None or median_h <= 0) and median_height_all > 0:
        median_h = median_height_all
    if (median_h is None or median_h <= 0) and y_diffs:
        median_h = statistics.median(y_diffs)
    if median_h is None or median_h <= 0:
        median_h = 4.0

    y_gap_limit = 0.75 * median_h if median_h > 0 else 3.0
    if y_gap_limit <= 0:
        y_gap_limit = 3.0

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
        if (
            abs(y_val - prev_y) <= y_gap_limit
            and abs(y_val - band_center) <= y_gap_limit
        ):
            current_band.append(record)
            current_sum_y += y_val
        else:
            raw_bands.append(current_band)
            current_band = [record]
            current_sum_y = y_val
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
            {"bands": [], "band_cells": [], "rows_txt_fallback": [], "qty_col": None, "ref_col": None, "desc_col": None, "y_eps": y_eps},
        )

    for idx, band in enumerate(bands[:10]):
        mean_y = sum(item["y"] for item in band) / len(band)
        print(f"[TABLE-Y] band#{idx} y≈{mean_y:.2f} lines={len(band)}")

    band_summaries: list[dict[str, Any]] = []
    band_cells_dump: list[dict[str, Any]] = []
    band_results: list[dict[str, Any]] = []

    for band_index, band in enumerate(bands):
        mean_y = sum(item["y"] for item in band) / len(band)
        band_summaries.append(
            {"index": band_index, "y_mean": mean_y, "line_count": len(band)}
        )
        sorted_band = sorted(band, key=lambda item: item["x"])
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

        columns: list[dict[str, Any]] = []
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
                continue
            columns.append(
                {
                    "items": [item],
                    "sum_x": x_pos,
                    "sum_y": item["y"],
                    "count": 1,
                    "center_x": x_pos,
                    "center_y": item["y"],
                }
            )

        cell_texts: list[str] = []
        preview_parts: list[str] = []
        for col_index, column in enumerate(columns):
            sorted_items = sorted(column["items"], key=lambda itm: itm["x"])
            cell_text = " ".join(item["text"] for item in sorted_items).strip()
            cell_texts.append(cell_text)
            band_cells_dump.append(
                {
                    "band": band_index,
                    "col": col_index,
                    "x_center": column["center_x"],
                    "y_center": column["center_y"],
                    "text": cell_text,
                }
            )
            preview_parts.append(
                f'C{col_index}="{_truncate_cell_preview(cell_text)}"'
            )

        if band_index < 10:
            preview_body = " | ".join(preview_parts)
            print(f"[TABLE-X] band#{band_index} cols={len(columns)} | {preview_body}")

        band_results.append(
            {
                "cells": cell_texts,
                "y_mean": mean_y,
                "line_count": len(band),
                "centers": [column["center_x"] for column in columns],
                "x_eps": x_eps,
            }
        )

    if not band_results:
        return (
            None,
            {"bands": band_summaries, "band_cells": band_cells_dump, "rows_txt_fallback": [], "qty_col": None, "ref_col": None, "desc_col": None, "y_eps": y_eps},
        )

    column_count = max(len(band["cells"]) for band in band_results)
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

    fallback_qty_col: int | None = None
    qty_candidates = [
        idx
        for idx, info in enumerate(metrics)
        if info["header_qty"] or (info["non_empty"] > 0 and info["numeric_ratio"] >= 0.6)
    ]
    if qty_candidates:
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
        if qty_x is not None and centers:
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

    ref_candidates = [
        idx
        for idx, info in enumerate(metrics)
        if info["header_ref"] or (info["non_empty"] > 0 and info["ref_ratio"] >= 0.4)
    ]
    ref_col = max(ref_candidates) if ref_candidates else None

    desc_candidates = [idx for idx in range(column_count) if idx != qty_col and idx != ref_col]
    if not desc_candidates:
        desc_candidates = list(range(column_count))

    def _desc_score(index: int) -> tuple[float, float, int]:
        data = metrics[index]
        return (float(data["action_ratio"]), float(data["avg_len"]), -index)

    desc_col = max(desc_candidates, key=_desc_score)

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
        qty_idx = band.get("qty_index")
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
        need_row_parse = (qty_val is None or qty_val <= 0 or qty_groups_count == 0)
        if need_row_parse and combined_row_text:
            qty_candidate, remainder = _extract_band_quantity(combined_row_text)
            if (qty_candidate is None or qty_candidate <= 0) and (
                qty_val is None or qty_val <= 0
            ):
                qty_candidate, remainder = _extract_row_quantity_and_remainder(
                    combined_row_text
                )
            if qty_candidate is not None and qty_candidate > 0:
                qty_val = qty_candidate
                fallback_desc = remainder.strip()
                used_qty_fallback = True
                print(
                    f"[QTY-FALLBACK] band#{band_index} used desc-parse qty={qty_val}"
                )
        if qty_val is None or qty_val <= 0:
            continue
        desc_text = _cell_at(cells, desc_col)
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
        ref_text_candidate = _cell_at(cells, ref_col)
        ref_cell_ref = ("", None)
        if ref_text_candidate:
            ref_cell_ref = _extract_row_reference(ref_text_candidate)
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
            side = _detect_row_side(fragment_desc)
            row_dict: dict[str, Any] = {
                "hole": "",
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


def read_text_table(doc) -> dict[str, Any]:
    helper = _resolve_app_callable("extract_hole_table_from_text")
    _print_helper_debug("text", helper)
    global _LAST_TEXT_TABLE_DEBUG
    _LAST_TEXT_TABLE_DEBUG = {
        "candidates": [],
        "band_cells": [],
        "bands": [],
        "rows": [],
        "raw_lines": [],
    }
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
        nonlocal columnar_table_info, columnar_debug_info
        if table_lines is not None:
            return table_lines

        collected_entries: list[dict[str, Any]] = []
        merged_rows = []
        parsed_rows = []
        text_rows_info = None
        rows_txt_initial = 0
        hint_logged = False
        attrib_count = 0

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

            def _process_entity(entity: Any, *, depth: int, from_block: bool) -> None:
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
                        }
                        counter += 1
                        collected_entries.append(entry)
                        kept_count += 1
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
                    if virtual_entities:
                        for child in virtual_entities:
                            _process_entity(child, depth=depth + 1, from_block=True)
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
                                _process_entity(child, depth=depth + 1, from_block=True)
                    for attribute in _iter_insert_attributes(entity):
                        _process_entity(attribute, depth=depth + 1, from_block=True)
                    if name_str:
                        visited_blocks.discard(name_str)

            for entity in base_entities:
                marker = id(entity)
                if marker in seen_entities:
                    continue
                seen_entities.add(marker)
                _process_entity(entity, depth=0, from_block=False)

            print(
                f"[TEXT-SCAN] layout={layout_name} text={text_fragments} "
                f"mtext={mtext_fragments} kept={kept_count} from_blocks={from_blocks_count}"
            )

        print(
            f"[TEXT-SCAN] attrib_lines={attrib_count} depth_max={_MAX_INSERT_DEPTH}"
        )

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
                table_candidate, debug_payload = _build_columnar_table_from_entries(raw_lines)
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
    hole_count = table_info.get("hole_count")
    if hole_count is None:
        hole_count = _sum_qty(rows)
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


def _load_doc_for_path(path: Path, *, use_oda: bool) -> Any:
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
        acad_info = read_acad_table(doc) or {}
    except Exception:
        acad_info = {}
    try:
        text_info = read_text_table(doc) or {}
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


def extract_geo_from_path(
    path: str,
    *,
    prefer_table: bool = True,
    use_oda: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load DWG/DXF at ``path`` and return a GEO dictionary."""

    path_obj = Path(path)
    try:
        doc = _load_doc_for_path(path_obj, use_oda=use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[EXTRACT] failed to load document: {exc}")
        return {"error": str(exc)}

    payload = read_geo(doc, prefer_table=prefer_table, feature_flags=feature_flags)
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
) -> dict[str, Any]:
    """Load DWG/DXF at ``path`` and return a GEO dictionary."""

    path_obj = Path(path)
    try:
        doc = _load_doc_for_path(path_obj, use_oda=use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[EXTRACT] failed to load document: {exc}")
        return {"error": str(exc)}

    return read_geo(
        doc,
        prefer_table=prefer_table,
        feature_flags=feature_flags,
    )


def get_last_text_table_debug() -> dict[str, Any] | None:
    if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
        return _LAST_TEXT_TABLE_DEBUG
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
]
