"""Hole table parsing and aggregation helpers."""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from collections.abc import Mapping as _MappingABC, MutableMapping as _MutableMappingABC
from fractions import Fraction
from statistics import median
from typing import Any, Callable, Iterator, MutableMapping as TypingMutableMapping, Sequence, NamedTuple, cast

from cad_quoter.utils.number_parse import (
    NUM_DEC_RE,
    VALUE_PATTERN,
    _to_inch,
    first_inch_value,
)
from cad_quoter.utils.numeric import parse_mixed_fraction

try:  # pragma: no cover - optional dependency during limited installs
    from cad_quoter.geometry.dxf_enrich import iter_spaces as _iter_spaces
except Exception:  # pragma: no cover - defensive fallback
    _iter_spaces = None

from cad_quoter.domain_models import (
    coerce_float_or_none as _coerce_float_or_none,
)


from .chart_lines import (
    _build_ops_rows_from_lines_fallback as _chart_build_ops_rows_from_lines_fallback,
    collect_chart_lines_context as _collect_chart_lines_context,
)


RE_TAP = re.compile(
    r"(\(\d+\)\s*)?("
    r"#\s*\d{1,2}-\d+"  # #10-32
    r"|(?:\d+/\d+)\s*-\s*\d+"  # 5/8-11
    r"|(?:\d+(?:\.\d+)?)\s*-\s*\d+"  # 0.190-32
    r"|M\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?"  # M8x1.25
    r")\s*TAP",
    re.I,
)
RE_NPT = re.compile(r"(\d+/\d+)\s*-\s*N\.?P\.?T\.?", re.I)
RE_THRU = re.compile(r"\bTHRU\b", re.I)
RE_CBORE = re.compile(r"C[’']?BORE|CBORE|COUNTERBORE", re.I)
RE_CSK = re.compile(r"CSK|C'SINK|COUNTERSINK", re.I)
_RE_DEPTH_OR_THICK = re.compile(
    rf"({VALUE_PATTERN})\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?",
    re.I,
)
RE_DEPTH = _RE_DEPTH_OR_THICK
RE_DIA = re.compile(
    rf"(?:%%[Cc]\s*|[Ø⌀\u00D8]\s*)?({VALUE_PATTERN})",
    re.I,
)
RE_FRONT_BACK = re.compile(
    r"FRONT\s*&\s*BACK|FRONT\s+AND\s+BACK|BOTH\s+SIDES|TWO\s+SIDES|2\s+SIDES|OPPOSITE\s+SIDE",
    re.I,
)

WIRE_GAGE_MAJOR_DIA_IN = {
    0: 0.06,
    1: 0.073,
    2: 0.086,
    3: 0.099,
    4: 0.112,
    5: 0.125,
    6: 0.138,
    7: 0.151,
    8: 0.164,
    9: 0.177,
    10: 0.19,
    11: 0.203,
    12: 0.216,
}

TAP_MINUTES_BY_CLASS = {
    "small": 0.22,
    "medium": 0.3,
    "large": 0.38,
    "xl": 0.45,
    "pipe": 0.55,
}

COUNTERDRILL_MIN_PER_SIDE_MIN = 0.12
CBORE_MIN_PER_SIDE_MIN = 0.15
CSK_MIN_PER_SIDE_MIN = 0.12

_SPOT_TOKENS = re.compile(r"(?:C['’]?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|SPOT)", re.I)
_THREAD_WITH_TPI_RE = re.compile(
    r"((?:#\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*-\s*(\d+)",
    re.I,
)
_THREAD_WITH_NPT_RE = re.compile(
    r"((?:#\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?))\s*-\s*(N\.?P\.?T\.?)",
    re.I,
)
_CBORE_RE = re.compile(
    rf"(?:^|[ ;])(?:%%[Cc]|Ø|⌀|DIA)?\s*({VALUE_PATTERN})\s*(?:C['’]?\s*BORE|CBORE|COUNTER\s*BORE)",
    re.I,
)
_SIDE_BOTH = re.compile(r"\b(FRONT\s*&\s*BACK|BOTH\s+SIDES)\b", re.I)
_SIDE_BACK = re.compile(r"\b(?:FROM\s+)?BACK\b", re.I)
_SIDE_FRONT = re.compile(r"\b(?:FROM\s+)?FRONT\b", re.I)

_OP_WORDS = {
    "cbore": r"(?:C['’]?\s*BORE|CBORE|COUNTER\s*BORE)",
    "csk": r"(?:C['’]?\s*SINK|CSK|COUNTER\s*SINK)",
    "cdrill": r"(?:C['’]?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|SPOT)",
    "tap": r"\bTAP\b",
    "thru": r"\bTHRU\b",
    "jig": r"\bJIG\s*GRIND\b",
}
_DEPTH_TOKEN = re.compile(r"[×xX]\s*([0-9.]+)\b")
_DIA_TOKEN = re.compile(
    rf"(?:%%[Cc]|Ø|⌀|REF|DIA)[^0-9]*({VALUE_PATTERN})",
    re.I,
)
_MIXED_FRACTION_ONLY = re.compile(r"^[+-]?\s*(?:\d+\s+)?\d+/\d+\s*$")

_MM_DIM_TOKEN = re.compile(
    r"(?:%%[Cc]|[Ø⌀\u00D8]|DIA|DIAM|REF)?\s*"
    r"((?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?))\s*"
    r"(?:MM|MILLIM(?:E|E)T(?:E|)RS?)",
    re.I,
)


def parse_dim(value: Any) -> float | None:
    """Parse a dimension string to inches, handling fractions and millimetres."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            val = float(value)
        except Exception:
            return None
        return val if math.isfinite(val) and val > 0 else None

    text = str(value).strip()
    if not text:
        return None

    mm_match = _MM_DIM_TOKEN.search(text)
    if mm_match:
        token = mm_match.group(1).replace(" ", "")
        try:
            mm_val = float(Fraction(token)) if "/" in token else float(token)
        except Exception:
            mm_val = None
        if mm_val is not None:
            inch_val = mm_val / 25.4
            if math.isfinite(inch_val) and inch_val > 0:
                return inch_val

    cleaned = (
        text.replace("%%C", "")
        .replace("%%c", "")
        .replace("\u00D8", "")
        .replace("Ø", "")
        .replace("⌀", "")
        .replace("IN", "")
        .replace("in", "")
        .strip("\"' ")
    )

    cleaned_for_mix = cleaned.replace("-", " ") if cleaned else ""
    if cleaned_for_mix and _MIXED_FRACTION_ONLY.match(cleaned_for_mix):
        mixed = parse_mixed_fraction(cleaned)
        if mixed and mixed > 0:
            return mixed

    for candidate in (cleaned, text):
        num = _to_inch(candidate)
        if num and num > 0:
            return num
        fallback = first_inch_value(candidate)
        if fallback and fallback > 0:
            return fallback

    return None


_SUMMARY_OP_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Tap",
        re.compile(
            r"""
            \b(
                TAP|
                NPT|
                \d+(?:/\d+)?\s*-\s*\d+(?:\.\d+)?(?:\s*(?:UNC|UNF|UNEF|UNS|UNJ|UNJC|UN|BSP|BSPT|BSPP|BSF|BSW|SAE|NPS|NPSM|NPSF|NPSL|NPTF))?|
                \#\s*\d+\s*-\s*\d+(?:\.\d+)?|
                M\d+(?:\.\d+)?\s*X\s*\d+(?:\.\d+)?
            )\b
            """,
            re.VERBOSE,
        ),
    ),
    (
        "C'bore",
        re.compile(
            r"\b(?:C'? ?BORE|CBORE|COUNTER ?BORE|SPOT ?FACE|SPOTFACE)\b",
        ),
    ),
    (
        "C'drill",
        re.compile(
            r"\b(?:C'? ?DRILL|C'? ?DRL|C'? ?SINK|C'? ?SK|COUNTER[- ]?DRILL|COUNTER ?SINK|CTR ?DRILL|CENTER ?DRILL|SPOT ?DRILL|SPOT|CSK)\b",
        ),
    ),
    (
        "Jig Grind",
        re.compile(r"\bJIG ?(?:GRIND|GRND)\b"),
    ),
    (
        "Drill",
        re.compile(
            r"""
            (
                (?<!COUNTER )(?<!CENTER )(?<!CTR )(?<!SPOT )\bDRILL\b|
                \bTHRU\b|
                \bLETTER\s+[A-Z]\s+DRILL\b|
                \b'?[A-Z]'?\s+DRILL\b
            )
            """,
            re.VERBOSE,
        ),
    ),
)

_SUMMARY_OP_TRANSLATE = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": "'",
        "”": "'",
        '"': "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "Ø": "O",
        "ø": "O",
        "⌀": "O",
        "×": "X",
        "✕": "X",
        "✖": "X",
        "⨯": "X",
    }
)


def _norm_txt(s: str) -> str:
    s = (s or "").replace("\u00D8", "Ø").replace("’", "'").upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_ops_desc(desc: str) -> str:
    text = (desc or "").translate(_SUMMARY_OP_TRANSLATE)
    text = text.upper()
    text = text.replace("Ø", "O").replace("⌀", "O")
    text = re.sub(r"\bN\.\s*P\.\s*T\.?\b", "NPT", text)
    text = re.sub(r"C\s*'\s*", "C'", text)
    text = re.sub(r"'([A-Z])'", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_summary_operation(desc: str) -> tuple[str, str]:
    normalized = _normalize_ops_desc(desc)
    for label, pattern in _SUMMARY_OP_RULES:
        if pattern.search(normalized):
            return label, normalized
    return "Unknown", normalized


def _ops_qty_from_value(value: Any) -> int:
    """Best-effort coercion of a HOLE TABLE quantity value to an int."""

    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            # Treat floats as intentional numeric quantities (e.g., 4.0 -> 4).
            return int(round(float(value)))
        except Exception:
            return 0
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(round(float(text)))
    except Exception:
        match = re.search(r"\d+", text)
        return int(match.group()) if match else 0


class _TextFragment(NamedTuple):
    x: float
    y: float
    height: float
    text: str


_QTY_ONLY_RE = re.compile(r"^\(?\s*(\d+)\s*\)?$")
_INLINE_QTY_RE = re.compile(r"^\(\s*(\d+)\s*\)\s*(.*)$")
_SIDE_TOKENS = {"FRONT", "BACK", "BOTH", "SIDE", "OPP", "OPPOSITE"}
_HEADER_QTY_TOKENS = {"QTY", "QUANTITY"}
_HEADER_REF_TOKENS = {"REF", "REFERENCE", "Ø", "DIA", "DIAM"}
_HEADER_DESC_TOKENS = {"DESC", "DESCRIPTION"}
_HEADER_SIDE_TOKENS = {"SIDE", "SIDES", "FACE", "FACES"}
_HEADER_HOLE_TOKENS = {"HOLE", "ID", "NO."}


def _normalize_fragment_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = (
        text.replace("\u00D8", "Ø")
        .replace("ø", "Ø")
        .replace("%%C", "Ø")
        .replace("%%c", "Ø")
    )
    return re.sub(r"\s+", " ", text).strip()


def _entity_world_point(entity: Any) -> tuple[float, float, float]:
    dxf = getattr(entity, "dxf", None)
    point = None
    for attr in ("insert", "alignment_point", "align_point", "start", "position"):
        candidate = getattr(dxf, attr, None) if dxf is not None else None
        if candidate is not None:
            point = candidate
            break
    if point is None:
        return (0.0, 0.0, 0.0)

    def _coord(value: Any, index: int) -> float:
        try:
            return float(value[index])
        except Exception:
            try:
                return float(getattr(value, "xyz"[index]))
            except Exception:
                return 0.0

    if hasattr(point, "xyz"):
        try:
            x_val, y_val, z_val = point.xyz
        except Exception:
            x_val = _coord(point, 0)
            y_val = _coord(point, 1)
            z_val = _coord(point, 2)
    else:
        x_val = _coord(point, 0)
        y_val = _coord(point, 1)
        z_val = _coord(point, 2)

    try:
        ocs = entity.ocs()
    except Exception:
        ocs = None
    if ocs is not None:
        try:
            x_val, y_val, z_val = ocs.to_wcs((x_val, y_val, z_val))
        except Exception:
            pass

    return float(x_val), float(y_val), float(z_val)


def _entity_text_height(entity: Any) -> float:
    dxf = getattr(entity, "dxf", None)
    if dxf is None:
        return 0.0
    for attr in ("height", "char_height"):
        candidate = getattr(dxf, attr, None)
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except Exception:
            continue
        if value > 0:
            return value
    return 0.0


def _iter_text_fragments_from_entity(entity: Any) -> Iterator[_TextFragment]:
    try:
        kind = entity.dxftype()
    except Exception:
        kind = ""

    if kind in {"TEXT", "MTEXT"}:
        raw_text: str | None = None
        plain = getattr(entity, "plain_text", None)
        if callable(plain):
            try:
                raw_text = plain()
            except Exception:
                raw_text = None
        if not raw_text:
            raw_text = getattr(getattr(entity, "dxf", None), "text", None)
        if not raw_text:
            return
        parts = re.split(r"\\P|\r?\n", str(raw_text)) if kind == "MTEXT" else [str(raw_text)]
        x_val, y_val, _ = _entity_world_point(entity)
        height = _entity_text_height(entity)
        for part in parts:
            normalized = _normalize_fragment_text(part)
            if not normalized:
                continue
            yield _TextFragment(x_val, y_val, height, normalized)
    elif kind == "INSERT":
        try:
            virtuals = entity.virtual_entities()
        except Exception:
            virtuals = []
        for child in virtuals or []:
            yield from _iter_text_fragments_from_entity(child)


def collect_text_table_fragments(doc: Any) -> list[tuple[float, float, float, str]]:
    """Collect raw text fragments for HOLE TABLE detection using world coordinates."""

    fragments: list[tuple[float, float, float, str]] = []
    if doc is None:
        return fragments

    spaces: list[Any] = []
    if callable(_iter_spaces):  # pragma: no branch - evaluated once
        try:
            spaces = list(_iter_spaces(doc))
        except Exception:
            spaces = []
    if not spaces:
        try:
            modelspace = doc.modelspace()
        except Exception:
            modelspace = None
        if modelspace is not None:
            spaces.append(modelspace)

    seen: set[int] = set()
    for space in spaces:
        try:
            entities = space.query("TEXT,MTEXT,INSERT")
        except Exception:
            entities = []
        for entity in entities:
            key = id(entity)
            if key in seen:
                continue
            seen.add(key)
            for fragment in _iter_text_fragments_from_entity(entity):
                fragments.append((fragment.x, fragment.y, fragment.height, fragment.text))

    return fragments


def _looks_like_hole_header(text: str) -> bool:
    upper = text.upper()
    has_hole = any(token in upper for token in _HEADER_HOLE_TOKENS)
    has_ref = any(token in upper for token in _HEADER_REF_TOKENS)
    has_qty = any(token in upper for token in _HEADER_QTY_TOKENS)
    has_desc = any(token in upper for token in _HEADER_DESC_TOKENS)
    has_side = any(token in upper for token in _HEADER_SIDE_TOKENS)
    score = sum(
        1
        for flag in (has_ref, has_desc, has_qty or has_side)
        if flag
    )
    return has_hole and score >= 2


def _assign_row_breaks(fragments: Sequence[_TextFragment]) -> list[list[_TextFragment]]:
    if not fragments:
        return []

    ordered = sorted(fragments, key=lambda frag: (-frag.y, frag.x))
    heights = [frag.height for frag in ordered if frag.height and frag.height > 0]
    med = median(heights) if heights else 0.0
    threshold = 0.75 * med if med > 0 else 0.75

    rows: list[list[_TextFragment]] = []
    current: list[_TextFragment] = []
    prev_y: float | None = None
    for fragment in ordered:
        if prev_y is None or (prev_y - fragment.y) > threshold:
            if current:
                rows.append(current)
            current = [fragment]
        else:
            current.append(fragment)
        prev_y = fragment.y
    if current:
        rows.append(current)

    for row in rows:
        row.sort(key=lambda frag: frag.x)
    return rows


def _kmeans_1d(points: Sequence[float], clusters: int) -> tuple[list[float], float]:
    if clusters <= 0:
        return [], 0.0
    pts = sorted(points)
    if not pts:
        return [], 0.0
    if len(pts) <= clusters:
        centers = sorted(set(pts))
        inertia = sum((pt - centers[min(range(len(centers)), key=lambda idx: abs(pt - centers[idx]))]) ** 2 for pt in pts)
        return centers, inertia

    step = (len(pts) - 1) / (clusters - 1)
    centers = [pts[int(round(step * i))] for i in range(clusters)]

    for _ in range(25):
        buckets: list[list[float]] = [[] for _ in range(clusters)]
        for pt in pts:
            idx = min(range(clusters), key=lambda j: abs(pt - centers[j]))
            buckets[idx].append(pt)
        new_centers: list[float] = []
        for idx, bucket in enumerate(buckets):
            if bucket:
                new_centers.append(sum(bucket) / len(bucket))
            else:
                fallback = max(pts, key=lambda value: min(abs(value - c) for c in centers))
                new_centers.append(fallback)
        if all(abs(new - old) < 1e-6 for new, old in zip(new_centers, centers)):
            centers = new_centers
            break
        centers = new_centers

    inertia = 0.0
    for pt in pts:
        idx = min(range(clusters), key=lambda j: abs(pt - centers[j]))
        inertia += (pt - centers[idx]) ** 2
    centers.sort()
    return centers, inertia


def _best_column_centers(rows: Sequence[Sequence[_TextFragment]]) -> list[float]:
    if not rows:
        return []
    target_row = max(rows, key=lambda row: len(row))
    xs = [frag.x for frag in target_row]
    if not xs:
        return []
    unique_x = sorted(set(xs))
    if len(unique_x) <= 1:
        return unique_x

    best_centers = unique_x
    best_inertia = float("inf")
    max_clusters = min(4, len(unique_x))
    for clusters in range(2, max_clusters + 1):
        centers, inertia = _kmeans_1d(xs, clusters)
        if not centers:
            continue
        if inertia < best_inertia - 1e-6 or (
            abs(inertia - best_inertia) <= 1e-6 and len(centers) > len(best_centers)
        ):
            best_centers = centers
            best_inertia = inertia
    return best_centers


def _snap_row_to_centers(row: Sequence[_TextFragment], centers: Sequence[float]) -> dict[int, list[_TextFragment]]:
    assignments: dict[int, list[_TextFragment]] = {idx: [] for idx in range(len(centers))}
    if not centers:
        return assignments
    for fragment in row:
        idx = min(range(len(centers)), key=lambda j: abs(fragment.x - centers[j]))
        assignments[idx].append(fragment)
    for value in assignments.values():
        value.sort(key=lambda frag: frag.x)
    return assignments


def _collect_column_texts(
    snapped_rows: Sequence[dict[int, list[_TextFragment]]],
    column_count: int,
    skip_row: int | None = None,
) -> dict[int, list[str]]:
    texts: dict[int, list[str]] = {idx: [] for idx in range(column_count)}
    for row_index, row in enumerate(snapped_rows):
        if skip_row is not None and row_index == skip_row:
            continue
        for idx in range(column_count):
            fragments = row.get(idx, [])
            if not fragments:
                continue
            cell_text = " ".join(fragment.text for fragment in fragments).strip()
            if cell_text:
                texts[idx].append(cell_text)
    return texts


def _score_qty_column(texts: Sequence[str]) -> int:
    score = 0
    for text in texts:
        if _QTY_ONLY_RE.match(text):
            score += 3
        elif re.search(r"\b\d+\b", text):
            score += 1
    return score


def _score_ref_column(texts: Sequence[str]) -> int:
    score = 0
    for text in texts:
        upper = text.upper()
        if any(token in upper for token in _HEADER_REF_TOKENS):
            score += 3
        if re.search(r"\d+\s*/\s*\d+", text):
            score += 2
        if re.search(r"\d+(?:\.\d+)?", text):
            score += 1
    return score


def _score_side_column(texts: Sequence[str]) -> int:
    score = 0
    for text in texts:
        upper = text.upper()
        if any(token in upper for token in _SIDE_TOKENS):
            score += 2
        if "&" in upper and "BACK" in upper:
            score += 1
    return score


def _assign_column_roles(
    snapped_rows: Sequence[dict[int, list[_TextFragment]]],
    centers: Sequence[float],
    header_index: int,
) -> dict[int, str]:
    column_count = len(centers)
    header_row = snapped_rows[header_index] if 0 <= header_index < len(snapped_rows) else {}
    header_text_by_col = {
        idx: " ".join(fragment.text for fragment in header_row.get(idx, [])).strip()
        for idx in range(column_count)
    }

    roles: dict[int, str] = {}
    assigned_roles: set[str] = set()

    def _try_assign(idx: int, role: str) -> None:
        if idx in roles or role in assigned_roles:
            return
        roles[idx] = role
        assigned_roles.add(role)

    for idx in range(column_count):
        header_text = header_text_by_col.get(idx, "")
        if not header_text:
            continue
        upper = header_text.upper()
        if any(token in upper for token in _HEADER_QTY_TOKENS):
            _try_assign(idx, "qty")
        elif any(token in upper for token in _HEADER_REF_TOKENS):
            _try_assign(idx, "ref")
        elif any(token in upper for token in _HEADER_SIDE_TOKENS):
            _try_assign(idx, "side")
        elif any(token in upper for token in _HEADER_DESC_TOKENS):
            _try_assign(idx, "desc")
        elif any(token in upper for token in _HEADER_HOLE_TOKENS):
            _try_assign(idx, "hole")

    column_texts = _collect_column_texts(snapped_rows, column_count, skip_row=header_index)
    available = [idx for idx in range(column_count) if idx not in roles]

    if "qty" not in assigned_roles:
        best_idx = None
        best_score = 0
        for idx in available:
            score = _score_qty_column(column_texts.get(idx, []))
            if score > best_score or (score == best_score and best_idx is not None and idx < best_idx):
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score > 0:
            _try_assign(best_idx, "qty")
            available.remove(best_idx)

    if "ref" not in assigned_roles:
        best_idx = None
        best_score = 0
        for idx in available:
            score = _score_ref_column(column_texts.get(idx, []))
            if score > best_score or (score == best_score and best_idx is not None and idx < best_idx):
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score > 0:
            _try_assign(best_idx, "ref")
            available.remove(best_idx)

    if "side" not in assigned_roles:
        best_idx = None
        best_score = 0
        for idx in available:
            score = _score_side_column(column_texts.get(idx, []))
            if score > best_score or (score == best_score and best_idx is not None and idx < best_idx):
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score > 0:
            _try_assign(best_idx, "side")
            available.remove(best_idx)

    if "desc" not in assigned_roles and available:
        idx = max(available)
        _try_assign(idx, "desc")
        available = [col for col in available if col != idx]

    if "hole" not in assigned_roles and available:
        idx = min(available)
        _try_assign(idx, "hole")
        available = [col for col in available if col != idx]

    for idx in available:
        roles[idx] = "desc"

    return roles


def _build_row_from_columns(row: dict[int, list[_TextFragment]], roles: Mapping[int, str]) -> dict[str, str]:
    ordered_indices = sorted(row.keys())
    segments: dict[str, list[str]] = defaultdict(list)
    for idx in ordered_indices:
        fragments = row.get(idx, [])
        if not fragments:
            continue
        text = " ".join(fragment.text for fragment in fragments).strip()
        if not text:
            continue
        role = roles.get(idx, "desc")
        segments.setdefault(role, []).append(text)

    def _joined(role: str) -> str:
        return " ".join(segments.get(role, [])).strip()

    qty_text = _joined("qty")
    if not qty_text:
        for key in ("hole", "ref", "desc"):
            items = segments.get(key, [])
            for pos, item in enumerate(list(items)):
                match = _INLINE_QTY_RE.match(item)
                if not match:
                    continue
                qty_text = match.group(1)
                remainder = match.group(2).strip()
                if remainder:
                    items[pos] = remainder
                else:
                    items.pop(pos)
                break
            if qty_text:
                break

    hole_text = " ".join(segments.get("hole", [])).strip()
    ref_text = " ".join(segments.get("ref", [])).strip()
    desc_text = " ".join(segments.get("desc", [])).strip()
    side_text = " ".join(segments.get("side", [])).strip()

    if side_text:
        side_upper = side_text.upper()
        if side_upper in {"FRONT", "BACK"}:
            desc_text = " ".join(filter(None, [desc_text, f"FROM {side_upper}"])).strip()
        elif any(token in side_upper for token in _SIDE_TOKENS):
            desc_text = " ".join(filter(None, [desc_text, side_text])).strip()
        else:
            desc_text = " ".join(filter(None, [side_text, desc_text])).strip()

    row_payload: dict[str, str] = {
        "hole": hole_text,
        "ref": ref_text,
        "qty": qty_text,
        "desc": desc_text,
    }
    if side_text:
        row_payload["side"] = side_text
    return row_payload


def parse_text_table_fragments(
    fragments: Sequence[tuple[float, float, float, str]] | Sequence[_TextFragment],
    *,
    min_rows: int = 5,
) -> list[dict[str, str]]:
    """Return HOLE TABLE rows parsed from normalized text fragments."""

    normalized: list[_TextFragment] = []
    for fragment in fragments:
        if isinstance(fragment, _TextFragment):
            text_value = fragment.text
            entry = fragment
        else:
            if len(fragment) != 4:
                continue
            x_val, y_val, height_val, text_value = fragment
            entry = _TextFragment(float(x_val), float(y_val), float(height_val or 0.0), _normalize_fragment_text(text_value))
        if not entry.text:
            continue
        normalized.append(entry)

    if not normalized:
        return []

    rows = _assign_row_breaks(normalized)
    if not rows:
        return []

    header_index = None
    for idx, row in enumerate(rows):
        row_text = " ".join(fragment.text for fragment in row)
        if _looks_like_hole_header(row_text):
            header_index = idx
            break
    if header_index is None:
        return []

    centers = _best_column_centers(rows)
    if not centers:
        return []

    snapped_rows = [_snap_row_to_centers(row, centers) for row in rows]
    roles = _assign_column_roles(snapped_rows, centers, header_index)

    parsed_rows: list[dict[str, str]] = []
    for idx, row in enumerate(snapped_rows):
        if idx <= header_index:
            continue
        combined = " ".join(
            fragment.text for fragments in row.values() for fragment in fragments if fragment.text
        ).strip()
        if not combined:
            continue
        if _looks_like_hole_header(combined):
            break
        parsed = _build_row_from_columns(row, roles)
        if not any(parsed.get(key) for key in ("hole", "ref", "qty", "desc")):
            continue
        parsed_rows.append(parsed)

    if len(parsed_rows) < min_rows:
        return []

    return parsed_rows


def _sanitize_ops_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the minimal row payload required for ops cards."""

    hole = str(row.get("hole") or row.get("id") or "").strip()
    ref = str(row.get("ref") or "").strip()
    desc = str(row.get("desc") or "").strip()
    qty = _ops_qty_from_value(row.get("qty"))
    return {"hole": hole, "ref": ref, "qty": qty, "desc": desc}


def _count_ops_card_rows(rows: Iterable[Mapping[str, Any]] | None) -> int:
    count = 0
    for entry in rows or []:
        if not isinstance(entry, _MappingABC):
            continue
        qty = _ops_qty_from_value(entry.get("qty"))
        desc = str(entry.get("desc") or "").strip()
        ref = str(entry.get("ref") or "").strip()
        if qty > 0 or desc or ref:
            count += 1
    return count


def _apply_built_rows(
    ops_summary: MutableMapping[str, Any] | Mapping[str, Any] | None,
    rows: Iterable[Mapping[str, Any]] | None,
) -> int:
    built_rows = _count_ops_card_rows(rows)
    if isinstance(ops_summary, _MutableMappingABC):
        cast(TypingMutableMapping[str, Any], ops_summary)["built_rows"] = int(built_rows)
    return int(built_rows)


def parse_ops_per_hole(desc: str) -> dict[str, int]:
    """Return ops per HOLE (not multiplied by QTY)."""

    U = _norm_txt(desc)
    ops = defaultdict(int)

    clauses = re.split(r"[;]+", U) if ";" in U else [U]
    for cl in clauses:
        if not cl.strip():
            continue
        side = (
            "both"
            if _SIDE_BOTH.search(cl)
            else (
                "back"
                if _SIDE_BACK.search(cl)
                else ("front" if _SIDE_FRONT.search(cl) else None)
            )
        )

        has_tap = re.search(_OP_WORDS["tap"], cl)
        has_thru = re.search(_OP_WORDS["thru"], cl)
        has_cbore = re.search(_OP_WORDS["cbore"], cl)
        has_csk = re.search(_OP_WORDS["csk"], cl)
        has_cdr = re.search(_OP_WORDS["cdrill"], cl)
        has_jig = re.search(_OP_WORDS["jig"], cl)

        if has_tap:
            ops["drill"] += 1
            if side == "back":
                ops["tap_back"] += 1
            elif side == "both":
                ops["tap_front"] += 1
                ops["tap_back"] += 1
            else:
                ops["tap_front"] += 1

        if has_thru and not has_tap:
            ops["drill"] += 1

        if has_cbore:
            if side == "back":
                ops["cbore_back"] += 1
            elif side == "both":
                ops["cbore_front"] += 1
                ops["cbore_back"] += 1
            else:
                ops["cbore_front"] += 1

        if has_csk:
            if side == "back":
                ops["csk_back"] += 1
            elif side == "both":
                ops["csk_front"] += 1
                ops["csk_back"] += 1
            else:
                ops["csk_front"] += 1

        if has_cdr:
            if side == "back":
                ops["spot_back"] += 1
            elif side == "both":
                ops["spot_front"] += 1
                ops["spot_back"] += 1
            else:
                ops["spot_front"] += 1

        if has_jig:
            ops["jig_grind"] += 1

    return dict(ops)


def _aggregate_ops_legacy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy aggregate implementation (regex-based) preserved for fallback."""

    totals: defaultdict[str, int] = defaultdict(int)
    rows_simple: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    simple_rows: list[dict[str, Any]] = []
    for r in rows:
        row_payload = _sanitize_ops_row(r)
        per = parse_ops_per_hole(row_payload.get("desc", ""))
        qty = row_payload.get("qty", 0) or 0
        row_total = {k: v * qty for k, v in per.items()}
        for k, v in row_total.items():
            totals[k] += v
        simple_row = {
            "hole": r.get("hole") or r.get("id") or "",
            "ref": (r.get("ref") or "").strip(),
            "qty": qty,
            "desc": str(r.get("desc", "")),
        }
        diameter_val = _coerce_float_or_none(r.get("diameter_in"))
        if diameter_val is not None:
            simple_row["diameter_in"] = float(diameter_val)
        rows_simple.append(simple_row)
        detail.append(
            {
                **row_payload,
                "per_hole": per,
                "total": row_total,
            }
        )
        simple_rows.append(row_payload)

    actions_total = sum(totals.values())
    back_ops_total = (
        totals.get("cbore_back", 0)
        + totals.get("csk_back", 0)
        + totals.get("tap_back", 0)
        + totals.get("spot_back", 0)
    )
    flip_required = back_ops_total > 0
    built_rows = _count_ops_card_rows(simple_rows)
    return {
        "totals": dict(totals),
        "rows": rows_simple,
        "rows_detail": detail,
        "actions_total": int(actions_total),
        "back_ops_total": int(back_ops_total),
        "flip_required": bool(flip_required),
        "built_rows": int(built_rows),
    }


def _parser_rules_v2_enabled(
    params_obj: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when parser_rules_v2 feature flag is enabled."""

    env_val = os.getenv("PARSER_RULES_V2")
    if env_val is not None:
        normalized = str(env_val).strip().lower()
        if normalized in {"", "0", "false", "off", "no"}:
            return False
        return True

    if isinstance(params_obj, _MappingABC):
        for key in ("parser_rules_v2", "ParserRulesV2", "parserRulesV2"):
            if key in params_obj:
                normalized = str(params_obj.get(key)).strip().lower()
                if normalized in {"", "0", "false", "off", "no"}:
                    return False
                return True

    return False


def _coerce_int_or_zero(value: Any) -> int:
    """Coerce ``value`` to an integer, returning ``0`` on failure."""

    coerced = _coerce_float_or_none(value)
    if coerced is None:
        return 0
    try:
        return int(coerced)
    except Exception:
        return 0


def _normalize_ops_entries(
    ops_entries: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize chart-derived operation entries into a consistent schema."""

    normalized: list[dict[str, Any]] = []
    if not ops_entries:
        return normalized

    def _truthy_flag(value: Any) -> bool:
        try:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            text = str(value).strip().lower()
            if not text:
                return False
            return text not in {"0", "false", "off", "no", "n"}
        except Exception:
            return False

    for entry in ops_entries:
        if not isinstance(entry, _MappingABC):
            continue
        raw_type = str(entry.get("type") or entry.get("op") or "").strip().lower()
        derived_ops: list[tuple[str, Mapping[str, Any]]] = []

        ops_payload = entry.get("ops")
        if isinstance(ops_payload, Iterable):
            for op_payload in ops_payload:
                if not isinstance(op_payload, _MappingABC):
                    continue
                payload_type = str(
                    op_payload.get("type") or op_payload.get("op") or ""
                ).strip()
                if not payload_type:
                    continue
                derived_ops.append((payload_type, op_payload))

        if not derived_ops:
            if raw_type:
                derived_ops.append((raw_type, entry))
            else:
                if entry.get("tap"):
                    derived_ops.append(("tap", entry))
                if _truthy_flag(entry.get("cbore")):
                    derived_ops.append(("cbore", entry))
                if _truthy_flag(entry.get("csk")):
                    derived_ops.append(("csk", entry))
                if _truthy_flag(entry.get("thru")):
                    derived_ops.append(("drill", entry))
                if _truthy_flag(entry.get("jig_grind")):
                    derived_ops.append(("jig_grind", entry))

        for derived_type, payload in derived_ops:
            op = dict(entry)
            op.pop("ops", None)
            if isinstance(payload, _MappingABC) and payload is not entry:
                for key, value in payload.items():
                    op[key] = value
            op_type_raw = derived_type.strip().lower()
            if op_type_raw in {"counterbore", "c'bore"}:
                op_type = "cbore"
            elif op_type_raw in {
                "counterdrill",
                "counter_drill",
                "counter-drill",
                "counter drill",
                "c drill",
                "c-drill",
            }:
                op_type = "counterdrill"
            elif op_type_raw in {"countersink", "csink", "c'sink", "csk"}:
                op_type = "csk"
            elif op_type_raw in {"spot", "spot_drill", "center_drill", "c'drill"}:
                op_type = "spot"
            elif op_type_raw in {"jig", "jig_grind", "jig-grind"}:
                op_type = "jig_grind"
            elif op_type_raw in {"tapping", "tap"}:
                op_type = "tap"
            elif op_type_raw in {"drill", "deep_drill"}:
                op_type = "drill"
            else:
                op_type = op_type_raw

            qty = _coerce_int_or_zero(op.get("qty"))
            if qty <= 0:
                continue

            side_raw = str(op.get("side") or op.get("face") or "").strip()
            from_face = str(entry.get("from_face") or "").strip()
            if not side_raw and from_face:
                side_raw = from_face
            side_upper = side_raw.upper()
            double_sided = _truthy_flag(op.get("double_sided")) or side_upper in {
                "FRONT & BACK",
                "FRONT AND BACK",
                "BOTH",
                "BOTH SIDES",
                "2 SIDES",
                "TWO SIDES",
            }
            if side_upper == "BACK" or _truthy_flag(op.get("from_back")):
                base_side = "BACK"
            elif side_upper == "FRONT":
                base_side = "FRONT"
            else:
                base_side = "FRONT"
            sides = ["FRONT", "BACK"] if double_sided else [base_side]

            ref = str(op.get("ref") or op.get("hole") or "").strip()
            thread = str(op.get("thread") or op.get("tap") or "").strip()

            depth_in = _coerce_float_or_none(op.get("depth_in"))
            if depth_in is None:
                depth_mm = _coerce_float_or_none(op.get("depth_mm"))
                if depth_mm is not None:
                    depth_in = float(depth_mm) / 25.4

            dia_in = _coerce_float_or_none(op.get("dia_in"))
            if dia_in is None:
                dia_in = _coerce_float_or_none(op.get("diameter_in"))
            if dia_in is None:
                dia_mm = _coerce_float_or_none(op.get("dia_mm"))
                if dia_mm is not None:
                    dia_in = float(dia_mm) / 25.4
            if dia_in is None:
                major_mm = _coerce_float_or_none(op.get("major_mm"))
                if major_mm is not None:
                    dia_in = float(major_mm) / 25.4
            if dia_in is None:
                dia_in = _coerce_float_or_none(op.get("ref_dia_in"))
            if dia_in is None and ref:
                ref_in = parse_dim(ref)
                if ref_in is not None:
                    dia_in = ref_in

            if thread and dia_in is None:
                dia_in = _coerce_float_or_none(entry.get("major_dia_in"))

            ref_label: str
            if thread:
                ref_label = thread
            elif dia_in is not None:
                ref_label = f"Ø{dia_in:.4f}"
            else:
                ref_label = ref

            normalized.append(
                {
                    "type": op_type,
                    "qty": int(qty),
                    "sides": sides,
                    "side": sides[0] if sides else "",
                    "double_sided": bool(double_sided),
                    "ref": ref,
                    "ref_label": ref_label,
                    "thread": thread,
                    "ref_dia_in": float(dia_in) if dia_in is not None else None,
                    "depth_in": float(depth_in) if depth_in is not None else None,
                    "thru": bool(_truthy_flag(entry.get("thru"))),
                    "source": entry.get("source"),
                    "claimed_by_tap": bool(op.get("claimed_by_tap")),
                    "pilot_for_thread": op.get("pilot_for_thread"),
                }
            )

    return normalized


def aggregate_ops(
    rows: list[dict[str, Any]],
    ops_entries: Iterable[Mapping[str, Any]] | None = None,
    *,
    parser_rules_params: Mapping[str, Any] | None = None,
    parser_rules_v2_enabled: bool | None = None,
) -> dict[str, Any]:
    legacy_summary = _aggregate_ops_legacy(rows)
    normalized_ops = _normalize_ops_entries(ops_entries)
    if not normalized_ops:
        return legacy_summary

    totals: defaultdict[str, int] = defaultdict(int)
    group_totals: defaultdict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    detail: list[dict[str, Any]] = []
    drill_claim_bins: Counter[float] = Counter()
    tap_claim_bins: Counter[float] = Counter()
    drill_group_refs: dict[float, list[dict[str, Any]]] = defaultdict(list)
    drill_detail_refs: dict[float, list[dict[str, Any]]] = defaultdict(list)
    claimed_pilot_diams: list[float] = []

    rows_simple = list(legacy_summary.get("rows") or [])
    built_rows = int(legacy_summary.get("built_rows") or _count_ops_card_rows(rows_simple))

    def _group_entry(
        type_key: str,
        ref_key: str,
        side_key: str,
        *,
        ref_label: str,
        ref_text: str,
        diameter_in: float | None,
    ) -> dict[str, Any]:
        type_bucket = group_totals.setdefault(type_key, {})
        ref_bucket = type_bucket.setdefault(ref_key, {})
        entry = ref_bucket.get(side_key)
        if entry is None:
            entry = {
                "type": type_key,
                "ref": ref_text,
                "ref_label": ref_label,
                "diameter_in": float(diameter_in) if diameter_in is not None else None,
                "qty": 0,
                "sources": [],
                "_depths": [],
            }
            ref_bucket[side_key] = entry
        return entry

    for op in normalized_ops:
        op_type = op["type"]
        qty = int(op.get("qty") or 0)
        if qty <= 0:
            continue
        sides = list(op.get("sides") or []) or ["FRONT"]
        ref_dia = _coerce_float_or_none(op.get("ref_dia_in"))
        ref_label = str(op.get("ref_label") or op.get("ref") or "").strip()
        if ref_dia is not None:
            ref_key = f"{ref_dia:.4f}"
        elif op.get("thread"):
            ref_key = str(op.get("thread")).strip()
        elif ref_label:
            ref_key = ref_label
        else:
            ref_key = op_type

        detail_entry = {
            "type": op_type,
            "qty": qty,
            "sides": sides,
            "side": sides[0] if sides else "",
            "double_sided": bool(op.get("double_sided")),
            "ref": str(op.get("ref") or ""),
            "ref_label": ref_label,
            "ref_dia_in": float(ref_dia) if ref_dia is not None else None,
            "depth_in": _coerce_float_or_none(op.get("depth_in")),
            "thread": str(op.get("thread") or op.get("tap") or ""),
            "thru": bool(op.get("thru")),
            "source": op.get("source"),
        }
        pilot_flag = bool(op.get("claimed_by_tap") or op.get("pilot_for_thread"))
        dia_key: float | None = None
        if ref_dia is not None and math.isfinite(ref_dia):
            dia_key = round(float(ref_dia), 4)

        if op.get("claimed_by_tap"):
            detail_entry["claimed_by_tap"] = True
        if op.get("pilot_for_thread"):
            detail_entry["pilot_for_thread"] = op.get("pilot_for_thread")
        detail.append(detail_entry)

        for side_key in sides:
            side_norm = "BACK" if side_key.upper() == "BACK" else "FRONT"
            bucket = _group_entry(
                op_type,
                ref_key,
                side_norm,
                ref_label=ref_label,
                ref_text=str(op.get("ref") or ""),
                diameter_in=ref_dia,
            )
            bucket["qty"] = int(bucket.get("qty", 0)) + qty
            depth_val = _coerce_float_or_none(op.get("depth_in"))
            if depth_val is not None:
                bucket.setdefault("_depths", []).append(float(depth_val))
            source_val = op.get("source")
            if source_val:
                try:
                    source_text = str(source_val)
                except Exception:
                    source_text = None
                if source_text and source_text not in bucket["sources"]:
                    bucket["sources"].append(source_text)

            if op_type == "tap":
                totals[f"tap_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            elif op_type == "cbore":
                totals[f"cbore_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            elif op_type == "csk":
                totals[f"csk_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            elif op_type == "spot":
                totals[f"spot_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            elif op_type == "counterdrill":
                totals[f"counterdrill_{'back' if side_norm == 'BACK' else 'front'}"] += qty
            if op_type == "drill" and pilot_flag and dia_key is not None:
                if bucket not in drill_group_refs.setdefault(dia_key, []):
                    drill_group_refs[dia_key].append(bucket)
        if op_type == "drill":
            totals["drill"] += qty
            if pilot_flag and dia_key is not None:
                drill_claim_bins[dia_key] += qty
                drill_detail_refs[dia_key].append(detail_entry)
        elif op_type == "jig_grind":
            totals["jig_grind"] += qty
        elif op_type == "counterdrill":
            totals["counterdrill"] += qty
        if op_type == "tap" and dia_key is not None and dia_key > 0:
            tap_claim_bins[dia_key] += qty

    totals["tap_total"] = totals.get("tap_front", 0) + totals.get("tap_back", 0)
    totals["cbore_total"] = totals.get("cbore_front", 0) + totals.get("cbore_back", 0)
    totals["csk_total"] = totals.get("csk_front", 0) + totals.get("csk_back", 0)
    totals["spot_total"] = totals.get("spot_front", 0) + totals.get("spot_back", 0)
    totals["counterdrill_total"] = (
        totals.get("counterdrill_front", 0) + totals.get("counterdrill_back", 0)
    )

    drill_subtracted_total = 0
    processed_dia_keys: set[float] = set()
    for dia_key, available in drill_claim_bins.items():
        tap_claim_qty = tap_claim_bins.get(dia_key, 0)
        subtract = min(available, tap_claim_qty) if tap_claim_qty else available
        if subtract <= 0:
            processed_dia_keys.add(dia_key)
            continue
        drill_subtracted_total += subtract
        claimed_pilot_diams.extend([float(dia_key)] * subtract)
        remaining = subtract
        for bucket in drill_group_refs.get(dia_key, []):
            qty_val = int(_coerce_float_or_none(bucket.get("qty")) or 0)
            if qty_val <= 0:
                continue
            take = min(qty_val, remaining)
            bucket["qty"] = qty_val - take
            remaining -= take
            if remaining <= 0:
                break
        remaining_detail = subtract
        for entry in drill_detail_refs.get(dia_key, []):
            entry_qty = int(_coerce_float_or_none(entry.get("qty")) or 0)
            if entry_qty <= 0:
                continue
            take = min(entry_qty, remaining_detail)
            entry["qty"] = entry_qty - take
            remaining_detail -= take
            if remaining_detail <= 0:
                break
        processed_dia_keys.add(dia_key)
    for dia_key, claim_qty in tap_claim_bins.items():
        if dia_key in processed_dia_keys:
            continue
        available = drill_claim_bins.get(dia_key, 0)
        subtract = min(available, claim_qty)
        if subtract <= 0:
            continue
        drill_subtracted_total += subtract
        claimed_pilot_diams.extend([float(dia_key)] * subtract)
        remaining = subtract
        for bucket in drill_group_refs.get(dia_key, []):
            qty_val = int(_coerce_float_or_none(bucket.get("qty")) or 0)
            if qty_val <= 0:
                continue
            take = min(qty_val, remaining)
            bucket["qty"] = qty_val - take
            remaining -= take
            if remaining <= 0:
                break
        remaining_detail = subtract
        for entry in drill_detail_refs.get(dia_key, []):
            entry_qty = int(_coerce_float_or_none(entry.get("qty")) or 0)
            if entry_qty <= 0:
                continue
            take = min(entry_qty, remaining_detail)
            entry["qty"] = entry_qty - take
            remaining_detail -= take
            if remaining_detail <= 0:
                break

    detail = [
        entry
        for entry in detail
        if int(_coerce_float_or_none(entry.get("qty")) or 0) > 0
    ]

    totals["drill"] = max(0, totals.get("drill", 0) - drill_subtracted_total)

    actions_total = (
        totals.get("drill", 0)
        + totals.get("tap_front", 0)
        + totals.get("tap_back", 0)
        + totals.get("cbore_front", 0)
        + totals.get("cbore_back", 0)
        + totals.get("csk_front", 0)
        + totals.get("csk_back", 0)
        + totals.get("spot_front", 0)
        + totals.get("spot_back", 0)
        + totals.get("counterdrill_front", 0)
        + totals.get("counterdrill_back", 0)
        + totals.get("jig_grind", 0)
    )

    back_ops_total = (
        totals.get("cbore_back", 0)
        + totals.get("csk_back", 0)
        + totals.get("tap_back", 0)
        + totals.get("spot_back", 0)
        + totals.get("counterdrill_back", 0)
    )

    grouped_final: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for type_key, ref_map in group_totals.items():
        grouped_final[type_key] = {}
        for ref_key, side_map in ref_map.items():
            grouped_final[type_key][ref_key] = {}
            for side_key, payload in side_map.items():
                qty_val = int(_coerce_float_or_none(payload.get("qty")) or 0)
                if qty_val <= 0:
                    continue
                depths = payload.pop("_depths", [])
                depth_vals = [
                    _coerce_float_or_none(val)
                    for val in depths
                    if _coerce_float_or_none(val) is not None
                ]
                if depth_vals:
                    depth_clean = [float(val) for val in depth_vals if val is not None]
                    if depth_clean:
                        payload["depth_in_avg"] = sum(depth_clean) / len(depth_clean)
                        payload["depth_in_max"] = max(depth_clean)
                        payload["depth_in_min"] = min(depth_clean)
                if not payload.get("sources"):
                    payload.pop("sources", None)
                grouped_final[type_key][ref_key][side_key] = payload

    summary = {
        "totals": {
            key: int(totals.get(key, 0))
            for key in (
                "drill",
                "tap_front",
                "tap_back",
                "tap_total",
                "cbore_front",
                "cbore_back",
                "cbore_total",
                "csk_front",
                "csk_back",
                "csk_total",
                "spot_front",
                "spot_back",
                "spot_total",
                "counterdrill_front",
                "counterdrill_back",
                "counterdrill_total",
                "jig_grind",
            )
        },
        "rows": rows_simple,
        "rows_detail": detail,
        "actions_total": int(actions_total),
        "back_ops_total": int(back_ops_total),
        "flip_required": bool(back_ops_total > 0),
        "built_rows": int(built_rows),
        "group_totals": grouped_final,
    }

    if claimed_pilot_diams:
        summary["claims"] = {"claimed_pilot_diams": [float(val) for val in claimed_pilot_diams]}

    if parser_rules_v2_enabled is None:
        parser_rules_v2_enabled = _parser_rules_v2_enabled(parser_rules_params)

    if parser_rules_v2_enabled:
        legacy_totals = (legacy_summary or {}).get("totals") or {}
        try:
            logging.info(
                "[counts] legacy=%s new=%s",
                {k: int(legacy_totals.get(k, 0)) for k in sorted(legacy_totals.keys())},
                summary["totals"],
            )
        except Exception:
            pass

    return summary


def _rows_from_ops_summary(
    geo: Mapping[str, Any] | None,
    *,
    result: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> list[dict]:
    geo_map: Mapping[str, Any] = geo if isinstance(geo, _MappingABC) else {}
    ops = (geo_map or {}).get("ops_summary") or {}
    rows = ops.get("rows") if isinstance(ops, dict) else None
    if rows:
        return list(rows)
    if isinstance(ops, dict):
        detail = ops.get("rows_detail")
        if isinstance(detail, list):
            fallback: list[dict[str, Any]] = []
            for entry in detail:
                if not isinstance(entry, _MappingABC):
                    continue
                base = {
                    "hole": entry.get("hole", ""),
                    "ref": entry.get("ref", ""),
                    "qty": entry.get("qty", 0),
                    "desc": entry.get("desc", ""),
                }
                if entry.get("diameter_in") is not None:
                    base["diameter_in"] = entry.get("diameter_in")
                fallback.append(base)
            if fallback:
                return fallback
    _containers: list[dict] = []
    for candidate in (result, breakdown, geo_map):
        if isinstance(candidate, dict):
            _containers.append(candidate)
        elif isinstance(candidate, _MappingABC):
            try:
                _containers.append(dict(candidate))
            except Exception:
                continue
    chart_lines = _collect_chart_lines_context(*_containers)
    if chart_lines:
        fallback_rows = _chart_build_ops_rows_from_lines_fallback(chart_lines)
        if fallback_rows:
            return fallback_rows
    return []


def _side_of(desc: str) -> str:
    if _SIDE_BACK.search(desc or ""):
        return "BACK"
    if _SIDE_FRONT.search(desc or ""):
        return "FRONT"
    return "FRONT"


def _major_diameter_from_thread(spec: str) -> float | None:
    if not spec:
        return None
    spec_clean = spec.strip().upper().replace(" ", "")
    if "NPT" in spec_clean:
        parts = spec_clean.split("NPT", 1)[0]
        parts = parts.rstrip("-")
        if not parts:
            return None
        try:
            return float(Fraction(parts))
        except Exception:
            pass
        try:
            return float(parts)
        except Exception:
            return None
    if spec_clean.startswith("#"):
        try:
            gauge = int(re.sub(r"[^0-9]", "", spec_clean.split("-", 1)[0]))
        except Exception:
            return None
        return WIRE_GAGE_MAJOR_DIA_IN.get(gauge)
    if spec_clean.startswith("M"):
        try:
            mm_val = float(re.findall(r"M(\d+(?:\.\d+)?)", spec_clean)[0])
        except Exception:
            return None
        return mm_val / 25.4
    lead = spec_clean.split("-", 1)[0]
    try:
        if "/" in lead:
            return float(Fraction(lead))
        return float(lead)
    except Exception:
        return None


def _classify_thread_spec(spec: str) -> tuple[str, float, bool]:
    major = _major_diameter_from_thread(spec)
    if spec and "NPT" in spec.upper():
        return "pipe", TAP_MINUTES_BY_CLASS["pipe"], True
    if major is None:
        return "unknown", TAP_MINUTES_BY_CLASS["medium"], False
    if major <= 0.2:
        return "small", TAP_MINUTES_BY_CLASS["small"], False
    if major <= 0.3125:
        return "medium", TAP_MINUTES_BY_CLASS["medium"], False
    if major <= 0.5:
        return "large", TAP_MINUTES_BY_CLASS["large"], False
    return "xl", TAP_MINUTES_BY_CLASS["xl"], False


def _normalize_hole_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip().upper()
    cleaned = cleaned.replace("%%C", "").replace("%%c", "").replace("Ø", "").replace("⌀", "")
    return cleaned


def _dedupe_hole_entries(
    existing_entries: Iterable[dict[str, Any]],
    new_entries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = {
        _normalize_hole_text(entry.get("raw"))
        for entry in existing_entries
        if isinstance(entry, dict)
    }
    unique_entries: list[dict[str, Any]] = []
    for entry in new_entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = _normalize_hole_text(entry.get("raw"))
        if fingerprint and fingerprint in seen:
            continue
        if fingerprint:
            seen.add(fingerprint)
        unique_entries.append(entry)
    return unique_entries


def build_ops_summary_rows_from_hole_rows(
    hole_rows: Iterable[Any] | None,
) -> list[dict[str, Any]]:
    """Convert parsed hole rows into ops-summary table rows."""

    summary_rows: list[dict[str, Any]] = []
    if not hole_rows:
        return summary_rows

    for row in hole_rows:
        if row is None:
            continue

        if isinstance(row, Mapping):
            qty_raw = row.get("qty")
            ref = (
                row.get("ref")
                or row.get("drill_ref")
                or row.get("pilot")
                or ""
            )
            desc = row.get("description") or row.get("desc") or ""
            features = row.get("features")
        else:
            qty_raw = getattr(row, "qty", 0)
            ref = (
                getattr(row, "ref", None)
                or getattr(row, "drill_ref", None)
                or getattr(row, "pilot", None)
                or ""
            )
            desc = (
                getattr(row, "description", None)
                or getattr(row, "desc", None)
                or ""
            )
            try:
                features = list(getattr(row, "features", []) or [])
            except Exception:
                features = []

        try:
            qty = int(qty_raw or 0)
        except Exception:
            qty = 0

        if not desc:
            parts: list[str] = []
            feature_iterable: Iterable[Any] | None = None
            if isinstance(features, Mapping):
                feature_iterable = features.values()
            elif isinstance(features, Iterable) and not isinstance(features, (str, bytes)):
                feature_iterable = features
            features_list: list[Mapping[str, Any]] = []
            if feature_iterable is not None:
                try:
                    features_list = [
                        feature
                        for feature in feature_iterable
                        if isinstance(feature, Mapping)
                    ]
                except Exception:
                    features_list = []
            for feature in features_list:
                feature_type = str(feature.get("type", "")).lower()
                side = str(feature.get("side", "")).upper()
                if feature_type == "tap":
                    thread = feature.get("thread") or ""
                    depth = feature.get("depth_in")
                    tap_desc = f"{thread} TAP" if thread else "TAP"
                    if isinstance(depth, (int, float)):
                        tap_desc += f" × {depth:.2f}\""
                    if side:
                        tap_desc += f" FROM {side}"
                    parts.append(tap_desc)
                elif feature_type == "cbore":
                    dia = feature.get("dia_in")
                    depth = feature.get("depth_in")
                    cbore_desc = ""
                    if isinstance(dia, (int, float)):
                        cbore_desc += f"{dia:.4f} "
                    cbore_desc += "C’BORE"
                    if isinstance(depth, (int, float)):
                        cbore_desc += f" × {depth:.2f}\""
                    if side:
                        cbore_desc += f" FROM {side}"
                    parts.append(cbore_desc)
                elif feature_type in {"csk", "countersink"}:
                    dia = feature.get("dia_in")
                    depth = feature.get("depth_in")
                    csk_desc = ""
                    if isinstance(dia, (int, float)):
                        csk_desc += f"{dia:.4f} "
                    csk_desc += "C’SINK"
                    if isinstance(depth, (int, float)):
                        csk_desc += f" × {depth:.2f}\""
                    if side:
                        csk_desc += f" FROM {side}"
                    parts.append(csk_desc)
                elif feature_type == "drill":
                    ref_local = feature.get("ref") or ref or ""
                    thru = " THRU" if feature.get("thru", True) else ""
                    parts.append(f"{ref_local}{thru}".strip())
                elif feature_type == "spot":
                    depth = feature.get("depth_in")
                    spot_desc = "C’DRILL"
                    if isinstance(depth, (int, float)):
                        spot_desc += f" × {depth:.2f}\""
                    parts.append(spot_desc)
                elif feature_type == "jig":
                    parts.append("JIG GRIND")
            desc = "; ".join(part for part in parts if part)

        summary_rows.append(
            {
                "hole": str(
                    getattr(row, "hole_id", "")
                    or getattr(row, "letter", "")
                    or (row.get("hole") if isinstance(row, Mapping) else None)
                    or ""
                ),
                "ref": str(ref or ""),
                "qty": int(qty),
                "desc": str(desc or ""),
            }
        )

    return summary_rows


def _summary_row_side(desc: str, *, normalized: str | None = None) -> str | None:
    text = normalized if normalized is not None else _normalize_ops_desc(desc)
    if "FRONT & BACK" in text or "FRONT AND BACK" in text or "BOTH SIDES" in text:
        return "BOTH"
    if "FROM BACK" in text:
        return "BACK"
    if "FROM FRONT" in text:
        return "FRONT"
    if "BACK" in text and "FRONT" in text:
        return "BOTH"
    if "BACK" in text:
        return "BACK"
    if "FRONT" in text:
        return "FRONT"
    return None


def _aggregate_summary_rows(
    rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    totals: defaultdict[str, int] = defaultdict(int)
    actions: defaultdict[str, int] = defaultdict(int)
    operation_totals: defaultdict[str, int] = defaultdict(int)

    if rows is None:
        rows = []

    for row in rows:
        try:
            qty = int(float(row.get("qty") or 0))
        except Exception:
            qty = 0
        if qty <= 0:
            continue

        desc = str(row.get("desc") or row.get("description") or "")
        label, normalized = _match_summary_operation(desc)
        side = _summary_row_side(desc, normalized=normalized)
        operation_totals[label] += qty

        if label == "Tap":
            totals["tap"] += qty
            if side == "BACK":
                totals["tap_back"] += qty
                actions["tap_back"] += qty
            elif side == "BOTH":
                totals["tap_front"] += qty
                totals["tap_back"] += qty
                actions["tap_front"] += qty
                actions["tap_back"] += qty
            else:
                totals["tap_front"] += qty
                actions["tap_front"] += qty
        elif label == "C'bore":
            totals["counterbore"] += qty
            if side == "BACK":
                totals["counterbore_back"] += qty
                actions["counterbore_back"] += qty
            elif side == "BOTH":
                totals["counterbore_front"] += qty
                totals["counterbore_back"] += qty
                actions["counterbore_front"] += qty
                actions["counterbore_back"] += qty
            else:
                totals["counterbore_front"] += qty
                actions["counterbore_front"] += qty
        elif label == "C'drill":
            totals["spot"] += qty
            actions["spot"] += qty
        elif label == "Jig Grind":
            totals["jig_grind"] += qty
            actions["jig_grind"] += qty
        elif label == "Drill":
            totals["drill"] += qty
            actions["drill"] += qty
        else:
            totals["unknown"] += qty

    back_ops_total = int(
        totals.get("counterbore_back", 0) + totals.get("tap_back", 0)
    )
    actions_total = int(sum(actions.values()))

    return {
        "totals": dict(totals),
        "actions_total": actions_total,
        "back_ops_total": back_ops_total,
        "flip_required": bool(back_ops_total > 0),
        "operation_totals": dict(operation_totals),
    }


def update_geo_ops_summary_from_hole_rows(
    geo: MutableMapping[str, Any],
    *,
    hole_rows: Iterable[Any] | None = None,
    summary_rows: Iterable[Mapping[str, Any]] | None = None,
    chart_lines: Iterable[str] | None = None,
    chart_source: str | None = None,
    ops_source: str | None = None,
    chart_summary: Mapping[str, Any] | None = None,
    apply_built_rows: Callable[[
        MutableMapping[str, Any] | Mapping[str, Any] | None,
        Iterable[Mapping[str, Any]] | None,
    ], int]
    | None = None,
    summary_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Populate geo["ops_summary"] with rows and totals derived from hole data."""

    ops_rows: list[dict[str, Any]] = []
    if summary_rows is not None:
        for row in summary_rows:
            if not isinstance(row, Mapping):
                continue
            entry = dict(row)
            if "hole" not in entry:
                entry.setdefault("hole", entry.get("id") or "")
            if "desc" not in entry and "description" in entry:
                entry["desc"] = entry.get("description")
            ops_rows.append(entry)
    else:
        ops_rows = build_ops_summary_rows_from_hole_rows(hole_rows)
    if not ops_rows and chart_lines:
        ops_rows = _chart_build_ops_rows_from_lines_fallback(chart_lines)

    if not ops_rows:
        return {"rows": [], "totals": {}}

    ops_summary_map = geo.setdefault("ops_summary", {})
    ops_summary_map["rows"] = ops_rows
    source_value = ops_source or chart_source or "chart_lines"
    ops_summary_map["source"] = source_value
    rows = ops_rows
    try:
        qty_sum = sum(int(r.get("qty") or 0) for r in rows)
    except Exception:
        qty_sum = 0
    print(f"[EXTRACTOR] wrote ops rows: {len(rows)} (qty_sum={qty_sum})")

    aggregates = _aggregate_summary_rows(ops_rows)
    totals_map = dict(aggregates.get("totals", {}))
    existing_totals = ops_summary_map.get("totals")
    if isinstance(existing_totals, Mapping):
        merged_totals = dict(existing_totals)
        merged_totals.update(totals_map)
        totals_map = merged_totals
    if totals_map:
        ops_summary_map["totals"] = totals_map

    op_totals = aggregates.get("operation_totals")
    if isinstance(op_totals, Mapping):
        ops_summary_map["operation_totals"] = dict(op_totals)

    for key in ("actions_total", "back_ops_total", "flip_required"):
        if key not in ops_summary_map and key in aggregates:
            ops_summary_map[key] = aggregates[key]

    totals_from_summary: dict[str, int] = {}

    if apply_built_rows:
        try:
            apply_built_rows(ops_summary_map, ops_rows)
        except Exception:
            pass

    if chart_summary and isinstance(chart_summary, Mapping):
        try:
            tap_qty = int(chart_summary.get("tap_qty") or 0)
            if tap_qty:
                totals_from_summary["tap_total"] = tap_qty
        except Exception:
            pass
        try:
            cbore_qty = int(chart_summary.get("cbore_qty") or 0)
            if cbore_qty:
                totals_from_summary["cbore_total"] = cbore_qty
        except Exception:
            pass
        try:
            csk_qty = int(chart_summary.get("csk_qty") or 0)
            if csk_qty:
                totals_from_summary["csk_total"] = csk_qty
        except Exception:
            pass

    if totals_from_summary:
        existing_totals = ops_summary_map.get("totals")
        if isinstance(existing_totals, Mapping):
            totals_map = dict(existing_totals)
            totals_map.update(totals_from_summary)
        else:
            totals_map = dict(totals_from_summary)
        ops_summary_map["totals"] = totals_map
        # Maintain legacy top-level keys for downstream compatibility.
        for key, value in totals_from_summary.items():
            ops_summary_map[key] = value

    if summary_metadata and isinstance(summary_metadata, Mapping):
        for key, value in summary_metadata.items():
            if key in {"rows", "totals"}:
                continue
            ops_summary_map.setdefault(key, value)

    return {
        "rows": ops_rows,
        "totals": ops_summary_map.get("totals", {}),
    }


def _parse_hole_line(line: str, to_in: float, *, source: str | None = None) -> dict[str, Any] | None:
    if not line:
        return None
    U = line.upper()
    if not any(k in U for k in ("HOLE", "TAP", "THRU", "CBORE", "C'BORE", "DRILL")):
        return None

    entry: dict[str, Any] = {
        "qty": None,
        "tap": None,
        "tap_class": None,
        "tap_minutes_per": None,
        "tap_is_npt": False,
        "thru": False,
        "cbore": False,
        "csk": False,
        "ref_dia_in": None,
        "depth_in": None,
        "side": None,
        "double_sided": False,
        "from_back": False,
        "raw": line,
    }
    if source:
        entry["source"] = source

    m_qty = re.search(r"\bQTY\b[:\s]*(\d+)", U)
    if m_qty:
        entry["qty"] = int(m_qty.group(1))

    mt = RE_TAP.search(U)
    if mt:
        thread_spec = None
        if mt.lastindex and mt.lastindex >= 2:
            thread_spec = mt.group(2)
        else:
            thread_spec = mt.group(1)
        if thread_spec:
            cleaned = thread_spec.replace(" ", "")
        else:
            cleaned = None
        entry["tap"] = cleaned
        if cleaned:
            cls, minutes_per, is_npt = _classify_thread_spec(cleaned)
            entry["tap_class"] = cls
            entry["tap_minutes_per"] = minutes_per
            entry["tap_is_npt"] = is_npt
    else:
        m_npt = RE_NPT.search(U)
        if m_npt:
            cleaned = m_npt.group(0).replace(" ", "")
            entry["tap"] = cleaned
            cls, minutes_per, is_npt = _classify_thread_spec(cleaned)
            entry["tap_class"] = cls
            entry["tap_minutes_per"] = minutes_per
            entry["tap_is_npt"] = is_npt
        else:
            entry["tap"] = None
    entry["thru"] = bool(RE_THRU.search(U))
    entry["cbore"] = bool(RE_CBORE.search(U))
    entry["csk"] = bool(RE_CSK.search(U))
    if RE_FRONT_BACK.search(U):
        entry["double_sided"] = True

    md = RE_DEPTH.search(U)
    if md:
        try:
            depth_token = md.group(1)
        except IndexError:
            depth_token = None
        depth = None
        if depth_token:
            depth_val = _to_inch(depth_token)
            if depth_val is not None:
                try:
                    depth = depth_val * float(to_in)
                except Exception:
                    depth = None
        side = (md.group(2) or "").upper() or None
        if depth is not None:
            entry["depth_in"] = depth
        if side:
            entry["side"] = side
            if side == "BACK":
                entry["from_back"] = True

    back_hint = bool(
        re.search(r"\((?:FROM\s+)?BACK\)", U)
        or re.search(r"\bFROM\s+BACK\b", U)
        or re.search(r"\bBACK\s*SIDE\b", U)
        or "BACKSIDE" in U
    )
    if back_hint and str(entry.get("side") or "").upper() != "BACK":
        entry["side"] = "BACK"
        entry["from_back"] = True
    if re.search(r"\b(FRONT\s*&\s*BACK|BOTH\s+SIDES)\b", U):
        entry["double_sided"] = True

    mref = re.search(r"REF\s*(?:%%[Cc]|[Ø⌀])\s*({VALUE_PATTERN})", U)
    if mref:
        ref_token = mref.group(1)
        ref_val = _to_inch(ref_token)
        if ref_val is None:
            ref_val = first_inch_value(mref.group(0))
        if ref_val is not None:
            try:
                entry["ref_dia_in"] = ref_val * float(to_in)
            except Exception:
                entry["ref_dia_in"] = None

    if entry.get("ref_dia_in") is None:
        mdia = RE_DIA.search(U)
        if mdia and ("Ø" in U or "⌀" in U or " REF" in U or "%%C" in U or "%%c" in U):
            val = _to_inch(mdia.group(1)) if mdia.lastindex else None
            if val is None:
                val = first_inch_value(mdia.group(0))
            if val is not None:
                try:
                    entry["ref_dia_in"] = val * float(to_in)
                except Exception:
                    entry["ref_dia_in"] = None

    def _int_or_zero(value: Any) -> int:
        if value in (None, ""):
            return 0
        try:
            return int(value)
        except Exception:
            try:
                return int(round(float(value)))
            except Exception:
                return 0

    def _base_op_payload() -> dict[str, Any]:
        qty = _int_or_zero(entry.get("qty"))
        side_raw = str(entry.get("side") or "").strip().upper()
        if not side_raw:
            side_raw = "BACK" if entry.get("from_back") else "FRONT"
        payload: dict[str, Any] = {
            "qty": qty,
            "side": side_raw,
            "double_sided": bool(entry.get("double_sided")),
            "from_back": bool(entry.get("from_back")),
            "depth_in": entry.get("depth_in"),
            "ref_dia_in": entry.get("ref_dia_in"),
            "thru": bool(entry.get("thru")),
            "source": entry.get("source"),
        }
        ref_val = entry.get("ref")
        if ref_val:
            payload["ref"] = ref_val
        return payload

    ops: list[dict[str, Any]] = []

    if entry.get("tap"):
        tap_payload = _base_op_payload()
        tap_payload.update(
            {
                "type": "tap",
                "thread": entry.get("tap"),
                "tap_class": entry.get("tap_class"),
                "tap_minutes_per": entry.get("tap_minutes_per"),
                "tap_is_npt": entry.get("tap_is_npt"),
            }
        )
        ops.append({k: v for k, v in tap_payload.items() if v not in (None, "")})

    if entry.get("cbore"):
        cbore_payload = _base_op_payload()
        cbore_payload["type"] = "cbore"
        ops.append({k: v for k, v in cbore_payload.items() if v not in (None, "")})

    if entry.get("csk"):
        csk_payload = _base_op_payload()
        csk_payload["type"] = "csk"
        ops.append({k: v for k, v in csk_payload.items() if v not in (None, "")})

    if entry.get("thru") or entry.get("ref_dia_in"):
        drill_payload = _base_op_payload()
        drill_payload["type"] = "drill"
        if entry.get("tap"):
            drill_payload["claimed_by_tap"] = True
            drill_payload["pilot_for_thread"] = entry.get("tap")
            if entry.get("tap_is_npt"):
                drill_payload["claimed_is_npt"] = True
        ops.append({k: v for k, v in drill_payload.items() if v not in (None, "")})

    if ops:
        entry["ops"] = ops
        if not entry.get("type"):
            entry["type"] = ops[0].get("type")

    return entry


def _aggregate_hole_entries(entries: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    hole_count = 0
    tap_qty = 0
    cbore_qty = 0
    csk_qty = 0
    max_depth_in = 0.0
    back_ops = False
    tap_details: dict[str, dict[str, Any]] = {}
    tap_minutes_total = 0.0
    tap_class_counter: Counter[str] = Counter()
    npt_qty = 0
    double_cbore = False
    double_csk = False
    cbore_minutes_total = 0.0
    csk_minutes_total = 0.0
    if entries:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            qty_val = entry.get("qty")
            if qty_val is None:
                qty = 0
            else:
                try:
                    qty = int(qty_val)
                except Exception:
                    try:
                        qty = int(round(float(qty_val)))
                    except Exception:
                        qty = 0
            qty = qty if qty > 0 else 1
            hole_count += qty
            if entry.get("tap"):
                tap_qty += qty
                spec = str(entry.get("tap") or "").strip()
                cls = entry.get("tap_class") or "unknown"
                minutes_per = entry.get("tap_minutes_per") or TAP_MINUTES_BY_CLASS.get("medium", 0.3)
                tap_minutes_total += qty * float(minutes_per)
                tap_class_counter[str(cls)] += qty
                detail = tap_details.setdefault(
                    spec,
                    {
                        "spec": spec,
                        "qty": 0,
                        "class": cls,
                        "minutes_per_hole": float(minutes_per),
                        "total_minutes": 0.0,
                        "is_npt": bool(entry.get("tap_is_npt")),
                    },
                )
                detail["qty"] += qty
                detail["total_minutes"] += qty * float(minutes_per)
                if entry.get("tap_is_npt"):
                    npt_qty += qty
            if entry.get("cbore"):
                ops_qty = qty * (2 if entry.get("double_sided") else 1)
                cbore_qty += ops_qty
                cbore_minutes_total += ops_qty * CBORE_MIN_PER_SIDE_MIN
                if entry.get("double_sided"):
                    double_cbore = True
            if entry.get("csk"):
                ops_qty = qty * (2 if entry.get("double_sided") else 1)
                csk_qty += ops_qty
                csk_minutes_total += ops_qty * CSK_MIN_PER_SIDE_MIN
                if entry.get("double_sided"):
                    double_csk = True
            depth = entry.get("depth_in")
            try:
                if depth and float(depth) > max_depth_in:
                    max_depth_in = float(depth)
            except Exception:
                continue
            side = str(entry.get("side") or "").upper()
            if (
                side == "BACK"
                or (entry.get("raw") and "BACK" in str(entry.get("raw")).upper())
                or entry.get("double_sided")
                and (entry.get("cbore") or entry.get("csk"))
            ):
                back_ops = True
    tap_details_list = []
    for spec, detail in tap_details.items():
        detail["qty"] = int(detail.get("qty", 0) or 0)
        detail["total_minutes"] = round(float(detail.get("total_minutes", 0.0)), 3)
        detail["minutes_per_hole"] = round(float(detail.get("minutes_per_hole", 0.0)), 3)
        tap_details_list.append(detail)
    tap_details_list.sort(key=lambda d: -d.get("qty", 0))
    tap_class_counts = {cls: int(qty) for cls, qty in tap_class_counter.items() if qty}
    return {
        "hole_count": hole_count if hole_count else None,
        "tap_qty": tap_qty,
        "cbore_qty": cbore_qty,
        "csk_qty": csk_qty,
        "deepest_hole_in": max_depth_in if max_depth_in > 0 else None,
        "provenance": "HOLE TABLE / NOTES" if hole_count else None,
        "from_back": back_ops,
        "tap_details": tap_details_list,
        "tap_minutes_hint": round(tap_minutes_total, 3) if tap_minutes_total else None,
        "tap_class_counts": tap_class_counts,
        "npt_qty": npt_qty,
        "cbore_minutes_hint": round(cbore_minutes_total, 3) if cbore_minutes_total else None,
        "csk_minutes_hint": round(csk_minutes_total, 3) if csk_minutes_total else None,
        "double_sided_cbore": double_cbore,
        "double_sided_csk": double_csk,
    }


def summarize_hole_chart_lines(lines: Iterable[str] | None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for raw in lines or []:
        text = str(raw or "")
        if not text.strip():
            continue
        entry = _parse_hole_line(text, 1.0, source="CHART")
        if not entry:
            continue
        if not entry.get("qty"):
            mqty = re.search(r"\((\d+)\)", text)
            if mqty:
                try:
                    entry["qty"] = int(mqty.group(1))
                except Exception:
                    pass
        entries.append(entry)
    agg = _aggregate_hole_entries(entries)
    return {
        "tap_qty": int(agg.get("tap_qty") or 0),
        "cbore_qty": int(agg.get("cbore_qty") or 0),
        "csk_qty": int(agg.get("csk_qty") or 0),
        "deepest_hole_in": agg.get("deepest_hole_in"),
        "from_back": bool(agg.get("from_back")),
        "tap_details": agg.get("tap_details") or [],
        "tap_minutes_hint": agg.get("tap_minutes_hint"),
        "tap_class_counts": agg.get("tap_class_counts") or {},
        "npt_qty": int(agg.get("npt_qty") or 0),
        "cbore_minutes_hint": agg.get("cbore_minutes_hint"),
        "csk_minutes_hint": agg.get("csk_minutes_hint"),
        "double_sided_cbore": bool(agg.get("double_sided_cbore")),
        "double_sided_csk": bool(agg.get("double_sided_csk")),
    }


def summarize_hole_chart_agreement(
    entity_holes_mm: Iterable[Any] | None, chart_ops: Iterable[dict] | None
) -> dict[str, Any]:
    """Compare entity-detected hole sizes with chart-derived operations."""

    def _as_qty(value: Any) -> int:
        try:
            qty = int(round(float(value)))
        except Exception:
            qty = 0
        return qty if qty > 0 else 1

    bin_key = lambda dia: round(float(dia), 1)

    ent_bins: Counter[float] = Counter()
    if entity_holes_mm:
        for value in entity_holes_mm:
            try:
                dia = float(value)
            except Exception:
                continue
            if dia > 0:
                ent_bins[bin_key(dia)] += 1

    chart_bins: Counter[float] = Counter()
    tap_qty = 0
    cbore_qty = 0
    csk_qty = 0
    if chart_ops:
        for op in chart_ops:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type") or "").lower()
            qty = _as_qty(op.get("qty"))
            if op_type == "tap":
                tap_qty += qty
            if op_type == "cbore":
                cbore_qty += qty
            if op_type in {"csk", "countersink"}:
                csk_qty += qty
            if op_type != "drill":
                continue
            dia_raw = op.get("dia_mm")
            if dia_raw is None:
                continue
            try:
                dia_val = float(dia_raw)
            except Exception:
                continue
            if dia_val <= 0:
                continue
            chart_bins[bin_key(dia_val)] += qty

    entity_total = sum(ent_bins.values())
    chart_total = sum(chart_bins.values())
    max_total = max(entity_total, chart_total)
    tolerance = max(5, 0.1 * max_total) if max_total else 5
    agreement = abs(entity_total - chart_total) <= tolerance

    return {
        "entity_bins": {float(k): int(v) for k, v in ent_bins.items()},
        "chart_bins": {float(k): int(v) for k, v in chart_bins.items()},
        "tap_qty": int(tap_qty),
        "cbore_qty": int(cbore_qty),
        "csk_qty": int(csk_qty),
        "agreement": bool(agreement),
        "entity_total": int(entity_total),
        "chart_total": int(chart_total),
    }


__all__ = [
    "RE_TAP",
    "RE_NPT",
    "RE_THRU",
    "RE_CBORE",
    "RE_CSK",
    "RE_DEPTH",
    "RE_DIA",
    "RE_FRONT_BACK",
    "TAP_MINUTES_BY_CLASS",
    "COUNTERDRILL_MIN_PER_SIDE_MIN",
    "CBORE_MIN_PER_SIDE_MIN",
    "CSK_MIN_PER_SIDE_MIN",
    "_SPOT_TOKENS",
    "_THREAD_WITH_TPI_RE",
    "_THREAD_WITH_NPT_RE",
    "_CBORE_RE",
    "_SIDE_BACK",
    "_SIDE_FRONT",
    "_DEPTH_TOKEN",
    "_DIA_TOKEN",
    "parse_dim",
    "_rows_from_ops_summary",
    "_side_of",
    "_major_diameter_from_thread",
    "_classify_thread_spec",
    "_normalize_hole_text",
    "_dedupe_hole_entries",
    "build_ops_summary_rows_from_hole_rows",
    "update_geo_ops_summary_from_hole_rows",
    "_parse_hole_line",
    "_aggregate_hole_entries",
    "_build_ops_rows_from_lines_fallback",
    "build_ops_rows_from_lines_fallback",
    "summarize_hole_chart_lines",
    "summarize_hole_chart_agreement",
]


def build_ops_rows_from_lines_fallback(lines: Iterable[str] | None) -> list[dict]:
    """Proxy to the chart-line fallback parser exposed for hole operations helpers."""

    seq = [str(s) for s in lines or [] if str(s)]
    if not seq:
        return []
    return _chart_build_ops_rows_from_lines_fallback(seq)


_build_ops_rows_from_lines_fallback = _chart_build_ops_rows_from_lines_fallback
