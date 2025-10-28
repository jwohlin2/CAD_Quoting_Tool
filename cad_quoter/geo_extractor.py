#GEO Extractor
"""Isolated GEO extraction helpers for DWG/DXF sources."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import csv
import json
from fractions import Fraction
import inspect
import math
from functools import lru_cache
import os
from pathlib import Path
import re
import statistics
from typing import Any, Callable
from fnmatch import fnmatchcase

from cad_quoter import geometry
from cad_quoter.geometry import convert_dwg_to_dxf
from cad_quoter.geometry.dxf_enrich import detect_units_scale
from cad_quoter.vendors import ezdxf as _ezdxf_vendor


NO_TEXT_ROWS_MESSAGE = "No text found before filtering; use --dump-ents to inspect."


class NoTextRowsError(RuntimeError):
    """Raised when neither ACAD nor text pipelines yield usable rows."""

    def __init__(self, message: str = NO_TEXT_ROWS_MESSAGE) -> None:
        super().__init__(message)


TransformMatrix = tuple[float, float, float, float, float, float]


@dataclass(slots=True)
class FlattenedEntity:
    """Metadata wrapper for entities yielded by :func:`flatten_entities`."""

    entity: Any
    transform: TransformMatrix
    from_block: bool
    block_name: str | None
    block_stack: tuple[str, ...]
    depth: int
    layer: str
    layer_upper: str
    effective_layer: str
    effective_layer_upper: str


_IDENTITY_TRANSFORM: TransformMatrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


_OPS_SEGMENT_SPLIT_RE = re.compile(r"[;•]+")
_FALLBACK_ROW_START_RE = re.compile(r"^\s*\(?(\d+)\)?\s+")
_TAP_WORD_TOKEN_RE = re.compile(r"\bTAP\b", re.IGNORECASE)
_TAP_THREAD_TOKEN_RE = re.compile(
    r"(?:#\s*\d+\s*-\s*\d+|\b\d+\s*/\s*\d+\s*-\s*\d+\b)",
    re.IGNORECASE,
)
_NPT_TOKEN_RE = re.compile(r"\bN\.?P\.?T\.?\b", re.IGNORECASE)
_PIPE_TAP_TOKEN_RE = re.compile(r"\bPIPE\s+TAP\b", re.IGNORECASE)
_COUNTERBORE_TOKEN_RE = re.compile(
    r"\b(?:C['’]?\s*BORE|CBORE|COUNTER\s*BORE)\b",
    re.IGNORECASE,
)
_COUNTERSINK_TOKEN_RE = re.compile(r"\b(?:C['’]?\s*SINK|CSK|COUNTERSINK)\b", re.IGNORECASE)
_COUNTERDRILL_TOKEN_RE = re.compile(
    r"\b(?:C['’]?\s*DRILL|COUNTER\s*DRILL|CTR\s*DRILL|C['’]DRILL|CENTER\s*DRILL)\b",
    re.IGNORECASE,
)
_JIG_GRIND_TOKEN_RE = re.compile(
    r"\b(?:JIG\s*GRIND|JIG[-\s]?GROUND|JG)\b",
    re.IGNORECASE,
)
_SPOT_TOKEN_RE = re.compile(r"\bSPOT\b", re.IGNORECASE)
_DRILL_TOKEN_RE = re.compile(r"\bDRILL(?:\s+THRU)?\b", re.IGNORECASE)
_DRILL_SIZE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"#\s*(\d+)", re.IGNORECASE),
    re.compile(r"NO\.?\s*(\d+)", re.IGNORECASE),
    re.compile(r"LETTER\s+([A-Z])", re.IGNORECASE),
    re.compile(r'"([A-Z])"'),
    re.compile(r"\bR\s*[.#]?\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE),
    re.compile(r"R\s*\(([^)]+)\)", re.IGNORECASE),
    re.compile(r"[\u00D8\u2300\u2A00⌀]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:IN\.?|MM|\"|DIA|DIAM)\b", re.IGNORECASE),
    re.compile(r"\b(\.[0-9]+)\b"),
    re.compile(r"\b([0-9]+\s*/\s*[0-9]+)\b"),
    re.compile(r"\b([A-Z])\b(?=[^A-Z0-9]*(?:DRILL|HOLE))", re.IGNORECASE),
    re.compile(r"\(([^(]*?([0-9]+(?:\.[0-9]+)?)[^)]*)\)", re.IGNORECASE),
)
_OPS_MANIFEST_KEYS = (
    "tap",
    "counterbore",
    "counterdrill",
    "csink",
    "drill",
    "jig_grind",
    "spot",
    "npt",
    "unknown",
)

_AUTHORITATIVE_TABLE_SOURCES = {"acad_table", "text_table", "text_fallback"}


def _table_source_is_authoritative(source: Any, row_count: int) -> bool:
    if row_count < 8:
        return False
    try:
        source_text = str(source or "")
    except Exception:
        source_text = ""
    return source_text.lower() in _AUTHORITATIVE_TABLE_SOURCES


def _matrix_multiply(a: TransformMatrix, b: TransformMatrix) -> TransformMatrix:
    """Return the matrix product ``a @ b`` for affine 2D transforms."""

    a00, a01, a02, a10, a11, a12 = a
    b00, b01, b02, b10, b11, b12 = b
    return (
        a00 * b00 + a01 * b10,
        a00 * b01 + a01 * b11,
        a00 * b02 + a01 * b12 + a02,
        a10 * b00 + a11 * b10,
        a10 * b01 + a11 * b11,
        a10 * b02 + a11 * b12 + a12,
    )


def _matrix_chain(*matrices: TransformMatrix) -> TransformMatrix:
    """Compose ``matrices`` left→right into a single transform."""

    transform = _IDENTITY_TRANSFORM
    for matrix in matrices:
        transform = _matrix_multiply(transform, matrix)
    return transform


def _matrix_translate(tx: float, ty: float) -> TransformMatrix:
    return (1.0, 0.0, float(tx), 0.0, 1.0, float(ty))


def _matrix_scale(sx: float, sy: float) -> TransformMatrix:
    return (float(sx), 0.0, 0.0, 0.0, float(sy), 0.0)


def _matrix_rotate(rad: float) -> TransformMatrix:
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0)


def _apply_transform_point(
    matrix: TransformMatrix, point: tuple[float | None, float | None]
) -> tuple[float | None, float | None]:
    x_val, y_val = point
    if not isinstance(x_val, (int, float)) or not isinstance(y_val, (int, float)):
        return (None, None)
    x = float(x_val)
    y = float(y_val)
    m00, m01, m02, m10, m11, m12 = matrix
    return (m00 * x + m01 * y + m02, m10 * x + m11 * y + m12)


def _transform_scale_hint(matrix: TransformMatrix) -> float:
    """Return an approximate scale factor encoded by ``matrix``."""

    m00, m01, _m02, m10, m11, _m12 = matrix
    scale_x = math.hypot(m00, m10)
    scale_y = math.hypot(m01, m11)
    candidates = [value for value in (scale_x, scale_y) if value > 0]
    if not candidates:
        return 1.0
    try:
        return float(statistics.median(candidates))
    except Exception:
        return candidates[0]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _point2d(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if hasattr(value, "xyz"):
        try:
            x_val, y_val, _ = value.xyz
            return (float(x_val), float(y_val))
        except Exception:
            return None
    for accessor in (("x", "y"), (0, 1)):
        try:
            x_candidate = getattr(value, accessor[0]) if isinstance(accessor[0], str) else value[accessor[0]]
        except Exception:
            x_candidate = None
        try:
            y_candidate = getattr(value, accessor[1]) if isinstance(accessor[1], str) else value[accessor[1]]
        except Exception:
            y_candidate = None
        if x_candidate is None and y_candidate is None:
            continue
        try:
            x_float = float(x_candidate) if x_candidate is not None else 0.0
            y_float = float(y_candidate) if y_candidate is not None else 0.0
        except Exception:
            continue
        return (x_float, y_float)
    try:
        sequence = list(value)
    except Exception:
        return None
    if len(sequence) >= 2:
        try:
            return (float(sequence[0]), float(sequence[1]))
        except Exception:
            return None
    return None


def _layer_name_is_excluded(layer_name: str | None) -> bool:
    if not layer_name:
        return False
    try:
        normalized = str(layer_name).strip()
    except Exception:
        normalized = ""
    if not normalized:
        return False
    upper_name = normalized.upper()
    for prefix in _GEO_CIRCLE_LAYER_EXCLUDE_PREFIXES:
        if upper_name.startswith(prefix):
            return True
    for pattern in _GEO_CIRCLE_LAYER_EXCLUDE_GLOBS:
        if fnmatchcase(upper_name, pattern):
            return True
    if _GEO_CIRCLE_LAYER_BLACKLIST_RE.search(normalized):
        return True
    return False


def _point3d(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if hasattr(value, "xyz"):
        try:
            x_val, y_val, z_val = value.xyz
            return (float(x_val), float(y_val), float(z_val))
        except Exception:
            return None
    for accessor in (("x", "y", "z"), (0, 1, 2)):
        coords: list[float] = []
        missing = True
        for key in accessor:
            try:
                candidate = getattr(value, key) if isinstance(key, str) else value[key]
            except Exception:
                candidate = None
            if candidate is None:
                coords.append(0.0)
            else:
                missing = False
                try:
                    coords.append(float(candidate))
                except Exception:
                    coords = []
                    break
        if coords and not missing:
            try:
                x_val, y_val, z_val = coords
                return (x_val, y_val, z_val)
            except Exception:
                continue
    return None


def _is_positive_z_normal(candidate: Any, tol: float = 1e-6) -> bool:
    vector = _point3d(candidate)
    if vector is None:
        return True
    nx, ny, nz = vector
    magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        return False
    nx /= magnitude
    ny /= magnitude
    nz /= magnitude
    if nz <= tol:
        return False
    if abs(nx) > tol or abs(ny) > tol:
        return False
    return True


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


def flatten_entities(
    layout: Any,
    depth: int = 5,
    *,
    include_block: Callable[[str | None], bool] | None = None,
) -> Iterable[FlattenedEntity]:
    """Yield entities from ``layout`` with accumulated block transforms.

    Args:
        include_block: Optional predicate invoked for each ``INSERT`` block
            reference. When provided, recursion into the block's entities is
            skipped unless the predicate returns ``True`` for the block name.
    """

    if layout is None:
        return

    max_depth = max(int(depth), 0)
    doc = getattr(layout, "doc", None)

    def _iter_container(container: Any) -> list[Any]:
        if container is None:
            return []
        try:
            return list(container)
        except Exception:
            result: list[Any] = []
            try:
                iterator = iter(container)
            except Exception:
                query = getattr(container, "query", None)
                if callable(query):
                    for spec in ("TEXT, MTEXT, RTEXT, MLEADER, INSERT", "*"):
                        try:
                            result = list(query(spec))
                        except Exception:
                            continue
                        if result:
                            break
                return result
            for item in iterator:
                result.append(item)
            return result

    def _extract_block_name(entity: Any) -> str | None:
        dxf_obj = getattr(entity, "dxf", None)
        name = None
        if dxf_obj is not None:
            name = getattr(dxf_obj, "name", None)
        if name is None:
            name = getattr(entity, "name", None)
        if not isinstance(name, str):
            return None
        text = name.strip()
        return text or None

    def _resolve_block_layout(entity: Any) -> Any | None:
        block_layout = None
        block_attr = getattr(entity, "block", None)
        if callable(block_attr):
            try:
                block_layout = block_attr()
            except Exception:
                block_layout = None
        if block_layout is not None:
            return block_layout
        name = _extract_block_name(entity)
        if not name:
            return None
        candidates: list[Any] = []
        for source in (getattr(entity, "doc", None), doc):
            if source is None:
                continue
            blocks = getattr(source, "blocks", None)
            if blocks is None:
                continue
            get_block = getattr(blocks, "get", None)
            if callable(get_block):
                try:
                    layout_obj = get_block(name)
                except Exception:
                    layout_obj = None
                if layout_obj is not None:
                    candidates.append(layout_obj)
        return candidates[0] if candidates else None

    def _block_base_point(block_layout: Any) -> tuple[float, float] | None:
        if block_layout is None:
            return None
        for source in (
            getattr(block_layout, "block", None),
            getattr(block_layout, "dxf", None),
            block_layout,
        ):
            if source is None:
                continue
            for attr in ("base_point", "insert", "origin"):
                coords = _point2d(getattr(source, attr, None))
                if coords is not None:
                    return coords
        return None

    def _insert_local_transform(entity: Any, block_layout: Any) -> TransformMatrix:
        dxf_obj = getattr(entity, "dxf", None)
        insert_point = None
        if dxf_obj is not None:
            insert_point = getattr(dxf_obj, "insert", None)
        if insert_point is None:
            insert_point = getattr(entity, "insert", None)
        insert_xy = _point2d(insert_point) or (0.0, 0.0)

        rotation_deg = 0.0
        if dxf_obj is not None:
            rotation_deg = _coerce_float(getattr(dxf_obj, "rotation", 0.0), 0.0)
        else:
            rotation_deg = _coerce_float(getattr(entity, "rotation", 0.0), 0.0)

        scale_uniform = 1.0
        if dxf_obj is not None:
            scale_uniform = _coerce_float(getattr(dxf_obj, "scale", 1.0), 1.0)
        else:
            scale_uniform = _coerce_float(getattr(entity, "scale", 1.0), 1.0)
        if scale_uniform == 0.0:
            scale_uniform = 1.0

        sx = scale_uniform
        sy = scale_uniform
        if dxf_obj is not None:
            sx_raw = getattr(dxf_obj, "xscale", None)
            sy_raw = getattr(dxf_obj, "yscale", None)
        else:
            sx_raw = getattr(entity, "xscale", None)
            sy_raw = getattr(entity, "yscale", None)
        if sx_raw not in (None, 0):
            sx *= _coerce_float(sx_raw, 1.0)
        if sy_raw not in (None, 0):
            sy *= _coerce_float(sy_raw, sx)
        if sx == 0.0:
            sx = 1.0
        if sy == 0.0:
            sy = sx

        base_point = _block_base_point(block_layout)
        if base_point is None:
            base_point = _point2d(getattr(dxf_obj, "block_base_point", None))

        transform_local = _matrix_chain(
            _matrix_translate(insert_xy[0], insert_xy[1]),
            _matrix_rotate(math.radians(rotation_deg)),
            _matrix_scale(sx, sy),
        )
        if base_point is not None:
            transform_local = _matrix_multiply(
                transform_local, _matrix_translate(-base_point[0], -base_point[1])
            )
        return transform_local

    def _flatten(
        entity: Any,
        transform: TransformMatrix,
        block_stack: tuple[str, ...],
        parent_effective_layer: str | None,
        level: int,
    ) -> Iterable[FlattenedEntity]:
        layer_name = _entity_layer(entity)
        layer_upper = layer_name.upper() if layer_name else ""
        effective_layer = layer_name or ""
        effective_layer_upper = layer_upper
        if not effective_layer_upper or effective_layer_upper == "0":
            candidate = parent_effective_layer or layer_name or ""
            effective_layer = candidate
            effective_layer_upper = candidate.upper() if candidate else ""

        flattened = FlattenedEntity(
            entity=entity,
            transform=transform,
            from_block=bool(block_stack),
            block_name=block_stack[-1] if block_stack else None,
            block_stack=block_stack,
            depth=level,
            layer=layer_name,
            layer_upper=layer_upper,
            effective_layer=effective_layer,
            effective_layer_upper=effective_layer_upper,
        )
        yield flattened

        dxftype = None
        try:
            dxftype = entity.dxftype()
        except Exception:
            dxftype = None
        kind = str(dxftype or "").upper()
        if kind != "INSERT" or level >= max_depth:
            return

        block_name = _extract_block_name(entity)
        if block_name and block_name in block_stack:
            return

        if include_block is not None:
            try:
                allow_block = bool(include_block(block_name))
            except Exception:
                allow_block = True
        else:
            allow_block = True

        if not allow_block:
            return

        block_layout = _resolve_block_layout(entity)
        local_transform = _insert_local_transform(entity, block_layout)
        child_transform = _matrix_multiply(transform, local_transform)
        child_stack = block_stack + (block_name,) if block_name else block_stack
        child_parent_layer = effective_layer or parent_effective_layer

        nested_entities: list[Any] = []
        use_local_transform = True
        if block_layout is not None:
            entity_space = getattr(block_layout, "entity_space", None)
            nested_entities = _iter_container(entity_space if entity_space is not None else block_layout)
        if not nested_entities:
            try:
                nested_entities = list(entity.virtual_entities())
            except Exception:
                nested_entities = []
            else:
                use_local_transform = False

        for child in nested_entities:
            next_transform = child_transform if use_local_transform else transform
            yield from _flatten(child, next_transform, child_stack, child_parent_layer, level + 1)

        for attribute in _iter_insert_attributes(entity):
            yield from _flatten(attribute, child_transform, child_stack, child_parent_layer, level + 1)

    base_entities = _iter_container(layout)
    for entity in base_entities:
        yield from _flatten(entity, _IDENTITY_TRANSFORM, tuple(), None, 0)

_HAS_ODAFC = bool(getattr(geometry, "HAS_ODAFC", False))


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _env_float(*names: str, default: float) -> float:
    for name in names:
        if not name:
            continue
        try:
            value = os.environ.get(name)
        except Exception:
            value = None
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except Exception:
            continue
    return float(default)


_DEFAULT_LAYER_ALLOWLIST = frozenset({"BALLOON"})
_GEO_EXCLUDE_LAYERS_DEFAULT = r"^(AM_BOR|DEFPOINTS|PAPER)$"
DEFAULT_TEXT_LAYER_EXCLUDE_REGEX: tuple[str, ...] = (
    _GEO_EXCLUDE_LAYERS_DEFAULT,
)

_GEOM_BLOCK_EXCLUDE_RE = re.compile(r"^(TITLE|BORDER|CHART|FRAME|AM_.*)$", re.IGNORECASE)
_GEO_CIRCLE_LAYER_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "AM_BOR",
    "BORDER",
    "TITLE",
    "FRAME",
    "CHART",
    "SHEET",
    "NOTES",
    "NOTE",
    "DIM",
    "ANNO",
    "CENTER",
    "CNTR",
    "SYMBOL",
    "SYMB",
    "DEFPOINTS",
    "PAPER",
)
_GEO_CIRCLE_LAYER_EXCLUDE_GLOBS: tuple[str, ...] = ("SHEET*",)
_GEO_CIRCLE_LAYER_BLACKLIST_RE = re.compile(
    r"^(AM_BOR|BORDER|TITLE|FRAME|CHART|SHEET|NOTES?|DIM|CENTER|CNTR|SY(M|MB)OL|DEFPOINTS|PAPER)$",
    re.IGNORECASE,
)

_GEO_STRICT_ANCHOR = _env_flag("GEO_STRICT_ANCHOR")
try:
    _GEO_H_ANCHOR_MIN = float(os.environ.get("GEO_H_ANCHOR_MIN", "0.10") or 0.10)
except Exception:
    _GEO_H_ANCHOR_MIN = 0.10
_GEO_H_ANCHOR_MIN = max(_GEO_H_ANCHOR_MIN, 0.0)
_GEO_H_ANCHOR_HARD_MIN = 0.04

_GEO_DIA_MIN_IN = max(
    _env_float("GEO_DIA_MIN_IN", "GEO_CIRCLE_DIAM_MIN_IN", default=0.04),
    0.0,
)
_GEO_DIA_MAX_IN = _env_float("GEO_DIA_MAX_IN", "GEO_CIRCLE_DIAM_MAX_IN", default=2.0)
if _GEO_DIA_MAX_IN <= 0.0:
    _GEO_DIA_MAX_IN = 2.0
if _GEO_DIA_MAX_IN < _GEO_DIA_MIN_IN:
    _GEO_DIA_MAX_IN = _GEO_DIA_MIN_IN
_GEO_DRILL_RADIUS_MIN_IN = max(_env_float("GEO_DRILL_RADIUS_MIN_IN", default=0.04), 0.0)
_GEO_DRILL_RADIUS_MAX_IN = _env_float("GEO_DRILL_RADIUS_MAX_IN", default=2.0)
if _GEO_DRILL_RADIUS_MAX_IN <= 0.0:
    _GEO_DRILL_RADIUS_MAX_IN = 2.0
if (
    _GEO_DRILL_RADIUS_MAX_IN
    and _GEO_DRILL_RADIUS_MIN_IN
    and _GEO_DRILL_RADIUS_MAX_IN < _GEO_DRILL_RADIUS_MIN_IN
):
    _GEO_DRILL_RADIUS_MAX_IN = _GEO_DRILL_RADIUS_MIN_IN
_GEO_CIRCLE_Z_ABS_MAX = 1e-6
_GEO_CIRCLE_DEDUP_DIGITS = 3
_GEO_CIRCLE_CENTER_GROUP_DIGITS = 3
_GEO_BBOX_MARGIN_IN = 0.25


@dataclass(slots=True)
class ExtractionState:
    published: bool = False
    anchor_authoritative: bool = False
    publish_logged: bool = False

    def mark_published(self) -> bool:
        """Mark the state as published, returning ``True`` if this is the first publish."""

        if self.publish_logged:
            return False
        self.publish_logged = True
        self.published = True
        return True

_PREFERRED_BLOCK_NAME_RE = re.compile(r"HOLE.*(?:CHART|TABLE)", re.IGNORECASE)
_FOLLOW_SHEET_DIRECTIVE_RE = re.compile(
    r"SEE\s+(?:SHEET|SHT)\s+(?P<target>[A-Z0-9]+)", re.IGNORECASE
)

_LAST_ACAD_TABLE_SCAN: dict[str, Any] | None = None
_TRACE_ACAD = False
_LAST_DXF_FALLBACK_INFO: dict[str, Any] | None = None


@dataclass(slots=True)
class TableHit:
    """Container describing an AutoCAD table discovered in the drawing."""

    owner: str
    layer: str
    handle: str
    rows: int
    cols: int
    dxftype: str
    table: Any
    source: str
    name: str | None
    scan_index: int
    transform: TransformMatrix = _IDENTITY_TRANSFORM
    block_stack: tuple[str, ...] = ()
    from_block: bool = False


def _get_table_dimension(entity: Any, names: tuple[str, ...]) -> int | None:
    dxf_obj = getattr(entity, "dxf", None)
    for name in names:
        for source in (entity, dxf_obj):
            if source is None:
                continue
            value = getattr(source, name, None)
            if value is None:
                continue
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            try:
                return int(float(value))
            except Exception:
                continue
    return None


def _entity_layer(entity: Any) -> str:
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


def _extract_layer(entity: Any) -> str:
    """Return the best-effort layer name for ``entity``."""

    return _entity_layer(entity)


def _entity_handle(entity: Any) -> str:
    dxf_obj = getattr(entity, "dxf", None)
    for source in (entity, dxf_obj):
        if source is None:
            continue
        handle = getattr(source, "handle", None)
        if handle is None:
            continue
        try:
            text = str(handle).strip()
        except Exception:
            continue
        if text:
            return text
    return ""



def _entity_owner_handle(entity: Any) -> str:
    dxf_obj = getattr(entity, "dxf", None)
    for source in (entity, dxf_obj):
        if source is None:
            continue
        owner = getattr(source, "owner", None)
        if owner is None:
            continue
        try:
            text = str(owner).strip()
        except Exception:
            continue
        if text:
            return text
    return ""



def _normalize_oda_version(version: str | None) -> str | None:
    if version is None:
        return None
    try:
        normalized = str(version).strip().upper()
    except Exception:
        return None
    mapping = {
        "ACAD2000": "ACAD2000",
        "2000": "ACAD2000",
        "ACAD2004": "ACAD2004",
        "2004": "ACAD2004",
        "ACAD2007": "ACAD2007",
        "2007": "ACAD2007",
        "ACAD2013": "ACAD2013",
        "2013": "ACAD2013",
        "ACAD2018": "ACAD2018",
        "2018": "ACAD2018",
    }
    return mapping.get(normalized, normalized)


def _normalize_layout_key(name: str | None) -> str:
    if name is None:
        return ""
    try:
        text = str(name)
    except Exception:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized.upper()


def _parse_layout_filter(
    layouts_arg: Mapping[str, Any] | Iterable[str] | str | None,
) -> tuple[bool, list[str]]:
    allow_all = True
    patterns: list[str] = []
    if isinstance(layouts_arg, Mapping):
        allow_all = bool(layouts_arg.get("all_layouts", True))
        raw_patterns = layouts_arg.get("patterns")
        if isinstance(raw_patterns, str):
            patterns = [raw_patterns]
        elif isinstance(raw_patterns, Iterable) and not isinstance(
            raw_patterns, (str, bytes, bytearray)
        ):
            patterns = [str(value) for value in raw_patterns if isinstance(value, str)]
        else:
            patterns = []
    elif isinstance(layouts_arg, str):
        allow_all = False
        patterns = [layouts_arg]
    elif isinstance(layouts_arg, Iterable) and not isinstance(
        layouts_arg, (str, bytes, bytearray)
    ):
        allow_all = False
        patterns = [str(value) for value in layouts_arg if isinstance(value, str)]
    cleaned: list[str] = []
    for pattern in patterns:
        text = pattern.strip()
        if text:
            cleaned.append(text)
    if not cleaned and not allow_all:
        allow_all = False
    return (allow_all, cleaned)


def iter_layouts(
    doc: Any,
    layouts_arg: Mapping[str, Any] | Iterable[str] | str | None,
    *,
    log: bool = True,
) -> list[tuple[str, Any]]:
    allow_all, pattern_texts = _parse_layout_filter(layouts_arg)
    layouts: list[tuple[str, Any]] = []
    if doc is None:
        if log:
            print("[TEXT-SCAN] layouts=<none>")
        raise RuntimeError("No DXF/DWG document loaded")

    modelspace = getattr(doc, "modelspace", None)
    if callable(modelspace):
        try:
            space = modelspace()
        except Exception:
            space = None
        if space is not None:
            layouts.append(("Model", space))

    layouts_manager = getattr(doc, "layouts", None)
    names: list[Any] = []
    if layouts_manager is not None:
        raw_names = getattr(layouts_manager, "names", None)
        try:
            if callable(raw_names):
                names_iter = raw_names()  # type: ignore[call-arg]
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
            layouts.append((name, layout_obj))

    # Deduplicate layouts by normalized name while preserving order.
    unique_layouts: list[tuple[str, Any]] = []
    seen_names: set[str] = set()
    for name, layout_obj in layouts:
        key = _normalize_layout_key(name)
        if key in seen_names:
            continue
        seen_names.add(key)
        unique_layouts.append((name, layout_obj))
    layouts = unique_layouts

    filtered_layouts = layouts
    if pattern_texts:
        compiled_patterns: list[re.Pattern[str]] = []
        for text in pattern_texts:
            try:
                compiled_patterns.append(re.compile(text, re.IGNORECASE))
            except re.error as exc:
                print(f"[TEXT-SCAN] layout regex error pattern={text!r} err={exc}")
        if compiled_patterns:
            filtered_layouts = []
            for name, layout_obj in layouts:
                label = str(name or "")
                if any(pattern.search(label) for pattern in compiled_patterns):
                    filtered_layouts.append((name, layout_obj))
        else:
            filtered_layouts = []
    elif not allow_all:
        filtered_layouts = []

    names_display = [str(name or "").strip() or "-" for name, _ in filtered_layouts]
    if log:
        display = ", ".join(names_display) if names_display else "<none>"
        print(f"[TEXT-SCAN] layouts={display}")

    if not filtered_layouts:
        raise RuntimeError("No layouts matched the requested filters")

    return filtered_layouts


_DEFAULT_HEIGHT_ATTRS: tuple[str, ...] = ("char_height", "text_height", "height", "size")
_DEFAULT_ROTATION_ATTRS: tuple[str, ...] = ("rotation", "rot")
_DEFAULT_INSERT_ATTRS: tuple[str, ...] = (
    "insert",
    "alignment_point",
    "location",
    "base_point",
    "defpoint",
    "center",
    "position",
)


def _first_text_value(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        try:
            text = str(value)
        except Exception:
            continue
        if text:
            return text
    return None


def _extract_numeric_attr(entity: Any, names: Sequence[str]) -> float | None:
    if entity is None:
        return None
    dxf_obj = getattr(entity, "dxf", None)
    sources = (entity, dxf_obj)
    for source in sources:
        if source is None:
            continue
        for name in names:
            if not hasattr(source, name):
                continue
            value = getattr(source, name, None)
            if value is None:
                continue
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            try:
                number = float(value)
            except Exception:
                continue
            if math.isfinite(number):
                return number
    return None


def _extract_insert_point(
    entity: Any, transform: TransformMatrix, attrs: Sequence[str] = _DEFAULT_INSERT_ATTRS
) -> tuple[float, float] | None:
    if entity is None:
        return None
    dxf_obj = getattr(entity, "dxf", None)
    sources = (entity, dxf_obj)
    for source in sources:
        if source is None:
            continue
        for name in attrs:
            if not hasattr(source, name):
                continue
            point = _point2d(getattr(source, name, None))
            if point is None:
                continue
            world = _apply_transform_point(transform, point)
            if world[0] is None or world[1] is None:
                continue
            try:
                return (float(world[0]), float(world[1]))
            except Exception:
                continue
    return None


def _normalize_text_output(value: Any) -> str:
    try:
        text = str(value)
    except Exception:
        text = ""
    if not text:
        return ""
    candidate = text.replace("\r\n", "\n").replace("\r", "\n")
    candidate = candidate.replace("\\P", "\n")
    return candidate.strip()


def _iter_text_layout_spaces(doc: Any, include_paperspace: bool) -> list[tuple[str, Any]]:
    spaces: list[tuple[str, Any]] = []
    if doc is None:
        return spaces

    modelspace = getattr(doc, "modelspace", None)
    if callable(modelspace):
        try:
            ms = modelspace()
        except Exception:
            ms = None
        if ms is not None:
            spaces.append(("Model", ms))

    if not include_paperspace:
        return spaces

    layouts_manager = getattr(doc, "layouts", None)
    if layouts_manager is None:
        return spaces

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
        is_paper = False
        layout_dxf = getattr(layout_obj, "dxf", None)
        if layout_dxf is not None:
            try:
                is_paper = bool(int(getattr(layout_dxf, "paperspace", 0)))
            except Exception:
                is_paper = False
        if not is_paper:
            continue
        block_obj = None
        block_method = getattr(layout_obj, "block", None)
        if callable(block_method):
            try:
                block_obj = block_method()
            except Exception:
                block_obj = None
        if block_obj is None:
            block_obj = getattr(layout_obj, "entity_space", None) or layout_obj
        spaces.append((name, block_obj))

    unique: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for label, space in spaces:
        key = _normalize_layout_key(label)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, space))
    return unique


def _collect_table_text(
    flattened: FlattenedEntity, layout_name: str, *, etype_override: str
) -> list[dict[str, Any]]:
    entity = flattened.entity
    try:
        virtual_entities = list(entity.virtual_entities())
    except Exception:
        virtual_entities = []

    records: list[dict[str, Any]] = []
    for child in virtual_entities:
        try:
            child_type = str(child.dxftype()).upper()
        except Exception:
            child_type = ""
        if child_type not in {"TEXT", "MTEXT"}:
            continue
        child_layer = _entity_layer(child) or flattened.effective_layer or flattened.layer
        effective_layer = child_layer or flattened.effective_layer or flattened.layer or ""
        child_flattened = FlattenedEntity(
            entity=child,
            transform=flattened.transform,
            from_block=flattened.from_block,
            block_name=flattened.block_name,
            block_stack=flattened.block_stack,
            depth=flattened.depth + 1,
            layer=child_layer or "",
            layer_upper=(child_layer or "").upper(),
            effective_layer=effective_layer,
            effective_layer_upper=effective_layer.upper(),
        )
        records.extend(
            _collect_text_from_flattened(
                child_flattened,
                layout_name,
                etype_override=etype_override,
            )
        )

    if records:
        return records

    rows = _get_table_dimension(entity, ("n_rows", "row_count", "nrows", "rows"))
    cols = _get_table_dimension(entity, ("n_cols", "col_count", "ncols", "columns"))
    if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:
        return records

    for row in range(rows):
        for col in range(cols):
            cell_text = _cell_text(entity, row, col)
            if not cell_text:
                continue
            record = _build_text_record(
                flattened,
                layout_name,
                etype=etype_override,
                text=cell_text,
                raw=cell_text,
                height_entity=None,
                rotation_entity=None,
                insert_entity=None,
            )
            if record is not None:
                records.append(record)
    return records


def _extract_entity_text_payload(
    flattened: FlattenedEntity, canonical_kind: str
) -> dict[str, Any] | None:
    entity = flattened.entity
    dxf_obj = getattr(entity, "dxf", None)

    if canonical_kind in {"TEXT", "ATTRIB", "ATTDEF"}:
        raw = _first_text_value(
            getattr(dxf_obj, "text", None),
            getattr(dxf_obj, "value", None),
            getattr(entity, "text", None),
            getattr(entity, "value", None),
            getattr(entity, "default", None),
        )
        if raw is None:
            return None
        return {
            "text": raw,
            "raw": raw,
            "height_entity": entity,
            "rotation_entity": entity,
            "insert_entity": entity,
        }

    if canonical_kind == "MTEXT":
        plain_text = None
        plain_method = getattr(entity, "plain_text", None)
        if callable(plain_method):
            try:
                plain_text = plain_method()
            except Exception:
                plain_text = None
        raw = _first_text_value(
            getattr(entity, "text", None),
            getattr(entity, "raw_text", None),
            getattr(dxf_obj, "text", None),
            getattr(dxf_obj, "content", None),
        )
        text_value = plain_text or raw
        if text_value is None:
            return None
        return {
            "text": text_value,
            "raw": raw if raw is not None else text_value,
            "height_entity": entity,
            "rotation_entity": entity,
            "insert_entity": entity,
        }

    if canonical_kind == "MLEADER":
        context = getattr(entity, "context", None)
        mtext = getattr(context, "mtext", None) if context is not None else None
        if mtext is None:
            return None
        plain_text = None
        plain_method = getattr(mtext, "plain_text", None)
        if callable(plain_method):
            try:
                plain_text = plain_method()
            except Exception:
                plain_text = None
        raw = _first_text_value(
            getattr(mtext, "text", None),
            getattr(getattr(mtext, "dxf", None), "text", None),
            getattr(mtext, "raw_text", None),
        )
        text_value = plain_text or raw
        if text_value is None:
            return None
        return {
            "text": text_value,
            "raw": raw if raw is not None else text_value,
            "height_entity": mtext,
            "rotation_entity": mtext,
            "insert_entity": mtext,
        }

    if canonical_kind == "DIM":
        raw = _first_text_value(
            getattr(dxf_obj, "text", None),
            getattr(entity, "text", None),
        )
        if raw is None:
            return None
        raw_stripped = str(raw).strip()
        if not raw_stripped or raw_stripped == "<>":
            return None
        return {
            "text": raw,
            "raw": raw,
            "height_entity": entity,
            "rotation_entity": entity,
            "insert_entity": entity,
        }

    return None


def _build_text_record(
    flattened: FlattenedEntity,
    layout_name: str,
    *,
    etype: str,
    text: Any,
    raw: Any = None,
    height_entity: Any | None = None,
    rotation_entity: Any | None = None,
    insert_entity: Any | None = None,
    layer: str | None = None,
) -> dict[str, Any] | None:
    text_value = _normalize_text_output(text)
    if not text_value:
        return None

    raw_value = None
    if raw is not None:
        try:
            raw_value = str(raw)
        except Exception:
            raw_value = repr(raw)

    height_value = None
    if height_entity is not None:
        height_raw = _extract_numeric_attr(height_entity, _DEFAULT_HEIGHT_ATTRS)
        if height_raw is None and height_entity is not flattened.entity:
            height_raw = _extract_numeric_attr(flattened.entity, _DEFAULT_HEIGHT_ATTRS)
        if height_raw is not None:
            try:
                height_value = abs(float(height_raw)) * _transform_scale_hint(flattened.transform)
            except Exception:
                height_value = None

    transform_angle = math.degrees(math.atan2(flattened.transform[3], flattened.transform[0]))

    rotation_value = None
    if rotation_entity is not None:
        rotation_raw = _extract_numeric_attr(rotation_entity, _DEFAULT_ROTATION_ATTRS)
        if rotation_raw is None and rotation_entity is not flattened.entity:
            rotation_raw = _extract_numeric_attr(flattened.entity, _DEFAULT_ROTATION_ATTRS)
        if rotation_raw is not None:
            try:
                rotation_value = float(rotation_raw) + transform_angle
            except Exception:
                rotation_value = float(rotation_raw)

    if rotation_value is None and transform_angle:
        rotation_value = transform_angle

    insert_value = None
    if insert_entity is not None:
        insert_value = _extract_insert_point(insert_entity, flattened.transform)
        if insert_value is None and insert_entity is not flattened.entity:
            insert_value = _extract_insert_point(flattened.entity, flattened.transform)

    layer_name = layer if layer is not None else flattened.effective_layer or flattened.layer or ""
    layout_label = str(layout_name or "").strip() or "-"

    return {
        "text": text_value,
        "raw": raw_value,
        "etype": etype,
        "layout": layout_label,
        "layer": layer_name,
        "height": height_value,
        "rotation": rotation_value,
        "insert": insert_value,
        "block_path": tuple(name for name in flattened.block_stack if name),
    }


def _collect_text_from_flattened(
    flattened: FlattenedEntity,
    layout_name: str,
    *,
    etype_override: str | None = None,
) -> list[dict[str, Any]]:
    entity = flattened.entity
    try:
        dxftype = str(entity.dxftype()).upper()
    except Exception:
        dxftype = ""

    if dxftype == "ACAD_TABLE":
        override = etype_override or "TABLECELL"
        return _collect_table_text(flattened, layout_name, etype_override=override)

    canonical = dxftype
    if canonical == "DIMENSION":
        canonical = "DIM"

    if canonical not in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "MLEADER", "DIM"}:
        return []

    payload = _extract_entity_text_payload(flattened, canonical)
    if payload is None:
        return []

    record = _build_text_record(
        flattened,
        layout_name,
        etype=etype_override or canonical,
        text=payload.get("text"),
        raw=payload.get("raw"),
        height_entity=payload.get("height_entity"),
        rotation_entity=payload.get("rotation_entity"),
        insert_entity=payload.get("insert_entity"),
    )
    return [record] if record is not None else []


def _compile_layer_patterns(patterns: Any) -> list[re.Pattern[str]]:
    if patterns in (None, ""):
        return []
    if isinstance(patterns, str):
        candidates = [patterns]
    elif isinstance(patterns, Iterable) and not isinstance(patterns, (str, bytes, bytearray)):
        candidates = [str(item) for item in patterns if str(item).strip()]
    else:
        candidates = [str(patterns)]
    compiled: list[re.Pattern[str]] = []
    for text in candidates:
        cleaned = text.strip()
        if not cleaned:
            continue
        try:
            compiled.append(re.compile(cleaned, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def collect_all_text(
    doc: Any,
    *,
    include_blocks: bool = True,
    include_paperspace: bool = True,
    min_height: float | None = None,
    layers_include: Any = None,
    layers_exclude: Any = None,
) -> list[dict[str, Any]]:
    layouts = _iter_text_layout_spaces(doc, include_paperspace)
    depth = _MAX_INSERT_DEPTH if include_blocks else 0
    records: list[dict[str, Any]] = []

    for layout_name, layout in layouts:
        if layout is None:
            continue
        try:
            flattened_iter = flatten_entities(layout, depth=depth)
        except Exception:
            flattened_iter = []
        for flattened in flattened_iter:
            records.extend(_collect_text_from_flattened(flattened, layout_name))

    include_patterns = _compile_layer_patterns(layers_include)
    exclude_patterns = _compile_layer_patterns(layers_exclude)
    min_height_value = None
    if min_height is not None:
        try:
            min_height_value = float(min_height)
        except Exception:
            min_height_value = None
        else:
            if not math.isfinite(min_height_value):
                min_height_value = None

    if not records:
        return []

    filtered: list[dict[str, Any]] = []
    for entry in records:
        layer_name = str(entry.get("layer") or "")
        if include_patterns and not any(pattern.search(layer_name) for pattern in include_patterns):
            continue
        if exclude_patterns and any(pattern.search(layer_name) for pattern in exclude_patterns):
            continue
        if min_height_value is not None:
            height_value = entry.get("height")
            if isinstance(height_value, (int, float)) and height_value < min_height_value:
                continue
        filtered.append(entry)

    return filtered


def set_trace_acad(enabled: bool) -> None:
    """Toggle verbose tracing for AutoCAD table discovery helpers."""

    global _TRACE_ACAD
    _TRACE_ACAD = bool(enabled)


def log_last_dxf_fallback(tables_found: Any) -> None:
    """Emit DXF fallback diagnostics if tracing is enabled."""

    global _LAST_DXF_FALLBACK_INFO
    info = _LAST_DXF_FALLBACK_INFO
    _LAST_DXF_FALLBACK_INFO = None
    if not info or not _TRACE_ACAD:
        return
    version = str(info.get("version") or "").strip() or "-"
    path = str(info.get("path") or "").strip() or "-"
    tables_count = 0
    if isinstance(tables_found, (int, float)):
        tables_count = int(tables_found)
    else:
        try:
            tables_count = int(float(tables_found))
        except Exception:
            tables_count = 0
    ok = info.get("ok") if isinstance(info, Mapping) else None
    ok_display = bool(ok) if isinstance(ok, (bool, int)) else False
    print(
        "[DXF-FALLBACK] version={version} path={path} tables={tables} ok={ok}".format(
            version=version,
            path=path,
            tables=tables_count,
            ok=ok_display,
        )
    )


def scan_tables_everywhere(doc) -> list[TableHit]:
    """Enumerate AutoCAD tables across layouts, paper space, and blocks."""

    global _LAST_ACAD_TABLE_SCAN

    hits: list[TableHit] = []
    scan_tables: list[dict[str, Any]] = []
    layouts_scanned: list[str] = []
    blocks_scanned: list[str] = []

    if doc is None:
        print("[ACAD-TABLE] scanned layouts=0 blocks=0 tables_found=0")
        _LAST_ACAD_TABLE_SCAN = {
            "layouts": [],
            "blocks": [],
            "tables_found": 0,
            "tables": [],
        }
        return hits

    containers: list[tuple[str, str, Any, str | None]] = []

    modelspace = getattr(doc, "modelspace", None)
    if callable(modelspace):
        try:
            ms = modelspace()
        except Exception:
            ms = None
        if ms is not None:
            containers.append(("MODEL", "MODEL", ms, None))
            layouts_scanned.append("MODEL")

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
            layout_obj = None
            if callable(get_layout):
                try:
                    layout_obj = get_layout(raw_name)
                except Exception:
                    layout_obj = None
            if layout_obj is None:
                continue
            is_paper = False
            layout_dxf = getattr(layout_obj, "dxf", None)
            if layout_dxf is not None:
                try:
                    is_paper = bool(int(getattr(layout_dxf, "paperspace", 0)))
                except Exception:
                    is_paper = False
            if not is_paper:
                continue
            block_obj = None
            block_method = getattr(layout_obj, "block", None)
            if callable(block_method):
                try:
                    block_obj = block_method()
                except Exception:
                    block_obj = None
            if block_obj is None:
                block_obj = layout_obj
            owner = f"PAPER:{name}"
            containers.append((owner, "PAPER", block_obj, name))
            layouts_scanned.append(name)

    blocks_manager = getattr(doc, "blocks", None)
    block_items: list[tuple[str, Any]] = []
    if blocks_manager is not None:
        try:
            iterator = list(blocks_manager)
        except Exception:
            iterator = []
        for block in iterator:
            if block is None:
                continue
            name = getattr(block, "name", None)
            if name is None and hasattr(block, "dxf"):
                name = getattr(block.dxf, "name", None)
            name_text = str(name).strip() if isinstance(name, str) else None
            block_items.append((name_text or "", block))
        for name_text, block in block_items:
            owner = f"BLOCK:{name_text}" if name_text else "BLOCK"
            containers.append((owner, "BLOCK", block, name_text or None))
            if name_text:
                blocks_scanned.append(name_text)

    seen_entities: set[int] = set()

    for scan_index, (owner_label, source, container, name) in enumerate(containers):
        if container is None:
            continue
        layout_label = name or owner_label
        for flattened in flatten_entities(container, depth=_MAX_INSERT_DEPTH):
            entity = flattened.entity
            try:
                dxftype = str(entity.dxftype()).upper()
            except Exception:
                dxftype = ""
            if dxftype not in {"ACAD_TABLE", "TABLE"}:
                continue
            marker = id(entity)
            if marker in seen_entities:
                continue
            seen_entities.add(marker)
            layer_name = flattened.layer or ""
            handle = _entity_handle(entity)
            owner_handle = _entity_owner_handle(entity)
            rows = _get_table_dimension(
                entity,
                ("n_rows", "row_count", "nrows", "rows"),
            )
            cols = _get_table_dimension(
                entity,
                ("n_cols", "col_count", "ncols", "columns"),
            )
            rows_val = rows if isinstance(rows, int) and rows >= 0 else 0
            cols_val = cols if isinstance(cols, int) and cols >= 0 else 0
            scan_entry = {
                "owner": owner_label,
                "layer": layer_name or "",
                "handle": handle or "",
                "rows": rows_val,
                "cols": cols_val,
                "type": dxftype or "",
                "from_block": flattened.from_block,
                "block_stack": list(flattened.block_stack),
            }
            scan_tables.append(scan_entry)
            if _TRACE_ACAD:
                layout_display = str(layout_label or "").strip() or owner_label
                owner_display = owner_handle or handle or "-"
                layer_display = layer_name or "-"
                print(
                    "[ACAD-TABLE] layout={layout} owner={owner} rows={rows} cols={cols} layer={layer}".format(
                        layout=layout_display,
                        owner=owner_display,
                        rows=rows_val,
                        cols=cols_val,
                        layer=layer_display,
                    )
                )
            hit = TableHit(
                owner=owner_label,
                layer=layer_name or "",
                handle=handle or "",
                rows=rows_val,
                cols=cols_val,
                dxftype=dxftype or "",
                table=entity,
                source=source,
                name=name,
                scan_index=len(scan_tables) - 1,
                transform=flattened.transform,
                block_stack=flattened.block_stack,
                from_block=flattened.from_block,
            )
            hits.append(hit)

    layouts_count = len({name for name in layouts_scanned if name})
    blocks_count = len({name for name in blocks_scanned if name})
    tables_found = len(hits)

    print(
        f"[ACAD-TABLE] scanned layouts={layouts_count} blocks={blocks_count} tables_found={tables_found}"
    )
    for entry in scan_tables:
        print(
            "[ACAD-TABLE] hit owner={owner} layer={layer} handle={handle} rows={rows} cols={cols} type={typ}".format(
                owner=entry.get("owner") or "-",
                layer=entry.get("layer") or "-",
                handle=entry.get("handle") or "-",
                rows=int(entry.get("rows") or 0),
                cols=int(entry.get("cols") or 0),
                typ=entry.get("type") or "-",
            )
        )

    _LAST_ACAD_TABLE_SCAN = {
        "layouts": sorted({name for name in layouts_scanned if name}),
        "blocks": sorted({name for name in blocks_scanned if name}),
        "tables_found": tables_found,
        "tables": scan_tables,
    }

    return hits


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
    if re.fullmatch(r"\d{1,3}", stripped):
        try:
            value = int(stripped)
        except Exception:
            value = None
        if value is not None and value > 0:
            return value
    if _DECIMAL_RE.search(candidate):
        return None
    loose_match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", candidate)
    if loose_match:
        try:
            return int(loose_match.group(1))
        except Exception:
            return None
    return None


def _detect_header_hits(cells: list[str]) -> dict[str, int]:
    header_token_re = re.compile(
        r"(QTY|QUANTITY|DESC|DESCRIPTION|REF|DIA|Ø|⌀|HOLE|ID|SIDE)",
        re.IGNORECASE,
    )
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


def _compute_table_bbox(
    entity: Any, *, transform: TransformMatrix | None = None
) -> tuple[float, float, float, float] | None:
    try:
        virtual_entities = list(entity.virtual_entities())
    except Exception:
        virtual_entities = []
    return _compute_entity_bbox(
        entity,
        include_virtual=True,
        virtual_entities=virtual_entities,
        transform=transform,
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
    try:
        default_height = float(getattr(entity, "default_text_height", 0.0))
    except Exception:
        default_height = 0.0
    if default_height > 0:
        return default_height
    return 0.0


def _rows_from_acad_table_with_info(
    entity: Any, *, transform: TransformMatrix | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {}
    if entity is None:
        return ([], info)

    n_rows = _get_table_dimension(entity, ("n_rows", "row_count", "nrows", "rows")) or 0
    n_cols = _get_table_dimension(entity, ("n_cols", "col_count", "ncols", "columns")) or 0
    info["n_rows"] = n_rows
    info["n_cols"] = n_cols
    if n_rows <= 0 or n_cols <= 0:
        return ([], info)

    dxf_obj = getattr(entity, "dxf", None)
    insert = getattr(dxf_obj, "insert", None) if dxf_obj is not None else None
    if insert is None:
        insert = getattr(entity, "insert", None)
    base_x = None
    base_y = None
    if insert is not None:
        try:
            base_x = float(getattr(insert, "x", None))
        except Exception:
            base_x = None
        try:
            base_y = float(getattr(insert, "y", None))
        except Exception:
            base_y = None
        if (base_x is None or base_y is None) and hasattr(insert, "__iter__"):
            try:
                parts = list(insert)
            except Exception:
                parts = []
            if base_x is None and len(parts) >= 1:
                try:
                    base_x = float(parts[0])
                except Exception:
                    base_x = None
            if base_y is None and len(parts) >= 2:
                try:
                    base_y = float(parts[1])
                except Exception:
                    base_y = None

    get_cell_extents = getattr(entity, "get_cell_extents", None)
    get_column_width = getattr(entity, "get_column_width", None)
    get_row_height = getattr(entity, "get_row_height", None)

    fallback_col_edges: list[float] | None = None
    if not callable(get_cell_extents) and callable(get_column_width):
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
        if len(edges) == int(n_cols) + 1:
            fallback_col_edges = edges

    fallback_row_edges: list[float] | None = None
    if not callable(get_cell_extents) and callable(get_row_height):
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
        if len(edges_y) == int(n_rows) + 1:
            fallback_row_edges = edges_y

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
                    if (
                        math.isfinite(x_min)
                        and math.isfinite(x_max)
                        and math.isfinite(y_min)
                        and math.isfinite(y_max)
                    ):
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
            if (
                math.isfinite(x_min)
                and math.isfinite(x_max)
                and math.isfinite(y_top)
                and math.isfinite(y_bottom)
            ):
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
                text_value = _cell_text(entity, row_idx, col_idx)
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

    if table_centers:
        if transform is not None:
            transformed_centers: list[list[tuple[float, float] | None]] = []
            for row_centers in table_centers:
                transformed_row: list[tuple[float, float] | None] = []
                for center in row_centers:
                    if center is None:
                        transformed_row.append(None)
                        continue
                    tx, ty = _apply_transform_point(transform, center)
                    if tx is None or ty is None:
                        transformed_row.append(None)
                    else:
                        transformed_row.append((tx, ty))
                transformed_centers.append(transformed_row)
            info["cell_centers"] = transformed_centers
        else:
            info["cell_centers"] = table_centers
    info["table_cells"] = table_cells
    info["row_count_raw"] = len(table_cells)

    header_map: dict[str, int] = {}
    header_row_idx: int | None = None
    header_valid = False
    for idx, row_cells in enumerate(table_cells):
        hits = _detect_header_hits(row_cells)
        if not hits:
            continue
        header_map = dict(hits)
        header_row_idx = idx
        combined_upper = " ".join(cell.upper() for cell in row_cells if cell)
        has_hole = "HOLE" in combined_upper
        has_desc = "DESC" in combined_upper or "DESCRIPTION" in combined_upper
        has_qty = "QTY" in combined_upper or "QUANTITY" in combined_upper
        has_ref = any(token in combined_upper for token in ("REF", "Ø", "⌀", "DIA"))
        header_valid = has_hole and has_desc and has_qty and has_ref
        break

    info["header_map"] = dict(header_map)
    info["header_row_index"] = header_row_idx
    info["header_valid"] = header_valid

    if "desc" not in header_map and table_cells:
        candidate_indices = list(range(len(table_cells[0])))
        for used in header_map.values():
            if used in candidate_indices:
                candidate_indices.remove(used)
        if candidate_indices:
            header_map["desc"] = max(candidate_indices)

    rows: list[dict[str, Any]] = []
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
        qty_val = None
        qty_idx = header_map.get("qty")
        if qty_idx is not None and qty_idx < len(row_cells):
            qty_val = _parse_qty_cell_text(row_cells[qty_idx])
        fallback_desc = ""
        combined_qty, combined_remainder = _extract_column_quantity_and_remainder(combined_text)
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
        if qty_val is not None and desc_text:
            cleaned_desc = re.sub(r"^(?:\(\s*\d+\s*\)|\d+\s*[Xx×])\s+", "", desc_text, count=1)
            if cleaned_desc:
                desc_text = cleaned_desc
        if not desc_text:
            continue

        ref_idx = header_map.get("ref")
        ref_cell_text = row_cells[ref_idx] if ref_idx is not None and ref_idx < len(row_cells) else ""
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

        fragment_desc = " ".join(desc_text.split())
        if not fragment_desc:
            continue
        ref_text, ref_value = _extract_row_reference(fragment_desc)
        if not ref_text and ref_cell_ref[0]:
            ref_text, ref_value = ref_cell_ref
        elif not ref_text and ref_cell_text:
            ref_text = " ".join(ref_cell_text.split())
            ref_value = None
        if qty_val is not None and fragment_desc:
            qty_prefix = re.match(r"^(\d+)\s*[Xx×]\s+(.*)", fragment_desc)
            if qty_prefix:
                prefix_qty = qty_prefix.group(1)
                try:
                    if int(prefix_qty) == int(qty_val):
                        fragment_desc = qty_prefix.group(2).strip()
                except Exception:
                    fragment_desc = qty_prefix.group(2).strip()
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
        rows.append(row_entry)
        if ref_value is not None:
            key = f"{ref_value:.4f}".rstrip("0").rstrip(".")
            families[key] = families.get(key, 0) + qty_val

    info["families"] = families
    info["row_count"] = len(rows)
    sum_qty = _sum_qty(rows)
    info["sum_qty"] = sum_qty

    bbox = _compute_table_bbox(entity, transform=transform)
    if bbox is not None:
        info["bbox"] = bbox
    median_height = _estimate_text_height(entity, n_rows)
    info["median_height"] = median_height

    return (rows, info)


def rows_from_acad_table(
    entity: Any, *, transform: TransformMatrix | None = None
) -> list[dict[str, Any]]:
    rows, _info = _rows_from_acad_table_with_info(entity, transform=transform)
    return rows


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
    transform: TransformMatrix | None = None,
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
    if transform is not None and points:
        transformed: list[tuple[float, float]] = []
        for x_val, y_val in points:
            try:
                x_float = float(x_val)
                y_float = float(y_val)
            except Exception:
                continue
            tx, ty = _apply_transform_point(transform, (x_float, y_float))
            if tx is None or ty is None:
                continue
            transformed.append((tx, ty))
        points = transformed
    if not points:
        return None
    xs = [pt[0] for pt in points if isinstance(pt[0], (int, float))]
    ys = [pt[1] for pt in points if isinstance(pt[1], (int, float))]
    if not xs or not ys:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


class _LayerAllowlist(Iterable[str]):
    __slots__ = ("_patterns",)

    def __init__(self, patterns: Iterable[str]):
        unique: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            upper = pattern.upper()
            if upper in seen:
                continue
            seen.add(upper)
            unique.append(upper)
        self._patterns = tuple(unique)

    def __iter__(self):
        return iter(self._patterns)

    def __contains__(self, value: object) -> bool:  # pragma: no cover - exercised via iteration
        if not self._patterns:
            return False
        text = "" if value is None else str(value)
        candidate = text.upper()
        for pattern in self._patterns:
            if fnmatchcase(candidate, pattern):
                return True
        return False

    def __len__(self) -> int:
        return len(self._patterns)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"_LayerAllowlist(patterns={self._patterns!r})"


def _normalize_layer_allowlist(
    layer_allowlist: Iterable[str] | None,
) -> _LayerAllowlist | None:
    if layer_allowlist is None or layer_allowlist is _DEFAULT_LAYER_ALLOWLIST:
        return None

    special_tokens = {"ALL", "*", "<ALL>"}
    normalized: list[str] = []
    seen: set[str] = set()
    allow_all = False

    def _consume(value: Any) -> None:
        nonlocal allow_all
        if allow_all or value is None:
            return
        if isinstance(value, str):
            raw_values = value.split(",")
        else:
            raw_values = [value]
        for item in raw_values:
            if allow_all:
                break
            text = str(item).strip()
            if not text:
                continue
            upper = text.upper()
            if upper in special_tokens:
                allow_all = True
                break
            if upper in seen:
                continue
            seen.add(upper)
            normalized.append(upper)

    sources: tuple[Iterable[Any] | Any, ...] = (
        _DEFAULT_LAYER_ALLOWLIST,
        layer_allowlist,
    )
    for source in sources:
        if allow_all:
            break
        if isinstance(source, Iterable) and not isinstance(source, (str, bytes, bytearray)):
            for value in source:
                _consume(value)
                if allow_all:
                    break
        else:
            _consume(source)

    if allow_all or not normalized:
        return None
    return _LayerAllowlist(normalized)


@lru_cache(maxsize=1)
def _load_app_module():
    import importlib

    return importlib.import_module("appV5")


@lru_cache(maxsize=None)
def _resolve_app_callable(name: str) -> Callable[..., Any] | None:
    try:
        module = _load_app_module()
    except Exception:
        module = None
    if module is not None:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    return globals().get(name)


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
_ADMIN_ROW_DROP_RE = re.compile(
    r"\b(SEE\s+SHEET|BREAK\s+ALL|RADIUS|DEBURR|TOLERANCE|SCALE|TITLE|DETAIL|FINISH|NOTE)\b",
    re.IGNORECASE,
)
_NUMERIC_LADDER_RE = re.compile(r"^(?:\d+\s+){8,}\d+$")
_FRAGMENT_SPLIT_RE = re.compile(r";|(?<=\))\s+(?=\d+\))")
_INCH_MARK_REF_RE = re.compile(r"\b(\d+(?:\.\d+)?)(?:\s*\"|[ ]?in\b)", re.IGNORECASE)
_DIA_SYMBOL_INLINE_RE = re.compile(r"[Ø⌀]\s*(\d+(?:\.\d+)?)")
_LETTER_DRILL_REF_RE = re.compile(r"\b([A-HJ-NP-Z])\"?\b")
_NUMBERED_THREAD_REF_RE = re.compile(r"#\d+\s*-\s*\d+", re.IGNORECASE)
_NUMBER_DRILL_REF_RE = re.compile(r"#\d+\b")
_PIPE_NPT_REF_RE = re.compile(
    r"\b((?:\d+\/\d+|\d+(?:\.\d+)?))\s*-\s*(N\.?P\.?T\.?)",
    re.IGNORECASE,
)
_THREAD_CALL_OUT_RE = re.compile(r"\b(\d+\/\d+|M\d+(?:\.\d+)?)-\d+\b", re.IGNORECASE)
_ROW_ANCHOR_RE = re.compile(r"^\(\s*(\d{1,3})\s*\)\s+", re.IGNORECASE)
_ROW_QUANTITY_PATTERNS = [
    _ROW_ANCHOR_RE,
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
_FALLBACK_LEADING_QTY_RE = re.compile(r"^\(\d+\)\s*")
_FALLBACK_JJ_NOISE_RE = re.compile(r"\bJ\s+J\b", re.IGNORECASE)
_FALLBACK_ETCH_NOISE_RE = re.compile(r"\bETCH ON DETAIL\b(?:\.)?", re.IGNORECASE)
_RE_TEXT_ROW_START = _ROW_ANCHOR_RE
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

_DEBUG_DIR = Path("debug")
_DEBUG_ROWS_PATH = _DEBUG_DIR / "hole_table_rows.csv"
_DEBUG_TOTALS_PATH = _DEBUG_DIR / "ops_table_totals.json"
_DEBUG_FIELDNAMES = (
    "qty",
    "kind",
    "side",
    "tool",
    "diam_token",
    "depth_token",
    "raw_text",
)
_DEBUG_DEPTH_RE = re.compile(
    r"([x×]\s*)?(\d+(?:\.\d+)?)\s*(?:DEEP|DEPTH|THK|THICK)\b",
    re.IGNORECASE,
)
_DEBUG_THRU_RE = re.compile(r"\bTHRU\b", re.IGNORECASE)
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
_PROMOTED_ROWS_LOGGED = False


def _promoted_rows_preview_limit() -> int:
    raw_value = os.environ.get("CAD_QUOTER_SHOW_ROWS")
    if raw_value in (None, ""):
        return 0
    try:
        limit = int(float(str(raw_value).strip()))
    except Exception:
        return 0
    return max(limit, 0)


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


_QTY_PREFIX = re.compile(r"^\(\d+\)\s*")


def _clean_desc_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    cleaned = _QTY_PREFIX.sub("", text)
    cleaned = cleaned.rstrip("; ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_for_dedupe(value: Any) -> str:
    return _clean_cell_text(value).upper()


def _classify_promoted_row(desc_upper: str) -> str:
    if not desc_upper:
        return "note"
    tap_tokens = ("TAP", "THREAD", "NPT", "N.P.T", "NPTF", "NPS")
    counterbore_tokens = ("COUNTERBORE", "COUNTER BORE", "C'BORE", "CBORE", "SPOTFACE", "SPOT FACE")
    drill_tokens = (
        "DRILL",
        "THRU",
        "Ø",
        "⌀",
        "DIA",
        "CSK",
        "C'SINK",
        "COUNTERSINK",
        "SPOT",
        "REAM",
        "JIG GRIND",
        "CENTER DRILL",
        "C DRILL",
        "C’DRILL",
    )
    if any(token in desc_upper for token in tap_tokens):
        return "tap"
    if any(token in desc_upper for token in counterbore_tokens):
        return "counterbore"
    drill_hit = any(token in desc_upper for token in drill_tokens)
    if ("NOTE" in desc_upper or desc_upper.startswith("SEE ")) and not drill_hit:
        return "note"
    return "drill"


def _prepare_columnar_promoted_rows(
    table_info: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    rows_raw = _normalize_table_rows(table_info.get("rows") if isinstance(table_info, Mapping) else None)
    seen: set[tuple[int, str, str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = {
        "tap": [],
        "counterbore": [],
        "drill": [],
    }
    for row in rows_raw:
        if not isinstance(row, Mapping):
            continue
        qty_raw = row.get("qty")
        try:
            qty_val = int(float(qty_raw or 0))
        except Exception:
            qty_val = 0
        if qty_val <= 0:
            continue
        desc_clean = _clean_desc_text(row.get("desc"))
        ref_clean = _clean_cell_text(row.get("ref"))
        side_clean = _clean_cell_text(row.get("side"))
        dedupe_key = (
            qty_val,
            _normalize_for_dedupe(ref_clean),
            _normalize_for_dedupe(side_clean),
            _normalize_for_dedupe(desc_clean),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        prepared_row: dict[str, Any] = dict(row)
        prepared_row["qty"] = qty_val
        prepared_row["desc"] = desc_clean
        prepared_row["ref"] = ref_clean
        if side_clean:
            prepared_row["side"] = side_clean
        elif "side" in prepared_row:
            prepared_row.pop("side")
        prepared_row["hole"] = _clean_cell_text(row.get("hole"))
        row_type = _classify_promoted_row(_normalize_for_dedupe(desc_clean))
        if row_type == "note":
            continue
        grouped[row_type].append(prepared_row)
    ordered_rows: list[dict[str, Any]] = []
    for kind in ("tap", "counterbore", "drill"):
        ordered_rows.extend(grouped[kind])
    qty_sum = sum(row.get("qty", 0) for row in ordered_rows if isinstance(row.get("qty"), int))
    return (ordered_rows, qty_sum)


def _print_promoted_rows_once(rows: Iterable[Mapping[str, Any]]) -> None:
    global _PROMOTED_ROWS_LOGGED
    if _PROMOTED_ROWS_LOGGED:
        return
    limit = _promoted_rows_preview_limit()
    if limit <= 0:
        return
    materialized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not materialized:
        return
    count = min(limit, len(materialized))
    print(f"[EXTRACT] promoted rows preview_count={count}")
    for idx, row in enumerate(materialized[:count]):
        qty_display = row.get("qty")
        ref_display = row.get("ref") or "-"
        side_display = row.get("side") or "-"
        desc_display = row.get("desc") or ""
        print(
            "[EXTRACT] promoted row[{idx:02d}] qty={qty} ref={ref} side={side} desc={desc}".format(
                idx=idx,
                qty=qty_display,
                ref=ref_display,
                side=side_display,
                desc=desc_display,
            )
        )
    _PROMOTED_ROWS_LOGGED = True


def _score_table(info: Mapping[str, Any] | None) -> tuple[int, int, int]:
    if not isinstance(info, Mapping):
        return (0, 0, 0)
    rows = info.get("rows") or []
    header_ok = 1 if info.get("header_validated") else 0
    return (header_ok, _sum_qty(rows), len(rows))


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
        module = None
        try:
            module = inspect.getmodule(helper)
        except Exception:
            module = None
        if module is None:
            try:
                module = _load_app_module()
            except Exception:
                module = None
        sentinel = object()
        prev_allow = sentinel
        prev_depth = sentinel
        if module is not None:
            try:
                prev_allow = getattr(module, "_ACAD_LAYER_ALLOW_OVERRIDE")
            except AttributeError:
                prev_allow = sentinel
            setattr(module, "_ACAD_LAYER_ALLOW_OVERRIDE", layer_allowlist)
            try:
                prev_depth = getattr(module, "_ACAD_DEPTH_MAX_OVERRIDE")
            except AttributeError:
                prev_depth = sentinel
            depth_override = None
            if feature_flags and isinstance(feature_flags, Mapping):
                depth_override = feature_flags.get("acad_depth_max")
            setattr(module, "_ACAD_DEPTH_MAX_OVERRIDE", depth_override)
        try:
            result = helper(doc) or {}
        except Exception as exc:
            print(f"[EXTRACT] acad helper error: {exc}")
            raise
        finally:
            if module is not None:
                if prev_allow is sentinel:
                    try:
                        delattr(module, "_ACAD_LAYER_ALLOW_OVERRIDE")
                    except AttributeError:
                        pass
                else:
                    setattr(module, "_ACAD_LAYER_ALLOW_OVERRIDE", prev_allow)
                if prev_depth is sentinel:
                    try:
                        delattr(module, "_ACAD_DEPTH_MAX_OVERRIDE")
                    except AttributeError:
                        pass
                else:
                    setattr(module, "_ACAD_DEPTH_MAX_OVERRIDE", prev_depth)
        if isinstance(result, Mapping):
            return dict(result)
        return {}

    allowlist = _normalize_layer_allowlist(layer_allowlist)
    hits = scan_tables_everywhere(doc)
    if not hits:
        return {}

    global _LAST_ACAD_TABLE_SCAN
    scan_tables_meta: list[dict[str, Any]] = []
    if isinstance(_LAST_ACAD_TABLE_SCAN, Mapping):
        raw_tables = _LAST_ACAD_TABLE_SCAN.get("tables")
        if isinstance(raw_tables, list):
            scan_tables_meta = raw_tables

    table_candidates: list[dict[str, Any]] = []

    for hit in hits:
        rows, table_info = _rows_from_acad_table_with_info(
            hit.table, transform=hit.transform
        )
        row_count = int(table_info.get("row_count") or len(rows) or 0)
        sum_qty_val = table_info.get("sum_qty")
        try:
            sum_qty_int = int(sum_qty_val)
        except Exception:
            sum_qty_int = _sum_qty(rows)
        layer_upper = hit.layer.upper() if hit.layer else ""
        header_valid = bool(table_info.get("header_valid"))
        families_raw = table_info.get("families")
        families = dict(families_raw) if isinstance(families_raw, Mapping) else {}
        bbox = table_info.get("bbox")
        median_height = table_info.get("median_height")
        roi_hint: dict[str, Any] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                xmin = float(bbox[0])
                xmax = float(bbox[1])
                ymin = float(bbox[2])
                ymax = float(bbox[3])
            except Exception:
                xmin = xmax = ymin = ymax = 0.0
            else:
                try:
                    median_height_float = float(median_height or 0.0)
                except Exception:
                    median_height_float = 0.0
                pad = 2.0 * median_height_float if median_height_float > 0 else 6.0
                roi_hint = {
                    "source": "ACAD_TABLE",
                    "handle": hit.handle,
                    "layer": hit.layer,
                    "bbox": [xmin, xmax, ymin, ymax],
                    "pad": pad,
                    "median_height": median_height_float,
                }
                if hit.name:
                    roi_hint["name"] = hit.name

        candidate = {
            "rows": rows,
            "row_count": row_count,
            "sum_qty": sum_qty_int,
            "layer": hit.layer,
            "layer_upper": layer_upper,
            "owner": hit.owner,
            "handle": hit.handle,
            "n_rows": table_info.get("n_rows"),
            "n_cols": table_info.get("n_cols"),
            "families": families,
            "roi_hint": roi_hint,
            "cell_centers": table_info.get("cell_centers"),
            "header_valid": header_valid,
        }
        if rows:
            table_candidates.append(candidate)

        if 0 <= hit.scan_index < len(scan_tables_meta):
            scan_entry = scan_tables_meta[hit.scan_index]
            if isinstance(scan_entry, dict):
                scan_entry.setdefault("owner", hit.owner)
                scan_entry.setdefault("layer", hit.layer)
                scan_entry.setdefault("handle", hit.handle)
                scan_entry["rows"] = row_count
                scan_entry["cols"] = table_info.get("n_cols") or scan_entry.get("cols", 0)
                scan_entry["sum_qty"] = sum_qty_int
                scan_entry["header_valid"] = header_valid


    if isinstance(_LAST_ACAD_TABLE_SCAN, dict):
        _LAST_ACAD_TABLE_SCAN["tables"] = scan_tables_meta
        _LAST_ACAD_TABLE_SCAN["tables_found"] = len(hits)

    if not table_candidates:
        return {}

    header_candidates = [cand for cand in table_candidates if cand.get("header_valid")]
    if not header_candidates:
        return {}

    if allowlist is not None:
        filtered = [cand for cand in header_candidates if cand.get("layer_upper") in allowlist]
    else:
        filtered = list(header_candidates)
    if not filtered:
        filtered = header_candidates

    best_candidate = max(
        filtered,
        key=lambda cand: (int(cand.get("row_count") or 0), int(cand.get("sum_qty") or 0)),
    )

    best_rows = list(best_candidate.get("rows") or [])
    if not best_rows:
        return {}

    sum_qty = int(best_candidate.get("sum_qty") or _sum_qty(best_rows))
    result: dict[str, Any] = {
        "rows": best_rows,
        "hole_count": sum_qty,
        "sum_qty": sum_qty,
        "provenance_holes": "HOLE TABLE",
        "layer": best_candidate.get("layer"),
        "owner": best_candidate.get("owner"),
        "handle": best_candidate.get("handle"),
        "n_rows": best_candidate.get("n_rows"),
        "n_cols": best_candidate.get("n_cols"),
        "source": "acad_table",
        "header_validated": True,
    }

    families_map = best_candidate.get("families")
    if isinstance(families_map, Mapping) and families_map:
        result["hole_diam_families_in"] = dict(families_map)

    roi_hint = best_candidate.get("roi_hint")
    if isinstance(roi_hint, Mapping):
        result["roi_hint"] = dict(roi_hint)

    cell_centers = best_candidate.get("cell_centers")
    if isinstance(cell_centers, list) and cell_centers:
        result["cell_centers"] = cell_centers

    print(
        "[ACAD-TABLE] chosen handle={handle} layer={layer} owner={owner} rows={rows} qty_sum={qty}".format(
            handle=result.get("handle") or "-",
            layer=result.get("layer") or "-",
            owner=result.get("owner") or "-",
            rows=len(best_rows),
            qty=sum_qty,
        )
    )

    return result



def _collect_table_text_lines(
    doc: Any,
    *,
    layout_filters: Mapping[str, Any] | Iterable[str] | str | None = None,
) -> list[str]:
    lines: list[str] = []
    seen_markers: set[int] = set()
    if doc is None:
        return lines

    try:
        spaces = iter_layouts(doc, layout_filters, log=False)
    except RuntimeError:
        raise

    for _name, space in spaces:
        if space is None:
            continue
        query = getattr(space, "query", None)
        if not callable(query):
            continue
        marker = id(space)
        if marker in seen_markers:
            continue
        seen_markers.add(marker)
        try:
            entities = list(query("TEXT, MTEXT, RTEXT"))
        except Exception:
            continue
        for entity in entities:
            fragments = list(_iter_entity_text_fragments(entity))
            for fragment, _ in fragments:
                normalized = _normalize_table_fragment(fragment)
                if normalized:
                    lines.append(normalized)
    return lines


def ensure_text_stream(
    doc_or_path: Any, log: Callable[[str], None] | None = None
) -> tuple[Any, list[str]]:
    """Run the text collector and retry via DXF conversion when needed."""

    def _log(message: str) -> None:
        if callable(log):
            try:
                log(message)
            except Exception:  # pragma: no cover - defensive logging
                pass

    doc_candidate = doc_or_path
    try:
        lines = _collect_table_text_lines(doc_candidate)
    except Exception:
        lines = []

    if lines:
        return doc_candidate, lines

    path_obj: Path | None = None
    if isinstance(doc_or_path, (str, os.PathLike)):
        try:
            path_obj = Path(doc_or_path)
        except Exception:
            path_obj = None
    else:
        for attr in ("filename", "filepath", "file_path", "_filename"):
            candidate = getattr(doc_or_path, attr, None)
            if not candidate:
                continue
            try:
                path_obj = Path(candidate)
            except Exception:
                continue
            if path_obj:
                break

    if path_obj is None or path_obj.suffix.lower() != ".dwg":
        return doc_candidate, lines

    try:
        dxf_path = convert_dwg_to_dxf(str(path_obj), out_ver="ACAD2013")
    except Exception as exc:  # pragma: no cover - defensive logging
        _log(f"[FALLBACK] text-stream convert failed path={path_obj} err={exc}")
        return doc_candidate, lines

    _log(
        "[FALLBACK] text-stream source={src} dxf={dst}".format(
            src=path_obj, dst=dxf_path
        )
    )

    fallback_doc: Any | None = None
    try:
        ezdxf_mod = geometry.require_ezdxf()
        readfile = getattr(ezdxf_mod, "readfile", None)
        if callable(readfile):
            fallback_doc = readfile(str(dxf_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        _log(f"[FALLBACK] text-stream read err={exc}")
        fallback_doc = None

    if fallback_doc is None:
        return doc_candidate, lines

    doc_candidate = fallback_doc
    try:
        lines = _collect_table_text_lines(doc_candidate)
    except Exception:
        lines = []
    return doc_candidate, lines


def _resolve_follow_sheet_layout(
    token: str, layout_names: Iterable[str]
) -> tuple[str, str | None, bool]:
    normalized = re.sub(r"[^A-Z0-9]", "", str(token or "")).upper()
    if not normalized:
        target_label = "SHEET ()"
        return (target_label, None, False)

    target_label = f"SHEET ({normalized})"

    lookup: dict[str, str] = {}
    for name in layout_names:
        if not isinstance(name, str):
            continue
        stripped = name.strip()
        if not stripped:
            continue
        lookup.setdefault(stripped.upper(), stripped)

    target_upper = target_label.upper()
    if target_upper in lookup:
        return (target_label, lookup[target_upper], True)

    alt_labels: list[str] = []
    if normalized.isdigit():
        trimmed = normalized.lstrip("0") or "0"
        if trimmed != normalized:
            alt_labels.append(f"SHEET ({trimmed})")
        alt_labels.append(f"SHEET {normalized}")
        if trimmed != normalized:
            alt_labels.append(f"SHEET {trimmed}")
        alt_labels.append(trimmed)
    else:
        alt_labels.append(f"SHEET {normalized}")
        alt_labels.append(normalized)

    seen: set[str] = set()
    for candidate in alt_labels:
        candidate_upper = candidate.upper()
        if candidate_upper in seen:
            continue
        seen.add(candidate_upper)
        if candidate_upper in lookup:
            return (target_label, lookup[candidate_upper], True)

    for upper_name, original in lookup.items():
        if normalized and normalized in upper_name:
            return (target_label, original, True)

    return (target_label, None, False)


def _count_tables_for_layout_name(layout_name: str) -> int:
    layout_upper = str(layout_name or "").strip().upper()
    if not layout_upper:
        return 0
    snapshot = _LAST_ACAD_TABLE_SCAN
    if not isinstance(snapshot, Mapping):
        return 0
    raw_tables = snapshot.get("tables")
    if not isinstance(raw_tables, list):
        return 0
    suffixes = (f":{layout_upper}",)
    count = 0
    for entry in raw_tables:
        if not isinstance(entry, Mapping):
            continue
        owner_upper = str(entry.get("owner") or "").strip().upper()
        if not owner_upper:
            continue
        if owner_upper == layout_upper:
            count += 1
            continue
        if any(owner_upper.endswith(suffix) for suffix in suffixes):
            count += 1
    return count


def _extract_layer(entity: Any) -> str:
    dxf_obj = getattr(entity, "dxf", None)
    layer_name = None
    if dxf_obj is not None:
        layer_name = getattr(dxf_obj, "layer", None)
    if not layer_name:
        layer_name = getattr(entity, "layer", None)
    try:
        return str(layer_name or "").strip()
    except Exception:
        return ""


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
    elif kind == "MLEADER":
        context = getattr(entity, "context", None)
        if context is None:
            return
        mtext = getattr(context, "mtext", None)
        if mtext is None:
            raw_text = getattr(context, "text", "")
            try:
                base = str(raw_text)
            except Exception:
                base = raw_text if isinstance(raw_text, str) else ""
            for piece in base.splitlines():
                if piece.strip():
                    yield (piece, True)
            return
        plain_text = getattr(mtext, "plain_text", None)
        content = None
        if callable(plain_text):
            try:
                content = plain_text()
            except Exception:
                content = None
        if content is None:
            content = getattr(mtext, "text", "")
        try:
            base = str(content)
        except Exception:
            base = content if isinstance(content, str) else ""
        for piece in base.splitlines():
            if piece.strip():
                yield (piece, True)
    elif kind in {"ATTRIB", "ATTDEF"}:
        dxf_obj = getattr(entity, "dxf", None)
        candidates: list[Any] = []
        if dxf_obj is not None:
            for attr in ("text", "value", "tag", "prompt", "default"):
                candidates.append(getattr(dxf_obj, attr, None))
        for attr in ("text", "value", "tag", "prompt", "default"):
            candidates.append(getattr(entity, attr, None))
        for raw in candidates:
            if not raw:
                continue
            try:
                text_value = str(raw)
            except Exception:
                text_value = raw if isinstance(raw, str) else ""
            for piece in text_value.splitlines():
                if piece.strip():
                    yield (piece, False)
    elif kind == "RTEXT":
        def _flatten_text_values(value: Any) -> Iterable[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, (bytes, bytearray)):
                try:
                    return [value.decode("utf-8")]
                except Exception:
                    try:
                        return [value.decode("latin-1")]
                    except Exception:
                        return []
            if isinstance(value, Mapping):
                results: list[str] = []
                for candidate in value.values():
                    results.extend(_flatten_text_values(candidate))
                return results
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
                results: list[str] = []
                for item in value:
                    if isinstance(item, tuple) and len(item) >= 2:
                        results.extend(_flatten_text_values(item[1]))
                    else:
                        results.extend(_flatten_text_values(item))
                return results
            try:
                text = str(value)
            except Exception:
                return []
            return [text]

        seen_fragments: set[str] = set()
        collected: list[str] = []

        def _collect_text(value: Any) -> None:
            for fragment in _flatten_text_values(value):
                cleaned = fragment.strip()
                if not cleaned:
                    continue
                if cleaned in seen_fragments:
                    continue
                seen_fragments.add(cleaned)
                collected.append(cleaned)

        dxf_obj = getattr(entity, "dxf", None)
        for source in (entity, dxf_obj):
            if source is None:
                continue
            for attr in (
                "raw_content",
                "raw_text",
                "stored_text",
                "text",
                "value",
                "content",
                "string",
            ):
                _collect_text(getattr(source, attr, None))
            plain_text = getattr(source, "plain_text", None)
            if callable(plain_text):
                try:
                    _collect_text(plain_text())
                except Exception:
                    pass

        get_xdata = getattr(entity, "get_xdata", None)
        if callable(get_xdata):
            for app in ("RTEXT", "ACAD_RTEXT", "ACAD_REACTORS", "ACAD"):
                try:
                    _collect_text(get_xdata(app))
                except Exception:
                    continue

        for attr_name in ("xdata", "extended_data", "appdata"):
            _collect_text(getattr(entity, attr_name, None))

        if not collected:
            raw_text = getattr(entity, "text", "")
            if not raw_text and dxf_obj is not None:
                raw_text = getattr(dxf_obj, "text", "")
            _collect_text(raw_text)

        if not collected:
            return

        longest = max(collected, key=len, default="")
        if not longest:
            return

        pieces = _split_mtext_plain_text(longest)
        if not pieces:
            pieces = [longest]
        for piece in pieces:
            cleaned_piece = piece.strip()
            if cleaned_piece:
                yield (cleaned_piece, True)
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
    text = f"{value:.4f}"
    text = text.rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return f"{text}\""


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
    return _ROW_ANCHOR_RE.match(candidate)


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
    if not text:
        return False
    return bool(_ROW_ANCHOR_RE.match(text))


def _roi_is_row_starter(text: str) -> bool:
    if not text:
        return False
    return bool(_ROW_ANCHOR_RE.match(text))


def _roi_is_admin_noise(text: str) -> bool:
    if not text:
        return False
    return bool(_ADMIN_ROW_DROP_RE.search(text))


def _roi_is_numeric_ladder(text: str) -> bool:
    if not text:
        return False
    return bool(_NUMERIC_LADDER_RE.match(text))


def _normalize_candidate_text(text: Any) -> str:
    try:
        base = str(text or "")
    except Exception:
        base = ""
    return " ".join(base.split())


def _is_numeric_ladder_line(text: str) -> bool:
    return bool(_NUMERIC_LADDER_RE.match(text or ""))


def _should_drop_candidate_line(text: Any) -> bool:
    normalized = _normalize_candidate_text(text)
    if not normalized:
        return False
    if _is_numeric_ladder_line(normalized):
        return True
    return bool(_ADMIN_ROW_DROP_RE.search(normalized))


def _extract_drill_size(segment: str) -> str | None:
    if not segment or not _DRILL_TOKEN_RE.search(segment):
        return None
    for pattern in _DRILL_SIZE_PATTERNS:
        match = pattern.search(segment)
        if not match:
            continue
        size_text = match.group(match.lastindex or 1)
        if not size_text:
            continue
        cleaned = re.sub(r"\s+", " ", str(size_text)).strip()
        cleaned = cleaned.strip("'\"")
        cleaned = cleaned.replace("Ø", "Ø").replace("⌀", "Ø")
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned
    return None


def classify_op_row(desc: str | None) -> list[dict[str, Any]]:
    """Return operation descriptors parsed from ``desc``.

    Each descriptor contains ``kind`` (one of the manifest buckets), ``qty``
    (initialized to ``0`` and expected to be overridden by callers), and an
    optional ``size`` token for sized drill operations.
    """

    if not desc:
        return []
    try:
        text = str(desc)
    except Exception:
        return []
    segments = [part.strip() for part in _OPS_SEGMENT_SPLIT_RE.split(text) if part.strip()]
    if not segments:
        segments = [text.strip()]

    results: list[dict[str, Any]] = []
    for segment in segments:
        if not segment:
            continue
        kinds: list[tuple[str, str | None]] = []
        is_pipe_tap = bool(_PIPE_TAP_TOKEN_RE.search(segment))
        is_npt = bool(_NPT_TOKEN_RE.search(segment) or is_pipe_tap)
        is_cdrill = bool(_COUNTERDRILL_TOKEN_RE.search(segment))
        has_thread_tap = bool(_TAP_THREAD_TOKEN_RE.search(segment))
        has_tap_word = bool(_TAP_WORD_TOKEN_RE.search(segment))
        has_tap = has_thread_tap or has_tap_word
        if is_npt:
            kinds.append(("npt", None))
        if is_npt or has_tap:
            kinds.append(("tap", None))
        if _COUNTERBORE_TOKEN_RE.search(segment):
            kinds.append(("counterbore", None))
        if _COUNTERSINK_TOKEN_RE.search(segment):
            kinds.append(("csink", None))
        if is_cdrill:
            kinds.append(("counterdrill", None))
        if _JIG_GRIND_TOKEN_RE.search(segment):
            kinds.append(("jig_grind", None))
        if _SPOT_TOKEN_RE.search(segment):
            kinds.append(("spot", None))
        drill_size = None if is_cdrill else _extract_drill_size(segment)
        if drill_size:
            kinds.append(("drill", drill_size))

        if not kinds:
            kinds.append(("unknown", None))

        seen_local: set[tuple[str, str | None]] = set()
        for kind, size_text in kinds:
            key = (kind, size_text if size_text is not None else None)
            if key in seen_local:
                continue
            seen_local.add(key)
            entry = {"kind": kind, "qty": 0, "size": size_text}
            if kind == "tap" and is_npt:
                entry["tap_type"] = "pipe"
            results.append(entry)

    return results


def _coerce_positive_int(value: Any) -> int | None:
    try:
        candidate = int(round(float(value)))
    except Exception:
        return None
    return candidate if candidate > 0 else None


def _hole_sets_total(candidate: Any) -> tuple[int, bool]:
    total = 0
    found = False
    if isinstance(candidate, Mapping):
        if "hole_sets" in candidate:
            return _hole_sets_total(candidate.get("hole_sets"))
        qty_val = _coerce_positive_int(candidate.get("qty"))
        if qty_val is not None:
            total += qty_val
            found = True
        return (total, found)
    if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes, bytearray)):
        for item in candidate:
            subtotal, subfound = _hole_sets_total(item)
            total += subtotal
            found = found or subfound
        return (total, found)
    qty_val = _coerce_positive_int(candidate)
    if qty_val is not None:
        return (qty_val, True)
    return (0, False)


def ops_manifest(
    chart_rows: Iterable[Mapping[str, Any]] | None,
    hole_sets: Any = None,
) -> dict[str, Any]:
    """Return a normalized operation manifest from chart rows and geometry."""

    table_totals: dict[str, int] = {key: 0 for key in _OPS_MANIFEST_KEYS}
    row_count = 0
    sized_drill_qty = 0

    if chart_rows is not None:
        for row in chart_rows:
            if not isinstance(row, Mapping):
                continue
            qty = _coerce_positive_int(row.get("qty")) or 0
            if qty <= 0:
                continue
            row_count += 1
            desc_value = row.get("desc") or row.get("description") or row.get("text")
            operations = classify_op_row(desc_value)
            if not operations:
                table_totals["unknown"] += qty
                continue
            for op in operations:
                kind = str(op.get("kind") or "unknown").strip().lower()
                if kind not in _OPS_MANIFEST_KEYS:
                    kind = "unknown"
                table_totals[kind] += qty
                if kind == "drill":
                    sized_drill_qty += qty

    geom_total, geom_found = _hole_sets_total(hole_sets)
    geom_drill_total = geom_total if geom_found else 0
    geom_unsized = max(geom_drill_total - sized_drill_qty, 0)

    total_totals = dict(table_totals)
    if geom_found:
        total_totals["drill"] = sized_drill_qty + geom_unsized
    manifest: dict[str, Any] = {
        "table": table_totals,
        "total": total_totals,
        "chart_row_count": row_count,
        "text": {"estimated_total_drills": int(table_totals.get("drill", 0))},
    }
    if geom_found:
        manifest["geom"] = {
            "drill": geom_drill_total,
            "residual_drill": geom_unsized,
            "drill_residual": geom_unsized,
            "total": geom_drill_total,
        }
        manifest["geom_drill_count"] = geom_drill_total
    manifest["chart_drill_sized"] = sized_drill_qty
    return manifest


def _norm_row_key(row: Mapping[str, Any] | Any) -> tuple[int, str]:
    qty_value: Any = None
    desc_source: str = ""
    if isinstance(row, Mapping):
        qty_value = row.get("qty")
        desc_source = str(row.get("desc") or "")
    else:
        qty_value = getattr(row, "qty", None)
        desc_source = str(getattr(row, "desc", "") or "")
    try:
        qty_int = int(float(qty_value or 0))
    except Exception:
        qty_int = 0
    desc_normalized = " ".join(desc_source.split()).upper()
    return (qty_int, desc_normalized)


def _unique_rows_in_order(
    row_sources: Iterable[Iterable[Mapping[str, Any]] | None]
) -> tuple[list[dict[str, Any]], int]:
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    dropped = 0
    for source in row_sources:
        if not source:
            continue
        for row in source:
            if not isinstance(row, Mapping):
                continue
            key = _norm_row_key(row)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            unique_rows.append(dict(row))
    return unique_rows, dropped


def _combine_text_rows(
    anchor_rows: Iterable[Mapping[str, Any]] | None,
    primary_rows: Iterable[Mapping[str, Any]] | None,
    roi_rows: Iterable[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], int, bool]:
    authoritative_rows: list[dict[str, Any]] = []
    if anchor_rows:
        for row in anchor_rows:
            if isinstance(row, Mapping):
                authoritative_rows.append(dict(row))
        if authoritative_rows:
            return authoritative_rows, 0, True

    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    dedup_dropped = 0
    for source in (primary_rows, roi_rows):
        if not source:
            continue
        for row in source:
            if not isinstance(row, Mapping):
                continue
            key = _norm_row_key(row)
            if key in seen:
                dedup_dropped += 1
                continue
            seen.add(key)
            merged.append(dict(row))
    return merged, dedup_dropped, False


def _compute_anchor_h(entries: Iterable[Mapping[str, Any]]) -> tuple[float, int]:
    heights: list[float] = []
    count = 0
    for entry in entries:
        text_value = entry.get("normalized_text") or entry.get("text") or ""
        normalized = _normalize_candidate_text(text_value)
        if not _ROW_ANCHOR_RE.match(normalized):
            continue
        height_val = entry.get("height")
        if not isinstance(height_val, (int, float)):
            continue
        height_float = float(height_val)
        if height_float <= 0:
            continue
        heights.append(height_float)
        count += 1
    if not heights:
        return (0.0, 0)
    try:
        anchor_h = float(statistics.median(heights))
    except Exception:
        anchor_h = float(heights[0])
    return (anchor_h, count)


def _filter_entries_by_anchor_h(
    entries: Iterable[Mapping[str, Any]],
    *,
    anchor_h: float,
    tolerance: float = 0.4,
) -> list[Mapping[str, Any]]:
    effective_anchor = float(anchor_h or 0.0)
    if effective_anchor > 0:
        effective_anchor = max(effective_anchor, _GEO_H_ANCHOR_MIN)
    lower = max(effective_anchor * (1.0 - tolerance), _GEO_H_ANCHOR_HARD_MIN)
    upper = effective_anchor * (1.0 + tolerance)
    filtered: list[Mapping[str, Any]] = []
    if effective_anchor <= 0:
        return list(entries)
    for entry in entries:
        height_val = entry.get("height")
        if not isinstance(height_val, (int, float)):
            filtered.append(entry)
            continue
        height_float = float(height_val)
        if height_float <= 0:
            filtered.append(entry)
            continue
        if lower <= height_float <= upper:
            filtered.append(entry)
    return filtered


def _extract_row_quantity_and_remainder(text: str) -> tuple[int | None, str]:
    base = (text or "").strip()
    if not base:
        return (None, "")

    primary_match = _match_row_quantity(base)
    if primary_match:
        qty_text = primary_match.group(1)
        try:
            qty_val = int(qty_text)
        except Exception:
            qty_val = None
        remainder = base[primary_match.end() :].strip()
        return (qty_val, remainder)

    return (None, base)


def _extract_column_quantity_and_remainder(text: str) -> tuple[int | None, str]:
    base = (text or "").strip()
    if not base:
        return (None, "")

    def _match_any(candidate: str) -> re.Match[str] | None:
        for pattern in _ROW_QUANTITY_PATTERNS:
            match = pattern.search(candidate)
            if match:
                return match
        return None

    def _strip_span(source: str, span: tuple[int, int]) -> str:
        start, end = span
        return (source[:start] + " " + source[end:]).strip()

    primary_match = _match_any(base)
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
        remainder_match = _match_any(remainder_body)
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
    search_space = desc or ""
    diameter = _extract_diameter(search_space)
    if diameter is not None and 0 < diameter <= 10:
        return (_format_ref_value(diameter), diameter)

    thread_match = _THREAD_CALL_OUT_RE.search(search_space)
    if thread_match:
        return (thread_match.group(0).upper(), None)

    pipe_match = _PIPE_NPT_REF_RE.search(search_space)
    if pipe_match:
        numeric_part = pipe_match.group(1)
        suffix = pipe_match.group(2)
        compact = f"{numeric_part}-{suffix}".upper().replace(" ", "")
        return (compact, None)

    numbered_thread = _NUMBERED_THREAD_REF_RE.search(search_space)
    if numbered_thread:
        raw_value = numbered_thread.group(0)
        normalized = raw_value.upper().replace(" ", "")
        return (normalized, None)

    number_drill = _NUMBER_DRILL_REF_RE.search(search_space)
    if number_drill:
        return (number_drill.group(0).upper(), None)

    letter_drill = _LETTER_DRILL_REF_RE.search(search_space)
    if letter_drill:
        return (letter_drill.group(1).upper(), None)

    inch_match = _INCH_MARK_REF_RE.search(search_space)
    if inch_match:
        value = _parse_number_token(inch_match.group(1))
        if value is not None and 0 < value <= 10:
            return (_format_ref_value(value), value)

    dia_inline = _DIA_SYMBOL_INLINE_RE.search(search_space)
    if dia_inline:
        value = _parse_number_token(dia_inline.group(1))
        if value is not None and 0 < value <= 10:
            return (_format_ref_value(value), value)

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


def _line_is_table_row_start(text: str) -> bool:
    if not text:
        return False
    candidate = str(text).strip()
    if not candidate:
        return False
    for pattern in _ROW_QUANTITY_PATTERNS:
        if pattern.match(candidate):
            return True
    if re.match(r"^\s*QTY\b", candidate, re.IGNORECASE):
        return True
    return False


def _merge_table_lines(lines: Iterable[str]) -> list[str]:
    merged: list[str] = []
    current: list[str] = []
    for raw_line in lines:
        candidate = " ".join(str(raw_line or "").split()).strip()
        if not candidate:
            continue
        if _roi_is_admin_noise(candidate) or _roi_is_numeric_ladder(candidate):
            continue
        starts_new_row = False
        if _ROW_QUANTITY_PATTERNS[0].match(candidate) or _ROW_QUANTITY_PATTERNS[1].match(candidate):
            starts_new_row = True
        elif not current and _line_is_table_row_start(candidate):
            starts_new_row = True
        if starts_new_row:
            if current:
                merged.append(" ".join(current))
            current = [candidate]
            continue
        if current:
            current.append(candidate)
        else:
            current = [candidate]
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


def _build_columnar_table_from_panel_entries(
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
                "rows_txt_fallback": [],
                "qty_col": None,
                "ref_col": None,
                "desc_col": None,
                "roi": None,
                "row_debug": [],
                "columns": [],
            },
        )

    base_records = list(records)
    records_all = list(base_records)
    roi_bounds: dict[str, float] | None = None
    roi_info: dict[str, Any] | None = None
    roi_median_height = 0.0
    follow_sheet_requests: dict[str, Any] = {}

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

    y_gap = 0.75 * median_h if median_h > 0 else 4.0
    if y_gap <= 0:
        y_gap = 4.0

    normalized_cells: list[tuple[float, float, float, str]] = []
    for record in records:
        text_value = str(record.get("text") or "").strip()
        if not text_value:
            continue
        x_raw = record.get("x")
        y_raw = record.get("y")
        if not isinstance(x_raw, (int, float)) or not isinstance(y_raw, (int, float)):
            continue
        height_raw = record.get("height")
        if isinstance(height_raw, (int, float)) and float(height_raw) > 0:
            height_val = float(height_raw)
        else:
            height_val = median_h if median_h > 0 else median_height_all or 4.0
        normalized_cells.append((float(x_raw), float(y_raw), float(height_val), text_value))

    if not normalized_cells:
        debug_info = {
            "rows_txt_fallback": [],
            "median_h": median_h,
            "roi": roi_info,
            "row_gap": y_gap,
            "columns": [],
        }
        return (None, debug_info)

    class _RowBuffer:
        __slots__ = ("cells", "y_values")

        def __init__(self, cell: tuple[float, float, float, str]) -> None:
            self.cells: list[tuple[float, float, float, str]] = [cell]
            self.y_values: list[float] = [cell[1]]

        def add(self, cell: tuple[float, float, float, str]) -> None:
            self.cells.append(cell)
            self.y_values.append(cell[1])

        @property
        def center(self) -> float:
            return sum(self.y_values) / len(self.y_values)

    sorted_cells = sorted(normalized_cells, key=lambda item: (-item[1], item[0]))
    row_buffers: list[_RowBuffer] = []
    for cell in sorted_cells:
        if not row_buffers:
            row_buffers.append(_RowBuffer(cell))
            continue
        current = row_buffers[-1]
        if abs(cell[1] - current.center) <= y_gap:
            current.add(cell)
            continue
        row_buffers.append(_RowBuffer(cell))

    def _row_span(buffer: _RowBuffer) -> float:
        xs = [cell[0] for cell in buffer.cells]
        if len(xs) < 2:
            return 0.0
        return max(xs) - min(xs)

    def _column_centers_from_rows(rows: list[_RowBuffer]) -> list[float]:
        if not rows:
            return []
        span_target = max(rows, key=lambda row: (len(row.cells), _row_span(row)))
        xs_sorted = sorted(cell[0] for cell in span_target.cells)
        centers: list[float] = []
        min_gap = max(2.0, 0.4 * (median_h if median_h > 0 else median_height_all or 4.0))
        for x_val in xs_sorted:
            if not centers:
                centers.append(x_val)
                continue
            if abs(x_val - centers[-1]) <= min_gap:
                centers[-1] = (centers[-1] + x_val) / 2.0
            else:
                centers.append(x_val)
        if not centers:
            centers = xs_sorted
        return centers

    column_centers = _column_centers_from_rows(row_buffers)
    if not column_centers:
        column_centers = sorted({cell[0] for cell in sorted_cells})

    def _snap_row(buffer: _RowBuffer, centers: list[float]) -> tuple[list[str], list[list[tuple[float, float, float, str]]]]:
        if not centers:
            return [], []
        assignments: list[list[tuple[float, float, float, str]]] = [
            [] for _ in centers
        ]
        for cell in sorted(buffer.cells, key=lambda item: item[0]):
            nearest_index = min(
                range(len(centers)),
                key=lambda idx: (abs(cell[0] - centers[idx]), idx),
            )
            assignments[nearest_index].append(cell)
        cell_texts = [
            " ".join(part[3] for part in bucket).strip() if bucket else ""
            for bucket in assignments
        ]
        return cell_texts, assignments

    snapped_rows: list[dict[str, Any]] = []
    for row_index, buffer in enumerate(row_buffers):
        cell_texts, assignments = _snap_row(buffer, column_centers)
        if not cell_texts and buffer.cells:
            cell_texts = [" ".join(part[3] for part in sorted(buffer.cells, key=lambda item: item[0])).strip()]
        row_center_y = sum(buffer.y_values) / len(buffer.y_values)
        snapped_rows.append(
            {
                "index": row_index,
                "y": row_center_y,
                "cells": cell_texts,
                "assignments": assignments,
            }
        )

    if not snapped_rows:
        debug_info = {
            "rows_txt_fallback": [],
            "median_h": median_h,
            "roi": roi_info,
            "row_gap": y_gap,
            "columns": column_centers,
        }
        return (None, debug_info)

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
            if "HOLE" in upper or re.search(r"ID", upper):
                hits.setdefault("hole", idx)
        return hits

    header_rows: dict[int, dict[str, int]] = {}
    header_cols: dict[str, int] = {}
    for row in snapped_rows:
        hits = _header_hits(row.get("cells", []))
        if not hits:
            continue
        if "qty" not in hits and len(hits) < 2:
            continue
        header_rows[row["index"]] = hits
        for field, col_idx in hits.items():
            header_cols.setdefault(field, col_idx)

    header_row_indices = set(header_rows)
    column_count = len(column_centers)

    def _normalize_cell(text: str) -> str:
        return " ".join(text.split())

    column_metrics: list[dict[str, Any]] = []
    for col_idx in range(column_count):
        values = []
        for row in snapped_rows:
            cells = row.get("cells", [])
            if col_idx < len(cells):
                values.append(cells[col_idx])
        non_empty = [value.strip() for value in values if value and value.strip()]
        numeric_hits = sum(1 for value in non_empty if _parse_qty_cell_text(value) is not None)
        qty_hits = sum(1 for value in non_empty if _ROW_QUANTITY_PATTERNS[0].match(value))
        ref_hits = sum(1 for value in non_empty if _cell_has_ref_marker(value))
        side_hits = sum(1 for value in non_empty if _detect_row_side(value))
        avg_len = statistics.mean(len(value) for value in non_empty) if non_empty else 0.0
        column_metrics.append(
            {
                "non_empty": len(non_empty),
                "numeric_hits": numeric_hits,
                "qty_hits": qty_hits,
                "ref_hits": ref_hits,
                "side_hits": side_hits,
                "avg_len": avg_len,
            }
        )

    qty_col = header_cols.get("qty")
    if qty_col is None and column_metrics:
        candidate = max(
            range(column_count),
            key=lambda idx: (
                column_metrics[idx]["qty_hits"],
                column_metrics[idx]["numeric_hits"],
                column_metrics[idx]["non_empty"],
                -idx,
            ),
        )
        if column_metrics[candidate]["qty_hits"] > 0 or column_metrics[candidate]["numeric_hits"] > 0:
            qty_col = candidate

    ref_col = header_cols.get("ref")
    if ref_col is None and column_metrics:
        ref_candidates = [idx for idx in range(column_count) if idx != qty_col]
        if not ref_candidates:
            ref_candidates = list(range(column_count))
        candidate = max(
            ref_candidates,
            key=lambda idx: (
                column_metrics[idx]["ref_hits"],
                column_metrics[idx]["non_empty"],
                -idx,
            ),
        )
        if column_metrics[candidate]["ref_hits"] > 0:
            ref_col = candidate

    side_col = header_cols.get("side")
    if side_col is None and column_metrics:
        side_candidates = [
            idx
            for idx in range(column_count)
            if idx not in {qty_col, ref_col}
        ]
        if not side_candidates:
            side_candidates = list(range(column_count))
        candidate = max(
            side_candidates,
            key=lambda idx: (
                column_metrics[idx]["side_hits"],
                column_metrics[idx]["non_empty"],
                -idx,
            ),
        )
        if column_metrics[candidate]["side_hits"] > 0:
            side_col = candidate

    desc_col = header_cols.get("desc")
    occupied = {idx for idx in (qty_col, ref_col, side_col) if isinstance(idx, int)}
    if desc_col is None and column_metrics:
        candidates = [idx for idx in range(column_count) if idx not in occupied]
        if not candidates:
            candidates = list(range(column_count))
        desc_col = max(
            candidates,
            key=lambda idx: (
                column_metrics[idx]["avg_len"],
                column_metrics[idx]["non_empty"],
                -idx,
            ),
        )

    row_debug_entries = [
        {
            "index": row["index"],
            "y": row["y"],
            "cells": row.get("cells", []),
        }
        for row in snapped_rows
    ]

    base_rows: list[dict[str, Any]] = []
    base_row_keys: set[tuple[int, str]] = set()

    for row in snapped_rows:
        row_index = row["index"]
        if row_index in header_row_indices:
            continue
        cells = [cell.strip() for cell in row.get("cells", [])]
        if not any(cells):
            continue
        qty_text = ""
        if isinstance(qty_col, int) and qty_col < len(cells):
            qty_text = cells[qty_col]
        qty_val = _parse_qty_cell_text(qty_text) if qty_text else None
        desc_idx = desc_col if isinstance(desc_col, int) else None
        desc_text = cells[desc_idx] if desc_idx is not None and desc_idx < len(cells) else ""
        combined_text = " ".join(value for value in cells if value)
        inline_qty_detail: dict[str, Any] | None = None
        if (
            isinstance(desc_idx, int)
            and desc_idx == qty_col
            and desc_text
        ):
            inline_qty_val, inline_remainder = _extract_column_quantity_and_remainder(desc_text)
            if inline_qty_val is not None and inline_qty_val > 0:
                inline_qty_detail = {
                    "value": int(inline_qty_val),
                    "source": "desc",
                    "remainder": inline_remainder.strip(),
                }
                if inline_qty_detail["remainder"]:
                    desc_text = inline_qty_detail["remainder"]
                if qty_val is None or qty_val <= 0:
                    qty_val = inline_qty_val
        if qty_val is None or qty_val <= 0:
            source = None
            inline_qty_val = None
            remainder = ""
            if desc_text:
                inline_qty_val, remainder = _extract_column_quantity_and_remainder(desc_text)
            source = "desc"
            if inline_qty_val is None or inline_qty_val <= 0:
                inline_qty_val, remainder = _extract_column_quantity_and_remainder(combined_text)
                source = "combined"
            if inline_qty_val is not None and inline_qty_val > 0:
                qty_val = inline_qty_val
                inline_qty_detail = {
                    "value": int(inline_qty_val),
                    "source": source,
                    "remainder": remainder.strip(),
                }
                if not desc_text or (desc_idx == qty_col and inline_qty_detail["remainder"]):
                    desc_text = inline_qty_detail["remainder"] or desc_text
        if qty_val is None or qty_val <= 0:
            continue
        try:
            qty_int = int(qty_val)
        except Exception:
            continue
        if qty_int <= 0:
            continue
        if not desc_text:
            fallback_parts = [
                value
                for idx, value in enumerate(cells)
                if idx != qty_col and value
            ]
            desc_text = " ".join(fallback_parts) if fallback_parts else combined_text
        desc_text = _normalize_cell(desc_text)
        ref_idx = header_cols.get("ref", ref_col)
        ref_cell_text = (
            cells[ref_idx]
            if isinstance(ref_idx, int) and ref_idx < len(cells)
            else ""
        )
        ref_text, ref_value = _extract_row_reference(ref_cell_text or desc_text)
        if not ref_text:
            alt_ref_text, alt_ref_value = _extract_row_reference(desc_text)
            if alt_ref_text:
                ref_text = alt_ref_text
            if ref_value is None and alt_ref_value is not None:
                ref_value = alt_ref_value
        side_idx = header_cols.get("side", side_col)
        side_cell_text = (
            cells[side_idx]
            if isinstance(side_idx, int) and side_idx < len(cells)
            else ""
        )
        side_value = _detect_row_side(" ".join(filter(None, [side_cell_text, desc_text])))
        row_entry: dict[str, Any] = {
            "hole": "",
            "qty": qty_int,
            "desc": desc_text,
            "ref": ref_text or "",
        }
        if side_value:
            row_entry["side"] = side_value
        if inline_qty_detail:
            row_entry["inline_qty"] = inline_qty_detail
        dedupe_key = (qty_int, " ".join(desc_text.split()).upper())
        if dedupe_key in base_row_keys:
            continue
        base_row_keys.add(dedupe_key)
        base_rows.append(row_entry)
        preview_cols = ", ".join(
            f"{idx}:{_truncate_cell_preview(value)}" for idx, value in enumerate(cells)
        )
        print(
            f"[TABLE-R] row#{row_index} qty={qty_int} cols=[{preview_cols}]"
        )

    rows_output: list[dict[str, Any]] = [dict(row_entry) for row_entry in base_rows]

    if not rows_output:
        debug_info = {
            "rows_txt_fallback": [],
            "median_h": median_h,
            "roi": roi_info,
            "row_gap": y_gap,
            "columns": column_centers,
            "header_rows": sorted(header_row_indices),
            "header_cols": header_cols,
        }
        return (None, debug_info)

    qty_sum = sum(int(row.get("qty") or 0) for row in rows_output)
    families: dict[str, int] = {}
    for row_entry in rows_output:
        ref_text, ref_value = _extract_row_reference(row_entry.get("ref") or row_entry.get("desc") or "")
        if ref_text:
            row_entry["ref"] = ref_text
        if ref_value is not None:
            key = f"{ref_value:.4f}".rstrip("0").rstrip(".")
            families[key] = families.get(key, 0) + int(row_entry.get("qty", 0))

    table_info: dict[str, Any] = {
        "rows": rows_output,
        "hole_count": qty_sum,
        "sum_qty": qty_sum,
        "provenance_holes": "HOLE TABLE",
        "source": "text_table",
    }
    if families:
        table_info["hole_diam_families_in"] = families

    debug_info = {
        "rows_txt_fallback": rows_output,
        "median_h": median_h,
        "row_gap": y_gap,
        "columns": column_centers,
        "header_rows": sorted(header_row_indices),
        "header_cols": header_cols,
        "qty_col": qty_col,
        "ref_col": ref_col,
        "desc_col": desc_col,
        "side_col": side_col,
        "qty_sum": qty_sum,
        "row_debug": row_debug_entries,
    }
    if roi_info is not None:
        debug_info["roi"] = roi_info
    return (table_info, debug_info)


def _extract_mechanical_table_from_blocks(doc: Any) -> Mapping[str, Any] | None:
    helper = _resolve_app_callable("extract_hole_table_from_text")
    if not callable(helper):
        return None

    blocks_section = getattr(doc, "blocks", None)
    if blocks_section is None:
        return None

    try:
        block_iter = list(blocks_section)
    except Exception:
        block_iter = []

    def _is_mechanical_name(name: str) -> bool:
        upper = name.upper()
        return upper.startswith("AM_") or upper.startswith("*U")

    def _extract_text(entity: Any) -> str:
        if entity is None:
            return ""
        plain = getattr(entity, "plain_text", None)
        text_value: Any = None
        if callable(plain):
            try:
                text_value = plain()
            except Exception:
                text_value = None
        if not text_value:
            dxf_obj = getattr(entity, "dxf", None)
            text_value = getattr(dxf_obj, "text", None) if dxf_obj is not None else None
        try:
            return str(text_value).strip()
        except Exception:
            return ""

    def _extract_xy(entity: Any) -> tuple[float | None, float | None]:
        if entity is None:
            return (None, None)
        dxf_obj = getattr(entity, "dxf", None)
        point = None
        for source in (entity, dxf_obj):
            if source is None:
                continue
            for attr in ("insert", "alignment_point", "align_point", "start", "position"):
                candidate = getattr(source, attr, None)
                if candidate is not None:
                    point = candidate
                    break
            if point is not None:
                break
        if point is None:
            return (None, None)

        def _coerce(value: Any, attr: str | None = None) -> float | None:
            target = value
            if attr is not None:
                target = getattr(value, attr, None)
            if target is None:
                return None
            try:
                return float(target)
            except Exception:
                return None

        if hasattr(point, "xyz"):
            try:
                x_val, y_val, _ = point.xyz
                return (float(x_val), float(y_val))
            except Exception:
                return (None, None)

        for accessor in ((0, 1), ("x", "y")):
            if isinstance(accessor[0], int):
                try:
                    x_val = float(point[accessor[0]])  # type: ignore[index]
                except Exception:
                    x_val = None
            else:
                x_val = _coerce(point, accessor[0])
            if isinstance(accessor[1], int):
                try:
                    y_val = float(point[accessor[1]])  # type: ignore[index]
                except Exception:
                    y_val = None
            else:
                y_val = _coerce(point, accessor[1])
            if x_val is not None or y_val is not None:
                return (x_val, y_val)
        return (None, None)

    def _extract_height(entity: Any) -> float | None:
        dxf_obj = getattr(entity, "dxf", None)
        candidates: list[Any] = []
        if dxf_obj is not None:
            candidates.extend(
                getattr(dxf_obj, attr, None) for attr in ("height", "char_height", "text_height")
            )
        candidates.append(getattr(entity, "height", None))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                value = float(candidate)
            except Exception:
                continue
            if value > 0:
                return value
        return None

    best_result: Mapping[str, Any] | None = None
    best_rows = 0

    for block in block_iter:
        name = getattr(block, "name", None)
        if not isinstance(name, str):
            continue
        if not _is_mechanical_name(name):
            continue
        try:
            entities = list(block)
        except Exception:
            entities = []
        texts: list[dict[str, Any]] = []
        for entity in entities:
            try:
                kind = entity.dxftype()
            except Exception:
                kind = None
            if str(kind or "").upper() not in {"TEXT", "MTEXT", "RTEXT"}:
                continue
            text_value = _extract_text(entity)
            if not text_value:
                continue
            x_val, y_val = _extract_xy(entity)
            height_val = _extract_height(entity)
            texts.append(
                {
                    "text": text_value,
                    "x": x_val,
                    "y": y_val,
                    "height": height_val,
                }
            )
        headers_detected: list[str] = []
        if texts:
            heights = [item["height"] for item in texts if isinstance(item.get("height"), (int, float))]
            median_height = statistics.median(heights) if heights else None
            y_tol = max((median_height or 0.0) * 2.0, 0.25)
            clusters: list[dict[str, Any]] = []
            for entry in texts:
                y_val = entry.get("y")
                if not isinstance(y_val, (int, float)):
                    continue
                upper = str(entry.get("text") or "").upper()
                tokens: set[str] = set()
                if "HOLE" in upper:
                    tokens.add("HOLE")
                if "REF" in upper or "Ø" in upper or "DIA" in upper:
                    tokens.add("REF")
                if "QTY" in upper or "QUANTITY" in upper:
                    tokens.add("QTY")
                if "DESC" in upper or "DESCRIPTION" in upper:
                    tokens.add("DESC")
                if not tokens:
                    continue
                placed = False
                for cluster in clusters:
                    center = cluster["y"]
                    if center is not None and abs(float(y_val) - float(center)) <= y_tol:
                        cluster["tokens"].update(tokens)
                        cluster["values"].append(entry)
                        placed = True
                        break
                if not placed:
                    clusters.append(
                        {
                            "y": float(y_val),
                            "tokens": set(tokens),
                            "values": [entry],
                        }
                    )
            clusters = [c for c in clusters if len(c["tokens"]) >= 1]
            best_cluster = None
            for cluster in clusters:
                if len(cluster["tokens"]) >= 3:
                    if best_cluster is None or len(cluster["tokens"]) > len(best_cluster["tokens"]):
                        best_cluster = cluster
            if best_cluster is not None:
                headers_detected = sorted(best_cluster["tokens"])
        headers_display = f"{headers_detected}" if headers_detected else "[]"
        print(f"[AMTABLE] block={name} texts={len(texts)} headers={headers_display}")
        if len(headers_detected) < 3:
            continue

        class _BlockTextEntity:
            __slots__ = ("dxf", "_kind", "_text")

            def __init__(self, record: Mapping[str, Any]):
                self._text = str(record.get("text") or "")
                self._kind = "MTEXT"
                x_val = record.get("x")
                y_val = record.get("y")
                if not isinstance(x_val, (int, float)):
                    x_val = 0.0
                if not isinstance(y_val, (int, float)):
                    y_val = 0.0
                self.dxf = SimpleNamespace(
                    text=self._text,
                    insert=SimpleNamespace(
                        x=float(x_val),
                        y=float(y_val),
                        xyz=(float(x_val), float(y_val), 0.0),
                    ),
                )

            def dxftype(self) -> str:
                return self._kind

            def plain_text(self) -> str:
                return self._text

        class _BlockTextSpace:
            __slots__ = ("_entities",)

            def __init__(self, records: Iterable[Mapping[str, Any]]):
                self._entities = [
                    _BlockTextEntity(rec)
                    for rec in records
                    if str(rec.get("text") or "").strip()
                ]

            def query(self, _pattern: str):
                return list(self._entities)

        class _BlockLayouts:
            __slots__ = ()

            def names_in_taborder(self):
                return []

            def get(self, _name: str):
                return SimpleNamespace(entity_space=None)

        class _BlockDoc:
            __slots__ = ("_space", "layouts")

            def __init__(self, records: Iterable[Mapping[str, Any]]):
                self._space = _BlockTextSpace(records)
                self.layouts = _BlockLayouts()

            def modelspace(self):
                return self._space

        fake_doc = _BlockDoc(texts)
        try:
            candidate = helper(fake_doc)
        except Exception:
            candidate = None
        if isinstance(candidate, Mapping) and candidate.get("rows"):
            row_count = len(candidate.get("rows", []))
            if row_count > best_rows:
                best_rows = row_count
                best_result = dict(candidate)

    return best_result


_FALLBACK_ACTION_KINDS = {
    "tap",
    "counterbore",
    "drill",
    "counterdrill",
    "jig_grind",
    "csink",
    "spot",
}
_FALLBACK_TRAILING_LADDER_RE = re.compile(r"(?:\s*\d+){3,}$")
_ANCHOR_TERMINATOR_RE = re.compile(
    r"^(?:NOTES?|TOLERANCES?|REVISION|TITLE|DRAWN\s+BY)\b",
    re.IGNORECASE,
)


def _prepare_fallback_lines(lines: Iterable[str]) -> list[str]:
    try:
        raw_lines = list(lines)
    except Exception:
        raw_lines = []

    cleaned: list[str] = []
    for raw_line in raw_lines:
        candidate = " ".join(str(raw_line or "").split()).strip()
        if not candidate:
            continue
        if _roi_is_admin_noise(candidate) or _roi_is_numeric_ladder(candidate):
            continue
        cleaned.append(candidate)

    stitched: list[str] = []
    current: list[str] = []
    for line in cleaned:
        if _FALLBACK_ROW_START_RE.match(line):
            if current:
                stitched.append(" ".join(current).strip())
            current = [line]
            continue
        if current:
            current.append(line)
    if current:
        stitched.append(" ".join(current).strip())

    if not stitched and cleaned:
        stitched = _merge_table_lines(cleaned)

    return stitched


def _coerce_float_optional(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if not math.isfinite(result):
        return None
    return result


def _coerce_layout_index(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def _normalize_entry_text(entry: Mapping[str, Any]) -> str:
    text = entry.get("normalized_text")
    if text:
        return _normalize_candidate_text(text)
    return _normalize_candidate_text(entry.get("text"))


def _entry_band_sort_key(entry: Mapping[str, Any]) -> tuple[float, float, int]:
    x_val = entry.get("x")
    y_val = entry.get("y")
    order_val = entry.get("order", 0)
    try:
        y_key = -float(y_val) if y_val is not None else float("inf")
    except Exception:
        y_key = float("inf")
    try:
        x_key = float(x_val) if x_val is not None else float("inf")
    except Exception:
        x_key = float("inf")
    try:
        order_key = int(order_val)
    except Exception:
        order_key = 0
    return (y_key, x_key, order_key)


def _extract_anchor_band_lines(context: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(context, Mapping):
        return []

    raw_entries = context.get("entries")
    raw_anchor_entries = context.get("anchor_entries")
    if not isinstance(raw_entries, Iterable) or not isinstance(raw_anchor_entries, Iterable):
        return []

    entries: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue
        entry_copy = dict(entry)
        entry_copy["normalized_text"] = _normalize_entry_text(entry_copy)
        entries.append(entry_copy)

    anchor_entries: list[dict[str, Any]] = []
    for entry in raw_anchor_entries:
        if not isinstance(entry, Mapping):
            continue
        entry_copy = dict(entry)
        entry_copy["normalized_text"] = _normalize_entry_text(entry_copy)
        anchor_entries.append(entry_copy)

    if not entries or not anchor_entries:
        return []

    anchor_layouts = {
        _coerce_layout_index(item.get("layout_index")) for item in anchor_entries
    }
    if not anchor_layouts:
        return []

    layout_order_raw = context.get("layout_order")
    layout_sequence: list[int]
    if isinstance(layout_order_raw, Iterable):
        layout_sequence = []
        for idx in layout_order_raw:
            layout_idx = _coerce_layout_index(idx)
            if layout_idx in anchor_layouts and layout_idx not in layout_sequence:
                layout_sequence.append(layout_idx)
    else:
        layout_sequence = []
    if not layout_sequence:
        layout_sequence = sorted(anchor_layouts)

    entries_by_layout: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        layout_idx = _coerce_layout_index(entry.get("layout_index"))
        if layout_idx in anchor_layouts:
            entries_by_layout[layout_idx].append(entry)

    anchors = [
        entry
        for entry in anchor_entries
        if _ROW_ANCHOR_RE.match(
            _normalize_candidate_text(
                entry.get("normalized_text") or entry.get("text") or ""
            )
        )
    ]
    anchor_h = 0.0
    try:
        anchor_h = float(
            context.get("anchor_h")
            if isinstance(context, Mapping)
            else 0.0
        )
        if anchor_h == 0.0 and isinstance(context, Mapping):
            legacy_anchor = context.get("anchor_height")
            if isinstance(legacy_anchor, (int, float)):
                anchor_h = float(legacy_anchor)
    except Exception:
        anchor_h = 0.0
    if anchor_h < 0:
        anchor_h = 0.0

    height_lower = anchor_h * 0.6 if anchor_h > 0 else None
    height_upper = anchor_h * 1.4 if anchor_h > 0 else None
    height_stop_lower = anchor_h * 0.3 if anchor_h > 0 else None
    height_stop_upper = anchor_h * 1.7 if anchor_h > 0 else None
    print(
        f"[TEXT-SCAN] anchors={len(anchors)} h_anchor={anchor_h:.2f}"
        if anchors
        else "[TEXT-SCAN] anchors=0"
    )

    anchor_y_values: list[float] = []
    anchor_x_bounds: dict[int, list[float | None]] = {}
    anchor_keys_by_layout: defaultdict[
        int, Counter[tuple[str, float | None, float | None]]
    ] = defaultdict(Counter)

    def _anchor_key(entry: Mapping[str, Any]) -> tuple[str, float | None, float | None] | None:
        normalized = _normalize_candidate_text(
            entry.get("normalized_text") or entry.get("text")
        )
        if not normalized:
            return None
        x_val = _coerce_float_optional(entry.get("x"))
        y_val = _coerce_float_optional(entry.get("y"))
        rounded_x = None if x_val is None else round(float(x_val), 3)
        rounded_y = None if y_val is None else round(float(y_val), 3)
        return (normalized.upper(), rounded_x, rounded_y)
    for entry in anchor_entries:
        y_val = _coerce_float_optional(entry.get("y"))
        if y_val is not None:
            anchor_y_values.append(y_val)
        layout_idx = _coerce_layout_index(entry.get("layout_index"))
        bounds = anchor_x_bounds.setdefault(layout_idx, [None, None])
        x_val = _coerce_float_optional(entry.get("x"))
        width_val = _coerce_float_optional(entry.get("width"))
        if x_val is None:
            continue
        right_val = x_val
        if width_val is not None and width_val > 0:
            right_val = x_val + width_val
        if bounds[0] is None or x_val < bounds[0]:
            bounds[0] = x_val
        if bounds[1] is None or (right_val is not None and right_val > bounds[1]):
            bounds[1] = right_val
        key = _anchor_key(entry)
        if key is not None:
            anchor_keys_by_layout[layout_idx][key] += 1

    anchor_center_y: float | None = None
    band_y_low: float | None = None
    band_y_high: float | None = None
    row_spacing_est = 0.0
    if anchor_y_values:
        sorted_anchor_y = sorted(anchor_y_values)
        try:
            anchor_center_y = float(statistics.median(sorted_anchor_y))
        except Exception:
            anchor_center_y = float(sorted_anchor_y[0])
        diffs = [
            abs(sorted_anchor_y[idx] - sorted_anchor_y[idx - 1])
            for idx in range(1, len(sorted_anchor_y))
            if abs(sorted_anchor_y[idx] - sorted_anchor_y[idx - 1]) > 0
        ]
        if diffs:
            try:
                row_spacing_est = float(statistics.median(diffs))
            except Exception:
                row_spacing_est = float(diffs[0])
        if (not row_spacing_est or row_spacing_est <= 0) and len(sorted_anchor_y) >= 2:
            span = abs(sorted_anchor_y[-1] - sorted_anchor_y[0])
            divider = max(len(sorted_anchor_y) - 1, 1)
            row_spacing_est = span / divider if divider else span
        if (not row_spacing_est or row_spacing_est <= 0) and anchor_h > 0:
            row_spacing_est = anchor_h * 2.5
        if not row_spacing_est or row_spacing_est <= 0:
            row_spacing_est = anchor_h if anchor_h > 0 else 1.0
        band_half_height = row_spacing_est * 8.0
        if anchor_h > 0:
            band_half_height = max(band_half_height, anchor_h * 12.0)
        band_y_low = anchor_center_y - band_half_height
        band_y_high = anchor_center_y + band_half_height
    elif anchor_h > 0:
        row_spacing_est = anchor_h * 2.5

    if row_spacing_est <= 0 and anchor_h > 0:
        row_spacing_est = anchor_h
    if row_spacing_est <= 0:
        row_spacing_est = 1.0

    layout_x_extents: dict[int, tuple[float | None, float | None]] = {}
    for layout_idx, layout_entries in entries_by_layout.items():
        min_x: float | None = None
        max_x: float | None = None
        for entry in layout_entries:
            x_val = _coerce_float_optional(entry.get("x"))
            width_val = _coerce_float_optional(entry.get("width"))
            if x_val is None:
                continue
            right_val = x_val
            if width_val is not None and width_val > 0:
                right_val = x_val + width_val
            if min_x is None or x_val < min_x:
                min_x = x_val
            if max_x is None or (right_val is not None and right_val > max_x):
                max_x = right_val
        layout_x_extents[layout_idx] = (min_x, max_x)

    x_margin = 0.0
    if anchor_h > 0:
        x_margin = max(x_margin, anchor_h * 4.0)
    if row_spacing_est > 0:
        x_margin = max(x_margin, row_spacing_est * 0.5)

    band_x_bounds: dict[int, tuple[float | None, float | None]] = {}
    for layout_idx in layout_sequence:
        anchor_bounds = anchor_x_bounds.get(layout_idx, [None, None])
        layout_bounds = layout_x_extents.get(layout_idx, (None, None))
        left = anchor_bounds[0] if anchor_bounds else None
        right = anchor_bounds[1] if anchor_bounds else None
        if left is None:
            left = layout_bounds[0]
        if right is None:
            right = layout_bounds[1]
        if left is not None and right is not None and right < left:
            left, right = right, left
        if left is not None and x_margin:
            left -= x_margin
        if right is not None and x_margin:
            right += x_margin
        band_x_bounds[layout_idx] = (left, right)

    band_lines: list[str] = []
    for layout_idx in layout_sequence:
        records = entries_by_layout.get(layout_idx, [])
        if not records:
            continue
        sorted_records = sorted(records, key=_entry_band_sort_key)
        layout_lines: list[str] = []
        layout_anchor_counts = Counter(anchor_keys_by_layout.get(layout_idx, {}))
        anchors_needed = min(sum(layout_anchor_counts.values()), 3)
        anchors_seen = 0
        band_started = anchors_needed == 0
        started = False
        x_bounds = band_x_bounds.get(layout_idx)
        x_left_bound = x_bounds[0] if x_bounds else None
        x_right_bound = x_bounds[1] if x_bounds else None
        if (
            x_left_bound is not None
            and x_right_bound is not None
            and x_right_bound < x_left_bound
        ):
            x_left_bound, x_right_bound = x_right_bound, x_left_bound
        for record in sorted_records:
            y_val = _coerce_float_optional(record.get("y"))
            if (
                band_y_low is not None
                and band_y_high is not None
                and y_val is not None
            ):
                low = band_y_low if band_y_low <= band_y_high else band_y_high
                high = band_y_high if band_y_high >= band_y_low else band_y_low
                if y_val < low or y_val > high:
                    continue

            if x_left_bound is not None or x_right_bound is not None:
                x_val = _coerce_float_optional(record.get("x"))
                width_val = _coerce_float_optional(record.get("width"))
                left = x_val
                right = x_val
                if left is not None and width_val is not None and width_val > 0:
                    right = left + width_val
                if left is not None and right is not None:
                    if x_left_bound is not None and right < x_left_bound:
                        continue
                    if x_right_bound is not None and left > x_right_bound:
                        continue

            text = _normalize_entry_text(record)
            if not text:
                continue
            if _roi_is_admin_noise(text):
                continue
            anchor_key = _anchor_key(record)
            if not band_started and anchors_needed:
                if (
                    anchor_key is not None
                    and layout_anchor_counts.get(anchor_key, 0) > 0
                ):
                    layout_anchor_counts[anchor_key] -= 1
                    anchors_seen += 1
                    if anchors_seen >= anchors_needed:
                        band_started = True
                    continue
                if anchors_seen < anchors_needed:
                    continue
                band_started = True
            if band_started and anchor_key is not None:
                original_count = anchor_keys_by_layout.get(layout_idx, Counter()).get(
                    anchor_key, 0
                )
                if original_count:
                    continue
            if not started:
                if _line_is_table_row_start(text) or re.search(r"\bQTY\b", text, re.IGNORECASE):
                    started = True
                else:
                    continue
            if _ANCHOR_TERMINATOR_RE.search(text):
                break
            height_val = record.get("height")
            height_float = (
                float(height_val)
                if isinstance(height_val, (int, float)) and math.isfinite(height_val)
                else None
            )
            if anchor_h > 0 and height_float is not None and height_float > 0:
                if (
                    height_stop_lower is not None
                    and height_stop_upper is not None
                    and (height_float < height_stop_lower or height_float > height_stop_upper)
                ):
                    if layout_lines or started:
                        break
                    continue
                if (
                    (height_lower is not None and height_float < height_lower)
                    or (height_upper is not None and height_float > height_upper)
                ):
                    continue
            layout_lines.append(text)
        band_lines.extend(layout_lines)

    return band_lines


def _build_fallback_rows_from_lines(lines: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    rows: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    total_qty = 0
    seen_keys: set[tuple[int, str, str, str]] = set()

    for entry in lines:
        normalized_desc = " ".join(str(entry or "").split())
        if not normalized_desc:
            continue
        qty_val, remainder = _extract_row_quantity_and_remainder(normalized_desc)
        if qty_val is None or qty_val <= 0:
            continue
        try:
            qty_int = int(qty_val)
        except Exception:
            continue
        if qty_int <= 0:
            continue
        remainder_clean = " ".join(remainder.split())
        desc_text = remainder_clean or normalized_desc
        desc_text = _FALLBACK_LEADING_QTY_RE.sub("", desc_text)
        desc_text = _FALLBACK_JJ_NOISE_RE.sub("", desc_text)
        desc_text = _FALLBACK_ETCH_NOISE_RE.sub("", desc_text)
        desc_text = " ".join(desc_text.split()).strip()
        desc_text = _FALLBACK_TRAILING_LADDER_RE.sub("", desc_text).strip()
        if not desc_text:
            continue

        operations = classify_op_row(desc_text)
        op_kinds = {str(op.get("kind") or "").strip().lower() for op in operations}
        fragments = split_actions(desc_text) or [desc_text]
        candidate_fragments: list[tuple[str, dict[str, Any]]] = []
        for fragment in fragments:
            cleaned_fragment = " ".join(str(fragment or "").split()).strip()
            cleaned_fragment = _FALLBACK_TRAILING_LADDER_RE.sub("", cleaned_fragment).strip()
            if not cleaned_fragment:
                continue
            if _roi_is_admin_noise(cleaned_fragment) or _roi_is_numeric_ladder(cleaned_fragment):
                continue
            action = classify_action(cleaned_fragment)
            kind = str(action.get("kind") or "").strip().lower()
            is_relevant = kind in _FALLBACK_ACTION_KINDS or action.get("npt")
            if not is_relevant and len(fragments) > 1 and (op_kinds & _FALLBACK_ACTION_KINDS):
                # Skip unrelated fragments when we have explicit action rows.
                continue
            candidate_fragments.append((cleaned_fragment, action))

        if not candidate_fragments:
            continue

        base_ref_text, base_ref_value = _extract_row_reference(desc_text)
        side_hint = _detect_row_side(desc_text) or _detect_row_side(normalized_desc)

        for fragment_text, action in candidate_fragments:
            ref_text, ref_value = _extract_row_reference(fragment_text)
            side_value = action.get("side") or _detect_row_side(fragment_text) or side_hint
            if side_value == "both":
                sides_to_emit = ["front", "back"]
            else:
                fallback_side = side_value or "front"
                sides_to_emit = [fallback_side]

            for side_option in sides_to_emit:
                normalized_key = (
                    qty_int,
                    fragment_text.upper(),
                    (side_option or "").upper(),
                    (ref_text or base_ref_text or "").upper(),
                )
                if normalized_key in seen_keys:
                    continue
                seen_keys.add(normalized_key)

                row: dict[str, Any] = {
                    "hole": "",
                    "ref": ref_text or base_ref_text or "",
                    "qty": qty_int,
                    "desc": fragment_text,
                }
                if side_option:
                    row["side"] = side_option
                if action.get("npt"):
                    row["npt"] = True
                tap_type = action.get("tap_type")
                if tap_type:
                    row["tap_type"] = tap_type
                rows.append(row)
                total_qty += qty_int

                value_for_family = ref_value if ref_value is not None else base_ref_value
                if value_for_family is not None:
                    key = f"{float(value_for_family):.4f}".rstrip("0").rstrip(".")
                    families[key] = families.get(key, 0) + qty_int

    return rows, families, total_qty


def _fallback_rows_sample(rows: Sequence[Mapping[str, Any]] | None, limit: int = 3) -> str:
    if not rows:
        return "[]"
    preview: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        qty_value = row.get("qty")
        qty_text = ""
        if isinstance(qty_value, (int, float)):
            qty_int = int(round(float(qty_value)))
            if qty_int > 0:
                qty_text = f"{qty_int}× "
        elif qty_value is not None:
            qty_text = f"{qty_value}× "
        desc_value = row.get("desc") or row.get("description") or row.get("text") or ""
        desc_preview = _truncate_cell_preview(str(desc_value), 48)
        combined = f"{qty_text}{desc_preview}".strip()
        if combined:
            preview.append(combined)
        if len(preview) >= limit:
            break
    if not preview:
        return "[]"
    return f"[{', '.join(preview)}]"


def _detect_thread_token(text: str) -> str:
    search_space = text or ""
    thread_match = _THREAD_CALL_OUT_RE.search(search_space)
    if thread_match:
        return thread_match.group(0).upper()
    pipe_match = _PIPE_NPT_REF_RE.search(search_space)
    if pipe_match:
        numeric = pipe_match.group(1)
        suffix = pipe_match.group(2)
        return f"{numeric}-{suffix}".upper().replace(" ", "")
    numbered_thread = _NUMBERED_THREAD_REF_RE.search(search_space)
    if numbered_thread:
        return numbered_thread.group(0).upper().replace(" ", "")
    return ""


def _detect_diameter_token(text: str) -> str:
    search_space = text or ""
    for pattern in (_DIAMETER_PREFIX_RE, _DIAMETER_SUFFIX_RE):
        match = pattern.search(search_space)
        if match:
            return match.group(0).strip()
    size = _extract_drill_size(search_space)
    return size.strip() if size else ""


def _detect_depth_token(text: str) -> str:
    search_space = text or ""
    match = _DEBUG_DEPTH_RE.search(search_space)
    if match:
        token = match.group(0).strip()
        return token
    if _DEBUG_THRU_RE.search(search_space):
        return "THRU"
    return ""


def _fallback_debug_records(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    qty_sum = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        qty = _coerce_positive_int(row.get("qty")) or 0
        desc = (
            row.get("desc")
            or row.get("description")
            or row.get("text")
            or row.get("raw")
            or ""
        )
        desc_text = str(desc)
        operations = classify_op_row(desc_text)
        kind = ""
        for op in operations:
            candidate = str(op.get("kind") or "")
            if candidate and candidate not in {"unknown", "npt"}:
                kind = candidate
                break
        if not kind:
            kind = str(classify_action(desc_text).get("kind") or "")
        record = {
            "qty": qty,
            "kind": kind.lower(),
            "side": str(row.get("side") or ""),
            "tool": "",
            "diam_token": _detect_diameter_token(desc_text),
            "depth_token": _detect_depth_token(desc_text),
            "raw_text": desc_text,
        }
        if record["kind"] == "tap":
            record["tool"] = _detect_thread_token(desc_text)
        elif record["kind"] == "drill":
            record["tool"] = record["diam_token"] or _detect_thread_token(desc_text)
        else:
            record["tool"] = record["diam_token"] or _detect_thread_token(desc_text) or record["kind"]
        records.append(record)
        qty_sum += qty
    return records, qty_sum


def _write_fallback_debug(records: list[dict[str, Any]], qty_sum: int) -> None:
    if not records:
        return
    totals = {key: 0 for key in ("drill", "tap", "counterbore", "counterdrill", "jig_grind")}
    for record in records:
        raw_kind = str(record.get("kind") or "").lower()
        kind = {
            "cbore": "counterbore",
            "counterbore": "counterbore",
            "cdrill": "counterdrill",
            "counterdrill": "counterdrill",
        }.get(raw_kind, raw_kind)
        if kind in totals:
            try:
                qty_val = int(record.get("qty") or 0)
            except Exception:
                qty_val = 0
            totals[kind] += qty_val
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        with _DEBUG_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_DEBUG_FIELDNAMES)
            writer.writeheader()
            writer.writerows(records)
        with _DEBUG_TOTALS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(totals, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"[TABLE-DUMP] rows={len(records)} qty_sum={qty_sum} -> {_DEBUG_ROWS_PATH.as_posix()}"
        )
        print(f"[OPS] table totals -> {_DEBUG_TOTALS_PATH.as_posix()}")
    except Exception:
        pass


def _fallback_text_table(lines: Iterable[str]) -> dict[str, Any]:
    merged = _prepare_fallback_lines(lines)
    rows, families, total_qty = _build_fallback_rows_from_lines(merged)

    if not rows:
        return {}

    sample = _fallback_rows_sample(rows)
    print(
        f"[TEXT-FALLBACK] rebuilt rows={len(rows)} qty_sum={total_qty} sample={sample}"
    )

    try:
        records, qty_sum = _fallback_debug_records(rows)
        _write_fallback_debug(records, qty_sum)
    except Exception:
        pass

    result: dict[str, Any] = {"rows": rows, "hole_count": total_qty}
    if families:
        result["hole_diam_families_in"] = families
    result["provenance_holes"] = "HOLE TABLE"
    result["source"] = "text_fallback"
    return result


def _publish_fallback_from_rows_txt(rows_txt: Iterable[Any]) -> dict[str, Any]:
    merged = _prepare_fallback_lines(rows_txt)
    rows, families, total_qty = _build_fallback_rows_from_lines(merged)

    if not rows:
        return {}

    sample = _fallback_rows_sample(rows)
    print(
        f"[TEXT-FALLBACK] rebuilt rows={len(rows)} qty_sum={total_qty} sample={sample}"
    )

    result: dict[str, Any] = {
        "rows": rows,
        "hole_count": total_qty,
        "provenance_holes": "HOLE TABLE (fallback)",
        "source": "text_fallback",
    }
    if families:
        result["hole_diam_families_in"] = families

    try:
        records, qty_sum = _fallback_debug_records(rows)
        _write_fallback_debug(records, qty_sum)
    except Exception:
        pass
    return result


def extract_hole_table_from_text(
    doc: Any,
    rows_txt: Iterable[Any] | None = None,
    *,
    min_rows: int = 2,
) -> dict[str, Any]:
    """Fallback text helper for hole table extraction.

    The helper mirrors the contract expected by :func:`read_text_table` while
    avoiding crashes when no usable rows are discovered. Callers may supply raw
    ``rows_txt`` (typically pre-merged lines). When omitted, the helper will
    attempt to scan the provided ``doc`` for text entities using
    :func:`_collect_table_text_lines`.
    """

    rows: list[dict[str, Any]] = []
    total_qty = 0

    if rows_txt is None:
        try:
            candidate_rows = _collect_table_text_lines(doc)
        except Exception:
            candidate_rows = []
    else:
        try:
            candidate_rows = list(rows_txt)
        except Exception:
            candidate_rows = []

    if not candidate_rows:
        return {"rows": rows, "hole_count": total_qty}

    fallback = _publish_fallback_from_rows_txt(candidate_rows)
    if not fallback:
        fallback = _fallback_text_table(candidate_rows)

    if not fallback:
        return {"rows": rows, "hole_count": total_qty}

    raw_rows = fallback.get("rows")
    if isinstance(raw_rows, list):
        rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    elif isinstance(raw_rows, Iterable) and not isinstance(
        raw_rows, (str, bytes, bytearray)
    ):
        rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]

    hole_count_value = fallback.get("hole_count")
    try:
        total_qty = int(hole_count_value)
    except Exception:
        total_qty = _sum_qty(rows)

    if total_qty <= 0 and len(rows) < min_rows:
        return {"rows": rows, "hole_count": total_qty}

    result: dict[str, Any] = {
        "rows": rows,
        "hole_count": total_qty,
        "provenance_holes": fallback.get("provenance_holes", "HOLE TABLE"),
        "source": fallback.get("source", "text_table"),
    }
    families = fallback.get("hole_diam_families_in")
    if isinstance(families, Mapping):
        result["hole_diam_families_in"] = dict(families)
    return result


def read_text_table(
    doc,
    *,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    roi_hint: Mapping[str, Any] | None = None,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
    layer_include_regex: Iterable[str] | str | None = None,
    layer_exclude_regex: Iterable[str] | str | None = DEFAULT_TEXT_LAYER_EXCLUDE_REGEX,
    layout_filters: Mapping[str, Any] | Iterable[str] | str | None = None,
    debug_layouts: bool = False,
    debug_scan: bool = False,
) -> dict[str, Any]:
    helper = _resolve_app_callable("extract_hole_table_from_text")
    _print_helper_debug("text", helper)
    global _LAST_TEXT_TABLE_DEBUG, _PROMOTED_ROWS_LOGGED
    _LAST_TEXT_TABLE_DEBUG = {
        "candidates": [],
        "rows": [],
        "raw_lines": [],
        "roi_hint": roi_hint,
        "roi": None,
        "preferred_blocks": [],
        "row_debug": [],
        "columns": [],
        "bands": [],
        "layout_filters": layout_filters,
    }
    debug_scan_enabled = bool(debug_scan)
    roi_hint_effective: Mapping[str, Any] | None = roi_hint
    resolved_allowlist = _normalize_layer_allowlist(layer_allowlist)
    normalized_block_allow = _normalize_block_allowlist(block_name_allowlist)
    block_regex_patterns = _compile_block_name_patterns(block_name_regex)
    allow_all_layouts, layout_filter_patterns = _parse_layout_filter(layout_filters)
    if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
        _LAST_TEXT_TABLE_DEBUG["layout_filters"] = {
            "all_layouts": allow_all_layouts,
            "patterns": list(layout_filter_patterns),
        }
        _LAST_TEXT_TABLE_DEBUG["debug_scan_requested"] = debug_scan_enabled

    def _compile_layer_patterns(
        patterns: Iterable[str] | str | None,
    ) -> list[re.Pattern[str]]:
        compiled: list[re.Pattern[str]] = []
        if isinstance(patterns, str):
            pattern_iter: Iterable[str] = [patterns]
        elif patterns is None:
            pattern_iter = []
        else:
            pattern_iter = patterns
        for candidate in pattern_iter:
            if not isinstance(candidate, str):
                continue
            text = candidate.strip()
            if not text:
                continue
            try:
                compiled.append(re.compile(text, re.IGNORECASE))
            except re.error as exc:
                print(f"[TEXT-SCAN] layer regex error pattern={text!r} err={exc}")
        return compiled

    include_patterns = _compile_layer_patterns(layer_include_regex)
    exclude_patterns = _compile_layer_patterns(layer_exclude_regex)
    base_exclude = re.compile(_GEO_EXCLUDE_LAYERS_DEFAULT, re.IGNORECASE)
    if not any(pattern.pattern == base_exclude.pattern for pattern in exclude_patterns):
        exclude_patterns.insert(0, base_exclude)
    am_bor_band_layouts = {"CHART", "SHEET (B)"}
    include_display = [pattern.pattern for pattern in include_patterns]
    exclude_display = [pattern.pattern for pattern in exclude_patterns]
    allowlist_display = (
        "None"
        if resolved_allowlist is None
        else "{" + ",".join(sorted(resolved_allowlist) or []) + "}"
    )
    if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
        _LAST_TEXT_TABLE_DEBUG["layer_regex_include"] = list(include_display)
        _LAST_TEXT_TABLE_DEBUG["layer_regex_exclude"] = list(exclude_display)
        _LAST_TEXT_TABLE_DEBUG["debug_layouts_requested"] = bool(debug_layouts)
    table_lines: list[str] | None = None
    fallback_candidate: Mapping[str, Any] | None = None
    best_candidate: Mapping[str, Any] | None = None
    best_score: tuple[int, int] = (0, 0)
    helper_score: tuple[int, int] = (0, 0)
    legacy_score: tuple[int, int] = (0, 0)
    text_rows_info: dict[str, Any] | None = None
    merged_rows: list[str] = []
    parsed_rows: list[dict[str, Any]] = []
    families: dict[str, int] = {}
    total_qty = 0
    columnar_table_info: dict[str, Any] | None = None
    columnar_debug_info: dict[str, Any] | None = None
    anchor_rows_primary: list[dict[str, Any]] = []
    anchor_qty_total = 0
    anchor_is_authoritative = False
    roi_rows_primary: list[dict[str, Any]] = []
    rows_txt_initial = 0
    confidence_high = False
    anchor_authoritative_result: dict[str, Any] | None = None
    secondary_anchor_candidate: dict[str, Any] | None = None
    anchor_band_context: dict[str, Any] | None = None

    helper_missing = helper is None
    if helper_missing and isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
        _LAST_TEXT_TABLE_DEBUG["layer_counts_pre"] = {}

    am_bor_included = False

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
        nonlocal rows_txt_initial, doc, resolved_allowlist
        nonlocal anchor_rows_primary, roi_rows_primary, anchor_authoritative_result
        nonlocal anchor_is_authoritative
        nonlocal anchor_band_context
        if table_lines is not None:
            return table_lines

        text_stream_doc, text_stream_lines = ensure_text_stream(doc, log=print)
        if text_stream_doc is not None:
            doc = text_stream_doc
        fallback_stream_lines = list(text_stream_lines or [])
        follow_sheet_target_layouts: list[str] = []
        collected_entries: list[dict[str, Any]] = []
        candidate_entries: list[dict[str, Any]] = []
        entries_by_layout: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        layout_names: dict[int, str] = {}
        layout_order: list[int] = []
        see_sheet_hint_text: str | None = None
        see_sheet_hint_logged = False
        ordered_layout_lines: list[tuple[str, str]] = []
        combined_fallback_lines: list[str] = []
        fallback_layout_names: list[str] = []
        fallback_follow_entries: list[dict[str, Any]] = []
        families: dict[str, int] = {}
        total_qty = 0
        layout_line_map: dict[str, list[str]] = {}

        def _filter_and_dedupe_row_texts(row_texts: Iterable[str]) -> list[str]:
            filtered: list[str] = []
            seen: set[tuple[str, str]] = set()
            for raw_text in row_texts:
                candidate = " ".join(str(raw_text or "").split()).strip()
                if not candidate:
                    continue
                if _ADMIN_ROW_DROP_RE.search(candidate):
                    continue
                match = _ROW_QUANTITY_PATTERNS[0].match(candidate)
                if not match:
                    continue
                qty_token = match.group(1)
                remainder = candidate[match.end() :].strip()
                normalized_desc = " ".join(remainder.split()).upper()
                key = (qty_token, normalized_desc)
                if key in seen:
                    continue
                seen.add(key)
                filtered.append(candidate)
            return filtered

        def _perform_text_scan(
            current_allowlist: _LayerAllowlist | None,
        ) -> tuple[int, int, int]:
            nonlocal table_lines, text_rows_info, merged_rows, parsed_rows
            nonlocal families, total_qty
            nonlocal columnar_table_info, columnar_debug_info, roi_hint_effective, rows_txt_initial
            nonlocal anchor_rows_primary, roi_rows_primary, anchor_authoritative_result
            nonlocal anchor_is_authoritative, secondary_anchor_candidate
            nonlocal anchor_band_context
            nonlocal allowlist_display, follow_sheet_target_layouts
            nonlocal collected_entries, candidate_entries, entries_by_layout, layout_names
            nonlocal layout_order, see_sheet_hint_text, see_sheet_hint_logged
            nonlocal am_bor_included
            families = {}
            total_qty = 0
            collected_entries = []
            candidate_entries = []
            entries_by_layout = defaultdict(list)
            layout_names = {}
            layout_order = []
            merged_rows = []
            parsed_rows = []
            total_qty = 0
            text_rows_info = None
            rows_txt_initial = 0
            anchor_rows_primary = []
            roi_rows_primary = []
            secondary_anchor_candidate = None
            anchor_band_context = None
            hint_logged = False
            attrib_count = 0
            mleader_count = 0
            preferred_block_names: list[str] = []
            preferred_block_rois: list[dict[str, Any]] = []
            block_height_samples: defaultdict[str, list[float]] = defaultdict(list)
            block_stats: defaultdict[str, dict[str, Any]] = defaultdict(
                lambda: {"texts": 0, "att": 0, "nested_inserts": 0}
            )
            layout_names_seen: list[str] = []
            layout_names_seen_set: set[str] = set()
            scanned_layers_map: dict[str, str] = {}
            follow_sheet_directive: dict[str, Any] | None = None
            follow_sheet_directives: list[dict[str, Any]] = []
            follow_sheet_target_layouts = []
            follow_sheet_target_layout: str | None = None
            follow_sheet_requests: dict[str, dict[str, Any]] = {}
            layout_lookup: dict[str, Any] = {}
            layout_name_lookup: dict[str, Any] = {}
            layout_index_lookup: dict[str, int] = {}
            visited_layout_keys: set[tuple[str, str]] = set()
            all_layout_names: set[str] = set()
            expanded_layouts: list[str] = []

            try:
                if current_allowlist is not None and "AM_BOR" in current_allowlist:
                    am_bor_included = True
            except Exception:
                pass

            allowlist_display = (
                "None"
                if current_allowlist is None
                else "{" + ",".join(sorted(current_allowlist) or []) + "}"
            )
    
            if doc is None:
                table_lines = []
                return (0, 0, 0)
    
            def _lookup_layer_count(counts: Mapping[str, int], target: str) -> int:
                total = 0
                for name, value in counts.items():
                    try:
                        candidate = str(name or "").upper()
                    except Exception:
                        candidate = str(name).upper() if name is not None else ""
                    if candidate == target:
                        try:
                            total += int(value)
                        except Exception:
                            continue
                return total
    
            def _scan_layout_body(
                layout_index: int,
                layout_name: Any,
                layout_obj: Any,
                *,
                source: str = "initial",
            ) -> bool:
                nonlocal follow_sheet_directive
    
                name_text = str(layout_name or "")
                name_clean = name_text.strip()
                key_upper = name_clean.upper()
                if isinstance(layout_name, str):
                    all_layout_names.add(layout_name)
                unique_key = (source, key_upper or f"#{layout_index}")
                if unique_key in visited_layout_keys:
                    return False
                visited_layout_keys.add(unique_key)
    
                if source != "block" and key_upper:
                    layout_lookup[key_upper] = layout_obj
                    layout_name_lookup.setdefault(key_upper, name_clean or layout_name)
                    layout_index_lookup.setdefault(key_upper, layout_index)
    
                layout_names[layout_index] = layout_name
                if layout_name not in layout_names_seen_set:
                    layout_names_seen_set.add(layout_name)
                    layout_names_seen.append(layout_name)
                if layout_index not in layout_order:
                    layout_order.append(layout_index)
    
                layout_str = str(layout_name or "")
                layout_label = layout_str.strip() or "-"
                display_name = layout_name if layout_name else layout_label
                if source == "follow":
                    display_name = f"{display_name} [FOLLOW]"
                elif source == "block":
                    display_name = f"{display_name} [BLOCK]"
                elif layout_obj is None:
                    display_name = f"{display_name} [UNRESOLVED]"
                expanded_layouts.append(display_name)
    
                if layout_obj is None:
                    return False
                return True
    
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
    
            block_stats: dict[str, dict[str, int]] = {}
    
            def _block_stats_entry(name: str | None) -> dict[str, int]:
                key = (name or "").strip()
                entry = block_stats.get(key)
                if entry is None:
                    entry = {"texts": 0, "att": 0, "nested_inserts": 0}
                    block_stats[key] = entry
                return entry
    
            def _scan_layout_entities(
                layout_index: int,
                layout_name: Any,
                layout_obj: Any,
                *,
                source: str = "initial",
            ) -> bool:
                nonlocal hint_logged, attrib_count, mleader_count, follow_sheet_directive
                nonlocal follow_sheet_directives
                if not _scan_layout_body(layout_index, layout_name, layout_obj, source=source):
                    return False
    
                layout_str = str(layout_name or "")
                layout_label = layout_str.strip() or "-"
                layout_tables = _count_tables_for_layout_name(layout_str)
                query = getattr(layout_obj, "query", None)
                base_entities: list[Any] = []
                layer_extractor = globals().get("_extract_layer")
                if callable(query):
                    try:
                        base_entities = list(query("TEXT, MTEXT, RTEXT, MLEADER, INSERT"))
                    except Exception:
                        base_entities = []
                    if not base_entities:
                        for spec in ("TEXT", "MTEXT", "RTEXT", "MLEADER", "INSERT"):
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
                    if _TRACE_ACAD:
                        print(f"[LAYOUT] {layout_label} texts=0/0 tables={layout_tables}")
                    return False
    
                seen_entities: set[int] = set()
                text_fragments = 0
                mtext_fragments = 0
                kept_count = 0
                from_blocks_count = 0
                counter = 0
    
                flattened_entities = list(flatten_entities(layout_obj, depth=_MAX_INSERT_DEPTH))
                if not flattened_entities and base_entities:
                    for entity in base_entities:
                        flattened_entities.append(
                            FlattenedEntity(
                                entity=entity,
                                transform=_IDENTITY_TRANSFORM,
                                from_block=False,
                                block_name=None,
                                block_stack=tuple(),
                                depth=0,
                                layer="",
                                layer_upper="",
                                effective_layer="",
                                effective_layer_upper="",
                            )
                        )
    
                for flattened in flattened_entities:
                    entity = flattened.entity
                    parent_effective_layer = getattr(flattened, "parent_effective_layer", None)
                    active_block = getattr(flattened, "block_name", None)
                    from_block = bool(getattr(flattened, "from_block", False))
                    marker = id(entity)
                    if marker in seen_entities:
                        continue
                    seen_entities.add(marker)
                    parent_effective_layer = flattened.effective_layer
                    active_block = flattened.block_name
                    from_block = bool(flattened.from_block)
                    try:
                        dxftype = entity.dxftype()
                    except Exception:
                        dxftype = None
                    kind = str(dxftype or "").upper()
                    parent_effective_layer = getattr(flattened, "effective_layer", None)
                    from_block = bool(getattr(flattened, "from_block", False))
                    active_block = getattr(flattened, "block_name", None)
                    layer_name = _extract_layer(entity)
                    layer_upper = layer_name.upper() if layer_name else ""
                    effective_layer = layer_name
                    effective_layer_upper = layer_upper
                    if not effective_layer_upper or effective_layer_upper == "0":
                        candidate = parent_effective_layer or layer_name or ""
                        effective_layer = candidate
                        effective_layer_upper = candidate.upper() if candidate else ""
                    if kind in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "MLEADER", "RTEXT"}:
                        entity_type = "ATTRIB" if kind in {"ATTRIB", "ATTDEF"} else kind
                        coords = _extract_coords(entity)
                        coords = _apply_transform_point(flattened.transform, coords)
                        text_height = _extract_text_height(entity)
                        if isinstance(text_height, (int, float)):
                            text_height = float(text_height) * _transform_scale_hint(flattened.transform)
                        counted_block_text = False
                        counted_block_attr = False
                        for fragment, is_mtext in _iter_entity_text_fragments(entity):
                            normalized = _normalize_table_fragment(fragment)
                            if not normalized:
                                continue
                            normalized_upper = normalized.upper()
                            if kind in {"ATTRIB", "ATTDEF"}:
                                attrib_count += 1
                                if active_block and not counted_block_attr:
                                    _block_stats_entry(active_block)["att"] += 1
                                    counted_block_attr = True
                            elif kind == "MLEADER":
                                mleader_count += 1
                            if active_block and not counted_block_text and kind not in {"ATTRIB", "ATTDEF"}:
                                _block_stats_entry(active_block)["texts"] += 1
                                counted_block_text = True
                            if (
                                not hint_logged
                                and "SEE SHEET 2 FOR HOLE CHART" in normalized_upper
                            ):
                                print(
                                    "[HINT] Chart may live on an alternate sheet/block; ensure its INSERT is present and not on a frozen/off layer."
                                )
                                hint_logged = True
                            match = _FOLLOW_SHEET_DIRECTIVE_RE.search(normalized)
                            if match:
                                directive_entry = {
                                    "layout": layout_name,
                                    "token": match.group("target"),
                                    "text": normalized,
                                }
                                follow_sheet_directive = directive_entry
                                follow_sheet_directives.append(directive_entry)
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
                                "block_stack": list(flattened.block_stack),
                                "entity_type": entity_type,
                            }
                            counter += 1
                            collected_entries.append(entry)
                            entries_by_layout[layout_index].append(entry)
                            layer_token = effective_layer or layer_name
                            if layer_token:
                                upper_token = layer_token.upper()
                                if upper_token and upper_token not in scanned_layers_map:
                                    scanned_layers_map[upper_token] = layer_token
                            kept_count += 1
                            if _TRACE_ACAD and active_block:
                                stats_entry = _block_stats_entry(active_block)
                                if kind in {"ATTRIB", "ATTDEF"}:
                                    stats_entry["att"] += 1
                                else:
                                    stats_entry["texts"] += 1
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
                    elif kind == "INSERT":
                        dxf_obj = getattr(entity, "dxf", None)
                        block_name = None
                        if dxf_obj is not None:
                            block_name = getattr(dxf_obj, "name", None)
                        if block_name is None:
                            block_name = getattr(entity, "name", None)
                        name_str = block_name.strip() if isinstance(block_name, str) else None
                        if active_block:
                            _block_stats_entry(active_block)["nested_inserts"] += 1
                        if name_str:
                            _block_stats_entry(name_str)
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
                                transform=flattened.transform,
                            )
                            if bbox is not None:
                                try:
                                    xmin, xmax, ymin, ymax = (
                                        float(bbox[0]),
                                        float(bbox[1]),
                                        float(bbox[2]),
                                        float(bbox[3]),
                                    )
                                except Exception:
                                    xmin = xmax = ymin = ymax = 0.0
                                bbox_entry = [xmin, xmax, ymin, ymax]
                                preferred_block_rois.append(
                                    {
                                        "name": name_str,
                                        "layer": effective_layer or layer_name,
                                        "bbox": bbox_entry,
                                        "block_stack": list(flattened.block_stack),
                                        "from_block": flattened.from_block,
                                    }
                                )
    
                print(
                    f"[TEXT-SCAN] layout={layout_name} text={text_fragments} "
                    f"mtext={mtext_fragments} kept={kept_count} from_blocks={from_blocks_count}"
                )
                if _TRACE_ACAD:
                    print(
                        f"[LAYOUT] {layout_label} texts={text_fragments}/{mtext_fragments} "
                        f"tables={layout_tables}"
                    )
                return True
    
            try:
                initial_layouts = iter_layouts(doc, layout_filters)
            except RuntimeError:
                table_lines = []
                raise
    
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                _LAST_TEXT_TABLE_DEBUG["layout_sequence"] = [
                    str(name or "") for name, _ in initial_layouts
                ]
    
            for name, _ in initial_layouts:
                if isinstance(name, str):
                    all_layout_names.add(name)
    
            layouts_manager = getattr(doc, "layouts", None)
            raw_names = getattr(layouts_manager, "names", None) if layouts_manager is not None else None
            if raw_names is not None:
                try:
                    names_iter = raw_names() if callable(raw_names) else raw_names
                except Exception:
                    names_iter = None
                if names_iter is not None:
                    try:
                        for candidate in names_iter:
                            if isinstance(candidate, str):
                                all_layout_names.add(candidate)
                    except Exception:
                        pass
    
            get_layout = getattr(layouts_manager, "get", None) if layouts_manager is not None else None
            layout_queue: deque[tuple[int, Any, Any, str]] = deque(
                (index, layout_name, layout_obj, "initial")
                for index, (layout_name, layout_obj) in enumerate(initial_layouts)
            )
            next_layout_index = len(initial_layouts)
            queued_layout_names: set[str] = {
                _normalize_layout_key(layout_name)
                for layout_name, _layout_obj in initial_layouts
            }
    
            def _enqueue_follow_layout(name: str) -> None:
                nonlocal next_layout_index
                normalized = _normalize_layout_key(name)
                layout_obj = None
                if callable(get_layout):
                    try:
                        layout_obj = get_layout(name)
                    except Exception:
                        layout_obj = None
                if normalized in queued_layout_names:
                    return
                queued_layout_names.add(normalized)
                layout_queue.append((next_layout_index, name, layout_obj, "follow"))
                next_layout_index += 1
    
            def _update_follow_targets() -> None:
                nonlocal follow_sheet_target_layouts
                if not (follow_sheet_directive or follow_sheet_target_layout):
                    return
                catalog = list(dict.fromkeys(name for name in all_layout_names if isinstance(name, str)))
                if not catalog:
                    catalog = layout_names_seen if layout_names_seen else list(layout_names.values())
                new_targets: list[str] = []
                token_value = None
                if isinstance(follow_sheet_directive, Mapping):
                    token_value = follow_sheet_directive.get("token")
                if token_value:
                    target_label, resolved_layout, resolved_found = _resolve_follow_sheet_layout(
                        token_value or "", catalog
                    )
                    follow_sheet_requests[target_label] = {
                        "token": token_value,
                        "target": target_label,
                        "resolved": resolved_layout,
                        "found": resolved_found,
                    }
                    if resolved_layout and resolved_found:
                        new_targets.append(resolved_layout)
                if follow_sheet_target_layout:
                    new_targets.append(follow_sheet_target_layout)
                if new_targets:
                    existing = set(follow_sheet_target_layouts)
                    for target in new_targets:
                        if target not in existing:
                            follow_sheet_target_layouts.append(target)
                            existing.add(target)
    
            while layout_queue:
                layout_index, layout_name, layout_obj, source = layout_queue.popleft()
                if not _scan_layout_entities(layout_index, layout_name, layout_obj, source=source):
                    continue
                _update_follow_targets()
                for target_layout in follow_sheet_target_layouts:
                    _enqueue_follow_layout(target_layout)
    
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                _LAST_TEXT_TABLE_DEBUG["layouts"] = list(expanded_layouts)
                _LAST_TEXT_TABLE_DEBUG["follow_sheet_targets"] = list(follow_sheet_target_layouts)
    
            if _TRACE_ACAD and block_stats:
                for block_name, stats in block_stats.items():
                    display_name = block_name or "-"
                    print(
                        f"[BLOCK] {display_name} texts={stats['texts']} "
                        f"att={stats['att']} nested_inserts={stats['nested_inserts']}"
                    )
    
            if preferred_block_names:
                print(f"[TEXT-SCAN] preferred_blocks={preferred_block_names}")
            _LAST_TEXT_TABLE_DEBUG["preferred_blocks"] = list(preferred_block_names)
            _LAST_TEXT_TABLE_DEBUG["attrib_lines"] = attrib_count
            _LAST_TEXT_TABLE_DEBUG["mleader_lines"] = mleader_count
            print(
                f"[TEXT-SCAN] attrib_lines={attrib_count} mleader_lines={mleader_count} "
                f"depth_max={_MAX_INSERT_DEPTH} allow_layers={allowlist_display}"
            )
    
            def _format_layer_summary(counts: Mapping[str, int]) -> str:
                if not counts:
                    return "{}"
                top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
                return "{" + ", ".join(f"{name or '-'}:{count}" for name, count in top) + "}"
    
            def _count_layers(entries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
                counts: dict[str, int] = defaultdict(int)
                for entry in entries:
                    layer_key = str(
                        entry.get("effective_layer")
                        or entry.get("layer")
                        or "",
                    ).strip()
                    counts[layer_key] += 1
                return dict(counts)
    
            def _count_layouts(entries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
                counts: dict[str, int] = defaultdict(int)
                for entry in entries:
                    layout_key = str(
                        entry.get("layout_name")
                        or entry.get("layout_index")
                        or "",
                    ).strip()
                    counts[layout_key] += 1
                return dict(counts)
    
            layer_counts_pre = _count_layers(collected_entries)
            total_layer_hits = sum(layer_counts_pre.values())
            if total_layer_hits == 0:
                if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                    _LAST_TEXT_TABLE_DEBUG["layer_counts_pre"] = dict(layer_counts_pre)
                    _LAST_TEXT_TABLE_DEBUG["layout_counts_pre"] = {}
                    _LAST_TEXT_TABLE_DEBUG["scanned_layers"] = sorted(
                        scanned_layers_map.values(), key=lambda value: value.upper()
                    )
                    _LAST_TEXT_TABLE_DEBUG["scanned_layouts"] = list(layout_names_seen)
                raise RuntimeError("No text found before layer filtering…")
            layout_counts_pre = _count_layouts(collected_entries)
            if debug_scan_enabled:
                layout_display = ", ".join(
                    str(name) for name in layout_names_seen if isinstance(name, str)
                )
                if not layout_display:
                    layout_display = "-"
                print(f"[TEXT-SCAN] layouts=[{layout_display}]")
                layout_type_counts: dict[str, Counter[str]] = {}
                for layout_index, layout_entries in entries_by_layout.items():
                    layout_name = layout_names.get(layout_index, layout_index)
                    name_str = str(layout_name)
                    counter = layout_type_counts.setdefault(name_str, Counter())
                    for entry in layout_entries:
                        entry_type = str(entry.get("entity_type") or "")
                        if entry_type == "ATTDEF":
                            entry_type = "ATTRIB"
                        if entry_type:
                            counter[entry_type] += 1
                preferred_types = ("TEXT", "MTEXT", "ATTRIB", "MLEADER", "RTEXT")
                reported: set[str] = set()
                for layout_name in layout_names_seen:
                    name_str = str(layout_name)
                    counter = layout_type_counts.get(name_str, Counter())
                    parts = [f"{type_name}={int(counter.get(type_name, 0))}" for type_name in preferred_types]
                    print(
                        "[TEXT-SCAN] layout={name} {summary}".format(
                            name=name_str or "-",
                            summary=" ".join(parts),
                        )
                    )
                    reported.add(name_str)
                for extra_name in sorted(set(layout_type_counts) - reported):
                    counter = layout_type_counts.get(extra_name, Counter())
                    parts = [f"{type_name}={int(counter.get(type_name, 0))}" for type_name in preferred_types]
                    print(
                        "[TEXT-SCAN] layout={name} {summary}".format(
                            name=extra_name or "-",
                            summary=" ".join(parts),
                        )
                    )
            print(f"[TEXT-SCAN] kept_by_layer(pre)={_format_layer_summary(layer_counts_pre)}")
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                _LAST_TEXT_TABLE_DEBUG["layer_counts_pre"] = dict(layer_counts_pre)
                _LAST_TEXT_TABLE_DEBUG["layout_counts_pre"] = dict(layout_counts_pre)
                _LAST_TEXT_TABLE_DEBUG["scanned_layers"] = sorted(
                    scanned_layers_map.values(), key=lambda value: value.upper()
                )
                _LAST_TEXT_TABLE_DEBUG["scanned_layouts"] = list(layout_names_seen)
                if debug_scan_enabled:
                    _LAST_TEXT_TABLE_DEBUG["layout_entity_counts"] = {
                        name: {key: int(value) for key, value in counter.items()}
                        for name, counter in layout_type_counts.items()
                    }
    
            if include_patterns or exclude_patterns:
                def _matches_any(
                    patterns: list[re.Pattern[str]],
                    values: list[str],
                ) -> bool:
                    for pattern in patterns:
                        for value in values:
                            if not value:
                                continue
                            match = pattern.search(value)
                            if not match:
                                continue
                            return True
                    return False

                regex_filtered: list[dict[str, Any]] = []
                for entry in collected_entries:
                    layer_text = str(
                        entry.get("effective_layer")
                        or entry.get("layer")
                        or "",
                    )
                    upper_text = str(
                        entry.get("effective_layer_upper")
                        or entry.get("layer_upper")
                        or layer_text.upper()
                    )
                    layout_name_value = entry.get("layout_name")
                    if not layout_name_value:
                        layout_idx = entry.get("layout_index")
                        layout_name_value = layout_names.get(layout_idx, layout_idx)
                    layout_upper = str(layout_name_value or "").strip().upper()
                    values = [layer_text, upper_text]
                    include_ok = True
                    if include_patterns:
                        include_ok = _matches_any(include_patterns, values)
                    exclude_hit = False
                    if include_ok and exclude_patterns:
                        exclude_hit = _matches_any(
                            list(exclude_patterns),
                            values,
                        )
                    if include_ok and not exclude_hit:
                        regex_filtered.append(entry)
                kept = len(regex_filtered)
                dropped = len(collected_entries) - kept
                print(
                    "[LAYER] regex include={incl} exclude={excl} kept={kept} dropped={dropped}".format(
                        incl=include_display or "-",
                        excl=exclude_display or "-",
                        kept=kept,
                        dropped=dropped,
                    )
                )
                collected_entries = regex_filtered
                layer_counts_regex = _count_layers(collected_entries)
                layout_counts_regex = _count_layouts(collected_entries)
                if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                    _LAST_TEXT_TABLE_DEBUG["layer_counts_post_regex"] = dict(layer_counts_regex)
                    _LAST_TEXT_TABLE_DEBUG["layout_counts_post_regex"] = dict(layout_counts_regex)
    
            if current_allowlist is not None:
                filtered_entries = []
                filtered_by_layout: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
                for entry in collected_entries:
                    layout_idx = entry.get("layout_index")
                    try:
                        layout_key = int(layout_idx)
                    except Exception:
                        layout_key = 0
                    filtered_by_layout[layout_key].append(entry)
                for layout_index in layout_order:
                    layout_entries = filtered_by_layout.get(layout_index, [])
                    if not layout_entries:
                        continue
                    layout_name = layout_names.get(layout_index, str(layout_index))
                    kept_for_layout = [
                        entry
                        for entry in layout_entries
                        if not (entry.get("effective_layer_upper") or "")
                        or (entry.get("effective_layer_upper") or "")
                        in current_allowlist
                    ]
                    kept_count = len(kept_for_layout)
                    if kept_count > 0:
                        print(
                            "[LAYER] layout={layout} allow={allow} kept={count}".format(
                                layout=layout_name,
                                allow=allowlist_display,
                                count=kept_count,
                            )
                        )
                        filtered_entries.extend(kept_for_layout)
                    else:
                        print(
                            "[LAYER] layout={layout} allow={allow} kept=0".format(
                                layout=layout_name,
                                allow=allowlist_display,
                            )
                        )
            else:
                filtered_entries = list(collected_entries)
    
            layer_counts_post = _count_layers(filtered_entries)
            layout_counts_post = _count_layouts(filtered_entries)
            print(f"[TEXT-SCAN] kept_by_layer(post)={_format_layer_summary(layer_counts_post)}")
    
            am_bor_pre_count = _lookup_layer_count(layer_counts_pre, "AM_BOR")
            am_bor_post_count = _lookup_layer_count(layer_counts_post, "AM_BOR")
            am_bor_drop_count = max(am_bor_pre_count - am_bor_post_count, 0)

            if am_bor_post_count > 0:
                raise AssertionError(
                    "AM_BOR layer must be excluded from text scan results",
                )
    
            collected_entries = filtered_entries
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                _LAST_TEXT_TABLE_DEBUG["layer_counts_post_allow"] = dict(layer_counts_post)
                _LAST_TEXT_TABLE_DEBUG["layout_counts_post_allow"] = dict(layout_counts_post)
                _LAST_TEXT_TABLE_DEBUG["collected_entities"] = [
                    {
                        "layout": str(entry.get("layout_name") or ""),
                        "layer": str(
                            entry.get("effective_layer")
                            or entry.get("layer")
                            or ""
                        ),
                        "type": str(entry.get("entity_type") or ""),
                        "x": entry.get("x"),
                        "y": entry.get("y"),
                        "height": entry.get("height"),
                        "text": str(entry.get("text") or ""),
                    }
                    for entry in collected_entries
                ]
    
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
                    bbox_vals = block_hint.get("bbox")
                    if (
                        isinstance(bbox_vals, (list, tuple))
                        and len(bbox_vals) == 4
                        and all(isinstance(val, (int, float)) for val in bbox_vals)
                    ):
                        print(
                            "[ROI] seeded_from=BLOCK name={name} layer={layer} box=[{xmin:.1f}..{xmax:.1f}, {ymin:.1f}..{ymax:.1f}]".format(
                                name=block_hint.get("name") or "-",
                                layer=block_hint.get("layer") or "-",
                                xmin=float(bbox_vals[0]),
                                xmax=float(bbox_vals[1]),
                                ymin=float(bbox_vals[2]),
                                ymax=float(bbox_vals[3]),
                            )
                        )
    
            if follow_sheet_target_layout:
                follow_sheet_target_layouts.append(follow_sheet_target_layout)
            follow_sheet_target_layouts = list(dict.fromkeys(follow_sheet_target_layouts))
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                _LAST_TEXT_TABLE_DEBUG["follow_sheet_targets"] = list(follow_sheet_target_layouts)
    
            if follow_sheet_requests:
                info_entries: list[dict[str, Any]] = []
                for request in follow_sheet_requests.values():
                    token_value = request.get("token")
                    target_label = request.get("target") or (
                        f"SHEET ({token_value})" if token_value else "SHEET ()"
                    )
                    resolved_layout = request.get("resolved")
                    resolved_found = bool(request.get("found")) and bool(resolved_layout)
                    print(
                        f"[HINT] follow-sheet target=\"{target_label}\" found={resolved_found}"
                    )
                    info_map = {
                        "target": target_label,
                        "resolved": resolved_layout if resolved_found else None,
                        "texts": 0,
                        "tables": 0,
                    }
                    if resolved_found and resolved_layout:
                        target_upper = str(resolved_layout).strip().upper()
                        texts_count = sum(
                            1
                            for entry in collected_entries
                            if str(entry.get("layout_name") or "").strip().upper()
                            == target_upper
                        )
                        tables_count = _count_tables_for_layout_name(resolved_layout)
                        print(
                            f"[LAYOUT] {resolved_layout} texts={texts_count} tables={tables_count}"
                        )
                        info_map["texts"] = texts_count
                        info_map["tables"] = tables_count
                    info_entries.append(info_map)
                if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                    if len(info_entries) == 1:
                        _LAST_TEXT_TABLE_DEBUG["follow_sheet_info"] = info_entries[0]
                    else:
                        _LAST_TEXT_TABLE_DEBUG["follow_sheet_info"] = info_entries
    
            if not collected_entries:
                fallback_lines = _collect_table_text_lines(doc, layout_filters=layout_filters)
                if fallback_lines:
                    combined_fallback_lines = list(fallback_lines)
                combined_fallback_lines = [
                    " ".join(str(line or "").split())
                    for line in combined_fallback_lines
                    if str(line or "").strip()
                ]
    
                fallback_entries: list[dict[str, Any]] = []
                if ordered_layout_lines:
                    entries_by_layout = defaultdict(list)
                    layout_names = {}
                    layout_order = []
                    for idx, (layout_label, text_value) in enumerate(ordered_layout_lines):
                        layout_key = layout_label or "FALLBACK"
                        layout_index_effective = layout_index_map.setdefault(
                            layout_key, len(layout_index_map)
                        )
                        if layout_index_effective not in layout_order:
                            layout_order.append(layout_index_effective)
                        layout_names[layout_index_effective] = layout_key
                        entry = {
                            "layout_index": layout_index_effective,
                            "layout_name": layout_key,
                            "text": text_value,
                            "x": None,
                            "y": None,
                            "order": idx,
                            "from_block": False,
                            "height": None,
                            "layer": "",
                            "layer_upper": "",
                            "effective_layer": "",
                            "effective_layer_upper": "",
                            "block_name": None,
                            "block_stack": [],
                        }
                        fallback_entries.append(entry)
                        entries_by_layout[layout_index_effective].append(entry)
                elif combined_fallback_lines:
                    entries_by_layout = defaultdict(list)
                    layout_names = {}
                    layout_order = []
                    for idx, text_value in enumerate(combined_fallback_lines):
                        entry = {
                            "layout_index": 0,
                            "layout_name": "FALLBACK",
                            "text": text_value,
                            "x": None,
                            "y": None,
                            "order": idx,
                            "from_block": False,
                            "height": None,
                            "layer": "",
                            "layer_upper": "",
                            "effective_layer": "",
                            "effective_layer_upper": "",
                            "block_name": None,
                            "block_stack": [],
                        }
                        fallback_entries.append(entry)
                    if fallback_entries:
                        entries_by_layout[0] = list(fallback_entries)
                        layout_names[0] = "FALLBACK"
                        layout_order.append(0)
    
                if fallback_entries:
                    collected_entries = fallback_entries
                    if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                        if ordered_layout_lines or combined_fallback_lines:
                            debug_lines = [
                                text for _, text in ordered_layout_lines
                            ] or combined_fallback_lines
                            _LAST_TEXT_TABLE_DEBUG["fallback_text_stream"] = list(debug_lines)
                        debug_layouts = list(_LAST_TEXT_TABLE_DEBUG.get("layouts") or [])
                        for name in fallback_layout_names or ["FALLBACK"]:
                            if name and name not in debug_layouts:
                                debug_layouts.append(name)
                        _LAST_TEXT_TABLE_DEBUG["layouts"] = debug_layouts
                        if fallback_follow_entries:
                            info_entries: list[dict[str, Any]] = []
                            for follow_entry in fallback_follow_entries:
                                target_label = follow_entry.get("target")
                                resolved_layout = follow_entry.get("resolved")
                                resolved_found = bool(follow_entry.get("found")) and bool(
                                    resolved_layout
                                )
                                info_map = {
                                    "target": target_label,
                                    "resolved": resolved_layout if resolved_found else None,
                                    "texts": len(
                                        layout_line_map.get(
                                            resolved_layout or target_label or "", []
                                        )
                                    ),
                                    "tables": 0,
                                }
                                info_entries.append(info_map)
                            if info_entries:
                                if len(info_entries) == 1:
                                    _LAST_TEXT_TABLE_DEBUG["follow_sheet_info"] = info_entries[0]
                                else:
                                    _LAST_TEXT_TABLE_DEBUG["follow_sheet_info"] = info_entries
                                targets_payload = []
                                for info in info_entries:
                                    resolved = info.get("resolved") or info.get("target")
                                    if resolved and resolved not in targets_payload:
                                        targets_payload.append(resolved)
                                if targets_payload:
                                    existing_targets = _LAST_TEXT_TABLE_DEBUG.get(
                                        "follow_sheet_targets"
                                    )
                                    if isinstance(existing_targets, list):
                                        targets = list(existing_targets)
                                    elif existing_targets:
                                        targets = [existing_targets]
                                    else:
                                        targets = []
                                    for candidate in targets_payload:
                                        if candidate not in targets:
                                            targets.append(candidate)
                                    _LAST_TEXT_TABLE_DEBUG["follow_sheet_targets"] = targets
                else:
                    _LAST_TEXT_TABLE_DEBUG["rows_txt_count"] = 0
                    _LAST_TEXT_TABLE_DEBUG["text_row_count"] = 0
                    table_lines = []
                    print("[TEXT-SCAN] rows_txt count=0")
                    print("[TEXT-SCAN] parsed rows: 0")
                    return (0, 0, 0)
    
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

            layout_hint_pattern = re.compile(r"(SHEET|CHART|SHT)", re.IGNORECASE)

            def _scan_secondary_layout(
                layout_index: int,
                layout_entries: list[dict[str, Any]],
            ) -> dict[str, Any] | None:
                filtered_entries = [
                    entry
                    for entry in layout_entries
                    if str(entry.get("normalized_text") or entry.get("text") or "").strip()
                ]
                if not filtered_entries:
                    return None
                ordered_entries = sorted(filtered_entries, key=_entry_sort_key)
                candidate_entries_local: list[dict[str, Any]] = []
                row_active = False
                continuation_budget = 0
                for idx, entry in enumerate(ordered_entries):
                    base_text = entry.get("normalized_text") or entry.get("text") or ""
                    stripped = str(base_text).strip()
                    if not stripped:
                        row_active = False
                        continuation_budget = 0
                        continue
                    if _ADMIN_ROW_DROP_RE.search(stripped):
                        row_active = False
                        continuation_budget = 0
                        continue
                    if idx + 1 < len(ordered_entries):
                        next_source = (
                            ordered_entries[idx + 1].get("normalized_text")
                            or ordered_entries[idx + 1].get("text")
                            or ""
                        )
                        next_text = str(next_source)
                    else:
                        next_text = None
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
                    if not keep_line:
                        continue
                    normalized_line = stripped
                    match = _match_row_quantity(normalized_line)
                    if match:
                        prefix = normalized_line[: match.start()].strip(" |")
                        suffix = normalized_line[match.end() :].strip()
                        row_token = match.group(0).strip()
                        parts = [row_token]
                        if prefix:
                            parts.append(prefix)
                        if suffix:
                            parts.append(suffix)
                        normalized_line = " ".join(parts)
                    normalized_line = " ".join(normalized_line.split())
                    if _should_drop_candidate_line(normalized_line):
                        continue
                    entry_copy = dict(entry)
                    entry_copy["normalized_text"] = normalized_line
                    candidate_entries_local.append(entry_copy)
                if not candidate_entries_local:
                    return None
                anchor_h_local, _anchor_count = _compute_anchor_h(
                    candidate_entries_local
                )
                anchor_h_local = float(anchor_h_local or 0.20)
                if _anchor_count > 0 and anchor_h_local > 0 and candidate_entries_local:
                    filtered_by_height = _filter_entries_by_anchor_h(
                        candidate_entries_local, anchor_h=anchor_h_local
                    )
                    if filtered_by_height:
                        candidate_entries_local = [dict(entry) for entry in filtered_by_height]
                table_lines_local = [
                    str(entry.get("normalized_text") or "") for entry in candidate_entries_local
                ]
                row_data: list[dict[str, Any]] = []
                current_texts: list[str] = []
                current_positions: list[float] = []
                for idx, entry in enumerate(candidate_entries_local):
                    line = str(entry.get("normalized_text", "")).strip()
                    if not line or _ADMIN_ROW_DROP_RE.search(line):
                        continue
                    if idx + 1 < len(candidate_entries_local):
                        next_line = str(
                            candidate_entries_local[idx + 1].get("normalized_text", "") or ""
                        )
                    else:
                        next_line = None
                    entry_y = entry.get("y")
                    if _is_row_start(line, next_text=next_line):
                        if current_texts:
                            text = " ".join(current_texts)
                            avg_y = (
                                sum(current_positions) / len(current_positions)
                                if current_positions
                                else None
                            )
                            row_data.append({"text": text, "y": avg_y})
                        current_texts = [line]
                        current_positions = []
                        if isinstance(entry_y, (int, float)):
                            current_positions.append(float(entry_y))
                    elif current_texts:
                        current_texts.append(line)
                        if isinstance(entry_y, (int, float)):
                            current_positions.append(float(entry_y))
                if current_texts:
                    text = " ".join(current_texts)
                    avg_y = sum(current_positions) / len(current_positions) if current_positions else None
                    row_data.append({"text": text, "y": avg_y})
                merged_texts = [row.get("text") for row in row_data if row.get("text")]
                deduped_rows = _filter_and_dedupe_row_texts(merged_texts)
                if len(deduped_rows) < 3:
                    return None
                y_positions: list[float] = []
                for row_text in deduped_rows:
                    match_row = next((row for row in row_data if row.get("text") == row_text), None)
                    if match_row is None:
                        continue
                    y_val = match_row.get("y")
                    if isinstance(y_val, (int, float)) and math.isfinite(float(y_val)):
                        y_positions.append(float(y_val))
                if len(y_positions) < 3:
                    return None
                y_positions_sorted = sorted(y_positions, reverse=True)
                diffs = [
                    abs(y_positions_sorted[i] - y_positions_sorted[i + 1])
                    for i in range(len(y_positions_sorted) - 1)
                    if abs(y_positions_sorted[i] - y_positions_sorted[i + 1]) > 0
                ]
                if not diffs:
                    return None
                try:
                    median_diff = float(statistics.median(diffs))
                except Exception:
                    median_diff = float(diffs[0]) if diffs else 0.0
                if median_diff <= 0:
                    return None
                tolerance = max(0.5, 0.35 * median_diff)
                consistent = sum(1 for diff in diffs if abs(diff - median_diff) <= tolerance)
                required = max(1, int(math.ceil(len(diffs) * 0.6)))
                if consistent < required:
                    return None
                anchor_count_effective = sum(
                    1
                    for entry in candidate_entries_local
                    if _ROW_ANCHOR_RE.match(str(entry.get("normalized_text") or ""))
                )
                layout_name = str(layout_names.get(layout_index, layout_index))
                score = (
                    float(consistent),
                    float(anchor_count_effective),
                    float(len(deduped_rows)),
                    -median_diff,
                )
                return {
                    "layout_index": layout_index,
                    "layout_name": layout_name,
                    "rows_txt": list(deduped_rows),
                    "entries": [dict(entry) for entry in candidate_entries_local],
                    "table_lines": list(table_lines_local),
                    "anchor_count": int(anchor_count_effective),
                    "score": score,
                    "median_gap": median_diff,
                    "y_positions": y_positions_sorted,
                }
    
            collected_entries.sort(key=_entry_sort_key)
    
            candidate_entries = []
            row_active = False
            continuation_budget = 0
            for idx, entry in enumerate(collected_entries):
                stripped = entry.get("text", "").strip()
                if not stripped:
                    row_active = False
                    continuation_budget = 0
                    continue
                if _ADMIN_ROW_DROP_RE.search(stripped):
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
                    block_display = entry.get("block_name") or "-"
                    print(
                        f"  [{idx:02d}] (x={x_display} y={y_display}) block={block_display} "
                        f"text=\"{entry.get('text', '')}\""
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
                if _should_drop_candidate_line(normalized_line):
                    normalized_upper = normalized_line.upper()
                    if (
                        "SEE SHEET" in normalized_upper
                        and "HOLE CHART" in normalized_upper
                        and see_sheet_hint_text is None
                    ):
                        see_sheet_hint_text = normalized_line
                        directive = _FOLLOW_SHEET_DIRECTIVE_RE.search(normalized_line)
                        if directive:
                            follow_sheet_target_layout = directive.group("target")
                    continue
                entry_copy = dict(entry)
                entry_copy["normalized_text"] = normalized_line
                normalized_entries.append(entry_copy)
                normalized_lines.append(normalized_line)

            candidate_entries = normalized_entries
            table_lines = normalized_lines

            anchors = [
                entry
                for entry in candidate_entries
                if _ROW_ANCHOR_RE.match(
                    _normalize_candidate_text(
                        entry.get("normalized_text") or entry.get("text") or ""
                    )
                )
            ]
            anchor_count = len(anchors)
            anchor_h = 0.0
            if anchor_count:
                anchor_h, counted = _compute_anchor_h(anchors)
                if counted <= 0:
                    anchor_h = 0.0
                anchor_h = float(anchor_h or 0.20)
            print(
                f"[TEXT-SCAN] anchors={len(anchors)} h_anchor={anchor_h:.2f}"
                if anchors
                else "[TEXT-SCAN] anchors=0"
            )
            total_by_height = len(candidate_entries)
            if anchor_count > 0 and anchor_h > 0 and candidate_entries:
                filtered_entries = _filter_entries_by_anchor_h(
                    candidate_entries, anchor_h=anchor_h
                )
                kept_count = len(filtered_entries)
                if kept_count != total_by_height:
                    candidate_entries = [dict(entry) for entry in filtered_entries]
                    table_lines = [
                        str(entry.get("normalized_text") or "") for entry in candidate_entries
                    ]
                total_kept = len(candidate_entries)
            else:
                total_kept = total_by_height
            print(f"[TEXT-SCAN] kept_by_height={total_kept}/{total_by_height}")
    
            debug_candidates: list[dict[str, Any]] = []
            for entry in candidate_entries:
                debug_candidates.append(
                    {
                        "layout": entry.get("layout_name"),
                        "in_block": bool(entry.get("from_block")),
                        "block": entry.get("block_name"),
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
                if _ADMIN_ROW_DROP_RE.search(line):
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

            merged_rows = _filter_and_dedupe_row_texts(merged_rows)
            rows_txt_initial = len(merged_rows)
            _LAST_TEXT_TABLE_DEBUG["rows_txt_count"] = rows_txt_initial
            _LAST_TEXT_TABLE_DEBUG["rows_txt_lines"] = list(merged_rows)

            if see_sheet_hint_text:
                best_secondary: dict[str, Any] | None = None
                best_score: tuple[float, float, float, float] | None = None
                for layout_index, layout_entries in entries_by_layout.items():
                    layout_name = str(layout_names.get(layout_index, layout_index))
                    if not layout_name or not layout_hint_pattern.search(layout_name):
                        continue
                    candidate = _scan_secondary_layout(layout_index, layout_entries)
                    if candidate is None:
                        continue
                    score_raw = candidate.get("score")
                    if (
                        not isinstance(score_raw, tuple)
                        or len(score_raw) != 4
                    ):
                        continue
                    score_tuple = (
                        float(score_raw[0]),
                        float(score_raw[1]),
                        float(score_raw[2]),
                        float(score_raw[3]),
                    )
                    if best_score is None or score_tuple > best_score:
                        best_secondary = candidate
                        best_score = score_tuple
                if best_secondary is not None:
                    merged_rows = list(best_secondary.get("rows_txt", []))
                    rows_txt_initial = len(merged_rows)
                    table_lines = list(best_secondary.get("table_lines", [])) or list(merged_rows)
                    candidate_entries = [
                        dict(entry) for entry in best_secondary.get("entries", [])
                    ]
                    debug_candidates = []
                    for entry in candidate_entries:
                        debug_candidates.append(
                            {
                                "layout": entry.get("layout_name"),
                                "in_block": bool(entry.get("from_block")),
                                "block": entry.get("block_name"),
                                "x": entry.get("x"),
                                "y": entry.get("y"),
                                "text": entry.get("normalized_text")
                                or entry.get("text")
                                or "",
                            }
                        )
                    if debug_candidates:
                        _LAST_TEXT_TABLE_DEBUG["candidates"] = debug_candidates
                    _LAST_TEXT_TABLE_DEBUG["rows_txt_count"] = rows_txt_initial
                    _LAST_TEXT_TABLE_DEBUG["rows_txt_lines"] = list(merged_rows)
                    _LAST_TEXT_TABLE_DEBUG["secondary_sheet_layout"] = best_secondary.get(
                        "layout_name"
                    )
                    _LAST_TEXT_TABLE_DEBUG["secondary_sheet_anchor_count"] = int(
                        best_secondary.get("anchor_count") or 0
                    )
                    _LAST_TEXT_TABLE_DEBUG["secondary_sheet_median_gap"] = float(
                        best_secondary.get("median_gap") or 0.0
                    )
                    secondary_anchor_candidate = dict(best_secondary)
                    see_sheet_hint_logged = True

            if (
                see_sheet_hint_text
                and not see_sheet_hint_logged
                and rows_txt_initial > 0
            ):
                preview = see_sheet_hint_text
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                print(
                    f"[TEXT-SCAN] hint: HOLE CHART may be on another sheet (\"{preview}\") "
                    f"rows here={rows_txt_initial}"
                )
                see_sheet_hint_logged = True

            return (am_bor_pre_count, am_bor_post_count, am_bor_drop_count)

        am_bor_pre_count, am_bor_post_count, am_bor_drop_count = _perform_text_scan(
            resolved_allowlist
        )
        if isinstance(anchor_authoritative_result, Mapping):
            anchor_result = dict(anchor_authoritative_result)
            rows_for_debug = anchor_result.get("rows")
            if isinstance(rows_for_debug, Iterable) and not isinstance(
                rows_for_debug, list
            ):
                rows_for_debug = list(rows_for_debug)
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                if isinstance(rows_for_debug, list):
                    _LAST_TEXT_TABLE_DEBUG["rows"] = list(rows_for_debug)
                    _LAST_TEXT_TABLE_DEBUG["text_row_count"] = len(rows_for_debug)
                else:
                    _LAST_TEXT_TABLE_DEBUG["rows"] = []
            return anchor_result
        def _parse_rows(row_texts: list[str]) -> tuple[list[dict[str, Any]], dict[str, int], int]:
            families: dict[str, int] = {}
            parsed: list[dict[str, Any]] = []
            total = 0
            seen_keys: set[tuple[int, str]] = set()
            for row_text in row_texts:
                text_value = " ".join((row_text or "").split()).strip()
                if not text_value:
                    continue
                if _ADMIN_ROW_DROP_RE.search(text_value):
                    continue
                match = _ROW_QUANTITY_PATTERNS[0].match(text_value)
                if not match:
                    continue
                qty_text = match.group(1)
                try:
                    qty_val = int(qty_text)
                except Exception:
                    continue
                if qty_val <= 0:
                    continue
                remainder = text_value[match.end() :].strip()
                desc_value = " ".join(remainder.split()) or text_value
                if _ADMIN_ROW_DROP_RE.search(desc_value):
                    continue
                side_value = _detect_row_side(text_value) or _detect_row_side(desc_value)
                ref_text, ref_value = _extract_row_reference(desc_value)
                normalized_key = (qty_val, " ".join(desc_value.split()).upper())
                if normalized_key in seen_keys:
                    continue
                seen_keys.add(normalized_key)
                row_dict: dict[str, Any] = {
                    "hole": "",
                    "qty": qty_val,
                    "desc": desc_value,
                    "ref": ref_text or "",
                }
                if side_value:
                    row_dict["side"] = side_value
                parsed.append(row_dict)
                total += qty_val
                if ref_value is not None:
                    key = f"{ref_value:.4f}".rstrip("0").rstrip(".")
                    families[key] = families.get(key, 0) + qty_val
            return (parsed, families, total)

        total_qty = 0
        parsed_rows, families, total_qty = _parse_rows(merged_rows)
        anchor_rows_primary = list(parsed_rows)
        anchor_qty_total = _sum_qty(anchor_rows_primary)
        anchor_is_authoritative = len(anchor_rows_primary) >= 2
        anchor_mode = "authoritative" if anchor_is_authoritative else "fallback"
        print(
            f"[TEXT-SCAN] pass=anchor rows={len(anchor_rows_primary)} ({anchor_mode})"
        )

        if anchor_is_authoritative:
            anchor_layouts: set[int] = set()
            layout_band_ranges: dict[int, tuple[float, float]] = {}
            for entry in candidate_entries:
                normalized = entry.get("normalized_text") or entry.get("text")
                if not _line_is_table_row_start(normalized):
                    continue
                layout_idx = _coerce_layout_index(entry.get("layout_index"))
                anchor_layouts.add(layout_idx)
                y_val = entry.get("y")
                if isinstance(y_val, (int, float)):
                    y_float = float(y_val)
                    low_high = layout_band_ranges.get(layout_idx)
                    if low_high is None:
                        layout_band_ranges[layout_idx] = (y_float, y_float)
                    else:
                        low, high = low_high
                        layout_band_ranges[layout_idx] = (
                            min(low, y_float),
                            max(high, y_float),
                        )
            anchor_layouts = {idx for idx in anchor_layouts if idx is not None}
            if anchor_layouts:
                anchor_band_entries: list[dict[str, Any]] = []
                band_drop_counts: defaultdict[int, int] = defaultdict(int)
                for entry in collected_entries:
                    if not isinstance(entry, Mapping):
                        continue
                    layout_idx = _coerce_layout_index(entry.get("layout_index"))
                    if layout_idx not in anchor_layouts:
                        continue
                    entry_copy = dict(entry)
                    entry_copy["normalized_text"] = _normalize_candidate_text(
                        entry_copy.get("normalized_text") or entry_copy.get("text")
                    )
                    layout_name_value = layout_names.get(layout_idx, layout_idx)
                    layout_upper = str(layout_name_value or "").strip().upper()
                    if layout_upper in am_bor_band_layouts:
                        band_limits = layout_band_ranges.get(layout_idx)
                        if band_limits is not None:
                            y_val = entry_copy.get("y")
                            if isinstance(y_val, (int, float)):
                                y_float = float(y_val)
                                lower, upper = band_limits
                                band_min = min(lower, upper)
                                band_max = max(lower, upper)
                                if anchor_h > 0:
                                    band_margin = max(anchor_h * 1.5, 6.0)
                                else:
                                    band_margin = max((band_max - band_min) * 0.5, 6.0)
                                lower_bound = band_min - band_margin
                                upper_bound = band_max + band_margin
                                if y_float < lower_bound or y_float > upper_bound:
                                    band_drop_counts[layout_idx] += 1
                                    continue
                    anchor_band_entries.append(entry_copy)
                if band_drop_counts:
                    for layout_idx, drop_count in band_drop_counts.items():
                        layout_name = layout_names.get(layout_idx, layout_idx)
                        print(
                            "[BAND] layout={layout} trimmed={count}".format(
                                layout=layout_name,
                                count=drop_count,
                            )
                        )
                anchor_band_context = {
                    "entries": anchor_band_entries,
                    "anchor_entries": [
                        dict(entry)
                        for entry in candidate_entries
                        if _line_is_table_row_start(
                            entry.get("normalized_text") or entry.get("text")
                        )
                    ],
                    "anchor_h": anchor_h,
                    "layout_order": [
                        idx for idx in layout_order if _coerce_layout_index(idx) in anchor_layouts
                    ],
                }
            anchor_payload_rows = [dict(row) for row in anchor_rows_primary]
            anchor_authoritative_result = {
                "rows": anchor_payload_rows,
                "hole_count": anchor_qty_total,
                "provenance_holes": "HOLE TABLE (anchor)",
                "source": "text_table",
                "header_validated": True,
                "anchor_authoritative": True,
            }
            text_rows_info = dict(anchor_authoritative_result)
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                _LAST_TEXT_TABLE_DEBUG["anchor_authoritative"] = True
            return (am_bor_pre_count, am_bor_post_count, am_bor_drop_count)

        if follow_sheet_target_layouts:
            follow_debug_entries: list[dict[str, Any]] = []
            for follow_layout in follow_sheet_target_layouts:
                target_upper = str(follow_layout).strip().upper()
                follow_entries: list[dict[str, Any]] = []
                for entry in collected_entries:
                    layout_value = str(entry.get("layout_name") or "").strip()
                    if layout_value.upper() != target_upper:
                        continue
                    text_value = str(entry.get("text") or "").strip()
                    if not text_value:
                        continue
                    follow_entries.append(
                        {
                            "layout_name": layout_value,
                            "from_block": bool(entry.get("from_block")),
                            "x": entry.get("x"),
                            "y": entry.get("y"),
                            "height": entry.get("height"),
                            "text": text_value,
                            "normalized_text": text_value,
                        }
                    )
                if not follow_entries:
                    continue
                follow_candidate, follow_debug_payload = _build_columnar_table_from_entries(
                    follow_entries, roi_hint=roi_hint_effective
                )
                if isinstance(follow_debug_payload, Mapping):
                    follow_debug_entry = dict(follow_debug_payload)
                    follow_debug_entry["layout"] = follow_layout
                    follow_debug_entries.append(follow_debug_entry)
                candidate_score = _score_table(follow_candidate)
                existing_score = _score_table(columnar_table_info)
                if candidate_score > existing_score:
                    columnar_table_info = follow_candidate
                    if isinstance(follow_debug_payload, Mapping):
                        columnar_debug_info = dict(follow_debug_payload)
            if follow_debug_entries and isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                if len(follow_debug_entries) == 1:
                    _LAST_TEXT_TABLE_DEBUG["follow_sheet_debug"] = follow_debug_entries[0]
                else:
                    _LAST_TEXT_TABLE_DEBUG["follow_sheet_debug"] = follow_debug_entries

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

        if "parsed_rows" not in locals():
            parsed_rows = []
        if "total_qty" not in locals():
            total_qty = 0

        if len(parsed_rows) < 8:
            clusters = _cluster_entries_by_y(candidate_entries)
            fallback_rows = _clusters_to_rows(clusters)
            fallback_rows = [row for row in fallback_rows if row]
            fallback_rows = _filter_and_dedupe_row_texts(fallback_rows)
            fallback_parsed, fallback_families, fallback_qty = _parse_rows(fallback_rows)
            print(
                f"[TEXT-SCAN] fallback clusters={len(clusters)} "
                f"chosen_rows={len(fallback_parsed)} qty_sum={fallback_qty}"
            )
            current_total_qty = locals().get("total_qty", 0)
            if fallback_parsed and (
                (fallback_qty, len(fallback_parsed))
                > (scan_totals.get("qty", 0), len(parsed_rows))
            ):
                merged_rows = fallback_rows
                parsed_rows = fallback_parsed
                scan_totals["families"] = fallback_families
                scan_totals["qty"] = fallback_qty

        if merged_rows and len(parsed_rows) == len(merged_rows):
            for idx, fallback_text in enumerate(merged_rows):
                if idx >= len(parsed_rows):
                    break
                fallback_clean = " ".join(str(fallback_text or "").split())
                if not fallback_clean:
                    continue
                current_desc = str(parsed_rows[idx].get("desc") or "")
                if len(fallback_clean) > len(current_desc):
                    if (
                        _RE_TEXT_ROW_START.match(fallback_clean)
                        and not _RE_TEXT_ROW_START.match(current_desc)
                    ):
                        continue
                    leading_token = fallback_clean.split("|", 1)[0].strip()
                    if (
                        leading_token
                        and len(leading_token) == 1
                        and leading_token.isalpha()
                        and not current_desc.startswith(f"{leading_token} |")
                    ):
                        continue
                    parsed_rows[idx]["desc"] = fallback_clean

        if rows_txt_initial > 0 and not parsed_rows:
            print("[PATH-GUARD] rows_txt>0 but text_rows==0; forcing band/column pass")

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
                "block_name": entry.get("block_name"),
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
                "block": item.get("block_name"),
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
                _LAST_TEXT_TABLE_DEBUG["row_debug"] = list(
                    debug_payload.get("row_debug", [])
                )
                _LAST_TEXT_TABLE_DEBUG["columns"] = list(
                    debug_payload.get("columns", [])
                )
                _LAST_TEXT_TABLE_DEBUG["rows"] = list(
                    debug_payload.get("rows_txt_fallback", [])
                )
                _LAST_TEXT_TABLE_DEBUG["bands"] = []
                if "roi" in debug_payload:
                    _LAST_TEXT_TABLE_DEBUG["roi"] = debug_payload.get("roi")

        _LAST_TEXT_TABLE_DEBUG["rows_txt_count"] = len(merged_rows)
        _LAST_TEXT_TABLE_DEBUG["rows_txt_lines"] = list(merged_rows)
        print(f"[TEXT-SCAN] rows_txt count={len(merged_rows)}")
        for idx, row_text in enumerate(merged_rows[:3]):
            print(f"  [{idx:02d}] {row_text}")

        _LAST_TEXT_TABLE_DEBUG["text_row_count"] = len(parsed_rows)
        print(f"[TEXT-SCAN] parsed rows: {len(parsed_rows)}")
        for idx, row in enumerate(parsed_rows[:3]):
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
            current_qty = int(scan_totals.get("qty", 0) or 0)
            text_rows_info = {
                "rows": parsed_rows,
                "hole_count": current_qty,
                "provenance_holes": "HOLE TABLE",
            }
            families_map = scan_totals.get("families")
            if isinstance(families_map, Mapping) and families_map:
                text_rows_info["hole_diam_families_in"] = dict(families_map)
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
    if not lines:
        lines = _collect_table_text_lines(doc, layout_filters=layout_filters)

    def _line_confident(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        if re.match(r"^\(\d+\)|^\d+[xX]?", stripped):
            return True
        upper = stripped.upper()
        if any(token in upper for token in ("Ø", "⌀", "TAP", "C'BORE", "C’BORE", "DRILL", "N.P.T", "NPT")):
            return True
        return False

    candidate_lines = merged_rows if merged_rows else lines
    confidence_high = any(_line_confident(line) for line in candidate_lines)
    _LAST_TEXT_TABLE_DEBUG["confidence_high"] = bool(confidence_high)
    force_columnar = False

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
            if (
                helper_score[1] == 0
                and rows_txt_initial >= 2
                and confidence_high
            ):
                force_columnar = True
                print(
                    "[PATH-GUARD] helper_rows=0 but rows_txt>=2; forcing band/column fallback"
                )

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

    if (
        anchor_is_authoritative
        and len(anchor_rows_primary) <= 3
        and helper_score[1] == 0
        and legacy_score[1] == 0
        and isinstance(anchor_band_context, Mapping)
    ):
        band_lines = _extract_anchor_band_lines(anchor_band_context)
        if band_lines:
            if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                _LAST_TEXT_TABLE_DEBUG["anchor_band_lines"] = list(band_lines)
            anchor_band_result = _fallback_text_table(band_lines)
            if anchor_band_result:
                fallback_candidate = anchor_band_result
                band_score = _score_table(anchor_band_result)
                if band_score[1] > 0 and band_score > best_score:
                    best_candidate = anchor_band_result
                    best_score = band_score
                anchor_authoritative_result = dict(anchor_band_result)
                anchor_rows_primary = _normalize_table_rows(
                    anchor_band_result.get("rows")
                )
                anchor_qty_total = _sum_qty(anchor_rows_primary)
                text_rows_info = dict(anchor_band_result)

    primary_result: dict[str, Any] | None = None
    if isinstance(anchor_authoritative_result, Mapping):
        primary_result = dict(anchor_authoritative_result)
    elif isinstance(best_candidate, Mapping):
        primary_result = dict(best_candidate)
    elif isinstance(text_rows_info, Mapping):
        primary_result = dict(text_rows_info)
    elif isinstance(fallback_candidate, Mapping):
        primary_result = dict(fallback_candidate)

    _PROMOTED_ROWS_LOGGED = False

    columnar_result: dict[str, Any] | None = None
    column_selected = False
    roi_rows_primary: list[dict[str, Any]] = []
    if not anchor_is_authoritative and not anchor_rows_primary:
        if isinstance(columnar_table_info, Mapping):
            columnar_result = dict(columnar_table_info)
            promoted_rows, promoted_qty_sum = _prepare_columnar_promoted_rows(
                columnar_result
            )
            columnar_result["rows"] = promoted_rows
            if promoted_qty_sum > 0:
                columnar_result["hole_count"] = promoted_qty_sum
            columnar_result["source_label"] = "text_table (column-mode+stripe)"

        if columnar_result:
            existing_score = _score_table(primary_result)
            fallback_score = _score_table(columnar_result)
            if fallback_score[1] > 0 and (
                fallback_score > existing_score or force_columnar
            ):
                primary_result = columnar_result
                column_selected = True
                _print_promoted_rows_once(columnar_result.get("rows", []))

        roi_rows_primary = (
            _normalize_table_rows(columnar_result.get("rows"))
            if columnar_result
            else []
        )
        print(f"[TEXT-SCAN] pass=roi rows={len(roi_rows_primary)}")

    if primary_result is None:
        fallback = _fallback_text_table(lines)
        if fallback:
            rows = fallback.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    desc_val = row.get("desc")
                    if isinstance(desc_val, str):
                        cleaned = _FALLBACK_LEADING_QTY_RE.sub("", desc_val).strip()
                        if cleaned:
                            cleaned = re.sub(r"^[A-Z]\s*\|\s*", "", cleaned)
                            cleaned = re.sub(r"\|\s*\(\d+\)\s*", "| ", cleaned)
                        if cleaned:
                            row["desc"] = cleaned
            fallback["provenance_holes"] = "HOLE TABLE"
            if am_bor_included:
                fallback["am_bor_included"] = True
                if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
                    _LAST_TEXT_TABLE_DEBUG["am_bor_included"] = True
            _LAST_TEXT_TABLE_DEBUG["rows"] = list(fallback.get("rows", []))
            return fallback
        _LAST_TEXT_TABLE_DEBUG["rows"] = []
        return {}

    primary_rows = _normalize_table_rows(primary_result.get("rows"))
    merged_unique_rows, dedup_dropped, anchor_authoritative = _combine_text_rows(
        anchor_rows_primary,
        primary_rows,
        roi_rows_primary,
    )
    if not anchor_authoritative:
        print(f"[TEXT-SCAN] merge_dedup final_rows={len(merged_unique_rows)}")
        print(f"[TEXT-SCAN] dedup dropped={dedup_dropped}")
    primary_result = dict(primary_result)
    primary_result["rows"] = merged_unique_rows
    final_qty_sum = _sum_qty(merged_unique_rows)
    if final_qty_sum > 0:
        primary_result["hole_count"] = final_qty_sum

    if column_selected:
        primary_result.setdefault("source", "text_table")
    if isinstance(columnar_debug_info, Mapping):
        row_debug_payload = columnar_debug_info.get("row_debug", [])
        columns_payload = columnar_debug_info.get("columns", [])

        def _as_list(value: Any) -> list[Any]:
            if isinstance(value, list):
                return list(value)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
                return list(value)
            if value in (None, ""):
                return []
            return [value]

        _LAST_TEXT_TABLE_DEBUG["row_debug"] = _as_list(row_debug_payload)
        _LAST_TEXT_TABLE_DEBUG["columns"] = _as_list(columns_payload)
        _LAST_TEXT_TABLE_DEBUG["bands"] = []
    else:
        _LAST_TEXT_TABLE_DEBUG.setdefault("row_debug", [])
        _LAST_TEXT_TABLE_DEBUG.setdefault("columns", [])
        _LAST_TEXT_TABLE_DEBUG.setdefault("bands", [])
    rows_materialized = list(primary_result.get("rows", []))
    if confidence_high and rows_materialized:
        primary_result["confidence_high"] = True
        if len(rows_materialized) >= 3 and not primary_result.get("header_validated"):
            primary_result["header_validated"] = True
    _LAST_TEXT_TABLE_DEBUG["rows"] = rows_materialized
    if am_bor_included:
        primary_result["am_bor_included"] = True
    if isinstance(_LAST_TEXT_TABLE_DEBUG, dict):
        _LAST_TEXT_TABLE_DEBUG["am_bor_included"] = bool(am_bor_included)
    return primary_result


def _cluster_panel_entries(
    entries: list[dict[str, Any]],
    *,
    roi_hint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not entries:
        return []

    usable_records: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
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
            "index": idx,
            "layout": entry.get("layout_name"),
            "from_block": bool(entry.get("from_block")),
            "block_name": entry.get("block_name"),
            "x": x_float,
            "y": y_float,
            "text": text_value,
            "height": entry.get("height"),
        }
        usable_records.append(record)

    if not usable_records:
        return []

    def _filter_entries(bounds: Mapping[str, float]) -> list[dict[str, Any]]:
        xmin = float(bounds.get("xmin", 0.0))
        xmax = float(bounds.get("xmax", 0.0))
        ymin = float(bounds.get("ymin", 0.0))
        ymax = float(bounds.get("ymax", 0.0))
        dx = float(bounds.get("dx", 0.0) or 0.0)
        dy = float(bounds.get("dy", 0.0) or 0.0)
        expanded_xmin = xmin - dx
        expanded_xmax = xmax + dx
        expanded_ymin = ymin - dy
        expanded_ymax = ymax + dy
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            x_val = entry.get("x")
            y_val = entry.get("y")
            try:
                x_float = float(x_val)
                y_float = float(y_val)
            except Exception:
                continue
            if (
                expanded_xmin <= x_float <= expanded_xmax
                and expanded_ymin <= y_float <= expanded_ymax
            ):
                filtered.append(entry)
        return filtered

    all_heights = [
        float(rec["height"])
        for rec in usable_records
        if isinstance(rec.get("height"), (int, float)) and float(rec["height"]) > 0
    ]
    median_height_all = statistics.median(all_heights) if all_heights else 0.0

    def _compute_bounds(
        cluster: list[dict[str, Any]],
        *,
        median_hint: float = 0.0,
    ) -> tuple[dict[str, float], float]:
        xs = [rec["x"] for rec in cluster]
        ys = [rec["y"] for rec in cluster]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        base_dx = 18.0 * median_height_all if median_height_all > 0 else 0.0
        base_dy = 24.0 * median_height_all if median_height_all > 0 else 0.0
        dx = max(40.0, base_dx)
        dy = max(50.0, base_dy)
        if median_hint and median_hint > 0:
            dx = max(dx, 18.0 * median_hint)
            dy = max(dy, 24.0 * median_hint)
        bounds = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "dx": dx,
            "dy": dy,
        }
        return bounds, median_hint

    def _summarize_meta(panel_entries: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        layout_counter: Counter[str] = Counter()
        block_counter: Counter[str] = Counter()
        for item in panel_entries:
            layout_value = str(item.get("layout_name") or "").strip()
            block_value = str(item.get("block_name") or "").strip()
            if layout_value:
                layout_counter[layout_value] += 1
            if block_value:
                block_counter[block_value] += 1
        layout_name = layout_counter.most_common(1)[0][0] if layout_counter else None
        block_name = block_counter.most_common(1)[0][0] if block_counter else None
        return layout_name, block_name

    panels: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    def _register_panel(
        *,
        source: str,
        bounds: Mapping[str, float],
        entries_subset: list[dict[str, Any]],
        roi_info: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        median_hint: float = 0.0,
    ) -> None:
        if not entries_subset:
            return
        key = (
            source,
            round(float(bounds.get("xmin", 0.0)), 1),
            round(float(bounds.get("xmax", 0.0)), 1),
            round(float(bounds.get("ymin", 0.0)), 1),
            round(float(bounds.get("ymax", 0.0)), 1),
        )
        if key in seen_keys:
            return
        seen_keys.add(key)
        layout_name, block_name = _summarize_meta(entries_subset)
        meta_payload = {
            "source": source,
            "layout": layout_name,
            "block": block_name,
        }
        if metadata:
            for k, v in metadata.items():
                if v is None or v == "":
                    continue
                meta_payload.setdefault(k, v)
        pad_val = max(float(bounds.get("dx", 0.0) or 0.0), float(bounds.get("dy", 0.0) or 0.0))
        roi_hint_payload: dict[str, Any] = {
            "source": source,
            "bbox": [
                float(bounds.get("xmin", 0.0)),
                float(bounds.get("xmax", 0.0)),
                float(bounds.get("ymin", 0.0)),
                float(bounds.get("ymax", 0.0)),
            ],
            "pad": pad_val,
        }
        if median_hint and median_hint > 0:
            roi_hint_payload["median_height"] = median_hint
        panel_entry = {
            "entries": list(entries_subset),
            "meta": meta_payload,
            "bounds": dict(bounds),
            "roi_info": dict(roi_info) if isinstance(roi_info, Mapping) else None,
            "roi_hint": roi_hint_payload,
        }
        panels.append(panel_entry)

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
            try:
                pad_val = float(roi_hint.get("pad") or 0.0)
            except Exception:
                pad_val = 0.0
            bounds = {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "dx": pad_val,
                "dy": pad_val,
            }
            roi_median_height = 0.0
            try:
                roi_median_height = float(roi_hint.get("median_height") or 0.0)
            except Exception:
                roi_median_height = 0.0
            subset = _filter_entries(bounds)
            if subset:
                source_label = str(roi_hint.get("source") or "ROI_HINT")
                roi_info = {
                    "source": source_label,
                    "bbox": [xmin, xmax, ymin, ymax],
                    "pad": pad_val,
                    "kept": len(subset),
                }
                metadata = {
                    "block": roi_hint.get("name"),
                    "layer": roi_hint.get("layer"),
                }
                _register_panel(
                    source=source_label,
                    bounds=bounds,
                    entries_subset=subset,
                    roi_info=roi_info,
                    metadata=metadata,
                    median_hint=roi_median_height,
                )

    anchor_lines = [rec for rec in usable_records if _ROI_ANCHOR_RE.search(rec["text"])]
    if anchor_lines:
        sorted_anchors = sorted(anchor_lines, key=lambda rec: -rec["y"])
        anchor_count = len(sorted_anchors)
        clusters: list[list[dict[str, Any]]] = []
        if sorted_anchors:
            height_values = [
                float(rec["height"])
                for rec in sorted_anchors
                if isinstance(rec.get("height"), (int, float)) and float(rec["height"]) > 0
            ]
            anchor_y_diffs = [
                abs(sorted_anchors[idx]["y"] - sorted_anchors[idx - 1]["y"])
                for idx in range(1, len(sorted_anchors))
                if abs(sorted_anchors[idx]["y"] - sorted_anchors[idx - 1]["y"]) > 0
            ]
            if height_values:
                median_height = statistics.median(height_values)
                roi_median_height = median_height
                y_anchor_eps = 1.8 * median_height if median_height > 0 else 0.0
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
        for cluster_index, cluster in enumerate(clusters):
            if not cluster:
                continue
            bounds, median_hint = _compute_bounds(cluster, median_hint=roi_median_height)
            entries_subset = _filter_entries(bounds)
            roi_info = {
                "anchors": len(cluster),
                "clusters": len(clusters),
                "bbox": [
                    bounds["xmin"],
                    bounds["xmax"],
                    bounds["ymin"],
                    bounds["ymax"],
                ],
                "total": len(usable_records),
                "cluster_index": cluster_index,
            }
            metadata = {
                "anchors": len(cluster),
                "cluster_index": cluster_index,
            }
            _register_panel(
                source="ANCHOR_CLUSTER",
                bounds=bounds,
                entries_subset=entries_subset,
                roi_info=roi_info,
                metadata=metadata,
                median_hint=median_hint,
            )

    block_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in usable_records:
        if not rec.get("from_block"):
            continue
        block_name = str(rec.get("block_name") or "").strip()
        if not block_name:
            continue
        layout_name = str(rec.get("layout") or "").strip()
        block_groups[(layout_name, block_name)].append(rec)
    for (layout_name, block_name), records_block in block_groups.items():
        if not records_block:
            continue
        heights = [
            float(rec.get("height"))
            for rec in records_block
            if isinstance(rec.get("height"), (int, float)) and float(rec.get("height")) > 0
        ]
        median_hint = statistics.median(heights) if heights else median_height_all
        bounds, median_val = _compute_bounds(records_block, median_hint=median_hint)
        entries_subset = _filter_entries(bounds)
        if not entries_subset:
            continue
        roi_info = {
            "source": "BLOCK_GROUP",
            "bbox": [
                bounds["xmin"],
                bounds["xmax"],
                bounds["ymin"],
                bounds["ymax"],
            ],
            "block": block_name,
            "layout": layout_name,
            "count": len(entries_subset),
        }
        metadata = {
            "block": block_name,
            "layout": layout_name,
        }
        _register_panel(
            source="BLOCK_GROUP",
            bounds=bounds,
            entries_subset=entries_subset,
            roi_info=roi_info,
            metadata=metadata,
            median_hint=median_val,
        )

    if not panels:
        layout_name, block_name = _summarize_meta(entries)
        fallback_meta = {
            "source": "FALLBACK",
            "layout": layout_name,
            "block": block_name,
        }
        fallback_hint = roi_hint if isinstance(roi_hint, Mapping) else None
        panels.append(
            {
                "entries": list(entries),
                "meta": fallback_meta,
                "bounds": None,
                "roi_info": None,
                "roi_hint": dict(fallback_hint) if isinstance(fallback_hint, Mapping) else None,
            }
        )

    return panels


def _build_columnar_table_from_entries(
    entries: list[dict[str, Any]],
    *,
    roi_hint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    panels = _cluster_panel_entries(entries, roi_hint=roi_hint)
    if len(panels) <= 1:
        sole_panel = panels[0] if panels else None
        panel_hint = sole_panel.get("roi_hint") if sole_panel else roi_hint
        panel_entries = sole_panel.get("entries") if sole_panel else entries
        return _build_columnar_table_from_panel_entries(
            list(panel_entries or []),
            roi_hint=panel_hint,
        )

    panel_results: list[dict[str, Any]] = []
    for idx, panel in enumerate(panels):
        panel_entries = list(panel.get("entries") or [])
        panel_hint = panel.get("roi_hint")
        candidate, debug_payload = _build_columnar_table_from_panel_entries(
            panel_entries,
            roi_hint=panel_hint,
        )
        normalized_rows = _normalize_table_rows(candidate.get("rows")) if candidate else []
        panel_debug = {
            "index": idx,
            "rows": len(normalized_rows),
        }
        meta = panel.get("meta") or {}
        panel_debug.update({k: v for k, v in meta.items() if v not in (None, "")})
        if isinstance(debug_payload, Mapping):
            panel_debug["roi"] = debug_payload.get("roi") or panel.get("roi_info")
        elif panel.get("roi_info") is not None:
            panel_debug["roi"] = panel.get("roi_info")
        panel_results.append(
            {
                "candidate": candidate,
                "debug": debug_payload,
                "rows": normalized_rows,
                "panel": panel,
                "panel_debug": panel_debug,
            }
        )

    best_candidate: dict[str, Any] | None = None
    best_debug: dict[str, Any] | None = None
    for result in panel_results:
        candidate = result.get("candidate")
        debug_payload = result.get("debug")
        if candidate is None:
            if best_candidate is None and isinstance(debug_payload, Mapping):
                best_debug = dict(debug_payload)
            continue
        if best_candidate is None or _score_table(candidate) > _score_table(best_candidate):
            best_candidate = candidate
            best_debug = dict(debug_payload) if isinstance(debug_payload, Mapping) else None

    merged_rows: list[dict[str, Any]] = []
    seen_row_keys: set[tuple[Any, ...]] = set()
    for result in panel_results:
        for row in result.get("rows", []):
            qty_val = row.get("qty")
            try:
                qty_key = int(qty_val) if qty_val is not None else None
            except Exception:
                qty_key = None
            desc_value = " ".join(str(row.get("desc") or "").split())
            key = (qty_key, desc_value.upper())
            if key in seen_row_keys:
                continue
            seen_row_keys.add(key)
            merged_rows.append(dict(row))

    if best_candidate is None:
        # Fallback to the highest scoring panel even if no candidate produced rows
        for result in panel_results:
            candidate = result.get("candidate")
            if candidate is not None:
                best_candidate = candidate
                best_debug = dict(result.get("debug") or {})
                break

    if best_candidate is None:
        return _build_columnar_table_from_panel_entries(entries, roi_hint=roi_hint)

    combined_candidate = dict(best_candidate)
    if merged_rows:
        combined_candidate["rows"] = merged_rows
        qty_total = 0
        for row in merged_rows:
            qty_val = row.get("qty")
            try:
                qty_int = int(qty_val)
            except Exception:
                qty_int = 0
            if qty_int > 0:
                qty_total += qty_int
        if qty_total > 0:
            combined_candidate["hole_count"] = qty_total

    aggregated_debug: dict[str, Any] = {}
    if isinstance(best_debug, Mapping):
        aggregated_debug.update(best_debug)
    aggregated_debug["panels"] = [result["panel_debug"] for result in panel_results]

    return combined_candidate, aggregated_debug


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


def split_actions(desc: str) -> list[str]:
    if not desc:
        return []
    pieces: list[str] = []
    for fragment in _OPS_SEGMENT_SPLIT_RE.split(str(desc)):
        cleaned = fragment.strip(" ;")
        if cleaned:
            pieces.append(cleaned)
    return pieces


def classify_action(fragment: str) -> dict[str, Any]:
    text = " ".join(str(fragment or "").split()).strip()
    upper = text.upper()
    result: dict[str, Any] = {
        "kind": "unknown",
        "qty": 1,
        "size": None,
        "side": _detect_row_side(text),
        "npt": False,
    }
    if not text:
        return result

    is_pipe_tap = bool(_PIPE_TAP_TOKEN_RE.search(text))
    is_npt = bool(_NPT_TOKEN_RE.search(text) or is_pipe_tap)
    if _TAP_WORD_TOKEN_RE.search(text) or _TAP_THREAD_TOKEN_RE.search(text) or is_npt:
        result["kind"] = "tap"
        if is_npt:
            result["npt"] = True
            result["tap_type"] = "pipe"
        return result

    if _COUNTERBORE_TOKEN_RE.search(upper):
        result["kind"] = "counterbore"
        return result

    if _COUNTERDRILL_TOKEN_RE.search(upper):
        result["kind"] = "counterdrill"
        return result

    if _COUNTERSINK_TOKEN_RE.search(upper):
        result["kind"] = "csink"
        return result

    if _JIG_GRIND_TOKEN_RE.search(upper):
        result["kind"] = "jig_grind"
        return result

    if _SPOT_TOKEN_RE.search(upper) and not _TAP_WORD_TOKEN_RE.search(text):
        result["kind"] = "spot"
        return result

    if _DRILL_TOKEN_RE.search(upper):
        drill_size: str | None = None
        for pattern in _DRILL_SIZE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            token = match.group(1) if match.lastindex else match.group(0)
            if token:
                drill_size = str(token).strip()
                break
        result["kind"] = "drill"
        if drill_size:
            result["size"] = drill_size
        return result

    return result


def _normalize_geom_holes_payload(
    geom_holes: Any = None, hole_sets: Any = None
) -> dict[str, Any]:
    groups_counter: defaultdict[float, int] = defaultdict(int)
    total_candidates: list[int] = []
    residual_centers: set[tuple[float, float]] = set()
    non_drill_centers: set[tuple[float, float]] = set()
    residual_hole_keys: set[tuple[float, float, float]] = set()
    residual_holes_payload: list[dict[str, float]] = []

    def _center_key(point: tuple[float, float]) -> tuple[float, float]:
        return (
            round(float(point[0]), _GEO_CIRCLE_CENTER_GROUP_DIGITS),
            round(float(point[1]), _GEO_CIRCLE_CENTER_GROUP_DIGITS),
        )

    def _consume_centers(source: Any, target: set[tuple[float, float]]) -> None:
        if source is None:
            return
        if isinstance(source, Mapping):
            if "x" in source or "y" in source:
                point = _point2d(source)
                if point is not None:
                    target.add(_center_key(point))
                return
            for key in ("center", "point"):
                if key in source:
                    point = _point2d(source.get(key))
                    if point is not None:
                        target.add(_center_key(point))
            for value in source.values():
                if isinstance(value, Mapping) or (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                ):
                    _consume_centers(value, target)
            return
        if isinstance(source, Sequence) and not isinstance(
            source, (str, bytes, bytearray)
        ):
            for item in source:
                _consume_centers(item, target)
            return
        point = _point2d(source)
        if point is not None:
            target.add(_center_key(point))

    def _ingest_group(entry: Mapping[str, Any]) -> None:
        dia_candidate = (
            entry.get("dia_in")
            or entry.get("diam_in")
            or entry.get("diameter_in")
            or entry.get("diam")
            or entry.get("dia")
        )
        qty_candidate = (
            entry.get("count")
            or entry.get("qty")
            or entry.get("quantity")
            or entry.get("total")
        )
        try:
            dia_value = float(dia_candidate)
        except Exception:
            return
        try:
            qty_value = int(float(qty_candidate))
        except Exception:
            return
        if qty_value <= 0 or dia_value <= 0:
            return
        dia_key = round(dia_value, 4)
        groups_counter[dia_key] += qty_value

    def _ingest_source(source: Any) -> None:
        if source is None:
            return
        if isinstance(source, Mapping):
            total_val = source.get("total")
            if total_val not in (None, ""):
                try:
                    total_candidates.append(int(float(total_val)))
                except Exception:
                    pass
            _consume_centers(source.get("residual_centers"), residual_centers)
            holes_value = source.get("residual_holes")
            _consume_centers(holes_value, residual_centers)
            if isinstance(holes_value, Sequence):
                for hole in holes_value:
                    if not isinstance(hole, Mapping):
                        continue
                    point = _point2d(hole)
                    if point is None:
                        continue
                    dia_val = (
                        hole.get("dia_in")
                        or hole.get("diam_in")
                        or hole.get("diameter_in")
                        or hole.get("diameter")
                    )
                    try:
                        hx, hy = float(point[0]), float(point[1])
                        hd = float(dia_val) if dia_val is not None else 0.0
                    except Exception:
                        continue
                    key = (
                        round(hx, _GEO_CIRCLE_CENTER_GROUP_DIGITS),
                        round(hy, _GEO_CIRCLE_CENTER_GROUP_DIGITS),
                        round(hd, 4),
                    )
                    if key in residual_hole_keys:
                        continue
                    residual_hole_keys.add(key)
                    residual_holes_payload.append({"x": hx, "y": hy, "dia_in": hd})
            _consume_centers(source.get("centers"), residual_centers)
            _consume_centers(source.get("non_drill_centers"), non_drill_centers)
            for key in ("hole_count", "hole_count_geom", "hole_count_geom_dedup"):
                value = source.get(key)
                if value not in (None, ""):
                    try:
                        total_candidates.append(int(float(value)))
                    except Exception:
                        continue
            groups_val = source.get("groups") or source.get("hole_groups")
            if isinstance(groups_val, Sequence):
                for item in groups_val:
                    if isinstance(item, Mapping):
                        _ingest_group(item)
            families = source.get("hole_diam_families_in")
            if isinstance(families, Mapping):
                for dia_text, qty_val in families.items():
                    try:
                        dia_value = float(dia_text)
                        qty_value = int(float(qty_val))
                    except Exception:
                        continue
                    if qty_value > 0 and dia_value > 0:
                        dia_key = round(dia_value, 4)
                        groups_counter[dia_key] += qty_value
            for key in ("sets", "hole_sets"):
                alt = source.get(key)
                if isinstance(alt, Sequence):
                    for item in alt:
                        if isinstance(item, Mapping):
                            _ingest_group(item)
        elif isinstance(source, Sequence):
            for item in source:
                if isinstance(item, Mapping):
                    _ingest_group(item)
                else:
                    _consume_centers(item, residual_centers)

    _ingest_source(geom_holes)
    _ingest_source(hole_sets)

    groups = [
        {"dia_in": float(dia_key), "count": count}
        for dia_key, count in sorted(groups_counter.items())
        if count > 0
    ]
    total = sum(entry["count"] for entry in groups)
    if total <= 0 and total_candidates:
        total = max(total_candidates)
    residual_center_list = sorted(residual_centers)
    non_drill_center_list = sorted(non_drill_centers)
    residual_candidate_set = {
        key for key in residual_centers if key not in non_drill_centers
    }
    payload: dict[str, Any] = {
        "groups": groups,
        "total": int(total),
        "center_count": len(residual_center_list),
        "residual_centers": [
            {"x": center[0], "y": center[1]} for center in residual_center_list
        ],
        "non_drill_centers": [
            {"x": center[0], "y": center[1]} for center in non_drill_center_list
        ],
        "residual_candidates": len(residual_candidate_set),
        "residual_holes": residual_holes_payload,
    }
    if not payload["total"] and residual_center_list:
        payload["total"] = len(residual_center_list)
    return payload


def ops_manifest(
    chart_rows: Iterable[Mapping[str, Any]] | None,
    geom_holes: Mapping[str, Any] | None = None,
    *,
    hole_sets: Any = None,
    authoritative_table: bool = False,
) -> dict[str, Any]:
    table_keys = (
        "drill",
        "tap",
        "counterbore",
        "counterdrill",
        "csink",
        "jig_grind",
        "spot",
        "unknown",
    )
    table_counts: dict[str, int] = {key: 0 for key in table_keys}
    details: dict[str, Any] = {"npt": 0, "drill_sized": 0, "drill_sizes": {}}

    rows_iter = chart_rows or []
    table_rows_present = False
    drill_groups: Counter[tuple[str, str]] = Counter()
    tap_implied_candidates: list[tuple[int, tuple[str, str] | None, Any]] = []

    def _row_value(row_obj: Any, key: str) -> Any:
        if isinstance(row_obj, Mapping):
            return row_obj.get(key)
        return getattr(row_obj, key, None)

    def _normalize_key_token(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text.upper()

    def _row_group_key(row_obj: Any, desc: str) -> tuple[str, str] | None:
        hole_token = _normalize_key_token(
            _row_value(row_obj, "hole") or _row_value(row_obj, "hole_id")
        )
        ref_token = _normalize_key_token(
            _row_value(row_obj, "ref")
            or _row_value(row_obj, "drill_ref")
            or _row_value(row_obj, "pilot")
        )
        if not ref_token:
            extracted_ref, _ = _extract_row_reference(desc)
            ref_token = _normalize_key_token(extracted_ref)
        if hole_token or ref_token:
            return (hole_token, ref_token)
        return None

    def _set_row_drill_implied(row_obj: Any, value: bool) -> None:
        if isinstance(row_obj, MutableMapping):
            tags_value = row_obj.get("tags")
            if not isinstance(tags_value, MutableMapping):
                if value:
                    row_obj["tags"] = {"drill_implied": True}
                else:
                    if "tags" in row_obj:
                        try:
                            row_obj.pop("tags")
                        except Exception:
                            pass
                return
            if value:
                tags_value["drill_implied"] = True
            else:
                tags_value.pop("drill_implied", None)
                if not tags_value:
                    try:
                        row_obj.pop("tags")
                    except Exception:
                        pass
            return

        tags_value = getattr(row_obj, "tags", None)
        if not isinstance(tags_value, MutableMapping):
            if value:
                try:
                    setattr(row_obj, "tags", {"drill_implied": True})
                except Exception:
                    pass
            else:
                if hasattr(row_obj, "tags"):
                    try:
                        delattr(row_obj, "tags")
                    except Exception:
                        pass
            return
        if value:
            tags_value["drill_implied"] = True
        else:
            tags_value.pop("drill_implied", None)
            if not tags_value:
                try:
                    delattr(row_obj, "tags")
                except Exception:
                    pass

    for row in rows_iter:
        if not isinstance(row, Mapping):
            continue
        qty_val = row.get("qty")
        try:
            qty = int(float(qty_val or 0))
        except Exception:
            qty = 0
        if qty <= 0:
            continue
        table_rows_present = True
        desc_source = None
        for key in ("desc", "description", "text", "hole"):
            if key in row and row[key]:
                desc_source = row[key]
                break
        desc_text = str(desc_source or "")
        group_key = _row_group_key(row, desc_text)
        row_has_drill = False
        row_has_tap = False
        fragments = split_actions(desc_text) or [desc_text]
        for fragment in fragments:
            action = classify_action(fragment)
            kind = action.get("kind") or ""
            if kind not in table_counts:
                kind = "unknown"
            table_counts[kind] += qty
            if kind == "tap" and action.get("npt"):
                details["npt"] += qty
                row_has_tap = True
            elif kind == "tap":
                row_has_tap = True
            if kind == "drill":
                row_has_drill = True
                size_token = action.get("size")
                if size_token:
                    details["drill_sized"] += qty
                    size_map = details.setdefault("drill_sizes", {})
                    size_map[size_token] = size_map.get(size_token, 0) + qty

        if row_has_drill and not row_has_tap and group_key:
            drill_groups[group_key] += qty
        if row_has_tap and not row_has_drill:
            tap_implied_candidates.append((qty, group_key, row))
        elif row_has_tap:
            _set_row_drill_implied(row, False)

    def _apply_aliases(counts: dict[str, int]) -> dict[str, int]:
        if "counterbore" in counts and "cbore" not in counts:
            counts["cbore"] = counts["counterbore"]
        if "counterdrill" in counts and "cdrill" not in counts:
            counts["cdrill"] = counts["counterdrill"]
        return counts

    table_counts = _apply_aliases(table_counts)

    implied_drill_total = 0
    if not authoritative_table:
        leftover_drill = Counter(drill_groups)
        for qty, key, row in tap_implied_candidates:
            matched = 0
            if key is not None and leftover_drill.get(key, 0) > 0:
                available = leftover_drill[key]
                matched = min(available, qty)
                leftover_drill[key] -= matched
            implied_qty = qty - matched
            if implied_qty > 0:
                implied_drill_total += implied_qty
                _set_row_drill_implied(row, True)
            else:
                _set_row_drill_implied(row, False)
    else:
        for _qty, _key, row in tap_implied_candidates:
            _set_row_drill_implied(row, False)
    details["drill_implied_from_taps"] = implied_drill_total

    geom_info = _normalize_geom_holes_payload(geom_holes, hole_sets)

    def _center_key(point: tuple[float, float]) -> tuple[float, float]:
        return (
            round(float(point[0]), _GEO_CIRCLE_CENTER_GROUP_DIGITS),
            round(float(point[1]), _GEO_CIRCLE_CENTER_GROUP_DIGITS),
        )

    def _collect_centers(candidate: Any) -> set[tuple[float, float]]:
        centers: set[tuple[float, float]] = set()
        if candidate is None:
            return centers
        if isinstance(candidate, Mapping):
            if "x" in candidate or "y" in candidate:
                point = _point2d(candidate)
                if point is not None:
                    centers.add(_center_key(point))
                    return centers
            for value in candidate.values():
                if isinstance(value, Mapping) or (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                ):
                    centers.update(_collect_centers(value))
            return centers
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            for item in candidate:
                centers.update(_collect_centers(item))
            return centers
        point = _point2d(candidate)
        if point is not None:
            centers.add(_center_key(point))
        return centers

    residual_centers_set = _collect_centers(
        geom_info.get("residual_centers") or geom_info.get("residual_holes")
    )
    non_drill_centers_set = _collect_centers(geom_info.get("non_drill_centers"))
    residual_candidate_set = {
        center for center in residual_centers_set if center not in non_drill_centers_set
    }

    geom_total = int(geom_info.get("total") or 0)
    center_count_val = int(geom_info.get("center_count") or 0)
    if center_count_val > geom_total:
        geom_total = center_count_val
    if len(residual_centers_set) > geom_total:
        geom_total = len(residual_centers_set)
    sized_drill_qty = int(details.get("drill_sized") or 0)
    text_drill_qty = _coerce_positive_int(table_counts.get("drill")) or 0
    residual_candidate_total = len(residual_candidate_set)
    if not residual_candidate_total:
        try:
            residual_candidate_total = int(geom_info.get("residual_candidates") or 0)
        except Exception:
            residual_candidate_total = 0
    geom_residual_base = residual_candidate_total if residual_candidate_total else geom_total
    geom_residual = (
        max(geom_residual_base - text_drill_qty, 0)
        if geom_residual_base and text_drill_qty
        else geom_residual_base
    )

    table_drill_only = int(table_counts.get("drill", 0))
    table_tap = int(table_counts.get("tap", 0))
    table_counterbore = int(table_counts.get("cbore", 0))
    table_counterdrill = int(table_counts.get("cdrill", 0))
    table_jig = int(table_counts.get("jig_grind", 0))

    table_manifest: dict[str, int] = {
        "drill_only": table_drill_only,
        "tap": table_tap,
        "counterbore": table_counterbore,
        "counterdrill": table_counterdrill,
        "jig_grind": table_jig,
        "drill": table_drill_only,
        "cbore": table_counterbore,
        "cdrill": table_counterdrill,
        "csink": int(table_counts.get("csink", 0)),
        "spot": int(table_counts.get("spot", 0)),
        "unknown": int(table_counts.get("unknown", 0)),
    }

    table_drill_total = table_drill_only + implied_drill_total
    if authoritative_table:
        total_drill = table_drill_only
    elif table_rows_present:
        total_drill = sized_drill_qty + geom_residual + implied_drill_total
    else:
        total_drill = max(geom_total, table_drill_total)

    total_counts: dict[str, int] = {
        "drill": total_drill,
        "tap": table_tap,
        "counterbore": table_counterbore,
        "counterdrill": table_counterdrill,
        "jig_grind": table_jig,
        "cbore": table_counterbore,
        "cdrill": table_counterdrill,
    }

    total_counts = _apply_aliases(total_counts)

    text_info = {"estimated_total_drills": int(table_counts.get("drill", 0))}

    geom_manifest = {
        "drill": geom_total,
        "groups": geom_info.get("groups", []),
        "total": geom_total,
        "drill_residual": geom_residual,
        "residual_drill": geom_residual,
    }

    manifest = {
        "table": table_manifest,
        "geom": geom_manifest,
        "total": total_counts,
        "details": details,
        "text": text_info,
    }
    if authoritative_table:
        manifest["authoritative_table"] = True
        manifest["table_authoritative"] = True
    return manifest


_LAST_GEO_OUTLINE_HINT: Mapping[str, Any] | None = None


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    if len(points) < 3:
        return area
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _polyline_vertices_in(
    entity: Any,
    transform: TransformMatrix,
    to_in: float,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    get_points = getattr(entity, "get_points", None)
    if callable(get_points):
        try:
            raw_pts = list(get_points("xy"))
        except TypeError:
            raw_pts = list(get_points())  # type: ignore[call-arg]
        except Exception:
            raw_pts = []
        for raw in raw_pts:
            if not isinstance(raw, Sequence) or len(raw) < 2:
                continue
            try:
                local = (float(raw[0]), float(raw[1]))
            except Exception:
                continue
            world = _apply_transform_point(transform, local)
            if world[0] is None or world[1] is None:
                continue
            pts.append((float(world[0]) * to_in, float(world[1]) * to_in))
    else:
        vertices = getattr(entity, "vertices", None)
        if vertices is not None:
            try:
                iterator = list(vertices)
            except Exception:
                iterator = []
            for vertex in iterator:
                location = getattr(getattr(vertex, "dxf", vertex), "location", None)
                point = _point2d(location)
                if point is None:
                    continue
                world = _apply_transform_point(transform, point)
                if world[0] is None or world[1] is None:
                    continue
                pts.append((float(world[0]) * to_in, float(world[1]) * to_in))
    return pts


def _find_polyline_bbox_in(
    layout: Any,
    to_in: float,
    exclude_patterns: Sequence[re.Pattern[str]],
) -> tuple[float, float, float, float] | None:
    best: tuple[float, tuple[float, float, float, float]] | None = None

    def _is_closed(candidate: Any) -> bool:
        if getattr(candidate, "closed", False):
            return True
        dxf_obj = getattr(candidate, "dxf", None)
        flags = getattr(dxf_obj, "flags", 0) if dxf_obj is not None else 0
        if isinstance(flags, int) and flags & 1:
            return True
        try:
            vertices = list(getattr(candidate, "vertices", []))
        except Exception:
            vertices = []
        if len(vertices) >= 2:
            first = _point2d(getattr(vertices[0], "dxf", vertices[0]).location)
            last = _point2d(getattr(vertices[-1], "dxf", vertices[-1]).location)
            if first and last and abs(first[0] - last[0]) < 1e-6 and abs(first[1] - last[1]) < 1e-6:
                return True
        return False

    def _maybe_update(candidate: Any) -> None:
        nonlocal best
        layer_name = (
            str(getattr(getattr(candidate, "dxf", object()), "layer", "") or "")
        ).upper()
        if layer_name and any(pattern.search(layer_name) for pattern in exclude_patterns):
            return
        if not _is_closed(candidate):
            return
        pts = _polyline_vertices_in(candidate, _IDENTITY_TRANSFORM, to_in)
        if len(pts) < 3:
            return
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        if not xs or not ys:
            return
        area = abs(_polygon_area(pts))
        if area <= 0.0:
            return
        bbox = (min(xs), max(xs), min(ys), max(ys))
        if best is None or area > best[0]:
            best = (area, bbox)

    if layout is None:
        return None

    for spec in ("LWPOLYLINE", "POLYLINE"):
        try:
            entities = list(layout.query(spec))
        except Exception:
            entities = []
        for entity in entities:
            _maybe_update(entity)

    return best[1] if best else None


def _collect_entity_points_in(
    flattened: FlattenedEntity,
    to_in: float,
) -> list[tuple[float, float]]:
    entity = flattened.entity
    try:
        dxftype = entity.dxftype()
    except Exception:
        return []
    dxftype_upper = str(dxftype or "").upper()
    points: list[tuple[float, float]] = []
    transform = flattened.transform

    if dxftype_upper == "LINE":
        for attr in ("start", "end"):
            point = _point2d(getattr(getattr(entity, "dxf", entity), attr, None))
            if point is None:
                continue
            world = _apply_transform_point(transform, point)
            if world[0] is None or world[1] is None:
                continue
            points.append((float(world[0]) * to_in, float(world[1]) * to_in))
    elif dxftype_upper in {"LWPOLYLINE", "POLYLINE"}:
        points.extend(_polyline_vertices_in(entity, transform, to_in))
    elif dxftype_upper == "ARC":
        center = _point2d(getattr(getattr(entity, "dxf", entity), "center", None))
        radius = getattr(getattr(entity, "dxf", entity), "radius", None)
        start_angle = getattr(getattr(entity, "dxf", entity), "start_angle", None)
        end_angle = getattr(getattr(entity, "dxf", entity), "end_angle", None)
        if (
            center is not None
            and isinstance(radius, (int, float))
            and isinstance(start_angle, (int, float))
            and isinstance(end_angle, (int, float))
        ):
            try:
                center_world = _apply_transform_point(transform, center)
            except Exception:
                center_world = (None, None)
            else:
                if center_world[0] is not None and center_world[1] is not None:
                    points.append(
                        (float(center_world[0]) * to_in, float(center_world[1]) * to_in)
                    )
            for angle_deg in (float(start_angle), float(end_angle)):
                angle_rad = math.radians(angle_deg)
                x = center[0] + float(radius) * math.cos(angle_rad)
                y = center[1] + float(radius) * math.sin(angle_rad)
                world = _apply_transform_point(transform, (x, y))
                if world[0] is None or world[1] is None:
                    continue
                points.append((float(world[0]) * to_in, float(world[1]) * to_in))
    return points


def _filter_dense_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 6:
        return list(points)
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    median_x = statistics.median(xs)
    median_y = statistics.median(ys)
    mad_x = statistics.median([abs(x - median_x) for x in xs]) or 0.0
    mad_y = statistics.median([abs(y - median_y) for y in ys]) or 0.0
    scale = 6.0
    filtered = [
        pt
        for pt in points
        if (mad_x == 0.0 or abs(pt[0] - median_x) <= scale * mad_x)
        and (mad_y == 0.0 or abs(pt[1] - median_y) <= scale * mad_y)
    ]
    if len(filtered) >= max(6, len(points) // 3):
        return filtered
    return list(points)


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _bbox_from_points(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    filtered = _filter_dense_points(points)
    hull = _convex_hull(filtered)
    if not hull:
        hull = filtered
    if not hull:
        return None
    xs = [pt[0] for pt in hull]
    ys = [pt[1] for pt in hull]
    if not xs or not ys:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def _coerce_positive(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val) or val <= 0.0:
        return None
    return val


def _extract_dims_hint(geo_hint: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(geo_hint, Mapping):
        return None

    def _from_mapping(candidate: Mapping[str, Any], keys: tuple[str, str]) -> tuple[float, float] | None:
        width = _coerce_positive(candidate.get(keys[0]))
        height = _coerce_positive(candidate.get(keys[1]))
        if width and height:
            return (float(width), float(height))
        return None

    outline = geo_hint.get("outline_bbox")
    if isinstance(outline, Mapping):
        dims = _from_mapping(outline, ("plate_wid_in", "plate_len_in"))
        if dims:
            return dims

    for key in ("bbox_in", "required_blank_in"):
        payload = geo_hint.get(key)
        if isinstance(payload, Mapping):
            dims = _from_mapping(payload, ("w", "h"))
            if dims:
                return dims

    plate_dims = (
        _coerce_positive(geo_hint.get("plate_wid_in")),
        _coerce_positive(geo_hint.get("plate_len_in")),
    )
    if plate_dims[0] and plate_dims[1]:
        return (float(plate_dims[0]), float(plate_dims[1]))

    return None


def _bbox_from_dims(
    dims: tuple[float, float] | None,
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if not dims or not points:
        return None
    width, height = dims
    if width <= 0.0 or height <= 0.0:
        return None
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    if not xs or not ys:
        return None
    center_x = statistics.median(xs)
    center_y = statistics.median(ys)
    half_w = width / 2.0
    half_h = height / 2.0
    return (
        center_x - half_w,
        center_x + half_w,
        center_y - half_h,
        center_y + half_h,
    )


def geom_hole_census(doc: Any) -> dict[str, Any]:
    blocks_included = 0
    blocks_skipped = 0
    groups_counter: defaultdict[float, int] = defaultdict(int)

    try:
        msp = doc.modelspace()
    except Exception:
        layouts = getattr(doc, "layouts", None)
        layout_get = getattr(layouts, "get", None) if layouts is not None else None
        if callable(layout_get):
            try:
                msp = layout_get("Model")
            except Exception:
                msp = None
        else:
            msp = None
        if msp is None:
            print(
                "[GEOM] counted circles: total=0 from model=0 paperspace=0 "
                "blocks_included=0 blocks_skipped=0"
            )
            return {"groups": [], "total": 0}

    global _LAST_GEO_OUTLINE_HINT
    geo_hint = _LAST_GEO_OUTLINE_HINT if isinstance(_LAST_GEO_OUTLINE_HINT, Mapping) else None
    _LAST_GEO_OUTLINE_HINT = None

    units = detect_units_scale(doc)
    to_in = float(units.get("to_in") or 1.0)
    exclude_patterns = [
        re.compile(pattern, re.IGNORECASE) for pattern in DEFAULT_TEXT_LAYER_EXCLUDE_REGEX
    ]

    poly_bbox_in = _find_polyline_bbox_in(msp, to_in, exclude_patterns)
    dims_hint = _extract_dims_hint(geo_hint)

    seen_circle_keys: set[tuple[float, float, float]] = set()
    circle_records: list[dict[str, float]] = []
    total_candidates = 0
    layer_filter_dropped = 0
    radius_guard_dropped = 0

    def _allow_block(name: str | None) -> bool:
        nonlocal blocks_included, blocks_skipped
        if not name:
            return True
        if _GEOM_BLOCK_EXCLUDE_RE.match(name):
            blocks_skipped += 1
            return False
        blocks_included += 1
        return True

    for flattened in flatten_entities(msp, depth=0, include_block=_allow_block):
        entity = flattened.entity
        try:
            dxftype = entity.dxftype()
        except Exception:
            continue
        if str(dxftype or "").upper() != "CIRCLE":
            continue
        dxf_obj = getattr(entity, "dxf", None)
        layer_upper = (
            getattr(flattened, "effective_layer_upper", "")
            or getattr(flattened, "layer_upper", "")
        )
        layer_name = (
            getattr(flattened, "effective_layer", None)
            or getattr(flattened, "layer", None)
            or layer_upper
            or ""
        )
        if _layer_name_is_excluded(layer_name) or _layer_name_is_excluded(layer_upper):
            layer_filter_dropped += 1
            continue
        if layer_upper:
            if any(pattern.search(layer_upper) for pattern in exclude_patterns):
                layer_filter_dropped += 1
                continue
        radius_val = getattr(dxf_obj, "radius", None)
        if radius_val is None:
            radius_val = getattr(entity, "radius", None)
        if not isinstance(radius_val, (int, float)):
            continue
        center_obj = getattr(dxf_obj, "center", None)
        if center_obj is None:
            center_obj = getattr(entity, "center", None)
        center_coords = _point3d(center_obj)
        if center_coords is None:
            continue
        cx_raw, cy_raw, cz_raw = center_coords
        if not math.isfinite(cz_raw) or abs(cz_raw) > _GEO_CIRCLE_Z_ABS_MAX:
            continue
        tx, ty = _apply_transform_point(flattened.transform, (cx_raw, cy_raw))
        if tx is None or ty is None:
            continue
        if not (math.isfinite(tx) and math.isfinite(ty)):
            continue
        normal_candidate = getattr(dxf_obj, "extrusion", None)
        if normal_candidate is None:
            normal_candidate = getattr(dxf_obj, "normal", None)
        if normal_candidate is None:
            normal_candidate = getattr(entity, "extrusion", None)
        if normal_candidate is None:
            normal_candidate = getattr(entity, "normal", None)
        if not _is_positive_z_normal(normal_candidate):
            continue
        scaled_radius = float(radius_val) * _transform_scale_hint(flattened.transform)
        diameter_in = 2.0 * scaled_radius * to_in
        if not math.isfinite(diameter_in) or diameter_in <= 0:
            continue
        if diameter_in < _GEO_DIA_MIN_IN:
            continue
        if _GEO_DIA_MAX_IN and diameter_in > _GEO_DIA_MAX_IN:
            continue
        radius_in = diameter_in / 2.0
        if radius_in < _GEO_DRILL_RADIUS_MIN_IN:
            radius_guard_dropped += 1
            continue
        if _GEO_DRILL_RADIUS_MAX_IN and radius_in > _GEO_DRILL_RADIUS_MAX_IN:
            radius_guard_dropped += 1
            continue
        total_candidates += 1
        tx_in = float(tx) * to_in
        ty_in = float(ty) * to_in
        dedup_key = (
            round(float(tx_in), _GEO_CIRCLE_DEDUP_DIGITS),
            round(float(ty_in), _GEO_CIRCLE_DEDUP_DIGITS),
            round(float(diameter_in), _GEO_CIRCLE_DEDUP_DIGITS),
        )
        if dedup_key in seen_circle_keys:
            continue
        seen_circle_keys.add(dedup_key)
        circle_records.append({"x": tx_in, "y": ty_in, "dia_in": float(diameter_in)})

    raw_unique_count = len(circle_records)

    def _cluster_bbox_from_circle_records(
        records: Sequence[Mapping[str, float]]
    ) -> tuple[float, float, float, float] | None:
        if not records:
            return None
        points = [(float(rec["x"]), float(rec["y"])) for rec in records]
        if len(points) <= 4:
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            return (min(xs), max(xs), min(ys), max(ys))
        diameters = [float(rec.get("dia_in", 0.0)) for rec in records if rec.get("dia_in")]
        try:
            median_dia = statistics.median(diameters) if diameters else 0.0
        except Exception:
            median_dia = 0.0
        cell_size = float(median_dia) * 4.0 if median_dia and math.isfinite(median_dia) else 0.0
        if not cell_size or cell_size <= 0.0:
            cell_size = 6.0
        cell_size = max(6.0, cell_size)
        grid: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (px, py) in enumerate(points):
            cell_x = int(math.floor(px / cell_size))
            cell_y = int(math.floor(py / cell_size))
            grid[(cell_x, cell_y)].append(idx)
        best_indices: set[int] = set()
        best_count = 0
        for (cell_x, cell_y), idxs in grid.items():
            candidate: set[int] = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = grid.get((cell_x + dx, cell_y + dy))
                    if neighbor:
                        candidate.update(neighbor)
            if len(candidate) > best_count:
                best_indices = candidate
                best_count = len(candidate)
        if not best_indices:
            best_indices = set(range(len(points)))
        chosen_points = [points[idx] for idx in sorted(best_indices)]
        xs = [pt[0] for pt in chosen_points]
        ys = [pt[1] for pt in chosen_points]
        return (min(xs), max(xs), min(ys), max(ys))

    part_bbox_in: tuple[float, float, float, float] | None = None
    if poly_bbox_in is not None:
        part_bbox_in = tuple(poly_bbox_in)
    else:
        cluster_bbox = _cluster_bbox_from_circle_records(circle_records)
        if cluster_bbox is not None:
            part_bbox_in = cluster_bbox
    if part_bbox_in is None and dims_hint:
        dims_bbox = _bbox_from_dims(
            tuple(dims_hint),
            [(float(rec["x"]), float(rec["y"])) for rec in circle_records],
        )
        if dims_bbox is not None:
            part_bbox_in = dims_bbox

    kept_records = list(circle_records)
    dropped_outside = 0
    applied_bbox: tuple[float, float, float, float] | None = None
    if part_bbox_in is not None:
        xmin, xmax, ymin, ymax = part_bbox_in
        margin = _GEO_BBOX_MARGIN_IN
        if margin and math.isfinite(margin) and margin > 0.0:
            xmin -= margin
            xmax += margin
            ymin -= margin
            ymax += margin
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        filtered: list[dict[str, float]] = []
        for rec in circle_records:
            px = float(rec["x"])
            py = float(rec["y"])
            radius = max(float(rec.get("dia_in", 0.0)) / 2.0, 0.0)
            circle_xmin = px - radius
            circle_xmax = px + radius
            circle_ymin = py - radius
            circle_ymax = py + radius
            if (
                circle_xmax < xmin
                or circle_xmin > xmax
                or circle_ymax < ymin
                or circle_ymin > ymax
            ):
                dropped_outside += 1
                continue
            filtered.append(rec)
        kept_records = filtered
        applied_bbox = (xmin, xmax, ymin, ymax)
        if dropped_outside > 0:
            print(
                "[GEOM] bbox=[{xmin:.1f}..{xmax:.1f}, {ymin:.1f}..{ymax:.1f}] "
                "kept={kept} dropped_outside={dropped}".format(
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    kept=len(kept_records),
                    dropped=dropped_outside,
                )
            )

    def _collapse_concentric(
        records: Sequence[Mapping[str, float]]
    ) -> tuple[list[dict[str, float]], int]:
        grouped: defaultdict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
        for rec in records:
            cx = round(float(rec.get("x", 0.0)), _GEO_CIRCLE_CENTER_GROUP_DIGITS)
            cy = round(float(rec.get("y", 0.0)), _GEO_CIRCLE_CENTER_GROUP_DIGITS)
            grouped[(cx, cy)].append(dict(rec))
        kept: list[dict[str, float]] = []
        dropped_count = 0
        for items in grouped.values():
            if not items:
                continue
            items_sorted = sorted(items, key=lambda item: float(item.get("dia_in", 0.0)))
            kept.append(items_sorted[0])
            dropped_count += max(len(items_sorted) - 1, 0)
        return kept, dropped_count

    collapsed_records, concentric_dropped = _collapse_concentric(kept_records)
    kept_records = collapsed_records

    final_unique_count = len(kept_records)
    if total_candidates or final_unique_count:
        print(
            "[GEOM] unique circles after dedup: {} (was {})".format(
                final_unique_count, total_candidates
            )
        )
    if concentric_dropped > 0 and raw_unique_count:
        print(
            "[GEOM] concentric collapse dropped={} from raw_unique={}".format(
                concentric_dropped, raw_unique_count
            )
        )
    print(f"[GEOM] layer-filter dropped={layer_filter_dropped}")
    if radius_guard_dropped > 0:
        print(f"[GEOM] radius-guard dropped={radius_guard_dropped}")

    groups_counter = defaultdict(int)
    residual_holes: list[dict[str, float]] = []
    for rec in kept_records:
        hole_record = {
            "x": float(rec.get("x", 0.0)),
            "y": float(rec.get("y", 0.0)),
            "dia_in": float(rec.get("dia_in", 0.0)),
        }
        residual_holes.append(hole_record)
        dia_key = round(float(hole_record.get("dia_in", 0.0)), 4)
        if dia_key > 0:
            groups_counter[dia_key] += 1

    groups = [
        {"dia_in": float(diameter), "count": count}
        for diameter, count in sorted(groups_counter.items())
        if count > 0
    ]
    circle_total = sum(groups_counter.values())
    total = circle_total
    model_circles = circle_total
    print(
        "[GEOM] counted circles: total={} from model={} paperspace=0 "
        "blocks_included={} blocks_skipped={}".format(
            circle_total, model_circles, blocks_included, blocks_skipped
        )
    )
    residual_centers_payload = [
        {"x": hole["x"], "y": hole["y"]}
        for hole in residual_holes
    ]
    payload: dict[str, Any] = {
        "groups": groups,
        "total": int(total),
        "center_count": len(residual_holes),
        "residual_candidates": len(residual_holes),
        "residual_holes": residual_holes,
        "residual_centers": residual_centers_payload,
        "non_drill_centers": [],
        "dropped_concentric": int(concentric_dropped),
        "dropped_outside_bbox": int(dropped_outside),
        "dropped_radius_guard": int(radius_guard_dropped),
        "raw_unique_count": int(raw_unique_count),
        "total_candidates": int(total_candidates),
    }
    if applied_bbox is not None:
        payload["bbox_in"] = applied_bbox
        payload["bbox_margin_in"] = float(_GEO_BBOX_MARGIN_IN)
    return payload


def promote_table_to_geo(
    geo: dict[str, Any],
    table_info: Mapping[str, Any],
    source_tag: str,
    *,
    log_publish: bool = True,
    geom_holes: Mapping[str, Any] | None = None,
    state: ExtractionState | None = None,
) -> None:
    helper = _resolve_app_callable("_persist_rows_and_totals")
    if callable(helper):
        try:
            helper(geo, table_info, src=source_tag)
            return
        except Exception:
            pass
    if not isinstance(table_info, Mapping):
        return
    rows_candidate = table_info.get("rows") or []
    if isinstance(rows_candidate, list):
        rows = rows_candidate
    elif isinstance(rows_candidate, Iterable):
        rows = list(rows_candidate)
    else:
        rows = []
    if not rows:
        return
    ops_summary = geo.setdefault("ops_summary", {})
    ops_summary["rows"] = list(rows)
    ops_summary["source"] = source_tag
    qty_sum = _sum_qty(rows)
    authoritative_table = _table_source_is_authoritative(source_tag, len(rows))
    manifest = ops_manifest(
        rows,
        geom_holes=geom_holes,
        authoritative_table=authoritative_table,
    )
    if manifest:
        ops_summary["manifest"] = manifest
        totals_map = manifest.get("total")
        if isinstance(totals_map, Mapping):
            ops_summary["totals"] = dict(totals_map)
    if source_tag == "text_table" and qty_sum > 0:
        geo["hole_count"] = qty_sum
        provenance = geo.setdefault("provenance", {})
        if isinstance(provenance, Mapping) and not isinstance(provenance, dict):
            provenance = dict(provenance)
            geo["provenance"] = provenance
        if isinstance(provenance, dict):
            provenance["holes"] = "HOLE TABLE"
        if log_publish:
            if state is None:
                print(
                    f"[PATH] publish=text_table rows={len(rows)} qty_sum={qty_sum}"
                )
            elif state.mark_published():
                print(
                    f"[PATH] publish=text_table rows={len(rows)} qty_sum={qty_sum}"
                )
    hole_count = table_info.get("hole_count")
    if qty_sum > 0:
        hole_count = qty_sum
    try:
        geo["hole_count"] = int(hole_count)
    except Exception:
        pass
    provenance = geo.setdefault("provenance", {})
    provenance["holes"] = "HOLE TABLE"


def extract_hole_table(
    doc_or_path: Any,
    *,
    opts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return standardized hole-table data for downstream pricing/app usage."""

    options = dict(opts or {})

    doc = doc_or_path
    if isinstance(doc_or_path, (str, os.PathLike, Path)):
        use_oda = bool(options.pop("use_oda", True))
        oda_version = options.pop("oda_version", None)
        path_obj = Path(doc_or_path)
        doc = _load_doc_for_path(path_obj, use_oda=use_oda, out_ver=oda_version)

    if doc is None:
        return {
            "chart_rows": [],
            "hole_count_table": 0,
            "provenance": "HOLE TABLE",
            "layouts_seen": [],
            "layer_policy": {"include": "-", "exclude": "-"},
        }

    allowed_keys = {
        "layer_allowlist",
        "roi_hint",
        "block_name_allowlist",
        "block_name_regex",
        "layer_include_regex",
        "layer_exclude_regex",
        "layout_filters",
        "debug_layouts",
        "debug_scan",
    }
    read_kwargs = {key: value for key, value in options.items() if key in allowed_keys}

    try:
        table_info = read_text_table(doc, **read_kwargs) or {}
    except NoTextRowsError:
        table_info = {}

    rows = _normalize_table_rows(table_info.get("rows"))
    chart_rows: list[dict[str, Any]] = []
    for row in rows:
        qty_val = 0
        try:
            qty_val = int(float(row.get("qty") or 0))
        except Exception:
            qty_val = 0
        desc_text = " ".join(str(row.get("desc") or "").split())
        if qty_val <= 0 and not desc_text:
            continue
        row_payload: dict[str, Any] = {}
        if qty_val > 0:
            row_payload["qty"] = qty_val
        if desc_text:
            row_payload["desc"] = desc_text
        ref_text = str(row.get("ref") or "").strip()
        if ref_text:
            row_payload["ref"] = ref_text
        side_text = str(row.get("side") or "").strip()
        if side_text:
            row_payload["side"] = side_text
        if not row_payload:
            continue
        chart_rows.append(row_payload)

    hole_count_table = sum(int(row.get("qty", 0)) for row in chart_rows)

    debug_snapshot = get_last_text_table_debug() or {}
    layout_candidates: list[str] = []
    if isinstance(debug_snapshot, Mapping):
        for key in ("scanned_layouts", "layouts"):
            raw_value = debug_snapshot.get(key)
            if isinstance(raw_value, list):
                for item in raw_value:
                    text = str(item or "").strip()
                    if text:
                        layout_candidates.append(text)
    layouts_seen: list[str] = []
    seen_layouts: set[str] = set()
    for name in layout_candidates:
        if name in seen_layouts:
            continue
        seen_layouts.add(name)
        layouts_seen.append(name)

    include_patterns: list[str] = []
    exclude_patterns: list[str] = []
    if isinstance(debug_snapshot, Mapping):
        raw_include = debug_snapshot.get("layer_regex_include")
        raw_exclude = debug_snapshot.get("layer_regex_exclude")
        if isinstance(raw_include, list):
            include_patterns = [str(item) for item in raw_include if str(item or "").strip()]
        if isinstance(raw_exclude, list):
            exclude_patterns = [str(item) for item in raw_exclude if str(item or "").strip()]
    if not exclude_patterns:
        exclude_patterns = [pattern for pattern in DEFAULT_TEXT_LAYER_EXCLUDE_REGEX if pattern]

    def _pattern_display(patterns: list[str]) -> str:
        if not patterns:
            return "-"
        return ", ".join(patterns)

    layer_policy = {
        "include": _pattern_display(include_patterns),
        "exclude": _pattern_display(exclude_patterns),
    }

    provenance = "HOLE TABLE"

    return {
        "chart_rows": chart_rows,
        "hole_count_table": hole_count_table,
        "provenance": provenance,
        "layouts_seen": layouts_seen,
        "layer_policy": layer_policy,
    }


# new façade — lightweight adapter
def extract_for_app(
    doc_or_path: Any,
    *,
    layouts: Mapping[str, Any] | Iterable[str] | str | None = None,
    text_layer_exclude: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """
    Returns:
      {
        "rows": [...],                # table rows with qty/desc
        "qty_sum": int,
        "source": "acad_table|text_table|text_fallback|geom",
        "provenance": str|None,       # e.g., "HOLE TABLE (anchor)"
        "geom": { "groups": [...], "total": int },   # dedup'd hole circles
        "manifest": { "table": {...}, "geom": {...}, "total": {...}, "details": {...}, "text": {...} }
      }
    """

    doc = doc_or_path
    if isinstance(doc_or_path, (str, os.PathLike, Path)):
        path_obj = Path(doc_or_path)
        try:
            doc = _load_doc_for_path(path_obj, use_oda=True)
        except Exception:
            doc = None

    if doc is None:
        geom_payload = _normalize_geom_holes_payload(None)
        manifest = ops_manifest([], geom_holes=geom_payload)
        return {
            "rows": [],
            "qty_sum": 0,
            "source": "geom",
            "provenance": None,
            "geom": geom_payload,
            "manifest": manifest,
        }

    selected_info: Mapping[str, Any] | dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    selected_source = "geom"

    try:
        acad_info = read_acad_table(doc) or {}
    except Exception:
        acad_info = {}
    acad_rows = _normalize_table_rows(acad_info.get("rows"))
    if acad_rows:
        selected_info = dict(acad_info)
        selected_rows = acad_rows
        selected_source = "acad_table"
    else:
        text_kwargs: dict[str, Any] = {}
        if layouts is not None:
            text_kwargs["layout_filters"] = layouts
        if text_layer_exclude is not None:
            text_kwargs["layer_exclude_regex"] = text_layer_exclude
        try:
            text_info = read_text_table(doc, **text_kwargs) or {}
        except (NoTextRowsError, RuntimeError):
            text_info = {}
        except Exception:
            text_info = {}
        text_rows = _normalize_table_rows(text_info.get("rows"))
        if text_rows:
            selected_info = dict(text_info)
            selected_rows = text_rows
            selected_source = "text_table"
        else:
            try:
                fallback_lines = _collect_table_text_lines(
                    doc, layout_filters=layouts
                )
            except Exception:
                fallback_lines = []
            fallback_info = _fallback_text_table(fallback_lines) or {}
            fallback_rows = _normalize_table_rows(fallback_info.get("rows"))
            if fallback_rows:
                selected_info = dict(fallback_info)
                selected_rows = fallback_rows
                selected_source = "text_fallback"

    provenance = None
    if isinstance(selected_info, Mapping):
        provenance_candidate = selected_info.get("provenance_holes")
        if provenance_candidate not in (None, ""):
            provenance_text = str(provenance_candidate).strip()
            provenance = provenance_text or None

    qty_sum = _sum_qty(selected_rows)

    try:
        geom_source = geom_hole_census(doc)
    except Exception:
        geom_source = None
    geom_payload = _normalize_geom_holes_payload(geom_source)

    authoritative_table = _table_source_is_authoritative(
        selected_source,
        len(selected_rows),
    )
    manifest = ops_manifest(
        selected_rows,
        geom_holes=geom_payload,
        authoritative_table=authoritative_table,
    )

    return {
        "rows": selected_rows,
        "qty_sum": qty_sum,
        "source": selected_source,
        "provenance": provenance,
        "geom": geom_payload,
        "manifest": manifest,
    }


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
    global _LAST_DXF_FALLBACK_INFO
    _LAST_DXF_FALLBACK_INFO = None
    ezdxf_mod = geometry.require_ezdxf()
    readfile = getattr(ezdxf_mod, "readfile", None)
    if not callable(readfile):
        raise AttributeError("ezdxf module does not expose a callable readfile")
    lower_suffix = path.suffix.lower()
    oda_version = _normalize_oda_version(out_ver)
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
        target_version = oda_version or "ACAD2018"
        dxf_path: str | None = None
        try:
            if oda_version:
                dxf_path = convert_dwg_to_dxf(str(path), out_ver=oda_version)
            else:
                dxf_path = convert_dwg_to_dxf(str(path), quiet=True)
        except Exception as exc:
            out_display = dxf_path or "-"
            error_text = str(exc)
            print(
                f"[DXF-FALLBACK] try={target_version} ok=False out={out_display} err={error_text}"
            )
            _LAST_DXF_FALLBACK_INFO = {
                "version": target_version,
                "path": str(path),
                "ok": False,
                "error": error_text,
            }
            raise
        ok = bool(dxf_path) and os.path.exists(dxf_path)
        out_display = dxf_path or "-"
        if oda_version or not ok:
            print(f"[DXF-FALLBACK] try={target_version} ok={ok} out={out_display}")
            _LAST_DXF_FALLBACK_INFO = {
                "version": target_version,
                "path": str(out_display),
                "ok": ok,
            }
        else:
            _LAST_DXF_FALLBACK_INFO = None
        if not ok:
            raise AssertionError(
                f"ODA fallback {target_version} did not produce a DXF at {out_display}"
            )
        return readfile(out_display)
    return readfile(str(path))


def _ensure_ops_summary_map(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {}


def _am_bor_included_from_candidates(*candidates: Mapping[str, Any] | None) -> bool:
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        flag = candidate.get("am_bor_included")
        if isinstance(flag, bool):
            if flag:
                return True
        elif flag:
            return True
    return False


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
    force_text: bool = False,
    pipeline: str = "auto",
    allow_geom: bool = False,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
    layer_include_regex: Iterable[str] | str | None = None,
    layer_exclude_regex: Iterable[str] | str | None = DEFAULT_TEXT_LAYER_EXCLUDE_REGEX,
    layout_filters: Mapping[str, Any] | Iterable[str] | str | None = None,
    debug_layouts: bool = False,
    debug_scan: bool = False,
    state: ExtractionState | None = None,
) -> dict[str, Any]:
    """Process a loaded DXF/DWG document into GEO payload details.

    Args:
        pipeline: Extraction pipeline to run. ``"auto"`` tries ACAD first and
            falls back to TEXT, ``"acad"`` and ``"text"`` force a specific path,
            and ``"geom"`` publishes geometry-derived rows directly.
        allow_geom: When ``True``, geometry rows may be emitted even when the
            pipeline is set to ``"auto"``.
    """

    if state is None:
        state = ExtractionState()

    del feature_flags  # placeholder for future feature toggles
    pipeline_normalized = str(pipeline or "auto").strip().lower()
    if pipeline_normalized not in {"auto", "acad", "text", "geom"}:
        pipeline_normalized = "auto"
    allow_geom_rows = bool(allow_geom or pipeline_normalized == "geom")
    geo = extract_geometry(doc)
    if not isinstance(geo, dict):
        geo = {}

    global _LAST_GEO_OUTLINE_HINT
    _LAST_GEO_OUTLINE_HINT = geo if isinstance(geo, Mapping) else None
    geom_census = geom_hole_census(doc)
    if isinstance(geo, dict):
        geo["geom_holes"] = geom_census

    use_tables = bool(
        prefer_table and pipeline_normalized in {"auto", "acad", "text"}
    )
    run_acad = pipeline_normalized in {"auto", "acad"}
    run_text = pipeline_normalized in {"auto", "text"}

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

    acad_info: Mapping[str, Any] | dict[str, Any]
    if run_acad:
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
    else:
        acad_info = {}
    acad_roi_hint: Mapping[str, Any] | None = None
    if isinstance(acad_info, Mapping):
        roi_candidate = acad_info.get("roi_hint")
        acad_roi_hint = roi_candidate if isinstance(roi_candidate, Mapping) else None
    text_layer_allowlist = layer_allowlist
    if text_layer_allowlist is _DEFAULT_LAYER_ALLOWLIST:
        text_layer_allowlist = None
    elif isinstance(text_layer_allowlist, Iterable) and not isinstance(
        text_layer_allowlist, (str, bytes, bytearray)
    ):
        try:
            if set(text_layer_allowlist) == set(_DEFAULT_LAYER_ALLOWLIST):
                text_layer_allowlist = None
        except TypeError:
            pass
    if run_text:
        try:
            text_info = read_text_table(
                doc,
                layer_allowlist=text_layer_allowlist,
                roi_hint=acad_roi_hint,
                block_name_allowlist=block_name_allowlist,
                block_name_regex=block_name_regex,
                layer_include_regex=layer_include_regex,
                layer_exclude_regex=layer_exclude_regex,
                layout_filters=layout_filters,
                debug_layouts=debug_layouts,
                debug_scan=debug_scan,
            ) or {}
        except TypeError as exc:
            message = str(exc)
            if any(
                key in message
                for key in ("layer_allowlist", "roi_hint", "layout_filters", "debug_scan")
            ):
                try:
                    text_info = read_text_table(doc) or {}
                except RuntimeError:
                    raise
                except Exception:
                    text_info = {}
            else:
                raise
        except RuntimeError:
            raise
        except Exception:
            text_info = {}
    else:
        text_info = {}

    if isinstance(text_info, Mapping) and text_info.get("anchor_authoritative"):
        state.anchor_authoritative = True
        state.published = True

    acad_rows_list: list[dict[str, Any]] = []
    if isinstance(acad_info, Mapping):
        acad_info = dict(acad_info)
        acad_rows_list = _normalize_table_rows(acad_info.get("rows"))
        acad_info["rows"] = acad_rows_list
    else:
        acad_info = {}
    acad_rows = len(acad_rows_list)
    if not run_acad:
        print("[PATH] acad=skip (pipeline=text/geom)")
    elif acad_rows == 0:
        print("[PATH] acad=0 (no tables found)")

    text_rows_list: list[dict[str, Any]] = []
    if isinstance(text_info, Mapping):
        text_info = dict(text_info)
        text_rows_list = _normalize_table_rows(text_info.get("rows"))
        text_info["rows"] = text_rows_list
    else:
        text_info = {}
    text_rows = len(text_rows_list)
    debug_snapshot = get_last_text_table_debug() or {}
    rows_txt_debug = 0
    if isinstance(debug_snapshot, Mapping):
        try:
            rows_txt_debug = int(float(debug_snapshot.get("rows_txt_count") or 0))
        except Exception:
            rows_txt_debug = 0
    rows_txt_lines: list[str] = []
    if isinstance(debug_snapshot, Mapping):
        raw_rows_txt = debug_snapshot.get("rows_txt_lines")
        candidates: Iterable[Any]
        if isinstance(raw_rows_txt, list):
            candidates = raw_rows_txt
        elif isinstance(raw_rows_txt, Iterable) and not isinstance(
            raw_rows_txt, (str, bytes, bytearray)
        ):
            candidates = list(raw_rows_txt)
        else:
            candidates = []
        for item in candidates:
            try:
                text_candidate = str(item)
            except Exception:
                text_candidate = ""
            normalized_candidate = " ".join(text_candidate.split())
            if normalized_candidate:
                rows_txt_lines.append(normalized_candidate)
    if run_text:
        print(f"[PATH] text=run (rows_txt={rows_txt_debug})")
    else:
        print("[PATH] text=skip (pipeline=acad/geom)")
    print(f"[EXTRACT] acad_rows={acad_rows} text_rows={text_rows}")

    publish_info: dict[str, Any] | None = None
    publish_source_tag: str | None = None
    fallback_info: dict[str, Any] | None = None
    fallback_rows_list: list[dict[str, Any]] = []
    fallback_qty_sum = 0
    if rows_txt_lines:
        fallback_candidate = _publish_fallback_from_rows_txt(rows_txt_lines)
        if isinstance(fallback_candidate, Mapping) and fallback_candidate.get("rows"):
            fallback_info = dict(fallback_candidate)
            fallback_rows_list = _normalize_table_rows(fallback_info.get("rows"))
            fallback_info["rows"] = fallback_rows_list
            fallback_qty_sum = _sum_qty(fallback_rows_list)
    no_text_rows_available = bool(
        use_tables
        and run_acad
        and run_text
        and acad_rows == 0
        and text_rows == 0
        and not fallback_rows_list
        and rows_txt_debug == 0
        and not allow_geom_rows
    )
    if no_text_rows_available:
        raise NoTextRowsError()
    force_text_mode = bool(
        run_text and force_text and fallback_info and fallback_rows_list
    )
    fallback_selected = False
    auto_text_fallback = bool(
        run_text
        and fallback_info
        and fallback_rows_list
        and (not run_acad or acad_rows == 0)
        and text_rows == 0
        and rows_txt_lines
    )
    if run_text and auto_text_fallback:
        publish_info = fallback_info
        publish_source_tag = "text_table"
        fallback_selected = True
    elif run_text and force_text_mode:
        publish_info = fallback_info
        publish_source_tag = "text_table"
        fallback_selected = True
    elif run_acad and acad_rows_list:
        publish_info = acad_info
        publish_source_tag = "acad_table"
    elif run_text and text_rows_list:
        publish_info = text_info
        publish_source_tag = "text_table"
    elif run_text and fallback_info and fallback_rows_list:
        publish_info = fallback_info
        publish_source_tag = "text_table"
        fallback_selected = True

    score_a = _score_table(acad_info)
    score_b = _score_table(text_info)
    score_c = _score_table(fallback_info)
    best_table = publish_info or choose_better_table(acad_info, text_info)
    if fallback_info and fallback_rows_list:
        best_score = _score_table(best_table) if isinstance(best_table, Mapping) else (0, 0, 0)
        if score_c > best_score:
            best_table = fallback_info
    if publish_info is None and isinstance(best_table, Mapping) and best_table.get("rows"):
        publish_info = dict(best_table)
        publish_info["rows"] = _normalize_table_rows(publish_info.get("rows"))
        if run_acad and best_table is acad_info:
            publish_source_tag = "acad_table"
        else:
            publish_source_tag = "text_table"
    publish_rows: list[dict[str, Any]] = []
    if isinstance(publish_info, Mapping):
        publish_rows = list(publish_info.get("rows") or [])
    skip_acad = bool(
        publish_rows
        and publish_source_tag
        and str(publish_source_tag).lower() == "text_table"
    )
    if pipeline_normalized in {"text", "geom"}:
        skip_acad = True
    if fallback_selected and fallback_rows_list:
        print(
            f"[TEXT-FALLBACK] promoted rows={len(fallback_rows_list)} "
            f"qty_sum={fallback_qty_sum} source=text"
        )

    table_used = False
    source_tag = publish_source_tag
    existing_score = _score_table(current_table_info)
    publish_score = _score_table(publish_info) if isinstance(publish_info, Mapping) else (0, 0, 0)
    if use_tables and publish_info and publish_rows:
        can_promote = False
        if not existing_is_table:
            can_promote = True
        elif publish_score > existing_score:
            can_promote = True
        elif (
            publish_info is text_info
            and acad_rows == 0
            and text_rows > 0
        ):
            can_promote = True
        if can_promote:
            promote_table_to_geo(
                geo,
                publish_info,
                publish_source_tag or "text_table",
                log_publish=False,
                geom_holes=geom_census,
                state=state,
            )
            table_used = True
            if publish_source_tag and publish_source_tag == "text_table":
                _print_promoted_rows_once(publish_rows)

    if not isinstance(best_table, Mapping) or not best_table.get("rows"):
        best_table = publish_info or best_table

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

    qty_sum = 0
    if table_used:
        qty_sum = _sum_qty(rows)
    elif publish_rows:
        qty_sum = _sum_qty(publish_rows)
        ops_summary["rows"] = list(publish_rows)
        if publish_source_tag:
            ops_summary["source"] = publish_source_tag
        if publish_source_tag == "text_table":
            provenance = geo.setdefault("provenance", {})
            if isinstance(provenance, Mapping) and not isinstance(provenance, dict):
                provenance = dict(provenance)
                geo["provenance"] = provenance
            if isinstance(provenance, dict):
                provenance["holes"] = (
                    "HOLE TABLE (anchor)"
                    if getattr(state, "anchor_authoritative", False)
                    else "HOLE TABLE"
                )
            authoritative_table = _table_source_is_authoritative(
                publish_source_tag,
                len(publish_rows),
            )
            manifest_payload = ops_manifest(
                publish_rows,
                geom_holes=geom_census,
                authoritative_table=authoritative_table,
            )
            if manifest_payload:
                ops_summary["manifest"] = manifest_payload
                totals_map = manifest_payload.get("total")
                if isinstance(totals_map, Mapping):
                    ops_summary["totals"] = dict(totals_map)
            geo["hole_count"] = qty_sum
    else:
        geometry_rows_present = bool(rows)
        if geometry_rows_present and not allow_geom_rows:
            ops_summary.pop("rows", None)
            rows = []
            if pipeline_normalized == "auto":
                print(
                    "[PATH] geom suppressed (use --pipeline geom or --allow-geom)"
                )
            ops_summary["source"] = pipeline_normalized
        else:
            if not geometry_rows_present:
                ops_summary.pop("rows", None)
                rows = []
            if geometry_rows_present and allow_geom_rows:
                ops_summary["source"] = "geom"
                qty_sum = _sum_qty(rows)
            else:
                ops_summary["source"] = pipeline_normalized

    if not table_used and not publish_rows:
        hole_count = _best_geo_hole_count(geo)
        if hole_count:
            geo["hole_count"] = hole_count

    if publish_rows and not table_used:
        try:
            hole_total = publish_info.get("hole_count") if isinstance(publish_info, Mapping) else None
            if hole_total in (None, ""):
                hole_total = qty_sum
            geo["hole_count"] = int(float(hole_total))
        except Exception:
            geo["hole_count"] = qty_sum

    if (table_used or publish_rows) and source_tag:
        ops_summary["source"] = source_tag
    totals = ops_summary.get("totals")
    if isinstance(totals, Mapping):
        ops_summary["totals"] = dict(totals)

    rows_for_log: list[Mapping[str, Any]] | list[Any]
    if table_used:
        rows_for_log = ops_summary.get("rows") or []
        if not isinstance(rows_for_log, list) and isinstance(rows_for_log, Iterable):
            rows_for_log = list(rows_for_log)
        qty_sum = _sum_qty(rows_for_log)
    elif publish_rows:
        rows_for_log = list(publish_rows)
        qty_sum = _sum_qty(rows_for_log)
    else:
        rows_for_log = rows
        if not isinstance(rows_for_log, list):
            if isinstance(rows_for_log, Iterable):
                rows_for_log = list(rows_for_log)
            else:
                rows_for_log = []
        if not rows_for_log and text_rows:
            rows_for_log = list(text_rows_list)

    manifest_rows = ops_summary.get("rows")
    if not isinstance(manifest_rows, list):
        manifest_rows = rows_for_log
    if not isinstance(manifest_rows, list):
        if isinstance(manifest_rows, Iterable):
            manifest_rows = list(manifest_rows)
        else:
            manifest_rows = []
    authoritative_table = _table_source_is_authoritative(
        ops_summary.get("source"),
        len(manifest_rows),
    )
    manifest_payload = ops_manifest(
        manifest_rows,
        geom_holes=geom_census,
        authoritative_table=authoritative_table,
    )
    if manifest_payload:
        ops_summary["manifest"] = manifest_payload
        totals_map = manifest_payload.get("total")
        if isinstance(totals_map, Mapping):
            ops_summary["totals"] = dict(totals_map)

        def _format_ops_counts(
            counts: Mapping[str, Any] | None,
            order: Sequence[tuple[str, str]],
        ) -> str:
            if not isinstance(counts, Mapping):
                return "Drill 0"
            parts: list[str] = []
            for key, label in order:
                value = counts.get(key)
                try:
                    value_int = int(round(float(value)))
                except Exception:
                    continue
                parts.append(f"{label} {value_int}")
            if not parts:
                parts.append("Drill 0")
            return " | ".join(parts)

        table_counts = manifest_payload.get("table") if isinstance(manifest_payload, Mapping) else {}
        geom_counts = manifest_payload.get("geom") if isinstance(manifest_payload, Mapping) else {}
        total_counts = manifest_payload.get("total") if isinstance(manifest_payload, Mapping) else {}
        print(
            "[OPS] table: "
            + _format_ops_counts(
                table_counts,
                (
                    ("drill_only", "Drill"),
                    ("tap", "Tap"),
                    ("counterbore", "C'bore"),
                    ("counterdrill", "C'drill"),
                    ("jig_grind", "Jig"),
                ),
            )
        )
        print(
            "[OPS] geom : "
            + _format_ops_counts(
                geom_counts,
                (
                    ("drill_residual", "Drill"),
                ),
            )
        )
        print(
            "[OPS] total: "
            + _format_ops_counts(
                total_counts,
                (
                    ("drill", "Drill"),
                    ("tap", "Tap"),
                    ("counterbore", "C'bore"),
                    ("counterdrill", "C'drill"),
                    ("jig_grind", "Jig"),
                ),
            )
        )
        
        def _int_from(value: Any) -> int:
            try:
                return int(round(float(value or 0)))
            except Exception:
                return 0

        text_drill_total = _int_from(table_counts.get("drill_only")) if isinstance(table_counts, Mapping) else 0
        text_cbore_total = _int_from(table_counts.get("counterbore")) if isinstance(table_counts, Mapping) else 0
        text_cdrill_total = _int_from(table_counts.get("counterdrill")) if isinstance(table_counts, Mapping) else 0
        text_ops_total = text_drill_total + text_cbore_total + text_cdrill_total
        text_manifest = manifest_payload.get("text") if isinstance(manifest_payload, Mapping) else {}
        text_estimated_total_drills = 0
        if isinstance(text_manifest, Mapping):
            try:
                text_estimated_total_drills = int(
                    float(text_manifest.get("estimated_total_drills") or 0)
                )
            except Exception:
                text_estimated_total_drills = 0
        geom_total = 0
        if isinstance(geom_counts, Mapping):
            geom_total_candidate = geom_counts.get("total")
            try:
                geom_total = int(float(geom_total_candidate or 0))
            except Exception:
                geom_total = 0
            if geom_total <= 0:
                try:
                    geom_total = int(float(geom_counts.get("drill") or 0))
                except Exception:
                    geom_total = 0
        am_bor_in_text_flow = _am_bor_included_from_candidates(
            text_info,
            fallback_info,
            publish_info,
            best_table,
            current_table_info,
        )
        total_drill_count = _int_from(total_counts.get("drill")) if isinstance(total_counts, Mapping) else 0
        if total_drill_count > 100 or (total_drill_count != 0 and total_drill_count < 50):
            print("[GEOM] suspect overcount – check layer blacklist or bbox guard")
        if (
            am_bor_in_text_flow
            and geom_total > 0
            and text_estimated_total_drills > 0
            and float(geom_total) > 2.0 * float(text_estimated_total_drills)
        ):
            suspect_payload: dict[str, Any] = {
                "geom_total": geom_total,
                "text_estimated_total_drills": text_estimated_total_drills,
                "am_bor_included": True,
                "logged": True,
            }
            print(
                "[OPS-GUARD] suspect geometry: "
                f"geom.total={geom_total} text.estimated_total_drills={text_estimated_total_drills}"
            )
            if isinstance(manifest_payload, dict):
                flags_map = manifest_payload.setdefault("flags", {})
                if isinstance(flags_map, dict):
                    flags_map["suspect_geometry"] = suspect_payload
            if isinstance(ops_summary, dict):
                flags_map = ops_summary.setdefault("flags", {})
                if isinstance(flags_map, dict):
                    flags_map["suspect_geometry"] = suspect_payload
    source_display = ops_summary.get("source") if isinstance(ops_summary, Mapping) else None
    source_lower = str(source_display or "").lower()
    if source_lower == "text_table":
        skip_acad = True
        publish_path = "text_table"
    elif source_lower == "acad_table":
        publish_path = "acad_table"
    elif source_lower:
        publish_path = source_lower
    else:
        publish_path = "geom"
    should_log_publish = True if state is None else state.mark_published()
    if should_log_publish:
        print(
            f"[PATH] publish={publish_path} rows={len(rows_for_log)} qty_sum={qty_sum}"
        )
    print(
        f"[EXTRACT] published rows={len(rows_for_log)} qty_sum={qty_sum} "
        f"source={ops_summary.get('source')}"
    )
    provenance_holes = None
    if isinstance(publish_info, Mapping):
        provenance_holes = publish_info.get("provenance_holes")
    if not provenance_holes:
        provenance = geo.get("provenance")
        if isinstance(provenance, Mapping):
            provenance_holes = provenance.get("holes")
    print(f"[EXTRACT] provenance={provenance_holes}")

    debug_payload = get_last_text_table_debug() or {}
    hole_count_val: int | None | float = None
    if isinstance(publish_info, Mapping):
        hole_count_val = publish_info.get("hole_count")
    if hole_count_val in (None, ""):
        try:
            hole_count_val = geo.get("hole_count") if isinstance(geo, Mapping) else None
        except Exception:
            hole_count_val = None
    if hole_count_val in (None, ""):
        hole_count_val = _best_geo_hole_count(geo) if isinstance(geo, Mapping) else None
    try:
        if hole_count_val not in (None, ""):
            hole_count_val = int(float(hole_count_val))
    except Exception:
        pass

    payload_rows: list[Mapping[str, Any]] = []
    if isinstance(rows_for_log, list):
        payload_rows = rows_for_log
    elif isinstance(rows_for_log, Iterable):
        payload_rows = list(rows_for_log)

    families_map: dict[str, int] | None = None
    candidates_for_families: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    for candidate in (publish_info, best_table, text_info, acad_info, current_table_info):
        if not isinstance(candidate, Mapping):
            continue
        marker = id(candidate)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)
        candidates_for_families.append(candidate)
    for candidate in candidates_for_families:
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

    table_used = table_used or bool(publish_rows)

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
        "skip_acad": skip_acad,
        "state_published": state.published,
    }

    if geom_census:
        result_payload["geom_holes"] = geom_census

    if families_map is not None:
        result_payload["hole_diam_families_in"] = families_map

    return result_payload


def _read_geo_payload_from_path(
    path_obj: Path,
    *,
    prefer_table: bool = True,
    use_oda: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
    force_text: bool = False,
    pipeline: str = "auto",
    allow_geom: bool = False,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
    layer_include_regex: Iterable[str] | str | None = None,
    layer_exclude_regex: Iterable[str] | str | None = DEFAULT_TEXT_LAYER_EXCLUDE_REGEX,
    debug_layouts: bool = False,
) -> dict[str, Any]:
    try:
        doc = _load_doc_for_path(path_obj, use_oda=use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[EXTRACT] failed to load document: {exc}")
        return {"error": str(exc)}

    state = ExtractionState()

    payload = read_geo(
        doc,
        prefer_table=prefer_table,
        feature_flags=feature_flags,
        force_text=force_text,
        pipeline=pipeline,
        allow_geom=allow_geom,
        layer_allowlist=layer_allowlist,
        block_name_allowlist=block_name_allowlist,
        block_name_regex=block_name_regex,
        layer_include_regex=layer_include_regex,
        layer_exclude_regex=layer_exclude_regex,
        debug_layouts=debug_layouts,
        state=state,
    )

    if isinstance(payload, Mapping) and payload.get("skip_acad"):
        return payload

    scan_info = get_last_acad_table_scan() or {}
    tables_found = 0
    try:
        tables_found = int(scan_info.get("tables_found", 0))  # type: ignore[arg-type]
    except Exception:
        tables_found = 0
    log_last_dxf_fallback(tables_found)
    published_rows_obj = payload.get("rows") if isinstance(payload, Mapping) else None
    if isinstance(published_rows_obj, Iterable) and not isinstance(
        published_rows_obj, list
    ):
        published_rows_list = list(published_rows_obj)
    elif isinstance(published_rows_obj, list):
        published_rows_list = published_rows_obj
    else:
        published_rows_list = []
    if published_rows_list:
        return payload
    if tables_found == 0 and path_obj.suffix.lower() == ".dwg":
        fallback_versions = [
            "ACAD2000",
            "ACAD2004",
            "ACAD2007",
            "ACAD2013",
            "ACAD2018",
        ]
        for version in fallback_versions:
            oda_version = _normalize_oda_version(version) or version
            print(f"[ACAD-TABLE] trying DXF fallback version={oda_version}")
            try:
                fallback_doc = _load_doc_for_path(
                    path_obj, use_oda=use_oda, out_ver=oda_version
                )
            except Exception as exc:
                print(f"[ACAD-TABLE] DXF fallback {oda_version} failed: {exc}")
                continue
            mechanical_table = _extract_mechanical_table_from_blocks(fallback_doc)
            global _LAST_GEO_OUTLINE_HINT
            _LAST_GEO_OUTLINE_HINT = None
            fallback_geom_census = geom_hole_census(fallback_doc)
            payload = read_geo(
                fallback_doc,
                prefer_table=prefer_table,
                feature_flags=feature_flags,
                force_text=force_text,
                pipeline=pipeline,
                allow_geom=allow_geom,
                layer_allowlist=layer_allowlist,
                block_name_allowlist=block_name_allowlist,
                block_name_regex=block_name_regex,
                state=state,
            )
            if isinstance(mechanical_table, Mapping) and mechanical_table.get("rows"):
                existing_rows_obj = payload.get("rows")
                if isinstance(existing_rows_obj, list):
                    existing_rows = existing_rows_obj
                elif isinstance(existing_rows_obj, Iterable):
                    existing_rows = list(existing_rows_obj)
                else:
                    existing_rows = []
                if not existing_rows:
                    geo_obj = payload.get("geo")
                    if isinstance(geo_obj, dict):
                        promote_table_to_geo(
                            geo_obj,
                            mechanical_table,
                            "text",
                            geom_holes=fallback_geom_census,
                            log_publish=not state.published,
                            state=state,
                        )
                        payload["state_published"] = state.published
                        ops_summary_obj = geo_obj.get("ops_summary")
                        ops_summary = dict(ops_summary_obj) if isinstance(ops_summary_obj, Mapping) else {}
                        payload["ops_summary"] = ops_summary
                        rows_obj = ops_summary.get("rows")
                        if isinstance(rows_obj, list):
                            rows_list = rows_obj
                        elif isinstance(rows_obj, Iterable):
                            rows_list = list(rows_obj)
                        else:
                            rows_list = []
                        payload["rows"] = rows_list
                        qty_sum = _sum_qty(rows_list)
                        payload["qty_sum"] = qty_sum
                        hole_count_val = mechanical_table.get("hole_count")
                        if isinstance(hole_count_val, (int, float)) and hole_count_val > 0:
                            try:
                                hole_count_int = int(float(hole_count_val))
                            except Exception:
                                hole_count_int = qty_sum
                        else:
                            hole_count_int = qty_sum
                        payload["hole_count"] = hole_count_int
                        try:
                            geo_obj["hole_count"] = hole_count_int
                        except Exception:
                            pass
                        payload["table_used"] = True
                        payload["source"] = ops_summary.get("source")
                        payload["provenance_holes"] = mechanical_table.get(
                            "provenance_holes", payload.get("provenance_holes")
                        )
                        payload["chart_lines"] = [
                            _format_chart_line(row)
                            for row in rows_list
                            if isinstance(row, Mapping)
                        ]
            scan_info = get_last_acad_table_scan() or {}
            try:
                tables_found = int(scan_info.get("tables_found", 0))
            except Exception:
                tables_found = 0
            log_last_dxf_fallback(tables_found)
            if tables_found:
                break
    return payload


def extract_for_app(doc: Any, *, opts: Mapping[str, Any] | None = None, **read_kwargs: Any) -> dict[str, Any]:
    """Return a normalized payload suitable for application diagnostics."""

    payload_raw = read_geo(doc, **read_kwargs)
    payload: dict[str, Any]
    if isinstance(payload_raw, Mapping):
        payload = dict(payload_raw)
    else:
        payload = {}

    geo_obj = payload.get("geo") if isinstance(payload, Mapping) else None
    if isinstance(geo_obj, Mapping):
        geo_map = geo_obj if isinstance(geo_obj, dict) else dict(geo_obj)
    else:
        geo_map = {}
    payload["geo"] = geo_map

    ops_summary_candidate = payload.get("ops_summary")
    ops_summary_map = _ensure_ops_summary_map(ops_summary_candidate)
    if not ops_summary_map and isinstance(geo_map, Mapping):
        ops_summary_map = _ensure_ops_summary_map(geo_map.get("ops_summary"))
    payload["ops_summary"] = ops_summary_map

    def _rows_list(value: Any) -> list[Mapping[str, Any]]:
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, Mapping)]
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [entry for entry in value if isinstance(entry, Mapping)]
        return []

    rows_list = _rows_list(payload.get("rows"))
    if not rows_list and isinstance(ops_summary_map, Mapping):
        rows_list = _rows_list(ops_summary_map.get("rows"))
    if rows_list:
        payload["rows"] = list(rows_list)

    qty_sum: int
    qty_value = payload.get("qty_sum")
    if isinstance(qty_value, (int, float)):
        qty_sum = int(float(qty_value))
    else:
        qty_sum = _sum_qty(rows_list)
    payload["qty_sum"] = qty_sum

    holes_source = payload.get("provenance_holes")
    if holes_source is None and isinstance(geo_map, Mapping):
        provenance = geo_map.get("provenance")
        if isinstance(provenance, Mapping):
            holes_source = provenance.get("holes")
    if holes_source is not None:
        payload["provenance_holes"] = holes_source

    hole_count = payload.get("hole_count")
    if hole_count in (None, "") and isinstance(geo_map, Mapping):
        hole_count = geo_map.get("hole_count")
    try:
        if hole_count not in (None, ""):
            hole_count = int(float(hole_count))
    except Exception:
        pass
    if hole_count in (None, ""):
        hole_count = qty_sum
    payload["hole_count"] = hole_count

    source = payload.get("source")
    if source in (None, "") and isinstance(ops_summary_map, Mapping):
        source = ops_summary_map.get("source")
    if source not in (None, ""):
        payload["source"] = source

    def _extract_hole_sets_from_geo(candidate: Mapping[str, Any] | None) -> Any:
        if not isinstance(candidate, Mapping):
            return None
        direct = candidate.get("hole_sets")
        nested = candidate.get("geo")
        if direct is not None:
            if (
                isinstance(direct, (Mapping, Sequence))
                and not isinstance(direct, (str, bytes))
                and not direct
                and isinstance(nested, Mapping)
            ):
                nested_result = _extract_hole_sets_from_geo(nested)
                if nested_result is not None:
                    return nested_result
            return direct
        if isinstance(nested, Mapping):
            return _extract_hole_sets_from_geo(nested)
        return None

    hole_sets_payload = _extract_hole_sets_from_geo(geo_map)
    geom_holes_payload = payload.get("geom_holes")
    if not isinstance(geom_holes_payload, Mapping) and isinstance(geo_map, Mapping):
        geom_candidate = geo_map.get("geom_holes")
        if isinstance(geom_candidate, Mapping):
            geom_holes_payload = geom_candidate

    authoritative_table = _table_source_is_authoritative(source, len(rows_list))
    manifest_payload = ops_manifest(
        rows_list,
        geom_holes=geom_holes_payload if isinstance(geom_holes_payload, Mapping) else None,
        hole_sets=hole_sets_payload,
        authoritative_table=authoritative_table,
    )
    payload["ops_manifest"] = dict(manifest_payload)

    opts_map = dict(opts or {})
    errors: dict[str, str] = {}
    try:
        hole_table = extract_hole_table(doc, opts=opts_map) or {}
    except Exception as exc:
        errors["extract_hole_table"] = str(exc)
        hole_table = {}
    if isinstance(hole_table, Mapping):
        payload["extract_hole_table"] = dict(hole_table)
    else:
        payload["extract_hole_table"] = {}

    result: dict[str, Any] = {
        "payload": payload,
        "rows": rows_list,
        "qty_sum": qty_sum,
        "hole_count": hole_count,
        "source": source,
        "provenance_holes": holes_source,
        "ops_summary": ops_summary_map,
        "geom_holes": geom_holes_payload if isinstance(geom_holes_payload, Mapping) else None,
        "hole_sets": hole_sets_payload,
        "manifest": manifest_payload,
        "ops_manifest": manifest_payload,
        "published": bool(rows_list),
        "extract_hole_table": payload.get("extract_hole_table", {}),
        "errors": errors,
    }
    return result


def extract_geo_from_path(
    path: str,
    *,
    prefer_table: bool = True,
    use_oda: bool = True,
    feature_flags: Mapping[str, Any] | None = None,
    force_text: bool = False,
    pipeline: str = "auto",
    allow_geom: bool = False,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
    layer_include_regex: Iterable[str] | str | None = None,
    layer_exclude_regex: Iterable[str] | str | None = DEFAULT_TEXT_LAYER_EXCLUDE_REGEX,
    debug_layouts: bool = False,
) -> dict[str, Any]:
    """Load DWG/DXF at ``path`` and return a GEO dictionary."""

    path_obj = Path(path)
    payload = _read_geo_payload_from_path(
        path_obj,
        prefer_table=prefer_table,
        use_oda=use_oda,
        feature_flags=feature_flags,
        force_text=force_text,
        pipeline=pipeline,
        allow_geom=allow_geom,
        layer_allowlist=layer_allowlist,
        block_name_allowlist=block_name_allowlist,
        block_name_regex=block_name_regex,
        layer_include_regex=layer_include_regex,
        layer_exclude_regex=layer_exclude_regex,
        debug_layouts=debug_layouts,
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
    force_text: bool = False,
    pipeline: str = "auto",
    allow_geom: bool = False,
    layer_allowlist: Iterable[str] | None = _DEFAULT_LAYER_ALLOWLIST,
    block_name_allowlist: Iterable[str] | None = None,
    block_name_regex: Iterable[str] | str | None = None,
    layer_include_regex: Iterable[str] | str | None = None,
    layer_exclude_regex: Iterable[str] | str | None = DEFAULT_TEXT_LAYER_EXCLUDE_REGEX,
    debug_layouts: bool = False,
) -> dict[str, Any]:
    """Load DWG/DXF at ``path`` and return a GEO dictionary."""

    path_obj = Path(path)
    return _read_geo_payload_from_path(
        path_obj,
        prefer_table=prefer_table,
        use_oda=use_oda,
        feature_flags=feature_flags,
        force_text=force_text,
        pipeline=pipeline,
        allow_geom=allow_geom,
        layer_allowlist=layer_allowlist,
        block_name_allowlist=block_name_allowlist,
        block_name_regex=block_name_regex,
        layer_include_regex=layer_include_regex,
        layer_exclude_regex=layer_exclude_regex,
        debug_layouts=debug_layouts,
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
    "NoTextRowsError",
    "NO_TEXT_ROWS_MESSAGE",
    "read_geo",
    "extract_for_app",
    "extract_geo_from_path",
    "read_acad_table",
    "rows_from_acad_table",
    "read_geo",
    "read_text_table",
    "read_geo",
    "choose_better_table",
    "promote_table_to_geo",
    "extract_hole_table",
    "extract_geometry",
    "read_geo",
    "get_last_text_table_debug",
    "get_last_acad_table_scan",
    "set_trace_acad",
    "log_last_dxf_fallback",
    "DEFAULT_TEXT_LAYER_EXCLUDE_REGEX",
    "collect_all_text",
]

