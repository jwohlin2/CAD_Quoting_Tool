"""Shared DXF enrichment helpers used by the CLI and GUI flows.

Also includes DWG/DXF punch feature extraction for automated quoting.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from cad_quoter.geo_extractor import collect_all_text

from cad_quoter.vendors import ezdxf as _ezdxf_vendor

try:  # pragma: no cover - optional helper available during packaging
    from cad_quoter.geometry.hole_operations import explode_rows_to_operations
except Exception:  # pragma: no cover - optional dependency unavailable
    explode_rows_to_operations = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    _EZDXF = _ezdxf_vendor.require_ezdxf()
except Exception:  # pragma: no cover - optional dependency
    _EZDXF = None

RE_TAP = re.compile(
    r"(\(\s*\d+\s*\)\s*)?(#\s*\d{1,2}-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?)\s*TAP",
    re.I,
)
RE_CBORE = re.compile(r"C[’']?BORE|CBORE|COUNTERBORE", re.I)
RE_CSK = re.compile(r"CSK|C['’]SINK|COUNTERSINK", re.I)
RE_THRU = re.compile(r"\bTHRU\b", re.I)
NUM_PATTERN = r"(?:\d*\.\d+|\d+)"

RE_DEPTH = re.compile(rf"({NUM_PATTERN})\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?", re.I)
RE_QTY = re.compile(r"\((\d+)\)")
RE_REF_D = re.compile(rf"\bREF\s*[Ø⌀]?\s*({NUM_PATTERN})", re.I)
RE_DIA = re.compile(rf"[Ø⌀\u00D8]?\s*({NUM_PATTERN})", re.I)
RE_FROMBK = re.compile(r"\bFROM\s+BACK\b", re.I)

RE_DIAM_TOKEN = re.compile(r"[Ø⌀\u00D8]\s*(\d+\s*/\s*\d+|\d+(?:\.\d+)?|\.\d+)", re.I)
RE_GENERIC_NUM = re.compile(r"(\d+\s*/\s*\d+|\d+(?:\.\d+)?|\.\d+)")
RE_DEPTH_TOKEN = re.compile(
    r"(?:X|×|DEEP|DEPTH)\s*(?:TO\s*)?(?:Ø|⌀)?\s*(\d+\s*/\s*\d+|\d+(?:\.\d+)?|\.\d+)",
    re.I,
)

_UHEX_RE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")

RE_MAT = re.compile(r"\b(MATERIAL|MAT)\b[:\s]*([A-Z0-9\-\s/\.]+)")
RE_COAT = re.compile(
    r"\b(ANODIZE|BLACK OXIDE|ZINC PLATE|NICKEL PLATE|PASSIVATE|HEAT TREAT|DLC|PVD|CVD)\b",
    re.I,
)
RE_TOL = re.compile(r"\bUNLESS OTHERWISE SPECIFIED\b.*?([±\+\-]\s*\d+\.\d+)", re.I | re.S)
RE_REV = re.compile(r"\bREV(ISION)?\b[:\s]*([A-Z0-9\-]+)")


def _decode_uplus(text: str) -> str:
    r"""Decode AutoCAD ``\U+XXXX`` sequences to Unicode characters."""

    def _replace(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except Exception:
            return match.group(0)

    return _UHEX_RE.sub(_replace, text or "")


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except Exception:
        return None
    return number


def _flatten_block_path(path_value: Any) -> str:
    if isinstance(path_value, (list, tuple)):
        return "/".join(str(part) for part in path_value if str(part))
    if isinstance(path_value, str):
        return path_value
    return ""


def detect_units_scale(doc: Any) -> Dict[str, float | int]:
    """Return INSUNITS metadata and inches conversion factor for ``doc``."""

    try:
        units = int(doc.header.get("$INSUNITS", 1))
    except Exception:
        units = 1
    to_in = 1.0 if units == 1 else (1 / 25.4) if units in (4, 13) else 1.0
    return {"insunits": units, "to_in": float(to_in)}


def iter_spaces(doc: Any) -> List[Any]:
    """Return unique entity spaces (model + layouts) for ``doc``."""

    spaces: List[Any] = []
    if doc is None:
        return spaces

    seen: set[int] = set()
    try:
        msp = doc.modelspace()
    except Exception:
        msp = None
    if msp is not None and id(msp) not in seen:
        seen.add(id(msp))
        spaces.append(msp)

    try:
        layout_names = list(doc.layouts.names_in_taborder())
    except Exception:
        layout_names = []

    for layout_name in layout_names:
        if layout_name.lower() in {"model", "defpoints"}:
            continue
        try:
            entity_space = doc.layouts.get(layout_name).entity_space
        except Exception:
            continue
        if entity_space is None or id(entity_space) in seen:
            continue
        seen.add(id(entity_space))
        spaces.append(entity_space)

    return spaces


_DEBUG_DIR = Path("debug")
_DEBUG_CHART_PATH = _DEBUG_DIR / "chart_text_raw.csv"
_DEBUG_CHART_FIELDNAMES = ("etype", "layer", "height", "x", "y", "text")


def _point_to_xy(point: Any) -> tuple[float | None, float | None]:
    try:
        if hasattr(point, "xyz"):
            x_val, y_val, _ = point.xyz
        else:
            x_val, y_val = point[0], point[1]
    except Exception:
        return (None, None)
    try:
        return (float(x_val), float(y_val))
    except Exception:
        return (None, None)


def _entity_xy(entity: Any) -> tuple[float | None, float | None]:
    candidates = ("insert", "alignment_point", "center", "start")
    for attr in candidates:
        try:
            value = getattr(entity.dxf, attr)
        except Exception:
            value = None
        if value is None:
            continue
        x_val, y_val = _point_to_xy(value)
        if x_val is not None or y_val is not None:
            return (x_val, y_val)
    return (None, None)


def _entity_height(entity: Any) -> float | None:
    for attr in ("char_height", "height"):
        try:
            value = getattr(entity.dxf, attr)
        except Exception:
            value = None
        if value in (None, ""):
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _entity_text(entity: Any) -> str:
    try:
        etype = entity.dxftype()
    except Exception:
        etype = ""
    if etype == "MTEXT":
        try:
            if hasattr(entity, "plain_text"):
                return str(entity.plain_text())
        except Exception:
            pass
        try:
            return str(entity.text)
        except Exception:
            return ""
    if etype == "TEXT":
        try:
            return str(entity.dxf.text)
        except Exception:
            return ""
    return ""


def _collect_chart_layout_records(doc: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if doc is None:
        return records
    try:
        layouts = list(doc.layouts.names_in_taborder())
    except Exception:
        layouts = []
    target_layout = None
    for layout_name in layouts:
        if layout_name and layout_name.upper() == "CHART":
            target_layout = layout_name
            break
    if not target_layout:
        return records
    try:
        layout = doc.layouts.get(target_layout)
    except Exception:
        return records
    try:
        space = layout.entity_space
    except Exception:
        return records
    try:
        iterator = iter(space)
    except Exception:
        iterator = []
    for entity in iterator:
        try:
            etype = entity.dxftype()
        except Exception:
            etype = ""
        try:
            layer = str(getattr(entity.dxf, "layer", "") or "")
        except Exception:
            layer = ""
        height = _entity_height(entity)
        x_val, y_val = _entity_xy(entity)
        text = _entity_text(entity)
        records.append(
            {
                "etype": etype or "",
                "layer": layer,
                "height": height,
                "x": x_val,
                "y": y_val,
                "text": text,
            }
        )
    return records


def _write_chart_layout_debug(records: list[dict[str, Any]]) -> None:
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        with _DEBUG_CHART_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_DEBUG_CHART_FIELDNAMES)
            writer.writeheader()
            for record in records:
                row = {
                    "etype": record.get("etype", ""),
                    "layer": record.get("layer", ""),
                    "height": record.get("height"),
                    "x": record.get("x"),
                    "y": record.get("y"),
                    "text": record.get("text", ""),
                }
                writer.writerow(row)
    except Exception:
        pass


def iter_table_entities(doc: Any) -> Iterator[Any]:
    if doc is None:
        return

    seen: set[int] = set()

    def _yield_from_insert(ins: Any) -> Iterator[Any]:
        try:
            virtual_entities = ins.virtual_entities()
        except Exception:
            return
        for sub in virtual_entities:
            try:
                dxftype = sub.dxftype()
            except Exception:
                dxftype = ""
            if dxftype == "TABLE":
                key = id(sub)
                if key not in seen:
                    seen.add(key)
                    yield sub
            elif dxftype == "INSERT":
                yield from _yield_from_insert(sub)

    for space in iter_spaces(doc):
        try:
            tables = space.query("TABLE")
        except Exception:
            tables = []
        for table in tables:
            key = id(table)
            if key not in seen:
                seen.add(key)
                yield table
        try:
            inserts = space.query("INSERT")
        except Exception:
            inserts = []
        for ins in inserts:
            yield from _yield_from_insert(ins)


def _collect_normalized_text_rows(doc: Any) -> list[dict[str, Any]]:
    """Return normalized text rows harvested from ``doc`` via ``collect_all_text``."""

    try:
        raw_rows = collect_all_text(doc, max_block_depth=8)
    except Exception:
        raw_rows = []

    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        text_raw = row.get("text")
        if not isinstance(text_raw, str):
            text = ""
        else:
            text = _decode_uplus(text_raw)
        text = (text or "").strip()
        if not text:
            continue

        normalized.append(
            {
                "layout": str(row.get("layout") or ""),
                "layer": str(row.get("layer") or ""),
                "etype": str(row.get("etype") or ""),
                "text": text,
                "x": _coerce_float(row.get("x")),
                "y": _coerce_float(row.get("y")),
                "height": _coerce_float(row.get("height")),
                "rotation": _coerce_float(row.get("rotation")),
                "in_block": bool(row.get("in_block")),
                "depth": int(row.get("depth") or 0),
                "block_path": _flatten_block_path(row.get("block_path")),
            }
        )

    return normalized


def iter_table_text(doc: Any) -> Iterator[str]:
    """Yield text strings discovered inside TABLE and TEXT entities."""

    normalized_rows = _collect_normalized_text_rows(doc)
    for row in normalized_rows:
        etype = row.get("etype", "").upper()
        if etype in {"PROXYTEXT", "MTEXT", "TEXT", "TABLECELL"}:
            text = str(row.get("text") or "").strip()
            if text:
                yield text


def harvest_plate_dimensions(doc: Any, to_in: float) -> Dict[str, Any]:
    xs: List[float] = []
    ys: List[float] = []

    for space in iter_spaces(doc):
        try:
            dimensions = space.query("DIMENSION")
        except Exception:
            continue
        for dim in dimensions:
            try:
                dimtype = int(dim.dxf.dimtype)
                is_ord = (dimtype & 6) == 6 or dimtype == 6
            except Exception:
                is_ord = False
            if not is_ord:
                continue
            try:
                measurement = float(dim.get_measurement()) * to_in
            except Exception:
                continue
            if not (0.05 <= measurement <= 10000):
                continue
            try:
                angle = float(getattr(dim.dxf, "text_rotation", 0.0)) % 180.0
            except Exception:
                angle = 0.0
            target = ys if 60.0 <= angle <= 120.0 else xs
            target.append(measurement)

    def _pick(values: Iterable[float]) -> float | None:
        uniq = sorted({round(v, 3) for v in values if v > 0.2})
        return uniq[-1] if uniq else None

    return {
        "plate_len_in": _pick(ys),
        "plate_wid_in": _pick(xs),
        "prov": "ORDINATE DIMENSIONS",
    }


def harvest_outline_metrics(doc: Any, to_in: float) -> Dict[str, Any]:
    perimeter = 0.0
    area_in2 = 0.0

    for space in iter_spaces(doc):
        try:
            polylines = space.query("LWPOLYLINE")
        except Exception:
            polylines = []
        for pline in polylines:
            if not getattr(pline, "closed", False):
                continue
            try:
                perimeter += float(pline.length()) * to_in
                area_in2 += float(pline.area()) * (to_in ** 2)
            except Exception:
                continue

        try:
            circles = space.query("CIRCLE")
        except Exception:
            circles = []
        for circle in circles:
            try:
                radius = float(circle.dxf.radius) * to_in
            except Exception:
                continue
            perimeter += 2 * math.pi * radius
            area_in2 += math.pi * radius * radius

        try:
            arcs = space.query("ARC")
        except Exception:
            arcs = []
        for arc in arcs:
            try:
                angle = abs(float(arc.dxf.end_angle) - float(arc.dxf.start_angle)) * math.pi / 180.0
                radius = float(arc.dxf.radius) * to_in
            except Exception:
                continue
            perimeter += radius * angle

    return {
        "edge_len_in": round(perimeter, 3) if perimeter else None,
        "outline_area_in2": round(area_in2, 3) if area_in2 else None,
        "prov": "CLOSED LWPOLYLINE/ARC",
    }


def harvest_hole_geometry(doc: Any, to_in: float) -> Dict[str, Any]:
    diameters: List[float] = []

    for space in iter_spaces(doc):
        try:
            circles = space.query("CIRCLE")
        except Exception:
            circles = []
        for circle in circles:
            try:
                diameter = 2.0 * float(circle.dxf.radius) * to_in
            except Exception:
                continue
            if 0.04 <= diameter <= 6.0:
                diameters.append(round(diameter, 4))

    families = Counter(diameters)
    return {
        "hole_count_geom": len(diameters),
        "hole_diam_families_in": dict(families.most_common()),
        "min_hole_in": min(diameters) if diameters else None,
        "max_hole_in": max(diameters) if diameters else None,
        "prov": "CIRCLE entities",
    }


def _fractional_to_float(token: str) -> float | None:
    try:
        return float(Fraction(token))
    except (ValueError, ZeroDivisionError):
        return None


def _numeric_from_token(token: str) -> float | None:
    stripped = token.strip().strip(" \"'()")
    stripped = stripped.replace("∅", "Ø").replace("⌀", "Ø")
    stripped = stripped.strip()
    if not stripped:
        return None
    if "/" in stripped:
        return _fractional_to_float(stripped)
    if stripped.startswith("."):
        stripped = f"0{stripped}"
    try:
        return float(stripped)
    except Exception:
        return None


def _diameter_from_fields(*tokens: Any) -> float | None:
    for raw in tokens:
        if raw in (None, ""):
            continue
        text = str(raw)
        match = RE_DIAM_TOKEN.search(text)
        if match:
            value = _numeric_from_token(match.group(1))
            if value is not None:
                return value
        match_generic = RE_GENERIC_NUM.search(text)
        if match_generic:
            value = _numeric_from_token(match_generic.group(1))
            if value is not None and value > 0:
                return value
    return None


def _depth_from_description(desc: str) -> float | None:
    if not desc:
        return None
    match = RE_DEPTH_TOKEN.search(desc)
    if match:
        return _numeric_from_token(match.group(1))
    return None


def _coerce_positive_int(value: Any) -> int:
    if isinstance(value, (int, float)):
        try:
            ivalue = int(round(float(value)))
        except Exception:
            return 0
        return ivalue if ivalue > 0 else 0
    text = str(value or "").strip()
    if not text:
        return 0
    match = re.search(r"-?\d+", text)
    if not match:
        return 0
    try:
        ivalue = int(match.group(0))
    except Exception:
        return 0
    return abs(ivalue)


def _normalize_description(desc: str, qty: int) -> str:
    text = str(desc or "").strip()
    if not text:
        return ""
    if qty > 0:
        pattern = re.compile(rf"^[\(\[]?{qty}[\)\]]?\s*(?:X|×)?\s*", re.I)
        text = pattern.sub("", text, count=1)
    leading_qty = re.match(r"^[\(\[]?(\d+)[\)\]]?\s*(?:X|×)?\s*", text)
    if leading_qty:
        text = text[leading_qty.end() :]
    return text.strip()


def _classify_operation(desc: str) -> str:
    upper = desc.upper()
    if "JIG" in upper:
        return "jig"
    if "COUNTERDRILL" in upper or "C'DRILL" in upper or "C DRILL" in upper or "SPOT DRILL" in upper:
        return "cdrill"
    if "COUNTERBORE" in upper or "C'BORE" in upper or "CBORE" in upper:
        return "cbore"
    if "COUNTERSINK" in upper or "C'SINK" in upper or "CSK" in upper:
        return "csk"
    if "TAP" in upper:
        return "tap"
    if "REAM" in upper or "DRILL" in upper:
        return "drill"
    # Detect slot/obround features
    if "SLOT" in upper or "OBROUND" in upper or "ELONGATED" in upper:
        return "slot"
    # Detect slot patterns like "R.094" with length indicators
    import re
    if re.search(r'\bR[\.\d]+\s*(?:X\s*[\d\.]+|OVER\s*R)', upper):
        return "slot"
    return "other"


def _side_from_description(desc: str) -> str | None:
    upper = desc.upper()
    if "FRONT" in upper and "BACK" in upper:
        return "BOTH"
    if "BACK" in upper:
        return "BACK"
    if "FRONT" in upper:
        return "FRONT"
    return None


def harvest_hole_table(doc: Any) -> Dict[str, Any]:
    taps = cbore = csk = 0
    deepest = 0.0
    from_back = False
    lines: List[str] = []
    family_guess: Counter[float] = Counter()
    structured_rows: list[dict[str, Any]] = []
    ops_rows: list[dict[str, Any]] = []
    totals_by_type: Counter[str] = Counter()
    hole_qty_by_name: dict[str, int] = {}
    structured_acc: dict[tuple[str, str], dict[str, Any]] = {}

    try:
        from cad_quoter.geo_dump import _find_hole_table_chunks, _parse_header, _split_descriptions
    except Exception:  # pragma: no cover - helpers unavailable
        _find_hole_table_chunks = _parse_header = _split_descriptions = None  # type: ignore[assignment]

    try:
        chart_debug_records = _collect_chart_layout_records(doc)
    except Exception:
        chart_debug_records = []
    try:
        _write_chart_layout_debug(chart_debug_records)
    except Exception:
        pass

    normalized_rows = _collect_normalized_text_rows(doc)
    candidate_rows = [
        row
        for row in normalized_rows
        if str(row.get("etype") or "").upper() in {"PROXYTEXT", "MTEXT", "TEXT", "TABLECELL"}
    ]

    header_chunks: List[str] = []
    body_chunks: List[str] = []
    if _find_hole_table_chunks is not None and candidate_rows:
        try:
            header_chunks, body_chunks = _find_hole_table_chunks(candidate_rows)
        except Exception:
            header_chunks, body_chunks = [], []

    text_rows = [chunk for chunk in header_chunks + body_chunks if chunk]

    if not text_rows:
        keywords = ("HOLE", "TAP", "THRU", "CBORE", "C'BORE", "DRILL", "Ø", "⌀")
        for row in candidate_rows:
            text_val = str(row.get("text") or "")
            upper = text_val.upper()
            if any(keyword in upper for keyword in keywords):
                text_rows.append(text_val)

    lines = list(text_rows)

    hole_letters: List[str] = []
    diam_tokens: List[str] = []
    qty_tokens: List[int] = []
    if header_chunks and _parse_header is not None and _split_descriptions is not None:
        try:
            hole_letters, diam_tokens, qty_tokens = _parse_header(header_chunks)
        except Exception:
            hole_letters, diam_tokens, qty_tokens = [], [], []
        if diam_tokens:
            try:
                descs = _split_descriptions(body_chunks, diam_tokens)
            except Exception:
                descs = [""] * len(diam_tokens)
            n = min(len(hole_letters), len(diam_tokens), len(qty_tokens))
            for idx in range(n):
                hole = hole_letters[idx]
                ref = diam_tokens[idx]
                qty_val = qty_tokens[idx]
                desc = (descs[idx] if idx < len(descs) else "").strip()
                key = (hole, ref)
                structured_acc[key] = {
                    "HOLE": hole,
                    "REF_DIAM": ref,
                    "QTY": qty_val,
                    "DESCRIPTION": [desc] if desc else [],
                    "_header_desc": bool(desc),
                }
                key_name = hole or ref
                if key_name:
                    current_qty = hole_qty_by_name.get(key_name, 0)
                    if qty_val > current_qty:
                        hole_qty_by_name[key_name] = qty_val

    for raw in text_rows:
        upper = raw.upper()
        if not any(
            keyword in upper for keyword in ("HOLE", "TAP", "THRU", "CBORE", "C'BORE", "DRILL", "Ø", "⌀")
        ):
            continue

        qty = 1
        match_qty = RE_QTY.search(upper)
        if match_qty:
            qty = int(match_qty.group(1))

        if RE_TAP.search(upper):
            taps += qty
        if RE_CBORE.search(upper):
            cbore += qty
        if RE_CSK.search(upper):
            csk += qty

        match_depth = RE_DEPTH.search(upper)
        if match_depth:
            try:
                deepest = max(deepest, float(match_depth.group(1)))
            except Exception:
                pass
            if (match_depth.group(2) or "").upper() == "BACK" or RE_FROMBK.search(upper):
                from_back = True

        match_ref = RE_REF_D.search(upper) if "REF" in upper else None
        if match_ref:
            try:
                family_guess[round(float(match_ref.group(1)), 4)] += qty
            except Exception:
                pass
        else:
            match_dia = RE_DIA.search(upper)
            if match_dia and ("Ø" in upper or "⌀" in upper):
                try:
                    family_guess[round(float(match_dia.group(1)), 4)] += qty
                except Exception:
                    pass

    raw_ops: Sequence[Sequence[Any] | Mapping[str, Any]] = []
    if text_rows and explode_rows_to_operations:
        try:
            raw_ops = explode_rows_to_operations(text_rows) or []
        except Exception:
            raw_ops = []

    if raw_ops:
        for entry in raw_ops:
            if isinstance(entry, Mapping):
                hole_id = str(entry.get("HOLE") or entry.get("hole") or "").strip()
                ref_token = str(entry.get("REF_DIAM") or entry.get("ref") or "").strip()
                qty_token: Any = entry.get("QTY") or entry.get("qty")
                desc_raw = entry.get("DESCRIPTION/DEPTH") or entry.get("desc") or ""
                desc_text = str(desc_raw)
            else:
                seq = [str(part) for part in entry]
                hole_id = seq[0].strip() if seq else ""
                ref_token = seq[1].strip() if len(seq) > 1 else ""
                qty_token = seq[2] if len(seq) > 2 else ""
                if len(seq) > 3:
                    desc_text = " ".join(part for part in seq[3:] if part)
                else:
                    desc_text = seq[3] if len(seq) > 3 else ""

            qty_val = _coerce_positive_int(qty_token)
            desc_norm = _normalize_description(desc_text, qty_val)
            if qty_val <= 0:
                leading = re.match(r"^[\(\[]?(\d+)[\)\]]?", desc_norm)
                if leading:
                    qty_val = _coerce_positive_int(leading.group(1))
                    desc_norm = _normalize_description(desc_norm, qty_val)
            if qty_val <= 0 and not desc_norm:
                continue

            op_type = _classify_operation(desc_norm)
            totals_by_type[op_type] += qty_val
            if op_type == "tap":
                taps = max(taps, totals_by_type[op_type])
            elif op_type == "cbore":
                cbore = max(cbore, totals_by_type[op_type])
            elif op_type == "csk":
                csk = max(csk, totals_by_type[op_type])

            diameter_in = _diameter_from_fields(ref_token, desc_norm)
            if diameter_in is not None:
                rounded = round(float(diameter_in), 4)
                existing_qty = family_guess.get(rounded, 0)
                family_guess[rounded] = max(existing_qty, qty_val if qty_val > 0 else existing_qty)

            depth_in = _depth_from_description(desc_norm)
            if depth_in is not None:
                try:
                    deepest = max(deepest, float(depth_in))
                except Exception:
                    pass

            side = _side_from_description(desc_norm)
            thru = "THRU" in desc_norm.upper()
            if side in {"BACK", "BOTH"}:
                from_back = True

            op_entry: dict[str, Any] = {
                "hole": hole_id,
                "ref": ref_token,
                "qty": qty_val,
                "desc": desc_norm,
                "type": op_type,
            }
            if diameter_in is not None:
                op_entry["diameter_in"] = float(round(diameter_in, 4))
            if depth_in is not None:
                op_entry["depth_in"] = float(round(depth_in, 4))
            if thru:
                op_entry["thru"] = True
            if side:
                op_entry["side"] = side

            ops_rows.append(op_entry)

            key = hole_id or ref_token
            if key:
                current = hole_qty_by_name.get(key, 0)
                if qty_val > current:
                    hole_qty_by_name[key] = qty_val

            if hole_id or ref_token:
                struct_key = (hole_id, ref_token)
                struct_entry = structured_acc.setdefault(
                    struct_key,
                    {
                        "HOLE": hole_id,
                        "REF_DIAM": ref_token,
                        "QTY": qty_val,
                        "DESCRIPTION": [],
                        "_header_desc": False,
                    },
                )
                if qty_val > struct_entry.get("QTY", 0):
                    struct_entry["QTY"] = qty_val
                if desc_norm:
                    desc_list = struct_entry.setdefault("DESCRIPTION", [])
                    if not isinstance(desc_list, list):
                        desc_list = [str(desc_list)]
                    header_seeded = bool(struct_entry.get("_header_desc")) and bool(desc_list)
                    if not header_seeded and desc_norm not in desc_list:
                        desc_list.append(desc_norm)
                    struct_entry["DESCRIPTION"] = desc_list

    if structured_acc:
        for entry in structured_acc.values():
            desc_list = entry.get("DESCRIPTION") or []
            if desc_list:
                deduped: list[str] = []
                seen: set[str] = set()
                for fragment in desc_list:
                    if fragment not in seen:
                        deduped.append(fragment)
                        seen.add(fragment)
                entry["DESCRIPTION"] = "; ".join(deduped)
            else:
                entry["DESCRIPTION"] = ""
            structured_rows.append(
                {
                    "HOLE": entry.get("HOLE", ""),
                    "REF_DIAM": entry.get("REF_DIAM", ""),
                    "QTY": entry.get("QTY", 0),
                    "DESCRIPTION": entry.get("DESCRIPTION", ""),
                }
            )
        structured_rows.sort(key=lambda row: (str(row.get("HOLE") or ""), str(row.get("REF_DIAM") or "")))

    ops_hole_total = sum(hole_qty_by_name.values())

    return {
        "tap_qty": taps,
        "cbore_qty": cbore,
        "csk_qty": csk,
        "deepest_hole_in": deepest or None,
        "holes_from_back": bool(from_back),
        "hole_table_families_in": dict(family_guess.most_common()) if family_guess else None,
        "chart_lines": lines,
        "prov": "HOLE TABLE / TEXT",
        "provenance": "table_ops" if ops_rows else "HOLE TABLE / TEXT",
        "structured": structured_rows,
        "ops": ops_rows,
        "ops_totals": dict(totals_by_type) if totals_by_type else None,
        "hole_count_ops": ops_hole_total or None,
        "cdrill_qty": totals_by_type.get("cdrill", 0),
        "jig_qty": totals_by_type.get("jig", 0),
        "drill_qty": totals_by_type.get("drill", 0),
    }


def hole_ops_to_drill_bins(
    ops_rows: Sequence[Mapping[str, Any]] | None, plate_thk_in: Any | None
) -> tuple[list[dict[str, Any]], int, int]:
    """Convert HOLE TABLE operations into drill bins and deep/std counts."""

    if not isinstance(ops_rows, Sequence):
        return ([], 0, 0)

    try:
        thickness = float(plate_thk_in) if plate_thk_in is not None else None
    except Exception:
        thickness = None
    if thickness is not None and (not math.isfinite(thickness) or thickness <= 0):
        thickness = None

    bins: dict[float, dict[str, Any]] = {}
    deep_qty = 0
    std_qty = 0

    for row in ops_rows:
        if not isinstance(row, Mapping):
            continue
        op_type = str(row.get("type") or "").lower()
        if op_type != "drill":
            continue
        qty_val = _coerce_positive_int(row.get("qty"))
        if qty_val <= 0:
            continue

        diameter_in = row.get("diameter_in")
        if diameter_in is None:
            diameter_in = _diameter_from_fields(row.get("ref"), row.get("desc"))
        try:
            dia_val = float(diameter_in)
        except Exception:
            continue
        if not math.isfinite(dia_val) or dia_val <= 0:
            continue

        depth_raw = row.get("depth_in")
        try:
            depth_val = float(depth_raw) if depth_raw is not None else None
        except Exception:
            depth_val = None
        if depth_val is not None and (not math.isfinite(depth_val) or depth_val <= 0):
            depth_val = None
        if depth_val is None:
            if bool(row.get("thru")) and thickness is not None:
                depth_val = thickness
            elif thickness is not None:
                depth_val = thickness

        is_deep = False
        if depth_val is not None and dia_val > 0:
            try:
                is_deep = float(depth_val) >= 3.0 * float(dia_val) - 1e-6
            except Exception:
                is_deep = False

        if is_deep:
            deep_qty += qty_val
        else:
            std_qty += qty_val

        key = round(float(dia_val), 4)
        bucket = bins.setdefault(
            key,
            {
                "diameter_in": float(round(dia_val, 4)),
                "qty": 0,
                "op": "drill",
                "source": "table_ops",
            },
        )
        bucket["qty"] += qty_val
        if depth_val is not None:
            bucket["depth_in"] = float(round(depth_val, 4))
        if is_deep:
            bucket["op"] = "deep_drill"

    ordered = [bins[key] for key in sorted(bins.keys())]
    return (ordered, deep_qty, std_qty)


def harvest_title_notes(doc: Any) -> Dict[str, Any]:
    text_dump: List[str] = []

    for space in iter_spaces(doc):
        try:
            inserts = space.query("INSERT")
        except Exception:
            inserts = []
        for ins in inserts:
            try:
                block_name = ins.dxf.name.upper()
            except Exception:
                continue
            for attr in getattr(ins, "attribs", []):
                try:
                    text_dump.append(attr.dxf.tag)
                    text_dump.append(attr.plain_text())
                except Exception:
                    continue
            try:
                block_def = doc.blocks.get(block_name)
            except Exception:
                block_def = None
            if block_def is None:
                continue
            try:
                block_text = block_def.query("TEXT, MTEXT")
            except Exception:
                block_text = []
            for entity in block_text:
                try:
                    text = entity.plain_text() if entity.dxftype() == "MTEXT" else entity.dxf.text
                except Exception:
                    continue
                text_dump.append(text)

    for line in iter_table_text(doc):
        text_dump.append(line)

    combined = "\n".join(str(item) for item in text_dump if item)
    upper = combined.upper()

    material_note = None
    match_mat = RE_MAT.search(upper)
    if match_mat:
        material_note = match_mat.group(2).strip()

    finishes = sorted({match.group(0).upper() for match in RE_COAT.finditer(upper)})

    tol_match = RE_TOL.search(upper)
    default_tol = tol_match.group(1).replace(" ", "") if tol_match else None

    rev_match = RE_REV.search(upper)
    revision = rev_match.group(2).strip() if rev_match else None

    return {
        "material_note": material_note,
        "finishes": finishes,
        "default_tol": default_tol,
        "revision": revision,
        "prov": "TITLE BLOCK / NOTES",
    }


def build_geo_from_doc(doc: Any) -> Dict[str, Any]:
    if doc is None:
        return {"ok": False, "error": "DXF document is unavailable"}

    units = detect_units_scale(doc)
    to_in = units["to_in"]

    dims = harvest_plate_dimensions(doc, to_in)
    outline = harvest_outline_metrics(doc, to_in)
    holes = harvest_hole_geometry(doc, to_in)
    hole_table = harvest_hole_table(doc)
    title = harvest_title_notes(doc)

    geo = {
        "ok": True,
        "units": units,
        "plate_len_in": dims.get("plate_len_in"),
        "plate_wid_in": dims.get("plate_wid_in"),
        "edge_len_in": outline.get("edge_len_in"),
        "outline_area_in2": outline.get("outline_area_in2"),
        "hole_count_geom": holes.get("hole_count_geom"),
        "hole_diam_families_in": holes.get("hole_diam_families_in"),
        "min_hole_in": holes.get("min_hole_in"),
        "max_hole_in": holes.get("max_hole_in"),
        "tap_qty": hole_table.get("tap_qty", 0),
        "cbore_qty": hole_table.get("cbore_qty", 0),
        "csk_qty": hole_table.get("csk_qty", 0),
        "deepest_hole_in": hole_table.get("deepest_hole_in"),
        "holes_from_back": hole_table.get("holes_from_back", False),
        "hole_table_families_in": hole_table.get("hole_table_families_in"),
        "chart_lines": hole_table.get("chart_lines"),
        "material_note": title.get("material_note"),
        "finishes": title.get("finishes", []),
        "default_tol": title.get("default_tol"),
        "revision": title.get("revision"),
        "provenance": {
            "plate_size": dims.get("prov"),
            "edge_len": outline.get("prov"),
            "holes_geom": holes.get("prov"),
            "hole_table": hole_table.get("provenance") or hole_table.get("prov"),
            "material": title.get("prov"),
        },
    }

    # Normalise hole table data with structured/ops outputs
    hole_table_structured = []
    raw_structured = hole_table.get("structured")
    if isinstance(raw_structured, Sequence):
        for row in raw_structured:
            if isinstance(row, Mapping):
                hole_table_structured.append(dict(row))

    hole_table_ops = []
    raw_ops = hole_table.get("ops")
    if isinstance(raw_ops, Sequence):
        for op in raw_ops:
            if isinstance(op, Mapping):
                hole_table_ops.append(dict(op))

    if hole_table_structured:
        geo["hole_table_structured"] = hole_table_structured

    if hole_table_ops:
        formatted_ops: list[dict[str, Any]] = []
        for op in hole_table_ops:
            formatted = {
                "HOLE": op.get("hole"),
                "REF_DIAM": op.get("ref"),
                "QTY": op.get("qty"),
                "DESCRIPTION/DEPTH": op.get("desc"),
            }
            if op.get("type"):
                formatted["TYPE"] = op.get("type")
            if op.get("diameter_in") is not None:
                formatted["DIAMETER_IN"] = op.get("diameter_in")
            if op.get("depth_in") is not None:
                formatted["DEPTH_IN"] = op.get("depth_in")
            if op.get("side"):
                formatted["SIDE"] = op.get("side")
            if op.get("thru") is not None:
                formatted["THRU"] = bool(op.get("thru"))
            formatted_ops.append(formatted)
        geo["hole_table_ops"] = formatted_ops

    hole_table_provenance = hole_table.get("provenance") or hole_table.get("prov")
    hole_table_summary = {
        "tap_qty": geo.get("tap_qty", 0),
        "cbore_qty": geo.get("cbore_qty", 0),
        "csk_qty": geo.get("csk_qty", 0),
        "cdrill_qty": hole_table.get("cdrill_qty", 0),
        "jig_qty": hole_table.get("jig_qty", 0),
        "drill_qty": hole_table.get("drill_qty", 0),
        "hole_count_ops": hole_table.get("hole_count_ops") or 0,
    }
    ops_totals = hole_table.get("ops_totals")
    if isinstance(ops_totals, Mapping):
        hole_table_summary["ops_totals"] = dict(ops_totals)

    hole_table_payload = {
        "structured": hole_table_structured,
        "ops": hole_table_ops,
        "lines": list(hole_table.get("chart_lines") or []),
        "summary": hole_table_summary,
        "provenance": hole_table_provenance,
    }
    geo["hole_table"] = hole_table_payload

    geo["cdrill_qty"] = hole_table.get("cdrill_qty", 0)
    geo["jig_qty"] = hole_table.get("jig_qty", 0)
    geo["drill_qty_table"] = hole_table.get("drill_qty", 0)

    bins_list, deep_qty, std_qty = hole_ops_to_drill_bins(
        hole_table_ops,
        geo.get("deepest_hole_in"),
    )
    if bins_list or deep_qty or std_qty:
        geo["drill"] = {
            "bins_list": bins_list,
            "deep_qty": deep_qty,
            "std_qty": std_qty,
            "source": "table_ops",
        }
        if bins_list and not geo.get("bins_list"):
            geo["bins_list"] = bins_list

    hole_total = hole_table.get("hole_count_ops")
    try:
        hole_total_int = int(round(float(hole_total))) if hole_total not in (None, "") else 0
    except Exception:
        hole_total_int = 0
    if hole_total_int > 0:
        geo["hole_count"] = hole_total_int
        geo["hole_count_provenance"] = "table_ops"
        prov_map = geo.get("provenance")
        if isinstance(prov_map, dict):
            prov_map["holes"] = "table_ops"
        else:
            geo["provenance"] = {"holes": "table_ops"}
    elif hole_table_ops:
        geo.setdefault("hole_count_provenance", "table_ops")

    return geo


def build_geo_from_dxf(path: str) -> Dict[str, Any]:
    """Load ``path`` with ezdxf and return geometry enrichment data."""

    if _EZDXF is None:  # pragma: no cover - optional dependency
        return {"ok": False, "error": "ezdxf not installed"}

    try:
        doc = _EZDXF.readfile(path)
    except Exception as exc:
        return {"ok": False, "error": f"DXF read failed: {exc}"}

    return build_geo_from_doc(doc)


# ============================================================================
# DWG PUNCH FEATURE EXTRACTION
# ============================================================================


def normalize_acad_mtext(line: str) -> str:
    """
    Normalize AutoCAD MTEXT formatting codes into simpler plain text.

    Handles:
    - Strip outer {...}
    - Remove \\Hxx; (height) and \\Cxx; (color)
    - Convert stacked text \\S+.005^ -.000; -> '+.005/-.000'
    - Remove leftover '{}' braces
    """
    if not line:
        return ""

    if line.startswith("{") and line.endswith("}"):
        line = line[1:-1]

    line = re.sub(r"\\H[0-9.]+x;", "", line)
    line = re.sub(r"\\C\d+;", "", line)

    def repl_stack(m):
        top = m.group(1).strip()
        bot = m.group(2).strip()
        return f"{top}/{bot}"

    line = re.sub(r"\\S([^\\^]+)\^([^;]+);", repl_stack, line)
    line = line.replace("{}", "").strip()

    return line


def units_to_inch_factor(insunits: int) -> float:
    """Convert DXF $INSUNITS code to inch conversion factor."""
    units_factors = {
        0: 1.0,
        1: 1.0,
        2: 12.0,
        4: 1.0 / 25.4,
        5: 1.0 / 2.54,
        6: 39.3701,
    }
    return units_factors.get(insunits, 1.0)


def resolved_dimension_text(dim, unit_factor: float) -> str:
    """Resolve dimension text with <> placeholder replaced by numeric measurement."""
    raw_text = dim.dxf.text if hasattr(dim.dxf, 'text') else ""

    try:
        meas = dim.get_measurement()
        if meas is None:
            meas = 0
        if hasattr(meas, 'magnitude'):
            meas = meas.magnitude
        elif hasattr(meas, 'x'):
            meas = abs(meas.x)
        meas = float(meas)
    except Exception:
        meas = 0

    value_in = meas * unit_factor
    nominal_str = f"{value_in:.4f}".rstrip("0").rstrip(".")

    if nominal_str.startswith("0.") and value_in < 1.0:
        nominal_str = nominal_str[1:]
    elif not nominal_str or nominal_str == ".":
        nominal_str = "0"

    text = normalize_acad_mtext(raw_text) if raw_text else ""

    if "<>" in text and nominal_str:
        text = text.replace("<>", nominal_str)
    elif not text and nominal_str:
        text = nominal_str

    return text.strip()


def cluster_values(values: List[float], tolerance: float = 0.0002) -> List[float]:
    """Cluster similar values within a tolerance and return representative values."""
    if not values:
        return []

    sorted_vals = sorted(values)
    clusters = []
    current_cluster = [sorted_vals[0]]

    for val in sorted_vals[1:]:
        cluster_mean = sum(current_cluster) / len(current_cluster)
        if abs(val - cluster_mean) <= tolerance:
            current_cluster.append(val)
        else:
            clusters.append(current_cluster)
            current_cluster = [val]

    clusters.append(current_cluster)
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _collect_diameters_from_dimensions(doc, unit_factor: float) -> List[float]:
    """Collect diameter measurements from DIMENSION entities."""
    diameters = []
    msp = doc.modelspace()

    for dim in msp.query("DIMENSION"):
        try:
            dimtype = dim.dimtype
            raw_text = dim.dxf.text if hasattr(dim.dxf, 'text') else ""

            is_diameter = (
                dimtype == 3 or
                "%%c" in raw_text.lower() or
                "Ø" in raw_text or
                "Ø" in raw_text or
                " DIA" in raw_text.upper() or
                "DIA " in raw_text.upper()
            )
            is_radius = (dimtype == 4 or "R" in raw_text[:5])

            if is_diameter or is_radius:
                meas = dim.get_measurement()
                if meas is None:
                    continue
                if hasattr(meas, 'magnitude'):
                    meas = meas.magnitude
                elif hasattr(meas, 'x'):
                    meas = abs(meas.x)

                meas = float(meas)
                meas_in = meas * unit_factor

                if is_radius and not is_diameter:
                    meas_in *= 2.0

                if 0.01 <= meas_in <= 20.0:
                    diameters.append(meas_in)
        except Exception:
            continue

    return diameters


@dataclass
class PunchFeatureSummary:
    """Comprehensive feature summary for punch parts extracted from DWG/DXF."""

    family: str = "round_punch"
    shape_type: str = "round"
    overall_length_in: float = 0.0
    max_od_or_width_in: float = 0.0
    body_width_in: Optional[float] = None
    body_thickness_in: Optional[float] = None
    form_length_in: Optional[float] = None
    num_ground_diams: int = 0
    total_ground_length_in: float = 0.0
    # Turning time model parameters (round parts)
    shank_length: float = 0.0  # Length of major diameter section
    pilot_length: float = 0.0  # Length of minor/pilot diameter section
    shoulder_count: int = 0  # Number of diameter transitions
    flange_thickness: float = 0.0  # Flange/head thickness (if present)
    # Grinding time model parameters (round parts)
    grind_pilot_len: float = 0.0  # Length of pilot section to be ground
    grind_shank_len: float = 0.0  # Length of shank section to be ground
    grind_head_faces: int = 0  # Number of head/flange faces to grind
    has_perp_face_grind: bool = False
    has_3d_surface: bool = False
    form_complexity_level: int = 0
    tap_count: int = 0
    tap_summary: List[Dict[str, Any]] = field(default_factory=list)
    num_undercuts: int = 0
    num_chamfers: int = 0
    num_small_radii: int = 0
    min_dia_tol_in: Optional[float] = None
    min_len_tol_in: Optional[float] = None
    has_polish_contour: bool = False
    has_no_step_permitted: bool = False
    has_sharp_edges: bool = False
    has_gdt: bool = False
    has_etch: bool = False
    material_callout: Optional[str] = None
    extraction_source: str = "dxf_geometry_and_text"
    confidence_score: float = 1.0
    warnings: List[str] = field(default_factory=list)


def extract_geometry_envelope(dxf_path: Path) -> Dict[str, Any]:
    """Extract bounding box and envelope dimensions from DXF geometry."""
    if _EZDXF is None:
        return {
            "overall_length_in": 0.0, "overall_width_in": 0.0, "overall_height_in": 0.0,
            "bbox_min": (0, 0, 0), "bbox_max": (0, 0, 0), "units": "inches",
            "error": "ezdxf not available"
        }

    try:
        from ezdxf.bbox import extents
        doc = _EZDXF.readfile(str(dxf_path))
        msp = doc.modelspace()

        outline_entities = msp.query("LINE ARC CIRCLE LWPOLYLINE POLYLINE SPLINE")
        if not outline_entities:
            return {
                "overall_length_in": 0.0, "overall_width_in": 0.0, "overall_height_in": 0.0,
                "bbox_min": (0, 0, 0), "bbox_max": (0, 0, 0), "units": "inches",
                "error": "no geometry entities found"
            }

        bbox = extents(outline_entities)
        min_pt = bbox.extmin
        max_pt = bbox.extmax

        length = max_pt.x - min_pt.x
        width = max_pt.y - min_pt.y
        height = max_pt.z - min_pt.z if len(max_pt) > 2 else 0.0

        insunits = doc.header.get("$INSUNITS", 1)
        measurement = doc.header.get("$MEASUREMENT", 0)

        is_metric = measurement == 1 or insunits == 4
        if insunits in (0, 1) and max(length, width) > 50:
            is_metric = True

        if is_metric:
            length /= 25.4
            width /= 25.4
            height /= 25.4

        return {
            "overall_length_in": length, "overall_width_in": width, "overall_height_in": height,
            "bbox_min": tuple(min_pt), "bbox_max": tuple(max_pt), "units": "inches",
        }
    except Exception as e:
        return {
            "overall_length_in": 0.0, "overall_width_in": 0.0, "overall_height_in": 0.0,
            "bbox_min": (0, 0, 0), "bbox_max": (0, 0, 0), "units": "inches", "error": str(e)
        }


def extract_punch_dimensions(dxf_path: Path) -> Dict[str, Any]:
    """Extract dimension measurements and tolerances from DIMENSION entities."""
    if _EZDXF is None:
        return {
            "linear_dims": [], "diameter_dims": [], "resolved_dim_texts": [],
            "max_linear_dim": 0.0, "max_diameter_dim": 0.0,
            "min_dia_tol": None, "min_len_tol": None, "all_tolerances": [],
            "error": "ezdxf not available"
        }

    try:
        doc = _EZDXF.readfile(str(dxf_path))
        msp = doc.modelspace()

        insunits = doc.header.get("$INSUNITS", 1)
        measurement = doc.header.get("$MEASUREMENT", 0)
        unit_factor = units_to_inch_factor(insunits)

        is_metric = measurement == 1
        if is_metric and insunits not in [4, 5, 6]:
            unit_factor = 1.0 / 25.4

        linear_dims = []
        diameter_dims = []
        all_tolerances = []
        resolved_dim_texts = []

        for dim in msp.query("DIMENSION"):
            try:
                text_resolved = resolved_dimension_text(dim, unit_factor)
                resolved_dim_texts.append(text_resolved)

                raw_text = dim.dxf.text if hasattr(dim.dxf, 'text') else ""
                meas = dim.get_measurement()
                if meas is None:
                    continue

                if hasattr(meas, 'magnitude'):
                    meas = meas.magnitude
                elif hasattr(meas, 'x'):
                    meas = abs(meas.x)

                meas = float(meas)
                meas_in = meas * unit_factor
                dimtype = dim.dimtype

                is_diameter = (
                    dimtype == 3 or "%%c" in raw_text.lower() or
                    "Ø" in raw_text or "Ø" in text_resolved or
                    "⌀" in raw_text or "⌀" in text_resolved or
                    "DIA" in raw_text.upper()
                )

                if is_diameter:
                    diameter_dims.append({"measurement": meas_in, "text": text_resolved, "raw_text": raw_text, "type": "diameter"})
                else:
                    linear_dims.append({"measurement": meas_in, "text": text_resolved, "raw_text": raw_text, "type": dimtype})

                tolerances = parse_punch_tolerances_from_text(text_resolved)
                all_tolerances.extend(tolerances)
            except Exception:
                continue

        max_linear = max([d["measurement"] for d in linear_dims], default=0.0)
        max_diameter = max([d["measurement"] for d in diameter_dims], default=0.0)

        if not is_metric and (max_linear > 50 or max_diameter > 10):
            for d in linear_dims:
                d["measurement"] /= 25.4
            for d in diameter_dims:
                d["measurement"] /= 25.4
            max_linear /= 25.4
            max_diameter /= 25.4

        return {
            "linear_dims": linear_dims, "diameter_dims": diameter_dims,
            "resolved_dim_texts": resolved_dim_texts,
            "max_linear_dim": max_linear, "max_diameter_dim": max_diameter,
            "min_dia_tol": min([parse_punch_tolerances_from_text(d["text"]) for d in diameter_dims], default=[None])[0] if diameter_dims else None,
            "min_len_tol": min([parse_punch_tolerances_from_text(d["text"]) for d in linear_dims], default=[None])[0] if linear_dims else None,
            "all_tolerances": all_tolerances,
        }
    except Exception as e:
        return {
            "linear_dims": [], "diameter_dims": [], "resolved_dim_texts": [],
            "max_linear_dim": 0.0, "max_diameter_dim": 0.0,
            "min_dia_tol": None, "min_len_tol": None, "all_tolerances": [], "error": str(e)
        }


def parse_punch_tolerances_from_text(text: str) -> List[float]:
    """Parse tolerance values from dimension text."""
    tolerances = []
    matched_ranges = []

    def add_match(start, end, values):
        for s, e in matched_ranges:
            if not (end <= s or start >= e):
                return False
        matched_ranges.append((start, end))
        tolerances.extend(values)
        return True

    pm_pattern = r'±\s*(\d*\.?\d+)'
    for match in re.finditer(pm_pattern, text):
        tol = float(match.group(1))
        add_match(match.start(), match.end(), [abs(tol)])

    slash_pattern = r'\+\s*(\d*\.?\d+)\s*/\s*-\s*(\d*\.?\d+)'
    for match in re.finditer(slash_pattern, text):
        tol_plus = float(match.group(1))
        tol_minus = float(match.group(2))
        add_match(match.start(), match.end(), [abs(tol_plus), abs(tol_minus)])

    plus_minus_pattern = r'\+\s*(\d*\.?\d+)\s*-\s*(\d*\.?\d+)'
    for match in re.finditer(plus_minus_pattern, text):
        if any(s <= match.start() < e or s < match.end() <= e for s, e in matched_ranges):
            continue
        tol_plus = float(match.group(1))
        tol_minus = float(match.group(2))
        add_match(match.start(), match.end(), [abs(tol_plus), abs(tol_minus)])

    return tolerances


def classify_punch_family(text_dump: str) -> Tuple[str, str]:
    """Classify punch family and shape type from text."""
    text_upper = text_dump.upper()
    family = None

    if "PILOT PIN" in text_upper or "PILOT-PIN" in text_upper:
        family = "pilot_pin"
    elif "SPRING PIN" in text_upper or "SPRING-PIN" in text_upper:
        family = "round_punch"
    elif "GUIDE POST" in text_upper:
        family = "guide_post"
    elif "GUIDE BUSHING" in text_upper:
        family = "bushing"
    elif "FORM PUNCH" in text_upper or "COIN PUNCH" in text_upper:
        family = "form_punch"
    elif "DIE SECTION" in text_upper:
        family = "die_section"

    if family is None:
        if "INSERT" in text_upper or "COIN" in text_upper:
            family = "form_punch" if "PUNCH" in text_upper else "die_insert"
        elif "FORM" in text_upper and ("PUNCH" in text_upper or "DETAIL" in text_upper):
            family = "form_punch"
        elif "SECTION" in text_upper:
            family = "die_section"

    if family is None:
        if "BUSHING" in text_upper:
            family = "bushing"
        elif "PUNCH" in text_upper:
            family = "round_punch"
        else:
            family = "round_punch"

    shape = "round"
    if "RECTANGULAR" in text_upper or "SQUARE" in text_upper:
        shape = "rectangular"
    elif ("THICKNESS" in text_upper or "THK" in text_upper) and ("WIDTH" in text_upper or " W " in text_upper):
        shape = "rectangular"

    return family, shape


def is_plate_geometry(geo_envelope: Dict[str, Any], diameter_dims: List[Dict[str, Any]]) -> bool:
    """Check if geometry suggests a plate rather than a round/lathe part.

    A part should be classified as a plate if:
    - Thickness (Z) is relatively small (< 1.5")
    - Both in-plane dimensions (X, Y) are larger than thickness
    - The part doesn't have significant axisymmetric features (few/no diameter dimensions)

    Args:
        geo_envelope: Geometry envelope with overall_length_in, overall_width_in, overall_height_in
        diameter_dims: List of diameter dimension dicts from drawing

    Returns:
        True if the geometry suggests a plate/insert rather than a round punch
    """
    length = geo_envelope.get("overall_length_in", 0.0)
    width = geo_envelope.get("overall_width_in", 0.0)
    height = geo_envelope.get("overall_height_in", 0.0)

    # If we don't have height data, can't determine plate geometry
    if height <= 0:
        return False

    # Plate characteristics:
    # 1. Thickness is the smallest dimension (or close to it)
    # 2. Thickness is relatively small (< 1.5")
    # 3. Both in-plane dimensions are larger than thickness

    min_dim = min(length, width, height) if all(d > 0 for d in [length, width, height]) else 0

    # Check if height/thickness is the smallest or nearly smallest dimension
    is_thin = height < 1.5 and height > 0
    in_plane_larger = length > height and width > height

    # Check for absence of significant axisymmetric features
    # Large diameter dimensions (> 0.5") suggest turning operations
    large_diameters = [d for d in diameter_dims if d.get("measurement", 0) > 0.5]
    has_few_large_diameters = len(large_diameters) <= 1

    # Additional check: aspect ratio suggests plate
    # A typical plate has L and W within 5x of each other, and thickness much smaller
    if length > 0 and width > 0:
        aspect_ratio_planar = max(length, width) / min(length, width)
        aspect_ratio_thick = max(length, width) / height if height > 0 else 0

        # Plate: planar aspect < 5, thickness aspect > 1.5
        is_plate_aspect = aspect_ratio_planar < 5 and aspect_ratio_thick > 1.5
    else:
        is_plate_aspect = False

    return is_thin and in_plane_larger and has_few_large_diameters and is_plate_aspect


def detect_punch_material(text_dump: str) -> Optional[str]:
    """Detect material callout from text."""
    text_upper = text_dump.upper()

    lines = text_dump.split('\n')
    for line in lines:
        line_upper = line.upper()
        if ' PUNCH' in line_upper:
            tokens = line_upper.split()
            if len(tokens) >= 3 and tokens[-1] == 'PUNCH':
                material_candidate = tokens[-2]
                if re.match(r'^[A-Z0-9-]+$', material_candidate):
                    return material_candidate

    materials = [
        (r'\bA-?2\b', 'A2'), (r'\bA-?6\b', 'A6'), (r'\bA-?10\b', 'A10'),
        (r'\bD-?2\b', 'D2'), (r'\bD-?3\b', 'D3'), (r'\bM-?2\b', 'M2'),
        (r'\bM-?4\b', 'M4'), (r'\bO-?1\b', 'O1'), (r'\bS-?7\b', 'S7'),
        (r'\bH-?13\b', 'H13'), (r'\bCARBIDE\b', 'CARBIDE'),
        (r'\b440-?C\b', '440C'), (r'\b17-4\b', '17-4'),
        (r'\b4140\b', '4140'), (r'\b4340\b', '4340'),
        (r'\bVM\s?-?\s?15\s?-?\s?M?\b', 'VM-15M'),  # VM-15M, VM15M, VM-15, VM15, VM 15M, VM 15
    ]

    for pattern, normalized in materials:
        if re.search(pattern, text_upper):
            return normalized

    return None


def _filter_struck_out_text(text: str) -> str:
    """Filter out struck-out/crossed-out text from AutoCAD annotations.

    Detects and removes:
    - %%O...%%O wrapped text (AutoCAD overline toggle, commonly used for struck-out)
    - Text that contains strikethrough indicators

    Args:
        text: Raw text from drawing that may contain struck-out content

    Returns:
        Text with struck-out sections removed
    """
    if not text:
        return text

    # Remove %%O...%%O wrapped text (overline toggle in AutoCAD MTEXT)
    # Pattern: %%O<struck text>%%O or %%o<struck text>%%o
    result = re.sub(r'%%[Oo]([^%]*)%%[Oo]', '', text)

    # Also handle case where %%O appears without closing (treat rest as struck)
    # This handles "%%OTHIS SURFACE TO BE GROUND" where the whole line is struck
    if '%%O' in result.upper():
        # Remove everything from %%O to end of that line segment
        result = re.sub(r'%%[Oo][^\n]*', '', result)

    return result


def _has_active_grind_note(text_upper: str) -> bool:
    """Check if text contains an active (non-struck-out) grinding requirement.

    Looks for explicit grinding callouts like:
    - "TO BE GROUND"
    - "GRIND" (standalone or as part of grinding instruction)
    - "GROUND SURFACE"
    - "GRINDING REQUIRED"

    Args:
        text_upper: Uppercase text that has already had struck-out content removed

    Returns:
        True if an active grind callout is present
    """
    grind_patterns = [
        r'\bTO\s+BE\s+GROUND\b',
        r'\bGRIND\s+(?:THIS\s+)?SURFACE\b',
        r'\bGROUND\s+SURFACE\b',
        r'\bGRINDING\s+REQUIRED\b',
        r'\bSURFACE\s+(?:TO\s+BE\s+)?GROUND\b',
        r'\bGRIND\s+(?:TO|FOR|ALL)\b',  # "GRIND TO .0001", "GRIND FOR FINISH", "GRIND ALL"
    ]

    for pattern in grind_patterns:
        if re.search(pattern, text_upper):
            return True

    return False


def detect_punch_ops_features(text_dump: str) -> Dict[str, Any]:
    """Detect operations-driving features from text."""
    # First filter out any struck-out/crossed-out text
    filtered_text = _filter_struck_out_text(text_dump)
    text_upper = filtered_text.upper()

    features = {
        "num_chamfers": 0, "num_small_radii": 0, "has_3d_surface": False,
        "has_perp_face_grind": False, "form_complexity_level": 0,
    }

    chamfer_qty_pattern = r'\((\d+)\)\s*(?:0)?\.?\d+\s*X\s*45'
    for match in re.finditer(chamfer_qty_pattern, text_upper):
        features["num_chamfers"] += int(match.group(1))

    single_chamfer = r'(?:0)?\.?\d+\s*X\s*45'
    single_count = len(re.findall(single_chamfer, text_upper))
    if single_count > features["num_chamfers"]:
        features["num_chamfers"] = single_count

    small_radius_patterns = [r'R\s*(?:0)?\.00\d+', r'(?:0)?\.00\d+\s*R']
    for pattern in small_radius_patterns:
        features["num_small_radii"] += len(re.findall(pattern, text_upper))

    has_3d_indicators = ["POLISH CONTOUR", "POLISHED CONTOUR", "POLISH", "FORM", "COIN", "CONTOUR", "OVER R", "OVER-R"]
    if any(kw in text_upper for kw in has_3d_indicators):
        features["has_3d_surface"] = True

    # Check for face grinding requirements
    # Only trigger has_perp_face_grind if there's an actual grind callout
    # "PERPENDICULAR TO CENTERLINE" alone is NOT a grinding requirement - it's an orientation callout
    # Grinding is only required if there's an explicit grind note or if combined with grind terminology
    has_grind_note = _has_active_grind_note(text_upper)

    if has_grind_note:
        # There's an explicit grind callout - set face grind flag
        features["has_perp_face_grind"] = True
    elif "PERPENDICULAR" in text_upper or "PERP" in text_upper:
        # Only trigger grinding if perpendicular is combined with grind-related terms
        # e.g., "GRIND PERPENDICULAR" or "PERPENDICULAR GRIND"
        # But NOT "PERPENDICULAR TO CENTERLINE" which is just orientation
        perp_grind_patterns = [
            r'\bGRIND\s+PERPENDICULAR\b',
            r'\bPERPENDICULAR\s+GRIND\b',
            r'\bGROUND\s+PERPENDICULAR\b',
        ]
        if any(re.search(p, text_upper) for p in perp_grind_patterns):
            features["has_perp_face_grind"] = True
        # Note: "PERPENDICULAR TO CENTERLINE" does NOT trigger grinding

    radius_count = len(re.findall(r'R\s*(?:0)?\.?\d+', text_upper))
    diameter_count = len(re.findall(r'[Ø]|%%C', text_upper, re.IGNORECASE))
    total_form_features = radius_count + diameter_count

    if total_form_features > 10:
        features["form_complexity_level"] = 3
    elif total_form_features > 5:
        features["form_complexity_level"] = 2
    elif total_form_features > 2:
        features["form_complexity_level"] = 1

    return features


def detect_edge_break_operation(text_dump: str) -> bool:
    """Detect if edge break/deburr operation is required from text.

    Returns True if "BREAK ALL OUTSIDE SHARP CORNERS" or similar text is found.
    """
    if not text_dump:
        return False

    # First filter out any struck-out/crossed-out text
    filtered_text = _filter_struck_out_text(text_dump)
    text_upper = filtered_text.upper()

    # Patterns for edge break operations
    edge_break_patterns = [
        r'BREAK\s+ALL\s+OUTSIDE\s+SHARP\s+CORNERS',
        r'BREAK\s+ALL\s+SHARP\s+CORNERS',
        r'BREAK\s+ALL\s+OUTSIDE\s+CORNERS',
        r'BREAK\s+ALL\s+EDGES',
        r'BREAK\s+SHARP\s+CORNERS',
        r'DEBURR\s+ALL\s+EDGES',
        r'DEBURR\s+ALL\s+CORNERS',
    ]

    for pattern in edge_break_patterns:
        if re.search(pattern, text_upper):
            return True

    return False


def detect_etch_operation(text_dump: str) -> bool:
    """Detect if etching operation is required from text.

    Returns True if "ETCH ON DETAIL" or similar text is found.
    """
    if not text_dump:
        return False

    # First filter out any struck-out/crossed-out text
    filtered_text = _filter_struck_out_text(text_dump)
    text_upper = filtered_text.upper()

    # Patterns for etching operations (use word boundaries to avoid false positives)
    etch_patterns = [
        r'\bETCH\s+ON\s+DETAIL',
        r'\bETCH\s+DETAIL',
        r'\bETCH.*VENDOR.*DRAWING',
        r'\bETCH.*DRAWING.*NO',
        r'\bETCH\s+PART\s+NUMBER',
        r'\bETCH\s+P/?N',
        r'\bMARK\s+ON\s+DETAIL',
        r'\bLASER\s+ETCH',
        r'\bELECTRO\s*-?\s*ETCH',
    ]

    for pattern in etch_patterns:
        if re.search(pattern, text_upper):
            return True

    return False


def detect_polish_contour_operation(text_dump: str) -> bool:
    """Detect if polish contour operation is required from text.

    Returns True if "POLISH CONTOUR" or similar text is found.
    """
    if not text_dump:
        return False

    # First filter out any struck-out/crossed-out text
    filtered_text = _filter_struck_out_text(text_dump)
    text_upper = filtered_text.upper()

    # Patterns for polish contour operations
    polish_patterns = [
        r'\bPOLISH\s+CONTOUR',
        r'\bPOLISH\s+CONTOURED',
        r'\bPOLISHED\s+CONTOUR',
        r'\bCONTOUR\s+POLISH',
        r'\bPOLISH.*FORM',
        r'\bPOLISH.*RADIUS',
        r'\bPOLISH.*SURFACE',
    ]

    for pattern in polish_patterns:
        if re.search(pattern, text_upper):
            return True

    return False


def detect_waterjet_openings(text_dump: str) -> tuple[bool, float]:
    """Detect if waterjet cutting of openings is required from text.

    Returns:
        Tuple of (has_waterjet_openings, tolerance_plusminus)
        Example: (True, 0.005) for "WATERJET ALL OPENINGS ±.005"
    """
    if not text_dump:
        return False, 0.0

    # First filter out any struck-out/crossed-out text
    filtered_text = _filter_struck_out_text(text_dump)
    text_upper = filtered_text.upper()

    # Patterns for waterjet openings
    openings_patterns = [
        r'\bWATERJET\s+ALL\s+OPENINGS',
        r'\bWATERJET\s+OPENINGS',
        r'\bWATER\s+JET\s+ALL\s+OPENINGS',
        r'\bWATER\s+JET\s+OPENINGS',
    ]

    has_waterjet = False
    match_obj = None
    for pattern in openings_patterns:
        match_obj = re.search(pattern, text_upper)
        if match_obj:
            has_waterjet = True
            break

    if not has_waterjet:
        return False, 0.0

    # Extract tolerance from text (e.g., "±.005", "±0.005", "+/-.003")
    # Note: Also match � (Unicode replacement char) for when ± gets corrupted during DWG extraction
    # Search only near the waterjet keyword to avoid matching diameter symbols (∅)
    tolerance = 0.005  # Default tolerance
    search_start = max(0, match_obj.start())
    search_end = min(len(text_upper), match_obj.end() + 100)
    nearby_text = text_upper[search_start:search_end]
    tol_match = re.search(r'(?:^|[^R\d])[±+/-�]\s*0*\.(\d+)', nearby_text)
    if tol_match:
        # Convert matched digits to float (e.g., "005" -> 0.005, "003" -> 0.003)
        digits = tol_match.group(1)
        tolerance = float(f"0.{digits}")

    return True, tolerance


def detect_waterjet_profile(text_dump: str) -> tuple[bool, float]:
    """Detect if waterjet cutting of profile is required from text.

    Returns:
        Tuple of (has_waterjet_profile, tolerance_plusminus)
        Example: (True, 0.003) for "WATERJET TO ±.003"
    """
    if not text_dump:
        return False, 0.0

    # First filter out any struck-out/crossed-out text
    filtered_text = _filter_struck_out_text(text_dump)
    text_upper = filtered_text.upper()

    # Patterns for waterjet profile cutting
    profile_patterns = [
        r'\bWATERJET\s+TO\s+[±+/-]',
        r'\bWATER\s+JET\s+TO\s+[±+/-]',
        r'\bWATERJET\s+PROFILE',
        r'\bWATER\s+JET\s+PROFILE',
        r'\bWATERJET\s+CUT',
        r'\bWATER\s+JET\s+CUT',
    ]

    has_waterjet = False
    match_obj = None
    for pattern in profile_patterns:
        match_obj = re.search(pattern, text_upper)
        if match_obj:
            has_waterjet = True
            break

    if not has_waterjet:
        return False, 0.0

    # Extract tolerance from text (e.g., "±.003", "±0.003", "+/-.005")
    # Note: Also match � (Unicode replacement char) for when ± gets corrupted during DWG extraction
    # Search only near the waterjet keyword to avoid matching diameter symbols (∅)
    tolerance = 0.003  # Default tolerance for profile (typically tighter)
    search_start = max(0, match_obj.start())
    search_end = min(len(text_upper), match_obj.end() + 100)
    nearby_text = text_upper[search_start:search_end]
    tol_match = re.search(r'(?:^|[^R\d])[±+/-�]\s*0*\.(\d+)', nearby_text)
    if tol_match:
        # Convert matched digits to float (e.g., "003" -> 0.003, "005" -> 0.005)
        digits = tol_match.group(1)
        tolerance = float(f"0.{digits}")

    return True, tolerance


def detect_punch_pain_flags(text_dump: str) -> Dict[str, bool]:
    """Detect quality/pain flags from text."""
    text_upper = text_dump.upper()

    has_polish = any(kw in text_upper for kw in [
        "POLISH CONTOUR", "POLISH CONTOURED", "POLISHED", "POLISH TO", " POLISH "
    ])
    has_no_step = any(kw in text_upper for kw in [
        "NO STEP PERMITTED", "NO STEP", "NO STEPS", "NO-STEP"
    ])
    has_sharp = any(kw in text_upper for kw in ["SHARP EDGE", "SHARP EDGES", " SHARP "])

    has_gdt_font = "\\Famgdt" in text_dump or "\\FAMGDT" in text_upper
    has_gdt_symbols = bool(re.search(
        r'[⏥⌭⏄⌯⊕⌖]|GD&T|PERPENDICULARITY|FLATNESS|POSITION|CONCENTRICITY|RUNOUT|TIR',
        text_upper
    ))

    return {
        "has_polish_contour": has_polish,
        "has_no_step_permitted": has_no_step,
        "has_sharp_edges": has_sharp,
        "has_gdt": has_gdt_font or has_gdt_symbols,
    }


def parse_punch_holes_from_text(text_dump: str) -> Dict[str, Any]:
    """Parse hole and tap specifications from free text."""
    taps = []
    holes = []

    tap_pattern = r'(\d+/\d+-\d+)\s+TAP\s+X\s+([\d\.]+)\s+DEEP'
    for match in re.finditer(tap_pattern, text_dump, re.IGNORECASE):
        taps.append({"size": match.group(1), "depth_in": float(match.group(2))})

    tap_pattern_no_depth = r'(\d+/\d+-\d+)\s+TAP'
    for match in re.finditer(tap_pattern_no_depth, text_dump, re.IGNORECASE):
        size = match.group(1)
        if not any(t["size"] == size for t in taps):
            taps.append({"size": size, "depth_in": None})

    hole_thru_pattern = r'Ø\s*([\d\.]+)\s+THRU'
    for match in re.finditer(hole_thru_pattern, text_dump, re.IGNORECASE):
        holes.append({"diameter": float(match.group(1)), "depth_in": None, "thru": True})

    hole_depth_pattern = r'Ø\s*([\d\.]+)\s+X\s+([\d\.]+)\s+(?:DP|DEEP)'
    for match in re.finditer(hole_depth_pattern, text_dump, re.IGNORECASE):
        holes.append({"diameter": float(match.group(1)), "depth_in": float(match.group(2)), "thru": False})

    return {"tap_count": len(taps), "tap_summary": taps, "hole_count": len(holes), "hole_summary": holes}


def extract_punch_features_from_dxf(dxf_path: Path, text_dump: str) -> PunchFeatureSummary:
    """Main function to extract punch features from DXF + text."""
    summary = PunchFeatureSummary()
    warnings = []

    family, shape = classify_punch_family(text_dump)
    summary.family = family
    summary.shape_type = shape
    summary.material_callout = detect_punch_material(text_dump)

    geo_envelope = extract_geometry_envelope(dxf_path)
    if "error" in geo_envelope:
        warnings.append(f"Geometry extraction: {geo_envelope['error']}")

    dim_data = extract_punch_dimensions(dxf_path)
    if "error" in dim_data:
        warnings.append(f"Dimension extraction: {dim_data['error']}")

    geo_length = geo_envelope.get("overall_length_in", 0.0)
    geo_width = geo_envelope.get("overall_width_in", 0.0)

    MAX_REASONABLE_PUNCH_LENGTH = 12.0
    MAX_REASONABLE_PUNCH_OD = 3.0

    linear_dims = dim_data.get("linear_dims", [])
    diameter_dims = dim_data.get("diameter_dims", [])

    reasonable_linear = [d["measurement"] for d in linear_dims if 0 < d["measurement"] <= MAX_REASONABLE_PUNCH_LENGTH]
    dim_length = max(reasonable_linear) if reasonable_linear else 0.0

    reasonable_diameters = [d["measurement"] for d in diameter_dims if 0 < d["measurement"] <= MAX_REASONABLE_PUNCH_OD]
    dim_diameter = max(reasonable_diameters) if reasonable_diameters else 0.0

    # Geometry-based plate detection: Override text-based classification if geometry
    # suggests this is a plate (thin, rectangular, no significant axisymmetric features)
    if is_plate_geometry(geo_envelope, diameter_dims):
        # Override to use plate/insert template instead of punch turning template
        # Keep the original family name but mark as rectangular for plate processing
        summary.shape_type = "rectangular"
        # If classified as form_punch or round_punch based on text but geometry says plate,
        # use die_insert which gets plate processing
        if summary.family in ("form_punch", "round_punch"):
            summary.family = "die_insert"
        warnings.append(f"Geometry override: Detected plate geometry (L={geo_length:.3f}, W={geo_width:.3f}, T={geo_envelope.get('overall_height_in', 0):.3f}), using rectangular/plate template")

    def select_punch_dimension(geo_val, dim_val, max_reasonable):
        if 0 < dim_val <= max_reasonable:
            return dim_val
        if 0 < geo_val <= max_reasonable:
            return geo_val
        if geo_val > 0 and dim_val > 0:
            return min(geo_val, dim_val)
        return dim_val if dim_val > 0 else geo_val

    summary.overall_length_in = select_punch_dimension(geo_length, dim_length, MAX_REASONABLE_PUNCH_LENGTH)

    if summary.shape_type == "round":
        summary.max_od_or_width_in = select_punch_dimension(geo_width, dim_diameter, MAX_REASONABLE_PUNCH_OD)
    else:
        summary.max_od_or_width_in = select_punch_dimension(geo_width, dim_diameter, MAX_REASONABLE_PUNCH_OD)
        summary.body_width_in = summary.max_od_or_width_in
        summary.body_thickness_in = geo_envelope.get("overall_height_in")

    unit_factor = 1.0
    try:
        doc = _EZDXF.readfile(str(dxf_path))
        insunits = doc.header.get("$INSUNITS", 1)
        measurement = doc.header.get("$MEASUREMENT", 0)
        unit_factor = units_to_inch_factor(insunits)
        if measurement == 1 and insunits not in [4, 5, 6]:
            unit_factor = 1.0 / 25.4

        raw_diameters = _collect_diameters_from_dimensions(doc, unit_factor)
        clustered_diameters = cluster_values(raw_diameters, tolerance=0.0002)
        summary.num_ground_diams = min(len(clustered_diameters), 6)

        if summary.num_ground_diams == 0 and summary.max_od_or_width_in > 0:
            summary.num_ground_diams = 1
    except Exception:
        diameter_dims = dim_data.get("diameter_dims", [])
        unique_diameters = set(round(d["measurement"], 4) for d in diameter_dims)
        summary.num_ground_diams = len(unique_diameters) if unique_diameters else 1

    ground_fraction = 0.3 if summary.family == "form_punch" else (0.7 if summary.num_ground_diams > 2 else 0.5)
    summary.total_ground_length_in = summary.overall_length_in * ground_fraction

    # Calculate turning time model parameters for round parts
    if summary.shape_type == "round" and summary.overall_length_in > 0:
        # shoulder_count: number of diameter transitions (one less than number of diameters)
        summary.shoulder_count = max(0, summary.num_ground_diams - 1)

        # For round parts, estimate shank/pilot split based on number of diameters
        if summary.num_ground_diams >= 2:
            # Multi-diameter part: assume 60% shank, 40% pilot
            summary.shank_length = summary.overall_length_in * 0.60
            summary.pilot_length = summary.overall_length_in * 0.40
        elif summary.num_ground_diams == 1:
            # Single diameter: all shank, no pilot
            summary.shank_length = summary.overall_length_in
            summary.pilot_length = 0.0
        else:
            # No diameters specified: conservative split
            summary.shank_length = summary.overall_length_in * 0.70
            summary.pilot_length = summary.overall_length_in * 0.30

        # flange_thickness: default to 0 (could be enhanced with geometry analysis)
        summary.flange_thickness = 0.0

        # Calculate grinding time model parameters for round parts
        # Grinding typically covers the precision-ground sections (subset of overall length)
        if summary.total_ground_length_in > 0:
            # Split grinding length proportionally to turning split
            if summary.num_ground_diams >= 2:
                # Multi-diameter: grind both shank and pilot sections
                summary.grind_shank_len = summary.total_ground_length_in * 0.60
                summary.grind_pilot_len = summary.total_ground_length_in * 0.40
            elif summary.num_ground_diams == 1:
                # Single diameter: grind shank only
                summary.grind_shank_len = summary.total_ground_length_in
                summary.grind_pilot_len = 0.0
            else:
                # Conservative split
                summary.grind_shank_len = summary.total_ground_length_in * 0.70
                summary.grind_pilot_len = summary.total_ground_length_in * 0.30

        # grind_head_faces will be set after has_perp_face_grind is detected

    ops_features = detect_punch_ops_features(text_dump)
    summary.num_chamfers = ops_features["num_chamfers"]
    summary.num_small_radii = ops_features["num_small_radii"]
    summary.has_3d_surface = ops_features["has_3d_surface"]
    summary.has_perp_face_grind = ops_features["has_perp_face_grind"]
    summary.form_complexity_level = ops_features["form_complexity_level"]

    # Set grind_head_faces based on perpendicular face grinding requirements
    if summary.shape_type == "round" and summary.has_perp_face_grind:
        summary.grind_head_faces = 2  # Typically top and bottom faces

    pain_flags = detect_punch_pain_flags(text_dump)
    summary.has_polish_contour = pain_flags["has_polish_contour"]
    summary.has_no_step_permitted = pain_flags["has_no_step_permitted"]
    summary.has_sharp_edges = pain_flags["has_sharp_edges"]
    summary.has_gdt = pain_flags["has_gdt"]

    # Detect etch operation requirement
    summary.has_etch = detect_etch_operation(text_dump)

    hole_data = parse_punch_holes_from_text(text_dump)
    summary.tap_count = hole_data["tap_count"]
    summary.tap_summary = hole_data["tap_summary"]

    summary.warnings = warnings
    confidence = 0.7
    if summary.overall_length_in > 0:
        confidence += 0.1
    if summary.max_od_or_width_in > 0:
        confidence += 0.1
    if summary.material_callout:
        confidence += 0.05
    summary.confidence_score = min(1.0, confidence)

    return summary


def extract_punch_features(dxf_path: str | Path, text_lines: Optional[List[str]] = None) -> PunchFeatureSummary:
    """Convenience function for extracting punch features."""
    dxf_path = Path(dxf_path)

    if text_lines is None:
        try:
            doc = _EZDXF.readfile(str(dxf_path)) if _EZDXF else None
            if doc:
                text_records = list(collect_all_text(doc))
                text_lines = [rec["text"] for rec in text_records if rec.get("text")]
            else:
                text_lines = []
        except Exception:
            text_lines = []

    text_dump = "\n".join(text_lines)
    return extract_punch_features_from_dxf(dxf_path, text_dump)


__all__ = [
    "detect_units_scale",
    "iter_spaces",
    "iter_table_text",
    "harvest_plate_dimensions",
    "harvest_outline_metrics",
    "harvest_hole_geometry",
    "harvest_hole_table",
    "hole_ops_to_drill_bins",
    "harvest_title_notes",
    "iter_table_entities",
    "build_geo_from_doc",
    "build_geo_from_dxf",
    # Punch extraction
    "PunchFeatureSummary",
    "extract_punch_features",
    "extract_punch_features_from_dxf",
    "extract_geometry_envelope",
    "extract_punch_dimensions",
    "classify_punch_family",
    "is_plate_geometry",
    "detect_punch_material",
    "detect_punch_ops_features",
    "detect_punch_pain_flags",
    "parse_punch_holes_from_text",
    "normalize_acad_mtext",
    "units_to_inch_factor",
    "cluster_values",
]
