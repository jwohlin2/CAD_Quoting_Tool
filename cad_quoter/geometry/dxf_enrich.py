"""Shared DXF enrichment helpers used by the CLI and GUI flows."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from cad_quoter.geo_extractor import collect_all_text

from cad_quoter.vendors import ezdxf as _ezdxf_vendor

try:  # pragma: no cover - optional helper available during packaging
    from tools.hole_ops import explode_rows_to_operations
except Exception:  # pragma: no cover - import fallback when tools/ is not a package
    try:
        from hole_ops import explode_rows_to_operations  # type: ignore
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
]
