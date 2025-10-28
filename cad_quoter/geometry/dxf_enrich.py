"""Shared DXF enrichment helpers used by the CLI and GUI flows."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from cad_quoter.vendors import ezdxf as _ezdxf_vendor

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

RE_MAT = re.compile(r"\b(MATERIAL|MAT)\b[:\s]*([A-Z0-9\-\s/\.]+)")
RE_COAT = re.compile(
    r"\b(ANODIZE|BLACK OXIDE|ZINC PLATE|NICKEL PLATE|PASSIVATE|HEAT TREAT|DLC|PVD|CVD)\b",
    re.I,
)
RE_TOL = re.compile(r"\bUNLESS OTHERWISE SPECIFIED\b.*?([±\+\-]\s*\d+\.\d+)", re.I | re.S)
RE_REV = re.compile(r"\bREV(ISION)?\b[:\s]*([A-Z0-9\-]+)")


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


def iter_table_text(doc: Any) -> Iterator[str]:
    """Yield text strings discovered inside TABLE and TEXT entities."""

    for table in iter_table_entities(doc):
        try:
            n_rows = int(getattr(table.dxf, "n_rows", 0))
            n_cols = int(getattr(table.dxf, "n_cols", 0))
        except Exception:
            n_rows = 0
            n_cols = 0
        if n_rows <= 0 or n_cols <= 0:
            continue
        try:
            for row_idx in range(n_rows):
                row: list[str] = []
                for col_idx in range(n_cols):
                    try:
                        cell = table.get_cell(row_idx, col_idx)
                    except Exception:
                        cell = None
                    if cell is None:
                        row.append("")
                        continue
                    try:
                        text = cell.get_text()
                    except Exception:
                        text = ""
                    row.append(text or "")
                line = " | ".join(fragment.strip() for fragment in row if fragment)
                if line:
                    yield line
        except Exception:
            continue

    for space in iter_spaces(doc):
        try:
            entities = space.query("MTEXT,TEXT")
        except Exception:
            entities = []
        for entity in entities:
            try:
                if entity.dxftype() == "MTEXT":
                    text = entity.plain_text()
                else:
                    text = entity.dxf.text
            except Exception:
                text = None
            if text:
                yield str(text)


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


def harvest_hole_table(doc: Any) -> Dict[str, Any]:
    taps = cbore = csk = 0
    deepest = 0.0
    from_back = False
    lines: List[str] = []
    family_guess: Counter[float] = Counter()

    try:
        chart_debug_records = _collect_chart_layout_records(doc)
    except Exception:
        chart_debug_records = []
    try:
        _write_chart_layout_debug(chart_debug_records)
    except Exception:
        pass

    for raw in iter_table_text(doc):
        upper = raw.upper()
        if not any(keyword in upper for keyword in ("HOLE", "TAP", "THRU", "CBORE", "C'BORE", "DRILL", "Ø", "⌀")):
            continue
        lines.append(raw)

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

    return {
        "tap_qty": taps,
        "cbore_qty": cbore,
        "csk_qty": csk,
        "deepest_hole_in": deepest or None,
        "holes_from_back": bool(from_back),
        "hole_table_families_in": dict(family_guess.most_common()) if family_guess else None,
        "chart_lines": lines,
        "prov": "HOLE TABLE / TEXT",
    }


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

    return {
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
            "hole_table": hole_table.get("prov"),
            "material": title.get("prov"),
        },
    }


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
    "harvest_title_notes",
    "iter_table_entities",
    "build_geo_from_doc",
    "build_geo_from_dxf",
]
