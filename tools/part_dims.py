"""Utilities for inferring stock dimensions from DXF drawings."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _read_texts(
    csv_path: Optional[str] = None, jsonl_path: Optional[str] = None
) -> List[str]:
    """Read raw text strings from a CSV or JSONL text dump."""

    lines: List[str] = []

    if csv_path:
        csv_file = Path(csv_path)
        if csv_file.is_file():
            try:
                import csv

                with csv_file.open(newline="", encoding="utf-8", errors="ignore") as fh:
                    reader = csv.reader(fh)
                    for row in reader:
                        if not row:
                            continue
                        if len(row) >= 4:
                            lines.append(row[3])
                        else:
                            lines.append(row[-1])
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[part-dims] failed to read CSV texts: {exc}")
        else:
            print(f"[part-dims] CSV not found: {csv_file}")

    if jsonl_path:
        jsonl_file = Path(jsonl_path)
        if jsonl_file.is_file():
            try:
                import json

                with jsonl_file.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = payload.get("text")
                        if isinstance(text, str):
                            lines.append(text)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[part-dims] failed to read JSONL texts: {exc}")
        else:
            print(f"[part-dims] JSONL not found: {jsonl_file}")

    return lines


_THK_TOKENS = r'(?:THK|T(?:H(?:I(?:C(?:K(?:NESS)?)?)?)?)?)\s*[:=]?\s*'
_NUM = r'(?:\d+(?:\.\d+)?|\d+\s*-\s*\d+\/\d+|\d+\/\d+)'

RE_THICKNESS_LINE = re.compile(
    rf'\b{_THK_TOKENS}(?P<t>{_NUM})(?:\s*(?P<u>in(?:ch(?:es)?)?|"))?\b',
    re.IGNORECASE
)


def _to_float_in(token: str) -> Optional[float]:
    """Convert supported numeric token forms to inches."""

    s = token.strip().lower().replace('"', '')
    if not s:
        return None

    mixed = re.match(r'^(\d+)\s*-\s*(\d+)\s*/\s*(\d+)$', s)
    if mixed:
        whole, num, den = map(int, mixed.groups())
        if den == 0:
            return None
        return whole + (num / den)

    frac = re.match(r'^(\d+)\s*/\s*(\d+)$', s)
    if frac:
        num, den = map(int, frac.groups())
        if den == 0:
            return None
        return num / den

    try:
        return float(s)
    except ValueError:
        return None


def _parse_thickness_from_text(lines: List[str]) -> Optional[float]:
    """Only accept thickness explicitly labeled (THK, T=, THICKNESS)."""

    for raw in lines:
        s = raw.strip()
        m = RE_THICKNESS_LINE.search(s)   # require keyword -> prevents 'SHEET 2'
        if not m:
            continue
        t = _to_float_in(m.group('t'))
        if t is not None:
            return t
    return None


_INSUNITS_TO_IN = {
    0: 1.0,  # unitless -> assume inch
    1: 1.0,  # inches
    2: 12.0,  # feet
    3: 63360.0,  # miles
    4: 1.0 / 25.4,  # millimeters
    5: 1.0 / 2.54,  # centimeters
    6: 39.37007874015748,  # meters
    7: 39370.07874015748,  # kilometers
    8: 3.937007874015748e-05,  # microinches
    9: 3.937007874015748,  # decimeters
    10: 39.37007874015748,  # decameters
    11: 393.7007874015748,  # hectometers
    12: 3937.007874015748,  # gigameters? (generic)
    13: 1550.0031000062,  # astronomical units
    14: 1.0e-08,  # nanometers
    15: 1.0e-05,  # microns
    16: 0.001,  # millimeters? (decimicrons) -> fallback
    17: 1.0e-10,  # angstroms
    18: 39.37007874015748,  # nanometers? keep fallback
    19: 15748031.496062992,  # parsecs -> not realistic but included
}


def _insunits_to_inch_factor(doc) -> float:
    unit_code = int(doc.header.get("$INSUNITS", 1))
    factor = _INSUNITS_TO_IN.get(unit_code)
    if factor is None:
        # assume inches for unknown values
        factor = 1.0
    return float(factor)


def _float_from_text(s: str) -> Optional[float]:
    try:
        return float(str(s).strip().replace(",", ""))
    except Exception:
        return None


def _max_ordinate_xy(msp) -> Tuple[Optional[float], Optional[float]]:
    """
    Get max X and max Y from ORDinate DIMENSIONs.
    Strategy:
      - Identify ORDinate by bit 64 in dim.dxf.dimtype.
      - Axis hint: many CADs store X/Y axis in dxf.azin (0=X, 1=Y). If missing, we guess by the leader vector.
      - Value: prefer dim.dxf.text (if numeric), else dim.get_measurement(), else scan virtual_entities() for TEXT/MTEXT.
    Returns (max_x, max_y) in drawing units, or (None, None) if not found.
    """
    max_x = None
    max_y = None

    for dim in msp.query("DIMENSION"):
        try:
            dt = int(dim.dxf.dimtype)
        except Exception:
            continue

        # ORDinate bit is 64
        if not (dt & 64):
            continue

        # axis: 0 = X-ordinate, 1 = Y-ordinate (AM/AutoCAD stores this in AZIN sometimes)
        axis = getattr(dim.dxf, "azin", None)  # may not exist

        # value from explicit text?
        txt = (dim.dxf.text or "").strip()
        val = None
        if txt and txt != "<>":
            m = re.search(r"([-+]?\d+(?:\.\d+)?)", txt)
            if m:
                val = _float_from_text(m.group(1))

        # try measurement
        if val is None:
            try:
                val = float(dim.get_measurement())
            except Exception:
                val = None

        # last resort: look inside the rendered/virtual block
        if val is None:
            try:
                for e in dim.virtual_entities():
                    if e.dxftype() in ("TEXT", "MTEXT"):
                        s = e.dxf.text if e.dxftype() == "TEXT" else e.plain_text()
                        m2 = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
                        if m2:
                            val = _float_from_text(m2.group(1))
                            break
            except Exception:
                pass

        if val is None:
            continue

        # If axis unknown, try to infer: compare the absolute of extension vector components if available
        if axis is None:
            try:
                # tip: many ORD dims use defpoint/defpoint2; the bigger delta usually indicates the axis
                p1 = dim.dxf.defpoint
                p2 = getattr(dim.dxf, "defpoint2", None)
                if p2:
                    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
                    axis = 0 if dx >= dy else 1
            except Exception:
                axis = 0  # default to X if we can’t tell

        if axis == 0:
            max_x = val if max_x is None else max(max_x, val)
        else:
            max_y = val if max_y is None else max(max_y, val)

    return max_x, max_y


def _max_linear_hw(msp) -> tuple[Optional[float], Optional[float]]:
    """
    Scan non-ordinate DIMENSIONs and return (max_horizontal, max_vertical) in drawing units.
    Heuristics:
      - Use dim.get_measurement() when available.
      - Determine orientation by angle when possible; otherwise by extents of defpoints.
      - Ignore obviously tiny notes (< 0.05 in) to avoid arrows and small callouts.
    """

    max_h: Optional[float] = None
    max_v: Optional[float] = None

    for dim in msp.query("DIMENSION"):
        try:
            dt = int(dim.dxf.dimtype)
        except Exception:
            continue

        if dt & 64:  # skip ordinates here
            continue

        val: Optional[float] = None
        try:
            val = float(dim.get_measurement())
        except Exception:
            pass

        if val is None:
            txt = (dim.dxf.text or "").strip()
            if txt and txt != "<>":
                m = re.search(r"([-+]?\d+(?:\.\d+)?)", txt)
                if m:
                    try:
                        val = float(m.group(1))
                    except Exception:
                        pass

        if val is None or val < 0.05:
            continue

        angle = getattr(dim.dxf, "angle", None)
        orient: Optional[str] = None  # 'H' or 'V'

        if angle is not None:
            try:
                a = abs(float(angle)) % 180.0
                orient = "H" if a <= 45 or a >= 135 else "V"
            except Exception:
                orient = None
        else:
            try:
                p1 = dim.dxf.defpoint
                p2 = getattr(dim.dxf, "defpoint2", None)
                if p2:
                    dx = abs(p2[0] - p1[0])
                    dy = abs(p2[1] - p1[1])
                    orient = "H" if dx >= dy else "V"
            except Exception:
                orient = None

        if orient == "H":
            max_h = val if max_h is None else max(max_h, val)
        elif orient == "V":
            max_v = val if max_v is None else max(max_v, val)
        else:
            max_h = val if max_h is None else max(max_h, val)
            max_v = val if max_v is None else max(max_v, val)

    return max_h, max_v


def _aabb_size(msp, include=None, exclude=None):
    from ezdxf.math import BoundingBox

    geom_types = {
        "LINE",
        "LWPOLYLINE",
        "POLYLINE",
        "ARC",
        "CIRCLE",
        "ELLIPSE",
        "SPLINE",
        "SOLID",
        "TRACE",
        "INSERT",
    }
    anno_types = {"DIMENSION", "TEXT", "MTEXT", "TABLE", "HATCH"}

    def layer_ok(ent) -> bool:
        layer = (ent.dxf.layer or "").upper()
        if include:
            if layer.upper() not in {s.upper() for s in include}:
                return False
        if exclude and layer.upper() in {s.upper() for s in exclude}:
            return False
        # heuristic: skip common annotation layers unless explicitly included
        if not include and layer.startswith(("AM_", "DEFPOINTS", "DIM", "ANNOT", "TITLE", "BORDER", "FRAME")):
            return False
        return True

    bb = BoundingBox()
    for e in msp:
        kind = e.dxftype()
        if kind in anno_types or kind not in geom_types:
            continue
        if not layer_ok(e):
            continue
        try:
            if kind == "INSERT":
                for ve in e.virtual_entities():
                    if layer_ok(ve):
                        bb.extend(ve.bbox())
            else:
                bb.extend(e.bbox())
        except Exception:
            try:
                bb.extend(list(e.vertices()))
            except Exception:
                pass

    if not bb.has_data:
        return None, None, None

    (xmin, ymin, zmin), (xmax, ymax, zmax) = bb.extmin, bb.extmax
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return (
        dx if dx > 1e-6 else None,
        dy if dy > 1e-6 else None,
        dz if dz > 1e-6 else None,
    )


def infer_part_dims(
    dxf_path: str,
    text_csv: Optional[str] = None,
    text_jsonl: Optional[str] = None,
    layer_include: Optional[List[str]] = None,
    layer_exclude: Optional[List[str]] = None,
) -> Dict[str, Optional[float]]:
    import ezdxf
    from ezdxf.math import BoundingBox  # noqa: F401 - ensure ezdxf dependency available

    input_path = Path(dxf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"DXF file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Expected a DXF file, but got a directory: {input_path}")

    doc = ezdxf.readfile(str(input_path))
    msp = doc.modelspace()
    f = _insunits_to_inch_factor(doc)

    # dimensions first
    ox = oy = None
    try:
        ox, oy = _max_ordinate_xy(msp)
    except Exception as e:
        print(f"[part-dims] ordinate read failed: {e}")
    lh = lv = None
    try:
        lh, lv = _max_linear_hw(msp)
    except Exception as e:
        print(f"[part-dims] linear read failed: {e}")
    dx = dy = dz = None
    try:
        dx, dy, dz = _aabb_size(msp, include=layer_include, exclude=layer_exclude)
    except Exception as e:
        print(f"[part-dims] aabb read failed: {e}")

    source: Optional[str] = None

    candidates_w = [v for v in [ox, lh, dx] if v is not None]
    candidates_h = [v for v in [oy, lv, dy] if v is not None]

    width_units = max(candidates_w) if candidates_w else None
    height_units = max(candidates_h) if candidates_h else None

    L: Optional[float] = None
    W: Optional[float] = None

    if width_units is not None and height_units is not None:
        width_in = width_units * f
        height_in = height_units * f
        L, W = (height_in, width_in) if height_in >= width_in else (width_in, height_in)
        parts = [
            ("ord", (ox is not None or oy is not None)),
            ("lin", (lh is not None or lv is not None)),
            ("aabb", (dx is not None or dy is not None)),
        ]
        source = "+".join(s for s, ok in parts if ok) or None
        src_label = source or "unknown"
        print(f"[part-dims] fused dimensions: L={L:.4f} in, W={W:.4f} in from {src_label}")
    else:
        L = W = None

    # thickness after text parse (with strict keywords)
    T: Optional[float] = None
    lines = _read_texts(text_csv, text_jsonl) if (text_csv or text_jsonl) else []
    if lines:
        T = _parse_thickness_from_text(lines)
        if T is not None:
            print(f"[part-dims] thickness from text: {T:.4f} in")
    if T is None and dz:
        T = dz * f
        source = source or "aabb"
        print(f"[part-dims] thickness from geometry: {T:.4f} in")

    result = {
        "length_in": L,
        "width_in": W,
        "thickness_in": T,
        "source": source or "none",
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Infer stock dimensions from DXF files")
    parser.add_argument("--dxf", required=True, help="Path to the DXF drawing")
    parser.add_argument("--csv", help="CSV file produced by the DXF text dump tool")
    parser.add_argument("--jsonl", help="JSONL text dump with a 'text' field")
    parser.add_argument("--include", nargs="*", help="Layer names to include (AABB)")
    parser.add_argument("--exclude", nargs="*", help="Layer names to exclude (AABB)")

    args = parser.parse_args(argv)

    result = infer_part_dims(
        args.dxf,
        text_csv=args.csv,
        text_jsonl=args.jsonl,
        layer_include=args.include,
        layer_exclude=args.exclude,
    )

    print(json.dumps(result, indent=2))

    output_base = Path(args.csv or args.dxf)
    output_path = output_base.with_name("stock_dims.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[part-dims] -> {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

