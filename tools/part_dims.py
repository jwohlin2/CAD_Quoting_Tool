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
        return float(s.strip().replace(",", ""))
    except Exception:
        return None


def _max_ordinate_xy(msp) -> Tuple[Optional[float], Optional[float]]:
    """
    Try to read X/Y-ordinate values from DIMENSION entities.
    Works without ezdxf.addons; if we can't confidently detect axis, we return (None, None)
    so the caller will fall back to the AABB path.
    """
    max_x = None
    max_y = None
    for dim in msp.query("DIMENSION"):
        # Ordinate test: bit 64 set in dimtype
        try:
            dimtype = int(dim.dxf.dimtype)
        except Exception:
            continue
        is_ordinate = bool(dimtype & 64)
        if not is_ordinate:
            continue

        # Try to get the displayed numeric value
        val = None
        txt = (dim.dxf.text or "").strip()
        if txt and txt != "<>":
            m = re.search(r'([-+]?\d+(?:\.\d+)?)', txt)
            if m:
                try:
                    val = float(m.group(1))
                except Exception:
                    val = None
        if val is None:
            try:
                # some ezdxf versions support this for ordinate too
                val = float(dim.get_measurement())
            except Exception:
                pass
        if val is None:
            # Last resort: scan virtual entities for TEXT/MTEXT showing the number
            try:
                for e in dim.virtual_entities():
                    if e.dxftype() in ("TEXT", "MTEXT"):
                        s = e.dxf.text if e.dxftype() == "TEXT" else e.plain_text()
                        m2 = re.search(r'([-+]?\d+(?:\.\d+)?)', s or "")
                        if m2:
                            try:
                                val = float(m2.group(1))
                                break
                            except Exception:
                                pass
            except Exception:
                pass
        if val is None:
            continue

        # Axis detection without addons: many drawings store 0=X,1=Y in AZIN (AM_ ordinate)
        axis_flag = getattr(dim.dxf, "azin", None)  # 0 = X-ordinate, 1 = Y-ordinate (if present)
        if axis_flag == 0:
            max_x = val if max_x is None else max(max_x, val)
        elif axis_flag == 1:
            max_y = val if max_y is None else max(max_y, val)
        else:
            # Unknown axis -> give up gracefully; let AABB handle it
            return None, None
    return max_x, max_y


def _aabb_size(
    msp,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute AABB over likely 'part' geometry.
    - Includes LINE, LWPOLYLINE, POLYLINE, ARC, CIRCLE, ELLIPSE, SPLINE, SOLID, TRACE, INSERT (expanded)
    - Excludes DIMENSION, TEXT, MTEXT, TABLE, HATCH by default (can be tuned)
    - Honors layer include/exclude filters if provided
    """
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

    bbox = BoundingBox()

    def _layer_ok(ent) -> bool:
        layer = (ent.dxf.layer or "").upper()
        if include:
            if all(layer.upper() != pat.upper() for pat in include):
                return False
        if exclude:
            if any(layer.upper() == pat.upper() for pat in exclude):
                return False
        # heuristic: skip common annotation layers if not explicitly included
        if not include and layer.startswith(("AM_", "DEFPOINTS", "DIM", "ANNOT", "TITLE", "BORDER", "FRAME")):
            return False
        return True

    # 1) add entities
    for e in msp:
        kind = e.dxftype()
        if kind in anno_types:
            continue
        if kind not in geom_types:
            continue
        if not _layer_ok(e):
            continue
        try:
            if kind == "INSERT":
                # expand block references
                for ve in e.virtual_entities():
                    if _layer_ok(ve):
                        bbox.extend(ve.bbox())  # ezdxf ≥ 1.0
            else:
                bbox.extend(e.bbox())
        except Exception:
            # last resort: accumulate explicit vertex points if available
            try:
                bbox.extend(list(e.vertices()))
            except Exception:
                pass

    if not bbox.has_data:
        return None, None, None

    (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox.extmin, bbox.extmax
    dx, dy, dz = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    # guard against garbage zeros
    dx = dx if dx > 1e-6 else None
    dy = dy if dy > 1e-6 else None
    dz = dz if dz > 1e-6 else None
    return dx, dy, dz


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
    L: Optional[float] = None
    W: Optional[float] = None
    source: Optional[str] = None

    if ox is not None and oy is not None:
        L, W = sorted([ox * f, oy * f], reverse=True)
        source = "dimensions"
        print(f"[part-dims] using ordinate dimensions: L={L:.4f} in, W={W:.4f} in")

    dx = dy = dz = None

    if L is None or W is None:
        dx, dy, dz = _aabb_size(msp, include=layer_include, exclude=layer_exclude)
        if dx and dy:
            L, W = sorted([dx * f, dy * f], reverse=True)
            source = source or "aabb"
            print(f"[part-dims] using AABB dimensions: L={L:.4f} in, W={W:.4f} in")

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

