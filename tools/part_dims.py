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


_THK_PATTERN_SUFFIX = r"(?P<num>(?:\d+\s*-\s*)?\d+(?:/\d+)?|\d*\.\d+|\.\d+)"
_THK_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(rf"(?:THK|T|Thickness)\s*[:=]?\s*{_THK_PATTERN_SUFFIX}", re.IGNORECASE),
    re.compile(rf"{_THK_PATTERN_SUFFIX}\s*(?:THK|T)\b", re.IGNORECASE),
)


def _parse_fraction(token: str) -> Optional[float]:
    """Convert decimal, fraction (13/32) or mixed number (1-1/2) to float."""

    token = token.strip().replace(" ", "")
    if not token:
        return None

    try:
        return float(token)
    except ValueError:
        pass

    if "-" in token:
        whole, _, frac = token.partition("-")
        try:
            whole_val = float(whole)
        except ValueError:
            return None
        frac_val = _parse_fraction(frac)
        if frac_val is None:
            return None
        return whole_val + frac_val

    if "/" in token:
        num, _, den = token.partition("/")
        try:
            return float(int(num)) / float(int(den))
        except (ValueError, ZeroDivisionError):
            return None

    return None


def _parse_thickness_from_text(lines: List[str]) -> Optional[float]:
    """Extract a stock thickness value from free-form text lines."""

    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        for pattern in _THK_PATTERNS:
            for match in pattern.finditer(text):
                token = match.group("num")
                if not token:
                    continue
                token = token.strip().rstrip(",.;")
                value = _parse_fraction(token)
                if value is not None and value > 0:
                    print(f"[part-dims] thickness from text: {value:.4f} in (line='{text}')")
                    return value
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


def _max_ordinate_xy(msp) -> Tuple[Optional[float], Optional[float], int, int]:
    try:
        import ezdxf
        from ezdxf import EzdxfError  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - ezdxf missing at runtime
        return None, None, 0, 0

    max_x: Optional[float] = None
    max_y: Optional[float] = None
    count_x = 0
    count_y = 0

    for dim in msp.query("DIMENSION"):
        try:
            is_ordinate = bool(getattr(dim, "is_ordinate", False))
            if not is_ordinate:
                dimtype = getattr(dim.dxf, "dimtype", 0)
                is_ordinate = (dimtype & 0x07) == 6
            if not is_ordinate:
                continue

            ord_type = getattr(dim.dxf, "ordtype", None)
            if ord_type not in (0, 1):
                # try alternate helpers if provided by ezdxf
                if hasattr(dim, "is_x_ordinate") and dim.is_x_ordinate:  # type: ignore[attr-defined]
                    ord_type = 0
                elif hasattr(dim, "is_y_ordinate") and dim.is_y_ordinate:  # type: ignore[attr-defined]
                    ord_type = 1
                else:
                    continue

            try:
                measurement = dim.get_measurement()
            except Exception:
                measurement = None

            if measurement is None:
                # fall back to rendered text
                try:
                    text = dim.plain_text()
                except Exception:
                    text = ""
                match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", text)
                measurement = float(match.group()) if match else None

            if measurement is None:
                continue

            measurement = float(measurement)
            if measurement < 0:
                measurement = abs(measurement)

            if ord_type == 0:
                count_x += 1
                max_x = measurement if max_x is None else max(max_x, measurement)
            else:
                count_y += 1
                max_y = measurement if max_y is None else max(max_y, measurement)
        except Exception:  # pragma: no cover - ignore malformed entities
            continue

    print(f"[part-dims] ordinate dimensions: X={count_x}, Y={count_y}")
    return max_x, max_y, count_x, count_y


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

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    unit_factor = _insunits_to_inch_factor(doc)

    # 1) ordinate dimensions first
    max_x, max_y, count_x, count_y = _max_ordinate_xy(msp)
    length_in: Optional[float] = None
    width_in: Optional[float] = None
    length_source: Optional[str] = None

    if max_x is not None and max_y is not None:
        dims = sorted([max_x * unit_factor, max_y * unit_factor], reverse=True)
        length_in, width_in = dims[0], dims[1]
        length_source = "dimensions"
        print(
            f"[part-dims] using ordinate dimensions: L={length_in:.4f} in, W={width_in:.4f} in"
        )

    # 2) AABB fallback
    dz: Optional[float] = None
    if length_in is None or width_in is None:
        dx, dy, dz = _aabb_size(
            msp, include=layer_include, exclude=layer_exclude
        )
        if dx is not None and dy is not None:
            dims = sorted([dx * unit_factor, dy * unit_factor], reverse=True)
            length_in, width_in = dims[0], dims[1]
            if length_source is None:
                length_source = "aabb"
                print(
                    f"[part-dims] using AABB dimensions: L={length_in:.4f} in, W={width_in:.4f} in"
                )

    # 3) thickness
    thickness_source: Optional[str] = None
    thickness_in: Optional[float] = None
    text_lines: List[str] = []
    if text_csv or text_jsonl:
        text_lines = _read_texts(text_csv, text_jsonl)
    if text_lines:
        thickness_in = _parse_thickness_from_text(text_lines)
        if thickness_in is not None:
            thickness_source = "text"

    if thickness_in is None and dz:
        thickness_val = dz * unit_factor
        if 0.05 <= thickness_val <= 20.0:
            thickness_in = thickness_val
            thickness_source = "aabb"
            print(f"[part-dims] thickness from geometry: {thickness_in:.4f} in")

    # Determine overall source label
    source: str
    if length_source and thickness_source and length_source != thickness_source:
        source = "mixed"
    elif thickness_source and not length_source:
        source = thickness_source
    elif length_source:
        source = length_source
    elif thickness_source:
        source = thickness_source
    else:
        source = "none"

    result = {
        "length_in": length_in,
        "width_in": width_in,
        "thickness_in": thickness_in,
        "source": source,
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

