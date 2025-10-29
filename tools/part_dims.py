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


_THK_TOKENS = r'(?:THK|T(?:H(?:I(?:C(?:K(?:NESS)?)?)?)?)?)[\s:=]*'
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


def _should_include_layer(layer: str, include: Optional[Iterable[str]], exclude: Optional[Iterable[str]]) -> bool:
    name = layer or ""
    lname = name.lower()

    if include:
        include_lower = {value.lower() for value in include}
        if lname not in include_lower:
            return False

    if exclude:
        exclude_lower = {value.lower() for value in exclude}
        if lname in exclude_lower:
            return False

    undesirable = ("title" in lname) or ("border" in lname) or ("frame" in lname) or ("sheet" in lname)
    if undesirable and (not include or lname not in {value.lower() for value in include}):
        return False

    return True


_AABB_ALLOWED = {
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

_AABB_EXCLUDED = {"DIMENSION", "TEXT", "MTEXT", "TABLE", "HATCH"}


def _extend_bbox(bbox, entity) -> None:
    try:
        entity_bbox = entity.bbox()
    except Exception:
        entity_bbox = None

    if entity_bbox is None:
        return

    extmin = getattr(entity_bbox, "extmin", None)
    extmax = getattr(entity_bbox, "extmax", None)

    if extmin is not None:
        bbox.extend(extmin)
    if extmax is not None:
        bbox.extend(extmax)


def _aabb_size(
    msp,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        import ezdxf
        from ezdxf import EzdxfError  # type: ignore[attr-defined]
        from ezdxf.math import BoundingBox
    except Exception:  # pragma: no cover - ezdxf missing at runtime
        return None, None, None

    bbox = BoundingBox()

    include = list(include) if include else None
    exclude = list(exclude) if exclude else None

    for entity in msp:
        dxftype = entity.dxftype()
        if dxftype in _AABB_EXCLUDED:
            continue
        if dxftype not in _AABB_ALLOWED:
            continue
        if not _should_include_layer(getattr(entity.dxf, "layer", ""), include, exclude):
            continue

        if dxftype == "INSERT":
            try:
                for sub_entity in entity.virtual_entities():
                    _extend_bbox(bbox, sub_entity)
            except Exception:
                continue
        else:
            _extend_bbox(bbox, entity)

    if not bbox.has_data:
        print("[part-dims] AABB: no geometry found")
        return None, None, None

    size = bbox.size
    dx = float(size.x)
    dy = float(size.y)
    dz = float(size.z)
    print(f"[part-dims] AABB size: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
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

