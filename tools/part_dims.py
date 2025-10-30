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
    Return (Hmax, Vmax) for LINEAR/ALIGNED dimensions only.
    Orientation priority:
      1) explicit rotation angle (0/180 ~= horizontal, 90 ~= vertical)
      2) vector from defpoint -> defpoint2 (larger delta axis wins)
      3) dimension line direction inferred from virtual_entities()
    """

    max_h = None
    max_v = None

    for dim in msp.query("DIMENSION"):
        try:
            dt = int(dim.dxf.dimtype)
        except Exception:
            continue

        base = dt & 7              # 0=linear(rotated), 1=aligned, 2=angular, 3=diameter, 4=radius, 5=angular3pt, 6=ordinate
        if base not in (0, 1):     # accept ONLY linear/aligned
            continue

        # value
        val = None
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

        # orientation: try rotation angle first
        orient: Optional[str] = None
        angle = getattr(dim.dxf, "angle", None)
        if angle is not None:
            a = abs(float(angle)) % 180.0
            orient = "H" if a <= 45.0 or a >= 135.0 else "V"

        # if still unknown, compare defpoints
        if orient is None:
            try:
                p1 = dim.dxf.defpoint
                p2 = getattr(dim.dxf, "defpoint2", None)
                if p2 is not None:
                    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
                    orient = "H" if dx >= dy else "V"
            except Exception:
                pass

        # last resort: inspect virtual entities (dimension line is usually a long LINE)
        if orient is None:
            try:
                best_len = 0.0
                best_orient = None
                for ve in dim.virtual_entities():
                    if ve.dxftype() == "LINE":
                        try:
                            sx, sy, _ = ve.dxf.start
                            ex, ey, _ = ve.dxf.end
                        except Exception:
                            continue
                        dx = abs(ex - sx); dy = abs(ey - sy)
                        l = (dx**2 + dy**2) ** 0.5
                        if l > best_len:
                            best_len = l
                            best_orient = "H" if dx >= dy else "V"
                orient = best_orient
            except Exception:
                pass

        # record
        if orient == "H":
            max_h = val if max_h is None else max(max_h, val)
        elif orient == "V":
            max_v = val if max_v is None else max(max_v, val)
        else:
            # unknown → count towards both (rare)
            max_h = val if max_h is None else max(max_h, val)
            max_v = val if max_v is None else max(max_v, val)

    return max_h, max_v


def _aabb_size(msp, include=None, exclude=None):
    """
    Compute an axis-aligned bounding box by explicitly walking geometry,
    recursing into INSERTed blocks. Works even if entity.bbox() is missing.
    Returns (dx, dy, dz) in drawing units or (None, None, None) if nothing found.
    """
    import math

    include_set = {s.upper() for s in include} if include else None
    exclude_set = {s.upper() for s in exclude} if exclude else set()

    # Heuristic: skip annotation-ish layers unless explicitly included
    def layer_ok(layer: str) -> bool:
        L = (layer or "").upper()
        if include_set is not None and L not in include_set:
            return False
        if L in exclude_set:
            return False
        if include_set is None and L.startswith(("AM_", "DEFPOINTS", "DIM", "ANNOT", "TITLE", "BORDER", "FRAME")):
            return False
        return True

    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    found = False

    def _upd(x, y, z=0.0):
        nonlocal xmin, ymin, zmin, xmax, ymax, zmax, found
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        zmin = min(zmin, z)
        xmax = max(xmax, x)
        ymax = max(ymax, y)
        zmax = max(zmax, z)
        found = True

    def _approx_arc(cx, cy, r, start_deg, end_deg, steps=72):
        # robust sampling (handles start>end wrap)
        start = math.radians(start_deg)
        end = math.radians(end_deg)
        if end < start:
            end += 2 * math.pi
        for i in range(steps + 1):
            t = start + (end - start) * (i / steps)
            yield (cx + r * math.cos(t), cy + r * math.sin(t))

    def _approx_ellipse(ent, steps=128):
        try:
            for p in ent.flattening(deviation=0.01, segments=steps):
                # p is Vec3
                yield float(p.x), float(p.y), float(p.z)
        except Exception:
            # very defensive: sample param 0..1
            for i in range(steps + 1):
                try:
                    p = ent.point_at(i / steps)
                    yield float(p.x), float(p.y), float(p.z)
                except Exception:
                    break

    def _approx_spline(ent, steps=128):
        try:
            for p in ent.approximate(steps):
                yield float(p.x), float(p.y), float(p.z)
        except Exception:
            # last resort: control points
            try:
                for p in ent.control_points:
                    yield float(p.x), float(p.y), float(p.z)
            except Exception:
                pass

    def _handle_entity(e, depth=0, max_depth=3):
        kind = e.dxftype()
        layer = getattr(e.dxf, "layer", "")
        if not layer_ok(layer):
            return
        try:
            if kind == "INSERT":
                if depth >= max_depth:
                    return
                for ve in e.virtual_entities():
                    _handle_entity(ve, depth + 1, max_depth)
                return

            if kind in ("LINE",):
                sx, sy, sz = map(float, e.dxf.start)
                ex, ey, ez = map(float, e.dxf.end)
                _upd(sx, sy, sz)
                _upd(ex, ey, ez)
                return

            if kind in ("LWPOLYLINE", "POLYLINE"):
                try:
                    # try vertices() (POLYLINE) first
                    for v in e.vertices():
                        x = float(v.dxf.location.x)
                        y = float(v.dxf.location.y)
                        z = float(v.dxf.location.z)
                        _upd(x, y, z)
                except Exception:
                    # LWPOLYLINE
                    try:
                        for x, y, *_ in e.get_points("xy"):
                            _upd(float(x), float(y), 0.0)
                    except Exception:
                        pass
                return

            if kind == "CIRCLE":
                cx, cy, cz = map(float, e.dxf.center)
                r = float(e.dxf.radius)
                _upd(cx - r, cy, cz)
                _upd(cx + r, cy, cz)
                _upd(cx, cy - r, cz)
                _upd(cx, cy + r, cz)
                return

            if kind == "ARC":
                cx, cy, cz = map(float, e.dxf.center)
                r = float(e.dxf.radius)
                sa = float(e.dxf.start_angle)
                ea = float(e.dxf.end_angle)
                for x, y in _approx_arc(cx, cy, r, sa, ea, steps=72):
                    _upd(x, y, cz)
                return

            if kind == "ELLIPSE":
                for x, y, z in _approx_ellipse(e):
                    _upd(x, y, z)
                return

            if kind == "SPLINE":
                for x, y, z in _approx_spline(e):
                    _upd(x, y, z)
                return

            if kind in ("SOLID", "TRACE"):
                for vx, vy, vz in e.wcs_vertices():
                    _upd(float(vx), float(vy), float(vz))
                return

            # ignore TEXT/MTEXT/TABLE/DIMENSION/HATCH here
        except Exception:
            # swallow any odd entity we don't know how to sample
            return

    # Walk modelspace
    for ent in msp:
        _handle_entity(ent, 0, 3)

    if not found:
        return None, None, None
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
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

    # --- AABB (view-targeted first, broad fallback) ---
    preferred_layer_sets = [
        ["PUN_SHOE-PLAN"],
        ["PUN_SHOE-DETAIL"],
        ["PUN_SHOE-PLAN_ASSY"],
        ["PUN_SHOE-ASSY"],
    ]
    dx = dy = dz = None

    if layer_include:
        # user passed explicit filters; honor them
        dx, dy, dz = _aabb_size(msp, include=layer_include, exclude=layer_exclude)
    else:
        # try each candidate view layer until we get real geometry
        for inc in preferred_layer_sets:
            dx, dy, dz = _aabb_size(msp, include=inc, exclude=layer_exclude)
            print(f"[part-dims] aabb try include={inc}: dx={dx} dy={dy} dz={dz}")
            if dx and dy:
                break
        if dx is None or dy is None:
            # last resort: everything (excluding annotation-ish layers)
            dx, dy, dz = _aabb_size(msp, include=None, exclude=layer_exclude)
            print(f"[part-dims] aabb broad: dx={dx} dy={dy} dz={dz}")

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
    print(f"[part-dims] linear: Hmax={lh} Vmax={lv}")
    source: Optional[str] = None

    def _band_clip(vals, ref, lo=0.85, hi=1.15):
        if ref is None:
            # no ref → ignore tiny feature dims; keep only values >= 3.0 in drawing units
            return [v for v in vals if (v is not None and v >= 3.0)]
        lo_v, hi_v = lo * ref, hi * ref
        return [v for v in vals if (v is not None and lo_v <= v <= hi_v)]

    # Use AABB sizes as the reference band (if present)
    cand_w = _band_clip([ox, lh, dx], dx)
    cand_h = _band_clip([oy, lv, dy], dy)

    # If nothing survives the band, keep the largest (but still ignore tiny ones)
    if not cand_w:
        cand_w = [v for v in [ox, lh, dx] if v is not None]
    if not cand_h:
        cand_h = [v for v in [oy, lv, dy] if v is not None]

    width_units = max(cand_w) if cand_w else None
    height_units = max(cand_h) if cand_h else None

    L: Optional[float] = None
    W: Optional[float] = None

    if width_units is not None and height_units is not None:
        W = width_units * f
        H = height_units * f
        L, W = (H, W) if H >= W else (W, H)
        src_bits = []
        if dx is not None or dy is not None:
            src_bits.append("aabb")
        if ox is not None or oy is not None:
            src_bits.append("ord")
        if lh is not None or lv is not None:
            src_bits.append("lin")
        source = "+".join(src_bits)
        print(f"[part-dims] fused dimensions: L={L:.4f} in, W={W:.4f} in from {source}")
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

