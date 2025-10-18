# geo_read_more.py
# Read more DWG/DXF detail into your GEO dict (ezdxf required for DXF).
from __future__ import annotations
import re
import math
from collections import Counter

from cad_quoter.vendors import ezdxf as _ezdxf_vendor

try:
    ezdxf = _ezdxf_vendor.require_ezdxf()
except Exception:
    ezdxf = None

# ---------- regex library (hole table & notes) ----------
RE_TAP    = re.compile(r"(\(\s*\d+\s*\)\s*)?(#\s*\d{1,2}-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?)\s*TAP", re.I)
RE_CBORE  = re.compile(r"C[’']?BORE|CBORE|COUNTERBORE", re.I)
RE_CSK    = re.compile(r"CSK|C['’]SINK|COUNTERSINK", re.I)
RE_THRU   = re.compile(r"\bTHRU\b", re.I)
NUM_PATTERN = r"(?:\d*\.\d+|\d+)"

RE_DEPTH  = re.compile(rf"({NUM_PATTERN})\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?", re.I)
RE_QTY    = re.compile(r"\((\d+)\)")
RE_REF_D  = re.compile(rf"\bREF\s*[Ø⌀]?\s*({NUM_PATTERN})", re.I)
RE_DIA    = re.compile(rf"[Ø⌀\u00D8]?\s*({NUM_PATTERN})")  # last resort
RE_FROMBK = re.compile(r"\bFROM\s+BACK\b", re.I)

RE_MAT    = re.compile(r"\b(MATERIAL|MAT)\b[:\s]*([A-Z0-9\-\s/\.]+)")
RE_COAT   = re.compile(r"\b(ANODIZE|BLACK OXIDE|ZINC PLATE|NICKEL PLATE|PASSIVATE|HEAT TREAT|DLC|PVD|CVD)\b", re.I)
RE_TOL    = re.compile(r"\bUNLESS OTHERWISE SPECIFIED\b.*?([±\+\-]\s*\d+\.\d+)", re.I|re.S)
RE_REV    = re.compile(r"\bREV(ISION)?\b[:\s]*([A-Z0-9\-]+)")

# ---------- utils ----------
def _units_scale(doc):
    """Return factor to convert doc units -> inches."""
    try:
        u = int(doc.header.get("$INSUNITS", 1))
    except Exception:
        u = 1
    to_in = 1.0 if u == 1 else (1/25.4) if u in (4, 13) else 1.0
    return {"insunits": u, "to_in": to_in}

def _spaces(doc):
    yield doc.modelspace()
    for n in doc.layouts.names_in_taborder():
        if n.lower() in ("model", "defpoints"): continue
        try: yield doc.layouts.get(n).entity_space
        except Exception: pass


def _all_tables(doc):
    """Yield every TABLE entity in model/layout spaces and inside block INSERTs."""

    if doc is None:
        return

    seen: set[int] = set()

    def _yield_from_insert(ins):
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

    for sp in _spaces(doc):
        try:
            tables = sp.query("TABLE")
        except Exception:
            tables = []
        for table in tables:
            key = id(table)
            if key not in seen:
                seen.add(key)
                yield table
        try:
            inserts = sp.query("INSERT")
        except Exception:
            inserts = []
        for ins in inserts:
            yield from _yield_from_insert(ins)

def _is_ordinate(dim) -> bool:
    try: return (int(dim.dxf.dimtype) & 6) == 6 or int(dim.dxf.dimtype) == 6
    except Exception: return False

def _dim_value_in(dim, to_in) -> float | None:
    try:
        v = float(dim.get_measurement()) * to_in
        return v if v == v and v > 0 else None
    except Exception:
        return None

# ---------- 1) Plate L/W from ORDINATE dims ----------
def harvest_plate_dims(doc, to_in: float):
    xs, ys = [], []
    for sp in _spaces(doc):
        for d in sp.query("DIMENSION"):
            if not _is_ordinate(d): continue
            v = _dim_value_in(d, to_in)
            if not v or v < 0.1 or v > 10000: continue
            ang = float(getattr(d.dxf, "text_rotation", 0.0)) % 180.0
            (ys if 60 <= ang <= 120 else xs).append(v)
    def pick(vals):
        uniq = sorted({round(v,3) for v in vals if v>0.2})
        return (uniq[-1] if uniq else None)
    return {
        "plate_len_in": pick(ys),    # dominant Y
        "plate_wid_in": pick(xs),    # dominant X
        "prov": "ORDINATE DIMENSIONS"
    }

# ---------- 2) Outline area / perimeter / deburr edge length ----------
def harvest_outline(doc, to_in: float):
    perim = 0.0
    area_in2 = 0.0
    for sp in _spaces(doc):
        # closed LWPOLYLINEs
        for pl in sp.query("LWPOLYLINE"):
            if not pl.closed: continue
            try:
                perim += float(pl.length()) * to_in
                area_in2 += float(pl.area()) * (to_in**2)
            except Exception:
                pass
        # add circles (rare for plate outline)
        for c in sp.query("CIRCLE"):
            r = float(c.dxf.radius) * to_in
            perim += 2*math.pi*r
            area_in2 += math.pi*r*r
        # arcs add to edge length but not to closed area unless stitched; skip area
        for a in sp.query("ARC"):
            ang = abs(float(a.dxf.end_angle) - float(a.dxf.start_angle)) * math.pi/180.0
            r = float(a.dxf.radius) * to_in
            perim += r*ang
    return {"edge_len_in": round(perim,3) if perim else None,
            "outline_area_in2": round(area_in2,3) if area_in2 else None,
            "prov": "CLOSED LWPOLYLINE/ARC"}

# ---------- 3) Hole geometry from entities (CIRCLEs) ----------
def harvest_holes_geometry(doc, to_in: float):
    dias = []
    for sp in _spaces(doc):
        for c in sp.query("CIRCLE"):
            d = 2.0*float(c.dxf.radius)*to_in
            # ignore giant rings (likely borders) and tiny artifacts
            if 0.04 <= d <= 6.0:
                dias.append(round(d,4))
    fam = Counter(dias)
    return {
        "hole_count_geom": len(dias),
        "hole_diam_families_in": dict(fam.most_common()),
        "min_hole_in": min(dias) if dias else None,
        "max_hole_in": max(dias) if dias else None,
        "prov": "CIRCLE entities"
    }

# ---------- 4) Hole table / text harvest ----------
def _iter_table_text(doc):
    if doc is None:
        return
    for t in _all_tables(doc):
        try:
            n_rows = int(getattr(t.dxf, "n_rows", 0))
            n_cols = int(getattr(t.dxf, "n_cols", 0))
        except Exception:
            n_rows = 0
            n_cols = 0
        if n_rows <= 0 or n_cols <= 0:
            continue
        try:
            for r in range(n_rows):
                row: list[str] = []
                for c in range(n_cols):
                    try:
                        cell = t.get_cell(r, c)
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
                s = " | ".join([x.strip() for x in row if x])
                if s.strip():
                    yield s
        except Exception:
            continue
    for sp in _spaces(doc):
        for e in sp.query("MTEXT,TEXT"):
            try:
                yield e.plain_text() if e.dxftype()=="MTEXT" else e.dxf.text
            except Exception:
                pass

def harvest_hole_table(doc):
    taps = cb = csk = 0
    deepest = 0.0
    from_back = False
    lines = []
    fam_guess = Counter()
    for raw in _iter_table_text(doc):
        u = raw.upper()
        if not any(k in u for k in ("HOLE","TAP","THRU","CBORE","C'BORE","DRILL","Ø","⌀")):
            continue
        lines.append(raw)
        qty = 1
        mq = RE_QTY.search(u)
        if mq: qty = int(mq.group(1))
        if RE_TAP.search(u):   taps += qty
        if RE_CBORE.search(u): cb   += qty
        if RE_CSK.search(u):   csk  += qty
        md = RE_DEPTH.search(u)
        if md:
            deepest = max(deepest, float(md.group(1)))
            if (md.group(2) or "").upper() == "BACK" or RE_FROMBK.search(u):
                from_back = True
        # Ø ref (group families)
        mref = RE_REF_D.search(u) or (None if "REF" not in u else None)
        if mref:
            fam_guess[round(float(mref.group(1)),4)] += qty
        else:
            mdia = RE_DIA.search(u)
            if mdia and ("Ø" in u or "⌀" in u):
                fam_guess[round(float(mdia.group(1)),4)] += qty

    return {
        "tap_qty": taps,
        "cbore_qty": cb,
        "csk_qty": csk,
        "deepest_hole_in": deepest or None,
        "holes_from_back": bool(from_back),
        "hole_table_families_in": dict(fam_guess.most_common()) if fam_guess else None,
        "chart_lines": lines,
        "prov": "HOLE TABLE / TEXT"
    }

# ---------- 5) Title block / global notes ----------
def harvest_title_notes(doc):
    text_dump = []
    # INSERT attributes + block internal text
    for sp in _spaces(doc):
        for ins in sp.query("INSERT"):
            try:
                bname = ins.dxf.name.upper()
            except Exception:
                continue
            # attributes
            for att in getattr(ins, "attribs", []):
                try:
                    text_dump.append(att.dxf.tag)
                    text_dump.append(att.plain_text())
                except Exception: pass
            # inside block
            try:
                bdef = doc.blocks.get(bname)
                for e in bdef.query("TEXT, MTEXT"):
                    txt = e.plain_text() if e.dxftype()=="MTEXT" else e.dxf.text
                    text_dump.append(txt)
            except Exception: pass

    # all free text again (paperspace notes)
    for s in _iter_table_text(doc):
        text_dump.append(s)

    big = "\n".join([str(s) for s in text_dump if s])
    U = big.upper()
    mat = None; coat = []; tol=None; rev=None
    m = RE_MAT.search(U);  mat = m.group(2).strip() if m else None
    coat = sorted({m.group(0).upper() for m in RE_COAT.finditer(U)})
    mt = RE_TOL.search(U); tol = mt.group(1).replace(" ","") if mt else None
    mr = RE_REV.search(U); rev = mr.group(2).strip() if mr else None
    return {"material_note": mat, "finishes": coat, "default_tol": tol, "revision": rev, "prov": "TITLE BLOCK / NOTES"}

# ---------- 6) Orchestrator ----------
def build_geo_from_dxf(dxf_path: str) -> dict:
    if not ezdxf:
        return {"error": "ezdxf not installed", "ok": False}
    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        return {"error": f"DXF read failed: {e}", "ok": False}

    units = _units_scale(doc)
    to_in = units["to_in"]

    dims  = harvest_plate_dims(doc, to_in)
    outline = harvest_outline(doc, to_in)
    holes_g = harvest_holes_geometry(doc, to_in)
    hole_tbl = harvest_hole_table(doc)
    title = harvest_title_notes(doc)

    # Merge with provenance
    geo = {
        "ok": True,
        "units": units,
        "plate_len_in": dims.get("plate_len_in"),
        "plate_wid_in": dims.get("plate_wid_in"),
        "edge_len_in": outline.get("edge_len_in"),
        "outline_area_in2": outline.get("outline_area_in2"),
        "hole_count_geom": holes_g.get("hole_count_geom"),
        "hole_diam_families_in": holes_g.get("hole_diam_families_in"),
        "min_hole_in": holes_g.get("min_hole_in"),
        "max_hole_in": holes_g.get("max_hole_in"),
        "tap_qty": hole_tbl.get("tap_qty", 0),
        "cbore_qty": hole_tbl.get("cbore_qty", 0),
        "csk_qty": hole_tbl.get("csk_qty", 0),
        "deepest_hole_in": hole_tbl.get("deepest_hole_in"),
        "holes_from_back": hole_tbl.get("holes_from_back", False),
        "hole_table_families_in": hole_tbl.get("hole_table_families_in"),
        "chart_lines": hole_tbl.get("chart_lines"),
        "material_note": title.get("material_note"),
        "finishes": title.get("finishes", []),
        "default_tol": title.get("default_tol"),
        "revision": title.get("revision"),
        "provenance": {
            "plate_size": dims.get("prov"),
            "edge_len": outline.get("prov"),
            "holes_geom": holes_g.get("prov"),
            "hole_table": hole_tbl.get("prov"),
            "material": title.get("prov"),
        },
    }
    return geo
