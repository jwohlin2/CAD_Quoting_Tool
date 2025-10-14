# -*- coding: utf-8 -*-
"""Geometry loading and enrichment utilities."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# Optional trimesh for STL
try:
    import trimesh  # type: ignore
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

from cad_quoter.vendors import ezdxf as _ezdxf_vendor
from cad_quoter.vendors.occt import (
    STACK,
    STACK_GPROP,
    BRepAdaptor_Curve,
    BRepAlgoAPI_Section,
    BRep_Builder,
    BRep_Tool,
    BRepCheck_Analyzer,
    BRepGProp,
    GProp_GProps,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Torus,
    GeomAdaptor_Surface,
    IFSelect_RetDone,
    IGESControl_Reader,
    STEPControl_Reader,
    ShapeAnalysis_Surface,
    ShapeFix_Shape,
    TopAbs_COMPOUND,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopExp,
    TopExp_Explorer,
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopoDS,
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Shape,
    Bnd_Box,
    bnd_add,
    brep_read,
    gp_Dir,
    gp_Pln,
    gp_Pnt,
    gp_Vec,
    uv_bounds,
)

try:
    from cad_quoter.vendors.occt import (
        BACKEND,
        STACK,
        STACK_GPROP,
        BRepAdaptor_Curve,
        BRepAlgoAPI_Section,
        BRep_Builder,
        BRep_Tool,
        BRepCheck_Analyzer,
        BRepGProp,
        BRepTools,
        GProp_GProps,
        GeomAbs_BSplineSurface,
        GeomAbs_BezierSurface,
        GeomAbs_Circle,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Torus,
        GeomAdaptor_Surface,
        IFSelect_RetDone,
        IGESControl_Reader,
        STEPControl_Reader,
        ShapeAnalysis_Surface,
        ShapeFix_Shape,
        TopAbs_COMPOUND,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_ShapeEnum,
        TopExp,
        TopExp_Explorer,
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopoDS,
        TopoDS_Compound,
        TopoDS_Face,
        TopoDS_Shape,
        Bnd_Box,
        bnd_add,
        brep_read,
        gp_Dir,
        gp_Pln,
        gp_Pnt,
        gp_Vec,
        uv_bounds,
    )
except Exception as _OCC_IMPORT_ERROR:  # pragma: no cover - optional OCC bindings missing

    class _MissingProxy:
        def __init__(self, name: str) -> None:
            self._name = name

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(f"{self._name} requires OCCT bindings. {_MISSING_HELP}") from _OCC_IMPORT_ERROR

        def __getattr__(self, attr: str) -> Any:
            raise ImportError(
                f"{self._name}.{attr} requires OCCT bindings. {_MISSING_HELP}"
            ) from _OCC_IMPORT_ERROR

        def __iter__(self):
            raise ImportError(f"{self._name} requires OCCT bindings. {_MISSING_HELP}") from _OCC_IMPORT_ERROR

        def __bool__(self) -> bool:  # pragma: no cover - sentinel helper
            return False

    def _missing_func(name: str) -> Callable[..., Any]:
        def _impl(*_: Any, **__: Any) -> Any:
            raise ImportError(f"{name} requires OCCT bindings. {_MISSING_HELP}") from _OCC_IMPORT_ERROR

        return _impl

    BACKEND = STACK = STACK_GPROP = "missing"  # type: ignore[assignment]
    for _name in [
        "BRepAdaptor_Curve",
        "BRepAlgoAPI_Section",
        "BRep_Builder",
        "BRep_Tool",
        "BRepCheck_Analyzer",
        "BRepGProp",
        "BRepTools",
        "GProp_GProps",
        "GeomAbs_BSplineSurface",
        "GeomAbs_BezierSurface",
        "GeomAbs_Circle",
        "GeomAbs_Cone",
        "GeomAbs_Cylinder",
        "GeomAbs_Plane",
        "GeomAbs_Torus",
        "GeomAdaptor_Surface",
        "IFSelect_RetDone",
        "IGESControl_Reader",
        "STEPControl_Reader",
        "ShapeAnalysis_Surface",
        "ShapeFix_Shape",
        "TopAbs_COMPOUND",
        "TopAbs_EDGE",
        "TopAbs_FACE",
        "TopAbs_SHELL",
        "TopAbs_SOLID",
        "TopAbs_ShapeEnum",
        "TopExp",
        "TopExp_Explorer",
        "TopTools_IndexedDataMapOfShapeListOfShape",
        "TopoDS",
        "TopoDS_Compound",
        "TopoDS_Face",
        "TopoDS_Shape",
        "Bnd_Box",
        "gp_Dir",
        "gp_Pln",
        "gp_Pnt",
        "gp_Vec",
    ]:
        globals()[_name] = _MissingProxy(_name)

    bnd_add = _missing_func("bnd_add")
    brep_read = _missing_func("brep_read")
    uv_bounds = _missing_func("uv_bounds")
from cad_quoter.vendors._occt_base import PREFIX as _OCCT_PREFIX, load_module as _occt_load_module

from appkit.occ_compat import (
    _MISSING_HELP,
    FACE_OF,
    as_face,
    ensure_face,
    ensure_shape,
    face_surface,
    iter_faces,
    linear_properties,
    list_iter,
    map_size,
    map_shapes_and_ancestors,
    to_edge,
    to_edge_safe,
    to_shell,
    to_solid,
)

_HAS_EZDXF = _ezdxf_vendor.HAS_EZDXF
_HAS_ODAFC = _ezdxf_vendor.HAS_ODAFC
_EZDXF_VER = _ezdxf_vendor.EZDXF_VERSION

# Hole chart helpers (optional)
try:
    from hole_table_parser import parse_hole_table_lines
except Exception:
    parse_hole_table_lines = None

try:
    from dxf_text_extract import extract_text_lines_from_dxf
except Exception:
    extract_text_lines_from_dxf = None


# numpy is optional for a few small calcs; degrade gracefully if missing
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore
# ---------- end compat ----------


# ---- tiny helpers you can use elsewhere --------------------------------------
def require_ezdxf():
    """Raise a clear error if ezdxf is missing."""
    return _ezdxf_vendor.require_ezdxf()

def get_dwg_converter_path() -> str:
    """Resolve a DWG?DXF converter path (.bat/.cmd/.exe)."""
    exe = os.environ.get("ODA_CONVERTER_EXE") or os.environ.get("DWG2DXF_EXE")
    # Fall back to a local wrapper next to this script if it exists.
    local = str(Path(__file__).with_name("dwg2dxf_wrapper.bat"))
    if not exe and Path(local).exists():
        exe = local
    return exe or ""

def have_dwg_support() -> bool:
    """True if we can open DWG (either odafc or an external converter is available)."""
    return _HAS_ODAFC or bool(get_dwg_converter_path())

def get_import_diagnostics_text() -> str:
    import sys
    import os
    lines = []
    lines.append(f"Python: {sys.executable}")
    try:
        import fitz  # PyMuPDF  # noqa: F401
        lines.append("PyMuPDF: OK")
    except Exception as e:
        lines.append(f"PyMuPDF: MISSING ({e})")

    try:
        ezdxf_mod = require_ezdxf()
    except Exception as e:  # pragma: no cover - diagnostic helper
        lines.append(f"ezdxf: MISSING ({e})")
    else:
        lines.append(f"ezdxf: {getattr(ezdxf_mod, '__version__', 'unknown')}")
        try:
            _ezdxf_vendor.require_odafc()
        except Exception as e:
            lines.append(f"ezdxf.addons.odafc: not available ({e})")
        else:
            lines.append("ezdxf.addons.odafc: OK")

    oda = os.environ.get("ODA_CONVERTER_EXE") or "(not set)"
    d2d = os.environ.get("DWG2DXF_EXE") or "(not set)"
    lines.append(f"ODA_CONVERTER_EXE: {oda}")
    lines.append(f"DWG2DXF_EXE: {d2d}")

    # local wrapper presence
    from pathlib import Path
    wrapper = Path(__file__).with_name("dwg2dxf_wrapper.bat")
    lines.append(f"Local wrapper present: {wrapper.exists()} ({wrapper})")
    return "\n".join(lines)
# Optional PDF stack
try:
    import fitz  # PyMuPDF  # noqa: F401
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False
def upsert_var_row(df, item, value, dtype="number"):
    """
    Upsert one row by Item name (case-insensitive).
    - Forces `item` and `value` to scalars.
    - Works on the sanitized 3-column df (Item, Example..., Data Type...).
    """
    import numpy as np
    import pandas as pd

    # force scalars
    if isinstance(item, pd.Series):
        item = item.iloc[0]
    item = str(item)

    if isinstance(value, pd.Series):
        value = value.iloc[0]
    # try to make numeric if it looks numeric; otherwise keep as-is
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            pass
        else:
            value = float(value)
    except Exception:
        # leave non-numerics (e.g., text) as-is
        pass

    # build a row with the exact df schema
    cols = list(df.columns)
    row = {c: "" for c in cols}
    if "Item" in row:
        row["Item"] = item
    if "Example Values / Options" in row:
        row["Example Values / Options"] = value
    if "Data Type / Input Method" in row:
        row["Data Type / Input Method"] = dtype

    # case-insensitive exact match on Item
    m = df["Item"].astype(str).str.casefold() == item.casefold()
    if m.any():
        df.loc[m, cols] = [row[c] for c in cols]
        return df

    # append
    new_row = pd.DataFrame([[row[c] for c in cols]], columns=cols)
    return pd.concat([df, new_row], ignore_index=True)

DIM_RE = re.compile(r"(?:ï¿½|DIAM|DIA)\s*([0-9.+-]+)|R\s*([0-9.+-]+)|([0-9.+-]+)\s*[xX]\s*([0-9.+-]+)")

def load_drawing(path: Path) -> Drawing:
    ezdxf_mod = require_ezdxf()
    if path.suffix.lower() == ".dwg":
        # Prefer explicit converter/wrapper if configured (works even if ODA isnï¿½t on PATH)
        exe = get_dwg_converter_path()
        if exe:
            dxf_path = convert_dwg_to_dxf(str(path))
            return ezdxf_mod.readfile(dxf_path)
        # Fallback: odafc (requires ODAFileConverter on PATH)
        if _HAS_ODAFC:

            odafc_mod = _ezdxf_vendor.require_odafc()
            return odafc_mod.readfile(str(path))
        raise RuntimeError(
            "DWG import needs ODA File Converter. Set ODA_CONVERTER_EXE to the exe "
            "or place dwg2dxf_wrapper.bat next to the script."
        )
    return ezdxf_mod.readfile(str(path))  # DXF directly


def dxf_to_structured(doc: Drawing) -> dict:
    msp = doc.modelspace()
    header = doc.header
    insunits = header.get("$INSUNITS", 0)  # 1 = inches, 6 = meters, etc.
    # Normalize a simple unit string for quoting
    UNITS = {0:"unitless",1:"in",2:"ft",3:"mi",4:"mm",5:"cm",6:"m"}
    units = UNITS.get(insunits, "unitless")

    texts = []
    for e in msp.query("TEXT"):
        texts.append({"type":"TEXT","text":e.dxf.text, "xy":tuple(e.dxf.insert[:2])})
    for e in msp.query("MTEXT"):
        # .text is formatted; .plain_text() strips inline codes
        texts.append({"type":"MTEXT","text":e.plain_text(), "xy":tuple(e.dxf.insert[:2])})
    for e in msp.query("MLEADER"):
        if e.context and e.context.mtext:
            texts.append({"type":"MLEADER","text":e.context.mtext.plain_text()})

    dims = []
    for d in msp.query("DIMENSION"):
        # d.dxf.text often contains the displayed text; fall back to raw if needed
        label = (d.dxf.text or "").strip()
        # Try to find diameter / radius / WxH patterns in the label
        m = DIM_RE.search(label.replace(",", "."))
        dims.append({"raw": label, "pattern": m.groups() if m else None})

    # Quick & dirty material/thickness/title block scrapes from text blobs
    material = next((t["text"].split(":")[-1].strip()
                     for t in texts if re.search(r"\\bMATERIAL\\b", t["text"], re.I)), None)
    thickness = next((re.search(r"(?:(?:THICK|THK)[\\s:=]*)([0-9.]+)\\s*(mm|in|inch|inches)?", t["text"], re.I)
                      for t in texts), None)
    if thickness:
        thickness = {"value": float(thickness.group(1)), "units": (thickness.group(2) or units)}

    return {
        "source_units": units,
        "texts": texts,
        "dimensions": dims,
        "material": material,
        "thickness": thickness,
    }

def pdf_to_structured(pdf_path: Path, page_index: int = 0, dpi: int = 300):
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    # 1) TEXT blocks (vector text if present)
    words = page.get_text("words")  # [(x0,y0,x1,y1,"word", block_no, line_no, word_no), ...]
    text_blocks = page.get_text("blocks")  # paragraph-level
    raw_text = page.get_text("text")       # simple text stream

    # 2) Tables (handy for title blocks / BOMs)
    tables = []
    try:
        tabs = page.find_tables()  # since PyMuPDF 1.23+
        for t in tabs.tables:
            tables.append(t.extract())
    except Exception:
        pass

    # 3) Vector line-art (for figure geometry / bounding clusters)
    drawings = page.get_drawings()  # vector paths, strokes, fills
    # Optional: cluster into groups that form a single figure (1.24+)
    # clusters = page.cluster_drawings()

    # 4) Page raster for Qwen
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_path = pdf_path.with_suffix(f".p{page_index}.png")
    pix.save(png_path)

    return {
        "raw_text": raw_text,
        "blocks": text_blocks,
        "tables": tables,
        "drawings_count": len(drawings),
        "image_path": str(png_path),
    }

QWEN_SYSTEM = """You are a quoting assistant. Only confirm or fill missing parameters.
Prefer vector-extracted numbers over reading pixels. If units conflict, explain and ask for clarification.
Return strict JSON with fields:
{overall_size_mm, thickness_mm, material, hole_counts_by_dia_mm, slots, threads, tolerance_notes, finish, qty, uncertainties[]}"""

def build_llm_payload(structured: dict, page_image_path: str | None):
    messages = [
        {"role":"system","content": QWEN_SYSTEM},
        {"role":"user","content":[
            {"type":"text","text":"Vector-extracted data (trusted):\n" + json.dumps(structured, ensure_ascii=False)},
        ]}
    ]
    if page_image_path:
        messages[1]["content"].append({"type":"image_url","image_url":{"url":"file://"+page_image_path}})
    return messages



# ==== OCCT compat delegated to cad_quoter.vendors.occt ====


def BRepTools_UVBounds(face):
    return uv_bounds(face)


def _brep_read(path: str) -> TopoDS_Shape:
    return brep_read(path)

def _shape_from_reader(reader):
    """Return a healed TopoDS_Shape from a STEP/IGES reader."""
    transfer_count = 0
    if hasattr(reader, "NbShapes"):
        try:
            transfer_count = reader.NbShapes()
        except Exception:
            transfer_count = 0
    if not transfer_count and hasattr(reader, "NbRootsForTransfer"):
        try:
            transfer_count = reader.NbRootsForTransfer()
        except Exception:
            transfer_count = 0
    if transfer_count <= 0:
        raise RuntimeError("Reader produced zero shapes")

    if transfer_count == 1:
        shape = reader.Shape(1)
    else:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        added = 0
        for i in range(1, transfer_count + 1):
            s = reader.Shape(i)
            if s is None or s.IsNull():
                continue
            builder.Add(compound, s)
            added += 1
        if added == 0:
            raise RuntimeError("Reader produced only null sub-shapes")
        shape = compound

    if shape is None or shape.IsNull():
        raise RuntimeError("Reader produced a null TopoDS_Shape")

    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    healed = fixer.Shape()
    if healed is None or healed.IsNull():
        raise RuntimeError("Shape healing failed (null shape)")

    try:
        analyzer = BRepCheck_Analyzer(healed)
        # we do not require validity, but invoking the analyzer surfaces issues early
        analyzer.IsValid()
    except Exception:
        pass

    return healed

def read_step_shape(path: str) -> TopoDS_Shape:
    rdr = STEPControl_Reader()
    if rdr.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("STEP read failed (file unreadable or unsupported).")
    rdr.TransferRoots()
    n = rdr.NbShapes()
    if n == 0:
        raise RuntimeError("STEP read produced zero shapes (no transferable roots).")

    if n == 1:
        shape = rdr.Shape(1)
    else:
        builder = BRep_Builder()
        comp = TopoDS_Compound()
        builder.MakeCompound(comp)
        for i in range(1, n + 1):
            s = rdr.Shape(i)
            if not s.IsNull():
                builder.Add(comp, s)
        shape = comp

    if shape.IsNull():
        raise RuntimeError("STEP produced a null TopoDS_Shape.")
    # Verify we truly pass a Shape to MapShapesAndAncestors
    print("[DBG] shape type:", type(shape).__name__, "IsNull:", getattr(shape, "IsNull", lambda: True)())
    amap = map_shapes_and_ancestors(shape, TopAbs_EDGE, TopAbs_FACE)
    print("[DBG] map size:", amap.Size())
    # DEBUG: sanity probe for STEP faces
    if os.environ.get("STEP_PROBE", "0") == "1":
        cnt = 0
        try:
            for f in iter_faces(shape):
                _surf, _loc = face_surface(f)
                cnt += 1
        except Exception as _e:
            # Keep debug non-fatal; report and continue
            print(f"[STEP_PROBE] error during face probe: {_e}")
        else:
            print(f"[STEP_PROBE] faces={cnt}")


    fx = ShapeFix_Shape(shape)
    fx.Perform()
    return fx.Shape()

def safe_bbox(shape: TopoDS_Shape):
    if shape is None or shape.IsNull():
        raise ValueError("Cannot compute bounding box of a null shape.")
    box = Bnd_Box()
    bnd_add(shape, box, True)  # <- uses whichever binding is available
    return box
def read_step_or_iges_or_brep(path: str) -> TopoDS_Shape:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".step", ".stp"):
        return read_step_shape(str(p))
    if ext in (".iges", ".igs"):
        ig = IGESControl_Reader()
        if ig.ReadFile(str(p)) != IFSelect_RetDone:
            raise RuntimeError("IGES read failed")
        ig.TransferRoots()
        return _shape_from_reader(ig)
    if ext == ".brep":
        return _brep_read(str(p))
    raise RuntimeError(f"Unsupported OCC format: {ext}")

def convert_dwg_to_dxf(dwg_path: str, *, out_ver="ACAD2018") -> str:
    """
    Robust DWG?DXF wrapper.
    Works with:
      - A .bat/.cmd wrapper that accepts:  <input.dwg> <output.dxf>
      - A custom exe that accepts:         <input.dwg> <output.dxf>
      - ODAFileConverter.exe (7-arg form): <in_dir> <out_dir> <ver> DXF 0 0 <filter>
    Looks for the converter via env var or common local paths and prints
    the exact command used on failure.
    """
    # 1) find converter
    exe = (os.environ.get("ODA_CONVERTER_EXE")
           or os.environ.get("DWG2DXF_EXE")
           or str(Path(__file__).with_name("dwg2dxf_wrapper.bat"))
           or r"D:\CAD_Quoting_Tool\dwg2dxf_wrapper.bat")

    if not exe or not Path(exe).exists():
        raise RuntimeError(
            "DWG import needs a DWG?DXF converter.\n"
            "Set ODA_CONVERTER_EXE (recommended) or DWG2DXF_EXE to a .bat/.cmd/.exe.\n"
            "Expected .bat signature:  <input.dwg> <output.dxf>"
        )

    dwg = Path(dwg_path)
    out_dir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
    out_dxf = out_dir / (dwg.stem + ".dxf")

    exe_lower = Path(exe).name.lower()
    try:
        if exe_lower.endswith(".bat") or exe_lower.endswith(".cmd"):
            # ? run batch via cmd.exe so it actually executes
            cmd = ["cmd", "/c", exe, str(dwg), str(out_dxf)]
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elif "odafileconverter" in exe_lower:
            # ? official ODAFileConverter CLI
            cmd = [exe, str(dwg.parent), str(out_dir), out_ver, "DXF", "0", "0", dwg.name]
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            # ? generic exe that accepts <in> <out>
            cmd = [exe, str(dwg), str(out_dxf)]
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "DWG?DXF conversion failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e

    # 2) resolve the produced DXF
    produced = out_dxf if out_dxf.exists() else (out_dir / (dwg.stem + ".dxf"))
    if not produced.exists():
        raise RuntimeError(
            "Converter returned success but DXF not found.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"checked: {out_dxf} | {produced}"
        )
    return str(produced)



ANG_TOL = math.radians(5.0)
DOT_TOL = math.cos(ANG_TOL)
SMALL = 1e-7



def iter_solids(shape: TopoDS_Shape):
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield to_solid(exp.Current())
        exp.Next()

def explode_compound(shape: TopoDS_Shape):
    """If the file is a big COMPOUND, break it into shapes (parts/bodies)."""
    exp = TopExp_Explorer(shape, TopAbs_COMPOUND)
    if exp.More():
        # Itï¿½s a compound ï¿½ return its shells/solids/faces as needed
        solids = list(iter_solids(shape))
        if solids:
            return solids
        # fallback to shells
        sh = TopExp_Explorer(shape, TopAbs_SHELL)
        shells = []
        while sh.More():
            shells.append(to_shell(sh.Current()))
            sh.Next()
        return shells
    return [shape]



def _bbox(shape):
    box = safe_bbox(shape)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (xmin, ymin, zmin, xmax, ymax, zmax)

def _area_of_face(face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass())

def _length_of_edge(edge) -> float:
    props = GProp_GProps()
    linear_properties(edge, props)
    return float(props.Mass())

def _face_midpoint_uv(face):
    umin, umax, vmin, vmax = BRepTools_UVBounds(face)
    return (0.5*(umin+umax), 0.5*(vmin+vmax))


def _face_normal(face):
    try:
        try:
            from OCP.BRepLProp import BRepLProp_SLProps
        except ImportError:
            from OCC.Core.BRepLProp import BRepLProp_SLProps
        u, v = _face_midpoint_uv(face)
        props = BRepLProp_SLProps(face_surface(face)[0], u, v, 1, SMALL)
        if props.IsNormalDefined():
            n = props.Normal()
            if face.Orientation().Name() == "REVERSED":
                n.Reverse()
            return gp_Dir(n.X(), n.Y(), n.Z())
    except Exception:
        pass
    return None

def _face_type(face) -> str:
    surf = face_surface(face)[0]
    ga = GeomAdaptor_Surface(surf)
    st = ga.GetType()
    if st == GeomAbs_Plane: return "planar"
    if st in (GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone): return "cylindrical"
    if st in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface): return "freeform"
    return "other"



def _cluster_normals(normals):
    clusters = []
    for n in normals:
        added = False
        for c in clusters:
            if abs(n.Dot(c)) > DOT_TOL:
                added = True; break
        if not added:
            clusters.append(n)
    return clusters

def _sum_edge_length_sharp(shape, angle_thresh_deg=175.0) -> float:
    angle_thresh = math.radians(angle_thresh_deg)
    edge2faces = map_shapes_and_ancestors(shape, TopAbs_EDGE, TopAbs_FACE)
    total = 0.0
    for i in range(1, map_size(edge2faces) + 1):
        edge = to_edge(edge2faces.FindKey(i))
        face_list = edge2faces.FindFromIndex(i)
        faces = [ensure_face(shp) for shp in list_iter(face_list)]
        if len(faces) < 2:
            continue
        f1, f2 = faces[0], faces[-1]
        n1 = _face_normal(f1)
        n2 = _face_normal(f2)
        if not n1 or not n2:
            continue
        ang = math.acos(max(-1.0, min(1.0, abs(n1.Dot(n2)))))
        if ang < (math.pi - angle_thresh):
            total += _length_of_edge(edge)
    return total

def _largest_planar_faces_and_normals(shape):
    largest_area = 0.0
    normals = []
    for f in iter_faces(shape):
        if _face_type(f) == "planar":
            a = _area_of_face(f)
            largest_area = max(largest_area, a)
            n = _face_normal(f)
            if n: normals.append(n)
    clusters = _cluster_normals(normals)
    return largest_area, clusters

def _surface_areas_by_type(shape):
    from collections import defaultdict
    areas = defaultdict(float)
    for f in iter_faces(shape):
        t = _face_type(f)
        areas[t] += _area_of_face(f)
    return areas

def _section_perimeter_len(shape, z_values):
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
    total = 0.0
    for z in z_values:
        plane = gp_Pln(gp_Pnt(0,0,z), gp_Dir(0,0,1))
        sec = BRepAlgoAPI_Section(shape, plane, False); sec.Build()
        if not sec.IsDone(): continue
        w = sec.Shape()
        it = TopExp_Explorer(w, TopAbs_EDGE)
        while it.More():
            e = to_edge(it.Current())
            total += _length_of_edge(e)
            it.Next()
    return total

def _min_wall_between_parallel_planes(shape):
    planes = []
    for f in iter_faces(shape):
        if _face_type(f) == "planar":
            n = _face_normal(f)
            if n:
                umin, umax, vmin, vmax = BRepTools_UVBounds(f)
                surf, _ = face_surface(f)
                sas = ShapeAnalysis_Surface(surf)
                pnt = sas.Value(0.5*(umin+umax), 0.5*(vmin+vmax))
                d = n.X()*pnt.X() + n.Y()*pnt.Y() + n.Z()*pnt.Z()
                planes.append((f, n, d, _bbox(f)))
    def overlap(a1, a2, b1, b2): return not (a2 < b1 or b2 < a1)
    min_th = None
    for i in range(len(planes)):
        f1, n1, d1, b1 = planes[i]
        for j in range(i+1, len(planes)):
            f2, n2, d2, b2 = planes[j]
            if abs(abs(n1.Dot(n2)) - 1.0) < 1e-3:
                nd = (abs(n1.X()), abs(n1.Y()), abs(n1.Z()))
                normal_axis = nd.index(max(nd))
                if normal_axis == 2:
                    ok = overlap(b1[0], b1[3], b2[0], b2[3]) and overlap(b1[1], b1[4], b2[1], b2[4])
                elif normal_axis == 1:
                    ok = overlap(b1[0], b1[3], b2[0], b2[3]) and overlap(b1[2], b1[5], b2[2], b2[5])
                else:
                    ok = overlap(b1[1], b1[4], b2[1], b2[4]) and overlap(b1[2], b1[5], b2[2], b2[5])
                if ok:
                    th = abs(d1 - d2)
                    if (min_th is None) or (th < min_th): min_th = th
    return min_th

def _hole_face_identifier(face, cylinder, face_bbox):
    centers = []
    try:
        exp = TopExp_Explorer(face, TopAbs_EDGE)
    except Exception:
        exp = None
    while exp and exp.More():
        try:
            edge = to_edge(exp.Current())
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() == GeomAbs_Circle:
                circ = curve.Circle()
                loc = circ.Location()
                centers.append((round(loc.X(), 3), round(loc.Y(), 3), round(loc.Z(), 3)))
        except Exception:
            pass
        finally:
            exp.Next()
    if centers:
        return ("edges", tuple(sorted(centers)))
    loc = cylinder.Axis().Location()
    center = (
        round(0.5 * (face_bbox[0] + face_bbox[3]), 3),
        round(0.5 * (face_bbox[1] + face_bbox[4]), 3),
        round(0.5 * (face_bbox[2] + face_bbox[5]), 3),
    )
    return (
        "fallback",
        (
            round(loc.X(), 3),
            round(loc.Y(), 3),
            round(loc.Z(), 3),
            *center,
        ),
    )


def _hole_groups_from_cylinders(shape, bbox=None):
    if bbox is None:
        bbox = _bbox(shape)
    groups = {}
    for f in iter_faces(shape):
        if _face_type(f) != "cylindrical":
            continue
        ga = GeomAdaptor_Surface(face_surface(f)[0])
        try:
            cyl = ga.Cylinder()
            r = abs(cyl.Radius())
            ax = cyl.Axis().Direction()
            fb = _bbox(f)

            def proj(x, y, z):
                return x * ax.X() + y * ax.Y() + z * ax.Z()

            span = abs(proj(fb[3], fb[4], fb[5]) - proj(fb[0], fb[1], fb[2]))
            dia = 2.0 * r
            bmin = proj(*bbox[:3])
            bmax = proj(*bbox[3:])
            bspan = abs(bmax - bmin)
            through = span > 0.9 * bspan
            key = (round(dia, 2), round(span, 2), through)
            hole_id = _hole_face_identifier(f, cyl, fb)
        except Exception:
            continue

        entry = groups.setdefault(
            key,
            {
                "dia_mm": round(dia, 2),
                "depth_mm": round(span, 2),
                "through": through,
                "count": 0,
                "_ids": set(),
            },
        )
        if hole_id in entry["_ids"]:
            continue
        entry["_ids"].add(hole_id)
        entry["count"] += 1
    for entry in groups.values():
        entry.pop("_ids", None)
    return list(groups.values())

def _turning_score(shape, areas_by_type):
    total = sum(areas_by_type.values()) or 1.0
    cyl_ratio = areas_by_type.get("cylindrical", 0.0)/total
    axes = []
    for f in iter_faces(shape):
        if _face_type(f) == "cylindrical":
            ga = GeomAdaptor_Surface(face_surface(f)[0])
            try: axes.append(ga.Cylinder().Axis().Direction())
            except Exception: pass
    if axes:
        ax = gp_Dir(sum(a.X() for a in axes) or 1e-9, sum(a.Y() for a in axes) or 1e-9, sum(a.Z() for a in axes) or 1e-9)
        align = sum(abs(ax.Dot(a)) for a in axes)/len(axes)
    else:
        align = 0.0
    score = max(0.0, min(1.0, 0.5*cyl_ratio + 0.5*align))
    xmin,ymin,zmin,xmax,ymax,zmax = _bbox(shape)
    Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin
    length = max(Lx, Ly, Lz)
    maxod = sorted([Lx, Ly, Lz])[-2]
    return {"GEO_Turning_Score_0to1": round(score,3), "GEO_MaxOD_mm": round(maxod,3), "GEO_Length_mm": round(length,3)}

def enrich_geo_occ(shape):
    xmin,ymin,zmin,xmax,ymax,zmax = _bbox(shape)
    L,W,H = xmax-xmin, ymax-ymin, zmax-zmin

    props = GProp_GProps()
    try:
        BRepGProp.VolumeProperties_s(shape, props); volume = float(props.Mass())
        com = props.CentreOfMass(); center = [com.X(), com.Y(), com.Z()]
    except Exception:
        volume, center = 0.0, [0.0,0.0,0.0]
    try:
        BRepGProp.SurfaceProperties_s(shape, props); area_total = float(props.Mass())
    except Exception:
        area_total = 0.0

    faces = 0
    for _ in iter_faces(shape): faces += 1

    areas_by_type = _surface_areas_by_type(shape)
    largest_planar, normal_clusters = _largest_planar_faces_and_normals(shape)
    unique_normals = len(normal_clusters)

    min_wall = _min_wall_between_parallel_planes(shape)
    thinwall_present = (min_wall is not None) and (min_wall < 1.0)

    z_vals = [zmin + 0.25*(zmax-zmin), zmin + 0.50*(zmax-zmin), zmin + 0.75*(zmax-zmin)]
    wedm_len = _section_perimeter_len(shape, z_vals)

    deburr_len = _sum_edge_length_sharp(shape, angle_thresh_deg=175.0)

    holes = _hole_groups_from_cylinders(shape, (xmin,ymin,zmin,xmax,ymax,zmax))

    bbox_vol = max(L*W*H, 1e-9)
    free_ratio = areas_by_type.get("freeform", 0.0)/max(area_total, 1e-9)
    complexity = (faces / (volume if volume>0 else bbox_vol))*100.0 + 30.0*free_ratio

    axis_dirs = [gp_Dir(1,0,0), gp_Dir(-1,0,0), gp_Dir(0,1,0), gp_Dir(0,-1,0), gp_Dir(0,0,1), gp_Dir(0,0,-1)]
    hit=0; tot=0
    for f in iter_faces(shape):
        n = _face_normal(f)
        if n:
            tot += 1
            if any(abs(n.Dot(a))>DOT_TOL for a in axis_dirs): hit += 1
    access = (hit/tot) if tot else 0.0

    geo = {
        "GEO-01_Length_mm": round(L,3),
        "GEO-02_Width_mm": round(W,3),
        "GEO-03_Height_mm": round(H,3),
        "GEO-Volume_mm3": round(volume,2),
        "GEO-SurfaceArea_mm2": round(area_total,2),
        "Feature_Face_Count": int(faces),
        "GEO_Area_Planar_mm2": round(areas_by_type.get("planar",0.0),2),
        "GEO_Area_Cyl_mm2": round(areas_by_type.get("cylindrical",0.0),2),
        "GEO_Area_Freeform_mm2": round(areas_by_type.get("freeform",0.0),2),
        "GEO_LargestPlane_Area_mm2": round(largest_planar,2),
        "GEO_Setup_UniqueNormals": int(unique_normals),
        "GEO_MinWall_mm": round(min_wall,3) if min_wall is not None else None,
        "GEO_ThinWall_Present": bool(thinwall_present),
        "GEO_WEDM_PathLen_mm": round(wedm_len,2),
        "GEO_Deburr_EdgeLen_mm": round(deburr_len,2),
        "GEO_Hole_Groups": holes,
        "GEO_Complexity_0to100": round(complexity,2),
        "GEO_3Axis_Accessible_Pct": round(access,3),
        "GEO_CenterOfMass": center,
        "OCC_Backend": "OCC"
    }
    geo.update(_turning_score(shape, areas_by_type))
    return geo

def enrich_geo_stl(path):
    import time
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting enrich_geo_stl for {path}")
    if not _HAS_TRIMESH:

        raise RuntimeError("trimesh not available to process STL")
    
    print(f"[{time.time() - start_time:.2f}s] Loading mesh...")
    mesh = trimesh.load(path, force='mesh')
    print(f"[{time.time() - start_time:.2f}s] Mesh loaded. Faces: {len(mesh.faces)}")
    
    if mesh.is_empty:
        raise RuntimeError("Empty STL mesh")
    
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    L, W, H = float(xmax-xmin), float(ymax-ymin), float(zmax-zmin)
    area_total = float(mesh.area)
    volume = float(mesh.volume) if mesh.is_volume else 0.0
    faces = int(len(mesh.faces))
    
    print(f"[{time.time() - start_time:.2f}s] Calculating WEDM length...")
    z_vals = [zmin + 0.25*(zmax-zmin), zmin + 0.50*(zmax-zmin), zmin + 0.75*(zmax-zmin)]
    wedm_len = 0.0
    for z in z_vals:
        sec = mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1])
        if sec is None:
            continue
        planar = sec.to_2D() # Changed to_planar() to to_2D()
        try:
            wedm_len += float(planar.length)
        except Exception:
            pass
    print(f"[{time.time() - start_time:.2f}s] WEDM length calculated.")
    
    print(f"[{time.time() - start_time:.2f}s] Calculating deburr length...")
    try:
        import numpy as np
        angles = mesh.face_adjacency_angles
        edges = mesh.face_adjacency_edges
        if len(angles) and len(edges):
            sharp = angles > math.radians(15.0)
            vec = mesh.vertices[edges[:,0]] - mesh.vertices[edges[:,1]]
            lengths = np.linalg.norm(vec, axis=1)
            deburr_len = float(lengths[sharp].sum())
        else:
            deburr_len = 0.0
    except Exception:
        deburr_len = 0.0
    print(f"[{time.time() - start_time:.2f}s] Deburr length calculated.")
    
    bbox_vol = max(L*W*H, 1e-9)
    complexity = (faces / max(volume, bbox_vol)) * 100.0
    
    print(f"[{time.time() - start_time:.2f}s] Calculating 3-axis accessibility...")
    try:
        n = mesh.face_normals
        area_faces = mesh.area_faces.reshape(-1,1)
        import numpy as np
        axes = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float).T
        dots = np.abs(n @ axes)  # (F,6)
        close = (dots > DOT_TOL).any(axis=1)
        access = float((area_faces[close].sum() / area_faces.sum())) if area_faces.sum() > 0 else 0.0
    except Exception:
        access = 0.0
    print(f"[{time.time() - start_time:.2f}s] 3-axis accessibility calculated.")
    
    try:
        center = list(map(float, mesh.center_mass))
    except Exception:
        center = [0.0,0.0,0.0]
    
    print(f"[{time.time() - start_time:.2f}s] Finished enrich_geo_stl.")
    return {
        "GEO-01_Length_mm": round(L,3),
        "GEO-02_Width_mm": round(W,3),
        "GEO-03_Height_mm": round(H,3),
        "GEO-Volume_mm3": round(volume,2),
        "GEO-SurfaceArea_mm2": round(area_total,2),
        "Feature_Face_Count": faces,
        "GEO_LargestPlane_Area_mm2": None,
        "GEO_Setup_UniqueNormals": None,
        "GEO_MinWall_mm": None,
        "GEO_ThinWall_Present": False,
        "GEO_WEDM_PathLen_mm": round(wedm_len,2),
        "GEO_Deburr_EdgeLen_mm": round(deburr_len,2),
        "GEO_Hole_Groups": [],
        "GEO_Complexity_0to100": round(complexity,2),
        "GEO_3Axis_Accessible_Pct": round(access,3),
        "GEO_CenterOfMass": center,
        "OCC_Backend": "trimesh (STL)"
    }


def load_cad_any(path: str) -> TopoDS_Shape:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in ("step", "stp", "iges", "igs", "brep"):
        return read_step_or_iges_or_brep(path)
    if ext == "dwg":
        dxf_path = convert_dwg_to_dxf(path)
        return read_dxf_as_occ_shape(dxf_path)
    if ext == "dxf":
        return read_dxf_as_occ_shape(path)

    raise RuntimeError(f"Unsupported file type for shape loading: {ext}")
def read_cad_any(path: str):
    ext = Path(path).suffix.lower()
    if ext in (".step", ".stp"):
        return read_step_shape(path)
    if ext in (".iges", ".igs"):
        ig = IGESControl_Reader()
        if ig.ReadFile(path) != IFSelect_RetDone:
            raise RuntimeError("IGES read failed")
        ig.TransferRoots()
        return _shape_from_reader(ig)
    if ext == ".brep":
        return brep_read(path)
    if ext == ".dxf":
        return read_dxf_as_occ_shape(path)
    if ext == ".dwg":
        conv = os.environ.get("ODA_CONVERTER_EXE") or os.environ.get("DWG2DXF_EXE")
        print(f"INFO: Using DWG converter: {conv}")
        dxf_path = convert_dwg_to_dxf(path)
        return read_dxf_as_occ_shape(dxf_path)
    raise RuntimeError(f"Unsupported CAD format: {ext}")


HAS_TRIMESH = _HAS_TRIMESH
HAS_EZDXF = _HAS_EZDXF
HAS_ODAFC = _HAS_ODAFC
EZDXF_VERSION = _EZDXF_VER


def load_model(path: str) -> TopoDS_Shape:
    """Load any supported CAD file into an OCC :class:`TopoDS_Shape`."""

    return load_cad_any(path)


def extract_features_with_occ(path: str) -> Optional[Dict[str, Any]]:
    """Best-effort OCC feature extraction from STEP/IGES/BREP inputs."""

    ext = Path(path).suffix.lower()
    if ext not in {".step", ".stp", ".iges", ".igs", ".brep"}:
        return None
    shape = read_step_or_iges_or_brep(path)
    return enrich_geo_occ(shape)


class GeometryService:
    """Facade exposing high-level geometry operations for the UI layer."""

    def load_model(self, path: str) -> TopoDS_Shape:
        return load_model(path)

    def read_model(self, path: str) -> TopoDS_Shape:
        return read_cad_any(path)

    def read_step(self, path: str) -> TopoDS_Shape:
        return read_step_shape(path)

    def convert_dwg(self, path: str, *, out_ver: str = "ACAD2018") -> str:
        return convert_dwg_to_dxf(path, out_ver=out_ver)

    def enrich_occ(self, shape: TopoDS_Shape) -> Dict[str, Any]:
        return enrich_geo_occ(shape)

    def enrich_stl(self, path: str) -> Dict[str, Any]:
        return enrich_geo_stl(path)

    def extract_occ_features(self, path: str) -> Optional[Dict[str, Any]]:
        return extract_features_with_occ(path)

    @property
    def has_trimesh(self) -> bool:
        return HAS_TRIMESH

    @property
    def has_ezdxf(self) -> bool:
        return HAS_EZDXF

    @property
    def has_odafc(self) -> bool:
        return HAS_ODAFC


__all__ = [
    "HAS_TRIMESH",
    "HAS_EZDXF",
    "HAS_ODAFC",
    "EZDXF_VERSION",
    "GeometryService",
    "load_model",
    "load_cad_any",
    "read_cad_any",
    "read_step_shape",
    "read_step_or_iges_or_brep",
    "convert_dwg_to_dxf",
    "enrich_geo_occ",
    "enrich_geo_stl",
    "safe_bbox",
    "iter_solids",
    "explode_compound",
    "extract_features_with_occ",
    "parse_hole_table_lines",
    "extract_text_lines_from_dxf",
    "require_ezdxf",
    "get_dwg_converter_path",
    "have_dwg_support",
    "get_import_diagnostics_text",
    "upsert_var_row",
    "map_geo_to_double_underscore",
    "collect_geo_features_from_df",
    "update_variables_df_with_geo",
]


def map_geo_to_double_underscore(geo: dict) -> dict:
    from cad_quoter.geometry_wrappers import map_geo_to_double_underscore as _impl

    return _impl(geo)


def collect_geo_features_from_df(df):
    from cad_quoter.geometry_wrappers import collect_geo_features_from_df as _impl

    return _impl(df)


def update_variables_df_with_geo(df, geo: dict):
    from cad_quoter.geometry_wrappers import update_variables_df_with_geo as _impl

    return _impl(df, geo)


def read_dxf_as_occ_shape(dxf_path: str):
    # minimal DXF?OCC triangulated shape (3DFACE/MESH/POLYFACE), fallback: extrude closed polyline
    import numpy as np
    if _OCCT_PREFIX is None:
        raise RuntimeError("OCCT bindings are required for DXF conversion")

    builder_api = _occt_load_module("BRepBuilderAPI")
    prim_api = _occt_load_module("BRepPrimAPI")
    shape_fix_mod = _occt_load_module("ShapeFix")

    def tri_face(p0,p1,p2):
        poly = builder_api.BRepBuilderAPI_MakePolygon()
        for p in (p0,p1,p2,p0):
            poly.Add(gp_Pnt(*p))
        return builder_api.BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()

    def sew(faces):
        sew = builder_api.BRepBuilderAPI_Sewing(1.0e-6, True, True, True, True)
        for f in faces: sew.Add(f)
        sew.Perform()
        return sew.SewedShape()

    ezdxf_mod = require_ezdxf()
    doc = ezdxf_mod.readfile(dxf_path)
    msp = doc.modelspace()
    INSUNITS = doc.header.get("$INSUNITS", 1)  # 1=in, 4=mm, 2=ft, 6=m
    u2mm = {1:25.4, 4:1.0, 2:304.8, 6:1000.0}.get(INSUNITS, 1.0)

    tris = []
    # 3DFACE
    for e in msp.query("3DFACE"):
        verts = [e.dxf.vtx0, e.dxf.vtx1, e.dxf.vtx2, e.dxf.vtx3]
        pts = [(vx * u2mm, vy * u2mm, vz * u2mm) for (vx, vy, vz) in verts]
        if len(pts) < 3:
            continue
        tris.append((pts[0], pts[1], pts[2]))
        if len(pts) == 4:
            v3 = np.array(pts[2])
            v4 = np.array(pts[3])
            if np.linalg.norm(v3 - v4) > 1e-12:
                tris.append((pts[0], pts[2], pts[3]))
    # POLYFACE
    for e in msp.query("POLYFACE"):
        for f in e.faces():
            pts=[(v.dxf.location.x*u2mm, v.dxf.location.y*u2mm, v.dxf.location.z*u2mm) for v in f.vertices()]
            if len(pts)>=3:
                tris.append((pts[0], pts[1], pts[2]))
                for k in range(3,len(pts)):
                    tris.append((pts[0], pts[k-1], pts[k]))
    # MESH
    for e in msp.query("MESH"):
        for f in e.faces_as_vertices():
            pts=[(v.x*u2mm, v.y*u2mm, v.z*u2mm) for v in f]
            if len(pts)>=3:
                tris.append((pts[0], pts[1], pts[2]))
                for k in range(3,len(pts)):
                    tris.append((pts[0], pts[k-1], pts[k]))

    faces = [tri_face(*t) for t in tris]
    shape = None
    if faces:
        sewed = sew(faces)
        try:
            solid = builder_api.BRepBuilderAPI_MakeSolid()
            exp = TopExp_Explorer(sewed, TopAbs_FACE)
            while exp.More():
                solid.Add(exp.Current()); exp.Next()
            fix = shape_fix_mod.ShapeFix_Solid(solid.Solid()); fix.Perform()
            shape = fix.Solid()
        except Exception:
            shape = sewed

    # Fallback: 2D closed polyline ? extrude small thickness
    if (shape is None) or shape.IsNull():
        for pl in msp.query("LWPOLYLINE"):
            if pl.closed:
                pts2d = [(x*u2mm, y*u2mm, 0.0) for x,y,_ in pl.get_points("xyb")]
                poly = builder_api.BRepBuilderAPI_MakePolygon()
                for q in pts2d: poly.Add(gp_Pnt(*q))
                poly.Close()
                face = builder_api.BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
                thk_mm = float(os.environ.get("DXF_EXTRUDE_THK_MM", "5.0"))
                shape = prim_api.BRepPrimAPI_MakePrism(face, gp_Vec(0,0,thk_mm)).Shape()
                break

    if (shape is None) or shape.IsNull():
        raise RuntimeError("DXF contained no 3D geometry I can use. Prefer STEP/SAT if possible.")
    return shape

# ---- 2D: PDF (PyMuPDF) -------------------------------------------------------
try:
    import fitz  # old import name  # noqa: F401
    _HAS_PYMUPDF = True
except Exception:
    try:
        import pymupdf as fitz  # new import name  # noqa: F401
        _HAS_PYMUPDF = True
    except Exception:
        fitz = None  # allow the rest of the app to import
