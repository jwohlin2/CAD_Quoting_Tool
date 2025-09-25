# -*- coding: utf-8 -*-
# app_gui_occ_flow_v8_single_autollm.py
"""
Single-file CAD Quoter (v8)
- LLM (Qwen via llama-cpp) is ENABLED by default.
- Auto-loads a GGUF model from:
    1) QWEN_GGUF_PATH env var (if a file)
    2) D:\CAD_Quoting_Tool\models\*.gguf  (your path)
    3) <script_dir>\models\*.gguf
    4) .\models\*.gguf (cwd)
- STEP/IGES via OCP, STL via trimesh
- Variables auto-detect, Overrides tab, LLM tab
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import textwrap
import tkinter as tk
import tkinter.font as tkfont
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import pandas as pd

SHOW_LLM_HOURS_DEBUG = False # set True only when debugging
if sys.platform == 'win32':
    occ_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(occ_bin):
        os.add_dll_directory(occ_bin)
from tkinter import ttk, filedialog, messagebox
import subprocess, tempfile, shutil


# --------------------- helpers: model discovery ---------------------
PREFERRED_MODEL_DIRS = [
    r"D:\CAD_Quoting_Tool\models",
]

def _pick_best_gguf(paths):
    # Prefer files with 'qwen' and 'instruct'; otherwise choose the largest.
    if not paths:
        return ""
    paths = [Path(p) for p in paths if Path(p).is_file() and p.lower().endswith(".gguf")]
    if not paths:
        return ""
    preferred = [p for p in paths if ("qwen" in p.name.lower() and "instr" in p.name.lower())]
    pool = preferred or paths
    # Largest by size
    try:
        best = max(pool, key=lambda p: p.stat().st_size)
    except Exception:
        best = pool[0]
    return str(best)

def find_default_qwen_model():
    # 1) Env
    envp = os.environ.get("QWEN_GGUF_PATH", "")
    if envp and Path(envp).is_file():
        return envp
    # 2) D:\CAD_Quoting_Tool\models
    for d in PREFERRED_MODEL_DIRS:
        dpath = Path(d)
        if dpath.is_dir():
            ggufs = list(dpath.glob("*.gguf"))
            choice = _pick_best_gguf([str(p) for p in ggufs])
            if choice: return choice
    # 3) script_dir\models
    try:
        sdir = Path(__file__).resolve().parent
        ggufs = list((sdir / "models").glob("*.gguf"))
        choice = _pick_best_gguf([str(p) for p in ggufs])
        if choice: return choice
    except Exception:
        pass
    # 4) cwd\models
    ggufs = list((Path.cwd() / "models").glob("*.gguf"))
    choice = _pick_best_gguf([str(p) for p in ggufs])
    if choice: return choice
    return ""

# Optional trimesh for STL
try:
    import trimesh  # type: ignore
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False

_HAS_EZDXF  = False
_HAS_ODAFC  = False      # ezdxf.addons.odafc (uses ODA File Converter)
_EZDXF_VER  = "unknown"

try:
    import ezdxf
    _EZDXF_VER = getattr(ezdxf, "__version__", "unknown")
    _HAS_EZDXF = True
except Exception:
    ezdxf = None  # keep name defined

# odafc is optional; don't fail if it's not present
try:
    if _HAS_EZDXF:
        from ezdxf.addons import odafc  # type: ignore
        _HAS_ODAFC = True
except Exception:
    _HAS_ODAFC = False

# numpy is optional for a few small calcs; degrade gracefully if missing
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore
# ---------- OCC / OCP compatibility ----------
STACK = None

def _make_bnd_add_ocp():
    # Try every known symbol name in OCP builds
    candidates = []
    try:
        # module-level function: Add(...)
        from OCP.BRepBndLib import Add as _add1
        candidates.append(_add1)
    except Exception:
        pass
    try:
        # module-level function: Add_s(...)
        from OCP.BRepBndLib import Add_s as _add1s
        candidates.append(_add1s)
    except Exception:
        pass
    try:
        # module-level function: BRepBndLib_Add(...)
        from OCP.BRepBndLib import BRepBndLib_Add as _add2
        candidates.append(_add2)
    except Exception:
        pass
    try:
        # module-level function: brepbndlib_Add(...)
        from OCP.BRepBndLib import brepbndlib_Add as _add3
        candidates.append(_add3)
    except Exception:
        pass
    try:
        from OCP.BRepBndLib import BRepBndLib as _klass
        for attr in ("Add", "Add_s", "AddClose_s", "AddOptimal_s", "AddOBB_s"):
            fn = getattr(_klass, attr, None)
            if fn:
                candidates.append(fn)
    except Exception:
        pass

    if not candidates:
        raise ImportError("No BRepBndLib.Add symbol found in OCP")

    # return a dispatcher that tries different call signatures
    def _bnd_add(shape, box, use_triangulation=True):
        last_err = None
        for fn in candidates:
            # Try 3-arg form
            try:
                return fn(shape, box, use_triangulation)
            except TypeError as e:
                last_err = e
            # Try 2-arg form
            try:
                return fn(shape, box)
            except TypeError as e:
                last_err = e
            except Exception as e:
                last_err = e
        # If we get here, none of the variants worked
        raise last_err or RuntimeError("No callable BRepBndLib.Add variant succeeded")

    return _bnd_add

try:
    # Prefer OCP (CadQuery/ocp bindings)
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCP.BRep import BRep_Builder
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Bnd import Bnd_Box

    try:
        bnd_add = _make_bnd_add_ocp()
        STACK = "ocp"
    except Exception:
        # Fallback: try dynamic dispatch at call time for OCP builds missing Add
        def bnd_add(shape, box, use_triangulation=True):
            candidates = [
                ("OCP.BRepBndLib", "Add"),
                ("OCP.BRepBndLib", "Add_s"),
                ("OCP.BRepBndLib", "BRepBndLib_Add"),
                ("OCP.BRepBndLib", "brepbndlib_Add"),
                ("OCC.Core.BRepBndLib", "Add"),
                ("OCC.Core.BRepBndLib", "Add_s"),
                ("OCC.Core.BRepBndLib", "BRepBndLib_Add"),
                ("OCC.Core.BRepBndLib", "brepbndlib_Add"),
            ]
            for mod_name, attr in candidates:
                try:
                    module = __import__(mod_name, fromlist=[attr])
                    fn = getattr(module, attr)
                    return fn(shape, box, use_triangulation)
                except Exception:
                    continue
            raise RuntimeError("No BRepBndLib.Add available in this build")
except Exception:
    # Fallback to pythonocc-core
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.ShapeFix import ShapeFix_Shape
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add as _brep_add

    def bnd_add(shape, box, use_triangulation=True):
        _brep_add(shape, box, use_triangulation)

    STACK = "pythonocc"
# ---------- end shim ----------
# ----- one-backend imports -----
try:
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import topods_Face, topods_Edge, topods_Shell, topods_Solid
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    BACKEND = "ocp"
except Exception:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods_Face, topods_Edge, topods_Shell, topods_Solid
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    BACKEND = "pythonocc"

def _typename(o):  # small helper
    return getattr(o, "__class__", type(o)).__name__

def as_face(obj):
    """
    Return a TopoDS_Face for the *current backend*.
    - If already a Face, return it (don't re-cast).
    - If a tuple (some APIs return (surf, loc) or similar), take first elem.
    - If null/None, raise cleanly.
    - Otherwise cast with topods_Face.
    """
    if obj is None:
        raise TypeError("Expected a shape, got None")

    # unwrap tuples (defensive)
    if isinstance(obj, tuple) and obj:
        obj = obj[0]

    # null guards
    if hasattr(obj, "IsNull") and obj.IsNull():
        raise TypeError("Expected non-null TopoDS_Shape")

    # already a Face? (avoid calling topods_Face on a Face)
    name = _typename(obj)
    if name in ("TopoDS_Face", "Face"):
        return obj
    if hasattr(obj, "ShapeType"):
        try:
            st = obj.ShapeType()
            if st == TopAbs_FACE:
                return obj
        except Exception:
            pass

    # cast Shape→Face using the same backend
    return topods_Face(obj)

def iter_faces(shape):
    """Explorer that yields proper Faces, safely cast."""
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield as_face(exp.Current())   # safe cast (no double-cast)
        exp.Next()

def face_surface(face_like):
    """
    Return (Geom_Surface, Location) across OCP/pythonocc variants.
    Handles Surface vs Surface_s; falls back to BRepAdaptor_Surface.
    """
    f = as_face(face_like)

    def _call_surface(fn):
        loc_from_tool = None
        if hasattr(BRep_Tool, "Location"):
            try:
                loc_from_tool = BRep_Tool.Location(f)
            except Exception:
                loc_from_tool = None

        # Prepare a location placeholder for signatures that expect one.
        loc_arg = None
        try:
            loc_arg = TopLoc_Location()
        except Exception:
            loc_arg = None

        # First try the simple call.
        try:
            surf = fn(f)
            loc = loc_from_tool if loc_from_tool is not None else (
                f.Location() if hasattr(f, "Location") else None)
            return surf, loc
        except TypeError:
            pass

        # Then try signatures that need a location object to be passed in.
        if loc_arg is not None:
            try:
                surf = fn(f, loc_arg)
                return surf, loc_arg
            except TypeError:
                pass

        # give up for this function
        raise

    for attr in ("Surface", "Surface_s"):
        fn = getattr(BRep_Tool, attr, None)
        if not fn:
            continue
        try:
            surf, loc = _call_surface(fn)
            if isinstance(surf, tuple):
                surf, loc = surf
            if hasattr(surf, "Surface"):
                try:
                    surf = surf.Surface()
                except Exception:
                    pass
            return surf, loc
        except Exception:
            continue

    # Fallback is very tolerant
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface as _BAS
    except Exception:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface as _BAS
    a = _BAS(f)
    s = a.Surface()
    if hasattr(s, "Surface"):  # unwrap handle if present
        s = s.Surface()
    loc = f.Location() if hasattr(f, "Location") else None
    return s, loc

# ---------- OCCT compat (OCP or pythonocc-core) ----------
# ---------- Robust casters that work on OCP and pythonocc ----------
# Lock topods casters to the active backend
if STACK == "ocp":
    # Resolve within OCP only (no cross-backend mixing)
    try:
        from OCP.TopoDS import TopoDS_Face
        def _TO_FACE(shape):
            return TopoDS_Face.DownCast(shape)
    except ImportError:
        try:
            from OCP.TopoDS import topods
            _TO_FACE = topods.Face
        except (ImportError, AttributeError):
            try:
                from OCP.TopoDS import Face as _TO_FACE
            except ImportError:
                raise ImportError("Could not import a valid 'Face' caster from OCP")
    try:
        from OCP.TopoDS import TopoDS_Edge
        def _TO_EDGE(shape):
            return TopoDS_Edge.DownCast(shape)
    except ImportError:
        try:
            from OCP.TopoDS import topods
            _TO_EDGE = topods.Edge
        except (ImportError, AttributeError):
            try:
                from OCP.TopoDS import Edge as _TO_EDGE
            except ImportError:
                raise ImportError("Could not import a valid 'Edge' caster from OCP")
    try:
        from OCP.TopoDS import TopoDS_Solid
        def _TO_SOLID(shape):
            return TopoDS_Solid.DownCast(shape)
    except ImportError:
        try:
            from OCP.TopoDS import topods
            _TO_SOLID = topods.Solid
        except (ImportError, AttributeError):
            try:
                from OCP.TopoDS import Solid as _TO_SOLID
            except ImportError:
                raise ImportError("Could not import a valid 'Solid' caster from OCP")
    try:
        from OCP.TopoDS import TopoDS_Shell
        def _TO_SHELL(shape):
            return TopoDS_Shell.DownCast(shape)
    except ImportError:
        try:
            from OCP.TopoDS import topods
            _TO_SHELL = topods.Shell
        except (ImportError, AttributeError):
            try:
                from OCP.TopoDS import Shell as _TO_SHELL
            except ImportError:
                raise ImportError("Could not import a valid 'Shell' caster from OCP")
else:
    # Resolve within OCC.Core only
    try:
        from OCC.Core.TopoDS import topods_Face as _TO_FACE
    except Exception:
        try:
            from OCC.Core.TopoDS import Face as _TO_FACE
        except Exception:
            from OCC.Core.TopoDS import topods as _occ_topods
            _TO_FACE = getattr(_occ_topods, "Face")
    try:
        from OCC.Core.TopoDS import topods_Edge as _TO_EDGE
    except Exception:
        try:
            from OCC.Core.TopoDS import Edge as _TO_EDGE
        except Exception:
            from OCC.Core.TopoDS import topods as _occ_topods
            _TO_EDGE = getattr(_occ_topods, "Edge")
    try:
        from OCC.Core.TopoDS import topods_Solid as _TO_SOLID
    except Exception:
        from OCC.Core.TopoDS import topods as _occ_topods
        _TO_SOLID = getattr(_occ_topods, "Solid")
    try:
        from OCC.Core.TopoDS import topods_Shell as _TO_SHELL
    except Exception:
        from OCC.Core.TopoDS import topods as _occ_topods
        _TO_SHELL = getattr(_occ_topods, "Shell")

# Type guards
def _is_named(obj, names: tuple[str, ...]) -> bool:
    try:
        return type(obj).__name__ in names
    except Exception:
        return False

def _unwrap_value(obj):
    # Some containers return nodes with .Value()
    return obj.Value() if hasattr(obj, "Value") and callable(obj.Value) else obj

def ensure_shape(obj):
    obj = _unwrap_value(obj)
    if obj is None:
        raise TypeError("Expected TopoDS_Shape, got None")
    if hasattr(obj, "IsNull") and obj.IsNull():
        raise TypeError("Expected non-null TopoDS_Shape")
    return obj

# Safe casters: no-ops if already cast; unwrap list nodes; check kind
def to_face_safe(obj):
    obj = _unwrap_value(obj)
    if _is_named(obj, ("TopoDS_Face", "Face")):
        return obj
    obj = ensure_shape(obj)
    # Only cast if it is actually a FACE
    if hasattr(obj, "ShapeType"):
        try:
            from OCP.TopAbs import TopAbs_FACE
        except Exception:
            from OCC.Core.TopAbs import TopAbs_FACE
        if obj.ShapeType() != TopAbs_FACE:
            raise TypeError(f"to_face_safe expected a FACE, got {obj.ShapeType()}")
    return _TO_FACE(obj)

def to_edge_safe(obj):
    obj = _unwrap_value(obj)
    if _is_named(obj, ("TopoDS_Edge", "Edge")):
        return obj
    obj = ensure_shape(obj)
    return _TO_EDGE(obj)

# Choose stack
try:
    # OCP / CadQuery bindings
    from OCP.BRepGProp import BRepGProp as _BRepGProp_mod
    from OCP.TopExp import TopExp as _TopExp_mod

    def _to_edge_ocp(s):
        try:
            from OCP.TopoDS import topods_Edge as _fn
        except Exception:
            from OCP.TopoDS import Edge as _fn
        return _fn(s)

    def _to_face_ocp(s):
        try:
            from OCP.TopoDS import topods_Face as _fn
        except Exception:
            from OCP.TopoDS import Face as _fn
        return _fn(s)

    _TO_EDGE = _to_edge_ocp
    _TO_FACE = _to_face_ocp
    STACK_GPROP = "ocp"
except Exception:
    # pythonocc-core
    try:
        from OCC.Core.BRepGProp import BRepGProp as _BRepGProp_mod
    except Exception:
        from types import SimpleNamespace
        from OCC.Core.BRepGProp import (
            brepgprop_LinearProperties as _lp,
            brepgprop_SurfaceProperties as _sp,
            brepgprop_VolumeProperties as _vp,
        )
        _BRepGProp_mod = SimpleNamespace(
            LinearProperties=_lp,
            SurfaceProperties=_sp,
            VolumeProperties=_vp,
        )
    from OCC.Core.TopExp import TopExp as _TopExp_mod

    def _to_edge_occ(s):
        try:
            from OCC.Core.TopoDS import topods_Edge as _fn
        except Exception:
            from OCC.Core.TopoDS import Edge as _fn
        return _fn(s)

    def _to_face_occ(s):
        try:
            from OCC.Core.TopoDS import topods_Face as _fn
        except Exception:
            from OCC.Core.TopoDS import Face as _fn
        return _fn(s)

    _TO_EDGE = _to_edge_occ
    _TO_FACE = _to_face_occ
    STACK_GPROP = "pythonocc"

# Resolve topods casters across bindings


# ---- modern wrappers (no deprecation warnings)
# ---- modern wrappers (no deprecation warnings)
def linear_properties(edge, gprops):
    """Linear properties across OCP/pythonocc names."""
    fn = getattr(_BRepGProp_mod, "LinearProperties", None)
    if fn is None:
        fn = getattr(_BRepGProp_mod, "LinearProperties_s", None)
    if fn is None:
        try:
            from OCC.Core.BRepGProp import brepgprop_LinearProperties as _old  # type: ignore
            return _old(edge, gprops)
        except Exception as e:
            raise
    return fn(edge, gprops)

def map_shapes_and_ancestors(shape, subshape_type, ancestor_type, amap):
    """TopExp.MapShapesAndAncestors(shape, sub, ancestor, amap) with fallback."""
    fn = getattr(_TopExp_mod, "MapShapesAndAncestors", None)
    if fn is None:
        # old function name
        from OCC.Core.TopExp import topexp_MapShapesAndAncestors as _old  # type: ignore
        return _old(shape, subshape_type, ancestor_type, amap)
    return fn(shape, subshape_type, ancestor_type, amap)

# modern topods casters: topods.Edge(shape) / topods.Face(shape)
# ---- Robust topods casters that are no-ops for already-cast objects ----
def _is_instance(obj, qualnames):
    """True if obj.__class__.__name__ matches any in qualnames across OCP/OCC."""
    try:
        name = type(obj).__name__
    except Exception:
        return False
    return name in qualnames  # e.g. ["TopoDS_Face", "Face"]

def ensure_face(obj):
    # already a Face class? return it
    if type(obj).__name__ in ("TopoDS_Face", "Face"):
        return obj

    # unwrap & null checks...
    obj = _unwrap_value(obj)
    if obj is None or (hasattr(obj, "IsNull") and obj.IsNull()):
        raise TypeError("Expected non-null TopoDS_Shape")

    # if it is a FACE, just return obj — don't cross-cast
    try:
        from OCP.TopAbs import TopAbs_FACE
    except Exception:
        from OCC.Core.TopAbs import TopAbs_FACE
    st = obj.ShapeType() if callable(getattr(obj, "ShapeType", None)) else None
    if st == TopAbs_FACE:
        return obj

    # otherwise it's not a face
    raise TypeError(f"Expected a Face, got {type(obj).__name__}")
def to_edge(s):
    if _is_instance(s, ["TopoDS_Edge", "Edge"]):
        return s
    return _TO_EDGE(s)

def to_face(s):
    if _is_instance(s, ["TopoDS_Face", "Face"]):
        return s
    return _TO_FACE(s)

def to_solid(s):
    if _is_instance(s, ["TopoDS_Solid", "Solid"]):
        return s
    return _TO_SOLID(s)

def to_shell(s):
    if _is_instance(s, ["TopoDS_Shell", "Shell"]):
        return s
    return _TO_SHELL(s)

# Generic size helpers so we never call Extent() on lists/maps again
def map_size(m):
    for name in ("Size", "Extent", "Length"):
        if hasattr(m, name):
            return getattr(m, name)()
    raise AttributeError(f"No size method on {type(m)}")

def list_iter(lst):
    """
    Iterate TopTools_ListOfShape across bindings.
    Use cbegin()/More()/Value()/Next() if available; otherwise fall back to python list conversion.
    """
    # OCP + pythonocc expose cbegin iterator on TopTools_ListOfShape
    if hasattr(lst, "cbegin"):
        it = lst.cbegin()
        while it.More():
            yield it.Value()
            it.Next()
    else:
        # last-resort: try Python iteration (some wheels support it)
        for v in list(lst):
            yield v
# ---------- end compat ----------


# ---- tiny helpers you can use elsewhere --------------------------------------
def require_ezdxf():
    """Raise a clear error if ezdxf is missing."""
    if not _HAS_EZDXF:
        raise RuntimeError("ezdxf not installed. Install with pip/conda (package name: 'ezdxf').")
    return ezdxf

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
    import sys, shutil, os
    lines = []
    lines.append(f"Python: {sys.executable}")
    try:
        import fitz  # PyMuPDF
        lines.append("PyMuPDF: OK")
    except Exception as e:
        lines.append(f"PyMuPDF: MISSING ({e})")

    try:
        import ezdxf
        lines.append(f"ezdxf: {getattr(ezdxf, '__version__', 'unknown')}")
        try:
            from ezdxf.addons import odafc
            lines.append("ezdxf.addons.odafc: OK")
        except Exception as e:
            lines.append(f"ezdxf.addons.odafc: not available ({e})")
    except Exception as e:
        lines.append(f"ezdxf: MISSING ({e})")

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
    import fitz  # PyMuPDF
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
    require_ezdxf()
    if path.suffix.lower() == ".dwg":
        # Prefer explicit converter/wrapper if configured (works even if ODA isnï¿½t on PATH)
        exe = get_dwg_converter_path()
        if exe:
            dxf_path = convert_dwg_to_dxf(str(path))
            return ezdxf.readfile(dxf_path)
        # Fallback: odafc (requires ODAFileConverter on PATH)
        if _HAS_ODAFC:
            return odafc.readfile(str(path))
        raise RuntimeError(
            "DWG import needs ODA File Converter. Set ODA_CONVERTER_EXE to the exe "
            "or place dwg2dxf_wrapper.bat next to the script."
        )
    return ezdxf.readfile(str(path))  # DXF directly


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



# ==== OpenCascade compat (works with OCP OR OCC.Core) ====
import tempfile, subprocess
from pathlib import Path

try:
    # ---- OCP branch ----
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader
    from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRep import BRep_Tool        # OCP version
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid,
    )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    
    # ADD THESE TWO IMPORTS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopoDS import (
        TopoDS_Shape, TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound
    )
    from OCP.TopExp import TopExp_Explorer, TopExp
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone,
        GeomAbs_BSplineSurface, GeomAbs_BezierSurface
    )
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface
    from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Pln
    BACKEND_OCC = "OCP"

    def BRepTools_UVBounds(face):
        return BRepTools.UVBounds(face)


    def _brep_read(path: str) -> TopoDS_Shape:
        s = TopoDS_Shape()
        builder = BRep_Builder()
        if hasattr(BRepTools, "Read_s"):
            ok = BRepTools.Read_s(s, str(path), builder)
        else:
            ok = BRepTools.Read(s, str(path), builder)
        if ok is False:
            raise RuntimeError("BREP read failed")
        return s



except Exception:
    # ---- OCC.Core branch ----
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.IGESControl import IGESControl_Reader
    from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRep import BRep_Tool          # ? OCC version
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid,
    )
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    # ADD TopTools import and TopoDS_Face for the fix below
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCC.Core.TopoDS import (
        TopoDS_Shape, TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound,
        TopoDS_Face
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND
    from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone,
        GeomAbs_BSplineSurface, GeomAbs_BezierSurface
    )
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Pln
    BACKEND_OCC = "OCC.Core"

    # BRepGProp shim (pythonocc uses free functions)
    from OCC.Core.BRepGProp import (
        brepgprop_SurfaceProperties,
        brepgprop_LinearProperties,
        brepgprop_VolumeProperties,
    )
    class _BRepGPropShim:
        @staticmethod
        def SurfaceProperties_s(shape_or_face, gprops): brepgprop_SurfaceProperties(shape_or_face, gprops)
        @staticmethod
        def LinearProperties_s(edge, gprops):          brepgprop_LinearProperties(edge, gprops)
        @staticmethod
        def VolumeProperties_s(shape, gprops):         brepgprop_VolumeProperties(shape, gprops)
    BRepGProp = _BRepGPropShim

    # ADD: TopExp shim to provide MapShapesAndAncestors
    from OCC.Core.TopExp import topexp_MapShapesAndAncestors
    class _TopExpShim:
        @staticmethod
        def MapShapesAndAncestors(shape, subshape_type, ancestor_type, a_map):
            topexp_MapShapesAndAncestors(shape, subshape_type, ancestor_type, a_map)
    TopExp = _TopExpShim

    # UV bounds and brep read are free functions
    from OCC.Core.BRepTools import BRepTools, breptools_Read as _occ_breptools_read
    def BRepTools_UVBounds(face):
        fn = getattr(BRepTools, "UVBounds", None)
        if fn is None:
            from OCC.Core.BRepTools import breptools_UVBounds as _legacy
            return _legacy(face)
        return fn(face)
    def _brep_read(path: str) -> TopoDS_Shape:
        s = TopoDS_Shape()
        ok = _occ_breptools_read(s, str(path), BRep_Builder())
        if ok is False:
            raise RuntimeError("BREP read failed")
        return s

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

    fx = ShapeFix_Shape(shape)
    fx.Perform()
    return fx.Shape()

from OCP.TopoDS import TopoDS_Shape  # or OCC.Core.TopoDS on pythonocc

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

def convert_dwg_to_dxf(dwg_path: str, *, out_ver="ACAD2013") -> str:
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



def safe_bounding_box(shape: TopoDS_Shape) -> Bnd_Box:
    """Public wrapper maintained for compatibility."""
    return safe_bbox(shape)

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
    edge2faces = TopTools_IndexedDataMapOfShapeListOfShape()
    map_shapes_and_ancestors(shape, TopAbs_EDGE, TopAbs_FACE, edge2faces)
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

def _hole_groups_from_cylinders(shape, bbox=None):
    if bbox is None: bbox = _bbox(shape)
    from collections import defaultdict
    groups = defaultdict(lambda: {"dia_mm":0.0,"depth_mm":0.0,"through":False,"count":0})
    for f in iter_faces(shape):
        if _face_type(f) == "cylindrical":
            ga = GeomAdaptor_Surface(face_surface(f)[0])
            try:
                cyl = ga.Cylinder(); r = abs(cyl.Radius()); ax = cyl.Axis().Direction()
                fb = _bbox(f)
                def proj(x,y,z): return x*ax.X() + y*ax.Y() + z*ax.Z()
                span = abs(proj(fb[3],fb[4],fb[5]) - proj(fb[0],fb[1],fb[2]))
                dia = 2.0*r
                bmin = proj(*bbox[:3]); bmax = proj(*bbox[3:]); bspan = abs(bmax-bmin)
                through = span > 0.9*bspan
                key = (round(dia,2), round(span,2), through)
                groups[key]["dia_mm"] = round(dia,2)
                groups[key]["depth_mm"] = round(span,2)
                groups[key]["through"] = through
                groups[key]["count"] += 1
            except Exception:
                pass
    return list(groups.values())

def _turning_score(shape, areas_by_type):
    total = sum(areas_by_type.values()) or 1.0
    cyl_ratio = areas_by_type.get("cylindrical", 0.0)/total
    axes = []
    for f in iter_faces(shape):
        if _face_type(f) == "cylindrical":
            ga = GeomAdaptor_Surface(brep_surface(f)[0])
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
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.IGESControl import IGESControl_Reader
    from OCP.TopoDS import TopoDS_Shape

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
        s = TopoDS_Shape()
        if not BRepTools.Read(s, path, None):
            raise RuntimeError("BREP read failed")
        return s
    if ext == ".dxf":
        return read_dxf_as_occ_shape(path)
    if ext == ".dwg":
        conv = os.environ.get("ODA_CONVERTER_EXE") or os.environ.get("DWG2DXF_EXE")
        print(f"INFO: Using DWG converter: {conv}")
        dxf_path = convert_dwg_to_dxf(path)
        return read_dxf_as_occ_shape(dxf_path)
    raise RuntimeError(f"Unsupported CAD format: {ext}")

# ----------------- Offline Qwen via llama-cpp -----------------
def _llama_missing_msg():
    return ("llama-cpp-python is not installed.\n" 
            "Install in your env:\n" 
            "  python -m pip install -U llama-cpp-python\n")

class _LocalLLM:
    """
    Minimal, tolerant wrapper around llama-cpp-python.
    - Chat if available; otherwise use completion.
    - Always returns best-effort JSON dict.
    """
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0, n_threads: int | None = None):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(_llama_missing_msg()) from e
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model GGUF not found: {model_path}")
        self._Llama = Llama
        self._llm = None
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads

    def _ensure(self):
        if self._llm is None:
            self._llm = self._Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                logits_all=False,
                verbose=False
            )

    @staticmethod
    def _extract_json(text: str) -> dict:
        import re, json
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S|re.I)
        if not m:
            m = re.search(r"(\{.*\})", text, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}

    def ask_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 768) -> dict:
        self._ensure()
        llm = self._llm
        try:
            out = llm.create_chat_completion(
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}],
                temperature=temperature, max_tokens=max_tokens
            )
            text = out["choices"][0]["message"]["content"]
            js = self._extract_json(text)
            if js: return js
        except Exception:
            pass
        try:
            prompt = f"<<SYS>>{system_prompt}<<SYS>>\n\n{user_prompt}\n\nJSON ONLY:"
            out = llm(prompt=prompt, temperature=temperature, max_tokens=max_tokens, stop=["\n\n"])
            text = out["choices"][0]["text"]
            js = self._extract_json(text)
            if js: return js
        except Exception:
            pass
        return {}

# ---- LLM hours inference ----
def infer_hours_and_overrides_from_geo(geo: dict, params: dict | None = None, rates: dict | None = None) -> dict:
    """
    Ask the local LLM to estimate HOURS for each process + a few knobs.
    Returns a dict with 'hours', 'setups', 'inspection', 'notes'.
    """
    params = params or {}
    rates = rates or {}

    system = (
        "You are a senior manufacturing estimator in a tool & die/CNC job shop. "
        "Given CAD GEO features (mm/mm2/mm3), estimate realistic HOURS for a small lot of machined parts. "
        "Prefer simple, conservative estimates. Output ONLY JSON."
    )

    # keep schema explicit and small; minutes only where noted
    schema = {
        "hours": {
            "Programming_Hours": 0.0,
            "CAM_Programming_Hours": 0.0,
            "Engineering_Hours": 0.0,
            "Fixture_Build_Hours": 0.0,

            "Roughing_Cycle_Time_hr": 0.0,
            "Semi_Finish_Cycle_Time_hr": 0.0,
            "Finishing_Cycle_Time_hr": 0.0,

            "OD_Turning_Hours": 0.0,
            "ID_Bore_Drill_Hours": 0.0,
            "Threading_Hours": 0.0,
            "Cutoff_Hours": 0.0,

            "WireEDM_Hours": 0.0,
            "SinkerEDM_Hours": 0.0,

            "Grinding_Surface_Hours": 0.0,
            "Grinding_ODID_Hours": 0.0,
            "Jig_Grind_Hours": 0.0,

            "Lapping_Hours": 0.0,
            "Deburr_Hours": 0.0,
            "Tumble_Hours": 0.0,
            "Blast_Hours": 0.0,
            "Laser_Mark_Hours": 0.0,
            "Masking_Hours": 0.0,

            "InProcess_Inspection_Hours": 0.0,
            "Final_Inspection_Hours": 0.0,
            "CMM_Programming_Hours": 0.0,
            "CMM_RunTime_min": 0.0,  # minutes

            "Saw_Waterjet_Hours": 0.0,
            "Assembly_Hours": 0.0,
            "Packaging_Labor_Hours": 0.0,

            "EHS_Hours": 0.0
        },
        "setups": {
            "Milling_Setups": 1,
            "Setup_Hours_per_Setup": 0.3
        },
        "inspection": {
            "FAIR_Required": False,
            "Source_Inspection_Required": False
        },
        "notes": []
    }

    prompt = f'''
GEO (mm / mm2 / mm3):
{json.dumps(geo, indent=2)}

Rules of thumb:
- Small rectangular blocks with few features: Programming 0.2ï¿½1.0 hr, CAM 0.2ï¿½1.0 hr, Engineering 0 hr.
- Use setups 1ï¿½2 unless 3-axis accessibility is low (<0.6) or faces with unique normals > 4.
- Deburr 0.1ï¿½0.4 hr unless thin walls/freeform edges (then up to 0.8 hr).
- Final inspection ~0.2ï¿½0.6 hr; CMM only if tolerances < 0.02 mm or GD&T heavy.
- Grinding/EDM/Turning hours should be 0 unless features clearly require them.
- Never return huge numbers for tiny parts (<80 mm max dim).

Return JSON with this structure (numbers only, minutes only for CMM_RunTime_min):
{json.dumps(schema, indent=2)}
'''

    # try local GGUF if available
    try:
        mp = os.environ.get("QWEN_GGUF_PATH")
        if mp and Path(mp).is_file():
            q = _LocalLLM(mp)
            out = q.ask_json(system, prompt, temperature=0.1, max_tokens=1024)
            if isinstance(out, dict) and "hours" in out:
                return out
    except Exception:
        pass

    # Fallback heuristics if LLM unavailable
    faces = float(geo.get("GEO__Face_Count", 0) or 0)
    max_dim = float(geo.get("GEO__MaxDim_mm", 0) or 0)
    small_simple = (max_dim and max_dim <= 80.0 and faces < 300)

    hours = {k: 0.0 for k in schema["hours"]}
    if small_simple:
        hours["Programming_Hours"] = 0.5
        hours["CAM_Programming_Hours"] = 0.5
        hours["Engineering_Hours"] = 0.0
        hours["Roughing_Cycle_Time_hr"] = 0.5
        hours["Finishing_Cycle_Time_hr"] = 0.2
        hours["Deburr_Hours"] = 0.2
        hours["Final_Inspection_Hours"] = 0.2
        setups = {"Milling_Setups": 1, "Setup_Hours_per_Setup": 0.3}
    else:
        setups = {"Milling_Setups": 2, "Setup_Hours_per_Setup": 0.4}

    return {"hours": hours, "setups": setups, "inspection": {"FAIR_Required": False, "Source_Inspection_Required": False}, "notes": ["heuristic fallback"]}

def clamp_llm_hours(est: dict, geo: dict, params: dict | None = None) -> dict:
    params = params or {}
    h = est.get("hours", {}).copy()

    max_dim = float(geo.get("GEO__MaxDim_mm", 0) or 0)
    setups  = int(round(est.get("setups", {}).get("Milling_Setups", 1)))
    simple  = (max_dim and max_dim <= 80.0 and setups <= 2)

    # default caps (you can expose these in Overrides UI)
    caps_simple = {
        "Programming_Hours": 1.0, "CAM_Programming_Hours": 1.0, "Engineering_Hours": 1.0,
        "Roughing_Cycle_Time_hr": 1.0, "Finishing_Cycle_Time_hr": 0.6, "Semi_Finish_Cycle_Time_hr": 0.4,
        "Deburr_Hours": 0.6, "Final_Inspection_Hours": 0.6,
        "Assembly_Hours": 0.5
    }
    caps_general = {
        "Programming_Hours": 4.0, "CAM_Programming_Hours": 4.0, "Engineering_Hours": 8.0,
        "Deburr_Hours": 2.0, "Final_Inspection_Hours": 2.0,
    }

    for k, v in list(h.items()):
        v = max(0.0, float(v or 0))
        cap = (caps_simple if simple else caps_general).get(k, None)
        if cap is not None:
            v = min(v, cap)
        h[k] = v

    # Minutes sanity
    if "CMM_RunTime_min" in h:
        h["CMM_RunTime_min"] = max(0.0, min(float(h["CMM_RunTime_min"] or 0), 120.0))

    est["hours"] = h
    return est

def apply_llm_hours_to_variables(df, est: dict, allow_overwrite_nonzero=False, log: dict | None = None):
    """
    Writes LLM hour estimates into the Variables df.
    Only touches time rows; if allow_overwrite_nonzero=False, it will skip rows that already have a non-zero value.
    """
    import pandas as pd
    mapping = {
        "Programming_Hours":              "Programming Hours",
        "CAM_Programming_Hours":          "CAM Programming Hours",
        "Engineering_Hours":              "Engineering (Docs/Fixture Design) Hours",
        "Fixture_Build_Hours":            "Fixture Build Hours",

        "Roughing_Cycle_Time_hr":         "Roughing Cycle Time",
        "Semi_Finish_Cycle_Time_hr":      "Semi-Finish Cycle Time",
        "Finishing_Cycle_Time_hr":        "Finishing Cycle Time",

        "OD_Turning_Hours":               "OD Turning Hours",
        "ID_Bore_Drill_Hours":            "ID Boring/Drilling Hours",
        "Threading_Hours":                "Threading Hours",
        "Cutoff_Hours":                   "Cut-Off / Parting Hours",

        "WireEDM_Hours":                  "WEDM Hours",
        "SinkerEDM_Hours":                "Sinker EDM Hours",

        "Grinding_Surface_Hours":         "Grinding (Surface) Hours",
        "Grinding_ODID_Hours":            "Grinding (OD/ID) Hours",
        "Jig_Grind_Hours":                "Jig Grind Hours",

        "Lapping_Hours":                  "Lapping Hours",
        "Deburr_Hours":                   "Deburr Hours",
        "Tumble_Hours":                   "Tumbling Hours",
        "Blast_Hours":                    "Bead Blasting Hours",
        "Laser_Mark_Hours":               "Laser Mark Hours",
        "Masking_Hours":                  "Masking Hours",

        "InProcess_Inspection_Hours":     "In-Process Inspection Hours",
        "Final_Inspection_Hours":         "Final Inspection Hours",
        "CMM_Programming_Hours":          "CMM Programming Hours",
        "CMM_RunTime_min":                "CMM Run Time min",

        "Saw_Waterjet_Hours":             "Sawing Hours",
        "Assembly_Hours":                 "Assembly Hours",
        "Packaging_Labor_Hours":          "Packaging Labor Hours",
        "EHS_Hours":                      "EHS Training/Compliance Hours"
    }

    def upsert(name, value, dtype="number"):
        nonlocal df
        # ensure core columns exist
        for c in ("Item", "Example Values / Options", "Data Type / Input Method"):
            if c not in df.columns: df[c] = ""
        m = df["Item"].astype(str).str.fullmatch(str(name), case=False, na=False)
        if m.any():
            cur = pd.to_numeric(df.loc[m, "Example Values / Options"], errors="coerce").fillna(0.0).iloc[0]
            if (not allow_overwrite_nonzero) and (float(cur) != 0.0):
                return False
            df.loc[m, "Example Values / Options"] = value
            df.loc[m, "Data Type / Input Method"] = dtype
        else:
            row = {c: None for c in df.columns}
            row["Item"] = name
            row["Example Values / Options"] = value
            row["Data Type / Input Method"] = dtype
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return True

    applied = {}
    for k, v in (est.get("hours") or {}).items():
        item = mapping.get(k)
        if not item: continue
        ok = upsert(item, round(float(v or 0), 3), "number")
        if ok: applied[item] = v

    # Setups
    setups = est.get("setups") or {}
    if "Milling_Setups" in setups:
        upsert("Number of Milling Setups", int(setups["Milling_Setups"]), "number")
    if "Setup_Hours_per_Setup" in setups:
        upsert("Setup Hours / Setup", round(float(setups["Setup_Hours_per_Setup"]), 3), "number")

    # Inspection flags ? set to 0/1 as numbers
    insp = est.get("inspection", {}) or {}
    if "FAIR_Required" in insp:
        upsert("FAIR Required", 1 if insp.get("FAIR_Required") else 0, "number")
    if "Source_Inspection_Required" in insp:
        upsert("Source Inspection Requirement", 1 if insp.get("Source_Inspection_Required") else 0, "number")

    if log is not None:
        log["llm_hours_applied"] = applied
    return df

# --- WHICH SHEET ROWS MATTER TO THE ESTIMATOR --------------------------------
def _estimator_patterns():
    pats = [
        r"\b(Qty|Lot Size|Quantity)\b",
        r"\b(Overhead|Shop Overhead)\b", r"\b(Margin|Profit Margin)\b",
        r"\b(G&A|General\s*&\s*Admin)\b", r"\b(Contingency|Risk\s*Adder)\b",
        r"\b(Expedite|Rush\s*Fee)\b",
        r"\b(Net\s*Volume|Volume_net|Volume\s*\(cm\^?3\))\b",
        r"\b(Density|Material\s*Density)\b", r"\b(Scrap\s*%|Expected\s*Scrap)\b",
        r"\b(Material\s*Price.*(per\s*g|/g)|Unit\s*Price\s*/\s*g)\b",
        r"\b(Supplier\s*Min\s*Charge|min\s*charge)\b",
        r"\b(Material\s*MOQ)\b", r"\b(Material\s*Surcharge|Volatility)\b",
        r"\b(Material\s*Cost|Raw\s*Material\s*Cost)\b",

        r"(Programming|CAM\s*Programming|2D\s*CAM|3D\s*CAM|Simulation|Verification|DFM\s*Review|Tool\s*Library|Setup\s*Sheets)",
        r"\b(CAM\s*Programming|CAM\s*Sim|Post\s*Processing)\b",
        r"(Fixture\s*Design|Process\s*Sheet|Traveler|Documentation|Complex\s*Assembly\s*Doc)",

        r"(Fixture\s*Build|Custom\s*Fixture\s*Build)", r"(Fixture\s*Material\s*Cost|Fixture\s*Hardware)",

        r"(Roughing\s*Cycle\s*Time|Adaptive|HSM)", r"(Semi[- ]?Finish|Rest\s*Milling)", r"(Finishing\s*Cycle\s*Time)",
        r"(Number\s*of\s*Milling\s*Setups|Milling\s*Setups)", r"(Setup\s*Time\s*per\s*Setup|Setup\s*Hours\s*/\s*Setup)",
        r"(Thin\s*Wall\s*Factor|Thin\s*Wall\s*Multiplier)", r"(Tolerance\s*Multiplier|Tight\s*Tolerance\s*Factor)",
        r"(Finish\s*Multiplier|Surface\s*Finish\s*Factor)",

        r"(OD\s*Turning|OD\s*Rough/Finish|Outer\s*Diameter)", r"(ID\s*Boring|Drilling|Reaming)",
        r"(Threading|Tapping|Single\s*Point)", r"(Cut[- ]?Off|Parting)",

        r"(WEDM\s*Hours|Wire\s*EDM\s*Hours|EDM\s*Burn\s*Time)",
        r"(EDM\s*Length_mm|WEDM\s*Length_mm|EDM\s*Perimeter_mm)",
        r"(EDM\s*Thickness_mm|Stock\s*Thickness_mm)", r"(EDM\s*Passes|WEDM\s*Passes)",
        r"(EDM\s*Cut\s*Rate_mm\^?2/min|WEDM\s*Cut\s*Rate)", r"(EDM\s*Edge\s*Factor)",
        r"(WEDM\s*Wire\s*Cost\s*/\s*m|Wire\s*Cost\s*/m)", r"(Wire\s*Usage\s*m\s*/\s*mm\^?2)",

        r"(Sinker\s*EDM\s*Hours|Ram\s*EDM\s*Hours|Burn\s*Time)", r"(Sinker\s*Burn\s*Volume_mm3|EDM\s*Volume_mm3)",
        r"(Sinker\s*MRR_mm\^?3/min|EDM\s*MRR)", r"(Electrode\s*Count)", r"(Electrode\s*(Cost|Material).*)",

        r"(Surface\s*Grind|Pre[- ]?Op\s*Grinding|Blank\s*Squaring)", r"(Jig\s*Grind)",
        r"(OD/ID\s*Grind|Cylindrical\s*Grind)", r"(Grind\s*Volume_mm3|Grinding\s*Volume)",
        r"(Grind\s*MRR_mm\^?3/min|Grinding\s*MRR)", r"(Grinding\s*Passes)",
        r"(Dress\s*Frequency\s*passes)", r"(Dress\s*Time\s*min(\s*/\s*pass)?)",
        r"(Grinding\s*Wheel\s*Cost)",

        r"(Lapping|Honing|Polishing)",

        r"(Deburr|Edge\s*Break)", r"(Tumbling|Vibratory)", r"(Bead\s*Blasting|Sanding)",
        r"(Laser\s*Mark|Engraving)", r"(Masking\s*for\s*Plating|Masking)",

        r"(In[- ]?Process\s*Inspection)", r"(Final\s*Inspection|Manual\s*Inspection)",
        r"(CMM\s*Programming)", r"(CMM\s*Run\s*Time\s*min)", r"(CMM\s*Run\s*Time)\b",
        r"(FAIR|ISIR|PPAP)", r"(Source\s*Inspection)",

        r"(Sawing|Waterjet|Blank\s*Prep)",

        r"(Assembly|Manual\s*Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)", r"(Hardware|BOM\s*Cost|Fasteners|Bearings|CFM\s*Hardware)",

        r"(Outsourced\s*Heat\s*Treat|Heat\s*Treat\s*Cost)", r"(Plating|Coating|Anodize|Black\s*Oxide|DLC|PVD|CVD)",
        r"(Passivation|Cleaning\s*Vendor)",

        r"(Packaging|Boxing|Crating\s*Labor)", r"(Custom\s*Crate\s*NRE)", r"(Packaging\s*Materials|Foam|Trays)",
        r"(Freight|Shipping\s*Cost)", r"(Insurance|Liability\s*Adder)",

        r"(EHS|Compliance|Training|Waste\s*Handling)", r"(Gauge|Check\s*Fixture\s*NRE)",
    ]
    import re
    return [re.compile(p, re.I) for p in pats]

def _items_used_by_estimator(df):
    pats = _estimator_patterns()
    items = df["Item"].astype(str)
    used = []
    import re as _re
    for it in items:
        if any(p.search(it) for p in pats):
            # Skip explicit Rate rows to prevent LLM from editing machine rates
            if _re.search(r"\brate\b", it, _re.I):
                continue
            used.append(it)
    return used

# --- Build prompt + run model -----------------------------------------------
def _parse_pct_like(x):
    try:
        v = float(x)
        return v/100.0 if v > 1.0 else v
    except Exception:
        return None

def build_llm_sheet_prompt(geo: dict, allowed_items: list[str], params_snapshot: dict) -> tuple[str, str]:
    sys = (
        "You are a manufacturing estimator for precision machining (milling/turning/EDM/grinding). "
        "You will receive:\n" 
        "1) GEO_* features extracted from CAD (units in mm/mm^2/mm^3), and\n" 
        "2) a list of allowed Variables sheet row names you may edit.\n" 
        "Return STRICT JSON only.\n" 
        "Rules:\n" 
        "- Only edit rows that are in the provided allowed_items list.\n" 
        "- Use numbers only (no text). Time is in HOURS unless the row name includes 'min'.\n" 
        "- Pass counts/integers for fields like '... Passes' or '... Count'.\n" 
        "- Percents must be decimals (e.g., 0.20 for 20%).\n" 
        "- Be conservative; if you are unsure, skip the edit.\n" 
        "- You may also suggest param nudges in 'params' for: OEE_EfficiencyPct, FiveAxisMultiplier, TightToleranceMultiplier,\n" 
        "  MillingConsumablesPerHr, TurningConsumablesPerHr, EDMConsumablesPerHr, GrindingConsumablesPerHr, InspectionConsumablesPerHr,\n" 
        "  UtilitiesPerSpindleHr, ConsumablesFlat. Do not include other keys.\n"
    )
    import json, textwrap
    u = {
        "geo": geo,
        "allowed_items": allowed_items,
        "params_snapshot": {k: params_snapshot.get(k) for k in [
            "OEE_EfficiencyPct","FiveAxisMultiplier","TightToleranceMultiplier",
            "MillingConsumablesPerHr","TurningConsumablesPerHr","EDMConsumablesPerHr",
            "GrindingConsumablesPerHr","InspectionConsumablesPerHr",
            "UtilitiesPerSpindleHr","ConsumablesFlat"
        ]},
        "output_shape": {
            "sheet_edits": [{"item": "<exact string from allowed_items>", "value": 0.0, "why": "<short reason, optional>"}],
            "params": {
                "OEE_EfficiencyPct": 1.0,
                "FiveAxisMultiplier": 1.0,
                "TightToleranceMultiplier": 1.0,
                "MillingConsumablesPerHr": 0.0,
                "TurningConsumablesPerHr": 0.0,
                "EDMConsumablesPerHr": 0.0,
                "GrindingConsumablesPerHr": 0.0,
                "InspectionConsumablesPerHr": 0.0,
                "UtilitiesPerSpindleHr": 0.0,
                "ConsumablesFlat": 0.0
                # you may also include optional reasons per param as:
                # "_why": {"OEE_EfficiencyPct":"...", "FiveAxisMultiplier":"...", ...}
            }
        }
    }
    user = textwrap.dedent(
        "Given the following, produce JSON ONLY in the shape shown above. "
        "Prefer these typical relationships:\n" 
        "- WEDM: more path length/thickness -> higher 'EDM Passes' (int) or 'WEDM Hours'; adjust 'EDM Cut Rate_mm^2/min' if too aggressive.\n" 
        "- Grinding: large grind volume -> adjust 'Grind MRR_mm^3/min', 'Grinding Passes', and 'Dress Frequency passes' / 'Dress Time min'.\n" 
        "- Milling setups: high face count or multiple normals -> increase 'Number of Milling Setups' and 'Setup Hours / Setup'.\n" 
        "- Small min wall or high area/volume -> consider 'Thin Wall Factor' and 'Tolerance Multiplier'.\n" 
        "- Inspection: very small parts or tight tolerance -> bump 'CMM Run Time min' or 'Final Inspection'.\n" 
        "- Only include edits you are confident about.\n\n"
        f"INPUT:\n{json.dumps(u, ensure_ascii=False)}"
    )
    return sys, user

def llm_sheet_and_param_overrides(geo: dict, df, params: dict, model_path: str) -> dict:
    allowed = _items_used_by_estimator(df)
    if not allowed:
        return {"sheet_edits": [], "params": {}, "meta": {}}

    sys, usr = build_llm_sheet_prompt(geo, allowed, params)
    prompt_sha = hashlib.sha256((sys + "\n" + usr).encode("utf-8")).hexdigest()

    error_text = ""
    try:
        llm = _LocalLLM(model_path)
        js = llm.ask_json(sys, usr, temperature=0.15, max_tokens=900)
        model_name = Path(model_path).name
    except Exception as e:
        js, model_name = {}, "LLM-unavailable"
        try:
            error_text = f"{type(e).__name__}: {e}"
        except Exception:
            error_text = "LLM error"

    sheet_edits = []
    for e in js.get("sheet_edits", []):
        item = str(e.get("item","")); item = item.strip()
        if item and item in allowed:
            val = e.get("value", None)
            why = e.get("why", "").strip() if isinstance(e.get("why",""), str) else ""
            if isinstance(val, (int, float, str)):
                try:
                    v = float(val)
                except Exception:
                    v = _parse_pct_like(val)
                    if v is None:
                        continue
                sheet_edits.append({"item": item, "value": v, "why": why})

    param_allow = {
        "OEE_EfficiencyPct","FiveAxisMultiplier","TightToleranceMultiplier",
        "MillingConsumablesPerHr","TurningConsumablesPerHr","EDMConsumablesPerHr",
        "GrindingConsumablesPerHr","InspectionConsumablesPerHr",
        "UtilitiesPerSpindleHr","ConsumablesFlat"
    }
    pmap = js.get("params", {}) if isinstance(js.get("params", {}), dict) else {}
    param_whys = pmap.get("_why", {}) if isinstance(pmap.get("_why", {}), dict) else {}
    param_edits = {}
    for k, v in pmap.items():
        if k == "_why":
            continue
        if k in param_allow:
            try:
                param_edits[k] = float(v)
            except Exception:
                pv = _parse_pct_like(v)
                if pv is not None:
                    param_edits[k] = pv

    meta = {"model": model_name, "prompt_sha256": prompt_sha, "allowed_items_count": len(allowed), "error": error_text}
    return {"sheet_edits": sheet_edits, "params": param_edits, "param_whys": param_whys, "allowed_items": allowed, "meta": meta}

# --- APPLY LLM OUTPUT ---------------------------------------------------------
def apply_sheet_edits_to_df(df, sheet_edits: list[dict]):
    if not sheet_edits:
        return df, []
    items = df["Item"].astype(str)
    applied = []
    for e in sheet_edits:
        item = e["item"]; value = e["value"]
        m = items.str.fullmatch(item, case=False, na=False)
        if m.any():
            i = df.index[m][0]
            df.at[i, "Example Values / Options"] = value
            df.at[i, "Data Type / Input Method"] = "number"
            applied.append((item, value))
    return df, applied

def apply_param_edits_to_overrides_ui(self, param_edits: dict):
    if not param_edits:
        return
    for k, v in param_edits.items():
        if hasattr(self, "param_vars") and k in self.param_vars:
            self.param_vars[k].set(str(v))
        else:
            self.params[k] = v
    if hasattr(self, "apply_overrides"):
        try:
            self.apply_overrides()
            return
        except Exception:
            pass
    # Fallback normalization if apply_overrides is unavailable
    def fnum(s, d=0.0):
        try: return float(str(s).strip())
        except Exception: return d
    p = {k:fnum(self.param_vars[k].get(), self.params.get(k,0.0)) for k in getattr(self, 'param_vars', {}).keys()}
    for kk in list(p.keys()):
        if kk.endswith("Pct"):
            params[kk] = parse_pct(params[kk])
    self.params.update(p)

# ================== LLM DECISION LOG / AUDIT ==================
import hashlib, json as _json_audit, time as _time_audit
LOGS_DIR = Path(r"D:\\CAD_Quoting_Tool\\Logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _now_iso():
    return _time_audit.strftime("%Y-%m-%dT%H:%M:%S")

def _coerce_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return x

def _used_item_values(df, used_items):
    vals = {}
    items = df["Item"].astype(str)
    for name in used_items:
        m = items.str.fullmatch(name, case=False, na=False)
        if m.any():
            v = df.loc[m, "Example Values / Options"].iloc[0]
            vals[name] = _coerce_num(v)
    return vals

def _diff_map(before: dict, after: dict):
    changes = []
    keys = set(before.keys()) | set(after.keys())
    for k in sorted(keys):
        bv = before.get(k, None)
        av = after.get(k, None)
        if bv != av:
            changes.append({"key": k, "before": bv, "after": av})
    return changes

def render_llm_log_text(log: dict) -> str:
    m  = lambda v: f"${{float(v):,.2f}}" if isinstance(v, (int, float)) else str(v)
    lines = []
    lines.append(f"LLM DECISION LOG ï¿½ {log['timestamp']}")
    lines.append(f"Model: {log.get('model','?')}  |  Prompt SHA256: {log.get('prompt_sha256','')[:12]}ï¿½")
    if log.get('llm_error'):
        lines.append(f"LLM error: {log['llm_error']}")
    lines.append(f"Allowed items: {log.get('allowed_items_count',0)} | Sheet edits suggested/applied: {log.get('sheet_edits_suggested',0)}/{log.get('sheet_edits_applied',0)}")
    lines.append("")
    lines.append(f"Price before: {m(log['price_before'])}   ?   after: {m(log['price_after'])}   " 
                 f"?: {m(log['price_after']-log['price_before'])} " 
                 f"({(((log['price_after']-log['price_before'])/log['price_before']*100) if log['price_before'] else 0):.2f}%)")
    lines.append("")
    if log.get("sheet_changes"):
        lines.append("Sheet changes:")
        for ch in log["sheet_changes"]:
            why = f"  ï¿½ why: {ch['why']}" if ch.get("why") else ""
            lines.append(f"  ï¿½ {ch['key']}: {ch['before']} ? {ch['after']}{why}")
        lines.append("")
    if log.get("param_changes"):
        lines.append("Param nudges:")
        for ch in log["param_changes"]:
            why = f"  ï¿½ why: {ch['why']}" if ch.get("why") else ""
            lines.append(f"  ï¿½ {ch['key']}: {ch['before']} ? {ch['after']}{why}")
        lines.append("")
    if log.get("geo_summary"):
        lines.append("GEO summary:")
        for k, v in log["geo_summary"]:
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines)
def save_llm_log_json(log: dict) -> Path:
    fname = f"llm_log_{_time_audit.strftime('%Y%m%d_%H%M%S')}.json"
    path = LOGS_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        _json_audit.dump(log, f, indent=2)
    return path
# =============================================================

# ----------------- Variables & quote -----------------
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

CORE_COLS = ["Item", "Example Values / Options", "Data Type / Input Method"]

def _coerce_core_types(df_core: "pd.DataFrame") -> "pd.DataFrame":
    """Light normalization for estimator expectations."""
    core = df_core.copy()
    core["Item"] = core["Item"].astype(str)
    core["Data Type / Input Method"] = core["Data Type / Input Method"].astype(str).str.lower()
    # Leave "Example Values / Options" as-is (can be text or number); estimator coerces later.
    return core

def sanitize_vars_df(df_full: "pd.DataFrame") -> "pd.DataFrame":
    """
    Return a copy containing only the 3 core columns the estimator needs.
    - Does NOT mutate or overwrite the original file.
    - Creates missing core columns as blanks.
    - Normalizes types.
    """
    # Try to map any variant header names to our canon names (case/space tolerant)
    canon = {str(c).strip().lower(): c for c in df_full.columns}

    # Build list of the *actual* columns that correspond to CORE_COLS (if present)
    actual = []
    for want in CORE_COLS:
        key = want.strip().lower()
        # allow loose matches for common variants
        candidates = [
            canon.get(key),
            canon.get(key.replace(" / ", " ").replace("/", " ")),
            canon.get(key.replace(" ", "")),
        ]
        col = next((c for c in candidates if c in df_full.columns), None)
        actual.append(col)

    # Start with what we can find; add any missing columns as empty
    core = pd.DataFrame()
    for want, col in zip(CORE_COLS, actual):
        if col is not None:
            core[want] = df_full[col]
        else:
            core[want] = "" if want != "Example Values / Options" else None

    return _coerce_core_types(core)

def read_variables_file(path: str, return_full: bool = False):
    """
    Read .xlsx/.csv, keep original data intact, and return a sanitized copy for the estimator.
    - If return_full=True, returns (core_df, full_df); otherwise returns core_df only.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas required (conda/pip install pandas)")

    lp = path.lower()
    if lp.endswith(".xlsx"):
        # Prefer a sheet named "Variables" if it exists; else first sheet
        xl = pd.ExcelFile(path)
        sheet_name = "Variables" if "Variables" in xl.sheet_names else xl.sheet_names[0]
        df_full = pd.read_excel(path, sheet_name=sheet_name)
    elif lp.endswith(".csv"):
        df_full = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError("Variables must be .xlsx or .csv")

    core = sanitize_vars_df(df_full)

    return (core, df_full) if return_full else core

def find_variables_near(cad_path: str):
    """Look for variables.* in the same folder, then one level up."""
    import os
    folder = os.path.dirname(cad_path)
    names = ["variables.xlsx", "variables.csv"]
    subs  = ["variables", "vars"]

    def _scan(dirpath):
        try:
            listing = os.listdir(dirpath)
        except Exception:
            return None
        low = {e.lower(): e for e in listing}
        for n in names:
            if n in low:
                return os.path.join(dirpath, low[n])
        for e in listing:
            le = e.lower()
            if le.endswith((".xlsx", ".csv")) and any(s in le for s in subs):
                return os.path.join(dirpath, e)
        return None

    hit = _scan(folder)
    if hit:
        return hit
    parent = os.path.dirname(folder)
    if os.path.isdir(parent):
        return _scan(parent)
    return None


def parse_pct(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v/100.0 if v > 1.0 else v

# ---------- Pretty printer for quote results ----------
def render_quote(
    result: dict,
    currency: str = "$",
    show_zeros: bool = False,
    llm_explanation: str = "",
    page_width: int = 74,
) -> str:
    breakdown = result.get("breakdown", {}) or {}
    price = float(result.get("price", 0.0))
    qty = int(breakdown.get("qty", 1) or 1)

    def _m(x) -> str:
        return f"{currency}{float(x):,.2f}"

    def _h(x) -> str:
        return f"{float(x):.2f} hr"

    def _pct(x) -> str:
        return f"{float(x or 0.0) * 100:.1f}%"

    totals = breakdown.get("totals", {}) or {}
    nre_detail = breakdown.get("nre_detail", {}) or {}
    prog_detail = nre_detail.get("programming", {}) or {}
    fix_detail = nre_detail.get("fixture", {}) or {}
    process_costs = breakdown.get("process_costs", {}) or {}
    pass_through = breakdown.get("pass_through", {}) or {}
    applied_pcts = breakdown.get("applied_pcts", {}) or {}

    process_meta = {str(k).lower(): v or {} for k, v in (breakdown.get("process_meta", {}) or {}).items()}
    pass_meta = {str(k).lower(): v or {} for k, v in (breakdown.get("pass_meta", {}) or {}).items()}

    lines: list[str] = []
    divider = "-" * page_width

    def write_line(left: str, right: str = "", indent: str = "") -> None:
        left = f"{indent}{left}"
        right = str(right)
        if not right:
            lines.append(left)
            return
        max_left = max(1, page_width - len(right) - 1)
        if len(left) > max_left:
            left = left[: max_left - 1] + "..."
        lines.append(left + " " * (page_width - len(left) - len(right)) + right)

    def row(label: str, value, indent: str = "") -> None:
        write_line(label, _m(value), indent)

    def add_process_notes(key: str, indent: str = "  ") -> None:
        meta = process_meta.get(key) or process_meta.get(key.replace('_', ' ').title())
        if not meta:
            return
        hr = float(meta.get("hr") or 0.0)
        rate = float(meta.get("rate") or 0.0)
        if hr:
            note = _h(hr)
            if rate:
                note += f" @ {_m(rate)}/hr"
            write_line(note, indent=indent + "    ")

    def add_pass_basis(key: str, indent: str = "  ") -> None:
        meta = pass_meta.get(key) or pass_meta.get(key.replace('_', ' ').title())
        if not meta:
            return
        basis = meta.get("basis")
        if basis:
            write_line(basis, indent=indent + "    ")

    # Header
    lines.append(f"QUOTE SUMMARY - Qty {qty}")
    lines.append(divider)
    row("Final Price per Part:", price)
    row("Total Labor Cost:", totals.get("labor_cost", 0.0))
    row("Total Direct Costs:", totals.get("direct_costs", 0.0))
    lines.append("")

    # NRE / Setup
    lines.append("NRE / Setup Costs (per lot)")
    lines.append(divider)
    if prog_detail:
        row("Programming & Eng:", prog_detail.get("per_lot", 0.0))
        if prog_detail.get('prog_hr'):
            write_line(f"- Programmer: {_h(prog_detail['prog_hr'])} @ {_m(prog_detail.get('prog_rate', 0))}/hr", indent="    ")
        if prog_detail.get('cam_hr'):
            write_line(f"- CAM: {_h(prog_detail['cam_hr'])} @ {_m(prog_detail.get('cam_rate', 0))}/hr", indent="    ")
    if fix_detail:
        row("Fixturing:", fix_detail.get("per_lot", 0.0))
        if fix_detail.get('build_hr'):
            write_line(f"- Build Labor: {_h(fix_detail['build_hr'])} @ {_m(fix_detail.get('build_rate', 0))}/hr", indent="    ")
        write_line(f"- Materials: {_m(fix_detail.get('mat_cost', 0))}", indent="    ")
    lines.append("")

    # Process & Labor
    lines.append("Process & Labor Costs")
    lines.append(divider)
    proc_total = 0.0
    for key, value in sorted(process_costs.items(), key=lambda kv: kv[1], reverse=True):
        if value > 0 or show_zeros:
            label = key.replace('_', ' ').title()
            row(label, value, indent="  ")
            add_process_notes(key, indent="  ")
            proc_total += float(value)
    row("Total", proc_total, indent="  ")
    lines.append("")

    # Pass-Through
    lines.append("Pass-Through & Direct Costs")
    lines.append(divider)
    pass_total = 0.0
    for key, value in pass_through.items():
        if value > 0 or show_zeros:
            label = key.replace('_', ' ').replace('hr', '/hr').title()
            row(label, value, indent="  ")
            add_pass_basis(key, indent="  ")
            pass_total += float(value)
    row("Total", pass_total, indent="  ")
    lines.append("")

    # Pricing ladder
    lines.append("Pricing Ladder")
    lines.append(divider)
    subtotal = float(totals.get("subtotal", 0.0))
    with_overhead = float(totals.get("with_overhead", subtotal))
    with_ga = float(totals.get("with_ga", with_overhead))
    with_contingency = float(totals.get("with_contingency", with_ga))
    with_expedite = float(totals.get("with_expedite", with_contingency))

    row("Subtotal (Labor + Directs):", subtotal)
    row(f"+ Overhead ({_pct(applied_pcts.get('OverheadPct'))}):", with_overhead - subtotal)
    row(f"+ G&A ({_pct(applied_pcts.get('GA_Pct'))}):", with_ga - with_overhead)
    row(f"+ Contingency ({_pct(applied_pcts.get('ContingencyPct'))}):", with_contingency - with_ga)
    if applied_pcts.get('ExpeditePct'):
        row(f"+ Expedite ({_pct(applied_pcts.get('ExpeditePct'))}):", with_expedite - with_contingency)
    row("= Subtotal before Margin:", with_expedite)
    row(f"Final Price with Margin ({_pct(applied_pcts.get('MarginPct'))}):", price)
    lines.append("")

    if llm_explanation:
        lines.append("Why This Price")
        lines.append(divider)
        import textwrap
        for snippet in textwrap.wrap(llm_explanation.strip(), width=page_width):
            lines.append(snippet)

    return "\n".join(lines)
# ===== QUOTE CONFIG (edit-friendly) ==========================================
RATES_DEFAULT = {
    "ProgrammingRate": 120.0,
    "MillingRate": 120.0,
    "TurningRate": 115.0,
    "WireEDMRate": 140.0,
    "SinkerEDMRate": 150.0,
    "SurfaceGrindRate": 120.0,
    "ODIDGrindRate": 125.0,
    "JigGrindRate": 150.0,
    "LappingRate": 130.0,
    "InspectionRate": 110.0,
    "FinishingRate": 100.0,
    "SawWaterjetRate": 100.0,
    "FixtureBuildRate": 120.0,
    "AssemblyRate": 110.0,
    "EngineerRate": 140.0,
    "CAMRate": 125.0,
}

PARAMS_DEFAULT = {
    "OverheadPct": 0.15,
    "GA_Pct": 0.08,
    "ContingencyPct": 0.00,
    "ExpeditePct": 0.00,
    "MarginPct": 0.35,
    "InsurancePct": 0.00,
    "VendorMarkupPct": 0.00,
    "MinLotCharge": 0.00,
    "Quantity": 1,
    # Programming heuristics
    "ProgSimpleDim_mm": 80.0,
    "ProgCapHr": 1.0,
    "ProgMaxToMillingRatio": 1.5,
    # Efficiency & multipliers
    "OEE_EfficiencyPct": 1.0,
    "FiveAxisMultiplier": 1.0,
    "TightToleranceMultiplier": 1.0,
    # Adders
    "MillingConsumablesPerHr": 4.0,
    "TurningConsumablesPerHr": 3.0,
    "EDMConsumablesPerHr": 6.0,
    "GrindingConsumablesPerHr": 3.0,
    "InspectionConsumablesPerHr": 1.0,
    "UtilitiesPerSpindleHr": 2.5,
    "ConsumablesFlat": 35.0,
    # Misc
    "MaterialOther": 50.0,
    "LLMModelPath": "",
}

# Common regex pieces (kept non-capturing to avoid pandas warnings)
TIME_RE = r"\b(?:hours?|hrs?|hr|time|min(?:ute)?s?)\b"
MONEY_RE = r"(?:rate|/hr|per\s*hour|per\s*hr|price|cost|$)"

# ===== QUOTE HELPERS ========================================================

def alt(*terms: str) -> str:
    """Build a non-capturing (?:a|b|c) regex; terms are already escaped/regex-ready."""
    return r"(?:%s)" % "|".join(terms)

def pct(value: Any, default: float = 0.0) -> float:
    """Accept 0-1 or 0-100 and return 0-1."""
    try:
        v = float(value)
        return v / 100.0 if v > 1.0 else v
    except Exception:
        return default

def compute_quote_from_df(df: pd.DataFrame,
                          params: Dict[str, Any] | None = None,
                          rates: Dict[str, float] | None = None) -> Dict[str, Any]:
    """
    Estimator that consumes variables from the sheet (Item, Example Values / Options, Data Type / Input Method).

    Steps:
      1) Tolerant lookup (regex on 'Item') for numbers/percents/strings
      2) Aggregate hours ï¿½ rates per process (milling/turning/EDM/grinding/ï¿½)
      3) Geometry-derived estimates (optional)
      4) NRE, vendors, packaging/shipping/insurance
      5) Overhead ? G&A ? Contingency ? Expedite ? Margin
      6) Return total and a transparent breakdown
    """
    # ---- merge configs (easy to edit) ---------------------------------------

    params = {**PARAMS_DEFAULT, **(params or {})}
    rates = {**RATES_DEFAULT, **(rates or {})}
    # ---- sheet views ---------------------------------------------------------
    items = df["Item"].astype(str)
    vals  = df["Example Values / Options"]
    dtt   = df["Data Type / Input Method"].astype(str).str.lower()
    # --- lookups --------------------------------------------------------------
    def contains(pattern: str):
        return items.str.contains(pattern, case=False, regex=True, na=False)

    def first_num(pattern: str, default: float = 0.0) -> float:
        m = contains(pattern)
        if not m.any():
            return float(default)
        s = pd.to_numeric(vals[m].iloc[0], errors="coerce")
        return float(s) if pd.notna(s) else float(default)

    def num(pattern: str, default: float = 0.0) -> float:
        m = contains(pattern)
        if not m.any():
            return float(default)
        return float(pd.to_numeric(vals[m], errors="coerce").fillna(0.0).sum())

    def strv(pattern: str, default: str = "") -> str:
        m = contains(pattern)
        return (str(vals[m].iloc[0]) if m.any() else default)

    def num_pct(pattern: str, default: float = 0.0) -> float:
        return pct(first_num(pattern, default * 100.0), default)

    def sum_time(pattern: str, default: float = 0.0, exclude_pattern: str | None = None) -> float:
        """Sum only time-like rows; minutes are converted to hours."""
        m  = contains(pattern)
        if not m.any():
            return float(default)

        looks_time   = items.str.contains(TIME_RE,   case=False, regex=True, na=False)
        looks_money  = items.str.contains(MONEY_RE,  case=False, regex=True, na=False)
        typed_money  = dtt.str.contains(r"(?:rate|currency|price|cost)", case=False, na=False)
        excl = looks_money | typed_money
        if exclude_pattern:
            excl = excl | contains(exclude_pattern)

        mm = m & ~excl & looks_time
        if not mm.any():
            return float(default)

        mins_mask = items.str.contains(r"\bmin(?:ute)?s?\b", case=False, regex=True, na=False) & mm
        hrs_mask  = mm & ~mins_mask
        hrs_sum  = pd.to_numeric(vals[hrs_mask],  errors="coerce").fillna(0.0).sum()
        mins_sum = pd.to_numeric(vals[mins_mask], errors="coerce").fillna(0.0).sum()
        return float(hrs_sum) + float(mins_sum) / 60.0

    # --- OPTIONAL: let sheet override params/rates if those rows exist ---
    def sheet_num(pat, default=None):
        v = first_num(pat, float('nan'))
        return v if v == v else default  # NaN check

    def sheet_pct(pat, default=None):
        from math import isnan
        v = first_num(pat, float('nan'))
        if v == v:  # not NaN
            return (v/100.0) if abs(v) > 1.0 else v
        return default

    def sheet_text(pat, default=None):
        t = strv(pat, "").strip()
        return t if t else default

    # Params overrides
    params["OEE_EfficiencyPct"]        = sheet_pct(r"\b(OEE\s*Efficiency\s*%|OEE)\b", params.get("OEE_EfficiencyPct"))
    params["VendorMarkupPct"]          = sheet_pct(r"\b(Vendor\s*Markup\s*%)\b", params.get("VendorMarkupPct"))
    params["MinLotCharge"]             = sheet_num(r"\b(Min\s*Lot\s*Charge)\b", params.get("MinLotCharge"))
    params["ProgCapHr"]                = sheet_num(r"\b(Programming\s*Cap\s*Hr)\b", params.get("ProgCapHr"))
    params["ProgSimpleDim_mm"]         = sheet_num(r"\b(Prog\s*Simple\s*Dim_mm)\b", params.get("ProgSimpleDim_mm"))
    params["ProgMaxToMillingRatio"]    = sheet_num(r"\b(Prog\s*Max\s*To\s*Milling\s*Ratio)\b", params.get("ProgMaxToMillingRatio"))
    params["MillingConsumablesPerHr"]  = sheet_num(r"(?i)Milling\s*Consumables\s*/\s*Hr", params.get("MillingConsumablesPerHr"))
    params["TurningConsumablesPerHr"]  = sheet_num(r"(?i)Turning\s*Consumables\s*/\s*Hr", params.get("TurningConsumablesPerHr"))
    params["EDMConsumablesPerHr"]      = sheet_num(r"(?i)EDM\s*Consumables\s*/\s*Hr", params.get("EDMConsumablesPerHr"))
    params["GrindingConsumablesPerHr"] = sheet_num(r"(?i)Grinding\s*Consumables\s*/\s*Hr", params.get("GrindingConsumablesPerHr"))
    params["InspectionConsumablesPerHr"]= sheet_num(r"(?i)Inspection\s*Consumables\s*/\s*Hr", params.get("InspectionConsumablesPerHr"))
    params["UtilitiesPerSpindleHr"]    = sheet_num(r"(?i)Utilities\s*Per\s*Spindle\s*Hr", params.get("UtilitiesPerSpindleHr"))
    params["ConsumablesFlat"]          = sheet_num(r"(?i)Consumables\s*Flat", params.get("ConsumablesFlat"))

    # Rates overrides
    def rate_from_sheet(label, key):
        v = sheet_num(label, None)
        if v is not None:
            rates[key] = v

    rate_from_sheet(r"Rate\s*-\s*Programming",   "ProgrammingRate")
    rate_from_sheet(r"Rate\s*-\s*CAM",           "CAMRate")
    rate_from_sheet(r"Rate\s*-\s*Engineer",      "EngineerRate")
    rate_from_sheet(r"Rate\s*-\s*Milling",       "MillingRate")
    rate_from_sheet(r"Rate\s*-\s*Turning",       "TurningRate")
    rate_from_sheet(r"Rate\s*-\s*Wire\s*EDM",    "WireEDMRate")
    rate_from_sheet(r"Rate\s*-\s*Sinker\s*EDM",  "SinkerEDMRate")
    rate_from_sheet(r"Rate\s*-\s*Surface\s*Grind","SurfaceGrindRate")
    rate_from_sheet(r"Rate\s*-\s*ODID\s*Grind",  "ODIDGrindRate")
    rate_from_sheet(r"Rate\s*-\s*Jig\s*Grind",   "JigGrindRate")
    rate_from_sheet(r"Rate\s*-\s*Lapping",       "LappingRate")
    rate_from_sheet(r"Rate\s*-\s*Inspection",    "InspectionRate")
    rate_from_sheet(r"Rate\s*-\s*Finishing",     "FinishingRate")
    rate_from_sheet(r"Rate\s*-\s*Saw/Waterjet",  "SawWaterjetRate")
    rate_from_sheet(r"Rate\s*-\s*Fixture\s*Build","FixtureBuildRate")
    rate_from_sheet(r"Rate\s*-\s*Assembly",      "AssemblyRate")
    rate_from_sheet(r"Rate\s*-\s*Packaging",     "PackagingRate")

    # ---- knobs & qty ---------------------------------------------------------
    OverheadPct    = num_pct(r"\b" + alt('Overhead','Shop Overhead') + r"\b", params["OverheadPct"])
    MarginPct      = num_pct(r"\b" + alt('Margin','Profit Margin') + r"\b", params["MarginPct"])
    GA_Pct         = num_pct(r"\b" + alt('G&A','General\s*&\s*Admin') + r"\b", params["GA_Pct"])
    ContingencyPct = num_pct(r"\b" + alt('Contingency','Risk\s*Adder') + r"\b",  params["ContingencyPct"])
    ExpeditePct    = num_pct(r"\b" + alt('Expedite','Rush\s*Fee') + r"\b",      params["ExpeditePct"])

    priority = strv(alt('PM-01_Quote_Priority','Quote\s*Priority'), "").strip().lower()
    if priority not in ("expedite", "critical"):
        ExpeditePct = 0.0

    Qty = max(1, int(first_num(r"\b" + alt('Qty','Lot Size','Quantity') + r"\b", 1)))
    amortize_programming = True  # single switch if you want to expose later

    # ---- geometry (optional) -------------------------------------------------
    GEO_wedm_len_mm = first_num(r"\bGEO__?(?:WEDM_PathLen_mm|EDM_Length_mm)\b", 0.0)
    GEO_vol_mm3     = first_num(r"\bGEO__?(?:Volume|Volume_mm3|Net_Volume_mm3)\b", 0.0)

    # ---- material ------------------------------------------------------------
    vol_cm3       = first_num(r"\b(?:Net\s*Volume|Volume_net|Volume\s*\(cm\^?3\))\b", GEO_vol_mm3 / 1000.0)
    density_g_cc  = first_num(r"\b(?:Density|Material\s*Density)\b", 14.5)
    scrap_pct     = num_pct(r"\b(?:Scrap\s*%|Expected\s*Scrap)\b", 0.0)
    mass_g        = max(0.0, vol_cm3 * density_g_cc * (1.0 + scrap_pct))

    unit_price_per_g  = first_num(r"\b(?:Material\s*Price.*(?:per\s*g|/g)|Unit\s*Price\s*/\s*g)\b", 0.0)
    supplier_min_charge    = first_num(r"\b(?:Supplier\s*Min\s*Charge|min\s*charge)\b", 0.0)
    surcharge_pct = num_pct(r"\b(?:Material\s*Surcharge|Volatility)\b", 0.0)
    explicit_mat  = num(r"\b(?:Material\s*Cost|Raw\s*Material\s*Cost)\b", 0.0)

    material_cost = max(unit_price_per_g * mass_g, supplier_min_charge) * (1.0 + surcharge_pct)
    material_cost = max(material_cost, explicit_mat)

    # ---- programming / cam / dfm --------------------------------------------
    prog_hr = sum_time(r"(?:Programming|2D\s*CAM|3D\s*CAM|Simulation|Verification|DFM|Setup\s*Sheets)", exclude_pattern=r"\bCMM\b")
    cam_hr  = sum_time(r"(?:CAM\s*Programming|CAM\s*Sim|Post\s*Processing)")
    eng_hr  = sum_time(r"(?:Fixture\s*Design|Process\s*Sheet|Traveler|Documentation)")

    # ---- milling hours & caps ------------------------------------------------
    rough_hr   = sum_time(r"(?:Roughing\s*Cycle\s*Time|Adaptive|HSM)")
    semi_hr    = sum_time(r"(?:Semi[- ]?Finish|Rest\s*Milling)")
    finish_hr  = sum_time(r"(?:Finishing\s*Cycle\s*Time)")
    setups     = first_num(r"(?:Number\s*of\s*Milling\s*Setups|Milling\s*Setups)", 0.0)
    setup_each = first_num(r"(?:Setup\s*Time\s*per\s*Setup|Setup\s*Hours\s*/\s*Setup)", 0.5)

    setups = max(1, min(int(round(setups or 1)), 3))
    setup_hr  = setups * setup_each
    milling_hr = rough_hr + semi_hr + finish_hr + setup_hr

    max_dim   = first_num(r"\bGEO__MaxDim_mm\b", 0.0)
    is_simple = (max_dim and max_dim <= params["ProgSimpleDim_mm"] and setups <= 2)
    if is_simple:
        prog_hr = min(prog_hr, params["ProgCapHr"])
    if milling_hr > 0:
        prog_hr = min(prog_hr, params["ProgMaxToMillingRatio"] * milling_hr)

    programming_cost   = prog_hr * rates["ProgrammingRate"] + cam_hr * rates["CAMRate"] + eng_hr * rates["EngineerRate"]
    prog_per_lot       = programming_cost
    programming_per_part = (prog_per_lot / Qty) if (amortize_programming and Qty > 1) else prog_per_lot

    # ---- fixture -------------------------------------------------------------
    fixture_build_hr = sum_time(r"(?:Fixture\s*Build|Custom\s*Fixture\s*Build)")
    fixture_mat_cost = num(r"(?:Fixture\s*Material\s*Cost|Fixture\s*Hardware)")
    fixture_cost     = fixture_build_hr * rates["FixtureBuildRate"] + fixture_mat_cost
    fixture_per_part = (fixture_cost / Qty) if Qty > 1 else fixture_cost

    nre_detail = {
        "programming": {
            "prog_hr": float(prog_hr), "prog_rate": rates["ProgrammingRate"],
            "cam_hr": float(cam_hr),   "cam_rate": rates["CAMRate"],
            "eng_hr": float(eng_hr),   "eng_rate": rates["EngineerRate"],
            "per_lot": prog_per_lot, "per_part": programming_per_part,
            "amortized": bool(amortize_programming and Qty > 1)
        },
        "fixture": {
            "build_hr": float(fixture_build_hr), "build_rate": rates["FixtureBuildRate"],
            "mat_cost": float(fixture_mat_cost),
            "per_lot": fixture_cost, "per_part": fixture_per_part
        }
    }

    # ---- primary processes ---------------------------------------------------
    # OEE-adjusted helper
    oee = max(1e-6, params["OEE_EfficiencyPct"])
    def eff(hr: float) -> float: 
        try: return float(hr) / oee
        except Exception: return hr

    # Turning
    od_turn_hr   = sum_time(r"(?:OD\s*Turning|OD\s*Rough/Finish|Outer\s*Diameter)")
    id_bore_hr   = sum_time(r"(?:ID\s*Boring|Drilling|Reaming)")
    threading_hr = sum_time(r"(?:Threading|Tapping|Single\s*Point)")
    cutoff_hr    = sum_time(r"(?:Cut[- ]?Off|Parting)")
    turning_hr   = eff(od_turn_hr + id_bore_hr + threading_hr + cutoff_hr)

    # Wire EDM
    wedm_hr_dir  = sum_time(r"(?:WEDM\s*Hours|Wire\s*EDM\s*Hours|EDM\s*Burn\s*Time)")
    wedm_len_mm  = first_num(r"(?:EDM\s*Length_mm|WEDM\s*Length_mm|EDM\s*Perimeter_mm)", GEO_wedm_len_mm)
    wedm_thk_mm  = first_num(r"(?:EDM\s*Thickness_mm|Stock\s*Thickness_mm)", 0.0)
    wedm_passes  = max(1, int(first_num(r"(?:EDM\s*Passes|WEDM\s*Passes)", 1)))
    cut_rate     = max(0.0001, first_num(r"(?:EDM\s*Cut\s*Rate_mm\^?2/min|WEDM\s*Cut\s*Rate)", 2.0))
    edge_factor  = max(1.0, first_num(r"(?:EDM\s*Edge\s*Factor)", 1.0))
    wedm_mm2     = wedm_len_mm * max(0.0, wedm_thk_mm) * wedm_passes
    wedm_hr_calc = (wedm_mm2 / cut_rate) / 60.0 * edge_factor if wedm_mm2 > 0 else 0.0
    wedm_hr      = eff(wedm_hr_dir if wedm_hr_dir > 0 else wedm_hr_calc)

    wire_cost_m  = first_num(r"(?:WEDM\s*Wire\s*Cost\s*/\s*m|Wire\s*Cost\s*/m)", 0.0)
    wire_per_mm2 = first_num(r"(?:Wire\s*Usage\s*m\s*/\s*mm\^?2)", 0.0)
    wire_cost    = (wedm_mm2 * wire_per_mm2 * wire_cost_m) if (wire_per_mm2 > 0 and wire_cost_m > 0) else 0.0

    # Sinker EDM
    sinker_dir   = sum_time(r"(?:Sinker\s*EDM\s*Hours|Ram\s*EDM\s*Hours|Burn\s*Time)")
    sinker_vol   = first_num(r"(?:Sinker\s*Burn\s*Volume_mm3|EDM\s*Volume_mm3)", 0.0)
    sinker_mrr   = max(0.0001, first_num(r"(?:Sinker\s*MRR_mm\^?3/min|EDM\s*MRR)", 8.0))
    sinker_calc  = (sinker_vol / sinker_mrr) / 60.0 if sinker_vol > 0 else 0.0
    sinker_hr    = eff(sinker_dir if sinker_dir > 0 else sinker_calc)

    electrodes_n = max(0, int(first_num(r"(?:Electrode\s*Count)", 0)))
    electrode_each = first_num(r"(?:Electrode\s*(?:Cost|Material)\s*(?:/ea|per\s*ea)?)", 0.0)
    electrodes_cost = electrodes_n * electrode_each

    # Grinding
    surf_grind_hr = eff(sum_time(r"(?:Surface\s*Grind|Pre[- ]?Op\s*Grinding|Blank\s*Squaring)"))
    jig_grind_hr  = eff(sum_time(r"(?:Jig\s*Grind)"))
    odid_grind_hr = eff(sum_time(r"(?:OD/ID\s*Grind|Cylindrical\s*Grind)"))

    grind_vol   = first_num(r"(?:Grind\s*Volume_mm3|Grinding\s*Volume)", 0.0)
    grind_mrr   = max(0.0001, first_num(r"(?:Grind\s*MRR_mm\^?3/min|Grinding\s*MRR)", 300.0))
    grind_pass  = max(0, int(first_num(r"(?:Grinding\s*Passes)", 0)))
    dress_freq  = max(1, int(first_num(r"(?:Dress\s*Frequency\s*passes)", 4)))
    dress_min   = first_num(r"(?:Dress\s*Time\s*min(?:\s*/\s*pass)?)", 0.0)
    grind_calc  = (grind_vol / grind_mrr) / 60.0 if grind_vol > 0 else 0.0
    dresses     = math.floor(grind_pass / max(1, dress_freq))
    dressing_hr = (dress_min * dresses) / 60.0
    grind_hr_calc = grind_calc  # for meta
    grinding_hr  = surf_grind_hr + jig_grind_hr + odid_grind_hr + grind_hr_calc + dressing_hr

    # Apply multipliers
    five_axis_mult = 1.0 + num_pct(r"(?:5[- ]?Axis|Five[- ]Axis|Multi[- ]Axis)", max(0.0, params["FiveAxisMultiplier"] - 1.0))
    tight_tol_mult = 1.0 + num_pct(r"(?:Tight\s*Tolerance|Very\s*Tight)",        max(0.0, params["TightToleranceMultiplier"] - 1.0))

    # Costs
    milling_cost = eff(milling_hr) * rates["MillingRate"] * five_axis_mult * tight_tol_mult
    turning_cost = turning_hr * rates["TurningRate"]
    wedm_cost    = wedm_hr * rates["WireEDMRate"] + wire_cost
    sinker_cost  = sinker_hr * rates["SinkerEDMRate"] + electrodes_cost

    grinding_cost = (
        surf_grind_hr * rates["SurfaceGrindRate"] +
        jig_grind_hr  * rates["JigGrindRate"]     +
        odid_grind_hr * rates["ODIDGrindRate"]    +
        (grind_hr_calc + dressing_hr) * rates["SurfaceGrindRate"] +
        first_num(r"(?:Grinding\s*Wheel\s*Cost)", 0.0)
    )

    # Lapping / Finishing
    lap_hr   = sum_time(r"(?:Lapping|Honing|Polishing)")
    lap_cost = lap_hr * rates["LappingRate"]

    deburr_hr     = sum_time(r"(?:Deburr|Edge\s*Break)")
    tumble_hr     = sum_time(r"(?:Tumbling|Vibratory)")
    blast_hr      = sum_time(r"(?:Bead\s*Blasting|Sanding)")
    laser_mark_hr = sum_time(r"(?:Laser\s*Mark|Engraving)")
    masking_hr    = sum_time(r"(?:Masking\s*for\s*Plating|Masking)")
    finishing_cost = (deburr_hr + tumble_hr + blast_hr + laser_mark_hr + masking_hr) * rates["FinishingRate"]

    # Inspection & docs
    inproc_hr   = sum_time(r"(?:In[- ]?Process\s*Inspection)")
    final_hr    = sum_time(r"(?:Final\s*Inspection|Manual\s*Inspection)")
    cmm_prog_hr = sum_time(r"(?:CMM\s*Programming)")
    cmm_run_hr  = sum_time(r"(?:CMM\s*Run\s*Time)\b") + sum_time(r"(?:CMM\s*Run\s*Time\s*min)")
    fair_hr     = sum_time(r"(?:FAIR|ISIR|PPAP)")
    srcinsp_hr  = sum_time(r"(?:Source\s*Inspection)")

    inspection_cost = (
        (inproc_hr + final_hr) * rates["InspectionRate"] +
        cmm_prog_hr * rates["InspectionRate"] +
        cmm_run_hr  * rates["InspectionRate"] +
        fair_hr     * rates["InspectionRate"] +
        srcinsp_hr  * rates["InspectionRate"]
    )

    # Consumables & utilities
    spindle_hr = (eff(milling_hr) + turning_hr + wedm_hr + sinker_hr + surf_grind_hr + jig_grind_hr + odid_grind_hr)
    consumables_hr_cost = (
        params["MillingConsumablesPerHr"]    * eff(milling_hr) +
        params["TurningConsumablesPerHr"]    * turning_hr +
        params["EDMConsumablesPerHr"]        * (wedm_hr + sinker_hr) +
        params["GrindingConsumablesPerHr"]   * (surf_grind_hr + jig_grind_hr + odid_grind_hr) +
        params["InspectionConsumablesPerHr"] * (inproc_hr + final_hr + cmm_prog_hr + cmm_run_hr + fair_hr + srcinsp_hr)
    )
    utilities_cost   = params["UtilitiesPerSpindleHr"] * spindle_hr
    consumables_flat = float(params["ConsumablesFlat"] or 0.0)

    # Saw / Waterjet
    sawing_hr = sum_time(r"(?:Sawing|Waterjet|Blank\s*Prep)")
    saw_cost  = sawing_hr * rates["SawWaterjetRate"]

    # Assembly / hardware
    assembly_hr   = sum_time(r"(?:Assembly|Manual\s*Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)")
    assembly_cost = assembly_hr * rates["AssemblyRate"]
    hardware_cost = num(r"(?:Hardware|BOM\s*Cost|Fasteners|Bearings|CFM\s*Hardware)")

    # Outsourced vendors
    heat_treat_cost  = num(r"(?:Outsourced\s*Heat\s*Treat|Heat\s*Treat\s*Cost)")
    coating_cost     = num(r"(?:Plating|Coating|Anodize|Black\s*Oxide|DLC|PVD|CVD)")
    passivation_cost = num(r"(?:Passivation|Cleaning\s*Vendor)")
    outsourced_costs = heat_treat_cost + coating_cost + passivation_cost

    # Packaging / logistics
    packaging_hr      = sum_time(r"(?:Packaging|Boxing|Crating\s*Labor)")
    crate_nre_cost    = num(r"(?:Custom\s*Crate\s*NRE)")
    packaging_mat     = num(r"(?:Packaging\s*Materials|Foam|Trays)")
    shipping_cost     = num(r"(?:Freight|Shipping\s*Cost)")
    insurance_pct     = num_pct(r"(?:Insurance|Liability\s*Adder)", params["InsurancePct"])
    packaging_cost    = packaging_hr * rates["AssemblyRate"] + crate_nre_cost + packaging_mat

    # EHS / compliance
    ehs_hr   = sum_time(r"(?:EHS|Compliance|Training|Waste\s*Handling)")
    ehs_cost = ehs_hr * rates["InspectionRate"]

    # ---- roll-ups ------------------------------------------------------------
    labor_cost = (
        programming_per_part + fixture_per_part +
        milling_cost + turning_cost + wedm_cost + sinker_cost +
        grinding_cost + lap_cost + finishing_cost + inspection_cost +
        saw_cost + assembly_cost + packaging_cost + ehs_cost
    )

    direct_costs   = (material_cost + hardware_cost + outsourced_costs + shipping_cost +
                      consumables_hr_cost + utilities_cost + consumables_flat)

    insurance_cost = insurance_pct * (labor_cost + direct_costs)

    vendor_markup     = params["VendorMarkupPct"]
    vendor_marked_add = vendor_markup * (outsourced_costs + shipping_cost)

    subtotal = labor_cost + direct_costs + insurance_cost + vendor_marked_add

    with_overhead = subtotal * (1.0 + OverheadPct)
    with_ga       = with_overhead * (1.0 + GA_Pct)
    with_cont     = with_ga * (1.0 + ContingencyPct)
    with_expedite = with_cont * (1.0 + ExpeditePct)

    price_before_margin = with_expedite
    price = price_before_margin * (1.0 + MarginPct)

    min_lot = float(params["MinLotCharge"] or 0.0)
    if price < min_lot:
        price = min_lot

    # ---- meta for pretty output ---------------------------------------------
    inspection_hr_total = inproc_hr + final_hr + cmm_prog_hr + cmm_run_hr + fair_hr + srcinsp_hr
    process_meta = {
    "milling":          {"hr": eff(milling_hr), "rate": rates.get("MillingRate", 0.0)},
    "turning":          {"hr": eff(turning_hr), "rate": rates.get("TurningRate", 0.0)},
    "wire_edm":         {"hr": eff(wedm_hr),    "rate": rates.get("WireEDMRate", 0.0)},
    "sinker_edm":       {"hr": eff(sinker_hr),  "rate": rates.get("SinkerEDMRate", 0.0)},
    "grinding":         {"hr": eff(grinding_hr),"rate": rates.get("SurfaceGrindRate", 0.0)},
    "lapping_honing":   {"hr": lap_hr,          "rate": rates.get("LappingRate", 0.0)},
    "finishing_deburr": {"hr": deburr_hr + tumble_hr + blast_hr + laser_mark_hr + masking_hr,
                         "rate": rates.get("FinishingRate", 0.0)},
    "inspection":       {"hr": inspection_hr_total, "rate": rates.get("InspectionRate", 0.0)},
    "saw_waterjet":     {"hr": sawing_hr,       "rate": rates.get("SawWaterjetRate", 0.0)},
    "assembly":         {"hr": assembly_hr,     "rate": rates.get("AssemblyRate", 0.0)},
    "packaging":        {"hr": packaging_hr,    "rate": rates.get("PackagingRate", rates.get("AssemblyRate", 0.0))},
}
    pass_meta = {
        "consumables_hr_cost": {"basis": "Machine & inspection hours $/hr"},
        "utilities_cost":      {"basis": "Spindle/inspection hours $/hrr"},
        "insurance_cost":      {"basis": "Applied at insurance pct"},
        "vendor_markup_added": {"basis": "Vendor + freight markup"},
        "consumables_flat":    {"basis": "Fixed shop supplies"},
    }

    breakdown = {
        "qty": Qty,
        "material": {
            "mass_g": mass_g,
            "unit_price_per_g": unit_price_per_g,
            "supplier_min_charge": supplier_min_charge,
            "surcharge_pct": surcharge_pct,
            "material_cost": material_cost,
        },
        "nre": {
            "programming_per_part": programming_per_part,
            "fixture_per_part": fixture_per_part,
            "extra_nre_cost": 0.0,  # (gauge/other NRE hook, keep for compat)
        },
        "nre_detail": nre_detail,
        "process_costs": {
            "milling": milling_cost,
            "turning": turning_cost,
            "wire_edm": wedm_cost,
            "sinker_edm": sinker_cost,
            "grinding": grinding_cost,
            "lapping_honing": lap_cost,
            "finishing_deburr": finishing_cost,
            "inspection": inspection_cost,
            "saw_waterjet": saw_cost,
            "assembly": assembly_cost,
            "packaging": packaging_cost,
            "ehs_compliance": ehs_cost,
        },
        "process_meta": process_meta,
        "pass_through": {
            "hardware_bom": hardware_cost,
            "outsourced_vendors": outsourced_costs,
            "shipping": shipping_cost,
            "insurance_cost": insurance_cost,
            "consumables_hr_cost": consumables_hr_cost,
            "utilities_cost": utilities_cost,
            "consumables_flat": consumables_flat,
            "vendor_markup_added": vendor_marked_add,
        },
        "pass_meta": pass_meta,
        "totals": {
            "labor_cost": labor_cost,
            "direct_costs": direct_costs,
            "subtotal": subtotal,
            "with_overhead": with_overhead,
            "with_ga": with_ga,
            "with_contingency": with_cont,
            "with_expedite": with_expedite,
            "price": price
        },
        "applied_pcts": {
            "OverheadPct": OverheadPct,
            "GA_Pct": GA_Pct,
            "ContingencyPct": ContingencyPct,
            "ExpeditePct": ExpeditePct,
            "MarginPct": MarginPct,
            "InsurancePct": insurance_pct,
            "VendorMarkupPct": vendor_markup
        }
    }

    return {"price": price, "labor": labor_cost, "with_overhead": with_overhead, "breakdown": breakdown}


# ---------- Tracing ----------
def trace_hours_from_df(df):
    import pandas as pd
    out={}
    def grab(regex,label):
        items=df["Item"].astype(str)
        m=items.str.contains(regex, flags=re.I, regex=True, na=False)
        looks_time=items.str.contains(r"(hours?|hrs?|hr|time|min(ute)?s?)\b", flags=re.I, regex=True, na=False)
        mm=m & looks_time
        if mm.any():
            sub=df.loc[mm, ["Item","Example Values / Options"]].copy()
            sub["Example Values / Options"]=pd.to_numeric(sub["Example Values / Options"], errors="coerce").fillna(0.0)
            out[label]=[(str(rates["Item"]), float(rates["Example Values / Options"])) for _,r in sub.iterrows()]
    grab(r"(Fixture\s*Design|Process\s*Sheet|Traveler|Documentation)","Engineering")
    grab(r"(CAM\s*Programming|CAM\s*Sim|Post\s*Processing)","CAM")
    grab(r"(Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)","Assembly")
    return out

# ---------- Explanation ----------
def explain_quote(breakdown: dict[str,Any], hour_trace=None) -> str:
    tot=breakdown.get("totals",{}); pro=breakdown.get("process_costs",{}); nre=breakdown.get("nre",{}); pt=breakdown.get("pass_through",{}); ap=breakdown.get("applied_pcts",{})
    lines=[f"Includes Overhead {ap.get('OverheadPct',0):.0%}, Margin {ap.get('MarginPct',0):.0%}."]
    if nre.get("programming_per_part",0)>0: lines.append(f"NRE spread across lot: ${nre['programming_per_part']:.2f}")
    major={k:v for k,v in pro.items() if v and v>0}
    if major:
        top=sorted(major.items(), key=lambda kv: kv[1], reverse=True)[:4]
        lines.append("Major processes: " + ", ".join(f"{k.replace('_',' ')} ${v:,.0f}" for k,v in top))
    if pt.get("hardware_bom",0) or pt.get("outsourced_vendors",0):
        lines.append(f"Pass-throughs: hardware ${pt.get('hardware_bom',0):.0f}, vendors ${pt.get('outsourced_vendors',0):.0f}, ship ${pt.get('shipping',0):.0f}")
    if hour_trace:
        for bucket, rows in hour_trace.items():
            total=sum(v for _,v in rows)
            if total>0:
                items="; ".join(f"{name}={val}" for name,val in rows[:3])
                lines.append(f"{bucket} time from: {items} (total ~{total:g} hr)")
    return "\n".join(lines)

# ---------- Redaction ----------
def redact_text(s: str) -> str:
    out=s
    for pat,repl in _REDACT_PATTERNS: out=pat.sub(repl,out)
    return out
def redact_df(df):
    out=df.copy()
    for c in out.columns:
        if out[c].dtype==object:
            out[c]=out[c].astype(str).map(redact_text)
    return out

# ---------- Process router ----------
def route_process_family(geo: dict[str,Any]) -> str:
    max_dim=float(geo.get("GEO__MaxDim_mm",0) or 0); wedm=float(geo.get("GEO_WEDM_PathLen_mm",0) or 0); free=float(geo.get("GEO_Area_Freeform_mm2",0) or 0); area=float(geo.get("GEO__SurfaceArea_mm2",1) or 1); fr=(free/area) if area>0 else 0.0
    if wedm>200 and max_dim<300: return "wire_edm"
    if fr>0.5: return "die"
    if max_dim>600: return "plate"
    return "cnc"

# ---------- Simple Tk editor ----------
def edit_variables_tk(df):
    import tkinter as tk
    from tkinter import ttk, messagebox
    cols=["Item","Example Values / Options","Data Type / Input Method"]
    for c in cols:
        if c not in df.columns: df[c]=""
    win=tk.Toplevel(); win.title("Edit Variables")
    tree=ttk.Treeview(win, columns=cols, show="headings")
    for c in cols: tree.heading(c,text=c); tree.column(c,width=220,anchor="w")
    for i,row in df[cols].iterrows(): tree.insert("","end",iid=str(i),values=[row[c] for c in cols])
    tree.pack(fill="both",expand=True)
    entry=ttk.Entry(win); entry.pack(fill="x")
    cur={"col":0}
    def on_click(e):
        item=tree.focus()
        if not item: return
        idx=int(tree.identify_column(e.x).replace("#",""))-1
        curates["col"]=idx; vals=list(tree.item(item,"values")); entry.delete(0,"end"); entry.insert(0,vals[idx])
    def on_return(e=None):
        item=tree.focus()
        if not item: return
        idx=curates["col"]; vals=list(tree.item(item,"values")); vals[idx]=entry.get(); tree.item(item,values=vals)
    def on_save():
        for iid in tree.get_children():
            vals=tree.item(iid,"values"); i=int(iid)
            for k,v in zip(cols, vals): df.at[i,k]=v
        messagebox.showinfo("Saved","Variables updated in memory; Save in app to persist.")
    tree.bind("<Button-1>", on_click); entry.bind("<Return>", on_return); ttk.Button(win,text="Save",command=on_save).pack()
    win.grab_set(); win.wait_window(); return df
def read_dxf_as_occ_shape(dxf_path: str):
    # minimal DXF?OCC triangulated shape (3DFACE/MESH/POLYFACE), fallback: extrude closed polyline
    import ezdxf, numpy as np
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid)
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE

    def tri_face(p0,p1,p2):
        poly = BRepBuilderAPI_MakePolygon()
        for p in (p0,p1,p2,p0):
            poly.Add(gp_Pnt(*p))
        return BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()

    def sew(faces):
        sew = BRepBuilderAPI_Sewing(1.0e-6, True, True, True, True)
        for f in faces: sew.Add(f)
        sew.Perform()
        return sew.SewedShape()

    doc = ezdxf.readfile(dxf_path)
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
            solid = BRepBuilderAPI_MakeSolid()
            exp = TopExp_Explorer(sewed, TopAbs_FACE)
            while exp.More():
                solid.Add(exp.Current()); exp.Next()
            fix = ShapeFix_Solid(solid.Solid()); fix.Perform()
            shape = fix.Solid()
        except Exception:
            shape = sewed

    # Fallback: 2D closed polyline ? extrude small thickness
    if (shape is None) or shape.IsNull():
        for pl in msp.query("LWPOLYLINE"):
            if pl.closed:
                pts2d = [(x*u2mm, y*u2mm, 0.0) for x,y,_ in pl.get_points("xyb")]
                poly = BRepBuilderAPI_MakePolygon()
                for q in pts2d: poly.Add(gp_Pnt(*q))
                poly.Close()
                face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
                thk_mm = float(os.environ.get("DXF_EXTRUDE_THK_MM", "5.0"))
                shape = BRepPrimAPI_MakePrism(face, gp_Vec(0,0,thk_mm)).Shape()
                break

    if (shape is None) or shape.IsNull():
        raise RuntimeError("DXF contained no 3D geometry I can use. Prefer STEP/SAT if possible.")
    return shape

# ---- 2D: PDF (PyMuPDF) -------------------------------------------------------
try:
    import fitz  # old import name
    _HAS_PYMUPDF = True
except Exception:
    try:
        import pymupdf as fitz  # new import name
        _HAS_PYMUPDF = True
    except Exception:
        fitz = None  # allow the rest of the app to import

def extract_2d_features_from_pdf_vector(pdf_path: str) -> dict:
    if not _HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) not installed. pip install pymupdf")

    import math
    page = fitz.open(pdf_path)[0]
    text = page.get_text("text").lower()
    drawings = page.get_drawings()

    # perimeter from vector segments (points are in PostScript points; 1 pt = 0.352777ï¿½ mm)
    pt_to_mm = 0.352777778
    per_pts = 0.0

    def _as_xy(obj):
        if hasattr(obj, "x") and hasattr(obj, "y"):
            return float(obj.x), float(obj.y)
        if isinstance(obj, (tuple, list)) and len(obj) >= 2:
            try:
                return float(obj[0]), float(obj[1])
            except Exception:
                return None
        return None

    def _add_polyline(points):
        nonlocal per_pts
        if len(points) < 2:
            return
        for a, b in zip(points, points[1:]):
            if a is None or b is None:
                continue
            per_pts += math.hypot(b[0] - a[0], b[1] - a[1])

    for d in drawings:
        for item in d["items"]:
            kind, *data = item
            if kind == "l":  # line
                p1 = _as_xy(data[0]) if len(data) >= 1 else None
                p2 = _as_xy(data[1]) if len(data) >= 2 else None
                if p1 and p2:
                    per_pts += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            elif kind in {"c", "b", "v", "y"}:  # bezier / curves
                pts = [_as_xy(part) for part in data]
                _add_polyline([pt for pt in pts if pt is not None])
            elif kind == "re" and data:
                rect = data[0]
                if isinstance(rect, (tuple, list)) and len(rect) >= 4:
                    w = float(rect[2])
                    h = float(rect[3])
                    per_pts += 2.0 * (abs(w) + abs(h))
            elif kind == "qu":  # quadratic curve (3 points)
                pts = [_as_xy(part) for part in data]
                _add_polyline([pt for pt in pts if pt is not None])

    # scrape thickness/material from text
    import re
    thickness_mm = None
    m = re.search(r"(thk|thickness)\s*[:=]?\s*([0-9.]+)\s*(mm|in|in\.|\")", text)
    if m:
        val = float(m.group(2))
        unit = m.group(3)
        thickness_mm = val * (25.4 if (unit.startswith("in") or unit == '"') else 1.0)

    material = None
    mm = re.search(r"(matl|material)\s*[:=]?\s*([a-z0-9 \-\+]+)", text)
    if mm:
        material = mm.group(2).strip()

    return {
        "kind": "2D", "source": "pdf",
        "profile_length_mm": round(per_pts * pt_to_mm, 2),
        "hole_diams_mm": [],
        "hole_count": 0,
        "thickness_mm": thickness_mm,
        "material": material,
    }
# ---------- PDF-driven variables + inference ----------
REQUIRED_COLS = ["Item", "Example Values / Options", "Data Type / Input Method"]

def default_variables_template() -> pd.DataFrame:
    rows = [
        ("Overhead %", 0.15, "number"),
        ("G&A %", 0.08, "number"),
        ("Contingency %", 0.0, "number"),
        ("Profit Margin %", 0.0, "number"),
        ("Programmer $/hr", 120.0, "number"),
        ("CAM Programmer $/hr", 125.0, "number"),
        ("Milling $/hr", 100.0, "number"),
        ("Inspection $/hr", 95.0, "number"),
        ("Deburr $/hr", 60.0, "number"),
        ("Packaging $/hr", 55.0, "number"),
        ("Programming Hours", 0.0, "number"),
        ("CAM Programming Hours", 0.0, "number"),
        ("Engineering (Docs/Fixture Design) Hours", 0.0, "number"),
        ("Fixture Build Hours", 0.0, "number"),
        ("Roughing Cycle Time", 0.0, "number"),
        ("Semi-Finish Cycle Time", 0.0, "number"),
        ("Finishing Cycle Time", 0.0, "number"),
        ("In-Process Inspection Hours", 0.0, "number"),
        ("Final Inspection Hours", 0.0, "number"),
        ("CMM Programming Hours", 0.0, "number"),
        ("CMM Run Time min", 0.0, "number"),
        ("Deburr Hours", 0.0, "number"),
        ("Tumbling Hours", 0.0, "number"),
        ("Bead Blasting Hours", 0.0, "number"),
        ("Laser Mark Hours", 0.0, "number"),
        ("Masking Hours", 0.0, "number"),
        ("Sawing Hours", 0.0, "number"),
        ("Assembly Hours", 0.0, "number"),
        ("Packaging Labor Hours", 0.0, "number"),
        ("Number of Milling Setups", 1, "number"),
        ("Setup Hours / Setup", 0.3, "number"),
        ("FAIR Required", 0, "number"),
        ("Source Inspection Requirement", 0, "number"),
        ("Quantity", 1, "number"),
        ("Material", "", "text"),
        ("Thickness (in)", 0.0, "number"),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLS)

def coerce_or_make_vars_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not set(REQUIRED_COLS).issubset(df.columns):
        return default_variables_template().copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
    return df

def extract_pdf_all(pdf_path: Path, dpi: int = 300) -> dict:
    if not _HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) not installed. pip install pymupdf")
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for idx, page in enumerate(doc):
        text = page.get_text("text") or ""
        blocks = page.get_text("blocks")
        tables = []
        try:
            found = page.find_tables()
            for table in getattr(found, "tables", []):
                tables.append(table.extract())
        except Exception:
            pass
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png_path = pdf_path.with_suffix(f".p{idx}.png")
        pix.save(png_path)
        pages.append({
            "index": idx,
            "text": text,
            "blocks": blocks,
            "tables": tables,
            "image_path": str(png_path),
            "char_count": len(text),
            "table_count": len(tables),
        })
    best = max(pages, key=lambda p: (params["table_count"] > 0, params["char_count"])) if pages else {"text": "", "image_path": ""}
    doc.close()
    return {"pages": pages, "best": best}

JSON_SCHEMA = {
    "part_name": "str|null",
    "material": {"name": "str|null", "thickness_in": "float|null"},
    "quantity": "int|null",
    "estimates": {
        "Programming Hours": "float|null",
        "CAM Programming Hours": "float|null",
        "Engineering (Docs/Fixture Design) Hours": "float|null",
        "Fixture Build Hours": "float|null",
        "Roughing Cycle Time": "float|null",
        "Semi-Finish Cycle Time": "float|null",
        "Finishing Cycle Time": "float|null",
        "In-Process Inspection Hours": "float|null",
        "Final Inspection Hours": "float|null",
        "Deburr Hours": "float|null",
        "Sawing Hours": "float|null",
        "Assembly Hours": "float|null",
        "Packaging Labor Hours": "float|null"
    },
    "flags": {"FAIR Required": "0|1", "Source Inspection Requirement": "0|1"},
    "reasoning_brief": "str"
}

def _truncate_text(text: str, max_chars: int = 5000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return f"{head}\n...\n{tail}"

def build_llm_prompt(best_page: dict) -> dict:
    text = _truncate_text(best_page.get("text", ""))
    schema = json.dumps(JSON_SCHEMA, indent=2)
    system = (
        "You are a manufacturing estimator. Read the drawing text and image and return JSON only. "
        "Estimate hours conservatively and do not invent dimensions."
    )
    user = (
        f"TEXT FROM PDF PAGE:\n{text}\n\n"
        f"REQUIRED JSON SHAPE:\n{schema}\n\n"
        "Return strictly JSON. If a field is unknown use null, and use 0/1 for flags."
    )
    return {"system": system, "user": user, "image_path": best_page.get("image_path", "")}

def run_llm_json(system: str, user: str, image_path: str) -> dict:
    """Adapter for the local VL model; override with real llama.cpp call."""
    # Placeholder: integrate with your Qwen VL llama.cpp wrapper and return parsed JSON.
    result_text = "{}"
    try:
        match = re.search(r"\{.*\}", result_text, flags=re.S)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}

def infer_pdf_estimate(structured: dict) -> dict:
    best_page = structured.get("best", {})
    prompt = build_llm_prompt(best_page)
    return run_llm_json(prompt["system"], prompt["user"], prompt["image_path"])

MAP_KEYS = {
    "Quantity": "quantity",
    "Material": ("material", "name"),
    "Thickness (in)": ("material", "thickness_in"),
    "Programming Hours": ("estimates", "Programming Hours"),
    "CAM Programming Hours": ("estimates", "CAM Programming Hours"),
    "Engineering (Docs/Fixture Design) Hours": ("estimates", "Engineering (Docs/Fixture Design) Hours"),
    "Fixture Build Hours": ("estimates", "Fixture Build Hours"),
    "Roughing Cycle Time": ("estimates", "Roughing Cycle Time"),
    "Semi-Finish Cycle Time": ("estimates", "Semi-Finish Cycle Time"),
    "Finishing Cycle Time": ("estimates", "Finishing Cycle Time"),
    "In-Process Inspection Hours": ("estimates", "In-Process Inspection Hours"),
    "Final Inspection Hours": ("estimates", "Final Inspection Hours"),
    "Deburr Hours": ("estimates", "Deburr Hours"),
    "Sawing Hours": ("estimates", "Sawing Hours"),
    "Assembly Hours": ("estimates", "Assembly Hours"),
    "Packaging Labor Hours": ("estimates", "Packaging Labor Hours"),
    "FAIR Required": ("flags", "FAIR Required"),
    "Source Inspection Requirement": ("flags", "Source Inspection Requirement"),
}

def _deep_get(d: dict, path):
    if isinstance(path, str):
        return d.get(path)
    cur = d
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

def merge_estimate_into_vars(vars_df: pd.DataFrame, estimate: dict) -> pd.DataFrame:
    for item, src in MAP_KEYS.items():
        value = _deep_get(estimate, src)
        if value is None:
            continue
        mask = vars_df["Item"].astype(str).str.fullmatch(re.escape(item), case=False, na=False)
        if not mask.any():
            dtype = "number" if isinstance(value, (int, float)) else "text"
            new_row = pd.DataFrame([{ "Item": item, "Example Values / Options": "", "Data Type / Input Method": dtype }])
            vars_df = pd.concat([vars_df, new_row], ignore_index=True)
            mask = vars_df["Item"].astype(str).str.fullmatch(re.escape(item), case=False, na=False)
        vars_df.loc[mask, "Example Values / Options"] = value
    return vars_df
# ---- 2D: DXF / DWG (ezdxf) ---------------------------------------------------
def extract_2d_features_from_dxf_or_dwg(path: str) -> dict:
    if not _HAS_EZDXF:
        raise RuntimeError("ezdxf not installed. pip/conda install ezdxf")

    # --- load doc ---
    if path.lower().endswith(".dwg"):
        if _HAS_ODAFC:
            # uses ODAFileConverter through ezdxf, no env var needed
            doc = odafc.readfile(path)
        else:
            dxf_path = convert_dwg_to_dxf(path)  # needs ODA_CONVERTER_EXE or DWG2DXF_EXE
            doc = ezdxf.readfile(dxf_path)
    else:
        doc = ezdxf.readfile(path)

    sp = doc.modelspace()
    ins = int(doc.header.get("$INSUNITS", 4))
    u2mm = {1:25.4, 4:1.0, 2:304.8, 6:1000.0}.get(ins, 1.0)

    # perimeter from lightweight polylines, polylines, arcs
    import math
    per = 0.0
    for e in sp.query("LWPOLYLINE"):
        pts = e.get_points("xyb")
        for i in range(len(pts)):
            x1,y1,_ = pts[i]; x2,y2,_ = pts[(i+1) % len(pts)]
            per += math.hypot(x2-x1, y2-y1)
    for e in sp.query("POLYLINE"):
        vs = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        for i in range(len(vs)-1):
            x1,y1 = vs[i]; x2,y2 = vs[i+1]
            per += math.hypot(x2-x1, y2-y1)
    for e in sp.query("ARC"):
        per += abs(e.dxf.end_angle - e.dxf.start_angle) * math.pi/180.0 * e.dxf.radius

    # holes from circles
    holes = list(sp.query("CIRCLE"))
    hole_diams_mm = [round(2.0 * c.dxf.radius * u2mm, 2) for c in holes]

    # scrape text for thickness/material
    txt = " ".join([t.dxf.text for t in sp.query("TEXT")] +
                   [m.plain_text() for m in sp.query("MTEXT")]).lower()
    import re
    thickness_mm = None
    m = re.search(r"(thk|thickness)\s*[:=]?\s*([0-9.]+)\s*(mm|in|in\.|\")", txt)
    if m:
        thickness_mm = float(m.group(2)) * (25.4 if (m.group(3).startswith("in") or m.group(3)=='"') else 1.0)
    material = None
    mm = re.search(r"(matl|material)\s*[:=]?\s*([a-z0-9 \-\+]+)", txt)
    if mm:
        material = mm.group(2).strip()

    return {
        "kind": "2D", "source": Path(path).suffix.lower().lstrip("."),
        "profile_length_mm": round(per * u2mm, 2),
        "hole_diams_mm": hole_diams_mm,
        "hole_count": len(hole_diams_mm),
        "thickness_mm": thickness_mm,
        "material": material,
    }
def apply_2d_features_to_variables(df, g2d: dict, *, params: dict, rates: dict):
    """Write a few cycle-time rows based on 2D perimeter/holes so compute_quote_from_df() can price it."""



def set_row(pattern: str, value: float):
    def _to_noncapturing(expr: str) -> str:
        out: list[str] = []
        i = 0
        while i < len(expr):
            ch = expr[i]
            prev = expr[i - 1] if i > 0 else ''
            nxt = expr[i + 1] if i + 1 < len(expr) else ''
            if ch == '(' and prev != '\\\\' and nxt != '?':
                out.append('(?:')
                i += 1
                continue
            out.append(ch)
            i += 1
        return ''.join(out)

    regex = _to_noncapturing(pattern)
    mask = df["Item"].astype(str).str.contains(regex, case=False, regex=True, na=False)
    if mask.any():
        df.loc[mask, "Example Values / Options"] = value
    else:
        df.loc[len(df)] = [pattern, value, "number"]
    L = float(g2d.get("profile_length_mm", 0.0))
    t = float(g2d.get("thickness_mm") or 6.0)
    holes = g2d.get("hole_diams_mm", [])
    # crude process pick
    use_jet = (t <= 12.0 and (L >= 300.0 or len(holes) >= 2))

    if use_jet:
        cut_min = L / (300.0 if t <= 10 else 120.0)   # mm/min ? minutes
        deburr_min = (L/1000.0) * 2.0
        set_row(r"(Sawing|Waterjet|Blank\s*Prep)", round(cut_min/60.0, 3))
        set_row(r"(Deburr|Edge\s*Break)", round(deburr_min/60.0, 3))
        set_row(r"(Programming|2D\s*CAM)", 0.5)
        set_row(r"(Setup\s*Time\s*per\s*Setup)", 0.25)
        set_row(r"(Milling\s*Setups)", 1)
    else:
        mill_min = L / 800.0
        drill_min = max(0.2, (t/50.0)) * max(1, len(holes))
        deburr_min = (L/1000.0) * 3.0
        set_row(r"(Finishing\s*Cycle\s*Time)", round(mill_min/60.0, 3))
        set_row(r"(ID\s*Boring|Drilling|Reaming)", round(drill_min/60.0, 3))
        set_row(r"(Deburr|Edge\s*Break)", round(deburr_min/60.0, 3))
        set_row(r"(Programming|2D\s*CAM)", 0.75)
        set_row(r"(Setup\s*Time\s*per\s*Setup)", 0.5)
        set_row(r"(Milling\s*Setups)", 1)

    set_row(r"(Final\s*Inspection|Manual\s*Inspection)", 0.2)
    set_row(r"(Packaging|Boxing|Crating\s*Labor)", 0.1)
    return df.reset_index(drop=True)

def get_llm_quote_explanation(result: dict, model_path: str) -> str:
    """
    Returns a single-paragraph, friendly explanation of the main cost drivers.
    Works with the local _LocalLLM.ask_json(), and has a safe fallback if no model.
    """
    import os, json, textwrap

    b = result.get("breakdown", {}) or {}
    totals = b.get("totals", {}) or {}
    pro    = b.get("process_costs", {}) or {}
    nre    = b.get("nre", {}) or {}
    nre_detail = b.get("nre_detail", {}) or {}

    # --- Heuristic fallback (no LLM / error)
    def _fallback() -> str:
        top = sorted([(k, v) for k, v in pro.items() if v], key=lambda kv: kv[1], reverse=True)[:3]
        bits = []
        if nre.get("programming_per_part", 0):
            bits.append(f"NRE (programming/fixturing) adds ${nre['programming_per_part']:.0f} per part")
        if top:
            parts = ", ".join(f"{k.replace('_',' ')} ${v:,.0f}" for k, v in top)
            bits.append(f"largest labor buckets are {parts}")
        if totals.get("direct_costs", 0):
            bits.append(f"directs (material/vendors/shipping) are ${totals['direct_costs']:.0f}")
        return " / ".join(bits) or "Costs are driven mostly by setup/NRE and the primary machining steps."

    # If no model on disk, just return fallback
    if not model_path or not os.path.isfile(model_path):
        return _fallback()

    summary = {
        "final_price": result.get("price", 0.0),
        "labor_cost": totals.get("labor_cost", 0.0),
        "direct_costs": totals.get("direct_costs", 0.0),
        "nre_programming_cost": (nre_detail.get("programming", {}) or {}).get("per_lot", 0.0),
        "nre_fixture_cost": (nre_detail.get("fixture", {}) or {}).get("per_lot", 0.0),
        "process_costs": pro,
    }

    system_prompt = (
        "You are a helpful manufacturing estimator. Explain the main cost drivers of this quote "
        "in one short paragraph (1ï¿½3 sentences), friendly and professional. "
        "Focus on why: big NRE, which processes dominate, and whether directs matter. "
        "Return JSON only: {\"explanation\": \"...\"}"
    )
    user_prompt = f"```json\n{json.dumps(summary, indent=2)}\n```"

    try:
        llm = _LocalLLM(model_path)  # your local wrapper
        out = llm.ask_json(system_prompt, user_prompt, temperature=0.4, max_tokens=256)
        if isinstance(out, dict) and isinstance(out.get("explanation", ""), str) and out["explanation"].strip():
            return out["explanation"].strip()
        return _fallback()
    except Exception:
        return _fallback()
# ----------------- GUI -----------------
# ---- scrollable frame helper -----------------------------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # Bind wheel only while cursor is over this widget
        self.inner.bind("<Enter>", self._bind_mousewheel)
        self.inner.bind("<Leave>", self._unbind_mousewheel)

    # Windows & macOS wheel
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-int(event.delta/120), "units")

    # Linux wheel
    def _on_mousewheel_linux(self, event):
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>",  self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>",  self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CAD Quoter ï¿½ SINGLE FILE (v8, LLM default ON)")
        self.geometry("1260x900")

        self.vars_df = None
        self.vars_df_full = None
        self.geo = None
        self.params = PARAMS_DEFAULT.copy()
        self.rates = RATES_DEFAULT.copy()

        # LLM defaults: ON + auto model discovery
        default_model = find_default_qwen_model()
        if default_model:
            os.environ["QWEN_GGUF_PATH"] = default_model
            self.params["LLMModelPath"] = default_model

        self.llm_enabled = tk.BooleanVar(value=True)
        self.apply_llm_adj = tk.BooleanVar(value=True)
        self.llm_model_path = tk.StringVar(value=default_model)

        # Create a Menu Bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Load Overrides...", command=self.load_overrides)
        file_menu.add_command(label="Save Overrides...", command=self.save_overrides)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="Diagnostics",
            command=lambda: messagebox.showinfo("Diagnostics", get_import_diagnostics_text())
        )
        # Simplified top toolbar
        top = ttk.Frame(self); top.pack(fill="x", pady=(6,8))
        ttk.Button(top, text="1. Load CAD & Vars", command=self.open_flow).pack(side="left", padx=5)
        ttk.Button(top, text="2. Generate Quote", command=self.gen_quote).pack(side="left", padx=5)

        # Tabs
        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True)
        self.tab_geo = ttk.Frame(self.nb); self.nb.add(self.tab_geo, text="GEO")
        self.tab_editor = ttk.Frame(self.nb); self.nb.add(self.tab_editor, text="Quote Editor")
        self.editor_scroll = ScrollableFrame(self.tab_editor)
        self.editor_scroll.pack(fill="both", expand=True)
        self.tab_out = ttk.Frame(self.nb); self.nb.add(self.tab_out, text="Output")
        self.tab_llm = ttk.Frame(self)

        # Dictionaries to hold editor variables
        self.quote_vars = {}
        self.param_vars = {}
        self.rate_vars = {}
        self.editor_widgets_frame = None

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5)
        status_bar.pack(side="bottom", fill="x")

        # GEO (single pane; CAD open handled by top bar)
        self.geo_txt = tk.Text(self.tab_geo, wrap="word"); self.geo_txt.pack(fill="both", expand=True)


        # LLM (hidden frame is still built to keep functionality without a visible tab)
        if hasattr(self, "_build_llm"):
            try:
                self._build_llm(self.tab_llm)
            except Exception:
                pass

        # Output
        self.out_txt = tk.Text(self.tab_out, wrap="word")
        self.out_txt.pack(fill="both", expand=True)
        try:
            self.out_txt.tag_configure("rcol", tabs=("4.8i right",), tabstyle="tabular")
        except tk.TclError:
            self.out_txt.tag_configure("rcol", tabs=("4.8i right",))
        #self.out_txt = tk.Text(self.tab_out, wrap="word", font=("Consolas", 10)); self.out_txt.pack(fill="both", expand=True)

    def _populate_editor_tab(self, df: pd.DataFrame) -> None:
        df = coerce_or_make_vars_df(df)
        """Rebuild the Quote Editor tab using the latest variables dataframe."""
        parent = self.editor_scroll.inner
        for child in parent.winfo_children():
            child.destroy()

        self.quote_vars.clear()
        self.param_vars.clear()
        self.rate_vars.clear()

        self.editor_widgets_frame = parent
        self.editor_widgets_frame.grid_columnconfigure(0, weight=1)

        def normalize_item(value: str) -> str:
            cleaned = re.sub(r"[^0-9a-z&$]+", " ", str(value).strip().lower())
            return re.sub(r"\s+", " ", cleaned).strip()

        items_series = df["Item"].astype(str)
        normalized_items = items_series.apply(normalize_item)
        qty_mask = normalized_items.isin({"quantity", "qty", "lot size"})
        if qty_mask.any():
            qty_raw = df.loc[qty_mask, "Example Values / Options"].iloc[0]
            try:
                qty_value = float(str(qty_raw).strip())
            except Exception:
                qty_value = float(self.params.get("Quantity", 1) or 1)
            if math.isnan(qty_value):
                qty_value = float(self.params.get("Quantity", 1) or 1)
            self.params["Quantity"] = max(1, int(round(qty_value)))

        raw_skip_items = {
            "Overhead %", "Overhead",
            "G&A %", "G&A", "GA %", "GA",
            "Contingency %", "Contingency",
            "Profit Margin %", "Profit Margin", "Margin %", "Margin",
            "Expedite %", "Expedite",
            "Insurance %", "Insurance",
            "Vendor Markup %", "Vendor Markup",
            "Min Lot Charge",
            "Programmer $/hr",
            "CAM Programmer $/hr",
            "Milling $/hr",
            "Inspection $/hr",
            "Deburr $/hr",
            "Packaging $/hr",
            "Quantity", "Qty", "Lot Size",
        }
        skip_items = {normalize_item(item) for item in raw_skip_items}


        current_row = 0

        quote_frame = ttk.Labelframe(self.editor_widgets_frame, text="Quote-Specific Variables", padding=(10, 5))
        quote_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1

        row_index = 0
        for _, row_data in df.iterrows():
            item_name = str(row_data["Item"])
            if normalize_item(item_name) in skip_items:
                continue
            ttk.Label(quote_frame, text=item_name, wraplength=400).grid(row=row_index, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(row_data["Example Values / Options"]))
            ttk.Entry(quote_frame, textvariable=var, width=30).grid(row=row_index, column=1, sticky="w", padx=5, pady=2)
            self.quote_vars[item_name] = var
            row_index += 1

        def create_global_entries(parent_frame: ttk.Labelframe, keys, data_source, var_dict, columns: int = 2) -> None:
            for i, key in enumerate(keys):
                row, col = divmod(i, columns)
                ttk.Label(parent_frame, text=key).grid(row=row, column=col * 2, sticky="e", padx=5, pady=2)
                var = tk.StringVar(value=str(data_source.get(key, "")))
                entry = ttk.Entry(parent_frame, textvariable=var, width=15)
                if "Path" in key:
                    entry.config(width=50)
                entry.grid(row=row, column=col * 2 + 1, sticky="w", padx=5, pady=2)
                var_dict[key] = var

        comm_frame = ttk.Labelframe(self.editor_widgets_frame, text="Global Overrides: Commercial & General", padding=(10, 5))
        comm_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1
        comm_keys = [
            "OverheadPct", "GA_Pct", "MarginPct", "ContingencyPct",
            "ExpeditePct", "VendorMarkupPct", "InsurancePct", "MinLotCharge", "Quantity",
        ]
        create_global_entries(comm_frame, comm_keys, self.params, self.param_vars)

        rates_frame = ttk.Labelframe(self.editor_widgets_frame, text="Global Overrides: Hourly Rates ($/hr)", padding=(10, 5))
        rates_frame.grid(row=current_row, column=0, sticky="ew", padx=10, pady=5)
        current_row += 1
        create_global_entries(rates_frame, sorted(self.rates.keys()), self.rates, self.rate_vars, columns=3)

    # ----- Full flow: CAD ? GEO ? LLM ? Quote ----- 
    def action_full_flow(self):
        # ---------- choose file ----------
        self.status_var.set("Opening CAD/Drawingï¿½")
        path = filedialog.askopenfilename(
            title="Select CAD/Drawing",
            filetypes=[
                ("CAD/Drawing", "*.step *.stp *.iges *.igs *.brep *.stl *.dwg *.dxf *.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            self.status_var.set("Ready")
            return

        ext = Path(path).suffix.lower()
        self.status_var.set(f"Processing {os.path.basename(path)}ï¿½")

        # ---------- 2D branch: PDF / DWG / DXF ----------
        if ext in (".pdf", ".dwg", ".dxf"):
            try:
                structured_pdf = None
                if ext == ".pdf":
                    structured_pdf = extract_pdf_all(Path(path))
                    g2d = extract_2d_features_from_pdf_vector(path)   # PyMuPDF vector-only MVP
                else:
                    g2d = extract_2d_features_from_dxf_or_dwg(path)   # ezdxf / ODA

                if self.vars_df is None:
                    vp = find_variables_near(path) or filedialog.askopenfilename(title="Select variables CSV/XLSX")
                    if vp:
                        try:
                            core_df, full_df = read_variables_file(vp, return_full=True)
                            self.vars_df = core_df
                            self.vars_df_full = full_df
                        except Exception as read_err:
                            messagebox.showerror("Variables", f"Failed to read variables file:\n{read_err}\n\nContinuing with defaults.")
                            self.vars_df = None
                            self.vars_df_full = None
                    else:
                        messagebox.showinfo("Variables", "No variables file provided; using defaults.")
                        self.vars_df = None
                self.vars_df = coerce_or_make_vars_df(self.vars_df)

                if structured_pdf:
                    try:
                        estimate = infer_pdf_estimate(structured_pdf)
                    except Exception:
                        estimate = {}
                    if estimate:
                        self.vars_df = merge_estimate_into_vars(self.vars_df, estimate)

                self.vars_df = apply_2d_features_to_variables(self.vars_df, g2d, params=self.params, rates=self.rates)
                self.geo = g2d
                self._log_geo(g2d)

                self._populate_editor_tab(self.vars_df)
                self.nb.select(self.tab_editor)
                self.status_var.set(f"{ext.upper()} variables loaded. Review the Quote Editor and generate the quote.")
                return
            except Exception as e:
                messagebox.showerror(
                    "2D Import Error",
                    f"Failed to process {ext.upper()} file:\n{e}\n\n"
                    "Tip (DWG): set ODA_CONVERTER_EXE or DWG2DXF_EXE to a converter that accepts <input.dwg> <output.dxf>."
                )
                self.status_var.set("Ready")
                return

        # ---------- 3D branch: STEP / IGES / BREP / STL ----------
        geo = None
        try:
            # Fast path (your OCC feature extractor for 3D)
            geo = extract_features_with_occ(path)  # handles STEP/IGES/BREP
        except Exception:
            geo = None

        if geo is None:
            if ext == ".stl":
                stl_geo = None
                try:
                    stl_geo = enrich_geo_stl(path)  # trimesh-based
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    messagebox.showerror(
                        "2D Import Error",
                        f"Failed to process STL file:\n{e}\n\n{tb}"
                    )
                    self.status_var.set("Ready")
                    return
                if stl_geo is None:
                    messagebox.showerror(
                        "2D Import Error",
                        "STL import produced no 2D geometry."
                    )
                    self.status_var.set("Ready")
                    return
                geo = _map_geo_to_double_underscore(stl_geo)
            else:
                try:
                    if ext in (".step", ".stp"):
                        shape = read_step_shape(path)
                    else:
                        shape = read_cad_any(path)            # IGES/BREP and others
                    _ = safe_bbox(shape)
                    g = enrich_geo_occ(shape)             # OCC-based geometry features
                    geo = _map_geo_to_double_underscore(g)
                except Exception as e:
                    messagebox.showerror(
                        "CAD Import Error",
                        f"Failed to read CAD file:\n{e}\n\n"
                        "Tip (DWG): upload DXF/STEP/SAT, or set ODA_CONVERTER_EXE to your dwg2dxf wrapper."
                    )
                    self.status_var.set("Ready")
                    return
        # ---------- variables + LLM hours + quote ----------
        if self.vars_df is None:
            vp = find_variables_near(path)
            if not vp:
                vp = filedialog.askopenfilename(title="Select variables CSV/XLSX")
            if not vp:
                messagebox.showinfo("Variables", "No variables file provided.")
                self.status_var.set("Ready")
                return
            try:
                core_df, full_df = read_variables_file(vp, return_full=True)
                self.vars_df = core_df
                self.vars_df_full = full_df
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                messagebox.showerror("Variables", f"Failed to read variables file:\n{e}\n\n{tb}")
                self.status_var.set("Ready")
                return

        self.vars_df = coerce_or_make_vars_df(self.vars_df)

        # Merge GEO rows
        try:
            for k, v in geo.items():
                self.vars_df = upsert_var_row(self.vars_df, k, v, dtype="number")
        except Exception as e:
            messagebox.showerror("Variables", f"Failed to update variables with GEO rows:\n{e}")
            self.status_var.set("Ready")
            return

        # LLM hour estimation
        self.status_var.set("Estimating hours with LLMï¿½")
        decision_log = {}
        est_raw = infer_hours_and_overrides_from_geo(geo, params=self.params, rates=self.rates)
        est = clamp_llm_hours(est_raw, geo, params=self.params)
        self.vars_df = apply_llm_hours_to_variables(self.vars_df, est, allow_overwrite_nonzero=True, log=decision_log)
        self.geo = geo
        self._log_geo(geo)

        self._populate_editor_tab(self.vars_df)
        self.nb.select(self.tab_editor)
        self.status_var.set("Variables loaded. Review the Quote Editor and click Generate Quote.")
        return

    # Back-compat: keep the old button working
    def open_flow(self):
        # Delegate to the new full-flow handler
        return self.action_full_flow()

    def apply_overrides(self, notify: bool = False) -> None:
        raw_param_values = {k: var.get() for k, var in self.param_vars.items()}

        updates = {}
        for key, raw_value in raw_param_values.items():
            if key == "LLMModelPath":
                updates[key] = raw_value
                continue
            try:
                updates[key] = float(str(raw_value).strip())
            except Exception:
                updates[key] = self.params.get(key, 0.0)

        quantity_val = None
        if "Quantity" in updates:
            qty_raw = raw_param_values.get("Quantity", "")
            try:
                qty_value = float(str(qty_raw).strip())
            except Exception:
                qty_value = float(self.params.get("Quantity", 1) or 1)
            if math.isnan(qty_value):
                qty_value = float(self.params.get("Quantity", 1) or 1)
            quantity_val = max(1, int(round(qty_value)))
            updates["Quantity"] = quantity_val

        for key, val in list(updates.items()):
            if key.endswith("Pct") and not isinstance(val, str):
                updates[key] = parse_pct(val)

        self.params.update(updates)

        if quantity_val is not None:
            if "Quantity" in self.param_vars:
                self.param_vars["Quantity"].set(str(quantity_val))

        if self.vars_df is not None and raw_param_values:
            def normalize_item(value: str) -> str:
                cleaned = re.sub(r"[^0-9a-z&$]+", " ", str(value).strip().lower())
                return re.sub(r"\s+", " ", cleaned).strip()
            normalized_items = self.vars_df["Item"].astype(str).apply(normalize_item)
            param_to_items = {
                "OverheadPct": ["overhead %", "overhead"],
                "GA_Pct": ["g&a %", "ga %"],
                "ContingencyPct": ["contingency %"],
                "MarginPct": ["profit margin %", "margin %"],
                "ExpeditePct": ["expedite %"],
                "InsurancePct": ["insurance %"],
                "VendorMarkupPct": ["vendor markup %"],
                "MinLotCharge": ["min lot charge"],
                "Quantity": ["quantity", "qty", "lot size"],
            }
            for key, labels in param_to_items.items():
                if key not in raw_param_values:
                    continue
                value = raw_param_values.get(key, "")
                if key == "Quantity" and quantity_val is not None:
                    value = str(quantity_val)
                elif isinstance(value, str) and not value.strip():
                    value = str(updates.get(key, self.params.get(key, "")))
                for label in labels:
                    mask = normalized_items == label
                    if mask.any():
                        self.vars_df.loc[mask, "Example Values / Options"] = value
                        break

        rate_raw_values = {k: var.get() for k, var in self.rate_vars.items()}
        rate_updates = {}
        for key, raw_value in rate_raw_values.items():
            try:
                rate_updates[key] = float(str(raw_value).strip())
            except Exception:
                rate_updates[key] = self.rates.get(key, 0.0)
        self.rates.update(rate_updates)

        if notify:
            messagebox.showinfo("Overrides", "Overrides applied.")
        self.status_var.set("Overrides applied.")

    def save_overrides(self):

        self.apply_overrides()
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")], initialfile="overrides.json")
        if not path: return
        data = {"params": self.params, "rates": self.rates}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Overrides", f"Saved to:\n{{path}}")
            self.status_var.set(f"Saved overrides to {path}")
        except Exception as e:
            messagebox.showerror("Overrides", f"Save failed:\n{{e}}")
            self.status_var.set("Failed to save overrides.")

    def load_overrides(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "params" in data: self.params.update(data["params"])
            if "rates" in data: self.rates.update(data["rates"])
            for k,v in self.param_vars.items(): v.set(str(self.params.get(k, "")))
            for k,v in self.rate_vars.items():  v.set(str(self.rates.get(k, "")))
            messagebox.showinfo("Overrides", "Overrides loaded.")
            self.status_var.set(f"Loaded overrides from {path}")
        except Exception as e:
            messagebox.showerror("Overrides", f"Load failed:\n{{e}}")
            self.status_var.set("Failed to load overrides.")

    # ----- LLM tab ----- 
    def _build_llm(self, parent):
        row=0
        ttk.Checkbutton(parent, text="Enable LLM (Qwen via llama-cpp, offline)", variable=self.llm_enabled).grid(row=row, column=0, sticky="w", pady=(6,2)); row+=1
        ttk.Label(parent, text="Qwen GGUF model path").grid(row=row, column=0, sticky="e", padx=5, pady=3)
        ttk.Entry(parent, textvariable=self.llm_model_path, width=80).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        ttk.Button(parent, text="Browse...", command=self._pick_model).grid(row=row, column=2, padx=5); row+=1
        ttk.Checkbutton(parent, text="Apply LLM adjustments to params", variable=self.apply_llm_adj).grid(row=row, column=0, sticky="w", pady=(0,6)); row+=1
        ttk.Button(parent, text="Run LLM on current GEO", command=self.run_llm).grid(row=row, column=0, sticky="w", padx=5, pady=6); row+=1
        self.llm_txt = tk.Text(parent, wrap="word", height=24); self.llm_txt.grid(row=row, column=0, columnspan=3, sticky="nsew")
        parent.grid_columnconfigure(1, weight=1); parent.grid_rowconfigure(row, weight=1)

    def _pick_model(self):
        p=filedialog.askopenfilename(title="Choose Qwen *.gguf", filetypes=[("GGUF","*.gguf"),("All","*.*")])
        if p:
            self.llm_model_path.set(p)
            os.environ["QWEN_GGUF_PATH"] = p

    def run_llm(self):
        self.llm_txt.delete("1.0","end")
        if not self.llm_enabled.get():
            self.llm_txt.insert("end","LLM disabled (toggle ON to use it).\n"); return
        if not self.geo:
            messagebox.showinfo("LLM","Load a CAD first so we have GEO context."); return
        mp=self.llm_model_path.get().strip() or os.environ.get("QWEN_GGUF_PATH","")
        if not (mp and Path(mp).is_file() and mp.lower().endswith(".gguf")):
            self.llm_txt.insert("end","No GGUF model found. Put one in D:\\CAD_Quoting_Tool\\models or set QWEN_GGUF_PATH.\n")
            return
        os.environ["QWEN_GGUF_PATH"]=mp
        try:
            out = infer_shop_overrides_from_geo(self.geo)
        except Exception as e:
            self.llm_txt.insert("end", f"LLM error: {{e}}\n"); return
        self.llm_txt.insert("end", json.dumps(out, indent=2))
        if self.apply_llm_adj.get() and isinstance(out, dict):
            adj = out.get("LLM_Adjustments", {})
            try:
                self.params["OverheadPct"] += float(adj.get("OverheadPct_add", 0.0) or 0.0)
                self.params["MarginPct"] += float(adj.get("MarginPct_add", 0.0) or 0.0)
                self.params["ConsumablesFlat"] += float(adj.get("ConsumablesFlat_add", 0.0) or 0.0)
                for k,v in self.param_vars.items(): v.set(str(self.params.get(k, "")))
                messagebox.showinfo("LLM", "Applied LLM adjustments to parameters.")
            except Exception:
                pass

    # ----- Flow + Output ----- 
    def _log_geo(self, d):
        self.geo_txt.delete("1.0","end")
        self.geo_txt.insert("end", json.dumps(d, indent=2))

    def _log_out(self, d):
        self.out_txt.insert("end", d+"\n")
        self.out_txt.see("end")

    def gen_quote(self) -> None:

        if self.vars_df is None:
            self.vars_df = coerce_or_make_vars_df(None)
        for item_name, string_var in self.quote_vars.items():
            mask = self.vars_df["Item"] == item_name
            if mask.any():
                self.vars_df.loc[mask, "Example Values / Options"] = string_var.get()

        self.apply_overrides(notify=False)

        res = compute_quote_from_df(self.vars_df, params=self.params, rates=self.rates)
        model_path = self.llm_model_path.get().strip()
        llm_explanation = get_llm_quote_explanation(res, model_path)
        report = render_quote(res, currency="$", show_zeros=False, llm_explanation=llm_explanation)
        self.out_txt.delete("1.0", "end")
        self.out_txt.insert("end", report, "rcol")

        self.nb.select(self.tab_out)
        self.status_var.set(f"Quote Generated! Final Price: ${res.get('price', 0):,.2f}")

# Helper: map our existing enrich_geo_* output to GEO__ keys
def _map_geo_to_double_underscore(g: dict) -> dict:
    g = dict(g or {})
    out = {}
    def getf(k):
        try:
            return float(g[k])
        except Exception:
            return None
    L = getf("GEO-01_Length_mm")
    W = getf("GEO-02_Width_mm")
    H = getf("GEO-03_Height_mm")
    if L is not None: out["GEO__BBox_X_mm"] = L
    if W is not None: out["GEO__BBox_Y_mm"] = W
    if H is not None: out["GEO__BBox_Z_mm"] = H
    dims = [d for d in [L,W,H] if d is not None]
    if dims:
        out["GEO__MaxDim_mm"] = max(dims)
        out["GEO__MinDim_mm"] = min(dims)
        out["GEO__Stock_Thickness_mm"] = min(dims)
    v = getf("GEO-Volume_mm3") or getf("GEO-Volume_mm3")  # keep both spellings if present
    if v is not None: out["GEO__Volume_mm3"] = v
    a = getf("GEO-SurfaceArea_mm2")
    if a is not None: out["GEO__SurfaceArea_mm2"] = a
    fc = getf("Feature_Face_Count") or getf("GEO_Face_Count")
    if fc is not None: out["GEO__Face_Count"] = fc
    wedm = getf("GEO_WEDM_PathLen_mm")
    if wedm is not None: out["GEO__WEDM_PathLen_mm"] = wedm
    # derived area/volume ratio if possible
    if a is not None and v:
        try:
            out["GEO__Area_to_Volume"] = a / v if v else 0.0
        except Exception:
            pass
    return out

def _collect_geo_features_from_df(df):
    geo = {}
    if df is None:
        return geo
    items = df["Item"].astype(str)
    vals  = df["Example Values / Options"]
    m = items.str.startswith("GEO__", na=False)
    for _, row in df.loc[m].iterrows():
        k = str(row["Item"]).strip()
        try:
            geo[k] = float(row["Example Values / Options"]) if row["Example Values / Options"] is not None else 0.0
        except Exception:
            continue
    return geo

def update_variables_df_with_geo(df, geo: dict):
    cols = ["Item","Example Values / Options","Data Type / Input Method"]
    for col in cols:
        if col not in df.columns:
            raise ValueError("Variables sheet missing column: " + col)
    items = df["Item"].astype(str)
    for k, v in geo.items():
        m = items.str.fullmatch(k, case=False)
        if m.any():
            df.loc[m, "Example Values / Options"] = v
            df.loc[m, "Data Type / Input Method"] = "number"
        else:
            df.loc[len(df)] = [k, v, "number"]
    return df

def _rule_based_overrides(geo: dict, params: dict, rates: dict):
    rp, rr = {}, {}
    thk = float(geo.get("GEO__Stock_Thickness_mm", 0.0) or 0.0)
    wedm = float(geo.get("GEO__WEDM_PathLen_mm", 0.0) or 0.0)
    if thk > 50: rparams["OverheadPct"] = params.get("OverheadPct", 0.15) + 0.02
    if wedm > 500: rrates["WireEDMRate"] = rates.get("WireEDMRate", 140.0) * 1.05
    return {"params": rp, "rates": rr}

def _run_llm_json_stub(prompt: str, model_path: str):
    return {"params": {}, "rates": {}}

def suggest_overrides_from_cad(df_vars, params, rates, model_path: str):
    geo = _collect_geo_features_from_df(df_vars)
    base = _rule_based_overrides(geo, params, rates)
    llm_json = _run_llm_json_stub("", model_path)
    merged_p = {**params, **base.get("params", {}), **llm_json.get("params", {})}
    merged_r = {**rates,  **base.get("rates", {}),  **llm_json.get("rates", {})}
    return {"params": merged_p, "rates": merged_r, "geo": geo}

if __name__ == "__main__":
    App().mainloop()
# Base dir for logs and resources
BASE_DIR = Path(__file__).resolve().parent


