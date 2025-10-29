# cad_quoter/geo_extractor.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Dict, Any, List, Tuple

try:
    # ezdxf ≥ 1.0 style
    from ezdxf.entities import ProxyGraphic  # fallback name in some versions  # noqa: F401
except Exception:
    ProxyGraphic = None

# ezdxf import (prefer your vendored copy if present)
try:
    from cad_quoter.vendors import ezdxf as ezdxf  # type: ignore
except Exception:
    import ezdxf  # type: ignore

# Optional DWG→DXF converter; if missing and input is DWG, we raise a clear error.
try:
    from cad_quoter.geometry import convert_dwg_to_dxf  # type: ignore
except Exception:
    convert_dwg_to_dxf = None  # type: ignore

# Try to use Matrix44 transforms for INSERT.virtual_entities(); if absent, we still work.
try:
    from ezdxf.math import Matrix44  # type: ignore
except Exception:
    Matrix44 = None  # type: ignore


# --------------------------- public data model ---------------------------

@dataclass(frozen=True)
class TextRecord:
    layout: str
    layer: str
    etype: str
    text: str
    x: float
    y: float
    height: float
    rotation: float
    in_block: bool
    depth: int
    block_path: Tuple[str, ...]


# --------------------------- open + layout helpers ---------------------------

def open_doc(path: Path):
    """Open DXF directly; DWG via ODA/Teigha converter if available."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    def _readfile(p: Path):
        """Return doc using whichever ezdxf interface is available."""
        reader = getattr(ezdxf, "readfile", None)
        if callable(reader):
            return reader(str(p))
        require_fn = getattr(ezdxf, "require_ezdxf", None)
        if callable(require_fn):
            ez_mod = require_fn("DXF reading")
            mod_reader = getattr(ez_mod, "readfile", None)
            if callable(mod_reader):
                return mod_reader(str(p))
        raise AttributeError("cad_quoter.vendors.ezdxf does not expose readfile()")

    if path.suffix.lower() == ".dxf":
        return _readfile(path)

    if path.suffix.lower() == ".dwg":
        if convert_dwg_to_dxf is None:
            raise RuntimeError(
                "DWG input requires a DWG→DXF converter (convert_dwg_to_dxf not available)."
            )
        dxf_path = Path(convert_dwg_to_dxf(str(path)))
        return _readfile(dxf_path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def iter_layouts(doc, names: Optional[Iterable[str]] = None) -> Iterator[Tuple[str, Any]]:
    """Yield (layout_name, layout_obj). If names is None → yield all layouts."""
    if names:
        name_set = {n.strip() for n in names if str(n).strip()}
        for n in name_set:
            if n in doc.layouts:
                yield n, doc.layouts.get(n)
        return

    # Default: all layouts (Model + all paperspace layouts)
    for layout in doc.layouts:
        yield layout.name, layout


_TABLE_TYPES = {"ACAD_TABLE", "TABLE", "MTABLE"}
_TEXT_TYPES = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "DIMENSION", "MLEADER"}


def _plain(s: str) -> str:
    if not s:
        return ""
    # Strip MTEXT control codes: \P newlines, %%c diameter symbol, etc.
    s = s.replace("\\P", "\n").replace("\\p", "\n")
    s = s.replace("%%c", "Ø").replace("%%d", "°").replace("%%p", "±")
    return re.sub(r"\s+", " ", s).strip()


def _mtext_to_str(m) -> str:
    raw = getattr(m, "text", "") or ""
    try:
        return _plain(m.plain_text())  # newer ezdxf
    except Exception:
        return _plain(raw)


def _collect_table_cells(tbl) -> List[str]:
    out: List[str] = []
    try:
        nrows = int(getattr(tbl, "nrows", getattr(tbl, "n_rows", 0)) or 0)
        ncols = int(getattr(tbl, "ncols", getattr(tbl, "n_cols", 0)) or 0)
        for r in range(nrows):
            for c in range(ncols):
                cell = None
                for getter in ("get_cell", "cell", "getCell"):
                    try:
                        g = getattr(tbl, getter)
                        cell = g((r, c)) if getter == "get_cell" else g(r, c)
                        break
                    except Exception:
                        continue
                txt = ""
                if cell is not None:
                    for attr in ("plain_text", "text", "value"):
                        v = getattr(cell, attr, None)
                        if v is None and hasattr(cell, "content"):
                            v = getattr(cell.content, attr, None)
                        if callable(v):
                            try:
                                v = v()
                            except Exception:
                                v = None
                        if v:
                            txt = str(v)
                            break
                if txt:
                    out.append(_plain(txt))
        if not out:
            # Fallback: some MTABLE exports expose .cells
            for cell in getattr(tbl, "cells", []):
                for attr in ("plain_text", "text", "value"):
                    v = getattr(cell, attr, None)
                    if callable(v):
                        v = v()
                    if v:
                        out.append(_plain(str(v)))
                        break
    except Exception:
        pass
    return out


def _proxy_texts(ent) -> List[str]:
    """
    Extract TEXT/MTEXT rendered inside ACAD_PROXY_ENTITY via proxy graphics.
    Tries multiple ezdxf APIs to stay compatible across versions.
    """

    texts: List[str] = []
    try:
        loaders = []
        try:
            # ezdxf 1.0+ common path
            from ezdxf.proxygraphic import load_proxy_graphic as _load_pg

            loaders.append(lambda e: _load_pg(e))
        except Exception:
            pass
        try:
            # older shape
            from ezdxf.proxygraphic import ProxyGraphic as _PG

            loaders.append(lambda e: _PG.load(e))
        except Exception:
            pass
        # last-ditch: some builds expose raw bytes at e.proxy_graphic
        loaders.append(lambda e: getattr(e, "proxy_graphic", None))

        for load in loaders:
            try:
                pg = load(ent)
                if pg is None:
                    continue
                # Normalize to an object with .virtual_entities()
                if hasattr(pg, "virtual_entities"):
                    viter = pg.virtual_entities()
                else:
                    # If we only got raw bytes, see if a ProxyGraphic class can parse them
                    from ezdxf.proxygraphic import ProxyGraphic as _PG

                    pg2 = _PG(bytes(pg))
                    viter = pg2.virtual_entities()
                for v in viter:
                    t = v.dxftype()
                    if t == "TEXT":
                        s = _plain(getattr(v.dxf, "text", "") or "")
                        if s:
                            texts.append(s)
                    elif t == "MTEXT":
                        s = _mtext_to_str(v)
                        if s:
                            texts.append(s)
                if texts:
                    break
            except Exception:
                continue
    except Exception:
        pass
    return texts


def iter_text_records(
    layout_name: str,
    layout_obj,
    include_layers=None,
    exclude_layers=None,
    max_block_depth: int = 8,
    min_height: float = 0.0,
) -> Iterator[TextRecord]:

    include_layers = include_layers or []
    exclude_layers = exclude_layers or []

    def layer_ok(layer: str) -> bool:
        if any(re.search(p, layer, flags=re.I) for p in exclude_layers):
            return False
        if include_layers and not any(re.search(p, layer, flags=re.I) for p in include_layers):
            return False
        return True

    stack: List[Tuple[Any, Tuple[str, ...], int, bool]] = [(e, tuple(), 0, False) for e in layout_obj]

    while stack:
        ent, block_path, depth, from_block = stack.pop()
        et = ent.dxftype()
        layer = str(getattr(ent.dxf, "layer", "") or "")

        # Recurse into blocks
        if et == "INSERT" and depth < max_block_depth:
            try:
                for v in ent.virtual_entities():
                    stack.append((v, block_path + (ent.dxf.name,), depth + 1, True))
            except Exception:
                pass
            continue

        # Tables
        if et in _TABLE_TYPES:
            if layer_ok(layer):
                for s in _collect_table_cells(ent):
                    yield TextRecord(
                        layout_name,
                        layer,
                        "TABLECELL",
                        s,
                        0.0,
                        0.0,
                        0.0,
                        float(getattr(ent.dxf, "rotation", 0.0) or 0.0),
                        from_block,
                        depth,
                        block_path,
                    )
            continue

        # AutoCAD Mechanical (and other) custom objects as proxies
        if et == "ACAD_PROXY_ENTITY":
            # Optional filter: many AM hole charts have class names w/ HOLECHART
            cla = str(getattr(ent.dxf, "proxy_entity_class", "") or "")
            if (not cla) or ("HOLE" in cla.upper() or "CHART" in cla.upper()):
                for s in _proxy_texts(ent):
                    yield TextRecord(
                        layout_name,
                        layer,
                        "PROXYTEXT",
                        s,
                        0.0,
                        0.0,
                        0.0,
                        float(getattr(ent.dxf, "rotation", 0.0) or 0.0),
                        from_block,
                        depth,
                        block_path,
                    )
            continue

        # Plain text-ish
        if et not in _TEXT_TYPES or not layer_ok(layer):
            continue

        txt, h, x, y = None, 0.0, 0.0, 0.0
        if et == "TEXT":
            txt = getattr(ent.dxf, "text", "") or ""
            h = float(getattr(ent.dxf, "height", 0.0) or 0.0)
            ins = getattr(ent.dxf, "insert", None)
            if ins is not None:
                x, y = float(ins.x), float(ins.y)
        elif et == "MTEXT":
            txt = _mtext_to_str(ent)
            h = float(getattr(ent.dxf, "char_height", 0.0) or 0.0)
            ins = getattr(ent.dxf, "insert", None)
            if ins is not None:
                x, y = float(ins.x), float(ins.y)
        elif et in {"ATTRIB", "ATTDEF"}:
            txt = getattr(ent.dxf, "text", "") or ""
            h = float(getattr(ent.dxf, "height", 0.0) or 0.0)
            ins = getattr(ent.dxf, "insert", None)
            if ins is not None:
                x, y = float(ins.x), float(ins.y)
        elif et == "MLEADER":
            ctx = getattr(ent, "context", None)
            mt = getattr(ctx, "mtext", None)
            if mt is not None:
                txt = _mtext_to_str(mt)
        elif et == "DIMENSION":
            # Some variants carry measurement text at .dxf.text
            txt = getattr(ent.dxf, "text", "") or ""

        if txt:
            txt = _plain(str(txt))
            if (not min_height) or (h >= float(min_height)) or txt:
                yield TextRecord(
                    layout_name,
                    layer,
                    et,
                    txt,
                    x,
                    y,
                    h,
                    float(getattr(ent.dxf, "rotation", 0.0) or 0.0),
                    from_block,
                    depth,
                    block_path,
                )


def iter_text(
    layout_obj,
    *,
    layout_name: str,
    max_block_depth: int = 3,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    min_height: float = 0.0,
) -> Iterator[TextRecord]:
    yield from iter_text_records(
        layout_name,
        layout_obj,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
        max_block_depth=max_block_depth,
        min_height=min_height,
    )


def collect_all_text(
    doc,
    *,
    layouts: Optional[Iterable[str]] = None,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    min_height: float = 0.0,
    max_block_depth: int = 3,
) -> List[Dict[str, Any]]:
    """Convenience wrapper: iterate requested layouts and return dict rows."""
    rows: List[Dict[str, Any]] = []
    for lname, lobj in iter_layouts(doc, layouts):
        for rec in iter_text(
            lobj,
            layout_name=lname,
            max_block_depth=max_block_depth,
            include_layers=include_layers,
            exclude_layers=exclude_layers,
            min_height=min_height,
        ):
            rows.append(
                {
                    "layout": rec.layout,
                    "layer": rec.layer,
                    "etype": rec.etype,
                    "text": rec.text,
                    "x": rec.x,
                    "y": rec.y,
                    "height": rec.height,
                    "rotation": rec.rotation,
                    "in_block": rec.in_block,
                    "depth": rec.depth,
                    "block_path": list(rec.block_path),
                }
            )
    return rows
