# cad_quoter/geo_extractor.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Dict, Any, List, Tuple

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


# --------------------------- text normalization ---------------------------

_PLAIN_REPLACERS = [
    (r"\\P", "\n"),   # MTEXT paragraph
    (r"\\~", " "),    # NBSP-ish
    (r"\\[Xx][\\;]", ""),  # strikeout on/off, etc. (best effort)
]

def _plain(s: str) -> str:
    if not s:
        return ""
    out = s
    for pat, rep in _PLAIN_REPLACERS:
        out = re.sub(pat, rep, out)
    out = out.replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return out.strip()


# --------------------------- entity flattening ---------------------------

_TEXT_TYPES = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "MLEADER"}
_TABLE_TYPES = {"ACAD_TABLE", "TABLE", "MTABLE"}  # cover modern + legacy tables

def _entity_text_fields(e) -> Tuple[str, float, float, float]:
    """(text, height, x, y). Best effort across TEXT/MTEXT/ATTRIB/MLEADER."""
    et = e.dxftype()
    txt = ""
    h = 0.0
    x = y = 0.0

    try:
        if et == "TEXT":
            txt = getattr(e.dxf, "text", "") or ""
            h = float(getattr(e.dxf, "height", 0.0) or 0.0)
            ins = getattr(e.dxf, "insert", None)
            if ins is not None:
                x, y = float(ins.x), float(ins.y)

        elif et == "ATTRIB" or et == "ATTDEF":
            txt = getattr(e.dxf, "text", "") or ""
            h = float(getattr(e.dxf, "height", 0.0) or 0.0)
            ins = getattr(e.dxf, "insert", None)
            if ins is not None:
                x, y = float(ins.x), float(ins.y)

        elif et == "MTEXT":
            # ezdxf MTEXT has .text (raw) and .plain_text() helpers in newer versions
            raw = getattr(e, "text", "") or ""
            try:
                txt = e.plain_text()  # type: ignore[attr-defined]
            except Exception:
                txt = raw
            txt = _plain(txt or raw)
            h = float(getattr(e.dxf, "char_height", 0.0) or 0.0)
            ins = getattr(e.dxf, "insert", None)
            if ins is not None:
                x, y = float(ins.x), float(ins.y)

        elif et == "MLEADER":
            # best-effort: some ezdxf versions expose context.mtext
            ctx = getattr(e, "context", None)
            mt = getattr(ctx, "mtext", None)
            if mt is not None:
                raw = getattr(mt, "text", "") or ""
                try:
                    txt = mt.plain_text()  # type: ignore[attr-defined]
                except Exception:
                    txt = raw
                txt = _plain(txt or raw)
            else:
                # fallback: description
                txt = _plain(str(getattr(e.dxf, "text", "") or ""))
            # position is fuzzy; grab the MLEADER’s insertion if present
            ins = getattr(e.dxf, "insert", None)
            if ins is not None:
                x, y = float(ins.x), float(ins.y)

    except Exception:
        pass

    return str(txt or ""), float(h or 0.0), float(x or 0.0), float(y or 0.0)


def _iter_virtual(e) -> List[Any]:
    """Return virtual children for INSERT, else []."""
    try:
        return list(e.virtual_entities())
    except Exception:
        return []


def _collect_table_cells(e) -> List[str]:
    """
    Return flattened cell texts for a TABLE-like entity, best-effort across
    ezdxf versions and entity variants (ACAD_TABLE, TABLE, MTABLE).
    """
    payload: List[str] = []
    try:
        # Try both naming schemes used by ezdxf over time
        nrows = int(getattr(e, "nrows", getattr(e, "n_rows", 0)) or 0)
        ncols = int(getattr(e, "ncols", getattr(e, "n_cols", 0)) or 0)

        # Preferred: explicit row/col scan
        for r in range(nrows):
            for c in range(ncols):
                cell = None
                # Try the most common getters first
                for getter in ("get_cell", "cell", "getCell"):
                    try:
                        g = getattr(e, getter)
                        cell = g((r, c)) if getter == "get_cell" else g(r, c)
                        break
                    except Exception:
                        continue

                text = ""
                if cell is not None:
                    # Newer ezdxf: cell.plain_text() or cell.content.plain_text()
                    for attr in ("plain_text", "text", "value"):
                        try:
                            v = getattr(cell, attr, None)
                            if v is None and hasattr(cell, "content"):
                                v = getattr(cell.content, attr, None)
                            if callable(v):
                                v = v()
                            if v:
                                text = str(v)
                                break
                        except Exception:
                            continue
                if text:
                    payload.append(_plain(text))

        # Fallback: some MTABLE exports store a linear list of cell-like items
        if not payload:
            try:
                for cell in getattr(e, "cells", []):
                    for attr in ("plain_text", "text", "value"):
                        v = getattr(cell, attr, None)
                        if callable(v):
                            v = v()
                        if v:
                            payload.append(_plain(str(v)))
                            break
            except Exception:
                pass
    except Exception:
        pass
    return payload


def iter_text(
    layout_obj,
    *,
    layout_name: str,
    max_block_depth: int = 3,
    include_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    min_height: float = 0.0,
) -> Iterator[TextRecord]:
    """
    Yield TextRecord for TEXT/MTEXT/ATTRIB/MLEADER and TABLE cell strings.
    - Follows INSERTs recursively via virtual_entities() up to max_block_depth.
    - No “anchor” or HOLE parsing. Just a raw dump.
    """

    include_layers = include_layers or []  # regex allowlist (any match)
    exclude_layers = exclude_layers or []  # regex blocklist (any match)

    def layer_allowed(layer: str) -> bool:
        if any(re.search(p, layer, flags=re.I) for p in exclude_layers):
            return False
        if include_layers and not any(re.search(p, layer, flags=re.I) for p in include_layers):
            return False
        return True

    stack: List[Tuple[Any, Tuple[str, ...], int, bool]] = []
    # seed with top-level entities
    for ent in layout_obj:
        stack.append((ent, tuple(), 0, False))

    while stack:
        ent, block_path, depth, from_block = stack.pop()
        et = ent.dxftype()
        layer = str(getattr(ent.dxf, "layer", "") or "")

        if et == "INSERT" and depth < max_block_depth:
            # Recurse into the virtualized children (already transformed by ezdxf).
            children = _iter_virtual(ent)
            child_block_name = str(getattr(ent.dxf, "name", "") or "")
            child_path = block_path + ((child_block_name or "INSERT"),)
            for ch in children:
                stack.append((ch, child_path, depth + 1, True))
            continue

        # TABLE-like → flatten cell texts
        if et in _TABLE_TYPES:
            if layer_allowed(layer):
                for s in _collect_table_cells(ent):
                    if s:
                        yield TextRecord(
                            layout=layout_name,
                            layer=layer,
                            etype="TABLECELL",
                            text=s,
                            x=float(getattr(getattr(ent.dxf, "insert", None) or 0, "x", 0.0) or 0.0),
                            y=float(getattr(getattr(ent.dxf, "insert", None) or 0, "y", 0.0) or 0.0),
                            height=0.0,
                            rotation=float(getattr(ent.dxf, "rotation", 0.0) or 0.0),
                            in_block=from_block,
                            depth=depth,
                            block_path=block_path,
                        )
            continue

        if et not in _TEXT_TYPES:
            continue

        if not layer_allowed(layer):
            continue

        text, height, x, y = _entity_text_fields(ent)
        if min_height and height < float(min_height):
            # keep SEE SHEET etc. anyway if height unknown
            if not text:
                continue

        if text is None or str(text).strip() == "":
            continue

        yield TextRecord(
            layout=layout_name,
            layer=layer,
            etype=et,
            text=_plain(str(text)),
            x=float(x or 0.0),
            y=float(y or 0.0),
            height=float(height or 0.0),
            rotation=float(getattr(ent.dxf, "rotation", 0.0) or 0.0),
            in_block=from_block,
            depth=depth,
            block_path=block_path,
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
