r"""
geo_extractor.py
================
Low-level CAD text extractor.

Purpose:
    Open DWG/DXF, walk entities, and normalize ALL human-visible text
    (TEXT, MTEXT, TABLE, ACAD_PROXY_ENTITY fragments, etc.) into a
    machine-friendly stream.

What it emits:
    - dxf_text_dump.csv   # rows like: Space,Layer,Kind,Text,x,y,rot,…
    - dxf_text_dump.jsonl # same content as JSON Lines

Key behaviors:
    - Reassembles proxy-entity fragments (HOLE TABLEs often live here).
    - Decodes odd encodings (e.g., "\\U+2205" → "∅"), normalizes quotes.
    - Leaves layout/geometry to higher layers; this module is a “CAD → plain text” vacuum.

Typical pipeline:
    DWG/DXF → geo_extractor → dxf_text_dump.(csv|jsonl) → geo_dump → hole_ops
"""

from __future__ import annotations

import re
import fractions
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

# Optional DWG→DXF converter imported lazily to avoid circular dependencies
convert_dwg_to_dxf = None  # Will be imported when needed

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
        """Return doc using whichever ezdxf interface is available.

        Tries normal readfile first, then falls back to recover.readfile
        if the DXF file has structure errors (e.g., missing ENDSEC tags).
        """
        reader = getattr(ezdxf, "readfile", None)
        if callable(reader):
            try:
                return reader(str(p))
            except Exception as e:
                # Try recover mode for malformed DXF files
                error_name = type(e).__name__
                if "Structure" in error_name or "ENDSEC" in str(e):
                    print(f"[INFO] DXF structure error, trying recovery mode: {e}")
                    recover = getattr(ezdxf, "recover", None)
                    if recover and hasattr(recover, "readfile"):
                        try:
                            doc, auditor = recover.readfile(str(p))
                            if auditor and auditor.has_errors:
                                print(f"[WARN] DXF recovered with {len(auditor.errors)} errors")
                            return doc
                        except Exception as recover_err:
                            print(f"[ERROR] Recovery also failed: {recover_err}")
                            raise e  # Raise original error
                raise  # Re-raise if not a structure error

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
        # Lazy import to avoid circular dependency
        try:
            from cad_quoter.geometry import convert_dwg_to_dxf as _convert
        except Exception:
            _convert = None

        if _convert is None:
            raise RuntimeError(
                "DWG input requires a DWG→DXF converter (convert_dwg_to_dxf not available)."
            )
        dxf_path = Path(_convert(str(path)))
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


_UHEX_RE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")


def _decode_uplus(s: str) -> str:
    def _sub(m):
        try:
            cp = int(m.group(1), 16)
            return chr(cp)
        except Exception:
            return m.group(0)

    return _UHEX_RE.sub(_sub, s)


def _plain(s: str) -> str:
    if not s:
        return ""
    # Strip MTEXT control codes: \P newlines, %%c diameter symbol, etc.
    s = s.replace("\\P", "\n").replace("\\p", "\n")
    s = s.replace("%%c", "Ø").replace("%%d", "°").replace("%%p", "±")

    # Handle AutoCAD Mechanical GD&T font character codes
    # Format: {\Famgdt|c0;X} where X is a character code mapping to GD&T symbols
    # Common mappings in amgdt font:
    #   q = Ø (diameter)
    #   h = ⌭ (perpendicularity)
    #   u = ⏥ (position)
    #   etc.
    gdt_char_map = {
        'q': 'Ø',  # diameter
        'h': '⌭',  # perpendicularity
        'u': '⏥',  # position
        'n': '⌖',  # concentricity
        'o': '⊕',  # all around
        'p': '⌯',  # flatness
        'r': '⏄',  # parallelism
    }

    # Pattern: {\Famgdt|c0;X} or {\FAMGDT|c0;X}
    def replace_gdt_char(m):
        char_code = m.group(1).lower()
        return gdt_char_map.get(char_code, char_code)

    s = re.sub(r'\{\\[Ff][Aa][Mm][Gg][Dd][Tt]\|c0;(.)\}', replace_gdt_char, s)

    return re.sub(r"\s+", " ", s).strip()


def _mtext_to_str(m) -> str:
    raw = getattr(m, "text", "") or ""
    try:
        return _decode_uplus(_plain(m.plain_text()))  # newer ezdxf
    except Exception:
        return _decode_uplus(_plain(raw))


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
                    out.append(_decode_uplus(_plain(txt)))
        if not out:
            # Fallback: some MTABLE exports expose .cells
            for cell in getattr(tbl, "cells", []):
                for attr in ("plain_text", "text", "value"):
                    v = getattr(cell, attr, None)
                    if callable(v):
                        v = v()
                    if v:
                        out.append(_decode_uplus(_plain(str(v))))
                        break
    except Exception:
        pass
    return out


def _merge_proxy_fragments(lines: Iterable[str]) -> List[str]:
    """Merge split proxy-text fragments that belong to the same logical row.

    AutoCAD's HOLE TABLE proxy graphics frequently break a single row across
    multiple short text snippets. The heuristics below concatenate fragments
    until the buffer "feels" complete: we keep merging when the row ends with a
    dangling token (``FROM``, ``&``, ``BACK`` …) or punctuation that usually
    indicates continuation, or when the following fragment is a short
    all-caps clause such as ``FROM FRONT`` or ``BACK AS SHOWN``.
    """

    merged: List[str] = []
    buf = ""

    def push() -> None:
        nonlocal buf
        if buf:
            merged.append(" ".join(buf.split()))
            buf = ""

    fragments = [s.strip() for s in lines if s and s.strip()]
    if not fragments:
        return merged

    dangling = {"FROM", "FRONT", "BACK", "&", "AS", "SHOWN", "DEEP"}

    def short_all_caps(s: str) -> bool:
        cleaned = re.sub(r"[^A-Z0-9& ]+", "", s.strip())
        if not cleaned:
            return False
        if cleaned != cleaned.upper():
            return False
        return len(cleaned) <= 18

    for idx, frag in enumerate(fragments):
        buf = f"{buf} {frag}".strip() if buf else frag

        stripped = buf.rstrip()
        if not stripped:
            continue
        end = stripped[-1:]
        last_word = stripped.split()[-1]

        if end in ";,(":
            continue
        if last_word.upper() in dangling:
            continue
        next_frag = fragments[idx + 1] if idx + 1 < len(fragments) else ""
        if next_frag and short_all_caps(next_frag):
            continue
        push()

    push()
    return merged


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
                        s = _decode_uplus(_plain(getattr(v.dxf, "text", "") or ""))
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
    if not texts:
        return texts
    return _merge_proxy_fragments(texts)


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
            # Get dimension type to detect radial/diameter dimensions
            # dimtype values: 0=linear, 1=aligned, 2=angular, 3=diameter, 4=radius, 6=ordinate
            dimtype = getattr(ent.dxf, "dimtype", 0) if hasattr(ent, "dxf") else 0

            # Some variants carry measurement text at .dxf.text
            txt = getattr(ent.dxf, "text", "") or ""

            # Resolve <> placeholder with actual dimension measurement
            if "<>" in txt or not txt:
                try:
                    meas = ent.get_measurement()
                    if meas is not None:
                        if hasattr(meas, "magnitude"):
                            meas = meas.magnitude
                        # Assume drawing units are inches
                        meas_in = float(meas)
                        meas_str = f"{meas_in:.4f}".rstrip("0").rstrip(".")
                        if meas_str.startswith("0."):
                            meas_str = meas_str[1:]

                        # Add prefix for radial/diameter dimensions if not already present
                        if dimtype == 4:  # Radial dimension
                            if not txt or txt == "<>":
                                meas_str = f"R{meas_str}"
                            elif "<>" in txt:
                                # Check if there's already an R prefix in the text
                                if not re.match(r'^\s*R\s*<>', txt, re.IGNORECASE):
                                    meas_str = f"R{meas_str}"
                        elif dimtype == 3:  # Diameter dimension
                            if not txt or txt == "<>":
                                meas_str = f"Ø{meas_str}"
                            elif "<>" in txt:
                                # Check if there's already a diameter symbol
                                if not any(c in txt for c in ["Ø", "%%c", "DIA"]):
                                    meas_str = f"Ø{meas_str}"

                        if "<>" in txt:
                            txt = txt.replace("<>", meas_str)
                        elif not txt:
                            txt = meas_str
                except Exception:
                    pass

        if txt:
            txt = _decode_uplus(_plain(str(txt)))
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


# --------------------------- hole row parsing helpers ---------------------------

RE_REF = re.compile(r'^\s*(?:REF\s*)?["\']?([A-Z])["\']?\b')
RE_DIAM = re.compile(r'[Ø⌀\u00D8]\s*([0-9./]+)')
RE_TAP = re.compile(r'([0-9./#]+)\s*[-–]\s*([0-9]+)\s*TAP', re.IGNORECASE)
RE_DEPTH = re.compile(r'[Xx]\s*([0-9.]+)\s*DEEP')
RE_SIDE = re.compile(r'FROM\s+(FRONT|BACK)', re.IGNORECASE)

_QTY_PATTERNS = (
    re.compile(r'\bQTY\s+["\']?([A-Z])["\']?\s*[:=]?\s*(\d+)', re.IGNORECASE),
    re.compile(r'\((\d+)\)\s*["\']?([A-Z])["\']?', re.IGNORECASE),
    re.compile(r'["\']?([A-Z])["\']?\s*(?:-|=)?\s*(\d+)\s*(?:X|EA|EACH|PL)', re.IGNORECASE),
)

# Pattern to extract overall part quantity from title blocks or parts lists
# Matches format like: "A 134 8 A2 SPRING PIN ... QTY MATERIAL..."
# where 8 is the quantity appearing before the QTY label
# This pattern finds all numbers, and we'll select the one closest to QTY
_PART_QTY_PATTERN = re.compile(
    r'\b(\d{1,4})\b(?=.*?\bQTY\b)',
    re.IGNORECASE
)


def _to_float(token: str) -> Optional[float]:
    token = (token or "").strip()
    if not token:
        return None
    token = token.replace("O/", "Ø")  # guard against mis-read Ø tokens
    if "/" in token:
        try:
            return float(fractions.Fraction(token))
        except Exception:
            pass
    try:
        return float(token)
    except Exception:
        return None


def parse_hole_row(text: str) -> Dict[str, Any]:
    """Extract structured details from a textual HOLE TABLE row."""

    raw = text or ""
    normalized = raw.strip()

    result: Dict[str, Any] = {
        "ref": None,
        "diam": None,
        "tap_thread": None,
        "op": None,
        "ops": [],
        "depth": None,
        "side": None,
        "raw": raw,
    }

    match = RE_REF.search(normalized)
    if match:
        result["ref"] = match.group(1).upper()

    match = RE_DIAM.search(normalized)
    if match:
        result["diam"] = _to_float(match.group(1))

    match = RE_TAP.search(normalized)
    if match:
        result["tap_thread"] = f"{match.group(1)}-{match.group(2)}"

    upper = normalized.upper()
    ops: List[str] = []
    if "C'BORE" in upper or "COUNTERBORE" in upper:
        ops.append("CBORE")
    if "C'DRILL" in upper or "COUNTERDRILL" in upper or "CTR DRILL" in upper:
        ops.append("CDRILL")
    if "TAP" in upper:
        ops.append("TAP")
    if "THRU" in upper:
        ops.append("THRU")

    if ops:
        result["ops"] = ops
        result["op"] = ops[0]

    match = RE_DEPTH.search(normalized)
    if match:
        result["depth"] = _to_float(match.group(1))

    match = RE_SIDE.search(normalized)
    if match:
        result["side"] = match.group(1).upper()

    return result


def _collect_qty_hints(lines: Iterable[str]) -> Dict[str, int]:
    qty: Dict[str, int] = {}
    for line in lines:
        if not line:
            continue
        for pattern in _QTY_PATTERNS:
            for match in pattern.finditer(line):
                ref = match.group(1)
                count_token = match.group(2)
                if not ref or not count_token:
                    continue
                try:
                    qty[ref.upper()] = int(count_token)
                except Exception:
                    continue
    return qty


def extract_part_quantity_from_text(text_records) -> int:
    """
    Extract overall part quantity from CAD text records.

    Looks for patterns like "A 134 8 A2 SPRING PIN ... QTY MATERIAL..."
    where a number appears before the "QTY" label (typically in title blocks
    or parts lists). When multiple numbers are found before QTY, prefers the
    one closest to the QTY keyword.

    Args:
        text_records: Iterable of text strings OR dicts with a 'text' field

    Returns:
        Extracted quantity if found and confident, otherwise 1 (default)

    Examples:
        >>> records = [{"text": "A 134 8 A2 SPRING PIN 56-58 ROCK C REV DETAIL QTY MATERIAL"}]
        >>> extract_part_quantity_from_text(records)
        8

        >>> text_list = ["A 134 8 A2 SPRING PIN 56-58 ROCK C REV DETAIL QTY MATERIAL"]
        >>> extract_part_quantity_from_text(text_list)
        8

        >>> records = [{"text": "NO QTY INFO HERE"}]
        >>> extract_part_quantity_from_text(records)
        1
    """
    candidates = []

    for record in text_records:
        # Handle both string and dict inputs
        if isinstance(record, str):
            text = record
        elif isinstance(record, dict):
            text = record.get("text", "")
        else:
            continue

        if not text or "QTY" not in text.upper():
            continue

        # Find position of QTY keyword (case-insensitive)
        qty_match = re.search(r'\bQTY\b', text, re.IGNORECASE)
        if not qty_match:
            continue
        qty_pos = qty_match.start()

        # Find all numbers before QTY and their positions
        text_before_qty = text[:qty_pos]
        number_matches = []

        # Match standalone numbers, but not those in ranges like "56-58" or fractions like "1/4"
        for match in re.finditer(r'(?<![0-9/-])\b(\d{1,4})\b(?![0-9/-])', text_before_qty):
            qty_str = match.group(1)
            try:
                qty = int(qty_str)
                # Sanity check: reasonable quantity range
                if 1 <= qty <= 9999:
                    # Store (quantity, distance_to_qty)
                    distance = qty_pos - match.end()
                    number_matches.append((qty, distance))
            except (ValueError, TypeError):
                continue

        # If we found numbers, prefer the one closest to QTY (smallest distance)
        if number_matches:
            # Sort by distance (ascending) - closest first
            number_matches.sort(key=lambda x: x[1])
            # Take the closest number
            candidates.append(number_matches[0][0])

    # If we found exactly one candidate, use it with confidence
    if len(candidates) == 1:
        return candidates[0]

    # If we found multiple candidates, check if they're all the same
    if len(candidates) > 1:
        unique_candidates = set(candidates)
        if len(unique_candidates) == 1:
            # All matches agree on the same quantity
            return candidates[0]
        # Multiple different quantities found - not confident, default to 1

    # No matches or ambiguous - default to 1
    return 1


def rebuild_structured_rows(lines: Iterable[str]) -> List[Dict[str, Any]]:
    """Derive structured HOLE TABLE style rows from free-form text fragments."""

    cache: Dict[str, Dict[str, Any]] = {}
    qty_map = _collect_qty_hints(lines)

    for line in lines:
        parsed = parse_hole_row(line)
        ref = parsed.get("ref")
        if not ref:
            continue
        ref = str(ref)
        existing = cache.get(ref)
        if existing is None:
            parsed["qty"] = qty_map.get(ref)
            parsed["raw_fragments"] = [parsed.get("raw")]
            cache[ref] = parsed
            continue

        existing_raw = existing.setdefault("raw_fragments", [])
        existing_raw.append(parsed.get("raw"))

        for key in ("diam", "tap_thread", "depth", "side"):
            value = parsed.get(key)
            if value and existing.get(key) in {None, ""}:
                existing[key] = value

        new_ops = [op for op in parsed.get("ops", []) if op]
        if new_ops:
            old_ops = existing.setdefault("ops", [])
            for op in new_ops:
                if op not in old_ops:
                    old_ops.append(op)
            if not existing.get("op"):
                existing["op"] = old_ops[0]

        if existing.get("qty") in {None, 0}:
            existing["qty"] = qty_map.get(ref)

    rows = list(cache.values())
    for row in rows:
        if "raw_fragments" not in row:
            row["raw_fragments"] = [row.get("raw")]
        if row.get("qty") in {None, 0}:
            ref = str(row.get("ref") or "")
            if ref:
                row["qty"] = qty_map.get(ref)
    return rows

