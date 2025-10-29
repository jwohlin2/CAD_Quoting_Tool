# cad_quoter/geo_dump.py
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cad_quoter import geo_extractor

# ---------- HOLE TABLE helpers ----------
_UHEX_RE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")
def _decode_uplus(s: str) -> str:
    """Turn \\U+2205 into Ø etc., keep original if malformed."""
    return _UHEX_RE.sub(lambda m: chr(int(m.group(1), 16)), s or "")

def _diameter_aliases(token: str) -> List[str]:
    """Given 'Ø1.7500', build aliases found in body text, including trailing-Ø fraction forms like '17/32∅'."""
    out = [token]
    if token.startswith(("Ø", "∅")):
        val_str = token[1:]
        try:
            val = float(val_str)
            frac = Fraction(val).limit_denominator(64)
            if abs(float(frac) - val) < 1e-4:
                n, d = frac.numerator, frac.denominator
                # Ø leading
                out += [f"Ø{n}/{d}", f"∅{n}/{d}"]
                # Ø trailing
                out += [f"{n}/{d}Ø", f"{n}/{d}∅"]
            comp = f"{val:.3f}".rstrip("0").rstrip(".")
            out += [f"Ø{comp}", f"∅{comp}", f"(Ø{comp})", f"(∅{comp})"]
        except Exception:
            pass
    return out

# Parse simple thread tokens
_THREAD_RE = re.compile(r'(?:(#\d+)|([A-Z])|(\d+/\d+)|(\d+(?:\.\d+)?))\s*[-–]\s*(\d+)', re.I)

# Common number drill majors + tap-drill approximations (inches)
# For routing #10-32 reliably to ~0.1590
_NUMBER_TAP_DRILL = {
    "#4-40": 0.0890, "#6-32": 0.1065, "#8-32": 0.1360, "#10-24": 0.1495, "#10-32": 0.1590,
    "#12-24": 0.1770, "#12-28": 0.1890,
}

# Major diameters for fractional and number sizes (minimal set; we only need when text lacks Ø)
_FRACTION_TO_IN = lambda s: (lambda n, d: float(n) / float(d))(*s.split("/"))
_LETTER_DRILL_DEC = {  # letter drills (subset used in your file)
    "A": .234, "B": .238, "C": .242, "D": .246, "E": .250, "F": .257, "G": .261,
    "H": .266, "I": .272, "J": .277, "K": .281, "L": .290, "M": .295, "N": .302,
    "Q": .332,
}


def _parse_thread_token(s: str) -> Optional[Tuple[float, int]]:
    """
    Returns (major_diam_in, tpi) if we can infer it from tokens like:
      5/8-11, 0.625-11, #10-32, I-24, Q-?? etc.
    """

    m = _THREAD_RE.search(s)
    if not m:
        return None
    num, letter, frac, decimal, tpi = m.groups()
    tpi = int(tpi)
    if num:
        # e.g., #10-32
        key = f"{num.upper()}-{tpi}"
        if key in _NUMBER_TAP_DRILL:
            # return "tap drill" as the 'major' stand-in (we'll snap to nearest)
            return (_NUMBER_TAP_DRILL[key], tpi)
        # unknown number size: approximate major from ANSI table? skip -> None
        return None
    if letter:
        # e.g., I-24
        L = letter.upper()
        major = _LETTER_DRILL_DEC.get(L)
        if major:
            return (major, tpi)
        return None
    if frac:
        return (_FRACTION_TO_IN(frac), tpi)
    if decimal:
        return (float(decimal), tpi)
    return None


def _tap_drill_from(major_in: float, tpi: int) -> float:
    """Simple UNC/UNF approximation: tap drill ≈ major − (1 / TPI)."""

    return max(0.0, major_in - 1.0 / float(tpi))


def _snap_to_nearest_index(value_in: float, diam_list: List[str]) -> int:
    """Return index of the REF_DIAM closest to the supplied numeric value."""

    best, err = 0, float("inf")
    for i, tok in enumerate(diam_list):
        try:
            v = float(tok[1:]) if tok.startswith(("Ø", "∅")) else float(tok)
        except Exception:
            continue
        delta = abs(v - value_in)
        if delta < err:
            best, err = i, delta
    return best


def _snap_to_nearest(value: float, diam_list: List[str]) -> int:
    """Backward-compatible wrapper that returns the nearest REF_DIAM index."""

    return _snap_to_nearest_index(value, diam_list)

def _find_hole_table_chunks(rows: List[dict]):
    """Return (header_chunks, body_chunks) from text rows when a HOLE TABLE is present."""
    # rows are dicts with at least: layout, layer, etype, text, ...
    # Work over PROXYTEXT/MTEXT/TEXT so we survive variant exports
    starts = [i for i, r in enumerate(rows)
              if r.get("etype") in ("PROXYTEXT", "MTEXT", "TEXT")
              and "HOLE TABLE" in (r.get("text", "").upper())]
    if not starts:
        return [], []
    i = starts[0]
    header_chunks: List[str] = []
    body_chunks: List[str] = []
    j = i
    saw_desc = False
    # Collect header lines until we see DESCRIPTION
    while j < len(rows) and rows[j].get("etype") in ("PROXYTEXT", "MTEXT", "TEXT"):
        header_chunks.append(rows[j].get("text", ""))
        if "DESCRIPTION" in (rows[j].get("text", "").upper()):
            saw_desc = True
            j += 1
            break
        j += 1
    if not saw_desc:
        # Be tolerant if CAD split a bit further
        k = j
        while k < min(j + 5, len(rows)) and rows[k].get("etype") in ("PROXYTEXT", "MTEXT", "TEXT"):
            header_chunks.append(rows[k].get("text", ""))
            if "DESCRIPTION" in (rows[k].get("text", "").upper()):
                j = k + 1
                break
            k += 1
    # Body: remaining contiguous HOLE TABLE text
    while j < len(rows) and rows[j].get("etype") in ("PROXYTEXT", "MTEXT", "TEXT"):
        body_chunks.append(rows[j].get("text", ""))
        j += 1
    return header_chunks, body_chunks


def _parse_header(header_chunks: List[str]):
    """
    Return (hole_letters, diam_tokens, qty_list).
    Handles: 'HOLE TABLE HOLE A B ... REF Ø Ø1.7500 ... QTY 4 2 ... DESCRIPTION'
    - Picks the LAST 'HOLE' before REF (not the one in 'HOLE TABLE')
    - Drops stray standalone Ø right after REF
    """
    header_text = re.sub(r"\s+", " ", " ".join(header_chunks)).strip().replace(",", " ")
    toks = header_text.split()

    def all_idx(name: str) -> List[int]:
        u = name.upper()
        return [i for i, t in enumerate(toks) if t.upper() == u]

    def find_one(name: str) -> int:
        u = name.upper()
        return next((i for i, t in enumerate(toks) if t.upper() == u), -1)

    idx_holes = all_idx("HOLE")
    i_ref = find_one("REF")
    i_qty = find_one("QTY")
    i_desc = find_one("DESCRIPTION")
    if i_desc == -1:
        i_desc = find_one("DESC")
    if i_ref == -1:
        # some drawings collapse REF header to a diameter symbol
        i_ref = find_one("Ø")

    # choose the HOLE that is actually the column header:
    # the last 'HOLE' occurring BEFORE REF
    i_hole = -1
    for i in idx_holes:
        if i < i_ref:
            i_hole = i
    if min(i_hole, i_ref, i_qty, i_desc) == -1:
        raise ValueError(f"Unexpected HOLE TABLE header: {header_text}")

    # slice ranges
    hole_letters = toks[i_hole + 1 : i_ref]
    diam_tokens = toks[i_ref + 1 : i_qty]
    qty_tokens = toks[i_qty + 1 : i_desc]

    # drop stray symbols that aren't real diameters
    diam_tokens = [d for d in diam_tokens if d not in ("Ø", "∅")]

    # parse quantities
    qty_list = []
    for q in qty_tokens:
        try:
            qty_list.append(int(q))
        except Exception:
            pass

    n = min(len(hole_letters), len(diam_tokens), len(qty_list))
    return hole_letters[:n], diam_tokens[:n], qty_list[:n]


def _redistribute_cross_hits(descs: List[str], diam_list: List[str]) -> List[str]:
    """
    Reassign sub-chunks inside each segment that actually belong to other hole diameters.
    Keeps the leading chunk with the original hole; moves each internal chunk to its diameter's hole.
    """
    # 1) Build alias -> hole_idx map (include decimals, fractions, parens, trailing-Ø)
    alias_to_idx: Dict[str, int] = {}
    for idx, tok in enumerate(diam_list):
        for a in _diameter_aliases(tok):
            if a:
                alias_to_idx[a] = idx
        # numeric-only fallbacks for decimals like .272 / 0.272 (and parens)
        if tok.startswith(("Ø", "∅")):
            s = tok[1:]
            try:
                f = float(s)
                comp = f"{f:.3f}".rstrip("0").rstrip(".")
            except Exception:
                comp = s
            for a in (comp, f"0{comp}", f"({comp})", f"(0{comp})"):
                alias_to_idx[a] = idx

    if not alias_to_idx:
        return descs

    # 2) Build a single regex alternation (longest-first to avoid partial overlaps)
    aliases_sorted = sorted(alias_to_idx.keys(), key=len, reverse=True)
    alt_union = "|".join(map(re.escape, aliases_sorted))
    re_markers = re.compile(alt_union)

    def strip_marker_prefix(s: str) -> str:
        # fractions first (Ø-leading and Ø-trailing), then decimals, then quoted REF like "Q"(Ø.332)
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9]+/[0-9]+\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[0-9]+/[0-9]+[Ø∅]\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9.]+\)?\s*',        '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*\((?:[Ø∅]\s*[0-9.]+)\)\s*', '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    new_descs = [""] * len(descs)

    for i, seg in enumerate(descs):
        s = (seg or "").strip()
        if not s:
            continue

        # find all internal markers with their owners
        hits = []
        for m in re_markers.finditer(s):
            alias = m.group(0)
            tgt = alias_to_idx.get(alias)
            if tgt is None or tgt == i:
                continue
            hits.append((m.start(), m.end(), tgt, alias))

        if not hits:
            new_descs[i] = s
            continue

        # slice: self text between foreign markers; move foreign chunks
        hits.sort(key=lambda t: t[0])
        cursor = 0
        self_parts = []
        for k, (a, b, tgt, alias) in enumerate(hits):
            if a > cursor:
                self_parts.append(s[cursor:a])            # keep for self
            next_a = hits[k+1][0] if k+1 < len(hits) else len(s)
            chunk_raw = s[a:next_a]
            chunk = strip_marker_prefix(chunk_raw)
            tgt_idx = tgt
            m_jg = re.search(r'[Ø∅]\s*([0-9.]+)\s+JIG\s*GRIND', chunk_raw, flags=re.I)
            if m_jg:
                try:
                    val = float(m_jg.group(1))
                    tgt_idx = _snap_to_nearest_index(val, diam_list)
                except Exception:
                    pass
            if chunk:
                new_descs[tgt_idx] = (new_descs[tgt_idx] + " " + chunk).strip() if new_descs[tgt_idx] else chunk
            cursor = next_a
        if cursor < len(s):
            self_parts.append(s[cursor:])                 # tail back to self

        self_text = strip_marker_prefix(" ".join(self_parts))
        new_descs[i] = self_text

    # merge any untouched originals
    for i in range(len(descs)):
        if not new_descs[i] and descs[i]:
            new_descs[i] = descs[i]

    # tidy trailing semicolons/space
    return [re.sub(r"\s*;\s*$", "", x or "").strip() for x in new_descs]


def _route_tap_only_chunks(descs: List[str], diam_list: List[str]) -> List[str]:
    """
    Find TAP chunks that lack any Ø marker and route them to the nearest REF_DIAM
    based on estimated tap-drill size. Handles examples:
      - '5/8-11 TAP THRU (FROM BACK)'  -> ~0.625 - 1/11 ≈ 0.534  -> C (Ø.5313)
      - '#10-32 TAP X .62 DEEP ...'    -> 0.1590                 -> M (Ø.1590)
    Also honors letter drills like 'I-24' by using their decimal drill size.
    """

    moved = [""] * len(descs)
    keep = [""] * len(descs)

    for i, s in enumerate(descs):
        txt = (s or "").strip()
        if not txt:
            keep[i] = ""
            continue

        # If segment already contains a hole diameter alias, leave it (handled earlier)
        if any(sym in txt for sym in ("Ø", "∅")):
            keep[i] = txt
            continue

        # Split into clauses so we can route partial pieces
        clauses = re.split(r'(?<=;)\s+|(?<=\))\s+', txt)  # split on semicolon boundaries or right-paren
        keep_parts: List[str] = []
        for cl in clauses:
            if "TAP" not in cl.upper():
                keep_parts.append(cl)
                continue
            # Try to read thread token and estimate tap drill
            parsed = _parse_thread_token(cl)
            if not parsed:
                keep_parts.append(cl)
                continue
            major_in, tpi = parsed
            tap_in = major_in if "#" in cl or re.search(r'#[0-9]+', cl) else _tap_drill_from(major_in, tpi)
            tgt_idx = _snap_to_nearest(tap_in, diam_list)
            # Move this clause to its target hole
            moved[tgt_idx] = (moved[tgt_idx] + " " + cl).strip() if moved[tgt_idx] else cl

        keep[i] = " ".join(p for p in (p.strip() for p in keep_parts) if p)

    # Merge moved text into destinations
    out: List[str] = []
    for i in range(len(descs)):
        pieces = []
        if keep[i]:
            pieces.append(keep[i])
        if moved[i]:
            pieces.append(moved[i])
        out.append(" ".join(pieces).strip())
    return out


# ---------- Operation explosion helpers ----------

# Regexes we’ll reuse
_RE_DIAM_LEAD   = re.compile(r'^[\(\s]*[Ø∅]\s*([0-9]+/[0-9]+|[0-9.]+)\)?\s*', re.I)
_RE_DIAM_TRAIL  = re.compile(r'^[\(\s]*([0-9]+/[0-9]+)\s*[Ø∅]\)?\s*', re.I)  # e.g., 13/32∅
_RE_DIAM_ANY    = re.compile(r'[Ø∅]\s*([0-9]+/[0-9]+|[0-9.]+)', re.I)
_RE_PAREN_JG    = re.compile(r'\(\s*[Ø∅]\s*([0-9]+/[0-9]+|[0-9.]+)\s+JIG\s+GRIND\s*\)', re.I)
_RE_TAP         = re.compile(r'\bTAP\b', re.I)
_RE_THRU        = re.compile(r'\bTHRU\b', re.I)
_RE_CBORE       = re.compile(r"C['’]BORE", re.I)
_RE_CDRILL      = re.compile(r"C['’]DRILL", re.I)
_RE_DEPTH_PHRASE= re.compile(r'[Xx]\s*([0-9.]+)\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?', re.I)
_RE_SIDE_PAIR   = re.compile(r'\bFROM\s+FRONT\s*&\s*BACK\b', re.I)
_RE_SIDE        = re.compile(r'\bFROM\s+(FRONT|BACK)\b', re.I)
_RE_QREF        = re.compile(r'^\s*"{1,2}[A-Z]"{1,2}\s*', re.I)   # leading "Q", "I", etc.


def _fmt_diam(token: str) -> str:
    """Normalize a numeric token to Ø... (keep fractions as-is)."""
    tok = token.strip()
    if '/' in tok:
        return f"Ø{tok}"
    # decimals: keep as entered, drop trailing zeros and dot for secondary ops
    try:
        f = float(tok)
        s = f"{f:.4f}".rstrip('0').rstrip('.')
        return f"Ø{s}"
    except Exception:
        return f"Ø{tok}"


def _fmt_primary_ref_diam(token: str) -> str:
    """Normalize primary REF_DIAM values to fixed four decimals when numeric."""

    tok = (token or "").strip()
    if not tok:
        return tok

    sym = ""
    rest = tok
    if tok[0] in ("Ø", "∅"):
        sym = tok[0]
        rest = tok[1:].strip()

    if '/' in rest:
        return f"{sym}{rest}" if sym else tok

    try:
        val = float(rest)
    except Exception:
        return tok

    formatted = f"{val:.4f}"
    if rest.startswith('.') and formatted.startswith('0'):
        formatted = formatted[1:]

    return f"{sym}{formatted}" if sym else formatted


def _extract_leading_diam(clause: str) -> Tuple[str, str]:
    """
    Return (diam_override, remainder). Supports:
      - Ø.623 ...   (leading symbol)
      - 13/32Ø ...  (trailing symbol on a fraction)
    """

    s = clause.lstrip()
    m = _RE_DIAM_LEAD.match(s)
    if m:
        return _fmt_diam(m.group(1)), s[m.end():].lstrip()
    m = _RE_DIAM_TRAIL.match(s)
    if m:
        return _fmt_diam(m.group(1)), s[m.end():].lstrip()
    return "", clause.strip()


def _explode_front_back(desc: str) -> List[str]:
    """
    Split '... FROM FRONT & BACK ...' into two descs with explicit sides.
    If only one side present, return as-is.
    """

    if _RE_SIDE_PAIR.search(desc):
        base = _RE_SIDE_PAIR.sub("", desc).strip(",; ").strip()
        left  = _RE_SIDE.sub("FROM FRONT", base)
        right = _RE_SIDE.sub("FROM BACK",  base)
        # ensure side text present
        if "FROM FRONT" not in left:  left  = (left + " FROM FRONT").strip()
        if "FROM BACK"  not in right: right = (right + " FROM BACK").strip()
        return [left, right]
    return [desc]


def _clean_clause_text(s: str) -> str:
    # strip leading quoted refs and tidy spaces/semicolons
    s = _RE_QREF.sub("", s).strip()
    s = re.sub(r"\bAS\s+SHOWN\b", "", s, flags=re.I)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*;\s*$', '', s)
    return s.strip(", ").strip()


def _parse_clause_to_ops(
    hole_idx: int,
    base_diam: str,
    qty: int,
    text: str,
    diam_list: List[str],
) -> List[Tuple[int, str, int, str]]:
    """
    Turn one clause into 1+ (hole_idx, diam, qty, 'DESCRIPTION/DEPTH') ops.
    - pulls out parenthetical (Ø.3750 JIG GRIND) as its own op (snapped to nearest hole)
    - honors leading/trailing Ø diam overrides (reassigning hole by nearest REF_DIAM)
    - splits FRONT & BACK into two ops
    - keeps TAP+THRU glued together when both appear
    """

    ops: List[Tuple[int, str, int, str]] = []

    s = text.strip()
    if not s:
        return ops

    # Parenthetical Jig Grind like (Ø.3750 JIG GRIND)
    for pm in list(_RE_PAREN_JG.finditer(s)):
        val = pm.group(1)
        dstr = _fmt_diam(val)
        try:
            tgt = _snap_to_nearest_index(float(val.replace('/', '0')), diam_list)
        except Exception:
            tgt = hole_idx
        ops.append((tgt, dstr, qty, "JIG GRIND"))
        s = s.replace(pm.group(0), " ")  # remove from main text

    # Diameter override at start (or fraction with trailing Ø)
    diam_override, rest = _extract_leading_diam(_clean_clause_text(s))
    if diam_override:
        try:
            v = diam_override[1:]
            tgt = _snap_to_nearest_index(float(v), diam_list)
        except Exception:
            tgt = hole_idx
        hole_for_clause = tgt
        diam_for_clause = diam_override
    else:
        hole_for_clause = hole_idx
        diam_for_clause = base_diam

    rest = _clean_clause_text(rest)

    if not rest:
        # Bare diameter line => treat as THRU
        ops.append((hole_for_clause, diam_for_clause, qty, "THRU"))
        return ops

    # Split explicit FRONT & BACK pair into two
    parts = _explode_front_back(rest)

    for part in parts:
        desc = _clean_clause_text(part)
        if not desc:
            continue

        # --- GLUE: keep TAP+THRU combined if both appear in same clause ---
        if _RE_TAP.search(desc) and _RE_THRU.search(desc):
            ops.append((hole_for_clause, diam_for_clause, qty, desc))
            continue

        if any(rx.search(desc) for rx in (_RE_CBORE, _RE_CDRILL, _RE_TAP, _RE_THRU)):
            ops.append((hole_for_clause, diam_for_clause, qty, desc))
        else:
            # generic fallback
            ops.append((hole_for_clause, diam_for_clause, qty, desc))

    return ops


def _explode_description_into_ops(
    hole_idx: int,
    hole: str,
    base_diam: str,
    qty: int,
    description: str,
    diam_list: List[str],
    hole_letters: List[str],
) -> List[Dict[str, str]]:
    """
    Split a row's DESCRIPTION into multiple atomic ops by ';' boundaries,
    parse each to (diam, qty, desc), and return ready-to-write dicts.
    """

    # split on semicolons, but keep content inside parentheses intact
    clauses = [c.strip() for c in re.split(r';', description) if c.strip()]
    out: List[Dict[str,str]] = []
    for cl in clauses:
        for tgt_idx, diam, q, desc in _parse_clause_to_ops(hole_idx, base_diam, qty, cl, diam_list):
            hole_tok = hole_letters[tgt_idx] if 0 <= tgt_idx < len(hole_letters) else hole
            out.append({
                "HOLE": hole_tok,
                "REF_DIAM": diam,
                "QTY": str(q),
                "DESCRIPTION/DEPTH": desc,
            })
    return out


def _split_descriptions(body_chunks: List[str], diam_list: List[str]) -> List[str]:
    """
    ORDER-AGNOSTIC splitter + redistribution of cross-hits.
    """
    blob = re.sub(r"\s+", " ", " ".join(body_chunks)).strip()
    # Cut off coordinate table if present
    m = re.search(r"\bLIST OF COORDINATES\b", blob, flags=re.I)
    if m:
        blob = blob[:m.start()].rstrip()

    # 1) find positions for every diameter alias anywhere in the blob
    positions: List[Tuple[int, int, str]] = []  # (pos, hole_idx, alias)
    for idx, token in enumerate(diam_list):
        alts = _diameter_aliases(token)
        # last-resort numeric payloads
        if token.startswith(("Ø", "∅")):
            s = token[1:]
            try:
                f = float(s)
                comp = f"{f:.3f}".rstrip("0").rstrip(".")
            except Exception:
                comp = s
            alts += [comp, f"0{comp}", f"({comp})", f"(0{comp})"]

        best_pos, best_alt = None, ""
        for a in alts:
            p = blob.find(a)
            if p != -1 and (best_pos is None or p < best_pos):
                best_pos, best_alt = p, a
        if best_pos is None:
            best_pos, best_alt = len(blob), ""
        positions.append((best_pos, idx, best_alt))

    # 2) slice by sorted positions
    positions.sort(key=lambda t: t[0])
    cuts = [p for (p, _, _) in positions]
    segments_in_order = []
    for i, start in enumerate(cuts):
        end = cuts[i+1] if i+1 < len(cuts) else len(blob)
        segments_in_order.append(blob[start:end])

    # 3) basic strip of leading marker
    def strip_leading_marker(s: str) -> str:
        s = s.strip()
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9]+/[0-9]+\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[0-9]+/[0-9]+[Ø∅]\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9.]+\)?\s*',        '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*\((?:[Ø∅]\s*[0-9.]+)\)\s*', '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*', '', s)
        return s.strip()

    # 4) map to the original hole order
    descs = [""] * len(diam_list)
    for (pos, hole_idx, _), seg in zip(positions, segments_in_order):
        if pos >= len(blob):
            continue
        descs[hole_idx] = strip_leading_marker(seg)

    # 5) redistribute any cross-hits (multi-hole bundles) to the right holes
    descs = _redistribute_cross_hits(descs, diam_list)

    # 6) NEW: route TAP-only clauses (no Ø markers) to the nearest hole by tap-drill estimate
    descs = _route_tap_only_chunks(descs, diam_list)

    return descs

DEFAULT_SAMPLE_PATH = REPO_ROOT / "Cad Files" / "301_redacted.dxf"


def _parse_csv_patterns(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Dump ALL text (incl. inside blocks) from DXF/DWG → CSV + JSONL."
    )
    ap.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_SAMPLE_PATH),
        help=f"Input .dxf or .dwg (default: {DEFAULT_SAMPLE_PATH})",
    )
    ap.add_argument("--outdir", default="debug", help="Output directory (default: debug)")
    ap.add_argument(
        "--layouts",
        help="Comma-separated layout names to scan (default: all, e.g., 'CHART,SHEET (B),Model')",
    )
    ap.add_argument("--include-layers", help="Regex allowlist CSV (default: none → all)")
    ap.add_argument("--exclude-layers", help="Regex blocklist CSV (default: none)")
    ap.add_argument("--min-height", type=float, default=0.0, help="Min text height filter")
    ap.add_argument("--block-depth", type=int, default=3, help="Max recursive INSERT depth")
    ap.add_argument("--csv-name", default="dxf_text_dump.csv", help="CSV filename")
    ap.add_argument("--jsonl-name", default="dxf_text_dump.jsonl", help="JSONL filename")
    args = ap.parse_args()

    in_path = Path(args.path).expanduser().resolve()
    if not in_path.exists():
        ap.error(f"Input path not found: {in_path}")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / args.csv_name
    jsonl_path = outdir / args.jsonl_name

    doc = geo_extractor.open_doc(in_path)

    layouts = None
    if args.layouts and args.layouts.strip():
        layouts = [p.strip() for p in args.layouts.split(",") if p.strip()]

    include_layers = _parse_csv_patterns(args.include_layers)
    exclude_layers = _parse_csv_patterns(args.exclude_layers)

    rows = geo_extractor.collect_all_text(
        doc,
        layouts=layouts,
        include_layers=include_layers,
        exclude_layers=exclude_layers,
        min_height=float(args.min_height or 0.0),
        max_block_depth=int(args.block_depth or 0),
    )

    # Decode \U+#### across all text rows (so everything renders with real symbols)
    for r in rows:
        if "text" in r and isinstance(r["text"], str):
            r["text"] = _decode_uplus(r["text"])

    # Write CSV
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "layout",
                "layer",
                "etype",
                "text",
                "x",
                "y",
                "height",
                "rotation",
                "in_block",
                "depth",
                "block_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.get("layout", ""),
                    r.get("layer", ""),
                    r.get("etype", ""),
                    r.get("text", ""),
                    float(r.get("x", 0.0) or 0.0),
                    float(r.get("y", 0.0) or 0.0),
                    float(r.get("height", 0.0) or 0.0),
                    float(r.get("rotation", 0.0) or 0.0),
                    int(bool(r.get("in_block", False))),
                    int(r.get("depth", 0) or 0),
                    "/".join(r.get("block_path", []) or []),
                ]
            )

    # Write JSONL
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Emit structured HOLE TABLE, if present ----
    try:
        header_chunks, body_chunks = _find_hole_table_chunks(rows)
        if header_chunks:
            hole_letters, diam_tokens, qtys = _parse_header(header_chunks)
            diam_tokens = [_fmt_primary_ref_diam(d) for d in diam_tokens]
            descriptions = _split_descriptions(body_chunks, diam_tokens)
            out_rows = []
            if len(hole_letters) != len(diam_tokens) or len(hole_letters) != len(qtys):
                print(
                    f"[HOLE-TABLE][warn] counts mismatch: holes={len(hole_letters)} "
                    f"diam={len(diam_tokens)} qty={len(qtys)}"
                )
            if any(not d.startswith(("Ø", "∅")) for d in diam_tokens):
                print("[HOLE-TABLE][warn] non-diameter token detected in REF_DIAM:", diam_tokens)
            for i, hole in enumerate(hole_letters):
                out_rows.append({
                    "HOLE": hole,
                    "REF_DIAM": diam_tokens[i],
                    "QTY": qtys[i],
                    "DESCRIPTION": (descriptions[i] if i < len(descriptions) else "").strip(),
                })
            # ---- Emit exploded ops file (ideal for machine-time calc) ----
            ops_rows: List[Dict[str,str]] = []
            for i, hole in enumerate(hole_letters):
                base_d = diam_tokens[i]
                q      = qtys[i]
                desc   = (descriptions[i] if i < len(descriptions) else "").strip()
                if not desc:
                    continue
                ops_rows += _explode_description_into_ops(i, hole, base_d, q, desc, diam_tokens, hole_letters)

            ops_csv = csv_path.parent / "hole_table_ops.csv"
            with ops_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["HOLE","REF_DIAM","QTY","DESCRIPTION/DEPTH"])
                w.writeheader()
                w.writerows(ops_rows)
            print(f"[HOLE-TABLE][ops] rows={len(ops_rows)} csv={ops_csv}")
            hole_csv = csv_path.parent / "hole_table_structured.csv"
            with hole_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["HOLE", "REF_DIAM", "QTY", "DESCRIPTION"])
                w.writeheader()
                w.writerows(out_rows)
            print(f"[HOLE-TABLE] rows={len(out_rows)} csv={hole_csv}")
        else:
            print("[HOLE-TABLE] none detected")
    except Exception as e:
        print(f"[HOLE-TABLE] parse failed: {e}")

    print(
        "[TEXT-DUMP] layouts={layouts} include_layers={inc} exclude_layers={exc} "
        "rows={rows} csv={csv} jsonl={jsonl}".format(
            layouts=",".join(layouts or []) if layouts else "<all>",
            inc=include_layers or "-",
            exc=exclude_layers or "-",
            rows=len(rows),
            csv=csv_path,
            jsonl=jsonl_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
