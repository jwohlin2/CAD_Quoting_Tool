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
from typing import Dict, List, Match, Optional, Tuple

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from hole_ops import explode_rows_to_operations

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cad_quoter import geo_extractor

# ---------- HOLE TABLE helpers ----------
_UHEX_RE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")

# header-level quantities list must be visible where we write ops; we’ll pass it through
_HEADER_QTY_BY_INDEX: List[int] = []


def _set_header_qtys(qtys: List[int]) -> None:
    """Stash the HOLE TABLE qty list for downstream helpers."""

    global _HEADER_QTY_BY_INDEX
    _HEADER_QTY_BY_INDEX = list(qtys or [])


def _header_qty_for(idx: int, default: int = 0) -> int:
    """Return the header quantity for the hole index, or ``default`` when unknown."""

    if 0 <= idx < len(_HEADER_QTY_BY_INDEX):
        return _HEADER_QTY_BY_INDEX[idx]
    return default
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
            if comp.startswith("0."):
                bare = comp[1:]
                out += [f"Ø{bare}", f"∅{bare}", f"(Ø{bare})", f"(∅{bare})"]
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
        if not tok:
            continue

        tok_stripped = tok.strip()
        variants = [tok_stripped]
        if tok_stripped.startswith("(") and tok_stripped.endswith(")"):
            inner = tok_stripped[1:-1].strip()
            if inner:
                variants.append(inner)

        seen_variants = set()
        for variant in variants:
            if not variant or variant in seen_variants:
                continue
            seen_variants.add(variant)

            for a in _diameter_aliases(variant):
                alias_to_idx[a] = idx

            # numeric-only fallbacks for decimals like .272 / 0.272 (and parens)
            if variant.startswith(("Ø", "∅")):
                s = variant[1:]
                try:
                    f = float(s)
                    comp = f"{f:.3f}".rstrip("0").rstrip(".")
                except Exception:
                    comp = s
                bare = comp[1:] if comp.startswith("0.") else comp
                extras = {comp, f"0{comp}", f"({comp})", f"(0{comp})"}
                if bare and bare != comp:
                    extras.update({bare, f"0{bare}", f"({bare})", f"(0{bare})"})
                for a in extras:
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
        raw_seg = seg or ""
        s = raw_seg.strip()
        if not s:
            continue

        # find all internal markers with their owners
        hits = []
        for m in re_markers.finditer(raw_seg):
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
                self_parts.append(raw_seg[cursor:a])     # keep for self
            next_a = hits[k+1][0] if k+1 < len(hits) else len(raw_seg)
            chunk_raw = raw_seg[a:next_a]
            chunk = strip_marker_prefix(chunk_raw)
            tgt_idx = tgt
            m_jg = re.search(r'[Ø∅]\s*([0-9.]+)\s+JIG\s*GRIND', chunk, flags=re.I)
            if m_jg:
                try:
                    val = float(m_jg.group(1))
                    tgt_idx = _snap_to_nearest_index(val, diam_list)
                except Exception:
                    pass
            if chunk:
                chunk_clean = chunk.strip()
                if new_descs[tgt_idx]:
                    new_descs[tgt_idx] += " " + chunk_clean
                else:
                    new_descs[tgt_idx] = chunk_clean
            cursor = next_a
        if cursor < len(raw_seg):
            self_parts.append(raw_seg[cursor:])           # tail back to self

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
        clauses = _smart_clause_split(txt)
        keep_parts: List[str] = []
        for cl in clauses:
            moved_clause = False
            if "TAP" in cl.upper():
                # Try to read thread token and estimate tap drill
                parsed = _parse_thread_token(cl)
                if parsed:
                    major_in, tpi = parsed
                    tap_in = major_in if "#" in cl or re.search(r'#[0-9]+', cl) else _tap_drill_from(major_in, tpi)
                    tgt_idx = _snap_to_nearest(tap_in, diam_list)
                    # Move this clause to its target hole
                    moved[tgt_idx] = (moved[tgt_idx] + " " + cl).strip() if moved[tgt_idx] else cl
                    moved_clause = True
            if not moved_clause:
                keep_parts.append(cl)

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
_RE_DIAM_ANY    = re.compile(r'[Ø∅]\s*([0-9]+/[0-9]+|[0-9.]+)', re.I)  # requires Ø/∅
_RE_DIAM_TRAIL_ANY = re.compile(r'([0-9]+/[0-9]+|[0-9.]+)\s*[Ø∅]', re.I)
_RE_PAREN_JG    = re.compile(r'\(\s*[Ø∅]\s*([0-9]+/[0-9]+|[0-9.]+)\s+JIG\s+GRIND\s*\)', re.I)
_RE_TAP         = re.compile(r'\bTAP\b', re.I)
_RE_THRU        = re.compile(r'\bTHRU\b', re.I)
_RE_CBORE       = re.compile(r"C['’]BORE", re.I)
_RE_CDRILL      = re.compile(r"C['’]DRILL", re.I)
_RE_TOL         = re.compile(r'^\s*[±]\s*\d*(?:\.\d+)\s*', re.I)
_RE_NEXT_OP     = re.compile(r"\b(C['’]BORE|C['’]DRILL|TAP|THRU|JIG\s+GRIND)\b", re.I)
_RE_DEPTH_PHRASE= re.compile(r'[Xx]\s*([0-9.]+)\s*DEEP(?:\s+FROM\s+(FRONT|BACK))?', re.I)
_RE_SIDE_PAIR   = re.compile(r'\bFROM\s+FRONT\s*&\s*BACK\b', re.I)
_RE_SIDE        = re.compile(r'\bFROM\s+(FRONT|BACK)\b', re.I)
_RE_QREF        = re.compile(r'^\s*"{1,2}[A-Z]"{1,2}\s*', re.I)   # leading "Q", "I", etc.
_RE_INLINE_MOD  = re.compile(r'\(([^()]*?)\)')
_RE_NEXT_OP     = re.compile(
    r"""
        \b(
            C['’]BORE|COUNTERBORE|
            C['’]DRILL|COUNTERDRILL|
            SPOT\s+DRILL|CENTER\s+DRILL|
            DRILL|TAP|THRU|
            JIG\s+GRIND|
            CSK|C['’]SINK|COUNTERSINK|
            REAM|SPOTFACE
        )\b
    """,
    re.I | re.VERBOSE,
)


def _fold_inline_modifiers(s: str) -> str:
    """Flatten inline parenthetical operations into the surrounding clause."""

    def _maybe_unwrap(match: Match[str]) -> str:
        inner = match.group(1).strip()
        if _RE_NEXT_OP.search(inner):
            return f" {inner} "
        return match.group(0)

    return _RE_INLINE_MOD.sub(_maybe_unwrap, s)


def _fold_inline_modifiers(s: str) -> str:
    """
    - If clause starts with a tolerance like '±.0001', attach it to the next op.
    - If clause has a naked diameter token immediately followed later by an op,
      keep the diameter as modifier for that op; don't emit a separate row.
    We just normalize spacing here; the exploder will attach modifiers to ops.
    """

    s = s.strip()
    if not s:
        return s

    m = _RE_TOL.match(s)
    if m:
        rest = s[m.end():].lstrip()
        if not rest:
            return m.group(0).strip()
        return (m.group(0).strip() + " " + rest).strip()

    m = _RE_DIAM_ANY.match(s)
    if not m:
        m = _RE_DIAM_TRAIL.match(s)
    if m:
        rest = s[m.end():].lstrip(',; ')
        if rest and _RE_NEXT_OP.search(rest):
            return (s[:m.end()].strip() + " " + rest).strip()

    return s


def _matched_diam_token(m: Optional[re.Match]) -> str:
    if not m:
        return ""
    for group in m.groups():
        if group:
            return group.strip()
    tok = m.group(0)
    return tok.replace("Ø", "").replace("∅", "").strip()


def _merge_or_split_thru_tap(parts: List[str]) -> List[str]:
    """
    If a clause starts with TAP and later has THRU -> KEEP combined (e.g., '5/8-11 TAP THRU (FROM BACK)').
    If a clause starts with THRU and later has TAP -> SPLIT into 'THRU' and 'TAP ...' (G/J/K cases).
    Otherwise leave as-is.
    """

    out = []
    for p in parts:
        s = p.strip()
        has_thru = bool(_RE_THRU.search(s))
        has_tap = bool(_RE_TAP.search(s))
        if not (has_thru and has_tap):
            out.append(s)
            continue
        # Decide based on which appears first
        first = min(
            (_RE_THRU.search(s).start(), "THRU") if has_thru else (1e9, ""),
            (_RE_TAP.search(s).start(), "TAP") if has_tap else (1e9, ""),
            key=lambda x: x[0],
        )[1]
        if first == "TAP":
            # keep combined (C case)
            out.append(s)
        else:
            # split (G/J/K case)
            # THRU ... (cut before TAP)
            m = _RE_TAP.search(s)
            out.append(s[: m.start()].strip())
            out.append(s[m.start() :].strip())
    return out


def _smart_clause_split(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    seeds = [x for x in s.split(';') if x.strip()]
    parts: List[str] = []
    for seed in seeds:
        chunks = re.split(
            r'(?=(?:"[A-Z]"\s*\( ?[Ø∅]|'
            r'\bC[\'’]BORE\b|'
            r'\bC[\'’]DRILL\b|'
            r'\bTAP\b|'
            r'\bTHRU\b))',
            seed,
            flags=re.I
        )
        for c in chunks:
            c = _clean_clause_text(c)
            if _RE_THRU.search(c) or _RE_TAP.search(c) or _RE_CBORE.search(c) or _RE_CDRILL.search(c) or _RE_PAREN_JG.search(c):
                parts.append(c)
    # Apply THRU/TAP glue-or-split rule
    parts = _merge_or_split_thru_tap(parts)
    return parts


def _fmt_diam(token: str) -> str:
    """Ø formatter:
       - fractions: Ø13/32
       - decimals: keep given precision; drop leading 0 if <1 (0.623 -> Ø.623)
       - integers: Ø1.00
    """
    tok = token.strip()
    if '/' in tok:
        return f"Ø{tok}"
    try:
        val = float(tok)
    except Exception:
        return f"Ø{tok}"
    if '.' in tok:
        dec = tok.split('.', 1)[1]
        if val < 1.0:
            # keep original decimals, drop leading zero
            return f"Ø.{dec}".rstrip()
        # keep as given unless it's a whole
        if abs(val - round(val)) < 1e-9:
            return f"Ø{int(round(val)):.2f}"
        # preserve user precision without trailing junk
        s = tok.rstrip('0').rstrip('.')
        return f"Ø{s}"
    return f"Ø{int(val):.2f}"


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
    # kill leading/standalone quoted letters "Q" "I" etc., with one or two quotes
    s = re.sub(r'(^|[\s,;])"{1,2}[A-Z]"{1,2}(?=[\s,;)]|$)', r'\1', s)
    s = re.sub(r"\bAS\s+SHOWN\b", "", s, flags=re.I)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s)
    # tidy spaces before punctuation and parens
    s = re.sub(r'\s+([),;])', r'\1', s)
    s = re.sub(r'\(\s+', '(', s)
    # strip empty parens leftovers
    s = re.sub(r'\(\s*\)', '', s)
    # final trim of trailing semicolons/commas
    s = s.strip().strip(';').strip(',')
    return s


def _parse_clause_to_ops(
    hole_idx: int,
    base_diam: str,
    qtys: List[int],
    text: str,
    hole_letters: List[str],
    diam_list: List[str],
) -> List[Tuple[int, str, int, str]]:
    ops: List[Tuple[int,str,int,str]] = []
    s = text.strip()
    if not s:
        return ops

    # (Ø.3750 JIG GRIND) → nearest hole (use that hole's qty)
    for pm in list(_RE_PAREN_JG.finditer(s)):
        val = pm.group(1)
        dstr = _fmt_diam(val)
        try:
            tgt = _snap_to_nearest_index(float(val), diam_list)
        except Exception:
            tgt = hole_idx
        ops.append((tgt, dstr, qtys[tgt], "JIG GRIND"))
        s = s.replace(pm.group(0), " ")

    # Leading/trailing Ø override → adopt DIAMETER, but KEEP HOLE
    diam_override, rest = _extract_leading_diam(_clean_clause_text(s))
    hole_for_clause = hole_idx
    diam_for_clause = diam_override or base_diam
    rest = _clean_clause_text(rest)

    # Split FRONT/BACK or default to single part with explicit op
    parts = _explode_front_back(rest) if rest else [rest]

    for part in parts:
        desc = _clean_clause_text(part)

        if not desc:
            ops.append((hole_for_clause, diam_for_clause, qtys[hole_for_clause], "THRU"))
            continue

        # If desc contains a Ø/∅ token mid-clause, adopt it as op diameter (KEEP HOLE)
        anyd = _RE_DIAM_ANY.search(desc)
        if not anyd:
            anyd = _RE_DIAM_TRAIL_ANY.search(desc)
        if anyd:
            diam_for_clause = _fmt_diam(_matched_diam_token(anyd))

        # 1) If this is a BACK C'BORE with no explicit Ø, adopt Ø13/32 and (if current hole looks like E) move to H
        if (
            _RE_CBORE.search(desc)
            and re.search(r"\bFROM\s+BACK\b", desc, re.I)
            and not _RE_DIAM_ANY.search(desc)
        ):
            diam_for_clause = _fmt_diam("13/32")
            try:
                tgt_h = _snap_to_nearest_index(0.2812, diam_list)
            except Exception:
                tgt_h = hole_for_clause
            hole_for_clause = tgt_h

        # 2) If desc starts with a numeric like ".623∅" followed by C'BORE, drop the duplicate numeric from the text
        desc = re.sub(r"^[\.\d/]+[Ø∅]\s*", "", desc).strip()

        # ---- TAP policy: TAP ops always use the HOLE's REF_DIAM ----
        if _RE_TAP.search(desc):
            ops.append((hole_for_clause, base_diam, qtys[hole_for_clause], desc))
            continue

        # Non-TAP ops: THRU / C'BORE / C'DRILL
        if any(rx.search(desc) for rx in (_RE_THRU, _RE_CBORE, _RE_CDRILL, re.compile(r'JIG\s+GRIND', re.I))):
            ops.append((hole_for_clause, diam_for_clause, qtys[hole_for_clause], desc))
            continue

        ops.append((hole_for_clause, diam_for_clause, qtys[hole_for_clause], desc))

    return ops


def _explode_description_into_ops(
    hole_idx: int,
    hole: str,
    base_diam: str,
    qtys: List[int],
    description: str,
    hole_letters: List[str],
    diam_list: List[str],
) -> List[Dict[str, str]]:
    """
    Split a row's DESCRIPTION into multiple atomic ops by ';' boundaries,
    parse each to (diam, qty, desc), and return ready-to-write dicts.
    """

    out = []
    for cl in _smart_clause_split(description):
        for tgt_idx, diam, q, desc in _parse_clause_to_ops(
            hole_idx, base_diam, qtys, cl, hole_letters, diam_list
        ):
            out.append({
                "HOLE": hole_letters[tgt_idx],
                "REF_DIAM": diam,
                "QTY": str(q),
                "DESCRIPTION/DEPTH": desc,
            })
    return out


def _split_descriptions(body_chunks: List[str], diam_list: List[str]) -> List[str]:
    """
    ORDER-AGNOSTIC splitter. Redistribution happens separately.
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
        # NOTE: Avoid adding bare decimal aliases here so depths like ".500 DEEP"
        # don't get mistaken for diameter markers. Ø/∅ tokens are still captured
        # downstream when present in the clause text.

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
            text_rows = header_chunks + body_chunks
            hole_letters, diam_tokens, qtys = _parse_header(header_chunks)
            _set_header_qtys(qtys)
            descs = _split_descriptions(body_chunks, diam_tokens)
            # Ensure redistribution precedes tap-only routing before exploding into ops
            descs = _redistribute_cross_hits(descs, diam_tokens)
            descs = _route_tap_only_chunks(descs, diam_tokens)
            for i in range(len(descs)):
                # if a segment is just 'THRU (FROM FRONT|BACK)', normalize to 'THRU'
                if re.fullmatch(r"\s*THRU\s+\(FROM\s+(FRONT|BACK)\)\s*", descs[i], flags=re.I):
                    descs[i] = "THRU"
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
                    "DESCRIPTION": (descs[i] if i < len(descs) else "").strip(),
                })
            # ---- Emit exploded ops file (ideal for machine-time calc) ----
            ops_rows: List[Dict[str,str]] = []
            for i, hole in enumerate(hole_letters):
                base_d = diam_tokens[i]
                desc   = (descs[i] if i < len(descs) else "").strip()
                if not desc:
                    continue
                ops_rows += _explode_description_into_ops(i, hole, base_d, qtys, desc, hole_letters, diam_tokens)

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
# >>> HOLE_TABLE_OPS START
            try:
                # `text_rows` must be the list of HOLE TABLE text lines (strings) already collected
                # right before/when you wrote `hole_table_structured.csv`.
                # If it has a different variable name in this file, use that variable instead.
                ops_rows = explode_rows_to_operations(text_rows)

                # Write ops CSV next to the structured CSV
                import csv, pathlib
                out_dir = pathlib.Path(csv_path).parent if 'csv_path' in globals() else pathlib.Path('.')
                ops_csv = out_dir / "hole_table_ops.csv"
                with ops_csv.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["HOLE", "REF_DIAM", "QTY", "DESCRIPTION/DEPTH"])
                    w.writerows(ops_rows)

                print(f"[HOLE-TABLE][ops] rows={len(ops_rows)} csv={ops_csv}")
            except Exception as e:
                print(f"[HOLE-TABLE][ops] emit failed: {e}")
# <<< HOLE_TABLE_OPS END
            print(f"[HOLE-TABLE] rows={len(out_rows)} csv={hole_csv}")
        else:
            print("[HOLE-TABLE] none detected")
    except Exception as e:
        print(f"[HOLE-TABLE] parse failed: {e}")
    finally:
        _set_header_qtys([])

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
