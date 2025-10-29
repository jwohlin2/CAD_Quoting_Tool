import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ---------------------------
# Parsing helpers
# ---------------------------

_NUM = r'(?:\d+(?:\.\d+)?|\d+\s*-\s*\d+\/\d+|\d+\/\d+)'   # 1.25 | 1-1/2 | 13/32
_UNIT = r'(?:in(?:ch(?:es)?)?|")'                          # in | inch | inches | "
_X = r'[x×]'                                              # x or ×

RE_TRIPLET = re.compile(
    rf'(?P<a>{_NUM})\s*{_X}\s*(?P<b>{_NUM})\s*{_X}\s*(?P<c>{_NUM})(?:\s*(?P<u>{_UNIT}))?',
    re.IGNORECASE
)

RE_THICKNESS = re.compile(
    rf'\b(?:T(?:H(?:I(?:C(?:K(?:NESS)?)?)?)?)?)[=:]?\s*(?P<t>{_NUM})(?:\s*(?P<u>{_UNIT}))?\b',
    re.IGNORECASE
)

RE_LABELED = re.compile(
    rf'\b(?:L(?:EN(?:GTH)?)?)[=:]?\s*(?P<L>{_NUM})\b.*?\b(?:W(?:ID(?:TH)?)?)[=:]?\s*(?P<W>{_NUM})\b.*?\b(?:T|THK|THICK(?:NESS)?)[:=]?\s*(?P<T>{_NUM})\b',
    re.IGNORECASE
)

RE_STOCK_KEY = re.compile(r'\b(STOCK|BLANK|REQUIRED\s+BLANK|SIZE)\b', re.IGNORECASE)

def _to_float_in(token: str) -> Optional[float]:
    """Convert a dimension token to inches. Supports mixed numbers and fractions. Returns None if not parseable."""
    s = token.strip().lower().replace('"', '')
    # mixed: 1-1/2
    m_mixed = re.match(r'^(\d+)\s*-\s*(\d+)\/(\d+)$', s)
    if m_mixed:
        whole, num, den = map(int, m_mixed.groups())
        return whole + (num / den if den else 0.0)
    # fraction: 13/32
    m_frac = re.match(r'^(\d+)\/(\d+)$', s)
    if m_frac:
        num, den = map(int, m_frac.groups())
        return num / den if den else None
    # decimal/integer
    try:
        return float(s)
    except:
        return None

def _take_triplet(a: str, b: str, c: str) -> Optional[Tuple[float, float, float]]:
    aa = _to_float_in(a); bb = _to_float_in(b); cc = _to_float_in(c)
    if aa is None or bb is None or cc is None:
        return None
    vals = sorted([aa, bb, cc])
    # Plate assumption: smallest = thickness
    T = vals[0]; L, W = vals[2], vals[1]
    return (L, W, T)

def _fmt(val: float) -> str:
    # 3 decimals for nice display; keep trailing zeros for stability
    return f"{val:.3f}".rstrip('0').rstrip('.') if not math.isclose(val, round(val)) else f"{val:.2f}"

# ---------------------------
# Core inference
# ---------------------------

def infer_stock_dims_from_lines(lines: Iterable[str]) -> Optional[Tuple[float, float, float]]:
    """
    Given plain text lines, try to infer (L, W, T) in inches.
    Heuristics priority:
      1) L=.. W=.. T=.. (labeled triple)
      2) Any 'stock/blank/size' line containing a triplet A x B x C
      3) Any line containing a triplet A x B x C
      4) Separate T parsed via THK/Thickness if only 2D triplet is present (rare)
    """
    lines = [clean_cad_text(s) for s in lines]

    # 1) L/W/T labeled on one line
    for s in lines:
        m = RE_LABELED.search(s)
        if m:
            L = _to_float_in(m.group('L'))
            W = _to_float_in(m.group('W'))
            T = _to_float_in(m.group('T'))
            if None not in (L, W, T):
                # ensure L >= W
                L, W = max(L, W), min(L, W)
                return (L, W, T)

    # 2) "Stock/Blank/Size" annotated triplet
    for s in lines:
        if RE_STOCK_KEY.search(s):
            m = RE_TRIPLET.search(s)
            if m:
                trip = _take_triplet(m.group('a'), m.group('b'), m.group('c'))
                if trip:
                    return trip

    # 3) Any triplet anywhere
    for s in lines:
        m = RE_TRIPLET.search(s)
        if m:
            trip = _take_triplet(m.group('a'), m.group('b'), m.group('c'))
            if trip:
                return trip

    # 4) (optional) If we have a thickness token somewhere and later find a 2-tuple, we could combine.
    # Keeping it simple for now—return None.
    return None

# ---------------------------
# CAD text normalization
# ---------------------------

def clean_cad_text(s: str) -> str:
    """Normalize CAD text for parsing: decode \\U+XXXX, unify separators, collapse spaces."""
    s = _decode_uplus(s)
    s = s.replace('×', 'x').replace('–', '-').replace('—', '-')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _decode_uplus(s: str) -> str:
    def _repl(m):
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)
    return re.sub(r'\\U\+([0-9A-Fa-f]{4})', _repl, s)

# ---------------------------
# File readers
# ---------------------------

def read_texts_from_csv(p: Path) -> List[str]:
    """Return the 4th field (Text) from dxf_text_dump.csv rows."""
    out = []
    with p.open('r', newline='', encoding='utf-8') as fh:
        rdr = csv.reader(fh)
        for row in rdr:
            if not row or len(row) < 4:
                continue
            out.append(row[3])
    return out

def read_texts_from_jsonl(p: Path) -> List[str]:
    """Return 'text' field from dxf_text_dump.jsonl rows."""
    out = []
    with p.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get('text') or obj.get('Text') or ''
                if t:
                    out.append(str(t))
            except Exception:
                continue
    return out

# ---------------------------
# CLI
# ---------------------------

def main(argv: List[str]) -> int:
    """
    Usage:
        python tools/stock_dims.py --csv path/to/dxf_text_dump.csv
        python tools/stock_dims.py --jsonl path/to/dxf_text_dump.jsonl
    Emits:
        - prints JSON: {"length_in": L, "width_in": W, "thickness_in": T}
        - writes stock_dims.json next to the input file
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="Path to dxf_text_dump.csv")
    ap.add_argument("--jsonl", type=str, help="Path to dxf_text_dump.jsonl")
    args = ap.parse_args(argv)

    lines: List[str] = []
    src: Optional[Path] = None
    if args.csv:
        src = Path(args.csv)
        lines = read_texts_from_csv(src)
    elif args.jsonl:
        src = Path(args.jsonl)
        lines = read_texts_from_jsonl(src)
    else:
        print("Provide --csv or --jsonl")
        return 2

    dims = infer_stock_dims_from_lines(lines)
    result = None
    if dims:
        L, W, T = dims
        result = {"length_in": float(f"{L:.6f}"), "width_in": float(f"{W:.6f}"), "thickness_in": float(f"{T:.6f}")}
    else:
        result = {"length_in": None, "width_in": None, "thickness_in": None}

    print(json.dumps(result, indent=2))

    if src:
        outp = src.parent / "stock_dims.json"
        with outp.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"[stock-dims] json={outp}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
