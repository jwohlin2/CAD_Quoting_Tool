# cad_quoter/geo_dump.py
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

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
    """Given a header diameter like 'Ø1.7500', build search aliases found in body text."""
    out = [token]
    if token.startswith(("Ø", "∅")):
        val_str = token[1:]
        try:
            val = float(val_str)
            frac = Fraction(val).limit_denominator(64)
            if abs(float(frac) - val) < 1e-4:
                out += [f"Ø{frac.numerator}/{frac.denominator}", f"∅{frac.numerator}/{frac.denominator}"]
            comp = f"{val:.3f}".rstrip("0").rstrip(".")
            out += [f"Ø{comp}", f"∅{comp}", f"(Ø{comp})", f"(∅{comp})"]
        except Exception:
            pass
    return out

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
    """Return (hole_letters, diam_tokens, qty_list). Raises if header is malformed."""
    header_text = re.sub(r"\s+", " ", " ".join(header_chunks)).strip().replace(",", " ")
    toks = header_text.split()

    def find_tok(name: str) -> int:
        name = name.upper()
        for idx, t in enumerate(toks):
            if t.upper() == name:
                return idx
        return -1

    i_hole = find_tok("HOLE")
    i_ref = find_tok("REF")
    i_qty = find_tok("QTY")
    i_desc = find_tok("DESCRIPTION") if find_tok("DESCRIPTION") != -1 else find_tok("DESC")
    if i_ref == -1:
        i_ref = find_tok("Ø")  # some drawings collapse REF column to a diameter symbol
    if min(i_hole, i_ref, i_qty, i_desc) == -1:
        raise ValueError(f"Unexpected HOLE TABLE header: {header_text}")
    hole_letters = toks[i_hole + 1 : i_ref]
    diam_tokens = toks[i_ref + 1 : i_qty]
    # only ints for QTY
    qty_list = []
    for q in toks[i_qty + 1 : i_desc]:
        try:
            qty_list.append(int(q))
        except Exception:
            pass
    n = min(len(hole_letters), len(diam_tokens), len(qty_list))
    return hole_letters[:n], diam_tokens[:n], qty_list[:n]


def _split_descriptions(body_chunks: List[str], diam_list: List[str]) -> List[str]:
    """Slice the big body text into one description per diameter marker."""
    blob = re.sub(r"\s+", " ", " ".join(body_chunks)).strip()
    needles = [_diameter_aliases(d) for d in diam_list]
    positions: List[int] = []
    cursor = 0
    for alts in needles:
        found = None
        for a in alts:
            p = blob.find(a, cursor)
            if p != -1:
                found = p
                break
        if found is None:
            # last resort: just look for numeric payload of the first alias
            num = re.sub(r"^[Ø∅]\s*", "", alts[0])
            p = blob.find(num, cursor)
            found = p if p != -1 else len(blob)
        positions.append(found)
        cursor = max(found + 1, cursor)
    segs: List[str] = []
    for k, start in enumerate(positions):
        end = positions[k + 1] if k + 1 < len(positions) else len(blob)
        seg = blob[start:end].strip()
        seg = re.sub(r"^[\(\s]*[Ø∅][0-9.]+[\)]?\s*", "", seg)
        seg = re.sub(r"^[\(\s]*[Ø∅][0-9]+/[0-9]+[\)]?\s*", "", seg)
        segs.append(seg)
    return segs

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
            descriptions = _split_descriptions(body_chunks, diam_tokens)
            out_rows = []
            for i, hole in enumerate(hole_letters):
                out_rows.append({
                    "HOLE": hole,
                    "REF_DIAM": diam_tokens[i],
                    "QTY": qtys[i],
                    "DESCRIPTION": (descriptions[i] if i < len(descriptions) else "").strip(),
                })
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
