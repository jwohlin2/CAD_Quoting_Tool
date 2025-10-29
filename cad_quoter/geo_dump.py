# cad_quoter/geo_dump.py
from __future__ import annotations

import argparse
import csv
import json
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
    Some segments still contain other hole markers (e.g., row F has '"Q"(Ø.332)...' which belongs to G).
    This pass finds all other diameter aliases inside each segment, cuts those sub-chunks out,
    and reassigns them to the correct hole index. The leading chunk (before the first internal marker)
    stays with the original hole.
    """
    # Build alias map: hole_idx -> [aliases...]
    alias_map: Dict[int, List[str]] = {}
    for idx, tok in enumerate(diam_list):
        alias_map[idx] = _diameter_aliases(tok)

        # also allow numeric-only fallbacks for decimals like .272 / 0.272
        if tok.startswith(("Ø", "∅")):
            s = tok[1:]
            try:
                f = float(s)
                comp = f"{f:.3f}".rstrip("0").rstrip(".")
                alias_map[idx] += [comp, f"0{comp}", f"({comp})", f"(0{comp})"]
            except Exception:
                pass

    # Precompile a big alternation -> (hole_idx, alias)
    # Sort aliases by length to prefer longer matches first and avoid partial overlaps
    alt_pairs: List[Tuple[int, str]] = []
    for j, alts in alias_map.items():
        for a in alts:
            if a:
                alt_pairs.append((j, re.escape(a)))
    # if nothing, return as-is
    if not alt_pairs:
        return descs

    # Longest-first in the alternation to reduce ambiguous overlaps
    alt_pairs.sort(key=lambda p: len(p[1]), reverse=True)
    alt_union = "|".join(a for _, a in alt_pairs)
    re_markers = re.compile(alt_union)

    def strip_marker_prefix(s: str) -> str:
        # fractions first (both Ø-leading and Ø-trailing), then decimals, then quoted REF like "Q"(Ø.332)
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9]+/[0-9]+\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[0-9]+/[0-9]+[Ø∅]\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9.]+\)?\s*',        '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*\((?:[Ø∅]\s*[0-9.]+)\)\s*', '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*', '', s)
        return s.strip()

    new_descs = [""] * len(descs)

    for i, seg in enumerate(descs):
        s = (seg or "").strip()
        if not s:
            continue

        # Find all internal markers and who they belong to
        hits = []
        for m in re_markers.finditer(s):
            alias = m.group(0)
            # which hole index is this alias from?
            target_idx = next((j for j, esc in alt_pairs if re.fullmatch(esc, re.escape(alias))), None)
            # fallback: O(n) resolve by raw compare
            if target_idx is None:
                for j, alts in alias_map.items():
                    if alias in alts:
                        target_idx = j
                        break
            if target_idx is not None and target_idx != i:
                hits.append((m.start(), m.end(), target_idx, alias))

        if not hits:
            # nothing to move; keep as-is
            new_descs[i] = s
            continue

        # Order hits left→right and slice the segment
        hits.sort(key=lambda t: t[0])
        out_self_parts = []
        cursor = 0
        for k, (a, b, tgt, alias) in enumerate(hits):
            # self-chunk before this foreign marker
            if a > cursor:
                out_self_parts.append(s[cursor:a])
            # foreign chunk: from marker to next marker (or end)
            next_a = hits[k+1][0] if k+1 < len(hits) else len(s)
            chunk = s[a:next_a]
            chunk = strip_marker_prefix(chunk)
            if chunk:
                # append to target hole
                if new_descs[tgt]:
                    new_descs[tgt] += " "
                new_descs[tgt] += chunk.strip()
            cursor = next_a
        # tail after the last marker belongs to self
        if cursor < len(s):
            out_self_parts.append(s[cursor:])

        self_text = strip_marker_prefix(" ".join(out_self_parts).strip())
        if self_text:
            new_descs[i] = self_text

    # Merge any original descs that had no edits
    for i in range(len(descs)):
        if not new_descs[i] and descs[i]:
            new_descs[i] = descs[i]

    # tidy punctuation/spacing
    new_descs = [re.sub(r"\s*;\s*$", "", x or "").strip() for x in new_descs]
    return new_descs


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
