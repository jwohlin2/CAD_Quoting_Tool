"""
geo_dump.py
===========
Text organizer + HOLE TABLE structurer.

Purpose:
    Read the normalized text stream from geo_extractor (dxf_text_dump.csv/jsonl),
    find HOLE TABLE blocks, and emit clean CSVs for quoting.

What it writes:
    - hole_table_structured.csv  # HOLE, REF_DIAM, QTY, DESCRIPTION (coarse per-hole view)
    - hole_table_ops.csv         # HOLE, REF_DIAM, QTY, DESCRIPTION/DEPTH (atomic ops)

How it works:
    1) Load dxf_text_dump.(csv|jsonl).
    2) Detect HOLE TABLE header/body lines.
    3) Parse headers (A..N, REF diameters, QTY), pair with coarse descriptions.
    4) Delegate operation splitting/logic to hole_ops.explode_rows_to_operations(text_rows).

Division of responsibilities:
    - geo_extractor: “CAD → plain text” (robust extraction/decoding).
    - geo_dump:      “text → structured tables” (find HOLE TABLE + I/O).
    - hole_ops:      “make it machinable” (split, redistribute, tap rules, depths).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for tools imports
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from tools.hole_ops import explode_rows_to_operations
    from tools.stock_dims import infer_stock_dims_from_lines, read_texts_from_csv
except ImportError:
    # Fallback if tools module isn't available
    explode_rows_to_operations = None  # type: ignore
    infer_stock_dims_from_lines = None  # type: ignore
    read_texts_from_csv = None  # type: ignore

from cad_quoter import geo_extractor

# Repository root (parent of cad_quoter directory)
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------- HOLE TABLE helpers ----------
_UHEX_RE = re.compile(r"\\U\+([0-9A-Fa-f]{4})")


def _decode_uplus(s: str) -> str:
    """Turn \\U+2205 into Ø etc., keep original if malformed."""
    return _UHEX_RE.sub(lambda m: chr(int(m.group(1), 16)), s or "")


def _diameter_aliases(token: str) -> List[str]:
    """Given 'Ø1.7500', build aliases found in body text."""
    out = [token]
    if token.startswith(("Ø", "∅")):
        val_str = token[1:]
        try:
            val = float(val_str)
            frac = Fraction(val).limit_denominator(64)
            if abs(float(frac) - val) < 1e-4:
                n, d = frac.numerator, frac.denominator
                out += [f"Ø{n}/{d}", f"∅{n}/{d}", f"{n}/{d}Ø", f"{n}/{d}∅"]
            comp = f"{val:.3f}".rstrip("0").rstrip(".")
            out += [f"Ø{comp}", f"∅{comp}", f"(Ø{comp})", f"(∅{comp})"]
            if comp.startswith("0."):
                bare = comp[1:]
                out += [f"Ø{bare}", f"∅{bare}", f"(Ø{bare})", f"(∅{bare})"]
        except Exception:
            pass
    return out


def _find_hole_table_chunks(rows: List[dict]):
    """Return (header_chunks, body_chunks) from text rows when a HOLE TABLE is present."""
    starts = [
        i
        for i, r in enumerate(rows)
        if r.get("etype") in ("PROXYTEXT", "MTEXT", "TEXT")
        and "HOLE TABLE" in (r.get("text", "").upper())
    ]
    if not starts:
        return [], []
    i = starts[0]
    header_chunks: List[str] = []
    body_chunks: List[str] = []
    j = i
    saw_desc = False
    while j < len(rows) and rows[j].get("etype") in ("PROXYTEXT", "MTEXT", "TEXT"):
        header_chunks.append(rows[j].get("text", ""))
        if "DESCRIPTION" in (rows[j].get("text", "").upper()):
            saw_desc = True
            j += 1
            break
        j += 1
    if not saw_desc:
        k = j
        while k < min(j + 5, len(rows)) and rows[k].get("etype") in ("PROXYTEXT", "MTEXT", "TEXT"):
            header_chunks.append(rows[k].get("text", ""))
            if "DESCRIPTION" in (rows[k].get("text", "").upper()):
                j = k + 1
                break
            k += 1
    while j < len(rows) and rows[j].get("etype") in ("PROXYTEXT", "MTEXT", "TEXT"):
        body_chunks.append(rows[j].get("text", ""))
        j += 1
    return header_chunks, body_chunks


def _parse_header(header_chunks: List[str]):
    """
    Return (hole_letters, diam_tokens, qty_list).
    Handles: 'HOLE TABLE HOLE A B ... REF Ø Ø1.7500 ... QTY 4 2 ... DESCRIPTION'
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
        i_ref = find_one("Ø")

    i_hole = -1
    for i in idx_holes:
        if i < i_ref:
            i_hole = i
    if min(i_hole, i_ref, i_qty, i_desc) == -1:
        raise ValueError(f"Unexpected HOLE TABLE header: {header_text}")

    hole_letters = toks[i_hole + 1 : i_ref]
    diam_tokens = toks[i_ref + 1 : i_qty]
    qty_tokens = toks[i_qty + 1 : i_desc]

    diam_tokens = [d for d in diam_tokens if d not in ("Ø", "∅")]

    qty_list = []
    for q in qty_tokens:
        try:
            qty_list.append(int(q))
        except Exception:
            pass

    n = min(len(hole_letters), len(diam_tokens), len(qty_list))
    return hole_letters[:n], diam_tokens[:n], qty_list[:n]


def _split_descriptions(body_chunks: List[str], diam_list: List[str]) -> List[str]:
    """ORDER-AGNOSTIC splitter. Redistribution happens separately."""
    blob = re.sub(r"\s+", " ", " ".join(body_chunks)).strip()
    m = re.search(r"\bLIST OF COORDINATES\b", blob, flags=re.I)
    if m:
        blob = blob[: m.start()].rstrip()

    positions: List[Tuple[int, int, str]] = []
    for idx, token in enumerate(diam_list):
        alts = _diameter_aliases(token)
        best_pos, best_alt = None, ""
        for a in alts:
            p = blob.find(a)
            if p != -1 and (best_pos is None or p < best_pos):
                best_pos, best_alt = p, a
        if best_pos is None:
            best_pos, best_alt = len(blob), ""
        positions.append((best_pos, idx, best_alt))

    positions.sort(key=lambda t: t[0])
    cuts = [p for (p, _, _) in positions]
    segments_in_order = []
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else len(blob)
        segments_in_order.append(blob[start:end])

    def strip_leading_marker(s: str) -> str:
        s = s.strip()
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9]+/[0-9]+\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[0-9]+/[0-9]+[Ø∅]\)?\s*', '', s)
        s = re.sub(r'^[\(\s]*[Ø∅]\s*[0-9.]+\)?\s*', '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*\((?:[Ø∅]\s*[0-9.]+)\)\s*', '', s)
        s = re.sub(r'^\s*"{1,2}[A-Z]"{1,2}\s*', '', s)
        return s.strip()

    descs = [""] * len(diam_list)
    for (pos, hole_idx, _), seg in zip(positions, segments_in_order):
        if pos >= len(blob):
            continue
        descs[hole_idx] = strip_leading_marker(seg)

    return descs


DEFAULT_SAMPLE_PATH = REPO_ROOT / "Cad Files" / "301_redacted.dxf"


def extract_hole_table_from_file(file_path: str | Path) -> List[dict]:
    """
    Public API: Extract hole table from a CAD file.

    Args:
        file_path: Path to DXF or DWG file

    Returns:
        List of dicts with keys: HOLE, REF_DIAM, QTY, DESCRIPTION
        Returns empty list if no hole table found.

    Example:
        >>> holes = extract_hole_table_from_file("301.dxf")
        >>> for hole in holes:
        ...     print(hole["HOLE"], hole["REF_DIAM"], hole["QTY"])
    """
    file_path = Path(file_path)

    # Open the CAD file
    doc = geo_extractor.open_doc(file_path)

    # Collect text records from all layouts with deep block exploration
    text_records = geo_extractor.collect_all_text(doc, max_block_depth=5)

    # Decode unicode characters
    for r in text_records:
        if "text" in r and isinstance(r["text"], str):
            r["text"] = _decode_uplus(r["text"])

    # Find and parse hole table
    header_chunks, body_chunks = _find_hole_table_chunks(text_records)

    if not header_chunks:
        return []

    hole_letters, diam_tokens, qtys = _parse_header(header_chunks)
    descs = _split_descriptions(body_chunks, diam_tokens)

    # Build structured hole data
    result = []
    n = min(len(hole_letters), len(diam_tokens), len(qtys))
    for i in range(n):
        result.append({
            "HOLE": hole_letters[i],
            "REF_DIAM": diam_tokens[i],
            "QTY": qtys[i],
            "DESCRIPTION": (descs[i] if i < len(descs) else "").strip(),
        })

    return result


def extract_hole_operations_from_file(file_path: str | Path) -> List[dict]:
    """
    Public API: Extract machining operations from hole table in a CAD file.

    This function parses the hole table and breaks it down into atomic
    machining operations (drill, tap, counterbore, etc.) using hole_ops logic.

    Args:
        file_path: Path to DXF or DWG file

    Returns:
        List of dicts with keys: HOLE, REF_DIAM, QTY, OPERATION
        Returns empty list if no hole table found or hole_ops unavailable.

    Example:
        >>> ops = extract_hole_operations_from_file("301.dxf")
        >>> for op in ops:
        ...     print(f"{op['HOLE']}: {op['OPERATION']}")
    """
    if not explode_rows_to_operations:
        return []  # hole_ops module not available

    file_path = Path(file_path)

    # Open the CAD file
    doc = geo_extractor.open_doc(file_path)

    # Collect text records from all layouts with deep block exploration
    text_records = geo_extractor.collect_all_text(doc, max_block_depth=5)

    # Decode unicode characters
    for r in text_records:
        if "text" in r and isinstance(r["text"], str):
            r["text"] = _decode_uplus(r["text"])

    # Find and parse hole table
    header_chunks, body_chunks = _find_hole_table_chunks(text_records)

    if not header_chunks:
        return []

    # Explode into operations using hole_ops
    text_rows = header_chunks + body_chunks
    ops_rows = explode_rows_to_operations(text_rows)

    # Convert to list of dicts
    result = []
    for row in ops_rows:
        if len(row) >= 4:
            result.append({
                "HOLE": row[0],
                "REF_DIAM": row[1],
                "QTY": row[2],
                "OPERATION": row[3],
            })

    return result


def _parse_csv_patterns(s: str | None) -> List[str]:
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

    for r in rows:
        if "text" in r and isinstance(r["text"], str):
            r["text"] = _decode_uplus(r["text"])

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

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # (Removed DIM-DUMP: no dims_all.csv/jsonl)

    # ---- Infer stock dimensions from extracted text ----
    if read_texts_from_csv and infer_stock_dims_from_lines:
        try:
            stock_texts = read_texts_from_csv(csv_path)
        except Exception as exc:
            print(f"[stock-dims] read failed: {exc}")
        else:
            dims = infer_stock_dims_from_lines(stock_texts)
            stock_csv = csv_path.parent / "stock_dims.csv"
            with stock_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["length_in", "width_in", "thickness_in"])
                if dims:
                    L, W, T = dims
                    print(f"[stock-dims] L={L:.3f} in, W={W:.3f} in, T={T:.3f} in")
                    writer.writerow([f"{L:.6f}", f"{W:.6f}", f"{T:.6f}"])
                else:
                    print("[stock-dims] no stock dimensions detected")
                    writer.writerow(["", "", ""])
            print(f"[stock-dims] csv={stock_csv}")
    else:
        print("[stock-dims] skipped (tools module not available)")

    # (Removed part-dims inference: no reading or computing of DIMENSION geometry)

    # ---- Emit structured HOLE TABLE + ops (via hole_ops) ----
    try:
        header_chunks, body_chunks = _find_hole_table_chunks(rows)
        if header_chunks:
            text_rows = header_chunks + body_chunks
            hole_letters, diam_tokens, qtys = _parse_header(header_chunks)

            # Structured: pair header columns with coarse descriptions (optional)
            descs = _split_descriptions(body_chunks, diam_tokens)
            out_rows = []
            n = min(len(hole_letters), len(diam_tokens), len(qtys))
            for i in range(n):
                out_rows.append(
                    {
                        "HOLE": hole_letters[i],
                        "REF_DIAM": diam_tokens[i],
                        "QTY": qtys[i],
                        "DESCRIPTION": (descs[i] if i < len(descs) else "").strip(),
                    }
                )

            # Write structured CSV
            hole_csv = csv_path.parent / "hole_table_structured.csv"
            with hole_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(
                    fh, fieldnames=["HOLE", "REF_DIAM", "QTY", "DESCRIPTION"]
                )
                w.writeheader()
                w.writerows(out_rows)
            print(f"[HOLE-TABLE] rows={len(out_rows)} csv={hole_csv}")

            # Ops: delegate to hole_ops (the source of truth)
            if explode_rows_to_operations:
                ops_rows = explode_rows_to_operations(text_rows)
                ops_csv = csv_path.parent / "hole_table_ops.csv"
                with ops_csv.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["HOLE", "REF_DIAM", "QTY", "DESCRIPTION/DEPTH"])
                    w.writerows(ops_rows)
                print(f"[HOLE-TABLE][ops] rows={len(ops_rows)} csv={ops_csv}")
            else:
                print("[HOLE-TABLE][ops] skipped (hole_ops module not available)")
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
