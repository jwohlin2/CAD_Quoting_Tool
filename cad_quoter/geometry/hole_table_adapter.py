"""
hole_table_adapter.py
Thin adapter that extracts HOLE-TABLE text from an ezdxf document, then delegates
to tools.hole_ops to produce structured rows + atomic operations (in-memory).
"""

from typing import List, Tuple, Dict

# If tools/ is not a package, add a path shim here or in the caller:
try:
    from tools.hole_ops import explode_rows_to_operations  # main entry
except Exception:  # fallback if needed
    import sys
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
    from hole_ops import explode_rows_to_operations  # type: ignore

# Import the same helpers you already use in geo_dump.py to find hole-table chunks.
# If geo_dump helpers are local-private, paste minimal clones here (recommended),
# or do a safe import if module layout allows it.
from cad_quoter.geo_dump import (
    _find_hole_table_chunks,
    _parse_header,
    _split_descriptions,
)  # reuse


def _collect_text_rows_from_doc(doc) -> List[str]:
    """
    Collect normalized text lines from the ezdxf document (model space).
    Reuse your existing extraction logic if it’s in a helper; otherwise do a simple pass.
    """
    rows: List[str] = []
    msp = doc.modelspace()
    # TEXT + MTEXT + TABLE (strings); proxy text is already lifted by your existing extractor;
    # if not, you can add a small ACAD_PROXY_ENTITY reader here later.
    for e in msp:
        try:
            if e.dxftype() in ("TEXT", "MTEXT"):
                s = str(getattr(e, "plain_text", None) or e.dxf.text or "")
                if s.strip():
                    rows.append(s)
            elif e.dxftype() == "TABLE":
                # naive: any table cells' text; improve later if needed
                if hasattr(e, "get_cell"):
                    for row in range(e.nrows):
                        buf = []
                        for col in range(e.ncolumns):
                            c = e.get_cell(row, col)
                            if c and getattr(c, "value", ""):
                                buf.append(str(c.value))
                        if buf:
                            rows.append(" ".join(buf))
        except Exception:
            continue
    return rows


def extract_hole_table_from_doc(
    doc,
) -> Tuple[List[Dict[str, str]], List[Tuple[str, str, int, str]]]:
    """
    Returns:
        structured_rows: List of dicts {HOLE, REF_DIAM, QTY, DESCRIPTION}
        ops_rows:        List of tuples (HOLE, REF_DIAM, QTY, DESCRIPTION/DEPTH)
    Raises:
        RuntimeError if no HOLE TABLE is found (caller may ignore).
    """
    text_rows = _collect_text_rows_from_doc(doc)
    header_chunks, body_chunks = _find_hole_table_chunks(text_rows)
    if not header_chunks:
        raise RuntimeError("No HOLE TABLE found")

    hole_letters, diam_tokens, qtys = _parse_header(header_chunks)
    descs = _split_descriptions(body_chunks, diam_tokens)

    structured: List[Dict[str, str]] = []
    n = min(len(hole_letters), len(diam_tokens), len(qtys))
    for i in range(n):
        structured.append(
            {
                "HOLE": hole_letters[i],
                "REF_DIAM": diam_tokens[i],
                "QTY": str(qtys[i]),
                "DESCRIPTION": (descs[i] if i < len(descs) else "").strip(),
            }
        )

    # Build ops from the exact text block (header + body), same as geo_dump
    ops_rows = explode_rows_to_operations(header_chunks + body_chunks)
    return structured, ops_rows
