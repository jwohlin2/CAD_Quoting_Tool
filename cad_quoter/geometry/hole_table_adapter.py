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
    _decode_uplus,
    _find_hole_table_chunks,
    _parse_header,
    _split_descriptions,
)  # reuse
from cad_quoter.geo_extractor import collect_all_text


def _collect_text_rows_from_doc(doc) -> List[Dict[str, str]]:
    """Collect normalized text rows from the ezdxf document via geo_extractor."""

    try:
        model_name = str(getattr(doc.modelspace(), "name", "Model") or "Model")
    except Exception:
        model_name = "Model"
    model_name_norm = model_name.strip().lower()

    rows: List[Dict[str, str]] = []
    for rec in collect_all_text(doc):
        layout = str(rec.get("layout", "") or "")
        if layout.strip().lower() != model_name_norm:
            continue

        etype = str(rec.get("etype", "") or "")
        if etype not in {"PROXYTEXT", "MTEXT", "TEXT"}:
            continue

        text_val = rec.get("text", "")
        if not isinstance(text_val, str):
            text_val = str(text_val)

        normalized = " ".join(_decode_uplus(text_val).split()).strip()
        if not normalized:
            continue

        row: Dict[str, str] = {"layout": layout, "etype": etype, "text": normalized}
        rows.append(row)

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
    text_records = _collect_text_rows_from_doc(doc)
    header_chunks, body_chunks = _find_hole_table_chunks(text_records)
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
