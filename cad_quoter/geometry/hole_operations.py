"""
hole_operations.py
==================
Consolidated module for hole table parsing and machining operations extraction.

This module combines functionality from:
- tools/hole_ops.py - Explode HOLE TABLE text into atomic machining operations
- cad_quoter/geometry/hole_table_adapter.py - Extract hole tables from ezdxf documents
- tools/backup_hole_finder.py - Backup extraction from dimension annotations

Core rules:
    - Parse diameters only when marked with Ø/∅ or fractionØ (e.g., '13/32∅').
    - Never treat plain decimals (.38, .62, .0001) as diameters (they're depths/tolerances).
    - TAP ops use the hole's REF_DIAM (not thread majors). 'THRU…TAP…' splits; 'TAP…THRU' stays combined.
    - Move quoted-letter cross-hits (e.g., '"Q"(Ø.332)') to the referenced hole; remove from source.
    - 'FROM FRONT & BACK' → two ops; '(Ø.3750 JIG GRIND)' → own op routed to nearest hole with that Ø.
    - Stop parsing at 'LIST OF COORDINATES'.

Public API:
    - explode_rows_to_operations(text_rows) -> List of operation tuples
    - extract_hole_table_from_doc(doc) -> (structured_rows, ops_rows)
    - extract_holes_from_text_records(text_records) -> List[BackupHoleFeature]
    - convert_to_hole_operations(holes) -> List of operation lists
    - parse_mtext_hole_description(mtext) -> BackupHoleFeature
    - validate_holes(part_number, filepath, extracted) -> HoleValidationResult
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import ezdxf
    from ezdxf.entities import DXFEntity
    EZDXF_AVAILABLE = True
except ImportError:
    ezdxf = None  # type: ignore
    DXFEntity = None  # type: ignore
    EZDXF_AVAILABLE = False

# Import from existing modules
try:
    from cad_quoter.geometry.mtext_normalizer import normalize_acad_mtext
except ImportError:
    # Fallback if import fails
    def normalize_acad_mtext(line: str) -> str:
        if not line:
            return ""
        if line.startswith("{") and line.endswith("}"):
            line = line[1:-1]
        line = re.sub(r"\\H[0-9.]+x;", "", line)
        line = re.sub(r"\\C\d+;", "", line)
        line = re.sub(r"\\S([^\\^]+)\^([^;]+);", lambda m: f"{m.group(1).strip()}/{m.group(2).strip()}", line)
        line = line.replace("{}", "").strip()
        return line

try:
    from cad_quoter.geometry.mtext_normalizer import units_to_inch_factor
except ImportError:
    def units_to_inch_factor(insunits: int) -> float:
        units_factors = {
            0: 1.0, 1: 1.0, 2: 12.0, 3: 63360.0,
            4: 1.0 / 25.4, 5: 1.0 / 2.54, 6: 39.3701,
            7: 39370.1, 8: 1.0e-6, 9: 0.001, 10: 36.0,
            13: 1.0 / 25400.0, 14: 3.93701,
        }
        return units_factors.get(insunits, 1.0)


# =============================================================================
# CONSTANTS AND REGEX PATTERNS
# =============================================================================

INCH_TO_MM = 25.4

DIAMETER_TOKEN_RE = re.compile(r"Ø\s*(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?|\.\d+)")
THREAD_SPEC_RE = re.compile(
    r"(#\s*\d+-\d+|\d+\s*/\s*\d+-\d+|\d+(?:\.\d+)?-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?)",
    re.I,
)
JIG_PAREN_RE = re.compile(
    r"\((?:Ø|⌀)?\s*(?P<diam>(?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?)|(?:\.\d+))\s+(?P<label>JIG\s+GRIND[^)]*)\)",
    re.I,
)
FRONT_BACK_RE = re.compile(
    r"FROM\s+FRONT\s*(?:&|AND)\s*BACK(?:\s+AS\s+SHOWN)?",
    re.I,
)
LIST_OF_COORDS_RE = re.compile(r"LIST\s+OF\s+COORDINATES", re.I)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HoleSpec:
    """Represents a single hole definition from the HOLE TABLE header."""

    name: str
    ref: str
    qty: str
    value: float
    aliases: set[str]


@dataclass
class BackupHoleFeature:
    """Represents a hole feature extracted by the backup finder."""

    qty: int = 1
    diameter: Optional[str] = None  # Raw diameter string (e.g., "7/32", ".375")
    diameter_mm: Optional[float] = None
    is_thru: bool = False
    depth_in: Optional[float] = None
    operations: List[str] = field(default_factory=list)  # JIG GRIND, C'BORE, TAP, etc.
    from_face: Optional[str] = None  # FRONT, BACK, or None
    raw_text: str = ""
    cbore_diameter: Optional[str] = None
    cbore_depth_in: Optional[float] = None
    thread_spec: Optional[str] = None  # Thread specification (e.g., "5/16-18", "#10-32")


@dataclass
class HoleValidationResult:
    """Result of comparing extracted holes against expected."""

    part_number: str
    file_path: str
    extracted_holes: List[BackupHoleFeature]
    expected_holes: List[BackupHoleFeature]
    matches: bool
    discrepancies: List[str] = field(default_factory=list)


class HoleTableParsingError(RuntimeError):
    """Raised when the HOLE TABLE header cannot be parsed."""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_ref_token(token: str) -> str:
    """Normalize a diameter token so downstream processing is consistent."""

    token = token.strip().strip("()")
    token = token.replace("∅", "Ø").replace("⌀", "Ø")
    if not token:
        return ""
    if not token.startswith("Ø"):
        token = f"Ø{token.lstrip('Ø') }"
    payload = token[1:].strip().replace(" ", "")
    if not payload:
        return ""
    if "/" in payload:
        num, den = payload.split("/", 1)
        return f"Ø{num.strip()}/{den.strip()}"
    if payload.startswith("."):
        return f"Ø{payload}"
    if payload.startswith("0."):
        return f"Ø{payload[1:]}"
    if payload.startswith("0") and "." in payload:
        payload = payload.lstrip("0")
        if not payload.startswith("."):
            payload = f".{payload}" if payload else payload
        return f"Ø{payload}"
    return f"Ø{payload}"


def diameter_value(ref: str) -> float | None:
    """Convert a diameter reference string to a float value."""
    ref = normalize_ref_token(ref)
    if not ref:
        return None
    payload = ref[1:]
    try:
        if "/" in payload:
            return float(Fraction(payload))
        return float(payload)
    except (ValueError, ZeroDivisionError):
        return None


def build_aliases(ref: str) -> set[str]:
    """Build a set of equivalent diameter representations."""
    aliases: set[str] = set()
    base = normalize_ref_token(ref)
    if not base:
        return aliases
    aliases.add(base)
    value = diameter_value(base)
    if value is not None:
        frac = Fraction(value).limit_denominator(64)
        aliases.add(normalize_ref_token(f"Ø{frac.numerator}/{frac.denominator}"))
        compact = f"{value:.4f}".rstrip("0").rstrip(".")
        if compact:
            if compact.startswith("0."):
                compact = compact[1:]
            aliases.add(normalize_ref_token(f"Ø{compact}"))
        compact3 = f"{value:.3f}".rstrip("0").rstrip(".")
        if compact3:
            if compact3.startswith("0."):
                compact3 = compact3[1:]
            aliases.add(normalize_ref_token(f"Ø{compact3}"))
    return aliases


# =============================================================================
# HOLE TABLE PARSING (from hole_ops.py)
# =============================================================================

def split_header_body(text_rows: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split HOLE TABLE text into header and body sections."""
    header: list[str] = []
    body: list[str] = []
    in_table = False
    header_complete = False
    for raw in text_rows:
        if LIST_OF_COORDS_RE.search(raw):
            break
        upper = raw.upper()
        if not in_table:
            if "HOLE TABLE" in upper:
                in_table = True
                header.append(raw)
                if "DESCRIPTION" in upper:
                    header_complete = True
                continue
            continue
        if not header_complete:
            header.append(raw)
            if "DESCRIPTION" in upper:
                header_complete = True
            continue
        body.append(raw)
    return header, body


def parse_header(header_rows: Sequence[str]) -> list[HoleSpec]:
    """Parse HOLE TABLE header into list of HoleSpec objects."""
    if not header_rows:
        raise HoleTableParsingError("HOLE TABLE header not found")
    header_text = re.sub(r"\s+", " ", " ".join(header_rows)).strip()
    header_text = header_text.replace(",", " ")
    match = re.search(
        r"HOLE TABLE HOLE (?P<holes>.+?) REF (?P<refs>.+?) QTY (?P<qty>.+?) DESCRIPTION",
        header_text,
        re.I,
    )
    if not match:
        raise HoleTableParsingError(f"Unexpected HOLE TABLE header: {header_text}")
    hole_tokens = match.group("holes").split()
    ref_tokens = [tok for tok in match.group("refs").split() if re.search(r"[0-9]", tok)]
    qty_tokens = [tok for tok in match.group("qty").split() if tok.strip()]
    count = min(len(hole_tokens), len(ref_tokens), len(qty_tokens))
    specs: list[HoleSpec] = []
    for idx in range(count):
        name = hole_tokens[idx].strip()
        ref_norm = normalize_ref_token(ref_tokens[idx])
        qty = qty_tokens[idx]
        value = diameter_value(ref_norm) or 0.0
        aliases = build_aliases(ref_norm)
        specs.append(HoleSpec(name=name, ref=ref_norm, qty=str(qty), value=value, aliases=aliases))
    return specs


def prepare_body_blob(body_rows: Sequence[str]) -> str:
    """Prepare body text for parsing."""
    blob = " ".join(body_rows)
    blob = blob.replace("∅", "Ø").replace("⌀", "Ø")
    blob = re.sub(
        r"((?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?)|(?:\.\d+))\s*Ø",
        lambda m: f"Ø{m.group(1).replace(' ', '')}",
        blob,
    )
    blob = re.sub(r"\s+", " ", blob)
    return blob.strip()


def find_hole_for_token(token: str, holes: Sequence[HoleSpec], current_index: int) -> int | None:
    """Find the hole index that matches a diameter token."""
    alias_matches = [idx for idx, spec in enumerate(holes) if token in spec.aliases]
    if alias_matches:
        for idx in alias_matches:
            if idx > current_index:
                return idx
        first = alias_matches[0]
        if first >= current_index:
            return first
        return None
    value = diameter_value(token)
    if value is None:
        return None
    best_idx: int | None = None
    best_delta = float("inf")
    for idx, spec in enumerate(holes):
        delta = abs(spec.value - value)
        if delta < best_delta - 1e-6:
            best_delta = delta
            best_idx = idx
        elif abs(delta - best_delta) <= 1e-6:
            if best_idx is None or idx >= current_index > best_idx:
                best_idx = idx
    if best_delta <= 6e-4:
        return best_idx
    return None


def add_operation(ops: Dict[str, List[List[str]]], hole: HoleSpec, ref: str, desc: str) -> None:
    """Add an operation to the ops dictionary."""
    desc_clean = " ".join(desc.strip().split())
    if not desc_clean or desc_clean == ";":
        return
    ops.setdefault(hole.name, []).append([hole.name, ref, hole.qty, desc_clean])


def extract_parenthetical_jig(
    clause: str,
    holes: Sequence[HoleSpec],
    ops: Dict[str, List[List[str]]],
) -> str:
    """Extract parenthetical JIG GRIND operations."""
    def repl(match: re.Match[str]) -> str:
        diam_text = match.group("diam")
        label = match.group("label") or "JIG GRIND"
        token = normalize_ref_token(f"Ø{diam_text}")
        value = diameter_value(token)
        nearest_idx: int | None = None
        if value is not None:
            best_delta = float("inf")
            for idx, spec in enumerate(holes):
                delta = abs(spec.value - value)
                if delta < best_delta - 1e-6 or (
                    abs(delta - best_delta) <= 1e-6 and (nearest_idx is None or idx < nearest_idx)
                ):
                    best_delta = delta
                    nearest_idx = idx
        if nearest_idx is not None:
            hole_spec = holes[nearest_idx]
            add_operation(ops, hole_spec, token, " ".join(label.strip().split()))
        return ""

    return JIG_PAREN_RE.sub(repl, clause)


def split_front_back(desc: str) -> List[str]:
    """Split operation on FROM FRONT & BACK into two operations."""
    match = FRONT_BACK_RE.search(desc)
    if not match:
        return [desc.strip()]

    def replace_with(target: str) -> str:
        replaced = FRONT_BACK_RE.sub(target, desc)
        replaced = re.sub(r"\bAS\s+SHOWN\b", "", replaced, flags=re.I)
        return " ".join(replaced.strip().split())

    return [replace_with("FROM FRONT"), replace_with("FROM BACK")]


def split_clause_on_thread(
    clause: str,
    holes: Sequence[HoleSpec],
    current_idx: int,
    ref_for_current: str,
    ops: Dict[str, List[List[str]]],
) -> tuple[str, int, str]:
    """Split clause on thread specifications."""
    while True:
        match = THREAD_SPEC_RE.search(clause)
        if not match or match.start() == 0:
            break
        before = clause[: match.start()].strip()
        if before:
            for chunk in split_front_back(before):
                add_operation(ops, holes[current_idx], ref_for_current, chunk)
        clause = clause[match.start() :].strip()
        current_idx = min(current_idx + 1, len(holes) - 1)
        ref_for_current = holes[current_idx].ref
    return clause, current_idx, ref_for_current


def handle_leading_thru_thread(
    clause: str,
    holes: Sequence[HoleSpec],
    current_idx: int,
    ref_for_current: str,
    ops: Dict[str, List[List[str]]],
) -> tuple[str, int, str, bool]:
    """Handle leading THRU followed by thread specification."""
    forced = False
    while True:
        upper = clause.upper()
        if not upper.startswith("THRU "):
            break
        remainder = clause[4:].strip()
        match = THREAD_SPEC_RE.match(remainder)
        if not match:
            break
        add_operation(ops, holes[current_idx], holes[current_idx].ref, "THRU")
        forced = True
        clause = remainder[match.end() :].strip()
        current_idx = min(current_idx + 1, len(holes) - 1)
        ref_for_current = holes[current_idx].ref
        clause = match.group(0).strip() + (" " + clause if clause else "")
    return clause, current_idx, ref_for_current, forced


def split_desc_on_tap(desc: str) -> List[str]:
    """Split description on TAP operation."""
    upper = desc.upper()
    if upper.startswith("THRU") and " TAP" in upper:
        tap_index = upper.index(" TAP")
        first = desc[:tap_index].strip()
        second = desc[tap_index:].strip()
        parts: List[str] = []
        if first:
            parts.append(first)
        if second:
            parts.append(second)
        return parts
    return [desc]


def clean_clause_text(clause: str) -> str:
    """Clean up clause text for parsing."""
    clause = clause.strip()
    clause = clause.strip(",")
    clause = re.sub(r"^\)+", "", clause).strip()
    clause = re.sub(r'\s*"[A-Za-z0-9]+"\s*\($', "", clause).strip()
    clause = clause.rstrip("(").strip()
    return clause


def find_nearest_hole(token: str, holes: Sequence[HoleSpec]) -> int | None:
    """Find the nearest hole by diameter value."""
    value = diameter_value(token)
    if value is None:
        return None
    best_idx: int | None = None
    best_delta = float("inf")
    for idx, spec in enumerate(holes):
        delta = abs(spec.value - value)
        if delta < best_delta - 1e-6 or (
            abs(delta - best_delta) <= 1e-6 and (best_idx is None or idx < best_idx)
        ):
            best_delta = delta
            best_idx = idx
    return best_idx


def process_segment(
    token: str,
    segment_text: str,
    holes: Sequence[HoleSpec],
    ops: Dict[str, List[List[str]]],
    thru_hints: Dict[str, str],
    current_idx: int,
    match_idx: int | None,
) -> tuple[int, str]:
    """Process a segment of the body blob."""
    segment_text = segment_text.strip()
    if not segment_text:
        ref_for_current = holes[current_idx].ref if match_idx is not None else token
        return current_idx, ref_for_current
    if "JIG GRIND" in segment_text.upper() and match_idx is None:
        target_idx = find_nearest_hole(token, holes)
        if target_idx is not None:
            add_operation(ops, holes[target_idx], token, "JIG GRIND")
        ref_for_current = holes[current_idx].ref if match_idx is not None else token
        return current_idx, ref_for_current
    clauses = [part.strip() for part in segment_text.split(";")]
    idx = current_idx if match_idx is None else match_idx
    if idx >= len(holes):
        idx = len(holes) - 1
    ref_for_current = holes[idx].ref if match_idx is not None else token
    forced_thru = False
    for raw_clause in clauses:
        clause = clean_clause_text(raw_clause)
        if not clause:
            continue
        clause = extract_parenthetical_jig(clause, holes, ops)
        if not clause:
            continue
        clause, idx, ref_for_current, forced = handle_leading_thru_thread(
            clause, holes, idx, ref_for_current, ops
        )
        forced_thru = forced_thru or forced
        if not clause:
            continue
        clause, idx, ref_for_current = split_clause_on_thread(clause, holes, idx, ref_for_current, ops)
        if not clause:
            continue
        for part in split_front_back(clause):
            for fragment in split_desc_on_tap(part):
                fragment = fragment.strip()
                if not fragment:
                    continue
                hole_spec = holes[idx]
                ref_for_op = ref_for_current
                if "TAP" in fragment.upper():
                    ref_for_op = hole_spec.ref
                add_operation(ops, hole_spec, ref_for_op, fragment)
                if (
                    fragment.upper() == "THRU"
                    and match_idx is not None
                    and holes[match_idx].name == hole_spec.name
                    and not forced_thru
                    and "/" in token
                ):
                    thru_hints[hole_spec.name] = token
    return idx, ref_for_current


def explode_rows_to_operations(text_rows: Iterable[str]) -> List[List[str]]:
    """
    Explode HOLE TABLE text into atomic machining operations.

    Args:
        text_rows: List of HOLE TABLE lines (header then body)

    Returns:
        List of [HOLE, REF_DIAM, QTY, DESCRIPTION/DEPTH] lists
    """
    rows = list(text_rows)
    header_rows, body_rows = split_header_body(rows)
    holes = parse_header(header_rows)
    body_blob = prepare_body_blob(body_rows)
    matches = list(DIAMETER_TOKEN_RE.finditer(body_blob))
    if not matches:
        return []
    ops: Dict[str, List[List[str]]] = {}
    thru_hints: Dict[str, str] = {}
    current_idx = 0
    ref_for_current = holes[current_idx].ref
    for idx, match in enumerate(matches):
        token = normalize_ref_token(match.group(0))
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(body_blob)
        segment_text = body_blob[match.end() : next_start]
        match_idx = find_hole_for_token(token, holes, current_idx)
        if match_idx is not None:
            current_idx = match_idx
        current_idx, ref_for_current = process_segment(
            token,
            segment_text,
            holes,
            ops,
            thru_hints,
            current_idx,
            match_idx,
        )
        if match_idx is not None:
            ref_for_current = holes[current_idx].ref
        else:
            ref_for_current = token
    ordered: List[List[str]] = []
    for spec in holes:
        entries = ops.get(spec.name, [])
        for entry in entries:
            if entry[3].upper() == "THRU":
                entry[1] = spec.ref
        hint = thru_hints.get(spec.name)
        if hint and len(entries) == 1 and not spec.ref.endswith("0"):
            entries[0][1] = hint
        ordered.extend(entries)
    return ordered


# =============================================================================
# DOCUMENT EXTRACTION (from hole_table_adapter.py)
# =============================================================================

def _collect_text_rows_from_doc(doc) -> List[Dict[str, str]]:
    """Collect normalized text rows from an ezdxf document."""
    # Import here to avoid circular imports
    from cad_quoter.geo_extractor import collect_all_text
    from cad_quoter.geo_dump import _decode_uplus

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
) -> Tuple[List[Dict[str, str]], List[List[str]]]:
    """
    Extract hole table from an ezdxf document.

    Args:
        doc: ezdxf document object

    Returns:
        Tuple of:
            structured_rows: List of dicts {HOLE, REF_DIAM, QTY, DESCRIPTION}
            ops_rows: List of [HOLE, REF_DIAM, QTY, DESCRIPTION/DEPTH] lists

    Raises:
        RuntimeError: If no HOLE TABLE is found
    """
    # Import here to avoid circular imports
    from cad_quoter.geo_dump import (
        _find_hole_table_chunks,
        _parse_header as geo_parse_header,
        _split_descriptions,
    )

    text_records = _collect_text_rows_from_doc(doc)
    header_chunks, body_chunks = _find_hole_table_chunks(text_records)
    if not header_chunks:
        raise RuntimeError("No HOLE TABLE found")

    hole_letters, diam_tokens, qtys = geo_parse_header(header_chunks)
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

    # Build ops from the exact text block (header + body)
    ops_rows = explode_rows_to_operations(header_chunks + body_chunks)
    return structured, ops_rows


# =============================================================================
# BACKUP HOLE FINDING (from backup_hole_finder.py)
# =============================================================================

# Expected holes for known parts (for validation)
EXPECTED_HOLES: Dict[str, List[str]] = {
    "108": [
        r"(2) Ø.2500 THRU\X(JIG GRIND)",
        r"\A1;(3) ∅7/32 THRU; ∅11/32 C'BORE\PX .100 DEEP FROM FRONT",
    ],
    "157": [
        r"\A1;5/16-18 TAP X 1.00 DEEP FROM BACK",
    ],
    "348": [
        r"\A1;5/16-18 TAP X 1.00 DEEP FROM BACK",
    ],
}


def parse_mtext_hole_description(mtext: str) -> BackupHoleFeature:
    """
    Parse an AutoCAD MTEXT hole description into a BackupHoleFeature.

    Examples:
        "(2) <> THRU\\X(JIG GRIND)" -> qty=2, thru=True, ops=[JIG GRIND]
        "\\A1;(3) ∅7/32 THRU; ∅11/32 C'BORE\\PX .100 DEEP FROM FRONT"
            -> qty=3, dia=7/32, thru=True, cbore_dia=11/32, cbore_depth=.100

    Args:
        mtext: Raw MTEXT string with formatting codes

    Returns:
        Parsed BackupHoleFeature
    """
    feature = BackupHoleFeature(raw_text=mtext)

    # Normalize the text for easier parsing
    text = mtext

    # Remove alignment codes like \A1;
    text = re.sub(r"\\A\d+;", "", text)

    # Replace \X with space (line break)
    text = text.replace(r"\X", " ")

    # Replace \P with space (paragraph)
    text = re.sub(r"\\P[A-Z]?", " ", text)

    # Normalize other MTEXT codes
    text = normalize_acad_mtext(text) if callable(normalize_acad_mtext) else text

    # Extract quantity: (2), (3), etc.
    qty_match = re.search(r"\((\d+)\)", text)
    if qty_match:
        feature.qty = int(qty_match.group(1))

    # Check for THRU
    if re.search(r"\bTHRU\b", text, re.IGNORECASE):
        feature.is_thru = True

    # Extract main diameter
    dia_patterns = [
        r"[∅Ø]\s*(\d+\s*/\s*\d+)",  # Fraction: ∅7/32
        r"[∅Ø]\s*(\.\d+)",          # Decimal: Ø.375
        r"[∅Ø]\s*(\d+\.\d+)",       # Decimal with leading: Ø0.375
        r"\)\s*(\.\d+)\s+THRU",     # Plain decimal after qty: (2) .2500 THRU
        r"\)\s*(\d+\.\d+)\s+THRU",  # Plain decimal with leading: (2) 0.2500 THRU
    ]

    for pattern in dia_patterns:
        dia_match = re.search(pattern, text)
        if dia_match:
            feature.diameter = dia_match.group(1).replace(" ", "")
            # Convert to mm
            if "/" in feature.diameter:
                try:
                    frac = Fraction(feature.diameter)
                    feature.diameter_mm = float(frac) * INCH_TO_MM
                except (ValueError, ZeroDivisionError):
                    pass
            else:
                try:
                    feature.diameter_mm = float(feature.diameter) * INCH_TO_MM
                except ValueError:
                    pass
            break

    # Check for <> placeholder (diameter from dimension)
    if "<>" in text and not feature.diameter:
        feature.diameter = "<>"

    # Extract JIG GRIND
    if re.search(r"\bJIG\s*GRIND\b", text, re.IGNORECASE):
        feature.operations.append("JIG GRIND")

    # Extract thread specification and TAP operation
    thread_match = re.search(
        r"(#\s*\d+-\d+|\d+\s*/\s*\d+-\d+|\d+(?:\.\d+)?-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?)\s*TAP",
        text,
        re.IGNORECASE
    )
    if thread_match:
        feature.thread_spec = thread_match.group(1).replace(" ", "")
        feature.operations.append("TAP")
    elif re.search(r"\bTAP\b", text, re.IGNORECASE):
        feature.operations.append("TAP")

    # Extract depth for TAP operations
    tap_depth_match = re.search(
        r"X\s*(\d*\.?\d+)\s*DEEP",
        text,
        re.IGNORECASE
    )
    if tap_depth_match:
        try:
            feature.depth_in = float(tap_depth_match.group(1))
        except ValueError:
            pass

    # Extract C'BORE
    cbore_match = re.search(
        r"[∅Ø]\s*(\d+\s*/\s*\d+|\.\d+|\d+\.\d+)\s*C['']?BORE",
        text,
        re.IGNORECASE
    )
    if cbore_match:
        feature.cbore_diameter = cbore_match.group(1).replace(" ", "")
        feature.operations.append("C'BORE")

    # Extract C'BORE depth
    cbore_depth_match = re.search(
        r"C['']?BORE.*?(\.\d+|\d+\.\d+)\s*DEEP",
        text,
        re.IGNORECASE
    )
    if cbore_depth_match:
        try:
            feature.cbore_depth_in = float(cbore_depth_match.group(1))
        except ValueError:
            pass

    # Also check for depth pattern without C'BORE prefix
    if not feature.cbore_depth_in:
        depth_match = re.search(r"(\.\d+|\d+\.\d+)\s*DEEP", text, re.IGNORECASE)
        if depth_match and "C'BORE" in text.upper():
            try:
                feature.cbore_depth_in = float(depth_match.group(1))
            except ValueError:
                pass

    # Extract FROM FRONT/BACK
    face_match = re.search(r"FROM\s+(FRONT|BACK)", text, re.IGNORECASE)
    if face_match:
        feature.from_face = face_match.group(1).upper()

    return feature


def _is_hole_description(text: str) -> bool:
    """Check if text appears to be a hole description."""

    upper = text.upper()

    # Check for quantity marker like (2), (3), etc.
    has_qty = bool(re.search(r"\(\d+\)", text))

    # Must have a diameter indicator or resolved decimal
    has_diameter = bool(
        re.search(r"[∅Ø]", text) or
        "<>" in text or
        re.search(r"\)\s*\.?\d+\.?\d*\s+THRU", text)
    )

    # Check for thread specification
    has_thread_spec = bool(
        re.search(
            r"(#\s*\d+-\d+|\d+\s*/\s*\d+-\d+|\d+(?:\.\d+)?-\d+|M\d+(?:\.\d+)?x\d+(?:\.\d+)?)\s*TAP",
            text,
            re.IGNORECASE
        )
    )

    # Must have a hole operation keyword
    hole_op_keywords = ["THRU", "JIG GRIND", "C'BORE", "CBORE", "C'DRILL", "TAP"]
    has_op_keyword = any(kw in upper for kw in hole_op_keywords)

    # For thread spec TAP patterns, qty is optional
    if has_thread_spec and has_op_keyword:
        return True

    # For other patterns, require qty and (diameter or thread spec)
    if not has_qty:
        return False

    return (has_diameter or has_thread_spec) and has_op_keyword


def extract_holes_from_text_records(text_records: List[Dict[str, Any]]) -> List[BackupHoleFeature]:
    """
    Extract hole features from geo_extractor text records.

    This is designed to work with text records from geo_dump/geo_extractor
    when no formal HOLE TABLE is found.

    Args:
        text_records: List of text record dicts with 'text' and 'etype' keys

    Returns:
        List of extracted BackupHoleFeature objects
    """
    holes: List[BackupHoleFeature] = []

    for record in text_records:
        text = record.get("text", "")
        if not text:
            continue

        if _is_hole_description(text):
            feature = parse_mtext_hole_description(text)
            if feature.diameter or feature.is_thru or feature.operations:
                holes.append(feature)

    return holes


def convert_to_hole_operations(holes: List[BackupHoleFeature]) -> List[List[str]]:
    """
    Convert BackupHoleFeature list to hole operations format.

    This matches the format expected by explode_rows_to_operations output:
    [HOLE_LETTER, REF_DIAM, QTY, OPERATION]

    Args:
        holes: List of BackupHoleFeature objects

    Returns:
        List of [hole_letter, ref_diam, qty, operation] lists
    """
    operations: List[List[str]] = []
    hole_letter = ord('A')

    for hole in holes:
        letter = chr(hole_letter)
        hole_letter += 1
        qty = str(hole.qty)

        # Format main hole diameter
        if hole.diameter and hole.diameter != "<>":
            main_diam = f"Ø{hole.diameter}"
        elif hole.diameter_mm:
            inches = hole.diameter_mm / INCH_TO_MM
            main_diam = f"Ø{inches:.4f}".rstrip("0").rstrip(".")
        else:
            main_diam = "Ø?"

        # Check for JIG GRIND operation
        if "JIG GRIND" in hole.operations:
            op_desc = "THRU (JIG GRIND)" if hole.is_thru else "(JIG GRIND)"
            operations.append([letter, main_diam, qty, op_desc])
            continue

        # Check for TAP operation
        if "TAP" in hole.operations:
            if hole.thread_spec:
                tap_ref = hole.thread_spec
            elif hole.diameter:
                tap_ref = f"Ø{hole.diameter}"
            else:
                tap_ref = "?"

            tap_desc = "TAP"
            if hole.depth_in:
                tap_desc += f" X {hole.depth_in:.2f} DEEP".rstrip("0").rstrip(".")
            if hole.from_face:
                tap_desc += f" FROM {hole.from_face}"

            operations.append([letter, tap_ref, qty, tap_desc])
            continue

        # For drill + C'BORE, create separate operations
        if hole.is_thru or hole.diameter:
            drill_desc = "THRU" if hole.is_thru else "DRILL"
            operations.append([letter, main_diam, qty, drill_desc])

        # C'BORE operation
        if "C'BORE" in hole.operations and hole.cbore_diameter:
            cbore_diam = f"Ø{hole.cbore_diameter}"
            cbore_desc = "C'BORE"
            if hole.cbore_depth_in:
                cbore_desc += f" X {hole.cbore_depth_in} DEEP"
            if hole.from_face:
                cbore_desc += f" FROM {hole.from_face}"
            operations.append([letter, cbore_diam, qty, cbore_desc])

    return operations


def extract_holes_from_dwg(filepath: str) -> List[BackupHoleFeature]:
    """
    Extract hole features from a DWG/DXF file using ezdxf.

    Args:
        filepath: Path to the DWG/DXF file

    Returns:
        List of extracted BackupHoleFeature objects
    """
    if not EZDXF_AVAILABLE:
        raise ImportError("ezdxf is required for DWG/DXF extraction")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    holes: List[BackupHoleFeature] = []

    try:
        doc = ezdxf.readfile(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read DWG/DXF file: {e}")

    # Get unit conversion factor
    try:
        insunits = doc.header.get("$INSUNITS", 0)
        unit_factor = units_to_inch_factor(insunits)
    except Exception:
        unit_factor = 1.0

    # Collect all text entities
    hole_texts: List[str] = []

    for entity in doc.modelspace():
        text = _extract_entity_text(entity, unit_factor)
        if text:
            hole_texts.append(text)

    # Parse hole descriptions
    for text in hole_texts:
        if _is_hole_description(text):
            feature = parse_mtext_hole_description(text)
            if feature.diameter or feature.is_thru or feature.operations:
                holes.append(feature)

    return holes


def _extract_entity_text(entity, unit_factor: float) -> str:
    """Extract text content from a DXF entity."""

    dxftype = entity.dxftype()

    if dxftype == "TEXT":
        return getattr(entity.dxf, "text", "")

    if dxftype == "MTEXT":
        return getattr(entity.dxf, "text", "")

    if dxftype == "DIMENSION":
        raw_text = getattr(entity.dxf, "text", "")

        # Handle <> placeholder
        if "<>" in raw_text:
            try:
                meas = entity.get_measurement()
                if meas is not None:
                    if hasattr(meas, "magnitude"):
                        meas = meas.magnitude
                    meas_in = float(meas) * unit_factor
                    meas_str = f"{meas_in:.4f}".rstrip("0").rstrip(".")
                    if meas_str.startswith("0."):
                        meas_str = meas_str[1:]
                    raw_text = raw_text.replace("<>", meas_str)
            except Exception:
                pass

        return raw_text

    return ""


# =============================================================================
# VALIDATION
# =============================================================================

def validate_holes(
    part_number: str,
    filepath: str,
    extracted: Optional[List[BackupHoleFeature]] = None,
) -> HoleValidationResult:
    """
    Validate extracted holes against expected values for a part.

    Args:
        part_number: Part identifier (e.g., "108")
        filepath: Path to the DWG/DXF file
        extracted: Pre-extracted holes (if None, will extract from file)

    Returns:
        HoleValidationResult with comparison details
    """
    # Get expected holes for this part
    expected_mtext = EXPECTED_HOLES.get(part_number, [])
    expected: List[BackupHoleFeature] = [
        parse_mtext_hole_description(mt) for mt in expected_mtext
    ]

    # Extract holes from file if not provided
    if extracted is None:
        try:
            extracted = extract_holes_from_dwg(filepath)
        except Exception as e:
            return HoleValidationResult(
                part_number=part_number,
                file_path=filepath,
                extracted_holes=[],
                expected_holes=expected,
                matches=False,
                discrepancies=[f"Extraction failed: {e}"],
            )

    # Compare extracted vs expected
    discrepancies: List[str] = []

    # Group by quantity for comparison
    extracted_by_qty = _group_holes_by_qty(extracted)
    expected_by_qty = _group_holes_by_qty(expected)

    # Check for matching quantities
    for qty, exp_holes in expected_by_qty.items():
        ext_holes = extracted_by_qty.get(qty, [])

        if len(ext_holes) != len(exp_holes):
            discrepancies.append(
                f"Qty {qty}: expected {len(exp_holes)} hole group(s), "
                f"found {len(ext_holes)}"
            )
            continue

        # Compare individual holes
        for i, (exp, ext) in enumerate(zip(exp_holes, ext_holes)):
            hole_diffs = _compare_holes(exp, ext)
            if hole_diffs:
                discrepancies.append(f"Qty {qty} hole {i+1}: {'; '.join(hole_diffs)}")

    # Check for unexpected holes
    for qty in extracted_by_qty:
        if qty not in expected_by_qty:
            discrepancies.append(f"Unexpected hole group with qty={qty}")

    matches = len(discrepancies) == 0

    return HoleValidationResult(
        part_number=part_number,
        file_path=filepath,
        extracted_holes=extracted,
        expected_holes=expected,
        matches=matches,
        discrepancies=discrepancies,
    )


def _group_holes_by_qty(holes: List[BackupHoleFeature]) -> Dict[int, List[BackupHoleFeature]]:
    """Group holes by their quantity."""
    groups: Dict[int, List[BackupHoleFeature]] = {}
    for hole in holes:
        if hole.qty not in groups:
            groups[hole.qty] = []
        groups[hole.qty].append(hole)
    return groups


def _compare_holes(expected: BackupHoleFeature, extracted: BackupHoleFeature) -> List[str]:
    """Compare two hole features and return list of differences."""

    diffs: List[str] = []

    # Compare THRU
    if expected.is_thru != extracted.is_thru:
        diffs.append(f"THRU: expected={expected.is_thru}, got={extracted.is_thru}")

    # Compare operations
    exp_ops = set(expected.operations)
    ext_ops = set(extracted.operations)

    missing_ops = exp_ops - ext_ops
    extra_ops = ext_ops - exp_ops

    if missing_ops:
        diffs.append(f"Missing operations: {', '.join(missing_ops)}")
    if extra_ops:
        diffs.append(f"Extra operations: {', '.join(extra_ops)}")

    # Compare diameter
    if expected.diameter and expected.diameter != "<>":
        if not extracted.diameter:
            diffs.append(f"Missing diameter: expected {expected.diameter}")
        elif expected.diameter_mm and extracted.diameter_mm:
            if abs(expected.diameter_mm - extracted.diameter_mm) > 0.1:
                diffs.append(
                    f"Diameter mismatch: expected {expected.diameter}, "
                    f"got {extracted.diameter}"
                )

    # Compare C'BORE
    if expected.cbore_diameter:
        if not extracted.cbore_diameter:
            diffs.append(f"Missing C'BORE diameter: expected {expected.cbore_diameter}")

    if expected.cbore_depth_in:
        if not extracted.cbore_depth_in:
            diffs.append(f"Missing C'BORE depth: expected {expected.cbore_depth_in}")
        elif abs(expected.cbore_depth_in - extracted.cbore_depth_in) > 0.001:
            diffs.append(
                f"C'BORE depth mismatch: expected {expected.cbore_depth_in}, "
                f"got {extracted.cbore_depth_in}"
            )

    # Compare FROM face
    if expected.from_face and expected.from_face != extracted.from_face:
        diffs.append(f"Face mismatch: expected {expected.from_face}, got {extracted.from_face}")

    return diffs


# =============================================================================
# CLI MAIN
# =============================================================================

def main():
    """Command-line interface for hole operations."""

    parser = argparse.ArgumentParser(
        description="Hole operations extraction and validation"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Path to DWG/DXF file"
    )
    parser.add_argument(
        "--part",
        help="Part number for validation (e.g., 108)"
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Show expected holes for the specified part"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--text",
        nargs="+",
        help="Raw MTEXT hole descriptions to parse (instead of file)"
    )

    args = parser.parse_args()

    # Show expected holes
    if args.show_expected:
        if args.part:
            if args.part not in EXPECTED_HOLES:
                print(f"No expected holes defined for part {args.part}")
                return
            print(f"\nExpected holes for part {args.part}:")
            print("=" * 60)
            for i, mtext in enumerate(EXPECTED_HOLES[args.part], 1):
                print(f"\nHole group {i}:")
                print(f"  Raw MTEXT: {mtext}")
                feature = parse_mtext_hole_description(mtext)
                print(f"  Parsed:")
                print(f"    Quantity: {feature.qty}")
                print(f"    Diameter: {feature.diameter or 'N/A'}")
                print(f"    Through: {feature.is_thru}")
                print(f"    Operations: {', '.join(feature.operations) or 'None'}")
                if feature.thread_spec:
                    print(f"    Thread spec: {feature.thread_spec}")
                if feature.depth_in:
                    print(f"    Depth: {feature.depth_in}")
                if feature.cbore_diameter:
                    print(f"    C'BORE dia: {feature.cbore_diameter}")
                if feature.cbore_depth_in:
                    print(f"    C'BORE depth: {feature.cbore_depth_in}")
                if feature.from_face:
                    print(f"    From face: {feature.from_face}")
        else:
            print("Available parts with expected holes:")
            for part in EXPECTED_HOLES:
                print(f"  - {part}")
        return

    # Parse text input
    if args.text:
        holes = [parse_mtext_hole_description(desc) for desc in args.text]
        result = {
            "file": "<text_input>",
            "holes": [
                {
                    "qty": h.qty,
                    "diameter": h.diameter,
                    "diameter_mm": h.diameter_mm,
                    "is_thru": h.is_thru,
                    "depth_in": h.depth_in,
                    "operations": h.operations,
                    "from_face": h.from_face,
                    "cbore_diameter": h.cbore_diameter,
                    "cbore_depth_in": h.cbore_depth_in,
                    "thread_spec": h.thread_spec,
                    "raw_text": h.raw_text,
                }
                for h in holes
            ],
        }
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"\nParsed {len(holes)} hole description(s):")
            for i, hole in enumerate(result["holes"], 1):
                print(f"\n  Hole {i}:")
                print(f"    Qty: {hole['qty']}")
                print(f"    Diameter: {hole['diameter'] or 'N/A'}")
                print(f"    Through: {hole['is_thru']}")
                print(f"    Operations: {', '.join(hole['operations']) or 'None'}")
        return

    # Extract from file
    if not args.filepath:
        parser.print_help()
        sys.exit(1)

    try:
        holes = extract_holes_from_dwg(args.filepath)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    result = {
        "file": args.filepath,
        "holes": [
            {
                "qty": h.qty,
                "diameter": h.diameter,
                "diameter_mm": h.diameter_mm,
                "is_thru": h.is_thru,
                "depth_in": h.depth_in,
                "operations": h.operations,
                "from_face": h.from_face,
                "cbore_diameter": h.cbore_diameter,
                "cbore_depth_in": h.cbore_depth_in,
                "thread_spec": h.thread_spec,
                "raw_text": h.raw_text,
            }
            for h in holes
        ],
    }

    # Add validation if part number provided
    if args.part and args.part in EXPECTED_HOLES:
        validation = validate_holes(args.part, args.filepath, holes)
        result["validation"] = {
            "matches": validation.matches,
            "expected_count": len(validation.expected_holes),
            "extracted_count": len(validation.extracted_holes),
            "discrepancies": validation.discrepancies,
        }

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\nHole Operations Results")
        print(f"File: {result['file']}")
        print("=" * 60)
        print(f"\nExtracted {len(result['holes'])} hole group(s):")
        for i, hole in enumerate(result["holes"], 1):
            print(f"\n  Hole group {i}:")
            print(f"    Qty: {hole['qty']}")
            print(f"    Diameter: {hole['diameter'] or 'N/A'}")
            print(f"    Through: {hole['is_thru']}")
            print(f"    Operations: {', '.join(hole['operations']) or 'None'}")
            if hole.get("thread_spec"):
                print(f"    Thread spec: {hole['thread_spec']}")
            if hole.get("depth_in"):
                print(f"    Depth: {hole['depth_in']}")
            if hole.get("cbore_diameter"):
                print(f"    C'BORE: {hole['cbore_diameter']}")
            if hole.get("cbore_depth_in"):
                print(f"    C'BORE depth: {hole['cbore_depth_in']}")
            if hole.get("from_face"):
                print(f"    From face: {hole['from_face']}")

        if result.get("validation"):
            val = result["validation"]
            print(f"\nValidation Results:")
            print(f"  Matches: {'YES' if val['matches'] else 'NO'}")
            print(f"  Expected: {val['expected_count']} group(s)")
            print(f"  Extracted: {val['extracted_count']} group(s)")
            if val["discrepancies"]:
                print(f"\n  Discrepancies:")
                for disc in val["discrepancies"]:
                    print(f"    - {disc}")


__all__ = [
    # Data structures
    "HoleSpec",
    "BackupHoleFeature",
    "HoleValidationResult",
    "HoleTableParsingError",
    # Utility functions
    "normalize_ref_token",
    "diameter_value",
    "build_aliases",
    # Core parsing
    "explode_rows_to_operations",
    "parse_header",
    "split_header_body",
    # Document extraction
    "extract_hole_table_from_doc",
    # Backup hole finding
    "parse_mtext_hole_description",
    "extract_holes_from_text_records",
    "convert_to_hole_operations",
    "extract_holes_from_dwg",
    # Validation
    "validate_holes",
    "EXPECTED_HOLES",
]


if __name__ == "__main__":
    main()
