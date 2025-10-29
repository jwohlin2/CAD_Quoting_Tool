"""Utilities for exploding hole table descriptions into operation rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Sequence


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


@dataclass
class HoleSpec:
    """Represents a single hole definition from the HOLE TABLE header."""

    name: str
    ref: str
    qty: str
    value: float
    aliases: set[str]


class HoleTableParsingError(RuntimeError):
    """Raised when the HOLE TABLE header cannot be parsed."""


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


def split_header_body(text_rows: Sequence[str]) -> tuple[list[str], list[str]]:
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
    desc_clean = " ".join(desc.strip().split())
    if not desc_clean or desc_clean == ";":
        return
    ops.setdefault(hole.name, []).append([hole.name, ref, hole.qty, desc_clean])


def extract_parenthetical_jig(
    clause: str,
    holes: Sequence[HoleSpec],
    ops: Dict[str, List[List[str]]],
) -> str:
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
    clause = clause.strip()
    clause = clause.strip(",")
    clause = re.sub(r"^\)+", "", clause).strip()
    clause = re.sub(r'\s*"[A-Za-z0-9]+"\s*\($', "", clause).strip()
    clause = clause.rstrip("(").strip()
    return clause


def process_segment(
    token: str,
    segment_text: str,
    holes: Sequence[HoleSpec],
    ops: Dict[str, List[List[str]]],
    thru_hints: Dict[str, str],
    current_idx: int,
    match_idx: int | None,
) -> tuple[int, str]:
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


def find_nearest_hole(token: str, holes: Sequence[HoleSpec]) -> int | None:
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


def explode_rows_to_operations(text_rows: Iterable[str]) -> List[List[str]]:
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


__all__ = ["explode_rows_to_operations"]
