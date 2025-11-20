"""Shared hole table parsing helpers."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LETTER_DRILLS_INCH = {
    "A": 0.234,
    "B": 0.238,
    "C": 0.242,
    "Q": 0.332,
    "R": 0.339,
    "S": 0.348,
    "T": 0.358,
    "U": 0.368,
    "V": 0.377,
    "W": 0.386,
    "X": 0.397,
    "Y": 0.404,
    "Z": 0.413,
}

NUMBER_DRILLS_INCH = {
    "#1": 0.228,
    "#2": 0.221,
    "#3": 0.213,
    "#4": 0.209,
    "#5": 0.205,
    "#6": 0.204,
    "#7": 0.201,
    "#8": 0.199,
    "#9": 0.196,
    "#10": 0.193,
    "#11": 0.191,
    "#12": 0.189,
    "#13": 0.185,
    "#14": 0.182,
    "#15": 0.18,
    "#16": 0.177,
    "#17": 0.173,
    "#18": 0.169,
    "#19": 0.166,
    "#20": 0.161,
}

INCH_TO_MM = 25.4

NUM_PATTERN = r"(?:\d*\.\d+|\d+)"

# Standard thread combinations: major diameter (inches) -> list of valid TPI values
# These are the common UNC (coarse), UNF (fine), and UNEF (extra fine) threads
STANDARD_THREADS: Dict[str, List[int]] = {
    # Number threads
    "#0": [80],
    "#1": [64, 72],
    "#2": [56, 64],
    "#3": [48, 56],
    "#4": [40, 48],
    "#5": [40, 44],
    "#6": [32, 40],
    "#8": [32, 36],
    "#10": [24, 32],
    "#12": [24, 28],
    # Fractional threads
    "1/4": [20, 28, 32],
    "5/16": [18, 24, 32],
    "3/8": [16, 24, 32],
    "7/16": [14, 20, 28],
    "1/2": [13, 20, 28],
    "9/16": [12, 18, 24],
    "5/8": [11, 18, 24],
    "3/4": [10, 16, 20],
    "7/8": [9, 14, 20],
    "1": [8, 12, 14, 20],
}

# Major diameters in inches for each nominal size
THREAD_MAJOR_DIAMETERS: Dict[str, float] = {
    "#0": 0.0600, "#1": 0.0730, "#2": 0.0860, "#3": 0.0990, "#4": 0.1120,
    "#5": 0.1250, "#6": 0.1380, "#8": 0.1640, "#10": 0.1900, "#12": 0.2160,
    "1/4": 0.2500, "5/16": 0.3125, "3/8": 0.3750, "7/16": 0.4375, "1/2": 0.5000,
    "9/16": 0.5625, "5/8": 0.6250, "3/4": 0.7500, "7/8": 0.8750, "1": 1.0000,
}


def validate_and_correct_thread(
    major: str,
    tpi: int,
    nearby_diameters: Optional[List[float]] = None
) -> tuple[str, int, bool]:
    """
    Validate a parsed thread specification and correct if invalid.

    Args:
        major: Parsed major diameter (e.g., "5/16", "#10", "0.2500")
        tpi: Parsed threads per inch
        nearby_diameters: List of nearby dimension values (boss/counterbore) in inches

    Returns:
        tuple of (corrected_major, corrected_tpi, was_corrected)
    """
    # Normalize major to standard format
    normalized_major = major.strip()

    # If major is a decimal, try to map to closest standard size
    if re.match(r"^\d*\.\d+$", normalized_major):
        decimal_val = float(normalized_major)
        closest_nom = None
        closest_diff = float('inf')

        for nom, dia in THREAD_MAJOR_DIAMETERS.items():
            diff = abs(decimal_val - dia)
            if diff < closest_diff:
                closest_diff = diff
                closest_nom = nom

        if closest_nom and closest_diff < 0.02:  # Within 0.02" tolerance
            normalized_major = closest_nom

    # Check if the thread spec is valid
    if normalized_major in STANDARD_THREADS:
        valid_tpis = STANDARD_THREADS[normalized_major]
        if tpi in valid_tpis:
            return (normalized_major, tpi, False)  # Valid as-is

        # Invalid TPI for this nominal - find closest valid TPI
        closest_tpi = min(valid_tpis, key=lambda t: abs(t - tpi))
        return (normalized_major, closest_tpi, True)

    # Major not in standard list - try to find best match using nearby dimensions
    if nearby_diameters:
        # Find the closest standard nominal to the largest nearby diameter
        # (boss diameter is typically larger than thread major by ~0.05-0.10")
        for nearby_dia in sorted(nearby_diameters, reverse=True):
            for nom, thread_dia in THREAD_MAJOR_DIAMETERS.items():
                # Boss diameter should be >= thread major diameter
                # Check if nearby diameter matches a thread's boss size
                # Typical clearance for tapped holes: major + 0.03" to + 0.10"
                if thread_dia <= nearby_dia <= thread_dia + 0.15:
                    if nom in STANDARD_THREADS:
                        # Use coarse thread as default (first in list)
                        default_tpi = STANDARD_THREADS[nom][0]
                        return (nom, default_tpi, True)

    # Last resort: try to correct based on the parsed decimal major diameter
    if re.match(r"^\d*\.\d+$", major):
        decimal_val = float(major)
        closest_nom = None
        closest_diff = float('inf')

        for nom, dia in THREAD_MAJOR_DIAMETERS.items():
            diff = abs(decimal_val - dia)
            if diff < closest_diff:
                closest_diff = diff
                closest_nom = nom

        if closest_nom and closest_diff < 0.1:
            # Use standard coarse thread
            default_tpi = STANDARD_THREADS[closest_nom][0]
            return (closest_nom, default_tpi, True)

    # Cannot correct - return as-is but flag as potentially invalid
    return (major, tpi, False)


def is_valid_thread_spec(major: str, tpi: int) -> bool:
    """Check if a thread specification is a valid standard combination."""
    # Normalize decimal to nominal if possible
    if re.match(r"^\d*\.\d+$", major):
        decimal_val = float(major)
        for nom, dia in THREAD_MAJOR_DIAMETERS.items():
            if abs(decimal_val - dia) < 0.01:
                major = nom
                break

    if major in STANDARD_THREADS:
        return tpi in STANDARD_THREADS[major]
    return False


def _frac_to_mm(n: int, d: int) -> float:
    return (n / d) * INCH_TO_MM


def parse_drill_token(tok: str) -> Optional[float]:
    """Return drill diameter in mm from tokens like Ø.3750, 3/8, #7, Q, 0.339."""

    t = tok.strip().upper().replace("Ø", "").replace("O/", "Ø").replace(" ", "")

    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", t)
    if match:
        return _frac_to_mm(int(match.group(1)), int(match.group(2)))

    match = re.fullmatch(r"(?:0?)(\.\d+)", t)
    if match:
        return float(match.group(1)) * INCH_TO_MM

    match = re.fullmatch(r"(\d+(?:\.\d+)?)MM", t)
    if match:
        return float(match.group(1))

    if t in LETTER_DRILLS_INCH:
        return LETTER_DRILLS_INCH[t] * INCH_TO_MM
    if t in NUMBER_DRILLS_INCH:
        return NUMBER_DRILLS_INCH[t] * INCH_TO_MM

    match = re.fullmatch(r"(\d+(?:\.\d+)?)", t)
    if match and float(match.group(1)) < 1.5:
        return float(match.group(1)) * INCH_TO_MM

    return None


@dataclass
class HoleRow:
    ref: str
    qty: int
    features: List[Dict[str, Any]]
    raw_desc: str


_DEBUG_DIR = Path("debug")
_DEBUG_ROWS_PATH = _DEBUG_DIR / "hole_table_rows.csv"
_DEBUG_TOTALS_PATH = _DEBUG_DIR / "ops_table_totals.json"
_DEBUG_ROW_FIELDNAMES = ("ref", "qty", "raw_desc", "features_json")

_OPS_TOTAL_KEYS = ("Drill", "Tap", "C'bore", "C'drill", "Jig", "NPT")
_NPT_TOKEN_RE = re.compile(r"N\.?P\.?T", re.IGNORECASE)


def _normalize_side_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if upper.startswith("BACK"):
        return "BACK"
    if upper.startswith("FRONT"):
        return "FRONT"
    if upper.startswith("BOTH"):
        return "BOTH"
    return upper


def _operation_bucket(record: Mapping[str, Any]) -> str | None:
    kind = str(record.get("kind") or "").strip().lower()
    if not kind:
        return None

    raw_text = str(record.get("raw_text") or "")
    tool_text = str(record.get("tool") or "")
    thread_text = str(record.get("thread") or "")
    tap_type = str(record.get("tap_type") or "").strip().lower()
    combined = " ".join(part for part in (raw_text, tool_text, thread_text) if part)

    if tap_type == "pipe" or _NPT_TOKEN_RE.search(combined):
        return "NPT"

    if kind == "tap":
        return "Tap"
    if kind in {"counterbore", "cbore"}:
        return "C'bore"
    if kind in {"counterdrill", "cdrill"}:
        return "C'drill"
    if kind in {"jig", "jig_grind", "jig grind"}:
        return "Jig"

    if kind == "drill":
        upper_text = combined.upper()
        if "TAP" in upper_text or _NPT_TOKEN_RE.search(upper_text):
            return None
        return "Drill"

    return None


def _format_ops_totals_line(totals: Mapping[str, int]) -> str:
    parts: list[str] = []
    for label in _OPS_TOTAL_KEYS:
        try:
            value = int(totals.get(label, 0))
        except Exception:
            value = 0
        if value:
            parts.append(f"{label} {value}")
    return " | ".join(parts) if parts else "—"


def _format_inches_token(value_mm: Optional[float]) -> str:
    if value_mm is None:
        return ""
    inches = value_mm / INCH_TO_MM
    token = f"{inches:.4f}".rstrip("0").rstrip(".")
    return f"{token}\"" if token else ""


def _format_depth_token(feature: Dict[str, Any]) -> str:
    depth_mm = feature.get("depth_mm")
    if depth_mm is not None:
        return _format_inches_token(depth_mm)
    if feature.get("thru"):
        return "THRU"
    return ""


def _format_tool_token(kind: str, feature: Dict[str, Any]) -> str:
    if kind == "tap":
        thread = feature.get("thread")
        if thread:
            return str(thread)
    dia_mm = feature.get("dia_mm")
    if dia_mm is not None:
        return _format_inches_token(dia_mm)
    major_mm = feature.get("major_mm")
    if major_mm is not None:
        return _format_inches_token(major_mm)
    return ""


def _hole_rows_debug_records(
    rows: List[HoleRow],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    row_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    qty_sum = 0
    for row in rows:
        qty = int(getattr(row, "qty", 0) or 0)
        raw_desc = getattr(row, "raw_desc", "")
        features = list(getattr(row, "features", []) or [])
        try:
            features_json = json.dumps(features, sort_keys=True, default=str)
        except Exception:
            features_json = json.dumps([], sort_keys=True)
        row_records.append(
            {
                "ref": getattr(row, "ref", ""),
                "qty": qty,
                "raw_desc": raw_desc,
                "features_json": features_json,
            }
        )
        if not features:
            qty_sum += qty
            continue
        for feature in features:
            if not isinstance(feature, dict):
                continue
            kind = str(feature.get("type") or "")
            side = str(feature.get("from_face") or "")
            diam_mm = feature.get("dia_mm") or feature.get("major_mm")
            diam_token = _format_inches_token(diam_mm) if diam_mm is not None else ""
            record = {
                "qty": qty,
                "kind": kind,
                "side": _normalize_side_token(side),
                "tool": _format_tool_token(kind, feature),
                "diam_token": diam_token,
                "depth_token": _format_depth_token(feature),
                "raw_text": raw_desc,
            }
            thread_val = feature.get("thread")
            if thread_val:
                record["thread"] = str(thread_val)
            tap_type_val = feature.get("tap_type")
            if tap_type_val:
                record["tap_type"] = str(tap_type_val)
            feature_records.append(record)
            qty_sum += qty
    return row_records, feature_records, qty_sum


def _hole_rows_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {key: 0 for key in _OPS_TOTAL_KEYS}
    for record in records:
        bucket = _operation_bucket(record)
        if not bucket:
            continue
        try:
            qty_val = int(record.get("qty") or 0)
        except Exception:
            qty_val = 0
        totals[bucket] += qty_val
    return totals


def _write_hole_table_debug(
    row_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
    qty_sum: int,
) -> None:
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        with _DEBUG_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_DEBUG_ROW_FIELDNAMES)
            writer.writeheader()
            writer.writerows(row_records)
        totals = _hole_rows_totals(feature_records)
        with _DEBUG_TOTALS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(totals, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"[TABLE-DUMP] rows={len(row_records)} qty_sum={qty_sum} -> {_DEBUG_ROWS_PATH.as_posix()}"
        )
        print(f"[OPS] table: {_format_ops_totals_line(totals)}")
        print(f"[OPS] table totals -> {_DEBUG_TOTALS_PATH.as_posix()}")
    except Exception:
        pass


def _depth_or_thru(desc: str) -> tuple[Optional[float], Optional[bool]]:
    depth_match = re.search(rf"({NUM_PATTERN})\s*DEEP", desc)
    if depth_match:
        depth_val = float(depth_match.group(1)) * INCH_TO_MM
    else:
        depth_val = None
    return depth_val, (True if "THRU" in desc else None)


def _from_face(desc: str) -> Optional[str]:
    match = re.search(r"FROM\s+(FRONT|BACK)", desc)
    if match:
        return match.group(1).lower()
    return None


def _parse_description(desc: str) -> List[Dict[str, Any]]:
    text = " ".join(desc.upper().split())
    tokens: List[Dict[str, Any]] = []
    has_tap = False

    # Improved thread parsing with validation
    # Priority 1: Standard fractional or number thread format (e.g., "5/16-18", "#10-32")
    # Priority 2: Decimal format immediately before TAP (e.g., "0.250-20 TAP")
    # The decimal format is more restrictive to avoid mis-parsing depth values

    if "TAP" in text:
        major = None
        pitch = None

        # Try standard thread format first (most reliable)
        std_tap_re = re.compile(r"(\d+/\d+|\#\d+)\s*-\s*(\d+)")
        match = std_tap_re.search(text)
        if match:
            major = match.group(1)
            pitch = match.group(2)
        else:
            # Try decimal format, but require it to be immediately before TAP
            # This avoids matching "X 1.00 DEEP" + some unrelated "80" elsewhere
            decimal_tap_re = re.compile(r"(\d+\.\d+)\s*-\s*(\d+)\s*TAP")
            match = decimal_tap_re.search(text)
            if match:
                major = match.group(1)
                pitch = match.group(2)

        if major and pitch:
            tpi = int(pitch)

            # Validate and correct the thread specification
            corrected_major, corrected_tpi, was_corrected = validate_and_correct_thread(
                major, tpi, nearby_diameters=None
            )

            # Use corrected values for calculations
            if was_corrected:
                # Log the correction for debugging
                import logging
                logging.debug(
                    f"Thread corrected: {major}-{tpi} -> {corrected_major}-{corrected_tpi}"
                )
                major = corrected_major
                pitch = str(corrected_tpi)

            # Calculate major_mm from the corrected major
            if corrected_major in THREAD_MAJOR_DIAMETERS:
                major_mm = THREAD_MAJOR_DIAMETERS[corrected_major] * INCH_TO_MM
            else:
                major_mm = parse_drill_token(major) if major else None

            depth_mm, thru = _depth_or_thru(text)
            from_face = _from_face(text)
            tokens.append(
                {
                    "type": "tap",
                    "thread": f"{major}-{pitch}",
                    "major_mm": major_mm,
                    "depth_mm": depth_mm,
                    "thru": thru,
                    "from_face": from_face,
                    "source": "desc",
                    "was_corrected": was_corrected,
                }
            )
            has_tap = True

    # Skip DRILL parsing if we already found a TAP operation
    # TAP operations include their own drill operation (tap drill), so we don't want to create duplicates
    if not has_tap:
        tol_pattern = rf"(?:\s*[±\+\-]\s*{NUM_PATTERN})?"
        drill_pattern = rf"Ø?\(?\s*({NUM_PATTERN})\s*\)?{tol_pattern}\s*(?:DRILL|THRU|TYP|$)"
        for match in re.finditer(drill_pattern, text):
            start = match.start(1)
            preceding = text[:start].rstrip()
            if preceding and preceding[-1] in {"±", "+", "-"}:
                continue
            dia_mm = parse_drill_token(match.group(1))
            if dia_mm:
                depth_mm, thru = _depth_or_thru(text)
                from_face = _from_face(text)
                tokens.append(
                    {
                        "type": "drill",
                        "dia_mm": dia_mm,
                        "depth_mm": depth_mm,
                        "thru": thru,
                        "from_face": from_face,
                        "source": "desc",
                    }
                )

    cbore_depth = re.search(rf"C['’]?BORE\s+X\s+({NUM_PATTERN})\s*DEEP", text)
    if cbore_depth:
        depth_val = float(cbore_depth.group(1)) * INCH_TO_MM
        dia_match = re.search(rf"({NUM_PATTERN})\s*[Ø⌀\u00D8]?\s*C['’]?BORE", text)
        dia_val = float(dia_match.group(1)) * INCH_TO_MM if dia_match else None
        tokens.append({"type": "cbore", "dia_mm": dia_val, "depth_mm": depth_val, "source": "desc"})

    csk_match = re.search(rf"({NUM_PATTERN})\s*[Ø⌀\u00D8]?\s*CSK", text)
    if csk_match:
        tokens.append({"type": "csk", "dia_mm": float(csk_match.group(1)) * INCH_TO_MM, "source": "desc"})

    spot_match = re.search(rf"({NUM_PATTERN})\s*SPOT\s*(?:DRILL|FACE)", text)
    if spot_match:
        tokens.append({"type": "spot", "dia_mm": float(spot_match.group(1)) * INCH_TO_MM, "source": "desc"})

    return tokens


def parse_hole_table_lines(lines: List[str]) -> List[HoleRow]:
    """Parse text lines of a HOLE TABLE into structured operations."""

    text = "\n".join(lines)
    row_pattern = re.compile(r"^\s*([A-Z])\s+(?:Ø?\s*([A-Z0-9#./]+))?\s+(\d+)\s+(.*)$", re.IGNORECASE | re.MULTILINE)
    rows: List[HoleRow] = []

    for match in row_pattern.finditer(text):
        ref = match.group(1).upper()
        qty = int(match.group(3))
        desc = match.group(4).strip()
        features = _parse_description(desc)
        drill_token = match.group(2)
        if drill_token and not any(feature["type"] == "drill" for feature in features):
            dia_mm = parse_drill_token(drill_token) or parse_drill_token(drill_token.replace("O", "Ø"))
            if dia_mm:
                features.append(
                    {
                        "type": "drill",
                        "dia_mm": dia_mm,
                        "thru": None,
                        "depth_mm": None,
                        "from_face": None,
                        "source": "refØ",
                    }
                )

        rows.append(HoleRow(ref=ref, qty=qty, features=features, raw_desc=desc))

    try:
        row_records, feature_records, qty_sum = _hole_rows_debug_records(rows)
        _write_hole_table_debug(row_records, feature_records, qty_sum)
    except Exception:
        pass

    return rows


__all__ = ["HoleRow", "parse_drill_token", "parse_hole_table_lines"]

