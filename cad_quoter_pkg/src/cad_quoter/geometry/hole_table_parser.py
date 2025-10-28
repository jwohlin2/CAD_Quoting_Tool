"""Shared hole table parsing helpers."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
_DEBUG_FIELDNAMES = (
    "qty",
    "kind",
    "side",
    "tool",
    "diam_token",
    "depth_token",
    "raw_text",
)


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


def _hole_rows_debug_records(rows: List[HoleRow]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    qty_sum = 0
    for row in rows:
        qty = int(getattr(row, "qty", 0) or 0)
        raw_desc = getattr(row, "raw_desc", "")
        features = list(getattr(row, "features", []) or [])
        if not features:
            records.append(
                {
                    "qty": qty,
                    "kind": "",
                    "side": "",
                    "tool": "",
                    "diam_token": "",
                    "depth_token": "",
                    "raw_text": raw_desc,
                }
            )
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
                "side": side,
                "tool": _format_tool_token(kind, feature),
                "diam_token": diam_token,
                "depth_token": _format_depth_token(feature),
                "raw_text": raw_desc,
            }
            records.append(record)
            qty_sum += qty
    return records, qty_sum


def _hole_rows_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {key: 0 for key in ("drill", "tap", "counterbore", "counterdrill", "jig_grind")}
    for record in records:
        raw_kind = str(record.get("kind") or "").lower()
        kind = {
            "cbore": "counterbore",
            "counterbore": "counterbore",
            "cdrill": "counterdrill",
            "counterdrill": "counterdrill",
        }.get(raw_kind, raw_kind)
        if kind in totals:
            try:
                qty_val = int(record.get("qty") or 0)
            except Exception:
                qty_val = 0
            totals[kind] += qty_val
    return totals


def _write_hole_table_debug(records: list[dict[str, Any]], qty_sum: int) -> None:
    if not records:
        return
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        with _DEBUG_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_DEBUG_FIELDNAMES)
            writer.writeheader()
            writer.writerows(records)
        totals = _hole_rows_totals(records)
        with _DEBUG_TOTALS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(totals, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"[TABLE-DUMP] rows={len(records)} qty_sum={qty_sum} -> {_DEBUG_ROWS_PATH.as_posix()}"
        )
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

    tap_re = re.compile(r"(\d+/\d+|\#\d+)\s*-\s*(\d+)|(\d+(?:\.\d+)?)\s*-\s*(\d+)")
    if "TAP" in text:
        match = tap_re.search(text)
        if match:
            major = match.group(1) or match.group(3)
            pitch = match.group(2) or match.group(4)
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
                }
            )

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
        records, qty_sum = _hole_rows_debug_records(rows)
        _write_hole_table_debug(records, qty_sum)
    except Exception:
        pass

    return rows


__all__ = ["HoleRow", "parse_drill_token", "parse_hole_table_lines"]

