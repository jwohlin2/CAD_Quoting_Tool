from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Letter/number drill charts (subset; extend as needed)
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


def inch(x: float) -> float:
    return x


def mm(x: float) -> float:
    return x


def to_mm(value: float, unit: str) -> float:
    return value * INCH_TO_MM if unit == "in" else value


def _num(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def parse_drill_token(tok: str) -> Optional[float]:
    """Return drill diameter in mm from tokens like Ø.3750, 3/8, #7, Q, 0.339"""
    t = tok.strip().upper().replace("Ø", "").replace("O/", "Ø").replace(" ", "")
    # fractional inch (e.g., 3/8)
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", t)
    if m:
        return _frac_to_mm(int(m.group(1)), int(m.group(2)))
    # decimal inch (.3750 or 0.3750)
    m = re.fullmatch(r"(?:0?)(\.\d+)", t)
    if m:
        return float(m.group(1)) * INCH_TO_MM
    # decimal mm (e.g., 5.50MM)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)MM", t)
    if m:
        return float(m.group(1))
    # number or letter drills
    if t in LETTER_DRILLS_INCH:
        return LETTER_DRILLS_INCH[t] * INCH_TO_MM
    if t in NUMBER_DRILLS_INCH:
        return NUMBER_DRILLS_INCH[t] * INCH_TO_MM
    # raw decimal without symbol (0.339)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", t)
    if m and float(m.group(1)) < 1.5:  # treat as inch if < 1.5
        return float(m.group(1)) * INCH_TO_MM
    return None


def _frac_to_mm(n: int, d: int) -> float:
    return (n / d) * INCH_TO_MM


@dataclass
class HoleRow:
    ref: str
    qty: int
    features: List[Dict[str, Any]]
    raw_desc: str


def parse_hole_table_lines(lines: List[str]) -> List[HoleRow]:
    """Parse text lines of a HOLE TABLE into structured ops."""
    txt = "\n".join(lines)
    row_re = re.compile(
        r"^\s*([A-Z])\s+(?:Ø?\s*([A-Z0-9#./]+))?\s+(\d+)\s+(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    out: List[HoleRow] = []
    for m in row_re.finditer(txt):
        ref = m.group(1).upper()
        qty = int(m.group(3))
        desc = m.group(4).strip()

        features = _parse_description(desc)
        tok = m.group(2)
        if tok and not any(f["type"] == "drill" for f in features):
            dia_mm = parse_drill_token(tok) or parse_drill_token(tok.replace("O", "Ø"))
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

        out.append(HoleRow(ref=ref, qty=qty, features=features, raw_desc=desc))
    return out


def _parse_description(desc: str) -> List[Dict[str, Any]]:
    s = " ".join(desc.upper().split())
    tokens: List[Dict[str, Any]] = []

    tap_re = re.compile(r"(\d+/\d+|\#\d+)\s*-\s*(\d+)|(\d+(?:\.\d+)?)\s*-\s*(\d+)")
    if "TAP" in s:
        m = tap_re.search(s)
        if m:
            major = m.group(1) or m.group(3)
            pitch = m.group(2) or m.group(4)
            major_mm = parse_drill_token(major) if major else None
            depth_mm, thru = _depth_or_thru(s)
            from_face = _from_face(s)
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

    for m in re.finditer(r"Ø?\(?\s*(\d+(?:\.\d+)?)\s*\)?\s*(?:DRILL|THRU|TYP|$)", s):
        d = parse_drill_token(m.group(1))
        if d:
            depth_mm, thru = _depth_or_thru(s)
            from_face = _from_face(s)
            tokens.append(
                {
                    "type": "drill",
                    "dia_mm": d,
                    "depth_mm": depth_mm,
                    "thru": thru,
                    "from_face": from_face,
                    "source": "desc",
                }
            )

    m = re.search(r"C['’]?BORE\s+X\s+(\d+(?:\.\d+)?)\s*DEEP", s)
    if m:
        depth = float(m.group(1)) * INCH_TO_MM
        d = None
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*C['’]?BORE", s)
        if m2:
            d = float(m2.group(1)) * INCH_TO_MM
        tokens.append({"type": "cbore", "dia_mm": d, "depth_mm": depth, "source": "desc"})

    for m in re.finditer(r"(\d+)\s*\)\s*\.?(\d+(?:\.\d+)?)\s*X\s*45", s):
        qty = int(m.group(1))
        size_in = float(m.group(2))
        tokens.append(
            {
                "type": "chamfer",
                "qty": qty,
                "size_mm": size_in * INCH_TO_MM,
                "source": "desc",
            }
        )

    return _dedup_ops(tokens)


def _depth_or_thru(s: str) -> tuple[Optional[float], Optional[bool]]:
    if " THRU" in s:
        return None, True
    m = re.search(r"X\s*\.?(\d+(?:\.\d+)?)\s*DEEP", s)
    if m:
        return float(m.group(1)) * INCH_TO_MM, False
    return None, None


def _from_face(s: str) -> Optional[str]:
    if "FROM FRONT" in s:
        return "front"
    if "FROM BACK" in s:
        return "back"
    return None


def _dedup_ops(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for it in items:
        key = (
            it.get("type"),
            round(it.get("dia_mm", -1.0), 3),
            it.get("thru"),
            round(it.get("depth_mm", -1.0), 1),
            it.get("thread"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

