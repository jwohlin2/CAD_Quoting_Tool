"""Shared hole table parsing helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

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
VALUE_PATTERN = rf"(?:\d+\s*/\s*\d+|{NUM_PATTERN})"


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


def _parse_length_token(tok: str | None) -> Optional[float]:
    """Parse a nominal depth/length token (defaults to inches)."""

    if not tok:
        return None
    cleaned = tok.strip().upper().replace("\"", "").replace("IN", "")
    cleaned = cleaned.replace("INCH", "").replace("INCHES", "")
    if not cleaned:
        return None
    if cleaned.endswith("MM"):
        try:
            return float(cleaned[:-2])
        except Exception:
            return None
    if "/" in cleaned:
        try:
            return float(Fraction(cleaned)) * INCH_TO_MM
        except Exception:
            return None
    try:
        return float(cleaned) * INCH_TO_MM
    except Exception:
        return None


def _normalize_side_token(token: str | None) -> str | None:
    if not token:
        return None
    token = token.strip().lower()
    if not token:
        return None
    if "back" in token:
        return "back"
    if "front" in token:
        return "front"
    return None


_SIDE_TOKEN_RE = re.compile(r"\bFROM\s+(FRONT|BACK)\b", re.I)
_SIDE_BOTH_RE = re.compile(
    r"FRONT\s*&\s*BACK|FRONT\s+AND\s+BACK|BOTH\s+SIDES|TWO\s+SIDES|2\s+SIDES",
    re.I,
)
_CBORE_RE = re.compile(
    rf"(?P<dia>(?:Ø|⌀)?\s*{VALUE_PATTERN})\s*(?:C['’]?\s*BORE|COUNTER\s*BORE)"
    rf"\s*(?:[×xX]\s*(?P<depth>{VALUE_PATTERN})\s*DEEP)?"
    r"(?P<tail>[^;]*)",
    re.I,
)
_CSK_RE = re.compile(
    rf"(?P<dia>(?:Ø|⌀)?\s*{VALUE_PATTERN})\s*(?:C['’]?\s*SINK|CSK|COUNTER\s*SINK)"
    r"(?P<tail>[^;]*)",
    re.I,
)
_SPOT_RE = re.compile(
    rf"(?P<dia>(?:Ø|⌀)?\s*{VALUE_PATTERN})?\s*SPOT(?:\s*(?:DRILL|FACE))?"
    rf"(?:\s*[×xX]\s*(?P<depth>{VALUE_PATTERN})\s*DEEP)?",
    re.I,
)
_DEPTH_RE = re.compile(rf"(?P<depth>{VALUE_PATTERN})\s*DEEP", re.I)
_TAP_RE = re.compile(
    rf"((?:\d+/\d+)|(?:#\s*\d+)|({NUM_PATTERN}))\s*-\s*(\d+)\s*TAP",
    re.I,
)


def _extract_sides(text: str) -> list[str]:
    sides: list[str] = []
    for match in _SIDE_TOKEN_RE.finditer(text):
        side = _normalize_side_token(match.group(1))
        if side and side not in sides:
            sides.append(side)
    if not sides and _SIDE_BOTH_RE.search(text):
        sides = ["front", "back"]
    return sides


def _sides_for_segment(segment: str, default_sides: Sequence[str]) -> list[str]:
    sides = _extract_sides(segment)
    if sides:
        return sides
    if default_sides:
        return list(default_sides)
    return ["front"]


def _feature_sides(feature: Mapping[str, Any]) -> list[str]:
    sides_raw = feature.get("sides")
    if isinstance(sides_raw, (list, tuple, set)):
        sides = [
            side
            for side in (
                _normalize_side_token(str(item)) if item is not None else None
                for item in sides_raw
            )
            if side
        ]
        if sides:
            return sides
    side = feature.get("side") or feature.get("from_face")
    norm = _normalize_side_token(side if isinstance(side, str) else None)
    if norm:
        return [norm]
    return ["front"]


def _summarise_features(features: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for feature in features:
        detail = {
            "type": feature.get("type"),
            "side": feature.get("side") or feature.get("sides"),
        }
        if feature.get("thread"):
            detail["thread"] = feature.get("thread")
        if feature.get("thru"):
            detail["thru"] = True
        if feature.get("depth_mm") is not None:
            detail["depth_mm"] = round(float(feature["depth_mm"]), 3)
        summary.append(detail)
    return summary


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

    tap_re = re.compile(rf"(\d+/\d+|\#\d+)\s*-\s*(\d+)|({NUM_PATTERN})\s*-\s*(\d+)")
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


@dataclass
class HoleRow:
    ref: str
    qty: int
    features: List[Dict[str, Any]]
    raw_desc: str
    block_thickness_in: float | None = None


def parse_hole_row(
    ref: str,
    qty: int,
    desc: str,
    drill_token: str | None = None,
    *,
    block_thickness_in: float | None = None,
) -> List[Dict[str, Any]]:
    """Parse a HOLE TABLE row description with enhanced rule handling."""

    text = desc or ""
    text_upper = text.upper()
    base_features = _parse_description(text)

    spot_only = bool(base_features) and all(f.get("type") == "spot" for f in base_features)
    default_sides = _extract_sides(text_upper)

    drill_dia_mm: float | None = None
    for feature in base_features:
        if feature.get("type") == "drill" and feature.get("dia_mm"):
            drill_dia_mm = float(feature["dia_mm"])
            break
    if drill_dia_mm is None and drill_token:
        drill_dia_mm = parse_drill_token(drill_token) or parse_drill_token(drill_token.replace("O", "Ø"))

    if drill_dia_mm and not spot_only and not any(f.get("type") == "drill" for f in base_features):
        base_features.append(
            {
                "type": "drill",
                "dia_mm": drill_dia_mm,
                "thru": True if "THRU" in text_upper else None,
                "depth_mm": None,
                "from_face": None,
                "source": "refØ",
            }
        )

    # Replace counterbore features with side-aware variants when detected.
    cbore_features: list[dict[str, Any]] = []
    for match in _CBORE_RE.finditer(text):
        dia_tok = (match.group("dia") or "").replace("Ø", "").replace("⌀", "").strip()
        depth_tok = match.group("depth")
        dia_mm = parse_drill_token(dia_tok) if dia_tok else None
        depth_mm = _parse_length_token(depth_tok)
        segment = match.group(0) + (match.group("tail") or "")
        for side in _sides_for_segment(segment, default_sides):
            cbore_features.append(
                {
                    "type": "cbore",
                    "dia_mm": dia_mm,
                    "depth_mm": depth_mm,
                    "side": side,
                    "source": "rules_v2",
                }
            )
    if cbore_features:
        base_features = [f for f in base_features if f.get("type") != "cbore"]
        base_features.extend(cbore_features)

    # Enrich countersink with side hints
    csk_features: list[dict[str, Any]] = []
    for match in _CSK_RE.finditer(text):
        dia_tok = (match.group("dia") or "").replace("Ø", "").replace("⌀", "").strip()
        dia_mm = parse_drill_token(dia_tok) if dia_tok else None
        segment = match.group(0) + (match.group("tail") or "")
        for side in _sides_for_segment(segment, default_sides):
            csk_features.append(
                {
                    "type": "csk",
                    "dia_mm": dia_mm,
                    "side": side,
                    "source": "rules_v2",
                }
            )
    if csk_features:
        base_features = [f for f in base_features if f.get("type") != "csk"]
        base_features.extend(csk_features)

    processed: list[dict[str, Any]] = []
    tap_features: list[dict[str, Any]] = []

    for feature in base_features:
        entry = dict(feature)
        f_type = entry.get("type")
        from_face = entry.pop("from_face", None)
        if from_face and not entry.get("side") and not entry.get("sides"):
            norm = _normalize_side_token(from_face)
            if norm:
                entry["side"] = norm
        if not entry.get("side") and not entry.get("sides"):
            sides_hint = list(default_sides)
            if f_type in {"cbore", "csk"} and len(default_sides) > 1:
                entry["sides"] = tuple(default_sides)
            elif sides_hint:
                entry["side"] = sides_hint[0]
            else:
                entry["side"] = "front"
        if f_type == "tap":
            thread = entry.get("thread")
            if isinstance(thread, str):
                entry["thread"] = thread.replace(" ", "")
            if entry.get("thru") and entry.get("depth_mm") is None and block_thickness_in:
                entry["depth_mm"] = (block_thickness_in + 0.05) * INCH_TO_MM
            tap_features.append(entry)
        elif f_type == "drill":
            if entry.get("dia_mm") is None and drill_dia_mm is not None:
                entry["dia_mm"] = drill_dia_mm
            if entry.get("thru") and entry.get("depth_mm") is None and block_thickness_in:
                entry["depth_mm"] = (block_thickness_in + 0.05) * INCH_TO_MM
        elif f_type == "spot":
            # Ensure spot rows carry a side hint when possible
            if not entry.get("side") and default_sides:
                entry["side"] = default_sides[0]
            if not entry.get("side"):
                entry["side"] = "front"
        processed.append(entry)

    if tap_features:
        existing_drills = [feat for feat in processed if feat.get("type") == "drill"]
        for tap in tap_features:
            tap_depth = tap.get("depth_mm")
            if tap.get("thru") and tap_depth is None and block_thickness_in:
                tap_depth = (block_thickness_in + 0.05) * INCH_TO_MM
            if tap_depth is not None:
                for drill in existing_drills:
                    if drill.get("depth_mm") is None:
                        drill["depth_mm"] = tap_depth
        if drill_dia_mm and not existing_drills:
            for tap in tap_features:
                sides = _feature_sides(tap)
                tap_depth = tap.get("depth_mm")
                if tap.get("thru") and tap_depth is None and block_thickness_in:
                    tap_depth = (block_thickness_in + 0.05) * INCH_TO_MM
                for side in sides:
                    processed.append(
                        {
                            "type": "drill",
                            "dia_mm": drill_dia_mm,
                            "depth_mm": tap_depth,
                            "thru": tap.get("thru"),
                            "side": side,
                            "source": "tap_pre_drill",
                        }
                    )

    # Ensure spot-only rows do not accidentally gain drill operations
    if spot_only:
        processed = [feat for feat in processed if feat.get("type") == "spot"]

    logger.info("[rules] row %s qty=%s -> %s", ref, qty, _summarise_features(processed))
    return processed


def parse_hole_table_lines(
    lines: List[str],
    *,
    rules_v2: bool = False,
    block_thickness_in: float | None = None,
) -> List[HoleRow]:
    """Parse text lines of a HOLE TABLE into structured operations."""

    text = "\n".join(lines)
    row_pattern = re.compile(r"^\s*([A-Z])\s+(?:Ø?\s*([A-Z0-9#./]+))?\s+(\d+)\s+(.*)$", re.IGNORECASE | re.MULTILINE)
    rows: List[HoleRow] = []

    if rules_v2:
        logger.info(
            "[rules] parser_rules_v2 enabled (thickness≈%s in)",
            f"{block_thickness_in:.3f}" if block_thickness_in else "?",
        )

    for match in row_pattern.finditer(text):
        ref = match.group(1).upper()
        qty = int(match.group(3))
        desc = match.group(4).strip()
        drill_token = match.group(2)
        if rules_v2:
            features = parse_hole_row(
                ref,
                qty,
                desc,
                drill_token,
                block_thickness_in=block_thickness_in,
            )
        else:
            features = _parse_description(desc)
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

        rows.append(
            HoleRow(
                ref=ref,
                qty=qty,
                features=features,
                raw_desc=desc,
                block_thickness_in=block_thickness_in,
            )
        )

    return rows


__all__ = [
    "HoleRow",
    "parse_drill_token",
    "parse_hole_row",
    "parse_hole_table_lines",
]
