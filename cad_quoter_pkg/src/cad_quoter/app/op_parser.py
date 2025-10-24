"""Shared helpers for parsing operation notes and claims."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Callable, Sequence

from cad_quoter.geometry.dxf_text import _clean_mtext as _shared_clean_mtext

__all__ = [
    "LETTER_DRILLS",
    "_CB_DIA_RE",
    "_DRILL_THRU",
    "_JIG_RE_TXT",
    "_LETTER_RE",
    "_SIZE_INCH_RE",
    "_SPOT_RE_TXT",
    "_TAP_RE",
    "_X_DEPTH_RE",
    "_parse_ops_and_claims",
    "_parse_qty",
    "_side",
]

_CB_DIA_RE = re.compile(
    r"(?:(?:[Ø⌀\u00D8]|%%[Cc])\s*)?(?P<numA>\d+(?:\.\d+)?|\.\d+|\d+\s*/\s*\d+)\s*(?:C[’']?\s*BORE|CBORE|COUNTER\s*BORE)"
    r"|(?P<numB>\d+(?:\.\d+)?|\.\d+|\d+\s*/\s*\d+)\s*(?:[Ø⌀\u00D8]|%%[Cc])\s*(?:C[’']?\s*BORE|CBORE|COUNTER\s*BORE)",
    re.I,
)
_X_DEPTH_RE = re.compile(r"[×xX]\s*([0-9]+(?:\.[0-9]+)?)")
_BACK_RE = re.compile(r"\bFROM\s+BACK\b", re.I)
_FRONT_RE = re.compile(r"\bFROM\s+FRONT\b", re.I)
_BOTH_RE = re.compile(r"\bFRONT\s*(?:[&/]|AND)\s*BACK|BOTH\s+SIDES|2\s+SIDES\b", re.I)
_SPOT_RE_TXT = re.compile(r"(?:C[’']?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|SPOT\b)", re.I)
_JIG_RE_TXT = re.compile(r"\bJIG\s*GRIND\b", re.I)
_COUNTERDRILL_RE = re.compile(
    r"\b(?:C[’']\s*DRILL|C[-\s]*DRILL|COUNTER[-\s]*DRILL)\b",
    re.I,
)
_CENTER_OR_SPOT_RE = re.compile(r"\b(CENTER\s*DRILL|SPOT\s*DRILL|SPOT)\b", re.I)
_TAP_RE = re.compile(
    r"\b(?:#?\d+[- ]\d+|[1-9]/\d+-\d+|M\d+(?:[.\s×xX]\d+)?|[\d/]+-NPT|N\.?P\.?T\.?)\s*TAP\b",
    re.I,
)
_DRILL_THRU = re.compile(r"\bDRILL(?:[-\s]+)THRU\b", re.I)
_SIZE_INCH_RE = re.compile(r"\((\d+(?:\.\d+)?|\.\d+)\)")
_LETTER_RE = re.compile(r"\b([A-Z])\b")

LETTER_DRILLS = {
    "Q": 0.3320,
    "R": 0.3390,
    "S": 0.3480,
    "T": 0.3580,
}


def _parse_qty(s: str) -> int:
    match = re.match(r"\s*\((\d+)\)\s*", s)
    if match:
        return int(match.group(1))
    match = re.search(r"(?<!\d)(\d+)\s*[xX×]\b", s)
    if match:
        return int(match.group(1))
    match = re.search(r"\bQTY[:\s]+(\d+)\b", s, re.I)
    if match:
        return int(match.group(1))
    return 1


def _side(U: str) -> str:
    has_front = bool(_FRONT_RE.search(U) or re.search(r"\bFRONT\b", U))
    has_back = bool(_BACK_RE.search(U) or re.search(r"\bBACK\b", U))
    if _BOTH_RE.search(U) or (has_front and has_back):
        return "BOTH"
    if has_back:
        return "BACK"
    if has_front:
        return "FRONT"
    return "FRONT"


def _parse_ops_and_claims(
    joined_lines: Sequence[str] | None,
    *,
    cleaner: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Collect operations and pilot drill claims from free-form text."""

    cb_groups: dict[tuple[float | None, str, float | None], int] = {}
    tap_qty = 0
    npt_qty = 0
    spot_qty = 0
    jig_qty = 0
    counterdrill_qty = 0
    claimed_pilot_diams: list[float] = []

    _clean = cleaner or _shared_clean_mtext

    for raw in joined_lines or []:
        s = _clean(str(raw or ""))
        if not s:
            continue
        U = s.upper()
        qty = _parse_qty(s)

        mcb = _CB_DIA_RE.search(s)
        if mcb:
            rawnum = (mcb.group("numA") or mcb.group("numB") or "").strip()
            dia: float | None = None
            if rawnum:
                if "/" in rawnum:
                    try:
                        dia = float(Fraction(rawnum))
                    except Exception:
                        dia = None
                else:
                    try:
                        dia = float(rawnum)
                    except Exception:
                        dia = None
            depth_match = _X_DEPTH_RE.search(s)
            depth: float | None = None
            if depth_match:
                try:
                    depth = float(depth_match.group(1))
                except Exception:
                    depth = None
            side = _side(U)
            if side == "BOTH":
                for sd in ("FRONT", "BACK"):
                    cb_groups[(dia, sd, depth)] = cb_groups.get((dia, sd, depth), 0) + qty
            else:
                cb_groups[(dia, side, depth)] = cb_groups.get((dia, side, depth), 0) + qty
            continue

        if (
            _COUNTERDRILL_RE.search(U)
            and not _CENTER_OR_SPOT_RE.search(U)
            and not _DRILL_THRU.search(U)
        ):
            counterdrill_qty += qty
            continue

        if _SPOT_RE_TXT.search(U) and ("TAP" not in U) and ("THRU" not in U):
            spot_qty += qty
            continue

        if _JIG_RE_TXT.search(U):
            jig_qty += qty
            continue

        if "NPT" in U or "N.P.T" in U:
            npt_qty += qty
            mdec = _SIZE_INCH_RE.search(s)
            if mdec:
                try:
                    claimed_pilot_diams.extend([float(mdec.group(1))] * qty)
                except Exception:
                    pass
            else:
                mlet = _LETTER_RE.search(s)
                if mlet and mlet.group(1) in LETTER_DRILLS:
                    claimed_pilot_diams.extend([LETTER_DRILLS[mlet.group(1)]] * qty)
            continue

        if _TAP_RE.search(U):
            tap_qty += qty
            mdec = _SIZE_INCH_RE.search(s)
            if mdec:
                try:
                    claimed_pilot_diams.extend([float(mdec.group(1))] * qty)
                except Exception:
                    pass
            else:
                mlet = _LETTER_RE.search(s)
                if mlet and mlet.group(1) in LETTER_DRILLS:
                    claimed_pilot_diams.extend([LETTER_DRILLS[mlet.group(1)]] * qty)
            continue

        if _DRILL_THRU.search(U):
            mdec = _SIZE_INCH_RE.search(s)
            if mdec:
                try:
                    claimed_pilot_diams.extend([float(mdec.group(1))] * qty)
                except Exception:
                    pass
            else:
                mlet = _LETTER_RE.search(s)
                if mlet and mlet.group(1) in LETTER_DRILLS:
                    claimed_pilot_diams.extend([LETTER_DRILLS[mlet.group(1)]] * qty)

    total_cb = int(sum(cb_groups.values()))
    front_cb = int(sum(q for (dia, side, _depth), q in cb_groups.items() if side == "FRONT"))
    back_cb = int(sum(q for (dia, side, _depth), q in cb_groups.items() if side == "BACK"))

    cleaned_claims: list[float] = []
    for value in claimed_pilot_diams:
        try:
            num = float(value)
        except Exception:
            continue
        cleaned_claims.append(num)

    return {
        "cb_groups": dict(cb_groups),
        "cb_total": total_cb,
        "cb_front": front_cb,
        "cb_back": back_cb,
        "tap": int(tap_qty),
        "npt": int(npt_qty),
        "spot": int(spot_qty),
        "jig": int(jig_qty),
        "counterdrill": int(counterdrill_qty),
        "claimed_pilot_diams": cleaned_claims,
    }
