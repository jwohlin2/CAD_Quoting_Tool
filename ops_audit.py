import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

_CB_DIA_RE = re.compile(
    r"(?:(?:[Ø⌀\u00D8]|%%[Cc])\s*)?"
    r"(?P<numA>\d+(?:\.\d+)?|\.\d+|\d+\s*/\s*\d+)\s*"
    r"(?:C[’']?\s*BORE|CBORE|COUNTER\s*BORE)"
    r"|"
    r"(?P<numB>\d+(?:\.\d+)?|\.\d+|\d+\s*/\s*\d+)\s*"
    r"(?:[Ø⌀\u00D8]|%%[Cc])\s*"
    r"(?:C[’']?\s*BORE|CBORE|COUNTER\s*BORE)",
    re.I,
)
_X_DEPTH_RE = re.compile(r"[×xX]\s*([0-9]+(?:\.[0-9]+)?)")
_BACK_RE = re.compile(r"\bFROM\s+BACK\b", re.I)
_FRONT_RE = re.compile(r"\bFROM\s+FRONT\b", re.I)
_BOTH_RE = re.compile(r"\bFRONT\s*&\s*BACK|BOTH\s+SIDES|2\s+SIDES\b", re.I)
_SPOT_RE_TXT = re.compile(r"(?:C[’']?\s*DRILL|CENTER\s*DRILL|SPOT\s*DRILL|SPOT\b)", re.I)
_JIG_RE_TXT = re.compile(r"\bJIG\s*GRIND\b", re.I)
_TAP_RE = re.compile(
    r"\b(?:#?\d+[- ]\d+|[1-9]/\d+-\d+|[1-9]/\d+|[\d/]+-NPT|N\.?P\.?T\.?)\s*TAP\b",
    re.I,
)
_DRILL_THRU = re.compile(r"\bDRILL\s+THRU\b", re.I)
_SIZE_INCH_RE = re.compile(r"\((\d+(?:\.\d+)?|\.\d+)\)")
_LETTER_RE = re.compile(r"\b([A-Z])\b")
_SIDE_RE = re.compile(r"\b(FRONT|BACK)\b", re.I)
_DRILL_ROW_RE = re.compile(r'^Dia\s+([0-9.]+)"\s+×\s+(\d+)', re.I)
_TAP_ROW_RE = re.compile(r'^\s*(#?\d+(?:-\d+)?|[0-9/]+-[0-9]+)\s+TAP.*×\s+(\d+)\s+\((FRONT|BACK)\)', re.I)


def _parse_qty(s: str) -> int:
    match = re.match(r"\s*\((\d+)\)\s*", s)
    if match:
        return int(match.group(1))
    match = re.search(r"(?<!\d)(\d+)\s*[xX×]", s)
    if match:
        return int(match.group(1))
    match = re.search(r"\bQTY[:\s]+(\d+)\b", s, re.I)
    if match:
        return int(match.group(1))
    return 1


def _side(U: str) -> str:
    if _BOTH_RE.search(U):
        return "BOTH"
    has_back = bool(_BACK_RE.search(U) or re.search(r"\bBACK\b", U))
    has_front = bool(_FRONT_RE.search(U) or re.search(r"\bFRONT\b", U))
    if has_back and has_front:
        return "BOTH"
    if has_back:
        return "BACK"
    if has_front:
        return "FRONT"
    return "FRONT"


def _side_of(text: str | None) -> str:
    match = _SIDE_RE.search(text or "")
    return (match.group(1) or "").upper() if match else "-"


def _row_kind(row: Any) -> str:
    if row is None:
        return ""
    if isinstance(row, Mapping):
        kind_val = row.get("kind")
        if isinstance(kind_val, str) and kind_val.strip():
            return kind_val.strip().lower()
        # fall back to name-based inference if possible
        name_val = row.get("name") or row.get("desc") or row.get("description")
    else:
        kind_val = getattr(row, "kind", None)
        if isinstance(kind_val, str) and kind_val.strip():
            return kind_val.strip().lower()
        name_val = getattr(row, "name", None) or getattr(row, "desc", None)
    if isinstance(name_val, str):
        text = name_val.strip()
        U = text.upper()
        if "TAP" in U or _TAP_RE.search(text):
            return "tap"
        if _CB_DIA_RE.search(text) or any(token in U for token in ("C'BORE", "CBORE", "COUNTER BORE")):
            return "counterbore"
        if (
            _SPOT_RE_TXT.search(text)
            and not _DRILL_THRU.search(text)
            and not ("TAP" in U or _TAP_RE.search(text))
        ):
            return "spot"
        if _JIG_RE_TXT.search(text):
            return "jig_grind"
        if "DRILL" in U:
            return "drill"
    return ""


def _row_qty(row: Any) -> int:
    if row is None:
        return 0
    if isinstance(row, Mapping):
        qty_val = row.get("qty")
    else:
        qty_val = getattr(row, "qty", None)
    try:
        return int(qty_val or 0)
    except Exception:
        try:
            return int(float(qty_val))  # type: ignore[arg-type]
        except Exception:
            return 0


def _row_side(row: Any) -> str:
    if row is None:
        return "-"
    if isinstance(row, Mapping):
        side_val = row.get("side") or row.get("face")
    else:
        side_val = getattr(row, "side", None) or getattr(row, "face", None)
    return _side_of(str(side_val) if side_val is not None else None)


def _iter_rows(rows: Any) -> Iterable[Any]:
    if rows is None:
        return []
    if isinstance(rows, Mapping):
        candidate = rows.get("rows")
        if isinstance(candidate, Sequence):
            return candidate
    if isinstance(rows, Sequence):
        return rows
    return []


def _extract_ops_from_text(text: str) -> dict[str, int]:
    counts = defaultdict(int)
    for raw in text.splitlines():
        s = (raw or "").strip()
        if not s:
            continue
        qty = _parse_qty(s)
        if qty <= 0:
            continue
        U = s.upper()
        side = _side(U)

        if "TAP" in U or _TAP_RE.search(s):
            counts["taps_total"] += qty
            if side == "BACK":
                counts["taps_back"] += qty
            elif side == "BOTH":
                counts["taps_front"] += qty
                counts["taps_back"] += qty
            else:
                counts["taps_front"] += qty

        if ("CBORE" in U) or ("C'BORE" in U) or ("COUNTER BORE" in U) or _CB_DIA_RE.search(s):
            counts["counterbores_total"] += qty
            if side == "BACK":
                counts["counterbores_back"] += qty
            elif side == "BOTH":
                counts["counterbores_front"] += qty
                counts["counterbores_back"] += qty
            else:
                counts["counterbores_front"] += qty

        if (
            _SPOT_RE_TXT.search(s)
            and not _DRILL_THRU.search(s)
            and not ("TAP" in U or _TAP_RE.search(s))
        ):
            counts["spot"] += qty

        if _JIG_RE_TXT.search(s):
            counts["jig_grind"] += qty

    return dict(counts)


def audit_operations(planner_ops_rows: Any, removal_sections_text: str | None) -> dict[str, int]:
    """Return aggregated counts for drilling/tapping/counterbore/etc operations."""

    counts = defaultdict(int)

    for row in _iter_rows(planner_ops_rows):
        kind = _row_kind(row)
        qty = _row_qty(row)
        side = _row_side(row)
        if qty <= 0 or not kind:
            continue
        if kind == "drill":
            counts["drills"] += qty
        elif kind in {"tap", "tapping"}:
            counts["taps_total"] += qty
            if side == "FRONT":
                counts["taps_front"] += qty
            elif side == "BACK":
                counts["taps_back"] += qty
        elif kind in {"counterbore", "c'bore", "cbore"}:
            counts["counterbores_total"] += qty
            if side == "FRONT":
                counts["counterbores_front"] += qty
            elif side == "BACK":
                counts["counterbores_back"] += qty
        elif kind in {"spot", "spot_drill", "spot-drill", "spot drill"}:
            counts["spot"] += qty
        elif kind in {"jig_grind", "jig-grind", "jig grind"}:
            counts["jig_grind"] += qty

    text = removal_sections_text or ""
    if counts["drills"] == 0:
        for line in text.splitlines():
            match = _DRILL_ROW_RE.match(line.strip())
            if match:
                counts["drills"] += int(match.group(2))

    tap_front = 0
    tap_back = 0
    for line in text.splitlines():
        match = _TAP_ROW_RE.match(line.strip())
        if not match:
            continue
        qty = int(match.group(2))
        side = (match.group(3) or "").upper()
        if side == "FRONT":
            tap_front += qty
        elif side == "BACK":
            tap_back += qty

    if tap_front or tap_back:
        counts["taps_total"] += tap_front + tap_back
        counts["taps_front"] += tap_front
        counts["taps_back"] += tap_back

    text_counts = _extract_ops_from_text(text)
    if text_counts.get("taps_total", 0) > counts["taps_total"]:
        counts["taps_total"] = text_counts.get("taps_total", 0)
        counts["taps_front"] = text_counts.get("taps_front", 0)
        counts["taps_back"] = text_counts.get("taps_back", 0)
    if text_counts.get("counterbores_total", 0) > counts["counterbores_total"]:
        counts["counterbores_total"] = text_counts.get("counterbores_total", 0)
        counts["counterbores_front"] = text_counts.get("counterbores_front", 0)
        counts["counterbores_back"] = text_counts.get("counterbores_back", 0)
    counts["spot"] = max(counts["spot"], text_counts.get("spot", 0))
    counts["jig_grind"] = max(counts["jig_grind"], text_counts.get("jig_grind", 0))

    counts["actions_total"] = (
        counts["drills"]
        + counts["taps_total"]
        + counts["counterbores_total"]
        + counts["spot"]
        + counts["jig_grind"]
    )

    return dict(counts)


__all__ = ["audit_operations"]
