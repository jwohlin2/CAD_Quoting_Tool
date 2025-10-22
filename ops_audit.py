import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

_SIDE_RE = re.compile(r"\b(FRONT|BACK)\b", re.I)
_DRILL_ROW_RE = re.compile(r'^Dia\s+([0-9.]+)"\s+×\s+(\d+)', re.I)
_TAP_ROW_RE = re.compile(r'^\s*(#?\d+(?:-\d+)?|[0-9/]+-[0-9]+)\s+TAP.*×\s+(\d+)\s+\((FRONT|BACK)\)', re.I)


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
        lowered = name_val.strip().lower()
        if "tap" in lowered:
            return "tap"
        if any(token in lowered for token in ("c'bore", "cbore", "counterbore")):
            return "counterbore"
        if "spot" in lowered or "c'drill" in lowered:
            return "spot"
        if "jig" in lowered and "grind" in lowered:
            return "jig_grind"
        if "drill" in lowered:
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

    counts["actions_total"] = (
        counts["drills"]
        + counts["taps_total"]
        + counts["counterbores_total"]
        + counts["spot"]
        + counts["jig_grind"]
    )

    return dict(counts)


__all__ = ["audit_operations"]
