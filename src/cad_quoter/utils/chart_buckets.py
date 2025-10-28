from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Tuple

__all__ = ["classify_chart_rows"]

_SEGMENT_SPLIT_RE = re.compile(r"[;•]+")
_DRILL_SPEC_PATTERNS = (
    r"[\u00D8\u2300]\s*\d+(?:\.\d+)?",
    r"\b\d+(?:\.\d+)?\s*(?:DIA|DIAM|IN\.?|MM)\b",
    r"\b\d+\s*/\s*\d+\b",
    r"\b#\s*\d+\b",
    r"\bNO\.?\s*\d+\b",
    r"\bLETTER\s+[A-Z]\b",
    r'"[A-Z]"',
)

_DRILL_SPEC_TOKEN_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _DRILL_SPEC_PATTERNS), re.IGNORECASE
)


def _coerce_qty(value: Any) -> int:
    try:
        qty = int(round(float(value)))
    except Exception:
        return 0
    return qty if qty > 0 else 0


def _normalize_desc(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _split_segments(desc: str) -> list[str]:
    if not desc:
        return []
    parts = [segment.strip() for segment in _SEGMENT_SPLIT_RE.split(desc) if segment.strip()]
    return parts or [desc.strip()]


def _segment_has_drill_spec(segment: str) -> bool:
    if not segment:
        return False
    text_upper = segment.upper()
    if "DRILL" not in text_upper:
        return False
    if _DRILL_SPEC_TOKEN_RE.search(segment):
        return True
    if re.search(r"\bDRILL\s+\"?[A-Z]\"?\b", text_upper):
        return True
    if re.search(r"([0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:IN\.?|MM|\")?\s*DRILL", segment, re.IGNORECASE):
        return True
    return False


def classify_chart_rows(
    rows: Iterable[Mapping[str, Any]] | None,
) -> Tuple[Dict[str, int], int, int]:
    """Classify chart rows into operation buckets.

    Returns a tuple of ``(buckets, row_count, qty_sum)`` where ``buckets`` maps
    bucket names (``tap``, ``cbore``, ``npt``, ``drill_spec``, ``unknown``) to
    total quantities.
    """

    totals: Dict[str, int] = {
        "tap": 0,
        "cbore": 0,
        "cdrill": 0,
        "csink": 0,
        "drill": 0,
        "jig_grind": 0,
        "spot": 0,
        "npt": 0,
        "unknown": 0,
    }
    row_count = 0
    qty_sum = 0

    if not rows:
        return {}, row_count, qty_sum

    try:
        from cad_quoter import geo_extractor as _geo_extractor  # type: ignore

        classify_op_row = getattr(_geo_extractor, "classify_op_row", None)
    except Exception:  # pragma: no cover - defensive import
        classify_op_row = None

    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        qty = _coerce_qty(raw_row.get("qty"))
        if qty <= 0:
            continue
        row_count += 1
        qty_sum += qty

        desc = _normalize_desc(
            raw_row.get("desc")
            or raw_row.get("description")
            or raw_row.get("text")
            or raw_row.get("name")
        )

        categories: set[str] = set()
        if callable(classify_op_row):
            operations = classify_op_row(desc)
            for op in operations:
                kind = str(op.get("kind") or "unknown").strip().lower()
                if kind not in totals:
                    kind = "unknown"
                if kind == "unknown" and kind in categories:
                    continue
                categories.add(kind)
        else:
            segments = _split_segments(desc)
            for segment in segments:
                segment_upper = re.sub(r"\s+", "", segment.upper())
                segment_plain_upper = segment.upper()
                if not segment_upper:
                    continue
                if "NPT" in segment_upper.replace(".", ""):
                    categories.add("npt")
                if "TAP" in segment_plain_upper:
                    categories.add("tap")
                if any(
                    token in segment_plain_upper
                    for token in ("C'BORE", "CBORE", "COUNTERBORE", "COUNTER BORE")
                ):
                    categories.add("cbore")
                if _segment_has_drill_spec(segment):
                    categories.add("drill")

        if not categories:
            categories.add("unknown")

        if "npt" in categories and "tap" in categories:
            categories.discard("tap")

        for category in categories:
            totals[category] = totals.get(category, 0) + qty

    if row_count == 0:
        return {}, row_count, qty_sum

    filtered = {key: value for key, value in totals.items() if value > 0}
    if totals.get("drill"):
        filtered["drill_spec"] = totals["drill"]
    return filtered, row_count, qty_sum
